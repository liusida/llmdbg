"""Pure-Python model programs backed by llmdbg's resident model runtime."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Iterator, Sequence

from llmdbg.model_runtime import ModelExecution, prepare_model_execution


@dataclass(frozen=True)
class Prompt:
    text: str


class TensorHandle(Sequence[float]):
    """A list-like handle to a vector owned by the model runtime."""

    def __init__(self, session: "DebugSession", layer: str, vector: str, values: tuple[float, ...]):
        self._session, self.layer, self.vector, self._values = session, layer, vector, values

    def __len__(self) -> int:
        return len(self._values)

    def __iter__(self) -> Iterator[float]:
        return iter(self._values)

    def __getitem__(self, index: int | slice) -> float | tuple[float, ...]:
        return self._values[index]

    def __setitem__(self, index: int | slice, value: float | Sequence[float]) -> None:
        indices = range(*index.indices(len(self))) if isinstance(index, slice) else (index,)
        values = value if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) else None
        for offset, dimension in enumerate(indices):
            replacement = float(values[offset] if values is not None else value)
            self._session.intervene(self.layer, self.vector, dimension, replacement)
        self._values = self._session.vector(self.layer, self.vector)

    def __add__(self, update: "TensorHandle") -> "TensorHandle":
        if not isinstance(update, TensorHandle):
            return NotImplemented
        return self._session.output(update.layer)

    def __repr__(self) -> str:
        preview = ", ".join(f"{value:+.5f}" for value in self._values[:6])
        suffix = ", ..." if len(self) > 6 else ""
        return f"TensorHandle([{preview}{suffix}], shape=({len(self)},))"


class LayerHandle:
    def __init__(self, session: "DebugSession", index: int):
        self.session, self.index = session, index

    def _name(self, kind: str) -> str:
        return f"transformer.h.{self.index}.{kind}"

    def layernorm(self, _x: TensorHandle) -> TensorHandle:
        caller = inspect.currentframe().f_back.f_code.co_name  # type: ignore[union-attr]
        kind = "mlp" if caller == "mlp" else "attention"
        return self.session.detail(self._name(kind), "normalized_input")

    def qkv_projection(self, _x: TensorHandle) -> TensorHandle:
        return self.session.detail(self._name("attention"), "attention_qkv")

    def attention_mixing(self, _qkv: TensorHandle) -> TensorHandle:
        return self.session.detail(self._name("attention"), "attention_mixing")

    def output_projection(self, _mixed: TensorHandle) -> TensorHandle:
        return self.session.detail(self._name("attention"), "attention_projection")

    def expansion(self, _x: TensorHandle) -> TensorHandle:
        return self.session.detail(self._name("mlp"), "mlp_expansion_pre_activation")

    def gelu(self, _hidden: TensorHandle) -> TensorHandle:
        return self.session.detail(self._name("mlp"), "mlp_expansion")

    def projection(self, _hidden: TensorHandle) -> TensorHandle:
        return self.session.detail(self._name("mlp"), "mlp_projection")


class RemoteModel:
    def __init__(self, model: str, *, revision: str, device: str, dtype: str):
        self.model, self.revision, self.device, self.dtype = model, revision, device, dtype
        self._session: DebugSession | None = None

    def tokenize(self, prompt: str) -> Prompt:
        return Prompt(prompt)

    @property
    def layers(self) -> tuple[LayerHandle, ...]:
        session = self._require_session()
        return tuple(LayerHandle(session, index) for index in range(session.layer_count))

    def token_embedding(self, _tokens: Prompt) -> TensorHandle:
        return self._require_session().output("token_embedding")

    def final_layernorm(self, _x: TensorHandle) -> TensorHandle:
        return self._require_session().output("transformer.ln_f")

    def vocabulary_projection(self, x: TensorHandle) -> TensorHandle:
        return self._require_session().vocabulary_projection(x)

    def _require_session(self) -> "DebugSession":
        if self._session is None:
            raise RuntimeError("model operations must run inside with debug(...)")
        return self._session


class DebugSession:
    def __init__(self, model: RemoteModel, tokens: Prompt, token: int):
        self.model, self.tokens, self.token = model, tokens, token
        self.execution: ModelExecution | None = None
        self._states: dict[str, Any] = {}

    def __enter__(self) -> "DebugSession":
        self.execution = prepare_model_execution(
            self.model.model, revision=self.model.revision, prompt=self.tokens.text,
            device=self.model.device, dtype=self.model.dtype,
        )
        self.model._session = self
        self._refresh()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.model._session = None

    @property
    def layer_count(self) -> int:
        assert self.execution is not None
        return len(self.execution.runtime.model.transformer.h)

    def _refresh(self) -> None:
        assert self.execution is not None
        trace = self.execution.trace_token(self.token)
        self._states = {state.layer.name: state for state in trace.states}

    def vector(self, layer: str, vector: str) -> tuple[float, ...]:
        state = self._states[layer]
        return state.output_vector if vector == "output" else state.details[vector]

    def output(self, layer: str) -> TensorHandle:
        return TensorHandle(self, layer, "output", self.vector(layer, "output"))

    def detail(self, layer: str, vector: str) -> TensorHandle:
        return TensorHandle(self, layer, vector, self.vector(layer, vector))

    def intervene(self, layer: str, vector: str, index: int, value: float) -> None:
        assert self.execution is not None
        self.execution.set_intervention(self.token, layer, vector, index, value)
        self._refresh()

    def vocabulary_projection(self, x: TensorHandle) -> TensorHandle:
        import torch
        assert self.execution is not None
        runtime = self.execution.runtime
        weight = runtime.model.lm_head.weight
        hidden = torch.tensor(x._values, device=weight.device, dtype=weight.dtype)
        logits = torch.mv(weight, hidden).float().cpu().tolist()
        return TensorHandle(self, "vocabulary", "logits", tuple(logits))


def gpt2(
    model: str,
    *,
    revision: str = "main",
    device: str = "cuda",
    dtype: str = "bfloat16",
) -> RemoteModel:
    return RemoteModel(model, revision=revision, device=device, dtype=dtype)


def debug(model: RemoteModel, tokens: Prompt, *, token: int = 0) -> DebugSession:
    return DebugSession(model, tokens, token)
