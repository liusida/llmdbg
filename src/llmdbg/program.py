"""Pure-Python model programs backed by llmdbg's resident model runtime."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Iterator, Sequence

from llmdbg.model_runtime import ModelExecution, TokenPrediction, prepare_model_execution


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

    def __mul__(self, other: "TensorHandle") -> "TensorHandle":
        if not isinstance(other, TensorHandle):
            return NotImplemented
        layer = self.layer if self.layer.endswith(".mlp") else other.layer
        return self._session.detail(layer, "mlp_expansion")

    def __repr__(self) -> str:
        preview = ", ".join(f"{value:+.5f}" for value in self._values[:6])
        suffix = ", ..." if len(self) > 6 else ""
        return f"TensorHandle([{preview}{suffix}], shape=({len(self)},))"


class LayerHandle:
    def __init__(self, session: "DebugSession", index: int):
        self.session, self.index = session, index

    def _name(self, kind: str) -> str:
        return self.session.layer_name(self.index, kind)

    @property
    def kind(self) -> str:
        return self.session.layer_kind(self.index)

    def layernorm(self, _x: TensorHandle) -> TensorHandle:
        caller = inspect.currentframe().f_back.f_code.co_name  # type: ignore[union-attr]
        kind = "mlp" if caller == "mlp" else "attention"
        return self.session.detail(self._name(kind), "normalized_input")

    def qkv_projection(self, _x: TensorHandle) -> TensorHandle:
        return self.session.detail(self._name("attention"), "attention_qkv")

    def attention_mixing(self, _qkv: TensorHandle) -> TensorHandle:
        return self.session.detail(self._name("attention"), "attention_mixing")

    def output_projection(self, _mixed: TensorHandle) -> TensorHandle:
        caller = inspect.currentframe().f_back.f_code.co_name  # type: ignore[union-attr]
        kind = caller if caller in {"full_attention", "gated_linear_attention"} else "attention"
        return self.session.detail(self._name(kind), "attention_projection")

    def expansion(self, _x: TensorHandle) -> TensorHandle:
        return self.session.detail(self._name("mlp"), "mlp_expansion_pre_activation")

    def gelu(self, _hidden: TensorHandle) -> TensorHandle:
        return self.session.detail(self._name("mlp"), "mlp_expansion")

    def projection(self, _hidden: TensorHandle) -> TensorHandle:
        return self.session.detail(self._name("mlp"), "mlp_projection")

    def input_layernorm(self, _x: TensorHandle) -> TensorHandle:
        return self.session.detail(self._name(self.kind), "normalized_input")

    def full_attention(self, _x: TensorHandle) -> TensorHandle:
        return self.session.detail(self._name("full_attention"), "full_attention_mixing")

    def gated_delta(self, _x: TensorHandle) -> TensorHandle:
        return self.session.detail(self._name("gated_linear_attention"), "gated_delta_mixing")

    def post_attention_layernorm(self, _x: TensorHandle) -> TensorHandle:
        return self.session.detail(self._name("mlp"), "normalized_input")

    def gate_projection(self, _x: TensorHandle) -> TensorHandle:
        return self.session.detail(self._name("mlp"), "mlp_gate_pre_activation")

    def silu(self, _gate: TensorHandle) -> TensorHandle:
        return self.session.detail(self._name("mlp"), "mlp_gate_activation")

    def up_projection(self, _x: TensorHandle) -> TensorHandle:
        return self.session.detail(self._name("mlp"), "mlp_up_projection")

    def down_projection(self, _hidden: TensorHandle) -> TensorHandle:
        return self.session.detail(self._name("mlp"), "mlp_projection")


class RemoteModel:
    def __init__(self, model: str, *, architecture: str, revision: str, device: str, dtype: str):
        self.model, self.architecture = model, architecture
        self.revision, self.device, self.dtype = revision, device, dtype
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
        session = self._require_session()
        return session.output(session.final_norm_name)

    def vocabulary_projection(self, x: TensorHandle) -> TensorHandle:
        return self._require_session().vocabulary_projection(x)

    def top_tokens(
        self, logits: TensorHandle, *, k: int = 10
    ) -> tuple[TokenPrediction, ...]:
        return self._require_session().top_tokens(logits, k=k)

    def _require_session(self) -> "DebugSession":
        if self._session is None:
            raise RuntimeError("model operations must run inside with debug(...)")
        return self._session


class DebugSession:
    def __init__(self, model: RemoteModel, tokens: Prompt, token: int | None):
        self.model, self.prompt = model, tokens
        self.token = 0 if token is None else token
        self.execution: ModelExecution | None = None
        self._states: dict[str, Any] = {}

    def __enter__(self) -> "DebugSession":
        self.execution = prepare_model_execution(
            self.model.model, revision=self.model.revision, prompt=self.prompt.text,
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
        return len(self.execution.runtime.layers)

    @property
    def tokens(self) -> Iterator[int]:
        """Yield real token-position locals while selecting each runtime trace."""
        assert self.execution is not None
        count = int(self.execution.input_ids.shape[1])
        for position in range(count):
            self.token = position
            self._refresh()
            yield position

    def layer_kind(self, index: int) -> str:
        assert self.execution is not None
        return self.execution.runtime.layer_kind(index)

    def layer_name(self, index: int, kind: str) -> str:
        assert self.execution is not None
        return self.execution.runtime.state_name(index, kind)

    @property
    def final_norm_name(self) -> str:
        assert self.execution is not None
        return self.execution.runtime.final_norm_name

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
        assert self.execution is not None
        logits, _ = self.execution.vocabulary_projection(x._values)
        return TensorHandle(self, "vocabulary_projection", "logits", logits)

    def top_tokens(
        self, logits: TensorHandle, *, k: int = 10
    ) -> tuple[TokenPrediction, ...]:
        assert self.execution is not None
        _, predictions = self.execution.vocabulary_projection(
            self.output(self.final_norm_name)._values, count=k
        )
        return predictions


def gpt2(
    model: str,
    *,
    revision: str = "main",
    device: str = "cuda",
    dtype: str = "bfloat16",
) -> RemoteModel:
    return RemoteModel(model, architecture="gpt2", revision=revision, device=device, dtype=dtype)


def qwen35(
    model: str,
    *,
    revision: str = "main",
    device: str = "cuda",
    dtype: str = "bfloat16",
) -> RemoteModel:
    return RemoteModel(model, architecture="qwen3_5", revision=revision, device=device, dtype=dtype)


def debug(
    model: RemoteModel, tokens: Prompt, *, token: int | None = None
) -> DebugSession:
    return DebugSession(model, tokens, token)


def nop() -> None:
    """Do nothing while providing a stable source location for a breakpoint."""


def end() -> None:
    """Compatibility alias for the historical terminal no-op."""
    nop()
