"""Capture inspectable layer states from Hugging Face causal language models."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from threading import RLock
from typing import Any

from llmdbg.demo_model import Layer, LayerState


@dataclass(frozen=True)
class ModelTrace:
    """Captured model states plus the resolved prompt tokens."""

    states: tuple[LayerState, ...]
    tokens: tuple[str, ...]


@dataclass
class ModelRuntime:
    """A tokenizer and GPU-resident model reusable across debug sessions."""

    model: Any
    tokenizer: Any


@dataclass(frozen=True)
class TokenScore:
    """One vocabulary token and its score for a neuron output direction."""

    token_id: int
    text: str
    score: float


@dataclass(frozen=True)
class NeuronProjection:
    """An activated MLP neuron and its direct vocabulary projections."""

    index: int
    activation: float
    direction_norm: float
    vocabulary: tuple[TokenScore, ...]
    contribution: tuple[TokenScore, ...]


@dataclass
class ModelExecution:
    """A prompt execution that advances one token at a time with a KV cache."""

    runtime: ModelRuntime
    input_ids: Any
    tokens: tuple[str, ...]
    device: str
    progress: Callable[[str], None]
    past_key_values: Any = None
    cached_through: int = -1
    interventions: dict[tuple[int, str, str, int], float] = field(default_factory=dict)

    @property
    def token_count(self) -> int:
        return len(self.tokens)

    def trace_token(self, position: int) -> ModelTrace:
        """Prefill up to ``position`` and capture one token's layer trace."""
        if position < 0 or position >= self.token_count:
            raise ValueError(
                f"token position {position} is outside the prompt range 0..{self.token_count - 1}"
            )
        with _runtime_lock:
            if position <= self.cached_through:
                self.progress(f"Resetting KV cache before token {position}.\n")
                self.past_key_values = None
                self.cached_through = -1
            if position > self.cached_through + 1:
                self._prefill(position)
            trace, next_cache = _capture_token_trace(
                self.runtime,
                input_ids=self.input_ids[:, position : position + 1],
                past_key_values=self.past_key_values,
                total_length=position + 1,
                device=self.device,
                progress=self.progress,
                token_position=position,
                interventions=self.interventions,
            )
            self.past_key_values = next_cache
            self.cached_through = position
            return ModelTrace(trace.states, self.tokens)

    def set_intervention(
        self,
        token_position: int,
        layer_name: str,
        vector_name: str,
        index: int,
        value: float,
    ) -> None:
        """Persist one vector override for subsequent replays of a token."""
        self.interventions[(token_position, layer_name, vector_name, index)] = value

    def _prefill(self, position: int) -> None:
        """Run uncaptured prompt tokens into the cache before a target token."""
        import torch

        start = self.cached_through + 1
        prefix = self.input_ids[:, start:position].to(self.device)
        if prefix.shape[1] == 0:
            return
        self.progress(f"Prefilling tokens {start}..{position - 1} into the KV cache.\n")
        attention_mask = torch.ones(
            (1, position), dtype=torch.long, device=self.device
        )
        with torch.inference_mode():
            output = self.runtime.model(
                input_ids=prefix,
                attention_mask=attention_mask,
                past_key_values=self.past_key_values,
                use_cache=True,
            )
        self.past_key_values = output.past_key_values
        self.cached_through = position - 1

    def top_mlp_neurons(
        self,
        layer_name: str,
        activation: tuple[float, ...],
        *,
        neuron_count: int = 5,
        token_count: int = 5,
    ) -> tuple[NeuronProjection, ...]:
        """Project the most active MLP neurons directly into vocabulary space."""
        import re

        import torch

        match = re.fullmatch(r"transformer\.h\.(\d+)\.mlp", layer_name)
        if match is None:
            return ()
        block = self.runtime.model.transformer.h[int(match.group(1))]
        down_projection = block.mlp.c_proj.weight
        unembedding = self.runtime.model.get_output_embeddings().weight
        with _runtime_lock, torch.inference_mode():
            activation_tensor = torch.tensor(
                activation,
                dtype=down_projection.dtype,
                device=down_projection.device,
            )
            active_values, active_indices = torch.topk(
                activation_tensor, min(neuron_count, activation_tensor.numel())
            )
            directions = down_projection.index_select(0, active_indices)
            vocabulary_scores = (directions @ unembedding.T).float()
            contribution_scores = vocabulary_scores * active_values.float().unsqueeze(1)
            vocabulary_values, vocabulary_ids = torch.topk(
                vocabulary_scores, min(token_count, vocabulary_scores.shape[1]), dim=1
            )
            contribution_values, contribution_ids = torch.topk(
                contribution_scores, min(token_count, contribution_scores.shape[1]), dim=1
            )
            direction_norms = directions.float().norm(dim=1)

        def token_scores(ids: Any, values: Any) -> tuple[TokenScore, ...]:
            return tuple(
                TokenScore(
                    int(token_id),
                    self.runtime.tokenizer.decode([int(token_id)]),
                    float(score),
                )
                for token_id, score in zip(ids.tolist(), values.tolist(), strict=True)
            )

        return tuple(
            NeuronProjection(
                index=int(active_indices[rank]),
                activation=float(active_values[rank]),
                direction_norm=float(direction_norms[rank]),
                vocabulary=token_scores(vocabulary_ids[rank], vocabulary_values[rank]),
                contribution=token_scores(
                    contribution_ids[rank], contribution_values[rank]
                ),
            )
            for rank in range(active_indices.shape[0])
        )


_runtime_lock = RLock()
_runtime_key: tuple[str, str, str, str] | None = None
_runtime: ModelRuntime | None = None


def _get_model_runtime(
    model_name_or_path: str,
    *,
    revision: str,
    device: str,
    dtype: str,
    progress: Callable[[str], None],
) -> ModelRuntime:
    """Return the matching resident model, loading it only on a cache miss."""
    global _runtime, _runtime_key
    key = (model_name_or_path, revision, device, dtype)
    if _runtime is not None and _runtime_key == key:
        progress(f"Reusing resident {model_name_or_path} model tensors.\n")
        return _runtime

    import torch
    from gb10_load_llm import from_pretrained_to_cuda
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if _runtime is not None:
        progress("Model configuration changed; releasing the previous resident model.\n")
        _runtime = None
        _runtime_key = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    torch_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[dtype]
    progress(f"Downloading tokenizer for {model_name_or_path}...\n")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, revision=revision)
    progress(f"Loading {model_name_or_path} tensors through gb10-load-llm...\n")
    model = from_pretrained_to_cuda(
        AutoModelForCausalLM,
        model_name_or_path,
        revision=revision,
        dtype=torch_dtype,
        device=device,
        touch="auto",
    )
    model.eval()
    runtime = ModelRuntime(model=model, tokenizer=tokenizer)
    _runtime = runtime
    _runtime_key = key
    return runtime


def _tensor_from_output(output: Any):
    """Return the hidden-state tensor from a module output."""
    if isinstance(output, tuple):
        return output[0]
    return output


def _last_token_vector(tensor) -> tuple[float, ...]:
    """Copy the last-token hidden state to a plain CPU tuple."""
    return tuple(float(value) for value in tensor[0, -1].detach().float().cpu().tolist())


def _logit_lens(model, tokenizer, hidden, *, apply_final_norm: bool) -> str:
    """Describe an intermediate vector using its strongest next-token predictions."""
    import torch

    with torch.inference_mode():
        projected = model.transformer.ln_f(hidden) if apply_final_norm else hidden
        logits = model.lm_head(projected)[0, -1].float()
        values, indices = torch.topk(logits, k=5)
    predictions = []
    for value, index in zip(values.tolist(), indices.tolist(), strict=True):
        token = tokenizer.decode([index]).replace("\n", "\\n")
        predictions.append(f"{token!r} ({value:+.2f})")
    vector = hidden[0, -1].detach().float()
    return (
        f"Logit-lens next-token evidence: {', '.join(predictions)}. "
        f"Vector mean={vector.mean().item():+.4f}, "
        f"std={vector.std().item():.4f}, norm={vector.norm().item():.4f}."
    )


def load_model_trace(
    model_name_or_path: str,
    *,
    revision: str = "main",
    prompt: str,
    device: str = "cuda",
    dtype: str = "bfloat16",
    progress: Callable[[str], None] | None = None,
) -> ModelTrace:
    """Load GPT-2 and return a layer-by-layer trace for one prompt.

    Model tensors are loaded through :mod:`gb10_load_llm`, as required for
    efficient cold-page transfers on GB10/GX10 systems.
    """
    execution = prepare_model_execution(
        model_name_or_path,
        revision=revision,
        prompt=prompt,
        device=device,
        dtype=dtype,
        progress=progress,
    )
    return execution.trace_token(execution.token_count - 1)


def prepare_model_execution(
    model_name_or_path: str,
    *,
    revision: str = "main",
    prompt: str,
    device: str = "cuda",
    dtype: str = "bfloat16",
    progress: Callable[[str], None] | None = None,
) -> ModelExecution:
    """Load/reuse a model and tokenize a prompt without running its forward pass."""
    try:
        import torch
        import gb10_load_llm  # noqa: F401 - validates the private runtime
        import transformers  # noqa: F401 - validates the private runtime
    except ImportError as error:
        raise RuntimeError(
            "The private llmdbg environment is missing model dependencies. "
            "Reinstall the VS Code extension or install gb10-load-llm into its runtime."
        ) from error

    if not prompt:
        raise ValueError("prompt must not be empty")
    if dtype not in {"bfloat16", "float16", "float32"}:
        raise ValueError(f"unsupported dtype: {dtype}")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false")

    report = progress or (lambda _message: None)
    with _runtime_lock:
        runtime = _get_model_runtime(
            model_name_or_path,
            revision=revision,
            device=device,
            dtype=dtype,
            progress=report,
        )
        if getattr(runtime.model.config, "model_type", None) != "gpt2":
            raise ValueError(
                "the prototype currently supports GPT-2 models, "
                f"got {runtime.model.config.model_type!r}"
            )
        encoded = runtime.tokenizer(prompt, return_tensors="pt")
        input_ids = encoded["input_ids"]
        tokens = tuple(
            runtime.tokenizer.decode([token_id]) for token_id in input_ids[0].tolist()
        )
    report(f"Prepared {len(tokens)} prompt tokens; forward execution is deferred.\n")
    return ModelExecution(runtime, input_ids, tokens, device, report)


def _capture_token_trace(
    runtime: ModelRuntime,
    *,
    input_ids: Any,
    past_key_values: Any,
    total_length: int,
    device: str,
    progress: Callable[[str], None],
    token_position: int,
    interventions: dict[tuple[int, str, str, int], float],
) -> tuple[ModelTrace, Any]:
    """Capture all layer states for one token using an existing KV cache."""
    import torch

    model = runtime.model
    tokenizer = runtime.tokenizer
    captured: list[tuple[str, str, str, Any, bool]] = []
    detailed: dict[str, dict[str, Any]] = {}
    handles = []

    def apply_interventions(tensor: Any, state_name: str, vector_name: str) -> Any:
        for (position, layer, vector, index), value in interventions.items():
            if position == token_position and layer == state_name and vector == vector_name:
                if index < 0 or index >= tensor.shape[-1]:
                    raise ValueError(
                        f"dimension {index} is outside {state_name}.{vector_name} "
                        f"range 0..{tensor.shape[-1] - 1}"
                    )
                tensor[..., index] = value
        return tensor

    def capture(name: str, operation: str, meaning: str, apply_final_norm: bool):
        def hook(_module, _inputs, output) -> None:
            tensor = _tensor_from_output(output)
            apply_interventions(tensor, name, "output")
            captured.append((name, operation, meaning, tensor.detach(), apply_final_norm))

        return hook

    def capture_input(name: str, operation: str, meaning: str):
        def hook(_module, inputs) -> None:
            hidden = inputs[0]
            apply_interventions(hidden, name, "output")
            captured.append((name, operation, meaning, hidden.detach(), True))

        return hook

    def capture_detail(state_name: str, detail_name: str):
        def hook(_module, _inputs, output) -> None:
            tensor = _tensor_from_output(output)
            apply_interventions(tensor, state_name, detail_name)
            detailed.setdefault(state_name, {})[detail_name] = tensor.detach()

        return hook

    handles.append(
        model.transformer.drop.register_forward_hook(
            capture(
                "token_embedding",
                "token + positional embedding",
                "Encodes the prompt tokens and their positions into the residual stream.",
                True,
            )
        )
    )
    for index, block in enumerate(model.transformer.h):
        attention_name = f"transformer.h.{index}.attention"
        mlp_name = f"transformer.h.{index}.mlp"
        handles.append(
            block.ln_1.register_forward_hook(
                capture_detail(attention_name, "normalized_input")
            )
        )
        handles.append(
            block.ln_2.register_forward_pre_hook(
                capture_input(
                    attention_name,
                    "causal self-attention + residual",
                    f"Block {index} has mixed information across prompt positions.",
                )
            )
        )
        handles.append(
            block.ln_2.register_forward_hook(
                capture_detail(mlp_name, "normalized_input")
            )
        )
        handles.append(
            block.mlp.c_fc.register_forward_hook(
                capture_detail(mlp_name, "mlp_expansion_pre_activation")
            )
        )
        if hasattr(block.mlp.act, "register_forward_hook"):
            handles.append(
                block.mlp.act.register_forward_hook(
                    capture_detail(mlp_name, "mlp_expansion")
                )
            )
        handles.append(
            block.mlp.c_proj.register_forward_hook(
                capture_detail(mlp_name, "mlp_projection")
            )
        )
        handles.append(
            block.register_forward_hook(
                capture(
                    mlp_name,
                    "feed-forward MLP + residual",
                    f"Block {index} has applied its learned feature transformation.",
                    True,
                )
            )
        )
    handles.append(
        model.transformer.ln_f.register_forward_hook(
            capture(
                "transformer.ln_f",
                "final layer normalization",
                "Normalizes the final residual stream before vocabulary projection.",
                False,
            )
        )
    )
    handles.append(
        model.transformer.ln_f.register_forward_hook(
            capture_detail("transformer.ln_f", "normalized_output")
        )
    )

    try:
        attention_mask = torch.ones(
            (1, total_length), dtype=torch.long, device=device
        )
        progress(f"Running token {token_position} as a one-token cached forward pass.\n")
        with torch.inference_mode():
            model_output = model(
                input_ids=input_ids.to(device),
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
    finally:
        for handle in handles:
            handle.remove()

    states: list[LayerState] = []
    previous: tuple[float, ...] | None = None
    for name, operation, meaning, hidden, apply_final_norm in captured:
        output = _last_token_vector(hidden)
        interpretation = (
            f"{meaning} "
            f"{_logit_lens(model, tokenizer, hidden, apply_final_norm=apply_final_norm)}"
        )
        states.append(
            LayerState(
                Layer(name, operation, meaning, 1.0, 0.0),
                previous if previous is not None else output,
                output,
                interpretation,
                {
                    detail_name: _last_token_vector(detail_tensor)
                    for detail_name, detail_tensor in detailed.get(name, {}).items()
                },
            )
        )
        previous = output

    progress(f"Captured {len(states)} layer states for token {token_position}.\n")
    return ModelTrace(tuple(states), ()), model_output.past_key_values
