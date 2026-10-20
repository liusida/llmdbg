"""Capture inspectable layer states from Hugging Face causal language models."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from threading import RLock
from typing import Any

from llmdbg.demo_model import Layer, LayerState


@dataclass(frozen=True)
class ModelTrace:
    """Captured model states plus the resolved prompt tokens."""

    states: tuple[LayerState, ...]
    tokens: tuple[str, ...]


@dataclass(frozen=True)
class ModelRuntime:
    """A tokenizer and GPU-resident model reusable across debug sessions."""

    model: Any
    tokenizer: Any


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

    # Forward hooks mutate model state temporarily, so loading and tracing share
    # one lock even if multiple editor windows connect to the host concurrently.
    with _runtime_lock:
        runtime = _get_model_runtime(
            model_name_or_path,
            revision=revision,
            device=device,
            dtype=dtype,
            progress=report,
        )
        return _capture_trace(runtime, prompt=prompt, device=device, progress=report)


def _capture_trace(
    runtime: ModelRuntime,
    *,
    prompt: str,
    device: str,
    progress: Callable[[str], None],
) -> ModelTrace:
    """Run one prompt through an already resident GPT-2 model."""
    import torch

    model = runtime.model
    tokenizer = runtime.tokenizer
    if getattr(model.config, "model_type", None) != "gpt2":
        raise ValueError(
            f"the prototype currently supports GPT-2 models, got {model.config.model_type!r}"
        )

    captured: list[tuple[str, str, str, Any, bool]] = []
    handles = []

    def capture(name: str, operation: str, meaning: str, apply_final_norm: bool):
        def hook(_module, _inputs, output) -> None:
            tensor = _tensor_from_output(output)
            captured.append((name, operation, meaning, tensor.detach(), apply_final_norm))

        return hook

    def capture_input(name: str, operation: str, meaning: str):
        def hook(_module, inputs) -> None:
            hidden = inputs[0]
            captured.append((name, operation, meaning, hidden.detach(), True))

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
        handles.append(
            block.ln_2.register_forward_pre_hook(
                capture_input(
                    f"transformer.h.{index}.attention",
                    "causal self-attention + residual",
                    f"Block {index} has mixed information across prompt positions.",
                )
            )
        )
        handles.append(
            block.register_forward_hook(
                capture(
                    f"transformer.h.{index}.mlp",
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

    try:
        inputs = tokenizer(prompt, return_tensors="pt")
        tokens = tuple(
            tokenizer.decode([token_id]) for token_id in inputs["input_ids"][0].tolist()
        )
        inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
        progress(f"Running forward pass for {len(inputs['input_ids'][0])} prompt tokens...\n")
        with torch.inference_mode():
            model(**inputs)
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
            )
        )
        previous = output

    progress(f"Captured {len(states)} GPT-2 layer states.\n")
    return ModelTrace(tuple(states), tokens)
