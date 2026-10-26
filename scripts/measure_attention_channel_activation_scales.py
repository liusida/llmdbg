#!/usr/bin/env python3
"""Measure attention-channel scales and per-head OV-projected norm references."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3.5-2B")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--prefix", default="Context:\n")
    parser.add_argument("--quantile", type=float, default=0.99)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plot-output", type=Path)
    args = parser.parse_args()
    if not 0.5 < args.quantile < 1.0:
        parser.error("--quantile must be between 0.5 and 1")
    return args


def plot_token_scales(
    *, model_id: str, token_scales: dict[int, list[float]], output: Path
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    layers = np.asarray(sorted(token_scales), dtype=float)
    matrix = np.asarray([token_scales[int(layer)] for layer in layers], dtype=float)
    if matrix.ndim != 2 or np.any(matrix <= 0) or not np.all(np.isfinite(matrix)):
        raise ValueError("token scale matrix must be finite, positive, and rectangular")
    p01, p10, median, p90, p99 = np.quantile(
        matrix, [0.01, 0.10, 0.50, 0.90, 0.99], axis=1
    )
    fig, ax = plt.subplots(figsize=(11.5, 6.2), constrained_layout=True)
    ax.fill_between(layers, p01, p99, color="#8b79c6", alpha=0.16, label="tokens p1–p99")
    ax.fill_between(layers, p10, p90, color="#694dae", alpha=0.24, label="tokens p10–p90")
    offsets = np.linspace(-0.32, 0.32, matrix.shape[1])
    for row, layer in enumerate(layers):
        ax.scatter(
            layer + offsets, matrix[row], s=5, color="#51358f", alpha=0.10,
            linewidths=0, rasterized=True,
        )
    ax.plot(layers, median, color="#d0643b", linewidth=1.5, alpha=0.9)
    ax.scatter(
        layers, median, marker="D", s=30, color="#d0643b", edgecolors="white",
        linewidths=0.5, label="token median (packaged scale)", zorder=4,
    )
    ax.set_yscale("log")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Token p99 of |activation| × |output direction| (log scale)")
    ax.set_title(f"{model_id} · attention-channel contribution calibration")
    ax.grid(axis="y", which="both", alpha=0.18)
    ax.legend(loc="upper left", frameon=False)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)


def projected_head_references(
    values: torch.Tensor, projection: torch.Tensor, head_count: int,
) -> list[dict[str, object]]:
    """Measure [tokens, channels] @ [channels, residual] per head, without bias.

    Accumulate in fp32, retaining only norms; chunking bounds intermediate memory.
    Inputs may be mixed outputs or unweighted values; callers keep references separate.
    """
    if (values.ndim != 2 or projection.ndim != 2 or head_count <= 0
            or values.shape[0] == 0 or values.shape[1] != projection.shape[0]
            or values.shape[1] % head_count):
        raise ValueError("Expected nonempty token/channel inputs and a compatible head layout")
    width = values.shape[1] // head_count
    records = []
    for head in range(head_count):
        start, stop = head * width, (head + 1) * width
        weight = projection[start:stop].detach().float()
        norms = torch.cat([
            (chunk[:, start:stop].detach().to(device=weight.device, dtype=torch.float32) @ weight).norm(dim=-1).cpu()
            for chunk in values.split(64)
        ])
        if not bool(torch.isfinite(norms).all()):
            raise ValueError(f"Non-finite projected norms for head {head}")
        percentiles = {f"p{int(p * 100):02d}": float(torch.quantile(norms, p))
                       for p in (0.01, 0.10, 0.50, 0.90, 0.99)}
        records.append({"head": head, "head_dimension": width,
                        "scale": percentiles["p50"], "analyzed_tokens": len(norms),
                        "token_norm_percentiles": percentiles})
    return records


def main() -> None:
    args = parse_args()
    from gb10_load_llm import load_model_to_cuda
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForMultimodalLM, AutoTokenizer
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    prompt_bytes = args.prompt_file.read_bytes()
    prompt = prompt_bytes.decode("utf-8").rstrip("\r\n")
    if not prompt:
        raise ValueError("prompt file must not be empty")

    config = AutoConfig.from_pretrained(args.model, revision=args.revision)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    full_prompt = args.prefix + prompt
    max_length = int(getattr(getattr(config, "text_config", config), "max_position_embeddings", tokenizer.model_max_length))
    encoded = tokenizer(
        full_prompt, add_special_tokens=False, return_tensors="pt", truncation=True,
        max_length=max_length,
    )
    untruncated_tokens = len(
        tokenizer(full_prompt, add_special_tokens=False, verbose=False).input_ids
    )
    prefix_ids = tokenizer(args.prefix, add_special_tokens=False).input_ids
    if encoded.input_ids.shape[1] <= len(prefix_ids):
        raise ValueError("No analysis tokens remain after excluding the prefix")

    print(f"Loading {args.model} through gb10-load-llm...", flush=True)
    model, touched = load_model_to_cuda(
        AutoModelForMultimodalLM if getattr(config, "vision_config", None) is not None else AutoModelForCausalLM,
        args.model,
        config=config,
        revision=args.revision,
        dtype=torch.bfloat16,
        attn_implementation="eager" if config.model_type == "gemma4" else "sdpa",
        touch="auto",
        return_touched=True,
    )
    model.eval()
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        architecture = "gpt2"
        layers = model.transformer.h
        modules_and_kinds = [(block.attn.c_proj, "attention") for block in layers]
    else:
        architecture = config.model_type
        layers = getattr(model.model, "language_model", model.model).layers
        text_config = getattr(config, "text_config", config)
        layer_types = tuple(getattr(text_config, "layer_types", ["full_attention"] * len(layers)))
        modules_and_kinds = []
        for block, raw_kind in zip(layers, layer_types, strict=True):
            if raw_kind == "linear_attention":
                modules_and_kinds.append(
                    (block.linear_attn.out_proj, "gated_linear_attention")
                )
            else:
                modules_and_kinds.append((block.self_attn.o_proj, "full_attention"))
    measurements: dict[int, dict[str, object]] = {}
    token_scales: dict[int, list[float]] = {}
    channel_sizes: set[int] = set()
    direction_norms = [
        torch.linalg.vector_norm(
            module.weight.detach(),
            ord=2,
            dim=1 if architecture == "gpt2" else 0,
            dtype=torch.float32,
        ).cpu()
        for module, _kind in modules_and_kinds
    ]

    def make_hook(layer_index: int, attention_kind: str):
        def hook(_module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
            captured = inputs[0][0].detach()[len(prefix_ids) :]
            values = captured.float().cpu()
            channel_sizes.add(int(values.shape[-1]))
            norms = direction_norms[layer_index]
            if norms.shape[0] != values.shape[-1]:
                raise RuntimeError(
                    f"layer {layer_index} has {values.shape[-1]} channels but "
                    f"{norms.shape[0]} output directions"
                )
            contributions = values * norms
            magnitudes = contributions.abs()
            per_token = torch.quantile(magnitudes, args.quantile, dim=1)
            token_scales[layer_index] = per_token.tolist()
            nearest = (magnitudes - per_token[:, None]).abs().argmin(dim=1)
            signed = contributions.gather(1, nearest[:, None]).squeeze(1)
            median = torch.quantile(per_token, 0.50)
            token_offset = int((per_token - median).abs().argmin())
            signed_reference = float(signed[token_offset])
            measurements[layer_index] = {
                "layer": layer_index,
                "attention_kind": attention_kind,
                "scale": float(median),
                "side": "positive" if signed_reference >= 0 else "negative",
                "signed_reference": signed_reference,
                "captured_token_position": token_offset + len(prefix_ids),
                "analysis_token_position": token_offset,
                "nearest_channel": int(nearest[token_offset]),
                "token_scale_percentiles": {
                    f"p{int(percentile * 100):02d}": float(
                        torch.quantile(per_token, percentile)
                    )
                    for percentile in (0.01, 0.10, 0.50, 0.90, 0.99)
                },
            }
            text_config = getattr(config, "text_config", config)
            head_count = int(text_config.linear_num_value_heads if attention_kind == "gated_linear_attention"
                             else text_config.n_head if architecture == "gpt2" else text_config.num_attention_heads)
            projection = _module.weight.detach()
            if architecture != "gpt2":
                projection = projection.T
            measurements[layer_index]["head_norm_scales"] = projected_head_references(
                captured, projection, head_count,
            )
            print(
                f"layer {layer_index:>2} ({attention_kind}): "
                f"scale={float(median):.8f} side={measurements[layer_index]['side']}",
                flush=True,
            )

        return hook

    handles = []
    value_measurements = {}
    active_delta = [None]
    patched_kernels = []
    def measure_values(index, tensor):
        kind = modules_and_kinds[index][1]
        cfg = getattr(config, 'text_config', config)
        heads = int(cfg.linear_num_value_heads if kind == 'gated_linear_attention'
                    else cfg.n_head if architecture == 'gpt2' else cfg.num_attention_heads)
        value_heads = heads if kind == 'gated_linear_attention' or architecture == 'gpt2' else int(cfg.num_key_value_heads)
        if heads % value_heads:
            raise ValueError('Unsupported grouped-query value layout')
        values = tensor.detach().reshape(tensor.shape[0], tensor.shape[1], value_heads, -1)[0, len(prefix_ids):]
        values = values.repeat_interleave(heads // value_heads, dim=1).flatten(1)
        projection = modules_and_kinds[index][0].weight.detach()
        if architecture != 'gpt2': projection = projection.T
        value_measurements[index] = projected_head_references(values, projection, heads)
    def capture_values(index, fused=False):
        def hook(_module, _inputs, output):
            measure_values(index, output.chunk(3,dim=-1)[2] if fused else output)
        return hook
    def enter_delta(index):
        def hook(_module, _inputs): active_delta[0] = index
        return hook
    def delta_values_kernel(original):
        def wrapped(query, key, value, *positional, **kwargs):
            if active_delta[0] is not None:
                measure_values(active_delta[0], value)
            return original(query,key,value,*positional,**kwargs)
        return wrapped
    for index, (module, kind) in enumerate(modules_and_kinds):
        handles.append(module.register_forward_pre_hook(make_hook(index, kind)))
        if kind == 'gated_linear_attention':
            mixer = layers[index].linear_attn
            handles.append(mixer.register_forward_pre_hook(enter_delta(index)))
            if not patched_kernels:
                import importlib
                implementation = importlib.import_module(type(mixer).__module__)
                for name in ('torch_chunk_gated_delta_rule','torch_recurrent_gated_delta_rule'):
                    original = getattr(implementation,name)
                    patched_kernels.append((implementation,name,original))
                    setattr(implementation,name,delta_values_kernel(original))
        elif architecture == 'gpt2':
            handles.append(layers[index].attn.c_attn.register_forward_hook(capture_values(index,True)))
        else:
            mixer = layers[index].self_attn
            value_module = getattr(mixer,'v_norm',None) if architecture == 'gemma4' else getattr(mixer,'v_proj',None)
            if value_module is not None:
                handles.append(value_module.register_forward_hook(capture_values(index)))
    try:
        device = next(model.parameters()).device
        with torch.inference_mode():
            model(
                input_ids=encoded.input_ids.to(device),
                attention_mask=encoded.attention_mask.to(device),
                use_cache=False,
                return_dict=False,
            )
    finally:
        for handle in handles:
            handle.remove()
        for implementation,name,original in reversed(patched_kernels):
            setattr(implementation,name,original)

    if len(measurements) != len(layers):
        raise RuntimeError(f"captured {len(measurements)} of {len(layers)} layers")
    for index, references in value_measurements.items():
        measurements[index]['value_head_norm_scales'] = references
    if architecture == 'qwen3_5' and len(value_measurements) != len(layers):
        raise RuntimeError(f'Captured value references for only {len(value_measurements)} of {len(layers)} layers')
    payload = {
        "schema_version": 2,
        "value_head_norm_calibration": {
            "version": 1,
            "quantity": "l2_norm(unweighted_value_head @ W_o_head)",
            "statistic": "per-layer, per-output-head median across analyzed tokens",
            "delta_value_site": "after causal convolution and activation; before memory update",
            "full_attention_value_site": "projected value (normalized value for Gemma4), before attention weighting",
            "grouped_query": "shared KV values projected separately through each query head output map",
            "input_modality": "text-only reference corpus",
            "dtype": "float32 projection and norm from captured model-dtype values",
            "bias_included": False,
            "output_gate_or_norm_included": False,
            "calibrated_layers": sorted(value_measurements),
            "score": "current projected value norm / this head's value scale",
        },
        "head_norm_calibration": {
            "version": 1,
            "quantity": "l2_norm(mixed_head @ W_o_head)",
            "statistic": "per-layer, per-output-head median across analyzed tokens (query heads for full attention; value heads for DeltaNet)",
            "dtype": "float32 projection and norm from captured model-dtype activations",
            "model_dtype": "bfloat16",
            "bias_included": False,
            "post_projection_normalization_included": False,
            "delta_net_included": any(kind == "gated_linear_attention" for _, kind in modules_and_kinds),
            "input_modality": "text-only reference corpus",
            "score": "current projected head norm / scale",
        },
        "model_id": args.model,
        "model_revision": getattr(config, "_commit_hash", None) or args.revision,
        "architecture": architecture,
        "activation": "input channels to attention o_proj/out_proj",
        "quantity": "activation * l2_norm(output-projection direction)",
        "statistic": (
            "median across analyzed tokens of token p99(abs(activation) * "
            "l2_norm(output-projection direction))"
        ),
        "quantile": args.quantile,
        "side_definition": (
            "sign of the activation-weighted contribution magnitude nearest p99 at "
            "the token supplying the layer median"
        ),
        "prompt_file": args.prompt_file.name,
        "prompt_sha256": hashlib.sha256(prompt_bytes).hexdigest(),
        "prefix": args.prefix,
        "excluded_prefix_tokens": len(prefix_ids),
        "analyzed_tokens": encoded.input_ids.shape[1] - len(prefix_ids),
        "captured_tokens": encoded.input_ids.shape[1],
        "untruncated_tokens": untruncated_tokens,
        "truncated_to_model_context": untruncated_tokens > encoded.input_ids.shape[1],
        "channel_size": next(iter(channel_sizes)) if len(channel_sizes) == 1 else None,
        "channel_sizes": sorted(channel_sizes),
        "touched_model_bytes": touched,
        "layers": [measurements[index] for index in range(len(layers))],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    plot_output = args.plot_output or args.output.with_suffix(".png")
    plot_token_scales(model_id=args.model, token_scales=token_scales, output=plot_output)
    print(f"Wrote {args.output.resolve()}")
    print(f"Wrote {plot_output.resolve()}")


if __name__ == "__main__":
    main()
