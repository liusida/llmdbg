#!/usr/bin/env python3
"""Measure per-layer median token-p99 MLP activation scales."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from gb10_load_llm import load_model_to_cuda
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--prefix", default="Context:\n")
    parser.add_argument("--quantile", type=float, default=0.99)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--plot-output",
        type=Path,
        help="default: <output stem>.png",
    )
    args = parser.parse_args()
    if not 0.5 < args.quantile < 1.0:
        parser.error("--quantile must be between 0.5 and 1")
    return args


def plot_token_scales(
    *, model_id: str, token_scales: dict[int, list[float]], output: Path
) -> None:
    """Render token p99 values as points plus percentile envelopes."""
    layers = np.asarray(sorted(token_scales), dtype=float)
    matrix = np.asarray([token_scales[int(layer)] for layer in layers], dtype=float)
    if matrix.ndim != 2 or np.any(matrix <= 0) or not np.all(np.isfinite(matrix)):
        raise ValueError("token scale matrix must be finite, positive, and rectangular")

    p01, p10, median, p90, p99 = np.quantile(
        matrix, [0.01, 0.10, 0.50, 0.90, 0.99], axis=1
    )
    fig, ax = plt.subplots(figsize=(11.5, 6.2), constrained_layout=True)
    ax.fill_between(
        layers,
        p01,
        p99,
        color="#78a9c2",
        alpha=0.18,
        label="tokens p1–p99",
        zorder=1,
    )
    ax.fill_between(
        layers,
        p10,
        p90,
        color="#32799a",
        alpha=0.24,
        label="tokens p10–p90",
        zorder=2,
    )
    token_offsets = np.linspace(-0.32, 0.32, matrix.shape[1])
    for row, layer in enumerate(layers):
        ax.scatter(
            layer + token_offsets,
            matrix[row],
            s=5,
            color="#245f7a",
            alpha=0.10,
            linewidths=0,
            rasterized=True,
            zorder=3,
        )
    ax.plot(
        layers,
        median,
        color="#c4522d",
        linewidth=1.5,
        alpha=0.9,
        zorder=4,
    )
    ax.scatter(
        layers,
        median,
        marker="D",
        s=30,
        color="#c4522d",
        edgecolors="white",
        linewidths=0.5,
        label="token median (packaged scale)",
        zorder=4,
    )
    ax.set_yscale("log")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Token p99 of |MLP hidden activation| (log scale)")
    ax.set_title(f"{model_id} · MLP activation-scale calibration")
    ax.grid(axis="y", which="both", alpha=0.18)
    ax.legend(loc="upper left", frameon=False)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    prompt = args.prompt_file.read_text(encoding="utf-8").rstrip("\r\n")
    if not prompt:
        raise ValueError("prompt file must not be empty")

    config = AutoConfig.from_pretrained(args.model, revision=args.revision)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    full_prompt = args.prefix + prompt
    max_length = int(
        getattr(config, "max_position_embeddings", tokenizer.model_max_length)
    )
    encoded = tokenizer(
        full_prompt,
        add_special_tokens=False,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    untruncated_token_count = len(
        tokenizer(full_prompt, add_special_tokens=False, verbose=False).input_ids
    )
    prefix_ids = tokenizer(args.prefix, add_special_tokens=False).input_ids
    if encoded.input_ids[0, : len(prefix_ids)].tolist() != prefix_ids:
        raise RuntimeError("prefix tokenization is not a prefix of the full prompt")

    print(f"Loading {args.model} through gb10-load-llm...", flush=True)
    model, touched = load_model_to_cuda(
        AutoModelForCausalLM,
        args.model,
        config=config,
        revision=args.revision,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        touch="auto",
        return_touched=True,
    )
    model.eval()
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        architecture = "gpt2"
        layers = model.transformer.h
        hook_modules = [layer.mlp.c_proj for layer in layers]
        activation_definition = "input to mlp.c_proj (gelu(c_fc(x)))"
    else:
        architecture = "qwen35"
        layers = model.model.layers
        hook_modules = [layer.mlp.down_proj for layer in layers]
        activation_definition = (
            "input to mlp.down_proj (silu(gate_proj(x)) * up_proj(x))"
        )
    measurements: dict[int, dict[str, object]] = {}
    token_scales: dict[int, list[float]] = {}
    intermediate_sizes: set[int] = set()

    def make_hook(layer_index: int):
        def hook(_module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
            values = inputs[0][0].detach().float().cpu()
            intermediate_sizes.add(int(values.shape[-1]))
            values = values[len(prefix_ids) :]
            magnitudes = values.abs()
            token_p99 = torch.quantile(magnitudes, args.quantile, dim=1)
            token_scales[layer_index] = token_p99.tolist()
            nearest = (magnitudes - token_p99[:, None]).abs().argmin(dim=1)
            signed_references = values.gather(1, nearest[:, None]).squeeze(1)
            median = torch.quantile(token_p99, 0.50)
            median_token_offset = int((token_p99 - median).abs().argmin())
            scale = float(median)
            signed_reference = float(signed_references[median_token_offset])
            measurements[layer_index] = {
                "layer": layer_index,
                "scale": scale,
                "side": "positive" if signed_reference >= 0 else "negative",
                "signed_reference": signed_reference,
                "captured_token_position": median_token_offset + len(prefix_ids),
                "analysis_token_position": median_token_offset,
                "nearest_neuron": int(nearest[median_token_offset]),
                "token_scale_percentiles": {
                    f"p{int(percentile * 100):02d}": float(
                        torch.quantile(token_p99, percentile)
                    )
                    for percentile in (0.01, 0.10, 0.50, 0.90, 0.99)
                },
            }
            print(
                f"layer {layer_index:>2}: scale={scale:.8f} "
                f"side={measurements[layer_index]['side']}",
                flush=True,
            )

        return hook

    handles = [
        module.register_forward_pre_hook(make_hook(index))
        for index, module in enumerate(hook_modules)
    ]
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

    if len(measurements) != len(layers):
        raise RuntimeError(f"captured {len(measurements)} of {len(layers)} layers")
    if len(intermediate_sizes) != 1:
        raise RuntimeError(f"captured inconsistent MLP widths: {intermediate_sizes}")
    payload = {
        "schema_version": 1,
        "model_id": args.model,
        "model_revision": getattr(config, "_commit_hash", None) or args.revision,
        "architecture": architecture,
        "activation": activation_definition,
        "statistic": "median across analyzed tokens of token p99(abs(activation))",
        "quantile": args.quantile,
        "side_definition": (
            "sign of the actual neuron activation nearest the p99 magnitude at "
            "the token supplying the layer median"
        ),
        "prompt_file": args.prompt_file.name,
        "prompt_sha256": hashlib.sha256(
            args.prompt_file.read_bytes()
        ).hexdigest(),
        "prefix": args.prefix,
        "excluded_prefix_tokens": len(prefix_ids),
        "analyzed_tokens": encoded.input_ids.shape[1] - len(prefix_ids),
        "captured_tokens": encoded.input_ids.shape[1],
        "untruncated_tokens": untruncated_token_count,
        "truncated_to_model_context": untruncated_token_count > encoded.input_ids.shape[1],
        "intermediate_size": intermediate_sizes.pop(),
        "touched_model_bytes": touched,
        "layers": [measurements[index] for index in range(len(layers))],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    plot_output = args.plot_output or args.output.with_suffix(".png")
    plot_token_scales(
        model_id=args.model,
        token_scales=token_scales,
        output=plot_output,
    )
    print(f"Wrote {args.output.resolve()}")
    print(f"Wrote {plot_output.resolve()}")


if __name__ == "__main__":
    main()
