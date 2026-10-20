#!/usr/bin/env python3
"""Measure per-layer median token p99 post-SwiGLU activation scales."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

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
    args = parser.parse_args()
    if not 0.5 < args.quantile < 1.0:
        parser.error("--quantile must be between 0.5 and 1")
    return args


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
    encoded = tokenizer(full_prompt, add_special_tokens=False, return_tensors="pt")
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
    layers = model.model.layers
    measurements: dict[int, dict[str, object]] = {}

    def make_hook(layer_index: int):
        def hook(_module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
            values = inputs[0][0].detach().float().cpu()
            values = values[len(prefix_ids) :]
            magnitudes = values.abs()
            token_p99 = torch.quantile(magnitudes, args.quantile, dim=1)
            nearest = (magnitudes - token_p99[:, None]).abs().argmin(dim=1)
            signed_references = values.gather(1, nearest[:, None]).squeeze(1)
            order = torch.argsort(token_p99)
            median_rank = len(order) // 2
            median_token_offset = int(order[median_rank])
            scale = float(token_p99[median_token_offset])
            signed_reference = float(signed_references[median_token_offset])
            measurements[layer_index] = {
                "layer": layer_index,
                "scale": scale,
                "side": "positive" if signed_reference >= 0 else "negative",
                "signed_reference": signed_reference,
                "captured_token_position": median_token_offset + len(prefix_ids),
                "analysis_token_position": median_token_offset,
                "nearest_neuron": int(nearest[median_token_offset]),
            }
            print(
                f"layer {layer_index:>2}: scale={scale:.8f} "
                f"side={measurements[layer_index]['side']}",
                flush=True,
            )

        return hook

    handles = [
        layer.mlp.down_proj.register_forward_pre_hook(make_hook(index))
        for index, layer in enumerate(layers)
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
    payload = {
        "schema_version": 1,
        "model_id": args.model,
        "model_revision": getattr(config, "_commit_hash", None) or args.revision,
        "activation": "input to mlp.down_proj (silu(gate_proj(x)) * up_proj(x))",
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
        "intermediate_size": int(layers[0].mlp.down_proj.in_features),
        "touched_model_bytes": touched,
        "layers": [measurements[index] for index in range(len(layers))],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
