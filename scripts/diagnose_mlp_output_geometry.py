#!/usr/bin/env python3
"""Measure how individual MLP neuron vectors compose the full MLP output."""

from __future__ import annotations

import argparse
import math

import torch
from gb10_load_llm import load_model_to_cuda
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3.5-2B")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--token", type=int, default=1)
    parser.add_argument("--layer", type=int, default=1)
    parser.add_argument("--top", type=int, default=10)
    return parser.parse_args()


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    denominator = left.norm() * right.norm()
    if float(denominator) == 0.0:
        return float("nan")
    return float(torch.dot(left, right) / denominator)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this diagnostic")

    config = AutoConfig.from_pretrained(args.model, revision=args.revision)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    encoded = tokenizer(args.prompt, add_special_tokens=False, return_tensors="pt")
    token_texts = [
        tokenizer.decode([int(token_id)]) for token_id in encoded.input_ids[0]
    ]
    if not 0 <= args.token < len(token_texts):
        raise ValueError(f"--token must be in 0..{len(token_texts) - 1}")

    print(f"prompt tokens: {list(enumerate(token_texts))}", flush=True)
    print(f"Loading {args.model} through gb10-load-llm...", flush=True)
    model = load_model_to_cuda(
        AutoModelForCausalLM,
        args.model,
        config=config,
        revision=args.revision,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        touch="auto",
    )
    model.eval()
    down_projection = model.model.layers[args.layer].mlp.down_proj
    captured: dict[str, torch.Tensor] = {}

    def capture(
        _module: torch.nn.Module,
        inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        captured["activations"] = inputs[0].detach()
        captured["output"] = output.detach()

    handle = down_projection.register_forward_hook(capture)
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
        handle.remove()

    activations = captured["activations"][0, args.token].float()
    output = captured["output"][0, args.token].float()
    directions = down_projection.weight.detach().T.float()
    contributions = activations[:, None] * directions
    magnitudes = contributions.norm(dim=1)
    count = min(args.top, contributions.shape[0])
    top_magnitudes, top_indices = torch.topk(magnitudes, count)
    output_norm = output.norm()
    output_unit = output / output_norm
    reconstructed = contributions.sum(dim=0)
    bias = getattr(down_projection, "bias", None)
    if bias is not None:
        reconstructed = reconstructed + bias.detach().float()

    print(f"source token: {args.token} {token_texts[args.token]!r}")
    print(f"layer: {args.layer}")
    print(f"neurons: {contributions.shape[0]}")
    print(f"|MLP output|: {float(output_norm):.8f}")
    print(f"|sum of all neuron vectors|: {float(reconstructed.norm()):.8f}")
    print(
        "reconstruction max absolute error: "
        f"{float((reconstructed - output).abs().max()):.3e}"
    )
    print("top neurons ranked by |a_i d_i|:")
    for rank, (index, magnitude) in enumerate(
        zip(top_indices.tolist(), top_magnitudes.tolist(), strict=True), start=1
    ):
        vector = contributions[index]
        cos = cosine(vector, output)
        signed_projection = float(torch.dot(vector, output_unit))
        perpendicular = math.sqrt(
            max(0.0, float(vector.norm()) ** 2 - signed_projection**2)
        )
        angle = math.degrees(math.acos(max(-1.0, min(1.0, cos))))
        print(
            f"  {rank:>2}. neuron {index:>5}  a={float(activations[index]):+.8f}  "
            f"|v|={magnitude:.8f}  cos(v, output)={cos:+.8f}  "
            f"angle={angle:7.3f}°  projection={signed_projection:+.8f}  "
            f"projection/output={signed_projection / float(output_norm):+.3%}  "
            f"perpendicular={perpendicular:.8f}"
        )

    top_sum = contributions.index_select(0, top_indices).sum(dim=0)
    remainder = output - top_sum
    print(f"top-{count} aggregate:")
    print(f"  |sum(top)|: {float(top_sum.norm()):.8f}")
    print(f"  cos(sum(top), output): {cosine(top_sum, output):+.8f}")
    print(
        "  signed projection onto output: "
        f"{float(torch.dot(top_sum, output_unit)):+.8f} "
        f"({float(torch.dot(top_sum, output_unit) / output_norm):+.3%})"
    )
    print(f"  |output - sum(top)|: {float(remainder.norm()):.8f}")
    print(f"  cos(remainder, output): {cosine(remainder, output):+.8f}")

    signed_projections = contributions @ output_unit
    projection_values, projection_indices = torch.topk(signed_projections, count)
    print(f"top-{count} neurons ranked by signed projection onto the MLP output:")
    for rank, (index, projection_value) in enumerate(
        zip(
            projection_indices.tolist(), projection_values.tolist(), strict=True
        ),
        start=1,
    ):
        vector = contributions[index]
        magnitude = float(vector.norm())
        cos = cosine(vector, output)
        print(
            f"  {rank:>2}. neuron {index:>5}  a={float(activations[index]):+.8f}  "
            f"|v|={magnitude:.8f}  cos(v, output)={cos:+.8f}  "
            f"projection={projection_value:+.8f}  "
            f"projection/output={projection_value / float(output_norm):+.3%}"
        )


if __name__ == "__main__":
    main()
