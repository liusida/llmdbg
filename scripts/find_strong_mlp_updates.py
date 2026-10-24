#!/usr/bin/env python3
"""Rank prompt token/layer sites by how strongly the MLP changes the residual."""

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
    parser.add_argument("--system", default="You are a helpful assistant.")
    parser.add_argument("--user", default="What is the capital of France?")
    parser.add_argument("--top", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    messages = [
        {"role": "system", "content": args.system},
        {"role": "user", "content": args.user},
    ]
    encoded = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
    )
    input_ids = encoded["input_ids"]
    token_texts = [tokenizer.decode([int(token)]) for token in input_ids[0]]
    print("tokens:", flush=True)
    for index, (token_id, text) in enumerate(zip(input_ids[0].tolist(), token_texts)):
        print(f"  {index:>3}: {token_id:>7} {text!r}", flush=True)

    config = AutoConfig.from_pretrained(args.model, revision=args.revision)
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
    rows: list[dict[str, float | int]] = []
    pending: dict[int, torch.Tensor] = {}
    handles = []

    for layer_index, block in enumerate(model.model.layers):
        def capture_mlp(_module, _inputs, output, *, layer=layer_index):
            pending[layer] = output.detach().float()

        def capture_post(_module, _inputs, output, *, layer=layer_index):
            post = (output[0] if isinstance(output, tuple) else output).detach().float()
            update = pending.pop(layer)
            mid = post - update
            mid_norm = mid.norm(dim=-1)
            post_norm = post.norm(dim=-1)
            update_norm = update.norm(dim=-1)
            cosine = torch.nn.functional.cosine_similarity(mid, post, dim=-1).clamp(-1, 1)
            angles = torch.rad2deg(torch.acos(cosine))
            for token in range(post.shape[1]):
                rows.append({
                    "layer": layer,
                    "token": token,
                    "mid_norm": float(mid_norm[0, token]),
                    "post_norm": float(post_norm[0, token]),
                    "update_norm": float(update_norm[0, token]),
                    "relative": float(update_norm[0, token] / mid_norm[0, token]),
                    "angle": float(angles[0, token]),
                })

        handles.append(block.mlp.register_forward_hook(capture_mlp))
        handles.append(block.register_forward_hook(capture_post))

    try:
        device = next(model.parameters()).device
        attention_mask = encoded.get("attention_mask")
        with torch.inference_mode():
            model(
                input_ids=input_ids.to(device),
                attention_mask=(attention_mask.to(device) if attention_mask is not None else None),
                use_cache=False,
                return_dict=False,
            )
    finally:
        for handle in handles:
            handle.remove()

    def show(title: str, key: str) -> None:
        print(f"\n{title}:")
        for rank, row in enumerate(sorted(rows, key=lambda item: item[key], reverse=True)[:args.top], 1):
            token = int(row["token"])
            print(
                f"  {rank:>2}. token {token:>2} {token_texts[token]!r}, "
                f"layer {int(row['layer']):>2}: angle={row['angle']:.2f}°, "
                f"|MLP|/|Mid|={row['relative']:.3f}, |Mid|={row['mid_norm']:.2f}, "
                f"|MLP|={row['update_norm']:.2f}, |Post|={row['post_norm']:.2f}"
            )

    show("Largest Mid-to-Post direction changes", "angle")
    show("Largest relative MLP updates", "relative")


if __name__ == "__main__":
    main()
