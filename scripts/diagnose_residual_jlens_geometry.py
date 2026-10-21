#!/usr/bin/env python3
"""Explain how a residual update changes geometry while preserving J-Lens tokens."""

from __future__ import annotations

import argparse

import torch
from gb10_load_llm import load_model_to_cuda
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from llmdbg.j_lens import source_map as j_lens_map


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3.5-2B")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--token", type=int, default=1)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--top", type=int, default=10)
    return parser.parse_args()


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    denominator = left.float().norm() * right.float().norm()
    return float(torch.dot(left.float(), right.float()) / denominator) if denominator else float("nan")


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this diagnostic")

    config = AutoConfig.from_pretrained(args.model, revision=args.revision)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    encoded = tokenizer(args.prompt, add_special_tokens=False, return_tensors="pt")
    token_texts = [tokenizer.decode([int(token_id)]) for token_id in encoded.input_ids[0]]
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
    block = model.model.layers[args.layer]
    text_config = getattr(config, "text_config", config)
    raw_kind = text_config.layer_types[args.layer]
    projection = block.linear_attn.out_proj if raw_kind == "linear_attention" else block.self_attn.o_proj
    captured: dict[str, torch.Tensor] = {}

    def capture_input(_module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        captured["input"] = inputs[0].detach()

    def capture_delta(
        _module: torch.nn.Module,
        _inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        captured["delta"] = output.detach()

    handles = [
        block.register_forward_pre_hook(capture_input),
        projection.register_forward_hook(capture_delta),
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

    residual_input = captured["input"][0, args.token].float()
    delta = captured["delta"][0, args.token].float()
    residual_output = residual_input + delta
    source = j_lens_map(args.model, args.layer, int(residual_input.numel()))
    if source is None:
        raise RuntimeError("no compatible J-Lens map is available")
    mapping, provenance = source
    mapping = mapping.to(device=residual_input.device, dtype=torch.float32)
    unembedding = model.get_output_embeddings().weight.detach().float()
    final_norm = model.model.norm

    def mapped(vector: torch.Tensor) -> torch.Tensor:
        return mapping @ vector.float()

    def logits(vector: torch.Tensor) -> torch.Tensor:
        transported = mapped(vector)
        normalized = final_norm(transported.to(final_norm.weight.dtype)).float()
        return (normalized @ unembedding.T).detach()

    vectors = {
        "input": residual_input,
        "delta": delta,
        "output": residual_output,
    }
    scores = {name: logits(vector) for name, vector in vectors.items()}
    mapped_vectors = {name: mapped(vector) for name, vector in vectors.items()}
    output_values, output_ids = torch.topk(scores["output"], args.top)
    top_ids = {
        name: torch.topk(value, args.top).indices.tolist() for name, value in scores.items()
    }

    print(f"source token: {args.token} {token_texts[args.token]!r}")
    print(f"layer: {args.layer} ({raw_kind})")
    print(f"J-Lens artifact: {provenance.get('path')}")
    print("raw residual geometry:")
    for name, vector in vectors.items():
        print(f"  |{name}| = {float(vector.norm()):.8f}")
    print(f"  |delta| / |input| = {float(delta.norm() / residual_input.norm()):.8f}")
    print(f"  cos(input, delta) = {cosine(residual_input, delta):+.8f}")
    print(f"  cos(input, output) = {cosine(residual_input, residual_output):+.8f}")
    print(f"  cos(delta, output) = {cosine(delta, residual_output):+.8f}")
    print("transported J-Lens geometry before final RMSNorm:")
    print(f"  cos(J input, J delta) = {cosine(mapped_vectors['input'], mapped_vectors['delta']):+.8f}")
    print(f"  cos(J input, J output) = {cosine(mapped_vectors['input'], mapped_vectors['output']):+.8f}")
    print(f"  cos(J delta, J output) = {cosine(mapped_vectors['delta'], mapped_vectors['output']):+.8f}")
    print(f"  linearity error |J(output) - J(input) - J(delta)|max = {float((mapped_vectors['output'] - mapped_vectors['input'] - mapped_vectors['delta']).abs().max()):.3e}")
    print("top-token overlap:")
    print(f"  input ∩ output: {len(set(top_ids['input']) & set(top_ids['output']))}/{args.top}")
    print(f"  delta ∩ output: {len(set(top_ids['delta']) & set(top_ids['output']))}/{args.top}")
    print("output J-Lens top tokens and their ranks/logits in each vector:")
    for rank, (token_id, output_logit) in enumerate(
        zip(output_ids.tolist(), output_values.tolist(), strict=True), start=1
    ):
        fields = []
        for name in ("input", "delta", "output"):
            score = scores[name]
            token_rank = int((score > score[token_id]).sum()) + 1
            fields.append(f"{name}={float(score[token_id]):+.5f} (rank {token_rank})")
        print(f"  {rank:>2}. {tokenizer.decode([token_id])!r:>18}  " + "  ".join(fields))


if __name__ == "__main__":
    main()
