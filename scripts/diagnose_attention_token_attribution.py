#!/usr/bin/env python3
"""Decompose one attention-output J-Lens token logit over output channels."""

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
    parser.add_argument("--token", type=int, default=3, help="source token position")
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--target", default=" France", help="one vocabulary token")
    parser.add_argument("--top", type=int, default=15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    config = AutoConfig.from_pretrained(args.model, revision=args.revision)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    encoded = tokenizer(args.prompt, add_special_tokens=False, return_tensors="pt")
    token_texts = [tokenizer.decode([int(token_id)]) for token_id in encoded.input_ids[0]]
    if not 0 <= args.token < len(token_texts):
        raise ValueError(f"--token must be in 0..{len(token_texts) - 1}")
    target_ids = tokenizer(args.target, add_special_tokens=False).input_ids
    if len(target_ids) != 1:
        raise ValueError(f"--target must encode to one token, got IDs {target_ids}")
    target_id = int(target_ids[0])

    print(f"prompt tokens: {list(enumerate(token_texts))}", flush=True)
    print(
        f"source token={args.token} {token_texts[args.token]!r}; "
        f"target token={target_id} {tokenizer.decode([target_id])!r}",
        flush=True,
    )
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
    projection = (
        block.linear_attn.out_proj
        if raw_kind == "linear_attention"
        else block.self_attn.o_proj
    )
    captured: dict[str, torch.Tensor] = {}

    def capture_input(_module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        captured["mixed"] = inputs[0].detach()

    def capture_output(
        _module: torch.nn.Module,
        _inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        captured["output"] = output.detach()

    handles = [
        projection.register_forward_pre_hook(capture_input),
        projection.register_forward_hook(capture_output),
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

    mixed = captured["mixed"][0, args.token].float()
    attention_output = captured["output"][0, args.token].float()
    directions = projection.weight.detach().T.float()
    bias = getattr(projection, "bias", None)
    reconstructed = mixed @ directions
    if bias is not None:
        reconstructed = reconstructed + bias.detach().float()
    reconstruction_error = float((reconstructed - attention_output).abs().max())

    source = j_lens_map(args.model, args.layer, int(attention_output.numel()))
    if source is None:
        raise RuntimeError("no compatible J-Lens map is available")
    mapping, provenance = source
    mapping = mapping.to(device=attention_output.device, dtype=torch.float32)
    mapped_output = mapping @ attention_output
    norm = model.model.norm
    # Qwen3.5 parameterizes the RMSNorm gain as (1 + weight).
    norm_weight = 1.0 + norm.weight.detach().float()
    variance = mapped_output.square().mean()
    rms = torch.sqrt(variance + float(getattr(norm, "eps", 1e-6)))
    unembedding = model.get_output_embeddings().weight.detach().float()
    target_readout = norm_weight * unembedding[target_id]

    # Each row is J @ (mixed[i] * W_O-input-direction[i]).
    transported_directions = directions @ mapping.T
    raw_contributions = mixed * (transported_directions @ target_readout)
    contributions = raw_contributions / rms
    bias_contribution = torch.tensor(0.0, device=contributions.device)
    if bias is not None:
        bias_contribution = ((mapping @ bias.detach().float()) @ target_readout) / rms
    exact_logits = (
        model.model.norm(mapped_output.to(norm_weight.dtype)) @ unembedding.T
    ).detach()
    exact_target_logit = float(exact_logits[target_id])
    decomposed_target_logit = float(contributions.sum() + bias_contribution)
    top_logits, top_ids = torch.topk(exact_logits, 10)
    target_rank = int((exact_logits > exact_logits[target_id]).sum()) + 1

    activation_order = torch.argsort(mixed.abs(), descending=True)
    positive_order = torch.argsort(contributions, descending=True)
    negative_order = torch.argsort(contributions)
    contribution_order = torch.argsort(contributions.abs(), descending=True)
    positive_indices = positive_order[contributions[positive_order] > 0]
    negative_indices = negative_order[contributions[negative_order] < 0]
    positive_total = float(contributions.clamp_min(0).sum())
    negative_total = float(contributions.clamp_max(0).sum())

    print(f"J-Lens artifact: {provenance.get('path')}")
    print(f"projection reconstruction max error: {reconstruction_error:.8g}")
    print(f"J-Lens RMS denominator: {float(rms):.8f}")
    print(
        f"target logit exact={exact_target_logit:+.6f}, "
        f"decomposed={decomposed_target_logit:+.6f}, "
        f"error={decomposed_target_logit - exact_target_logit:+.3e}, rank={target_rank}"
    )
    print(f"positive evidence={positive_total:+.6f}; negative evidence={negative_total:+.6f}")
    print("overall J-Lens top tokens:")
    for rank, (token_id, value) in enumerate(
        zip(top_ids.tolist(), top_logits.tolist(), strict=True), start=1
    ):
        print(f"  {rank:>2}. {token_id:>6} {tokenizer.decode([token_id])!r:>18} {value:+.6f}")

    def row(index: int) -> str:
        activation_rank = int((activation_order == index).nonzero()[0]) + 1
        contribution_rank = int((contribution_order == index).nonzero()[0]) + 1
        return (
            f"ch {index:>4}  activation={float(mixed[index]):+9.6f} "
            f"(|a| rank {activation_rank:>4})  |d|={float(directions[index].norm()):.6f}  "
            f"France={float(contributions[index]):+10.6f} "
            f"(|c| rank {contribution_rank:>4})"
        )

    print(f"top {args.top} positive contributors to {args.target!r}:")
    for index in positive_indices[: args.top].tolist():
        print("  " + row(index))
    print(f"top {min(10, args.top)} suppressors of {args.target!r}:")
    for index in negative_indices[: min(10, args.top)].tolist():
        print("  " + row(index))
    print("five channels selected by absolute activation:")
    selected = activation_order[:5]
    for index in selected.tolist():
        print("  " + row(index))
    selected_sum = float(contributions.index_select(0, selected).sum())
    print(
        f"activation-top-5 net France contribution={selected_sum:+.6f} "
        f"({selected_sum / exact_target_logit:+.2%} of exact target logit)"
    )
    requested_counts = (1, 5, 10, 25, 50, 100, 256, 512, 1024)
    counts = [count for count in requested_counts if count < len(positive_indices)]
    counts.append(len(positive_indices))
    for count in counts:
        support = float(contributions.index_select(0, positive_indices[:count]).sum())
        print(
            f"top {count:>4} positive channels: {support:+.6f} "
            f"({support / positive_total:.2%} of positive evidence)"
        )
    print(
        f"supporting channels={len(positive_indices)}; "
        f"suppressing channels={len(negative_indices)}"
    )


if __name__ == "__main__":
    main()
