#!/usr/bin/env python3
"""Diagnose one Qwen3.5 gated-attention output channel with counterfactuals."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from typing import Any, Iterator

import torch
from gb10_load_llm import load_model_to_cuda
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3.5-2B")
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--token", type=int, default=3)
    parser.add_argument("--channel", type=int, default=600)
    return parser.parse_args()


@contextmanager
def capture_mixer(mixer: torch.nn.Module) -> Iterator[dict[str, torch.Tensor]]:
    captured: dict[str, torch.Tensor] = {}

    def save(name: str):
        def hook(_module: torch.nn.Module, inputs: tuple[Any, ...]) -> None:
            captured[name] = inputs[0].detach().float().cpu()

        return hook

    def save_norm_inputs(
        _module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]
    ) -> None:
        captured["core"] = inputs[0].detach().float().cpu()
        captured["gate"] = inputs[1].detach().float().cpu()

    def save_mixer_input(
        _module: torch.nn.Module,
        inputs: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        hidden = inputs[0] if inputs else kwargs["hidden_states"]
        captured["input"] = hidden.detach().float().cpu()

    handles = [
        mixer.register_forward_pre_hook(save_mixer_input, with_kwargs=True),
        mixer.norm.register_forward_pre_hook(save_norm_inputs),
        mixer.norm.register_forward_hook(
            lambda _module, _inputs, output: captured.__setitem__(
                "norm_output", output.detach().float().cpu()
            )
        ),
        mixer.out_proj.register_forward_pre_hook(save("mixed")),
    ]
    try:
        yield captured
    finally:
        for handle in handles:
            handle.remove()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    config = AutoConfig.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    encoded = tokenizer(args.prompt, add_special_tokens=False, return_tensors="pt")
    token_ids = encoded.input_ids[0].tolist()
    token_text = [tokenizer.decode([token_id]) for token_id in token_ids]
    print(f"tokens: {list(enumerate(zip(token_ids, token_text, strict=True)))}")

    print(f"Loading {args.model} through gb10-load-llm...", flush=True)
    model = load_model_to_cuda(
        AutoModelForCausalLM,
        args.model,
        config=config,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        touch="auto",
    )
    model.eval()
    block = model.model.layers[args.layer]
    mixer = block.linear_attn
    device = next(model.parameters()).device

    def model_capture(ids: torch.Tensor) -> dict[str, torch.Tensor]:
        with capture_mixer(mixer) as captured, torch.inference_mode():
            model(input_ids=ids.to(device), use_cache=False, return_dict=False)
        return captured

    def mixer_capture(hidden: torch.Tensor) -> dict[str, torch.Tensor]:
        with capture_mixer(mixer) as captured, torch.inference_mode():
            mixer(hidden.to(device=device, dtype=next(mixer.parameters()).dtype))
        return captured

    baseline = model_capture(encoded.input_ids)
    token = args.token
    channel = args.channel
    mixed = baseline["mixed"]
    baseline_value = float(mixed[0, token, channel])
    direction_norm = float(
        torch.linalg.vector_norm(
            mixer.out_proj.weight[:, channel].detach(), dtype=torch.float32
        ).cpu()
    )
    print(
        f"baseline layer={args.layer} token={token} {token_text[token]!r} "
        f"channel={channel}: a={baseline_value:+.8f}, |d|={direction_norm:.8f}, "
        f"|delta_x|={abs(baseline_value) * direction_norm:.8f}"
    )

    head_dim = int(mixer.head_v_dim)
    value_heads = int(mixer.num_v_heads)
    head, offset = divmod(channel, head_dim)
    flat_row = token * value_heads + head
    core_row = baseline["core"][flat_row]
    gate_row = baseline["gate"][flat_row]
    core_value = float(core_row[offset])
    rms = float(torch.sqrt(core_row.square().mean() + mixer.norm.variance_epsilon))
    normalized = core_value / rms
    norm_weight = float(mixer.norm.weight[offset].detach().float().cpu())
    gate_value = float(gate_row[offset])
    silu_gate = float(torch.nn.functional.silu(torch.tensor(gate_value)))
    reconstructed = normalized * norm_weight * silu_gate
    print("channel factorization after recurrent mixing:")
    print(f"  head={head}, within-head offset={offset}, head_dim={head_dim}")
    print(f"  recurrent/core value={core_value:+.8f}")
    print(f"  per-head RMS denominator={rms:.8f}")
    print(f"  normalized core={normalized:+.8f}")
    print(f"  norm weight={norm_weight:+.8f}")
    print(f"  z preactivation={gate_value:+.8f}; SiLU(z)={silu_gate:+.8f}")
    print(
        f"  product={reconstructed:+.8f}; captured={baseline_value:+.8f}; "
        f"error={reconstructed - baseline_value:+.3e}"
    )

    hidden = baseline["input"].to(dtype=torch.bfloat16)
    through_target = hidden[:, : token + 1]
    current_only = mixer_capture(hidden[:, token : token + 1])
    zero_history_input = through_target.clone()
    zero_history_input[:, :token] = 0
    zero_history = mixer_capture(zero_history_input)
    zero_current_input = through_target.clone()
    zero_current_input[:, token] = 0
    zero_current = mixer_capture(zero_current_input)

    def selected(capture: dict[str, torch.Tensor], position: int) -> float:
        return float(capture["mixed"][0, position, channel])

    def mixer_factors(
        capture: dict[str, torch.Tensor], position: int
    ) -> tuple[float, float]:
        row = position * value_heads + head
        core_vector = capture["core"][row]
        core_rms = torch.sqrt(
            core_vector.square().mean() + mixer.norm.variance_epsilon
        )
        return (
            float(core_vector[offset] / core_rms),
            float(torch.nn.functional.silu(capture["gate"][row, offset])),
        )

    print("layer-20 mixer counterfactuals (same trained weights):")
    print(f"  full history + current residual: {baseline_value:+.8f}")
    for label, capture, position in (
        ("current residual only (zero recurrent/conv history)", current_only, 0),
        ("zero earlier layer-20 inputs, keep current residual", zero_history, token),
        ("keep earlier layer-20 inputs, zero current residual", zero_current, token),
    ):
        normalized_core, gate = mixer_factors(capture, position)
        print(
            f"  {label}: {selected(capture, position):+.8f}; "
            f"normalized_core={normalized_core:+.4f}, SiLU(z)={gate:+.4f}"
        )

    def factors(capture: dict[str, torch.Tensor], position: int) -> tuple[float, float]:
        row = position * value_heads + head
        core_vector = capture["core"][row]
        core_rms = torch.sqrt(
            core_vector.square().mean() + mixer.norm.variance_epsilon
        )
        normalized_core = float(core_vector[offset] / core_rms)
        gate = float(torch.nn.functional.silu(capture["gate"][row, offset]))
        return normalized_core, gate

    print("replace only token 3 while keeping token IDs 0..2 fixed:")
    for replacement in (
        " France", " French", " Paris", " Lyon", " Marseille", " Germany",
        " Berlin", " Italy", " Rome", " China", " London", " Europe", " country",
        " city", " apple", " dog",
    ):
        replacement_ids = tokenizer.encode(replacement, add_special_tokens=False)
        if len(replacement_ids) != 1:
            print(f"  {replacement!r}: skipped ({len(replacement_ids)} tokens)")
            continue
        changed = encoded.input_ids.clone()
        changed[0, token] = replacement_ids[0]
        capture = model_capture(changed)
        value = float(capture["mixed"][0, token, channel])
        impact = abs(value) * direction_norm
        normalized_core, gate = factors(capture, token)
        print(
            f"  {replacement!r:>12}: a={value:+.8f}, |delta_x|={impact:.8f}, "
            f"normalized_core={normalized_core:+.4f}, SiLU(z)={gate:+.4f}"
        )

    print("change the three earlier tokens while keeping token 3 = ' France':")
    prefix_sets = (
        ("The", " capital", " of"),
        ("A", " dog", " ate"),
        ("Yesterday", " we", " discussed"),
        ("1", " 2", " 3"),
    )
    france_id = tokenizer.encode(" France", add_special_tokens=False)[0]
    is_id = tokenizer.encode(" is", add_special_tokens=False)[0]
    for pieces in prefix_sets:
        ids = [tokenizer.encode(piece, add_special_tokens=False) for piece in pieces]
        if any(len(item) != 1 for item in ids):
            print(f"  {pieces!r}: skipped (not three single tokens)")
            continue
        sequence = torch.tensor([[item[0] for item in ids] + [france_id, is_id]])
        capture = model_capture(sequence)
        value = float(capture["mixed"][0, token, channel])
        normalized_core, gate = factors(capture, token)
        print(
            f"  {''.join(pieces)!r:>28}: a={value:+.8f}, "
            f"normalized_core={normalized_core:+.4f}, SiLU(z)={gate:+.4f}"
        )


if __name__ == "__main__":
    main()
