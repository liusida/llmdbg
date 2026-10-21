#!/usr/bin/env python3
"""Verify an llmdbg component removal with ordinary full-prompt forwards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from gb10_load_llm import from_pretrained_to_cuda
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "log",
        type=Path,
        help="Machine-readable .json file paired with an llmdbg experiment note.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of next-token predictions to print (default: 10).",
    )
    return parser.parse_args()


def projection_for(block: Any, activation_vector: str) -> torch.nn.Module:
    if activation_vector == "mlp_expansion":
        return block.mlp.down_proj
    if activation_vector == "full_attention_mixing":
        return block.self_attn.o_proj
    if activation_vector == "gated_delta_mixing":
        return block.linear_attn.out_proj
    raise ValueError(f"unsupported activation vector: {activation_vector}")


def top_predictions(
    tokenizer: Any, logits: torch.Tensor, count: int
) -> list[dict[str, Any]]:
    logits = logits.detach().float()
    values, token_ids = torch.topk(logits, min(count, logits.numel()))
    probabilities = torch.softmax(logits, dim=-1).index_select(0, token_ids)
    return [
        {
            "tokenId": int(token_id),
            "text": tokenizer.decode([int(token_id)]),
            "logit": float(logit),
            "probability": float(probability),
        }
        for token_id, logit, probability in zip(
            token_ids.tolist(), values.tolist(), probabilities.tolist(), strict=True
        )
    ]


def run_full_forward(
    model: Any,
    tokenizer: Any,
    encoded: Any,
    *,
    token_position: int,
    layer_index: int,
    activation_vector: str,
    component: int,
    intervene: bool,
    top_k: int,
) -> dict[str, Any]:
    """Run the entire prompt once, optionally zeroing one component at one token."""
    projection = projection_for(model.model.layers[layer_index], activation_vector)
    captured: dict[str, float] = {}

    def remove_component(
        _module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]
    ) -> tuple[torch.Tensor, ...] | None:
        activation = inputs[0]
        captured["before"] = float(
            activation[0, token_position, component].detach().float().cpu()
        )
        if not intervene:
            captured["after"] = captured["before"]
            return None

        modified = activation.clone()
        modified[0, token_position, component] = 0
        captured["after"] = float(
            modified[0, token_position, component].detach().float().cpu()
        )
        return (modified, *inputs[1:])

    handle = projection.register_forward_pre_hook(remove_component)
    try:
        device = next(model.parameters()).device
        with torch.inference_mode():
            output = model(
                input_ids=encoded.input_ids.to(device),
                attention_mask=encoded.attention_mask.to(device),
                use_cache=False,
            )
    finally:
        handle.remove()

    logits = output.logits[0, token_position]
    return {
        "activationBefore": captured["before"],
        "activationAfter": captured["after"],
        "logits": logits.detach().float().cpu(),
        "probabilities": torch.softmax(logits.float(), dim=-1).detach().cpu(),
        "predictions": top_predictions(tokenizer, logits, top_k),
    }


def print_predictions(label: str, result: dict[str, Any]) -> None:
    print(f"\n{label}")
    print(
        f"  selected activation: {result['activationBefore']:+.8f}"
        f" -> {result['activationAfter']:+.8f}"
    )
    for rank, prediction in enumerate(result["predictions"], start=1):
        print(
            f"  {rank:>2}. {prediction['text']!r:<18}"
            f" logit={prediction['logit']:+.4f}"
            f" p={prediction['probability']:.4%}"
        )


def main() -> None:
    args = parse_args()
    record = json.loads(args.log.expanduser().resolve().read_text(encoding="utf-8"))
    if record.get("schema") != "llmdbg.projected-component-intervention.v1":
        raise ValueError("log is not a projected-component intervention record")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to reproduce this CUDA experiment")

    intervention = record["intervention"]
    model_info = record["model"]
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[model_info["actualDtype"]]
    model_id = model_info["id"]
    revision = model_info["revision"]

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    encoded = tokenizer(record["prompt"], return_tensors="pt")
    observed_tokens = [
        tokenizer.decode([int(token_id)]) for token_id in encoded.input_ids[0]
    ]
    if observed_tokens != record["tokens"]:
        raise RuntimeError(f"tokenization differs from the log: {observed_tokens!r}")

    print(f"Loading {model_id} through gb10-load-llm...", flush=True)
    model = from_pretrained_to_cuda(
        AutoModelForCausalLM,
        model_id,
        revision=revision,
        dtype=dtype,
        device="cuda",
        touch="auto",
    )
    model.eval()

    common = {
        "model": model,
        "tokenizer": tokenizer,
        "encoded": encoded,
        "token_position": intervention["tokenPosition"],
        "layer_index": intervention["transformerLayer"],
        "activation_vector": intervention["activationVector"],
        "component": intervention["component"],
        "top_k": args.top_k,
    }
    baseline = run_full_forward(**common, intervene=False)
    intervened = run_full_forward(**common, intervene=True)

    print(
        f"\nPrompt: {record['prompt']!r}\n"
        f"Target: token {intervention['tokenPosition']}"
        f" {intervention['token']!r}, layer {intervention['transformerLayer']},"
        f" component {intervention['component']}"
    )
    print_predictions("Ordinary full forward — baseline", baseline)
    print_predictions("Ordinary full forward — component removed", intervened)

    token_id = baseline["predictions"][0]["tokenId"]
    baseline_logit = float(baseline["logits"][token_id])
    intervened_logit = float(intervened["logits"][token_id])
    baseline_probability = float(baseline["probabilities"][token_id])
    intervened_probability = float(intervened["probabilities"][token_id])
    print(
        f"\nEffect on baseline top prediction"
        f" {tokenizer.decode([token_id])!r}:\n"
        f"  logit: {baseline_logit:+.4f} -> {intervened_logit:+.4f}"
        f" (delta {intervened_logit - baseline_logit:+.4f})\n"
        f"  probability: {baseline_probability:.4%} ->"
        f" {intervened_probability:.4%}"
        f" (delta {intervened_probability - baseline_probability:+.4%})"
    )

    expected_activation = float(intervention["activationBefore"])
    print(
        "\nExecution-mode comparison:\n"
        f"  full-forward activation: {baseline['activationBefore']:+.8f}\n"
        f"  cached-debugger activation: {expected_activation:+.8f}\n"
        f"  difference: {baseline['activationBefore'] - expected_activation:+.8f}"
    )
    if baseline["activationBefore"] != intervened["activationBefore"]:
        raise AssertionError("baseline and intervention forwards saw different inputs")
    if intervened["activationAfter"] != 0.0:
        raise AssertionError("the selected component was not zeroed")


if __name__ == "__main__":
    main()
