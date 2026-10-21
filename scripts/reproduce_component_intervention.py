#!/usr/bin/env python3
"""Reproduce a logged llmdbg neuron or attention-channel intervention."""

from __future__ import annotations

import argparse
import hashlib
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
        "--strict-hashes",
        action="store_true",
        help="Fail if reproduced module-output hashes differ byte-for-byte.",
    )
    return parser.parse_args()


def tensor_from_output(output: Any) -> torch.Tensor:
    return output[0] if isinstance(output, tuple) else output


def vector_record(vector: torch.Tensor) -> dict[str, Any]:
    values = tuple(float(value) for value in vector.detach().float().cpu().tolist())
    encoded = json.dumps(values, separators=(",", ":")).encode("utf-8")
    return {
        "values": values,
        "norm": sum(value * value for value in values) ** 0.5,
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


def predictions(
    model: Any,
    tokenizer: Any,
    final_hidden: torch.Tensor,
    *,
    count: int = 10,
) -> list[dict[str, Any]]:
    with torch.inference_mode():
        weight = model.get_output_embeddings().weight
        logits = torch.mv(weight, final_hidden.to(dtype=weight.dtype)).float()
        values, ids = torch.topk(logits, min(count, logits.numel()))
        probabilities = torch.softmax(logits, dim=0).index_select(0, ids)
    return [
        {
            "tokenId": int(token_id),
            "text": tokenizer.decode([int(token_id)]),
            "logit": float(logit),
            "probability": float(probability),
        }
        for token_id, logit, probability in zip(
            ids.tolist(), values.tolist(), probabilities.tolist(), strict=True
        )
    ]


def run_current_token(
    model: Any,
    tokenizer: Any,
    input_ids: torch.Tensor,
    *,
    token_position: int,
    layer_index: int,
    activation_vector: str,
    component: int,
    intervene: bool,
) -> dict[str, Any]:
    """Run one cached token exactly as llmdbg does and capture the intervention."""
    device = next(model.parameters()).device
    prefix_ids = input_ids[:, :token_position].to(device)
    current_id = input_ids[:, token_position : token_position + 1].to(device)
    with torch.inference_mode():
        prefix = model(
            input_ids=prefix_ids,
            attention_mask=torch.ones(
                (1, token_position), dtype=torch.long, device=device
            ),
            use_cache=True,
        )

    block = model.model.layers[layer_index]
    if activation_vector == "mlp_expansion":
        projection = block.mlp.down_proj
    elif activation_vector == "full_attention_mixing":
        projection = block.self_attn.o_proj
    elif activation_vector == "gated_delta_mixing":
        projection = block.linear_attn.out_proj
    else:
        raise ValueError(f"unsupported activation vector: {activation_vector}")

    captured: dict[str, Any] = {}

    def alter_projection_input(
        _module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]
    ) -> None:
        activation = inputs[0]
        captured["activation_before"] = float(
            activation[0, -1, component].detach().float().cpu()
        )
        if intervene:
            activation[0, -1, component] = 0
        captured["activation_after"] = float(
            activation[0, -1, component].detach().float().cpu()
        )

    def capture_block_output(
        _module: torch.nn.Module,
        _inputs: tuple[torch.Tensor, ...],
        output: Any,
    ) -> None:
        captured["module_output"] = tensor_from_output(output)[0, -1].detach()

    def capture_final_hidden(
        _module: torch.nn.Module,
        _inputs: tuple[torch.Tensor, ...],
        output: Any,
    ) -> None:
        captured["final_hidden"] = tensor_from_output(output)[0, -1].detach()

    handles = [
        projection.register_forward_pre_hook(alter_projection_input),
        block.register_forward_hook(capture_block_output),
        model.model.norm.register_forward_hook(capture_final_hidden),
    ]
    try:
        with torch.inference_mode():
            model(
                input_ids=current_id,
                attention_mask=torch.ones(
                    (1, token_position + 1), dtype=torch.long, device=device
                ),
                past_key_values=prefix.past_key_values,
                use_cache=True,
            )
    finally:
        for handle in handles:
            handle.remove()

    output = vector_record(captured["module_output"])
    return {
        "activationBefore": captured["activation_before"],
        "activationAfter": captured["activation_after"],
        "moduleOutput": output,
        "predictions": predictions(
            model, tokenizer, captured["final_hidden"], count=10
        ),
    }


def print_comparison(
    label: str,
    actual: dict[str, Any],
    expected_output: dict[str, Any],
    expected_predictions: list[dict[str, Any]],
) -> bool:
    hash_matches = actual["moduleOutput"]["sha256"] == expected_output["sha256"]
    print(f"\n{label}")
    print(
        f"  activation: {actual['activationBefore']:+.8f}"
        f" -> {actual['activationAfter']:+.8f}"
    )
    print(
        f"  module output norm: {actual['moduleOutput']['norm']:.8f}"
        f" (logged {expected_output['norm']:.8f})"
    )
    print(f"  module output hash: {actual['moduleOutput']['sha256']}")
    print(f"  logged output hash: {expected_output['sha256']}")
    print(f"  exact hash match: {hash_matches}")
    print("  predictions:")
    for rank, (observed, expected) in enumerate(
        zip(actual["predictions"], expected_predictions, strict=True), start=1
    ):
        print(
            f"    {rank:>2}. {observed['text']!r:<18}"
            f" logit={observed['logit']:+.4f}"
            f" p={observed['probability']:.4%}"
            f" | logged {expected['text']!r}"
            f" {expected['logit']:+.4f} {expected['probability']:.4%}"
        )
    return hash_matches


def main() -> None:
    args = parse_args()
    record = json.loads(args.log.expanduser().resolve().read_text(encoding="utf-8"))
    if record.get("schema") != "llmdbg.projected-component-intervention.v1":
        raise ValueError("log is not a projected-component intervention record")
    intervention = record["intervention"]
    model_info = record["model"]
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[model_info["actualDtype"]]
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to reproduce this CUDA experiment")

    model_id = model_info["id"]
    revision = model_info["revision"]
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    encoded = tokenizer(record["prompt"], return_tensors="pt")
    observed_tokens = [
        tokenizer.decode([int(token_id)]) for token_id in encoded.input_ids[0]
    ]
    if observed_tokens != record["tokens"]:
        raise RuntimeError(
            f"tokenization differs from the log: {observed_tokens!r}"
        )

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
        "input_ids": encoded.input_ids,
        "token_position": intervention["tokenPosition"],
        "layer_index": intervention["transformerLayer"],
        "activation_vector": intervention["activationVector"],
        "component": intervention["component"],
    }
    baseline = run_current_token(**common, intervene=False)
    intervened = run_current_token(**common, intervene=True)

    baseline_match = print_comparison(
        "Baseline",
        baseline,
        record["moduleOutputBefore"],
        record["baselinePredictions"],
    )
    intervention_match = print_comparison(
        "Intervention",
        intervened,
        record["moduleOutputAfter"],
        record["intervenedPredictions"],
    )
    expected_before = float(intervention["activationBefore"])
    if abs(baseline["activationBefore"] - expected_before) > 1e-5:
        raise AssertionError(
            f"activation mismatch: reproduced {baseline['activationBefore']},"
            f" logged {expected_before}"
        )
    if intervened["activationAfter"] != 0.0:
        raise AssertionError("the intervention did not zero the selected component")
    if args.strict_hashes and not (baseline_match and intervention_match):
        raise AssertionError("one or more exact output hashes differ from the log")


if __name__ == "__main__":
    main()
