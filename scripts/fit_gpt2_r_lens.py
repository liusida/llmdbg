#!/usr/bin/env python3
"""Fit a RelP R-Lens transport map for GPT-2."""

from __future__ import annotations

import argparse
import itertools
import json
import logging
from pathlib import Path
from types import MethodType
from typing import Any

import torch
from gb10_load_llm import load_model_to_cuda
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.gpt2.modeling_gpt2 import GPT2MLP

DATASET_ID = "NeelNanda/pile-10k"
PROJECT_ROOT = Path(__file__).resolve().parents[1]


class _IdentityActivation(torch.autograd.Function):
    """Keep GELU's value while using the RelP secant rule backward."""

    @staticmethod
    def forward(ctx: Any, values: torch.Tensor, activated: torch.Tensor) -> torch.Tensor:
        ratio = torch.where(values != 0, activated / values, torch.full_like(values, 0.5))
        ctx.save_for_backward(ratio)
        return activated

    @staticmethod
    def backward(ctx: Any, gradient: torch.Tensor) -> tuple[torch.Tensor, None]:
        (ratio,) = ctx.saved_tensors
        return gradient * ratio, None


def install_relp_rules(model: nn.Module) -> tuple[int, int]:
    """Install value-preserving RelP rules for GPT-2 LayerNorm and GELU."""
    norms = mlps = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.LayerNorm) and (
            name.endswith(".ln_1") or name.endswith(".ln_2") or name == "transformer.ln_f"
        ):
            def relp_norm(current: nn.LayerNorm, values: torch.Tensor) -> torch.Tensor:
                dims = tuple(range(values.ndim - len(current.normalized_shape), values.ndim))
                mean = values.float().mean(dims, keepdim=True).detach()
                variance = (values.float() - mean).pow(2).mean(dims, keepdim=True).detach()
                normalized = (values.float() - mean) * torch.rsqrt(variance + current.eps)
                if current.elementwise_affine:
                    normalized = normalized * current.weight.float() + current.bias.float()
                return normalized.type_as(values)

            module.forward = MethodType(relp_norm, module)
            norms += 1
        elif isinstance(module, GPT2MLP):
            def relp_mlp(current: GPT2MLP, values: torch.Tensor) -> torch.Tensor:
                expanded = current.c_fc(values)
                activated = _IdentityActivation.apply(expanded, current.act(expanded))
                return current.dropout(current.c_proj(activated))

            module.forward = MethodType(relp_mlp, module)
            mlps += 1
    return norms, mlps


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="openai-community/gpt2")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--prompts", type=int, default=25)
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--skip-first", type=int, default=4)
    parser.add_argument("--dim-batch", type=int, default=32)
    parser.add_argument("--checkpoint", type=Path)
    return parser.parse_args()


def calibration_prompts(count: int) -> list[str]:
    from datasets import load_dataset

    dataset = load_dataset(DATASET_ID, split="train", streaming=True)
    texts = (str(row["text"]) for row in dataset if str(row["text"]).strip())
    result = list(itertools.islice(texts, count))
    if len(result) != count:
        raise RuntimeError(f"dataset supplied {len(result)} of {count} requested prompts")
    return result


def main() -> None:
    args = arguments()
    import jlens

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = load_model_to_cuda(
        AutoModelForCausalLM,
        args.model,
        device="cuda",
        dtype=torch.float32,
        touch="auto",
        low_cpu_mem_usage=True,
    )
    model.eval().requires_grad_(False)
    layer_count = int(model.config.num_hidden_layers)
    target_layer = layer_count - 1
    output = args.output or PROJECT_ROOT / "build/r_lens/artifacts/gpt2.pt"
    checkpoint = args.checkpoint or PROJECT_ROOT / "build/r_lens/gpt2.checkpoint.pt"
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)

    parity_ids = tokenizer("R-Lens parity check.", return_tensors="pt")["input_ids"].cuda()
    with torch.inference_mode():
        ordinary = model(parity_ids).logits
    norm_count, mlp_count = install_relp_rules(model)
    if (norm_count, mlp_count) != (2 * layer_count + 1, layer_count):
        raise RuntimeError(f"unexpected module counts: norms={norm_count}, mlps={mlp_count}")
    with torch.inference_mode():
        patched = model(parity_ids).logits
    if not torch.allclose(ordinary, patched, atol=1e-6, rtol=1e-5):
        raise RuntimeError(
            f"RelP rules changed forward values (max delta {(ordinary-patched).abs().max().item():.3e})"
        )

    fitted = jlens.fit(
        jlens.from_hf(model, tokenizer),
        calibration_prompts(args.prompts),
        source_layers=list(range(target_layer)),
        target_layer=target_layer,
        dim_batch=args.dim_batch,
        max_seq_len=args.max_seq_len,
        skip_first=args.skip_first,
        checkpoint_path=str(checkpoint),
        checkpoint_every=1,
        resume=True,
    )
    fitted.jacobians[target_layer] = torch.eye(fitted.d_model)
    fitted.source_layers = sorted(fitted.jacobians)
    fitted.save(str(output), dtype=torch.float16)
    artifact = torch.load(output, map_location="cpu", weights_only=True)
    artifact["provenance"] = {
        "model_id": args.model,
        "model_revision": getattr(model.config, "_commit_hash", None),
        "dataset_id": DATASET_ID,
        "target_layer": target_layer,
        "max_seq_len": args.max_seq_len,
        "n_prompts": fitted.n_prompts,
        "skip_first": args.skip_first,
        "estimator": "relp",
        "rules": {"ln_rule": True, "identity_rule": True, "half_rule": False},
    }
    torch.save(artifact, output)
    output.with_suffix(".json").write_text(json.dumps(artifact["provenance"], indent=2) + "\n")
    logging.info("Saved R-Lens artifact to %s", output)


if __name__ == "__main__":
    main()
