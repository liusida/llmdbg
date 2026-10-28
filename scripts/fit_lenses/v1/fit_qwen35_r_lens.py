#!/usr/bin/env python3
"""Legacy format-v1 R-Lens fitter retained for implementation reference.

Do not use this script for new artifacts. Use ``fit_lenses/v2/fit_lenses.py`` to
fit the current, site-aware format-v2 Workspace Lens instead.
"""

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
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3.modeling_qwen3 import Qwen3MLP, Qwen3RMSNorm
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5MLP, Qwen3_5RMSNorm

DATASET_ID = "NeelNanda/pile-10k"
PROJECT_ROOT = Path(__file__).resolve().parents[3]
OBSOLETE_ERROR = """\
ERROR: this script fits the obsolete format-v1 R-Lens artifact.
It does not capture the Embed/Mid/Post maps required by the current format-v2
Workspace Lens, and a v1 result cannot be repackaged as v2.

Use scripts/fit_lenses/v2/fit_lenses.py --method r for new fits. If the requested
model is not listed there yet, add and test its architecture support first.
"""


class _IdentityActivation(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, values: torch.Tensor, activated: torch.Tensor) -> torch.Tensor:
        ratio = torch.where(values != 0, activated / values, torch.full_like(values, 0.5))
        ctx.save_for_backward(ratio)
        return activated

    @staticmethod
    def backward(ctx: Any, gradient: torch.Tensor) -> tuple[torch.Tensor, None]:
        (ratio,) = ctx.saved_tensors
        return gradient * ratio, None


class _HalfProduct(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(left, right)
        return left * right

    @staticmethod
    def backward(ctx: Any, gradient: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        left, right = ctx.saved_tensors
        return 0.5 * gradient * right, 0.5 * gradient * left


def install_relp_rules(model: torch.nn.Module) -> tuple[int, int]:
    """Install value-preserving RelP rules for Qwen RMSNorm and SwiGLU."""
    norms = mlps = 0
    for name, module in model.named_modules():
        residual_norm = (
            name.endswith(".input_layernorm")
            or name.endswith(".post_attention_layernorm")
            or name == "model.norm"
        )
        if isinstance(module, Qwen3RMSNorm) and residual_norm:
            def relp_qwen3_norm(
                current: Qwen3RMSNorm, values: torch.Tensor
            ) -> torch.Tensor:
                input_dtype = values.dtype
                inverse_rms = torch.rsqrt(
                    values.float().pow(2).mean(-1, keepdim=True)
                    + current.variance_epsilon
                ).detach()
                normalized = values.float() * inverse_rms
                return current.weight * normalized.to(input_dtype)

            module.forward = MethodType(relp_qwen3_norm, module)
            norms += 1
        elif isinstance(module, Qwen3_5RMSNorm) and residual_norm:
            def relp_qwen35_norm(
                current: Qwen3_5RMSNorm, values: torch.Tensor
            ) -> torch.Tensor:
                inverse_rms = torch.rsqrt(
                    values.float().pow(2).mean(-1, keepdim=True) + current.eps
                ).detach()
                return (
                    values.float() * inverse_rms * (1 + current.weight.float())
                ).type_as(values)

            module.forward = MethodType(relp_qwen35_norm, module)
            norms += 1
        elif isinstance(module, (Qwen3MLP, Qwen3_5MLP)):
            def relp_mlp(
                current: Qwen3MLP | Qwen3_5MLP, values: torch.Tensor
            ) -> torch.Tensor:
                gate_values = current.gate_proj(values)
                gate = _IdentityActivation.apply(gate_values, current.act_fn(gate_values))
                return current.down_proj(_HalfProduct.apply(gate, current.up_proj(values)))

            module.forward = MethodType(relp_mlp, module)
            mlps += 1
    return norms, mlps


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B-Base")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--prompts", type=int, default=25)
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--skip-first", type=int, default=4)
    parser.add_argument("--dim-batch", type=int, default=32)
    parser.add_argument("--target-layer", type=int)
    parser.add_argument("--layers", default="all")
    parser.add_argument("--checkpoint", type=Path)
    return parser.parse_args()


def slug(model_id: str) -> str:
    return model_id.rstrip("/").split("/")[-1].lower().replace(".", "_")


def prompts(count: int) -> list[str]:
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise SystemExit("Install the optional dependency: uv pip install datasets") from error
    dataset = load_dataset(DATASET_ID, split="train", streaming=True)
    texts = (str(row["text"]) for row in dataset if str(row["text"]).strip())
    result = list(itertools.islice(texts, count))
    if len(result) != count:
        raise RuntimeError(f"dataset supplied {len(result)} of {count} requested prompts")
    return result


def main() -> None:
    arguments()
    raise SystemExit(OBSOLETE_ERROR)


def legacy_main(args: argparse.Namespace) -> None:
    """Retained v1 implementation; unreachable from the command-line entry point."""
    try:
        import jlens
    except ImportError as error:
        raise SystemExit(
            "Install the development dependencies with: uv sync"
        ) from error
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
    config = model.config if hasattr(model.config, "num_hidden_layers") else model.config.get_text_config()
    layer_count = int(config.num_hidden_layers)
    target_layer = layer_count - 1 if args.target_layer is None else args.target_layer
    if args.layers == "all":
        source_layers = list(range(target_layer + 1))
    else:
        source_layers = sorted({int(value) for value in args.layers.split(",")})
    if not source_layers or min(source_layers) < 0 or max(source_layers) > target_layer:
        raise ValueError(f"source layers must be within 0..{target_layer}")

    root = PROJECT_ROOT / "build/lenses/v1" / args.model.replace("/", "--") / "r"
    output = args.output or root / "fitted.pt"
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = args.checkpoint or root / "fit.checkpoint.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)

    parity_ids = tokenizer("R-Lens parity check.", return_tensors="pt")["input_ids"].cuda()
    with torch.inference_mode():
        ordinary = model(parity_ids).logits
    norm_count, mlp_count = install_relp_rules(model)
    if (norm_count, mlp_count) != (2 * layer_count + 1, layer_count):
        raise RuntimeError(f"unexpected module counts: norms={norm_count}, mlps={mlp_count}")
    with torch.inference_mode():
        patched = model(parity_ids).logits
    if not torch.equal(ordinary, patched):
        raise RuntimeError(
            f"RelP rules changed forward values (max delta {(ordinary-patched).abs().max().item():.3e})"
        )

    wrapped = jlens.from_hf(model, tokenizer)
    fitted = jlens.fit(
        wrapped,
        prompts(args.prompts),
        source_layers=[layer for layer in source_layers if layer != target_layer],
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
        "rules": {"ln_rule": True, "identity_rule": True, "half_rule": True},
    }
    torch.save(artifact, output)
    output.with_suffix(".json").write_text(json.dumps(artifact["provenance"], indent=2) + "\n")
    logging.info("Saved R-Lens artifact to %s", output)


if __name__ == "__main__":
    main()
