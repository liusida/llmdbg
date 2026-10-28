#!/usr/bin/env python3
"""Legacy format-v1 J-Lens fitter retained for implementation reference.

Do not use this script for new artifacts. Use ``fit_lenses/v2/fit_lenses.py`` to
fit the current, site-aware format-v2 Workspace Lens instead.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
from pathlib import Path

import torch
from gb10_load_llm import load_model_to_cuda
from transformers import AutoModelForCausalLM, AutoTokenizer

DATASET_ID = "NeelNanda/pile-10k"
PROJECT_ROOT = Path(__file__).resolve().parents[3]
OBSOLETE_ERROR = """\
ERROR: this script fits the obsolete format-v1 J-Lens artifact.
It does not capture the Embed/Mid/Post maps required by the current format-v2
Workspace Lens, and a v1 result cannot be repackaged as v2.

Use scripts/fit_lenses/v2/fit_lenses.py --method j for new fits. If the requested
model is not listed there yet, add and test its architecture support first.
"""


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
            'Install the fitter: uv pip install "git+https://github.com/anthropics/jacobian-lens.git"'
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
    config = (
        model.config
        if hasattr(model.config, "num_hidden_layers")
        else model.config.get_text_config()
    )
    layer_count = int(config.num_hidden_layers)
    target_layer = layer_count - 1 if args.target_layer is None else args.target_layer
    source_layers = (
        list(range(target_layer + 1))
        if args.layers == "all"
        else sorted({int(value) for value in args.layers.split(",")})
    )
    if not source_layers or min(source_layers) < 0 or max(source_layers) > target_layer:
        raise ValueError(f"source layers must be within 0..{target_layer}")

    root = PROJECT_ROOT / "build/lenses/v1" / args.model.replace("/", "--") / "j"
    output = args.output or root / "fitted.pt"
    checkpoint = args.checkpoint or root / "fit.checkpoint.pt"
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)

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
        "estimator": "ordinary_jacobian",
        "rules": {"ln_rule": False, "identity_rule": False, "half_rule": False},
    }
    torch.save(artifact, output)
    output.with_suffix(".json").write_text(
        json.dumps(artifact["provenance"], indent=2) + "\n"
    )
    logging.info("Saved J-Lens artifact to %s", output)


if __name__ == "__main__":
    main()
