#!/usr/bin/env python3
"""Legacy format-v1 GPT-2 J-Lens fitter retained for reference."""

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
It does not capture the Embed/Mid/Post maps required by format v2, and a v1
result cannot be repackaged as v2. Use scripts/fit_lenses/v2/fit_lenses.py --method j.
"""


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
    arguments()
    raise SystemExit(OBSOLETE_ERROR)


def legacy_main(args: argparse.Namespace) -> None:
    """Retained v1 implementation; unreachable from the command-line entry point."""
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
    root = PROJECT_ROOT / "build/lenses/v1/openai-community--gpt2/j"
    output = args.output or root / "fitted.pt"
    checkpoint = args.checkpoint or root / "fit.checkpoint.pt"
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)

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
        "estimator": "ordinary_jacobian",
        "rules": {"ln_rule": False, "identity_rule": False, "half_rule": False},
    }
    torch.save(artifact, output)
    output.with_suffix(".json").write_text(json.dumps(artifact["provenance"], indent=2) + "\n")
    logging.info("Saved J-Lens artifact to %s", output)


if __name__ == "__main__":
    main()
