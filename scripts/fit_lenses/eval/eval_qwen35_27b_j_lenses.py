#!/usr/bin/env python3
"""Compare Neuronpedia, v1, and v2 Qwen3.5-27B Post J-Lenses."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Any

import torch
from gb10_load_llm import load_model_to_cuda
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.fit_lenses.eval import eval_qwen3_8b_j_lenses as common


MODEL_ID = "Qwen/Qwen3.5-27B"
MODEL_REVISION = "fc05daec18b0a78c049392ed2e771dde82bdf654"
HIDDEN_SIZE = 5120

V1_REPOSITORY = "camilablank/workspace-lenses"
V1_REVISION = "d740106d1e0f95456dc8718fba2895e9c8ffd6ef"
V1_FILENAME = "qwen3.5-27b/j-lens/lens.pt"

V2_REPOSITORY = "sida/llmdbg"
V2_REPOSITORY_TYPE = "dataset"
V2_FILENAME = "lenses/v2/Qwen--Qwen3.5-27B/j/lens.safetensors"

NEURONPEDIA_REPOSITORY = "neuronpedia/jacobian-lens"
NEURONPEDIA_REVISION = common.NEURONPEDIA_REVISION
NEURONPEDIA_FILENAME = (
    "qwen3.5-27b/jlens/Salesforce-wikitext/Qwen3.5-27B_jacobian_lens.pt"
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    parser.add_argument(
        "--slugs", nargs="+", choices=common.DEFAULT_SLUGS, default=list(common.DEFAULT_SLUGS)
    )
    parser.add_argument("--ks", nargs="+", type=int, default=[1, 5, 10, 50])
    parser.add_argument("--limit", type=int, help="Limit each eval for a smoke test")
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--score-batch-size", type=int, default=8)
    parser.add_argument("--include-incorrect", action="store_true")
    parser.add_argument("--neuronpedia", type=Path)
    parser.add_argument("--ours-v1", type=Path)
    parser.add_argument("--ours-v2", type=Path)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=common.PROJECT_ROOT / "build/lenses/eval/anthropic" / common.ANTHROPIC_REVISION,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=common.PROJECT_ROOT / "build/lenses/eval/qwen35-27b-j-lens",
    )
    return parser.parse_args()


def hub_or_override(
    override: Path | None,
    *,
    repo_id: str,
    filename: str,
    revision: str | None = None,
    repo_type: str = "model",
) -> Path:
    if override is not None:
        path = override.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    return Path(
        hf_hub_download(
            repo_id=repo_id,
            repo_type=repo_type,
            revision=revision,
            filename=filename,
            library_name="llmdbg",
        )
    )


def pytorch_j_maps(path: Path) -> dict[int, torch.Tensor]:
    artifact = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(artifact, dict) or not isinstance(artifact.get("J"), dict):
        raise ValueError(f"invalid legacy J-Lens artifact: {path}")
    return {int(layer): matrix for layer, matrix in artifact["J"].items()}


def v2_post_maps(path: Path) -> dict[int, torch.Tensor]:
    maps: dict[int, torch.Tensor] = {}
    with safe_open(path, framework="pt", device="cpu") as tensors:
        for name in tensors.keys():
            if name.startswith("post."):
                maps[int(name.removeprefix("post."))] = tensors.get_tensor(name)
    if not maps:
        raise ValueError(f"v2 J-Lens has no Post maps: {path}")
    return maps


def load_lenses(args: argparse.Namespace) -> list[common.LensArtifact]:
    neuronpedia = hub_or_override(
        args.neuronpedia,
        repo_id=NEURONPEDIA_REPOSITORY,
        revision=NEURONPEDIA_REVISION,
        filename=NEURONPEDIA_FILENAME,
    )
    ours_v1 = hub_or_override(
        args.ours_v1,
        repo_id=V1_REPOSITORY,
        revision=V1_REVISION,
        filename=V1_FILENAME,
    )
    ours_v2 = hub_or_override(
        args.ours_v2,
        repo_id=V2_REPOSITORY,
        repo_type=V2_REPOSITORY_TYPE,
        filename=V2_FILENAME,
    )
    return [
        common.LensArtifact("neuronpedia", neuronpedia, pytorch_j_maps(neuronpedia)),
        common.LensArtifact("ours-v1", ours_v1, pytorch_j_maps(ours_v1)),
        common.LensArtifact("ours-v2-post", ours_v2, v2_post_maps(ours_v2)),
    ]


def validate_lenses(lenses: list[common.LensArtifact]) -> list[int]:
    layers = sorted(set.intersection(*(set(lens.maps) for lens in lenses)))
    if not layers:
        raise ValueError("the J-Lenses have no common Post layers")
    for lens in lenses:
        for layer, matrix in lens.maps.items():
            if tuple(matrix.shape) != (HIDDEN_SIZE, HIDDEN_SIZE):
                raise ValueError(
                    f"{lens.name} layer {layer} has {tuple(matrix.shape)}, expected "
                    f"{(HIDDEN_SIZE, HIDDEN_SIZE)}"
                )
    return layers


def main() -> int:
    args = arguments()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if any(k <= 0 for k in args.ks):
        raise ValueError("all --ks values must be positive")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")

    common.download_eval_data(args.data_dir, args.slugs)
    items = common.load_items(args.data_dir, args.slugs, args.limit)
    lenses = load_lenses(args)
    layers = validate_lenses(lenses)

    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.model_revision)
    model = load_model_to_cuda(
        AutoModelForCausalLM,
        args.model,
        revision=args.model_revision,
        device="cuda",
        dtype=torch.bfloat16,
        touch="auto",
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )
    model.eval().requires_grad_(False)
    residuals, correctness = common.collect_residuals(
        model, tokenizer, items, layers, args.max_seq_len
    )
    included = common.selected_indices(correctness, args.include_incorrect)
    logging.info("Correctness filter kept %d/%d items", len(included), len(items))
    ranks, dropped = common.score_lenses(
        model,
        tokenizer,
        items,
        residuals,
        lenses,
        layers,
        included,
        args.score_batch_size,
    )
    summary = common.summarize(ranks, sorted(set(args.ks)))

    args.output.mkdir(parents=True, exist_ok=True)
    common.write_csv(
        args.output / "ranks.csv",
        ranks,
        ["eval", "lens", "item", "intermediate", "layer", "rank", "n_candidates"],
    )
    common.write_csv(
        args.output / "summary.csv",
        summary,
        ["eval", "lens", "k", "pass_at_k", "items"],
    )
    results: dict[str, Any] = {
        "model_id": args.model,
        "model_revision": args.model_revision,
        "anthropic_revision": common.ANTHROPIC_REVISION,
        "slugs": args.slugs,
        "ks": sorted(set(args.ks)),
        "layers": layers,
        "site": "post",
        "items_total": len(items),
        "items_scored": len(included),
        "include_incorrect": args.include_incorrect,
        "correctness": [
            {"eval": item.slug, "item": item.name, "correct": correct}
            for item, correct in zip(items, correctness, strict=True)
        ],
        "dropped_intermediates": dropped,
        "lenses": [{"name": lens.name, "path": str(lens.path)} for lens in lenses],
        "summary": summary,
    }
    (args.output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(f"Wrote {args.output / 'summary.csv'}")
    for row in summary:
        if row["eval"] == "macro":
            print(f"{row['lens']:14s} pass@{row['k']:<3d} {row['pass_at_k']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
