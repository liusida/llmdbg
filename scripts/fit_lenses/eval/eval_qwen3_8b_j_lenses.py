#!/usr/bin/env python3
"""Compare the Neuronpedia and llmdbg v1 Qwen3-8B J-Lenses.

This implements the readout evaluation released with Anthropic's Jacobian
Lens: a known latent intermediate is scored by its best vocabulary rank over
the fitted layers.  The script emits per-layer ranks and item-weighted pass@k
scores, using the same prompts, readout positions, case/spacing variants, and
order-of-operations synonyms as the public J++ evaluation implementation.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import urllib.request
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
from gb10_load_llm import load_model_to_cuda
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_ID = "Qwen/Qwen3-8B"
MODEL_SLUG = "Qwen--Qwen3-8B"
MODEL_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
ANTHROPIC_REPOSITORY = "anthropics/jacobian-lens"
ANTHROPIC_REVISION = "581d398613e5602a5af361e1c34d3a92ea82ba8e"
NEURONPEDIA_REPOSITORY = "neuronpedia/jacobian-lens"
NEURONPEDIA_REVISION = "e166deb5ad540932100ef803c8c9bcc2a5453694"
NEURONPEDIA_FILENAME = (
    "qwen3-8b/jlens/Salesforce-wikitext/Qwen3-8B_jacobian_lens.pt"
)
DEFAULT_SLUGS = (
    "multihop",
    "multilingual",
    "poetry",
    "order-ops",
    "association",
    "typo",
)

READOUT_RULES = {
    "multihop": "final_token",
    "multilingual": "final_token",
    "poetry": "last_newline",
    "order-ops": "final_token",
    "association": "final_token",
    "typo": "final_token",
}
CORRECTNESS_TARGET = {
    "multihop": "target",
    "multilingual": "target",
    "poetry": "first_intermediate",
    "order-ops": "target",
    "association": None,
    "typo": None,
}
ORDER_OPS_SYNONYMS: dict[str, tuple[str, ...]] = {
    "3": ("3", "three"),
    "4": ("4", "four"),
    "5": ("5", "five"),
    "6": ("6", "six"),
    "7": ("7", "seven"),
    "8": ("8", "eight"),
    "9": ("9", "nine"),
    "10": ("10", "ten"),
    "11": ("11", "eleven"),
    "12": ("12", "twelve", "dozen"),
    "13": ("13", "thirteen"),
    "15": ("15", "fifteen"),
    "16": ("16", "sixteen"),
    "20": ("20", "twenty"),
    "24": ("24", "twenty-four"),
    "addition": ("addition", "add", "plus", "sum", "and", "+"),
    "subtraction": (
        "subtraction", "subtract", "minus", "difference", "take", "diff", "-",
    ),
    "multiplication": (
        "multiplication", "multiply", "times", "product", "mult", "prod", "by", "*", "×",
    ),
    "division": ("division", "divide", "quotient", "/", "÷"),
    "mod": ("mod", "modulo", "remainder", "%"),
    "squared": ("squared", "square", "power", "exponent", "^", "**"),
}


@dataclass(frozen=True)
class EvalItem:
    slug: str
    name: str
    prompt: str
    target: str | None
    intermediates: tuple[str, ...]

    def correctness_target(self) -> str | None:
        source = CORRECTNESS_TARGET[self.slug]
        if source == "target":
            return self.target
        if source == "first_intermediate":
            return self.intermediates[0]
        return None


@dataclass(frozen=True)
class LensArtifact:
    name: str
    path: Path
    maps: dict[int, torch.Tensor]


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    parser.add_argument("--slugs", nargs="+", choices=DEFAULT_SLUGS, default=list(DEFAULT_SLUGS))
    parser.add_argument("--ks", nargs="+", type=int, default=[1, 5, 10, 50])
    parser.add_argument("--limit", type=int, help="Limit each eval for a smoke test")
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--score-batch-size", type=int, default=16)
    parser.add_argument(
        "--include-incorrect",
        action="store_true",
        help="Also score gradeable items that Qwen3-8B does not answer correctly",
    )
    parser.add_argument(
        "--ours",
        type=Path,
        default=PROJECT_ROOT / "build/hf-upload/lenses/v1" / MODEL_SLUG / "j/lens.safetensors",
    )
    parser.add_argument("--neuronpedia", type=Path)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "build/lenses/eval/anthropic" / ANTHROPIC_REVISION,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "build/lenses/eval/qwen3-8b-j-lens",
    )
    return parser.parse_args()


def download_eval_data(data_dir: Path, slugs: Sequence[str]) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    base = (
        "https://raw.githubusercontent.com/"
        f"{ANTHROPIC_REPOSITORY}/{ANTHROPIC_REVISION}/data/evaluations"
    )
    for slug in slugs:
        destination = data_dir / f"lens-eval-{slug}.json"
        if destination.is_file():
            continue
        logging.info("Downloading %s", destination.name)
        with urllib.request.urlopen(f"{base}/{destination.name}") as response:
            payload = response.read()
        parsed = json.loads(payload)
        if not isinstance(parsed.get("items"), list):
            raise ValueError(f"invalid evaluation data at {destination.name}")
        temporary = destination.with_suffix(".json.tmp")
        temporary.write_bytes(payload)
        temporary.replace(destination)


def load_items(data_dir: Path, slugs: Sequence[str], limit: int | None) -> list[EvalItem]:
    items: list[EvalItem] = []
    for slug in slugs:
        raw = json.loads((data_dir / f"lens-eval-{slug}.json").read_text())
        selected = raw["items"][:limit] if limit is not None else raw["items"]
        items.extend(
            EvalItem(
                slug=slug,
                name=str(item["name"]),
                prompt=str(item["prompt"]),
                target=str(item["target"]) if item.get("target") is not None else None,
                intermediates=tuple(str(value) for value in item["intermediates"]),
            )
            for item in selected
        )
    return items


def surface_forms(word: str) -> list[str]:
    cased = {word, word.capitalize(), word.lower()}
    return [prefix + form for form in sorted(cased) for prefix in ("", " ")]


def candidate_token_ids(tokenizer: Any, item: EvalItem, intermediate: str) -> list[int]:
    synonyms = (
        ORDER_OPS_SYNONYMS.get(intermediate, (intermediate,))
        if item.slug == "order-ops"
        else (intermediate,)
    )
    result: set[int] = set()
    for synonym in synonyms:
        for surface in surface_forms(synonym):
            ids = tokenizer(surface, add_special_tokens=False).input_ids
            if len(ids) == 1:
                result.add(int(ids[0]))
    return sorted(result)


def readout_position(input_ids: torch.Tensor, tokenizer: Any, rule: str) -> int:
    if rule == "final_token":
        return input_ids.shape[1] - 1
    positions = [
        index
        for index, token_id in enumerate(input_ids[0].tolist())
        if "\n" in tokenizer.decode([token_id])
    ]
    if not positions:
        raise ValueError("poetry prompt contains no newline token")
    return positions[-1]


def next_token_matches_target(decoded: str, target: str) -> bool:
    stripped = decoded.lstrip()
    return bool(stripped) and target.strip().startswith(stripped)


def neuronpedia_path(override: Path | None) -> Path:
    if override is not None:
        return override.expanduser().resolve()
    return Path(
        hf_hub_download(
            repo_id=NEURONPEDIA_REPOSITORY,
            revision=NEURONPEDIA_REVISION,
            filename=NEURONPEDIA_FILENAME,
            library_name="llmdbg",
        )
    )


def load_lenses(ours_path: Path, neuronpedia_override: Path | None) -> list[LensArtifact]:
    ours_path = ours_path.expanduser().resolve()
    if not ours_path.is_file():
        raise FileNotFoundError(f"llmdbg v1 J-Lens not found: {ours_path}")
    ours_tensors = load_file(ours_path, device="cpu")
    ours = {
        int(name.removeprefix("J.")): tensor
        for name, tensor in ours_tensors.items()
        if name.startswith("J.")
    }

    np_path = neuronpedia_path(neuronpedia_override)
    artifact = torch.load(np_path, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(artifact, dict) or not isinstance(artifact.get("J"), dict):
        raise ValueError(f"invalid Neuronpedia J-Lens: {np_path}")
    neuronpedia = {int(layer): tensor for layer, tensor in artifact["J"].items()}
    return [
        LensArtifact("ours-v1", ours_path, ours),
        LensArtifact("neuronpedia", np_path, neuronpedia),
    ]


@torch.inference_mode()
def collect_residuals(
    model: Any,
    tokenizer: Any,
    items: Sequence[EvalItem],
    layers: Sequence[int],
    max_seq_len: int,
) -> tuple[torch.Tensor, list[bool | None]]:
    rows: list[torch.Tensor] = []
    correctness: list[bool | None] = []
    device = next(model.parameters()).device
    for index, item in enumerate(items):
        encoded = tokenizer(
            item.prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_seq_len,
        )["input_ids"].to(device)
        output = model(encoded, output_hidden_states=True, use_cache=False)
        position = readout_position(encoded, tokenizer, READOUT_RULES[item.slug])
        rows.append(
            torch.stack(
                [output.hidden_states[layer + 1][0, position].detach().cpu() for layer in layers]
            )
        )
        target = item.correctness_target()
        if target is None:
            correctness.append(None)
        else:
            decoded = tokenizer.decode([int(output.logits[0, -1].argmax())])
            correctness.append(next_token_matches_target(decoded, target))
        if (index + 1) % 25 == 0 or index + 1 == len(items):
            logging.info("Captured item %d/%d", index + 1, len(items))
    return torch.stack(rows), correctness


def selected_indices(
    correctness: Sequence[bool | None], include_incorrect: bool
) -> list[int]:
    if include_incorrect:
        return list(range(len(correctness)))
    return [index for index, value in enumerate(correctness) if value is not False]


@torch.inference_mode()
def score_lenses(
    model: Any,
    tokenizer: Any,
    items: Sequence[EvalItem],
    residuals: torch.Tensor,
    lenses: Sequence[LensArtifact],
    layers: Sequence[int],
    item_indices: Sequence[int],
    batch_size: int,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype
    final_norm = model.model.norm
    lm_head = model.lm_head
    rows: list[dict[str, Any]] = []
    dropped: list[dict[str, str]] = []
    candidates: dict[int, dict[str, list[int]]] = {}
    for item_index in item_indices:
        item = items[item_index]
        candidates[item_index] = {}
        for intermediate in item.intermediates:
            ids = candidate_token_ids(tokenizer, item, intermediate)
            if ids:
                candidates[item_index][intermediate] = ids
            else:
                dropped.append({"eval": item.slug, "item": item.name, "intermediate": intermediate})

    item_index_tensor = torch.tensor(item_indices, dtype=torch.long)
    for lens in lenses:
        logging.info("Scoring %s from %s", lens.name, lens.path)
        for layer_offset, layer in enumerate(layers):
            matrix = lens.maps[layer].to(device=device, dtype=torch.float32)
            layer_residuals = residuals[item_index_tensor, layer_offset]
            for start in range(0, len(item_indices), batch_size):
                batch_item_indices = item_indices[start : start + batch_size]
                hidden = layer_residuals[start : start + batch_size].to(
                    device=device, dtype=torch.float32
                )
                transported = (hidden @ matrix.T).to(model_dtype)
                logits = lm_head(final_norm(transported)).float()
                for batch_offset, item_index in enumerate(batch_item_indices):
                    item = items[item_index]
                    item_logits = logits[batch_offset]
                    for intermediate, ids in candidates[item_index].items():
                        candidate_scores = item_logits[torch.tensor(ids, device=device)]
                        best_score = candidate_scores.max()
                        rank = 1 + int((item_logits > best_score).sum())
                        rows.append(
                            {
                                "eval": item.slug,
                                "lens": lens.name,
                                "item": item.name,
                                "intermediate": intermediate,
                                "layer": layer,
                                "rank": rank,
                                "n_candidates": len(ids),
                            }
                        )
                del logits, transported, hidden
            del matrix
            logging.info("%s: scored layer %d", lens.name, layer)
    return rows, dropped


def summarize(rows: Sequence[dict[str, Any]], ks: Sequence[int]) -> list[dict[str, Any]]:
    minimum: dict[tuple[str, str, str, str], int] = {}
    for row in rows:
        key = (row["eval"], row["lens"], row["item"], row["intermediate"])
        minimum[key] = min(minimum.get(key, row["rank"]), row["rank"])

    item_values: dict[tuple[str, str, str, int], list[bool]] = defaultdict(list)
    for (slug, lens, item, _intermediate), rank in minimum.items():
        for k in ks:
            item_values[(slug, lens, item, k)].append(rank <= k)

    per_eval: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    for (slug, lens, _item, k), hits in item_values.items():
        per_eval[(slug, lens, k)].append(sum(hits) / len(hits))

    summary: list[dict[str, Any]] = []
    macro: dict[tuple[str, int], list[float]] = defaultdict(list)
    for (slug, lens, k), values in sorted(per_eval.items()):
        score = sum(values) / len(values)
        summary.append(
            {"eval": slug, "lens": lens, "k": k, "pass_at_k": score, "items": len(values)}
        )
        macro[(lens, k)].append(score)
    for (lens, k), values in sorted(macro.items()):
        summary.append(
            {"eval": "macro", "lens": lens, "k": k, "pass_at_k": sum(values) / len(values), "items": None}
        )
    return summary


def write_csv(path: Path, rows: Sequence[dict[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = arguments()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if any(k <= 0 for k in args.ks):
        raise ValueError("all --ks values must be positive")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")

    download_eval_data(args.data_dir, args.slugs)
    items = load_items(args.data_dir, args.slugs, args.limit)
    lenses = load_lenses(args.ours, args.neuronpedia)
    common_layers = sorted(set.intersection(*(set(lens.maps) for lens in lenses)))
    if not common_layers:
        raise ValueError("the two J-Lenses have no common layers")
    widths = {
        int(matrix.shape[0])
        for lens in lenses
        for matrix in lens.maps.values()
    }
    if widths != {4096}:
        raise ValueError(f"expected Qwen3-8B width 4096, found {sorted(widths)}")

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
    residuals, correctness = collect_residuals(
        model, tokenizer, items, common_layers, args.max_seq_len
    )
    included = selected_indices(correctness, args.include_incorrect)
    logging.info("Correctness filter kept %d/%d items", len(included), len(items))
    ranks, dropped = score_lenses(
        model,
        tokenizer,
        items,
        residuals,
        lenses,
        common_layers,
        included,
        args.score_batch_size,
    )
    summary = summarize(ranks, sorted(set(args.ks)))

    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(
        args.output / "ranks.csv",
        ranks,
        ["eval", "lens", "item", "intermediate", "layer", "rank", "n_candidates"],
    )
    write_csv(
        args.output / "summary.csv",
        summary,
        ["eval", "lens", "k", "pass_at_k", "items"],
    )
    metadata = {
        "model_id": args.model,
        "model_revision": args.model_revision,
        "anthropic_repository": ANTHROPIC_REPOSITORY,
        "anthropic_revision": ANTHROPIC_REVISION,
        "slugs": args.slugs,
        "ks": sorted(set(args.ks)),
        "layers": common_layers,
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
    (args.output / "results.json").write_text(json.dumps(metadata, indent=2) + "\n")

    print(f"Wrote {args.output / 'summary.csv'}")
    for row in summary:
        if row["eval"] == "macro":
            print(f"{row['lens']:12s} pass@{row['k']:<3d} {row['pass_at_k']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
