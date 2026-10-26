#!/usr/bin/env python3
"""Find ICA components whose signed contribution reads as a requested concept."""

from __future__ import annotations

import argparse
from pathlib import Path


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("../llmdbg-usecase/Qwen35-2B-Vision.llmdbg.py"),
    )
    parser.add_argument("--token", type=int, default=281)
    parser.add_argument("--layer", type=int, default=19)
    parser.add_argument("--concept", default="cat")
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def resolve_local_images(messages: list[dict], config: Path) -> list[dict]:
    resolved = []
    for message in messages:
        copied = dict(message)
        content = copied.get("content")
        if isinstance(content, list):
            parts = []
            for part in content:
                item = dict(part)
                source_key = "path" if "path" in item else "url"
                source = item.get(source_key)
                if (
                    item.get("type") == "image"
                    and isinstance(source, str)
                    and not source.startswith(("http://", "https://", "data:"))
                ):
                    item[source_key] = str((config.parent / source).resolve())
                parts.append(item)
            copied["content"] = parts
        resolved.append(copied)
    return resolved


def main() -> None:
    args = arguments()
    import numpy as np
    import torch

    from llmdbg.config import load_debug_config
    from llmdbg.ica_lens import _layer_artifact, artifact_repository
    from llmdbg.j_lens import source_map
    from llmdbg.model_runtime import prepare_model_execution
    from llmdbg.residual_vocabulary_lenses import activate

    config_path = args.config.resolve()
    config = load_debug_config(str(config_path))
    messages = resolve_local_images(config["messages"], config_path)
    execution = prepare_model_execution(
        config["model"],
        revision=config["revision"],
        messages=messages,
        add_generation_prompt=config["addGenerationPrompt"],
        device=config["device"],
        dtype=config["dtype"],
        progress=lambda message: print(message, end=""),
    )

    print(f"Initial prompt: {execution.token_count} tokens")
    while execution.token_count <= args.token:
        trace = execution.trace_token(execution.token_count - 1)
        _, predictions = execution.vocabulary_projection(trace.states[-1].output_vector)
        prediction = predictions[0]
        execution.append_generated_token(prediction.token_id)
        print(
            f"Generated token {execution.token_count - 1}: "
            f"{prediction.text!r} (p={prediction.probability:.4f})"
        )

    trace = execution.trace_token(args.token)
    state_name = f"model.layers.{args.layer}.mlp"
    state = next(state for state in trace.states if state.layer.name == state_name)
    residual = np.asarray(state.output_vector, dtype=np.float32)
    model_id = str(execution.runtime.model.config._name_or_path)
    repository = artifact_repository(model_id)
    if repository is None:
        raise SystemExit(f"No ICA artifact is configured for {model_id}")
    center, reading, writing = _layer_artifact(repository, args.layer, print)
    scores = (residual - center) @ reading.T
    energy_order = np.argsort(np.abs(scores))[::-1]
    energy_rank = np.empty_like(energy_order)
    energy_rank[energy_order] = np.arange(1, scores.size + 1)
    total_energy = float(np.square(scores, dtype=np.float64).sum())

    tokenizer = execution.runtime.tokenizer
    concept = args.concept.casefold()
    concept_ids = [
        token_id
        for token_id in range(len(tokenizer))
        if tokenizer.decode([token_id]).strip().casefold() in {concept, concept + "s"}
    ]
    if not concept_ids:
        raise SystemExit(f"No vocabulary tokens decode to {args.concept!r}")
    print(
        f"Searching {scores.size} ICA components for {args.concept!r}; "
        f"token IDs={concept_ids}"
    )

    with activate("j-neuronpedia"):
        resolved = source_map(model_id, args.layer, int(residual.size))
    if resolved is None:
        raise SystemExit(f"Neuronpedia J Lens layer {args.layer} is unavailable")
    mapping, metadata = resolved
    print(f"Neuronpedia J Lens: {metadata['path']}")

    device = execution.device
    dtype = execution.runtime.model.get_output_embeddings().weight.dtype
    directions = torch.from_numpy(writing.T.copy()).to(device=device, dtype=torch.float32)
    coefficient = torch.from_numpy(scores.copy()).to(device=device, dtype=torch.float32)
    contributions = directions * coefficient[:, None]
    mapping = mapping.to(device=device, dtype=torch.float32)
    unembedding = execution.runtime.model.get_output_embeddings().weight
    concept_tensor = torch.tensor(concept_ids, device=device, dtype=torch.long)
    candidates = []
    with torch.inference_mode():
        for start in range(0, scores.size, args.batch_size):
            stop = min(start + args.batch_size, scores.size)
            mapped = contributions[start:stop] @ mapping.T
            normalized = execution.runtime.final_norm(mapped.to(dtype=dtype)[:, None, :])[:, 0]
            logits = (normalized @ unembedding.T).float()
            concept_logits = logits.index_select(1, concept_tensor)
            best_values, best_local_ids = concept_logits.max(dim=1)
            ranks = (logits > best_values[:, None]).sum(dim=1) + 1
            top_values, top_ids = torch.topk(logits, 10, dim=1)
            for offset in range(stop - start):
                index = start + offset
                candidates.append(
                    {
                        "index": index,
                        "concept_rank": int(ranks[offset]),
                        "concept_token": tokenizer.decode(
                            [concept_ids[int(best_local_ids[offset])]]
                        ),
                        "concept_logit": float(best_values[offset]),
                        "score": float(scores[index]),
                        "energy_rank": int(energy_rank[index]),
                        "energy": float(scores[index] ** 2 / total_energy),
                        "top": [
                            tokenizer.decode([int(token_id)])
                            for token_id in top_ids[offset].tolist()
                        ],
                    }
                )

    candidates.sort(key=lambda item: (item["concept_rank"], -item["concept_logit"]))
    print("\nBest cat-oriented signed ICA contributions:")
    for item in candidates[: args.count]:
        print(
            f"C{item['index']}: {item['concept_token']!r} rank={item['concept_rank']}, "
            f"score={item['score']:+.3f}, energy-rank={item['energy_rank']}, "
            f"energy={item['energy']:.4%}; top={item['top']}"
        )

    print("\nMost active ICA components:")
    by_index = {item["index"]: item for item in candidates}
    for index in energy_order[: args.count]:
        item = by_index[int(index)]
        print(
            f"C{item['index']}: score={item['score']:+.3f}, "
            f"energy={item['energy']:.4%}, {item['concept_token']!r} "
            f"rank={item['concept_rank']}; top={item['top']}"
        )


if __name__ == "__main__":
    main()
