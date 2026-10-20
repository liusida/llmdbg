#!/usr/bin/env python3
"""Convert a fitted torch lens to the llmdbg Hub safetensors format."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import save_file

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("j", "r"))
    parser.add_argument("artifact", type=Path)
    parser.add_argument("hub_path")
    parser.add_argument("--compatible", action="append", default=[])
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = arguments()
    artifact = torch.load(args.artifact, map_location="cpu", weights_only=True)
    maps = artifact.get("J")
    provenance = dict(artifact.get("provenance", {}))
    if not isinstance(maps, dict) or not maps:
        raise ValueError(f"{args.artifact} does not contain J maps")
    layers = sorted(int(layer) for layer in maps)
    tensors = {f"J.{layer}": maps[layer].contiguous() for layer in layers}
    destination = PROJECT_ROOT / "build/hf-upload" / args.hub_path.strip("/")
    destination.mkdir(parents=True, exist_ok=True)
    tensor_path = destination / "lens.safetensors"
    save_file(tensors, tensor_path)

    method = "ordinary_jacobian" if args.kind == "j" else "relp"
    manifest = {
        "format_version": 1,
        "artifact": tensor_path.name,
        "sha256": sha256(tensor_path),
        "model_id": provenance["model_id"],
        "compatible_models": args.compatible,
        "model_revision": provenance.get("model_revision"),
        "method": method,
        "method_version": 1,
        "hidden_size": int(artifact.get("d_model", maps[layers[0]].shape[0])),
        "layers": layers,
        "target_layer": int(provenance["target_layer"]),
        "dtype": str(maps[layers[0]].dtype).removeprefix("torch."),
        "dataset": provenance["dataset_id"],
        "prompts": int(provenance["n_prompts"]),
        "max_seq_len": int(provenance["max_seq_len"]),
        "skip_first": int(provenance["skip_first"]),
        "rules": provenance["rules"],
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    index_path = PROJECT_ROOT / "build/hf-upload/index.json"
    index = json.loads(index_path.read_text())
    section = "j_lenses" if args.kind == "j" else "r_lenses"
    entry = {
        "model_id": provenance["model_id"],
        "compatible_models": args.compatible,
        "method": method,
        "version": 1,
        "path": args.hub_path.strip("/"),
    }
    entries = [item for item in index[section] if item.get("model_id") != entry["model_id"]]
    entries.append(entry)
    index[section] = entries
    index_path.write_text(json.dumps(index, indent=2) + "\n")
    print(f"Packaged {args.artifact} as {destination}")


if __name__ == "__main__":
    main()
