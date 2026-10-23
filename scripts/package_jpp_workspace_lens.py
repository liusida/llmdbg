#!/usr/bin/env python3
"""Convert a site-aware reference J++ fit into the llmdbg v2 artifact format."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import torch
from safetensors.torch import save_file


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("fit_manifest", type=Path)
    parser.add_argument("output", type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def site_key(index: int, target_index: int) -> str:
    if index == 0:
        return "embed"
    layer, offset = divmod(index - 1, 2)
    key = f"mid.{layer}" if offset == 0 else f"post.{layer}"
    if index >= target_index:
        raise ValueError(f"source index {index} is not before target {target_index}")
    return key


def main() -> int:
    args = arguments()
    artifact = torch.load(args.artifact, map_location="cpu", weights_only=True, mmap=True)
    parameters = artifact.get("parameters", {})
    maps = parameters.get("jacobians", artifact.get("jacobians"))
    if not isinstance(maps, dict) or not maps:
        raise ValueError(f"{args.artifact} has no J++ Jacobians")
    fit = json.loads(args.fit_manifest.read_text())
    target = int(fit["target_layer"])
    tensors = {
        site_key(int(index), target): matrix.half().contiguous()
        for index, matrix in maps.items()
    }
    args.output.mkdir(parents=True, exist_ok=True)
    tensor_path = args.output / "lens.safetensors"
    save_file(tensors, tensor_path)
    first = next(iter(tensors.values()))
    manifest = {
        "format_version": 2,
        "artifact": tensor_path.name,
        "sha256": sha256(tensor_path),
        "model_id": fit["model_id"],
        "method": "jpp",
        "method_name": "filtered_expert_relp_jacobian",
        "provenance": "llmdbg_refit",
        "target": {"site": "post", "layer": (target - 2) // 2},
        "sites": {
            "embed": "embed" in tensors,
            "mid": sorted(int(key.split(".")[1]) for key in tensors if key.startswith("mid.")),
            "post": sorted(int(key.split(".")[1]) for key in tensors if key.startswith("post.")),
        },
        "hidden_size": int(first.shape[0]),
        "dtype": "float16",
        "artifact_bytes": tensor_path.stat().st_size,
        "tensor_shapes": {key: list(value.shape) for key, value in tensors.items()},
        "model_revision": fit.get("model_revision"),
        "tokenizer_revision": fit.get("tokenizer_revision"),
        "reference_repo": fit["reference_repo"],
        "reference_commit": fit["reference_commit"],
        "router": fit["router"],
        "fit": fit["fit"],
        "lrp": fit["lrp"],
        "readout_filter": "contains_letter_or_digit",
        "created_at": datetime.now(UTC).isoformat(),
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"packaged {args.artifact} as {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
