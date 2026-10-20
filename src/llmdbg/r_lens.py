"""Discover, download, validate, and cache R-Lens transport maps."""

from __future__ import annotations

import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

HUB_REPO_ID = "sida/llmdbg"
HUB_REPO_TYPE = "dataset"


def _slug(model_id: str) -> str:
    return model_id.rstrip("/").split("/")[-1].lower().replace(".", "_")


def _local_candidates(model_id: str) -> tuple[Path, ...]:
    override = os.environ.get("LLMDBG_R_LENS_PATH")
    data = Path(__file__).with_name("data") / "r_lens"
    names = [_slug(model_id)]
    names.append(names[0][:-5] if names[0].endswith("-base") else f"{names[0]}-base")
    paths = [Path(override).expanduser()] if override else []
    paths.extend(data / f"{name}.pt" for name in names)
    return tuple(paths)


def _compatible(source_model: str, target_model: str) -> tuple[bool, str | None]:
    if source_model == target_model:
        return True, None
    base_transfer = (
        source_model.lower().endswith("-base")
        and not target_model.lower().endswith("-base")
        and source_model.lower().removesuffix("-base")
        == target_model.lower().removesuffix("-base")
    )
    return base_transfer, "base_to_instruct" if base_transfer else None


def _load_local(path: Path, model_id: str) -> dict[str, Any]:
    import torch

    artifact = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(artifact, dict) or not isinstance(artifact.get("J"), dict):
        raise ValueError(f"invalid R-Lens artifact at {path}: missing J maps")
    source_model = str(artifact.get("provenance", {}).get("model_id", ""))
    compatible, transfer = _compatible(source_model, model_id)
    if not compatible:
        raise ValueError(
            f"R-Lens artifact model {source_model!r} is incompatible with {model_id!r}"
        )
    artifact["_path"] = str(path)
    artifact["_transfer"] = transfer
    return artifact


def _hub_entry(index: dict[str, Any], model_id: str) -> dict[str, Any] | None:
    for entry in index.get("r_lenses", []):
        if model_id == entry.get("model_id") or model_id in entry.get(
            "compatible_models", []
        ):
            return entry
    return None


def _download_from_hub(model_id: str) -> dict[str, Any] | None:
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    download = {
        "repo_id": HUB_REPO_ID,
        "repo_type": HUB_REPO_TYPE,
        "library_name": "llmdbg",
    }
    index_path = Path(hf_hub_download(filename="index.json", **download))
    entry = _hub_entry(json.loads(index_path.read_text()), model_id)
    if entry is None:
        return None
    directory = str(entry["path"]).strip("/")
    manifest_path = Path(
        hf_hub_download(filename=f"{directory}/manifest.json", **download)
    )
    manifest = json.loads(manifest_path.read_text())
    source_model = str(manifest.get("model_id", ""))
    compatible, transfer = _compatible(source_model, model_id)
    if not compatible or (
        source_model != model_id and model_id not in manifest.get("compatible_models", [])
    ):
        raise ValueError(
            f"R-Lens manifest model {source_model!r} is incompatible with {model_id!r}"
        )
    lens_path = Path(
        hf_hub_download(filename=f"{directory}/{manifest['artifact']}", **download)
    )
    digest = hashlib.sha256(lens_path.read_bytes()).hexdigest()
    if digest != manifest.get("sha256"):
        raise ValueError(
            f"R-Lens checksum mismatch for {lens_path}: {digest} != {manifest.get('sha256')}"
        )
    tensors = load_file(lens_path, device="cpu")
    maps = {
        int(name.removeprefix("J.")): tensor
        for name, tensor in tensors.items()
        if name.startswith("J.")
    }
    return {
        "J": maps,
        "d_model": int(manifest["hidden_size"]),
        "source_layers": sorted(maps),
        "provenance": manifest,
        "_path": str(lens_path),
        "_transfer": transfer,
    }


@lru_cache(maxsize=4)
def load_r_lens(model_id: str) -> dict[str, Any] | None:
    """Load a local override or lazily download a compatible Hub artifact."""
    for path in _local_candidates(model_id):
        if path.is_file():
            return _load_local(path, model_id)
    try:
        return _download_from_hub(model_id)
    except OSError as error:
        # This exception is not cached, so Refresh can retry a transient failure.
        raise RuntimeError(f"Could not download R-Lens data from the Hub: {error}") from error


def source_map(model_id: str, layer: int, hidden_size: int) -> tuple[Any, dict[str, Any]] | None:
    """Return one validated transport map and display metadata."""
    artifact = load_r_lens(model_id)
    if artifact is None:
        return None
    if int(artifact.get("d_model", 0)) != hidden_size:
        raise ValueError(
            f"R-Lens width {artifact.get('d_model')} does not match model width {hidden_size}"
        )
    maps = artifact["J"]
    matrix = maps.get(layer, maps.get(str(layer)))
    if matrix is None:
        return None
    if tuple(matrix.shape) != (hidden_size, hidden_size):
        raise ValueError(
            f"R-Lens layer {layer} map has shape {tuple(matrix.shape)}, "
            f"expected {(hidden_size, hidden_size)}"
        )
    provenance = dict(artifact.get("provenance", {}))
    provenance["path"] = artifact["_path"]
    provenance["transfer"] = artifact["_transfer"]
    return matrix, provenance
