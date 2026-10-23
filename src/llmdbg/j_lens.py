"""Discover, download, validate, and cache ordinary J-Lens transport maps."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

HUB_REPO_ID = "sida/llmdbg"
HUB_REPO_TYPE = "dataset"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _slug(model_id: str) -> str:
    return model_id.rstrip("/").split("/")[-1].lower().replace(".", "_")


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


def _local_candidates(model_id: str) -> tuple[Path, ...]:
    override = os.environ.get("LLMDBG_J_LENS_PATH")
    data = Path(__file__).with_name("data") / "j_lens"
    names = [_slug(model_id)]
    names.append(names[0][:-5] if names[0].endswith("-base") else f"{names[0]}-base")
    paths = [Path(override).expanduser()] if override else []
    paths.extend(data / f"{name}.pt" for name in names)
    return tuple(paths)


def _load_local(path: Path, model_id: str) -> dict[str, Any]:
    import torch

    artifact = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(artifact, dict) or not isinstance(artifact.get("J"), dict):
        raise ValueError(f"invalid J-Lens artifact at {path}: missing J maps")
    source_model = str(artifact.get("provenance", {}).get("model_id", ""))
    compatible, transfer = _compatible(source_model, model_id)
    if not compatible:
        raise ValueError(
            f"J-Lens artifact model {source_model!r} is incompatible with {model_id!r}"
        )
    artifact["_path"] = str(path)
    artifact["_transfer"] = transfer
    return artifact


def _hub_entry(index: dict[str, Any], model_id: str) -> dict[str, Any] | None:
    for entry in index.get("j_lenses", []):
        if model_id == entry.get("model_id") or model_id in entry.get(
            "compatible_models", []
        ):
            return entry
    return None


def _download_from_hub(model_id: str) -> dict[str, Any] | None:
    from safetensors.torch import load_file

    from llmdbg.residual_vocabulary_lenses import active_lens, source_file

    configured = active_lens("j")
    source = configured or {"source": f"hf://{HUB_REPO_TYPE}s/{HUB_REPO_ID}"}
    index_path = source_file(source, "index.json")
    index = json.loads(index_path.read_text())
    section = str(configured.get("indexSection", "j_lenses")) if configured else "j_lenses"
    entry = next((item for item in index.get(section, []) if model_id == item.get("model_id") or model_id in item.get("compatible_models", [])), None)
    if entry is None:
        return None
    if entry.get("source_repo_id"):
        import torch
        from huggingface_hub import hf_hub_download

        lens_path = Path(
            hf_hub_download(
                repo_id=str(entry["source_repo_id"]),
                repo_type=str(entry.get("source_repo_type", "model")),
                revision=str(entry["source_revision"]),
                filename=str(entry["source_filename"]),
                library_name="llmdbg",
            )
        )
        expected_size = entry.get("source_size")
        if expected_size is not None and lens_path.stat().st_size != int(expected_size):
            raise ValueError(
                f"J-Lens size mismatch for {lens_path}: "
                f"{lens_path.stat().st_size} != {expected_size}"
            )
        expected_sha256 = entry.get("source_sha256")
        if expected_sha256 is not None:
            actual_sha256 = _sha256(lens_path)
            if actual_sha256 != expected_sha256:
                raise ValueError(
                    f"J-Lens checksum mismatch for {lens_path}: "
                    f"{actual_sha256} != {expected_sha256}"
                )
        artifact = torch.load(
            lens_path, map_location="cpu", weights_only=True, mmap=True
        )
        if not isinstance(artifact, dict) or not isinstance(artifact.get("J"), dict):
            raise ValueError(f"invalid J-Lens artifact at {lens_path}: missing J maps")
        source_model = str(artifact.get("provenance", {}).get("model_id", ""))
        compatible, transfer = _compatible(source_model, model_id)
        if not compatible:
            raise ValueError(
                f"J-Lens artifact model {source_model!r} is incompatible with {model_id!r}"
            )
        artifact["_path"] = str(lens_path)
        artifact["_transfer"] = transfer
        return artifact
    directory = str(entry["path"]).strip("/")
    manifest_path = source_file(source, f"{directory}/manifest.json")
    manifest = json.loads(manifest_path.read_text())
    source_model = str(manifest.get("model_id", ""))
    compatible, transfer = _compatible(source_model, model_id)
    if not compatible or (
        source_model != model_id and model_id not in manifest.get("compatible_models", [])
    ):
        raise ValueError(
            f"J-Lens manifest model {source_model!r} is incompatible with {model_id!r}"
        )
    lens_path = source_file(source, f"{directory}/{manifest['artifact']}")
    digest = _sha256(lens_path)
    if digest != manifest.get("sha256"):
        raise ValueError(
            f"J-Lens checksum mismatch for {lens_path}: {digest} != {manifest.get('sha256')}"
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


_artifact_cache: dict[str, dict[str, Any]] = {}


def load_j_lens(model_id: str) -> dict[str, Any] | None:
    """Load a compatible artifact, caching successes but retrying prior misses."""
    from llmdbg.residual_vocabulary_lenses import active_lens

    configured = active_lens("j")
    cache_key = f"{configured.get('id', 'default') if configured else 'default'}:{model_id}"
    if cache_key in _artifact_cache:
        return _artifact_cache[cache_key]
    local_candidates = (
        _local_candidates(model_id)
        if configured is None or configured.get("id") == "j"
        else ()
    )
    for path in local_candidates:
        if path.is_file():
            artifact = _load_local(path, model_id)
            _artifact_cache[cache_key] = artifact
            return artifact
    try:
        artifact = _download_from_hub(model_id)
    except OSError as error:
        raise RuntimeError(f"Could not download J-Lens data from the Hub: {error}") from error
    if artifact is not None:
        _artifact_cache[cache_key] = artifact
    return artifact


def _clear_cache() -> None:
    _artifact_cache.clear()


load_j_lens.cache_clear = _clear_cache  # type: ignore[attr-defined]


def source_map(model_id: str, layer: int, hidden_size: int) -> tuple[Any, dict[str, Any]] | None:
    """Return one validated ordinary-Jacobian transport map and metadata."""
    artifact = load_j_lens(model_id)
    if artifact is None:
        return None
    if int(artifact.get("d_model", 0)) != hidden_size:
        raise ValueError(
            f"J-Lens width {artifact.get('d_model')} does not match model width {hidden_size}"
        )
    maps = artifact["J"]
    matrix = maps.get(layer, maps.get(str(layer)))
    if matrix is None:
        return None
    if tuple(matrix.shape) != (hidden_size, hidden_size):
        raise ValueError(
            f"J-Lens layer {layer} map has shape {tuple(matrix.shape)}, "
            f"expected {(hidden_size, hidden_size)}"
        )
    provenance = dict(artifact.get("provenance", {}))
    provenance["path"] = artifact["_path"]
    provenance["transfer"] = artifact["_transfer"]
    return matrix, provenance
