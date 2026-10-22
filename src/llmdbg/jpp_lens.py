"""Load a local fitted J++ Lens or the official Qwen 3.5 9B artifact."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

SUPPORTED_MODEL = "Qwen/Qwen3.5-9B"
REPOSITORY = "koayon/jpp-lenses"
REVISION = "237dab6ea0e611d964bf407f13ba1655f1a6ae43"
FILENAME = "qwen3.5-9b/lens.pt"
SIZE = 2_080_385_107
SHA256 = "2b7e99817e66393a444e73d808a5ae79fadf87a2d085db789afb96b76d1d6e37"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _maps(artifact: dict[str, Any]) -> dict[Any, Any] | None:
    parameters = artifact.get("parameters")
    if isinstance(parameters, dict) and isinstance(parameters.get("jacobians"), dict):
        return parameters["jacobians"]
    if isinstance(artifact.get("jacobians"), dict):
        return artifact["jacobians"]
    return None


def _load(path: Path, model_id: str) -> dict[str, Any]:
    import torch

    artifact = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(artifact, dict) or not _maps(artifact):
        raise ValueError(f"invalid J++ Lens artifact at {path}: missing Jacobians")
    config = artifact.get("config", {})
    artifact_model = config.get("hf_model_name") if isinstance(config, dict) else None
    if artifact_model and artifact_model.lower() != model_id.lower():
        raise ValueError(
            f"J++ Lens artifact for {artifact_model!r} is incompatible with {model_id!r}"
        )
    artifact["_path"] = str(path)
    return artifact


_artifact_cache: dict[str, dict[str, Any]] = {}


def load_jpp_lens(model_id: str) -> dict[str, Any] | None:
    """Load an explicit local artifact, else the pinned official 9B artifact."""
    if model_id in _artifact_cache:
        return _artifact_cache[model_id]
    override = os.environ.get("LLMDBG_JPP_LENS_PATH")
    if override:
        path = Path(override).expanduser()
    else:
        if model_id.lower() != SUPPORTED_MODEL.lower():
            return None
        from huggingface_hub import hf_hub_download

        path = Path(
            hf_hub_download(
                repo_id=REPOSITORY,
                revision=REVISION,
                filename=FILENAME,
                library_name="llmdbg",
            )
        )
        if path.stat().st_size != SIZE:
            raise ValueError(
                f"J++ Lens size mismatch for {path}: {path.stat().st_size} != {SIZE}"
            )
        digest = _sha256(path)
        if digest != SHA256:
            raise ValueError(f"J++ Lens checksum mismatch for {path}: {digest} != {SHA256}")
    artifact = _load(path, model_id)
    artifact["_artifact_kind"] = "local" if override else "official"
    _artifact_cache[model_id] = artifact
    return artifact


def _clear_cache() -> None:
    _artifact_cache.clear()


load_jpp_lens.cache_clear = _clear_cache  # type: ignore[attr-defined]


def display_name(model_id: str) -> str:
    """Return a UI label that identifies the artifact's actual provenance."""
    artifact = load_jpp_lens(model_id)
    if artifact is None:
        return "J++ Lens"
    if artifact.get("_artifact_kind") == "official":
        return "Official J++ Lens"
    return "Local J++ Lens"


def configured_display_name(model_id: str) -> str:
    """Return the expected label without loading or downloading the artifact."""
    if os.environ.get("LLMDBG_JPP_LENS_PATH"):
        return "Local J++ Lens"
    if model_id.lower() == SUPPORTED_MODEL.lower():
        return "Official J++ Lens"
    return "J++ Lens"


def source_map(
    model_id: str, layer: int, hidden_size: int
) -> tuple[Any, dict[str, Any]] | None:
    """Return one validated J++ transport map and its provenance."""
    artifact = load_jpp_lens(model_id)
    if artifact is None:
        return None
    maps = _maps(artifact)
    assert maps is not None
    matrix = maps.get(layer, maps.get(str(layer)))
    if matrix is None:
        return None
    if tuple(matrix.shape) != (hidden_size, hidden_size):
        raise ValueError(
            f"J++ Lens layer {layer} map has shape {tuple(matrix.shape)}, "
            f"expected {(hidden_size, hidden_size)}"
        )
    artifact_kind = str(artifact.get("_artifact_kind", "local"))
    return matrix, {
        "model_id": model_id,
        "method": "jpp",
        "artifact_kind": artifact_kind,
        "display_name": display_name(model_id),
        "repository": REPOSITORY if artifact_kind == "official" else None,
        "revision": REVISION if artifact_kind == "official" else None,
        "path": artifact["_path"],
        "readout_filter": "contains_letter_or_digit",
    }
