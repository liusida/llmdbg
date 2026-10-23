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
    from llmdbg.residual_vocabulary_lenses import active_lens, source_file

    configured = active_lens("jpp")
    configured_id = str(configured.get("id", "default")) if configured else "default"
    cache_key = f"{configured_id}:{model_id}"
    if cache_key in _artifact_cache:
        return _artifact_cache[cache_key]
    override = os.environ.get("LLMDBG_JPP_LENS_PATH")
    if override:
        path = Path(override).expanduser()
    else:
        models = configured.get("models") if configured else [SUPPORTED_MODEL]
        if isinstance(models, list) and not any(
            str(candidate).lower() in {model_id.lower(), "*"} for candidate in models
        ):
            return None
        configured_source = str(configured.get("source", "")) if configured else ""
        is_local_source = bool(configured_source) and not configured_source.startswith("hf://")
        if configured and not configured.get("path") and not is_local_source:
            return None
        filename = (
            str(configured.get("path") or "lens.pt")
            if configured
            else FILENAME
        )
        source = configured or {
            "source": f"hf://models/{REPOSITORY}",
            "revision": REVISION,
        }
        path = source_file(source, filename)
        expected_size = configured.get("size") if configured else SIZE
        if expected_size and path.stat().st_size != int(expected_size):
            raise ValueError(
                f"J++ Lens size mismatch for {path}: {path.stat().st_size} != {expected_size}"
            )
        expected_sha = configured.get("sha256") if configured else SHA256
        if expected_sha:
            digest = _sha256(path)
            if digest != expected_sha:
                raise ValueError(f"J++ Lens checksum mismatch for {path}: {digest} != {expected_sha}")
    artifact = _load(path, model_id)
    artifact["_artifact_kind"] = (
        "local" if override or (configured_source and not configured_source.startswith("hf://"))
        else "official"
    )
    artifact["_configured_id"] = configured_id
    artifact["_configured_label"] = (
        "Local J++ Lens"
        if override
        else str(configured.get("label", "J++ Lens")) if configured else "Official J++ Lens"
    )
    _artifact_cache[cache_key] = artifact
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
    return str(artifact.get("_configured_label") or "Local J++ Lens")


def configured_display_name(model_id: str) -> str:
    """Return the expected label without loading or downloading the artifact."""
    from llmdbg.residual_vocabulary_lenses import active_lens

    if os.environ.get("LLMDBG_JPP_LENS_PATH"):
        return "Local J++ Lens"
    configured = active_lens("jpp")
    if configured is not None:
        return str(configured.get("label") or "J++ Lens")
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
