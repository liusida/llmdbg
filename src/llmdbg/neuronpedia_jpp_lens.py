"""Adapter for Neuronpedia's Post-only J++ Lens artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

REPO_ID = "neuronpedia/jacobian-lens"
REVISION = "e166deb5ad540932100ef803c8c9bcc2a5453694"
SOURCE = f"hf://models/{REPO_ID}"

ARTIFACTS: dict[str, str] = {
    "Qwen/Qwen3.5-0.8B": (
        "qwen3.5-0.8b/jpp/Salesforce-wikitext/"
        "Qwen3.5-0.8B_jpp_lens.pt"
    ),
    "Qwen/Qwen3.5-2B": (
        "qwen3.5-2b/jpp/Salesforce-wikitext/Qwen3.5-2B_jpp_lens.pt"
    ),
    "Qwen/Qwen3.5-9B": (
        "qwen3.5-9b/jpp/Salesforce-wikitext/Qwen3.5-9B_jpp_lens.pt"
    ),
}


def is_source(item: dict[str, Any] | None) -> bool:
    """Return whether a configured Lens entry selects this adapter."""
    from llmdbg.neuronpedia_discovery import is_source as discovered_source

    return discovered_source(item, "jpp")


def artifact_path(model_id: str, item: dict[str, Any] | None = None) -> str | None:
    from llmdbg.neuronpedia_discovery import artifact_path as discover

    return discover(item, model_id, "jpp", known=ARTIFACTS)


def load(item: dict[str, Any], model_id: str) -> dict[str, Any] | None:
    """Download and validate one exact-checkpoint Neuronpedia J++ artifact."""
    import torch

    from llmdbg.residual_vocabulary_lenses import source_file

    filename = artifact_path(model_id, item)
    if filename is None:
        return None
    path = Path(source_file(item, filename))
    # This is a compact Lens artifact rather than model weights; mmap keeps its
    # matrices CPU-backed until an individual layer is moved for a readout.
    artifact = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(artifact, dict) or not isinstance(artifact.get("J"), dict):
        raise ValueError(f"invalid Neuronpedia J++ Lens artifact at {path}: missing J maps")
    maps = artifact["J"]
    width = int(artifact.get("d_model", 0))
    if width <= 0 or any(tuple(matrix.shape) != (width, width) for matrix in maps.values()):
        raise ValueError(f"invalid Neuronpedia J++ Lens tensor shapes at {path}")
    return {
        **artifact,
        "provenance": {
            **(artifact.get("provenance") or {}),
            "model_id": model_id,
            "source_repo_id": REPO_ID,
            "source_revision": str(item.get("revision") or "main"),
            "source_filename": filename,
            "capture_point": "post",
            "target": "final_block_output",
            "legacy": True,
        },
        "_path": str(path),
    }
