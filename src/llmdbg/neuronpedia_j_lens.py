"""Adapter for Neuronpedia's Anthropic-format, Post-only Jacobian lenses."""

from __future__ import annotations

from pathlib import Path
from typing import Any

REPO_ID = "neuronpedia/jacobian-lens"
REVISION = "e166deb5ad540932100ef803c8c9bcc2a5453694"
SOURCE = f"hf://models/{REPO_ID}"

ARTIFACTS: dict[str, str] = {
    "google/gemma-2-2b-it": (
        "gemma-2-2b-it/jlens/Salesforce-wikitext/"
        "gemma-2-2b-it_jacobian_lens.pt"
    ),
    "openai-community/gpt2": (
        "gpt2-small/jlens/Salesforce-wikitext/gpt2_jacobian_lens.pt"
    ),
    "Qwen/Qwen3.5-0.8B": (
        "qwen3.5-0.8b/jlens/Salesforce-wikitext/"
        "Qwen3.5-0.8B_jacobian_lens.pt"
    ),
    "Qwen/Qwen3.5-2B": (
        "qwen3.5-2b/jlens/Salesforce-wikitext/Qwen3.5-2B_jacobian_lens.pt"
    ),
    "Qwen/Qwen3.5-2B-Base": (
        "qwen3.5-2b-pt/jlens/Salesforce-wikitext/"
        "Qwen3.5-2B-Base_jacobian_lens.pt"
    ),
    "Qwen/Qwen3.5-27B": (
        "qwen3.5-27b/jlens/Salesforce-wikitext/Qwen3.5-27B_jacobian_lens.pt"
    ),
}


def is_source(item: dict[str, Any] | None) -> bool:
    """Return whether a configured Lens entry selects the Neuronpedia adapter."""
    from llmdbg.neuronpedia_discovery import is_source as discovered_source

    return discovered_source(item, "j")


def artifact_path(model_id: str, item: dict[str, Any] | None = None) -> str | None:
    from llmdbg.neuronpedia_discovery import artifact_path as discover

    return discover(item, model_id, "j", known=ARTIFACTS)


def load(item: dict[str, Any], model_id: str) -> dict[str, Any] | None:
    """Download and validate one exact-checkpoint Neuronpedia J-Lens artifact."""
    import torch

    from llmdbg.residual_vocabulary_lenses import source_file

    filename = artifact_path(model_id, item)
    if filename is None:
        return None
    path = Path(source_file(item, filename))
    artifact = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(artifact, dict) or not isinstance(artifact.get("J"), dict):
        raise ValueError(f"invalid Neuronpedia J-Lens artifact at {path}: missing J maps")
    maps = artifact["J"]
    width = int(artifact.get("d_model", 0))
    if width <= 0 or any(tuple(matrix.shape) != (width, width) for matrix in maps.values()):
        raise ValueError(f"invalid Neuronpedia J-Lens tensor shapes at {path}")
    return {
        **artifact,
        "provenance": {
            "model_id": model_id,
            "source_repo_id": REPO_ID,
            "source_revision": str(item.get("revision") or "main"),
            "source_filename": filename,
            "capture_point": "post",
            "target": "final_block_output",
            "legacy": True,
        },
        "_path": str(path),
        "_transfer": None,
    }
