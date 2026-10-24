"""Cached discovery of model-specific Lens artifacts in Neuronpedia's Hub tree."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

REPO_ID = "neuronpedia/jacobian-lens"
SOURCE = f"hf://models/{REPO_ID}"


def is_source(item: dict[str, Any] | None, method: str) -> bool:
    return bool(
        item
        and str(item.get("method", "")).lower() == method
        and str(item.get("source", "")).strip().rstrip("/").lower() == SOURCE.lower()
    )


@lru_cache(maxsize=8)
def repository_files(revision: str) -> tuple[str, ...]:
    """List files once per revision; callers gracefully handle an offline cache miss."""
    from huggingface_hub import HfApi

    return tuple(
        entry.path
        for entry in HfApi().list_repo_tree(
            REPO_ID,
            recursive=True,
            revision=revision,
            repo_type="model",
        )
        if hasattr(entry, "size")
    )


def artifact_path(
    item: dict[str, Any] | None,
    model_id: str,
    method: str,
    *,
    known: dict[str, str] | None = None,
) -> str | None:
    """Find an exact model-folder artifact, retaining aliases for historical names."""
    normalized = model_id.lower()
    if known:
        fixed = next(
            (path for candidate, path in known.items() if candidate.lower() == normalized),
            None,
        )
        if fixed is not None:
            return fixed
    folder = model_id.rsplit("/", 1)[-1].lower()
    lens_dir = "jlens" if method == "j" else "jpp"
    suffix = "_jacobian_lens.pt" if method == "j" else "_jpp_lens.pt"
    revision = str((item or {}).get("revision") or "main")
    try:
        matches = [
            path for path in repository_files(revision)
            if path.lower().startswith(f"{folder}/{lens_dir}/")
            and path.lower().endswith(suffix)
        ]
    except Exception:
        return None
    preferred = [path for path in matches if "/salesforce-wikitext/" in path.lower()]
    candidates = preferred or matches
    return sorted(candidates)[0] if candidates else None
