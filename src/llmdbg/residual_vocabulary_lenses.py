"""Configuration registry for Residual-to-Vocabulary Lens artifact sources."""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator

ENVIRONMENT_VARIABLE = "LLMDBG_RESIDUAL_TO_VOCABULARY_LENSES"

DEFAULT_LENSES: tuple[dict[str, Any], ...] = (
    {
        "id": "j",
        "label": "J-Lens",
        "method": "j",
        "source": "hf://datasets/sida/llmdbg",
        "indexSection": "j_lenses",
        "enabled": True,
    },
    {
        "id": "r",
        "label": "R-Lens",
        "method": "r",
        "source": "hf://datasets/sida/llmdbg",
        "indexSection": "r_lenses",
        "enabled": True,
    },
    {
        "id": "jpp-official",
        "label": "Official J++ Lens",
        "method": "jpp",
        "source": "hf://models/koayon/jpp-lenses",
        "revision": "237dab6ea0e611d964bf407f13ba1655f1a6ae43",
        "path": "qwen3.5-9b/lens.pt",
        "models": ["Qwen/Qwen3.5-9B"],
        "enabled": True,
    },
    {
        "id": "j-neuronpedia",
        "label": "Neuronpedia J Lens",
        "method": "j",
        "source": "hf://models/neuronpedia/jacobian-lens",
        "enabled": True,
    },
    {
        "id": "jpp-neuronpedia",
        "label": "Neuronpedia J++ Lens",
        "method": "jpp",
        "source": "hf://models/neuronpedia/jacobian-lens",
        "enabled": True,
    },
)

_active_id: ContextVar[str | dict[str, str] | None] = ContextVar(
    "llmdbg_residual_vocabulary_lens", default=None
)


def configured_lenses() -> tuple[dict[str, Any], ...]:
    raw = os.environ.get(ENVIRONMENT_VARIABLE)
    values: Any = json.loads(raw) if raw else DEFAULT_LENSES
    if not isinstance(values, list | tuple):
        raise ValueError("Residual-to-Vocabulary Lenses must be an array")
    result = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, dict):
            raise ValueError("each Residual-to-Vocabulary Lens must be an object")
        item = dict(value)
        lens_id = str(item.get("id", "")).strip()
        method = str(item.get("method", "")).strip().lower()
        if not lens_id or lens_id in seen:
            raise ValueError(f"invalid or duplicate Residual-to-Vocabulary Lens ID {lens_id!r}")
        if method not in {"j", "r", "jpp"}:
            raise ValueError(f"Lens {lens_id!r} has unsupported method {method!r}")
        if not str(item.get("source", "")).strip():
            raise ValueError(f"Lens {lens_id!r} has no source")
        seen.add(lens_id)
        item.update({"id": lens_id, "method": method, "label": str(item.get("label") or lens_id)})
        result.append(item)
    return tuple(result)


def lens(lens_id: str) -> dict[str, Any] | None:
    return next((item for item in configured_lenses() if item["id"] == lens_id), None)


def active_lens(method: str) -> dict[str, Any] | None:
    selected = _active_id.get()
    if isinstance(selected, dict):
        selected = selected.get(method)
    entries = configured_lenses()
    if selected:
        item = next((entry for entry in entries if entry["id"] == selected), None)
        if item is not None and item["method"] == method:
            return item
    return next(
        (entry for entry in entries if entry["method"] == method and entry.get("enabled", True)),
        None,
    )


@contextmanager
def activate(lens_id: str | dict[str, str] | None) -> Iterator[None]:
    token = _active_id.set(lens_id)
    try:
        yield
    finally:
        _active_id.reset(token)


def source_parts(item: dict[str, Any]) -> tuple[str, str]:
    source = str(item["source"]).strip().rstrip("/")
    if source.startswith("hf://datasets/"):
        return source.removeprefix("hf://datasets/"), "dataset"
    if source.startswith("hf://models/"):
        return source.removeprefix("hf://models/"), "model"
    raise ValueError(
        f"unsupported remote Lens source {source!r}; use hf://datasets/... or hf://models/..."
    )


def source_file(
    item: dict[str, Any],
    filename: str,
    *,
    library_name: str = "llmdbg",
) -> Path:
    """Resolve one file from a Hub source or a local repository-shaped directory."""
    source = str(item["source"]).strip()
    if source.startswith(("hf://datasets/", "hf://models/")):
        from huggingface_hub import hf_hub_download

        repo_id, repo_type = source_parts(item)
        kwargs: dict[str, Any] = {
            "repo_id": repo_id,
            "repo_type": repo_type,
            "filename": filename,
            "library_name": library_name,
        }
        if item.get("revision"):
            kwargs["revision"] = str(item["revision"])
        return Path(hf_hub_download(**kwargs))
    if source.startswith("file://"):
        source = source.removeprefix("file://")
    root = Path(source).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"local Lens source directory does not exist: {root}")
    target = (root / filename).resolve()
    if not target.is_relative_to(root):
        raise ValueError(f"Lens artifact path escapes local source directory: {filename!r}")
    if not target.is_file():
        raise FileNotFoundError(f"local Lens source file does not exist: {target}")
    return target


@lru_cache(maxsize=32)
def available_lenses(model_id: str) -> tuple[dict[str, str], ...]:
    """Return enabled entries declared compatible with the active model.

    Entries without a model allowlist remain candidates; their downloaded
    manifest is still validated before use.
    """
    normalized = model_id.lower()
    result = []
    for item in configured_lenses():
        if not item.get("enabled", True):
            continue
        local_jpp = item["method"] == "jpp" and bool(os.environ.get("LLMDBG_JPP_LENS_PATH"))
        models = item.get("models")
        if not local_jpp and isinstance(models, list) and not any(
            str(candidate).lower() in {normalized, "*"} for candidate in models
        ):
            continue
        from llmdbg.neuronpedia_discovery import is_source as is_neuronpedia_source

        if is_neuronpedia_source(item, item["method"]):
            if not isinstance(models, list):
                if item["method"] == "j":
                    from llmdbg.neuronpedia_j_lens import artifact_path
                else:
                    from llmdbg.neuronpedia_jpp_lens import artifact_path
                if artifact_path(model_id, item) is None:
                    continue
            result.append({"id": item["id"], "label": item["label"], "method": item["method"]})
            continue
        if not isinstance(models, list):
            try:
                index = json.loads(source_file(item, "index.json").read_text())
                sections = [str(item.get("indexSection", "workspace_lenses"))]
                if "workspace_lenses" not in sections:
                    sections.append("workspace_lenses")
                candidates = [
                    entry
                    for section in sections
                    for entry in index.get(section, [])
                    if section != "workspace_lenses"
                    or str(entry.get("method", "")).lower() == item["method"]
                ]
                if not any(
                    str(entry.get("model_id", "")).lower() == normalized
                    or normalized in {str(value).lower() for value in entry.get("compatible_models", [])}
                    for entry in candidates
                ):
                    continue
            except (OSError, ValueError, KeyError, json.JSONDecodeError):
                source = str(item.get("source", ""))
                local_root = Path(source.removeprefix("file://")).expanduser()
                if item["method"] != "jpp" or not (local_root / "lens.pt").is_file():
                    continue
                fit_manifest = local_root / "fit-manifest.json"
                if fit_manifest.is_file():
                    try:
                        fitted_model = str(
                            json.loads(fit_manifest.read_text()).get("model_id", "")
                        ).lower()
                    except (OSError, json.JSONDecodeError):
                        continue
                    if fitted_model and fitted_model != normalized:
                        continue
        result.append({
            "id": item["id"],
            "label": "Local J++ Lens" if local_jpp else item["label"],
            "method": item["method"],
        })
    return tuple(result)
