"""Shared site-aware loading and resolution for J, R, and J++ lenses."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

Method = Literal["j", "r", "jpp"]
Site = Literal["embed", "mid", "post"]

HUB_REPO_ID = "sida/llmdbg"
HUB_REPO_TYPE = "dataset"
FORMAT_VERSION = 2


@dataclass(frozen=True)
class LensSource:
    """A resolved transport map and its exact activation-site provenance."""

    matrix: Any
    metadata: dict[str, Any]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_view_site(view: str, layer: int) -> tuple[Site, int | None]:
    """Map an llmdbg view to the activation site where its vector is injected."""
    view = view.removeprefix("ica-")
    if view in {"residual-pre", "residual-input", "residual-stream"}:
        return ("embed", None) if layer == 0 else ("post", layer - 1)
    if view in {"residual-mid", "attention-output", "attention-neurons"}:
        return "mid", layer
    if view in {"residual-post", "residual-output", "mlp-output", "neurons"}:
        return "post", layer
    raise ValueError(f"unknown Lens view {view!r}")


def tensor_key(site: Site, layer: int | None) -> str:
    if site == "embed":
        if layer is not None:
            raise ValueError("embed transport must not specify a layer")
        return "embed"
    if layer is None or layer < 0:
        raise ValueError(f"{site} transport requires a non-negative layer")
    return f"{site}.{layer}"


def _load_v2_directory(
    path: Path,
    model_id: str,
    method: Method,
    progress: Callable[[str], None],
) -> dict[str, Any]:
    from safetensors.torch import load_file

    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if int(manifest.get("format_version", 0)) != FORMAT_VERSION:
        raise ValueError(f"unsupported Workspace Lens format at {manifest_path}")
    if str(manifest.get("model_id", "")).lower() != model_id.lower():
        raise ValueError(
            f"Workspace Lens artifact for {manifest.get('model_id')!r} is incompatible "
            f"with {model_id!r}"
        )
    if manifest.get("method") != method:
        raise ValueError(
            f"Workspace Lens method {manifest.get('method')!r} does not match {method!r}"
        )
    tensor_path = path / str(manifest["artifact"])
    expected = manifest.get("sha256")
    if expected and _sha256(tensor_path) != expected:
        raise ValueError(f"Workspace Lens checksum mismatch for {tensor_path}")
    size_mib = tensor_path.stat().st_size / (1024 * 1024)
    progress(
        f"Loading {method.upper()} Lens tensors from {tensor_path} "
        f"({size_mib:.1f} MiB)...\n"
    )
    tensors = load_file(tensor_path, device="cpu")
    progress(f"Loaded {method.upper()} Lens with {len(tensors)} site maps.\n")
    return {"manifest": manifest, "tensors": tensors}


_cache: dict[tuple[str, str, str], dict[str, Any] | None] = {}


def load_v2(
    model_id: str,
    method: Method,
    *,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any] | None:
    """Load an exact-checkpoint v2 artifact from an override or the Hub index."""
    from llmdbg.residual_vocabulary_lenses import active_lens, source_file

    configured = active_lens(method)
    configured_id = str(configured.get("id", "default")) if configured else "default"
    key = (model_id.lower(), method, configured_id)
    if key in _cache:
        return _cache[key]
    if method == "j":
        from llmdbg.neuronpedia_j_lens import is_source as is_neuronpedia_source

        if is_neuronpedia_source(configured):
            # Neuronpedia publishes Anthropic-format block-output maps. They
            # are resolved by the explicit legacy Post-only adapter below.
            _cache[key] = None
            return None
    if method == "jpp":
        from llmdbg.neuronpedia_jpp_lens import is_source as is_neuronpedia_jpp_source

        if is_neuronpedia_jpp_source(configured):
            _cache[key] = None
            return None
    report = progress or (lambda _message: None)
    report(
        f"Resolving {method.upper()} Lens artifact for {model_id}; "
        "uncached files may be downloaded.\n"
    )
    override = os.environ.get("LLMDBG_WORKSPACE_LENS_PATH")
    if override:
        root = Path(override).expanduser()
        candidates = (
            root / model_id.replace("/", "--") / method,
            root / method,
            root,
        )
        for candidate in candidates:
            if (candidate / "manifest.json").is_file():
                loaded = _load_v2_directory(candidate, model_id, method, report)
                _cache[key] = loaded
                return loaded
    try:
        source = configured or {
            "source": f"hf://{HUB_REPO_TYPE}s/{HUB_REPO_ID}",
        }
        index_path = source_file(source, "index.json")
        index = json.loads(index_path.read_text())
        section = str(configured.get("indexSection", "workspace_lenses")) if configured else "workspace_lenses"
        # The built-in J/R entries also serve legacy artifacts from j_lenses /
        # r_lenses. Their v2 maps live in workspace_lenses in the same repo.
        if section in {"j_lenses", "r_lenses"}:
            section = "workspace_lenses"
        entry = next(
            (
                item
                for item in index.get(section, [])
                if str(item.get("model_id", "")).lower() == model_id.lower()
                and item.get("method") == method
                and int(item.get("format_version", 0)) == FORMAT_VERSION
            ),
            None,
        )
        if entry is None:
            report(f"No compatible site-aware {method.upper()} Lens artifact was found.\n")
            _cache[key] = None
            return None
        directory = str(entry["path"]).strip("/")
        manifest_path = source_file(source, f"{directory}/manifest.json")
        manifest = json.loads(manifest_path.read_text())
        tensor_path = source_file(source, f"{directory}/{manifest['artifact']}")
        loaded = _load_v2_directory(tensor_path.parent, model_id, method, report)
        _cache[key] = loaded
        return loaded
    except OSError as error:
        configured_source = str(configured.get("source", "")) if configured else ""
        if configured_source and not configured_source.startswith("hf://"):
            _cache[key] = None
            report(f"No compatible site-aware {method.upper()} Lens artifact was found.\n")
            return None
        raise RuntimeError(f"Could not download Workspace Lens data: {error}") from error


def source_map(
    method: Method,
    model_id: str,
    site: Site,
    layer: int | None,
    hidden_size: int,
    *,
    layer_count: int,
    model_revision: str | None = None,
    tokenizer_revision: str | None = None,
    progress: Callable[[str], None] | None = None,
) -> LensSource | None:
    """Resolve a v2 map; the final Post site is exactly the identity transport."""
    if site == "post" and layer == layer_count - 1:
        import torch

        return LensSource(
            torch.eye(hidden_size),
            {
                "format_version": FORMAT_VERSION,
                "model_id": model_id,
                "method": method,
                "site": site,
                "layer": layer,
                "artifact_kind": "identity",
            },
        )
    artifact = load_v2(model_id, method, progress=progress)
    if artifact is None:
        return None
    manifest = artifact["manifest"]
    for label, expected, actual in (
        ("model", manifest.get("model_revision"), model_revision),
        ("tokenizer", manifest.get("tokenizer_revision"), tokenizer_revision),
    ):
        if expected and actual and str(expected) != str(actual):
            raise ValueError(
                f"{method.upper()} Lens {label} revision {expected!r} does not match "
                f"the loaded checkpoint revision {actual!r}"
            )
    key = tensor_key(site, layer)
    matrix = artifact["tensors"].get(key)
    fallback_from: str | None = None
    if matrix is None and method in {"j", "jpp"} and site == "mid" and layer is not None:
        fallback_from = key
        key = tensor_key("post", layer)
        matrix = artifact["tensors"].get(key)
        if matrix is None and layer == layer_count - 1:
            import torch

            matrix = torch.eye(hidden_size)
    if matrix is None:
        return None
    if tuple(matrix.shape) != (hidden_size, hidden_size):
        raise ValueError(
            f"{method.upper()} Lens {key} map has shape {tuple(matrix.shape)}, "
            f"expected {(hidden_size, hidden_size)}"
        )
    metadata = dict(artifact["manifest"])
    metadata.update({"site": key.split(".", 1)[0], "layer": layer, "tensor_key": key})
    if fallback_from is not None and layer == layer_count - 1:
        metadata["artifact_kind"] = "identity"
    if fallback_from is not None:
        metadata.update({
            "fallback": True,
            "fallback_from_site": site,
            "fallback_from_tensor_key": fallback_from,
        })
    return LensSource(matrix, metadata)


def clear_cache() -> None:
    _cache.clear()

