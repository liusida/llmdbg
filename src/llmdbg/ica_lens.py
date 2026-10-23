"""Published ICA bases used by the optional ICA sections in the Lens panel."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Callable


_ARTIFACTS = {
    "openai-community/gpt2": "sida/icalens-gpt2-small-pile10k",
    "gpt2": "sida/icalens-gpt2-small-pile10k",
    "qwen/qwen3.5-2b": "sida/icalens-qwen3.5-2b-ultrachat-1m",
    "qwen/qwen3.5-2b-base": "sida/icalens-qwen3.5-2b-base-pile10k",
    "qwen/qwen3.5-9b": "sida/icalens-qwen3.5-9b-base-pile10k",
    "qwen/qwen3.5-9b-base": "sida/icalens-qwen3.5-9b-base-pile10k",
}


def artifact_repository(model_id: str) -> str | None:
    """Return the published ICA repository compatible with ``model_id``."""
    normalized = model_id.strip().rstrip("/").lower()
    return _ARTIFACTS.get(normalized)


_artifact_cache: OrderedDict[tuple[str, int], tuple[Any, Any, Any]] = OrderedDict()


def _download_layer_artifact(repository: str, layer: int) -> tuple[Any, Any, Any]:
    """Download and cache one layer's center/read/write matrices as NumPy arrays."""
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    path = hf_hub_download(
        repository,
        filename=f"artifacts/resid_post/layer_{layer:02d}.safetensors",
    )
    with safe_open(path, framework="numpy") as handle:
        return (
            handle.get_tensor("center"),
            handle.get_tensor("reading_matrix"),
            handle.get_tensor("writing_matrix"),
        )


def _layer_artifact(
    repository: str,
    layer: int,
    progress: Callable[[str], None] | None = None,
) -> tuple[Any, Any, Any]:
    """Load an ICA layer while announcing a cache miss before slow I/O."""
    report = progress or (lambda _message: None)
    key = (repository, layer)
    cached = _artifact_cache.get(key)
    if cached is not None:
        _artifact_cache.move_to_end(key)
        return cached
    report(
        f"Resolving ICA basis for layer {layer} from {repository}; "
        "uncached files may be downloaded.\n"
    )
    artifact = _download_layer_artifact(repository, layer)
    _artifact_cache[key] = artifact
    _artifact_cache.move_to_end(key)
    while len(_artifact_cache) > 2:
        _artifact_cache.popitem(last=False)
    size_mib = sum(array.nbytes for array in artifact) / (1024 * 1024)
    report(f"Loaded ICA basis for layer {layer} ({size_mib:.1f} MiB).\n")
    return artifact


def clear_cache() -> None:
    _artifact_cache.clear()


def top_components(
    model_id: str,
    layer: int,
    vector: tuple[float, ...],
    *,
    is_delta: bool,
    count: int = 3,
    include_indices: tuple[int, ...] = (),
    progress: Callable[[str], None] | None = None,
) -> tuple[dict[str, Any], ...]:
    """Decompose a residual state or update in a layer's resid-post ICA basis.

    Residual states use the fitted affine transform. Updates are displacement
    vectors, so they use only the transform's linear part and are not centered.
    """
    import numpy as np

    repository = artifact_repository(model_id)
    if repository is None:
        return ()
    center, reading, writing = _layer_artifact(repository, layer, progress)
    values = np.asarray(vector, dtype=np.float32)
    if values.shape != center.shape:
        raise ValueError(
            f"ICA artifact expects {center.size} dimensions, got {values.size}"
        )
    scores = (values if is_delta else values - center) @ reading.T
    ranked = (
        np.argsort(np.abs(scores))[-min(count, scores.size) :][::-1].tolist()
        if count > 0
        else []
    )
    indices = ranked + [index for index in include_indices if index not in ranked]
    total_energy = float(np.square(scores, dtype=np.float64).sum())
    results = []
    for rank, index in enumerate(indices, start=1):
        if index < 0 or index >= scores.size:
            raise ValueError(f"ICA component {index} is outside range 0..{scores.size - 1}")
        score = float(scores[index])
        component = writing[:, index].astype(np.float32, copy=False) * score
        results.append(
            {
                "rank": rank,
                "index": int(index),
                "score": score,
                "energy": (
                    float(score * score / total_energy) if total_energy > 0.0 else 0.0
                ),
                "vector": tuple(float(value) for value in component),
            }
        )
    return tuple(results)


def remove_component(
    model_id: str,
    layer: int,
    tensor: Any,
    component: int,
    *,
    progress: Callable[[str], None] | None = None,
) -> Any:
    """Zero one ICA score and reconstruct a residual tensor in place."""
    import torch

    repository = artifact_repository(model_id)
    if repository is None:
        raise ValueError(f"ICA intervention is not available for {model_id}")
    center, reading, writing = _layer_artifact(repository, layer, progress)
    if component < 0 or component >= reading.shape[0]:
        raise ValueError(
            f"ICA component {component} is outside range 0..{reading.shape[0] - 1}"
        )
    center_tensor = torch.as_tensor(center, device=tensor.device, dtype=tensor.dtype)
    reading_tensor = torch.as_tensor(reading, device=tensor.device, dtype=tensor.dtype)
    writing_tensor = torch.as_tensor(writing, device=tensor.device, dtype=tensor.dtype)
    scores = (tensor - center_tensor) @ reading_tensor.T
    scores[..., component] = 0
    tensor.copy_(center_tensor + scores @ writing_tensor.T)
    return tensor
