"""Packaged empirical scales for model-internal activation vectors."""

from __future__ import annotations

import json
import re
from functools import lru_cache
from importlib.resources import files
from typing import Any


_SCALE_FILES = {
    "openai-community/gpt2": "gpt2_mlp_activation_scales.json",
    "Qwen/Qwen3.5-0.8B": "qwen35_0_8b_mlp_activation_scales.json",
    "Qwen/Qwen3.5-2B": "qwen35_2b_mlp_activation_scales.json",
    "Qwen/Qwen3.5-9B": "qwen35_9b_mlp_activation_scales.json",
    "Qwen/Qwen3.5-27B": "qwen35_27b_mlp_activation_scales.json",
}


@lru_cache(maxsize=len(_SCALE_FILES))
def _model_scales(model_id: str) -> dict[int, dict[str, Any]]:
    path = files("llmdbg").joinpath(f"data/{_SCALE_FILES[model_id]}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {int(item["layer"]): item for item in payload["layers"]}


def mlp_activation_scale(model_id: str, layer_name: str) -> dict[str, Any] | None:
    """Return the measured post-SwiGLU scale for a supported model layer."""
    if model_id not in _SCALE_FILES:
        return None
    pattern = (
        r"transformer\.h\.(\d+)\.mlp"
        if model_id == "openai-community/gpt2"
        else r"model\.layers\.(\d+)\.mlp"
    )
    match = re.fullmatch(pattern, layer_name)
    if match is None:
        return None
    return _model_scales(model_id).get(int(match.group(1)))
