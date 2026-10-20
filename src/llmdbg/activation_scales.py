"""Packaged empirical scales for model-internal activation vectors."""

from __future__ import annotations

import json
import re
from functools import lru_cache
from importlib.resources import files
from typing import Any


@lru_cache(maxsize=1)
def _qwen35_0_8b_scales() -> dict[int, dict[str, Any]]:
    path = files("llmdbg").joinpath("data/qwen35_0_8b_mlp_activation_scales.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {int(item["layer"]): item for item in payload["layers"]}


def mlp_activation_scale(model_id: str, layer_name: str) -> dict[str, Any] | None:
    """Return the measured post-SwiGLU scale for a supported model layer."""
    if model_id != "Qwen/Qwen3.5-0.8B":
        return None
    match = re.fullmatch(r"model\.layers\.(\d+)\.mlp", layer_name)
    if match is None:
        return None
    return _qwen35_0_8b_scales().get(int(match.group(1)))
