"""Packaged empirical scales for additive component contributions."""

from __future__ import annotations

import json
import math
import re
from functools import lru_cache
from importlib.resources import files
from typing import Any


_SCALE_FILES = {
    "google/gemma-4-E2B-it": "gemma4_e2b_it_mlp_activation_scales.json",
    "openai-community/gpt2": "gpt2_mlp_activation_scales.json",
    "openai-community/gpt2-medium": "gpt2_medium_mlp_activation_scales.json",
    "Qwen/Qwen3.5-0.8B": "qwen35_0_8b_mlp_activation_scales.json",
    "Qwen/Qwen3.5-2B": "qwen35_2b_mlp_activation_scales.json",
    "Qwen/Qwen3.5-9B": "qwen35_9b_mlp_activation_scales.json",
    "Qwen/Qwen3.5-27B": "qwen35_27b_mlp_activation_scales.json",
}

_ATTENTION_SCALE_FILES = {
    "google/gemma-2-2b-it": "gemma2_2b_it_attention_channel_activation_scales.json",
    "Qwen/Qwen3-VL-2B-Instruct": "qwen3_vl_2b_instruct_attention_channel_activation_scales.json",
    "google/gemma-4-E2B-it": "gemma4_e2b_it_attention_channel_activation_scales.json",
    "openai-community/gpt2": "gpt2_attention_channel_activation_scales.json",
    "openai-community/gpt2-medium": "gpt2_medium_attention_channel_activation_scales.json",
    "Qwen/Qwen3.5-0.8B": "qwen35_0_8b_attention_channel_activation_scales.json",
    "Qwen/Qwen3.5-2B": "qwen35_2b_attention_channel_activation_scales.json",
    "Qwen/Qwen3.5-2B-Base": "qwen35_2b_base_attention_channel_activation_scales.json",
    "Qwen/Qwen3.5-9B": "qwen35_9b_attention_channel_activation_scales.json",
    "Qwen/Qwen3.5-27B": "qwen35_27b_attention_channel_activation_scales.json",
}


@lru_cache(maxsize=len(_SCALE_FILES))
def _model_scales(model_id: str) -> dict[int, dict[str, Any]]:
    path = files("llmdbg").joinpath(f"data/{_SCALE_FILES[model_id]}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {int(item["layer"]): item for item in payload["layers"]}


def mlp_activation_scale(model_id: str, layer_name: str) -> dict[str, Any] | None:
    """Return the measured MLP residual-contribution scale for a model layer."""
    if model_id not in _SCALE_FILES:
        return None
    pattern = (
        r"transformer\.h\.(\d+)\.mlp"
        if model_id in {"openai-community/gpt2", "openai-community/gpt2-medium"}
        else r"model\.layers\.(\d+)\.mlp"
    )
    match = re.fullmatch(pattern, layer_name)
    if match is None:
        return None
    return _model_scales(model_id).get(int(match.group(1)))


@lru_cache(maxsize=len(_ATTENTION_SCALE_FILES))
def _model_attention_scales(model_id: str) -> dict[int, dict[str, Any]]:
    path = files("llmdbg").joinpath(f"data/{_ATTENTION_SCALE_FILES[model_id]}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {int(item["layer"]): item for item in payload["layers"]}


def attention_channel_activation_scale(
    model_id: str, layer_name: str
) -> dict[str, Any] | None:
    """Return the measured attention-channel residual-contribution scale."""
    if model_id not in _ATTENTION_SCALE_FILES:
        return None
    pattern = (
        r"transformer\.h\.(\d+)\.attention"
        if model_id in {"openai-community/gpt2", "openai-community/gpt2-medium"}
        else r"model\.layers\.(\d+)\.(?:full_attention|gated_linear_attention)"
    )
    match = re.fullmatch(pattern, layer_name)
    if match is None:
        return None
    return _model_attention_scales(model_id).get(int(match.group(1)))


def attention_head_norm_scale(model_id: str, layer_name: str, head: int, *, value: bool = False) -> dict[str, Any] | None:
    """Return an exact per-head reference, never substitute a channel/layer scale.

    With value=True, use the separate unweighted-value reference. Never fall
    back to mixed-output references. Older packaged calibrations return None.
    """
    record = attention_channel_activation_scale(model_id, layer_name)
    if record is None or type(head) is not int or head < 0:
        return None
    for item in record.get("value_head_norm_scales" if value else "head_norm_scales", []):
        if item.get("head") == head:
            scale = item.get("scale")
            return item if isinstance(scale, (int, float)) and math.isfinite(scale) and scale > 0 else None
    return None
