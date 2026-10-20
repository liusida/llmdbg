"""A deterministic model used to exercise the debugger without dependencies."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Layer:
    """One inspectable transformation in the demo model."""

    name: str
    operation: str
    meaning: str
    scale: float
    bias: float


@dataclass(frozen=True)
class LayerState:
    """The input, output, and interpretation of one layer."""

    layer: Layer
    input_vector: tuple[float, ...]
    output_vector: tuple[float, ...]
    interpretation: str


LAYERS = (
    Layer(
        "token_embedding",
        "embedding lookup + positional signal",
        "Encodes token identity and position into the residual stream.",
        1.00,
        0.08,
    ),
    Layer(
        "attention_0",
        "causal self-attention",
        "Moves context-related information into the active token representation.",
        1.18,
        -0.03,
    ),
    Layer(
        "mlp_0",
        "feed-forward feature expansion",
        "Strengthens detected semantic features and suppresses weak ones.",
        1.42,
        0.02,
    ),
    Layer(
        "attention_1",
        "causal self-attention",
        "Combines the refined features with information from earlier tokens.",
        1.09,
        0.01,
    ),
    Layer(
        "lm_head",
        "projection to token logits",
        "Converts the final representation into next-token evidence.",
        1.31,
        -0.04,
    ),
)

FEATURES = (
    "syntax",
    "entity",
    "sentiment",
    "position",
    "instruction",
    "context",
    "prediction",
    "uncertainty",
)


def _transform(vector: tuple[float, ...], layer: Layer, index: int) -> tuple[float, ...]:
    """Apply a stable nonlinear transform that resembles a layer update."""
    size = len(vector)
    return tuple(
        math.tanh(
            layer.scale * value
            + layer.bias
            + 0.12 * vector[(position - 1) % size]
            - 0.05 * vector[(position + index + 1) % size]
        )
        for position, value in enumerate(vector)
    )


def _interpret(vector: tuple[float, ...], layer: Layer) -> str:
    ranked = sorted(zip(FEATURES, vector), key=lambda item: abs(item[1]), reverse=True)
    dominant = ", ".join(f"{name}={value:+.2f}" for name, value in ranked[:3])
    return f"{layer.meaning} Dominant dimensions: {dominant}."


def run_demo() -> tuple[LayerState, ...]:
    """Return the complete trace for a tiny deterministic language model."""
    vector = (0.62, -0.15, 0.31, 0.77, 0.48, -0.22, 0.19, 0.36)
    states: list[LayerState] = []
    for index, layer in enumerate(LAYERS):
        output = _transform(vector, layer, index)
        states.append(LayerState(layer, vector, output, _interpret(output, layer)))
        vector = output
    return tuple(states)
