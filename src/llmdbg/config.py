"""Read user-authored .llmdbg TOML launch configurations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib


def load_debug_config(path: str) -> dict[str, Any]:
    """Load a .llmdbg file and return flattened DAP launch arguments."""
    config_path = Path(path).expanduser().resolve()
    if config_path.suffix != ".llmdbg":
        raise ValueError(f"debug configuration must use the .llmdbg extension: {config_path}")
    if not config_path.is_file():
        raise FileNotFoundError(f"llmdbg configuration does not exist: {config_path}")

    with config_path.open("rb") as file:
        document = tomllib.load(file)

    model = document.get("model", {})
    input_config = document.get("input", {})
    debug = document.get("debug", {})
    if not isinstance(model, dict) or not isinstance(input_config, dict) or not isinstance(debug, dict):
        raise ValueError("[model], [input], and [debug] must be TOML tables")

    model_id = model.get("id")
    prompt = input_config.get("prompt")
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("[model].id must be a non-empty string")
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("[input].prompt must be a non-empty string")

    return {
        "configPath": str(config_path),
        "model": model_id,
        "revision": model.get("revision", "main"),
        "device": model.get("device", "cuda"),
        "dtype": model.get("dtype", "bfloat16"),
        "prompt": prompt,
        "stopOnEntry": debug.get("stop_on_entry", True),
    }
