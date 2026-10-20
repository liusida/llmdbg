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
    steps = document.get("step", [])
    if not isinstance(model, dict) or not isinstance(input_config, dict) or not isinstance(debug, dict):
        raise ValueError("[model], [input], and [debug] must be TOML tables")
    if not isinstance(steps, list) or not steps:
        raise ValueError("at least one [[step]] table is required")

    model_id = model.get("id")
    prompt = input_config.get("prompt")
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("[model].id must be a non-empty string")
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("[input].prompt must be a non-empty string")

    step_paths: list[str] = []
    for index, step in enumerate(steps):
        path_value = step.get("path") if isinstance(step, dict) else None
        if not isinstance(path_value, str) or not path_value:
            raise ValueError(f"[[step]] #{index + 1} must have a non-empty path")
        step_paths.append(path_value)
    if len(set(step_paths)) != len(step_paths):
        raise ValueError("[[step]].path values must be unique")

    step_lines: dict[str, int] = {}
    for line_number, line in enumerate(config_path.read_text(encoding="utf-8").splitlines(), start=1):
        if line.lstrip().startswith("path") and "=" in line:
            try:
                value = tomllib.loads(line).get("path")
            except tomllib.TOMLDecodeError:
                continue
            if isinstance(value, str) and value in step_paths:
                step_lines[value] = line_number
    if len(step_lines) != len(step_paths):
        raise ValueError("each [[step]].path must appear on its own line")

    return {
        "configPath": str(config_path),
        "model": model_id,
        "revision": model.get("revision", "main"),
        "device": model.get("device", "cuda"),
        "dtype": model.get("dtype", "bfloat16"),
        "prompt": prompt,
        "stopOnEntry": debug.get("stop_on_entry", True),
        "stepPaths": step_paths,
        "stepLines": step_lines,
    }
