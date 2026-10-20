"""Parse the compact, ordered llmdbg source language."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


_ASSIGNMENT = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+?)\s*$")
_BLOCK_SUBMODULE = re.compile(r"^transformer\.h\.(\d+)\.(attention|mlp)$")
_FUNCTION = re.compile(r"^def\s+([A-Za-z_][A-Za-z0-9_]*)\(\):\s*$")
_LAYER_CALL = re.compile(r"^(attention|mlp)\(\)\s*$")
_MLP_OPERATIONS = ("layernorm", "expansion", "gelu", "projection", "residual_add")
_GLOBAL_KEYS = {
    "model",
    "revision",
    "device",
    "dtype",
    "prompt",
    "token",
    "stop_on_entry",
}


def _parse_value(source: str, line_number: int) -> Any:
    if source == "true":
        return True
    if source == "false":
        return False
    if re.fullmatch(r"-?\d+", source):
        return int(source)
    try:
        value = json.loads(source)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"line {line_number}: values must be integers, booleans, or double-quoted strings"
        ) from error
    if not isinstance(value, str):
        raise ValueError(f"line {line_number}: unsupported value {source!r}")
    return value


def load_debug_config(path: str) -> dict[str, Any]:
    """Load an ordered .llmdbg program and return flattened DAP arguments."""
    config_path = Path(path).expanduser().resolve()
    if config_path.suffix != ".llmdbg":
        raise ValueError(f"debug source must use the .llmdbg extension: {config_path}")
    if not config_path.is_file():
        raise FileNotFoundError(f"llmdbg source does not exist: {config_path}")

    globals_: dict[str, Any] = {}
    step_paths: list[str] = []
    step_lines: dict[str, int] = {}
    current_layer: int | None = None
    function_name: str | None = None
    functions: dict[str, dict[str, int]] = {}

    for line_number, raw_line in enumerate(
        config_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        function_match = _FUNCTION.fullmatch(line)
        if function_match:
            function_name = function_match.group(1)
            if function_name != "mlp":
                raise ValueError(f"line {line_number}: only def mlp(): is currently supported")
            if function_name in functions:
                raise ValueError(f"line {line_number}: duplicate function {function_name!r}")
            functions[function_name] = {}
            continue
        if raw_line[:1].isspace():
            if function_name is None:
                raise ValueError(f"line {line_number}: indented operation is outside a function")
            if line not in _MLP_OPERATIONS:
                raise ValueError(f"line {line_number}: unknown MLP operation {line!r}")
            if line in functions[function_name]:
                raise ValueError(f"line {line_number}: duplicate MLP operation {line!r}")
            functions[function_name][line] = line_number
            continue
        function_name = None
        call_match = _LAYER_CALL.fullmatch(line)
        if call_match:
            if current_layer is None:
                raise ValueError(
                    f"line {line_number}: {call_match.group(1)}() requires layer = N first"
                )
            path = f"transformer.h.{current_layer}.{call_match.group(1)}"
            if path in step_lines:
                raise ValueError(f"line {line_number}: duplicate submodule {path!r}")
            step_paths.append(path)
            step_lines[path] = line_number
            continue
        match = _ASSIGNMENT.fullmatch(line)
        if not match:
            raise ValueError(f"line {line_number}: expected name = value")
        name, raw_value = match.groups()
        value = _parse_value(raw_value, line_number)

        if name in _GLOBAL_KEYS:
            if name in globals_:
                raise ValueError(f"line {line_number}: {name} may only be assigned once")
            globals_[name] = value
        elif name == "layer":
            if not isinstance(value, int) or value < 0:
                raise ValueError(f"line {line_number}: layer must be a non-negative integer")
            current_layer = value
        elif name == "submodule":
            if not isinstance(value, str) or not value:
                raise ValueError(f"line {line_number}: submodule must be a non-empty string")
            block_match = _BLOCK_SUBMODULE.fullmatch(value)
            if block_match and current_layer != int(block_match.group(1)):
                raise ValueError(
                    f"line {line_number}: {value} does not belong to layer {current_layer}"
                )
            if value in step_lines:
                raise ValueError(f"line {line_number}: duplicate submodule {value!r}")
            step_paths.append(value)
            step_lines[value] = line_number
        else:
            raise ValueError(f"line {line_number}: unknown llmdbg statement {name!r}")

    model = globals_.get("model")
    prompt = globals_.get("prompt")
    if not isinstance(model, str) or not model:
        raise ValueError('model must be assigned a non-empty string, for example model = "gpt2"')
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("prompt must be assigned a non-empty string")
    default_token = globals_.get("token")
    if default_token is not None and (
        not isinstance(default_token, int) or isinstance(default_token, bool) or default_token < 0
    ):
        raise ValueError("token must be assigned a non-negative integer")
    if not step_paths:
        raise ValueError("at least one submodule statement is required")
    if "mlp" in functions and tuple(functions["mlp"]) != _MLP_OPERATIONS:
        raise ValueError(
            "def mlp() operations must be, in order: " + ", ".join(_MLP_OPERATIONS)
        )

    return {
        "configPath": str(config_path),
        "model": model,
        "revision": globals_.get("revision", "main"),
        "device": globals_.get("device", "cuda"),
        "dtype": globals_.get("dtype", "bfloat16"),
        "prompt": prompt,
        "defaultToken": default_token,
        "stopOnEntry": globals_.get("stop_on_entry", True),
        "stepPaths": step_paths,
        "stepLines": step_lines,
        "functions": functions,
    }
