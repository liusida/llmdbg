"""Parse the compact, ordered llmdbg source language."""

from __future__ import annotations

import json
import ast
import re
from pathlib import Path
from typing import Any


_ASSIGNMENT = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+?)\s*$")
_BLOCK_SUBMODULE = re.compile(r"^transformer\.h\.(\d+)\.(attention|mlp)$")
_FUNCTION = re.compile(r"^def\s+([A-Za-z_][A-Za-z0-9_]*)\(\):\s*$")
_LAYER_CALL = re.compile(r"^(attention|mlp)\(\)\s*$")
_MLP_OPERATIONS = ("layernorm", "expansion", "gelu", "projection", "residual_add")
_ATTENTION_OPERATIONS = (
    "layernorm",
    "qkv_projection",
    "attention_mixing",
    "output_projection",
    "residual_add",
)
_FUNCTION_OPERATIONS = {"mlp": _MLP_OPERATIONS, "attention": _ATTENTION_OPERATIONS}
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
    if config_path.name.endswith(".llmdbg.py"):
        return _load_python_debug_config(config_path)
    if config_path.suffix != ".llmdbg":
        raise ValueError(f"debug source must use .llmdbg or .llmdbg.py: {config_path}")
    if not config_path.is_file():
        raise FileNotFoundError(f"llmdbg source does not exist: {config_path}")

    globals_: dict[str, Any] = {}
    step_paths: list[str] = []
    step_lines: dict[str, int] = {}
    current_layer: int | None = None
    function_name: str | None = None
    functions: dict[str, dict[str, int]] = {}
    function_bindings: dict[str, list[dict[str, Any]]] = {}

    for line_number, raw_line in enumerate(
        config_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        function_match = _FUNCTION.fullmatch(line)
        if function_match:
            function_name = function_match.group(1)
            if function_name not in _FUNCTION_OPERATIONS:
                raise ValueError(
                    f"line {line_number}: only def attention(): and def mlp(): are supported"
                )
            if function_name in functions:
                raise ValueError(f"line {line_number}: duplicate function {function_name!r}")
            functions[function_name] = {}
            continue
        if raw_line[:1].isspace():
            if function_name is None:
                raise ValueError(f"line {line_number}: indented operation is outside a function")
            if line not in _FUNCTION_OPERATIONS[function_name]:
                raise ValueError(
                    f"line {line_number}: unknown {function_name} operation {line!r}"
                )
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
    if "attention" in functions and tuple(functions["attention"]) != _ATTENTION_OPERATIONS:
        raise ValueError(
            "def attention() operations must be, in order: "
            + ", ".join(_ATTENTION_OPERATIONS)
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


def _literal(node: ast.AST, name: str) -> Any:
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError) as error:
        raise ValueError(f"{name} must be a Python literal") from error


def _load_python_debug_config(path: Path) -> dict[str, Any]:
    """Read a real-Python llmdbg program without importing or executing user code."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    model_call: ast.Call | None = None
    prompt: str | None = None
    token: int | None = None
    stop_on_entry = True
    functions: dict[str, dict[str, int]] = {}
    function_bindings: dict[str, list[dict[str, Any]]] = {}
    call_lines: dict[str, int] = {}
    globals_: dict[str, Any] = {}

    for statement in tree.body:
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            target = statement.targets[0]
            if isinstance(target, ast.Name):
                if target.id == "model" and isinstance(statement.value, ast.Call):
                    model_call = statement.value
                elif target.id in {"prompt", "token", "stop_on_entry"}:
                    globals_[target.id] = _literal(statement.value, target.id)
        elif isinstance(statement, ast.FunctionDef) and statement.name in _FUNCTION_OPERATIONS:
            operations: dict[str, int] = {}
            bindings: list[dict[str, Any]] = []
            for node in ast.walk(statement):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    operation = node.func.attr
                    if operation in _FUNCTION_OPERATIONS[statement.name]:
                        operations.setdefault(operation, node.lineno)
                if (
                    isinstance(node, ast.Assign)
                    and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name)
                    and isinstance(node.value, ast.Call)
                    and isinstance(node.value.func, ast.Attribute)
                    and node.value.func.attr in _FUNCTION_OPERATIONS[statement.name]
                ):
                    bindings.append(
                        {
                            "name": node.targets[0].id,
                            "operation": node.value.func.attr,
                            "line": node.lineno,
                        }
                    )
                elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
                    operations.setdefault("residual_add", node.lineno)
            functions[statement.name] = operations
            function_bindings[statement.name] = sorted(bindings, key=lambda item: item["line"])

    for node in ast.walk(tree):
        if isinstance(node, ast.With):
            for item in node.items:
                if isinstance(item.context_expr, ast.Call) and isinstance(item.context_expr.func, ast.Name) and item.context_expr.func.id == "debug":
                    for keyword in item.context_expr.keywords:
                        if keyword.arg == "token":
                            if "token" in globals_:
                                token = int(globals_["token"])
                            elif isinstance(keyword.value, ast.Name) and keyword.value.id == "token":
                                # Allow `# token = 3` while leaving `token=token`
                                # in the debug call: no value means all positions.
                                token = None
                            else:
                                token = int(_literal(keyword.value, "token"))
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                if node.func.attr == "tokenize" and node.args:
                    value = node.args[0]
                    prompt = str(
                        globals_["prompt"]
                        if "prompt" in globals_
                        else _literal(value, "prompt")
                    )
                elif node.func.attr in {"token_embedding", "final_layernorm"}:
                    call_lines.setdefault(node.func.attr, node.lineno)
            elif isinstance(node.func, ast.Name) and node.func.id in {"attention", "mlp"}:
                call_lines.setdefault(node.func.id, node.lineno)

    if model_call is None or not model_call.args:
        raise ValueError("Python llmdbg program must assign model = gpt2(\"model-id\", ...)")
    model = _literal(model_call.args[0], "model id")
    keywords = {keyword.arg: _literal(keyword.value, keyword.arg or "argument") for keyword in model_call.keywords}
    if not isinstance(model, str) or not model:
        raise ValueError("gpt2 model id must be a non-empty string")
    if prompt is None:
        prompt = globals_.get("prompt")
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("Python llmdbg program must tokenize a non-empty prompt")
    for name, expected in _FUNCTION_OPERATIONS.items():
        if name in functions and tuple(functions[name]) != expected:
            raise ValueError(f"def {name}() operations must be, in order: {', '.join(expected)}")

    return {
        "configPath": str(path), "model": model,
        "revision": keywords.get("revision", "main"),
        "device": keywords.get("device", "cuda"), "dtype": keywords.get("dtype", "bfloat16"),
        "prompt": prompt, "defaultToken": token if token is not None else globals_.get("token"),
        "stopOnEntry": globals_.get("stop_on_entry", stop_on_entry),
        "pythonProgram": True, "callLines": call_lines, "functions": functions,
        "functionBindings": function_bindings,
    }
