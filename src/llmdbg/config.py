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
_QWEN_MLP_OPERATIONS = (
    "post_attention_layernorm", "gate_projection", "silu",
    "up_projection", "gated_multiply", "down_projection", "residual_add",
)
_GEMMA_MLP_OPERATIONS = (
    "pre_feedforward_layernorm", "gate_projection", "gelu",
    "up_projection", "gated_multiply", "down_projection",
    "post_feedforward_layernorm", "residual_add",
)
_FULL_ATTENTION_OPERATIONS = (
    "input_layernorm", "query_projection", "key_projection", "value_projection",
    "cached_keys", "cached_values", "attention_weights", "mix_values",
    "output_projection", "residual_add",
)
_FULL_ATTENTION_LEGACY_OPERATIONS = (
    "input_layernorm", "full_attention", "output_projection", "residual_add",
)
_GEMMA_ATTENTION_OPERATIONS = (
    "input_layernorm", "full_attention", "output_projection",
    "post_attention_layernorm", "residual_add",
)
_GEMMA4_ATTENTION_OPERATIONS = (
    "input_layernorm", "query_projection", "cached_keys", "cached_values",
    "attention_weights", "mix_values", "output_projection", "post_attention_layernorm", "residual_add",
)
_GATED_LINEAR_ATTENTION_OPERATIONS = (
    "input_layernorm", "gated_delta", "output_projection", "residual_add",
)
_FUNCTION_VARIANTS = {
    "mlp": (_MLP_OPERATIONS, _QWEN_MLP_OPERATIONS, _GEMMA_MLP_OPERATIONS),
    "attention": (_ATTENTION_OPERATIONS, ("query_projection", "key_projection", "value_projection", "cached_keys", "cached_values", "attention_weights", "mix_values")),
    "gated_delta": (("convolved_qkv", "gates", "cached_memory", "update_memory", "read_memory", "gated_output_norm"),),
    "full_attention": (
        _FULL_ATTENTION_OPERATIONS,
        _FULL_ATTENTION_LEGACY_OPERATIONS,
        _GEMMA_ATTENTION_OPERATIONS,
        _GEMMA4_ATTENTION_OPERATIONS,
        ("input_layernorm", "attention", "output_projection", "residual_add"),
    ),
    "gated_linear_attention": (_GATED_LINEAR_ATTENTION_OPERATIONS,),
}
_FUNCTION_OPERATIONS = {
    name: tuple(dict.fromkeys(operation for variant in variants for operation in variant))
    for name, variants in _FUNCTION_VARIANTS.items()
}
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
        "device": globals_.get("device", "auto"),
        "dtype": globals_.get("dtype", "auto"),
        "prompt": prompt,
        "defaultToken": default_token,
        "stopOnEntry": globals_.get("stop_on_entry", False),
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
    messages: list[dict[str, str]] | None = None
    add_generation_prompt = True
    max_new_tokens = 0
    token: int | None = None
    stop_on_entry = False
    functions: dict[str, dict[str, int]] = {}
    function_bindings: dict[str, list[dict[str, Any]]] = {}
    call_lines: dict[str, Any] = {}
    function_returns: dict[str, dict[str, Any]] = {}
    globals_: dict[str, Any] = {}

    for statement in tree.body:
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            target = statement.targets[0]
            if isinstance(target, ast.Name):
                if target.id == "model" and isinstance(statement.value, ast.Call):
                    model_call = statement.value
                elif target.id in {"prompt", "messages", "token", "stop_on_entry"}:
                    globals_[target.id] = _literal(statement.value, target.id)
        elif isinstance(statement, ast.FunctionDef) and statement.name in _FUNCTION_OPERATIONS:
            operations: dict[str, int] = {}
            bindings: list[dict[str, Any]] = []
            if statement.body and isinstance(statement.body[-1], ast.Return) and isinstance(statement.body[-1].value, ast.Name):
                function_returns[statement.name] = {"line": statement.body[-1].lineno, "name": statement.body[-1].value.id}
            for node in ast.walk(statement):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCTION_OPERATIONS[statement.name]:
                    operations.setdefault(node.func.id, node.lineno)
                if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.value, ast.Call):
                    operation = node.value.func.attr if isinstance(node.value.func, ast.Attribute) else node.value.func.id if isinstance(node.value.func, ast.Name) else ""
                    if operation in _FUNCTION_OPERATIONS[statement.name]:
                        if isinstance(node.targets[0], ast.Tuple):
                            vectors = {"convolved_qkv": ("delta_query", "delta_key", "delta_value"), "gates": ("delta_decay", "delta_beta", "delta_gate")}.get(operation, ())
                            for target, vector in zip(node.targets[0].elts, vectors):
                                if isinstance(target, ast.Name):
                                    bindings.append({"name": target.id, "operation": operation, "vector": vector, "line": node.lineno})
                        elif isinstance(node.targets[0], ast.Name) and isinstance(node.value.func, ast.Name):
                            bindings.append({"name": node.targets[0].id, "operation": operation, "line": node.lineno})
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
                elif (
                    isinstance(node, ast.Assign)
                    and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name)
                    and isinstance(node.value, ast.BinOp)
                    and isinstance(node.value.op, ast.Mult)
                ):
                    bindings.append(
                        {
                            "name": node.targets[0].id,
                            "operation": "gated_multiply",
                            "line": node.lineno,
                        }
                    )
                elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
                    operations.setdefault("residual_add", node.lineno)
                elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
                    operations.setdefault("gated_multiply", node.lineno)
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
                elif node.func.attr == "chat" and node.args:
                    value = (
                        globals_[node.args[0].id]
                        if isinstance(node.args[0], ast.Name) and node.args[0].id in globals_
                        else _literal(node.args[0], "messages")
                    )
                    if not isinstance(value, list) or not value:
                        raise ValueError("messages must be a non-empty list")
                    for message in value:
                        if (
                            not isinstance(message, dict)
                            or message.get("role") not in {"system", "user", "assistant"}
                            or not _valid_message_content(message.get("content"))
                        ):
                            raise ValueError(
                                "each message must contain a supported role and either string "
                                "content or image/text content parts"
                            )
                    messages = value
                    for keyword in node.keywords:
                        if keyword.arg == "add_generation_prompt":
                            add_generation_prompt = bool(
                                _literal(keyword.value, "add_generation_prompt")
                            )
                        elif keyword.arg == "max_new_tokens":
                            max_new_tokens = int(
                                _literal(keyword.value, "max_new_tokens")
                            )
                            if max_new_tokens < 0:
                                raise ValueError("max_new_tokens must be non-negative")
                elif node.func.attr in {
                    "token_embedding", "final_layernorm", "vocabulary_projection",
                    "top_tokens", "visual_residual", "per_layer_residual", "layer_scale",
                }:
                    call_lines.setdefault(node.func.attr, node.lineno)
            elif isinstance(node.func, ast.Name) and node.func.id in _FUNCTION_OPERATIONS:
                call_lines.setdefault(node.func.id, node.lineno)
            elif isinstance(node.func, ast.Name) and node.func.id == "nop":
                call_lines.setdefault("nop", []).append(node.lineno)
            elif isinstance(node.func, ast.Name) and node.func.id == "end":
                call_lines.setdefault("end", node.lineno)

    if model_call is None or not model_call.args:
        raise ValueError("Python llmdbg program must assign model = gpt2(\"model-id\", ...)")
    model = _literal(model_call.args[0], "model id")
    keywords = {keyword.arg: _literal(keyword.value, keyword.arg or "argument") for keyword in model_call.keywords}
    if not isinstance(model, str) or not model:
        raise ValueError("gpt2 model id must be a non-empty string")
    if prompt is None and messages is None:
        prompt = globals_.get("prompt")
    if messages is None and (not isinstance(prompt, str) or not prompt):
        raise ValueError("Python llmdbg program must tokenize a prompt or chat messages")
    model_kind = model_call.func.id if isinstance(model_call.func, ast.Name) else "gpt2"
    if model_kind not in {"gpt2", "qwen35", "qwen3vl", "gemma2", "gemma4"}:
        raise ValueError("model constructor must be gpt2(...), qwen35(...), qwen3vl(...), gemma2(...), or gemma4(...)")

    from llmdbg.source_plan import compile_plans
    program_calls = {node.func.id for statement in tree.body if not isinstance(statement,(ast.FunctionDef,ast.AsyncFunctionDef))
                     for node in ast.walk(statement) if isinstance(node,ast.Call) and isinstance(node.func,ast.Name)}
    source_plans = compile_plans(tree, model_kind, program_calls)
    reachable = {frame['name'] for plan in source_plans.values() for step in plan['steps'] for frame in step['frames']}
    for name, plan in source_plans.items():
        functions[name] = {step['operation']:step['line'] for step in plan['steps'] if step['kind']=='operation'}
    for name, variants in _FUNCTION_VARIANTS.items():
        if name in functions and name not in reachable and tuple(functions[name]) not in variants:
            expected = " or ".join(", ".join(variant) for variant in variants)
            raise ValueError(f"def {name}() operations must be, in order: {expected}")

    return {
        "configPath": str(path), "model": model,
        "revision": keywords.get("revision", "main"),
        "device": keywords.get("device", "auto"), "dtype": keywords.get("dtype", "auto"),
        "prompt": prompt or "", "messages": messages,
        "addGenerationPrompt": add_generation_prompt,
        "maxNewTokens": max_new_tokens,
        "defaultToken": token if token is not None else globals_.get("token"),
        "stopOnEntry": globals_.get("stop_on_entry", stop_on_entry),
        "pythonProgram": True, "callLines": call_lines, "functions": functions,
        "functionBindings": function_bindings, "functionReturns": function_returns, "modelKind": model_kind,
        "sourcePlans": source_plans,
    }


def _valid_message_content(content: object) -> bool:
    if isinstance(content, str):
        return True
    if not isinstance(content, list) or not content:
        return False
    for part in content:
        if not isinstance(part, dict) or part.get("type") not in {"text", "image"}:
            return False
        if part["type"] == "text" and not isinstance(part.get("text"), str):
            return False
        if part["type"] == "image" and not isinstance(
            part.get("url", part.get("path")), str
        ):
            return False
    return True
