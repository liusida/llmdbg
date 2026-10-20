"""Minimal Debug Adapter Protocol implementation for llmdbg."""

from __future__ import annotations

import json
import ast
import base64
import io
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

from llmdbg.config import load_debug_config
from llmdbg.demo_model import FEATURES, LayerState, run_demo
from llmdbg.model_runtime import ModelExecution, NeuronProjection, prepare_model_execution


@dataclass(frozen=True)
class SourceBreakpoint:
    """A model-step source line with an optional token-position condition."""

    line: int
    token_position: int | None = None


class ProtocolStream:
    """Read and write Content-Length framed DAP messages."""

    def __init__(self, reader: BinaryIO, writer: BinaryIO) -> None:
        self.reader = reader
        self.writer = writer

    def read(self) -> dict[str, Any] | None:
        headers: dict[str, str] = {}
        while True:
            line = self.reader.readline()
            if not line:
                return None
            if line in (b"\r\n", b"\n"):
                break
            name, value = line.decode("ascii").split(":", 1)
            headers[name.lower()] = value.strip()
        length = int(headers["content-length"])
        payload = self.reader.read(length)
        if len(payload) != length:
            raise EOFError("incomplete DAP message")
        return json.loads(payload.decode("utf-8"))

    def write(self, message: dict[str, Any]) -> None:
        payload = json.dumps(message, separators=(",", ":")).encode("utf-8")
        self.writer.write(f"Content-Length: {len(payload)}\r\n\r\n".encode("ascii"))
        self.writer.write(payload)
        self.writer.flush()


class LLMDebugAdapter:
    """Expose a layer trace through the core DAP requests used by VS Code."""

    THREAD_ID = 1
    FRAME_ID = 1
    SCOPE_TOKEN = 50
    SCOPE_TOKENS = 60
    SCOPE_LAYER = 100
    SCOPE_INPUT = 200
    SCOPE_RESIDUAL = 300
    SCOPE_DELTA = 400
    SCOPE_TOP_NEURONS = 1000
    NEURON_REFERENCE = 1100
    TOKEN_REFERENCE = 1200
    LOCAL_VECTOR_REFERENCE = 2000
    DETAIL_SCOPES = {
        500: ("normalized_input", "LayerNorm output"),
        600: ("mlp_expansion_pre_activation", "MLP expansion (pre-activation)"),
        700: ("mlp_expansion", "MLP expansion (after activation)"),
        800: ("mlp_projection", "MLP projection"),
        900: ("normalized_output", "Normalized output"),
        1300: ("attention_qkv", "QKV projection"),
        1400: ("attention_mixing", "Attention mixing"),
        1500: ("attention_projection", "Attention output projection"),
    }
    SOURCE_REFERENCE = 1
    MLP_FRAME_ID = 2
    MLP_STEPS = (
        ("layernorm", "normalized_input", "Normalize the residual stream for the MLP."),
        ("expansion", "mlp_expansion_pre_activation", "Expand hidden width with W_up + bias."),
        ("gelu", "mlp_expansion", "Apply the GELU nonlinearity."),
        ("projection", "mlp_projection", "Project active neurons back with W_down."),
        ("residual_add", "output", "Add the MLP update to the residual stream."),
    )
    ATTENTION_STEPS = (
        ("layernorm", "normalized_input", "Normalize the residual stream for attention."),
        ("qkv_projection", "attention_qkv", "Project the input into query, key, and value vectors."),
        ("attention_mixing", "attention_mixing", "Mix value vectors across cached token positions and heads."),
        ("output_projection", "attention_projection", "Project concatenated attention heads back to model width."),
        ("residual_add", "output", "Add the attention update to the residual stream."),
    )

    def __init__(self, stream: ProtocolStream) -> None:
        self.stream = stream
        self.sequence = 1
        self.trace: tuple[LayerState, ...] = ()
        self.execution: ModelExecution | None = None
        self.tokens: tuple[str, ...] = ()
        self.prompt = ""
        self.model_name = "built-in demo"
        self.token_index = 0
        self.default_token: int | None = None
        self.layer_index = 0
        self.launched = False
        self.stop_on_entry = True
        self.source_content = ""
        self.source_lines: tuple[int, ...] = ()
        self.step_paths: tuple[str, ...] = ()
        self.source_name = "model.execution.llmdbg"
        self.source_path: str | None = None
        self.breakpoints: tuple[SourceBreakpoint, ...] = ()
        self.neuron_cache: dict[tuple[int, str], tuple[NeuronProjection, ...]] = {}
        self.inside_mlp = False
        self.mlp_step_index = 0
        self.mlp_source_content = ""
        self.mlp_source_lines: tuple[int, ...] = ()
        self.inside_attention = False
        self.attention_source_lines: tuple[int, ...] = ()
        self.function_bindings: dict[str, list[dict[str, Any]]] = {}

    @property
    def internal_steps(self) -> tuple[tuple[str, str, str], ...]:
        return self.ATTENTION_STEPS if getattr(self, "inside_attention", False) else self.MLP_STEPS

    @property
    def inside_internal(self) -> bool:
        return getattr(self, "inside_mlp", False) or getattr(self, "inside_attention", False)

    @property
    def state(self) -> LayerState:
        outer = self.trace[self.layer_index]
        if not getattr(self, "inside_internal", False):
            return outer
        step_name, _vector_name, meaning = self.internal_steps[self.mlp_step_index]
        input_vector = self._mlp_step_input(self.mlp_step_index)
        return LayerState(
            layer=outer.layer.__class__(
                f"{outer.layer.name}.{step_name}",
                step_name,
                meaning,
                1.0,
                0.0,
            ),
            input_vector=input_vector,
            output_vector=input_vector,
            interpretation=meaning,
        )

    @property
    def previous_state(self) -> LayerState | None:
        if getattr(self, "inside_internal", False):
            if self.mlp_step_index == 0:
                return None
            outer = self.trace[self.layer_index]
            step_name, vector_name, meaning = self.internal_steps[self.mlp_step_index - 1]
            output = self._mlp_vector(vector_name)
            input_vector = self._mlp_step_input(self.mlp_step_index - 1)
            details = {} if vector_name == "output" else {vector_name: output}
            return LayerState(
                layer=outer.layer.__class__(
                    f"{outer.layer.name}.{step_name}",
                    step_name,
                    meaning,
                    1.0,
                    0.0,
                ),
                input_vector=input_vector,
                output_vector=output,
                interpretation=meaning,
                details=details,
            )
        if self.layer_index == 0:
            return None
        return self.trace[self.layer_index - 1]

    @property
    def outer_state(self) -> LayerState:
        return self.trace[self.layer_index]

    def _mlp_vector(self, vector_name: str) -> tuple[float, ...]:
        outer = self.outer_state
        return outer.output_vector if vector_name == "output" else outer.details[vector_name]

    def _mlp_step_input(self, step_index: int) -> tuple[float, ...]:
        if step_index == 0:
            return self.outer_state.input_vector
        return self._mlp_vector(self.internal_steps[step_index - 1][1])

    def send_response(
        self,
        request: dict[str, Any],
        body: dict[str, Any] | None = None,
        *,
        success: bool = True,
        message: str | None = None,
    ) -> None:
        response: dict[str, Any] = {
            "seq": self.sequence,
            "type": "response",
            "request_seq": request["seq"],
            "success": success,
            "command": request["command"],
        }
        if body is not None:
            response["body"] = body
        if message is not None:
            response["message"] = message
        self.sequence += 1
        self.stream.write(response)

    def send_event(self, event: str, body: dict[str, Any] | None = None) -> None:
        message: dict[str, Any] = {
            "seq": self.sequence,
            "type": "event",
            "event": event,
        }
        if body is not None:
            message["body"] = body
        self.sequence += 1
        self.stream.write(message)

    def stop(self, reason: str) -> None:
        self.send_event(
            "stopped",
            {
                "reason": reason,
                "description": (
                    f"Stopped at token {self.token_index}, {self.state.layer.name}"
                ),
                "threadId": self.THREAD_ID,
                "allThreadsStopped": True,
            },
        )
        self.send_event(
            "llmdbg/vectorChanged",
            {"tokenPosition": self.token_index, "layer": self.state.layer.name},
        )

    def terminate(self) -> None:
        self.send_event("output", {"category": "console", "output": "llmdbg: model trace complete\n"})
        self.send_event("terminated")

    def handle(self, request: dict[str, Any]) -> bool:
        command = request.get("command", "")
        arguments = request.get("arguments") or {}

        if command == "initialize":
            self.send_response(
                request,
                {
                    "supportsConfigurationDoneRequest": True,
                    "supportsEvaluateForHovers": True,
                    "supportsConditionalBreakpoints": True,
                    "supportsSetExpression": True,
                    "supportsSetVariable": True,
                    "supportsInvalidatedEvent": True,
                    "supportsStepBack": False,
                    "supportsRestartRequest": False,
                },
            )
        elif command == "launch":
            if arguments.get("config"):
                file_arguments = load_debug_config(str(arguments["config"]))
                arguments = {**file_arguments, **arguments}
            model = arguments.get("model")
            self.model_name = str(model) if model else "built-in demo"
            self.prompt = str(arguments.get("prompt", "The capital of France is"))
            if model:
                self.send_event(
                    "output",
                    {"category": "console", "output": f"llmdbg: loading {model}\n"},
                )
                self.execution = prepare_model_execution(
                    str(model),
                    revision=str(arguments.get("revision", "main")),
                    prompt=self.prompt,
                    device=str(arguments.get("device", "cuda")),
                    dtype=str(arguments.get("dtype", "bfloat16")),
                    progress=lambda output: self.send_event(
                        "output", {"category": "console", "output": f"llmdbg: {output}"}
                    ),
                )
                self.tokens = self.execution.tokens
                self._configure_model_source(arguments)
                loaded_model = self.execution.runtime.model
                model_bytes = int(loaded_model.get_memory_footprint(return_buffers=True))
                cuda_allocated_bytes = 0
                cuda_reserved_bytes = 0
                if str(arguments.get("device", "cuda")).startswith("cuda"):
                    import torch

                    if torch.cuda.is_available():
                        cuda_allocated_bytes = int(torch.cuda.memory_allocated())
                        cuda_reserved_bytes = int(torch.cuda.memory_reserved())
                self.send_event(
                    "llmdbg/modelLoaded",
                    {
                        "model": str(model),
                        "revision": str(arguments.get("revision", "main")),
                        "device": str(arguments.get("device", "cuda")),
                        "dtype": str(arguments.get("dtype", "bfloat16")),
                        "modelBytes": model_bytes,
                        "cudaAllocatedBytes": cuda_allocated_bytes,
                        "cudaReservedBytes": cuda_reserved_bytes,
                    },
                )
            else:
                self.trace = run_demo()
                self.tokens = ("demo",)
                self._build_virtual_source(arguments, "built-in-demo", ())
            self.token_index = 0
            self.default_token = arguments.get("defaultToken")
            self.layer_index = 0
            self.launched = True
            self.stop_on_entry = bool(arguments.get("stopOnEntry", True))
            self.send_response(request)
            self.send_event("initialized")
            trace_name = str(model) if model else "built-in demo"
            self.send_event(
                "output",
                {
                    "category": "console",
                    "output": (
                        f"llmdbg: prepared {trace_name}; forward execution will begin "
                        "after breakpoints are configured\n"
                    ),
                },
            )
        elif command == "configurationDone":
            self.send_response(request)
            if not self.launched:
                return True
            if self.stop_on_entry:
                self._load_token(0)
                self.layer_index = 0
                self.stop("entry")
            else:
                self._continue_to_breakpoint(include_current=True, before_start=True)
        elif command == "threads":
            self.send_response(request, {"threads": [{"id": self.THREAD_ID, "name": "LLM forward pass"}]})
        elif command == "stackTrace":
            source: dict[str, Any] = {"name": self.source_name}
            if self.source_path:
                source["path"] = self.source_path
            else:
                source.update(
                    {
                        "sourceReference": self.SOURCE_REFERENCE,
                        "origin": "generated by llmdbg",
                    }
                )
            frame = {
                "id": self.FRAME_ID,
                "name": (
                    f"token {self.token_index}: {self.tokens[self.token_index]!r} · "
                    f"{self.layer_index:02d}: {self.state.layer.name}"
                ),
                "source": source,
                "line": (
                    (self.attention_source_lines if self.inside_attention else self.mlp_source_lines)[self.mlp_step_index]
                    if self.inside_internal else self.source_lines[self.layer_index]
                ),
                "column": 1,
                "presentationHint": "normal",
            }
            if self.inside_internal:
                frame["id"] = self.MLP_FRAME_ID
                frame["name"] = f"{self.outer_state.layer.name} · {self.internal_steps[self.mlp_step_index][0]}"
                parent = {
                    "id": self.FRAME_ID,
                    "name": f"token {self.token_index}: {self.tokens[self.token_index]!r} · {self.outer_state.layer.name}",
                    "source": source,
                    "line": self.source_lines[self.layer_index],
                    "column": 1,
                }
                self.send_response(request, {"stackFrames": [frame, parent], "totalFrames": 2})
            else:
                self.send_response(request, {"stackFrames": [frame], "totalFrames": 1})
        elif command == "source":
            reference = int(arguments.get("sourceReference", 0))
            if reference != self.SOURCE_REFERENCE:
                self.send_response(request, success=False, message="Unknown source reference")
            else:
                self.send_response(
                    request,
                    {"content": self.source_content, "mimeType": "text/plain"},
                )
        elif command == "scopes":
            scopes = [
                {"name": "Local variables", "variablesReference": self.SCOPE_TOKEN, "expensive": False},
            ]
            result_state = self.previous_state
            if result_state is not None:
                scopes.extend(
                    [
                        {
                            "name": f"Previous output — {result_state.layer.name}",
                            "variablesReference": self.SCOPE_RESIDUAL,
                            "expensive": False,
                        },
                        {"name": "Previous delta", "variablesReference": self.SCOPE_DELTA, "expensive": False},
                    ]
                )
                scopes.extend(
                    {
                        "name": label,
                        "variablesReference": reference,
                        "expensive": len(result_state.details[detail_name]) > 1024,
                    }
                    for reference, (detail_name, label) in self.DETAIL_SCOPES.items()
                    if detail_name in result_state.details
                )
            if (
                result_state is not None
                and "mlp_expansion" in result_state.details
                and self.execution is not None
            ):
                scopes.append(
                    {
                        "name": "Top activated neurons",
                        "variablesReference": self.SCOPE_TOP_NEURONS,
                        "expensive": True,
                    }
                )
            self.send_response(request, {"scopes": scopes})
        elif command == "variables":
            reference = int(arguments.get("variablesReference", 0))
            self.send_response(request, {"variables": self._variables(reference)})
        elif command == "setVariable":
            reference = int(arguments.get("variablesReference", 0))
            name = str(arguments.get("name", ""))
            expression = self._variable_expression(reference, name)
            value = float(str(arguments.get("value", "")).strip())
            updated = self._set_expression(expression, value)
            self.send_response(
                request,
                {"value": f"{updated:+.8f}", "type": "float", "variablesReference": 0},
            )
            self.send_event("invalidated", {"areas": ["variables", "watch"]})
        elif command == "evaluate":
            expression = str(arguments.get("expression", "")).strip().lower()
            body = self._evaluate_body(expression)
            if body is None:
                self.send_response(request, success=False, message=f"Unknown expression: {expression}")
            else:
                self.send_response(request, body)
        elif command == "setExpression":
            expression = str(arguments.get("expression", "")).strip().lower()
            value = float(str(arguments.get("value", "")).strip())
            updated = self._set_expression(expression, value)
            self.send_response(
                request,
                {"value": f"{updated:+.8f}", "type": "float", "variablesReference": 0},
            )
            self.send_event("invalidated", {"areas": ["variables", "watch"]})
        elif command == "llmdbg/vector":
            self.send_response(request, self._vector_snapshot())
        elif command == "llmdbg/plot":
            self.send_response(
                request,
                self._render_plot(
                    str(arguments.get("expression", "residual_stream")),
                    str(arguments.get("code", "")),
                ),
            )
        elif command == "llmdbg/neuronLens":
            self.send_response(
                request,
                self._neuron_lens_snapshot(),
            )
        elif command in ("next", "stepIn", "stepOut"):
            self.send_response(request)
            can_enter_mlp = self.outer_state.layer.name.endswith(".mlp") and self.mlp_source_lines
            can_enter_attention = self.outer_state.layer.name.endswith(".attention") and self.attention_source_lines
            if command == "stepIn" and not self.inside_internal and (can_enter_mlp or can_enter_attention):
                self.inside_mlp = bool(can_enter_mlp)
                self.inside_attention = bool(can_enter_attention)
                self.mlp_step_index = 0
                self.stop("step")
            elif command == "next" and not self.inside_internal and self._stop_at_internal_breakpoint():
                pass
            elif self.inside_internal and command != "stepOut" and self.mlp_step_index + 1 < len(self.internal_steps):
                self.mlp_step_index += 1
                self.stop("step")
            else:
                self.inside_mlp = False
                self.inside_attention = False
                self.mlp_step_index = 0
                self._advance_outer()
        elif command == "continue":
            self.send_response(request, {"allThreadsContinued": True})
            self._continue_to_breakpoint(include_current=False)
        elif command == "pause":
            self.send_response(request)
            self.stop("pause")
        elif command == "setBreakpoints":
            requested = [int(item.get("line", 0)) for item in arguments.get("breakpoints", [])]
            valid_lines = sorted(
                set(self.source_lines)
                | set(self.mlp_source_lines)
                | set(self.attention_source_lines)
            )
            resolved = [
                min(valid_lines, key=lambda candidate: abs(candidate - line))
                if valid_lines and min(abs(candidate - line) for candidate in valid_lines) <= 2
                else None
                for line in requested
            ]
            configured: list[SourceBreakpoint] = []
            breakpoints = []
            requested_items = arguments.get("breakpoints", [])
            for item, requested_line, resolved_line in zip(
                requested_items, requested, resolved, strict=True
            ):
                breakpoint: dict[str, Any] = {"verified": resolved_line is not None}
                if resolved_line is None:
                    breakpoint.update(
                        {
                            "line": requested_line,
                            "message": (
                                "Breakpoints must be set on a model operation call "
                                "or an attention/MLP operation"
                            ),
                        }
                    )
                else:
                    condition = str(item.get("condition", "")).strip()
                    token_position = (
                        self._parse_token_condition(condition)
                        if condition
                        else self.default_token
                    )
                    if condition and token_position is None:
                        breakpoint.update(
                            {
                                "verified": False,
                                "line": resolved_line,
                                "message": (
                                    "Use a token condition such as token=4"
                                ),
                            }
                        )
                        breakpoints.append(breakpoint)
                        continue
                    if token_position is not None and token_position >= len(self.tokens):
                        breakpoint.update(
                            {
                                "verified": False,
                                "line": resolved_line,
                                "message": (
                                    f"Token position {token_position} is outside the prompt "
                                    f"range 0..{len(self.tokens) - 1}"
                                ),
                            }
                        )
                        breakpoints.append(breakpoint)
                        continue
                    breakpoint["line"] = resolved_line
                    configured.append(SourceBreakpoint(resolved_line, token_position))
                breakpoints.append(breakpoint)
            self.breakpoints = tuple(configured)
            self.send_response(request, {"breakpoints": breakpoints})
        elif command == "setExceptionBreakpoints":
            self.send_response(request)
        elif command in ("disconnect", "terminate"):
            self.send_response(request)
            return False
        else:
            self.send_response(request, success=False, message=f"Unsupported request: {command}")
        return True

    def _build_virtual_source(
        self,
        arguments: dict[str, Any],
        model: str,
        tokens: tuple[str, ...],
    ) -> None:
        """Build a fallback execution view for the dependency-free demo."""
        safe_model_name = model.rstrip("/").split("/")[-1] or "model"
        self.source_name = f"{safe_model_name}.execution.llmdbg"
        self.source_path = None
        lines = [
            "# Generated llmdbg demo source. This view is read-only.",
            f"model = {json.dumps(model)}",
            f"revision = {json.dumps(str(arguments.get('revision', 'main')))}",
            f"device = {json.dumps(str(arguments.get('device', 'cuda')))}",
            f"dtype = {json.dumps(str(arguments.get('dtype', 'bfloat16')))}",
            f"prompt = {json.dumps(str(arguments.get('prompt', '')))}",
        ]
        source_lines = []
        previous_layer: str | None = None
        for state in self.trace:
            lines.append("")
            source_lines.append(len(lines) + 1)
            block_match = re.match(r"^transformer\.h\.(\d+)\.", state.layer.name)
            layer = block_match.group(1) if block_match else None
            if layer is not None and layer != previous_layer:
                lines.append(f"layer = {layer}")
                source_lines[-1] += 1
                previous_layer = layer
            lines.append(f"submodule = {json.dumps(state.layer.name)}")
        self.source_content = "\n".join(lines) + "\n"
        self.source_lines = tuple(source_lines)

    def _configure_model_source(self, arguments: dict[str, Any]) -> None:
        """Bind authored submodule statements to the model execution order."""
        if self.execution is None:
            raise RuntimeError("model execution is not prepared")
        model = self.execution.runtime.model
        available = ["token_embedding"]
        for index in range(len(model.transformer.h)):
            available.extend(
                [f"transformer.h.{index}.attention", f"transformer.h.{index}.mlp"]
            )
        available.append("transformer.ln_f")
        if arguments.get("pythonProgram"):
            call_lines = arguments.get("callLines", {})
            step_paths = []
            step_lines: dict[str, int] = {}
            if "token_embedding" in call_lines:
                step_paths.append("token_embedding")
                step_lines["token_embedding"] = int(call_lines["token_embedding"])
            for index in range(len(model.transformer.h)):
                for kind in ("attention", "mlp"):
                    if kind in call_lines:
                        name = f"transformer.h.{index}.{kind}"
                        step_paths.append(name)
                        step_lines[name] = int(call_lines[kind])
            if "final_layernorm" in call_lines:
                step_paths.append("transformer.ln_f")
                step_lines["transformer.ln_f"] = int(call_lines["final_layernorm"])
        else:
            step_paths = list(arguments.get("stepPaths", []))
            step_lines = arguments.get("stepLines", {})
        selected = [path for path in available if path in set(step_paths)]
        if selected != step_paths:
            raise ValueError(
                "submodules must be a valid subsequence of model execution; "
                f"available paths are {available!r}"
            )
        path = Path(str(arguments["configPath"]))
        self.source_name = path.name
        self.source_path = str(path)
        self.source_lines = tuple(int(step_lines[name]) for name in step_paths)
        self.step_paths = tuple(step_paths)
        functions = arguments.get("functions", {})
        self.function_bindings = dict(arguments.get("functionBindings", {}))
        mlp_lines = functions.get("mlp", {}) if isinstance(functions, dict) else {}
        self.mlp_source_lines = tuple(int(mlp_lines.get(name, 0)) for name, _, _ in self.MLP_STEPS)
        if any(line <= 0 for line in self.mlp_source_lines):
            self.mlp_source_lines = ()
        attention_lines = functions.get("attention", {}) if isinstance(functions, dict) else {}
        self.attention_source_lines = tuple(
            int(attention_lines.get(name, 0)) for name, _, _ in self.ATTENTION_STEPS
        )
        if any(line <= 0 for line in self.attention_source_lines):
            self.attention_source_lines = ()

    def _advance_outer(self) -> None:
        """Step over the current authored submodule."""
        if self.layer_index + 1 < len(self.trace):
            self.layer_index += 1
            self.stop("step")
        elif self.token_index + 1 < len(self.tokens):
            self._load_token(self.token_index + 1)
            self.layer_index = 0
            self.stop("step")
        else:
            self.terminate()

    def _load_token(self, position: int) -> None:
        """Execute one prompt token and select the authored model steps."""
        if self.execution is None:
            self.token_index = 0
            return
        model_trace = self.execution.trace_token(position)
        states_by_name = {state.layer.name: state for state in model_trace.states}
        self.trace = tuple(states_by_name[path] for path in self.step_paths)
        self.token_index = position

    @staticmethod
    def _parse_token_condition(condition: str) -> int | None:
        if not condition:
            return None
        match = re.fullmatch(
            r"token\s*=\s*(\d+)",
            condition,
            flags=re.IGNORECASE,
        )
        return int(match.group(1)) if match else None

    def _continue_to_breakpoint(
        self,
        *,
        include_current: bool,
        before_start: bool = False,
    ) -> None:
        """Advance in token-major order, including authored function-body operations."""
        current_location = self.mlp_step_index + 1 if self.inside_internal else 0
        current = (self.token_index, self.layer_index, current_location)
        for token_position in range(len(self.tokens)):
            for layer_index in range(len(self.source_lines)):
                locations: list[tuple[int, int | None]] = [
                    (self.source_lines[layer_index], None)
                ]
                path = self.step_paths[layer_index] if layer_index < len(self.step_paths) else ""
                if path.endswith(".attention"):
                    locations.extend(
                        (line, index)
                        for index, line in enumerate(self.attention_source_lines)
                    )
                elif path.endswith(".mlp"):
                    locations.extend(
                        (line, index) for index, line in enumerate(self.mlp_source_lines)
                    )
                for location_index, (line, internal_index) in enumerate(locations):
                    position = (token_position, layer_index, location_index)
                    if not before_start and (
                        position < current or (position == current and not include_current)
                    ):
                        continue
                    if any(
                        breakpoint.line == line
                        and (
                            breakpoint.token_position is None
                            or breakpoint.token_position == token_position
                        )
                        for breakpoint in self.breakpoints
                    ):
                        self._load_token(token_position)
                        self.layer_index = layer_index
                        self.inside_attention = (
                            path.endswith(".attention") and internal_index is not None
                        )
                        self.inside_mlp = path.endswith(".mlp") and internal_index is not None
                        self.mlp_step_index = internal_index or 0
                        self.stop("breakpoint")
                        return
        self.terminate()

    def _stop_at_internal_breakpoint(self) -> bool:
        """Honor breakpoints inside the current call even while stepping over it."""
        path = (
            self.step_paths[self.layer_index]
            if self.layer_index < len(self.step_paths)
            else self.outer_state.layer.name
        )
        if path.endswith(".attention"):
            lines = self.attention_source_lines
        elif path.endswith(".mlp"):
            lines = self.mlp_source_lines
        else:
            return False
        for index, line in enumerate(lines):
            if any(
                breakpoint.line == line
                and (
                    breakpoint.token_position is None
                    or breakpoint.token_position == self.token_index
                )
                for breakpoint in self.breakpoints
            ):
                self.inside_attention = path.endswith(".attention")
                self.inside_mlp = path.endswith(".mlp")
                self.mlp_step_index = index
                self.stop("breakpoint")
                return True
        return False

    def _local_vector_names(self) -> list[str]:
        """Return vector-valued names that exist at the current source line."""
        names: list[str] = []
        path = (
            self.step_paths[self.layer_index]
            if self.layer_index < len(self.step_paths)
            else self.outer_state.layer.name
        )
        if self.inside_internal or self.layer_index > 0 or path != "token_embedding":
            names.append("x")
        if self.inside_internal:
            function_name = "attention" if self.inside_attention else "mlp"
            completed = {
                operation for operation, _vector, _meaning in self.internal_steps[: self.mlp_step_index]
            }
            latest: dict[str, str] = {}
            for binding in self.function_bindings.get(function_name, []):
                if binding.get("operation") in completed:
                    latest[str(binding["name"])] = str(binding["operation"])
            names.extend(name for name in latest if name not in names)
        return names

    def _local_variables(self) -> list[dict[str, Any]]:
        """Build Python-like locals for the active authored source frame."""
        variables: list[dict[str, Any]] = [
            {
                "name": "model", "value": repr(self.model_name), "type": "model",
                "variablesReference": 0,
            },
            {
                "name": "prompt", "value": repr(self.prompt), "type": "str",
                "variablesReference": 0, "evaluateName": "prompt",
            },
            {
                "name": "tokens", "value": f"{len(self.tokens)} tokens", "type": "tokens",
                "variablesReference": self.SCOPE_TOKENS,
            },
            {
                "name": "token_pos", "value": str(self.token_index), "type": "int",
                "variablesReference": 0, "evaluateName": "token_pos",
            },
        ]
        path = self.step_paths[self.layer_index] if self.layer_index < len(self.step_paths) else ""
        if self.inside_internal or ".attention" in path or ".mlp" in path:
            variables.append(
                {
                    "name": "layer", "value": repr(self.outer_state.layer.name), "type": "layer",
                    "variablesReference": 0, "evaluateName": "layer",
                }
            )
        for offset, name in enumerate(self._local_vector_names()):
            vector = self._vector(self._source_expression(name))
            variables.append(
                {
                    "name": name, "value": f"{len(vector)}D vector", "type": "TensorHandle",
                    "variablesReference": self.LOCAL_VECTOR_REFERENCE + offset,
                    "evaluateName": name,
                }
            )
        variables.extend(
            [
                {
                    "name": "token_text", "value": repr(self.tokens[self.token_index]), "type": "str",
                    "variablesReference": 0,
                },
                {
                    "name": "token_id",
                    "value": str(
                        int(self.execution.input_ids[0, self.token_index].item())
                        if self.execution is not None else 0
                    ),
                    "type": "int", "variablesReference": 0,
                },
            ]
        )
        return variables

    def _variables(self, reference: int) -> list[dict[str, Any]]:
        if reference == self.SCOPE_TOKEN:
            return self._local_variables()
        if reference == self.SCOPE_TOKENS:
            return [
                {
                    "name": f"[{index}]", "value": repr(text), "type": "token",
                    "variablesReference": 0,
                }
                for index, text in enumerate(self.tokens)
            ]
        if reference >= self.LOCAL_VECTOR_REFERENCE:
            offset = reference - self.LOCAL_VECTOR_REFERENCE
            names = self._local_vector_names()
            if offset >= len(names):
                return []
            name = names[offset]
            vector = self._vector(self._source_expression(name))
            indices = sorted(range(len(vector)), key=lambda index: abs(vector[index]), reverse=True)[:16]
            return [
                {
                    "name": f"dimension[{index}]", "value": f"{vector[index]:+.6f}",
                    "type": "float", "variablesReference": 0,
                    "evaluateName": f"{name}[{index}]",
                }
                for index in indices
            ]
        if reference == self.SCOPE_LAYER:
            variables = [
                {"name": "index", "value": str(self.layer_index), "type": "int", "variablesReference": 0},
                {"name": "name", "value": self.state.layer.name, "type": "str", "variablesReference": 0},
                {"name": "phase", "value": "before execution", "type": "str", "variablesReference": 0},
                {"name": "operation", "value": self.state.layer.operation, "type": "str", "variablesReference": 0},
                {"name": "meaning", "value": self.state.layer.meaning, "type": "str", "variablesReference": 0},
            ]
            if self.previous_state is not None:
                variables.append(
                    {
                        "name": "previous",
                        "value": self.previous_state.layer.name,
                        "type": "completed operation",
                        "variablesReference": 0,
                    }
                )
            return variables
        if reference == self.SCOPE_TOP_NEURONS:
            return [
                {
                    "name": f"neuron[{neuron.index}]",
                    "value": f"activation {neuron.activation:+.6f}",
                    "type": "MLP neuron",
                    "variablesReference": self.NEURON_REFERENCE + rank,
                }
                for rank, neuron in enumerate(self._top_neurons())
            ]
        if self.NEURON_REFERENCE <= reference < self.NEURON_REFERENCE + 5:
            rank = reference - self.NEURON_REFERENCE
            neurons = self._top_neurons()
            if rank >= len(neurons):
                return []
            neuron = neurons[rank]
            token_reference = self.TOKEN_REFERENCE + rank * 2
            return [
                {
                    "name": "index",
                    "value": str(neuron.index),
                    "type": "int",
                    "variablesReference": 0,
                },
                {
                    "name": "activation",
                    "value": f"{neuron.activation:+.8f}",
                    "type": "float",
                    "variablesReference": 0,
                    "evaluateName": f"top_neurons[{rank}].activation",
                },
                {
                    "name": "output direction norm",
                    "value": f"{neuron.direction_norm:.8f}",
                    "type": "float",
                    "variablesReference": 0,
                },
                {
                    "name": "vocabulary projection (W_down → W_unembed)",
                    "value": "5 tokens",
                    "type": "direct vocabulary projection",
                    "variablesReference": token_reference,
                },
                {
                    "name": "activation-scaled token contribution",
                    "value": "5 tokens",
                    "type": "direct logit contribution",
                    "variablesReference": token_reference + 1,
                },
            ]
        if self.TOKEN_REFERENCE <= reference < self.TOKEN_REFERENCE + 10:
            offset = reference - self.TOKEN_REFERENCE
            rank, score_kind = divmod(offset, 2)
            neurons = self._top_neurons()
            if rank >= len(neurons):
                return []
            scores = (
                neurons[rank].vocabulary
                if score_kind == 0
                else neurons[rank].contribution
            )
            return [
                {
                    "name": f"token[{item.token_id}] {item.text!r}",
                    "value": f"{item.score:+.8f}",
                    "type": "logit",
                    "variablesReference": 0,
                }
                for item in scores
            ]
        vector_name: str | None = None
        if reference == self.SCOPE_INPUT:
            vector = self.state.input_vector
            vector_name = "input"
        elif reference == self.SCOPE_RESIDUAL:
            if self.previous_state is None:
                return []
            vector = self.previous_state.output_vector
            vector_name = "output"
        elif reference == self.SCOPE_DELTA:
            if self.previous_state is None:
                return []
            vector = tuple(
                output - input_
                for input_, output in zip(
                    self.previous_state.input_vector,
                    self.previous_state.output_vector,
                    strict=True,
                )
            )
        elif reference in self.DETAIL_SCOPES:
            if self.previous_state is None:
                return []
            detail_name = self.DETAIL_SCOPES[reference][0]
            if detail_name not in self.previous_state.details:
                return []
            vector = self.previous_state.details[detail_name]
            vector_name = detail_name
        else:
            return []
        if len(vector) == len(FEATURES):
            displayed = list(zip(FEATURES, vector, strict=True))
        else:
            indices = sorted(range(len(vector)), key=lambda index: abs(vector[index]), reverse=True)[:16]
            displayed = [(f"dimension[{index}]", vector[index]) for index in indices]
        return [
            {
                "name": feature,
                "value": f"{value:+.6f}",
                "type": "float",
                "variablesReference": 0,
                **(
                    {
                        "evaluateName": (
                            f"{vector_name}["
                            f"{feature.removeprefix('dimension[').removesuffix(']')}]"
                        )
                    }
                    if vector_name is not None and feature.startswith("dimension[")
                    else {}
                ),
            }
            for feature, value in displayed
        ]

    def _variable_expression(self, reference: int, name: str) -> str:
        """Resolve a Variables-pane Set Value request to a Watch expression."""
        dimension = re.fullmatch(r"dimension\[(\d+)\]", name)
        if dimension:
            vector_names = {
                self.SCOPE_INPUT: "input",
                self.SCOPE_RESIDUAL: "output",
                **{
                    scope_reference: detail_name
                    for scope_reference, (detail_name, _label) in self.DETAIL_SCOPES.items()
                },
            }
            vector_name = vector_names.get(reference)
            if vector_name is not None:
                return f"{vector_name}[{dimension.group(1)}]"
        if (
            self.NEURON_REFERENCE <= reference < self.NEURON_REFERENCE + 5
            and name == "activation"
        ):
            return f"top_neurons[{reference - self.NEURON_REFERENCE}].activation"
        raise ValueError(f"{name!r} cannot be changed in this scope")

    def _set_expression(self, expression: str, value: float) -> float:
        """Record an intervention and replay the token to refresh downstream values."""
        if self.execution is None:
            raise ValueError("vector intervention requires a loaded model")
        result_state = self.previous_state
        if result_state is None:
            raise ValueError("step over an operation before changing its output")

        expression = self._source_expression(expression)
        neuron_match = re.fullmatch(r"top_neurons\[(\d+)\]\.activation", expression)
        if neuron_match:
            rank = int(neuron_match.group(1))
            neurons = self._top_neurons()
            if rank >= len(neurons):
                raise ValueError(f"top neuron rank must be in 0..{len(neurons) - 1}")
            vector_name = "mlp_expansion"
            index = neurons[rank].index
        else:
            indexed = re.fullmatch(
                r"(input|output|residual|residual_stream|layernorm|normalized_input|"
                r"normalized_output|mlp_expansion_pre_activation|mlp_expansion|mlp_projection|"
                r"attention_qkv|attention_mixing|attention_projection)"
                r"\[(\d+)\]",
                expression,
            )
            if indexed is None:
                raise ValueError(
                    "set a vector dimension such as output[42], mlp_expansion[10], "
                    "or top_neurons[0].activation"
                )
            vector_name, raw_index = indexed.groups()
            index = int(raw_index)
            if vector_name == "input":
                vector_name = "output"
            elif vector_name in ("residual", "residual_stream"):
                vector_name = "output"
            elif vector_name == "layernorm":
                vector_name = "normalized_input"

        if getattr(self, "inside_internal", False) and vector_name == "output":
            vector_name = self.internal_steps[self.mlp_step_index - 1][1]

        source_vector = (
            result_state.output_vector
            if vector_name == "output"
            else result_state.details.get(vector_name)
        )
        if source_vector is None:
            raise ValueError(f"{vector_name} is not available for {result_state.layer.name}")
        if index < 0 or index >= len(source_vector):
            raise ValueError(
                f"dimension {index} is outside {vector_name} range 0..{len(source_vector) - 1}"
            )

        target_layer = (
            self.outer_state.layer.name
            if getattr(self, "inside_internal", False)
            else result_state.layer.name
        )
        self.execution.set_intervention(
            self.token_index,
            target_layer,
            vector_name,
            index,
            value,
        )
        current_layer = self.layer_index
        self._load_token(self.token_index)
        self.layer_index = current_layer
        self.neuron_cache.clear()
        return value

    def _top_neurons(self) -> tuple[NeuronProjection, ...]:
        """Return cached direct vocabulary projections for the current MLP state."""
        result_state = self.previous_state
        if (
            self.execution is None
            or result_state is None
            or "mlp_expansion" not in result_state.details
        ):
            return ()
        layer_name = (
            self.outer_state.layer.name
            if getattr(self, "inside_internal", False)
            else result_state.layer.name
        )
        key = (self.token_index, layer_name)
        if key not in self.neuron_cache:
            self.neuron_cache[key] = self.execution.top_mlp_neurons(
                layer_name,
                result_state.details["mlp_expansion"],
            )
        return self.neuron_cache[key]

    def _evaluate(self, expression: str) -> str | None:
        expression = self._source_expression(expression)
        if expression == "prompt":
            return self.prompt
        if expression in ("token", "token.text"):
            return self.tokens[self.token_index]
        if expression in ("token_pos", "position", "token.position"):
            return str(self.token_index)
        if expression in ("layer", "name"):
            return self.state.layer.name
        if expression in ("meaning", "interpretation"):
            return self.state.layer.meaning
        if expression == "operation":
            return self.state.layer.operation
        if expression == "input":
            return repr(self.state.input_vector)
        if expression == "x":
            return repr(self._vector("x"))
        if expression in ("output", "residual", "residual_stream"):
            return repr(self._vector("output"))
        if expression == "delta":
            return repr(self._vector("delta"))
        detail_expression = "normalized_input" if expression == "layernorm" else expression
        if self.previous_state is not None and detail_expression in self.previous_state.details:
            return repr(self.previous_state.details[detail_expression])
        indexed = re.fullmatch(
            r"(x|input|output|residual|residual_stream|delta|layernorm|normalized_input|"
            r"normalized_output|mlp_expansion_pre_activation|mlp_expansion|mlp_projection|"
            r"attention_qkv|attention_mixing|attention_projection)"
            r"\[(\d+)\]",
            expression,
        )
        if indexed:
            vector_name, raw_index = indexed.groups()
            vector = self._vector(vector_name)
            index = int(raw_index)
            if index >= len(vector):
                raise ValueError(
                    f"dimension {index} is outside the vector range 0..{len(vector) - 1}"
                )
            return f"{vector[index]:+.8f}"
        sliced = re.fullmatch(
            r"([a-z_][a-z0-9_]*)\[(\d*)?:(\d*)?(?::(\d+))?\]",
            expression,
        )
        if sliced:
            vector_name, raw_start, raw_stop, raw_step = sliced.groups()
            vector = self._vector(vector_name)
            selection = vector[
                slice(
                    int(raw_start) if raw_start else None,
                    int(raw_stop) if raw_stop else None,
                    int(raw_step) if raw_step else None,
                )
            ]
            return repr(selection)
        return None

    def _source_expression(self, expression: str) -> str:
        """Resolve Python frame locals and debugger aliases to runtime vectors."""
        aliases = {"x": "x", "residual_stream": "x"}
        if not getattr(self, "inside_internal", False):
            return self._replace_expression_name(expression, aliases)
        function_name = "attention" if getattr(self, "inside_attention", False) else "mlp"
        bindings = getattr(self, "function_bindings", {}).get(function_name, [])
        operation_vectors = {
            operation: vector for operation, vector, _meaning in self.internal_steps
        }
        completed = {
            operation for operation, _vector, _meaning in self.internal_steps[: self.mlp_step_index]
        }
        for binding in bindings:
            if binding.get("operation") in completed:
                aliases[str(binding["name"])] = operation_vectors[str(binding["operation"])]
        if not bindings and function_name == "attention":
            aliases.update({
                "normalized": "normalized_input",
                "qkv": "attention_qkv",
                "mixed": "attention_mixing",
                "update": "attention_projection",
            })
        elif not bindings:
            hidden = (
                "mlp_expansion_pre_activation"
                if self.mlp_step_index <= 2
                else "mlp_expansion"
            )
            aliases = {
                "normalized": "normalized_input",
                "hidden": hidden,
                "update": "mlp_projection",
            }
        return self._replace_expression_name(expression, aliases)

    @staticmethod
    def _replace_expression_name(expression: str, aliases: dict[str, str]) -> str:
        for alias, vector in aliases.items():
            if expression == alias or expression.startswith(alias + "["):
                return vector + expression[len(alias):]
        return expression

    def _evaluate_body(self, expression: str) -> dict[str, Any] | None:
        """Build a DAP evaluate result, including expandable synthetic values."""
        if expression in ("top_neurons", "top_activated_neurons"):
            result_state = self.previous_state
            if (
                self.execution is None
                or result_state is None
                or "mlp_expansion" not in result_state.details
            ):
                return {
                    "result": "available at MLP stops",
                    "type": "MLP neurons",
                    "variablesReference": 0,
                }
            return {
                "result": "top 5 post-GELU activations",
                "type": "MLP neurons",
                "variablesReference": self.SCOPE_TOP_NEURONS,
                "namedVariables": 5,
            }
        neuron_match = re.fullmatch(
            r"(?:top_neurons|top_activated_neurons)\[(\d+)\]"
            r"(?:\.(index|activation|direction_norm|vocabulary|contribution))?",
            expression,
        )
        if neuron_match:
            rank = int(neuron_match.group(1))
            attribute = neuron_match.group(2)
            neurons = self._top_neurons()
            if rank >= len(neurons):
                return {
                    "result": f"rank {rank} is outside 0..{len(neurons) - 1}",
                    "variablesReference": 0,
                }
            neuron = neurons[rank]
            if attribute is None:
                return {
                    "result": f"neuron[{neuron.index}] activation {neuron.activation:+.8f}",
                    "type": "MLP neuron",
                    "variablesReference": self.NEURON_REFERENCE + rank,
                }
            if attribute == "index":
                return {"result": str(neuron.index), "type": "int", "variablesReference": 0}
            if attribute == "activation":
                return {
                    "result": f"{neuron.activation:+.8f}",
                    "type": "float",
                    "variablesReference": 0,
                }
            if attribute == "direction_norm":
                return {
                    "result": f"{neuron.direction_norm:.8f}",
                    "type": "float",
                    "variablesReference": 0,
                }
            token_reference = self.TOKEN_REFERENCE + rank * 2
            return {
                "result": "5 tokens",
                "type": "direct vocabulary projection",
                "variablesReference": token_reference + (attribute == "contribution"),
            }
        result = self._evaluate(expression)
        if result is None:
            return None
        return {"result": result, "variablesReference": 0}

    def _vector(self, name: str) -> tuple[float, ...]:
        if name == "x":
            return self.outer_state.input_vector
        if name == "input":
            return self.state.input_vector
        if name in ("output", "residual", "residual_stream"):
            return (
                self.previous_state.output_vector
                if self.previous_state is not None
                else self.state.input_vector
            )
        if name == "delta":
            if self.previous_state is None:
                return tuple(0.0 for _ in self.state.input_vector)
            return tuple(
                output - input_
                for input_, output in zip(
                    self.previous_state.input_vector,
                    self.previous_state.output_vector,
                    strict=True,
                )
            )
        if name == "layernorm":
            name = "normalized_input"
        if getattr(self, "inside_internal", False) and name in self.outer_state.details:
            return self.outer_state.details[name]
        if self.previous_state is not None and name in self.previous_state.details:
            return self.previous_state.details[name]
        raise ValueError(f"unknown vector: {name}")

    def _vector_snapshot(self) -> dict[str, Any]:
        input_vector = self._vector("input")
        residual = self._vector("residual_stream")
        delta = self._vector("delta")
        vectors = {
            "input": list(input_vector),
            "residual_stream": list(residual),
            "delta": list(delta),
        }
        if self.previous_state is not None:
            vectors.update(
                {name: list(vector) for name, vector in self.previous_state.details.items()}
            )
        return {
            "tokenPosition": self.token_index,
            "token": self.tokens[self.token_index],
            "layer": self.state.layer.name,
            "previousLayer": (
                self.previous_state.layer.name if self.previous_state is not None else None
            ),
            "dimensions": len(residual),
            "vectors": vectors,
        }

    def _plot_vector(self, expression: str) -> tuple[float, ...]:
        expression = self._source_expression(expression.strip().lower())
        if expression.startswith("[") or expression.startswith("("):
            value = ast.literal_eval(expression)
            if not isinstance(value, (list, tuple)):
                raise ValueError("pasted vector must be a list or tuple")
            return tuple(float(item) for item in value)
        match = re.fullmatch(
            r"([a-z_][a-z0-9_]*)(?:\[(\d*)?:(\d*)?(?::(\d+))?\])?",
            expression,
        )
        if match is None:
            raise ValueError("vector must be a vector name, slice, list, or tuple")
        name, raw_start, raw_stop, raw_step = match.groups()
        vector = self._vector(name)
        if "[" not in expression:
            return vector
        return vector[
            slice(
                int(raw_start) if raw_start else None,
                int(raw_stop) if raw_stop else None,
                int(raw_step) if raw_step else None,
            )
        ]

    def _render_plot(self, expression: str, code: str) -> dict[str, Any]:
        """Run an explicitly user-selected Matplotlib snippet and return a PNG."""
        if not code.strip():
            raise ValueError("plot code is empty")
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        vector = self._plot_vector(expression)
        plt.close("all")
        namespace: dict[str, Any] = {
            "plt": plt,
            "np": np,
            "vector": np.asarray(vector, dtype=float),
            "vector_name": expression,
        }
        exec(compile(code, "<llmdbg plot snippet>", "exec"), namespace, namespace)
        plot = namespace.get("plot")
        if callable(plot):
            figure = plot(namespace["vector"], expression, plt, np) or plt.gcf()
        else:
            # Keep old snippets usable when sent by an older extension.
            figure = namespace.get("fig") or plt.gcf()
        output = io.BytesIO()
        figure.savefig(output, format="png", dpi=120, bbox_inches="tight")
        plt.close("all")
        return {
            "image": base64.b64encode(output.getvalue()).decode("ascii"),
            "meta": (
                f"{expression} · {len(vector)}D · token {self.token_index} "
                f"{self.tokens[self.token_index]!r} · {self.state.layer.name}"
            ),
        }

    def _neuron_lens_snapshot(self) -> dict[str, Any]:
        """Describe top MLP neurons through W_down and the unembedding matrix."""
        neurons = self._top_neurons()
        if not neurons:
            raise ValueError(
                "Neuron Lens is available after an MLP GELU activation has completed"
            )
        layer = (
            self.outer_state.layer.name
            if self.inside_internal
            else self.previous_state.layer.name  # type: ignore[union-attr]
        )
        return {
            "tokenPosition": self.token_index,
            "token": self.tokens[self.token_index],
            "layer": layer,
            "neurons": [
                {
                    "rank": rank,
                    "index": neuron.index,
                    "activation": neuron.activation,
                    "directionNorm": neuron.direction_norm,
                    "tokens": [
                        {
                            "rank": token_rank,
                            "id": token.token_id,
                            "text": token.text,
                            "logit": token.score,
                        }
                        for token_rank, token in enumerate(neuron.vocabulary, start=1)
                    ],
                }
                for rank, neuron in enumerate(neurons, start=1)
            ],
        }

    def run(self) -> None:
        while request := self.stream.read():
            if request.get("type") != "request":
                continue
            try:
                if not self.handle(request):
                    break
            except Exception as error:  # keep protocol errors visible to the client
                self.send_response(request, success=False, message=str(error))


def main() -> None:
    """Run the debug adapter over stdin/stdout."""
    LLMDebugAdapter(ProtocolStream(sys.stdin.buffer, sys.stdout.buffer)).run()


if __name__ == "__main__":
    main()
