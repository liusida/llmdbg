"""Minimal Debug Adapter Protocol implementation for llmdbg."""

from __future__ import annotations

import json
import ast
import base64
import hashlib
import importlib.metadata
import io
import platform
import re
import sys
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

from llmdbg.config import load_debug_config
from llmdbg.activation_scales import (
    attention_channel_activation_scale,
    mlp_activation_scale,
)
from llmdbg.demo_model import FEATURES, Layer, LayerState, run_demo
from llmdbg.model_runtime import ModelExecution, NeuronProjection, prepare_model_execution


@dataclass(frozen=True)
class SourceBreakpoint:
    """A model-step source line with optional token and transformer-layer filters."""

    line: int
    token_position: int | None = None
    transformer_layer: int | None = None
    token_operator: str = "=="
    layer_operator: str = "=="


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
    SCOPE_PREDICTIONS = 70
    SCOPE_BREAKPOINT_CONDITIONS = 80
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
        1600: ("full_attention_mixing", "Full-attention mixing"),
        1700: ("gated_delta_mixing", "Gated DeltaNet mixing"),
        1800: ("mlp_gate_pre_activation", "MLP gate (pre-activation)"),
        1900: ("mlp_gate_activation", "MLP gate (after SiLU)"),
        1950: ("mlp_up_projection", "MLP value branch"),
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
    FULL_ATTENTION_STEPS = (
        ("input_layernorm", "normalized_input", "RMS-normalize the residual stream."),
        ("full_attention", "full_attention_mixing", "Apply causal multi-head attention and its output gate."),
        ("output_projection", "attention_projection", "Project the mixed attention heads to model width."),
        ("residual_add", "output", "Add the full-attention update to the residual stream."),
    )
    GATED_LINEAR_ATTENTION_STEPS = (
        ("input_layernorm", "normalized_input", "RMS-normalize the residual stream."),
        ("gated_delta", "gated_delta_mixing", "Apply the recurrent Gated DeltaNet token mixer."),
        ("output_projection", "attention_projection", "Project the recurrent mixer output to model width."),
        ("residual_add", "output", "Add the linear-attention update to the residual stream."),
    )
    QWEN_MLP_STEPS = (
        ("post_attention_layernorm", "normalized_input", "RMS-normalize the post-attention residual stream."),
        ("gate_projection", "mlp_gate_pre_activation", "Project into the SwiGLU gate branch."),
        ("silu", "mlp_gate_activation", "Apply SiLU to the gate branch."),
        ("up_projection", "mlp_up_projection", "Project into the SwiGLU value branch."),
        ("gated_multiply", "mlp_expansion", "Multiply the activated gate and value branches."),
        ("down_projection", "mlp_projection", "Project active features back to model width."),
        ("residual_add", "output", "Add the MLP update to the residual stream."),
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
        self.stop_on_entry = False
        self.source_content = ""
        self.source_lines: tuple[int, ...] = ()
        self.step_paths: tuple[str, ...] = ()
        self.source_name = "model.execution.llmdbg"
        self.source_path: str | None = None
        self.breakpoints: tuple[SourceBreakpoint, ...] = ()
        self.breakpoints_by_source: dict[str, tuple[SourceBreakpoint, ...]] = {}
        self.neuron_cache: dict[tuple[Any, ...], tuple[NeuronProjection, ...]] = {}
        self.lens_snapshot_cache: dict[tuple[Any, ...], dict[str, Any]] = {}
        self.measurement_vectors: dict[str, tuple[str, tuple[float, ...]]] = {}
        self.inside_mlp = False
        self.mlp_step_index = 0
        self.mlp_source_content = ""
        self.mlp_source_lines: tuple[int, ...] = ()
        self.inside_attention = False
        self.attention_source_lines: tuple[int, ...] = ()
        self.function_bindings: dict[str, list[dict[str, Any]]] = {}
        self.function_source_lines: dict[str, tuple[int, ...]] = {}
        self.predictions: tuple[Any, ...] = ()
        self.model_revision = "main"
        self.requested_device = "auto"
        self.requested_dtype = "auto"

    @property
    def internal_steps(self) -> tuple[tuple[str, str, str], ...]:
        function_name = self._function_name_for_path(self.outer_state.layer.name)
        return {
            "attention": self.ATTENTION_STEPS,
            "full_attention": self.FULL_ATTENTION_STEPS,
            "gated_linear_attention": self.GATED_LINEAR_ATTENTION_STEPS,
            "mlp": self.QWEN_MLP_STEPS if self.outer_state.layer.name.startswith("model.layers.") else self.MLP_STEPS,
        }.get(function_name, self.MLP_STEPS)

    @staticmethod
    def _function_name_for_path(path: str) -> str:
        for name in ("gated_linear_attention", "full_attention", "attention", "mlp"):
            if path.endswith("." + name):
                return name
        return ""

    def _internal_source_lines(self, path: str | None = None) -> tuple[int, ...]:
        target = path or self.outer_state.layer.name
        function_name = self._function_name_for_path(target)
        configured = getattr(self, "function_source_lines", {})
        if function_name in configured:
            return configured[function_name]
        if function_name == "mlp":
            return getattr(self, "mlp_source_lines", ())
        if function_name == "attention":
            return getattr(self, "attention_source_lines", ())
        return ()

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
        getattr(self, "lens_snapshot_cache", {}).clear()
        getattr(self, "measurement_vectors", {}).clear()
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
            self.model_revision = str(arguments.get("revision", "main"))
            self.requested_device = str(arguments.get("device", "auto"))
            self.requested_dtype = str(arguments.get("dtype", "auto"))
            if model:
                self.send_event(
                    "output",
                    {"category": "console", "output": f"llmdbg: loading {model}\n"},
                )
                self.execution = prepare_model_execution(
                    str(model),
                    revision=str(arguments.get("revision", "main")),
                    prompt=self.prompt,
                    device=str(arguments.get("device", "auto")),
                    dtype=str(arguments.get("dtype", "auto")),
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
                if self.execution.device.startswith("cuda"):
                    import torch

                    if torch.cuda.is_available():
                        cuda_allocated_bytes = int(torch.cuda.memory_allocated())
                        cuda_reserved_bytes = int(torch.cuda.memory_reserved())
                self.send_event(
                    "llmdbg/modelLoaded",
                    {
                        "model": str(model),
                        "revision": str(arguments.get("revision", "main")),
                        "device": self.execution.device,
                        "dtype": self.execution.dtype,
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
            self.stop_on_entry = bool(arguments.get("stopOnEntry", False))
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
                    self._internal_source_lines()[self.mlp_step_index]
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
                {
                    "name": "Breakpoint Conditions",
                    "variablesReference": self.SCOPE_BREAKPOINT_CONDITIONS,
                    "expensive": False,
                },
                {"name": "Local variables", "variablesReference": self.SCOPE_TOKEN, "expensive": False},
            ]
            result_state = self.previous_state
            if result_state is not None:
                scopes.append(
                    {
                        "name": f"Previous output — {result_state.layer.name}",
                        "variablesReference": self.SCOPE_RESIDUAL,
                        "expensive": False,
                    }
                )
                if len(result_state.input_vector) == len(result_state.output_vector):
                    scopes.append(
                        {
                            "name": "Previous delta",
                            "variablesReference": self.SCOPE_DELTA,
                            "expensive": False,
                        }
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
            mode = str(arguments.get("mode", "logit"))
            view = str(arguments.get("view", "neurons"))
            cache_key = (
                self.token_index,
                self.layer_index,
                self.inside_mlp,
                self.inside_attention,
                self.mlp_step_index,
                mode,
                view,
            )
            snapshot = self.lens_snapshot_cache.get(cache_key)
            if snapshot is None:
                snapshot = self._neuron_lens_snapshot(
                    view=view,
                    include_logit_lens=mode in {"logit", "all"},
                    include_j_lens=mode in {"j-lens", "all"},
                    include_r_lens=mode in {"r-lens", "all"},
                    include_jpp_lens=mode in {"jpp-lens", "all"},
                )
                self.lens_snapshot_cache[cache_key] = snapshot
            self.send_response(request, snapshot)
        elif command == "llmdbg/measureCosine":
            self.send_response(
                request,
                self._measure_cosine(
                    str(arguments.get("vectorA", "")),
                    str(arguments.get("vectorB", "")),
                ),
            )
        elif command == "llmdbg/icaSupport":
            from llmdbg.ica_lens import artifact_repository
            from llmdbg.jpp_lens import configured_display_name

            repository = artifact_repository(str(self.model_name))
            self.send_response(
                request,
                {
                    "supported": repository is not None,
                    "repository": repository,
                    "jppLabel": configured_display_name(str(self.model_name)),
                },
            )
        elif command == "llmdbg/removeIcaComponent":
            self.send_event(
                "llmdbg/computationStarted", {"command": command}
            )
            result = self._remove_ica_component(
                str(arguments.get("view", "")),
                int(arguments.get("component", -1)),
            )
            self.send_response(request, result)
            self.send_event("invalidated", {"areas": ["variables", "watch"]})
            self.send_event(
                "llmdbg/vectorChanged",
                {"tokenPosition": self.token_index, "layer": self.state.layer.name},
            )
        elif command == "llmdbg/removeProjectedComponent":
            self.send_event(
                "llmdbg/computationStarted", {"command": command}
            )
            result = self._remove_projected_component(
                str(arguments.get("view", "")),
                int(arguments.get("component", -1)),
            )
            self.send_response(request, result)
            self.send_event("invalidated", {"areas": ["variables", "watch"]})
            self.send_event(
                "llmdbg/vectorChanged",
                {"tokenPosition": self.token_index, "layer": self.state.layer.name},
            )
        elif command in ("next", "stepIn", "stepOut"):
            self.send_response(request)
            self.send_event("llmdbg/computationStarted", {"command": command})
            function_name = self._function_name_for_path(self.outer_state.layer.name)
            internal_lines = self._internal_source_lines()
            can_enter_mlp = function_name == "mlp" and internal_lines
            can_enter_attention = function_name in {
                "attention", "full_attention", "gated_linear_attention"
            } and internal_lines
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
            self.send_event("llmdbg/computationStarted", {"command": command})
            self._continue_to_breakpoint(include_current=False)
        elif command == "pause":
            self.send_response(request)
            self.stop("pause")
        elif command == "setBreakpoints":
            source = arguments.get("source") or {}
            requested_path = source.get("path")
            source_key = self._breakpoint_source_key(requested_path)
            current_key = self._breakpoint_source_key(self.source_path)
            is_current_source = source_key == current_key
            requested = [int(item.get("line", 0)) for item in arguments.get("breakpoints", [])]
            valid_lines = sorted(
                set(self.source_lines)
                | {
                    line
                    for lines in self.function_source_lines.values()
                    for line in lines
                }
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
                    parsed_condition = self._parse_breakpoint_condition(condition)
                    if condition and parsed_condition is None:
                        breakpoint.update(
                            {
                                "verified": False,
                                "line": resolved_line,
                                "message": (
                                    "Use conditions such as token=4, layer>=15, "
                                    "or token=4, layer>=15"
                                ),
                            }
                        )
                        breakpoints.append(breakpoint)
                        continue
                    token_filter, layer_filter = parsed_condition or (None, None)
                    condition_token = token_filter[1] if token_filter is not None else None
                    token_operator = token_filter[0] if token_filter is not None else "=="
                    transformer_layer = layer_filter[1] if layer_filter is not None else None
                    layer_operator = layer_filter[0] if layer_filter is not None else "=="
                    token_position = (
                        condition_token
                        if condition_token is not None
                        else self.default_token
                    )
                    if (
                        token_position is not None
                        and token_operator == "=="
                        and token_position >= len(self.tokens)
                    ):
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
                    configured.append(
                        SourceBreakpoint(
                            resolved_line,
                            token_position,
                            transformer_layer,
                            token_operator,
                            layer_operator,
                        )
                    )
                breakpoints.append(breakpoint)
            self.breakpoints_by_source[source_key] = tuple(configured)
            if is_current_source:
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
            f"device = {json.dumps(str(arguments.get('device', 'auto')))}",
            f"dtype = {json.dumps(str(arguments.get('dtype', 'auto')))}",
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
        runtime = self.execution.runtime
        runtime_layers = getattr(runtime, "layers", None)
        if runtime_layers is None:
            runtime_layers = getattr(
                getattr(getattr(runtime, "model", None), "transformer", None),
                "h",
                (),
            )
        runtime_architecture = getattr(runtime, "architecture", "gpt2")

        def layer_kind(index: int) -> str:
            return runtime.layer_kind(index) if hasattr(runtime, "layer_kind") else "attention"

        def state_name(index: int, kind: str) -> str:
            return (
                runtime.state_name(index, kind)
                if hasattr(runtime, "state_name")
                else f"transformer.h.{index}.{kind}"
            )

        final_norm_name = getattr(runtime, "final_norm_name", "transformer.ln_f")
        available = ["token_embedding"]
        for index in range(len(runtime_layers)):
            available.extend(
                [
                    state_name(index, layer_kind(index)),
                    state_name(index, "mlp"),
                ]
            )
        available.extend(
            [final_norm_name, "vocabulary_projection", "predictions", "end"]
        )
        if arguments.get("pythonProgram"):
            call_lines = arguments.get("callLines", {})
            step_paths = []
            step_lines: dict[str, int] = {}
            nop_lines = [int(line) for line in call_lines.get("nop", ())]
            token_line = int(call_lines.get("token_embedding", 0))
            final_norm_line = int(call_lines.get("final_layernorm", 1 << 30))
            loop_nops = [
                (index, line) for index, line in enumerate(nop_lines)
                if token_line < line < final_norm_line
            ]
            if "token_embedding" in call_lines:
                step_paths.append("token_embedding")
                step_lines["token_embedding"] = int(call_lines["token_embedding"])
            for index in range(len(runtime_layers)):
                mixer_kind = layer_kind(index)
                layer_entries: list[tuple[int, str]] = []
                for kind in (mixer_kind, "mlp"):
                    if kind in call_lines:
                        name = state_name(index, kind)
                        step_lines[name] = int(call_lines[kind])
                        layer_entries.append((step_lines[name], name))
                for nop_index, nop_line in loop_nops:
                    nop_path = f"nop:{nop_index}:{index}"
                    step_lines[nop_path] = nop_line
                    layer_entries.append((nop_line, nop_path))
                layer_entries.sort(key=lambda entry: entry[0])
                step_paths.extend(path for _line, path in layer_entries)
            if "final_layernorm" in call_lines:
                step_paths.append(final_norm_name)
                step_lines[final_norm_name] = int(call_lines["final_layernorm"])
            if "vocabulary_projection" in call_lines:
                step_paths.append("vocabulary_projection")
                step_lines["vocabulary_projection"] = int(
                    call_lines["vocabulary_projection"]
                )
            if "top_tokens" in call_lines:
                step_paths.append("predictions")
                step_lines["predictions"] = int(call_lines["top_tokens"])
            if "end" in call_lines:
                step_paths.append("end")
                step_lines["end"] = int(call_lines["end"])
            loop_nop_indexes = {index for index, _line in loop_nops}
            for nop_index, nop_line in enumerate(nop_lines):
                if nop_index in loop_nop_indexes:
                    continue
                nop_path = f"nop:{nop_index}"
                insert_at = next(
                    (
                        index for index, path in enumerate(step_paths)
                        if step_lines[path] > int(nop_line)
                    ),
                    len(step_paths),
                )
                step_paths.insert(insert_at, nop_path)
                step_lines[nop_path] = int(nop_line)
        else:
            step_paths = list(arguments.get("stepPaths", []))
            step_lines = arguments.get("stepLines", {})
        executable_paths = [path for path in step_paths if not path.startswith("nop:")]
        selected = [path for path in available if path in set(executable_paths)]
        if selected != executable_paths:
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
        step_sets = {
            "attention": self.ATTENTION_STEPS,
            "full_attention": self.FULL_ATTENTION_STEPS,
            "gated_linear_attention": self.GATED_LINEAR_ATTENTION_STEPS,
            "mlp": self.QWEN_MLP_STEPS if runtime_architecture == "qwen3_5" else self.MLP_STEPS,
        }
        self.function_source_lines = {}
        if isinstance(functions, dict):
            for function_name, steps in step_sets.items():
                operation_lines = functions.get(function_name, {})
                lines = tuple(int(operation_lines.get(name, 0)) for name, _, _ in steps)
                if lines and all(line > 0 for line in lines):
                    self.function_source_lines[function_name] = lines
        self.mlp_source_lines = self.function_source_lines.get("mlp", ())
        self.attention_source_lines = self.function_source_lines.get("attention", ())

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
        self.predictions = ()
        if "vocabulary_projection" in self.step_paths:
            final_norm_name = self.execution.runtime.final_norm_name
            final_state = states_by_name[final_norm_name]
            logits, self.predictions = self.execution.vocabulary_projection(
                final_state.output_vector
            )
            states_by_name["vocabulary_projection"] = LayerState(
                Layer(
                    "vocabulary_projection", "vocabulary_projection",
                    "Project the final hidden state into vocabulary logits.", 1.0, 0.0,
                ),
                final_state.output_vector,
                logits,
                "Produces one logit for every vocabulary token.",
            )
            states_by_name["predictions"] = LayerState(
                Layer(
                    "predictions", "top_tokens",
                    "Select and decode the highest-scoring next tokens.", 1.0, 0.0,
                ),
                logits,
                logits,
                "Ranks the vocabulary logits without changing them.",
            )
            states_by_name["end"] = LayerState(
                Layer(
                    "end", "end", "End of the authored model program.", 1.0, 0.0,
                ),
                logits,
                logits,
                "Terminal breakpoint with predictions available.",
            )
        previous = None
        for path in self.step_paths:
            if path.startswith("nop:"):
                if previous is None:
                    raise ValueError("nop() requires an earlier model operation")
                states_by_name[path] = LayerState(
                    Layer(
                        path, "nop", "No operation; stable breakpoint location.",
                        1.0, 0.0,
                    ),
                    previous.output_vector,
                    previous.output_vector,
                    "Preserves the current value without executing a model operation.",
                )
            previous = states_by_name[path]
        self.trace = tuple(states_by_name[path] for path in self.step_paths)
        self.token_index = position

    @staticmethod
    def _parse_token_condition(condition: str) -> int | None:
        parsed = LLMDebugAdapter._parse_breakpoint_condition(condition)
        if parsed is None or parsed[0] is None or parsed[0][0] != "==":
            return None
        return parsed[0][1]

    @staticmethod
    def _parse_breakpoint_condition(
        condition: str,
    ) -> tuple[tuple[str, int] | None, tuple[str, int] | None] | None:
        """Parse declarative breakpoint filters without evaluating Python code."""
        if not condition.strip():
            return None, None
        values: dict[str, tuple[str, int]] = {}
        terms = re.split(r"\s*(?:,|&&|\band\b)\s*", condition, flags=re.IGNORECASE)
        for term in terms:
            match = re.fullmatch(
                r"(token|layer)\s*(==|=|<=|>=|<|>)\s*(\d+)",
                term,
                flags=re.IGNORECASE,
            )
            if match is None:
                return None
            name = match.group(1).lower()
            if name in values:
                return None
            operator = "==" if match.group(2) == "=" else match.group(2)
            values[name] = (operator, int(match.group(3)))
        return values.get("token"), values.get("layer")

    @staticmethod
    def _transformer_layer_index(path: str) -> int | None:
        match = re.match(r"(?:transformer\.h|model\.layers)\.(\d+)(?:\.|$)", path)
        if match is not None:
            return int(match.group(1))
        nop_match = re.fullmatch(r"nop:\d+:(\d+)", path)
        if nop_match is not None:
            return int(nop_match.group(1))
        demo_match = re.fullmatch(r"(?:attention|mlp)_(\d+)", path)
        return int(demo_match.group(1)) if demo_match is not None else None

    def _current_lens_phase(self) -> int:
        """Return the explicit block phase: pre, attention, mid, MLP, or post."""
        path = self.outer_state.layer.name
        function_name = self._function_name_for_path(path)
        if function_name in {"attention", "full_attention", "gated_linear_attention"}:
            if (
                self.inside_internal
                and self.internal_steps[self.mlp_step_index][0] == "residual_add"
            ):
                return 2
            return 1 if self.inside_internal else 0
        if function_name == "mlp":
            if (
                self.inside_internal
                and self.internal_steps[self.mlp_step_index][0] == "residual_add"
            ):
                return 4
            return 3 if self.inside_internal else 2
        block = self._transformer_layer_index(path)
        if path.startswith("nop:") and block is not None:
            prior_function = ""
            next_function = ""
            for index in range(self.layer_index - 1, -1, -1):
                candidate = self.trace[index]
                if self._transformer_layer_index(candidate.layer.name) == block:
                    prior_function = self._function_name_for_path(candidate.layer.name)
                    if prior_function:
                        break
            for index in range(self.layer_index + 1, len(self.trace)):
                candidate = self.trace[index]
                if self._transformer_layer_index(candidate.layer.name) == block:
                    next_function = self._function_name_for_path(candidate.layer.name)
                    if next_function:
                        break
            if prior_function == "mlp":
                return 4
            if prior_function in {
                "attention", "full_attention", "gated_linear_attention"
            }:
                return 2
            if next_function in {
                "attention", "full_attention", "gated_linear_attention"
            }:
                return 0
        return 5

    def _lens_result_is_saved(self, view: str, result_layer: str | None = None) -> bool:
        lens_phase = {
            "residual-pre": 0,
            "residual-input": 0,
            "residual-stream": 0,
            "attention-neurons": 1,
            "attention-output": 1,
            "residual-mid": 2,
            "neurons": 3,
            "mlp-output": 3,
            "residual-post": 4,
            "residual-output": 4,
        }[view]
        if result_layer is not None:
            active_block = self._transformer_layer_index(self.outer_state.layer.name)
            result_block = self._transformer_layer_index(result_layer)
            if (
                active_block is not None
                and result_block is not None
                and active_block != result_block
            ):
                return True
        return lens_phase < self._current_lens_phase()

    def _breakpoint_matches(
        self,
        breakpoint: SourceBreakpoint,
        token_position: int,
        trace_layer_index: int,
    ) -> bool:
        def compare(actual: int, operator: str, expected: int) -> bool:
            return {
                "==": actual == expected,
                "<": actual < expected,
                "<=": actual <= expected,
                ">": actual > expected,
                ">=": actual >= expected,
            }[operator]

        if (
            breakpoint.token_position is not None
            and not compare(
                token_position,
                breakpoint.token_operator,
                breakpoint.token_position,
            )
        ):
            return False
        if breakpoint.transformer_layer is None:
            return True
        path = (
            self.step_paths[trace_layer_index]
            if trace_layer_index < len(self.step_paths)
            else ""
        )
        actual_layer = self._transformer_layer_index(path)
        return (
            actual_layer is not None
            and compare(
                actual_layer,
                breakpoint.layer_operator,
                breakpoint.transformer_layer,
            )
        )

    def _breakpoint_source_key(self, path: str | None) -> str:
        """Return a stable DAP source key without letting other files clear this source."""
        if not path:
            return self.source_path or f"reference:{self.SOURCE_REFERENCE}"
        return str(Path(path).expanduser().resolve())

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
                locations.extend(
                    (line, index)
                    for index, line in enumerate(self._internal_source_lines(path))
                )
                for location_index, (line, internal_index) in enumerate(locations):
                    position = (token_position, layer_index, location_index)
                    if not before_start and (
                        position < current or (position == current and not include_current)
                    ):
                        continue
                    if any(
                        breakpoint.line == line
                        and self._breakpoint_matches(
                            breakpoint, token_position, layer_index
                        )
                        for breakpoint in self.breakpoints
                    ):
                        self._load_token(token_position)
                        self.layer_index = layer_index
                        function_name = self._function_name_for_path(path)
                        self.inside_attention = (
                            function_name in {"attention", "full_attention", "gated_linear_attention"}
                            and internal_index is not None
                        )
                        self.inside_mlp = function_name == "mlp" and internal_index is not None
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
        lines = self._internal_source_lines(path)
        if not lines:
            return False
        for index, line in enumerate(lines):
            if any(
                breakpoint.line == line
                and self._breakpoint_matches(
                    breakpoint, self.token_index, self.layer_index
                )
                for breakpoint in self.breakpoints
            ):
                function_name = self._function_name_for_path(path)
                self.inside_attention = function_name in {
                    "attention", "full_attention", "gated_linear_attention"
                }
                self.inside_mlp = function_name == "mlp"
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
        if path == "vocabulary_projection":
            return ["x"]
        if path == "predictions" or self._predictions_are_available():
            return ["logits"]
        if self.inside_internal or self.layer_index > 0 or path != "token_embedding":
            names.append("x")
        if self.inside_internal:
            function_name = self._function_name_for_path(self.outer_state.layer.name)
            completed = {
                operation for operation, _vector, _meaning in self.internal_steps[: self.mlp_step_index]
            }
            latest: dict[str, str] = {}
            for binding in self.function_bindings.get(function_name, []):
                if binding.get("operation") in completed:
                    latest[str(binding["name"])] = str(binding["operation"])
            names.extend(name for name in latest if name not in names)
        return names

    def _predictions_are_available(self) -> bool:
        """Whether the current source location follows model.top_tokens(...)."""
        step_paths = getattr(self, "step_paths", ())
        try:
            predictions_index = step_paths.index("predictions")
        except ValueError:
            return False
        return self.layer_index > predictions_index

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
                "name": "token", "value": str(self.token_index), "type": "int",
                "variablesReference": 0, "evaluateName": "token",
            },
        ]
        step_paths = getattr(self, "step_paths", ())
        path = step_paths[self.layer_index] if self.layer_index < len(step_paths) else ""
        transformer_layer = self._transformer_layer_index(path)
        if transformer_layer is not None:
            variables.extend(
                [
                    {
                        "name": "layer", "value": str(transformer_layer), "type": "int",
                        "variablesReference": 0, "evaluateName": "layer",
                    },
                    {
                        "name": "block", "value": f"model.layers[{transformer_layer}]",
                        "type": "LayerHandle", "variablesReference": 0,
                        "evaluateName": "block",
                    },
                ]
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
        path = self.step_paths[self.layer_index] if self.layer_index < len(self.step_paths) else ""
        if self._predictions_are_available():
            variables.append(
                {
                    "name": "predictions", "value": f"{len(self.predictions)} tokens",
                    "type": "next-token predictions",
                    "variablesReference": self.SCOPE_PREDICTIONS,
                    "evaluateName": "predictions",
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
        if reference == self.SCOPE_BREAKPOINT_CONDITIONS:
            transformer_layer = self._transformer_layer_index(self.outer_state.layer.name)
            return [
                {
                    "name": "token",
                    "value": str(self.token_index),
                    "type": "breakpoint condition · token=N",
                    "variablesReference": 0,
                    "evaluateName": "token",
                },
                {
                    "name": "layer",
                    "value": (
                        str(transformer_layer)
                        if transformer_layer is not None else "not in a transformer block"
                    ),
                    "type": "breakpoint condition · layer=N",
                    "variablesReference": 0,
                    "evaluateName": "layer",
                },
            ]
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
        if reference == self.SCOPE_PREDICTIONS:
            return [
                {
                    "name": f"[{index}]",
                    "value": (
                        f"{prediction.text!r} · logit {prediction.logit:.2f} · "
                        f"{prediction.probability:.2%}"
                    ),
                    "type": "token prediction", "variablesReference": 0,
                    "evaluateName": f"predictions[{index}]",
                }
                for index, prediction in enumerate(self.predictions)
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
                    "value": (
                        f"activation {neuron.activation:+.6f} · "
                        f"|delta| {abs(neuron.activation) * neuron.direction_norm:.6f}"
                    ),
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
                r"mlp_gate_pre_activation|mlp_gate_activation|mlp_up_projection|"
                r"attention_qkv|attention_mixing|attention_projection|"
                r"full_attention_mixing|gated_delta_mixing)"
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
        getattr(self, "lens_snapshot_cache", {}).clear()
        getattr(self, "measurement_vectors", {}).clear()
        return value

    def _remove_ica_component(self, view: str, component: int) -> dict[str, Any]:
        """Ablate one residual ICA component and replay the current token."""
        if self.execution is None:
            raise ValueError("ICA intervention requires a loaded model")
        if view not in {
            "ica-residual-pre", "ica-residual-mid", "ica-residual-post"
        }:
            raise ValueError("ICA removal is available only for residual vectors")
        if component < 0:
            raise ValueError("ICA component must be non-negative")
        snapshot = self._neuron_lens_snapshot(
            view=view,
            include_logit_lens=False,
            include_j_lens=False,
            include_r_lens=False,
        )
        if snapshot.get("saved"):
            raise ValueError(
                "ICA intervention is unavailable because this block has already executed"
            )
        layer = str(snapshot["layer"])
        vector_name = "input" if view == "ica-residual-pre" else "output"
        before_vector = self._ica_residual_vector(view, layer)
        stop_details = {
            "source": self.source_path,
            "line": (
                self._internal_source_lines()[self.mlp_step_index]
                if self.inside_internal and self._internal_source_lines()
                else self.source_lines[self.layer_index]
            ),
            "traceIndex": self.layer_index,
            "state": self.state.layer.name,
            "operation": self.state.layer.operation,
            "insideInternalFunction": self.inside_internal,
            "internalStepIndex": self.mlp_step_index if self.inside_internal else None,
        }
        self.execution.remove_ica_component(
            self.token_index, layer, vector_name, component
        )
        current_layer = self.layer_index
        self._load_token(self.token_index)
        self.layer_index = current_layer
        after_vector = self._ica_residual_vector(view, layer)
        self.neuron_cache.clear()
        getattr(self, "lens_snapshot_cache", {}).clear()
        getattr(self, "measurement_vectors", {}).clear()
        log_paths = self._write_ica_intervention_log(
            view=view,
            layer=layer,
            vector_name=vector_name,
            component=component,
            stop_details=stop_details,
            before=before_vector,
            after=after_vector,
        )
        self.send_event(
            "output",
            {
                "category": "console",
                "output": f"llmdbg: ICA intervention log: {log_paths['note']}\n",
            },
        )
        return {
            "removed": True,
            "view": view,
            "component": component,
            "tokenPosition": self.token_index,
            "layer": layer,
            "log": log_paths,
        }

    def _ica_residual_vector(
        self, view: str, layer: str
    ) -> tuple[float, ...]:
        """Resolve the effective residual at an ICA intervention boundary."""
        if self.execution is None:
            raise ValueError("ICA intervention requires a loaded model")
        state = next(
            (candidate for candidate in self.trace if candidate.layer.name == layer),
            None,
        )
        if state is None:
            raise ValueError(f"residual state {layer} is unavailable")
        if view != "ica-residual-pre":
            return tuple(float(value) for value in state.output_vector)
        active = sorted(
            key
            for key in self.execution.ica_interventions
            if key[0] == self.token_index and key[1] == layer and key[2] == "input"
        )
        if active:
            observation = self.execution.ica_observations.get(active[-1])
            if observation is not None:
                return observation[1]
        return tuple(float(value) for value in state.input_vector)

    def _write_ica_intervention_log(
        self,
        *,
        view: str,
        layer: str,
        vector_name: str,
        component: int,
        stop_details: dict[str, Any],
        before: tuple[float, ...],
        after: tuple[float, ...],
    ) -> dict[str, str]:
        """Write a reproducible JSON record and readable experiment note."""
        if self.execution is None:
            raise ValueError("ICA intervention requires a loaded model")
        def vector_record(values: tuple[float, ...]) -> dict[str, Any]:
            encoded = json.dumps(values, separators=(",", ":")).encode("utf-8")
            return {
                "dimensions": len(values),
                "norm": sum(value * value for value in values) ** 0.5,
                "sha256": hashlib.sha256(encoded).hexdigest(),
                "values": values,
            }

        def package_version(name: str) -> str:
            try:
                return importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                return "unknown"

        transformer_layer = self._transformer_layer_index(layer)
        from llmdbg.ica_lens import top_components

        before_component = top_components(
            self.model_name, int(transformer_layer), before,
            is_delta=False, count=0, include_indices=(component,),
        )[0]
        after_component = top_components(
            self.model_name, int(transformer_layer), after,
            is_delta=False, count=0, include_indices=(component,),
        )[0]
        start = next(
            (index for index, state in enumerate(self.trace) if state.layer.name == layer),
            0,
        )
        downstream = []
        for state in self.trace[start:]:
            output = tuple(float(value) for value in state.output_vector)
            delta_norm = None
            if len(state.input_vector) == len(state.output_vector):
                delta_norm = sum(
                    (float(right) - float(left)) ** 2
                    for left, right in zip(
                        state.input_vector, state.output_vector, strict=True
                    )
                ) ** 0.5
            downstream.append(
                {
                    "state": state.layer.name,
                    "operation": state.layer.operation,
                    "outputNorm": sum(value * value for value in output) ** 0.5,
                    "deltaNorm": delta_norm,
                    "outputSha256": hashlib.sha256(
                        json.dumps(output, separators=(",", ":")).encode("utf-8")
                    ).hexdigest(),
                }
            )
        record = {
            "schema": "llmdbg.ica-intervention.v1",
            "createdAt": datetime.now(timezone.utc).isoformat(),
            "model": {
                "id": self.model_name,
                "revision": self.model_revision,
                "requestedDevice": self.requested_device,
                "actualDevice": self.execution.device,
                "requestedDtype": self.requested_dtype,
                "actualDtype": self.execution.dtype,
            },
            "environment": {
                "python": platform.python_version(),
                "torch": package_version("torch"),
                "transformers": package_version("transformers"),
                "llmdbg": package_version("llmdbg"),
            },
            "debugProgram": {
                "path": self.source_path,
                "source": (
                    Path(self.source_path).read_text(encoding="utf-8")
                    if self.source_path and Path(self.source_path).is_file()
                    else self.source_content
                ),
            },
            "prompt": self.prompt,
            "tokens": list(self.tokens),
            "stop": stop_details,
            "intervention": {
                "kind": "remove_ica_component",
                "tokenPosition": self.token_index,
                "token": self.tokens[self.token_index],
                "view": view,
                "stage": view.removeprefix("ica-residual-"),
                "state": layer,
                "transformerLayer": transformer_layer,
                "basis": "resid_post",
                "component": component,
                "scoreBefore": before_component["score"],
                "scoreAfter": after_component["score"],
            },
            "activeInterventions": [
                {
                    "tokenPosition": position,
                    "state": active_layer,
                    "vector": active_vector,
                    "component": active_component,
                    "transformerLayer": self._transformer_layer_index(active_layer),
                    "basis": "resid_post",
                }
                for position, active_layer, active_vector, active_component
                in sorted(self.execution.ica_interventions)
            ],
            "residualBefore": vector_record(before),
            "residualAfter": vector_record(after),
            "downstream": downstream,
            "predictions": [
                {
                    "tokenId": prediction.token_id,
                    "text": prediction.text,
                    "logit": prediction.logit,
                    "probability": prediction.probability,
                }
                for prediction in self.predictions
            ],
            "reproduce": {
                "model": self.model_name,
                "revision": self.model_revision,
                "prompt": self.prompt,
                "tokenPosition": self.token_index,
                "layer": transformer_layer,
                "stage": view.removeprefix("ica-residual-"),
                "icaBasis": "resid_post",
                "component": component,
                "activeInterventions": [
                    {
                        "tokenPosition": position,
                        "state": active_layer,
                        "vector": active_vector,
                        "component": active_component,
                    }
                    for position, active_layer, active_vector, active_component
                    in sorted(self.execution.ica_interventions)
                ],
            },
        }
        project = Path(self.source_path).parent if self.source_path else Path.cwd()
        log_dir = project / ".logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        stem = f"{timestamp}-ica-remove-t{self.token_index}-l{transformer_layer}-c{component}"
        json_path = log_dir / f"{stem}.json"
        note_path = log_dir / f"{stem}.md"
        json_path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        predictions = ", ".join(
            f"{item['text']!r} ({item['logit']:+.4f})"
            for item in record["predictions"][:10]
        ) or "not available"
        note_path.write_text(
            "\n".join(
                [
                    "# llmdbg ICA intervention",
                    "",
                    f"- Time: `{record['createdAt']}`",
                    f"- Model: `{self.model_name}` at `{self.model_revision}`",
                    f"- Prompt: `{self.prompt}`",
                    f"- Stop: token {self.token_index} {self.tokens[self.token_index]!r}, `{stop_details['state']}`, source line {stop_details['line']}",
                    f"- Intervention: remove `L{transformer_layer}:C{component}` from Residual {view.removeprefix('ica-residual-').title()} using the resid-post ICA basis",
                    f"- ICA score: {before_component['score']:+.8f} → {after_component['score']:+.8f}",
                    f"- Residual norm: {record['residualBefore']['norm']:.8f} → {record['residualAfter']['norm']:.8f}",
                    f"- Residual hashes: `{record['residualBefore']['sha256']}` → `{record['residualAfter']['sha256']}`",
                    f"- Final predictions: {predictions}",
                    "",
                    f"Machine-readable record: `{json_path.name}`",
                    "The JSON record contains the exact before/after residual vectors and downstream state hashes for standalone reproduction.",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        return {"note": str(note_path), "json": str(json_path)}

    def _remove_projected_component(
        self, view: str, component: int
    ) -> dict[str, Any]:
        """Zero one MLP neuron or attention channel and replay the token."""
        if self.execution is None:
            raise ValueError("component intervention requires a loaded model")
        if view not in {"neurons", "attention-neurons"}:
            raise ValueError("component removal requires a neuron or channel view")
        if component < 0:
            raise ValueError("component must be non-negative")
        snapshot = self._neuron_lens_snapshot(
            view=view,
            include_logit_lens=False,
            include_j_lens=False,
            include_r_lens=False,
        )
        if snapshot.get("saved"):
            raise ValueError(
                "component intervention is unavailable because this block has already executed"
            )
        selected = next(
            (item for item in snapshot["neurons"] if item["index"] == component),
            None,
        )
        if selected is None:
            raise ValueError(f"component {component} is not available in this Lens")
        layer = str(snapshot["layer"])
        function_name = self._function_name_for_path(layer)
        vector_name = (
            "mlp_expansion"
            if view == "neurons"
            else {
                "attention": "attention_mixing",
                "full_attention": "full_attention_mixing",
                "gated_linear_attention": "gated_delta_mixing",
            }[function_name]
        )
        before_state = next(state for state in self.trace if state.layer.name == layer)
        before_output = tuple(float(value) for value in before_state.output_vector)
        before_predictions = tuple(self.predictions)
        before_activation = float(before_state.details[vector_name][component])
        stop_details = {
            "source": self.source_path,
            "line": (
                self._internal_source_lines()[self.mlp_step_index]
                if self.inside_internal and self._internal_source_lines()
                else self.source_lines[self.layer_index]
            ),
            "traceIndex": self.layer_index,
            "state": self.state.layer.name,
            "operation": self.state.layer.operation,
        }
        self.execution.set_intervention(
            self.token_index, layer, vector_name, component, 0.0
        )
        current_layer = self.layer_index
        self._load_token(self.token_index)
        self.layer_index = current_layer
        after_state = next(state for state in self.trace if state.layer.name == layer)
        after_output = tuple(float(value) for value in after_state.output_vector)
        after_activation = float(after_state.details[vector_name][component])
        self.neuron_cache.clear()
        getattr(self, "lens_snapshot_cache", {}).clear()
        getattr(self, "measurement_vectors", {}).clear()
        log_paths = self._write_projected_component_log(
            view=view,
            layer=layer,
            vector_name=vector_name,
            component=component,
            before_activation=before_activation,
            after_activation=after_activation,
            before_output=before_output,
            after_output=after_output,
            before_predictions=before_predictions,
            stop_details=stop_details,
        )
        self.send_event(
            "output",
            {
                "category": "console",
                "output": f"llmdbg: component intervention log: {log_paths['note']}\n",
            },
        )
        return {
            "removed": True,
            "view": view,
            "component": component,
            "tokenPosition": self.token_index,
            "layer": layer,
            "log": log_paths,
        }

    def _write_projected_component_log(
        self,
        *,
        view: str,
        layer: str,
        vector_name: str,
        component: int,
        before_activation: float,
        after_activation: float,
        before_output: tuple[float, ...],
        after_output: tuple[float, ...],
        before_predictions: tuple[Any, ...],
        stop_details: dict[str, Any],
    ) -> dict[str, str]:
        """Record a neuron/channel ablation and its downstream result."""
        if self.execution is None:
            raise ValueError("component intervention requires a loaded model")

        def digest(values: tuple[float, ...]) -> str:
            return hashlib.sha256(
                json.dumps(values, separators=(",", ":")).encode("utf-8")
            ).hexdigest()

        def predictions(items: tuple[Any, ...]) -> list[dict[str, Any]]:
            return [
                {
                    "tokenId": item.token_id,
                    "text": item.text,
                    "logit": item.logit,
                    "probability": item.probability,
                }
                for item in items
            ]

        transformer_layer = self._transformer_layer_index(layer)
        kind = "attention_channel" if view == "attention-neurons" else "mlp_neuron"
        operation = "channel-remove" if view == "attention-neurons" else "neuron-remove"
        start = next(
            (index for index, state in enumerate(self.trace) if state.layer.name == layer),
            0,
        )
        downstream = [
            {
                "state": state.layer.name,
                "operation": state.layer.operation,
                "outputNorm": sum(float(value) ** 2 for value in state.output_vector) ** 0.5,
                "outputSha256": digest(tuple(float(value) for value in state.output_vector)),
            }
            for state in self.trace[start:]
        ]
        record = {
            "schema": "llmdbg.projected-component-intervention.v1",
            "createdAt": datetime.now(timezone.utc).isoformat(),
            "model": {
                "id": self.model_name,
                "revision": self.model_revision,
                "requestedDevice": self.requested_device,
                "actualDevice": self.execution.device,
                "requestedDtype": self.requested_dtype,
                "actualDtype": self.execution.dtype,
            },
            "prompt": self.prompt,
            "tokens": list(self.tokens),
            "stop": stop_details,
            "intervention": {
                "kind": f"remove_{kind}",
                "tokenPosition": self.token_index,
                "token": self.tokens[self.token_index],
                "state": layer,
                "transformerLayer": transformer_layer,
                "activationVector": vector_name,
                "component": component,
                "activationBefore": before_activation,
                "activationAfter": after_activation,
            },
            "activeInterventions": [
                {
                    "tokenPosition": position,
                    "state": active_layer,
                    "vector": active_vector,
                    "component": active_component,
                    "value": value,
                }
                for (position, active_layer, active_vector, active_component), value
                in sorted(self.execution.interventions.items())
            ],
            "moduleOutputBefore": {
                "norm": sum(value * value for value in before_output) ** 0.5,
                "sha256": digest(before_output),
                "values": before_output,
            },
            "moduleOutputAfter": {
                "norm": sum(value * value for value in after_output) ** 0.5,
                "sha256": digest(after_output),
                "values": after_output,
            },
            "baselinePredictions": predictions(before_predictions),
            "intervenedPredictions": predictions(tuple(self.predictions)),
            "downstream": downstream,
            "reproduce": {
                "model": self.model_name,
                "revision": self.model_revision,
                "prompt": self.prompt,
                "tokenPosition": self.token_index,
                "layer": transformer_layer,
                "activationVector": vector_name,
                "component": component,
                "replacement": 0.0,
            },
        }
        project = Path(self.source_path).parent if self.source_path else Path.cwd()
        log_dir = project / ".logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        stem = f"{timestamp}-{operation}-t{self.token_index}-l{transformer_layer}-c{component}"
        json_path = log_dir / f"{stem}.json"
        note_path = log_dir / f"{stem}.md"
        json_path.write_text(
            json.dumps(record, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        baseline = predictions(before_predictions)
        intervened = predictions(tuple(self.predictions))
        note_path.write_text(
            "\n".join(
                [
                    f"# llmdbg {kind.replace('_', ' ')} intervention",
                    "",
                    f"- Time: `{record['createdAt']}`",
                    f"- Model: `{self.model_name}` at `{self.model_revision}`",
                    f"- Prompt: `{self.prompt}`",
                    f"- Stop: token {self.token_index} {self.tokens[self.token_index]!r}, `{stop_details['state']}`, source line {stop_details['line']}",
                    f"- Intervention: set `{vector_name}[{component}]` to zero at layer {transformer_layer}",
                    f"- Activation: {before_activation:+.8f} → {after_activation:+.8f}",
                    f"- Module output norm: {record['moduleOutputBefore']['norm']:.8f} → {record['moduleOutputAfter']['norm']:.8f}",
                    f"- Baseline top prediction: {baseline[0]['text']!r} ({baseline[0]['probability']:.2%})" if baseline else "- Baseline top prediction: unavailable",
                    f"- Intervened top prediction: {intervened[0]['text']!r} ({intervened[0]['probability']:.2%})" if intervened else "- Intervened top prediction: unavailable",
                    "",
                    f"Machine-readable record: `{json_path.name}`",
                    "The JSON record contains exact module output vectors, baseline/intervened predictions, active interventions, and downstream hashes.",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        return {"note": str(note_path), "json": str(json_path)}

    def _top_neurons(
        self,
        include_logit_lens: bool = True,
        include_j_lens: bool = False,
        include_r_lens: bool = False,
        include_jpp_lens: bool = False,
    ) -> tuple[NeuronProjection, ...]:
        """Return cached direct vocabulary projections for the current MLP state."""
        inside_mlp = (
            self.inside_internal
            and self._function_name_for_path(self.outer_state.layer.name) == "mlp"
        )
        result_state = (
            self.outer_state
            if inside_mlp
            else self._prior_operation_state("mlp")
        )
        expansion_complete = not inside_mlp or any(
            vector == "mlp_expansion"
            for _operation, vector, _meaning in self.internal_steps[: self.mlp_step_index]
        )
        if (
            self.execution is None
            or result_state is None
            or not expansion_complete
            or "mlp_expansion" not in result_state.details
        ):
            return ()
        layer_name = result_state.layer.name
        removed_indices = self._removed_projected_components(
            layer_name, "mlp_expansion"
        )
        key = (
            self.token_index, layer_name,
            include_logit_lens, include_j_lens, include_r_lens, include_jpp_lens,
            removed_indices,
        )
        if key not in self.neuron_cache:
            self.neuron_cache[key] = self.execution.top_mlp_neurons(
                layer_name,
                result_state.details["mlp_expansion"],
                output=result_state.details.get("mlp_projection"),
                residual=result_state.input_vector,
                include_logit_lens=include_logit_lens,
                include_j_lens=include_j_lens,
                include_r_lens=include_r_lens,
                include_jpp_lens=include_jpp_lens,
                include_indices=removed_indices,
            )
        return self.neuron_cache[key]

    def _top_attention_channels(
        self,
        include_logit_lens: bool = True,
        include_j_lens: bool = False,
        include_r_lens: bool = False,
        include_jpp_lens: bool = False,
    ) -> tuple[NeuronProjection, ...]:
        """Return the strongest mixed-attention channels and their W_O directions."""
        function_names = {"attention", "full_attention", "gated_linear_attention"}
        inside_attention = (
            self.inside_internal
            and self._function_name_for_path(self.outer_state.layer.name) in function_names
        )
        result_state = (
            self.outer_state
            if inside_attention
            else self._prior_operation_state(*function_names)
        )
        if self.execution is None or result_state is None:
            return ()
        function_name = self._function_name_for_path(result_state.layer.name)
        vector_name = {
            "attention": "attention_mixing",
            "full_attention": "full_attention_mixing",
            "gated_linear_attention": "gated_delta_mixing",
        }[function_name]
        mixing_complete = not inside_attention or any(
            detail == vector_name
            for _operation, detail, _meaning in self.internal_steps[: self.mlp_step_index]
        )
        activation = result_state.details.get(vector_name)
        if not mixing_complete or activation is None:
            return ()
        layer_name = result_state.layer.name
        removed_indices = self._removed_projected_components(
            layer_name, vector_name
        )
        key = (
            self.token_index, "attention:" + layer_name,
            include_logit_lens, include_j_lens, include_r_lens, include_jpp_lens,
            removed_indices,
        )
        if key not in self.neuron_cache:
            self.neuron_cache[key] = self.execution.top_attention_channels(
                layer_name, activation,
                output=result_state.details.get("attention_projection"),
                residual=result_state.input_vector,
                include_logit_lens=include_logit_lens,
                include_j_lens=include_j_lens,
                include_r_lens=include_r_lens,
                include_jpp_lens=include_jpp_lens,
                include_indices=removed_indices,
            )
        return self.neuron_cache[key]

    def _removed_projected_components(
        self, layer_name: str, vector_name: str
    ) -> tuple[int, ...]:
        if self.execution is None:
            return ()
        interventions = getattr(self.execution, "interventions", {})
        return tuple(
            sorted(
                index
                for (position, layer, vector, index), value
                in interventions.items()
                if position == self.token_index
                and layer == layer_name
                and vector == vector_name
                and value == 0.0
            )
        )

    def _prior_operation_state(self, *function_names: str) -> LayerState | None:
        """Return the latest completed matching operation before the current line."""
        for index in range(self.layer_index - 1, -1, -1):
            candidate = self.trace[index]
            if self._function_name_for_path(candidate.layer.name) in function_names:
                return candidate
        return None

    def _evaluate(self, expression: str) -> str | None:
        expression = self._source_expression(expression)
        if expression == "prompt":
            return self.prompt
        if expression == "token":
            return str(self.token_index)
        if expression in ("token_text", "token.text"):
            return self.tokens[self.token_index]
        if expression in ("token_pos", "position", "token.position"):
            return str(self.token_index)
        if expression == "layer":
            transformer_layer = self._transformer_layer_index(
                self.outer_state.layer.name
            )
            return (
                str(transformer_layer)
                if transformer_layer is not None else "not in a transformer block"
            )
        if expression in ("block", "name"):
            transformer_layer = self._transformer_layer_index(
                self.outer_state.layer.name
            )
            return (
                f"model.layers[{transformer_layer}]"
                if transformer_layer is not None else self.state.layer.name
            )
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
            r"mlp_gate_pre_activation|mlp_gate_activation|mlp_up_projection|"
            r"attention_qkv|attention_mixing|attention_projection|"
            r"full_attention_mixing|gated_delta_mixing)"
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
        function_name = self._function_name_for_path(self.outer_state.layer.name)
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
        step_paths = getattr(self, "step_paths", ())
        path = step_paths[self.layer_index] if self.layer_index < len(step_paths) else ""
        if expression == "predictions":
            if not self._predictions_are_available():
                return {
                    "result": "available after model.top_tokens(...) executes",
                    "type": "next-token predictions", "variablesReference": 0,
                }
            return {
                "result": f"{len(self.predictions)} tokens",
                "type": "next-token predictions",
                "variablesReference": self.SCOPE_PREDICTIONS,
                "indexedVariables": len(self.predictions),
            }
        prediction_match = re.fullmatch(
            r"predictions\[(\d+)\](?:\.(token|token_id|logit|probability))?",
            expression,
        )
        if prediction_match:
            index = int(prediction_match.group(1))
            attribute = prediction_match.group(2)
            if not self._predictions_are_available() or index >= len(self.predictions):
                return {
                    "result": f"prediction {index} is not available",
                    "variablesReference": 0,
                }
            prediction = self.predictions[index]
            values = {
                "token": repr(prediction.text),
                "token_id": str(prediction.token_id),
                "logit": f"{prediction.logit:.8f}",
                "probability": f"{prediction.probability:.8f}",
            }
            if attribute is not None:
                return {"result": values[attribute], "variablesReference": 0}
            return {
                "result": (
                    f"{prediction.text!r} · logit {prediction.logit:.2f} · "
                    f"{prediction.probability:.2%}"
                ),
                "type": "token prediction", "variablesReference": 0,
            }
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
                "result": "top 5 residual-contribution magnitudes",
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
        if name == "logits":
            for state in self.trace:
                if state.layer.name == "vocabulary_projection":
                    return state.output_vector
            raise ValueError("logits are not available before vocabulary projection")
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

    def _register_measurement_vector(
        self, label: str, vector: tuple[float, ...]
    ) -> str:
        """Retain a full-width vector behind a compact webview-safe identifier."""
        vectors = getattr(self, "measurement_vectors", None)
        if vectors is None:
            vectors = self.measurement_vectors = {}
        vector_id = f"t{self.token_index}-s{self.layer_index}-v{len(vectors)}"
        vectors[vector_id] = (
            label,
            tuple(float(value) for value in vector),
        )
        return vector_id

    def _measure_cosine(self, vector_a: str, vector_b: str) -> dict[str, Any]:
        """Measure the exact cosine and angle between two retained Lens vectors."""
        if vector_a not in self.measurement_vectors or vector_b not in self.measurement_vectors:
            raise ValueError("one or both selected vectors are no longer available")
        label_a, a = self.measurement_vectors[vector_a]
        label_b, b = self.measurement_vectors[vector_b]
        if len(a) != len(b):
            raise ValueError("selected vectors have different dimensions")
        norm_a = sum(value * value for value in a) ** 0.5
        norm_b = sum(value * value for value in b) ** 0.5
        if norm_a == 0.0 or norm_b == 0.0:
            raise ValueError("cosine is undefined for a zero-length vector")
        cosine = max(
            -1.0,
            min(1.0, sum(x * y for x, y in zip(a, b, strict=True)) / (norm_a * norm_b)),
        )
        import math

        return {
            "labelA": label_a,
            "labelB": label_b,
            "cosine": cosine,
            "angleDegrees": math.degrees(math.acos(cosine)),
            "normA": norm_a,
            "normB": norm_b,
            "dimensions": len(a),
        }

    def _neuron_lens_snapshot(
        self,
        view: str = "neurons",
        include_logit_lens: bool = True,
        include_j_lens: bool = False,
        include_r_lens: bool = False,
        include_jpp_lens: bool = False,
    ) -> dict[str, Any]:
        """Describe top MLP neurons through direct and residual-transport lenses."""
        jpp_lens_label = "J++ Lens"
        if include_jpp_lens:
            from llmdbg.jpp_lens import display_name

            jpp_lens_label = display_name(str(self.model_name))
        requested_view = view
        is_ica = view.startswith("ica-")
        if is_ica:
            view = view.removeprefix("ica-")
            if view not in {
                "residual-pre", "attention-output", "residual-mid",
                "mlp-output", "residual-post",
            }:
                raise ValueError(f"unknown ICA lens view: {requested_view}")
        if view not in {
            "neurons", "attention-neurons", "attention-output",
            "mlp-output", "residual-pre", "residual-mid", "residual-post",
            "residual-input", "residual-output", "residual-stream",
        }:
            raise ValueError(f"unknown lens view: {view}")
        previous_state = self.previous_state
        inside_mlp = (
            self.inside_internal
            and self._function_name_for_path(self.outer_state.layer.name) == "mlp"
        )
        result_state = self.outer_state if inside_mlp else previous_state
        layer = result_state.layer.name if result_state is not None else ""

        def prior_operation(*function_names: str) -> LayerState | None:
            return self._prior_operation_state(*function_names)

        def next_operation(*function_names: str) -> LayerState | None:
            for index in range(self.layer_index + 1, len(self.trace)):
                candidate = self.trace[index]
                function_name = self._function_name_for_path(candidate.layer.name)
                if function_name in function_names:
                    return candidate
                if function_name:
                    return None
            return None

        def block_operation(*function_names: str) -> LayerState | None:
            """Resolve an operation by block identity, independent of nearby no-ops."""
            block = self._transformer_layer_index(self.outer_state.layer.name)
            if block is None:
                return None
            for candidate in self.trace:
                if (
                    self._transformer_layer_index(candidate.layer.name) == block
                    and self._function_name_for_path(candidate.layer.name)
                    in function_names
                ):
                    return candidate
            return None

        def residual_geometry(
            state: LayerState | None, transition: str
        ) -> dict[str, float | str] | None:
            """Return the measured geometry of x + delta for a residual update."""
            if state is None or len(state.input_vector) != len(state.output_vector):
                return None
            input_vector = tuple(float(value) for value in state.input_vector)
            delta = tuple(
                float(output) - float(input_value)
                for input_value, output in zip(
                    state.input_vector, state.output_vector, strict=True
                )
            )
            input_norm = sum(value * value for value in input_vector) ** 0.5
            delta_norm = sum(value * value for value in delta) ** 0.5
            output_norm = sum(
                float(value) * float(value) for value in state.output_vector
            ) ** 0.5
            cosine = None
            if input_norm > 0.0 and delta_norm > 0.0:
                cosine = max(
                    -1.0,
                    min(
                        1.0,
                        sum(
                            input_value * delta_value
                            for input_value, delta_value in zip(
                                input_vector, delta, strict=True
                            )
                        )
                        / (input_norm * delta_norm),
                    ),
                )
            return {
                "transition": transition,
                "inputNorm": input_norm,
                "deltaNorm": delta_norm,
                "outputNorm": output_norm,
                "deltaRatio": delta_norm / input_norm if input_norm > 0.0 else 0.0,
                "cosine": cosine,
            }

        residual_geometry_data = None
        component_geometry_data = None

        if view == "neurons" and not inside_mlp:
            result_state = prior_operation("mlp")
            layer = result_state.layer.name if result_state is not None else ""
        elif view == "attention-neurons":
            attention_names = (
                "attention", "full_attention", "gated_linear_attention"
            )
            inside_attention = (
                self.inside_internal
                and self._function_name_for_path(self.outer_state.layer.name)
                in attention_names
            )
            result_state = (
                self.outer_state
                if inside_attention
                else prior_operation(*attention_names)
            )
            layer = result_state.layer.name if result_state is not None else ""

        if view not in {"neurons", "attention-neurons"}:
            if self.execution is None:
                raise ValueError("a loaded model is required for vector lenses")
            if view in {"residual-pre", "residual-input", "residual-stream"}:
                attention_names = (
                    "attention", "full_attention", "gated_linear_attention"
                )
                current_function = self._function_name_for_path(
                    self.outer_state.layer.name
                )
                if current_function in attention_names:
                    result_state = self.outer_state
                else:
                    result_state = block_operation(*attention_names)
                    if result_state is None:
                        result_state = next_operation(*attention_names)
                    if result_state is None:
                        result_state = prior_operation(*attention_names)
                    if (
                        result_state is None
                        and self._function_name_for_path(self.outer_state.layer.name)
                        == "mlp"
                    ):
                        result_state = self.outer_state
                if result_state is None:
                    raise ValueError(
                        "Residual Lens · Pre requires an attention operation in the current block"
                    )
                layer = result_state.layer.name
                vector = result_state.input_vector
                if is_ica:
                    vector = self._ica_residual_vector(requested_view, layer)
                label = "Residual Lens · Pre"
            elif view == "residual-mid":
                inside_attention = (
                    self.inside_internal
                    and self._function_name_for_path(self.outer_state.layer.name)
                    in {"attention", "full_attention", "gated_linear_attention"}
                )
                if inside_attention:
                    if self.internal_steps[self.mlp_step_index][0] != "residual_add":
                        raise ValueError(
                            "Residual Lens · Mid is available after attention residual addition"
                        )
                    result_state = self.outer_state
                else:
                    result_state = prior_operation(
                        "attention", "full_attention", "gated_linear_attention"
                    )
                    if result_state is None:
                        raise ValueError(
                            "Residual Lens · Mid is available after attention completes"
                        )
                layer = result_state.layer.name
                vector = result_state.output_vector
                label = "Residual Lens · Mid"
                residual_geometry_data = residual_geometry(
                    result_state, "attention update"
                )
            elif view in {"residual-post", "residual-output"}:
                if self.inside_internal:
                    if (
                        self._function_name_for_path(self.outer_state.layer.name) != "mlp"
                        or self.internal_steps[self.mlp_step_index][0] != "residual_add"
                    ):
                        raise ValueError(
                            "Residual Lens · Post is available after MLP residual addition"
                        )
                    result_state = self.outer_state
                else:
                    result_state = prior_operation("mlp")
                if (
                    result_state is None
                    or self._function_name_for_path(result_state.layer.name) != "mlp"
                ):
                    raise ValueError(
                        "Residual Lens · Post is available after an MLP completes"
                    )
                layer = result_state.layer.name
                vector = result_state.output_vector
                label = "Residual Lens · Post"
                residual_geometry_data = residual_geometry(result_state, "MLP update")
            elif view == "attention-output":
                inside_attention = (
                    self.inside_internal
                    and self._function_name_for_path(self.outer_state.layer.name)
                    in {"attention", "full_attention", "gated_linear_attention"}
                )
                attention_state = self.outer_state if inside_attention else None
                if attention_state is None and self.layer_index > 0:
                    attention_state = prior_operation(
                        "attention", "full_attention", "gated_linear_attention"
                    )
                if attention_state is None:
                    raise ValueError(
                        "Attention Output Lens is available after attention output projection completes"
                    )
                projection_complete = not inside_attention or any(
                    detail == "attention_projection"
                    for _operation, detail, _meaning in self.internal_steps[: self.mlp_step_index]
                )
                vector = attention_state.details.get("attention_projection")
                if not projection_complete or vector is None:
                    raise ValueError(
                        "Attention Output Lens is available after attention output projection completes"
                    )
                result_state = attention_state
                layer = attention_state.layer.name
                label = "Attention output"
                function_name = self._function_name_for_path(layer)
                activation_name = {
                    "attention": "attention_mixing",
                    "full_attention": "full_attention_mixing",
                    "gated_linear_attention": "gated_delta_mixing",
                }[function_name]
                activation = attention_state.details.get(activation_name)
                geometry = getattr(self.execution, "component_fan_geometry", None)
                if activation is not None and callable(geometry):
                    component_geometry_data = geometry(layer, activation, vector)
            else:
                if not inside_mlp:
                    result_state = prior_operation("mlp")
                    layer = result_state.layer.name if result_state is not None else ""
                if result_state is None or not layer.endswith(".mlp"):
                    raise ValueError("MLP Output Lens is available after an MLP completes")
                projection_complete = not inside_mlp or any(
                    vector == "mlp_projection"
                    for _operation, vector, _meaning in self.internal_steps[: self.mlp_step_index]
                )
                vector = result_state.details.get("mlp_projection")
                label = "MLP output"
                if not projection_complete or vector is None:
                    raise ValueError("MLP Output Lens is available after down projection completes")
                activation = result_state.details.get("mlp_expansion")
                geometry = getattr(self.execution, "component_fan_geometry", None)
                if activation is not None and callable(geometry):
                    component_geometry_data = geometry(layer, activation, vector)
            def tokens(items: Any) -> list[dict[str, Any]]:
                return [
                    {
                        "rank": rank, "id": token.token_id,
                        "text": token.text, "logit": token.score,
                    }
                    for rank, token in enumerate(items, start=1)
                ]

            if is_ica:
                intervention_vector = (
                    "input" if view == "residual-pre" else "output"
                )
                removed_indices = tuple(
                    component
                    for position, intervention_layer, vector_name, component
                    in self.execution.ica_interventions
                    if position == self.token_index
                    and intervention_layer == layer
                    and vector_name == intervention_vector
                ) if view.startswith("residual-") else ()
                components = self.execution.ica_components(
                    layer,
                    vector,
                    is_delta=view in {"attention-output", "mlp-output"},
                    include_logit_lens=include_logit_lens,
                    include_j_lens=include_j_lens,
                    include_r_lens=include_r_lens,
                    include_jpp_lens=include_jpp_lens,
                    include_indices=removed_indices,
                )
                if not components:
                    raise ValueError(
                        f"ICA Lens is not available for {self.model_name}"
                    )
                return {
                    "view": requested_view,
                    "tokenPosition": self.token_index,
                    "token": self.tokens[self.token_index],
                    "layer": layer,
                    "transformerLayer": self._transformer_layer_index(layer),
                    "saved": self._lens_result_is_saved(view, layer),
                    "vectorLabel": f"ICA {label}",
                    "vectorNorm": sum(float(value) ** 2 for value in vector) ** 0.5,
                    "jLensResolved": include_j_lens,
                    "rLensResolved": include_r_lens,
                    "jppLensResolved": include_jpp_lens,
                    "jppLensLabel": jpp_lens_label,
                    "icaBasis": "resid_post",
                    "icaComponents": [
                        {
                            "rank": component.rank,
                            "index": component.index,
                            "measurementId": self._register_measurement_vector(
                                f"{label} ICA L{self._transformer_layer_index(layer)}:C{component.index}",
                                component.vector,
                            ),
                            "score": component.score,
                            "energy": component.energy,
                            "removed": component.index in removed_indices,
                            "componentNorm": component.projection.norm,
                            "logitTokens": tokens(component.projection.vocabulary),
                            "jLensTokens": tokens(component.projection.j_lens),
                            "rLensTokens": tokens(component.projection.r_lens),
                            "jppLensTokens": tokens(
                                getattr(component.projection, "jpp_lens", ())
                            ),
                        }
                        for component in components
                    ],
                }

            projection = self.execution.vector_lens(
                layer,
                vector,
                include_logit_lens=include_logit_lens,
                include_j_lens=include_j_lens,
                include_r_lens=include_r_lens,
                include_jpp_lens=include_jpp_lens,
            )

            return {
                "view": requested_view,
                "tokenPosition": self.token_index,
                "token": self.tokens[self.token_index],
                "layer": layer,
                "transformerLayer": self._transformer_layer_index(layer),
                "saved": self._lens_result_is_saved(view, layer),
                "vectorLabel": label,
                "vectorNorm": projection.norm,
                "measurementId": self._register_measurement_vector(
                    f"{label} [layer {self._transformer_layer_index(layer)}]",
                    vector,
                ),
                "residualGeometry": residual_geometry_data,
                "componentGeometry": component_geometry_data,
                "jLensResolved": include_j_lens,
                "rLensResolved": include_r_lens,
                "jppLensResolved": include_jpp_lens,
                "jppLensLabel": jpp_lens_label,
                "logitTokens": tokens(projection.vocabulary),
                "jLensTokens": tokens(projection.j_lens),
                "rLensTokens": tokens(projection.r_lens),
                "jppLensTokens": tokens(getattr(projection, "jpp_lens", ())),
            }
        if view == "attention-neurons":
            neurons = self._top_attention_channels(
                include_logit_lens=include_logit_lens,
                include_j_lens=include_j_lens, include_r_lens=include_r_lens,
                include_jpp_lens=include_jpp_lens,
            )
            if not neurons:
                raise ValueError(
                    "Attention Output Channel Lens is available after attention mixing completes"
                )
            scale_record = attention_channel_activation_scale(
                getattr(self, "model_name", ""), layer
            )
        else:
            neurons = self._top_neurons(
                include_logit_lens=include_logit_lens,
                include_j_lens=include_j_lens, include_r_lens=include_r_lens,
                include_jpp_lens=include_jpp_lens,
            )
            scale_record = mlp_activation_scale(getattr(self, "model_name", ""), layer)
        if not neurons:
            raise ValueError(
                "MLP Neuron Lens is available after an MLP activation has completed"
            )
        scale = float(scale_record["scale"]) if scale_record is not None else None
        return {
            "view": view,
            "componentLabel": (
                "channel" if view == "attention-neurons" else "MLP neuron"
            ),
            "tokenPosition": self.token_index,
            "token": self.tokens[self.token_index],
            "layer": layer,
            "transformerLayer": self._transformer_layer_index(layer),
            "saved": self._lens_result_is_saved(view, layer),
            "jLensResolved": include_j_lens,
            "rLensResolved": include_r_lens,
            "jppLensResolved": include_jpp_lens,
            "jppLensLabel": jpp_lens_label,
            "neurons": [
                {
                    "rank": rank,
                    "index": neuron.index,
                    "measurementId": self._register_measurement_vector(
                        (
                            f"Attention channel L{self._transformer_layer_index(layer)}:C{neuron.index}"
                            if view == "attention-neurons"
                            else f"MLP neuron L{self._transformer_layer_index(layer)}:N{neuron.index}"
                        ),
                        neuron.vector,
                    ),
                    "activation": neuron.activation,
                    "removed": neuron.index in self._removed_projected_components(
                        layer,
                        (
                            {
                                "attention": "attention_mixing",
                                "full_attention": "full_attention_mixing",
                                "gated_linear_attention": "gated_delta_mixing",
                            }[self._function_name_for_path(layer)]
                            if view == "attention-neurons"
                            else "mlp_expansion"
                        ),
                    ),
                    "contributionMagnitude": abs(neuron.activation) * neuron.direction_norm,
                    "outputProjection": neuron.output_projection,
                    "componentGeometry": {
                        "inputNorm": neuron.residual_norm,
                        "deltaNorm": abs(neuron.activation) * neuron.direction_norm,
                        "outputNorm": neuron.potential_norm,
                        "deltaRatio": (
                            abs(neuron.activation) * neuron.direction_norm
                            / neuron.residual_norm
                            if neuron.residual_norm > 0.0 else 0.0
                        ),
                        "cosine": neuron.residual_cosine,
                    },
                    "relativeContribution": (
                        neuron.activation * neuron.direction_norm / scale
                        if scale is not None and scale > 0 else None
                    ),
                    "contributionScale": scale,
                    # Compatibility aliases for older extension builds.
                    "relativeActivation": (
                        neuron.activation * neuron.direction_norm / scale
                        if scale is not None and scale > 0 else None
                    ),
                    "activationScale": scale,
                    "scaleSide": (
                        str(scale_record["side"]) if scale_record is not None else None
                    ),
                    "directionNorm": neuron.direction_norm,
                    "logitTokens": [
                        {
                            "rank": token_rank,
                            "id": token.token_id,
                            "text": token.text,
                            "logit": token.score,
                        }
                        for token_rank, token in enumerate(neuron.vocabulary, start=1)
                    ],
                    # Compatibility for older extension builds.
                    "tokens": [
                        {
                            "rank": token_rank,
                            "id": token.token_id,
                            "text": token.text,
                            "logit": token.score,
                        }
                        for token_rank, token in enumerate(neuron.vocabulary, start=1)
                    ],
                    "rLensTokens": [
                        {
                            "rank": token_rank,
                            "id": token.token_id,
                            "text": token.text,
                            "logit": token.score,
                        }
                        for token_rank, token in enumerate(neuron.r_lens, start=1)
                    ],
                    "jLensTokens": [
                        {
                            "rank": token_rank,
                            "id": token.token_id,
                            "text": token.text,
                            "logit": token.score,
                        }
                        for token_rank, token in enumerate(neuron.j_lens, start=1)
                    ],
                    "jppLensTokens": [
                        {
                            "rank": token_rank,
                            "id": token.token_id,
                            "text": token.text,
                            "logit": token.score,
                        }
                        for token_rank, token in enumerate(
                            getattr(neuron, "jpp_lens", ()), start=1
                        )
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
