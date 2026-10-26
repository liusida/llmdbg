"""Minimal Debug Adapter Protocol implementation for llmdbg."""

from __future__ import annotations

import json
import keyword
import ast
import base64
import hashlib
import importlib.metadata
import io
import platform
import os
import operator
import re
import sys
import traceback
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

from llmdbg.config import load_debug_config
from llmdbg.activation_scales import (
    attention_channel_activation_scale,
    attention_head_norm_scale,
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


class DebugActionError(ValueError):
    """Expected validation or availability failure for a debugger action."""


class LensUnavailable(DebugActionError):
    """A requested Lens vector has not been produced at the current step."""


class PlotShapeError(DebugActionError):
    """A one-dimensional heatmap requires the user to choose an orientation."""


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
    TENSOR_REFERENCE = 10000
    SOURCE_FRAME_REFERENCE = 6000
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
        1900: ("mlp_gate_activation", "MLP gate (after activation)"),
        1257: ("attention_projection_raw", "Attention projection before output normalization"),
        1950: ("mlp_up_projection", "MLP value branch"),
        1251: ("attention_query", "Current-token query (all heads)"),
        1252: ("attention_key", "Current-token key (all KV heads)"),
        1253: ("attention_value", "Current-token value (all KV heads)"),
        1254: ("attention_cached_keys", "Cached keys through the current token"),
        1255: ("attention_cached_values", "Cached values through the current token"),
        1256: ("attention_weights", "Attention weights by head and source position"),
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
        ("query_projection", "attention_query", "Project the current token into one query per attention head."),
        ("key_projection", "attention_key", "Project the current token into keys for the KV heads."),
        ("value_projection", "attention_value", "Project the current token into values for the KV heads."),
        ("cached_keys", "attention_cached_keys", "Append the current keys to keys cached from earlier positions."),
        ("cached_values", "attention_cached_values", "Append the current values to values cached from earlier positions."),
        ("attention_weights", "attention_weights", "Compare each query with cached keys and apply softmax."),
        ("mix_values", "full_attention_mixing", "Mix cached values using attention weights; Qwen3.5 also applies its output gate."),
        ("output_projection", "attention_projection", "Project the mixed attention heads to model width."),
        ("residual_add", "output", "Add the full-attention update to the residual stream."),
    )
    LEGACY_FULL_ATTENTION_STEPS = (
        ("input_layernorm", "normalized_input", "RMS-normalize the residual stream."),
        ("full_attention", "full_attention_mixing", "Apply causal attention over the cached sequence."),
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
    GEMMA_ATTENTION_STEPS = (
        ("input_layernorm", "normalized_input", "RMS-normalize the residual stream."),
        ("full_attention", "full_attention_mixing", "Apply causal multi-head attention."),
        ("output_projection", "attention_projection", "Project attention back to model width."),
        ("post_attention_layernorm", "attention_projection", "RMS-normalize the attention update."),
        ("residual_add", "output", "Add the attention update to the residual stream."),
    )
    GEMMA_MLP_STEPS = (
        ("pre_feedforward_layernorm", "normalized_input", "RMS-normalize the post-attention residual stream."),
        ("gate_projection", "mlp_gate_pre_activation", "Project into the GeGLU gate branch."),
        ("gelu", "mlp_gate_activation", "Apply approximate GELU to the gate branch."),
        ("up_projection", "mlp_up_projection", "Project into the GeGLU value branch."),
        ("gated_multiply", "mlp_expansion", "Multiply the activated gate and value branches."),
        ("down_projection", "mlp_projection", "Project active features back to model width."),
        ("post_feedforward_layernorm", "mlp_projection", "RMS-normalize the MLP update."),
        ("residual_add", "output", "Add the MLP update to the residual stream."),
    )
    GEMMA4_ATTENTION_STEPS = (
        ("input_layernorm", "normalized_input", "Normalize the attention input."),
        ("query_projection", "attention_query", "Project and normalize the current query heads."),
        ("cached_keys", "attention_cached_keys", "Read the actual keys, including KV sharing and the sliding window."),
        ("cached_values", "attention_cached_values", "Read the actual normalized values used by this layer."),
        ("attention_weights", "attention_weights", "Compute masked softmax weights over the source positions."),
        ("mix_values", "full_attention_mixing", "Mix source values using the attention weights."),
        ("output_projection", "attention_projection_raw", "Project attention heads back to model width."),
        ("post_attention_layernorm", "attention_projection", "Normalize the attention output."),
        ("residual_add", "output", "Add the attention update to the residual."),
    )

    def __init__(self, stream: ProtocolStream) -> None:
        self.stream = stream
        self.sequence = 1
        self.trace: tuple[LayerState, ...] = ()
        self.execution: ModelExecution | None = None
        self.tokens: tuple[str, ...] = ()
        self.prompt = ""
        self.messages: list[dict[str, Any]] = []
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
        self.max_new_tokens = 0
        self.generated_tokens = 0
        self.transformer_layer_count = 0

    @property
    def internal_steps(self) -> tuple[tuple[str, str, str], ...]:
        function_name = self._function_name_for_path(self.outer_state.layer.name)
        configured = getattr(self, "configured_internal_steps", {}).get(function_name)
        if configured:
            return configured
        architecture = getattr(
            getattr(getattr(self, "execution", None), "runtime", None),
            "architecture", "",
        )
        if architecture == "gemma4" and function_name == "full_attention":
            return self.GEMMA4_ATTENTION_STEPS
        if architecture in {"gemma2", "gemma4"}:
            if function_name == "full_attention":
                return self.GEMMA_ATTENTION_STEPS
            if function_name == "mlp":
                return self.GEMMA_MLP_STEPS
        if function_name == "full_attention" and len(
            self._internal_source_lines(self.outer_state.layer.name)
        ) == len(self.LEGACY_FULL_ATTENTION_STEPS):
            return self.LEGACY_FULL_ATTENTION_STEPS
        return {
            "attention": self.ATTENTION_STEPS,
            "full_attention": self.FULL_ATTENTION_STEPS,
            "gated_linear_attention": self.GATED_LINEAR_ATTENTION_STEPS,
            "mlp": self.QWEN_MLP_STEPS if self.outer_state.layer.name.startswith("model.layers.") else self.MLP_STEPS,
        }.get(function_name, self.MLP_STEPS)

    @staticmethod
    def _function_name_for_path(path: str) -> str:
        for name in ("gated_linear_attention", "full_attention", "attention", "mlp", "per_layer_residual", "layer_scale"):
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

    def _core_info(self):
        if not self.inside_internal:
            return None
        root = self._function_name_for_path(self.outer_state.layer.name)
        if root in getattr(self,'source_plan_steps',{}):
            calls = getattr(self,'source_calls',{}).get(root,{})
            active = [info for info in calls.values() if info['start'] <= self.mlp_step_index <= info['end']]
            return max(active,key=lambda info:info['call']) if active else None
        info = getattr(self, "nested_cores", {}).get(self._function_name_for_path(self.outer_state.layer.name))
        return info if info and info["start"] <= self.mlp_step_index <= info["end"] else None

    def _active_function_name(self):
        frame = self._source_frame()
        if frame is not None:
            return frame['name']
        info = self._core_info()
        return info["name"] if info else self._function_name_for_path(self.outer_state.layer.name)

    def _source_frames(self):
        if not self.inside_internal:
            return []
        root = self._function_name_for_path(self.outer_state.layer.name)
        steps = getattr(self,'source_plan_steps',{}).get(root,())
        return steps[self.mlp_step_index]['frames'] if self.mlp_step_index < len(steps) else []

    def _source_frame(self):
        frames = self._source_frames()
        index = getattr(self,'inspection_frame_index',-1)
        return frames[index] if frames and -len(frames) <= index < len(frames) else None

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
            previous_index = self.mlp_step_index - 1
            core = self._core_info()
            if core and self.mlp_step_index == core["start"]:
                previous_index = core["call"] - 1
            if previous_index < 0:
                return None
            step_name, vector_name, meaning = self.internal_steps[previous_index]
            output = self._mlp_vector(vector_name)
            input_vector = self._mlp_step_input(previous_index)
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
        if vector_name == 'input':
            return outer.input_vector
        return outer.output_vector if vector_name == "output" else outer.details[vector_name]

    def _mlp_step_input(self, step_index: int) -> tuple[float, ...]:
        if step_index == 0:
            return self.outer_state.input_vector
        previous_index = step_index - 1
        core = self._core_info()
        if core and step_index == core["start"]:
            previous_index = core["call"] - 1
        if previous_index < 0:
            return self.outer_state.input_vector
        return self._mlp_vector(self.internal_steps[previous_index][1])

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
        self._last_response_request = request["seq"]

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

    def _console(self, message: str, *, warning: bool = False) -> None:
        """Write one actionable progress line to the Debug Console."""
        if not hasattr(self, "stream") or not hasattr(self, "sequence"):
            return
        self.send_event(
            "output",
            {"category": "console" if warning else "stdout", "output": f"llmdbg: {message.rstrip()}\n"},
        )

    def stop(self, reason: str) -> None:
        self.inspection_references = {}
        getattr(self, "lens_snapshot_cache", {}).clear()
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
        self.send_event("output", {"category": "stdout", "output": "llmdbg: model trace complete\n"})
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
            runtime_profile = os.environ.get("LLMDBG_RUNTIME_PROFILE")
            if runtime_profile:
                self.send_event("output", {"category": "stdout", "output": f"llmdbg: Runtime profile: {runtime_profile}; Python: {sys.executable}\n"})
            if arguments.get("config"):
                file_arguments = load_debug_config(str(arguments["config"]))
                arguments = {**file_arguments, **arguments}
            model = arguments.get("model")
            self.model_name = str(model) if model else "built-in demo"
            self.prompt = str(arguments.get("prompt", "The capital of France is"))
            self.messages = list(arguments.get("messages") or [])
            self.model_revision = str(arguments.get("revision", "main"))
            self.requested_device = str(arguments.get("device", "auto"))
            self.requested_dtype = str(arguments.get("dtype", "auto"))
            self.max_new_tokens = max(0, int(arguments.get("maxNewTokens", 0)))
            self.generated_tokens = 0
            if model:
                self.send_event(
                    "output",
                    {"category": "stdout", "output": f"llmdbg: loading {model}\n"},
                )
                self.execution = prepare_model_execution(
                    str(model),
                    revision=str(arguments.get("revision", "main")),
                    prompt=self.prompt or None,
                    messages=arguments.get("messages"),
                    add_generation_prompt=bool(
                        arguments.get("addGenerationPrompt", True)
                    ),
                    enable_thinking=bool(arguments.get('enableThinking', False)),
                    device=str(arguments.get("device", "auto")),
                    dtype=str(arguments.get("dtype", "auto")),
                    progress=lambda output: self.send_event(
                        "output", {"category": "stdout", "output": f"llmdbg: {output}"}
                    ),
                )
                self.prompt = self.execution.rendered_prompt
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
                    "category": "stdout",
                    "output": (
                        f"llmdbg: prepared {trace_name}; waiting for VS Code to finish "
                        "configuring breakpoints\n"
                    ),
                },
            )
        elif command == "configurationDone":
            self.send_response(request)
            if not self.launched:
                return True
            self._console("Breakpoints configured; starting forward execution.")
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
                frames = [frame, parent]
                core = self._core_info()
                source_frames = self._source_frames()
                if source_frames:
                    frame['name'] = source_frames[-1]['name']+' · '+self.internal_steps[self.mlp_step_index][0]
                    frames = [frame]
                    for depth,index in enumerate(range(len(source_frames)-2,-1,-1),1):
                        frames.append({'id':self.MLP_FRAME_ID+100*depth,'name':source_frames[index]['name'],
                                       'source':source,'line':source_frames[index+1]['callLine'],'column':1})
                    frames.append(parent)
                elif core:
                    frames.insert(1, {"id": self.MLP_FRAME_ID + 100, "name": self._function_name_for_path(self.outer_state.layer.name),
                                      "source": source, "line": self._internal_source_lines()[core["call"]], "column": 1})
                self.send_response(request, {"stackFrames": frames, "totalFrames": len(frames)})
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
            source_frames = self._source_frames()
            frame_id = int(arguments.get('frameId',self.MLP_FRAME_ID))
            frame_index = len(source_frames)-1-(frame_id-self.MLP_FRAME_ID)//100
            local_reference = (self.SOURCE_FRAME_REFERENCE+frame_index
                               if source_frames and frame_id >= self.MLP_FRAME_ID and 0 <= frame_index < len(source_frames)
                               else self.SCOPE_TOKEN)
            scopes = [
                {
                    "name": "Breakpoint Conditions",
                    "variablesReference": self.SCOPE_BREAKPOINT_CONDITIONS,
                    "expensive": False,
                },
                {"name": "Local variables", "variablesReference": local_reference, "expensive": False},
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
            frame_reference = self.SOURCE_FRAME_REFERENCE <= reference < self.SOURCE_FRAME_REFERENCE+16
            if frame_reference:
                self.inspection_frame_index = reference-self.SOURCE_FRAME_REFERENCE
            try:
                variables = (
                    self._tensor_variables(reference, arguments)
                    if reference >= self.TENSOR_REFERENCE else self._local_variables() if frame_reference else self._variables(reference)
                )
            finally:
                self.inspection_frame_index = -1
            self.send_response(request, {"variables": variables})
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
            expression = str(arguments.get("expression", "")).strip()
            source_frames = self._source_frames()
            frame_id = int(arguments.get('frameId',self.MLP_FRAME_ID))
            if source_frames and frame_id >= self.MLP_FRAME_ID:
                self.inspection_frame_index = len(source_frames)-1-(frame_id-self.MLP_FRAME_ID)//100
            try:
                body = self._evaluate_body(expression)
            finally:
                self.inspection_frame_index = -1
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
        elif command == "llmdbg/conceptTrace":
            from llmdbg.concept_trace import dispatch
            try:
                result = dispatch(self, arguments)
            except ValueError as error:
                raise DebugActionError(str(error)) from error
            self.send_response(request, result)
        elif command == "llmdbg/promptTokens":
            self.send_response(
                request,
                {
                    "prompt": self.prompt,
                    "tokens": list(self.tokens),
                    "currentToken": self.token_index,
                    "images": self._prompt_images(),
                },
            )
        elif command == "llmdbg/plot":
            self.send_response(
                request,
                self._render_plot(
                    str(arguments.get("expression", "residual_stream")),
                    str(arguments.get("code", "")),
                    shape_action=arguments.get("shapeAction"),
                ),
            )

        elif command == "llmdbg/decodeExpression":
            self.send_response(request, self._decode_expressions(
                str(arguments.get("expression", "x")),
                arguments.get("loop"),
            ))

        elif command == "llmdbg/neuronLens":
            mode = str(arguments.get("mode", "logit"))
            from llmdbg.residual_vocabulary_lenses import activate, available_lenses, lens

            selected_lens = lens(mode.removeprefix("lens:")) if mode.startswith("lens:") else None
            method = str(selected_lens.get("method")) if selected_lens else None
            available = available_lenses(str(self.model_name))
            available_methods = {item["method"] for item in available}
            active_selection: str | dict[str, str] | None = (
                str(selected_lens["id"]) if selected_lens else None
            )
            if mode == "all":
                by_method: dict[str, str] = {}
                for item in available:
                    by_method.setdefault(item["method"], item["id"])
                active_selection = by_method
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
                self._console(
                    f"Computing {mode} readouts for {view}; the first request may "
                    "download and load Lens artifacts."
                )
                with activate(active_selection):
                    snapshot = self._neuron_lens_snapshot(
                        view=view,
                        include_logit_lens=mode in {"logit", "all"},
                        include_j_lens=mode == "j-lens" or method == "j" or (mode == "all" and "j" in available_methods),
                        include_r_lens=mode == "r-lens" or method == "r" or (mode == "all" and "r" in available_methods),
                        include_jpp_lens=mode == "jpp-lens" or method == "jpp" or (mode == "all" and "jpp" in available_methods),
                    )
                if selected_lens:
                    active_metadata = {
                        "id": selected_lens["id"], "label": selected_lens["label"],
                        "method": selected_lens["method"],
                    }
                    snapshot["activeResidualVocabularyLens"] = active_metadata
                    for child in snapshot.get("icaComponents", snapshot.get("neurons", [])):
                        child["activeResidualVocabularyLens"] = active_metadata
                self.lens_snapshot_cache[cache_key] = snapshot
                self._console(f"Finished computing {mode} readouts for {view}.")
            self.send_response(request, snapshot)
        elif command == "llmdbg/deltaMemoryContributions":
            self.send_response(request, self._delta_memory_contributions(
                str(arguments.get('layer', '')), int(arguments.get('head', -1)),
                int(arguments.get('tokenPosition', -1))))
        elif command == "llmdbg/attentionTokenContributions":
            self.send_response(request, self._attention_token_contributions(
                str(arguments.get("layer", "")),
                int(arguments.get("channel", -1)),
                int(arguments.get("tokenPosition", -1)),
            ))
        elif command == "llmdbg/measureCosine":
            self.send_response(
                request,
                self._measure_cosine(
                    str(arguments.get("vectorA", "")),
                    str(arguments.get("vectorB", "")),
                ),
            )
        elif command == "llmdbg/measureCosineAll":
            reference = str(arguments.get("reference", ""))
            reference_vector = arguments.get("referenceVector")
            if isinstance(reference_vector, list):
                reference = self._register_measurement_vector(
                    str(arguments.get("referenceLabel", "Pinned reference")),
                    tuple(float(value) for value in reference_vector),
                )
            targets = arguments.get("targets", [])
            if not isinstance(targets, list):
                raise DebugActionError("cosine comparison targets must be a list")
            available = [
                str(target)
                for target in targets
                if str(target) in self.measurement_vectors
            ]
            self.send_response(
                request,
                {
                    "reference": reference,
                    "results": [
                        {
                            "target": target,
                            **self._measure_cosine(reference, target),
                        }
                        for target in available
                    ],
                    "unavailable": len(targets) - len(available),
                },
            )
        elif command == "llmdbg/getMeasurementVector":
            vector_id = str(arguments.get("vector", ""))
            if vector_id not in self.measurement_vectors:
                raise DebugActionError("the selected vector is no longer available")
            label, vector = self.measurement_vectors[vector_id]
            norm = sum(value * value for value in vector) ** 0.5
            match = re.match(r"^t(\d+)-", vector_id)
            token_position = int(match.group(1)) if match is not None else self.token_index
            self.send_response(
                request,
                {
                    "label": label,
                    "vector": list(vector),
                    "dimensions": len(vector),
                    "norm": norm,
                    "tokenPosition": token_position,
                },
            )
        elif command == "llmdbg/patchVector":
            self.send_event("llmdbg/computationStarted", {"command": command})
            self._console(
                f"Replaying token {self.token_index} with the copied vector patch."
            )
            result = self._patch_vector(
                str(arguments.get("view", "")),
                tuple(float(value) for value in arguments.get("vector", [])),
                str(arguments.get("sourceLabel", "Copied vector")),
            )
            self.send_response(request, result)
            self.send_event("invalidated", {"areas": ["variables", "watch"]})
            self.send_event(
                "llmdbg/vectorChanged",
                {"tokenPosition": self.token_index, "layer": self.state.layer.name},
            )
        elif command == "llmdbg/getTokenReferenceVectors":
            vectors: dict[str, dict[str, Any]] = {}
            attention_names = {"attention", "full_attention", "gated_linear_attention"}
            for state in self.trace:
                layer_index = self._transformer_layer_index(state.layer.name)
                if layer_index is None:
                    continue
                function_name = self._function_name_for_path(state.layer.name)
                candidates: tuple[tuple[str, Any, str], ...] = ()
                if function_name in attention_names:
                    candidates = (
                        ("residual-pre", state.input_vector, "Residual Lens · Pre"),
                        ("attention-output", state.details.get("attention_projection"), "Attention output"),
                        ("residual-mid", state.output_vector, "Residual Lens · Mid"),
                    )
                elif function_name == "mlp":
                    candidates = (
                        ("mlp-output", state.details.get("mlp_projection"), "MLP output"),
                        ("residual-post", state.output_vector, "Residual Lens · Post"),
                    )
                for view, vector, label in candidates:
                    if vector is None:
                        continue
                    values = tuple(float(value) for value in vector)
                    vectors[f"{view}|{layer_index}"] = {
                        "label": f"{label} [layer {layer_index}]",
                        "vector": list(values),
                        "norm": sum(value * value for value in values) ** 0.5,
                    }
            self.send_response(
                request,
                {"tokenPosition": self.token_index, "vectors": vectors},
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
        elif command == "llmdbg/residualToVocabularyLensSupport":
            from llmdbg.residual_vocabulary_lenses import available_lenses

            self.send_response(
                request,
                {"lenses": list(available_lenses(str(self.model_name)))},
            )
        elif command == "llmdbg/removeIcaComponent":
            self.send_event(
                "llmdbg/computationStarted", {"command": command}
            )
            self._console(
                f"Loading the ICA basis if needed, then replaying token {self.token_index} "
                "with the selected component removed."
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
            self._console(
                f"Replaying token {self.token_index} with the selected component removed."
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
            elif self.inside_internal and self._core_info() and command == "stepOut":
                self.mlp_step_index = self._core_info()["end"] + 1
                if self.mlp_step_index < len(self.internal_steps):
                    self.stop("step")
                else:
                    self.inside_mlp = self.inside_attention = False
                    self.mlp_step_index = 0
                    self._advance_outer()
            elif self.inside_internal and command != "stepOut" and self.mlp_step_index + 1 < len(self.internal_steps):
                core = (getattr(self,'source_calls',{}).get(function_name,{}).get(self.mlp_step_index)
                        if function_name in getattr(self,'source_plan_steps',{}) else getattr(self, "nested_cores", {}).get(function_name))
                if core and self.mlp_step_index == core["call"] and command == "next":
                    self.mlp_step_index = core["end"] + 1
                    for index in range(core["start"], core["end"] + 1):
                        if any(bp.line == self._internal_source_lines()[index] and self._breakpoint_matches(bp, self.token_index, self.layer_index) for bp in self.breakpoints):
                            self.mlp_step_index = index
                            break
                else:
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
                        and token_position >= len(self.tokens) + self.max_new_tokens
                    ):
                        maximum = len(self.tokens) + self.max_new_tokens - 1
                        range_kind = (
                            "prompt and generation"
                            if self.max_new_tokens
                            else "prompt"
                        )
                        breakpoint.update(
                            {
                                "verified": False,
                                "line": resolved_line,
                                "message": (
                                    f"Token position {token_position} is outside the configured "
                                    f"{range_kind} range 0..{maximum}"
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

    def _prompt_images(self) -> list[dict[str, Any]]:
        """Describe multimodal inputs and their contiguous visual-token spans."""
        sources: list[str] = []
        for message in self.messages:
            content = message.get("content") if isinstance(message, dict) else None
            if not isinstance(content, list):
                continue
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image":
                    source = part.get("url", part.get("path"))
                    if isinstance(source, str):
                        sources.append(source)

        spans: list[tuple[int, int]] = []
        runtime = getattr(getattr(self, "execution", None), "runtime", None)
        image_id = getattr(getattr(getattr(runtime, "model", None), "config", None), "image_token_id", None)
        image_token = runtime.tokenizer.decode([image_id]) if image_id is not None else None
        start: int | None = None
        for index, token in enumerate((*self.tokens, "")):
            visual = "image_pad" in token or (image_token is not None and token == image_token)
            if visual and start is None:
                start = index
            elif not visual and start is not None:
                spans.append((start, index - 1))
                start = None
        return [
            {
                "source": source,
                **self._image_grid_metadata(index, spans),
                "startToken": spans[index][0] if index < len(spans) else None,
                "endToken": spans[index][1] if index < len(spans) else None,
                "tokenCount": (
                    spans[index][1] - spans[index][0] + 1
                    if index < len(spans) else None
                ),
            }
            for index, source in enumerate(sources)
        ]

    def _image_grid_metadata(self, index: int, spans: list[tuple[int, int]]) -> dict[str, Any]:
        grids = getattr(getattr(self, "execution", None), "image_grids", ())
        merge = getattr(getattr(self, "execution", None), "image_merge_size", 1)
        if index >= len(grids) or index >= len(spans) or merge <= 0:
            return {}
        frames, height, width = grids[index]
        rows, columns = height // merge, width // merge
        if height % merge or width % merge or frames * rows * columns != spans[index][1] - spans[index][0] + 1:
            return {}
        return {"gridRows": rows, "gridColumns": columns, "gridFrames": frames,
                "regionMapping": "merged-row-major"}

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
        self.transformer_layer_count = len(runtime_layers)
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
            if index < getattr(runtime, "visual_layer_count", 0):
                available.append(state_name(index, "visual_residual"))
            if getattr(runtime, "has_per_layer_inputs", False):
                available.extend([state_name(index, "per_layer_residual"), state_name(index, "layer_scale")])
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
                kinds = (mixer_kind, "mlp", "visual_residual") if index < getattr(runtime, "visual_layer_count", 0) else (mixer_kind, "mlp")
                if getattr(runtime, "has_per_layer_inputs", False):
                    kinds += ("per_layer_residual", "layer_scale")
                for kind in kinds:
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
            raise DebugActionError(
                "submodules must be a valid subsequence of model execution; "
                f"available paths are {available!r}"
            )
        path = Path(str(arguments["configPath"]))
        self.source_name = path.name
        self.source_path = str(path)
        self.source_lines = tuple(int(step_lines[name]) for name in step_paths)
        self.step_paths = tuple(step_paths)
        functions = arguments.get("functions", {})
        if self.execution is not None:
            self.execution.capture_delta_core = "gated_delta" in functions
        self.function_bindings = dict(arguments.get("functionBindings", {}))
        self.source_plan_steps = {}
        self.source_calls = {}
        source_plans = arguments.get('sourcePlans',{})
        if source_plans:
            self.function_source_lines = {}
            self.configured_internal_steps = {}
            self.nested_cores = {}
            for root, plan in source_plans.items():
                steps = plan['steps']
                self.source_plan_steps[root] = steps
                self.configured_internal_steps[root] = tuple((step['operation'],step['vector'],
                    f"{step['frames'][-1]['name']}: {step['operation']}") for step in steps)
                self.function_source_lines[root] = tuple(step['line'] for step in steps)
                calls = {int(index):info for index,info in plan['calls'].items()}
                self.source_calls[root] = calls
                if calls:
                    self.nested_cores[root] = calls[min(calls)]  # legacy introspection compatibility
            if self.execution is not None:
                self.execution.capture_delta_core = any(step['operation']=='convolved_qkv'
                    for steps in self.source_plan_steps.values() for step in steps)
            self.mlp_source_lines = self.function_source_lines.get('mlp',())
            self.attention_source_lines = self.function_source_lines.get('attention',())
            return
        step_sets = {
            "attention": self.ATTENTION_STEPS,
            "full_attention": (
                self.GEMMA4_ATTENTION_STEPS
                if runtime_architecture == "gemma4" else self.GEMMA_ATTENTION_STEPS
                if runtime_architecture == "gemma2" else self.FULL_ATTENTION_STEPS
            ),
            "gated_linear_attention": self.GATED_LINEAR_ATTENTION_STEPS,
            "mlp": (
                self.GEMMA_MLP_STEPS
                if runtime_architecture in {"gemma2", "gemma4"}
                else self.QWEN_MLP_STEPS
                if runtime_architecture in {"qwen3_5", "qwen3_vl"}
                else self.MLP_STEPS
            ),
        }
        if runtime_architecture not in {"gemma2", "gemma4"} and isinstance(functions, dict):
            full_attention_lines = functions.get("full_attention", {})
            if isinstance(full_attention_lines, dict) and "query_projection" not in full_attention_lines:
                step_sets["full_attention"] = self.LEGACY_FULL_ATTENTION_STEPS
        self.function_source_lines = {}
        self.configured_internal_steps = {}
        self.nested_cores = {}
        core_steps = {
            "attention": self.FULL_ATTENTION_STEPS[1:-2],
            "gated_delta": (
                ("convolved_qkv", "delta_qkv", "Causal convolution, Q/K normalization, head expansion and query scaling."),
                ("gates", "delta_beta", "Compute decay, beta update strength, and SiLU output gate input."),
                ("cached_memory", "delta_previous_memory", "Read the previous recurrent memory, or zeros for the first token."),
                ("update_memory", "delta_updated_memory", "Decay memory and correct its prediction using the current key and value; save to cache."),
                ("read_memory", "delta_readout", "Read the updated memory with the current scaled query."),
                ("gated_output_norm", "gated_delta_mixing", "RMS-normalize each head and multiply by SiLU(gate)."),
            ),
        }
        for outer, core_name in (("full_attention", "attention"), ("gated_linear_attention", "gated_delta")):
            outer_lines = functions.get(outer, {}) if isinstance(functions, dict) else {}
            nested_lines = functions.get(core_name, {}) if isinstance(functions, dict) else {}
            if core_name not in outer_lines or not nested_lines:
                continue
            # The four-line outer function stays readable; insert the authored
            # core locations in execution order for stepping and breakpoints.
            outer_steps = (self.LEGACY_FULL_ATTENTION_STEPS if outer == "full_attention" else self.GATED_LINEAR_ATTENTION_STEPS)
            outer_steps = (outer_steps[0], (core_name, outer_steps[1][1], outer_steps[1][2]), *outer_steps[2:])
            nested_steps = core_steps[core_name]
            return_info = arguments.get('functionReturns', {}).get(core_name)
            # A named return is a real pause after the final assignment, with
            # the core frame and its newly assigned locals still in scope.
            return_binding = next((binding for binding in reversed(self.function_bindings.get(core_name, []))
                                   if return_info and binding['name'] == return_info['name']), None)
            return_line = None
            if return_binding and return_binding['operation'] == nested_steps[-1][0]:
                return_line = int(return_info['line'])
                nested_steps = (*nested_steps, ('return', nested_steps[-1][1], 'Return the completed mixer output to the caller.'))
            steps = (*outer_steps[:2], *nested_steps, *outer_steps[2:])
            lines = [outer_lines[outer_steps[0][0]], outer_lines[core_name]]
            lines.extend(nested_lines[operation] for operation, _, _ in core_steps[core_name])
            if return_line is not None:
                lines.append(return_line)
            lines.extend(outer_lines[operation] for operation, _, _ in outer_steps[2:])
            self.configured_internal_steps[outer] = steps
            self.function_source_lines[outer] = tuple(lines)
            self.nested_cores[outer] = {"name": core_name, "call": 1, "start": 2, "end": 1 + len(nested_steps)}
        if isinstance(functions, dict):
            for function_name, steps in step_sets.items():
                if function_name in self.configured_internal_steps:
                    continue
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
        elif self._append_next_generated_token():
            self._load_token(self.token_index + 1)
            self.layer_index = 0
            self.stop("step")
        else:
            self.terminate()

    def _append_next_generated_token(self) -> bool:
        """Greedily extend a chat run after its current final token."""
        if (
            self.execution is None
            or self.generated_tokens >= self.max_new_tokens
            or not self.predictions
        ):
            return False
        prediction = self.predictions[0]
        eos = getattr(self.execution.runtime.tokenizer, "eos_token_id", None)
        eos_ids = set(eos if isinstance(eos, (list, tuple, set)) else [eos])
        if prediction.token_id in eos_ids:
            self.send_event(
                "output",
                {"category": "stdout", "output": "llmdbg: generation reached EOS\n"},
            )
            return False
        text = self.execution.append_generated_token(prediction.token_id)
        self.generated_tokens += 1
        self.tokens = self.execution.tokens
        self.send_event(
            "output",
            {
                "category": "stdout",
                "output": text,
            },
        )
        return True

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
                    raise DebugActionError("nop() requires an earlier model operation")
                states_by_name[path] = LayerState(
                    Layer(
                        path, "nop", "No operation; stable breakpoint location.",
                        1.0, 0.0,
                    ),
                    previous.output_vector,
                    previous.output_vector,
                    "Preserves the current value without executing a model operation.",
                    tensors={key: previous.tensors["output"] for key in ("input", "output")
                             if "output" in previous.tensors},
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

    def _is_final_transformer_layer(self, path: str) -> bool:
        index = self._transformer_layer_index(path)
        return (
            index is not None
            and getattr(self, "transformer_layer_count", 0) > 0
            and index == self.transformer_layer_count - 1
        )

    def _current_lens_phase(self) -> int:
        """Return the explicit block phase: pre, attention, mid, MLP, or post."""
        path = self.outer_state.layer.name
        function_name = self._function_name_for_path(path)
        if function_name in {"attention", "full_attention", "gated_linear_attention"}:
            if (
                self.inside_internal
                and any(step[0]=='residual_add' for step in self.internal_steps[:self.mlp_step_index+1])
            ):
                return 2
            return 1 if self.inside_internal else 0
        if function_name == "mlp":
            if (
                self.inside_internal
                and any(step[0]=='residual_add' for step in self.internal_steps[:self.mlp_step_index+1])
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
            "attention-heads": 1,
            "attention-head-values": 1,
            "delta-memory-write": 1,
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
        self._console(
            f"Searching {len(self.tokens)} token positions and "
            f"{len(self.source_lines)} execution locations for the next breakpoint."
        )
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
                        self._console(
                            f"Matched breakpoint at token {token_position}, source line {line}; "
                            "preparing its model forward pass."
                        )
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
        if self.token_index + 1 < len(self.tokens):
            self._console(
                "No configured breakpoint matched; executing the final prompt token "
                "before completing the trace."
            )
            self._load_token(len(self.tokens) - 1)
            self.layer_index = len(self.trace) - 1
        if self._append_next_generated_token():
            self._load_token(self.token_index + 1)
            self.layer_index = 0
            self.inside_mlp = False
            self.inside_attention = False
            self.mlp_step_index = 0
            self._continue_to_breakpoint(include_current=True, before_start=False)
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
        frame = self._source_frame()
        if frame is not None:
            return list(frame['bindings'])
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
            function_name = self._active_function_name()
            completed = {
                operation for operation, _vector, _meaning in self.internal_steps[: self.mlp_step_index]
            }
            latest: dict[str, str] = {}
            if self._core_info():
                names.append("normalized")
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
            try:
                inspection = self._inspect_expression(name)
            except DebugActionError as error:
                variables.append({'name':name,'evaluateName':name,'value':str(error),'type':'expression error','variablesReference':0})
                continue
            if inspection is not None:
                variables.append({"name": name, "evaluateName": name,
                                  **self._tensor_description(inspection, name, variable=True)})
                continue
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
        raise DebugActionError(f"{name!r} cannot be changed in this scope")

    def _set_expression(self, expression: str, value: float) -> float:
        """Record an intervention and replay the token to refresh downstream values."""
        if self.execution is None:
            raise DebugActionError("vector intervention requires a loaded model")
        result_state = self.previous_state
        if result_state is None:
            raise DebugActionError("step over an operation before changing its output")

        expression = self._source_expression(expression)
        neuron_match = re.fullmatch(r"top_neurons\[(\d+)\]\.activation", expression)
        if neuron_match:
            rank = int(neuron_match.group(1))
            neurons = self._top_neurons()
            if rank >= len(neurons):
                raise DebugActionError(f"top neuron rank must be in 0..{len(neurons) - 1}")
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
                raise DebugActionError(
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
            raise DebugActionError(f"{vector_name} is not available for {result_state.layer.name}")
        if index < 0 or index >= len(source_vector):
            raise DebugActionError(
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
            raise DebugActionError("ICA intervention requires a loaded model")
        if view not in {
            "ica-residual-pre", "ica-residual-mid", "ica-residual-post"
        }:
            raise DebugActionError("ICA removal is available only for residual vectors")
        if component < 0:
            raise DebugActionError("ICA component must be non-negative")
        snapshot = self._neuron_lens_snapshot(
            view=view,
            include_logit_lens=False,
            include_j_lens=False,
            include_r_lens=False,
        )
        if snapshot.get("saved"):
            raise DebugActionError(
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
                "category": "stdout",
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

    def _patch_vector(
        self, view: str, replacement: tuple[float, ...], source_label: str
    ) -> dict[str, Any]:
        """Replace an active model-width vector and replay the current token."""
        if self.execution is None:
            raise DebugActionError("vector patch requires a loaded model")
        normalized_view = view.removeprefix("ica-")
        targets = {
            "residual-pre": "input",
            "residual-mid": "output",
            "residual-post": "output",
            "attention-output": "attention_projection",
            "mlp-output": "mlp_projection",
        }
        vector_name = targets.get(normalized_view)
        if vector_name is None:
            raise DebugActionError(f"Patch: Paste to replace is unavailable for {view!r}")
        snapshot = self._neuron_lens_snapshot(
            view=normalized_view,
            include_logit_lens=False,
            include_j_lens=False,
            include_r_lens=False,
            include_jpp_lens=False,
        )
        if snapshot.get("saved"):
            raise DebugActionError(
                "vector patch is unavailable because this block has already executed"
            )
        measurement_id = str(snapshot.get("measurementId", ""))
        if measurement_id not in self.measurement_vectors:
            raise DebugActionError("target vector is unavailable")
        _label, before = self.measurement_vectors[measurement_id]
        if len(replacement) != len(before):
            raise DebugActionError(
                f"copied vector has {len(replacement)} dimensions but target has {len(before)}"
            )
        layer = str(snapshot["layer"])
        baseline_predictions = tuple(self.predictions)
        self.execution.patch_vector(
            self.token_index, layer, vector_name, replacement
        )
        current_layer = self.layer_index
        self._load_token(self.token_index)
        self.layer_index = current_layer
        self.neuron_cache.clear()
        getattr(self, "lens_snapshot_cache", {}).clear()
        getattr(self, "measurement_vectors", {}).clear()
        log_paths = self._write_vector_patch_log(
            view=normalized_view,
            layer=layer,
            vector_name=vector_name,
            source_label=source_label,
            before=before,
            replacement=replacement,
            baseline_predictions=baseline_predictions,
        )
        self.send_event(
            "output",
            {"category": "stdout", "output": f"llmdbg: vector patch log: {log_paths['note']}\n"},
        )
        return {
            "patched": True,
            "view": normalized_view,
            "layer": layer,
            "tokenPosition": self.token_index,
            "log": log_paths,
        }

    def _write_vector_patch_log(
        self,
        *,
        view: str,
        layer: str,
        vector_name: str,
        source_label: str,
        before: tuple[float, ...],
        replacement: tuple[float, ...],
        baseline_predictions: tuple[Any, ...],
    ) -> dict[str, str]:
        """Write a reproducible record for a full-vector patch intervention."""
        def digest(values: tuple[float, ...]) -> str:
            return hashlib.sha256(
                json.dumps(values, separators=(",", ":")).encode("utf-8")
            ).hexdigest()

        def predictions(items: tuple[Any, ...]) -> list[dict[str, Any]]:
            return [
                {"tokenId": item.token_id, "text": item.text, "logit": item.logit,
                 "probability": item.probability}
                for item in items
            ]

        transformer_layer = self._transformer_layer_index(layer)
        record = {
            "schema": "llmdbg.vector-patch-intervention.v1",
            "createdAt": datetime.now(timezone.utc).isoformat(),
            "model": {"id": self.model_name, "revision": self.model_revision,
                      "device": self.execution.device if self.execution else None,
                      "dtype": self.execution.dtype if self.execution else None},
            "prompt": self.prompt,
            "tokens": list(self.tokens),
            "intervention": {"kind": "patch_vector", "tokenPosition": self.token_index,
                             "token": self.tokens[self.token_index], "view": view,
                             "state": layer, "transformerLayer": transformer_layer,
                             "vector": vector_name, "sourceLabel": source_label},
            "targetBefore": {"norm": sum(value * value for value in before) ** 0.5,
                             "sha256": digest(before), "values": before},
            "replacement": {"norm": sum(value * value for value in replacement) ** 0.5,
                            "sha256": digest(replacement), "values": replacement},
            "baselinePredictions": predictions(baseline_predictions),
            "intervenedPredictions": predictions(tuple(self.predictions)),
        }
        project = Path(self.source_path).parent if self.source_path else Path.cwd()
        log_dir = project / ".logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        stem = f"{timestamp}-vector-patch-t{self.token_index}-l{transformer_layer}"
        json_path = log_dir / f"{stem}.json"
        note_path = log_dir / f"{stem}.md"
        json_path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
        note_path.write_text("\n".join([
            "# llmdbg vector patch intervention", "",
            f"- Time: `{record['createdAt']}`",
            f"- Model: `{self.model_name}` at `{self.model_revision}`",
            f"- Prompt: `{self.prompt}`",
            f"- Target: token {self.token_index}, `{view}`, `{layer}.{vector_name}`",
            f"- Source: `{source_label}`",
            f"- Norm: {record['targetBefore']['norm']:.8f} → {record['replacement']['norm']:.8f}",
            "", f"Machine-readable record: `{json_path.name}`", "",
        ]), encoding="utf-8")
        return {"note": str(note_path), "json": str(json_path)}

    def _ica_residual_vector(
        self, view: str, layer: str
    ) -> tuple[float, ...]:
        """Resolve the effective residual at an ICA intervention boundary."""
        if self.execution is None:
            raise DebugActionError("ICA intervention requires a loaded model")
        state = next(
            (candidate for candidate in self.trace if candidate.layer.name == layer),
            None,
        )
        if state is None:
            raise DebugActionError(f"residual state {layer} is unavailable")
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
            raise DebugActionError("ICA intervention requires a loaded model")
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
            raise DebugActionError("component intervention requires a loaded model")
        if view not in {"neurons", "attention-neurons"}:
            raise DebugActionError("component removal requires a neuron or channel view")
        if component < 0:
            raise DebugActionError("component must be non-negative")
        snapshot = self._neuron_lens_snapshot(
            view=view,
            include_logit_lens=False,
            include_j_lens=False,
            include_r_lens=False,
        )
        if snapshot.get("saved"):
            raise DebugActionError(
                "component intervention is unavailable because this block has already executed"
            )
        selected = next(
            (item for item in snapshot["neurons"] if item["index"] == component),
            None,
        )
        if selected is None:
            raise DebugActionError(f"component {component} is not available in this Lens")
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
                "category": "stdout",
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
            raise DebugActionError("component intervention requires a loaded model")

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
        head_index: int | None = None,
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
            detail == vector_name and not (self._core_info() and operation == self._core_info()["name"])
            for operation, detail, _meaning in self.internal_steps[: self.mlp_step_index]
        )
        activation = result_state.details.get(vector_name)
        if not mixing_complete or activation is None:
            return ()
        layer_name = result_state.layer.name
        removed_indices = self._removed_projected_components(
            layer_name, vector_name
        )
        key = (
            self.token_index, "attention:" + layer_name, head_index,
            include_logit_lens, include_j_lens, include_r_lens, include_jpp_lens,
            removed_indices,
        )
        if key not in self.neuron_cache:
            selection = {"head_index": head_index, "channel_count": 3} if head_index is not None else {}
            self.neuron_cache[key] = self.execution.top_attention_channels(
                layer_name, activation,
                **selection,
                output=result_state.details.get("attention_projection"),
                residual=result_state.input_vector,
                include_logit_lens=include_logit_lens,
                include_j_lens=include_j_lens,
                include_r_lens=include_r_lens,
                include_jpp_lens=include_jpp_lens,
                include_indices=removed_indices,
            )
        return self.neuron_cache[key]

    @staticmethod
    def _attention_channel_head(state: Any, channel: int) -> dict[str, int]:
        weights = state.tensors.get("attention_weights")
        values = state.tensors.get("attention_cached_values")
        if weights is None or values is None or weights.ndim != 4 or values.ndim != 4:
            return {}
        query_heads, kv_heads, width = int(weights.shape[1]), int(values.shape[1]), int(values.shape[3])
        if kv_heads <= 0 or width <= 0 or query_heads % kv_heads or not 0 <= channel < query_heads * width:
            return {}
        head, dimension = divmod(channel, width)
        return {"queryHead": head, "kvHead": head // (query_heads // kv_heads),
                "dimension": dimension, "headDimension": width}

    def _delta_memory_contributions(self, layer: str, head: int, token_position: int) -> dict[str, Any]:
        if token_position != self.token_index:
            raise DebugActionError('This head belongs to another token. Refresh the Lens panel and try again.')
        state = next((s for s in self.trace if s.layer.name == layer), None)
        if state is None or self.execution is None or not layer.endswith('.gated_linear_attention'):
            raise DebugActionError('Memory contributions require a captured DeltaNet head')
        try:
            data = self.execution.delta_memory_contributions(self._transformer_layer_index(layer), head, token_position)
        except ValueError as error:
            raise DebugActionError(str(error)) from error
        # Do not present a replay of a different numerical/intervention history
        # as an attribution of the paused snapshot.
        import torch
        mixed = state.tensors.get('gated_delta_mixing')
        if mixed is not None:
            width = data['headDimension']
            weight = self._attention_output_weight(self._transformer_layer_index(layer))[head*width:(head+1)*width].float()
            actual = mixed.flatten()[head*width:(head+1)*width].float() @ weight
            reproduced = torch.tensor(data['outputVector'], device=actual.device)
            if float((actual-reproduced).norm()) > .03*float(actual.norm())+.001:
                raise DebugActionError('Replay differs from this paused head snapshot. Restart at this token with a fresh prefix cache before requesting memory contributions.')
        return {**{k:v for k,v in data.items() if k != 'outputVector'}, 'layer':layer, 'transformerLayer':self._transformer_layer_index(layer),
                'tokenPosition':token_position, 'token':self.tokens[token_position], 'images':self._prompt_images()}

    def _attention_token_contributions(
        self, layer: str, channel: int, token_position: int
    ) -> dict[str, Any]:
        """Decompose one full-attention input channel by cached source token."""
        if token_position != self.token_index:
            raise DebugActionError("This channel belongs to another token. Refresh the Lens panel and try again.")
        state = next((state for state in self.trace if state.layer.name == layer), None)
        if state is None or not layer.endswith(".full_attention"):
            raise DebugActionError("Token contributions require a captured full-attention layer.")
        weights = state.tensors.get("attention_weights")
        values = state.tensors.get("attention_cached_values")
        if weights is None or values is None:
            raise DebugActionError("Attention weights and cached values are unavailable for this layer.")
        if weights.ndim != 4 or values.ndim != 4:
            raise DebugActionError("Unsupported attention tensor layout for token contributions.")
        query_heads, kv_heads, width = int(weights.shape[1]), int(values.shape[1]), int(values.shape[3])
        if kv_heads <= 0 or query_heads % kv_heads:
            raise DebugActionError("Unsupported query/KV head grouping for token contributions.")
        if not 0 <= channel < query_heads * width:
            raise DebugActionError(f"Channel must be in 0..{query_heads * width - 1}.")
        count = int(values.shape[2])
        source_positions = state.tensors.get("attention_source_positions")
        if source_positions is None:
            positions = list(range(count))
        else:
            positions = [int(position) for position in source_positions.tolist()]
        if int(weights.shape[3]) != count or len(positions) != count or (source_positions is None and count != token_position + 1):
            raise DebugActionError("Cached source-token positions are unavailable for this attention layout.")
        head, dimension = divmod(channel, width)
        kv_head = head // (query_heads // kv_heads)
        selected_weights = weights[0, head, -1, :].detach().float().cpu().tolist()
        selected_values = values[0, kv_head, :, dimension].detach().float().cpu().tolist()
        rows = [
            {"position": index, "token": self.tokens[index], "weight": weight,
             "value": value, "contribution": weight * value}
            for index, weight, value in zip(positions, selected_weights, selected_values)
        ]
        mixed = state.tensors.get("full_attention_mixing")
        mixed_value = float(mixed[0, -1, channel].item()) if mixed is not None else None
        has_output_gate = getattr(getattr(self.execution, "runtime", None), "architecture", "qwen3_5") == "qwen3_5"
        return {
            "layer": layer, "transformerLayer": self._transformer_layer_index(layer),
            "tokenPosition": token_position, "token": self.tokens[token_position],
            "channel": channel, "queryHead": head, "kvHead": kv_head,
            "dimension": dimension, "headDimension": width,
            "weightsShape": list(weights.shape), "valuesShape": list(values.shape),
            "rows": rows, "sum": sum(row["contribution"] for row in rows),
            "positiveSum": sum(max(0., row["contribution"]) for row in rows),
            "negativeSum": sum(min(0., row["contribution"]) for row in rows),
            "mixedValue": mixed_value,
            "images": self._prompt_images(),
            "stage": "before-output-gate" if has_output_gate else "before-output-projection",
            "outputGate": has_output_gate,
        }

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
                raise DebugActionError(
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
        frame = self._source_frame()
        if frame is not None:
            aliases = {name:value if value.isidentifier() else '('+value+')' for name,value in frame['bindings'].items()}
            return self._replace_expression_name(expression, aliases)
        aliases = {"x": "x", "residual_stream": "x"}
        if not getattr(self, "inside_internal", False):
            return self._replace_expression_name(expression, aliases)
        function_name = self._active_function_name()
        bindings = getattr(self, "function_bindings", {}).get(function_name, [])
        operation_vectors = {
            operation: vector for operation, vector, _meaning in self.internal_steps
        }
        completed = {
            operation for operation, _vector, _meaning in self.internal_steps[: self.mlp_step_index]
        }
        for binding in bindings:
            if binding.get("operation") in completed:
                aliases[str(binding["name"])] = binding.get("vector", operation_vectors[str(binding["operation"])])
        if self._core_info():
            aliases["normalized"] = "normalized_input"
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
            if expression == alias or expression.startswith(alias + "[") or expression.startswith(alias + "."):
                return vector + expression[len(alias):]
        return expression

    def _evaluate_body(self, expression: str) -> dict[str, Any] | None:
        """Build a DAP evaluate result, including expandable synthetic values."""
        inspection = self._inspect_expression(expression)
        if inspection is not None:
            return self._tensor_description(inspection, expression)
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

    def _attention_output_weight(self, layer: int | None = None) -> Any:
        """Current layer's bias-free output map, oriented input × residual width."""
        if self.execution is None:
            raise DebugActionError("W_o requires a paused model debug session.")
        if layer is None:
            layer = self._transformer_layer_index(self.outer_state.layer.name)
        runtime = self.execution.runtime
        if layer is None or not 0 <= layer < len(runtime.layers):
            raise DebugActionError("W_o is available inside a transformer layer.")
        block = runtime.layers[layer]
        if runtime.architecture == "gpt2":
            weight = block.attn.c_proj.weight
        else:
            mixer = block.self_attn if runtime.layer_kind(layer) == "full_attention" else block.linear_attn
            projection = mixer.o_proj if hasattr(mixer, "o_proj") else mixer.out_proj
            weight = projection.weight.T
        # Never expose a live parameter to inspection clients.
        return weight.detach().clone()

    def _inspect_expression(self, expression: str) -> Any:
        """Evaluate a restricted tensor expression against captured snapshots."""
        try:
            tree = ast.parse(expression, mode="eval").body
        except SyntaxError:
            return None

        def resolve(node, captured=False):
            if isinstance(node, ast.Constant) and type(node.value) in {int, float}:
                return node.value
            if isinstance(node, (ast.Tuple,ast.List)):
                return tuple(resolve(item,captured) for item in node.elts)
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
                value = resolve(node.operand,captured)
                return value if isinstance(node.op, ast.UAdd) else -value
            if isinstance(node, ast.BinOp):
                operation = {ast.Add: operator.add, ast.Sub: operator.sub,
                             ast.Mult: operator.mul, ast.Div: operator.truediv,
                             ast.MatMult: operator.matmul}.get(type(node.op))
                if operation is None:
                    raise DebugActionError("Supported tensor operators: +, -, *, /, @, and parentheses")
                return operation(resolve(node.left,captured), resolve(node.right,captured))
            if isinstance(node, ast.Name):
                if node.id == "W_o":
                    return self._attention_output_weight()
                frame = self._source_frame() if not captured else None
                if frame is not None:
                    if node.id in frame['bindings']:
                        return resolve(ast.parse(frame['bindings'][node.id],mode='eval').body,True)
                    if node.id in frame['localNames']:
                        raise LookupError
                name = node.id if captured else self._source_expression(node.id)
                if name in {"x", "input", "residual", "residual_stream"}:
                    name = "input"
                state = self.outer_state if self.inside_internal or name == "input" else self.previous_state
                if state is None:
                    raise LookupError
                tensor = state.tensors.get(name)
                if tensor is None:
                    raise LookupError
                return tensor
            if isinstance(node, ast.Attribute) and node.attr in {"shape", "dtype", "device", "ndim", "requires_grad"}:
                return getattr(resolve(node.value,captured), node.attr)
            if isinstance(node, ast.Subscript):
                return resolve(node.value,captured)[index(node.slice,captured)]
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr in {"size", "numel", "dim"} and not node.keywords:
                    args = [resolve(arg,captured) for arg in node.args]
                    if any(not isinstance(arg, int) for arg in args):
                        raise DebugActionError("tensor method arguments must be integers")
                    return getattr(resolve(node.func.value,captured), node.func.attr)(*args)
            raise LookupError

        def index(node,captured=False):
            if isinstance(node, ast.Slice):
                return slice(*(index(part,captured) if part is not None else None
                               for part in (node.lower, node.upper, node.step)))
            if isinstance(node, ast.Tuple):
                return tuple(index(part,captured) for part in node.elts)
            if isinstance(node, (ast.BinOp, ast.UnaryOp,ast.Name)):
                value = resolve(node,captured)
                if type(value) is not int:
                    raise DebugActionError("Slice arithmetic must produce an integer.")
                return value
            return ast.literal_eval(node)

        try:
            return resolve(tree)
        except IndexError as error:
            raise DebugActionError(str(error)) from error
        except (LookupError, AttributeError):
            return None
        except (ValueError, TypeError, RuntimeError, ZeroDivisionError, OverflowError) as error:
            raise DebugActionError(str(error)) from error

    def _tensor_description(self, value: Any, expression: str, *, variable: bool = False) -> dict[str, Any]:
        is_tensor = hasattr(value, "detach") and hasattr(value, "shape")
        is_sequence = isinstance(value, (tuple, list))
        expandable = is_tensor or is_sequence
        reference = 0
        if expandable:
            references = getattr(self, "inspection_references", None)
            if references is None:
                references = self.inspection_references = {}
            reference = self.TENSOR_REFERENCE + len(references)
            references[reference] = (value, expression)
        if is_tensor:
            preview = (
                repr(value.item()) if value.ndim == 0 else
                f"tensor(shape={tuple(value.shape)}, dtype={value.dtype}, device='{value.device}')"
            )
            type_name = "torch.Tensor"
        else:
            preview, type_name = str(value), type(value).__name__
        result = {"value" if variable else "result": preview,
                  "type": type_name, "variablesReference": reference}
        if expandable:
            result["indexedVariables"] = int(value.shape[0]) if is_tensor and value.ndim else (len(value) if is_sequence else 0)
            if is_tensor:
                result["namedVariables"] = 6
        return result

    def _tensor_variables(self, reference: int, arguments: dict[str, Any]) -> list[dict[str, Any]]:
        entry = getattr(self, "inspection_references", {}).get(reference)
        if entry is None:
            return []
        value, expression = entry
        is_tensor = hasattr(value, "detach")
        variables = []
        if is_tensor and arguments.get("filter") != "indexed":
            for name, item in (("shape", value.shape), ("dtype", value.dtype),
                               ("device", value.device), ("ndim", value.ndim),
                               ("requires_grad", value.requires_grad), ("numel", value.numel())):
                child = f"{expression}.{name}" + ("()" if name == "numel" else "")
                variables.append({"name": name, "evaluateName": child,
                                  **self._tensor_description(item, child, variable=True)})
        if arguments.get("filter") != "named":
            length = int(value.shape[0]) if is_tensor and value.ndim else (len(value) if not is_tensor else 0)
            start = max(0, int(arguments.get("start", 0)))
            count = min(256, max(0, int(arguments.get("count", 100))))
            for position in range(start, min(length, start + count)):
                child = f"{expression}[{position}]"
                variables.append({"name": f"[{position}]", "evaluateName": child,
                                  **self._tensor_description(value[position], child, variable=True)})
        return variables

    def _vector(self, name: str) -> tuple[float, ...]:
        if name == "logits":
            for state in self.trace:
                if state.layer.name == "vocabulary_projection":
                    return state.output_vector
            raise DebugActionError("logits are not available before vocabulary projection")
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
        raise DebugActionError(f"unknown vector: {name}")

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
                raise DebugActionError("pasted vector must be a list or tuple")
            return tuple(float(item) for item in value)
        match = re.fullmatch(
            r"([a-z_][a-z0-9_]*)(?:\[(\d*)?:(\d*)?(?::(\d+))?\])?",
            expression,
        )
        if match is None:
            raise DebugActionError("vector must be a vector name, slice, list, or tuple")
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

    def _decode_expressions(self, expression: str, loop: Any = None) -> dict[str, Any]:
        if loop is None:
            return self._decode_expression(expression)
        if not isinstance(loop, dict):
            raise DebugActionError("Loop settings must contain a variable, start, and stop.")
        variable = loop.get("variable", "i")
        if not isinstance(variable, str) or not variable.isidentifier() or keyword.iskeyword(variable):
            raise DebugActionError("Loop variable must be a valid Python identifier.")
        start, stop = loop.get("start", 0), loop.get("stop")
        if type(start) is not int or type(stop) is not int or not 1 <= stop - start <= 64:
            raise DebugActionError("Loop needs integer bounds and 1–64 iterations; stop is exclusive.")
        try:
            tree = ast.parse(expression, mode="eval")
        except SyntaxError as error:
            raise DebugActionError("Invalid tensor expression.") from error
        if not any(isinstance(node, ast.Name) and node.id == variable for node in ast.walk(tree)):
            raise DebugActionError(f"Use the loop variable {variable} in the tensor expression.")
        results = []
        for value in range(start, stop):
            class BindVariable(ast.NodeTransformer):
                def visit_Name(self, node):
                    return ast.copy_location(ast.Constant(value=value), node) if node.id == variable else node
            bound = ast.unparse(BindVariable().visit(ast.parse(expression, mode="eval")))
            try:
                result = self._decode_expression(bound)
            except DebugActionError as error:
                result = {"expression": bound, "error": str(error)}
            results.append({**result, "loopValue": value})
        return {"expression": expression, "loop": {"variable": variable, "start": start, "stop": stop}, "results": results}

    def _decode_expression(self, expression: str) -> dict[str, Any]:
        """Read a model-width expression through every configured vocabulary lens."""
        import torch
        from llmdbg.residual_vocabulary_lenses import activate, configured_lenses
        from llmdbg.workspace_lens import resolve_view_site
        if self.execution is None:
            raise DebugActionError("Start and pause a model debug session first.")
        tensor = self._inspect_expression(expression)
        if tensor is None or not torch.is_tensor(tensor):
            raise DebugActionError("Enter a tensor expression, for example normalized - x.")
        vector = tensor.detach().squeeze()
        width = int(self.execution.runtime.model.get_output_embeddings().weight.shape[1])
        if vector.ndim != 1 or vector.numel() != width:
            raise DebugActionError(f"Decoder needs one model-width vector ({width} values); got shape {tuple(tensor.shape)}. Select a slice or project it into residual space; tensors are not flattened automatically.")
        if not bool(torch.isfinite(vector).all()):
            raise DebugActionError("Vector contains NaN or infinity.")
        current_layer = self._transformer_layer_index(self.outer_state.layer.name)
        layer = current_layer
        layer = (0 if self.outer_state.layer.name == "token_embedding" else len(self.execution.runtime.layers) - 1) if layer is None else layer
        if not 0 <= layer < len(self.execution.runtime.layers):
            raise DebugActionError("Layer is outside the model's layer range.")
        phase = self._current_lens_phase()
        site = "embed" if self.outer_state.layer.name == "token_embedding" else "pre" if phase <= 1 else "mid" if phase <= 3 else "post"
        source_site, source_layer = ("embed", None) if site == "embed" else resolve_view_site("residual-" + site, layer)
        values = tuple(vector.float().cpu().tolist())
        entries = [{"id": "logit", "label": "Logit Lens", "method": "logit"}]
        entries.extend(item for item in configured_lenses() if item.get("enabled", True))
        rows = []
        for item in entries:
            method = item["method"]
            row = {"id": item["id"], "label": item["label"], "method": method}
            self._console(f"Decoding expression with {item['label']}; uncached lens artifacts may be downloaded and loaded.")
            try:
                if item.get("models") and self.model_name not in item["models"]:
                    raise LensUnavailable("This lens is not configured for the current model.")
                with activate(None if method == "logit" else item["id"]):
                    projection = self.execution.vector_lens(
                        self.outer_state.layer.name, values, source_site=source_site, source_layer=source_layer,
                        include_logit_lens=method == "logit", include_j_lens=method == "j",
                        include_r_lens=method == "r", include_jpp_lens=method == "jpp",
                    )
                tokens = getattr(projection, {"logit": "vocabulary", "j": "j_lens", "r": "r_lens", "jpp": "jpp_lens"}[method])
                if not tokens:
                    raise LensUnavailable(f"No fitted map for {self.model_name} at the current context (layer {layer}).")
                metadata = getattr(projection, method + "_lens_metadata", None) if method != "logit" else None
                row.update(tokens=[{"id": t.token_id, "text": t.text, "score": t.score} for t in tokens], metadata=metadata)
            except (LensUnavailable, FileNotFoundError):
                continue  # A lens without a fitted map has no row in this panel.
            except Exception as error:
                row["error"] = " ".join(str(error).splitlines())
            rows.append(row)
        lines = self._internal_source_lines() if self.inside_internal else self.source_lines
        line_index = self.mlp_step_index if self.inside_internal else self.layer_index
        return {"expression": expression, "shape": list(tensor.shape), "norm": float(vector.float().norm()),
                "tokenPosition": self.token_index, "token": self.tokens[self.token_index], "layer": layer,
                "site": site, "sourceSite": source_site, "sourceLayer": source_layer,
                "sourceLine": lines[line_index] if line_index < len(lines) else None,
                "rows": rows}

    def _render_plot(self, expression: str, code: str, *, shape_action: str | None = None) -> dict[str, Any]:
        """Run an explicitly user-selected Matplotlib snippet and return a PNG."""
        if not code.strip():
            raise DebugActionError("plot code is empty")
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        tensor = self._inspect_expression(expression)
        if tensor is not None and hasattr(tensor, "detach"):
            original_shape = list(tensor.shape)
            vector = tensor.detach().reshape(-1).float().cpu().numpy()
        else:
            vector = self._plot_vector(expression)
            original_shape = [len(vector)]
        plt.close("all")
        namespace: dict[str, Any] = {
            "plt": plt,
            "np": np,
            "vector": np.asarray(vector, dtype=float),
            "vector_name": expression,
            "original_shape": tuple(original_shape),
            "PlotInputError": DebugActionError,
            "PlotShapeError": PlotShapeError,
            "heatmap_axis": {"prepend": 0, "append": -1}.get(shape_action),
        }
        exec(compile(code, "<llmdbg plot snippet>", "exec"), namespace, namespace)
        plot = namespace.get("plot")
        if callable(plot):
            try:
                figure = plot(namespace["vector"], expression, plt, np) or plt.gcf()
            except PlotShapeError as error:
                plt.close("all")
                self._console(f"Warning: {error}", warning=True)
                return {
                    "error": str(error),
                    "shapeActions": [
                        {"action": "prepend", "label": f"Add dimension at front → [1, {len(vector)}]"},
                        {"action": "append", "label": f"Add dimension at end → [{len(vector)}, 1]"},
                    ],
                }
        else:
            # Keep old snippets usable when sent by an older extension.
            figure = namespace.get("fig") or plt.gcf()
        output = io.BytesIO()
        figure.savefig(output, format="png", dpi=120, bbox_inches="tight")
        plot_shape = getattr(figure, "_llmdbg_plot_shape", None)
        shape_description = (
            f"shape {original_shape} → heatmap {list(plot_shape)} · {len(vector)} values"
            if plot_shape is not None else
            f"shape {original_shape} · {len(vector)} values (flattened)"
        )
        plt.close("all")
        return {
            "image": base64.b64encode(output.getvalue()).decode("ascii"),
            "meta": (
                f"{expression} · {shape_description} · token {self.token_index} "
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
            raise DebugActionError("one or both selected vectors are no longer available")
        label_a, a = self.measurement_vectors[vector_a]
        label_b, b = self.measurement_vectors[vector_b]
        if len(a) != len(b):
            raise DebugActionError("selected vectors have different dimensions")
        norm_a = sum(value * value for value in a) ** 0.5
        norm_b = sum(value * value for value in b) ** 0.5
        if norm_a == 0.0 or norm_b == 0.0:
            raise DebugActionError("cosine is undefined for a zero-length vector")
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

    @staticmethod
    def _delta_memory_write(state: Any) -> Any:
        """Beta-scaled value correction before the key-addressed outer product."""
        required = ('delta_previous_memory', 'delta_key', 'delta_value', 'delta_decay', 'delta_beta')
        if any(name not in state.tensors for name in required):
            raise LensUnavailable("DeltaNet Memory Write Lens requires expanded DeltaNet memory captures; restart the debug run with convolved_qkv and update_memory")
        memory, key, value, decay, beta = (state.tensors[name].detach().float() for name in required)
        key, value = key[:, 0], value[:, 0]
        decayed = memory * decay[:, 0, :, None, None]
        predicted = (decayed * key.unsqueeze(-1)).sum(-2)
        return (beta[:, 0, :, None] * (value - predicted)).flatten()

    def _attention_head_lens_snapshot(self, value_mode: bool = False, write_mode: bool = False, **lens_flags: bool) -> dict[str, Any]:
        """Decode each mixed head after its bias-free output projection."""
        import torch
        names = {"attention", "full_attention", "gated_linear_attention"}
        inside = self.inside_internal and self._function_name_for_path(self.outer_state.layer.name) in names
        state = self.outer_state if inside else self._prior_operation_state(*names)
        if self.execution is None or state is None:
            raise LensUnavailable("Attention head outputs are available after attention mixing completes")
        function = self._function_name_for_path(state.layer.name)
        delta = function == "gated_linear_attention"
        if write_mode and not delta:
            raise LensUnavailable("DeltaNet Memory Write Lens requires a DeltaNet layer")
        raw_mode = value_mode or write_mode
        detail = "gated_delta_mixing" if delta else "attention_mixing" if function == "attention" else "full_attention_mixing"
        if value_mode:
            detail = "delta_value" if delta else "attention_qkv" if function == "attention" else "attention_value"
        if write_mode:
            detail = "delta_updated_memory"
        complete = not inside or any(
            captured == ("delta_qkv" if value_mode and delta else detail) and not (self._core_info() and operation == self._core_info()["name"])
            for operation, captured, _meaning in self.internal_steps[:self.mlp_step_index]
        )
        if not complete or detail not in state.details:
            if write_mode:
                raise LensUnavailable("DeltaNet Memory Write Lens is available after update_memory completes and requires expanded DeltaNet captures")
            raise LensUnavailable("Head values are available after value projection (DeltaNet requires expanded convolved_qkv capture)" if value_mode
                                  else "Attention head outputs are available after attention mixing completes")
        layer = self._transformer_layer_index(state.layer.name)
        runtime = self.execution.runtime
        weights = self._attention_output_weight(layer)
        config = runtime.model.config
        config = getattr(config, "text_config", config)
        heads = int(config.linear_num_value_heads if delta else config.n_head if runtime.architecture == "gpt2" else config.num_attention_heads)
        mixed = (self._delta_memory_write(state).to(weights.device) if write_mode else
                 torch.tensor(state.details[detail], device=weights.device, dtype=weights.dtype).flatten())
        value_heads = heads
        if value_mode:
            if function == "attention":
                mixed = mixed.chunk(3)[2]
            elif not delta:
                value_heads = int(config.num_key_value_heads)
                if heads % value_heads:
                    raise LensUnavailable("Unsupported grouped-query value head layout")
                mixed = mixed.reshape(value_heads, -1).repeat_interleave(heads // value_heads, dim=0).flatten()
        if heads <= 0 or mixed.numel() != weights.shape[0] or mixed.numel() % heads:
            raise LensUnavailable("Unsupported attention head layout")
        width = mixed.numel() // heads
        results = []
        def tokens(items):
            return [{"rank": rank, "id": item.token_id, "text": item.text, "logit": item.score}
                    for rank, item in enumerate(items[:5], 1)]
        projected_heads = []
        for head in range(heads):
            start, stop = head * width, (head + 1) * width
            vector = mixed[start:stop].float() @ weights[start:stop].float()
            norm = float(vector.norm())
            reference = None if write_mode else attention_head_norm_scale(self.model_name, state.layer.name, head, **({'value': True} if value_mode else {}))
            scale = float(reference["scale"]) if reference else None
            projected_heads.append((head, norm, tuple(vector.cpu().tolist()), scale))
        calibrated = all(item[3] is not None for item in projected_heads)
        projected_heads.sort(key=lambda item: (-(item[1] / item[3] if calibrated else item[1]), item[0]))
        for head, norm, vector, scale in projected_heads[:5]:
            projection = self.execution.vector_lens(
                state.layer.name, vector, source_site="mid", source_layer=layer, token_count=5, **lens_flags,
            )
            result = {"index": head, "headDimension": width, "vectorNorm": norm,
                      "normScale": scale, "relativeNorm": norm / scale if scale else None,
                      "calibrationError": None if scale or write_mode else "Missing calibrated value-head reference (× unavailable)" if value_mode else "Missing calibrated head reference (× unavailable)",
                      "measurementId": self._register_measurement_vector(f"{'Memory write' if write_mode else 'Head value' if value_mode else 'Attention head'} {head} [layer {layer}]", tuple(vector))}
            if raw_mode:
                result['valueHead'] = head // (heads // value_heads)
            if delta and not raw_mode:
                result['memoryTokenContributions'] = {'layer':state.layer.name, 'head':head, 'tokenPosition':self.token_index}
            for method, attribute, key in (("logit", "vocabulary", "logit"), ("j", "j_lens", "jLens"), ("r", "r_lens", "rLens"), ("jpp", "jpp_lens", "jppLens")):
                result[key + "Tokens"] = tokens(getattr(projection, attribute, ()))
                metadata = getattr(projection, method + "_lens_metadata", None)
                result[key + "Fallback"] = bool(metadata and metadata.get("fallback"))
            results.append(result)
            result['channels'] = [] if raw_mode else self._neuron_lens_snapshot(
                view='attention-neurons', attention_head=head, **lens_flags,
            )['neurons'][:3]
        view = "delta-memory-write" if write_mode else "attention-head-values" if value_mode else "attention-heads"
        return {"view": view, "label": "DeltaNet Memory Write Lens" if write_mode else "Attention Head Value Lens" if value_mode else "Attention Head Lens",
                "tokenPosition": self.token_index, "token": self.tokens[self.token_index],
                "layer": state.layer.name, "transformerLayer": layer,
                "isFinalLayer": self._is_final_transformer_layer(state.layer.name),
                "saved": self._lens_result_is_saved(view, state.layer.name),
                "ranking": "raw-write-norm" if write_mode else "calibrated-value-norm" if value_mode and calibrated else "calibrated-norm" if calibrated else "raw-norm-missing-calibration",
                "heads": results}

    def _neuron_lens_snapshot(
        self,
        view: str = "neurons",
        include_logit_lens: bool = True,
        include_j_lens: bool = False,
        include_r_lens: bool = False,
        include_jpp_lens: bool = False,
        attention_head: int | None = None,
    ) -> dict[str, Any]:
        """Describe top MLP neurons through direct and residual-transport lenses."""
        if not self.trace:
            raise LensUnavailable(
                "Lens data is not available until execution reaches its first breakpoint"
            )
        if view in {"attention-heads", "attention-head-values", "delta-memory-write"}:
            return self._attention_head_lens_snapshot(
                value_mode=view == "attention-head-values",
                write_mode=view == "delta-memory-write",
                include_logit_lens=include_logit_lens, include_j_lens=include_j_lens,
                include_r_lens=include_r_lens, include_jpp_lens=include_jpp_lens,
            )
        jpp_lens_label = "J++ Lens"
        if include_jpp_lens:
            from llmdbg.jpp_lens import configured_display_name

            # Naming a configured v2 Lens must not load it through the legacy
            # single-file J++ artifact path. The actual site-aware artifact is
            # resolved below by ModelExecution._workspace_lens_source.
            jpp_lens_label = configured_display_name(str(self.model_name))
        requested_view = view
        is_ica = view.startswith("ica-")
        if is_ica:
            view = view.removeprefix("ica-")
            if view not in {
                "residual-pre", "attention-output", "residual-mid",
                "mlp-output", "residual-post",
            }:
                raise DebugActionError(f"unknown ICA lens view: {requested_view}")
        if view not in {
            "neurons", "attention-neurons", "attention-output",
            "mlp-output", "residual-pre", "residual-mid", "residual-post",
            "residual-input", "residual-output", "residual-stream",
        }:
            raise DebugActionError(f"unknown lens view: {view}")
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
                raise DebugActionError("a loaded model is required for vector lenses")
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
                    raise LensUnavailable(
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
                    if not any(step[0]=='residual_add' for step in self.internal_steps[:self.mlp_step_index+1]):
                        raise LensUnavailable(
                            "Residual Lens · Mid is available after attention residual addition"
                        )
                    result_state = self.outer_state
                else:
                    result_state = prior_operation(
                        "attention", "full_attention", "gated_linear_attention"
                    )
                    if result_state is None:
                        raise LensUnavailable(
                            "Residual Lens · Mid is available after attention completes"
                        )
                layer = result_state.layer.name
                vector = result_state.output_vector
                label = "Residual Lens · Mid"
                residual_geometry_data = residual_geometry(
                    result_state, "attention update"
                )
            elif view in {"residual-post", "residual-output"}:
                is_gemma4 = getattr(getattr(self.execution, "runtime", None), "architecture", "") == "gemma4"
                if is_gemma4:
                    result_state = prior_operation("layer_scale")
                    if result_state is None:
                        raise LensUnavailable("Residual Lens · Post is available after per-layer embedding and layer scaling complete")
                elif self.inside_internal:
                    if (
                        self._function_name_for_path(self.outer_state.layer.name) != "mlp"
                        or not any(step[0]=='residual_add' for step in self.internal_steps[:self.mlp_step_index+1])
                    ):
                        raise LensUnavailable(
                            "Residual Lens · Post is available after MLP residual addition"
                        )
                    result_state = self.outer_state
                else:
                    result_state = prior_operation("mlp")
                if (
                    result_state is None
                    or self._function_name_for_path(result_state.layer.name) != ("layer_scale" if is_gemma4 else "mlp")
                ):
                    raise LensUnavailable(
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
                    raise LensUnavailable(
                        "Attention Output Lens is available after attention output projection completes"
                    )
                projection_complete = not inside_attention or any(
                    detail == "attention_projection"
                    for _operation, detail, _meaning in self.internal_steps[: self.mlp_step_index]
                )
                vector = attention_state.details.get("attention_projection")
                if not projection_complete or vector is None:
                    raise LensUnavailable(
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
                    raise LensUnavailable("MLP Output Lens is available after an MLP completes")
                projection_complete = not inside_mlp or any(
                    vector == "mlp_projection"
                    for _operation, vector, _meaning in self.internal_steps[: self.mlp_step_index]
                )
                vector = result_state.details.get("mlp_projection")
                label = "MLP output"
                if not projection_complete or vector is None:
                    raise LensUnavailable("MLP Output Lens is available after down projection completes")
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

            from llmdbg.workspace_lens import resolve_view_site

            transformer_layer = self._transformer_layer_index(layer)
            if transformer_layer is None:
                raise ValueError(f"cannot resolve transformer layer for {layer!r}")
            source_site, source_layer = resolve_view_site(view, transformer_layer)

            def missing_message(method: str, values: Any) -> str | None:
                if values:
                    return None
                location = (
                    "embedding output"
                    if source_site == "embed"
                    else f"{source_site.title()} layer {source_layer}"
                )
                return f"No {method} {location} map is available."

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
                removed_vectors = {}
                for removed_index in removed_indices:
                    observation = self.execution.ica_observations.get(
                        (
                            self.token_index,
                            layer,
                            intervention_vector,
                            removed_index,
                        )
                    )
                    if observation is not None:
                        before, after = observation
                        removed_vectors[removed_index] = tuple(
                            float(left) - float(right)
                            for left, right in zip(before, after, strict=True)
                        )
                components = self.execution.ica_components(
                    layer,
                    vector,
                    is_delta=view in {"attention-output", "mlp-output"},
                    source_site=source_site,
                    source_layer=source_layer,
                    include_logit_lens=include_logit_lens,
                    include_j_lens=include_j_lens,
                    include_r_lens=include_r_lens,
                    include_jpp_lens=include_jpp_lens,
                    include_indices=removed_indices,
                    vector_overrides=removed_vectors,
                )
                if not components:
                    raise DebugActionError(
                        f"ICA Lens is not available for {self.model_name}"
                    )
                return {
                    "view": requested_view,
                    "tokenPosition": self.token_index,
                    "token": self.tokens[self.token_index],
                    "layer": layer,
                    "transformerLayer": self._transformer_layer_index(layer),
                    "isFinalLayer": self._is_final_transformer_layer(layer),
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
                            "jLensFallback": bool(
                                getattr(component.projection, "j_lens_metadata", None)
                                and component.projection.j_lens_metadata.get("fallback")
                            ),
                            "rLensFallback": bool(
                                getattr(component.projection, "r_lens_metadata", None)
                                and component.projection.r_lens_metadata.get("fallback")
                            ),
                            "jppLensFallback": bool(
                                getattr(component.projection, "jpp_lens_metadata", None)
                                and component.projection.jpp_lens_metadata.get("fallback")
                            ),
                            "rLensTokens": tokens(component.projection.r_lens),
                            "jppLensTokens": tokens(
                                getattr(component.projection, "jpp_lens", ())
                            ),
                            "jLensUnavailable": missing_message(
                                "J-Lens", component.projection.j_lens
                            ) if include_j_lens else None,
                            "rLensUnavailable": missing_message(
                                "R-Lens", component.projection.r_lens
                            ) if include_r_lens else None,
                            "jppLensUnavailable": missing_message(
                                "J++ Lens", component.projection.jpp_lens
                            ) if include_jpp_lens else None,
                        }
                        for component in components
                    ],
                }

            projection = self.execution.vector_lens(
                layer,
                vector,
                source_site=source_site,
                source_layer=source_layer,
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
                "isFinalLayer": self._is_final_transformer_layer(layer),
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
                "jLensFallback": bool(
                    getattr(projection, "j_lens_metadata", None)
                    and projection.j_lens_metadata.get("fallback")
                ),
                "rLensFallback": bool(
                    getattr(projection, "r_lens_metadata", None)
                    and projection.r_lens_metadata.get("fallback")
                ),
                "jppLensFallback": bool(
                    getattr(projection, "jpp_lens_metadata", None)
                    and projection.jpp_lens_metadata.get("fallback")
                ),
                "rLensTokens": tokens(projection.r_lens),
                "jppLensTokens": tokens(getattr(projection, "jpp_lens", ())),
                "jLensUnavailable": missing_message(
                    "J-Lens", projection.j_lens
                ) if include_j_lens else None,
                "rLensUnavailable": missing_message(
                    "R-Lens", projection.r_lens
                ) if include_r_lens else None,
                "jppLensUnavailable": missing_message(
                    "J++ Lens", projection.jpp_lens
                ) if include_jpp_lens else None,
            }
        if view == "attention-neurons":
            neurons = self._top_attention_channels(
                head_index=attention_head,
                include_logit_lens=include_logit_lens,
                include_j_lens=include_j_lens, include_r_lens=include_r_lens,
                include_jpp_lens=include_jpp_lens,
            )
            if not neurons:
                raise LensUnavailable(
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
            raise LensUnavailable(
                "MLP Neuron Lens is available after an MLP activation has completed"
            )
        scale = float(scale_record["scale"]) if scale_record is not None else None
        from llmdbg.workspace_lens import resolve_view_site

        component_layer = self._transformer_layer_index(layer)
        if component_layer is None:
            raise ValueError(f"cannot resolve transformer layer for {layer!r}")
        component_site, component_source_layer = resolve_view_site(view, component_layer)
        component_location = (
            "embedding output"
            if component_site == "embed"
            else f"{component_site.title()} layer {component_source_layer}"
        )
        return {
            "view": view,
            "componentLabel": (
                "channel" if view == "attention-neurons" else "MLP neuron"
            ),
            "tokenPosition": self.token_index,
            "token": self.tokens[self.token_index],
            "layer": layer,
            "transformerLayer": self._transformer_layer_index(layer),
            "isFinalLayer": self._is_final_transformer_layer(layer),
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
                    "channelTokenContributions": (
                        {"layer": layer, "channel": neuron.index, "tokenPosition": self.token_index,
                         "hasImages": bool(self._prompt_images()),
                         **self._attention_channel_head(result_state, neuron.index)}
                        if view == "attention-neurons" and result_state is not None
                        and "attention_weights" in result_state.tensors
                        and "attention_cached_values" in result_state.tensors
                        else None
                    ),
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
                    "jLensFallback": bool(
                        getattr(neuron, "j_lens_metadata", None)
                        and neuron.j_lens_metadata.get("fallback")
                    ),
                    "rLensFallback": bool(
                        getattr(neuron, "r_lens_metadata", None)
                        and neuron.r_lens_metadata.get("fallback")
                    ),
                    "jppLensFallback": bool(
                        getattr(neuron, "jpp_lens_metadata", None)
                        and neuron.jpp_lens_metadata.get("fallback")
                    ),
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
                    "jLensUnavailable": (
                        f"No J-Lens {component_location} map is available."
                        if include_j_lens and not neuron.j_lens else None
                    ),
                    "rLensUnavailable": (
                        f"No R-Lens {component_location} map is available."
                        if include_r_lens and not neuron.r_lens else None
                    ),
                    "jppLensUnavailable": (
                        f"No J++ Lens {component_location} map is available."
                        if include_jpp_lens and not neuron.jpp_lens else None
                    ),
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
            except LensUnavailable as error:
                self._console(f"Warning: {' '.join(str(error).splitlines())}", warning=True)
                if request.get("command") == "llmdbg/neuronLens":
                    view = str(request.get("arguments", {}).get("view", "neurons"))
                    snapshot = {
                        "view": view, "error": str(error), "unavailable": True,
                    }
                    mode = str(request.get("arguments", {}).get("mode", "logit"))
                    key = (self.token_index, self.layer_index, self.inside_mlp,
                           self.inside_attention, self.mlp_step_index, mode, view)
                    self.lens_snapshot_cache[key] = snapshot
                    self.send_response(request, snapshot)
                else:
                    self.send_response(request, success=False, message=str(error))
            except Exception as error:  # keep protocol errors visible to the client
                if isinstance(error, DebugActionError):
                    self._console(f"Warning: {' '.join(str(error).splitlines())}", warning=True)
                else:
                    self.send_event("output", {
                        "category": "stderr",
                        "output": f"llmdbg: {request.get('command')} failed:\n{traceback.format_exc()}",
                    })
                if getattr(self, "_last_response_request", None) != request["seq"]:
                    self.send_response(request, success=False, message=str(error))
                if request.get("command") in {
                    "next", "stepIn", "stepOut", "continue", "configurationDone", "pause",
                }:
                    # A failed internal step may reference a missing detail vector.
                    # Recover at its enclosing layer, which still has valid residuals.
                    self.inside_mlp = False
                    self.inside_attention = False
                    self.mlp_step_index = 0
                    if self.trace and 0 <= self.layer_index < len(self.trace):
                        self.stop("exception")
                    else:
                        self.send_event("terminated")
                self.send_event("llmdbg/computationFailed", {"message": str(error)})


def main() -> None:
    """Run the debug adapter over stdin/stdout."""
    LLMDebugAdapter(ProtocolStream(sys.stdin.buffer, sys.stdout.buffer)).run()


if __name__ == "__main__":
    main()
