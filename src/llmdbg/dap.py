"""Minimal Debug Adapter Protocol implementation for llmdbg."""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

from llmdbg.config import load_debug_config
from llmdbg.demo_model import FEATURES, LayerState, run_demo
from llmdbg.model_runtime import ModelExecution, prepare_model_execution


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
    SCOPE_LAYER = 100
    SCOPE_INPUT = 200
    SCOPE_OUTPUT = 300
    SOURCE_REFERENCE = 1

    def __init__(self, stream: ProtocolStream) -> None:
        self.stream = stream
        self.sequence = 1
        self.trace: tuple[LayerState, ...] = ()
        self.execution: ModelExecution | None = None
        self.tokens: tuple[str, ...] = ()
        self.token_index = 0
        self.layer_index = 0
        self.launched = False
        self.stop_on_entry = True
        self.source_content = ""
        self.source_lines: tuple[int, ...] = ()
        self.step_paths: tuple[str, ...] = ()
        self.source_name = "model.execution.llmdbg"
        self.source_path: str | None = None
        self.breakpoints: tuple[SourceBreakpoint, ...] = ()

    @property
    def state(self) -> LayerState:
        return self.trace[self.layer_index]

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
                    "supportsStepBack": False,
                    "supportsRestartRequest": False,
                },
            )
        elif command == "launch":
            if arguments.get("config"):
                file_arguments = load_debug_config(str(arguments["config"]))
                arguments = {**file_arguments, **arguments}
            model = arguments.get("model")
            if model:
                self.send_event(
                    "output",
                    {"category": "console", "output": f"llmdbg: loading {model}\n"},
                )
                self.execution = prepare_model_execution(
                    str(model),
                    revision=str(arguments.get("revision", "main")),
                    prompt=str(arguments.get("prompt", "The capital of France is")),
                    device=str(arguments.get("device", "cuda")),
                    dtype=str(arguments.get("dtype", "bfloat16")),
                    progress=lambda output: self.send_event(
                        "output", {"category": "console", "output": f"llmdbg: {output}"}
                    ),
                )
                self.tokens = self.execution.tokens
                self._configure_model_source(arguments)
            else:
                self.trace = run_demo()
                self.tokens = ("demo",)
                self._build_virtual_source(arguments, "built-in-demo", ())
            self.token_index = 0
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
                "line": self.source_lines[self.layer_index],
                "column": 1,
                "presentationHint": "normal",
            }
            self.send_response(request, {"stackFrames": [frame], "totalFrames": 1})
        elif command == "source":
            reference = int(arguments.get("sourceReference", 0))
            if reference != self.SOURCE_REFERENCE:
                self.send_response(request, success=False, message="Unknown source reference")
            else:
                self.send_response(
                    request,
                    {"content": self.source_content, "mimeType": "text/x-toml"},
                )
        elif command == "scopes":
            scopes = [
                {"name": "Token", "variablesReference": self.SCOPE_TOKEN, "expensive": False},
                {"name": "Layer", "variablesReference": self.SCOPE_LAYER, "expensive": False},
                {"name": "Input vector", "variablesReference": self.SCOPE_INPUT, "expensive": False},
                {"name": "Output vector", "variablesReference": self.SCOPE_OUTPUT, "expensive": False},
            ]
            self.send_response(request, {"scopes": scopes})
        elif command == "variables":
            reference = int(arguments.get("variablesReference", 0))
            self.send_response(request, {"variables": self._variables(reference)})
        elif command == "evaluate":
            expression = str(arguments.get("expression", "")).strip().lower()
            result = self._evaluate(expression)
            if result is None:
                self.send_response(request, success=False, message=f"Unknown expression: {expression}")
            else:
                self.send_response(request, {"result": result, "variablesReference": 0})
        elif command in ("next", "stepIn", "stepOut"):
            self.send_response(request)
            if self.layer_index + 1 < len(self.trace):
                self.layer_index += 1
                self.stop("step")
            elif self.token_index + 1 < len(self.tokens):
                self._load_token(self.token_index + 1)
                self.layer_index = 0
                self.stop("step")
            else:
                self.terminate()
        elif command == "continue":
            self.send_response(request, {"allThreadsContinued": True})
            self._continue_to_breakpoint(include_current=False)
        elif command == "pause":
            self.send_response(request)
            self.stop("pause")
        elif command == "setBreakpoints":
            requested = [int(item.get("line", 0)) for item in arguments.get("breakpoints", [])]
            valid_lines = sorted(set(self.source_lines))
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
                            "message": "Breakpoints must be set inside a [[step]] table",
                        }
                    )
                else:
                    condition = str(item.get("condition", "")).strip()
                    token_position = self._parse_token_condition(condition)
                    if condition and token_position is None:
                        breakpoint.update(
                            {
                                "verified": False,
                                "line": resolved_line,
                                "message": (
                                    "Use a token condition such as token.position == 4"
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
        """Build the read-only TOML execution view served through DAP."""
        safe_model_name = model.rstrip("/").split("/")[-1] or "model"
        self.source_name = f"{safe_model_name}.execution.llmdbg"
        self.source_path = None
        lines = [
            "# Generated by llmdbg. This virtual source is read-only.",
            "[session]",
            f"model = {json.dumps(model)}",
            f"revision = {json.dumps(str(arguments.get('revision', 'main')))}",
            f"device = {json.dumps(str(arguments.get('device', 'cuda')))}",
            f"dtype = {json.dumps(str(arguments.get('dtype', 'bfloat16')))}",
            f"prompt = {json.dumps(str(arguments.get('prompt', '')))}",
            f"tokens = {json.dumps(list(tokens))}",
        ]
        source_lines = []
        for index, state in enumerate(self.trace):
            lines.append("")
            source_lines.append(len(lines) + 1)
            if state.layer.name == "token_embedding":
                kind = "embedding"
            elif state.layer.name.endswith(".attention"):
                kind = "attention"
            elif state.layer.name.endswith(".mlp"):
                kind = "mlp"
            elif state.layer.name.endswith("ln_f"):
                kind = "normalization"
            else:
                kind = "layer"
            lines.extend(
                [
                    "[[step]]",
                    f"index = {index}",
                    f"kind = {json.dumps(kind)}",
                    f"path = {json.dumps(state.layer.name)}",
                    f"operation = {json.dumps(state.layer.operation)}",
                    f"description = {json.dumps(state.layer.meaning)}",
                ]
            )
        self.source_content = "\n".join(lines) + "\n"
        self.source_lines = tuple(source_lines)

    def _configure_model_source(self, arguments: dict[str, Any]) -> None:
        """Bind authored [[step]] declarations to the model execution order."""
        if self.execution is None:
            raise RuntimeError("model execution is not prepared")
        model = self.execution.runtime.model
        available = ["token_embedding"]
        for index in range(len(model.transformer.h)):
            available.extend(
                [f"transformer.h.{index}.attention", f"transformer.h.{index}.mlp"]
            )
        available.append("transformer.ln_f")
        step_paths = list(arguments.get("stepPaths", []))
        selected = [path for path in available if path in set(step_paths)]
        if selected != step_paths:
            raise ValueError(
                "[[step]] paths must be a valid subsequence of model execution; "
                f"available paths are {available!r}"
            )
        path = Path(str(arguments["configPath"]))
        step_lines = arguments.get("stepLines", {})
        self.source_name = path.name
        self.source_path = str(path)
        self.source_lines = tuple(int(step_lines[name]) for name in step_paths)
        self.step_paths = tuple(step_paths)

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
            r"(?:token(?:\.position)?|position)\s*==\s*(\d+)",
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
        """Advance in token-major order to the next matching source breakpoint."""
        token_count = len(self.tokens)
        if before_start:
            start_token, start_layer = 0, 0
        elif include_current:
            start_token, start_layer = self.token_index, self.layer_index
        else:
            start_token, start_layer = self.token_index, self.layer_index + 1
        for token_position in range(start_token, token_count):
            layer_start = start_layer if token_position == start_token else 0
            for layer_index in range(layer_start, len(self.source_lines)):
                line = self.source_lines[layer_index]
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
                    self.stop("breakpoint")
                    return
        self.terminate()

    def _variables(self, reference: int) -> list[dict[str, Any]]:
        if reference == self.SCOPE_TOKEN:
            token_id = (
                int(self.execution.input_ids[0, self.token_index].item())
                if self.execution is not None
                else 0
            )
            prefix = "".join(self.tokens[: self.token_index + 1])
            return [
                {
                    "name": "position",
                    "value": str(self.token_index),
                    "type": "int",
                    "variablesReference": 0,
                },
                {
                    "name": "id",
                    "value": str(token_id),
                    "type": "int",
                    "variablesReference": 0,
                },
                {
                    "name": "text",
                    "value": repr(self.tokens[self.token_index]),
                    "type": "str",
                    "variablesReference": 0,
                },
                {
                    "name": "prefix",
                    "value": repr(prefix),
                    "type": "str",
                    "variablesReference": 0,
                },
            ]
        if reference == self.SCOPE_LAYER:
            return [
                {"name": "index", "value": str(self.layer_index), "type": "int", "variablesReference": 0},
                {"name": "name", "value": self.state.layer.name, "type": "str", "variablesReference": 0},
                {"name": "operation", "value": self.state.layer.operation, "type": "str", "variablesReference": 0},
                {"name": "meaning", "value": self.state.interpretation, "type": "str", "variablesReference": 0},
            ]
        vector = self.state.input_vector if reference == self.SCOPE_INPUT else self.state.output_vector
        if reference not in (self.SCOPE_INPUT, self.SCOPE_OUTPUT):
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
            }
            for feature, value in displayed
        ]

    def _evaluate(self, expression: str) -> str | None:
        if expression in ("token", "token.text"):
            return self.tokens[self.token_index]
        if expression in ("position", "token.position"):
            return str(self.token_index)
        if expression in ("layer", "name"):
            return self.state.layer.name
        if expression in ("meaning", "interpretation"):
            return self.state.interpretation
        if expression == "operation":
            return self.state.layer.operation
        if expression == "input":
            return repr(self.state.input_vector)
        if expression == "output":
            return repr(self.state.output_vector)
        return None

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
