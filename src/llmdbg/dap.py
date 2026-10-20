"""Minimal Debug Adapter Protocol implementation for llmdbg."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, BinaryIO

from llmdbg.demo_model import FEATURES, LayerState, run_demo


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
    SCOPE_LAYER = 100
    SCOPE_INPUT = 200
    SCOPE_OUTPUT = 300

    def __init__(self, stream: ProtocolStream) -> None:
        self.stream = stream
        self.sequence = 1
        self.trace: tuple[LayerState, ...] = ()
        self.layer_index = 0
        self.launched = False
        self.stop_on_entry = True

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
                "description": f"Stopped after {self.state.layer.name}",
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
                    "supportsStepBack": False,
                    "supportsRestartRequest": False,
                },
            )
            self.send_event("initialized")
        elif command == "launch":
            self.trace = run_demo()
            self.layer_index = 0
            self.launched = True
            self.stop_on_entry = bool(arguments.get("stopOnEntry", True))
            self.send_response(request)
            self.send_event(
                "output",
                {
                    "category": "console",
                    "output": "llmdbg: loaded built-in demo trace; use Step Over to advance one layer\n",
                },
            )
        elif command == "configurationDone":
            self.send_response(request)
            if not self.launched:
                return True
            if self.stop_on_entry:
                self.stop("entry")
            else:
                self.terminate()
        elif command == "threads":
            self.send_response(request, {"threads": [{"id": self.THREAD_ID, "name": "LLM forward pass"}]})
        elif command == "stackTrace":
            source = str(Path(__file__).with_name("demo_model.py"))
            frame = {
                "id": self.FRAME_ID,
                "name": f"{self.layer_index:02d}: {self.state.layer.name}",
                "source": {"name": "demo_model.py", "path": source},
                "line": 1,
                "column": 1,
                "presentationHint": "normal",
            }
            self.send_response(request, {"stackFrames": [frame], "totalFrames": 1})
        elif command == "scopes":
            scopes = [
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
            if self.layer_index + 1 >= len(self.trace):
                self.terminate()
            else:
                self.layer_index += 1
                self.stop("step")
        elif command == "continue":
            self.send_response(request, {"allThreadsContinued": True})
            self.terminate()
        elif command == "pause":
            self.send_response(request)
            self.stop("pause")
        elif command in ("setBreakpoints", "setExceptionBreakpoints"):
            body = {"breakpoints": []} if command == "setBreakpoints" else None
            self.send_response(request, body)
        elif command in ("disconnect", "terminate"):
            self.send_response(request)
            return False
        else:
            self.send_response(request, success=False, message=f"Unsupported request: {command}")
        return True

    def _variables(self, reference: int) -> list[dict[str, Any]]:
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
        return [
            {
                "name": feature,
                "value": f"{value:+.6f}",
                "type": "float",
                "variablesReference": 0,
            }
            for feature, value in zip(FEATURES, vector)
        ]

    def _evaluate(self, expression: str) -> str | None:
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
