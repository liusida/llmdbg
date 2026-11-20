"""Shared llmdbg sessions: let attached clients drive an already-paused session.

The debug adapter that VS Code (or the terminal client) launched remains the
session owner. When sharing is enabled, the adapter also listens on a loopback
port. A client that presents the session token becomes an *attached client*:
its requests are queued onto the same adapter, responses go back to it alone,
and every event (``stopped``, ``output``, ``llmdbg/vectorChanged`` ...) is sent
to the owner and all attached clients. Stepping from an attached client is
therefore reflected in the owner's UI exactly like a breakpoint hit.

Discovery files live in ``$XDG_CACHE_HOME/llmdbg/sessions`` (owner-only) and
contain the port and token. ``python -m llmdbg.shared_session`` is a minimal
attached client.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import secrets
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Any

from llmdbg.dap import ProtocolStream

ATTACH_COMMAND = "llmdbg/attach"
# Requests that configure or end the owner's session stay with the owner.
# setBreakpoints from an attached client replaces the session's breakpoints for
# that source; the adapter emits llmdbg/breakpointsChanged so editors show them.
OWNER_ONLY = frozenset({
    "initialize", "launch", "attach", "configurationDone",
    "setExceptionBreakpoints", "terminate", "restart",
    # The person's controls over agents.
    "llmdbg/setAgentAccess", "llmdbg/disconnectClient",
})
EXECUTION_COMMANDS = frozenset({"next", "stepIn", "stepOut", "continue"})


def sessions_directory() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "llmdbg" / "sessions"


def pump(stream: ProtocolStream, inbox: queue.Queue) -> None:
    """Forward one client's requests into the adapter inbox; None marks EOF."""
    try:
        while (message := stream.read()) is not None:
            if message.get("type") == "request":
                inbox.put((stream, message))
    except (OSError, ValueError, EOFError):
        pass
    inbox.put((stream, None))


class SharedSession:
    """Loopback listener plus discovery file for one adapter session."""

    def __init__(self, inbox: queue.Queue, *, host: str = "127.0.0.1", port: int = 0,
                 description: dict[str, Any] | None = None) -> None:
        self.inbox = inbox
        self.token = secrets.token_urlsafe(24)
        self.pending: set[ProtocolStream] = set()
        self.attached: list[ProtocolStream] = []
        self.labels: dict[ProtocolStream, str] = {}
        self._next_number = 1
        self._connections: dict[ProtocolStream, socket.socket] = {}
        self.listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.listener.bind((host, port))
        self.listener.listen()
        self.host, self.port = self.listener.getsockname()[:2]
        directory = sessions_directory()
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.path = directory / f"{os.getpid()}-{self.port}.json"
        record = {"host": self.host, "port": self.port, "token": self.token,
                  "pid": os.getpid(), "started": time.time(), **(description or {})}
        descriptor = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(descriptor, "w") as handle:
            json.dump(record, handle)
        threading.Thread(target=self._accept, daemon=True, name="llmdbg-share").start()

    def _accept(self) -> None:
        while True:
            try:
                connection, _address = self.listener.accept()
            except OSError:
                return
            stream = ProtocolStream(connection.makefile("rb"), connection.makefile("wb"))
            self._connections[stream] = connection
            self.pending.add(stream)
            threading.Thread(target=pump, args=(stream, self.inbox), daemon=True).start()

    def owns(self, stream: ProtocolStream) -> bool:
        return stream in self.pending or stream in self.attached

    def authenticate(self, stream: ProtocolStream, token: Any, name: Any = None, pid: Any = None) -> bool:
        if not isinstance(token, str) or not secrets.compare_digest(token, self.token):
            return False
        self.pending.discard(stream)
        self.attached.append(stream)
        # Self-reported and unauthenticated: a label for logs, not an identity check.
        label = " ".join(str(name or "client").split())[:48] or "client"
        self.labels[stream] = f"{label} #{self._next_number}" + (f" (pid {int(pid)})" if str(pid).isdigit() else "")
        self._next_number += 1
        return True

    def label(self, stream: ProtocolStream) -> str:
        return self.labels.get(stream, "attached client")

    def detach(self, stream: ProtocolStream) -> None:
        self.pending.discard(stream)
        if stream in self.attached:
            self.attached.remove(stream)
        self.labels.pop(stream, None)
        connection = self._connections.pop(stream, None)
        if connection is not None:
            try:
                connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            connection.close()

    def broadcast(self, message: dict[str, Any]) -> None:
        for stream in tuple(self.attached):
            try:
                stream.write(message)
            except (OSError, ValueError):
                self.detach(stream)

    def close(self) -> None:
        try:
            self.listener.close()
        except OSError:
            pass
        for stream in tuple(self.pending) + tuple(self.attached):
            self.detach(stream)
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass


# ---------------------------------------------------------------- minimal client

class AttachedClient:
    def __init__(self, record: dict[str, Any], timeout: float = 600, name: str | None = None) -> None:
        self.connection = socket.create_connection((record["host"], record["port"]), timeout=timeout)
        self.stream = ProtocolStream(self.connection.makefile("rb"), self.connection.makefile("wb"))
        self.sequence = 1
        self.events: list[dict[str, Any]] = []
        response = self.request(ATTACH_COMMAND, {
            "token": record["token"], "pid": os.getpid(),
            "clientName": name or os.environ.get("LLMDBG_CLIENT_NAME") or "llmdbg.shared_session",
        })
        if not response.get("success"):
            raise RuntimeError(response.get("message") or "attach refused")

    def request(self, command: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        seq = self.sequence
        self.sequence += 1
        self.stream.write({"seq": seq, "type": "request", "command": command,
                           "arguments": arguments or {}})
        while True:
            message = self.stream.read()
            if message is None:
                raise EOFError("session closed")
            if message.get("type") == "response" and message.get("request_seq") == seq:
                return message
            if message.get("type") == "event":
                self.events.append(message)

    def wait_event(self, names: set[str]) -> dict[str, Any]:
        for index, event in enumerate(self.events):
            if event.get("event") in names:
                return self.events.pop(index)
        while True:
            message = self.stream.read()
            if message is None:
                raise EOFError("session closed")
            if message.get("type") == "event":
                if message.get("event") in names:
                    return message
                self.events.append(message)

    def where(self) -> dict[str, Any]:
        frames = self.request("stackTrace", {"threadId": 1}).get("body", {}).get("stackFrames", [])
        return {"frames": [{"name": frame.get("name"), "line": frame.get("line")} for frame in frames]}

    def close(self) -> None:
        try:
            self.request("disconnect")
        except (OSError, EOFError):
            pass
        self.connection.close()


def discover(path: str | None = None) -> dict[str, Any]:
    if path:
        return json.loads(Path(path).read_text())
    records = []
    for candidate in sessions_directory().glob("*.json"):
        try:
            record = json.loads(candidate.read_text())
            os.kill(int(record["pid"]), 0)
        except (OSError, ValueError, KeyError):
            continue
        records.append(record)
    if not records:
        raise SystemExit("No shared llmdbg session found. Launch with \"shareSession\": true.")
    return max(records, key=lambda record: record.get("started", 0))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Minimal attached client for a shared llmdbg session")
    parser.add_argument("action", choices=["where", "next", "stepIn", "stepOut", "continue"])
    parser.add_argument("--session", help="Discovery JSON file (default: newest live session)")
    arguments = parser.parse_args(argv)
    client = AttachedClient(discover(arguments.session))
    try:
        if arguments.action == "where":
            result = client.where()
        else:
            response = client.request(arguments.action, {"threadId": 1})
            if not response.get("success"):
                raise SystemExit(response.get("message") or f"{arguments.action} failed")
            stopped = client.wait_event({"stopped", "terminated"})
            result = {"event": stopped.get("event"), **(stopped.get("body") or {}), **client.where()}
        json.dump(result, sys.stdout, indent=2)
        print()
    finally:
        client.close()


if __name__ == "__main__":
    main()
