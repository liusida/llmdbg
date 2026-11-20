"""Persistent TCP host for reusable llmdbg model runtimes.

Every connection must authenticate before it can do anything beyond
``initialize``: its ``launch`` request carries ``hostToken``, a secret the
host receives through the ``LLMDBG_HOST_TOKEN`` environment variable (never
the command line, which other users can read with ``ps``). Without it, any
local account could start a session on the host and run code as its owner.
"""

from __future__ import annotations

import argparse
import os
import secrets
import socket
import sys
import threading

from llmdbg.dap import LLMDebugAdapter, ProtocolStream

TOKEN_ENVIRONMENT = "LLMDBG_HOST_TOKEN"
# Unauthenticated connections are closed after this many seconds, so an idle
# connection cannot hold the host's one-session-at-a-time queue.
AUTHENTICATION_TIMEOUT = float(os.environ.get("LLMDBG_HOST_AUTH_TIMEOUT", "30"))


def _close_if_unauthenticated(adapter: LLMDebugAdapter, connection: socket.socket) -> None:
    if not adapter.host_authenticated:
        try:
            connection.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass


def serve(host: str, port: int, token: str) -> None:
    """Accept sequential DAP sessions while keeping model globals resident."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((host, port))
        listener.listen()
        actual_port = listener.getsockname()[1]
        print(f"LLMDBG_HOST_READY {actual_port}", flush=True)
        while True:
            connection, _address = listener.accept()
            with connection:
                reader = connection.makefile("rb")
                writer = connection.makefile("wb")
                adapter = LLMDebugAdapter(ProtocolStream(reader, writer), host_token=token)
                watchdog = threading.Timer(
                    AUTHENTICATION_TIMEOUT, _close_if_unauthenticated, (adapter, connection)
                )
                watchdog.daemon = True
                watchdog.start()
                try:
                    adapter.run()
                finally:
                    watchdog.cancel()
                    writer.close()
                    reader.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the persistent llmdbg model host")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    arguments = parser.parse_args()
    # Remove the secret from this process's environment so children do not inherit it.
    token = os.environ.pop(TOKEN_ENVIRONMENT, "")
    if not token:
        token = secrets.token_urlsafe(32)
        # Started by hand: stdout is the starter's own terminal or pipe.
        print(f"{TOKEN_ENVIRONMENT} {token}", flush=True)
    try:
        serve(arguments.host, arguments.port, token)
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    main()
