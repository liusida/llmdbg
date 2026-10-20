"""Persistent TCP host for reusable llmdbg model runtimes."""

from __future__ import annotations

import argparse
import socket
import sys

from llmdbg.dap import LLMDebugAdapter, ProtocolStream


def serve(host: str, port: int) -> None:
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
                try:
                    LLMDebugAdapter(ProtocolStream(reader, writer)).run()
                finally:
                    writer.close()
                    reader.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the persistent llmdbg model host")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    arguments = parser.parse_args()
    try:
        serve(arguments.host, arguments.port)
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    main()
