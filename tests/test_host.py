from __future__ import annotations

import json
import socket
import subprocess
import sys
import unittest


def frame(message: dict[str, object]) -> bytes:
    payload = json.dumps(message).encode()
    return f"Content-Length: {len(payload)}\r\n\r\n".encode() + payload


class PersistentHostTest(unittest.TestCase):
    def test_accepts_consecutive_debug_sessions_in_one_process(self) -> None:
        process = subprocess.Popen(
            [sys.executable, "-m", "llmdbg.host", "--port", "0"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            assert process.stdout is not None
            ready = process.stdout.readline().strip()
            self.assertTrue(ready.startswith("LLMDBG_HOST_READY "), ready)
            port = int(ready.rsplit(" ", 1)[1])

            for session in range(2):
                requests = [
                    {"seq": 1, "type": "request", "command": "initialize", "arguments": {}},
                    {"seq": 2, "type": "request", "command": "launch", "arguments": {}},
                    {"seq": 3, "type": "request", "command": "disconnect", "arguments": {}},
                ]
                with socket.create_connection(("127.0.0.1", port), timeout=5) as connection:
                    connection.sendall(b"".join(frame(request) for request in requests))
                    connection.shutdown(socket.SHUT_WR)
                    response = bytearray()
                    while chunk := connection.recv(65536):
                        response.extend(chunk)
                self.assertIn(b'"command":"launch"', response, f"session {session}")
                self.assertIsNone(process.poll())
        finally:
            process.terminate()
            process.wait(timeout=5)
            if process.stdout is not None:
                process.stdout.close()
            if process.stderr is not None:
                process.stderr.close()


if __name__ == "__main__":
    unittest.main()
