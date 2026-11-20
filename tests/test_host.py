"""The persistent TCP host: resident sessions, and refusal of unauthenticated clients."""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def frame(message: dict[str, object]) -> bytes:
    payload = json.dumps(message).encode()
    return f"Content-Length: {len(payload)}\r\n\r\n".encode() + payload


def request(seq: int, command: str, **arguments) -> dict[str, object]:
    return {"seq": seq, "type": "request", "command": command, "arguments": arguments}


class HostProcess:
    def __init__(self, token: str | None = "secret-for-tests", **environment: str) -> None:
        env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), **environment)
        env.pop("LLMDBG_HOST_TOKEN", None)
        if token is not None:
            env["LLMDBG_HOST_TOKEN"] = token
        self.process = subprocess.Popen(
            [sys.executable, "-m", "llmdbg.host", "--port", "0"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env,
        )
        assert self.process.stdout is not None
        first = self.process.stdout.readline().strip()
        self.printed_token = None
        if first.startswith("LLMDBG_HOST_TOKEN "):
            self.printed_token = first.split(" ", 1)[1]
            first = self.process.stdout.readline().strip()
        assert first.startswith("LLMDBG_HOST_READY "), first
        self.port = int(first.rsplit(" ", 1)[1])

    def exchange(self, *messages: dict[str, object], timeout: float = 10) -> list[dict[str, object]]:
        with socket.create_connection(("127.0.0.1", self.port), timeout=timeout) as connection:
            connection.sendall(b"".join(frame(message) for message in messages))
            connection.shutdown(socket.SHUT_WR)
            data = bytearray()
            while chunk := connection.recv(65536):
                data.extend(chunk)
        replies = []
        while data:
            header, _, rest = bytes(data).partition(b"\r\n\r\n")
            length = int(header.split(b":", 1)[1])
            replies.append(json.loads(rest[:length]))
            data = bytearray(rest[length:])
        return replies

    def close(self) -> None:
        self.process.terminate()
        self.process.wait(timeout=5)
        for stream in (self.process.stdout, self.process.stderr):
            if stream is not None:
                stream.close()


class PersistentHostTest(unittest.TestCase):
    def setUp(self) -> None:
        self.host = HostProcess()
        self.addCleanup(self.host.close)

    def test_accepts_consecutive_authenticated_sessions_in_one_process(self) -> None:
        self.assertIsNone(self.host.printed_token)  # an inherited token is never echoed
        for session in range(2):
            replies = self.host.exchange(
                request(1, "initialize"),
                request(2, "launch", hostToken="secret-for-tests", shareSession=False),
                request(3, "disconnect"),
            )
            launch = next(r for r in replies if r.get("command") == "launch")
            self.assertTrue(launch["success"], f"session {session}: {launch}")
            self.assertIsNone(self.host.process.poll())

    def test_refuses_missing_wrong_or_premature_requests(self) -> None:
        for messages in (
            [request(1, "initialize"), request(2, "launch")],
            [request(1, "initialize"), request(2, "launch", hostToken="wrong")],
            [request(1, "threads")],
            [request(1, "initialize"), request(2, "evaluate", expression="x")],
        ):
            replies = self.host.exchange(*messages)
            last = [r for r in replies if r.get("type") == "response"][-1]
            self.assertFalse(last["success"], messages)
            self.assertIn("host token", last["message"])
            # Refusal closes the session: nothing after it is processed.
            self.assertEqual(len([r for r in replies if r.get("type") == "response"]), len(messages))
        self.assertIsNone(self.host.process.poll())

    def test_plot_code_injection_is_refused_without_the_token(self) -> None:
        """The local-attacker scenario: another account reaches the port and sends plot code."""
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "marker.txt"
            code = (f"open({str(marker)!r}, 'w').write('ran')\n"
                    "def plot(vector, name, plt, np):\n    plt.plot(vector); return plt.gcf()\n")
            replies = self.host.exchange(
                request(1, "initialize"),
                request(2, "launch", stopOnEntry=True, shareSession=False),
                request(3, "configurationDone"),
                request(4, "llmdbg/plot", expression="residual_stream", code=code),
            )
            self.assertFalse(marker.exists())
            self.assertNotIn("llmdbg/plot", [r.get("command") for r in replies])

    def test_the_same_plot_request_runs_with_the_token(self) -> None:
        replies = self.host.exchange(
            request(1, "initialize"),
            request(2, "launch", hostToken="secret-for-tests", stopOnEntry=True, shareSession=False),
            request(3, "configurationDone"),
            request(4, "llmdbg/plot", expression="residual_stream",
                    code="def plot(vector, name, plt, np):\n    plt.plot(vector); return plt.gcf()\n"),
            request(5, "disconnect"),
        )
        plot = next(r for r in replies if r.get("command") == "llmdbg/plot")
        self.assertTrue(plot["success"], plot)


class HostStartupTest(unittest.TestCase):
    def test_host_started_by_hand_prints_a_generated_token(self) -> None:
        host = HostProcess(token=None)
        try:
            self.assertGreaterEqual(len(host.printed_token or ""), 32)
            replies = host.exchange(request(1, "launch", hostToken=host.printed_token, shareSession=False),
                                    request(2, "disconnect"))
            self.assertTrue(replies[0]["success"], replies)
        finally:
            host.close()

    def test_idle_unauthenticated_connection_is_closed(self) -> None:
        host = HostProcess(LLMDBG_HOST_AUTH_TIMEOUT="0.5")
        try:
            started = time.monotonic()
            with socket.create_connection(("127.0.0.1", host.port), timeout=10) as connection:
                self.assertEqual(connection.recv(1), b"")  # closed by the host, not by us
            self.assertLess(time.monotonic() - started, 5)
            # The host is free for the next (authenticated) client.
            replies = host.exchange(request(1, "launch", hostToken="secret-for-tests", shareSession=False),
                                    request(2, "disconnect"))
            self.assertTrue(replies[0]["success"])
        finally:
            host.close()


if __name__ == "__main__":
    unittest.main()
