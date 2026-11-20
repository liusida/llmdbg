"""Shared sessions: an attached client steps the owner's paused session."""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from llmdbg.dap import LLMDebugAdapter, ProtocolStream
from llmdbg.shared_session import AttachedClient, discover


class OwnerClient:
    """The session owner (VS Code's role) on one end of a socket pair."""

    def __init__(self, connection: socket.socket) -> None:
        connection.settimeout(30)
        self.stream = ProtocolStream(connection.makefile("rb"), connection.makefile("wb"))
        self.seq = 0
        self.received: list[dict] = []

    def send(self, command: str, **arguments) -> int:
        self.seq += 1
        self.stream.write({"seq": self.seq, "type": "request", "command": command, "arguments": arguments})
        return self.seq

    def until(self, predicate) -> dict:
        while True:
            message = self.stream.read()
            assert message is not None, "adapter closed the owner stream"
            self.received.append(message)
            if predicate(message):
                return message


def is_event(name):
    return lambda message: message.get("type") == "event" and message.get("event") == name


class SharedSessionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.cache = tempfile.TemporaryDirectory()
        patcher = mock.patch.dict(os.environ, {"XDG_CACHE_HOME": self.cache.name})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self.cache.cleanup)
        adapter_end, owner_end = socket.socketpair()
        self.adapter = LLMDebugAdapter(ProtocolStream(adapter_end.makefile("rb"), adapter_end.makefile("wb")))
        self.thread = threading.Thread(target=self.adapter.run, daemon=True)
        self.thread.start()
        self.owner = OwnerClient(owner_end)
        self.owner.send("initialize")
        self.owner.send("launch", stopOnEntry=True)  # sharing is on by default
        self.owner.send("configurationDone")
        self.owner.until(is_event("stopped"))

    def tearDown(self) -> None:
        self.owner.send("disconnect")
        self.thread.join(timeout=10)
        self.assertFalse(self.thread.is_alive())
        self.assertEqual(list((discover.__globals__["sessions_directory"]()).glob("*.json")), [])

    def test_attached_step_is_reported_to_the_owner(self) -> None:
        client = AttachedClient(discover())
        before = client.where()
        response = client.request("next", {"threadId": 1})
        self.assertTrue(response["success"])
        stopped = client.wait_event({"stopped"})
        self.assertEqual(stopped["body"]["reason"], "step")
        # The owner sees the step exactly as it would a breakpoint hit.
        self.owner.until(is_event("continued"))
        self.owner.until(is_event("stopped"))
        self.assertFalse(any(m.get("type") == "response" and m.get("command") == "next"
                             for m in self.owner.received))
        self.assertNotEqual(before, client.where())
        client.close()

    def test_bad_token_and_owner_only_requests_are_refused(self) -> None:
        record = discover()
        with self.assertRaises(RuntimeError):
            AttachedClient({**record, "token": "wrong"})
        client = AttachedClient(record)
        refused = client.request("launch", {})
        self.assertFalse(refused["success"])
        self.assertIn("owner", refused["message"])
        client.close()
        # The owner session is still alive and responsive after the client leaves.
        seq = self.owner.send("threads")
        self.owner.until(lambda m: m.get("type") == "response" and m.get("request_seq") == seq)

    def owner_request(self, command: str, **arguments) -> dict:
        seq = self.owner.send(command, **arguments)
        return self.owner.until(lambda m: m.get("type") == "response" and m.get("request_seq") == seq)

    def test_attached_client_manages_the_sessions_breakpoints(self) -> None:
        lines = sorted(set(self.adapter.source_lines))
        owner_line, client_line = lines[-1], lines[2]
        self.owner_request("setBreakpoints", source={}, breakpoints=[{"line": owner_line}])
        client = AttachedClient(discover())
        response = client.request("setBreakpoints", {"source": {}, "breakpoints": [{"line": client_line}]})
        self.assertTrue(response["body"]["breakpoints"][0]["verified"])
        # The client's set replaces the owner's, and the owner is told to show it.
        self.assertEqual([bp.line for bp in self.adapter.breakpoints], [client_line])
        changed = self.owner.until(is_event("llmdbg/breakpointsChanged"))
        self.assertEqual(changed["body"]["breakpoints"], [{"line": client_line}])
        client.request("continue", {"threadId": 1})
        self.assertEqual(client.wait_event({"stopped"})["body"]["reason"], "breakpoint")
        self.assertEqual(client.where()["frames"][0]["line"], client_line)
        client.close()
        self.owner_request("threads")
        self.assertEqual([bp.line for bp in self.adapter.breakpoints], [client_line])

    def test_terminal_attach_mode_drives_the_owner_session(self) -> None:
        root = Path(__file__).resolve().parents[1]
        env = dict(os.environ, PYTHONPATH=str(root / "src"), PYTHONWARNINGS="ignore")
        result = subprocess.run(
            [sys.executable, "-m", "llmdbg", "debug", "--attach", "--client-name", "test agent", "--json",
             "-c", "where", "-c", "next", "-c", "print residual_stream", "-c", "launch x"],
            text=True, capture_output=True, cwd=root, env=env, timeout=60,
        )
        records = [json.loads(line) for line in result.stdout.splitlines()]
        self.assertEqual([r["command"] for r in records], ["attach", "where", "next", "print residual_stream", "launch x"])
        self.assertTrue(all(r["ok"] for r in records[:4]), result.stdout + result.stderr)
        self.assertEqual(records[2]["body"]["reason"], "step")
        self.assertRegex(records[0]["body"]["as"], r"^test agent #\d+ \(pid \d+\)$")
        self.assertIn("(by test agent #", records[2]["body"]["description"])
        self.assertFalse(records[4]["ok"])
        self.owner.until(is_event("continued"))
        self.owner.until(is_event("stopped"))
        self.owner_request("threads")  # owner session survives the client leaving
        log = "".join(m.get("body", {}).get("output", "") for m in self.owner.received if m.get("event") == "output")
        self.assertIn("[test agent #", log)
        self.assertIn("] requested next.", log)

    def test_session_file_tells_an_agent_how_to_connect(self) -> None:
        record = discover()
        self.assertEqual(record["protocol"], 1)
        self.assertEqual(record["client"]["command"][1:], ["-m", "llmdbg", "debug", "--attach"])
        self.assertTrue(Path(record["client"]["env"]["PYTHONPATH"], "llmdbg", "dap.py").exists())
        self.assertIn("Never print, log, or send the token", Path(record["docs"]).read_text())

    def test_activity_log_records_agent_actions_and_results(self) -> None:
        client = AttachedClient(discover(), name="test-agent")
        client.request("evaluate", {"expression": "residual_stream[0]"})
        client.request("evaluate", {"expression": "no_such_thing"})
        client.request("next", {"threadId": 1})
        client.wait_event({"stopped"})
        client.close()
        state = self.owner_request("llmdbg/controlState")["body"]
        rows = [(row["actor"].split(" #")[0], row["command"], row["ok"]) for row in state["activity"]]
        self.assertEqual(rows, [("test-agent", "attach", True), ("test-agent", "evaluate", True),
                                ("test-agent", "evaluate", False), ("test-agent", "next", True),
                                ("test-agent", "disconnect", True)])
        self.assertIn("→ token 0", state["activity"][3]["summary"])
        self.assertEqual(state["clients"], [])
        self.assertTrue(any(m.get("event") == "llmdbg/controlState" for m in self.owner.received))

    def test_read_only_and_off_modes(self) -> None:
        self.owner_request("llmdbg/setAgentAccess", mode="readonly")
        client = AttachedClient(discover(), name="reader")
        self.assertTrue(client.request("evaluate", {"expression": "residual_stream[0]"})["success"])
        for command, arguments in [("next", {"threadId": 1}), ("llmdbg/assign", {"target": "x", "expression": "x"}),
                                   ("llmdbg/plot", {"expression": "x", "code": "pass"})]:
            refused = client.request(command, arguments)
            self.assertFalse(refused["success"], command)
            self.assertIn("read-only", refused["message"])
        # Agents cannot change their own access.
        self.assertFalse(client.request("llmdbg/setAgentAccess", {"mode": "full"})["success"])
        turned_off = self.owner_request("llmdbg/setAgentAccess", mode="off")["body"]
        self.assertEqual([label.split(" #")[0] for label in turned_off["disconnected"]], ["reader"])
        with self.assertRaises((EOFError, OSError, RuntimeError)):
            client.request("threads")
        with self.assertRaises(RuntimeError):
            AttachedClient(discover(), name="late")
        self.owner_request("llmdbg/setAgentAccess", mode="full")
        AttachedClient(discover(), name="allowed-again").close()

    def test_owner_can_disconnect_one_agent(self) -> None:
        first, second = AttachedClient(discover(), name="one"), AttachedClient(discover(), name="two")
        self.owner_request("threads")
        labels = self.owner_request("llmdbg/controlState")["body"]["clients"]
        target = next(label for label in labels if label.startswith("one #"))
        self.assertTrue(self.owner_request("llmdbg/disconnectClient", label=target)["body"]["disconnected"])
        with self.assertRaises((EOFError, OSError)):
            first.request("threads")
        self.assertTrue(second.request("threads")["success"])
        second.close()


class SharingOptOutTest(unittest.TestCase):
    def test_share_session_false_opens_no_listener(self) -> None:
        import io
        from tests.test_dap import frame, parse_frames
        with tempfile.TemporaryDirectory() as cache, mock.patch.dict(os.environ, {"XDG_CACHE_HOME": cache}):
            requests = [
                {"seq": 1, "type": "request", "command": "launch", "arguments": {"shareSession": False}},
                {"seq": 2, "type": "request", "command": "disconnect"},
            ]
            output = io.BytesIO()
            LLMDebugAdapter(ProtocolStream(io.BytesIO(b"".join(map(frame, requests))), output)).run()
            text = " ".join(m.get("body", {}).get("output", "") for m in parse_frames(output.getvalue()))
            self.assertNotIn("Shared session listening", text)
            self.assertFalse((discover.__globals__["sessions_directory"]()).exists())


if __name__ == "__main__":
    unittest.main()
