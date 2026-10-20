from __future__ import annotations

import json
import subprocess
import sys
import unittest


def frame(message: dict[str, object]) -> bytes:
    payload = json.dumps(message).encode()
    return f"Content-Length: {len(payload)}\r\n\r\n".encode() + payload


def parse_frames(data: bytes) -> list[dict[str, object]]:
    messages = []
    while data:
        header, data = data.split(b"\r\n\r\n", 1)
        length = int(header.decode().split(":", 1)[1].strip())
        payload, data = data[:length], data[length:]
        messages.append(json.loads(payload))
    return messages


class DAPIntegrationTest(unittest.TestCase):
    def test_layer_step_and_variables(self) -> None:
        requests = [
            {"seq": 1, "type": "request", "command": "initialize", "arguments": {}},
            {"seq": 2, "type": "request", "command": "launch", "arguments": {"stopOnEntry": True}},
            {"seq": 3, "type": "request", "command": "configurationDone", "arguments": {}},
            {"seq": 4, "type": "request", "command": "threads", "arguments": {}},
            {"seq": 5, "type": "request", "command": "stackTrace", "arguments": {"threadId": 1}},
            {"seq": 6, "type": "request", "command": "scopes", "arguments": {"frameId": 1}},
            {"seq": 7, "type": "request", "command": "variables", "arguments": {"variablesReference": 100}},
            {"seq": 8, "type": "request", "command": "next", "arguments": {"threadId": 1}},
            {"seq": 9, "type": "request", "command": "evaluate", "arguments": {"expression": "layer"}},
            {"seq": 10, "type": "request", "command": "source", "arguments": {"sourceReference": 1}},
            {"seq": 11, "type": "request", "command": "disconnect", "arguments": {}},
        ]
        process = subprocess.run(
            [sys.executable, "-m", "llmdbg.dap"],
            input=b"".join(frame(request) for request in requests),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertEqual(process.returncode, 0, process.stderr.decode())
        messages = parse_frames(process.stdout)
        responses = {message.get("request_seq"): message for message in messages if message["type"] == "response"}
        self.assertTrue(all(response["success"] for response in responses.values()))
        self.assertEqual(responses[4]["body"]["threads"][0]["name"], "LLM forward pass")
        self.assertIn("token_embedding", responses[5]["body"]["stackFrames"][0]["name"])
        source = responses[5]["body"]["stackFrames"][0]["source"]
        self.assertEqual(source["sourceReference"], 1)
        self.assertTrue(source["name"].endswith(".execution.llmdbg"))
        layer_variables = responses[7]["body"]["variables"]
        self.assertTrue(any(variable["name"] == "meaning" for variable in layer_variables))
        self.assertEqual(responses[9]["body"]["result"], "attention_0")
        virtual_source = responses[10]["body"]["content"]
        self.assertIn("[session]", virtual_source)
        self.assertIn("[[step]]", virtual_source)
        self.assertIn('kind = "embedding"', virtual_source)
        stopped = [message for message in messages if message.get("event") == "stopped"]
        self.assertEqual([event["body"]["reason"] for event in stopped], ["entry", "step"])


if __name__ == "__main__":
    unittest.main()
