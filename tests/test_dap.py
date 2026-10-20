from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from llmdbg.dap import LLMDebugAdapter, SourceBreakpoint
from llmdbg.demo_model import Layer, LayerState
from llmdbg.model_runtime import LogitLensToken, NeuronProjection, TokenScore


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
    def test_detailed_vectors_support_indexed_watches(self) -> None:
        adapter = object.__new__(LLMDebugAdapter)
        adapter.trace = (
            LayerState(
                Layer("mlp", "mlp", "mlp", 1.0, 0.0),
                (1.0, 2.0),
                (1.5, 1.0),
                "test",
                {
                    "normalized_input": (0.1, 0.2),
                    "mlp_expansion": (3.0, 4.0, 5.0, 6.0),
                },
            ),
            LayerState(
                Layer("next", "next", "next", 1.0, 0.0),
                (1.5, 1.0),
                (1.5, 1.0),
                "next",
            ),
        )
        adapter.layer_index = 1
        self.assertEqual(adapter._evaluate("layernorm[1]"), "+0.20000000")
        self.assertEqual(adapter._evaluate("mlp_expansion[2]"), "+5.00000000")
        self.assertEqual(adapter._vector("delta"), (0.5, -1.0))

    def test_top_mlp_neurons_are_expandable(self) -> None:
        adapter = object.__new__(LLMDebugAdapter)
        adapter.trace = (
            LayerState(
                Layer("transformer.h.0.mlp", "mlp", "mlp", 1.0, 0.0),
                (1.0, 2.0),
                (1.5, 1.0),
                "test",
                {"mlp_expansion": (0.1, 3.0, 0.2)},
            ),
            LayerState(
                Layer("transformer.h.1.attention", "attention", "attention", 1.0, 0.0),
                (1.5, 1.0),
                (1.5, 1.0),
                "next",
            ),
        )
        projection = NeuronProjection(
            index=1,
            activation=3.0,
            direction_norm=2.5,
            vocabulary=(TokenScore(42, " answer", 1.25),),
            contribution=(TokenScore(42, " answer", 3.75),),
        )
        adapter.execution = SimpleNamespace(
            top_mlp_neurons=lambda *_args, **_kwargs: (projection,)
        )
        adapter.layer_index = 1
        adapter.token_index = 3
        adapter.neuron_cache = {}

        top = adapter._variables(adapter.SCOPE_TOP_NEURONS)
        self.assertEqual(top[0]["name"], "neuron[1]")
        neuron = adapter._variables(adapter.NEURON_REFERENCE)
        self.assertEqual(neuron[1]["value"], "+3.00000000")
        vocabulary = adapter._variables(adapter.TOKEN_REFERENCE)
        self.assertEqual(vocabulary[0]["name"], "token[42] ' answer'")
        self.assertEqual(vocabulary[0]["value"], "+1.25000000")
        watch = adapter._evaluate_body("top_neurons")
        self.assertEqual(watch["variablesReference"], adapter.SCOPE_TOP_NEURONS)
        self.assertEqual(watch["namedVariables"], 5)

    def test_setting_watch_expression_records_an_intervention(self) -> None:
        adapter = object.__new__(LLMDebugAdapter)
        adapter.trace = (
            LayerState(
                Layer("transformer.h.0.mlp", "mlp", "mlp", 1.0, 0.0),
                (1.0, 2.0),
                (1.5, 1.0),
                "test",
                {"mlp_expansion": (0.1, 3.0, 0.2)},
            ),
            LayerState(
                Layer("transformer.h.1.attention", "attention", "attention", 1.0, 0.0),
                (1.5, 1.0),
                (1.5, 1.0),
                "next",
            ),
        )
        interventions: list[tuple[object, ...]] = []
        adapter.execution = SimpleNamespace(
            set_intervention=lambda *args: interventions.append(args)
        )
        adapter.layer_index = 1
        adapter.token_index = 3
        adapter.neuron_cache = {}
        adapter._load_token = lambda _position: None

        value = adapter._set_expression("mlp_expansion[1]", 0.0)

        self.assertEqual(value, 0.0)
        self.assertEqual(
            interventions,
            [(3, "transformer.h.0.mlp", "mlp_expansion", 1, 0.0)],
        )

    def test_token_condition_parser(self) -> None:
        self.assertEqual(LLMDebugAdapter._parse_token_condition("token=4"), 4)
        self.assertEqual(LLMDebugAdapter._parse_token_condition("token = 2"), 2)
        self.assertIsNone(LLMDebugAdapter._parse_token_condition("token.position == 1"))
        self.assertIsNone(LLMDebugAdapter._parse_token_condition("token.text == 'x'"))

    def test_mlp_step_into_uses_authored_function_lines(self) -> None:
        adapter = object.__new__(LLMDebugAdapter)
        adapter.trace = (
            LayerState(
                Layer("transformer.h.0.mlp", "mlp", "mlp", 1.0, 0.0),
                (1.0, 2.0),
                (1.5, 2.5),
                "mlp",
                {
                    "normalized_input": (0.1, 0.2),
                    "mlp_expansion_pre_activation": (3.0, 4.0, 5.0),
                    "mlp_expansion": (2.0, 4.0, 5.0),
                    "mlp_projection": (0.5, 0.5),
                },
            ),
        )
        adapter.layer_index = 0
        adapter.inside_mlp = True
        adapter.mlp_step_index = 0

        self.assertEqual(adapter.state.layer.operation, "layernorm")
        self.assertIsNone(adapter.previous_state)
        adapter.mlp_step_index = 1
        self.assertEqual(adapter.state.input_vector, (0.1, 0.2))
        self.assertEqual(adapter.previous_state.output_vector, (0.1, 0.2))
        adapter.mlp_step_index = 3
        self.assertEqual(adapter.state.layer.operation, "projection")
        self.assertEqual(adapter.previous_state.output_vector, (2.0, 4.0, 5.0))

    def test_setting_internal_output_targets_completed_mlp_operation(self) -> None:
        adapter = object.__new__(LLMDebugAdapter)
        adapter.trace = (
            LayerState(
                Layer("transformer.h.0.mlp", "mlp", "mlp", 1.0, 0.0),
                (1.0, 2.0),
                (1.5, 2.5),
                "mlp",
                {"normalized_input": (0.1, 0.2)},
            ),
        )
        adapter.layer_index = 0
        adapter.inside_mlp = True
        adapter.mlp_step_index = 1
        adapter.token_index = 3
        adapter.neuron_cache = {}
        interventions: list[tuple[object, ...]] = []
        adapter.execution = SimpleNamespace(
            set_intervention=lambda *args: interventions.append(args)
        )
        adapter._load_token = lambda _position: None

        adapter._set_expression("output[1]", 9.0)

        self.assertEqual(
            interventions,
            [(3, "transformer.h.0.mlp", "normalized_input", 1, 9.0)],
        )

    def test_attention_step_into_exposes_real_internal_vectors(self) -> None:
        adapter = object.__new__(LLMDebugAdapter)
        adapter.trace = (
            LayerState(
                Layer("transformer.h.0.attention", "attention", "attention", 1.0, 0.0),
                (1.0, 2.0),
                (1.5, 2.5),
                "attention",
                {
                    "normalized_input": (0.1, 0.2),
                    "attention_qkv": (1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
                    "attention_mixing": (0.3, 0.4),
                    "attention_projection": (0.5, 0.5),
                },
            ),
        )
        adapter.layer_index = 0
        adapter.inside_mlp = False
        adapter.inside_attention = True
        adapter.mlp_step_index = 2

        self.assertEqual(adapter.state.layer.operation, "attention_mixing")
        self.assertEqual(adapter.state.input_vector, (1.0, 2.0, 3.0, 4.0, 5.0, 6.0))
        self.assertEqual(adapter.previous_state.layer.operation, "qkv_projection")
        self.assertEqual(adapter._source_expression("qkv[2]"), "attention_qkv[2]")

    def test_python_source_watch_alias_and_slice(self) -> None:
        adapter = object.__new__(LLMDebugAdapter)
        adapter.trace = (
            LayerState(
                Layer("transformer.h.0.mlp", "mlp", "mlp", 1.0, 0.0),
                (1.0, 2.0), (1.5, 2.5), "mlp",
                {
                    "normalized_input": (0.1, 0.2),
                    "mlp_expansion_pre_activation": (1.0, 2.0, 3.0),
                    "mlp_expansion": (1.5, 2.5, 3.5),
                },
            ),
        )
        adapter.layer_index = 0
        adapter.inside_mlp = True
        adapter.inside_attention = False
        adapter.mlp_step_index = 3
        adapter.function_bindings = {
            "mlp": [
                {"name": "norm", "operation": "layernorm", "line": 1},
                {"name": "features", "operation": "expansion", "line": 2},
                {"name": "features", "operation": "gelu", "line": 3},
            ]
        }
        self.assertEqual(adapter._evaluate("features[1]"), "+2.50000000")
        self.assertEqual(adapter._evaluate("features[:2]"), repr((1.5, 2.5)))
        self.assertEqual(adapter._evaluate("x[1]"), "+2.00000000")
        self.assertEqual(adapter._plot_vector("features[:2]"), (1.5, 2.5))
        self.assertEqual(adapter._plot_vector("x"), (1.0, 2.0))
        self.assertEqual(adapter._plot_vector("[1, 2.5, -3]"), (1.0, 2.5, -3.0))

    def test_logit_lens_snapshot_contains_ranked_tokens(self) -> None:
        adapter = object.__new__(LLMDebugAdapter)
        adapter.trace = (
            LayerState(
                Layer("transformer.h.0.mlp", "mlp", "mlp", 1.0, 0.0),
                (1.0, 2.0), (1.5, 2.5), "mlp",
            ),
        )
        adapter.layer_index = 0
        adapter.token_index = 0
        adapter.tokens = (" test",)
        adapter.inside_mlp = False
        adapter.inside_attention = False
        adapter.execution = SimpleNamespace(
            logit_lens=lambda vector: (
                LogitLensToken(42, " answer", 8.5, 0.25),
            )
        )

        result = adapter._lens_snapshot("residual_stream")

        self.assertEqual(result["dimensions"], 2)
        self.assertEqual(result["tokens"][0]["text"], " answer")
        self.assertEqual(result["tokens"][0]["probability"], 0.25)

    def test_continue_stops_at_a_source_breakpoint(self) -> None:
        requests = [
            {"seq": 1, "type": "request", "command": "initialize", "arguments": {}},
            {
                "seq": 2,
                "type": "request",
                "command": "launch",
                "arguments": {"stopOnEntry": False, "defaultToken": 0},
            },
            {
                "seq": 3,
                "type": "request",
                "command": "setBreakpoints",
                "arguments": {"breakpoints": [{"line": 10}]},
            },
            {"seq": 4, "type": "request", "command": "configurationDone", "arguments": {}},
            {"seq": 5, "type": "request", "command": "stackTrace", "arguments": {}},
            {"seq": 6, "type": "request", "command": "disconnect", "arguments": {}},
        ]
        process = subprocess.run(
            [sys.executable, "-m", "llmdbg.dap"],
            input=b"".join(frame(request) for request in requests),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        messages = parse_frames(process.stdout)
        responses = {
            message.get("request_seq"): message
            for message in messages
            if message["type"] == "response"
        }
        self.assertTrue(responses[3]["body"]["breakpoints"][0]["verified"])
        stopped = [message for message in messages if message.get("event") == "stopped"]
        self.assertEqual(stopped[0]["body"]["reason"], "breakpoint")
        self.assertEqual(responses[5]["body"]["stackFrames"][0]["line"], 10)

    def test_continue_stops_at_an_internal_mlp_breakpoint(self) -> None:
        adapter = object.__new__(LLMDebugAdapter)
        adapter.tokens = ("a", "b")
        adapter.source_lines = (10, 20)
        adapter.step_paths = ("token_embedding", "transformer.h.0.mlp")
        adapter.mlp_source_lines = (24, 25, 26, 27, 28)
        adapter.attention_source_lines = ()
        adapter.breakpoints = (SourceBreakpoint(27, 1),)
        adapter.token_index = 0
        adapter.layer_index = 0
        adapter.inside_mlp = False
        adapter.inside_attention = False
        adapter.mlp_step_index = 0
        adapter._load_token = lambda position: setattr(adapter, "token_index", position)
        stops: list[str] = []
        adapter.stop = stops.append

        adapter._continue_to_breakpoint(include_current=True, before_start=True)

        self.assertEqual(stops, ["breakpoint"])
        self.assertEqual((adapter.token_index, adapter.layer_index), (1, 1))
        self.assertTrue(adapter.inside_mlp)
        self.assertFalse(adapter.inside_attention)
        self.assertEqual(adapter.mlp_step_index, 3)

    def test_continue_from_internal_breakpoint_reaches_later_internal_line(self) -> None:
        adapter = object.__new__(LLMDebugAdapter)
        adapter.tokens = ("a",)
        adapter.source_lines = (20,)
        adapter.step_paths = ("transformer.h.0.mlp",)
        adapter.mlp_source_lines = (24, 25, 26, 27, 28)
        adapter.attention_source_lines = ()
        adapter.breakpoints = (SourceBreakpoint(27), SourceBreakpoint(28))
        adapter.token_index = 0
        adapter.layer_index = 0
        adapter.inside_mlp = True
        adapter.inside_attention = False
        adapter.mlp_step_index = 3
        adapter._load_token = lambda position: setattr(adapter, "token_index", position)
        stops: list[str] = []
        adapter.stop = stops.append

        adapter._continue_to_breakpoint(include_current=False)

        self.assertEqual(stops, ["breakpoint"])
        self.assertTrue(adapter.inside_mlp)
        self.assertEqual(adapter.mlp_step_index, 4)

    def test_step_over_outer_call_honors_internal_breakpoint(self) -> None:
        adapter = object.__new__(LLMDebugAdapter)
        adapter.tokens = ("a",)
        adapter.source_lines = (35,)
        adapter.step_paths = ("transformer.h.2.mlp",)
        adapter.mlp_source_lines = (24, 25, 26, 27, 28)
        adapter.attention_source_lines = ()
        adapter.breakpoints = (SourceBreakpoint(27),)
        adapter.token_index = 0
        adapter.layer_index = 0
        adapter.inside_mlp = False
        adapter.inside_attention = False
        adapter.mlp_step_index = 0
        stops: list[str] = []
        adapter.stop = stops.append

        self.assertTrue(adapter._stop_at_internal_breakpoint())

        self.assertEqual(stops, ["breakpoint"])
        self.assertTrue(adapter.inside_mlp)
        self.assertEqual(adapter.mlp_step_index, 3)

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
            {"seq": 11, "type": "request", "command": "evaluate", "arguments": {"expression": "output[2]"}},
            {"seq": 12, "type": "request", "command": "evaluate", "arguments": {"expression": "residual_stream[2]"}},
            {"seq": 13, "type": "request", "command": "evaluate", "arguments": {"expression": "delta[2]"}},
            {"seq": 14, "type": "request", "command": "llmdbg/vector", "arguments": {}},
            {"seq": 15, "type": "request", "command": "disconnect", "arguments": {}},
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
        self.assertEqual(responses[11]["body"]["result"], responses[12]["body"]["result"])
        self.assertRegex(responses[13]["body"]["result"], r"^[+-]\d+\.\d{8}$")
        snapshot = responses[14]["body"]
        self.assertEqual(snapshot["layer"], "attention_0")
        self.assertEqual(snapshot["dimensions"], len(snapshot["vectors"]["residual_stream"]))
        self.assertEqual(set(snapshot["vectors"]), {"input", "residual_stream", "delta"})
        virtual_source = responses[10]["body"]["content"]
        self.assertIn('model = "built-in-demo"', virtual_source)
        self.assertIn('submodule = "token_embedding"', virtual_source)
        stopped = [message for message in messages if message.get("event") == "stopped"]
        self.assertEqual([event["body"]["reason"] for event in stopped], ["entry", "step"])

    def test_config_file_is_used_as_the_debug_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "gpt.llmdbg"
            config.write_text('model = "openai-community/gpt2"\n', encoding="utf-8")
            adapter = object.__new__(LLMDebugAdapter)
            adapter.source_name = "model.execution.llmdbg"
            adapter.source_path = None
            adapter.execution = SimpleNamespace(
                runtime=SimpleNamespace(
                    model=SimpleNamespace(
                        transformer=SimpleNamespace(h=[object()])
                    )
                )
            )
            adapter._configure_model_source(
                {
                    "configPath": str(config),
                    "stepPaths": ["token_embedding", "transformer.h.0.attention"],
                    "stepLines": {
                        "token_embedding": 2,
                        "transformer.h.0.attention": 3,
                    },
                }
            )

        self.assertEqual(adapter.source_name, "gpt.llmdbg")
        self.assertEqual(adapter.source_path, str(config))
        self.assertEqual(adapter.source_lines, (2, 3))


if __name__ == "__main__":
    unittest.main()
