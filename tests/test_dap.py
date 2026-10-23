from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from llmdbg.dap import LLMDebugAdapter, SourceBreakpoint
from llmdbg.activation_scales import (
    attention_channel_activation_scale,
    mlp_activation_scale,
)
from llmdbg.demo_model import Layer, LayerState
from llmdbg.model_runtime import NeuronProjection, TokenScore


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
    def test_lens_before_first_breakpoint_has_a_clear_error(self) -> None:
        adapter = object.__new__(LLMDebugAdapter)
        adapter.trace = ()
        with self.assertRaisesRegex(ValueError, "first breakpoint"):
            adapter._neuron_lens_snapshot("residual-pre")

    def test_repeatable_nops_are_inserted_in_authored_execution_order(self) -> None:
        adapter = object.__new__(LLMDebugAdapter)
        adapter.execution = SimpleNamespace(
            runtime=SimpleNamespace(
                layers=(), architecture="gpt2", final_norm_name="transformer.ln_f"
            )
        )

        adapter._configure_model_source(
            {
                "configPath": "/tmp/nop.llmdbg.py",
                "pythonProgram": True,
                "callLines": {
                    "token_embedding": 10,
                    "final_layernorm": 20,
                    "nop": [21, 24],
                    "vocabulary_projection": 22,
                    "top_tokens": 23,
                },
                "functions": {},
            }
        )

        self.assertEqual(
            adapter.step_paths,
            (
                "token_embedding", "transformer.ln_f", "nop:0",
                "vocabulary_projection", "predictions", "nop:1",
            ),
        )
        self.assertEqual(adapter.source_lines, (10, 20, 21, 22, 23, 24))

    def test_nops_inside_layer_loop_expand_for_every_transformer_layer(self) -> None:
        adapter = object.__new__(LLMDebugAdapter)
        adapter.execution = SimpleNamespace(
            runtime=SimpleNamespace(
                layers=(object(), object()), architecture="gpt2",
                final_norm_name="transformer.ln_f",
                layer_kind=lambda _index: "attention",
                state_name=lambda index, kind: f"transformer.h.{index}.{kind}",
            )
        )

        adapter._configure_model_source(
            {
                "configPath": "/tmp/loop-nop.llmdbg.py",
                "pythonProgram": True,
                "callLines": {
                    "token_embedding": 10,
                    "attention": 20,
                    "mlp": 22,
                    "nop": [19, 21, 23],
                    "final_layernorm": 30,
                },
                "functions": {},
            }
        )

        self.assertEqual(
            adapter.step_paths,
            (
                "token_embedding",
                "nop:0:0", "transformer.h.0.attention", "nop:1:0",
                "transformer.h.0.mlp", "nop:2:0",
                "nop:0:1", "transformer.h.1.attention", "nop:1:1",
                "transformer.h.1.mlp", "nop:2:1",
                "transformer.ln_f",
            ),
        )

    def test_qwen_source_dispatches_each_hybrid_mixer(self) -> None:
        arguments = {
            "configPath": "/tmp/Qwen35-0.8B.llmdbg.py",
            "pythonProgram": True,
            "callLines": {
                "token_embedding": 45, "full_attention": 48,
                "gated_linear_attention": 50, "mlp": 51, "final_layernorm": 52,
                "vocabulary_projection": 53, "top_tokens": 54, "end": 55,
            },
            "functions": {
                "full_attention": {
                    "input_layernorm": 17, "full_attention": 18,
                    "output_projection": 19, "residual_add": 20,
                },
                "gated_linear_attention": {
                    "input_layernorm": 25, "gated_delta": 26,
                    "output_projection": 27, "residual_add": 28,
                },
                "mlp": {
                    "post_attention_layernorm": 33, "gate_projection": 34,
                    "silu": 35, "up_projection": 36, "gated_multiply": 37,
                    "down_projection": 38, "residual_add": 39,
                },
            },
            "functionBindings": {},
        }
        runtime = SimpleNamespace(
            layers=[object(), object(), object(), object()],
            architecture="qwen3_5",
            layer_kind=lambda index: "full_attention" if index == 3 else "gated_linear_attention",
            state_name=lambda index, kind: f"model.layers.{index}.{kind}",
            final_norm_name="model.norm",
        )
        adapter = object.__new__(LLMDebugAdapter)
        adapter.execution = SimpleNamespace(runtime=runtime)
        adapter._configure_model_source(arguments)

        self.assertEqual(
            adapter.step_paths[:5],
            (
                "token_embedding",
                "model.layers.0.gated_linear_attention", "model.layers.0.mlp",
                "model.layers.1.gated_linear_attention", "model.layers.1.mlp",
            ),
        )
        self.assertIn("model.layers.3.full_attention", adapter.step_paths)
        self.assertEqual(
            adapter.step_paths[-3:],
            ("vocabulary_projection", "predictions", "end"),
        )
        self.assertEqual(len(adapter.function_source_lines["gated_linear_attention"]), 4)
        self.assertEqual(len(adapter.function_source_lines["full_attention"]), 4)
        self.assertEqual(len(adapter.function_source_lines["mlp"]), 7)

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

    def test_predictions_are_available_at_terminal_line(self) -> None:
        adapter = object.__new__(LLMDebugAdapter)
        adapter.step_paths = ("vocabulary_projection", "predictions", "end")
        adapter.layer_index = 2
        adapter.predictions = (
            SimpleNamespace(token_id=7, text=" Paris", logit=9.25, probability=0.42),
        )

        watch = adapter._evaluate_body("predictions")
        first = adapter._evaluate_body("predictions[0]")
        variables = adapter._variables(LLMDebugAdapter.SCOPE_PREDICTIONS)

        self.assertEqual(watch["variablesReference"], LLMDebugAdapter.SCOPE_PREDICTIONS)
        self.assertIn("Paris", first["result"])
        self.assertEqual(variables[0]["evaluateName"], "predictions[0]")
        self.assertIn("42.00%", variables[0]["value"])

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

    def test_patch_vector_replays_active_target_with_full_replacement(self) -> None:
        adapter = object.__new__(LLMDebugAdapter)
        patches = []
        adapter.execution = SimpleNamespace(
            patch_vector=lambda *args: patches.append(args),
            device="cpu",
            dtype="float32",
        )
        adapter.token_index = 3
        adapter.layer_index = 7
        adapter.tokens = ("a", "b", "c", "d")
        adapter.predictions = ()
        adapter.neuron_cache = {}
        adapter.lens_snapshot_cache = {}
        adapter.measurement_vectors = {"target": ("target", (1.0, 2.0))}
        adapter._neuron_lens_snapshot = lambda **_kwargs: {
            "saved": False,
            "measurementId": "target",
            "layer": "model.layers.2.full_attention",
        }
        adapter._load_token = lambda _position: None
        adapter._write_vector_patch_log = lambda **_kwargs: {
            "note": "/tmp/note.md", "json": "/tmp/note.json"
        }
        adapter.send_event = lambda *_args, **_kwargs: None

        result = adapter._patch_vector("residual-pre", (4.0, 5.0), "source")

        self.assertEqual(
            patches,
            [(3, "model.layers.2.full_attention", "input", (4.0, 5.0))],
        )
        self.assertTrue(result["patched"])

    def test_breakpoint_condition_parser_supports_layer_and_combinations(self) -> None:
        parse = LLMDebugAdapter._parse_breakpoint_condition
        self.assertEqual(parse("layer=6"), (None, ("==", 6)))
        self.assertEqual(parse("token=3, layer=6"), (("==", 3), ("==", 6)))
        self.assertEqual(
            parse("layer >= 6 and token < 3"), (("<", 3), (">=", 6))
        )
        self.assertIsNone(parse("l=6"))
        self.assertIsNone(parse("layer=6, layer=7"))

    def test_breakpoint_comparison_filters_match_actual_token_and_layer(self) -> None:
        adapter = object.__new__(LLMDebugAdapter)
        adapter.step_paths = ("model.layers.14.mlp", "model.layers.15.mlp")
        breakpoint = SourceBreakpoint(
            20,
            token_position=2,
            transformer_layer=15,
            token_operator=">=",
            layer_operator=">=",
        )

        self.assertFalse(adapter._breakpoint_matches(breakpoint, 1, 1))
        self.assertFalse(adapter._breakpoint_matches(breakpoint, 2, 0))
        self.assertTrue(adapter._breakpoint_matches(breakpoint, 2, 1))

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
        adapter.model_name = "openai-community/gpt2"
        adapter.prompt = "hello"
        adapter.tokens = ("hello",)
        adapter.token_index = 0
        adapter.step_paths = ("transformer.h.0.mlp",)
        adapter.execution = None
        self.assertEqual(adapter._evaluate("features[1]"), "+2.50000000")
        self.assertEqual(adapter._evaluate("features[:2]"), repr((1.5, 2.5)))
        self.assertEqual(adapter._evaluate("x[1]"), "+2.00000000")
        self.assertEqual(adapter._plot_vector("features[:2]"), (1.5, 2.5))
        self.assertEqual(adapter._plot_vector("x"), (1.0, 2.0))
        self.assertEqual(adapter._plot_vector("[1, 2.5, -3]"), (1.0, 2.5, -3.0))
        locals_by_name = {item["name"]: item for item in adapter._local_variables()}
        self.assertEqual(locals_by_name["layer"]["evaluateName"], "layer")
        self.assertIn("x", locals_by_name)
        self.assertIn("norm", locals_by_name)
        self.assertIn("features", locals_by_name)

    def test_neuron_lens_snapshot_contains_direction_tokens(self) -> None:
        adapter = object.__new__(LLMDebugAdapter)
        adapter.trace = (
            LayerState(
                Layer("transformer.h.0.mlp", "mlp", "mlp", 1.0, 0.0),
                (1.0, 2.0), (1.5, 2.5), "mlp",
                {"mlp_expansion": (0.1, 3.0)},
            ),
            LayerState(
                Layer("transformer.h.1.attention", "attention", "attention", 1.0, 0.0),
                (1.5, 2.5), (1.5, 2.5), "attention",
            ),
        )
        adapter.layer_index = 1
        adapter.token_index = 0
        adapter.tokens = (" test",)
        adapter.inside_mlp = False
        adapter.inside_attention = False
        projection = NeuronProjection(
            index=1,
            activation=3.0,
            direction_norm=2.5,
            vocabulary=(TokenScore(42, " answer", 8.5),),
            contribution=(),
            j_lens=(TokenScore(21, " jacobian", 7.0),),
            r_lens=(TokenScore(84, " transported", 6.25),),
        )
        adapter.execution = SimpleNamespace(
            top_mlp_neurons=lambda *_args, **_kwargs: (projection,)
        )
        adapter.neuron_cache = {}

        result = adapter._neuron_lens_snapshot()

        self.assertEqual(result["neurons"][0]["index"], 1)
        self.assertEqual(result["neurons"][0]["tokens"][0]["text"], " answer")
        self.assertEqual(result["neurons"][0]["tokens"][0]["logit"], 8.5)
        self.assertEqual(result["neurons"][0]["logitTokens"][0]["text"], " answer")
        self.assertEqual(result["neurons"][0]["jLensTokens"][0]["text"], " jacobian")
        self.assertEqual(result["neurons"][0]["rLensTokens"][0]["text"], " transported")

    def test_vector_lens_snapshot_supports_mlp_output_and_residual_stream(self) -> None:
        adapter = object.__new__(LLMDebugAdapter)
        adapter.trace = (
            LayerState(
                Layer("transformer.h.0.mlp", "mlp", "mlp", 1.0, 0.0),
                (1.0, 2.0), (1.5, 2.5), "mlp",
                {"mlp_projection": (0.5, 0.5), "mlp_expansion": (0.1, 3.0)},
            ),
            LayerState(
                Layer("transformer.h.1.attention", "attention", "attention", 1.0, 0.0),
                (1.5, 2.5), (1.5, 2.5), "attention",
            ),
        )
        adapter.layer_index = 1
        adapter.token_index = 0
        adapter.tokens = (" test",)
        adapter.inside_mlp = False
        adapter.inside_attention = False
        calls = []
        projection = SimpleNamespace(
            norm=3.0,
            vocabulary=(TokenScore(1, " direct", 4.0),),
            j_lens=(TokenScore(2, " jacobian", 3.0),),
            r_lens=(TokenScore(3, " relp", 2.0),),
        )
        adapter.execution = SimpleNamespace(
            vector_lens=lambda layer, vector, **kwargs: calls.append(
                (layer, vector, kwargs)
            ) or projection
        )

        mlp = adapter._neuron_lens_snapshot(
            view="mlp-output", include_j_lens=True, include_r_lens=True
        )
        residual = adapter._neuron_lens_snapshot(view="residual-stream")

        self.assertEqual(calls[0][1], (0.5, 0.5))
        self.assertEqual(calls[1][1], (1.5, 2.5))
        self.assertEqual(mlp["jLensTokens"][0]["text"], " jacobian")
        self.assertEqual(mlp["rLensTokens"][0]["text"], " relp")
        self.assertEqual(residual["vectorLabel"], "Residual Lens · Pre")

    def test_lenses_at_residual_add_use_neurons_update_and_pre_add_residual(self) -> None:
        adapter = object.__new__(LLMDebugAdapter)
        adapter.trace = (
            LayerState(
                Layer("model.layers.0.mlp", "mlp", "mlp", 1.0, 0.0),
                (1.0, 2.0), (1.5, 2.5), "mlp",
                {"mlp_projection": (0.5, 0.5), "mlp_expansion": (-3.0, 0.2)},
            ),
        )
        adapter.layer_index = 0
        adapter.token_index = 0
        adapter.tokens = (" test",)
        adapter.inside_mlp = True
        adapter.inside_attention = False
        adapter.mlp_step_index = 6
        adapter.model_name = "Qwen/Qwen3.5-0.8B"
        adapter.neuron_cache = {}
        calls = []
        neuron = NeuronProjection(
            index=0, activation=-3.0, direction_norm=1.0,
            vocabulary=(TokenScore(1, " neuron", 1.0),), contribution=(),
        )
        vector_projection = SimpleNamespace(
            norm=1.0, vocabulary=(TokenScore(2, " vector", 1.0),),
            j_lens=(), r_lens=(),
        )
        adapter.execution = SimpleNamespace(
            top_mlp_neurons=lambda *_args, **_kwargs: (neuron,),
            vector_lens=lambda layer, vector, **kwargs: calls.append(vector) or vector_projection,
        )

        neuron_result = adapter._neuron_lens_snapshot()
        adapter._neuron_lens_snapshot(view="mlp-output")
        residual = adapter._neuron_lens_snapshot(view="residual-stream")

        self.assertEqual(neuron_result["neurons"][0]["activation"], -3.0)
        self.assertEqual(calls, [(0.5, 0.5), (1.0, 2.0)])
        self.assertEqual(residual["vectorLabel"], "Residual Lens · Pre")

    def test_residual_lens_is_available_before_attention_executes(self) -> None:
        adapter = object.__new__(LLMDebugAdapter)
        adapter.trace = (
            LayerState(
                Layer("model.layers.0.gated_linear_attention", "attention", "attention", 1.0, 0.0),
                (1.0, 2.0), (1.5, 2.5), "attention",
            ),
        )
        adapter.layer_index = 0
        adapter.token_index = 0
        adapter.tokens = (" test",)
        adapter.inside_mlp = False
        adapter.inside_attention = False
        calls = []
        projection = SimpleNamespace(
            norm=3.0, vocabulary=(TokenScore(1, " direct", 4.0),),
            j_lens=(), r_lens=(),
        )
        adapter.execution = SimpleNamespace(
            vector_lens=lambda layer, vector, **kwargs: calls.append(
                (layer, vector, kwargs)
            ) or projection
        )

        residual = adapter._neuron_lens_snapshot(view="residual-stream")

        self.assertEqual(calls[0][0], "model.layers.0.gated_linear_attention")
        self.assertEqual(calls[0][1], (1.0, 2.0))
        self.assertEqual(residual["vectorLabel"], "Residual Lens · Pre")

    def test_residual_output_lens_is_available_at_residual_add(self) -> None:
        adapter = object.__new__(LLMDebugAdapter)
        adapter.trace = (
            LayerState(
                Layer("model.layers.2.mlp", "mlp", "mlp", 1.0, 0.0),
                (1.0, 2.0), (1.5, 2.5), "mlp",
                {
                    "normalized_input": (1.0, 2.0),
                    "mlp_gate_pre_activation": (1.0, 2.0),
                    "mlp_gate_activation": (1.0, 2.0),
                    "mlp_up_projection": (1.0, 2.0),
                    "mlp_expansion": (1.0, 2.0),
                    "mlp_projection": (0.5, 0.5),
                },
            ),
        )
        adapter.layer_index = 0
        adapter.token_index = 0
        adapter.tokens = (" test",)
        adapter.inside_mlp = True
        adapter.inside_attention = False
        adapter.mlp_step_index = len(adapter.QWEN_MLP_STEPS) - 1
        calls = []
        projection = SimpleNamespace(
            norm=1.0, vocabulary=(), j_lens=(), r_lens=(),
        )
        adapter.execution = SimpleNamespace(
            vector_lens=lambda layer, vector, **kwargs: calls.append(
                (layer, vector, kwargs)
            ) or projection
        )

        output = adapter._neuron_lens_snapshot(view="residual-output")

        self.assertEqual(calls[0][1], (1.5, 2.5))
        self.assertEqual(output["vectorLabel"], "Residual Lens · Post")
        self.assertEqual(output["transformerLayer"], 2)

    def test_residual_mid_lens_is_available_at_nop_after_attention(self) -> None:
        attention = LayerState(
            Layer(
                "model.layers.3.gated_linear_attention", "attention", "attention",
                1.0, 0.0,
            ),
            (1.0, 2.0),
            (1.5, 2.5),
            "attention",
            {"attention_projection": (0.5, 0.5)},
        )
        nop = LayerState(
            Layer("nop:0", "nop", "no operation", 1.0, 0.0),
            attention.output_vector,
            attention.output_vector,
            "no operation",
        )
        adapter = object.__new__(LLMDebugAdapter)
        adapter.trace = (attention, nop)
        adapter.layer_index = 1
        adapter.token_index = 0
        adapter.tokens = (" test",)
        adapter.inside_mlp = False
        adapter.inside_attention = False
        calls = []
        projection = SimpleNamespace(
            norm=1.0, vocabulary=(), j_lens=(), r_lens=(),
        )
        adapter.execution = SimpleNamespace(
            vector_lens=lambda layer, vector, **kwargs: calls.append(
                (layer, vector, kwargs)
            ) or projection
        )

        result = adapter._neuron_lens_snapshot(view="residual-mid")
        pre = adapter._neuron_lens_snapshot(view="residual-pre")

        self.assertEqual(
            calls[0][0], "model.layers.3.gated_linear_attention"
        )
        self.assertEqual(calls[0][1], (1.5, 2.5))
        self.assertEqual(result["vectorLabel"], "Residual Lens · Mid")
        self.assertEqual(result["transformerLayer"], 3)
        self.assertEqual(result["residualGeometry"]["transition"], "attention update")
        self.assertAlmostEqual(result["residualGeometry"]["deltaNorm"], 2 ** -0.5)
        self.assertAlmostEqual(result["residualGeometry"]["deltaRatio"], 10 ** -0.5)
        self.assertEqual(calls[1][1], (1.0, 2.0))
        self.assertEqual(pre["vectorLabel"], "Residual Lens · Pre")
        self.assertEqual(pre["transformerLayer"], 3)

    def test_residual_pre_and_post_share_block_identity_at_post_nop(self) -> None:
        attention = LayerState(
            Layer("model.layers.0.full_attention", "attention", "attention", 1.0, 0.0),
            (1.0, 2.0), (1.5, 2.5), "attention",
        )
        mlp = LayerState(
            Layer("model.layers.0.mlp", "mlp", "mlp", 1.0, 0.0),
            (1.5, 2.5), (2.0, 3.0), "mlp",
        )
        post_nop = LayerState(
            Layer("nop:2:0", "nop", "no operation", 1.0, 0.0),
            mlp.output_vector, mlp.output_vector, "no operation",
        )
        next_attention = LayerState(
            Layer("model.layers.1.full_attention", "attention", "attention", 1.0, 0.0),
            (2.0, 3.0), (2.5, 3.5), "attention",
        )
        adapter = object.__new__(LLMDebugAdapter)
        adapter.trace = (attention, mlp, post_nop, next_attention)
        adapter.layer_index = 2
        adapter.token_index = 0
        adapter.tokens = (" test",)
        adapter.inside_mlp = False
        adapter.inside_attention = False
        projection = SimpleNamespace(
            norm=1.0, vocabulary=(), j_lens=(), r_lens=(),
        )
        adapter.execution = SimpleNamespace(
            vector_lens=lambda *_args, **_kwargs: projection
        )

        pre = adapter._neuron_lens_snapshot(view="residual-pre")
        post = adapter._neuron_lens_snapshot(view="residual-post")

        self.assertEqual(pre["transformerLayer"], 0)
        self.assertEqual(post["transformerLayer"], 0)
        self.assertEqual(pre["layer"], "model.layers.0.full_attention")
        self.assertEqual(post["layer"], "model.layers.0.mlp")
        self.assertEqual(adapter._current_lens_phase(), 4)
        self.assertTrue(adapter._lens_result_is_saved("attention-output"))
        self.assertTrue(adapter._lens_result_is_saved("residual-mid"))
        self.assertTrue(adapter._lens_result_is_saved("mlp-output"))
        self.assertFalse(adapter._lens_result_is_saved("residual-post"))
        self.assertTrue(pre["saved"])
        self.assertFalse(post["saved"])
        self.assertIsNone(pre["residualGeometry"])
        self.assertEqual(post["residualGeometry"]["transition"], "MLP update")
        self.assertAlmostEqual(post["residualGeometry"]["deltaNorm"], 2 ** -0.5)

    def test_lenses_find_latest_completed_mlp_when_continue_skips_intermediate_stops(
        self,
    ) -> None:
        attention = LayerState(
            Layer("model.layers.2.full_attention", "attention", "attention", 1.0, 0.0),
            (1.0, 2.0), (1.5, 2.5), "attention",
        )
        mid_nop = LayerState(
            Layer("nop:1:2", "nop", "no operation", 1.0, 0.0),
            attention.output_vector, attention.output_vector, "no operation",
        )
        mlp = LayerState(
            Layer("model.layers.2.mlp", "mlp", "mlp", 1.0, 0.0),
            (1.5, 2.5), (2.0, 3.0), "mlp",
            {"mlp_expansion": (0.4, -0.2), "mlp_projection": (0.5, 0.5)},
        )
        post_nop = LayerState(
            Layer("nop:2:2", "nop", "no operation", 1.0, 0.0),
            mlp.output_vector, mlp.output_vector, "no operation",
        )
        next_pre_nop = LayerState(
            Layer("nop:0:3", "nop", "no operation", 1.0, 0.0),
            mlp.output_vector, mlp.output_vector, "no operation",
        )
        next_attention = LayerState(
            Layer(
                "model.layers.3.full_attention", "attention", "attention", 1.0, 0.0
            ),
            mlp.output_vector, (2.5, 3.5), "attention",
        )
        adapter = object.__new__(LLMDebugAdapter)
        adapter.trace = (
            attention, mid_nop, mlp, post_nop, next_pre_nop, next_attention,
        )
        adapter.layer_index = 4
        adapter.token_index = 0
        adapter.tokens = (" test",)
        adapter.inside_mlp = False
        adapter.inside_attention = False
        adapter.neuron_cache = {}
        adapter.model_name = "unsupported/test-model"
        calls = []
        neuron = NeuronProjection(
            index=7, activation=0.4, direction_norm=0.5,
            vocabulary=(), contribution=(),
        )
        vector_projection = SimpleNamespace(
            norm=1.0, vocabulary=(), j_lens=(), r_lens=(),
        )
        adapter.execution = SimpleNamespace(
            top_mlp_neurons=lambda layer, activation, **_kwargs: calls.append(
                ("neurons", layer, activation)
            ) or (neuron,),
            vector_lens=lambda layer, vector, **_kwargs: calls.append(
                ("output", layer, vector)
            ) or vector_projection,
        )

        neurons = adapter._neuron_lens_snapshot(view="neurons")
        output = adapter._neuron_lens_snapshot(view="mlp-output")

        self.assertEqual(neurons["transformerLayer"], 2)
        self.assertEqual(output["transformerLayer"], 2)
        self.assertTrue(neurons["saved"])
        self.assertTrue(output["saved"])
        self.assertEqual(calls[0], ("neurons", "model.layers.2.mlp", (0.4, -0.2)))
        self.assertEqual(calls[1], ("output", "model.layers.2.mlp", (0.5, 0.5)))

    def test_breakpoint_conditions_scope_exposes_token_and_transformer_layer(self) -> None:
        adapter = object.__new__(LLMDebugAdapter)
        adapter.trace = (
            LayerState(
                Layer("model.layers.7.mlp", "mlp", "mlp", 1.0, 0.0),
                (1.0,), (2.0,), "mlp",
            ),
        )
        adapter.layer_index = 0
        adapter.token_index = 3

        variables = adapter._variables(adapter.SCOPE_BREAKPOINT_CONDITIONS)

        self.assertEqual([(item["name"], item["value"]) for item in variables], [
            ("token", "3"), ("layer", "7"),
        ])

    def test_loop_nop_retains_transformer_layer_for_conditions(self) -> None:
        adapter = object.__new__(LLMDebugAdapter)
        adapter.trace = (
            LayerState(
                Layer("nop:1:6", "nop", "no operation", 1.0, 0.0),
                (1.0,), (1.0,), "no operation",
            ),
        )
        adapter.layer_index = 0
        adapter.token_index = 4

        variables = adapter._variables(adapter.SCOPE_BREAKPOINT_CONDITIONS)

        self.assertEqual([(item["name"], item["value"]) for item in variables], [
            ("token", "4"), ("layer", "6"),
        ])
        self.assertIsNone(LLMDebugAdapter._transformer_layer_index("nop:2"))

    def test_source_loop_token_layer_and_block_are_real_locals(self) -> None:
        adapter = object.__new__(LLMDebugAdapter)
        adapter.trace = (
            LayerState(
                Layer("nop:1:6", "nop", "no operation", 1.0, 0.0),
                (1.0,), (1.0,), "no operation",
            ),
        )
        adapter.step_paths = ("nop:1:6",)
        adapter.layer_index = 0
        adapter.token_index = 4
        adapter.tokens = tuple(f" token-{index}" for index in range(5))
        adapter.model_name = "test/model"
        adapter.prompt = "test prompt"
        adapter.execution = None
        adapter.inside_mlp = False
        adapter.inside_attention = False
        adapter.predictions = ()

        variables = adapter._local_variables()
        by_name = {variable["name"]: variable for variable in variables}

        self.assertEqual(by_name["token"]["value"], "4")
        self.assertEqual(by_name["layer"]["value"], "6")
        self.assertEqual(by_name["block"]["value"], "model.layers[6]")
        self.assertEqual(adapter._evaluate("token"), "4")
        self.assertEqual(adapter._evaluate("layer"), "6")
        self.assertEqual(adapter._evaluate("block"), "model.layers[6]")

    def test_attention_output_lens_is_available_at_following_mlp(self) -> None:
        adapter = object.__new__(LLMDebugAdapter)
        adapter.trace = (
            LayerState(
                Layer("model.layers.0.full_attention", "attention", "attention", 1.0, 0.0),
                (1.0, 2.0), (1.5, 2.5), "attention",
                {"attention_projection": (0.5, 0.5)},
            ),
            LayerState(
                Layer("model.layers.0.mlp", "mlp", "mlp", 1.0, 0.0),
                (1.5, 2.5), (2.0, 3.0), "mlp",
            ),
        )
        adapter.layer_index = 1
        adapter.token_index = 0
        adapter.tokens = (" test",)
        adapter.inside_mlp = False
        adapter.inside_attention = False
        calls = []
        projection = SimpleNamespace(
            norm=1.0, vocabulary=(TokenScore(1, " attention", 4.0),),
            j_lens=(), r_lens=(),
        )
        adapter.execution = SimpleNamespace(
            vector_lens=lambda layer, vector, **kwargs: calls.append(
                (layer, vector, kwargs)
            ) or projection
        )

        result = adapter._neuron_lens_snapshot(view="attention-output")

        self.assertEqual(calls[0][0], "model.layers.0.full_attention")
        self.assertEqual(calls[0][1], (0.5, 0.5))
        self.assertEqual(result["vectorLabel"], "Attention output")

    def test_attention_neuron_lens_projects_mixed_channels(self) -> None:
        adapter = object.__new__(LLMDebugAdapter)
        adapter.trace = (
            LayerState(
                Layer("model.layers.0.full_attention", "attention", "attention", 1.0, 0.0),
                (1.0, 2.0), (1.5, 2.5), "attention",
                {
                    "full_attention_mixing": (-3.0, 0.2),
                    "attention_projection": (0.5, 0.5),
                },
            ),
            LayerState(
                Layer("model.layers.0.mlp", "mlp", "mlp", 1.0, 0.0),
                (1.5, 2.5), (2.0, 3.0), "mlp",
            ),
        )
        adapter.layer_index = 1
        adapter.token_index = 0
        adapter.tokens = (" test",)
        adapter.inside_mlp = False
        adapter.inside_attention = False
        adapter.neuron_cache = {}
        adapter.model_name = "Qwen/Qwen3.5-2B"
        channel = NeuronProjection(
            index=0, activation=-3.0, direction_norm=2.0,
            vocabulary=(TokenScore(1, " channel", 4.0),), contribution=(),
        )
        calls = []
        adapter.execution = SimpleNamespace(
            top_attention_channels=lambda layer, activation, **kwargs: calls.append(
                (layer, activation, kwargs)
            ) or (channel,)
        )

        result = adapter._neuron_lens_snapshot(view="attention-neurons")

        self.assertEqual(calls[0][0], "model.layers.0.full_attention")
        self.assertEqual(calls[0][1], (-3.0, 0.2))
        self.assertEqual(result["componentLabel"], "channel")
        self.assertEqual(result["neurons"][0]["activation"], -3.0)
        scale = attention_channel_activation_scale(
            "Qwen/Qwen3.5-2B", "model.layers.0.full_attention"
        )["scale"]
        self.assertAlmostEqual(
            result["neurons"][0]["relativeContribution"],
            -3.0 * 2.0 / scale,
        )
        self.assertEqual(result["neurons"][0]["contributionMagnitude"], 6.0)
        self.assertEqual(
            result["neurons"][0]["scaleSide"],
            attention_channel_activation_scale(
                "Qwen/Qwen3.5-2B", "model.layers.0.full_attention"
            )["side"],
        )

    def test_qwen_neuron_lens_reports_relative_activation_and_scale_side(self) -> None:
        adapter = object.__new__(LLMDebugAdapter)
        adapter.trace = (
            LayerState(
                Layer("model.layers.0.mlp", "mlp", "mlp", 1.0, 0.0),
                (1.0,), (2.0,), "mlp", {"mlp_expansion": (0.25,)},
            ),
            LayerState(
                Layer("model.layers.1.full_attention", "attention", "attention", 1.0, 0.0),
                (2.0,), (2.0,), "attention",
            ),
        )
        adapter.layer_index = 1
        adapter.token_index = 0
        adapter.tokens = (" test",)
        adapter.inside_mlp = False
        adapter.inside_attention = False
        adapter.model_name = "Qwen/Qwen3.5-0.8B"
        projection = NeuronProjection(
            index=0, activation=0.2561914622783661, direction_norm=1.0,
            vocabulary=(TokenScore(1, " token", 1.0),), contribution=(),
        )
        adapter.execution = SimpleNamespace(
            top_mlp_neurons=lambda *_args, **_kwargs: (projection,)
        )
        adapter.neuron_cache = {}

        neuron = adapter._neuron_lens_snapshot()["neurons"][0]

        scale_record = mlp_activation_scale(
            "Qwen/Qwen3.5-0.8B", "model.layers.0.mlp"
        )
        self.assertAlmostEqual(
            neuron["relativeContribution"],
            projection.activation * projection.direction_norm / scale_record["scale"],
        )
        self.assertEqual(neuron["scaleSide"], scale_record["side"])
        self.assertGreater(neuron["activationScale"], 0)

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

    def test_continue_filters_loop_breakpoint_by_transformer_layer(self) -> None:
        adapter = object.__new__(LLMDebugAdapter)
        adapter.tokens = ("a",)
        adapter.source_lines = (10, 20, 20, 20, 20)
        adapter.step_paths = (
            "token_embedding",
            "model.layers.0.full_attention", "model.layers.0.mlp",
            "model.layers.1.gated_linear_attention", "model.layers.1.mlp",
        )
        adapter.function_source_lines = {}
        adapter.mlp_source_lines = ()
        adapter.attention_source_lines = ()
        adapter.breakpoints = (SourceBreakpoint(20, None, 1),)
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
        self.assertEqual(adapter.layer_index, 3)

    def test_breakpoints_from_another_source_do_not_clear_active_source(self) -> None:
        adapter = object.__new__(LLMDebugAdapter)
        adapter.sequence = 1
        responses: list[dict[str, object]] = []
        adapter.stream = SimpleNamespace(write=responses.append)
        adapter.source_path = "/workspace/GPT-2.llmdbg.py"
        adapter.source_lines = (31, 33, 34, 35)
        adapter.function_source_lines = {"mlp": (23, 24, 25, 26, 27)}
        adapter.default_token = None
        adapter.tokens = ("The",)
        adapter.breakpoints = ()
        adapter.breakpoints_by_source = {}

        adapter.handle(
            {
                "seq": 1,
                "type": "request",
                "command": "setBreakpoints",
                "arguments": {
                    "source": {"path": "/workspace/GPT-2.llmdbg.py"},
                    "breakpoints": [{"line": 26}],
                },
            }
        )
        adapter.handle(
            {
                "seq": 2,
                "type": "request",
                "command": "setBreakpoints",
                "arguments": {
                    "source": {"path": "/workspace/other.llmdbg.py"},
                    "breakpoints": [],
                },
            }
        )

        self.assertEqual(adapter.breakpoints, (SourceBreakpoint(26),))
        self.assertEqual(
            adapter.breakpoints_by_source["/workspace/GPT-2.llmdbg.py"],
            (SourceBreakpoint(26),),
        )

    def test_generated_token_breakpoint_is_verified_within_generation_budget(self) -> None:
        adapter = object.__new__(LLMDebugAdapter)
        adapter.sequence = 1
        responses: list[dict[str, object]] = []
        adapter.stream = SimpleNamespace(write=responses.append)
        adapter.source_path = "/workspace/chat.llmdbg.py"
        adapter.source_lines = (52,)
        adapter.function_source_lines = {}
        adapter.default_token = None
        adapter.tokens = tuple(f"token-{index}" for index in range(20))
        adapter.max_new_tokens = 20
        adapter.breakpoints = ()
        adapter.breakpoints_by_source = {}

        adapter.handle({
            "seq": 1,
            "type": "request",
            "command": "setBreakpoints",
            "arguments": {
                "source": {"path": adapter.source_path},
                "breakpoints": [
                    {"line": 52, "condition": "token=33"},
                    {"line": 52, "condition": "token=40"},
                ],
            },
        })

        breakpoints = responses[-1]["body"]["breakpoints"]
        self.assertTrue(breakpoints[0]["verified"])
        self.assertFalse(breakpoints[1]["verified"])
        self.assertIn("range 0..39", breakpoints[1]["message"])

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

    def test_continue_after_generation_does_not_rescan_previous_tokens(self) -> None:
        adapter = object.__new__(LLMDebugAdapter)
        adapter.tokens = ("old",)
        adapter.source_lines = (20,)
        adapter.step_paths = ("model.layers.0.mlp",)
        adapter.mlp_source_lines = ()
        adapter.attention_source_lines = ()
        adapter.breakpoints = (SourceBreakpoint(20),)
        adapter.token_index = 0
        adapter.layer_index = 0
        adapter.trace = (object(),)
        adapter.inside_mlp = False
        adapter.inside_attention = False
        adapter.mlp_step_index = 0
        loaded: list[int] = []

        def load_token(position: int) -> None:
            loaded.append(position)
            adapter.token_index = position

        generated = False

        def append_generated_token() -> bool:
            nonlocal generated
            if generated:
                return False
            generated = True
            adapter.tokens = ("old", "new")
            return True

        adapter._load_token = load_token
        adapter._append_next_generated_token = append_generated_token
        stops: list[str] = []
        adapter.stop = stops.append

        adapter._continue_to_breakpoint(include_current=False)

        self.assertEqual(stops, ["breakpoint"])
        self.assertEqual(adapter.token_index, 1)
        self.assertNotIn(0, loaded)

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
        scope_names = {scope["name"] for scope in responses[6]["body"]["scopes"]}
        self.assertIn("Local variables", scope_names)
        self.assertNotIn("Next operation", scope_names)
        self.assertNotIn("Current input", scope_names)
        source = responses[5]["body"]["stackFrames"][0]["source"]
        self.assertEqual(source["sourceReference"], 1)
        self.assertTrue(source["name"].endswith(".execution.llmdbg"))
        layer_variables = responses[7]["body"]["variables"]
        self.assertTrue(any(variable["name"] == "meaning" for variable in layer_variables))
        self.assertEqual(responses[9]["body"]["result"], "0")
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

    def test_prompt_token_position_and_layer_are_first_class_context(self) -> None:
        requests = [
            {"seq": 1, "type": "request", "command": "initialize", "arguments": {}},
            {
                "seq": 2, "type": "request", "command": "launch",
                "arguments": {"stopOnEntry": True, "prompt": "Hello model"},
            },
            {"seq": 3, "type": "request", "command": "configurationDone", "arguments": {}},
            {
                "seq": 4, "type": "request", "command": "variables",
                "arguments": {"variablesReference": LLMDebugAdapter.SCOPE_TOKEN},
            },
            {"seq": 5, "type": "request", "command": "evaluate", "arguments": {"expression": "prompt"}},
            {"seq": 6, "type": "request", "command": "evaluate", "arguments": {"expression": "token_pos"}},
            {"seq": 7, "type": "request", "command": "evaluate", "arguments": {"expression": "layer"}},
            {"seq": 8, "type": "request", "command": "llmdbg/promptTokens", "arguments": {}},
            {"seq": 9, "type": "request", "command": "disconnect", "arguments": {}},
        ]
        process = subprocess.run(
            [sys.executable, "-m", "llmdbg.dap"],
            input=b"".join(frame(request) for request in requests),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        responses = {
            message.get("request_seq"): message
            for message in parse_frames(process.stdout)
            if message["type"] == "response"
        }
        context = {item["name"]: item for item in responses[4]["body"]["variables"]}
        self.assertEqual(context["prompt"]["evaluateName"], "prompt")
        self.assertEqual(context["token"]["value"], "0")
        self.assertNotIn("layer", context)
        self.assertNotIn("x", context)
        self.assertEqual(responses[5]["body"]["result"], "Hello model")
        self.assertEqual(responses[6]["body"]["result"], "0")
        self.assertTrue(responses[7]["body"]["result"])
        self.assertEqual(responses[8]["body"]["prompt"], "Hello model")
        self.assertEqual(responses[8]["body"]["tokens"], ["demo"])
        self.assertEqual(responses[8]["body"]["currentToken"], 0)

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

    def test_measure_cosine_uses_full_retained_vectors(self) -> None:
        adapter = object.__new__(LLMDebugAdapter)
        adapter.token_index = 4
        adapter.layer_index = 7
        adapter.measurement_vectors = {}
        vector_a = adapter._register_measurement_vector("Residual A", (1.0, 0.0, 0.0))
        vector_b = adapter._register_measurement_vector("Residual B", (1.0, 1.0, 0.0))

        result = adapter._measure_cosine(vector_a, vector_b)

        self.assertEqual(result["labelA"], "Residual A")
        self.assertEqual(result["labelB"], "Residual B")
        self.assertEqual(result["dimensions"], 3)
        self.assertAlmostEqual(result["cosine"], 2 ** -0.5)
        self.assertAlmostEqual(result["angleDegrees"], 45.0)

    def test_stop_keeps_measurement_vectors_for_saved_lens_cards(self) -> None:
        adapter = object.__new__(LLMDebugAdapter)
        adapter.lens_snapshot_cache = {("old",): {"view": "residual-post"}}
        adapter.measurement_vectors = {"old-vector": ("Residual", (1.0, 0.0))}
        adapter.sequence = 1
        adapter.token_index = 0
        adapter.layer_index = 0
        adapter.inside_mlp = False
        adapter.inside_attention = False
        adapter.trace = (
            SimpleNamespace(layer=SimpleNamespace(name="model.layers.0.mlp")),
        )
        adapter.stream = SimpleNamespace(write=lambda _message: None)

        adapter.stop("step")

        self.assertEqual(adapter.lens_snapshot_cache, {})
        self.assertIn("old-vector", adapter.measurement_vectors)


if __name__ == "__main__":
    unittest.main()
