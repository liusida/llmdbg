from __future__ import annotations

import json
import io
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from llmdbg.dap import LLMDebugAdapter, LensUnavailable, ProtocolStream, SourceBreakpoint
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
    def test_delta_memory_write_lens_decodes_beta_scaled_prediction_error(self) -> None:
        import torch
        from unittest.mock import Mock, patch
        from llmdbg.dap import LensUnavailable
        adapter = LLMDebugAdapter(ProtocolStream(io.BytesIO(), io.BytesIO()))
        adapter.tokens = ('cat',)
        adapter.model_name = 'test/model'
        memory = torch.arange(8.).reshape(1,2,2,2)
        key = torch.tensor([[[[1.,0.],[0.,1.]]]])
        value = torch.tensor([[[[3.,5.],[4.,9.]]]])
        decay, beta = torch.tensor([[[.5,.25]]]), torch.tensor([[[.2,.8]]])
        predicted = torch.einsum('bhk,bhkv->bhv',key[:,0], memory*decay[:,0,:,None,None])
        correction = beta[:,0,:,None]*(value[:,0]-predicted)
        state = LayerState(Layer('model.layers.0.gated_linear_attention','attention','',1.,0.),(0.,)*3,(0.,)*3,'',
                           {'delta_updated_memory': (0.,)*8})
        state.tensors.update(delta_previous_memory=memory,delta_key=key,delta_value=value,delta_decay=decay,delta_beta=beta)
        torch.testing.assert_close(adapter._delta_memory_write(state),correction.flatten())
        updated = memory*decay[:,0,:,None,None]+key[:,0].unsqueeze(-1)*correction.unsqueeze(-2)
        torch.testing.assert_close(updated-memory*decay[:,0,:,None,None],
                                   key[:,0].unsqueeze(-1)*adapter._delta_memory_write(state).reshape(1,2,1,2))
        adapter.trace = (state,LayerState(Layer('model.layers.0.mlp','mlp','',1.,0.),(0.,)*3,(0.,)*3,''))
        adapter.layer_index = 1
        adapter.execution = SimpleNamespace(runtime=SimpleNamespace(architecture='qwen35',model=SimpleNamespace(
            config=SimpleNamespace(linear_num_value_heads=2))), vector_lens=Mock(return_value=SimpleNamespace(
                vocabulary=(TokenScore(1,'cat',1.),),j_lens=(),r_lens=(),jpp_lens=())))
        weight = torch.arange(12.).reshape(4,3)
        with patch.object(adapter,'_attention_output_weight',return_value=weight):
            result = adapter._neuron_lens_snapshot(view='delta-memory-write')
            self.assertEqual(result['label'],'DeltaNet Memory Write Lens')
            self.assertEqual(result['ranking'],'raw-write-norm')
            for head,call in zip(result['heads'],adapter.execution.vector_lens.call_args_list):
                h=head['index']
                torch.testing.assert_close(torch.tensor(call.args[1]),correction[0,h] @ weight[h*2:(h+1)*2])
                self.assertIsNone(head['relativeNorm'])
                self.assertIsNone(head['calibrationError'])
                self.assertEqual(head['channels'],[])
            adapter.layer_index=0
            adapter.inside_attention=True
            adapter.configured_internal_steps={'gated_linear_attention':(
                ('update_memory','delta_updated_memory',''),('read_memory','delta_readout',''))}
            adapter.mlp_step_index=0
            with self.assertRaisesRegex(LensUnavailable,'after update_memory'):
                adapter._neuron_lens_snapshot(view='delta-memory-write')
            adapter.mlp_step_index=1
            self.assertEqual(len(adapter._neuron_lens_snapshot(view='delta-memory-write')['heads']),2)
        del state.tensors['delta_key']
        with self.assertRaisesRegex(LensUnavailable,'expanded DeltaNet'):
            adapter._delta_memory_write(state)

    def test_attention_head_value_lens_projects_values_and_expands_shared_kv(self) -> None:
        import torch
        from unittest.mock import Mock, patch
        for architecture, function, detail, values in (
            ('gpt2', 'attention', 'attention_qkv', (99.,)*8+(1.,2.,3.,4.)),
            ('qwen3_vl', 'full_attention', 'attention_value', (1.,2.)),
            ('qwen35', 'gated_linear_attention', 'delta_value', (1.,2.,3.,4.)),
        ):
            adapter = LLMDebugAdapter(ProtocolStream(io.BytesIO(), io.BytesIO()))
            adapter.model_name = 'test/model'
            path = ('transformer.h.0.' if architecture == 'gpt2' else 'model.layers.0.')+function
            adapter.trace = (LayerState(Layer(path, 'attention', '', 1., 0.), (0.,)*3, (0.,)*3, '', {detail: values}),
                             LayerState(Layer('model.layers.0.mlp', 'mlp', '', 1., 0.), (0.,)*3, (0.,)*3, ''))
            adapter.layer_index = 1
            adapter.tokens = ('cat',)
            runtime = SimpleNamespace(architecture=architecture, model=SimpleNamespace(config=SimpleNamespace(
                n_head=2, num_attention_heads=2, num_key_value_heads=1, linear_num_value_heads=2)))
            adapter.execution = SimpleNamespace(runtime=runtime, vector_lens=Mock(return_value=SimpleNamespace(
                vocabulary=(TokenScore(1,'cat',1.),), j_lens=(), r_lens=(), jpp_lens=())))
            weight = torch.arange(12.).reshape(4,3)
            with patch.object(adapter, '_attention_output_weight', return_value=weight):
                result = adapter._neuron_lens_snapshot(view='attention-head-values')
            self.assertEqual(result['label'], 'Attention Head Value Lens')
            self.assertEqual(result['ranking'], 'raw-norm-missing-calibration')
            expected = torch.tensor([1.,2.,1.,2.] if function=='full_attention' else [1.,2.,3.,4.])
            for head, call in zip(result['heads'],adapter.execution.vector_lens.call_args_list):
                start = head['index']*2
                torch.testing.assert_close(torch.tensor(call.args[1]), expected[start:start+2] @ weight[start:start+2])
                self.assertIsNone(head['relativeNorm'])
                self.assertIn('Missing calibrated value-head',head['calibrationError'])
                self.assertEqual(head['channels'], [])
                self.assertEqual(head['valueHead'], 0 if function=='full_attention' else head['index'])
            with patch.object(adapter,'_attention_output_weight',return_value=weight), patch(
                'llmdbg.dap.attention_head_norm_scale',side_effect=lambda model,layer,head,*,value: {'scale':1. if head==0 else 10000.}) as lookup:
                calibrated=adapter._neuron_lens_snapshot(view='attention-head-values')
            self.assertTrue(all(call.kwargs=={'value':True} for call in lookup.call_args_list))
            self.assertEqual(calibrated['ranking'],'calibrated-value-norm')
            self.assertEqual(calibrated['heads'][0]['index'],0)
            self.assertIsNone(calibrated['heads'][0]['calibrationError'])

    def test_attention_head_lenses_project_every_head_without_bias(self) -> None:
        import torch
        from unittest.mock import Mock, patch
        from llmdbg.dap import LensUnavailable
        for architecture, function, detail in (("gpt2", "attention", "attention_mixing"), ("qwen3_vl", "full_attention", "full_attention_mixing"), ("qwen35", "gated_linear_attention", "gated_delta_mixing")):
            adapter = LLMDebugAdapter(ProtocolStream(io.BytesIO(), io.BytesIO()))
            adapter.model_name = 'test/model'
            path = "transformer.h.0.attention" if architecture == "gpt2" else f"model.layers.0.{function}"
            mixed = torch.arange(4.) + 1
            weight = torch.arange(12.).reshape(4, 3)
            projection = SimpleNamespace(weight=weight if architecture == "gpt2" else weight.T, bias=torch.ones(3)*100)
            block = SimpleNamespace(attn=SimpleNamespace(c_proj=projection), self_attn=SimpleNamespace(o_proj=projection), linear_attn=SimpleNamespace(out_proj=projection))
            runtime = SimpleNamespace(architecture=architecture, layers=[block], layer_kind=lambda _: function, model=SimpleNamespace(config=SimpleNamespace(n_head=2, num_attention_heads=8 if function == 'gated_linear_attention' else 2, linear_num_value_heads=2)))
            adapter.execution = SimpleNamespace(runtime=runtime, vector_lens=Mock(return_value=SimpleNamespace(norm=1., vocabulary=(TokenScore(1, "cat", 2.),), j_lens=(), r_lens=(), jpp_lens=())))
            def head_channels(layer, activation, **kwargs):
                count = runtime.model.config.n_head
                width = len(activation)//count
                start = kwargs['head_index']*width
                return tuple(NeuronProjection(index=i, activation=activation[i], direction_norm=1., vocabulary=(), contribution=()) for i in range(start, min(start+3,start+width)))
            adapter.execution.top_attention_channels = Mock(side_effect=head_channels)
            adapter.trace = (LayerState(Layer(path, "attention", "", 1., 0.), (0.,)*3, (0.,)*3, "", {detail: tuple(mixed.tolist())}), LayerState(Layer("model.layers.0.mlp", "mlp", "", 1., 0.), (0.,)*3, (0.,)*3, ""))
            adapter.tokens = ("test",)
            adapter.trace[0].details['attention_projection'] = (0.,)*3
            adapter.layer_index = 1
            result = adapter._neuron_lens_snapshot(view="attention-heads")
            self.assertEqual([head["index"] for head in result["heads"]], [1, 0])
            calls = adapter.execution.vector_lens.call_args_list
            vectors = [torch.tensor(call.args[1]) for call in calls]
            torch.testing.assert_close(vectors[0], mixed[2:] @ weight[2:])
            torch.testing.assert_close(vectors[1], mixed[:2] @ weight[:2])
            torch.testing.assert_close(sum(vectors), mixed @ weight)
            self.assertEqual(calls[0].kwargs["source_site"], "mid")
            self.assertEqual(calls[0].kwargs["token_count"], 5)
            self.assertIsNone(result["heads"][0]["relativeNorm"])
            self.assertIn('unavailable', result["heads"][0]["calibrationError"])
            self.assertEqual(result["heads"][0]["logitTokens"][0]["text"], "cat")
            adapter.layer_index = 0
            adapter.inside_attention = True
            adapter.mlp_step_index = 0
            with self.assertRaisesRegex(LensUnavailable, "mixing completes"):
                adapter._neuron_lens_snapshot(view="attention-heads")
            adapter.mlp_step_index = len(adapter.internal_steps) - 1
            available = adapter._neuron_lens_snapshot(view="attention-heads")
            self.assertEqual(len(available["heads"]), 2)
            projection.weight = torch.ones(14, 3) if architecture == "gpt2" else torch.ones(3, 14)
            runtime.model.config.n_head = runtime.model.config.num_attention_heads = 7
            runtime.model.config.linear_num_value_heads = 7
            adapter.trace[0].details[detail] = tuple(float(i) for i in range(14))
            adapter.execution.vector_lens.reset_mock()
            adapter.execution.vector_lens.return_value.vocabulary = tuple(TokenScore(i, str(i), float(i)) for i in range(10))
            top = adapter._neuron_lens_snapshot(view="attention-heads")
            self.assertEqual([head["index"] for head in top["heads"]], [6, 5, 4, 3, 2])
            for head in top['heads']:
                self.assertLessEqual(len(head['channels']), 3)
                self.assertTrue(all(channel['index']//2 == head['index'] for channel in head['channels']))
            self.assertEqual(adapter.execution.vector_lens.call_count, 5)
            self.assertTrue(all(len(head["logitTokens"]) == 5 for head in top["heads"]))
            with patch('llmdbg.dap.attention_head_norm_scale', side_effect=lambda model, layer, head: {"scale": 1. if head == 0 else 10000.}):
                calibrated = adapter._neuron_lens_snapshot(view='attention-heads')
            self.assertEqual(calibrated['ranking'], 'calibrated-norm')
            self.assertEqual(calibrated['heads'][0]['index'], 0)
            self.assertAlmostEqual(calibrated['heads'][0]['relativeNorm'], 3. ** .5, places=5)

    def test_expression_decoder_loop(self) -> None:
        from unittest.mock import Mock
        from llmdbg.dap import DebugActionError
        adapter = LLMDebugAdapter(ProtocolStream(io.BytesIO(), io.BytesIO()))
        adapter._decode_expression = Mock(side_effect=[{"rows": []}, DebugActionError("bad slice"), {"rows": []}])
        result = adapter._decode_expressions("mixed[0, 0, i*64:(i+1)*64] @ W_o[i*64:(i+1)*64, :]", {"variable": "i", "start": 0, "stop": 3})
        self.assertEqual([item["loopValue"] for item in result["results"]], [0, 1, 2])
        self.assertEqual(result["results"][1]["error"], "bad slice")
        self.assertIn("0 * 64", adapter._decode_expression.call_args_list[0].args[0])
        self.assertIn("2 * 64", adapter._decode_expression.call_args_list[2].args[0])
        for loop in ({"variable": "for", "stop": 2}, {"variable": "i", "stop": 65}, {"variable": "i", "stop": 0}, {"variable": "i", "stop": 2.5}, {"variable": "j", "stop": 2}):
            with self.assertRaises(DebugActionError):
                adapter._decode_expressions("mixed[i]", loop)
        adapter._decode_expression = Mock(return_value={"rows": []})
        adapter._decode_expressions("mixed", None)
        adapter._decode_expression.assert_called_once_with("mixed")

    def test_expression_decoder_keeps_custom_lenses_separate_and_isolates_failures(self) -> None:
        import torch
        from unittest.mock import Mock, patch
        from llmdbg.model_runtime import VectorProjection
        from llmdbg.dap import DebugActionError
        adapter = LLMDebugAdapter(ProtocolStream(io.BytesIO(), io.BytesIO()))
        adapter.tokens = ("test",)
        tensor = torch.arange(4.).reshape(1, 1, 4)
        adapter.trace = (LayerState(Layer("model.layers.1.full_attention", "", "", 1., 0.), (1.,), (2.,), "", details={"normalized_input": (0., 1., 2., 3.)}, tensors={"input": torch.ones_like(tensor), "normalized_input": tensor, "matrix": torch.ones(2, 4)}),)
        adapter.inside_attention = True
        adapter.mlp_step_index = 1
        adapter.function_bindings = {"full_attention": [{"name": "normalized", "operation": "input_layernorm", "line": 1}]}
        adapter.execution = SimpleNamespace(runtime=SimpleNamespace(layers=[None] * 2, model=SimpleNamespace(get_output_embeddings=lambda: SimpleNamespace(weight=torch.zeros(8, 4)))))
        token = TokenScore(1, "cat", 2.)
        def project(_layer, values, **kwargs):
            self.assertEqual(values, (-1., 0., 1., 2.))
            if kwargs["include_r_lens"]:
                raise RuntimeError("corrupt artifact")
            return VectorProjection(1., (token,), j_lens=(token,), jpp_lens=(token,))
        adapter.execution.vector_lens = Mock(side_effect=project)
        entries = [{"id": method, "label": method, "method": method} for method in ("j", "r", "jpp")]
        entries.append({"id": "other-j", "label": "Another J Lens", "method": "j"})
        with patch("llmdbg.residual_vocabulary_lenses.configured_lenses", return_value=tuple(entries)):
            result = adapter._decode_expression("normalized - x")
        self.assertEqual(len(result["rows"]), 5)
        self.assertEqual(result["sourceSite"], "post")
        self.assertEqual(result["sourceLayer"], 0)
        self.assertEqual(result["layer"], 1)
        self.assertEqual(result["rows"][0]["tokens"][0]["text"], "cat")
        self.assertIn("corrupt artifact", result["rows"][2]["error"])
        self.assertEqual(result["rows"][-1]["label"], "Another J Lens")
        self.assertEqual(adapter.execution.vector_lens.call_count, 5)
        adapter.execution.vector_lens.side_effect = lambda *args, **kwargs: VectorProjection(1., (token,))
        with patch("llmdbg.residual_vocabulary_lenses.configured_lenses", return_value=tuple(entries)):
            missing = adapter._decode_expression("normalized - x")
        self.assertEqual([row["id"] for row in missing["rows"]], ["logit"])
        with self.assertRaisesRegex(DebugActionError, "not flattened"):
            adapter._decode_expression("matrix")
        with self.assertRaisesRegex(DebugActionError, "NaN or infinity"):
            adapter._decode_expression("normalized / 0")

    def test_attention_output_weight_expression(self) -> None:
        import torch
        from llmdbg.dap import DebugActionError
        adapter = LLMDebugAdapter(ProtocolStream(io.BytesIO(), io.BytesIO()))
        with self.assertRaisesRegex(DebugActionError, "paused model"):
            adapter._inspect_expression("W_o")
        mixed = torch.arange(4.).reshape(1, 1, 4)
        adapter.trace = (LayerState(
            Layer("transformer.h.0.attention", "attention", "attention", 1., 0.),
            (1.,), (2.,), "test", tensors={"attention_mixing": mixed},
        ),)
        adapter.inside_attention = True
        weight = torch.nn.Parameter(torch.arange(16.).reshape(4, 4))
        runtime = SimpleNamespace(architecture="gpt2", layers=[SimpleNamespace(
            attn=SimpleNamespace(c_proj=SimpleNamespace(weight=weight)))])
        adapter.execution = SimpleNamespace(runtime=runtime)
        torch.testing.assert_close(adapter._inspect_expression("mixed[0, 0, :2] @ W_o[:2, :]"), mixed[0, 0, :2] @ weight[:2])
        inspected = adapter._inspect_expression("W_o")
        self.assertFalse(inspected.requires_grad)
        inspected.zero_()
        self.assertNotEqual(weight.sum().item(), 0)
        for kind, attribute, projection in (("full_attention", "self_attn", "o_proj"), ("gated_linear_attention", "linear_attn", "out_proj")):
            runtime.architecture = "qwen3_5"
            runtime.layer_kind = lambda layer, kind=kind: kind
            runtime.layers = [SimpleNamespace(**{attribute: SimpleNamespace(**{projection: SimpleNamespace(weight=weight)})})]
            torch.testing.assert_close(adapter._inspect_expression("W_o"), weight.T)

    def test_tensor_arithmetic_for_watch_and_plot_is_non_mutating(self) -> None:
        import torch
        from llmdbg.dap import DebugActionError
        adapter = LLMDebugAdapter(ProtocolStream(io.BytesIO(), io.BytesIO()))
        adapter.tokens = ("test",)
        tensor = torch.arange(4.).reshape(1, 1, 4)
        adapter.trace = (LayerState(
            Layer("model.layers.3.full_attention", "attention", "attention", 1., 0.),
            (1.,), (2.,), "test", details={"normalized_input": (0., 1., 2., 3.)}, tensors={"input": torch.ones_like(tensor),
            "normalized_input": tensor, "matrix": torch.ones(4, 2)},
        ),)
        adapter.inside_attention = True
        adapter.mlp_step_index = 1
        adapter.function_bindings = {"full_attention": [{"name": "normalized", "operation": "input_layernorm", "line": 1}]}
        for expression, expected in (
            ("normalized-x", tensor - 1),
            ("(normalized + x) * 2 / 4", (tensor + 1) / 2),
            ("-normalized + 2", -tensor + 2),
            ("(normalized-x)[0, 0, :2]", (tensor - 1)[0, 0, :2]),
            ("normalized[0, 0, 1*2:(1+1)*2]", tensor[0, 0, 2:4]),
            ("normalized[0, 0, ::1+1]", tensor[0, 0, ::2]),
            ("normalized[0, 0, (1+1)*1]", tensor[0, 0, 2]),
            ("normalized @ matrix", tensor @ torch.ones(4, 2)),
        ):
            torch.testing.assert_close(adapter._inspect_expression(expression), expected)
        self.assertIsNone(adapter._inspect_expression("normalized.add_(1)"))
        self.assertIsNone(adapter._inspect_expression("__import__('os')"))
        with self.assertRaises(DebugActionError):
            adapter._inspect_expression("normalized ** 2")
        with self.assertRaises(DebugActionError):
            adapter._inspect_expression("normalized + matrix")
        with self.assertRaisesRegex(DebugActionError, "integer"):
            adapter._inspect_expression("normalized[0, 0, :4/2]")
        torch.testing.assert_close(tensor, torch.arange(4.).reshape(1, 1, 4))
        try:
            import matplotlib
        except ImportError:
            return
        result = adapter._render_plot("normalized-x", "fig, ax = plt.subplots(); ax.plot(vector)")
        self.assertIn("shape [1, 1, 4]", result["meta"])
        self.assertTrue(result["image"])

    def test_nested_qwen_mixer_step_in_over_out_and_tuple_locals(self) -> None:
        from llmdbg.config import load_debug_config
        from llmdbg.demo_model import Layer, LayerState
        from unittest.mock import Mock
        source = Path(__file__).parents[2] / "llmdbg-usecase" / "Qwen35-2B-Vision.llmdbg.py"
        adapter = LLMDebugAdapter(ProtocolStream(io.BytesIO(), io.BytesIO()))
        adapter.execution = SimpleNamespace(runtime=SimpleNamespace(
            layers=[object()], architecture="qwen3_5", layer_kind=lambda _: "gated_linear_attention",
            state_name=lambda i, k: f"model.layers.{i}.{k}", final_norm_name="model.norm",
        ))
        adapter._configure_model_source(load_debug_config(str(source)))
        adapter.trace = [LayerState(Layer("model.layers.0.gated_linear_attention", "", "", 1., 0.), (0.,), (0.,), "")]
        adapter.layer_index = 0
        adapter.inside_attention = True
        adapter.stop = Mock()
        adapter.mlp_step_index = 1
        adapter.handle({"seq": 1, "command": "next"})
        self.assertEqual(adapter.mlp_step_index, 9)
        adapter.mlp_step_index = 1
        adapter.handle({"seq": 2, "command": "stepIn"})
        self.assertEqual(adapter._core_info()["name"], "gated_delta")
        self.assertEqual(adapter.mlp_step_index, 2)
        adapter.handle({"seq": 3, "command": "next"})
        self.assertEqual(adapter._source_expression("query"), "delta_query")
        adapter.handle({"seq": 4, "command": "next"})
        self.assertEqual(adapter._source_expression("beta"), "delta_beta")
        adapter.handle({"seq": 5, "command": "stepOut"})
        self.assertEqual(adapter.mlp_step_index, 9)
        self.assertIsNone(adapter._core_info())
        adapter.trace = [LayerState(Layer("model.layers.0.full_attention", "", "", 1., 0.), (0.,), (0.,), "")]
        adapter.mlp_step_index = 1
        adapter.handle({"seq": 6, "command": "stepIn"})
        self.assertEqual(adapter._core_info()["name"], "attention")
        adapter.handle({"seq": 7, "command": "next"})
        self.assertEqual(adapter._source_expression("query"), "attention_query")
        adapter.handle({"seq": 8, "command": "stepOut"})
        self.assertEqual(adapter.mlp_step_index, 10)

    def test_nested_mixer_return_keeps_completed_mixed_local_and_breakpoint(self) -> None:
        import torch
        from unittest.mock import Mock, patch
        from llmdbg.model_runtime import VectorProjection
        source = Path(__file__).parents[2] / 'llmdbg-usecase/Qwen35-2B-Vision.llmdbg.py'
        from llmdbg.config import load_debug_config
        config = load_debug_config(source)
        adapter = LLMDebugAdapter(ProtocolStream(io.BytesIO(), io.BytesIO()))
        adapter.execution = SimpleNamespace(runtime=SimpleNamespace(
            layers=[object()], architecture='qwen3_5', layer_kind=lambda _: 'gated_linear_attention',
            state_name=lambda i,k:f'model.layers.{i}.{k}', final_norm_name='model.norm',
            model=SimpleNamespace(get_output_embeddings=lambda:SimpleNamespace(weight=torch.zeros(3,4))),
        ),vector_lens=Mock(return_value=VectorProjection(1.,(TokenScore(1,'cat',2.),))))
        adapter._configure_model_source(config)
        mixed = torch.arange(4.).reshape(1,1,4)
        adapter.trace = [LayerState(Layer('model.layers.0.gated_linear_attention','','',1.,0.),(0.,)*4,(0.,)*4,'',
            details={'gated_delta_mixing':tuple(mixed.flatten().tolist())},tensors={'gated_delta_mixing':mixed})]
        adapter.tokens=('test',)
        adapter.inside_attention=True
        adapter.stop=Mock()
        core=adapter.nested_cores['gated_linear_attention']
        adapter.mlp_step_index=core['end']-1
        self.assertNotIn('mixed',adapter._local_vector_names())
        adapter.handle({'seq':1,'command':'next'})
        self.assertEqual(adapter._active_function_name(),'gated_delta')
        self.assertEqual(adapter.internal_steps[adapter.mlp_step_index][0],'return')
        self.assertEqual(adapter._internal_source_lines()[adapter.mlp_step_index],config['functionReturns']['gated_delta']['line'])
        self.assertIn('mixed',adapter._local_vector_names())
        torch.testing.assert_close(adapter._inspect_expression('mixed'),mixed)
        with patch('llmdbg.residual_vocabulary_lenses.configured_lenses',return_value=()):
            self.assertEqual(adapter._decode_expression('mixed')['shape'],[1,1,4])
        adapter.handle({'seq':2,'command':'next'})
        self.assertIsNone(adapter._core_info())
        self.assertEqual(adapter.internal_steps[adapter.mlp_step_index][0],'output_projection')
        adapter.breakpoints=(SourceBreakpoint(config['functionReturns']['gated_delta']['line']),)
        adapter.mlp_step_index=core['call']
        adapter.handle({'seq':3,'command':'next'})
        self.assertEqual(adapter.mlp_step_index,core['end'])
        adapter.handle({'seq':4,'command':'stepOut'})
        self.assertEqual(adapter.mlp_step_index,core['end']+1)

        # The earlier inline-return form remains valid and adds no duplicate stop.
        text=source.read_text().replace('mixed = block.gated_output_norm(readout, gate)\n    return mixed',
                                        'return block.gated_output_norm(readout, gate)')
        with tempfile.TemporaryDirectory() as directory:
            old_source=Path(directory)/'old.llmdbg.py'
            old_source.write_text(text)
            old_config=load_debug_config(old_source)
            adapter._configure_model_source(old_config)
        self.assertNotIn('gated_delta',old_config['functionReturns'])
        self.assertEqual(adapter.nested_cores['gated_linear_attention']['end'],7)

    def test_attention_channel_details_map_query_heads_to_shared_value_heads(self) -> None:
        import torch
        state = SimpleNamespace(tensors={
            "attention_weights": torch.empty(1, 8, 1, 3),
            "attention_cached_values": torch.empty(1, 2, 3, 256),
        })
        self.assertEqual(LLMDebugAdapter._attention_channel_head(state, 1303), {
            "queryHead": 5, "kvHead": 1, "dimension": 23, "headDimension": 256,
        })

    def test_gemma4_source_exposes_shared_attention_and_complete_block_output(self) -> None:
        from llmdbg.config import load_debug_config
        source = Path(__file__).parents[2] / "llmdbg-usecase" / "Gemma4-E2B.llmdbg.py"
        adapter = LLMDebugAdapter(ProtocolStream(io.BytesIO(), io.BytesIO()))
        adapter.execution = SimpleNamespace(runtime=SimpleNamespace(
            layers=[object()] * 35, architecture="gemma4", has_per_layer_inputs=True,
            layer_kind=lambda index: "full_attention",
            state_name=lambda index, kind: f"model.layers.{index}.{kind}",
            final_norm_name="model.norm",
        ))
        adapter._configure_model_source(load_debug_config(str(source)))
        for operation in ("full_attention", "mlp", "per_layer_residual", "layer_scale"):
            self.assertEqual(sum(path.endswith("." + operation) for path in adapter.step_paths), 35)
        self.assertEqual(len(adapter.function_source_lines["full_attention"]), 9)
        self.assertEqual(len(adapter.function_source_lines["mlp"]), 8)

    def test_qwen3vl_source_exposes_all_attention_and_deepstack_steps(self) -> None:
        from llmdbg.config import load_debug_config
        source = Path(__file__).parents[2] / "llmdbg-usecase" / "Qwen3-VL-2B-Instruct.llmdbg.py"
        adapter = LLMDebugAdapter(ProtocolStream(io.BytesIO(), io.BytesIO()))
        adapter.execution = SimpleNamespace(runtime=SimpleNamespace(
            layers=[object()] * 28, architecture="qwen3_vl", visual_layer_count=3,
            layer_kind=lambda index: "full_attention",
            state_name=lambda index, kind: f"model.layers.{index}.{kind}",
            final_norm_name="model.norm",
        ))
        adapter._configure_model_source(load_debug_config(str(source)))
        self.assertEqual(sum(path.endswith(".full_attention") for path in adapter.step_paths), 28)
        self.assertEqual(sum(path.endswith(".visual_residual") for path in adapter.step_paths), 3)
        self.assertEqual(len(adapter.function_source_lines["full_attention"]), 10)
        self.assertEqual(len(adapter.function_source_lines["mlp"]), 7)

    def test_visual_token_regions_use_processor_grid_and_validate_token_count(self) -> None:
        adapter = LLMDebugAdapter(ProtocolStream(io.BytesIO(), io.BytesIO()))
        adapter.messages = [{"role": "user", "content": [
            {"type": "image", "url": "first.png"},
            {"type": "image", "url": "second.png"},
        ]}]
        adapter.tokens = ("start",) + ("<|image_pad|>",) * 6 + ("between",) + ("<|image_pad|>",) * 4
        adapter.execution = SimpleNamespace(image_grids=((1, 4, 6), (1, 4, 4)), image_merge_size=2)
        first, second = adapter._prompt_images()
        self.assertEqual((first["gridRows"], first["gridColumns"], first["startToken"], first["endToken"]), (2, 3, 1, 6))
        self.assertEqual((second["gridRows"], second["gridColumns"], second["startToken"], second["endToken"]), (2, 2, 8, 11))
        adapter.execution.image_grids = ((1, 4, 4), (1, 4, 4))
        self.assertNotIn("regionMapping", adapter._prompt_images()[0])
        self.assertEqual(adapter._prompt_images()[1]["regionMapping"], "merged-row-major")

    def test_attention_token_contributions_use_correct_gqa_head_and_signed_values(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch is needed for tensor inspection")
        layer = "model.layers.3.full_attention"
        weights = torch.zeros(1, 8, 1, 3)
        weights[0, 6, 0] = torch.tensor([.25, .5, .25])
        values = torch.zeros(1, 2, 3, 256)
        values[0, 0, :, 97] = 100.  # The other KV head must not be used.
        values[0, 1, :, 97] = torch.tensor([2., -4., 1.])
        mixed = torch.zeros(1, 1, 2048)
        mixed[0, 0, 1633] = -.625  # Output gate is distinct from raw token contributions.
        adapter = LLMDebugAdapter(ProtocolStream(io.BytesIO(), io.BytesIO()))
        adapter.tokens = ("first", "second", "current")
        adapter.token_index = 2
        adapter.trace = (LayerState(Layer(layer, "attention", "test", 1., 0.), (), (), "test", tensors={
            "attention_weights": weights, "attention_cached_values": values,
            "full_attention_mixing": mixed,
        }),)
        result = adapter._attention_token_contributions(layer, 1633, 2)
        self.assertEqual((result["queryHead"], result["kvHead"], result["dimension"]), (6, 1, 97))
        self.assertEqual([row["contribution"] for row in result["rows"]], [.5, -2., .25])
        self.assertEqual(result["sum"], -1.25)
        self.assertEqual(result["positiveSum"], .75)
        self.assertEqual(result["negativeSum"], -2.)
        self.assertEqual(result["mixedValue"], -.625)
        self.assertEqual(result["stage"], "before-output-gate")
        with self.assertRaisesRegex(ValueError, "another token"):
            adapter._attention_token_contributions(layer, 1633, 1)
        with self.assertRaisesRegex(ValueError, "Channel must be"):
            adapter._attention_token_contributions(layer, 2048, 2)
        adapter.trace = (LayerState(Layer(layer, "attention", "test", 1., 0.), (), (), "test"),)
        with self.assertRaisesRegex(ValueError, "unavailable"):
            adapter._attention_token_contributions(layer, 1633, 2)

    def test_tensor_watch_preserves_shape_and_supports_indexing_and_expansion(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch is needed for tensor inspection")
        adapter = LLMDebugAdapter(ProtocolStream(io.BytesIO(), io.BytesIO()))
        tensor = torch.arange(24, dtype=torch.float16).reshape(1, 2, 3, 4)
        adapter.trace = (LayerState(
            Layer("model.layers.3.full_attention", "attention", "attention", 1., 0.),
            (1.,), (2.,), "test", tensors={"attention_cached_keys": tensor},
        ),)
        adapter.inside_attention = True
        adapter.mlp_step_index = 5
        adapter.function_bindings = {"full_attention": [
            {"name": "keys", "operation": "cached_keys", "line": 5},
        ]}
        self.assertEqual(adapter._evaluate_body("keys.shape")["result"], "torch.Size([1, 2, 3, 4])")
        self.assertEqual(adapter._evaluate_body("keys.dtype")["result"], "torch.float16")
        self.assertEqual(adapter._evaluate_body("keys.size(-1)")["result"], "4")
        self.assertEqual(adapter._evaluate_body("keys.numel()")["result"], "24")
        self.assertEqual(adapter._evaluate_body("keys[0, 1, 2, 3]")["result"], "23.0")
        sliced = adapter._evaluate_body("keys[0, 1, :, :2]")
        retained = adapter.inspection_references[sliced["variablesReference"]][0]
        self.assertTrue(torch.equal(retained, tensor[0, 1, :, :2]))
        children = adapter._tensor_variables(sliced["variablesReference"], {"filter": "indexed", "start": 1, "count": 1})
        self.assertEqual([child["name"] for child in children], ["[1]"])
        self.assertEqual(children[0]["type"], "torch.Tensor")
        self.assertIsNone(adapter._inspect_expression("keys.zero_()"))
        self.assertEqual(tensor[0, 1, 2, 3].item(), 23.)

    def test_tensor_snapshot_is_independent_and_keeps_dtype(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch is needed for tensor inspection")
        from llmdbg.model_runtime import _inspection_snapshots
        source = torch.arange(6, dtype=torch.float16).reshape(1, 1, 6)
        snapshots = _inspection_snapshots(source, None, {"keys": source})
        source.zero_()
        self.assertEqual(snapshots["keys"].sum().item(), 15.)
        self.assertEqual(snapshots["keys"].dtype, torch.float16)
        self.assertEqual(snapshots["keys"].shape, (1, 1, 6))
        self.assertFalse(snapshots["keys"].requires_grad)

    def test_invalid_comparison_warns_once_and_keeps_session_responsive(self) -> None:
        requests = [
            {"seq": 1, "type": "request", "command": "llmdbg/measureCosine", "arguments": {}},
            {"seq": 2, "type": "request", "command": "disconnect"},
        ]
        output = io.BytesIO()
        adapter = LLMDebugAdapter(ProtocolStream(io.BytesIO(b"".join(map(frame, requests))), output))
        adapter.run()
        messages = parse_frames(output.getvalue())
        responses = [message for message in messages if message["type"] == "response"]
        self.assertEqual([message["request_seq"] for message in responses], [1, 2])
        self.assertFalse(responses[0]["success"])
        self.assertTrue(responses[1]["success"])
        warnings = [message for message in messages if message.get("event") == "output"]
        self.assertEqual(len(warnings), 1)
        self.assertIn("Warning:", warnings[0]["body"]["output"])
        self.assertNotIn("Traceback", warnings[0]["body"]["output"])
        self.assertTrue(any(message.get("event") == "llmdbg/computationFailed" for message in messages))

    def test_unavailable_lens_does_not_report_an_internal_error(self) -> None:
        request = {"seq": 1, "type": "request", "command": "llmdbg/neuronLens",
                   "arguments": {"view": "residual-post", "mode": "logit"}}
        output = io.BytesIO()
        adapter = LLMDebugAdapter(ProtocolStream(io.BytesIO(frame(request)), output))
        def unavailable(_request):
            raise LensUnavailable("Residual Lens · Post is available after MLP residual addition")
        adapter.handle = unavailable
        adapter.run()
        messages = parse_frames(output.getvalue())
        responses = [message for message in messages if message["type"] == "response"]
        warnings = [message for message in messages if message.get("event") == "output"]
        self.assertEqual(len(responses), 1)
        self.assertTrue(responses[0]["success"])
        self.assertTrue(responses[0]["body"]["unavailable"])
        self.assertEqual(len(warnings), 1)
        self.assertEqual(warnings[0]["body"]["output"].count("\n"), 1)
        self.assertIn("Warning:", warnings[0]["body"]["output"])
        self.assertNotIn("Traceback", warnings[0]["body"]["output"])

    def test_failed_acknowledged_step_recovers_and_accepts_disconnect(self) -> None:
        requests = [
            {"seq": 1, "type": "request", "command": "stepIn"},
            {"seq": 2, "type": "request", "command": "disconnect"},
        ]
        output = io.BytesIO()
        adapter = LLMDebugAdapter(ProtocolStream(io.BytesIO(b"".join(map(frame, requests))), output))
        adapter.trace = (LayerState(Layer("test", "test", "test", 1., 0.), (1.,), (2.,), "test"),)
        adapter.layer_index = 0
        adapter.token_index = 0
        original = adapter.handle

        def fail_step(request):
            if request["command"] == "stepIn":
                adapter.send_response(request)
                adapter.inside_attention = True
                raise KeyError("attention_query")
            return original(request)

        adapter.handle = fail_step
        adapter.run()
        messages = parse_frames(output.getvalue())
        responses = [message for message in messages if message["type"] == "response"]
        self.assertEqual([message["request_seq"] for message in responses], [1, 2])
        self.assertTrue(any(message.get("event") == "stopped" and message["body"]["reason"] == "exception" for message in messages))
        self.assertTrue(any(message.get("event") == "output" and "KeyError" in message["body"]["output"] for message in messages))
        self.assertFalse(adapter.inside_attention)

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
