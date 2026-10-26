from __future__ import annotations

import ast
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from llmdbg.config import load_debug_config
from llmdbg.dap import LLMDebugAdapter, ProtocolStream, SourceBreakpoint
from llmdbg.demo_model import Layer, LayerState
from llmdbg.source_plan import compile_plans


SOURCE = '''from llmdbg import debug, qwen35
model = qwen35("Qwen/Qwen3.5-2B")
prompt = "hello"
tokens = model.tokenize(prompt)

def inspect_copy(Tensor):
    saved = Tensor
    return saved

def memory_core(normalized, block):
    query, key, value = block.convolved_qkv(normalized)
    decay, beta, gate = block.gates(normalized)
    memory = block.cached_memory()
    memory = block.update_memory(memory, key, value, decay, beta)
    readout = block.read_memory(query, memory)
    mixed = block.gated_output_norm(readout, gate)
    copy = mixed
    difference = copy - normalized
    copy = copy * 2
    width = mixed.size(2)
    first = mixed[0, 0, :width-1]
    inspected = inspect_copy(mixed)
    return mixed

def gated_linear_attention(Stream, Engine):
    Norm = Engine.input_layernorm(Stream)
    Mix = memory_core(normalized=Norm, block=Engine)
    Update = Engine.output_projection(Mix)
    Stream = Stream + Update
    return Stream

with debug(model, tokens):
    x = model.token_embedding(tokens)
    for block in model.layers:
        x = gated_linear_attention(x, block)
    x = model.final_layernorm(x)
'''


class SourcePlanTest(unittest.TestCase):
    def test_direct_program_capture_does_not_depend_on_helper_name(self):
        from llmdbg.program import _needs_delta_capture
        def differently_named_core(x,block):
            return block.convolved_qkv(x)
        self.assertTrue(_needs_delta_capture(SimpleNamespace(f_globals={},f_locals={'core':differently_named_core})))
        self.assertFalse(_needs_delta_capture(SimpleNamespace(f_globals={'gated_delta':lambda x:x},f_locals={})))

    def config(self, text=SOURCE):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'model.llmdbg.py'
            path.write_text(text)
            return load_debug_config(path)

    def adapter(self, config):
        import torch
        from llmdbg.model_runtime import TokenScore, VectorProjection
        adapter = LLMDebugAdapter(ProtocolStream(io.BytesIO(),io.BytesIO()))
        adapter.execution = SimpleNamespace(runtime=SimpleNamespace(
            architecture='qwen3_5', layers=[object()], layer_kind=lambda _: 'gated_linear_attention',
            state_name=lambda index,kind:f'model.layers.{index}.{kind}',final_norm_name='model.norm',
            model=SimpleNamespace(get_output_embeddings=lambda:SimpleNamespace(weight=torch.zeros(3,4))),
        ),input_ids=torch.zeros(1,1,dtype=torch.long),vector_lens=Mock(return_value=VectorProjection(1.,(TokenScore(1,'cat',2.),))))
        # Wire roundtrip: plan call indices become JSON string keys.
        adapter._configure_model_source(json.loads(json.dumps(config)))
        snapshots={'input':torch.ones(1,1,4),'normalized_input':torch.ones(1,1,4)*3,
                   'gated_delta_mixing':torch.arange(4.).reshape(1,1,4),
                   'attention_projection':torch.ones(1,1,4)*2,'output':torch.ones(1,1,4)*3,
                   'delta_qkv':torch.ones(4),'delta_query':torch.ones(1,1,16,128),
                   'delta_key':torch.ones(1,1,16,128),'delta_value':torch.ones(1,1,16,128),
                   'delta_beta':torch.ones(1,1,16),'delta_decay':torch.ones(1,1,16),
                   'delta_gate':torch.ones(1,1,16,128),'delta_readout':torch.ones(1,1,16,128),
                   'delta_previous_memory':torch.zeros(1,16,128,128),'delta_updated_memory':torch.ones(1,16,128,128)}
        details={name:tuple(tensor.flatten().tolist()) for name,tensor in snapshots.items() if name not in {'input','output'}}
        adapter.trace=(LayerState(Layer('model.layers.0.gated_linear_attention','','',1.,0.),(1.,)*4,(3.,)*4,'',details,snapshots),)
        adapter.tokens=('hello',)
        adapter.inside_attention=True
        adapter.stop=Mock()
        return adapter

    def step(self, adapter, line):
        adapter.mlp_step_index=adapter._internal_source_lines().index(line)

    def test_source_locals_aliases_expressions_and_reassignment(self):
        import torch
        config=self.config()
        adapter=self.adapter(config)
        steps=config['sourcePlans']['gated_linear_attention']['steps']
        return_step=next(step for step in steps if step['operation']=='return' and step['frames'][-1]['name']=='memory_core')
        self.step(adapter,return_step['line'])
        mixed=adapter.outer_state.tensors['gated_delta_mixing']
        torch.testing.assert_close(adapter._inspect_expression('mixed'),mixed)
        torch.testing.assert_close(adapter._inspect_expression('copy'),mixed*2)
        torch.testing.assert_close(adapter._inspect_expression('difference'),mixed-3)
        torch.testing.assert_close(adapter._inspect_expression('first'),mixed[0,0,:3])
        self.assertEqual(adapter._inspect_expression('width'),4)
        torch.testing.assert_close(adapter._inspect_expression('inspected'),mixed)
        self.assertIn('mixed',adapter._local_vector_names())
        self.assertIsNone(adapter._inspect_expression('Norm'))  # caller local, not this frame
        before=next(step for step in steps if step['operation']=='gated_output_norm')
        self.step(adapter,before['line'])
        self.assertNotIn('mixed',adapter._local_vector_names())
        self.assertIsNone(adapter._inspect_expression('mixed'))
        with patch('llmdbg.residual_vocabulary_lenses.configured_lenses',return_value=()):
            self.step(adapter,return_step['line'])
            self.assertEqual(adapter._decode_expression('difference')['shape'],[1,1,4])

    def test_arbitrary_nested_helper_frames_step_over_out_and_return_breakpoint(self):
        import torch
        config=self.config()
        adapter=self.adapter(config)
        adapter.send_response=Mock()
        calls=adapter.source_calls['gated_linear_attention']
        outer=next(call for call in calls.values() if call['name']=='memory_core')
        inner=next(call for call in calls.values() if call['name']=='inspect_copy')
        adapter.mlp_step_index=inner['call']
        adapter.handle({'seq':1,'command':'stepIn'})
        self.assertEqual(adapter._active_function_name(),'inspect_copy')
        adapter.handle({'seq':2,'command':'next'})
        self.assertEqual(adapter.internal_steps[adapter.mlp_step_index][0],'return')
        self.assertEqual(adapter._active_function_name(),'inspect_copy')
        adapter.handle({'seq':3,'command':'stackTrace'})
        frames=adapter.send_response.call_args.args[1]['stackFrames']
        self.assertEqual([f['name'].split(' · ')[0] for f in frames[:3]],['inspect_copy','memory_core','gated_linear_attention'])
        # Evaluate the parent's original local, not the child's bindings; preserve case.
        adapter.handle({'seq':4,'command':'evaluate','arguments':{'expression':'Norm','frameId':202}})
        self.assertTrue(adapter.send_response.call_args.kwargs.get('success',True))
        self.assertIn('torch.Tensor',adapter.send_response.call_args.args[1]['type'])
        self.assertIsNone(adapter._inspect_expression('Norm'))
        adapter.handle({'seq':5,'command':'stepOut'})
        self.assertEqual(adapter._active_function_name(),'memory_core')
        self.assertEqual(adapter.mlp_step_index,inner['end']+1)
        adapter.mlp_step_index=outer['call']
        adapter.handle({'seq':6,'command':'next'})
        self.assertEqual(adapter.mlp_step_index,outer['end']+1)
        adapter.breakpoints=(SourceBreakpoint(adapter._internal_source_lines()[inner['end']]),)
        adapter.mlp_step_index=outer['call']
        adapter.handle({'seq':7,'command':'next'})
        self.assertEqual(adapter.mlp_step_index,inner['end'])

    def test_rejects_changed_computation_and_unsafe_or_unrepresentable_helpers(self):
        cases=[
            (SOURCE.replace('Engine.output_projection(Mix)','Engine.output_projection(Mix * 2)'), 'inputs differ'),
            (SOURCE.replace('return Stream','return Stream + 2'), 'altered return'),
            (SOURCE.replace('copy = mixed','copy = mixed.zero_()',1), 'read-only'),
            (SOURCE.replace('copy = mixed','copy = __import__("os")',1), 'numeric constants|read-only'),
            (SOURCE.replace('copy = mixed','copy = missing_variable',1), 'not defined'),
            (SOURCE.replace('copy = mixed','for i in range(3):\n        copy = mixed',1), 'straight-line'),
            (SOURCE.replace('saved = Tensor','saved = inspect_copy(Tensor)'), 'recursive'),
            (SOURCE.replace('copy = mixed','copy = block.unknown_method(mixed)',1), 'capture adapter'),
            (SOURCE.replace('copy = mixed','mixed[0] = 0',1), 'mutation'),
        ]
        for source,message in cases:
            with self.subTest(message=message),self.assertRaisesRegex(ValueError,message):
                self.config(source)

    def test_all_authored_examples_have_source_derived_plans(self):
        examples=Path(__file__).parents[2]/'llmdbg-usecase'
        for path in examples.glob('*.llmdbg.py'):
            config=load_debug_config(path)
            self.assertTrue(config['sourcePlans'],path.name)
            for plan in config['sourcePlans'].values():
                self.assertTrue(all(step['line']>0 and step['frames'] for step in plan['steps']))

    def test_tail_helper_returns_and_equivalent_residual_diagnostics(self):
        source=SOURCE.replace('    Stream = Stream + Update\n    return Stream',
                              '    diagnostic = Stream + Update\n    return finish(Stream, Update)')
        source=source.replace('with debug(model, tokens):',
                              'def finish(a, b):\n    total = a + b\n    return total\n\nwith debug(model, tokens):')
        config=self.config(source)
        adapter=self.adapter(config)
        finish=next(call for call in adapter.source_calls['gated_linear_attention'].values() if call['name']=='finish')
        adapter.mlp_step_index=finish['end']
        adapter.handle({'seq':1,'command':'stepOut'})
        self.assertEqual(adapter._active_function_name(),'gated_linear_attention')
        self.assertEqual(adapter.internal_steps[adapter.mlp_step_index][0],'return')
        self.assertEqual(adapter._current_lens_phase(),2)

    def test_gemma_cannot_silently_skip_its_output_normalization(self):
        path=Path(__file__).parents[2]/'llmdbg-usecase/Gemma2-2B-IT.llmdbg.py'
        text=path.read_text().replace('update = block.post_attention_layernorm(projected)','update = projected')
        with self.assertRaisesRegex(ValueError,'native output normalization'):
            self.config(text)

    def test_alias_before_first_operation_and_initial_helper_frame_use_input(self):
        text=SOURCE.replace('    Norm = Engine.input_layernorm(Stream)',
                            '    saved = Stream\n    Norm = Engine.input_layernorm(saved)')
        config=self.config(text)
        adapter=self.adapter(config)
        adapter.mlp_step_index=1
        self.assertEqual(adapter.state.input_vector,(1.,)*4)
        self.assertEqual(adapter.previous_state.output_vector,(1.,)*4)
        # Move the complete root into a helper and tail-call it as the first line.
        text=SOURCE.replace('def gated_linear_attention(Stream, Engine):','def complete(Stream, Engine):')
        text=text.replace('with debug(model, tokens):',
                          'def gated_linear_attention(Stream, Engine):\n    return complete(Stream, Engine)\n\nwith debug(model, tokens):')
        adapter=self.adapter(self.config(text))
        adapter.mlp_step_index=1
        self.assertEqual(adapter._active_function_name(),'complete')
        self.assertEqual(adapter.state.input_vector,(1.,)*4)
        self.assertIsNone(adapter.previous_state)

    def test_inspection_expression_error_does_not_hide_other_locals(self):
        config=self.config(SOURCE.replace('difference = copy - normalized','difference = copy @ normalized'))
        adapter=self.adapter(config)
        step=next(i for i,item in enumerate(config['sourcePlans']['gated_linear_attention']['steps'])
                  if item['kind']=='return' and item['frames'][-1]['name']=='memory_core')
        adapter.mlp_step_index=step
        variables={variable['name']:variable for variable in adapter._local_variables()}
        self.assertEqual(variables['difference']['type'],'expression error')
        self.assertEqual(variables['mixed']['type'],'torch.Tensor')

    def test_multiline_assignment_and_helper_call_use_statement_line(self):
        text=SOURCE.replace('    Norm = Engine.input_layernorm(Stream)',
                            '    Norm = (\n        Engine.input_layernorm(Stream)\n    )')
        text=text.replace('    Mix = memory_core(normalized=Norm, block=Engine)',
                          '    Mix = (\n        memory_core(normalized=Norm, block=Engine)\n    )')
        config=self.config(text)
        steps=config['sourcePlans']['gated_linear_attention']['steps']
        lines=text.splitlines()
        norm_line=next(i+1 for i,line in enumerate(lines) if line=='    Norm = (')
        call_line=next(i+1 for i,line in enumerate(lines) if line=='    Mix = (')
        self.assertEqual(steps[0]['line'],norm_line)
        self.assertEqual(steps[1]['line'],call_line)
        self.assertEqual(steps[2]['frames'][-1]['callLine'],call_line)

    def test_annotated_and_chained_alias_assignments(self):
        import torch
        text=SOURCE.replace('    copy = mixed','    annotated: Tensor = mixed\n    copy = other = annotated',1)
        config=self.config(text)
        adapter=self.adapter(config)
        adapter.mlp_step_index=next(i for i,step in enumerate(config['sourcePlans']['gated_linear_attention']['steps'])
                                   if step['kind']=='return' and step['frames'][-1]['name']=='memory_core')
        mixed=adapter.outer_state.tensors['gated_delta_mixing']
        torch.testing.assert_close(adapter._inspect_expression('other'),mixed)
        torch.testing.assert_close(adapter._inspect_expression('annotated'),mixed)


if __name__=='__main__':
    unittest.main()
