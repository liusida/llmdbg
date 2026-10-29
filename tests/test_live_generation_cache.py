import io
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import torch
from llmdbg.dap import LLMDebugAdapter, ProtocolStream, SourceBreakpoint


class Execution:
    def __init__(self):
        self.input_ids=torch.tensor([[1,2]])
        self.tokens=('one','two')
        self.auto_chat=None
        self.interventions={};self.ica_interventions=set();self.vector_patches={}
        self.runtime=SimpleNamespace(tokenizer=SimpleNamespace(decode=lambda ids:str(ids),eos_token_id=99))
    @property
    def token_count(self):return len(self.tokens)
    def append_generated_token(self,token):
        self.input_ids=torch.cat((self.input_ids,torch.tensor([[token]])),dim=1)
        self.tokens+= (str(token),)
        return str(token)


class LiveGenerationCacheTest(unittest.TestCase):
    def adapter(self):
        a=LLMDebugAdapter(ProtocolStream(io.BytesIO(),io.BytesIO()))
        a.execution=Execution();a.tokens=a.execution.tokens
        a.max_new_tokens=10;a.generated_tokens=0
        a.source_lines=(10,);a.step_paths=('embedding',)
        a._internal_source_lines=Mock(return_value=())
        a._live_generation_cache=dict(execution=a.execution,start=2,key='key',persistent=True,
            metadata={'vocabulary_size':100})
        return a

    def test_continue_stops_at_cached_token_breakpoint_without_generating(self):
        a=self.adapter();a.breakpoints=(SourceBreakpoint(10,4),)
        a._load_token=Mock();a.stop=Mock()
        with patch('llmdbg.generation_cache.load',return_value={'ids':[3,4,5,6],'eos':True}):
            a._continue_to_breakpoint(include_current=True,before_start=True)
        self.assertEqual(a.execution.input_ids.tolist(),[[1,2,3,4,5]])
        a._load_token.assert_called_once_with(4)
        a.stop.assert_called_once_with('breakpoint')
        self.assertEqual(a.generated_tokens,3)

    def test_prompt_breakpoint_does_not_materialize_future(self):
        a=self.adapter();a.breakpoints=(SourceBreakpoint(10,0),)
        with patch('llmdbg.generation_cache.load',return_value={'ids':[3,4],'eos':True}):
            a._reuse_live_generation_until_breakpoint(include_current=True,before_start=True)
        self.assertEqual(a.execution.token_count,2)

    def test_budget_limits_cached_prefix(self):
        a=self.adapter();a.max_new_tokens=2
        with patch('llmdbg.generation_cache.load',return_value={'ids':[3,4,5],'eos':True}):
            a._reuse_live_generation_until_breakpoint(include_current=True,before_start=True)
        self.assertEqual(a.execution.input_ids.tolist(),[[1,2,3,4]])

    def test_intervention_disables_cache_even_after_removal(self):
        a=self.adapter();a.execution.interventions={'x':1}
        self.assertIsNone(a._live_generation_context())
        a.execution.interventions={}
        self.assertIsNone(a._live_generation_context())

    def test_conflicting_prefix_is_not_reused(self):
        a=self.adapter();a.execution.append_generated_token(9);a.tokens=a.execution.tokens;a.generated_tokens=1
        with patch('llmdbg.generation_cache.load',return_value={'ids':[3,4],'eos':True}):
            a._reuse_live_generation_until_breakpoint(include_current=True,before_start=True)
        self.assertEqual(a.execution.input_ids.tolist(),[[1,2,9]])
        self.assertFalse(a._live_generation_cache['persistent'])

    def test_cache_hit_honors_eos_without_predicting_more(self):
        a=self.adapter();a.execution.append_generated_token(3);a.tokens=a.execution.tokens;a.generated_tokens=1
        a.predictions=(SimpleNamespace(token_id=4),)
        with patch('llmdbg.generation_cache.load',return_value={'ids':[3],'eos':True}),patch('llmdbg.generation_cache.save'):
            self.assertFalse(a._append_next_generated_token())
        self.assertEqual(a.execution.token_count,3)

    def test_normal_generation_saves_ids_and_eos(self):
        a=self.adapter();a.predictions=(SimpleNamespace(token_id=3),)
        with patch('llmdbg.generation_cache.load',return_value=None),patch('llmdbg.generation_cache.save') as save:
            self.assertTrue(a._append_next_generated_token())
            self.assertEqual(save.call_args.args[1:3],([3],False))
            a.predictions=(SimpleNamespace(token_id=99),)
            self.assertFalse(a._append_next_generated_token())
            self.assertEqual(save.call_args.args[1:3],([3],True))

    def test_continue_after_current_breakpoint_uses_next_safe_prefix(self):
        a=self.adapter();a.token_index=1;a.layer_index=0
        a.breakpoints=(SourceBreakpoint(10,1),SourceBreakpoint(10,3))
        with patch('llmdbg.generation_cache.load',return_value={'ids':[3,4,5],'eos':False}):
            a._reuse_live_generation_until_breakpoint(include_current=False,before_start=False)
        self.assertEqual(a.execution.input_ids.tolist(),[[1,2,3,4]])

    def test_internal_breakpoint_limits_materialization(self):
        a=self.adapter();a._internal_source_lines=Mock(return_value=(11,12))
        a.breakpoints=(SourceBreakpoint(12,2),)
        with patch('llmdbg.generation_cache.load',return_value={'ids':[3,4,5],'eos':False}):
            a._reuse_live_generation_until_breakpoint(include_current=True,before_start=True)
        self.assertEqual(a.execution.input_ids.tolist(),[[1,2,3]])


if __name__=='__main__':unittest.main()
