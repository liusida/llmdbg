from __future__ import annotations

import io
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from llmdbg.concept_trace import Snapshot, cosine_scores, dispatch, target_support, target_tokens
from llmdbg.dap import LLMDebugAdapter, ProtocolStream
from llmdbg.model_runtime import ModelExecution, ModelRuntime


class ConceptTraceTest(unittest.TestCase):
    def cache_adapter(self,budget=2):
        adapter=self.adapter()
        adapter.execution.runtime.model.config._commit_hash='unit-test-checkpoint'
        adapter.execution.runtime.tokenizer.eos_token_id=63
        adapter.max_new_tokens=budget
        return adapter

    def test_generation_cache_survives_sessions_extends_budget_and_preserves_eos(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ,{'LLMDBG_GENERATION_CACHE_DIR':directory}):
            first=self.cache_adapter()
            with patch.object(Snapshot,'_next_token',side_effect=[5,6]):
                original=Snapshot(first,{'target':' cat'})
            second=self.cache_adapter()
            with patch.object(Snapshot,'_next_token',side_effect=AssertionError('cached IDs must not regenerate')):
                reused=Snapshot(second,{'target':'two pieces','tokenId':4})
            self.assertEqual(reused.generation_key,original.generation_key)
            self.assertEqual(reused.result()['generationCache'],'disk')
            self.assertEqual(reused.result()['reusedGeneratedTokens'],2)
            self.assertEqual(reused.newly_generated,0)
            self.assertFalse(reused.activations_reused)
            self.assertEqual(reused.result()['generatedTokens'],['5','6'])
            third=self.cache_adapter(4)
            with patch.object(Snapshot,'_next_token',side_effect=[7,63]) as predict:
                extended=Snapshot(third,{'target':' cat'})
            self.assertEqual(predict.call_count,2)
            self.assertEqual(extended.reused_generated,2)
            self.assertEqual(extended.newly_generated,1)
            self.assertTrue(extended.eos_stopped)
            self.assertEqual(extended.result()['generatedTokens'],['5','6','7'])
            with patch.object(Snapshot,'_next_token',side_effect=AssertionError('cached EOS must stop generation')):
                ended=Snapshot(self.cache_adapter(20),{'target':' cat'})
            self.assertTrue(ended.eos_stopped)
            self.assertEqual(ended.stop,7)
            with patch.object(Snapshot,'_next_token',side_effect=[8,63]):
                fresh=Snapshot(first,{'target':' cat','regenerate':True})
            self.assertEqual(fresh.result()['generatedTokens'],['8'])
            self.assertFalse(fresh.activations_reused)

    def test_target_and_lens_changes_reuse_captures_without_forward_pass(self):
        adapter=self.adapter()
        first=Snapshot(adapter,{'target':' cat'})
        selected={'id':'test-j','method':'j','label':'Test J','enabled':True}
        with patch.object(Snapshot,'_complete_tokens',side_effect=AssertionError('must not generate')), \
             patch.object(Snapshot,'_capture',side_effect=AssertionError('must not recapture')):
            other=Snapshot(adapter,{'target':'two pieces','tokenId':4})
            self.assertTrue(other.activations_reused)
            self.assertIs(first.vectors['post:0'],other.vectors['post:0'])
            self.assertNotEqual(first.rows[0]['scores'],other.rows[0]['scores'])
            with patch('llmdbg.concept_trace.lens',return_value=selected), \
                 patch.object(adapter.execution,'_workspace_lens_source',return_value=(torch.eye(32),{})):
                changed_lens=Snapshot(adapter,{'target':' cat','mode':'lens:test-j'})
            self.assertTrue(changed_lens.activations_reused)
            self.assertEqual(changed_lens.method,'j')

    def test_conflicting_live_continuation_is_not_persisted_and_regenerate_restarts_prompt(self):
        from llmdbg.generation_cache import load
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ,{'LLMDBG_GENERATION_CACHE_DIR':directory}):
            with patch.object(Snapshot,'_next_token',side_effect=[5,63]):
                original=Snapshot(self.cache_adapter(),{'target':' cat'})
            adapter=self.cache_adapter(3)
            adapter.execution.append_generated_token(9)
            adapter.generated_tokens=1
            with patch.object(Snapshot,'_next_token',side_effect=[10,63]):
                conflict=Snapshot(adapter,{'target':' cat'})
            self.assertFalse(conflict.persistent_cache)
            self.assertEqual(load(original.generation_key,64)['ids'],[5])
            with patch.object(Snapshot,'_next_token',side_effect=[11,63]):
                fresh=Snapshot(adapter,{'target':' cat','regenerate':True})
            self.assertEqual(fresh.result()['generatedTokens'],['11'])
            self.assertEqual(adapter.execution.input_ids[0,-1].item(),9)

    def test_cache_identity_changes_with_inputs_revision_and_interventions_not_source(self):
        from llmdbg.generation_cache import identity
        adapter=self.cache_adapter()
        key,persistent,_=identity(adapter,4)
        self.assertTrue(persistent)
        adapter.source_path='/different/source.llmdbg.py'
        self.assertEqual(identity(adapter,4)[0],key)
        adapter.execution.rendered_prompt='different thinking template'
        self.assertNotEqual(identity(adapter,4)[0],key)
        adapter.execution.rendered_prompt=''
        adapter.execution.runtime.model.config._commit_hash='other-checkpoint'
        self.assertNotEqual(identity(adapter,4)[0],key)
        adapter.execution.runtime.model.config._commit_hash='unit-test-checkpoint'
        adapter.execution.input_ids[0,0]=9
        self.assertNotEqual(identity(adapter,4)[0],key)
        adapter.execution.interventions[(0,'model.layers.0.mlp','input',0)]=0.
        self.assertFalse(identity(adapter,4)[1])

    def test_snapshot_accepts_5000_positions_and_rejects_5001(self):
        adapter = self.adapter()
        adapter.max_new_tokens = 6000
        with patch.object(Snapshot, '_complete_tokens', side_effect=RuntimeError('range accepted')):
            for stop in (2049, 5000):
                with self.assertRaisesRegex(RuntimeError, 'range accepted'):
                    Snapshot(adapter, {'target':' cat', 'start':0, 'stop':stop})
            with self.assertRaisesRegex(ValueError, 'at most 5000'):
                Snapshot(adapter, {'target':' cat', 'start':0, 'stop':5001})

    def adapter(self, architecture='qwen'):
        if architecture == 'gpt2':
            from transformers import GPT2Config, GPT2LMHeadModel
            config = GPT2Config(vocab_size=64, n_embd=32, n_layer=2, n_head=4, n_positions=32)
            with torch.random.fork_rng():
                torch.manual_seed(9); model = GPT2LMHeadModel(config).eval()
        else:
            from transformers import Qwen3_5TextConfig, Qwen3_5ForCausalLM
            config = Qwen3_5TextConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
                num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=8,
                linear_value_head_dim=8, layer_types=['linear_attention','full_attention'])
            with torch.random.fork_rng():
                torch.manual_seed(9); model = Qwen3_5ForCausalLM(config).eval()
        tokenizer = SimpleNamespace(encode=lambda text, **kw: [3] if text == ' cat' else [3,4],
                                    decode=lambda ids: ' cat' if ids[0] == 3 else str(ids[0]))
        execution = ModelExecution(ModelRuntime(model, tokenizer), torch.tensor([[1,2,3,4]]),
                                   ('one','two','three','four'), 'cpu', lambda _:None)
        adapter = LLMDebugAdapter(ProtocolStream(io.BytesIO(),io.BytesIO()))
        adapter.execution, adapter.tokens, adapter.model_name, adapter.token_index = execution, execution.tokens, 'test/model', 3
        return adapter

    def test_score_is_cosine_of_actual_lens_pipeline_not_token_embeddings(self):
        adapter = self.adapter()
        vectors = torch.randn(5,32)
        mapping = torch.randn(32,32)
        scores = cosine_scores(adapter.execution,vectors,3,mapping)
        norm = adapter.execution.runtime.final_norm
        weight = adapter.execution.runtime.model.get_output_embeddings().weight
        decoded = norm((vectors @ mapping.T)[:,None,:])[:,0]
        expected = torch.nn.functional.cosine_similarity(decoded,weight[3][None,:],dim=-1)
        torch.testing.assert_close(torch.tensor(scores),expected)
        support, score, bias = target_support(adapter.execution,vectors,3,mapping)
        self.assertAlmostEqual(float(support.sum())+bias,score,places=5)

    def test_native_replay_heads_edges_and_inspection_preserve_paused_execution(self):
        adapter=self.adapter()
        execution=adapter.execution
        execution.trace_token(3)
        cache, through=execution.past_key_values,execution.cached_through
        memory=cache.layers[0].recurrent_states[0].clone()
        with patch.object(execution,'vector_lens',wraps=execution.vector_lens):
            result=dispatch(adapter,{'target':' cat','mode':'logit','start':0,'stop':4})
        snapshot=adapter._concept_trace_snapshot
        self.assertEqual([row['id'] for row in result['rows']],['embed','post:0','post:1'])
        self.assertTrue(all(len(row['scores'])==4 for row in result['rows']))
        self.assertIs(execution.past_key_values,cache)
        self.assertEqual(execution.cached_through,through)
        torch.testing.assert_close(cache.layers[0].recurrent_states[0],memory,atol=0,rtol=0)
        for layer in (0,1):
            rows=snapshot.heads(layer)['rows']
            self.assertEqual(len(rows),8)
            self.assertEqual([row['kind'] for row in rows[:2]],['value','output'])
            inspected=snapshot.inspect(f'output:{layer}:1',3)
            self.assertEqual(len(inspected['tokens']),10)
            edges=snapshot.edges(layer,1,3)
            self.assertEqual(len(edges['rows']),4)
            self.assertAlmostEqual(edges['supportSum']+edges['normalizationBiasSupport'],edges['targetScore'],places=5)
            self.assertAlmostEqual(edges['targetScore'],inspected['score'],places=4)
            self.assertIs(snapshot.edges(layer,1,3),edges)
        self.assertIs(execution.past_key_values,cache)
        torch.testing.assert_close(cache.layers[0].recurrent_states[0],memory,atol=0,rtol=0)
        self.assertEqual(snapshot.inspect('post:0',2)['position'],2)
        with self.assertRaisesRegex(ValueError,'expired'):
            dispatch(adapter,{'action':'inspect','snapshotId':'old','rowId':'post:0','position':2})

    def test_gpt2_residuals_value_heads_and_final_norm_bias(self):
        adapter=self.adapter('gpt2')
        result=Snapshot(adapter,{'target':' cat','mode':'logit','start':1,'stop':4})
        self.assertEqual(len(result.result()['rows'][0]['scores']),3)
        self.assertEqual(len(result.heads(0)['rows']),8)
        self.assertEqual(result.inspect('value:0:2',2)['position'],2)
        parts=torch.randn(4,32)
        adapter.execution.runtime.final_norm.bias.data.fill_(.3)
        support,score,bias=target_support(adapter.execution,parts,3)
        self.assertAlmostEqual(float(support.sum())+bias,score,places=5)
        with self.assertRaisesRegex(ValueError,'GPT-2'):
            result.edges(0,0,2)

    def test_missing_maps_are_hollow_not_zero_and_hooks_restore_on_failure(self):
        adapter=self.adapter()
        selected={'id':'test-j','method':'j','label':'Test J','enabled':True}
        with patch('llmdbg.concept_trace.lens',return_value=selected), patch.object(adapter.execution,'_workspace_lens_source',return_value=None):
            result=Snapshot(adapter,{'target':' cat','mode':'lens:test-j'})
            self.assertTrue(all(row['scores']==[None]*4 and not row['available'] for row in result.rows))
        modules=list(adapter.execution.runtime.model.modules())
        counts=[(len(m._forward_pre_hooks),len(m._forward_hooks)) for m in modules]
        with patch.object(ModelExecution,'_prefill',side_effect=RuntimeError('failure')):
            with self.assertRaisesRegex(RuntimeError,'failure'):
                Snapshot(adapter,{'target':' cat','mode':'logit','regenerate':True})
        self.assertEqual(counts,[(len(m._forward_pre_hooks),len(m._forward_hooks)) for m in modules])

    def test_target_token_and_range_validation(self):
        adapter=self.adapter()
        self.assertEqual(target_tokens(adapter.execution,' cat'),[{'id':3,'text':' cat'}])
        for args in ({'target':''},{'target':'two pieces'},{'target':' cat','tokenId':4},
                     {'target':' cat','start':2,'stop':1},{'target':' cat','stop':10}):
            with self.assertRaises(ValueError): Snapshot(adapter,args)
        snapshot=Snapshot(adapter,{'target':'two pieces','tokenId':4})
        self.assertEqual(snapshot.token_id,4)

    def test_authored_generation_budget_extends_private_replay_only(self):
        adapter=self.adapter()
        adapter.max_new_tokens=3
        adapter.execution.runtime.tokenizer.eos_token_id=63
        execution=adapter.execution
        execution.trace_token(3)
        cache,through=execution.past_key_values,execution.cached_through
        ids=execution.input_ids.clone();tokens=execution.tokens
        with patch.object(Snapshot,'_next_token',side_effect=[5,6,7]) as next_token:
            snapshot=Snapshot(adapter,{'target':' cat','mode':'logit','maxNewTokens':1000})
        self.assertEqual(next_token.call_count,3)
        result=snapshot.result()
        self.assertEqual((result['promptTokenCount'],result['maxNewTokens'],result['generatedTokenCount']),(4,3,3))
        self.assertEqual(result['newlyGeneratedTokens'],3)
        self.assertEqual(result['generatedTokenStart'],4)
        self.assertEqual(result['generatedTokens'],['5','6','7'])
        self.assertEqual(''.join(result['generatedTokens']),result['generatedText'])
        self.assertEqual(result['stop'],7)
        self.assertEqual(len(result['rows'][0]['scores']),7)
        self.assertEqual(len(snapshot.heads(0)['rows'][0]['scores']),7)
        self.assertEqual(snapshot.inspect('post:0',6)['token'],'7')
        self.assertEqual(len(snapshot.edges(0,1,6)['rows']),7)
        self.assertIs(execution.past_key_values,cache)
        self.assertEqual(execution.cached_through,through)
        self.assertEqual(execution.tokens,tokens)
        torch.testing.assert_close(execution.input_ids,ids)
        self.assertEqual(adapter.generated_tokens,0)
        with self.assertRaisesRegex(ValueError,'max_new_tokens=3'):
            Snapshot(adapter,{'target':' cat','stop':8})

    def test_text_only_embeddings_generate_without_explicit_position_ids(self):
        adapter=self.adapter()
        execution=adapter.execution
        execution.inputs_embeds=execution.runtime.text_model.get_input_embeddings()(execution.input_ids).detach()
        self.assertIsNone(execution.position_ids)
        original=execution.inputs_embeds.clone()
        adapter.max_new_tokens=2
        execution.runtime.tokenizer.eos_token_id=63
        with patch.object(Snapshot,'_next_token',side_effect=[5,6]):
            snapshot=Snapshot(adapter,{'target':' cat','mode':'logit'})
        self.assertEqual(snapshot.result()['newlyGeneratedTokens'],2)
        self.assertEqual(snapshot.replay.input_ids.shape[1],6)
        self.assertIsNone(snapshot.replay.position_ids)
        self.assertEqual(len(snapshot.result()['rows'][0]['scores']),6)
        torch.testing.assert_close(execution.inputs_embeds,original)
        self.assertEqual(execution.input_ids.shape[1],4)

    def test_existing_generated_tokens_count_toward_the_total_budget(self):
        adapter=self.adapter()
        adapter.max_new_tokens=3
        adapter.generated_tokens=1
        adapter.execution.runtime.tokenizer.eos_token_id=63
        with patch.object(Snapshot,'_next_token',side_effect=[5,6]) as next_token:
            result=Snapshot(adapter,{'target':' cat'}).result()
        self.assertEqual(next_token.call_count,2)
        self.assertEqual((result['promptTokenCount'],result['generatedTokenCount'],result['newlyGeneratedTokens']),(3,3,2))
        self.assertEqual(result['stop'],6)
        self.assertEqual(adapter.generated_tokens,1)

    def test_eos_stops_before_budget_and_explicit_prompt_ranges_do_not_generate(self):
        adapter=self.adapter()
        adapter.max_new_tokens=3
        adapter.execution.runtime.tokenizer.eos_token_id=63
        with patch.object(Snapshot,'_next_token',side_effect=[5,63]) as next_token:
            result=Snapshot(adapter,{'target':' cat'}).result()
        self.assertEqual(next_token.call_count,2)
        self.assertTrue(result['stoppedAtEos'])
        self.assertEqual(result['generatedTokenCount'],1)
        self.assertEqual(result['stop'],5)
        with patch.object(Snapshot,'_next_token',side_effect=AssertionError('unexpected generation')):
            result=Snapshot(adapter,{'target':' cat','stop':3}).result()
        self.assertEqual(result['stop'],3)
        self.assertEqual(result['newlyGeneratedTokens'],0)


if __name__=='__main__':unittest.main()
