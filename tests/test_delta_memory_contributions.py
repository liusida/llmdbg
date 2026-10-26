from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from llmdbg.model_runtime import ModelExecution, ModelRuntime, delta_write_readout_contributions


class DeltaMemoryContributionsTest(unittest.TestCase):
    def test_decomposition_reconstructs_recurrence_and_accounts_for_overwriting(self):
        with torch.random.fork_rng():
            torch.manual_seed(4)
            keys = torch.nn.functional.normalize(torch.randn(13, 5), dim=-1)
            values = torch.randn(13, 3)
            decays, betas, query = torch.rand(13), torch.rand(13), torch.randn(5)
        memory = torch.zeros(5, 3)
        for k, v, a, b in zip(keys, values, decays, betas):
            memory = a * memory
            memory = memory + b * k[:, None] * (v - k @ memory)[None, :]
        rows = delta_write_readout_contributions(keys, values, decays, betas, query)
        torch.testing.assert_close(rows.sum(0), query @ memory, atol=1e-6, rtol=1e-5)
        rows = delta_write_readout_contributions(torch.tensor([[1., 0.], [1., 0.]]),
            torch.tensor([[2., 3.], [7., 8.]]), torch.ones(2), torch.ones(2), torch.tensor([1., 0.]))
        torch.testing.assert_close(rows, torch.tensor([[0., 0.], [7., 8.]]))

    def test_on_demand_replay_preserves_cache_and_caches_only_matching_inputs(self):
        from transformers import Qwen3_5TextConfig, Qwen3_5ForCausalLM
        config = Qwen3_5TextConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
            linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=8,
            linear_value_head_dim=8, layer_types=['linear_attention', 'full_attention'])
        with torch.random.fork_rng():
            torch.manual_seed(5)
            model = Qwen3_5ForCausalLM(config).eval()
        execution = ModelExecution(ModelRuntime(model, SimpleNamespace(decode=lambda ids:str(ids[0]))),
            torch.tensor([[1, 2, 3, 4]]), ('a', 'b', 'c', 'd'), 'cpu', lambda _:None)
        self.assertFalse(hasattr(execution, '_delta_contribution_result'))
        execution.trace_token(3)
        cache, through = execution.past_key_values, execution.cached_through
        memory = cache.layers[0].recurrent_states[0].clone()
        observations = dict(execution.ica_observations)
        result = execution.delta_memory_contributions(0, 2, 3)
        self.assertEqual(len(result['rows']), 4)
        self.assertAlmostEqual(result['sum'], result['outputNorm'], places=4)
        self.assertLess(result['reconstructionError'], 1e-5)
        self.assertIs(execution.past_key_values, cache)
        self.assertEqual(execution.cached_through, through)
        self.assertEqual(execution.ica_observations, observations)
        torch.testing.assert_close(cache.layers[0].recurrent_states[0], memory)
        with patch.object(ModelExecution, 'trace_token', side_effect=AssertionError('unexpected replay')):
            self.assertIs(execution.delta_memory_contributions(0, 2, 3), result)
        execution.input_ids = torch.tensor([[1, 2, 3, 5]])
        changed = execution.delta_memory_contributions(0, 2, 3)
        self.assertIsNot(changed, result)
        torch.testing.assert_close(cache.layers[0].recurrent_states[0], memory)
        import importlib
        implementation = importlib.import_module(type(model.model.layers[0].linear_attn).__module__)
        original = implementation.torch_chunk_gated_delta_rule
        mixer = model.model.layers[0].linear_attn
        pre_count, post_count = len(mixer._forward_pre_hooks), len(mixer._forward_hooks)
        with patch.object(ModelExecution, 'trace_token', side_effect=RuntimeError('replay failed')):
            with self.assertRaisesRegex(RuntimeError, 'replay failed'):
                execution.delta_memory_contributions(0, 2, 2)
        self.assertIs(implementation.torch_chunk_gated_delta_rule, original)
        self.assertEqual((len(mixer._forward_pre_hooks),len(mixer._forward_hooks)),(pre_count,post_count))
        # Changed intervention state cannot reuse the earlier cached attribution.
        execution.interventions[(3,'model.layers.0.gated_linear_attention','input',0)] = 0.
        with patch.object(ModelExecution, 'trace_token', side_effect=RuntimeError('new replay')):
            with self.assertRaisesRegex(RuntimeError, 'new replay'):
                execution.delta_memory_contributions(0, 2, 3)
        for layer, head, position in ((1,0,3),(0,4,3),(0,0,4)):
            with self.assertRaises(ValueError): execution.delta_memory_contributions(layer,head,position)


if __name__ == '__main__':
    unittest.main()
