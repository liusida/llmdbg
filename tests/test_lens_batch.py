import io
import unittest
from unittest.mock import patch

from llmdbg.dap import LLMDebugAdapter, ProtocolStream, LensUnavailable, DebugActionError
from test_dap import frame, parse_frames


class LensBatchTest(unittest.TestCase):
    def test_batched_r_readout_matches_individual_vectors_with_one_normalization(self):
        import torch
        from types import SimpleNamespace
        from transformers import Qwen3Config, Qwen3ForCausalLM
        from llmdbg.model_runtime import ModelRuntime, ModelExecution
        config = Qwen3Config(vocab_size=64, hidden_size=16, intermediate_size=32,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, head_dim=8)
        with torch.random.fork_rng():
            torch.manual_seed(3)
            model = Qwen3ForCausalLM(config).eval()
            vectors = tuple(tuple(row.tolist()) for row in torch.randn(5, 16))
            mapping = torch.randn(16, 16)
        execution = ModelExecution(ModelRuntime(model, SimpleNamespace(decode=lambda ids: str(ids[0]))),
            torch.tensor([[1]]), ('one',), 'cpu', lambda _: None)
        flags = dict(source_site='mid', source_layer=0, token_count=5,
                     include_logit_lens=False, include_j_lens=False, include_r_lens=True)
        with patch.object(execution, '_workspace_lens_source', return_value=(mapping, {})), patch.object(
            execution.runtime.final_norm, 'forward', wraps=execution.runtime.final_norm.forward) as normalize:
            batch = execution._vectors_lens('model.layers.0.full_attention', vectors, **flags)
            self.assertEqual(normalize.call_count, 1)
            individual = [execution.vector_lens('model.layers.0.full_attention', vector, **flags) for vector in vectors]
            self.assertEqual(normalize.call_count, 6)
        for left, right in zip(batch, individual, strict=True):
            self.assertEqual([t.token_id for t in left.r_lens], [t.token_id for t in right.r_lens])
            torch.testing.assert_close(torch.tensor([t.score for t in left.r_lens]),
                                       torch.tensor([t.score for t in right.r_lens]), atol=1e-5, rtol=1e-5)

    def adapter(self, requests):
        output = io.BytesIO()
        adapter = LLMDebugAdapter(ProtocolStream(io.BytesIO(b''.join(frame(r) for r in requests)), output))
        return adapter, output

    def test_wire_batch_discovers_once_caches_sections_and_has_no_routine_logs(self):
        views = ['residual-pre', 'attention-head-values', 'attention-heads', 'attention-output',
                 'residual-mid', 'neurons', 'mlp-output', 'residual-post']
        entries = [{'mode': 'lens:r', 'view': view} for view in views]
        requests = [{'seq': i, 'type': 'request', 'command': 'llmdbg/neuronLensBatch',
                     'arguments': {'requests': entries}} for i in (1, 2)]
        requests.append({'seq': 3, 'type': 'request', 'command': 'llmdbg/neuronLens',
                         'arguments': entries[0]})
        adapter, output = self.adapter(requests)
        with patch('llmdbg.residual_vocabulary_lenses.available_lenses', return_value=(
            {'id': 'r', 'method': 'r', 'label': 'R-Lens'},)) as discovery, patch.object(
            adapter, '_neuron_lens_snapshot', side_effect=lambda **kw: {'view': kw['view']}) as compute:
            adapter.run()
        self.assertEqual(discovery.call_count, 3)  # Once per request, never per section.
        self.assertEqual(compute.call_count, 8)  # Repeated batch and legacy request hit cache.
        self.assertTrue(all(call.kwargs['include_r_lens'] for call in compute.call_args_list))
        messages = parse_frames(output.getvalue())
        self.assertTrue(all(message['type'] == 'response' and message['success'] for message in messages))
        self.assertEqual([item['view'] for item in messages[0]['body']['items']], views)
        self.assertEqual(messages[0]['body'], messages[1]['body'])
        self.assertEqual(messages[2]['body'], messages[0]['body']['items'][0]['item'])

    def test_batch_keeps_valid_sections_when_others_are_unavailable_or_invalid(self):
        adapter, _ = self.adapter([])
        def compute(**kwargs):
            if kwargs['view'] == 'missing':
                raise LensUnavailable('not reached yet')
            if kwargs['view'] == 'invalid':
                raise DebugActionError('invalid view')
            return {'view': kwargs['view']}
        entries = [{'mode': 'logit', 'view': view} for view in ('good', 'missing', 'invalid', 'missing')]
        with patch('llmdbg.residual_vocabulary_lenses.available_lenses', return_value=()), patch.object(
            adapter, '_neuron_lens_snapshot', side_effect=compute) as calls:
            result = adapter._lens_readout_batch(entries)['items']
        self.assertEqual(calls.call_count, 3)
        self.assertEqual(result[0]['item']['view'], 'good')
        self.assertTrue(result[1]['item']['unavailable'])
        self.assertEqual(result[2]['error'], 'invalid view')
        self.assertEqual(result[1], result[3])

    def test_batch_validates_before_discovery_and_empty_batch_is_free(self):
        adapter, _ = self.adapter([])
        with patch('llmdbg.residual_vocabulary_lenses.available_lenses') as discovery:
            self.assertEqual(adapter._lens_readout_batch([]), {'items': []})
            for invalid in ({}, [None], [{'view': 'neurons'}], [{'mode': 3, 'view': 'neurons'}], [{}]*257):
                with self.subTest(invalid=type(invalid).__name__), self.assertRaises(DebugActionError):
                    adapter._lens_readout_batch(invalid)
            discovery.assert_not_called()
