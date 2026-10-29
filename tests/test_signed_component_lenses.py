"""Activated neuron/channel labels must follow the actual signed residual vector."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from llmdbg.model_runtime import ModelExecution, ModelRuntime


class SignedComponentLensTest(unittest.TestCase):
    def test_all_lenses_decode_signed_component_and_keep_raw_direction_watch(self):
        unembedding = torch.tensor([[1., -1.], [-1., 1.]])
        model = SimpleNamespace(config=SimpleNamespace(model_type='qwen3', _name_or_path='fake/model'),
            model=SimpleNamespace(norm=torch.nn.RMSNorm(2, eps=1e-6)),
            get_output_embeddings=lambda: SimpleNamespace(weight=unembedding))
        execution = ModelExecution(ModelRuntime(model, SimpleNamespace(decode=lambda ids: ['heat', 'ice'][ids[0]])),
            None, ('one',), 'cpu', lambda _: None)
        directions = torch.tensor([[1., -1.]])
        with patch.object(execution, '_workspace_lens_source', return_value=(torch.eye(2), {})), patch.object(
            execution, '_semantic_token_mask', return_value=torch.ones(2, dtype=torch.bool)):
            for activation, expected in ((2., 0), (-3., 1), (0., None)):
                with self.subTest(activation=activation):
                    result = execution._top_projected_components(0, (activation,), directions,
                        component_count=1, token_count=2, include_logit_lens=True,
                        include_j_lens=True, include_r_lens=True, include_jpp_lens=True)[0]
                    # Explicit positive-direction vocabulary remains unchanged.
                    self.assertEqual(result.vocabulary[0].token_id, 0)
                    self.assertEqual(result.vector, tuple((activation*directions[0]).tolist()))
                    if expected is None:
                        self.assertEqual(result.logit_lens, ())
                        self.assertEqual(result.j_lens, ())
                        self.assertEqual(result.r_lens, ())
                        self.assertEqual(result.jpp_lens, ())
                        continue
                    direct = execution.vector_lens('model.layers.0.mlp', result.vector,
                        token_count=2, include_logit_lens=True, include_j_lens=True,
                        include_r_lens=True, include_jpp_lens=True)
                    for attribute, reference in (('logit_lens', 'vocabulary'), ('j_lens', 'j_lens'),
                                                 ('r_lens', 'r_lens'), ('jpp_lens', 'jpp_lens')):
                        tokens = getattr(result, attribute)
                        self.assertEqual(tokens[0].token_id, expected)
                        self.assertEqual([t.token_id for t in tokens], [t.token_id for t in getattr(direct, reference)])
                        torch.testing.assert_close(torch.tensor([t.score for t in tokens]),
                            torch.tensor([t.score for t in getattr(direct, reference)]))
                    self.assertEqual(result.contribution[0].token_id, expected)
