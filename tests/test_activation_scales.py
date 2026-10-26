from __future__ import annotations

import unittest

from llmdbg.activation_scales import (
    attention_channel_activation_scale,
    attention_head_norm_scale,
    mlp_activation_scale,
)


class ActivationScalesTest(unittest.TestCase):
    def test_qwen35_2b_value_references_cover_all_heads_separately(self) -> None:
        for layer in range(24):
            delta=layer%4!=3
            name=f'model.layers.{layer}.'+('gated_linear_attention' if delta else 'full_attention')
            for head in range(16 if delta else 8):
                value=attention_head_norm_scale('Qwen/Qwen3.5-2B',name,head,value=True)
                output=attention_head_norm_scale('Qwen/Qwen3.5-2B',name,head)
                self.assertIsNotNone(value)
                self.assertGreater(value['scale'],0)
                self.assertEqual(value['head_dimension'],128 if delta else 256)
                self.assertEqual(value['analyzed_tokens'],444)
                self.assertNotEqual(value['scale'],output['scale'])

    def test_all_packaged_attention_models_have_complete_head_references(self) -> None:
        import math
        from llmdbg.activation_scales import _ATTENTION_SCALE_FILES, _model_attention_scales
        for model in _ATTENTION_SCALE_FILES:
            records = _model_attention_scales(model)
            self.assertTrue(records)
            for layer, record in records.items():
                if record['attention_kind'] == 'gated_linear_attention':
                    if model != 'Qwen/Qwen3.5-2B':
                        continue  # Other checkpoints still need a fresh DeltaNet run.
                    self.assertEqual(len(record.get('head_norm_scales', [])), 16)
                    self.assertTrue(all(h['head_dimension'] == 128 for h in record['head_norm_scales']))
                heads = record.get('head_norm_scales', [])
                self.assertTrue(heads, (model, layer))
                self.assertEqual([head['head'] for head in heads], list(range(len(heads))))
                suffix = 'attention' if model.startswith('openai-community/') else 'full_attention'
                if record['attention_kind'] == 'gated_linear_attention':
                    suffix = 'gated_linear_attention'
                prefix = 'transformer.h' if suffix == 'attention' else 'model.layers'
                for head in heads:
                    scale = attention_head_norm_scale(model, f'{prefix}.{layer}.{suffix}', head['head'])
                    self.assertIsNotNone(scale, (model, layer, head['head']))
                    self.assertTrue(math.isfinite(scale['scale']) and scale['scale'] > 0)

    def test_projected_head_calibration_matches_explicit_projection(self) -> None:
        import runpy
        from pathlib import Path
        import torch
        measure = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/measure_attention_channel_activation_scales.py"))["projected_head_references"]
        # Opposing directions must cancel within a head, not sum channel norms.
        values = torch.tensor([[1., 1., 2., 1.], [2., 1., 3., 2.], [3., 1., 4., 3.]]).repeat(23, 1)
        weight = torch.tensor([[1., 0.], [-1., 0.], [0., 1.], [0., 2.]])
        records = measure(values, weight, 2)
        self.assertEqual(len(records), 2)
        for head, record in enumerate(records):
            start = head * 2
            norms = (values[:, start:start+2] @ weight[start:start+2]).norm(dim=-1)
            self.assertAlmostEqual(record["scale"], torch.quantile(norms, .5).item())
            self.assertAlmostEqual(record["token_norm_percentiles"]["p99"], torch.quantile(norms, .99).item())
            self.assertEqual(record["head"], head)
            self.assertEqual(record["analyzed_tokens"], 69)
            self.assertEqual(record["head_dimension"], 2)
        self.assertEqual(records[0]["scale"], 1.)
        for bad_values, bad_weight, count in ((values[:0], weight, 2), (values, weight, 3), (values, weight[:2], 2)):
            with self.assertRaises(ValueError):
                measure(bad_values, bad_weight, count)

    def test_head_norm_lookup_requires_exact_positive_reference(self) -> None:
        from unittest.mock import patch
        with patch("llmdbg.activation_scales.attention_channel_activation_scale", return_value={"scale": 1.}):
            self.assertIsNone(attention_head_norm_scale("model", "layer", 0))
        self.assertIsNone(attention_head_norm_scale("unknown", "model.layers.0.full_attention", 0))
        record = {"head_norm_scales": [{"head": 1, "scale": 2.5}, {"head": 2, "scale": 0.}, {"head": 3, "scale": float("nan")}]}
        with patch("llmdbg.activation_scales.attention_channel_activation_scale", return_value=record):
            self.assertEqual(attention_head_norm_scale("model", "layer", 1)["scale"], 2.5)
            self.assertIsNone(attention_head_norm_scale('model','layer',1,value=True))
            for head in (0, 2, 3, -1, True):
                self.assertIsNone(attention_head_norm_scale("model", "layer", head))

    def test_gpt2_medium_has_its_own_scales_for_all_24_layers(self) -> None:
        import math
        for layer in range(24):
            for head in range(16):
                head_record = attention_head_norm_scale('openai-community/gpt2-medium', f'transformer.h.{layer}.attention', head)
                self.assertIsNotNone(head_record)
                self.assertGreater(head_record['scale'], 0)
                self.assertEqual(head_record['scale'], head_record['token_norm_percentiles']['p50'])
            for lookup, suffix in ((attention_channel_activation_scale, "attention"), (mlp_activation_scale, "mlp")):
                record = lookup("openai-community/gpt2-medium", f"transformer.h.{layer}.{suffix}")
                self.assertIsNotNone(record)
                self.assertEqual(record["layer"], layer)
                self.assertTrue(math.isfinite(record["scale"]) and record["scale"] > 0)
                self.assertIn(record["side"], {"positive", "negative"})
                self.assertIsNone(lookup("openai-community/gpt2-medium", f"model.layers.{layer}.{suffix}"))

    def test_gemma4_calibrations_cover_all_layers_and_not_base_checkpoint(self) -> None:
        import math
        for layer in range(35):
            for lookup, suffix in ((attention_channel_activation_scale, "full_attention"), (mlp_activation_scale, "mlp")):
                record = lookup("google/gemma-4-E2B-it", f"model.layers.{layer}.{suffix}")
                self.assertIsNotNone(record)
                self.assertTrue(math.isfinite(record["scale"]) and record["scale"] > 0)
                self.assertIsNone(lookup("google/gemma-4-E2B", f"model.layers.{layer}.{suffix}"))

    def test_attention_channel_scales_cover_every_supported_model(self) -> None:
        layer_counts = {
            "openai-community/gpt2": 12,
            "Qwen/Qwen3.5-0.8B": 24,
            "Qwen/Qwen3.5-2B": 24,
            "Qwen/Qwen3.5-9B": 32,
            "Qwen/Qwen3.5-27B": 64,
        }
        for model_id, layer_count in layer_counts.items():
            records = [
                attention_channel_activation_scale(
                    model_id,
                    (
                        f"transformer.h.{layer}.attention"
                        if model_id == "openai-community/gpt2"
                        else f"model.layers.{layer}."
                        + ("full_attention" if layer % 4 == 3 else "gated_linear_attention")
                    ),
                )
                for layer in range(layer_count)
            ]
            with self.subTest(model=model_id):
                self.assertTrue(all(record is not None for record in records))
                self.assertTrue(all(float(record["scale"]) > 0 for record in records))
                self.assertTrue(
                    all(record["side"] in {"positive", "negative"} for record in records)
                )
        self.assertIsNone(
            attention_channel_activation_scale(
                "Qwen/Qwen3.5-2B", "model.layers.0.mlp"
            )
        )

    def test_supported_qwen35_models_have_every_layer(self) -> None:
        layer_counts = {
            "openai-community/gpt2": 12,
            "Qwen/Qwen3.5-0.8B": 24,
            "Qwen/Qwen3.5-2B": 24,
            "Qwen/Qwen3.5-9B": 32,
            "Qwen/Qwen3.5-27B": 64,
        }
        for model_id, layer_count in layer_counts.items():
            with self.subTest(model=model_id):
                records = [
                    mlp_activation_scale(
                        model_id,
                        (
                            f"transformer.h.{layer}.mlp"
                            if model_id == "openai-community/gpt2"
                            else f"model.layers.{layer}.mlp"
                        ),
                    )
                    for layer in range(layer_count)
                ]
                self.assertTrue(all(record is not None for record in records))
                self.assertTrue(all(float(record["scale"]) > 0 for record in records))
                self.assertTrue(
                    all(
                        abs(
                            float(record["scale"])
                            - float(record["token_scale_percentiles"]["p50"])
                        )
                        < 1e-6
                        for record in records
                    )
                )
                self.assertTrue(
                    all(record["side"] in {"positive", "negative"} for record in records)
                )

    def test_unknown_model_or_layer_is_not_silently_approximated(self) -> None:
        self.assertIsNone(
            mlp_activation_scale("Qwen/Qwen3.5-2B-Base", "model.layers.0.mlp")
        )
        self.assertIsNone(
            mlp_activation_scale("Qwen/Qwen3.5-2B", "model.layers.99.mlp")
        )
        self.assertIsNone(mlp_activation_scale("other/model", "model.layers.0.mlp"))


if __name__ == "__main__":
    unittest.main()
