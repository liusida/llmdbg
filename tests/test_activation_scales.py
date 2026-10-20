from __future__ import annotations

import unittest

from llmdbg.activation_scales import mlp_activation_scale


class ActivationScalesTest(unittest.TestCase):
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
