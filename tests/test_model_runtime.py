from __future__ import annotations

import unittest

import llmdbg.model_runtime as model_runtime


class ModelRuntimeCacheTest(unittest.TestCase):
    def test_qwen_runtime_exposes_hybrid_layer_names(self) -> None:
        from types import SimpleNamespace

        model = SimpleNamespace(
            config=SimpleNamespace(
                model_type="qwen3_5",
                layer_types=["linear_attention", "full_attention"],
            ),
            model=SimpleNamespace(layers=[object(), object()], norm=object()),
        )
        runtime = model_runtime.ModelRuntime(model=model, tokenizer=object())

        self.assertEqual(runtime.architecture, "qwen3_5")
        self.assertEqual(runtime.layer_kind(0), "gated_linear_attention")
        self.assertEqual(runtime.layer_kind(1), "full_attention")
        self.assertEqual(
            runtime.state_name(1, runtime.layer_kind(1)),
            "model.layers.1.full_attention",
        )
        self.assertEqual(runtime.final_norm_name, "model.norm")

    def test_matching_runtime_is_reused_without_reloading_dependencies(self) -> None:
        runtime = model_runtime.ModelRuntime(model=object(), tokenizer=object())
        original_runtime = model_runtime._runtime
        original_key = model_runtime._runtime_key
        messages: list[str] = []
        try:
            model_runtime._runtime = runtime
            model_runtime._runtime_key = ("gpt2", "main", "cuda", "bfloat16")
            result = model_runtime._get_model_runtime(
                "gpt2",
                revision="main",
                device="cuda",
                dtype="bfloat16",
                progress=messages.append,
            )
        finally:
            model_runtime._runtime = original_runtime
            model_runtime._runtime_key = original_key

        self.assertIs(result, runtime)
        self.assertIn("Reusing resident gpt2", messages[0])


if __name__ == "__main__":
    unittest.main()
