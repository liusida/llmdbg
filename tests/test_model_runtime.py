from __future__ import annotations

import unittest

import llmdbg.model_runtime as model_runtime


class ModelRuntimeCacheTest(unittest.TestCase):
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
