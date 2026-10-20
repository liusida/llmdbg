from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from llmdbg.config import load_debug_config


class DebugConfigTest(unittest.TestCase):
    def test_loads_ordered_llmdbg_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gpt2.llmdbg"
            path.write_text(
                """
model = "openai-community/gpt2"
revision = "main"
device = "cuda"
dtype = "bfloat16"
prompt = "The capital of France is"
stop_on_entry = true

submodule = "token_embedding"
layer = 0
submodule = "transformer.h.0.attention"
submodule = "transformer.h.0.mlp"
""".strip()
            )
            config = load_debug_config(str(path))

        self.assertEqual(config["model"], "openai-community/gpt2")
        self.assertEqual(config["prompt"], "The capital of France is")
        self.assertTrue(config["stopOnEntry"])
        self.assertEqual(
            config["stepPaths"],
            ["token_embedding", "transformer.h.0.attention", "transformer.h.0.mlp"],
        )

    def test_rejects_missing_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.llmdbg"
            path.write_text('prompt = "hello"\nsubmodule = "token_embedding"\n')
            with self.assertRaisesRegex(ValueError, r"model must be assigned"):
                load_debug_config(str(path))

    def test_rejects_submodule_under_the_wrong_layer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.llmdbg"
            path.write_text(
                'model = "gpt2"\nprompt = "hello"\nlayer = 1\n'
                'submodule = "transformer.h.0.attention"\n'
            )
            with self.assertRaisesRegex(ValueError, r"does not belong to layer 1"):
                load_debug_config(str(path))
