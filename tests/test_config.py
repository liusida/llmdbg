from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from llmdbg.config import load_debug_config


class DebugConfigTest(unittest.TestCase):
    def test_loads_llmdbg_toml(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gpt2.llmdbg"
            path.write_text(
                """
[model]
id = "openai-community/gpt2"
revision = "main"
device = "cuda"
dtype = "bfloat16"

[input]
prompt = "The capital of France is"

[debug]
stop_on_entry = true

[[step]]
kind = "embedding"
path = "token_embedding"
""".strip()
            )
            config = load_debug_config(str(path))

        self.assertEqual(config["model"], "openai-community/gpt2")
        self.assertEqual(config["prompt"], "The capital of France is")
        self.assertTrue(config["stopOnEntry"])
        self.assertEqual(config["stepPaths"], ["token_embedding"])

    def test_rejects_missing_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.llmdbg"
            path.write_text('[input]\nprompt = "hello"\n[[step]]\npath = "token_embedding"\n')
            with self.assertRaisesRegex(ValueError, r"\[model\]\.id"):
                load_debug_config(str(path))
