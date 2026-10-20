from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from llmdbg.config import load_debug_config


class DebugConfigTest(unittest.TestCase):
    def test_loads_pure_python_llmdbg_program(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gpt2.llmdbg.py"
            path.write_text(
                '''from llmdbg import debug, gpt2
model = gpt2("openai-community/gpt2", device="cuda", dtype="bfloat16")
prompt = "hello world"
token = 1
tokens = model.tokenize(prompt)
def attention(x, layer):
    normalized = layer.layernorm(x)
    qkv = layer.qkv_projection(normalized)
    mixed = layer.attention_mixing(qkv)
    update = layer.output_projection(mixed)
    return x + update
def mlp(x, layer):
    normalized = layer.layernorm(x)
    hidden = layer.expansion(normalized)
    hidden = layer.gelu(hidden)
    update = layer.projection(hidden)
    return x + update
with debug(model, tokens, token=token):
    x = model.token_embedding(tokens)
    for layer in model.layers:
        x = attention(x, layer)
        x = mlp(x, layer)
    x = model.final_layernorm(x)
''',
                encoding="utf-8",
            )
            config = load_debug_config(str(path))
        self.assertTrue(config["pythonProgram"])
        self.assertEqual(config["defaultToken"], 1)
        self.assertEqual(config["callLines"]["attention"], 21)
        self.assertEqual(config["functions"]["mlp"]["gelu"], 15)

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
token = 3
stop_on_entry = true

def attention():
    layernorm
    qkv_projection
    attention_mixing
    output_projection
    residual_add

def mlp():
    layernorm
    expansion
    gelu
    projection
    residual_add

submodule = "token_embedding"
layer = 0
attention()
mlp()
""".strip()
            )
            config = load_debug_config(str(path))

        self.assertEqual(config["model"], "openai-community/gpt2")
        self.assertEqual(config["prompt"], "The capital of France is")
        self.assertEqual(config["defaultToken"], 3)
        self.assertTrue(config["stopOnEntry"])
        self.assertEqual(
            list(config["functions"]["mlp"]),
            ["layernorm", "expansion", "gelu", "projection", "residual_add"],
        )
        self.assertEqual(
            list(config["functions"]["attention"]),
            ["layernorm", "qkv_projection", "attention_mixing", "output_projection", "residual_add"],
        )
        self.assertEqual(
            config["stepPaths"],
            ["token_embedding", "transformer.h.0.attention", "transformer.h.0.mlp"],
        )
        self.assertEqual(config["stepLines"]["transformer.h.0.attention"], 25)
        self.assertEqual(config["stepLines"]["transformer.h.0.mlp"], 26)

    def test_rejects_layer_call_without_layer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.llmdbg"
            path.write_text(
                'model = "gpt2"\nprompt = "hello"\nattention()\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, r"attention\(\) requires layer = N"):
                load_debug_config(str(path))

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

    def test_rejects_negative_default_token(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.llmdbg"
            path.write_text(
                'model = "gpt2"\nprompt = "hello"\ntoken = -1\n'
                'submodule = "token_embedding"\n'
            )
            with self.assertRaisesRegex(ValueError, r"token must be assigned"):
                load_debug_config(str(path))
