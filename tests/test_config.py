from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from llmdbg.config import load_debug_config


class DebugConfigTest(unittest.TestCase):
    def test_loads_gemma4_vision_program(self) -> None:
        source = Path(__file__).parents[2] / "llmdbg-usecase" / "Gemma4-E2B.llmdbg.py"
        config = load_debug_config(str(source))
        self.assertEqual(config["modelKind"], "gemma4")
        self.assertEqual(config["model"], "google/gemma-4-E2B-it")
        self.assertEqual(len(config["functions"]["full_attention"]), 9)
        self.assertEqual(len(config["functions"]["mlp"]), 8)

    def test_loads_gemma2_chat_program(self) -> None:
        source = Path(__file__).parents[2] / "llmdbg-usecase" / "Gemma2-2B-IT.llmdbg.py"
        config = load_debug_config(str(source))

        self.assertEqual(config["modelKind"], "gemma2")
        self.assertEqual(config["model"], "google/gemma-2-2b-it")
        self.assertEqual(
            list(config["functions"]["full_attention"]),
            [
                "input_layernorm", "full_attention", "output_projection",
                "post_attention_layernorm", "residual_add",
            ],
        )
        self.assertEqual(
            list(config["functions"]["mlp"]),
            [
                "pre_feedforward_layernorm", "gate_projection", "gelu",
                "up_projection", "gated_multiply", "down_projection",
                "post_feedforward_layernorm", "residual_add",
            ],
        )

    def test_loads_qwen35_chat_program(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "Qwen35-2B-Chat.llmdbg.py"
            source.write_text(
                '''from llmdbg import debug, qwen35
model = qwen35("Qwen/Qwen3.5-2B", revision="pinned")
messages = [
    {"role": "system", "content": "Be concise."},
    {"role": "user", "content": "Hello"},
]
tokens = model.chat(messages, add_generation_prompt=True, max_new_tokens=12)
with debug(model, tokens):
    x = model.token_embedding(tokens)
''',
                encoding="utf-8",
            )
            config = load_debug_config(str(source))

        self.assertEqual(config["model"], "Qwen/Qwen3.5-2B")
        self.assertEqual(config["revision"], "pinned")
        self.assertEqual(config["messages"], [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Hello"},
        ])
        self.assertTrue(config["addGenerationPrompt"])
        self.assertEqual(config["maxNewTokens"], 12)
        self.assertEqual(config["prompt"], "")

    def test_loads_qwen35_multimodal_chat_program(self) -> None:
        source = Path(__file__).parents[2] / "llmdbg-usecase" / "Qwen35-2B-Vision.llmdbg.py"
        config = load_debug_config(str(source))

        self.assertEqual(config["model"], "Qwen/Qwen3.5-2B")
        content = config["messages"][1]["content"]
        image = next(part for part in content if part["type"] == "image")
        image_source = image.get("url", image.get("path"))
        self.assertIsInstance(image_source, str)
        self.assertTrue(image_source.lower().endswith((".jpg", ".jpeg", ".png", ".webp")))
        self.assertTrue(any(part["type"] == "text" for part in content))
        self.assertEqual(config["maxNewTokens"], 20)

    def test_loads_qwen3vl_full_attention_vision_example(self) -> None:
        source = Path(__file__).parents[2] / "llmdbg-usecase" / "Qwen3-VL-2B-Instruct.llmdbg.py"
        config = load_debug_config(str(source))
        self.assertEqual(config["model"], "Qwen/Qwen3-VL-2B-Instruct")
        self.assertEqual(config["modelKind"], "qwen3vl")
        self.assertIn("visual_residual", config["callLines"])
        self.assertIn("query_projection", config["functions"]["full_attention"])

    def test_loads_qwen35_hybrid_python_program(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "Qwen35-0.8B.llmdbg.py"
            source.write_text(
                '''from llmdbg import debug, qwen35
model = qwen35("Qwen/Qwen3.5-0.8B")
prompt = "hello"
tokens = model.tokenize(prompt)
def full_attention(x, layer):
    normalized = layer.input_layernorm(x)
    mixed = layer.full_attention(normalized)
    update = layer.output_projection(mixed)
    return x + update
def gated_linear_attention(x, layer):
    normalized = layer.input_layernorm(x)
    mixed = layer.gated_delta(normalized)
    update = layer.output_projection(mixed)
    return x + update
def mlp(x, layer):
    normalized = layer.post_attention_layernorm(x)
    gate = layer.gate_projection(normalized)
    gate = layer.silu(gate)
    up = layer.up_projection(normalized)
    hidden = gate * up
    update = layer.down_projection(hidden)
    return x + update
with debug(model, tokens, token=token):
    x = model.token_embedding(tokens)
    for layer in model.layers:
        if layer.kind == "full_attention":
            x = full_attention(x, layer)
        else:
            x = gated_linear_attention(x, layer)
        x = mlp(x, layer)
    x = model.final_layernorm(x)
''',
                encoding="utf-8",
            )
            config = load_debug_config(str(source))

        self.assertEqual(config["modelKind"], "qwen35")
        self.assertEqual(config["model"], "Qwen/Qwen3.5-0.8B")
        self.assertLess(
            config["callLines"]["full_attention"],
            config["callLines"]["gated_linear_attention"],
        )
        self.assertEqual(
            list(config["functions"]["mlp"]),
            [
                "post_attention_layernorm", "gate_projection", "silu",
                "up_projection", "gated_multiply", "down_projection", "residual_add",
            ],
        )
        self.assertEqual(
            config["functionBindings"]["mlp"][4],
            {"name": "hidden", "operation": "gated_multiply", "line": 20},
        )

    def test_loads_pure_python_llmdbg_program(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gpt2.llmdbg.py"
            path.write_text(
                '''from llmdbg import debug, gpt2, nop
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
    nop()
    logits = model.vocabulary_projection(x)
    predictions = model.top_tokens(logits, k=10)
    nop()
''',
                encoding="utf-8",
            )
            config = load_debug_config(str(path))
        self.assertTrue(config["pythonProgram"])
        self.assertEqual(config["defaultToken"], 1)
        self.assertFalse(config["stopOnEntry"])
        self.assertEqual(config["callLines"]["attention"], 21)
        self.assertEqual(config["functions"]["mlp"]["gelu"], 15)
        self.assertEqual(config["callLines"]["nop"], [24, 27])
        self.assertEqual(config["callLines"]["vocabulary_projection"], 25)
        self.assertEqual(config["callLines"]["top_tokens"], 26)
        self.assertEqual(
            config["functionBindings"]["mlp"],
            [
                {"name": "normalized", "operation": "layernorm", "line": 13},
                {"name": "hidden", "operation": "expansion", "line": 14},
                {"name": "hidden", "operation": "gelu", "line": 15},
                {"name": "update", "operation": "projection", "line": 16},
            ],
        )

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

    def test_commented_python_token_runs_all_positions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "all-tokens.llmdbg.py"
            path.write_text(
                '''from llmdbg import debug, gpt2
model = gpt2("openai-community/gpt2")
prompt = "hello world"
# token = 1
tokens = model.tokenize(prompt)
with debug(model, tokens, token=token):
    x = model.token_embedding(tokens)
''',
                encoding="utf-8",
            )
            config = load_debug_config(str(path))

        self.assertIsNone(config["defaultToken"])
        self.assertFalse(config["stopOnEntry"])

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
