from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "fit_workspace_lenses.py"


def load_script():
    spec = importlib.util.spec_from_file_location("fit_workspace_lenses_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TinyBlock(torch.nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.attn = torch.nn.Linear(width, width, bias=False)
        self.ln_2 = torch.nn.Identity()
        self.mlp = torch.nn.Linear(width, width, bias=False)

    def forward(self, values: torch.Tensor) -> tuple[torch.Tensor]:
        mid = values + self.attn(values)
        output = mid + self.mlp(self.ln_2(mid))
        return (output,)


class TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        width = 3
        self.config = SimpleNamespace(model_type="gpt2", hidden_size=width, n_embd=width)
        self.transformer = SimpleNamespace(
            drop=torch.nn.Embedding(16, width),
            h=torch.nn.ModuleList([TinyBlock(width), TinyBlock(width)]),
            ln_f=torch.nn.Identity(),
        )
        self.embedding = self.transformer.drop
        self.blocks = self.transformer.h

    def forward(self, input_ids: torch.Tensor, use_cache: bool = False) -> Any:
        values = self.embedding(input_ids)
        for block in self.blocks:
            values = block(values)[0]
        return SimpleNamespace(last_hidden_state=values)


class TinyTokenizer:
    def __call__(self, _prompt: str, **_kwargs):
        return {"input_ids": torch.tensor([[1, 2, 3, 4]])}


class FitWorkspaceLensesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_script()

    def test_prompt_fit_captures_embed_mid_and_post_sites(self) -> None:
        model = TinyModel().eval().requires_grad_(False)
        maps, sequence_length, positions = self.module.prompt_jacobians(
            model,
            TinyTokenizer(),
            prompt="test",
            dim_batch=2,
            max_seq_len=8,
            skip_first=0,
        )

        self.assertEqual(sequence_length, 4)
        self.assertEqual(positions, 3)
        self.assertEqual(set(maps), {"embed", "mid.0", "post.0", "mid.1"})
        self.assertTrue(all(tuple(matrix.shape) == (3, 3) for matrix in maps.values()))

    def test_model_specific_prompt_defaults(self) -> None:
        self.assertEqual(self.module.DEFAULT_PROMPTS["openai-community/gpt2"], 100)
        self.assertEqual(self.module.DEFAULT_PROMPTS["Qwen/Qwen3.5-0.8B"], 100)
        self.assertEqual(self.module.DEFAULT_PROMPTS["Qwen/Qwen3.5-2B-Base"], 100)
        self.assertEqual(self.module.DEFAULT_PROMPTS["Qwen/Qwen3.5-2B"], 100)
        self.assertEqual(self.module.DEFAULT_PROMPTS["Qwen/Qwen3.5-9B"], 25)
        self.assertEqual(self.module.DEFAULT_PROMPTS["Qwen/Qwen3.5-27B"], 25)


if __name__ == "__main__":
    unittest.main()
