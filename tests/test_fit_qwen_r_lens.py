from __future__ import annotations

import importlib.util
from pathlib import Path

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/fit_lenses/v1/fit_qwen35_r_lens.py"
SPEC = importlib.util.spec_from_file_location("fit_qwen35_r_lens", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
install_relp_rules = MODULE.install_relp_rules


def test_qwen3_relp_rules_preserve_forward_values() -> None:
    config = Qwen3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
    )
    model = Qwen3ForCausalLM(config).eval()
    input_ids = torch.tensor([[1, 2, 3, 4]])

    with torch.inference_mode():
        ordinary = model(input_ids).logits

    assert install_relp_rules(model) == (5, 2)

    with torch.inference_mode():
        patched = model(input_ids).logits

    torch.testing.assert_close(patched, ordinary, rtol=0, atol=0)
