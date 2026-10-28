from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

import torch
from safetensors.torch import save_file


SCRIPT = Path(__file__).parents[1] / "scripts/fit_lenses/eval/eval_qwen3_8b_j_lenses.py"
SPEC = importlib.util.spec_from_file_location("eval_qwen3_8b_j_lenses", SCRIPT)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


class FakeEncoding:
    def __init__(self, input_ids):
        self.input_ids = input_ids


class FakeTokenizer:
    table = {
        "Brazil": [1],
        " brazil": [2],
        "5": [3],
        " five": [4],
        "multiplication": [5],
        " times": [6],
        "too many": [7, 8],
    }

    def __call__(self, text, add_special_tokens=False):
        return FakeEncoding(self.table.get(text, [99, 100]))


class JLensEvalTests(unittest.TestCase):
    def test_candidate_ids_include_case_spacing_and_order_ops_synonyms(self):
        normal = module.EvalItem("multihop", "x", "p", "t", ("Brazil",))
        order = module.EvalItem("order-ops", "x", "p", "t", ("5", "multiplication"))
        self.assertEqual(module.candidate_token_ids(FakeTokenizer(), normal, "Brazil"), [1, 2])
        self.assertEqual(module.candidate_token_ids(FakeTokenizer(), order, "5"), [3, 4])
        self.assertEqual(
            module.candidate_token_ids(FakeTokenizer(), order, "multiplication"), [5, 6]
        )

    def test_summary_is_item_weighted_and_uses_best_layer_rank(self):
        rows = [
            {"eval": "a", "lens": "one", "item": "x", "intermediate": "i", "layer": 0, "rank": 8},
            {"eval": "a", "lens": "one", "item": "x", "intermediate": "i", "layer": 1, "rank": 2},
            {"eval": "a", "lens": "one", "item": "x", "intermediate": "j", "layer": 0, "rank": 20},
            {"eval": "a", "lens": "one", "item": "y", "intermediate": "i", "layer": 0, "rank": 3},
        ]
        summary = module.summarize(rows, [5])
        per_eval = next(row for row in summary if row["eval"] == "a")
        # Item x has 1/2 hits and item y has 1/1: item-weighted mean is 0.75.
        self.assertEqual(per_eval["pass_at_k"], 0.75)

    def test_correctness_filter_keeps_ungradeable_items(self):
        self.assertEqual(module.selected_indices([True, False, None], False), [0, 2])
        self.assertEqual(module.selected_indices([True, False, None], True), [0, 1, 2])

    def test_v2_evaluator_selects_only_post_maps(self):
        other_path = Path(__file__).parents[1] / "scripts/fit_lenses/eval/eval_qwen35_27b_j_lenses.py"
        other_spec = importlib.util.spec_from_file_location("eval_qwen35_27b_j_lenses", other_path)
        assert other_spec and other_spec.loader
        other = importlib.util.module_from_spec(other_spec)
        sys.modules[other_spec.name] = other
        other_spec.loader.exec_module(other)
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            path = Path(directory) / "lens.safetensors"
            save_file(
                {
                    "embed": torch.eye(2),
                    "mid.0": torch.ones(2, 2),
                    "post.0": torch.eye(2),
                    "post.1": torch.ones(2, 2),
                },
                path,
            )
            maps = other.v2_post_maps(path)
        self.assertEqual(sorted(maps), [0, 1])
        torch.testing.assert_close(maps[0], torch.eye(2))


if __name__ == "__main__":
    unittest.main()
