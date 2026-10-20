from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from llmdbg.r_lens import load_r_lens, source_map


class RLensArtifactTest(unittest.TestCase):
    def tearDown(self) -> None:
        load_r_lens.cache_clear()

    def test_base_artifact_can_feed_matching_instruct_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifact_path = Path(directory) / "lens.pt"
            torch.save(
                {
                    "J": {0: torch.eye(2, dtype=torch.float16)},
                    "d_model": 2,
                    "provenance": {"model_id": "Qwen/Qwen3.5-0.8B-Base"},
                },
                artifact_path,
            )
            with patch.dict(os.environ, {"LLMDBG_R_LENS_PATH": str(artifact_path)}):
                loaded = source_map("Qwen/Qwen3.5-0.8B", 0, 2)

        self.assertIsNotNone(loaded)
        matrix, provenance = loaded
        self.assertEqual(tuple(matrix.shape), (2, 2))
        self.assertEqual(provenance["transfer"], "base_to_instruct")

    def test_rejects_incompatible_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifact_path = Path(directory) / "lens.pt"
            torch.save(
                {
                    "J": {0: torch.eye(2)},
                    "d_model": 2,
                    "provenance": {"model_id": "some/other-model"},
                },
                artifact_path,
            )
            with patch.dict(os.environ, {"LLMDBG_R_LENS_PATH": str(artifact_path)}):
                with self.assertRaisesRegex(ValueError, "incompatible"):
                    load_r_lens("Qwen/Qwen3.5-0.8B")


if __name__ == "__main__":
    unittest.main()
