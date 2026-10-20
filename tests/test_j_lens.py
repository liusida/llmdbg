from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from llmdbg.j_lens import load_j_lens, source_map


class JLensArtifactTest(unittest.TestCase):
    def tearDown(self) -> None:
        load_j_lens.cache_clear()

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
            with patch.dict(os.environ, {"LLMDBG_J_LENS_PATH": str(artifact_path)}):
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
            with patch.dict(os.environ, {"LLMDBG_J_LENS_PATH": str(artifact_path)}):
                with self.assertRaisesRegex(ValueError, "incompatible"):
                    load_j_lens("Qwen/Qwen3.5-0.8B")

    def test_loads_pinned_external_torch_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index = root / "index.json"
            lens = root / "lens.pt"
            torch.save({
                "J": {0: torch.eye(2)}, "d_model": 2,
                "provenance": {"model_id": "Qwen/Qwen3.5-9B"},
            }, lens)
            index.write_text(json.dumps({"j_lenses": [{
                "model_id": "Qwen/Qwen3.5-9B",
                "source_repo_id": "authors/lenses",
                "source_revision": "abc123",
                "source_filename": "qwen/j-lens/lens.pt",
                "source_size": lens.stat().st_size,
                "source_sha256": hashlib.sha256(lens.read_bytes()).hexdigest(),
            }]}))

            def download(*, filename: str, **_kwargs: object) -> str:
                return str(index if filename == "index.json" else lens)

            with patch("huggingface_hub.hf_hub_download", side_effect=download):
                loaded = source_map("Qwen/Qwen3.5-9B", 0, 2)

        self.assertIsNotNone(loaded)
        self.assertEqual(tuple(loaded[0].shape), (2, 2))

    def test_missing_hub_artifact_is_retried(self) -> None:
        artifact = {"J": {0: torch.eye(2)}}
        with patch("llmdbg.j_lens._download_from_hub", side_effect=[None, artifact]) as download:
            self.assertIsNone(load_j_lens("new/model"))
            self.assertIs(load_j_lens("new/model"), artifact)
        self.assertEqual(download.call_count, 2)


if __name__ == "__main__":
    unittest.main()
