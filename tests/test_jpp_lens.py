from __future__ import annotations

import os
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from llmdbg.jpp_lens import configured_display_name, display_name, load_jpp_lens, source_map
from llmdbg.residual_vocabulary_lenses import activate


class JppLensArtifactTest(unittest.TestCase):
    def tearDown(self) -> None:
        load_jpp_lens.cache_clear()

    def test_loads_supported_local_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "lens.pt"
            torch.save(
                {"parameters": {"jacobians": {0: torch.eye(2)}}}, path
            )
            with patch.dict(os.environ, {"LLMDBG_JPP_LENS_PATH": str(path)}):
                loaded = source_map("Qwen/Qwen3.5-9B", 0, 2)
        self.assertIsNotNone(loaded)
        self.assertEqual(tuple(loaded[0].shape), (2, 2))
        self.assertEqual(loaded[1]["method"], "jpp")
        self.assertEqual(loaded[1]["artifact_kind"], "local")
        self.assertEqual(loaded[1]["display_name"], "Local J++ Lens")
        self.assertIsNone(loaded[1]["repository"])

    def test_only_supports_qwen35_9b(self) -> None:
        self.assertIsNone(load_jpp_lens("Qwen/Qwen3.5-2B"))

    def test_configured_labels_do_not_load_artifacts(self) -> None:
        with patch.dict(os.environ, {"LLMDBG_JPP_LENS_PATH": "/tmp/local.pt"}):
            self.assertEqual(configured_display_name("Qwen/Qwen3.5-2B"), "Local J++ Lens")
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(configured_display_name("Qwen/Qwen3.5-9B"), "Official J++ Lens")

    def test_site_aware_repository_uses_configured_label_without_lens_pt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "index.json").write_text('{"workspace_lenses": []}')
            configured = [{
                "id": "jpp", "label": "J++ Lens", "method": "jpp",
                "source": directory, "indexSection": "workspace_lenses",
            }]
            with patch.dict(os.environ, {
                "LLMDBG_RESIDUAL_TO_VOCABULARY_LENSES": json.dumps(configured),
            }, clear=True):
                self.assertEqual(configured_display_name("Qwen/Test"), "J++ Lens")

    def test_loads_model_matched_local_2b_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "lens.pt"
            torch.save(
                {
                    "parameters": {"jacobians": {0: torch.eye(2)}},
                    "config": {"hf_model_name": "Qwen/Qwen3.5-2B"},
                },
                path,
            )
            with patch.dict(os.environ, {"LLMDBG_JPP_LENS_PATH": str(path)}):
                loaded = source_map("Qwen/Qwen3.5-2B", 0, 2)
                label = display_name("Qwen/Qwen3.5-2B")
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded[1]["model_id"], "Qwen/Qwen3.5-2B")
        self.assertEqual(label, "Local J++ Lens")

    def test_rejects_local_artifact_for_another_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "lens.pt"
            torch.save(
                {
                    "parameters": {"jacobians": {0: torch.eye(2)}},
                    "config": {"hf_model_name": "Qwen/Qwen3.5-2B"},
                },
                path,
            )
            with patch.dict(os.environ, {"LLMDBG_JPP_LENS_PATH": str(path)}):
                with self.assertRaisesRegex(ValueError, "incompatible"):
                    source_map("Qwen/Qwen3.5-9B", 0, 2)

    def test_rejects_wrong_width(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "lens.pt"
            torch.save({"parameters": {"jacobians": {0: torch.eye(2)}}}, path)
            with patch.dict(os.environ, {"LLMDBG_JPP_LENS_PATH": str(path)}):
                with self.assertRaisesRegex(ValueError, "shape"):
                    source_map("Qwen/Qwen3.5-9B", 0, 3)

    def test_loads_neuronpedia_jpp_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "Qwen3.5-2B_jpp_lens.pt"
            torch.save(
                {
                    "J": {0: torch.eye(2)},
                    "source_layers": [0],
                    "d_model": 2,
                    "n_prompts": 96,
                    "provenance": {"method": "jpp"},
                },
                path,
            )
            configured = [{
                "id": "jpp-neuronpedia",
                "label": "Neuronpedia J++ Lens",
                "method": "jpp",
                "source": "hf://models/neuronpedia/jacobian-lens",
                "revision": "e166deb5ad540932100ef803c8c9bcc2a5453694",
                "models": ["Qwen/Qwen3.5-2B"],
                "enabled": True,
            }]
            with patch.dict(os.environ, {
                "LLMDBG_RESIDUAL_TO_VOCABULARY_LENSES": json.dumps(configured),
            }, clear=True), patch(
                "llmdbg.residual_vocabulary_lenses.source_file", return_value=path
            ):
                with activate("jpp-neuronpedia"):
                    loaded = source_map("Qwen/Qwen3.5-2B", 0, 2)
                    label = display_name("Qwen/Qwen3.5-2B")
        self.assertIsNotNone(loaded)
        self.assertTrue(torch.equal(loaded[0], torch.eye(2)))
        self.assertEqual(loaded[1]["artifact_kind"], "custom")
        self.assertEqual(label, "Neuronpedia J++ Lens")


if __name__ == "__main__":
    unittest.main()
