from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
from safetensors.torch import save_file

from llmdbg import workspace_lens


class WorkspaceLensTest(unittest.TestCase):
    def tearDown(self) -> None:
        workspace_lens.clear_cache()

    def test_view_sites_are_resolved_at_the_vector_injection_boundary(self) -> None:
        self.assertEqual(workspace_lens.resolve_view_site("residual-pre", 0), ("embed", None))
        self.assertEqual(workspace_lens.resolve_view_site("residual-pre", 3), ("post", 2))
        self.assertEqual(workspace_lens.resolve_view_site("attention-output", 3), ("mid", 3))
        self.assertEqual(workspace_lens.resolve_view_site("attention-neurons", 3), ("mid", 3))
        self.assertEqual(workspace_lens.resolve_view_site("mlp-output", 3), ("post", 3))
        self.assertEqual(workspace_lens.resolve_view_site("neurons", 3), ("post", 3))

    def test_final_post_is_identity_without_an_artifact(self) -> None:
        with mock.patch.dict("os.environ", {}, clear=False), mock.patch.object(
            workspace_lens, "load_v2", return_value=None
        ):
            source = workspace_lens.source_map(
                "j", "fake/model", "post", 3, 2, layer_count=4
            )
        assert source is not None
        self.assertTrue(torch.equal(source.matrix, torch.eye(2)))
        self.assertEqual(source.metadata["artifact_kind"], "identity")

    def test_v2_artifact_requires_exact_model_and_explicit_site(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "Qwen--Exact" / "j"
            artifact.mkdir(parents=True)
            save_file({"embed": torch.eye(2), "mid.0": torch.ones(2, 2)}, artifact / "lens.safetensors")
            manifest = {
                "format_version": 2,
                "artifact": "lens.safetensors",
                "model_id": "Qwen/Exact",
                "method": "j",
            }
            (artifact / "manifest.json").write_text(json.dumps(manifest))
            with mock.patch.dict(
                "os.environ", {"LLMDBG_WORKSPACE_LENS_PATH": str(root)}
            ):
                embed = workspace_lens.source_map(
                    "j", "Qwen/Exact", "embed", None, 2, layer_count=2
                )
                missing_post = workspace_lens.source_map(
                    "j", "Qwen/Exact", "post", 0, 2, layer_count=2
                )
            assert embed is not None
            self.assertTrue(torch.equal(embed.matrix, torch.eye(2)))
            self.assertIsNone(missing_post)

    def test_v2_artifact_rejects_a_different_checkpoint_revision(self) -> None:
        artifact = {
            "manifest": {"model_revision": "fitted-sha"},
            "tensors": {"embed": torch.eye(2)},
        }
        with mock.patch.object(workspace_lens, "load_v2", return_value=artifact):
            with self.assertRaisesRegex(ValueError, "fitted-sha.*runtime-sha"):
                workspace_lens.source_map(
                    "j", "Qwen/Exact", "embed", None, 2, layer_count=2,
                    model_revision="runtime-sha",
                )

    def test_local_direct_jpp_folder_falls_back_when_v2_index_is_absent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "lens.pt").write_bytes(b"legacy")
            configured = [{
                "id": "jpp",
                "label": "J++ Lens",
                "method": "jpp",
                "source": directory,
                "enabled": True,
            }]
            with mock.patch.dict(
                "os.environ",
                {"LLMDBG_RESIDUAL_TO_VOCABULARY_LENSES": json.dumps(configured)},
            ):
                self.assertIsNone(workspace_lens.load_v2("Qwen/Test", "jpp"))

    def test_v2_load_reports_slow_work_before_and_after_tensor_loading(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "Qwen--Exact" / "j"
            artifact.mkdir(parents=True)
            save_file({"embed": torch.eye(2)}, artifact / "lens.safetensors")
            manifest = {
                "format_version": 2,
                "artifact": "lens.safetensors",
                "model_id": "Qwen/Exact",
                "method": "j",
            }
            (artifact / "manifest.json").write_text(json.dumps(manifest))
            messages: list[str] = []
            with mock.patch.dict(
                "os.environ", {"LLMDBG_WORKSPACE_LENS_PATH": str(root)}
            ):
                workspace_lens.load_v2("Qwen/Exact", "j", progress=messages.append)
                workspace_lens.load_v2("Qwen/Exact", "j", progress=messages.append)

        self.assertIn("Resolving J Lens artifact", messages[0])
        self.assertIn("Loading J Lens tensors", messages[1])
        self.assertIn("Loaded J Lens with 1 site maps", messages[2])
        self.assertEqual(len(messages), 3, "cache hits should not repeat progress messages")
