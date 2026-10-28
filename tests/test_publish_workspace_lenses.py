from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/fit_lenses/v2/publish_lenses.py"
SPEC = importlib.util.spec_from_file_location("publish_lenses_v2_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
publish = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(publish)


class PublishWorkspaceLensesTest(unittest.TestCase):
    def test_update_index_preserves_unrelated_legacy_sections(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            stage = Path(directory)
            legacy = {"format_version": 1, "j_lenses": [{"model_id": "legacy"}]}
            (stage / "index.json").write_text(json.dumps(legacy))
            entry = {
                "model_id": "test/model", "method": "j", "format_version": 2,
                "path": "lenses/v2/test--model/j",
            }

            index = json.loads(publish.update_index(stage, [entry]).read_text())

            self.assertEqual(index["format_version"], 1)
            self.assertEqual(index["j_lenses"], legacy["j_lenses"])
            self.assertEqual(index["workspace_lenses"], [entry])

    def test_stage_validate_and_index(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            stage = root / "stage"
            source.mkdir()
            tensor_path = source / "lens.safetensors"
            save_file({"embed": torch.eye(2), "mid.0": torch.ones(2, 2)}, tensor_path)
            manifest = {
                "format_version": 2, "artifact": tensor_path.name,
                "sha256": publish.digest(tensor_path), "model_id": "test/model",
                "model_revision": "model-sha", "tokenizer_revision": "tokenizer-sha",
                "method": "j", "sites": {"embed": True, "mid": [0], "post": []},
                "hidden_size": 2, "dtype": "float16", "provenance": "test",
            }
            (source / "manifest.json").write_text(json.dumps(manifest))

            entries = publish.stage_artifacts([source], stage)
            publish.clean_cache_test(stage, entries)
            index_path = publish.update_index(stage, entries)

            index = json.loads(index_path.read_text())
            self.assertEqual(index["workspace_lenses"], entries)
            self.assertTrue((stage / entries[0]["path"] / "lens.safetensors").is_file())

