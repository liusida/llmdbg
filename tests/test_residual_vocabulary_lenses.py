from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from llmdbg.residual_vocabulary_lenses import (
    active_lens,
    activate,
    available_lenses,
    configured_lenses,
    source_file,
)


class ResidualVocabularyLensRegistryTest(unittest.TestCase):
    def test_custom_lens_can_be_selected_without_replacing_its_method(self) -> None:
        configured = [
            {
                "id": "custom-j",
                "label": "My J-Lens",
                "method": "j",
                "source": "hf://datasets/example/lenses",
                "indexSection": "workspace_lenses",
            }
        ]
        with patch.dict(
            os.environ,
            {"LLMDBG_RESIDUAL_TO_VOCABULARY_LENSES": json.dumps(configured)},
        ):
            self.assertEqual(configured_lenses()[0]["label"], "My J-Lens")
            with activate("custom-j"):
                self.assertEqual(active_lens("j")["source"], "hf://datasets/example/lenses")

    def test_duplicate_ids_are_rejected(self) -> None:
        entry = {"id": "same", "label": "Lens", "method": "j", "source": "hf://datasets/a/b"}
        with patch.dict(
            os.environ,
            {"LLMDBG_RESIDUAL_TO_VOCABULARY_LENSES": json.dumps([entry, entry])},
        ):
            with self.assertRaisesRegex(ValueError, "duplicate"):
                configured_lenses()

    def test_local_source_resolves_repository_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index = root / "index.json"
            index.write_text('{"workspace_lenses": []}')
            self.assertEqual(source_file({"source": directory}, "index.json"), index)

    def test_local_source_cannot_escape_repository(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "escapes"):
                source_file({"source": directory}, "../outside.json")

    def test_method_mapping_activates_one_lens_per_method(self) -> None:
        configured = [
            {"id": "j-one", "label": "J", "method": "j", "source": "hf://datasets/a/b"},
            {"id": "jpp-one", "label": "J++", "method": "jpp", "source": "hf://datasets/a/b"},
        ]
        with patch.dict(os.environ, {"LLMDBG_RESIDUAL_TO_VOCABULARY_LENSES": json.dumps(configured)}):
            with activate({"j": "j-one", "jpp": "jpp-one"}):
                self.assertEqual(active_lens("j")["id"], "j-one")
                self.assertEqual(active_lens("jpp")["id"], "jpp-one")

    def test_local_jpp_fit_is_only_available_for_manifest_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "lens.pt").write_bytes(b"lens")
            (root / "fit-manifest.json").write_text(
                json.dumps({"model_id": "Qwen/Qwen3.5-2B"})
            )
            configured = [{
                "id": "jpp", "label": "J++ Lens", "method": "jpp", "source": directory,
            }]
            with patch.dict(os.environ, {"LLMDBG_RESIDUAL_TO_VOCABULARY_LENSES": json.dumps(configured)}):
                available_lenses.cache_clear()
                self.assertEqual(available_lenses("Qwen/Qwen3.5-2B")[0]["label"], "J++ Lens")
                self.assertEqual(available_lenses("Qwen/Qwen3.5-9B"), ())
                available_lenses.cache_clear()


if __name__ == "__main__":
    unittest.main()
