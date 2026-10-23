from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

from llmdbg import ica_lens


class ICALensTest(unittest.TestCase):
    def tearDown(self) -> None:
        ica_lens.clear_cache()

    def test_supported_model_aliases_resolve_to_published_artifacts(self) -> None:
        self.assertEqual(
            ica_lens.artifact_repository("Qwen/Qwen3.5-2B"),
            "sida/icalens-qwen3.5-2b-ultrachat-1m",
        )
        self.assertEqual(
            ica_lens.artifact_repository("Qwen/Qwen3.5-2B-Base"),
            "sida/icalens-qwen3.5-2b-base-pile10k",
        )
        self.assertEqual(
            ica_lens.artifact_repository("Qwen/Qwen3.5-9B-Base"),
            "sida/icalens-qwen3.5-9b-base-pile10k",
        )
        self.assertEqual(
            ica_lens.artifact_repository("openai-community/gpt2"),
            "sida/icalens-gpt2-small-pile10k",
        )
        self.assertIsNone(ica_lens.artifact_repository("Qwen/Qwen3.5-27B"))

    def test_residual_is_centered_but_delta_is_not(self) -> None:
        center = np.asarray([10.0, 20.0], dtype=np.float32)
        reading = np.eye(2, dtype=np.float32)
        writing = np.eye(2, dtype=np.float32)
        artifact = (center, reading, writing)
        with mock.patch.object(ica_lens, "_layer_artifact", return_value=artifact):
            residual = ica_lens.top_components(
                "gpt2", 0, (11.0, 24.0), is_delta=False, count=2
            )
            delta = ica_lens.top_components(
                "gpt2", 0, (1.0, 4.0), is_delta=True, count=2
            )

        self.assertEqual([item["index"] for item in residual], [1, 0])
        self.assertEqual([item["score"] for item in residual], [4.0, 1.0])
        self.assertEqual([item["score"] for item in delta], [4.0, 1.0])
        self.assertAlmostEqual(residual[0]["energy"], 16.0 / 17.0)

    def test_explicit_component_is_retained_after_it_leaves_top_k(self) -> None:
        artifact = (
            np.zeros(3, dtype=np.float32),
            np.eye(3, dtype=np.float32),
            np.eye(3, dtype=np.float32),
        )
        with mock.patch.object(ica_lens, "_layer_artifact", return_value=artifact):
            components = ica_lens.top_components(
                "gpt2", 0, (3.0, 2.0, 0.0), is_delta=False,
                count=1, include_indices=(2,),
            )
        self.assertEqual([item["index"] for item in components], [0, 2])

    def test_zero_count_returns_only_explicit_components(self) -> None:
        artifact = (
            np.zeros(3, dtype=np.float32),
            np.eye(3, dtype=np.float32),
            np.eye(3, dtype=np.float32),
        )
        with mock.patch.object(ica_lens, "_layer_artifact", return_value=artifact):
            components = ica_lens.top_components(
                "gpt2", 0, (3.0, 2.0, 1.0), is_delta=False,
                count=0, include_indices=(2,),
            )
        self.assertEqual([item["index"] for item in components], [2])
        self.assertEqual(components[0]["score"], 1.0)

    def test_remove_component_zeros_its_score_and_reconstructs(self) -> None:
        import torch

        artifact = (
            np.asarray([10.0, 20.0], dtype=np.float32),
            np.eye(2, dtype=np.float32),
            np.eye(2, dtype=np.float32),
        )
        vector = torch.tensor([11.0, 24.0])
        with mock.patch.object(ica_lens, "_layer_artifact", return_value=artifact):
            result = ica_lens.remove_component("gpt2", 0, vector, 1)
        self.assertIs(result, vector)
        self.assertEqual(vector.tolist(), [11.0, 20.0])

    def test_layer_artifact_reports_only_on_cache_miss(self) -> None:
        artifact = (
            np.zeros(2, dtype=np.float32),
            np.eye(2, dtype=np.float32),
            np.eye(2, dtype=np.float32),
        )
        messages: list[str] = []
        with mock.patch.object(
            ica_lens, "_download_layer_artifact", return_value=artifact
        ) as download:
            ica_lens._layer_artifact("owner/repo", 3, messages.append)
            ica_lens._layer_artifact("owner/repo", 3, messages.append)

        download.assert_called_once_with("owner/repo", 3)
        self.assertIn("Resolving ICA basis for layer 3", messages[0])
        self.assertIn("Loaded ICA basis for layer 3", messages[1])
        self.assertEqual(len(messages), 2)


if __name__ == "__main__":
    unittest.main()
