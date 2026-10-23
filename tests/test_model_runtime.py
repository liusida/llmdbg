from __future__ import annotations

import unittest
from unittest import mock

import llmdbg.model_runtime as model_runtime


class ModelRuntimeCacheTest(unittest.TestCase):
    def test_tokenizers_encoding_ids_are_normalized_to_batched_tensor(self) -> None:
        import torch

        class Encoding:
            ids = [10, 11, 12]

        input_ids = model_runtime._normalize_input_ids(Encoding(), torch)

        self.assertEqual(input_ids.dtype, torch.long)
        self.assertEqual(input_ids.tolist(), [[10, 11, 12]])

    def test_append_generated_token_extends_debug_sequence(self) -> None:
        import torch
        from types import SimpleNamespace

        execution = model_runtime.ModelExecution(
            runtime=model_runtime.ModelRuntime(
                model=SimpleNamespace(),
                tokenizer=SimpleNamespace(decode=lambda ids: f"token-{ids[0]}"),
            ),
            input_ids=torch.tensor([[10, 11]], dtype=torch.long),
            tokens=("a", "b"),
            device="cpu",
            progress=lambda _message: None,
        )

        text = execution.append_generated_token(12)

        self.assertEqual(text, "token-12")
        self.assertEqual(execution.input_ids.tolist(), [[10, 11, 12]])
        self.assertEqual(execution.tokens, ("a", "b", "token-12"))

    def test_auto_device_prefers_cuda_and_selects_supported_dtype(self) -> None:
        import torch

        with (
            mock.patch.object(torch.cuda, "is_available", return_value=True),
            mock.patch.object(torch.cuda, "is_bf16_supported", return_value=True),
            mock.patch.object(torch.cuda, "get_device_capability", return_value=(6, 1)),
        ):
            self.assertEqual(
                model_runtime._resolve_device_and_dtype("auto", "auto"),
                ("cuda", "float16"),
            )

    def test_auto_dtype_uses_bfloat16_only_on_ampere_or_newer(self) -> None:
        import torch

        with (
            mock.patch.object(torch.cuda, "is_available", return_value=True),
            mock.patch.object(torch.cuda, "is_bf16_supported", return_value=True),
            mock.patch.object(torch.cuda, "get_device_capability", return_value=(8, 0)),
        ):
            self.assertEqual(
                model_runtime._resolve_device_and_dtype("auto", "auto"),
                ("cuda", "bfloat16"),
            )

    def test_explicit_bfloat16_rejects_pascal_gpu(self) -> None:
        import torch

        with (
            mock.patch.object(torch.cuda, "is_available", return_value=True),
            mock.patch.object(torch.cuda, "is_bf16_supported", return_value=True),
            mock.patch.object(torch.cuda, "get_device_capability", return_value=(6, 1)),
        ):
            with self.assertRaisesRegex(RuntimeError, "compute capability 6.1"):
                model_runtime._resolve_device_and_dtype("cuda", "bfloat16")

    def test_auto_device_uses_mps_then_cpu(self) -> None:
        import torch

        with (
            mock.patch.object(torch.cuda, "is_available", return_value=False),
            mock.patch.object(torch.backends.mps, "is_available", return_value=True),
        ):
            self.assertEqual(
                model_runtime._resolve_device_and_dtype("auto", "auto"),
                ("mps", "float16"),
            )
        with (
            mock.patch.object(torch.cuda, "is_available", return_value=False),
            mock.patch.object(torch.backends.mps, "is_available", return_value=False),
        ):
            self.assertEqual(
                model_runtime._resolve_device_and_dtype("auto", "auto"),
                ("cpu", "float32"),
            )

    def test_projected_components_rank_by_residual_contribution_norm(self) -> None:
        import torch
        from types import SimpleNamespace

        model = SimpleNamespace(
            config=SimpleNamespace(_name_or_path="fake/model"),
            get_output_embeddings=lambda: SimpleNamespace(weight=torch.eye(2)),
        )
        tokenizer = SimpleNamespace(decode=lambda ids: str(ids[0]))
        execution = model_runtime.ModelExecution(
            runtime=model_runtime.ModelRuntime(model=model, tokenizer=tokenizer),
            input_ids=None,
            tokens=("x",),
            device="cpu",
            progress=lambda _message: None,
        )
        projection = torch.tensor([[1.0, 0.0], [0.0, 3.0]])

        result = execution._top_projected_components(
            0,
            (2.0, 1.0),
            projection,
            component_count=1,
            token_count=1,
            include_logit_lens=True,
            include_j_lens=False,
            include_r_lens=False,
        )

        self.assertEqual(result[0].index, 1)
        self.assertEqual(result[0].activation, 1.0)
        self.assertEqual(result[0].direction_norm, 3.0)

    def test_projected_components_retain_explicitly_removed_index(self) -> None:
        import torch
        from types import SimpleNamespace

        model = SimpleNamespace(
            config=SimpleNamespace(_name_or_path="fake/model"),
            get_output_embeddings=lambda: SimpleNamespace(weight=torch.eye(2)),
        )
        execution = model_runtime.ModelExecution(
            runtime=model_runtime.ModelRuntime(
                model=model,
                tokenizer=SimpleNamespace(decode=lambda ids: str(ids[0])),
            ),
            input_ids=None,
            tokens=("x",),
            device="cpu",
            progress=lambda _message: None,
        )
        result = execution._top_projected_components(
            0,
            (3.0, 0.0),
            torch.eye(2),
            component_count=1,
            token_count=1,
            include_logit_lens=False,
            include_j_lens=False,
            include_r_lens=False,
            include_indices=(1,),
        )
        self.assertEqual([item.index for item in result], [0, 1])
        self.assertEqual(result[1].activation, 0.0)

    def test_removed_ica_component_uses_frozen_vector_for_readout(self) -> None:
        from types import SimpleNamespace

        execution = model_runtime.ModelExecution(
            runtime=model_runtime.ModelRuntime(
                model=SimpleNamespace(config=SimpleNamespace(_name_or_path="fake/model")),
                tokenizer=None,
            ),
            input_ids=None,
            tokens=("x",),
            device="cpu",
            progress=lambda _message: None,
        )
        seen_vectors = []
        execution._vectors_lens = lambda _layer, vectors, **_kwargs: (
            seen_vectors.extend(vectors)
            or tuple(
                model_runtime.VectorProjection(norm=1.0, vocabulary=())
                for _vector in vectors
            )
        )
        current = {
            "index": 7, "rank": 4, "score": 1e-6, "energy": 1e-12,
            "vector": (1e-6, 0.0),
        }
        with mock.patch("llmdbg.ica_lens.top_components", return_value=(current,)):
            result = execution.ica_components(
                "model.layers.2.full_attention",
                (1.0, 2.0),
                is_delta=False,
                include_indices=(7,),
                vector_overrides={7: (3.0, 4.0)},
            )

        self.assertEqual(seen_vectors, [(3.0, 4.0)])
        self.assertEqual(result[0].vector, (3.0, 4.0))
        self.assertEqual(result[0].score, 0.0)
        self.assertEqual(result[0].energy, 0.0)

    def test_component_fan_uses_top_component_as_unit_and_projects_full_output(self) -> None:
        import torch
        from types import SimpleNamespace

        projection = torch.tensor(
            [[2.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
        )
        block = SimpleNamespace(
            attn=SimpleNamespace(c_proj=SimpleNamespace(weight=projection))
        )
        model = SimpleNamespace(
            config=SimpleNamespace(model_type="gpt2"),
            transformer=SimpleNamespace(h=[block]),
        )
        execution = model_runtime.ModelExecution(
            runtime=model_runtime.ModelRuntime(model=model, tokenizer=None),
            input_ids=None,
            tokens=("x",),
            device="cpu",
            progress=lambda _message: None,
        )

        geometry = execution.component_fan_geometry(
            "transformer.h.0.attention",
            (1.0, 1.0, 0.5),
            (2.5, 1.5),
            component_count=3,
        )

        self.assertIsNotNone(geometry)
        assert geometry is not None
        self.assertEqual(geometry["topIndex"], 0)
        self.assertAlmostEqual(geometry["topNorm"], 2.0)
        self.assertAlmostEqual(geometry["result"]["x"], 1.25)
        self.assertAlmostEqual(geometry["result"]["y"], 0.75)
        self.assertEqual([item["index"] for item in geometry["components"]], [2, 1])
        self.assertAlmostEqual(geometry["components"][0]["x"], 0.25)
        self.assertAlmostEqual(geometry["components"][0]["y"], 0.25)

    def test_projected_components_rank_by_signed_projection_onto_output(self) -> None:
        import torch
        from types import SimpleNamespace

        model = SimpleNamespace(
            config=SimpleNamespace(_name_or_path="fake/model"),
            get_output_embeddings=lambda: SimpleNamespace(weight=torch.eye(2)),
        )
        tokenizer = SimpleNamespace(decode=lambda ids: str(ids[0]))
        execution = model_runtime.ModelExecution(
            runtime=model_runtime.ModelRuntime(model=model, tokenizer=tokenizer),
            input_ids=None,
            tokens=("x",),
            device="cpu",
            progress=lambda _message: None,
        )
        projection = torch.tensor([[1.0, 0.0], [0.0, 3.0]])

        result = execution._top_projected_components(
            0,
            (2.0, 1.0),
            projection,
            component_count=1,
            token_count=1,
            include_logit_lens=True,
            include_j_lens=False,
            include_r_lens=False,
            output=(2.0, 0.0),
            residual=(1.0, 1.0),
        )

        self.assertEqual(result[0].index, 0)
        self.assertAlmostEqual(result[0].output_projection, 2.0)
        self.assertAlmostEqual(result[0].residual_norm, 2 ** 0.5)
        self.assertAlmostEqual(result[0].residual_cosine, 2 ** -0.5)
        self.assertAlmostEqual(result[0].potential_norm, 10 ** 0.5)

    def test_vocabulary_projection_returns_ranked_predictions(self) -> None:
        import torch
        from types import SimpleNamespace

        weight = torch.tensor([[1.0, 0.0], [0.0, 1.0], [2.0, 1.0]])
        model = SimpleNamespace(
            get_output_embeddings=lambda: SimpleNamespace(weight=weight)
        )
        tokenizer = SimpleNamespace(
            decode=lambda ids: ["zero", "one", "two"][ids[0]]
        )
        execution = model_runtime.ModelExecution(
            runtime=model_runtime.ModelRuntime(model=model, tokenizer=tokenizer),
            input_ids=None,
            tokens=("x",),
            device="cpu",
            progress=lambda _message: None,
        )

        logits, predictions = execution.vocabulary_projection((1.0, 2.0), count=2)

        self.assertEqual(logits, (1.0, 2.0, 4.0))
        self.assertEqual(
            [prediction.text for prediction in predictions], ["two", "one"]
        )
        self.assertGreater(predictions[0].probability, predictions[1].probability)

    def test_vector_lens_projects_a_model_width_vector(self) -> None:
        import torch
        from types import SimpleNamespace

        weight = torch.tensor([[1.0, 0.0], [0.0, 1.0], [2.0, 1.0]])
        model = SimpleNamespace(
            config=SimpleNamespace(model_type="gpt2", _name_or_path="fake/model"),
            transformer=SimpleNamespace(h=[object()], ln_f=torch.nn.Identity()),
            get_output_embeddings=lambda: SimpleNamespace(weight=weight),
        )
        tokenizer = SimpleNamespace(
            decode=lambda ids: ["zero", "one", "two"][ids[0]]
        )
        execution = model_runtime.ModelExecution(
            runtime=model_runtime.ModelRuntime(model=model, tokenizer=tokenizer),
            input_ids=None,
            tokens=("x",),
            device="cpu",
            progress=lambda _message: None,
        )

        projection = execution.vector_lens("transformer.h.0.mlp", (1.0, 2.0))

        self.assertEqual(projection.vocabulary[0].text, "two")
        self.assertAlmostEqual(projection.norm, 5 ** 0.5)

    def test_vector_lens_batches_vectors_and_skips_unselected_logit_readout(self) -> None:
        import torch
        from types import SimpleNamespace

        class CountingNorm(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.calls = 0

            def forward(self, values):
                self.calls += 1
                return values

        norm = CountingNorm()
        model = SimpleNamespace(
            config=SimpleNamespace(model_type="gpt2", _name_or_path="fake/model"),
            transformer=SimpleNamespace(h=[object()], ln_f=norm),
            get_output_embeddings=lambda: SimpleNamespace(weight=torch.eye(2)),
        )
        execution = model_runtime.ModelExecution(
            runtime=model_runtime.ModelRuntime(
                model=model,
                tokenizer=SimpleNamespace(decode=lambda ids: str(ids[0])),
            ),
            input_ids=None,
            tokens=("x",),
            device="cpu",
            progress=lambda _message: None,
        )

        skipped = execution._vectors_lens(
            "transformer.h.0.mlp",
            ((1.0, 0.0), (0.0, 1.0), (1.0, 1.0)),
            token_count=1,
            include_logit_lens=False,
            include_j_lens=False,
            include_r_lens=False,
        )
        self.assertEqual(norm.calls, 0)
        self.assertTrue(all(not item.vocabulary for item in skipped))

        projected = execution._vectors_lens(
            "transformer.h.0.mlp",
            ((1.0, 0.0), (0.0, 1.0), (1.0, 1.0)),
            token_count=1,
            include_logit_lens=True,
            include_j_lens=False,
            include_r_lens=False,
        )
        self.assertEqual(norm.calls, 1)
        self.assertTrue(all(len(item.vocabulary) == 1 for item in projected))

    def test_jpp_lens_transports_and_filters_nonsemantic_tokens(self) -> None:
        import torch
        from types import SimpleNamespace
        from unittest.mock import patch

        class Tokenizer:
            values = ("!", " word", "42")

            def __len__(self):
                return len(self.values)

            def decode(self, ids):
                return self.values[ids[0]]

        model = SimpleNamespace(
            config=SimpleNamespace(model_type="gpt2", _name_or_path="Qwen/Qwen3.5-9B"),
            transformer=SimpleNamespace(h=[object()], ln_f=torch.nn.Identity()),
            get_output_embeddings=lambda: SimpleNamespace(
                weight=torch.tensor([[10.0, 0.0], [2.0, 0.0], [1.0, 0.0]])
            ),
        )
        execution = model_runtime.ModelExecution(
            runtime=model_runtime.ModelRuntime(model=model, tokenizer=Tokenizer()),
            input_ids=None,
            tokens=("x",),
            device="cpu",
            progress=lambda _message: None,
        )
        with patch(
            "llmdbg.jpp_lens.source_map",
            return_value=(torch.eye(2), {"method": "jpp"}),
        ):
            projection = execution.vector_lens(
                "transformer.h.0.mlp",
                (1.0, 0.0),
                token_count=2,
                include_logit_lens=False,
                include_jpp_lens=True,
            )

        self.assertEqual([item.text for item in projection.jpp_lens], [" word", "42"])

    def test_qwen_runtime_exposes_hybrid_layer_names(self) -> None:
        from types import SimpleNamespace

        model = SimpleNamespace(
            config=SimpleNamespace(
                model_type="qwen3_5",
                layer_types=["linear_attention", "full_attention"],
            ),
            model=SimpleNamespace(layers=[object(), object()], norm=object()),
        )
        runtime = model_runtime.ModelRuntime(model=model, tokenizer=object())

        self.assertEqual(runtime.architecture, "qwen3_5")
        self.assertEqual(runtime.layer_kind(0), "gated_linear_attention")
        self.assertEqual(runtime.layer_kind(1), "full_attention")
        self.assertEqual(
            runtime.state_name(1, runtime.layer_kind(1)),
            "model.layers.1.full_attention",
        )
        self.assertEqual(runtime.final_norm_name, "model.norm")

    def test_matching_runtime_is_reused_without_reloading_dependencies(self) -> None:
        runtime = model_runtime.ModelRuntime(model=object(), tokenizer=object())
        original_runtime = model_runtime._runtime
        original_key = model_runtime._runtime_key
        messages: list[str] = []
        try:
            model_runtime._runtime = runtime
            model_runtime._runtime_key = ("gpt2", "main", "cuda", "bfloat16")
            result = model_runtime._get_model_runtime(
                "gpt2",
                revision="main",
                device="cuda",
                dtype="bfloat16",
                progress=messages.append,
            )
        finally:
            model_runtime._runtime = original_runtime
            model_runtime._runtime_key = original_key

        self.assertIs(result, runtime)
        self.assertIn("Reusing resident gpt2", messages[0])


if __name__ == "__main__":
    unittest.main()
