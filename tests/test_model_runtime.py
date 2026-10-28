from __future__ import annotations

import unittest
from unittest import mock

import llmdbg.model_runtime as model_runtime


class ModelRuntimeCacheTest(unittest.TestCase):
    def test_qwen3_dense_cached_trace_matches_native_forward(self) -> None:
        import torch
        from types import SimpleNamespace
        from transformers import Qwen3Config, Qwen3ForCausalLM
        from llmdbg import qwen3
        self.assertEqual(qwen3().architecture, 'qwen3')
        config = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=8)
        with torch.random.fork_rng():
            torch.manual_seed(42)
            model = Qwen3ForCausalLM(config).eval()
        runtime = model_runtime.ModelRuntime(model, SimpleNamespace(decode=lambda ids: str(ids[0])))
        ids = torch.tensor([[1, 2, 3]])
        execution = model_runtime.ModelExecution(runtime, ids, ('a', 'b', 'c'), 'cpu', lambda _: None)
        with torch.inference_mode():
            expected = model.model(input_ids=ids, use_cache=False).last_hidden_state
        for position in (0, 2):
            trace = execution.trace_token(position)
            self.assertTrue(all(runtime.layer_kind(i) == 'full_attention' for i in range(2)))
            self.assertTrue(any(s.layer.name.endswith('.full_attention') for s in trace.states))
            torch.testing.assert_close(torch.tensor(trace.states[-1].output_vector),
                                       expected[0, position], atol=1e-5, rtol=1e-4)

    def test_delta_channels_group_by_value_heads_not_attention_heads(self) -> None:
        import torch
        from types import SimpleNamespace
        execution = object.__new__(model_runtime.ModelExecution)
        execution.runtime = SimpleNamespace(
            architecture='qwen35',
            layers=[SimpleNamespace(linear_attn=SimpleNamespace(out_proj=SimpleNamespace(weight=torch.ones(3, 2048))))],
            model=SimpleNamespace(config=SimpleNamespace(text_config=SimpleNamespace(
                num_attention_heads=8, linear_num_value_heads=16))),
        )
        with mock.patch.object(model_runtime.ModelExecution, '_top_projected_components', return_value=()) as project:
            execution.top_attention_channels('model.layers.0.gated_linear_attention', (1.,)*2048, head_index=15)
        self.assertEqual(project.call_args.kwargs['channel_range'], (1920, 2048))

    def test_delta_core_snapshots_reconstruct_native_memory_and_readout(self) -> None:
        import torch
        from types import SimpleNamespace
        from transformers import Qwen3_5TextConfig, Qwen3_5ForCausalLM
        config = Qwen3_5TextConfig(
            vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2, head_dim=8,
            linear_num_key_heads=2, linear_num_value_heads=4,
            linear_key_head_dim=8, linear_value_head_dim=8,
            layer_types=["linear_attention", "full_attention"],
        )
        with torch.random.fork_rng():
            torch.manual_seed(42)
            model = Qwen3_5ForCausalLM(config).eval()
            runtime = model_runtime.ModelRuntime(model, SimpleNamespace(decode=lambda ids: str(ids[0])))
            ids = torch.tensor([[1, 2, 3]])
            execution = model_runtime.ModelExecution(runtime, ids, ("a", "b", "c"), "cpu", lambda _: None)
            execution.capture_delta_core = True
            with torch.inference_mode():
                expected = model.model(input_ids=ids, use_cache=False).last_hidden_state
            for position in (0, 2):
                trace = execution.trace_token(position)
                state = next(s for s in trace.states if s.layer.name.endswith(".gated_linear_attention"))
                t = state.tensors
                q, k, v = (t["delta_" + name][:, 0].float() for name in ("query", "key", "value"))
                memory = t["delta_previous_memory"] * t["delta_decay"][:, 0, :, None, None]
                prediction = (memory * k.unsqueeze(-1)).sum(-2)
                correction = (v - prediction) * t["delta_beta"][:, 0, :, None]
                updated = memory + k.unsqueeze(-1) * correction.unsqueeze(-2)
                torch.testing.assert_close(updated, t["delta_updated_memory"], atol=1e-5, rtol=1e-4)
                readout = (updated * q.unsqueeze(-1)).sum(-2)
                torch.testing.assert_close(readout, t["delta_readout"][:, 0], atol=1e-5, rtol=1e-4)
                torch.testing.assert_close(torch.tensor(trace.states[-1].output_vector), expected[0, position], atol=1e-5, rtol=1e-4)

    def test_gemma4_cached_trace_matches_native_shared_sliding_attention(self) -> None:
        import torch
        from types import SimpleNamespace
        from transformers import Gemma4TextConfig, Gemma4ForCausalLM

        config = Gemma4TextConfig(
            vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
            num_attention_heads=4, num_key_value_heads=1, head_dim=8, global_head_dim=16,
            layer_types=["sliding_attention", "full_attention"] * 2,
            num_kv_shared_layers=2, sliding_window=4,
            hidden_size_per_layer_input=8, vocab_size_per_layer_input=64,
            final_logit_softcapping=30.,
        )
        with torch.random.fork_rng():
            torch.manual_seed(42)
            model = Gemma4ForCausalLM(config).eval()
            model.model.config._attn_implementation = "eager"
            runtime = model_runtime.ModelRuntime(model, SimpleNamespace(decode=lambda ids: str(ids[0])))
            ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9]])
            with torch.inference_mode():
                embeddings = model.model.get_input_embeddings()(ids)
                ple = model.model.get_per_layer_inputs(ids, embeddings)
                reference = model.model(input_ids=ids, use_cache=False).last_hidden_state
            execution = model_runtime.ModelExecution(
                runtime, ids, tuple(str(i) for i in range(9)), "cpu", lambda _: None,
                inputs_embeds=embeddings, position_ids=torch.arange(9).view(1, -1),
                per_layer_inputs=ple,
            )
            for position in (1, 6, 8):
                trace = execution.trace_token(position)
                torch.testing.assert_close(torch.tensor(trace.states[-1].output_vector), reference[0, position], rtol=1e-4, atol=1e-5)
                self.assertEqual(len([s for s in trace.states if s.layer.name.endswith(".per_layer_residual")]), 4)
                self.assertEqual(len([s for s in trace.states if s.layer.name.endswith(".layer_scale")]), 4)

    def test_qwen3vl_cached_trace_matches_native_deepstack_forward(self) -> None:
        import torch
        from types import SimpleNamespace
        try:
            from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration, Qwen3VLTextConfig, Qwen3VLVisionConfig
        except ImportError:
            self.skipTest("Qwen3-VL requires current Transformers")
        text = Qwen3VLTextConfig(
            vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
            num_attention_heads=4, num_key_value_heads=2, head_dim=8,
            rope_parameters={"rope_type": "default", "rope_theta": 10000., "mrope_section": [1, 1, 2]},
        )
        vision = Qwen3VLVisionConfig(
            depth=2, hidden_size=16, intermediate_size=32, num_heads=2,
            out_hidden_size=32, num_position_embeddings=16, deepstack_visual_indexes=[0, 1],
        )
        with torch.random.fork_rng():
            torch.manual_seed(42)
            model = Qwen3VLForConditionalGeneration(Qwen3VLConfig(
                text_config=text.to_dict(), vision_config=vision.to_dict(),
            )).eval()
            model.model.language_model.config._attn_implementation = "eager"
            runtime = model_runtime.ModelRuntime(model, SimpleNamespace(decode=lambda ids: str(ids[0])))
            ids = torch.tensor([[1, 2, 3, 4, 5]])
            with torch.inference_mode():
                embeddings = runtime.text_model.get_input_embeddings()(ids)
                features = [torch.randn(2, 32), torch.randn(2, 32)]
            mask = torch.tensor([[False, True, False, True, False]])
            positions = torch.arange(5).view(1, 1, 5).expand(4, 1, 5)
            execution = model_runtime.ModelExecution(
                runtime, ids, ("a", "image1", "b", "image2", "c"), "cpu", lambda _: None,
                inputs_embeds=embeddings, position_ids=positions,
                visual_pos_masks=mask, deepstack_visual_embeds=features,
            )
            with torch.inference_mode():
                reference = runtime.text_model(
                    inputs_embeds=embeddings, position_ids=positions, visual_pos_masks=mask,
                    deepstack_visual_embeds=features, use_cache=False,
                ).last_hidden_state
            for position in (1, 3, 4):
                trace = execution.trace_token(position)
                actual = torch.tensor(trace.states[-1].output_vector)
                torch.testing.assert_close(actual, reference[0, position], rtol=1e-4, atol=1e-5)
                visual_states = [state for state in trace.states if state.layer.name.endswith(".visual_residual")]
                self.assertEqual(len(visual_states), 2)

    def test_qwen3vl_slices_deepstack_features_by_visual_token_positions(self) -> None:
        import torch
        execution = model_runtime.ModelExecution(
            runtime=None, input_ids=torch.tensor([[1, 2, 3, 4, 5]]),
            tokens=("a", "image", "b", "image", "c"), device="cpu", progress=lambda _: None,
            visual_pos_masks=torch.tensor([[False, True, False, True, False]]),
            deepstack_visual_embeds=[torch.tensor([[10., 11.], [20., 21.]])],
        )
        first = execution._visual_slice(0, 3)
        self.assertEqual(first["visual_pos_masks"].tolist(), [[False, True, False]])
        self.assertEqual(first["deepstack_visual_embeds"][0].tolist(), [[10., 11.]])
        second = execution._visual_slice(3, 4)
        self.assertEqual(second["deepstack_visual_embeds"][0].tolist(), [[20., 21.]])
        text = execution._visual_slice(4, 5)
        self.assertEqual(tuple(text["deepstack_visual_embeds"][0].shape), (0, 2))

    def test_qwen3vl_runtime_uses_full_attention_and_deepstack_count(self) -> None:
        from types import SimpleNamespace
        runtime = model_runtime.ModelRuntime(model=SimpleNamespace(
            config=SimpleNamespace(model_type="qwen3_vl", vision_config=SimpleNamespace(deepstack_visual_indexes=[8, 16, 24])),
            model=SimpleNamespace(language_model=SimpleNamespace(layers=[object()] * 28)),
        ), tokenizer=None)
        self.assertEqual(runtime.architecture, "qwen3_vl")
        self.assertEqual(runtime.visual_layer_count, 3)
        self.assertTrue(all(runtime.layer_kind(index) == "full_attention" for index in range(28)))

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

    def test_append_embedded_tokens_with_implicit_and_explicit_positions(self) -> None:
        import torch
        from types import SimpleNamespace
        embedding = torch.nn.Embedding(32, 4)
        for positions in (None, torch.tensor([[5,6]]), torch.tensor([[[5,6]],[[7,8]],[[9,10]]])):
            original = positions.clone() if positions is not None else None
            execution = model_runtime.ModelExecution(
                runtime=SimpleNamespace(text_model=SimpleNamespace(get_input_embeddings=lambda:embedding),
                                        tokenizer=SimpleNamespace(decode=lambda ids:str(ids[0]))),
                input_ids=torch.tensor([[10,11]]), tokens=('a','b'),device='cpu',progress=lambda _:None,
                inputs_embeds=embedding(torch.tensor([[10,11]])).detach(),position_ids=positions)
            execution.append_generated_token(12)
            execution.append_generated_token(13)
            self.assertEqual(execution.input_ids.tolist(),[[10,11,12,13]])
            self.assertEqual(execution.inputs_embeds.shape,(1,4,4))
            torch.testing.assert_close(execution.inputs_embeds[:,2:],embedding(torch.tensor([[12,13]])))
            if original is None:
                self.assertIsNone(execution.position_ids)
            else:
                torch.testing.assert_close(execution.position_ids,
                    torch.cat((original,original[...,-1:]+1,original[...,-1:]+2),dim=-1))

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

        # Restrict candidates before ranking, while retaining global indices.
        projection = torch.tensor([[100., 0.], [90., 0.], [1., 0.], [2., 0.], [3., 0.], [4., 0.]])
        grouped = execution._top_projected_components(
            0, (1.,)*6, projection, channel_range=(2, 6), component_count=3,
            token_count=1, include_logit_lens=True, include_j_lens=False,
            include_r_lens=False, include_indices=(0,),
        )
        self.assertEqual([item.index for item in grouped], [5, 4, 3])

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

    def test_legacy_j_mid_uses_same_layer_post_map_as_fallback(self) -> None:
        import torch
        from types import SimpleNamespace

        messages = []
        model = SimpleNamespace(
            config=SimpleNamespace(
                _name_or_path="Qwen/Test", _commit_hash="revision",
                model_type="qwen3_5", layer_types=["full_attention"] * 4,
            ),
            model=SimpleNamespace(layers=[object()] * 4),
        )
        execution = model_runtime.ModelExecution(
            runtime=model_runtime.ModelRuntime(
                model=model,
                tokenizer=SimpleNamespace(_commit_hash="revision"),
            ),
            input_ids=None,
            tokens=("x",),
            device="cpu",
            progress=messages.append,
        )
        matrix = torch.eye(2)
        with mock.patch(
            "llmdbg.workspace_lens.source_map", return_value=None
        ), mock.patch(
            "llmdbg.j_lens.source_map", return_value=(matrix, {"model_id": "Qwen/Test"})
        ):
            source = execution._workspace_lens_source(
                "j", "Qwen/Test", "mid", 2, 2
            )

        assert source is not None
        self.assertIs(source[0], matrix)
        self.assertTrue(source[1]["fallback"])
        self.assertEqual(source[1]["fallback_from_tensor_key"], "mid.2")
        self.assertTrue(any("Post layer 2 J-Lens" in message for message in messages))

    def test_legacy_jpp_mid_uses_same_layer_post_map_as_fallback(self) -> None:
        import torch
        from types import SimpleNamespace

        messages = []
        model = SimpleNamespace(
            config=SimpleNamespace(
                _name_or_path="Qwen/Test", _commit_hash="revision",
                model_type="qwen3_5", layer_types=["full_attention"] * 4,
            ),
            model=SimpleNamespace(layers=[object()] * 4),
        )
        execution = model_runtime.ModelExecution(
            runtime=model_runtime.ModelRuntime(
                model=model,
                tokenizer=SimpleNamespace(_commit_hash="revision"),
            ),
            input_ids=None,
            tokens=("x",),
            device="cpu",
            progress=messages.append,
        )
        matrix = torch.eye(2)
        with mock.patch(
            "llmdbg.workspace_lens.source_map", return_value=None
        ), mock.patch(
            "llmdbg.jpp_lens.source_map", return_value=(matrix, {"model_id": "Qwen/Test"})
        ):
            source = execution._workspace_lens_source(
                "jpp", "Qwen/Test", "mid", 2, 2
            )

        assert source is not None
        self.assertIs(source[0], matrix)
        self.assertTrue(source[1]["fallback"])
        self.assertEqual(source[1]["fallback_from_tensor_key"], "mid.2")
        self.assertTrue(any("Post layer 2 J++ Lens" in message for message in messages))

    def test_legacy_final_jpp_mid_uses_post_identity_without_artifact(self) -> None:
        import torch
        from types import SimpleNamespace

        messages = []
        model = SimpleNamespace(
            config=SimpleNamespace(
                _name_or_path="Qwen/Test", _commit_hash="revision",
                model_type="qwen3_5", layer_types=["full_attention"] * 4,
            ),
            model=SimpleNamespace(layers=[object()] * 4),
        )
        execution = model_runtime.ModelExecution(
            runtime=model_runtime.ModelRuntime(
                model=model,
                tokenizer=SimpleNamespace(_commit_hash="revision"),
            ),
            input_ids=None,
            tokens=("x",),
            device="cpu",
            progress=messages.append,
        )
        with mock.patch("llmdbg.workspace_lens.source_map", return_value=None):
            source = execution._workspace_lens_source(
                "jpp", "Qwen/Test", "mid", 3, 2
            )

        assert source is not None
        self.assertTrue(torch.equal(source[0], torch.eye(2)))
        self.assertEqual(source[1]["artifact_kind"], "identity")
        self.assertTrue(source[1]["fallback"])
        self.assertTrue(any("final Post identity" in message for message in messages))

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
