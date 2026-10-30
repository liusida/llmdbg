"""Tests for the consolidated train_s_lens.py command and its online pipeline.

All model fixtures are tiny, local CPU models; no pretrained weights are loaded.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

import torch

from scripts.fit_lenses.v2 import train_s_lens as train
from scripts.fit_lenses.v2.publish_lenses import validate_artifact


class TrainSLensCLITest(unittest.TestCase):
    def test_defaults(self):
        args = train.arguments([])
        self.assertEqual(args.source, "online")
        self.assertFalse(hasattr(args, "cache"))
        self.assertEqual(args.epochs, 5)
        self.assertEqual(args.batch_size, 8192)

    def test_batch_argument(self):
        args = train.arguments(["--model", "Qwen/Qwen3.5-2B", "--batch-size", "32"])
        self.assertEqual(args.batch_size, 32)

    def test_snapshots_can_be_disabled(self):
        self.assertFalse(train.arguments([]).no_snapshots)
        self.assertTrue(train.arguments(['--no-snapshots']).no_snapshots)

    def test_removed_modes_are_rejected(self):
        import contextlib
        import io
        for options in [["--source", "cache"], ["--loss", "s-cos-future"]]:
            with self.subTest(options=options), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    train.arguments(options)

    def test_kl_alias_and_chunk_size(self):
        args = train.arguments(["--loss", "tuned", "--kl-chunk-size", "64"])
        self.assertEqual(args.loss, "kl")
        self.assertEqual(args.kl_chunk_size, 64)
        self.assertEqual(args.source, "online")


class TrainSLensOnlineCaptureTest(unittest.TestCase):
    def test_padding_masks_last_real_token_and_preserves_budget_prefix(self):
        documents = [{"ids": [1, 2, 3, 4, 5]}, {"ids": [6, 7, 8]}]
        ids, attention, valid = train.padded_batch(documents, 0, 1, "cpu")
        self.assertEqual(ids.tolist(), [[1, 2, 3, 4, 5], [6, 7, 8, 0, 0]])
        self.assertEqual(attention.tolist(), [[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]])
        self.assertEqual(ids[valid].tolist(), [2, 3, 4, 7])
        self.assertEqual(ids[valid][:2].tolist(), [2, 3])

    def test_batched_capture_matches_independent_sequences(self):
        from tests.test_fit_workspace_lenses import TinyModel
        class MaskedTinyModel(TinyModel):
            def forward(self, input_ids, attention_mask=None, use_cache=False):
                return super().forward(input_ids=input_ids, use_cache=use_cache)
        model = MaskedTinyModel().eval()
        documents = [{"ids": [1, 2, 3, 4, 5]}, {"ids": [6, 7, 8]}]
        keys = train.site_keys(2, ["embed", "mid", "post"])
        batch = train.forward_batch(model, keys, documents, 0, 1, "gpt2", "cpu")
        singles = [train.forward_batch(model, keys, [d], 0, 1, "gpt2", "cpu") for d in documents]
        for key in batch:
            expected = torch.cat([single[key] for single in singles])
            torch.testing.assert_close(batch[key], expected)
            torch.testing.assert_close(batch[key].T @ batch[key], expected.T @ expected)

    def test_capture_is_gradient_free_and_cleans_up(self):
        from tests.test_fit_workspace_lenses import TinyModel
        model = TinyModel()
        keys = train.site_keys(2, ["embed", "mid", "post"])
        with torch.inference_mode(), train.capture_sites(model, keys) as values:
            model(input_ids=torch.tensor([[1, 2, 3, 4]]), use_cache=False)
        self.assertEqual(set(values), set(keys) | {"post.1"})
        self.assertTrue(all(not value.requires_grad for value in values.values()))
        self.assertFalse(model.transformer.drop._forward_hooks)

    def test_selected_sites_exclude_final_identity_map(self):
        args = train.arguments(["--model", "Qwen/Qwen3.5-2B"])
        self.assertEqual(args.tokens, 1_000_000)
        keys = train.site_keys(24, args.sites)
        self.assertEqual(len(keys), 48)
        self.assertIn("mid.23", keys)
        self.assertNotIn("post.23", keys)
        with self.assertRaises(ValueError):
            train.site_keys(24, ["post"], [24])

    def test_online_source_exact_budget_and_frozen_model(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from tests.test_fit_workspace_lenses import TinyModel
        class MaskedTinyModel(TinyModel):
            def forward(self, input_ids, attention_mask=None, use_cache=False):
                return super().forward(input_ids=input_ids, use_cache=use_cache)
        model = MaskedTinyModel()
        docs = [{"ids": [1, 2, 3, 4, 5], "hash": "a"}, {"ids": [6, 7, 8, 9, 10], "hash": "b"}]
        batches = train.document_batches(docs, 2, 1, 5)
        self.assertEqual(sum(take for _, take in batches), 5)
        args = train.arguments(["--model", "openai-community/gpt2", "--tokens", "5",
                                "--skip-first", "1", "--forward-batch-size", "2"])
        original_forward = train.forward_batch
        with patch.object(train.AutoConfig, "from_pretrained", return_value=SimpleNamespace(model_type="gpt2", _commit_hash="sha")), \
             patch.object(train.AutoTokenizer, "from_pretrained", return_value=SimpleNamespace(pad_token_id=0)), \
             patch.object(train, "corpus", return_value=(docs[:1], docs[1:], docs, "fingerprint")), \
             patch.object(train, "load_model_to_cuda", return_value=model), \
             patch.object(train, "forward_batch", wraps=original_forward) as forward:
            source = train.OnlineSource(args)
            shard = source.manifest["splits"]["train"][0]
            values = source.load(None, shard)
            self.assertEqual(len(values["target"]), 5)
            self.assertTrue(all(not p.requires_grad for p in model.parameters()))
            maps = train.make_maps(3, len(source.keys))
            optimizer = torch.optim.Adam(maps.parameters())
            train.train_batch(maps, optimizer, source.keys, values, "target", 5)
            self.assertTrue(all(p.grad is None for p in model.parameters()))
            source.load(None, shard)
            self.assertEqual(forward.call_count, 2)

    def test_previous_target_does_not_cross_document_boundary(self):
        from tests.test_fit_workspace_lenses import TinyModel
        class MaskedModel(TinyModel):
            def forward(self,input_ids,attention_mask=None,use_cache=False):
                return super().forward(input_ids=input_ids,use_cache=use_cache)
        model=MaskedModel().eval()
        documents=[{'ids':[1,2,3,4,5]},{'ids':[6,7,8]}]
        keys=['post.0']
        result=train.forward_batch(model,keys,documents,0,1,'gpt2','cpu',include_previous_target=True)
        expected=[]
        for document in documents:
            values=train.forward_batch(model,keys,[document],0,0,'gpt2','cpu')
            expected.append(values['post.1'][:-1])
        torch.testing.assert_close(result['previous_target'],torch.cat(expected))
        self.assertEqual(result['current_token_ids'].tolist(),[2,3,4,7])
        self.assertEqual(len(result['previous_target']),4)

    def test_gate_checks_frozen_previous_prediction(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        head=torch.nn.Linear(2,2,bias=False)
        with torch.no_grad():head.weight.copy_(torch.eye(2))
        model=SimpleNamespace(get_output_embeddings=lambda:head)
        previous=torch.tensor([[10.,1.],[1.,10.],[4.,1.]])
        with patch.object(train,'model_parts',return_value=(None,[],torch.nn.Identity(),'test')):
            gate=train.previous_prediction_gate(model,previous,torch.tensor([0,0,1]),chunk_size=2)
        self.assertEqual(gate.tolist(),[True,False,False])

    def test_epoch_shuffle_is_deterministic(self):
        shards = [{} for _ in range(20)]
        self.assertEqual(train.epoch_order(shards, 42, 1), train.epoch_order(shards, 42, 1))
        self.assertNotEqual(train.epoch_order(shards, 42, 1), train.epoch_order(shards, 42, 2))


class TrainSLensGeometricLossTest(unittest.TestCase):
    def test_cosine_loss_is_directional_and_has_gradients(self):
        target = torch.tensor([[1., 0.], [0., 1.]])
        prediction = torch.tensor([[1., 1.], [1., 1.]], requires_grad=True)
        loss, mse, cosine = train.readout_loss(prediction, target, "cosine")
        self.assertAlmostEqual(loss.item(), 1-2**-.5, places=6)
        scaled_loss, _, _ = train.readout_loss(prediction*7, target*3, "cosine")
        torch.testing.assert_close(loss, scaled_loss)
        loss.backward()
        self.assertGreater(prediction.grad.abs().sum().item(), 0)
        zero_loss, _, _ = train.readout_loss(torch.zeros_like(target), target, "cosine")
        self.assertTrue(torch.isfinite(zero_loss))
        args = train.arguments(["--loss", "cos"])
        self.assertEqual(args.loss, "cosine")
        self.assertEqual(args.source, "online")

    def test_cosine_updates_improve_alignment(self):
        torch.manual_seed(42)
        maps = train.make_maps(3, 1)
        optimizer = torch.optim.Adam(maps.parameters(), lr=.03)
        x = torch.randn(32, 3)
        values = {"embed": x, "target": x @ torch.tensor([[0., 1., 0.], [-1., 0., 0.], [0., 0., 1.]])}
        first = train.train_batch(maps, optimizer, ["embed"], values, "target", 32, "cosine")
        for _ in range(80):
            last = train.train_batch(maps, optimizer, ["embed"], values, "target", 32, "cosine")
        self.assertLess(last["loss"], first["loss"] * .1)
        self.assertGreater(last["cosine"], first["cosine"])

    def test_gradient_updates_reduce_loss(self):
        torch.manual_seed(42)
        maps = train.make_maps(3, 1)
        optimizer = torch.optim.Adam(maps.parameters(), lr=.03)
        values = {"embed": torch.randn(20, 3)}
        values["post.1"] = values["embed"] * 2
        first = train.train_batch(maps, optimizer, ["embed"], values, "post.1", 20)
        for _ in range(60):
            last = train.train_batch(maps, optimizer, ["embed"], values, "post.1", 20)
        self.assertLess(last["mse"], first["mse"] * .1)
        self.assertTrue(all(value.grad is None for value in values.values()))

    def test_gated_91_loss_weights_and_fallback(self):
        prediction = torch.tensor([[1.,0.],[1.,0.]],requires_grad=True)
        target = torch.tensor([[1.,0.],[1.,0.]])
        previous = torch.tensor([[0.,1.],[0.,1.]])
        loss,_ = train.gated_cosine_loss(prediction,target,previous,torch.tensor([True,False]))
        self.assertAlmostEqual(loss.item(), .05, places=6)
        loss46,_ = train.gated_cosine_loss(prediction,target,previous,torch.tensor([True,False]),next_weight=.4)
        self.assertAlmostEqual(loss46.item(), .3, places=6)
        loss.backward()
        self.assertGreater(prediction.grad[0].abs().sum().item(),0)
        self.assertEqual(prediction.grad[1].abs().sum().item(),0)
        loss,_ = train.gated_cosine_loss(prediction,target,previous,torch.tensor([False,False]))
        self.assertAlmostEqual(loss.item(),0,places=6)
        args=train.arguments(['--loss','s-cos-next-current-91'])
        self.assertEqual(args.source,'online')

    def test_gated_training_accepts_frozen_inference_activations(self):
        maps = train.make_maps(3, 1)
        optimizer = torch.optim.Adam(maps.parameters(), lr=.001)
        with torch.inference_mode():
            values = {'embed': torch.randn(4, 3), 'target': torch.randn(4, 3),
                      'previous_target': torch.randn(4, 3),
                      'current_gate': torch.tensor([True, False, True, False])}
        scores = train.train_batch(maps, optimizer, ['embed'], values, 'target', 4,
                                   loss_kind='s-cos-next-current-91')
        self.assertEqual(scores['current_gate_fraction'], .5)
        self.assertTrue(torch.isfinite(torch.tensor(scores['loss'])))


class TrainSLensKLLossTest(unittest.TestCase):
    def readout(self):
        return train.FrozenReadout(torch.nn.LayerNorm(3), torch.nn.Linear(3, 7, bias=False))

    def test_teacher_to_student_kl_and_detached_teacher(self):
        student_logits = torch.tensor([[2., -1., .5], [0., 1., -1.]], requires_grad=True)
        teacher_logits = torch.tensor([[0., 1., 2.], [2., -1., 0.]], requires_grad=True)
        teacher_log = teacher_logits.log_softmax(-1)
        student_log = student_logits.log_softmax(-1)
        loss = train.distillation_kl(student_log, teacher_log)
        expected = (teacher_log.exp() * (teacher_log - student_log)).sum(-1).mean()
        torch.testing.assert_close(loss, expected)
        self.assertGreater(abs(loss.item() - train.distillation_kl(teacher_log, student_log).item()), .01)
        loss.backward()
        self.assertIsNone(teacher_logits.grad)
        self.assertGreater(student_logits.grad.abs().sum().item(), 0)

    def test_chunked_gradients_and_single_update_match_dense(self):
        torch.manual_seed(42)
        readout = self.readout()
        chunked = train.make_maps(3, 2)
        dense = copy.deepcopy(chunked)
        keys = ["mid.0", "post.0"]
        values = {key: torch.randn(7, 3) for key in keys + ["target"]}
        chunked_optimizer = torch.optim.SGD(chunked.parameters(), lr=.01)
        dense_optimizer = torch.optim.SGD(dense.parameters(), lr=.01)
        score = train.train_batch(chunked, chunked_optimizer, keys, values, "target", 7,
                                  "kl", readout=readout, kl_chunk_size=3)
        with torch.no_grad():
            teacher = readout(values["target"])
        losses = []
        for key, layer in zip(keys, dense):
            loss = train.distillation_kl(readout(layer(values[key])), teacher)
            losses.append(loss.item())
            loss.backward()
        dense_optimizer.step()
        self.assertAlmostEqual(score["loss"], sum(losses)/len(losses), places=6)
        for a, b in zip(chunked, dense):
            torch.testing.assert_close(a.weight, b.weight, rtol=1e-6, atol=1e-7)
            torch.testing.assert_close(a.weight.grad, b.weight.grad, rtol=1e-5, atol=1e-6)
            self.assertIsNone(a.bias)
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in readout.parameters()))

    def test_kl_training_reduces_loss(self):
        torch.manual_seed(4)
        readout = self.readout()
        maps = train.make_maps(3, 1)
        optimizer = torch.optim.Adam(maps.parameters(), lr=.03)
        x = torch.randn(12, 3)
        values = {"mid.0": x, "target": x @ torch.tensor([[0., 1., 0.], [-1., 0., 0.], [0., 0., 1.]])}
        first = train.train_batch(maps, optimizer, ["mid.0"], values, "target", 12, "kl", readout, 5)
        for _ in range(50):
            last = train.train_batch(maps, optimizer, ["mid.0"], values, "target", 12, "kl", readout, 5)
        self.assertLess(last["loss"], first["loss"] * .2)
        self.assertEqual(last["loss"], last["kl"])

    def test_evaluation_reports_kl_of_exported_weights(self):
        torch.manual_seed(42)
        readout = self.readout()
        maps = train.make_maps(3, 1)
        values = {"mid.0": torch.randn(7, 3), "target": torch.randn(7, 3)}
        result = train.evaluate(maps, ["mid.0"], None, [{}], loader=lambda *_: values,
                                readout=readout, kl_chunk_size=3)
        expected = train.distillation_kl(readout(values["mid.0"]), readout(values["target"]))
        self.assertAlmostEqual(result["mid.0"]["kl_divergence"], expected.item(), places=6)
        self.assertEqual(result["mid.0"]["positions"], 7)


class TrainSLensCheckpointTest(unittest.TestCase):
    def test_export_is_v2_compatible(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            metadata = {"model_id": "test/model", "model_revision": "sha",
                        "tokenizer_revision": "sha", "hidden_size": 2,
                        "sites": {"embed": True, "mid": [], "post": []}}
            train.export(path, {"embed": torch.eye(2)}, metadata, {})
            manifest = validate_artifact(path)
            self.assertEqual(manifest["method"], "s")
            self.assertEqual(manifest["dtype"], "float16")

    def test_retained_snapshots_record_actual_update_boundary(self):
        self.assertEqual(train.crossed_milestones(0, 8192), [1000])
        self.assertEqual(train.crossed_milestones(8192, 16384), [10_000])
        self.assertEqual(train.crossed_milestones(1000, 1000), [])
        self.assertEqual(train.crossed_milestones(0, 20000), [1000, 10_000])
        with tempfile.TemporaryDirectory() as directory:
            metadata = {"model_id": "test/model", "model_revision": "sha",
                        "tokenizer_revision": "sha", "hidden_size": 3,
                        "sites": {"embed": True, "mid": [], "post": []}}
            train.retain_snapshot(Path(directory), "positions-0000001000", train.make_maps(3, 1),
                                  ["embed"], metadata, 8192, 1, milestone=1000)
            manifest = validate_artifact(Path(directory) / "checkpoints/positions-0000001000")
            self.assertEqual(manifest["snapshot"]["actual_positions"], 8192)
            self.assertEqual(manifest["snapshot"]["requested_positions"], 1000)
            self.assertTrue(manifest["snapshot"]["weights_only"])

    def test_optimizer_resume_matches_uninterrupted_update(self):
        torch.manual_seed(42)
        maps = train.make_maps(3, 1)
        optimizer = torch.optim.Adam(maps.parameters(), lr=.01, foreach=False)
        values = {"embed": torch.randn(20, 3), "post.1": torch.randn(20, 3)}
        train.train_batch(maps, optimizer, ["embed"], values, "post.1", 7)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.safetensors"
            train.save_state(path, maps, optimizer, "recipe", 7, 2)
            restored = train.make_maps(3, 1)
            count, cursor, states = train.restore_maps(path, restored, "recipe")
            self.assertEqual((count, cursor), (7, 2))
            resumed_optimizer = torch.optim.Adam(restored.parameters(), lr=.01, foreach=False)
            resumed_optimizer.state[restored[0].weight] = states[0]
            train.train_batch(maps, optimizer, ["embed"], values, "post.1", 11)
            train.train_batch(restored, resumed_optimizer, ["embed"], values, "post.1", 11)
            torch.testing.assert_close(maps[0].weight, restored[0].weight, rtol=0, atol=0)
            with self.assertRaises(ValueError):
                train.restore_maps(path, restored, "different recipe")


class TrainSLensMonitoringTest(unittest.TestCase):
    def test_tensorboard_named_run_and_scalar_tags(self):
        try:
            from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        except ImportError:
            self.skipTest("optional training dependency not installed")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "output"
            output.mkdir()
            args = train.arguments(["--run-name", "adam-lr001",
                                    "--tensorboard-dir", str(root / "events")])
            writer = train.create_writer(args, output, {}, 0)
            train.log_training(writer, {"loss": .25, "cosine": .75, "mse": .5, "rel_mse": .8,
                                       "sites": {"post.12": {"mse": .4, "rel_mse": .7}}},
                               8192, 1, 1000, .001)
            train.log_evaluation(writer, "validation", {"post.12": {
                "mean_cosine": .6, "relative_squared_error": .7}}, 8192)
            writer.close()
            events = EventAccumulator(str(root / "events/adam-lr001")).Reload()
            self.assertAlmostEqual(events.Scalars("train/loss_mse")[0].value, .5)
            self.assertAlmostEqual(events.Scalars("train/loss")[0].value, .25)
            self.assertAlmostEqual(events.Scalars("train/cosine_similarity")[0].value, .75)
            self.assertEqual(events.Scalars("train/loss_mse")[0].step, 8192)
            self.assertIn("train_sites/post.12/mse", events.Tags()["scalars"])
            self.assertIn("validation_sites/post.12/mean_cosine", events.Tags()["scalars"])
            args.resume, args.run_name = True, None
            resumed = train.create_writer(args, output, {}, 8192)
            resumed.close()
            self.assertEqual(json.loads((output / "tensorboard-run.json").read_text())["run_name"], "adam-lr001")


class TrainSLensCommandTest(unittest.TestCase):
    def test_online_command_cosine_and_kl_export_and_resume(self):
        import contextlib
        import io
        from types import SimpleNamespace
        from unittest.mock import patch
        from transformers import GPT2Config, GPT2LMHeadModel

        docs = [{"ids": [1, 2, 3, 4, 5, 6], "hash": str(i)} for i in range(4)]
        config = GPT2Config(n_layer=2, n_head=1, n_embd=8, n_positions=8,
                            vocab_size=16, pad_token_id=0, bos_token_id=1, eos_token_id=1)
        config._commit_hash = "tiny-local-revision"
        for loss in ["cosine", "kl"]:
            with self.subTest(loss=loss), tempfile.TemporaryDirectory() as directory:
                model = GPT2LMHeadModel(config).eval()
                options = ["--model", "openai-community/gpt2", "--tokens", "5",
                           "--max-seq-len", "8", "--skip-first", "1", "--sites", "post",
                           "--layers", "0", "--forward-batch-size", "2", "--batch-size", "3",
                           "--validation-prompts", "1", "--test-prompts", "1", "--epochs", "1",
                           "--loss", loss, "--kl-chunk-size", "2", "--no-tensorboard",
                           "--no-snapshots", "--output", directory]
                with patch.object(train.AutoConfig, "from_pretrained", return_value=config), \
                     patch.object(train.AutoTokenizer, "from_pretrained", return_value=SimpleNamespace(pad_token_id=0)), \
                     patch.object(train, "corpus", return_value=(docs[:1], docs[1:2], docs[2:], "tiny")), \
                     patch.object(train, "load_model_to_cuda", return_value=model), \
                     patch.object(train, "move_model_to_cuda", side_effect=lambda value, **_: value), \
                     contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    train.main(options)
                    output = Path(directory)
                    manifest = validate_artifact(output)
                    self.assertEqual(manifest["optimizer_positions"], 5)
                    self.assertEqual(manifest["training_positions"], 5)
                    self.assertEqual(manifest["method_name"], f"supervised_adam_{loss}")
                    self.assertEqual(set(manifest["tensor_shapes"]), {"post.0"})
                    self.assertTrue(all(not p.requires_grad and p.grad is None for p in model.parameters()))
                    self.assertTrue((output / "optimizer-state.safetensors").is_file())
                    train.main(options + ["--resume"])
                    resumed = validate_artifact(output)
                    self.assertEqual(resumed["sha256"], manifest["sha256"])
                    self.assertEqual(resumed["optimizer_positions"], 5)


if __name__ == "__main__":
    unittest.main()
