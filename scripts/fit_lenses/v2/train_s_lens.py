#!/usr/bin/env python3
"""Train S-Lens maps in one place: frozen capture, losses, Adam, and v2 export."""
from __future__ import annotations

import argparse
import atexit
from contextlib import contextmanager
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import random
import re
import sys
import time
from typing import Any

import torch
from gb10_load_llm import load_model_to_cuda, move_model_to_cuda
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm.auto import tqdm
from torch.nn import functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PILE_DATASET_ID = "NeelNanda/pile-10k"
SUPPORTED_MODELS = (
    "openai-community/gpt2",
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3.5-0.8B",
    "Qwen/Qwen3.5-2B-Base",
    "Qwen/Qwen3.5-2B",
    "Qwen/Qwen3.5-9B",
    "Qwen/Qwen3.5-27B",
)


def model_path(model_id: str) -> str:
    return model_id.strip("/").replace("/", "--")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolved_revision(value: Any) -> str | None:
    direct = getattr(value, "_commit_hash", None)
    if direct:
        return str(direct)
    revision = getattr(value, "init_kwargs", {}).get("_commit_hash")
    return str(revision) if revision else None


def model_parts(model: Any) -> tuple[Any, Any, Any, str]:
    model_type = str(getattr(model.config, "model_type", ""))
    if model_type == "gpt2":
        return model.transformer.drop, model.transformer.h, model.transformer.ln_f, "gpt2"
    if model_type == "qwen3":
        return model.model.embed_tokens, model.model.layers, model.model.norm, "qwen3"
    if model_type in {"qwen3_5", "qwen3_5_text"}:
        return model.model.embed_tokens, model.model.layers, model.model.norm, "qwen3_5"
    raise ValueError(f"unsupported architecture {model_type!r}")


def hidden_size(model: Any) -> int:
    config = getattr(model.config, "text_config", model.config)
    return int(getattr(config, "hidden_size", getattr(config, "n_embd", 0)))


def site_keys(count, sites, layers=None):
    selected = list(range(count)) if layers is None else sorted(set(layers))
    if any(i < 0 or i >= count for i in selected):
        raise ValueError("layer outside model range")
    keys = (["embed"] if "embed" in sites else [])
    keys += [f"mid.{i}" for i in selected if "mid" in sites]
    keys += [f"post.{i}" for i in selected if "post" in sites and i < count - 1]
    if not keys:
        raise ValueError("no non-identity source sites selected")
    return keys


def padded_batch(documents, pad_token_id, skip_first, device):
    """Right-pad independent sequences; select only original valid positions.

    Flattening the selection preserves document order and the exact token-budget
    prefix used by unbatched training. Each document's last real token is excluded.
    """
    lengths = torch.tensor([len(d["ids"]) for d in documents], device=device)
    positions = torch.arange(int(lengths.max()), device=device)[None, :]
    attention = positions < lengths[:, None]
    ids = torch.full(attention.shape, pad_token_id, dtype=torch.long, device=device)
    for i, document in enumerate(documents):
        ids[i, :len(document["ids"])] = torch.tensor(document["ids"], device=device)
    valid = (positions >= skip_first) & (positions < lengths[:, None] - 1)
    return ids, attention.long(), valid


@torch.inference_mode()
def forward_batch(model, keys, documents, pad_token_id, skip_first, architecture, device="cuda",
                  include_previous_target=False):
    if include_previous_target and skip_first < 1:
        raise ValueError("previous-target capture requires skip_first >= 1")
    ids, attention, valid = padded_batch(documents, pad_token_id, skip_first, device)
    with capture_sites(model, keys) as captured:
        kwargs = {} if architecture == "gpt2" else {"logits_to_keep": 1}
        model(input_ids=ids, attention_mask=attention, use_cache=False, **kwargs)
    result = {k: v[valid] for k, v in captured.items()}
    if include_previous_target:
        _, layers, _, _ = model_parts(model)
        # Shift inside each sequence before masking/flattening. Never borrow
        # the last token of another document or a previous optimizer batch.
        result['previous_target'] = captured[f'post.{len(layers)-1}'][:, :-1][valid[:, 1:]]
        result['current_token_ids'] = ids[valid]
    return result


@contextmanager
def capture_sites(model, keys):
    embed, layers, _, architecture = model_parts(model)
    wanted = set(keys) | {f"post.{len(layers)-1}"}
    values, handles = {}, []
    def output(key):
        def hook(module, inputs, result):
            values[key] = (result if torch.is_tensor(result) else result[0]).detach()
        return hook
    def incoming(key):
        def hook(module, inputs):
            values[key] = inputs[0].detach()
        return hook
    try:
        if "embed" in wanted:
            handles.append(embed.register_forward_hook(output("embed")))
        for i, layer in enumerate(layers):
            if f"post.{i}" in wanted:
                handles.append(layer.register_forward_hook(output(f"post.{i}")))
            if f"mid.{i}" in wanted:
                norm = layer.ln_2 if architecture == "gpt2" else layer.post_attention_layernorm
                handles.append(norm.register_forward_pre_hook(incoming(f"mid.{i}")))
        yield values
    finally:
        for handle in handles:
            handle.remove()


def corpus(tokenizer, args):
    from datasets import load_dataset
    dataset = load_dataset(PILE_DATASET_ID, split="train")
    indices = list(range(args.dataset_offset, len(dataset)))
    random.Random(args.seed).shuffle(indices)
    documents, seen = [], set()
    for index in indices:
        ids = tokenizer(str(dataset[index]["text"]), truncation=True,
                        max_length=args.max_seq_len)["input_ids"]
        digest = hashlib.sha256(json.dumps(ids).encode()).hexdigest()
        if len(ids) <= args.skip_first + 1 or digest in seen:
            continue
        seen.add(digest)
        documents.append({"row": index, "hash": digest, "ids": ids})
    split = args.validation_prompts + args.test_prompts
    if len(documents) <= split:
        raise ValueError("not enough distinct documents for held-out splits")
    train = documents[split:]
    if sum(len(d["ids"]) - args.skip_first - 1 for d in train) < args.tokens:
        raise ValueError("corpus cannot supply requested training token budget")
    return (documents[:args.validation_prompts], documents[args.validation_prompts:split],
            train, dataset._fingerprint)


def export(directory, maps, metadata, report):
    directory.mkdir(parents=True, exist_ok=True)
    artifact = directory / "lens.safetensors"
    save_file({k: v.half().contiguous() for k, v in maps.items()}, str(artifact))
    manifest = dict(metadata, format_version=2, artifact=artifact.name,
                    sha256=sha256(artifact), method="s",
                    method_name=metadata.get("method_name", "supervised_adam_mse"),
                    provenance="llmdbg_supervised_fit", dtype="float16",
                    tensor_shapes={k: list(v.shape) for k, v in maps.items()},
                    artifact_bytes=artifact.stat().st_size,
                    created_at=datetime.now(UTC).isoformat(), evaluation=report)
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", choices=["online"], default="online")
    p.add_argument("--model", default="Qwen/Qwen3.5-2B", choices=SUPPORTED_MODELS)
    p.add_argument("--revision", default="main")
    p.add_argument("--tokens", type=int, default=1_000_000)
    p.add_argument("--forward-batch-size", type=int, default=32, help="online documents per frozen forward")
    p.add_argument("--max-seq-len", type=int, default=256)
    p.add_argument("--skip-first", type=int, default=4)
    p.add_argument("--dataset-offset", type=int, default=1000)
    p.add_argument("--validation-prompts", type=int, default=20)
    p.add_argument("--test-prompts", type=int, default=50)
    p.add_argument("--sites", nargs="+", choices=["embed", "mid", "post"], default=["embed", "mid", "post"])
    p.add_argument("--layers", nargs="+", type=int)
    p.add_argument("--batch-size", type=int, default=8192, help="activation positions per S-Lens update")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--loss", choices=["mse", "cosine", "cos", "kl", "tuned", "s-cos-next-current-91", "s-cos-46"], default="mse",
                   help="MSE, cosine, gated next/current cosine, or bias-free Tuned Lens KL; tuned aliases kl")
    p.add_argument("--kl-chunk-size", type=int, default=128,
                   help="positions per full-vocabulary KL chunk; optimizer batch size is unchanged")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--checkpoint-every", type=int, default=100_000)
    p.add_argument("--no-snapshots", action="store_true",
                   help="skip retained milestone/epoch artifacts; keep latest resumable optimizer state and final artifact")
    p.add_argument("--output", type=Path)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--run-name", help="TensorBoard run label (letters, numbers, dot, dash, underscore)")
    p.add_argument("--tensorboard-dir", type=Path, help="event root; default: build/tensorboard/s-lens")
    p.add_argument("--no-tensorboard", action="store_true")
    a = p.parse_args(argv)
    if a.loss == "cos":
        a.loss = "cosine"
    if a.loss == "tuned":
        a.loss = "kl"
    if a.kl_chunk_size < 1:
        p.error("kl-chunk-size must be positive")
    if a.loss in ('s-cos-next-current-91', 's-cos-46') and a.skip_first < 1:
        p.error("s-cos-next-current-91 requires skip-first >= 1")
    if min(a.batch_size, a.epochs, a.checkpoint_every) < 1 or a.learning_rate <= 0:
        p.error("batch-size, epochs, and learning-rate must be positive")
    if a.run_name and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", a.run_name):
        p.error("run-name must be a simple label, not a path")
    if (min(a.tokens, a.forward_batch_size, a.validation_prompts, a.test_prompts) < 1
            or a.skip_first < 0 or a.max_seq_len <= a.skip_first+1 or a.dataset_offset < 0):
        p.error("invalid online capture settings")
    return a


@torch.inference_mode()
def previous_prediction_gate(model, previous_target, current_token_ids, chunk_size=128):
    _, _, norm, _ = model_parts(model)
    head = model.get_output_embeddings()
    matches = []
    for start in range(0, len(previous_target), chunk_size):
        logits = head(norm(previous_target[start:start+chunk_size].to(head.weight.dtype)))
        matches.append(logits.argmax(-1) == current_token_ids[start:start+chunk_size])
    return torch.cat(matches)


def document_batches(documents, batch_size, skip_first, budget=None):
    batches, remaining = [], budget
    for start in range(0, len(documents), batch_size):
        if remaining is not None and remaining <= 0:
            break
        batch = documents[start:start+batch_size]
        positions = sum(max(0, len(d["ids"])-skip_first-1) for d in batch)
        if not positions:
            continue
        take = positions if remaining is None else min(positions, remaining)
        batches.append((batch, take))
        if remaining is not None:
            remaining -= take
    if remaining is not None and remaining > 0:
        raise ValueError("not enough documents for training budget")
    return batches


class OnlineSource:
    def __init__(self, args):
        config = AutoConfig.from_pretrained(args.model, revision=args.revision)
        revision = resolved_revision(config) or args.revision
        tokenizer = AutoTokenizer.from_pretrained(args.model, revision=revision)
        validation, test, train, fingerprint = corpus(tokenizer, args)
        self.model = load_model_to_cuda(
            AutoModelForCausalLM, args.model, revision=revision,
            dtype=torch.float32 if config.model_type == "gpt2" else torch.bfloat16,
            low_cpu_mem_usage=True, attn_implementation="sdpa", touch="auto")
        self.model.eval().requires_grad_(False)
        _, layers, _, self.architecture = model_parts(self.model)
        self.keys = site_keys(len(layers), args.sites, args.layers)
        self.final = f"post.{len(layers)-1}"
        self.skip_first = args.skip_first
        self.with_previous = args.loss in ('s-cos-next-current-91', 's-cos-46')
        self.pad = tokenizer.pad_token_id
        if self.pad is None:
            self.pad = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
        self.batches, splits = {}, {}
        for split, docs in [("train", train), ("validation", validation), ("test", test)]:
            self.batches[split] = document_batches(docs, args.forward_batch_size, args.skip_first,
                                                  args.tokens if split == "train" else None)
            if not self.batches[split]:
                raise ValueError(f"{split} split has no eligible source positions")
            splits[split] = [{"split": split, "index": i, "positions": take}
                             for i, (_, take) in enumerate(self.batches[split])]
        capture_recipe = {name: getattr(args, name) for name in [
            "model", "revision", "tokens", "forward_batch_size", "max_seq_len", "skip_first",
            "dataset_offset", "seed", "validation_prompts", "test_prompts", "sites", "layers"]}
        capture_recipe.update(model_revision=revision, dataset_fingerprint=fingerprint,
                              document_hashes=[d["hash"] for d in validation+test+train])
        self.manifest = {"model_id": args.model, "model_revision": revision,
                         "tokenizer_revision": resolved_revision(tokenizer) or revision,
                         "hidden_size": hidden_size(self.model), "keys": self.keys,
                         "target": {"site": "post", "layer": len(layers)-1},
                         "capture_recipe": capture_recipe, "splits": splits}

    def load(self, _directory, shard):
        documents, take = self.batches[shard["split"]][shard["index"]]
        kwargs = {'include_previous_target': True} if self.with_previous else {}
        values = forward_batch(self.model, self.keys, documents, self.pad,
                               self.skip_first, self.architecture,
                               device=next(self.model.parameters()).device, **kwargs)
        # Only the trainable maps have a downstream gradient graph.
        result = {**{k: values[k][:take] for k in self.keys}, "target": values[self.final][:take]}
        if self.with_previous:
            result['previous_target'] = values['previous_target'][:take]
            result['current_gate'] = previous_prediction_gate(
                self.model, result['previous_target'], values['current_token_ids'][:take])
        return result


class FrozenReadout(torch.nn.Module):
    """Reuse the resident model's final norm/head; add no learned parameters."""
    def __init__(self, final_norm, output_head):
        super().__init__()
        self.final_norm = final_norm
        self.output_head = output_head
        self.eval().requires_grad_(False)

    def forward(self, vectors):
        logits = self.output_head(self.final_norm(vectors.to(self.output_head.weight.dtype)))
        return logits.float().log_softmax(dim=-1)


def distillation_kl(student_log_probs, teacher_log_probs):
    """Mean KL(teacher || student), in nats per token, at temperature 1."""
    return F.kl_div(student_log_probs, teacher_log_probs.detach(),
                    log_target=True, reduction="batchmean")


def train_kl_batch(maps, optimizer, keys, values, final, take, readout, chunk_size):
    if readout is None:
        raise ValueError("KL loss requires the frozen final norm and output head")
    if take < 1 or chunk_size < 1:
        raise ValueError("KL batch and chunk size must be positive")
    optimizer.zero_grad(set_to_none=True)
    totals = {key: {"loss": 0., "mse": 0., "cosine": 0.} for key in keys}
    target = values[final][:take].float().clone()
    energy = target.square().mean().item()
    for start in range(0, take, chunk_size):
        y = target[start:start+chunk_size]
        weight = len(y) / take
        with torch.no_grad():
            teacher = readout(y)
        # Share teacher logits across sites, and free each student's vocabulary
        # graph immediately. Do not retain B x vocabulary graphs for all sites.
        for key, layer in zip(keys, maps):
            x = values[key][start:start+len(y)].float().clone()
            with torch.autocast(device_type=x.device.type, dtype=torch.bfloat16, enabled=x.is_cuda):
                prediction = layer(x)
            loss = distillation_kl(readout(prediction), teacher)
            if not torch.isfinite(loss):
                raise ValueError(f"non-finite KL loss at {key}; optimizer not advanced")
            (loss * weight).backward()
            with torch.no_grad():
                totals[key]["loss"] += loss.item() * weight
                totals[key]["mse"] += (prediction.float()-y).square().mean().item() * weight
                totals[key]["cosine"] += F.cosine_similarity(prediction.float(), y, dim=-1).mean().item() * weight
    # One optimizer step for the entire token batch, not one per vocabulary chunk.
    optimizer.step()
    for scores in totals.values():
        scores["kl"] = scores["loss"]
        scores["rel_mse"] = scores["mse"] / max(energy, 1e-30)
    result = {name: sum(scores[name] for scores in totals.values()) / len(keys)
              for name in ["loss", "kl", "mse", "rel_mse", "cosine"]}
    result["sites"] = totals
    return result


def create_writer(args, output, recipe, count):
    if args.no_tensorboard:
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as error:
        raise RuntimeError("TensorBoard requires: uv sync --group training (or use --no-tensorboard)") from error
    settings_path = output / "tensorboard-run.json"
    previous = json.loads(settings_path.read_text()) if args.resume and settings_path.exists() else {}
    name = args.run_name or previous.get("run_name") or datetime.now(UTC).strftime("s-%Y%m%d-%H%M%S-%f")
    root = args.tensorboard_dir or Path(previous.get("root", ROOT / "build/tensorboard/s-lens"))
    directory = root / name
    if not args.resume and directory.exists() and any(directory.iterdir()):
        raise ValueError(f"TensorBoard run already exists: {directory}; choose another --run-name")
    writer = SummaryWriter(str(directory), purge_step=count+1 if args.resume else None, flush_secs=10)
    atexit.register(writer.close)
    settings_path.write_text(json.dumps({"run_name": name, "root": str(root.resolve())}, indent=2) + "\n")
    writer.add_text("configuration", json.dumps(recipe, indent=2), global_step=count)
    print(f"TensorBoard run: {name}\nEvents: {directory}", flush=True)
    return writer


def log_training(writer, scores, count, epoch, rate, learning_rate):
    if writer is None:
        return
    writer.add_scalar("train/loss_mse", scores["mse"], count)
    writer.add_scalar("train/loss", scores.get("loss", scores["mse"]), count)
    if "cosine" in scores:
        writer.add_scalar("train/cosine_similarity", scores["cosine"], count)
    if "kl" in scores:
        writer.add_scalar("train/kl_divergence", scores["kl"], count)
    for metric in ['current_gate_fraction', 'current_cosine']:
        if metric in scores:
            writer.add_scalar(f'train/{metric}', scores[metric], count)
    writer.add_scalar("train/relative_mse", scores["rel_mse"], count)
    writer.add_scalar("performance/positions_per_second", rate, count)
    writer.add_scalar("train/epoch", epoch, count)
    writer.add_scalar("train/learning_rate", learning_rate, count)
    for key, values in scores.get("sites", {}).items():
        writer.add_scalar(f"train_sites/{key}/mse", values["mse"], count)
        writer.add_scalar(f"train_sites/{key}/relative_mse", values["rel_mse"], count)
        writer.add_scalar(f"train_sites/{key}/loss", values.get("loss", values["mse"]), count)
        if "cosine" in values:
            writer.add_scalar(f"train_sites/{key}/cosine_similarity", values["cosine"], count)
        if "kl" in values:
            writer.add_scalar(f"train_sites/{key}/kl_divergence", values["kl"], count)
        if 'current_cosine' in values:
            writer.add_scalar(f'train_sites/{key}/current_cosine', values['current_cosine'], count)


def log_evaluation(writer, split, scores, count):
    if writer is None:
        return
    metrics = ["mean_cosine", "relative_squared_error"]
    if all("kl_divergence" in value for value in scores.values()):
        metrics.append("kl_divergence")
    for metric in ['current_mean_cosine', 'current_gate_fraction', 'weighted_cosine_loss']:
        if all(metric in value for value in scores.values()):
            metrics.append(metric)
    for metric in metrics:
        writer.add_scalar(f"{split}/{metric}", sum(v[metric] for v in scores.values()) / len(scores), count)
        for key, values in scores.items():
            writer.add_scalar(f"{split}_sites/{key}/{metric}", values[metric], count)
    writer.flush()


def make_maps(width, count):
    maps = torch.nn.ModuleList([torch.nn.Linear(width, width, bias=False) for _ in range(count)])
    with torch.no_grad():
        for layer in maps:
            layer.weight.copy_(torch.eye(width))
    return maps


def readout_loss(prediction, target, kind):
    prediction, target = prediction.float(), target.float()
    mse = (prediction-target).square().mean()
    cosine = torch.nn.functional.cosine_similarity(prediction, target, dim=-1, eps=1e-8).mean()
    if kind not in {"mse", "cosine"}:
        raise ValueError(f"unknown loss {kind}")
    return (mse if kind == "mse" else 1-cosine), mse, cosine


def gated_cosine_loss(prediction, target, previous_target, gate, next_weight=.9):
    next_cos = torch.nn.functional.cosine_similarity(prediction.float(), target.float(), dim=-1, eps=1e-8)
    current_cos = torch.nn.functional.cosine_similarity(prediction.float(), previous_target.float(), dim=-1, eps=1e-8)
    next_loss, current_loss = 1-next_cos, 1-current_cos
    # No batch-wise gate renormalization: each eligible token gets the 90/10
    # mixture, every other token gets 100% next loss.
    losses = torch.where(gate.bool(), next_weight*next_loss + (1-next_weight)*current_loss, next_loss)
    return losses.mean(), current_cos


def train_batch(maps, optimizer, keys, values, final, take, loss_kind="mse", readout=None, kl_chunk_size=128):
    if loss_kind == "kl":
        return train_kl_batch(maps, optimizer, keys, values, final, take, readout, kl_chunk_size)
    optimizer.zero_grad(set_to_none=True)
    y = values[final][:take].float().clone()
    total_error, energy, site_scores = 0., y.square().sum().item(), {}
    total_loss, total_cosine = 0., 0.
    mixed = loss_kind in ('s-cos-next-current-91', 's-cos-46')
    if mixed:
        previous = values['previous_target'][:take].float().clone()
        gate = values['current_gate'][:take].bool().clone()
    current_cos_sum = 0.
    for key, layer in zip(keys, maps):
        x = values[key][:take].float().clone()
        with torch.autocast(device_type=x.device.type, dtype=torch.bfloat16, enabled=x.is_cuda):
            prediction = layer(x)
        loss, mse_tensor, cosine_tensor = readout_loss(prediction, y, 'cosine' if mixed else loss_kind)
        if mixed:
            loss, current_cos = gated_cosine_loss(prediction, y, previous, gate, .4 if loss_kind == 's-cos-46' else .9)
        if not torch.isfinite(loss):
            raise ValueError(f"non-finite loss at {key}; checkpoint not advanced")
        loss.backward()
        mse, cosine, loss_value = mse_tensor.detach().item(), cosine_tensor.detach().item(), loss.detach().item()
        total_loss += loss_value
        total_cosine += cosine
        total_error += mse * y.numel()
        site_scores[key] = {"loss": loss_value, "cosine": cosine,
                            "mse": mse, "rel_mse": mse * y.numel() / max(energy, 1e-30)}
        if mixed:
            eligible_mean = current_cos[gate].mean().item() if gate.any() else 0.
            current_cos_sum += eligible_mean
            site_scores[key]['current_cosine'] = eligible_mean
    optimizer.step()
    result = {"mse": total_error / (len(keys) * y.numel()),
            "rel_mse": total_error / max(len(keys) * energy, 1e-30), "sites": site_scores,
            "loss": total_loss / len(keys), "cosine": total_cosine / len(keys)}
    if mixed:
        result.update(current_gate_fraction=gate.float().mean().item(), current_cosine=current_cos_sum/len(keys))
    return result


def save_state(path, maps, optimizer, identity, count, cursor, algorithm="adam_online_mse_v1"):
    tensors = {}
    for i, layer in enumerate(maps):
        tensors[f"weight.{i}"] = layer.weight.detach().cpu().contiguous()
        for name, value in optimizer.state[layer.weight].items():
            tensors[f"adam.{i}.{name}"] = value.detach().cpu().contiguous()
    temporary = path.with_suffix(".tmp")
    save_file(tensors, str(temporary), metadata={"recipe_hash": identity, "count": str(count),
                                               "cursor": str(cursor), "algorithm": algorithm})
    temporary.replace(path)


def restore_maps(path, maps, identity, algorithm="adam_online_mse_v1"):
    # Restore CPU weights first, then use GB10-aware module transfer.
    with safe_open(str(path), framework="pt", device="cpu") as stored:
        metadata = stored.metadata()
        if metadata.get("algorithm") != algorithm or metadata["recipe_hash"] != identity:
            raise ValueError("checkpoint algorithm/cache/recipe does not match")
        with torch.no_grad():
            for i, layer in enumerate(maps):
                layer.weight.copy_(stored.get_tensor(f"weight.{i}"))
        states = [{name: stored.get_tensor(f"adam.{i}.{name}")
                   for name in ["step", "exp_avg", "exp_avg_sq"]} for i in range(len(maps))]
    return int(metadata["count"]), int(metadata["cursor"]), states


def epoch_order(shards, seed, epoch):
    order = list(range(len(shards)))
    random.Random(seed + epoch).shuffle(order)
    return order


def crossed_milestones(previous, count):
    return [value for value in (1000, 10_000, 100_000, 1_000_000) if previous < value <= count]


def retain_snapshot(output, name, maps, keys, metadata, count, epoch, milestone=None, evaluation=None):
    """Retained v2 readout weights; latest optimizer checkpoint handles resume."""
    detail = dict(metadata, optimizer_positions=count,
                  snapshot={"epoch": epoch, "requested_positions": milestone,
                            "actual_positions": count, "weights_only": True})
    export(output / "checkpoints" / name,
                  {key: layer.weight.detach().cpu() for key, layer in zip(keys, maps)},
                  detail, evaluation or {})


@torch.no_grad()
def evaluate(maps, keys, directory, shards, loader=None, readout=None, kl_chunk_size=128, loss_kind='mse'):
    totals = {k: [0., 0., 0., 0] for k in keys}
    kl_totals = {k: 0. for k in keys}
    previous_totals = {k: [0., 0, 0.] for k in keys}
    has_previous = False
    for shard in shards:
        values = loader(directory, shard)
        y = values["target"].float()
        for key, layer in zip(keys, maps):
            # Evaluate the actual FP16-exported matrix, using FP32 arithmetic.
            prediction = values[key].float() @ layer.weight.detach().half().float().T
            item = totals[key]
            item[0] += torch.nn.functional.cosine_similarity(prediction, y).sum().item()
            item[1] += (prediction-y).square().sum().item()
            item[2] += y.square().sum().item()
            item[3] += len(y)
            if 'previous_target' in values:
                has_previous = True
                gate = values['current_gate'].bool()
                losses, previous_cos = gated_cosine_loss(prediction, y, values['previous_target'], gate, .4 if loss_kind == 's-cos-46' else .9)
                previous_totals[key][0] += previous_cos[gate].sum().item()
                previous_totals[key][1] += int(gate.sum())
                previous_totals[key][2] += losses.item()*len(y)
        if readout is not None:
            for start in range(0, len(y), kl_chunk_size):
                target = y[start:start+kl_chunk_size]
                teacher = readout(target)
                for key, layer in zip(keys, maps):
                    prediction = values[key][start:start+len(target)].float() @ layer.weight.detach().half().float().T
                    kl_totals[key] += distillation_kl(readout(prediction), teacher).item() * len(target)
    result = {key: {"mean_cosine": c/n, "relative_squared_error": e/max(t, 1e-30), "positions": n}
              for key, (c, e, t, n) in totals.items()}
    if readout is not None:
        for key in keys:
            result[key]["kl_divergence"] = kl_totals[key] / result[key]["positions"]
    if has_previous:
        for key, (cos_sum, eligible, loss_sum) in previous_totals.items():
            result[key].update(current_mean_cosine=cos_sum/eligible if eligible else 0.,
                               current_gate_fraction=eligible/result[key]['positions'],
                               weighted_cosine_loss=loss_sum/result[key]['positions'])
    return result


def main(argv=None):
    args = arguments(argv)
    default_subdir = 's'
    anticipated_output = args.output
    if anticipated_output is None:
        anticipated_output = ROOT / "build/lenses/v2" / model_path(args.model) / default_subdir
    if anticipated_output and anticipated_output.exists() and any(anticipated_output.iterdir()) and not args.resume:
        raise ValueError(f"{anticipated_output} is not empty; choose --output or --resume")
    print("Source: online frozen forwards (no activation-cache I/O)", flush=True)
    source = OnlineSource(args)
    manifest, loader = source.manifest, source.load
    readout = None
    if args.loss == "kl":
        _, _, norm, _ = model_parts(source.model)
        readout = FrozenReadout(norm, source.model.get_output_embeddings())
    keys, width = manifest["keys"], manifest["hidden_size"]
    model_id = manifest["model_id"]
    output = args.output or ROOT / "build/lenses/v2" / model_path(model_id) / default_subdir
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise ValueError(f"{output} is not empty; choose --output or --resume")
    # Preserve existing online recipe hashes and resume compatibility.
    shared_options = ["batch_size", "epochs", "learning_rate", "seed", "checkpoint_every"]
    recipe = {k: getattr(args, k) for k in shared_options}
    recipe.update(source="online", capture_recipe=manifest["capture_recipe"], algorithm="adam_online_mse_v1")
    if args.loss != "mse":
        recipe.update(loss=args.loss, algorithm=f"adam_online_{args.loss}_v1")
    if args.loss == "kl":
        recipe.update(kl_chunk_size=args.kl_chunk_size, kl_temperature=1., teacher="final distribution",
                      readout="frozen model final norm and output head")
    if args.loss in ('s-cos-next-current-91', 's-cos-46'):
        next_weight = .4 if args.loss == 's-cos-46' else .9
        recipe.update(next_weight=next_weight, current_weight=round(1-next_weight, 1), current_target='same-document final residual at t-1',
                      current_gate='argmax(frozen_head(final_norm(final[t-1]))) == input_ids[t]',
                      ungated_weighting='100% next loss', current_gate_chunk_size=128)
    objective = {"mse": "unweighted MSE", "cosine": "mean(1-cosine(mapped source, final)); eps=1e-8",
                 "kl": "mean KL(final distribution || mapped distribution); full vocabulary; no translator bias",
                 's-cos-next-current-91': 'gated .9*(1-cos(mapped[t],final[t])) + .1*(1-cos(mapped[t],final[t-1])); otherwise next only',
                 's-cos-46': 'gated .4*(1-cos(mapped[t],final[t])) + .6*(1-cos(mapped[t],final[t-1])); otherwise next only'}[args.loss]
    method_name = f"supervised_adam_{args.loss}"
    identity = hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()
    torch.manual_seed(args.seed)
    maps = make_maps(width, len(keys))
    state_path = output / "optimizer-state.safetensors"
    count, completed_shards, states = 0, 0, None
    if args.resume:
        count, completed_shards, states = restore_maps(state_path, maps, identity, recipe["algorithm"])
    output.mkdir(parents=True, exist_ok=True)
    writer = create_writer(args, output, recipe, count)
    maps = move_model_to_cuda(maps, touch="auto")
    optimizer = torch.optim.Adam(maps.parameters(), lr=args.learning_rate, foreach=False)
    if states is not None:
        for layer, state in zip(maps, states):
            optimizer.state[layer.weight] = {k: v if k == "step" else v.to(layer.weight.device) for k, v in state.items()}
    output.mkdir(parents=True, exist_ok=True)
    shards = manifest["splits"]["train"]
    positions_per_epoch = sum(s["positions"] for s in shards)
    snapshot_metadata = {k: manifest[k] for k in ["model_id", "model_revision", "tokenizer_revision", "hidden_size", "target"]}
    snapshot_metadata.update(method_name=method_name, dataset=PILE_DATASET_ID,
                             training_positions=positions_per_epoch,
                             fitting_recipe=dict(recipe, capture_recipe=manifest["capture_recipe"],
                                                 bias=False, initialization="identity", loss=objective),
                             sites={"embed": "embed" in keys,
                                    "mid": [int(k.split(".")[1]) for k in keys if k.startswith("mid.")],
                                    "post": [int(k.split(".")[1]) for k in keys if k.startswith("post.")]})
    progress = tqdm(total=positions_per_epoch * args.epochs, initial=count, unit="positions")
    last_checkpoint_count = count
    last_batch_time = time.perf_counter()
    for epoch in range(completed_shards // len(shards), args.epochs):
        for offset, shard_index in enumerate(epoch_order(shards, args.seed, epoch)):
            if epoch * len(shards) + offset < completed_shards:
                continue
            shard = shards[shard_index]
            values = loader(None, shard)
            device = maps[0].weight.device
            generator = torch.Generator(device=device).manual_seed(args.seed + epoch * len(shards) + shard_index)
            permutation = torch.randperm(shard["positions"], generator=generator, device=device)
            for start in range(0, shard["positions"], args.batch_size):
                indices = permutation[start:start + args.batch_size]
                batch = {k: v[indices] for k, v in values.items()}
                scores = train_batch(maps, optimizer, keys, batch, "target", len(indices), args.loss,
                                     readout=readout, kl_chunk_size=args.kl_chunk_size)
                previous, count = count, count + len(indices)
                now = time.perf_counter()
                log_training(writer, scores, count, epoch+1,
                             len(indices) / max(now-last_batch_time, 1e-9), args.learning_rate)
                last_batch_time = now
                progress.update(len(indices))
                postfix = dict(epoch=epoch+1, loss=f'{scores["loss"]:.4f}', cos=f'{scores["cosine"]:.4f}', rel_mse=f'{scores["rel_mse"]:.4f}')
                progress.set_postfix(**postfix)
                with (output / "training-log.jsonl").open("a") as stream:
                    stream.write(json.dumps({"positions": count, "epoch": epoch+1, **scores}) + "\n")
                for milestone in ([] if args.no_snapshots else crossed_milestones(previous, count)):
                    retain_snapshot(output, f"positions-{milestone:010d}", maps, keys,
                                    snapshot_metadata, count, epoch+1, milestone=milestone)
            completed_shards = epoch * len(shards) + offset + 1
            # Resume regenerates deterministic permutations and replays any
            # shards processed since the last atomic checkpoint.
            if count - last_checkpoint_count >= args.checkpoint_every or offset == len(shards)-1:
                save_state(state_path, maps, optimizer, identity, count, completed_shards, recipe["algorithm"])
                if writer is not None:
                    writer.flush()
                last_checkpoint_count = count
        val_scores = evaluate(maps, keys, None, manifest["splits"]["validation"], loader, readout, args.kl_chunk_size, args.loss)
        log_evaluation(writer, "validation", val_scores, count)
        if not args.no_snapshots:
            retain_snapshot(output, f"epoch-{epoch+1:03d}", maps, keys, snapshot_metadata,
                            count, epoch+1, evaluation={k: {"validation": v} for k, v in val_scores.items()})
        with (output / "training-log.jsonl").open("a") as stream:
            stream.write(json.dumps({"positions": count, "epoch": epoch+1, "validation": val_scores}) + "\n")
    progress.close()
    val_scores = evaluate(maps, keys, None, manifest["splits"]["validation"], loader, readout, args.kl_chunk_size, args.loss)
    test_scores = evaluate(maps, keys, None, manifest["splits"]["test"], loader, readout, args.kl_chunk_size, args.loss)
    log_evaluation(writer, "validation", val_scores, count)
    log_evaluation(writer, "test", test_scores, count)
    metadata = {k: manifest[k] for k in ["model_id", "model_revision", "tokenizer_revision", "hidden_size", "target"]}
    metadata.update(method_name=method_name, dataset=PILE_DATASET_ID,
                    training_positions=positions_per_epoch, optimizer_positions=count,
                    fitting_recipe=dict(recipe, capture_recipe=manifest["capture_recipe"],
                                        bias=False, initialization="identity", loss=objective),
                    sites={"embed": "embed" in keys,
                           "mid": [int(k.split(".")[1]) for k in keys if k.startswith("mid.")],
                           "post": [int(k.split(".")[1]) for k in keys if k.startswith("post.")]})
    report = {k: {"validation": val_scores[k], "test": test_scores[k]} for k in keys}
    export(output, {k: m.weight.detach().cpu() for k, m in zip(keys, maps)}, metadata, report)
    if writer is not None:
        writer.close()
    print(f"Saved {output / 'lens.safetensors'} and manifest.json", flush=True)


if __name__ == "__main__":
    main()
