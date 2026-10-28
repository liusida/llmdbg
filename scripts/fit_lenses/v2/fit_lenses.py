#!/usr/bin/env python3
"""Fit site-aware format-v2 J/R lenses with resumable accumulation."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import statistics
import sys
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator

import torch
from gb10_load_llm import load_model_to_cuda
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
PILE_DATASET_ID = "NeelNanda/pile-10k"
WIKITEXT_DATASET_ID = "Salesforce/wikitext"
WIKITEXT_CONFIG = "wikitext-103-raw-v1"
SUPPORTED_MODELS = (
    "openai-community/gpt2",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3.5-0.8B",
    "Qwen/Qwen3.5-2B-Base",
    "Qwen/Qwen3.5-2B",
    "Qwen/Qwen3.5-9B",
    "Qwen/Qwen3.5-27B",
)
DEFAULT_PROMPTS = {
    "openai-community/gpt2": 100,
    "Qwen/Qwen3-8B": 25,
    "Qwen/Qwen3.5-0.8B": 100,
    "Qwen/Qwen3.5-2B-Base": 100,
    "Qwen/Qwen3.5-2B": 100,
    "Qwen/Qwen3.5-9B": 25,
    "Qwen/Qwen3.5-27B": 25,
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=SUPPORTED_MODELS, required=True)
    parser.add_argument("--method", choices=("j", "r"), required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--prompts", type=int,
        help="Jacobian calibration sequences (model-specific default: 25 or 100)",
    )
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--skip-first", type=int, default=4)
    parser.add_argument(
        "--dataset",
        choices=("pile-10k", "wikitext-103"),
        default="pile-10k",
        help=(
            "calibration corpus; wikitext-103 reproduces Neuronpedia's "
            "2,000-character concatenation/chunking"
        ),
    )
    parser.add_argument("--dim-batch", type=int, default=16)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--calibration-rows", type=int, default=64)
    parser.add_argument("--calibration-repeats", type=int, default=2)
    args = parser.parse_args()
    if args.prompts is None:
        args.prompts = DEFAULT_PROMPTS[args.model]
    return args


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


def calibration_prompts(count: int, dataset_name: str) -> list[str]:
    from datasets import load_dataset

    if dataset_name == "pile-10k":
        dataset = load_dataset(PILE_DATASET_ID, split="train", streaming=True)
        texts = (str(row["text"]) for row in dataset if str(row["text"]).strip())
        prompts = list(itertools.islice(texts, count))
    elif dataset_name == "wikitext-103":
        dataset = load_dataset(
            WIKITEXT_DATASET_ID,
            WIKITEXT_CONFIG,
            split="train",
            streaming=True,
        )
        prompts = []
        buffer = ""
        for row in dataset:
            text = str(row.get("text", "")).strip()
            if not text or text.startswith("="):
                continue
            buffer += " " + text
            while len(buffer) > 2000:
                prompts.append(buffer[:2000].strip())
                buffer = buffer[2000:]
                if len(prompts) >= count:
                    break
            if len(prompts) >= count:
                break
    else:
        raise ValueError(f"unsupported dataset {dataset_name!r}")
    if len(prompts) != count:
        raise RuntimeError(f"dataset supplied {len(prompts)} of {count} prompts")
    return prompts


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


def install_relp(model: Any, architecture: str) -> dict[str, Any]:
    if architecture == "gpt2":
        from scripts.fit_lenses.v1.fit_gpt2_r_lens import install_relp_rules

        norms, mlps = install_relp_rules(model)
        return {"ln_rule": True, "identity_rule": True, "half_rule": False,
                "norm_modules": norms, "mlp_modules": mlps}
    from scripts.fit_lenses.v1.fit_qwen35_r_lens import install_relp_rules

    norms, mlps = install_relp_rules(model)
    return {"ln_rule": True, "identity_rule": True, "half_rule": True,
            "norm_modules": norms, "mlp_modules": mlps}


@contextmanager
def record_sites(embed: Any, layers: Any, architecture: str) -> Iterator[dict[str, Any]]:
    """Capture embed, post-attention Mid, and full-block Post tensors."""
    values: dict[str, Any] = {}
    handles = []

    def capture(key: str, *, graph_root: bool = False):
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            tensor = output if torch.is_tensor(output) else output[0]
            if graph_root:
                tensor.requires_grad_(True)
            values[key] = tensor
        return hook

    def capture_input(key: str):
        def hook(_module: Any, inputs: Any) -> None:
            values[key] = inputs[0]
        return hook

    handles.append(embed.register_forward_hook(capture("embed", graph_root=True)))
    for index, block in enumerate(layers):
        mid_module = block.ln_2 if architecture == "gpt2" else block.post_attention_layernorm
        handles.append(mid_module.register_forward_pre_hook(capture_input(f"mid.{index}")))
        handles.append(block.register_forward_hook(capture(f"post.{index}")))
    try:
        yield values
    finally:
        for handle in handles:
            handle.remove()


def prompt_jacobians(
    model: Any,
    tokenizer: Any,
    *,
    prompt: str,
    dim_batch: int,
    max_seq_len: int,
    skip_first: int,
    calibration_rows: int | None = None,
) -> tuple[dict[str, torch.Tensor], int, int]:
    embed, layers, _norm, architecture = model_parts(model)
    encoded = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=max_seq_len
    )["input_ids"].to(next(model.parameters()).device)
    sequence_length = int(encoded.shape[1])
    positions = torch.arange(sequence_length)
    positions = positions[(positions >= skip_first) & (positions < sequence_length - 1)]
    if positions.numel() == 0:
        raise ValueError("prompt has no valid positions after filtering")
    d_model = hidden_size(model)
    row_count = min(d_model, calibration_rows or d_model)
    jacobians: dict[str, torch.Tensor] = {}
    with record_sites(embed, layers, architecture) as captured, torch.enable_grad():
        model(encoded.expand(dim_batch, -1), use_cache=False)
        target = captured[f"post.{len(layers) - 1}"]
        sources = {
            key: value
            for key, value in captured.items()
            if key != f"post.{len(layers) - 1}"
        }
        jacobians = {
            key: torch.zeros(row_count, d_model, dtype=torch.float32)
            for key in sources
        }
        device_positions = positions.to(target.device)
        batches = torch.arange(dim_batch, device=target.device)
        cotangent = torch.zeros_like(target)
        for start in range(0, row_count, dim_batch):
            count = min(dim_batch, row_count - start)
            cotangent.zero_()
            cotangent[
                batches[:count, None], device_positions[None, :],
                start + batches[:count, None],
            ] = 1
            gradients = torch.autograd.grad(
                target,
                tuple(sources.values()),
                grad_outputs=cotangent,
                retain_graph=start + count < row_count,
            )
            for (key, _source), gradient in zip(sources.items(), gradients, strict=True):
                source_positions = positions.to(gradient.device)
                jacobians[key][start : start + count] = (
                    gradient[:count, source_positions, :].float().mean(1).cpu()
                )
    return jacobians, sequence_length, int(positions.numel())


def save_checkpoint(path: Path, sums: dict[str, torch.Tensor], completed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({"sums": sums, "completed": completed}, temporary)
    temporary.replace(path)


def main() -> int:
    args = arguments()
    output = args.output or (
        PROJECT_ROOT / "build" / "lenses" / "v2" / model_path(args.model) / args.method
    )
    checkpoint = args.checkpoint or output / "fit.checkpoint.pt"
    output.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    is_gpt2 = args.model == "openai-community/gpt2"
    model = load_model_to_cuda(
        AutoModelForCausalLM,
        args.model,
        device="cuda",
        dtype=torch.float32 if is_gpt2 else torch.bfloat16,
        touch="auto",
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )
    model.eval().requires_grad_(False)
    _embed, layers, _norm, architecture = model_parts(model)
    rules = {"ln_rule": False, "identity_rule": False, "half_rule": False}
    if args.method == "r":
        parity_ids = tokenizer("Workspace Lens parity.", return_tensors="pt")["input_ids"].cuda()
        with torch.inference_mode():
            before = model(parity_ids).logits
        rules = install_relp(model, architecture)
        with torch.inference_mode():
            after = model(parity_ids).logits
        if not torch.allclose(before, after, atol=1e-5, rtol=1e-4):
            raise RuntimeError(f"RelP changed forward values by {(before-after).abs().max().item():.3e}")

    requested_prompts = 1 if args.calibrate else args.prompts
    prompts = calibration_prompts(requested_prompts, args.dataset)
    if args.calibrate:
        # Warm model kernels and the CUDA context before timing. Without this,
        # the smaller first shard can appear slower than the larger shard.
        prompt_jacobians(
            model, tokenizer, prompt=prompts[0], dim_batch=args.dim_batch,
            max_seq_len=args.max_seq_len, skip_first=args.skip_first,
            calibration_rows=args.dim_batch,
        )
        samples: list[dict[str, float | int]] = []
        current: dict[str, torch.Tensor] = {}
        for rows in sorted({max(args.dim_batch, args.calibration_rows // 2), args.calibration_rows}):
            timings = []
            peak_allocated = []
            peak_reserved = []
            free_before = []
            for _ in range(args.calibration_repeats):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    free_before.append(int(torch.cuda.mem_get_info()[0]))
                sample_started = time.perf_counter()
                current, _token_count, _position_count = prompt_jacobians(
                    model, tokenizer, prompt=prompts[0], dim_batch=args.dim_batch,
                    max_seq_len=args.max_seq_len, skip_first=args.skip_first,
                    calibration_rows=rows,
                )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    peak_allocated.append(int(torch.cuda.max_memory_allocated()))
                    peak_reserved.append(int(torch.cuda.max_memory_reserved()))
                timings.append(time.perf_counter() - sample_started)
            samples.append({
                "rows": rows,
                "seconds": statistics.median(timings),
                "minimum_seconds": min(timings),
                "maximum_seconds": max(timings),
                "peak_cuda_allocated_bytes": max(peak_allocated, default=0),
                "peak_cuda_reserved_bytes": max(peak_reserved, default=0),
                "minimum_cuda_free_before_bytes": min(free_before, default=0),
            })
        first, last = samples[0], samples[-1]
        row_delta = int(last["rows"]) - int(first["rows"])
        measured_slope = (
            (float(last["seconds"]) - float(first["seconds"])) / row_delta
            if row_delta else float(last["seconds"]) / int(last["rows"])
        )
        # A non-positive slope is timing noise. Fall back to the conservative
        # zero-fixed-cost estimate instead of emitting a negative duration.
        seconds_per_row = (
            measured_slope
            if measured_slope > 0
            else float(last["seconds"]) / int(last["rows"])
        )
        fixed_seconds = max(
            0.0, float(last["seconds"]) - seconds_per_row * int(last["rows"])
        )
        full_rows = hidden_size(model)
        projected_per_prompt = fixed_seconds + seconds_per_row * full_rows
        projected = projected_per_prompt * args.prompts
        estimate = {
            "model_id": args.model, "method": args.method,
            "samples": samples,
            "fixed_seconds_per_prompt": fixed_seconds,
            "seconds_per_row": seconds_per_row,
            "projected_prompts": args.prompts,
            "projected_rows": full_rows,
            "dimension_batch": args.dim_batch,
            "projected_seconds": projected,
            "projected_hours": projected / 3600,
            "checkpoint": str(checkpoint),
            "estimated_checkpoint_bytes": len(current) * full_rows * full_rows * 4,
        }
        (output / "calibration.json").write_text(json.dumps(estimate, indent=2) + "\n")
        print(json.dumps(estimate, indent=2))
        return 0
    sums: dict[str, torch.Tensor] = {}
    completed = 0
    if checkpoint.is_file() and not args.calibrate:
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        sums, completed = saved["sums"], int(saved["completed"])
    started = time.perf_counter()
    valid_positions = 0
    sequence_tokens = 0
    for prompt_index, prompt in enumerate(prompts[completed:], start=completed):
        current, token_count, position_count = prompt_jacobians(
            model, tokenizer, prompt=prompt, dim_batch=args.dim_batch,
            max_seq_len=args.max_seq_len, skip_first=args.skip_first,
            calibration_rows=None,
        )
        for key, tensor in current.items():
            sums[key] = sums.get(key, torch.zeros_like(tensor)) + tensor
        completed = prompt_index + 1
        sequence_tokens += token_count
        valid_positions += position_count
        if not args.calibrate and (
            completed % args.checkpoint_every == 0 or completed == requested_prompts
        ):
            save_checkpoint(checkpoint, sums, completed)
        print(f"completed {completed}/{requested_prompts}", flush=True)

    elapsed = time.perf_counter() - started
    averaged = {key: (value / completed).half().contiguous() for key, value in sums.items()}
    tensor_path = output / "lens.safetensors"
    save_file(averaged, tensor_path)
    manifest = {
        "format_version": 2,
        "artifact": tensor_path.name,
        "sha256": sha256(tensor_path),
        "model_id": args.model,
        "model_revision": resolved_revision(model.config),
        "tokenizer_revision": resolved_revision(tokenizer),
        "method": args.method,
        "method_name": "ordinary_jacobian" if args.method == "j" else "relp",
        "provenance": "llmdbg_refit",
        "target": {"site": "post", "layer": len(layers) - 1},
        "sites": {"embed": True, "mid": list(range(len(layers))),
                  "post": list(range(len(layers) - 1))},
        "hidden_size": hidden_size(model),
        "dtype": "float16",
        "artifact_bytes": tensor_path.stat().st_size,
        "tensor_shapes": {key: list(value.shape) for key, value in averaged.items()},
        "dataset": (
            PILE_DATASET_ID
            if args.dataset == "pile-10k"
            else f"{WIKITEXT_DATASET_ID}/{WIKITEXT_CONFIG}"
        ),
        "dataset_recipe": args.dataset,
        "prompts": completed,
        "max_seq_len": args.max_seq_len,
        "skip_first": args.skip_first,
        "valid_positions_last_run": valid_positions,
        "sequence_tokens_last_run": sequence_tokens,
        "rules": rules,
        "fitting_recipe": {
            "prompts": completed,
            "max_seq_len": args.max_seq_len,
            "skip_first": args.skip_first,
            "dimension_batch": args.dim_batch,
            "model_compute_dtype": "float32" if is_gpt2 else "bfloat16",
            "accumulator_dtype": "float32",
        },
        "created_at": datetime.now(UTC).isoformat(),
        "elapsed_seconds_last_run": elapsed,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"saved {tensor_path} and manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
