#!/usr/bin/env python3
"""Fit J++ lenses for llmdbg's supported models and compare the 9B fit.

This is a reproducible, resumable driver around the reference J++ pipeline.  It
uses ``gb10-load-llm`` instead of Accelerate/device_map, fits every residual
source layer, grades each model to create its own correctness filter, and scores
the fitted lens on a held-out split.  For Qwen3.5-9B it also evaluates the
published lens on exactly the same examples and writes paired significance
statistics.

The reference implementation is pinned by default.  Install its package once:

  uv pip install datasets pandas pyarrow scikit-learn tqdm
  uv pip install --no-deps \
    "jpp-lens @ git+https://github.com/safety-research/jpp_lens@d7efc0de372e20589e899188c0cfe8434acc63ec"

Examples:

  python scripts/fit_jpp_lenses.py --model qwen2b
  python scripts/fit_jpp_lenses.py --model all --num-shards 4
  python scripts/fit_jpp_lenses.py --model qwen9b --stage compare

Pass ``--dry-run`` to print every command without loading a model.  Existing
stage outputs are reused unless ``--force`` is given.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import MethodType
from typing import Any, Sequence

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_COMMIT = "d7efc0de372e20589e899188c0cfe8434acc63ec"
REFERENCE_REPO = "https://github.com/safety-research/jpp_lens.git"
OFFICIAL_9B_REPO = "koayon/jpp-lenses"
OFFICIAL_9B_REVISION = "237dab6ea0e611d964bf407f13ba1655f1a6ae43"
OFFICIAL_9B_FILE = "qwen3.5-9b/lens.pt"


@dataclass(frozen=True)
class ModelProfile:
    model_id: str
    slug: str
    gpt2_relp: bool = False


PROFILES = {
    "gpt2": ModelProfile("openai-community/gpt2", "gpt2", True),
    "qwen0.8b": ModelProfile("Qwen/Qwen3.5-0.8B", "qwen3.5-0.8b"),
    "qwen2b": ModelProfile("Qwen/Qwen3.5-2B", "qwen3.5-2b"),
    "qwen9b": ModelProfile("Qwen/Qwen3.5-9B", "qwen3.5-9b"),
    "qwen27b": ModelProfile("Qwen/Qwen3.5-27B", "qwen3.5-27b"),
}


class _IdentityActivation(torch.autograd.Function):
    """Keep GELU's forward value and use its secant slope in backward."""

    @staticmethod
    def forward(ctx: Any, values: torch.Tensor, activated: torch.Tensor) -> torch.Tensor:
        ratio = torch.where(values != 0, activated / values, torch.full_like(values, 0.5))
        ctx.save_for_backward(ratio)
        return activated

    @staticmethod
    def backward(ctx: Any, gradient: torch.Tensor) -> tuple[torch.Tensor, None]:
        (ratio,) = ctx.saved_tensors
        return gradient * ratio, None


def install_gpt2_relp_rules(model: torch.nn.Module) -> tuple[int, int]:
    """Install the value-preserving GPT-2 RelP rules used by llmdbg's R-Lens."""
    from transformers.models.gpt2.modeling_gpt2 import GPT2MLP

    norms = mlps = 0
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.LayerNorm) and (
            name.endswith(".ln_1") or name.endswith(".ln_2") or name == "transformer.ln_f"
        ):

            def relp_norm(current: torch.nn.LayerNorm, values: torch.Tensor) -> torch.Tensor:
                dims = tuple(range(values.ndim - len(current.normalized_shape), values.ndim))
                mean = values.float().mean(dims, keepdim=True).detach()
                variance = (values.float() - mean).pow(2).mean(dims, keepdim=True).detach()
                normalized = (values.float() - mean) * torch.rsqrt(variance + current.eps)
                if current.elementwise_affine:
                    normalized = normalized * current.weight.float() + current.bias.float()
                return normalized.type_as(values)

            module.forward = MethodType(relp_norm, module)
            norms += 1
        elif isinstance(module, GPT2MLP):

            def relp_mlp(current: GPT2MLP, values: torch.Tensor) -> torch.Tensor:
                expanded = current.c_fc(values)
                activated = _IdentityActivation.apply(expanded, current.act(expanded))
                return current.dropout(current.c_proj(activated))

            module.forward = MethodType(relp_mlp, module)
            mlps += 1
    return norms, mlps


def load_lens_model(model_id: str, *, gpt2_relp: bool) -> Any:
    """Load one model through GB10 and adapt it to the reference LensModel API."""
    from gb10_load_llm import load_model_to_cuda
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import jlens

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = load_model_to_cuda(
        AutoModelForCausalLM,
        model_id,
        device="cuda",
        # The GPT-2 RelP rules are parity-tested in fp32. Qwen fits follow the
        # reference recipe's bf16 model precision.
        dtype=torch.float32 if gpt2_relp else torch.bfloat16,
        touch="auto",
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )
    model.eval().requires_grad_(False)
    if gpt2_relp:
        input_ids = tokenizer("J++ forward parity.", return_tensors="pt")["input_ids"].cuda()
        with torch.inference_mode():
            ordinary = model(input_ids).logits
        norms, mlps = install_gpt2_relp_rules(model)
        expected_layers = int(model.config.num_hidden_layers)
        if (norms, mlps) != (2 * expected_layers + 1, expected_layers):
            raise RuntimeError(f"unexpected GPT-2 RelP modules: norms={norms}, mlps={mlps}")
        with torch.inference_mode():
            patched = model(input_ids).logits
        if not torch.allclose(ordinary, patched, atol=1e-5, rtol=1e-4):
            delta = float((ordinary - patched).abs().max())
            raise RuntimeError(f"GPT-2 RelP rules changed forward values (max delta {delta:.3e})")
    return jlens.from_hf(model, tokenizer)


def load_reference_cli(root: Path) -> Any:
    """Load the pinned reference CLI as a module, without copying its internals."""
    source_root = root / "src"
    cli_path = root / "scripts" / "jpp_cli.py"
    if not cli_path.is_file():
        raise SystemExit(
            f"J++ reference checkout not found at {root}. Run with --bootstrap-reference "
            "or pass --jpp-root."
        )
    sys.path.insert(0, str(source_root))
    spec = importlib.util.spec_from_file_location("llmdbg_reference_jpp_cli", cli_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {cli_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ensure_reference(root: Path, bootstrap: bool) -> None:
    if (root / "scripts" / "jpp_cli.py").is_file():
        return
    if not bootstrap:
        return
    root.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", REFERENCE_REPO, str(root)], check=True)
    subprocess.run(["git", "checkout", REFERENCE_COMMIT], cwd=root, check=True)


def reference_head(root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, text=True, capture_output=True
    )
    return result.stdout.strip()


def profile_for_model_id(model_id: str) -> ModelProfile:
    for profile in PROFILES.values():
        if profile.model_id == model_id:
            return profile
    raise ValueError(f"unsupported model {model_id!r}")


def run_reference_stage(argv: Sequence[str], reference_root: Path) -> int:
    """Internal subprocess entry point used to guarantee model memory is released."""
    if not argv:
        raise SystemExit("missing reference stage")
    cli = load_reference_cli(reference_root)
    parsed = cli.build_parser().parse_args(list(argv))
    if hasattr(parsed, "hf_model_name"):
        profile = profile_for_model_id(parsed.hf_model_name)

        def gb10_loader(_: argparse.Namespace) -> Any:
            return load_lens_model(profile.model_id, gpt2_relp=profile.gpt2_relp)

        cli._load_model = gb10_loader
    return cli.main(list(argv))


def model_layers(model_id: str) -> list[int]:
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model_id)
    text_config = getattr(config, "text_config", config)
    count = int(getattr(text_config, "num_hidden_layers", getattr(text_config, "n_layer", 0)))
    if count < 2:
        raise RuntimeError(f"cannot determine transformer layer count for {model_id}")
    return list(range(count - 1))


def fit_router_in_layer_batches(args: argparse.Namespace) -> int:
    """Fit identical per-layer routers without retaining every layer's pool at once."""
    cli = load_reference_cli(args.jpp_root)
    from workspace_lens.routing.cluster import collect_valid_position_activations
    from workspace_lens.routing.router import ActivationRouterCollection

    profile = profile_for_model_id(args.model)
    model = load_lens_model(profile.model_id, gpt2_relp=False)
    prompts = cli.load_fit_prompts(args.num_prompts)
    if args.out.exists():
        router = ActivationRouterCollection.load(str(args.out))
        if (router.num_clusters, router.projection_dim, router.seed) != (
            args.num_clusters,
            args.projection_dim,
            args.seed,
        ):
            raise RuntimeError(f"partial router settings do not match requested settings: {args.out}")
    else:
        router = ActivationRouterCollection(
            args.num_clusters, args.projection_dim, seed=args.seed
        )
    completed = set(router.layer_routers_L_dict)
    remaining = [layer for layer in args.layers if layer not in completed]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(remaining), args.layer_batch_size):
        layers = remaining[start : start + args.layer_batch_size]
        print(f"router layer batch {layers}; already complete={sorted(completed)}", flush=True)
        activations = collect_valid_position_activations(
            model,
            prompts,
            layers,
            skip_first_n_positions=args.skip_first_n_positions,
            max_seq_len=args.max_seq_len,
        )
        partial = ActivationRouterCollection(
            args.num_clusters, args.projection_dim, seed=args.seed
        )
        partial.fit(activations)
        router.layer_routers_L_dict.update(partial.layer_routers_L_dict)
        router.save(str(args.out))
        completed.update(layers)
        del activations, partial
        torch.cuda.empty_cache()
    return 0


def run_command(command: list[str], *, dry_run: bool) -> None:
    print("+", " ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, check=True)


def newest_checkpoint_dir(base: Path, checkpoint_name: str, shard: int, shards: int) -> Path:
    filename = "experts_K8_checkpoint.pt"
    matches = sorted(
        base.glob(f"*/{checkpoint_name}/shard{shard}of{shards}/{filename}"),
        key=lambda path: path.stat().st_mtime,
    )
    if not matches:
        raise FileNotFoundError(f"no {filename} below {base} for shard {shard}/{shards}")
    return matches[-1].parent


def exact_mcnemar_p(discordant_a: int, discordant_b: int) -> float:
    """Two-sided exact McNemar/binomial p-value without scipy."""
    n = discordant_a + discordant_b
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(discordant_a, discordant_b) + 1)) / 2**n
    return min(1.0, 2.0 * tail)


def write_paired_comparison(ranks_path: Path, output: Path) -> None:
    import pandas as pd

    frame = pd.read_csv(ranks_path)
    names = list(dict.fromkeys(frame["lens"].tolist()))
    if len(names) != 2:
        raise ValueError(f"expected exactly two lenses, got {names}")
    keys = ["eval", "item", "intermediate", "layer"]
    wide = frame.pivot(index=keys, columns="lens", values="rank").dropna()
    rows = []
    for k in (1, 10):
        local = wide[names[0]] <= k
        official = wide[names[1]] <= k
        local_only = int((local & ~official).sum())
        official_only = int((~local & official).sum())
        rows.append(
            {
                "k": k,
                "local_lens": names[0],
                "official_lens": names[1],
                "n_paired_readouts": len(wide),
                "local_recall": float(local.mean()),
                "official_recall": float(official.mean()),
                "difference": float(local.mean() - official.mean()),
                "local_only_hits": local_only,
                "official_only_hits": official_only,
                "mcnemar_exact_p": exact_mcnemar_p(local_only, official_only),
            }
        )
    pd.DataFrame(rows).to_csv(output, index=False)


def write_manifest(path: Path, profile: ModelProfile, args: argparse.Namespace, layers: list[int]) -> None:
    payload = {
        "created_at": datetime.now(UTC).isoformat(),
        "model_id": profile.model_id,
        "profile": profile.slug,
        "source_layers": layers,
        "target_layer": max(layers) + 1,
        "reference_repo": REFERENCE_REPO,
        "reference_commit": REFERENCE_COMMIT,
        "router": {"clusters": args.num_clusters, "projection_dim": args.projection_dim},
        "fit": {
            "router_prompts": args.router_prompts,
            "expert_prompts": args.expert_prompts,
            "fit_fraction": args.fit_fraction,
            "split_seed": args.seed,
        },
        "lrp": (
            {
                "mode": "external-gpt2-relp",
                "rules": ["LayerNorm detached statistics", "GELU secant backward"],
                "note": "J++-style fit; GPT-2 rules are llmdbg additions, not upstream J++ rules.",
            }
            if profile.gpt2_relp
            else {"mode": "upstream-rlens"}
        ),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


def orchestrate(profile: ModelProfile, args: argparse.Namespace) -> None:
    root = args.output_root / profile.slug
    root.mkdir(parents=True, exist_ok=True)
    layers = model_layers(profile.model_id)
    layers_arg = ",".join(map(str, layers))
    manifest = root / "fit-manifest.json"
    write_manifest(manifest, profile, args, layers)

    wrapper = [sys.executable, str(Path(__file__).resolve()), "_reference", "--jpp-root", str(args.jpp_root)]
    common = ["--hf-model-name", profile.model_id, "--layers", layers_arg]
    router = root / "router.pt"
    experts = root / "experts.pt"
    lens = root / "lens.pt"
    correctness = root / "model_correctness.csv"
    ranks = root / "held_out_ranks.csv"
    checkpoint_name = f"{profile.slug}-jpp"
    checkpoints_base = root / "checkpoints"

    def stage_enabled(name: str) -> bool:
        return args.stage in ("all", name) or (args.stage == "compare" and name == "compare")

    if stage_enabled("router") and (args.force or not router.exists()):
        run_command(
            [sys.executable, str(Path(__file__).resolve()), "_router",
                "--jpp-root", str(args.jpp_root), "--model", profile.model_id,
                "--layers", layers_arg]
            + [
                "--num-prompts", str(args.router_prompts),
                "--num-clusters", str(args.num_clusters),
                "--projection-dim", str(args.projection_dim),
                "--seed", str(args.seed),
                "--layer-batch-size", str(args.router_layer_batch),
                "--out", str(router),
            ],
            dry_run=args.dry_run,
        )

    checkpoint_dirs: list[Path] = []
    if stage_enabled("experts") and (args.force or not experts.exists()):
        for shard in range(args.num_shards):
            command = (
                wrapper
                + ["fit-shard"]
                + common
                + [
                    "--router-path", str(router),
                    "--shard-idx", str(shard),
                    "--num-shards", str(args.num_shards),
                    "--num-prompts", str(args.expert_prompts),
                    "--lrp-mode", "none" if profile.gpt2_relp else "rlens",
                    "--artifacts-base-dir", str(checkpoints_base),
                    "--checkpoint-name", checkpoint_name,
                ]
            )
            # The reference trainer resumes its own checkpoint, so rerunning is safe.
            run_command(command, dry_run=args.dry_run)
            checkpoint_dirs.append(
                checkpoints_base / "<date>" / checkpoint_name / f"shard{shard}of{args.num_shards}"
                if args.dry_run
                else newest_checkpoint_dir(
                    checkpoints_base, checkpoint_name, shard, args.num_shards
                )
            )
        run_command(
            wrapper
            + [
                "merge-experts",
                "--checkpoint-dirs", *map(str, checkpoint_dirs),
                "--num-clusters", str(args.num_clusters),
                "--checkpoint-name", checkpoint_name,
                "--out", str(experts),
            ],
            dry_run=args.dry_run,
        )

    if stage_enabled("weights"):
        if args.force or not correctness.exists():
            run_command(
                [sys.executable, str(Path(__file__).resolve()), "_grade", "--jpp-root", str(args.jpp_root),
                 "--model", profile.model_id, "--layers", layers_arg, "--out", str(correctness)],
                dry_run=args.dry_run,
            )
        if args.force or not lens.exists():
            run_command(
                wrapper
                + ["fit-weights"]
                + common
                + [
                    "--experts", str(experts),
                    "--correctness-csv", str(correctness),
                    "--eval-data-dir", str(args.jpp_root / "data/jlens/evaluations"),
                    "--items", f"fit:{args.seed}",
                    "--fit-fraction", str(args.fit_fraction),
                    "--cache-path", str(root / "readout_residuals.pt"),
                    "--checkpoint-name", checkpoint_name,
                    "--out", str(lens),
                ],
                dry_run=args.dry_run,
            )

    if stage_enabled("evaluate") and (args.force or not ranks.exists()):
        run_command(
            wrapper
            + ["evaluate"]
            + common
            + [
                "--lens", str(lens),
                "--correctness-csv", str(correctness),
                "--eval-data-dir", str(args.jpp_root / "data/jlens/evaluations"),
                "--items", f"held-out:{args.seed}",
                "--fit-fraction", str(args.fit_fraction),
                "--out", str(ranks),
            ],
            dry_run=args.dry_run,
        )

    if stage_enabled("compare") and profile.slug == "qwen3.5-9b":
        official = root / "official-9b.pt"
        local_named = root / "local-9b.pt"
        if not args.dry_run:
            from huggingface_hub import hf_hub_download

            official_cache = Path(
                hf_hub_download(
                    OFFICIAL_9B_REPO,
                    OFFICIAL_9B_FILE,
                    revision=OFFICIAL_9B_REVISION,
                )
            )
            if not official.exists():
                try:
                    official.symlink_to(official_cache)
                except OSError:
                    shutil.copy2(official_cache, official)
            if not local_named.exists():
                try:
                    local_named.symlink_to(lens.name)
                except OSError:
                    shutil.copy2(lens, local_named)
        comparison_ranks = root / "official_comparison.csv"
        run_command(
            wrapper
            + ["evaluate"]
            + common
            + [
                "--lens", str(local_named), str(official),
                "--correctness-csv", str(correctness),
                "--eval-data-dir", str(args.jpp_root / "data/jlens/evaluations"),
                "--items", f"held-out:{args.seed}",
                "--fit-fraction", str(args.fit_fraction),
                "--out", str(comparison_ranks),
            ],
            dry_run=args.dry_run,
        )
        if not args.dry_run:
            write_paired_comparison(comparison_ranks, root / "official_comparison_paired.csv")


def grade_correctness(args: argparse.Namespace) -> int:
    cli = load_reference_cli(args.jpp_root)
    from lens_evals.readout_evals.readout_eval_items import MACRO_EVALS

    profile = profile_for_model_id(args.model)
    model = load_lens_model(profile.model_id, gpt2_relp=False)
    items = cli.load_readout_eval_items(
        str(args.jpp_root / "data/jlens/evaluations"), slugs=MACRO_EVALS
    )
    runner = cli.readout_runner(model, layers=args.layers, max_seq_len=cli.RECIPE_READOUT_MAX_SEQ_LEN)
    frame = runner.grade_model_correctness(items, hf_model_name=profile.model_id)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=[*PROFILES, "all"], default="all")
    parser.add_argument("--stage", choices=("all", "router", "experts", "weights", "evaluate", "compare"), default="all")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "build/jpp_lens")
    parser.add_argument("--jpp-root", type=Path, default=PROJECT_ROOT / "build/vendor/jpp_lens")
    parser.add_argument("--bootstrap-reference", action="store_true")
    parser.add_argument("--router-prompts", type=int, default=1000)
    parser.add_argument("--expert-prompts", type=int, default=64)
    parser.add_argument("--num-clusters", type=int, default=8)
    parser.add_argument("--projection-dim", type=int, default=64)
    parser.add_argument("--router-layer-batch", type=int, default=2)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--fit-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def internal_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    subparsers = parser.add_subparsers(dest="internal", required=True)
    ref = subparsers.add_parser("_reference")
    ref.add_argument("--jpp-root", type=Path, required=True)
    ref.add_argument("reference_args", nargs=argparse.REMAINDER)
    grade = subparsers.add_parser("_grade")
    grade.add_argument("--jpp-root", type=Path, required=True)
    grade.add_argument("--model", required=True)
    grade.add_argument("--layers", type=lambda value: [int(x) for x in value.split(",")], required=True)
    grade.add_argument("--out", type=Path, required=True)
    router = subparsers.add_parser("_router")
    router.add_argument("--jpp-root", type=Path, required=True)
    router.add_argument("--model", required=True)
    router.add_argument("--layers", type=lambda value: [int(x) for x in value.split(",")], required=True)
    router.add_argument("--num-prompts", type=int, default=1000)
    router.add_argument("--num-clusters", type=int, default=8)
    router.add_argument("--projection-dim", type=int, default=64)
    router.add_argument("--seed", type=int, default=0)
    router.add_argument("--layer-batch-size", type=int, default=2)
    router.add_argument("--skip-first-n-positions", type=int, default=16)
    router.add_argument("--max-seq-len", type=int, default=128)
    router.add_argument("--out", type=Path, required=True)
    return parser


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1].startswith("_"):
        args = internal_parser().parse_args()
        if args.internal == "_reference":
            return run_reference_stage(args.reference_args, args.jpp_root)
        if args.internal == "_router":
            return fit_router_in_layer_batches(args)
        return grade_correctness(args)

    args = build_parser().parse_args()
    ensure_reference(args.jpp_root, args.bootstrap_reference)
    if not (args.jpp_root / "scripts/jpp_cli.py").is_file():
        raise SystemExit("reference checkout missing; pass --bootstrap-reference or --jpp-root")
    head = reference_head(args.jpp_root)
    if head != REFERENCE_COMMIT:
        raise SystemExit(f"reference checkout is {head}, expected pinned {REFERENCE_COMMIT}")
    selected = list(PROFILES.values()) if args.model == "all" else [PROFILES[args.model]]
    if args.stage == "compare" and args.model != "qwen9b":
        raise SystemExit("--stage compare requires --model qwen9b")
    for profile in selected:
        orchestrate(profile, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
