# Workspace Lens v2 implementation and execution plan

## Scope

J, R, and J++ share one site-aware artifact, resolver, loader, and release path.
Artifacts contain `embed`, `mid.L`, and `post.L` maps. `Residual Pre [L>0]`
resolves to `post.L-1`; final Post is an identity rule and is not fitted.
Version-1 artifacts remain explicitly Post-only.

## Current implementation status

- Implemented the shared v2 resolver, exact model/revision checks, identity rule,
  Post-only legacy fallback, and exact missing-site messages.
- Implemented unified resumable J/R fitting and site-aware J++ fitting/packaging.
- Implemented hash/shape/schema validation, clean-cache staging, and index-last
  publication to `sida/llmdbg`.
- Current automated result: 100 tests plus 10 subtests pass.
- No full-quality refit and no Hub upload has started yet.

## Runtime observation (2026-10-22)

The new site-aware job is larger than the artifacts that felt fast to fit
earlier. The old J/R scripts defaulted to **25 prompts**, skipped the first 4
positions, fitted only one Post source per transformer block, and used 32
Jacobian rows per vector-Jacobian batch. The revised v2 recipe also skips the
first **4** positions and fits Embed plus Mid and Post sites. It uses **100 J/R
prompts for GPT-2, Qwen 0.8B, and both Qwen 2B checkpoints**, and **25 prompts
for Qwen 9B and 27B**. The 100-prompt jobs are evaluated at checkpoints 25,
50, and 100; extension to 250 requires held-out quality or map-convergence
evidence. The 25-prompt large-model jobs are not automatically extended.

The prompt count makes jobs through 2B **4x** the old default; the 9B and 27B
jobs retain the old 25-prompt budget. The current
16-row batch requires twice as many backward launches as the old 32-row batch,
but each launch is half as wide, so this does **not** double mathematical
backward FLOPs. Batch size affects utilization, launch overhead, and memory and
must be timed rather than treated as a work multiplier. Capturing Mid and Post
roughly doubles artifact and
accumulator storage; it does not strictly double the whole backward traversal,
because gradients for all requested sites are returned from the same sweep.
For J++, the paper-faithful statistical recipe is **64 Jacobian sequences** of
128 tokens, **1,000 separate router/clustering sequences**, eight experts,
the same **4-position skip**, disjoint expert-weight data, and held-out
evaluation. Exposing Embed/Mid/Post
still creates almost twice as many routed source sites as a Post-only fit, so
its router/expert work grows materially.

| Work factor | Earlier J/R default | v2 full-quality recipe | Effect |
|---|---:|---:|---:|
| J/R calibration prompts | 25 | 100 through 2B; 25 for 9B/27B | 4x through 2B; unchanged for large models |
| Jacobian dimension batch | 32 | tuned per model | same row work; different overhead/memory |
| Source sites | Post only | Embed + Mid + Post | about 2x stored maps; smaller compute effect |
| Leading positions skipped | 4 | 4 | directly comparable token-position policy |

The core estimator itself has not become intrinsically slower: both versions
perform one replicated forward and `ceil(d_model / dim_batch)` vector-Jacobian
backward sweeps per prompt, requesting all source activations in each sweep.
Through 2B, the large difference is therefore chiefly the deliberately larger
statistical recipe, followed by the tuned dimension batch and Qwen kernel
fallback. For 9B and 27B, additional site coverage and model cost dominate;
their prompt budget itself has not increased.

Expected v2 accumulator/artifact storage from the exact model configurations:

| Model | Maps | Resumable fp32 checkpoint | Published fp16 artifact |
|---|---:|---:|---:|
| GPT-2 | 24 | 0.053 GiB | 0.026 GiB |
| Qwen3.5 0.8B | 48 | 0.188 GiB | 0.094 GiB |
| Qwen3.5 2B Base | 48 | 0.750 GiB | 0.375 GiB |
| Qwen3.5 2B Chat | 48 | 0.750 GiB | 0.375 GiB |
| Qwen3.5 9B Chat | 64 | 4.000 GiB | 2.000 GiB |
| Qwen3.5 27B Chat | 128 | 12.500 GiB | 6.250 GiB |

Qwen timing is additionally affected by the installed environment falling back
to reference causal-convolution and linear-attention kernels. Estimates based
on that environment must be labeled as fallback-kernel measurements until the
optimized kernels are installed and remeasured.

On Qwen3.5 0.8B with the current fallback kernels, a warmed 16/32/64 batch
sweep projected 51.4/55.9/59.7 minutes for 100 J prompts. Batch 16 is therefore
the proper choice in this environment; larger batches are slower, not faster.

The corrected two-shard estimator warms CUDA first, separates fixed forward
cost from per-Jacobian-row cost, and now records CUDA allocation, reservation,
and free-memory headroom. The measurements below were collected with
`skip_first=16`; they must be refreshed with the final value of 4 before a
full fit starts. Small-model rows project 100 prompts. The original large-model
measurements also projected 100 prompts; their scheduling projections below
are scaled to the newly selected 25-prompt budget and are marked accordingly:

| Job | Batch | Measured rows | Projected runtime | Checkpoint | Checkpoint path |
|---|---:|---:|---:|---:|---|
| GPT-2 J | 16 | 32/64, twice | 6.7 min | 0.053 GiB | `build/lenses/v2/openai-community--gpt2/j/fit.checkpoint.pt` |
| GPT-2 R | 16 | 32/64, twice | 3.6 min | 0.053 GiB | `build/lenses/v2/openai-community--gpt2/r/fit.checkpoint.pt` |
| Qwen 0.8B J | 16 | 32/64, twice | 51.7 min | 0.188 GiB | `build/lenses/v2/Qwen--Qwen3.5-0.8B/j/fit.checkpoint.pt` |
| Qwen 0.8B R | 16 | 32/64, twice | 52.4 min | 0.188 GiB | `build/lenses/v2/Qwen--Qwen3.5-0.8B/r/fit.checkpoint.pt` |
| Qwen 2B Base J | 16 | 32/64, twice | 2.13 h | 0.750 GiB | `build/lenses/v2/Qwen--Qwen3.5-2B-Base/j/fit.checkpoint.pt` |
| Qwen 2B Base R | 16 | 32/64, twice | 1.98 h | 0.750 GiB | `build/lenses/v2/Qwen--Qwen3.5-2B-Base/r/fit.checkpoint.pt` |
| Qwen 2B Chat J | 16 | 32/64, twice | 2.15 h | 0.750 GiB | `build/lenses/v2/Qwen--Qwen3.5-2B/j/fit.checkpoint.pt` |
| Qwen 2B Chat R | 16 | 32/64, twice | 2.04 h | 0.750 GiB | `build/lenses/v2/Qwen--Qwen3.5-2B/r/fit.checkpoint.pt` |
| Qwen 9B J | 4 | 8/16, twice | 2.82 h (25 prompts, scaled) | 4.000 GiB | `build/lenses/v2/Qwen--Qwen3.5-9B/j/fit.checkpoint.pt` |
| Qwen 9B R | 4 | 8/16, twice | 2.61 h (25 prompts, scaled) | 4.000 GiB | `build/lenses/v2/Qwen--Qwen3.5-9B/r/fit.checkpoint.pt` |
| Qwen 27B J | 2 | 2/4, once | 11.72 h (25 prompts, scaled) | 12.500 GiB | `build/lenses/v2/Qwen--Qwen3.5-27B/j/fit.checkpoint.pt` |
| Qwen 27B R | 1 | 2/4, twice | 13.18 h (25 prompts, scaled) | 12.500 GiB | `build/lenses/v2/Qwen--Qwen3.5-27B/r/fit.checkpoint.pt` |

These are projections, not completed fits. Remaining rows will be added only
after their corrected timing shards run.

Memory-guided batch checks found that larger is not faster with the current
fallback Qwen kernels. Qwen 2B J used 26.0 GB peak at batch 32 and 48.2 GB at
batch 64, but both projected slower than batch 16. Qwen 9B used 31.7 GB at
batch 8 and also projected slower than batch 4. Qwen 27B batch 2 used 63.7 GB
peak with about 30.2 GB free before the timed sample and improved its J
projection over batch 1; no larger 27B batch is needed for queue classification.

J++ timings remain pending a site-aware router/expert shard. Its work must not
be inferred by multiplying the J/R prompt count: the agreed recipe is 1,000
router sequences and 64 expert-Jacobian sequences. A completed **Post-only**
Qwen 2B Chat reference run on this machine provides a concrete lower-bound
basis: roughly 4 minutes for its router, 69 minutes for experts/merge, and 7
minutes for weight fitting plus evaluation (about 80 minutes total). The v2 run
has 48 Embed/Mid/Post sources instead of 23 Post sources. Because the expert
Jacobians share backward sweeps but routing/clustering and stored gradients
grow with site count, the provisional scheduling range is **1.5–2.0 hours**;
this will be replaced by the site-aware timing shard before the fit starts.

## Execution gates

1. Finish the Qwen3.5-2B Chat set first: J, R, then J++ (router, experts,
   learned composition, held-out evaluation). This explicit priority overrides
   the usual shortest-job ordering; every stage remains resumable. Current
   scheduling estimates are J 2.15 h, R 2.04 h, and J++ provisionally 1.5–2.0 h.
2. Then run the remaining short queue: GPT-2 R/J, Qwen 0.8B J/R, Qwen 0.8B
   J++, Qwen 2B Base R/J/J++, and GPT-2 J++.
3. Recheck Qwen estimates with the intended optimized kernels when available.
4. Before any remaining job projected above two hours, report its measurement basis,
   duration, storage, and resumable checkpoint path.
5. Run the final long queue sequentially: Qwen 9B R/J, then Qwen 27B J/R.
   Do not run 27B J++. Keep official 9B J++
   pinned and Post-only.
6. Run held-out transport/logit tests and J++ recall@10 evaluation.
7. Stage files, verify from a clean cache, upload artifacts first, and update
   the Hub index last.
8. Install the updated extension locally only after clean-cache integration
   tests pass.

