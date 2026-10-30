# Format-v2 Lens fitting

- `fit_lenses.py`: J/R backward-rule fitting.
- `train_s_lens.py`: self-contained S-Lens training: frozen model loading,
  dataset splits, activation capture, MSE/cosine/KL losses, Adam, monitoring,
  checkpoints and v2 export.
- `publish_lenses.py`: validate and publish saved artifacts.

There are no separate S-Lens helper modules. The old ridge fitter, HDD capture,
online helper, KL helper and failed S Cos Future experiment are archived under
`to_delete/`. Cache training and `--loss s-cos-future` are no longer supported.
The archived files are historical references, not supported commands.

The trainer's tests are consolidated in `tests/test_train_s_lens.py`, grouped
by CLI, online capture, geometric/KL losses, checkpoints, monitoring and command
execution. They use tiny CPU fixtures, not pretrained model downloads:

```bash
.venv/bin/python -m unittest tests.test_train_s_lens -v
```

## S Cos training

Run from the project root:

```bash
uv run --group training python scripts/fit_lenses/v2/train_s_lens.py \
  --model Qwen/Qwen3.5-2B --loss cosine --tokens 1000000 \
  --forward-batch-size 32 --batch-size 8192 --epochs 5 \
  --output build/lenses/v2/Qwen--Qwen3.5-2B/s-cos-loss --run-name cos-loss
```

Each batch passes through the frozen model under inference mode; captured
activations feed matrix updates immediately and are then discarded. No HDD
activation cache is read or written. Loading uses `gb10-load-llm`, not
Accelerate. `--source online` is accepted for compatibility and is the only
supported source.

`--forward-batch-size` counts documents per model forward. `--batch-size` caps
activation positions per optimizer update within that forward batch; the
actual update can be smaller. Larger batches can change update counts and
numerical results. The defaults are 32 documents and 8192 positions.

The text dataset is `NeelNanda/pile-10k`: shuffle from row 1000 with seed 2026,
deduplicate truncated token IDs, reserve 20 validation and 50 test documents.
Default context length is 256. Exclude padding, the first four positions and
each document's last real token. Epochs revisit the same exact training-position
budget, shuffling batch order and positions within each batch. Validation and
test also use online forwards. Dataset/model download caches may still be used.

Select source sites with `--sites embed mid post` (default: all) and optionally
`--layers 12`. The final Post site is excluded because its map is identity.
The 24-layer 2B default has 48 learned maps.

## Supported losses

- `--loss cosine` (alias `cos`): mean
  `1 - cosine(source @ A.T, final)` at the same position; FP32 loss, epsilon
  1e-8. Only A is trained; vector lengths are not optimized directly.
- `--loss mse` (the unchanged default): mean squared error to the raw final
  residual. Identity-initialized, bias-free maps, Adam, FP32 weights/moments,
  BF16 CUDA map multiplication, FP32 loss.
- `--loss kl` (alias `tuned`): mean `KL(final distribution || mapped distribution)`
  over the full vocabulary, temperature 1. The model's final norm and output
  head are frozen, and teacher probabilities are detached. The translator has
  **no bias**: this is a Tuned-Lens-style variant, not a reproduction of the
  published affine/optimizer recipe.
- `--loss s-cos-next-current-91` and `s-cos-46`: retained gated next/current
  experiments, described below.

Progress and TensorBoard report the selected objective as `loss`, alongside
MSE, relative MSE and cosine. Geometric improvement is not proof of improved
token decoding. The loss is part of resume identity.

### KL vocabulary chunks

```bash
uv run --group training python scripts/fit_lenses/v2/train_s_lens.py \
  --model Qwen/Qwen3.5-2B --loss kl --tokens 1000000 \
  --forward-batch-size 32 --batch-size 8192 --kl-chunk-size 128 --epochs 5 \
  --output build/lenses/v2/Qwen--Qwen3.5-2B/s-kl-loss --run-name kl-loss
```

`--kl-chunk-size` limits positions scored against the full vocabulary at once.
Teacher logits are shared across sites; each student graph is backpropagated
immediately with weight `chunk_positions / actual_batch_positions`. Adam steps
once per full activation batch, not per vocabulary chunk. There is no top-k
approximation. KL metrics are nats per token. Validation/test use the actual
FP16-exported maps.

## TensorBoard

TensorBoard is enabled by default; `--no-tensorboard` disables it.
Use `--run-name` to distinguish experiments and a separate `--output` directory
for each new run. A missing name is generated from the timestamp. Resume
reuses the saved name unless overridden. `--tensorboard-dir` changes the event
root without affecting checkpoint compatibility.

```bash
uv run --group training tensorboard --logdir build/tensorboard/s-lens \
  --host 127.0.0.1 --port 6006
```

Scalars use cumulative processed positions. They include global/per-site
training loss, MSE, relative MSE, cosine and KL (when enabled), epoch validation,
final test scores, learning rate and throughput. Throughput includes intervening
capture, checkpointing and evaluation. Events flush every ten seconds and at
checkpoints. Resume uses TensorBoard's purge marker beyond the restored count.

## Checkpoints and artifact compatibility

The default output is `build/lenses/v2/<model-id>/s/`. Use `--output` to keep
experiments separate. Existing nonempty outputs are rejected unless resuming.

`optimizer-state.safetensors` is atomically replaced roughly every 100k
processed positions at completed document-batch boundaries, and at epoch ends.
`--resume` restores weights, Adam moments/steps, and deterministic batch cursor.
Work after the latest checkpoint is repeated. Ordinary online MSE, cosine,
KL and gated-loss recipe hashes remain compatible with the pre-consolidation
trainer; cached/ridge/future runs are not supported.

Retained weights-only snapshots are written after the first update crossing
1k, 10k, 100k and 1M positions, plus each epoch end. Directory names describe
requested thresholds; the manifest records the actual boundary. For example,
an 8192-position first update crosses the 1k milestone at 8192, not exactly
1000. These snapshots cannot resume Adam. `--no-snapshots` disables them but
keeps the latest optimizer checkpoint, metrics and final artifact.

All losses export square FP16 `lens.safetensors` maps and a checksummed
`manifest.json`: method `s`, algorithm and loss in provenance, convention
`mapped = source @ A.T`. Vocabulary readout applies the frozen final norm and
output head. Validation runs each epoch; test is evaluated at completion and
is never used to select training settings automatically.

## Gated next/current experiments

`s-cos-next-current-91` uses 90% next-position cosine loss and 10% current-token
cosine loss when the previous final-layer prediction matches the current
input token. `s-cos-46` uses 40% / 60% under the same gate. Otherwise both
use 100% next-position loss.

The next target is `final[t]`; the current target is same-document
`final[t-1]`. The gate is
`argmax(head(final_norm(final[t-1]))) == input_ids[t]`.
Capture happens before flattening/shuffling and never crosses document
boundaries. These losses require `--skip-first >= 1`. Monitoring records gate
fraction and eligible current cosine alongside next-target metrics.
