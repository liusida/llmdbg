# J-Lens evaluation

`eval_qwen3_8b_j_lenses.py` compares the local llmdbg format-v1 Qwen3-8B
J-Lens with Neuronpedia's Qwen3-8B J-Lens on Anthropic's six public readout
evaluations.

It downloads the pinned evaluation JSON files, keeps only gradeable prompts
that Qwen3-8B answers correctly, ranks every accepted single-token surface of
each latent intermediate at every common fitted layer, and reports the
official item-weighted pass@k metric. Association and typo items have no
answer-correctness filter and are always retained.

The benchmark pins both Anthropic's evaluation-data revision and the local
Qwen3-8B checkpoint revision. These can be overridden explicitly when needed.

Run a one-item-per-task smoke test first:

```bash
uv run --group dev python scripts/fit_lenses/eval/eval_qwen3_8b_j_lenses.py --limit 1
```

Run the complete benchmark:

```bash
uv run --group dev python scripts/fit_lenses/eval/eval_qwen3_8b_j_lenses.py
```

Results are written to `build/lenses/eval/qwen3-8b-j-lens/`:

- `summary.csv`: pass@1/5/10/50 by evaluation and macro-average
- `ranks.csv`: every per-layer intermediate rank
- `results.json`: provenance, correctness decisions, dropped intermediates,
  artifact paths, and summary results

The evaluator uses `gb10-load-llm` for model tensors. It never uses
`accelerate` or `device_map="auto"`.

## Qwen3.5-27B

`eval_qwen35_27b_j_lenses.py` evaluates three checkpoint-matched Post lenses:

- Neuronpedia's legacy J-Lens
- the externally hosted format-v1 J-Lens registered by llmdbg
- llmdbg's format-v2 `post.*` J-Lens maps

The v2 artifact also contains Embed and Mid maps, but those are intentionally
excluded from this comparison because legacy lenses only contain Post maps.

Smoke test after other model-heavy work has stopped:

```bash
uv run --group dev python scripts/fit_lenses/eval/eval_qwen35_27b_j_lenses.py --limit 1
```

Full benchmark:

```bash
uv run --group dev python scripts/fit_lenses/eval/eval_qwen35_27b_j_lenses.py
```

The three large lens artifacts are downloaded from their pinned/default Hub
locations when they are not already cached. Results go to
`build/lenses/eval/qwen35-27b-j-lens/`.
