# Playground debugger tests

Normal tests include config/source-plan validation (`test_playground.py`) and
a real DAP round trip with a tiny locally created CPU checkpoint
(`test_playground_dap.py`). These do not download checkpoints or allocate GPU
memory. The latter also checks consecutive sessions on a private persistent host.

## Before a major release

Explicitly enable the real-checkpoint suite from the repository root:

```bash
LLMDBG_RELEASE_TESTS=1 uv run python -m unittest discover -s tests -p test_playground_gpu.py -v
```

Without `LLMDBG_RELEASE_TESTS=1`, this suite is skipped, including during ordinary
test discovery. Only enable it on a CUDA machine with sufficient free memory;
close other resident model sessions first. It can download model weights and
requires the normal llmdbg model-runtime dependencies (including Transformers
and gb10-load-llm).

The suite launches **all five curated files** through a private DAP host using
their configured model IDs, messages, images, and chat settings. It tests entry,
attention and MLP stepping, source frames, local variables, finite residual
vectors, predictions, and generated-token execution. It also checks image region
metadata, the thinking prefix, both AUTO_REPLY turns, and the final assistant
prefill. GPT-2 receives a two-token generation budget in the launch request
because its raw `tokenize()` demo has no generation setting. Other demos retain
their configured budgets. This is a smoke test, not a check of answer quality
or completion of the entire thinking response.

Models run sequentially; shared Qwen3/Qwen3.5 weights are reused across adjacent
demos. Tests do not attach to or terminate a user's existing debugger host.
Generation caches are isolated under the run's log directory.

Requests have a finite timeout (default 1800 seconds), and intermediate-turn
resolution has an additional deadline. Override the request timeout if needed:

```bash
LLMDBG_RELEASE_TESTS=1 LLMDBG_RELEASE_TIMEOUT=2400 uv run python -m unittest discover -s tests -p test_playground_gpu.py -v
```

Results are saved under `build/test-results/playground-<timestamp>-<pid>/`:

- `summary.json`: pass/fail, model metadata, token counts, elapsed times, errors.
- `*.dap.jsonl`: complete requests, responses, events, and debugger output.
- `host-stdout.log` / `host-stderr.log`: host diagnostics.

A failed or timed-out host is stopped before subsequent demos run. Successful
cases disconnect normally. A nonzero test exit means the release gate failed.
