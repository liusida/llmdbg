# Changelog

## 0.2.0 (Release) — 2026-11-20

### Security

- The local model host now requires a per-host secret. Before, any account on a
  shared machine could connect to the host's `127.0.0.1` port, start a session,
  and run code as the extension's user (e.g. through a plot snippet). The secret
  travels in `LLMDBG_HOST_TOKEN` and the launch request (`hostToken`), never on
  the command line; unauthenticated connections get only `initialize` and are
  closed after 30 seconds. `llmdbg debug --connect` now reads the token from
  `LLMDBG_HOST_TOKEN`.

### Added

- **Shared sessions for agents.** Every debug session can be joined by local
  programs that hold its token (stored in an owner-only file under
  `~/.cache/llmdbg/sessions/`), e.g. `llmdbg debug --attach --client-name claude`.
  Attached clients can step, continue, set breakpoints (mirrored into VS Code),
  evaluate, intervene, and send any `llmdbg/*` request; VS Code follows every
  stop. Only the session owner can launch, configure, or end the session.
  Opt out with `"shareSession": false`.
- **llmdbg Control panel** (sliders icon), opened when a session starts:
  agent access (Full / Read-only / Off), connected agents with Disconnect,
  session changes with Reset weights and Clear activation edits, a copyable
  agent prompt, and an activity feed of what each agent did.
- A bundled agent guide (`llmdbg/data/agent-guide.md`); session files describe
  the client command, `PYTHONPATH`, and guide location.
- **Expression assignment:** `let x = x + 4 * steer`, `let x[0, 0, 46] = 0`,
  `let x *= 2` rewrite the residual stream at the paused stop. Named vectors:
  `vec NAME = EXPR`, `vec NAME from FILE`, `vecs`. Expressions gain `.norm()`,
  `.sum()`, `.mean()`, `.abs()`.
- **Session-scoped weight editing:** `let weight["<parameter>"][...] = EXPR` and
  `let W_o[...] *= 0`, with `params`, `weights`, and `undo weights`. Edits
  recompute the prompt from token 0, bypass generation caches, and are undone
  when the debug session ends.
- Terminal commands `set`, `let`, `vec`, `vecs`, `params`, `weights`,
  `undo weights`, and options `--attach`, `--client-name`.
- Every llmdbg panel tab shows its toolbar icon.

### Changed

- The generation budget belongs to the run:
  `debug(model, inputs, max_new_tokens=N)`. `model.chat(..., max_new_tokens=)`
  still works; a value on `debug(...)` takes precedence. Starters name the input
  `inputs` and set the budget on `debug(...)`. Running a program as plain
  Python now honors the budget too.
- The Lens panel no longer opens automatically; the Control panel does
  (setting `llmdbg.openControlPanelOnStart`).
- Positions read as `token 4 ' is' · layer 5` in the Control panel and activity.

### Fixed

- Prompt Tokens shows tokens generated after the last stop when a run continues
  to the end.

## 0.1.5 (Pre-release) — 2026-10-29

### Added

- Terminal DAP client (`llmdbg debug`) with interactive commands, conditional
  breakpoints, persistent-host connections, JSONL output, and batch/script support.

- Persistent generated-token caching for ordinary debugging, with breakpoint-aware
  fast prefill on Continue. Stepping remains unchanged; interventions and conflicting
  continuations disable cache reuse.
- Curated GPT-2, Qwen3 chat/multi-turn, and Qwen3.5 vision/thinking starter files,
  including a shared editable launch configuration.
- Concept Trace keyboard shortcuts: Home/End select the first/last token;
  PageUp/PageDown select the first/last residual layer.
- Qwen3-4B support in the workspace Lens fitter and Qwen3-8B attention/MLP
  magnitude calibration.
- A local extension build/install helper and opt-in real-model starter release tests.

### Fixed

- Neuron/component Lens readouts now respect the sign of the active contribution.
- Batched Lens panel readouts reduce repeated projection work.
- Direct debugger launches resolve local images relative to the configuration file.
- Prompt Tokens stays synchronized when AUTO_REPLY appends follow-up turns.
