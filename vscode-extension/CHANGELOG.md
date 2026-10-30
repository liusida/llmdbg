# Changelog

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
