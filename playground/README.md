# Curated debugger demos

Open a demo, set a breakpoint, and start an llmdbg debug session with F5.
Run **llmdbg: Create Starter Files** to copy the five curated demos, their image,
and the shared `.vscode/launch.json` into your workspace. Existing files are
left unchanged. The command opens the GPT-2 demo by default.
The model ID near the top of each file is editable. First use may download
model weights. These are debugger source programs, not ordinary standalone
inference scripts.

| Demo | Purpose |
| --- | --- |
| `GPT2-Text.llmdbg.py` | Smallest starting point: attention, MLPs, residuals, next-token predictions |
| `Qwen3-Chat.llmdbg.py` | Dense Qwen3 text chat with full attention |
| `Qwen35-Vision.llmdbg.py` | Image patches, visual tokens, full attention and DeltaNet |
| `Qwen35-Thinking.llmdbg.py` | Thinking-enabled text reasoning |
| `Qwen3-MultiTurn.llmdbg.py` | Lazy AUTO_REPLY turns, follow-ups, assistant prefill |

The vision demo uses the bundled `images/cat.jpg`; image paths are relative to
the demo file. Keep the image alongside the demo when copying it elsewhere.
The image is copied from the repository workspace's existing cat example.

Config checks and a tiny CPU DAP smoke test run normally. Real-checkpoint debugger
tests are opt-in before major releases; see [tests/README.md](../tests/README.md).

`starters.json` is the packaging allowlist used by Create Starter Files. Other
playground files, local experiments, and other `.vscode` settings are not bundled.
Edit `playground/.vscode/launch.json` to change the default launch configuration;
packaging and Create Starter Files copy that file without generating a duplicate.

Chat and vision use short generation budgets; thinking has a 4096-token budget
shared by reasoning and the final answer. Multi-turn budgets apply separately
to each reply. Model-specific fitted lenses or calibration data may not be
available for every model ID; the model execution examples do not supply them.
