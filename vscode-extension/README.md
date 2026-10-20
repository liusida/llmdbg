# llmdbg for VS Code

This extension connects VS Code's debugger UI to the bundled `llmdbg` Python
Debug Adapter Protocol implementation. On first use it creates a private Python
environment in VS Code's per-extension storage. It does not use or modify the
opened workspace's virtual environment. The private environment owns the model
runtime dependencies, including `gb10-load-llm`, Transformers, and PyTorch.
Environment creation and dependency installation use `uv`, allowing cached
packages to be shared efficiently across environments.

## Prototype quick start

1. Install this extension's VSIX.
2. Open **Run and Debug**, select **llmdbg: Demo model**, and press **F5**.
3. Wait for the private debugger environment to be created on first use.
4. Expand **Variables** and use **Step Over** to advance through model layers.

With a `model` launch setting, the prototype loads GPT-2 through
`gb10-load-llm`, captures embeddings and transformer-block outputs, and shows a
logit-lens interpretation at every stop. Without `model`, it runs a
deterministic built-in trace. The Debug Console supports `layer`, `operation`,
`meaning`, `input`, and `output`.

During extension development, set `adapterSource` in `launch.json` to the
directory containing the `llmdbg` Python package. Python changes are then used
on the next debug session without rebuilding the VSIX.
