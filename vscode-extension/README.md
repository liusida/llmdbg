# llmdbg for VS Code

This extension connects VS Code's debugger UI to the bundled `llmdbg` Python
Debug Adapter Protocol implementation. On first use it creates a private Python
environment in VS Code's per-extension storage. It does not use or modify the
opened workspace's virtual environment. The private environment owns the model
runtime dependencies, including `gb10-load-llm`, Transformers, and PyTorch.
Environment creation and dependency installation use `uv`, allowing cached
packages to be shared efficiently across environments.

## Prototype quick start

1. Create or open a `.llmdbg` file containing the model and input settings.
2. Press **F5** while that file is active. If prompted, select **llmdbg**.
3. Wait for the private debugger environment to be created on first use.
4. Expand **Variables** and use **Step Over** to advance through model layers.

For example:

```toml
[model]
id = "openai-community/gpt2"
revision = "main"
device = "cuda"
dtype = "bfloat16"

[input]
prompt = "The capital of France is"

[debug]
stop_on_entry = true
```

The extension can start directly from the active `.llmdbg` file, so a
`launch.json` is optional. If one is used, it only needs a `llmdbg` launch with
`"config": "${file}"`.

The prototype loads GPT-2 through `gb10-load-llm`, captures embedding,
attention, and MLP outputs, and shows a logit-lens interpretation at every
stop. The Debug Console supports `layer`, `operation`, `meaning`, `input`, and
`output`.

The model is lazy-loaded on the first F5 and kept in a persistent host owned by
the extension. Ending a debug session does not unload its GPU tensors. Later F5
runs with the same model, revision, device, and dtype reuse the resident model
and perform only the prompt forward pass. Changing any of those model settings
replaces the resident model; closing or reloading VS Code releases it.

During extension development, set `llmdbg.developmentSourcePath` in workspace
settings to the directory containing the `llmdbg` Python package. Python
changes are then used on the next debug session without rebuilding the VSIX.
