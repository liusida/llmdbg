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

```text
model = "openai-community/gpt2"
revision = "main"
device = "cuda"
dtype = "bfloat16"
prompt = "The capital of France is"
token = 3
stop_on_entry = false

submodule = "token_embedding"

layer = 0
submodule = "transformer.h.0.attention"
submodule = "transformer.h.0.mlp"

layer = 1
submodule = "transformer.h.1.attention"
submodule = "transformer.h.1.mlp"
```

The extension can start directly from the active `.llmdbg` file, so a
`launch.json` is optional. If one is used, it only needs a `llmdbg` launch with
`"config": "${file}"`.

The debugger keeps that same `.llmdbg` document active while stepping. Runtime
state appears in the debugger panels; it does not create or switch to a second
`*.execution.llmdbg` editor and never rewrites the configuration file.
Each `submodule` statement selects a model execution stop and can receive a
breakpoint. Repeated assignments are intentionally valid in the llmdbg source
language; this file is not TOML. Unlisted model operations are omitted from
the debug trace.

Set `token = 3` in the source to make that zero-based position the default for
ordinary breakpoints. A conditional breakpoint such as `token=4` overrides the
default for one submodule. llmdbg prefills earlier tokens into the KV cache and
executes the selected token as a one-token cached forward pass. The **Token**
scope shows its position, token ID, decoded text, and prompt prefix.

Watch one zero-based dimension with `residual_stream[42]`, `output[42]`,
`input[42]`, or `delta[42]`. Run **llmdbg: Plot Current Vector** from the
Command Palette or debug toolbar for an auto-updating dimension plot,
activation histogram, and top-absolute-activation view. The debugger also
exposes explicit **Residual stream (output)** and **Delta** scopes.

Attention stops expose their 768D LayerNorm output. MLP stops additionally
expose the 3072D linear expansion before activation, the 3072D expansion after
GELU, and the 768D projection. These appear as debugger scopes and plot choices
and support indexed Watches: `layernorm[42]`,
`mlp_expansion_pre_activation[42]`, `mlp_expansion[42]`, and
`mlp_projection[42]`.

At an MLP stop, expand **Top activated neurons** to inspect the five largest
post-GELU activations. Every neuron includes its output-direction norm, five
direct vocabulary projections through `W_down @ W_unembed`, and five scores
scaled by the current activation.
Add `top_neurons` to **Watch** to keep the expandable neuron and token results
updated while stepping between MLP stops.

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
