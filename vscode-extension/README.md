# llmdbg for VS Code

This extension connects VS Code's debugger UI to the bundled `llmdbg` Python
Debug Adapter Protocol implementation. On first use it creates a private Python
environment in VS Code's per-extension storage. It does not use or modify the
opened workspace's virtual environment. The private environment owns the model
runtime dependencies, including `gb10-load-llm`, Transformers, and PyTorch.
Environment creation and dependency installation use `uv`, allowing cached
packages to be shared efficiently across environments.
After this release is active, installing a newer llmdbg VSIX automatically
reloads the VS Code window by default. Set `llmdbg.autoReloadAfterUpdate` to
`false` to opt out.

## Prototype quick start

1. Create or open a `.llmdbg.py` Python model program (recommended) or legacy `.llmdbg` file.
2. Press **F5** while that file is active. If prompted, select **llmdbg**.
3. Wait for the private debugger environment to be created on first use.
4. Expand **Variables** and use **Step Over** to advance through model layers.

VS Code assigns `*.llmdbg.py` the dedicated **llmdbg Python** language mode.
This keeps Python highlighting but makes F5 select llmdbg; ordinary `*.py`
files continue to use the normal Python debugger.

The Python format uses `from llmdbg import gpt2, debug`, ordinary functions,
loops, indexing, and slicing. See the repository `gpt2.llmdbg.py` example.
Tensor handles keep model tensors in llmdbg's runtime, and assignments such as
`hidden[1843] = 0` become replayable interventions. The compact format remains
available; for example:

```text
model = "openai-community/gpt2"
revision = "main"
device = "cuda"
dtype = "bfloat16"
prompt = "The capital of France is"
token = 3
stop_on_entry = false

def attention():
    layernorm
    qkv_projection
    attention_mixing
    output_projection
    residual_add

def mlp():
    layernorm
    expansion
    gelu
    projection
    residual_add

submodule = "token_embedding"

layer = 0
attention()
mlp()

layer = 1
attention()
mlp()
```

The extension can start directly from the active `.llmdbg` file, so a
`launch.json` is optional. If one is used, it only needs a `llmdbg` launch with
`"config": "${file}"`.

The debugger keeps that same `.llmdbg` document active while stepping. Runtime
state appears in the debugger panels; it does not create or switch to a second
`*.execution.llmdbg` editor and never rewrites the configuration file.
After `layer = N`, each `attention()` or `mlp()` call selects the corresponding
transformer-block operation and can receive a breakpoint. The older explicit
`submodule = "transformer.h.N.mlp"` spelling remains supported. Repeated
assignments are intentionally valid in the llmdbg source language; this file
is not TOML. Unlisted model operations are omitted from the debug trace.

Use **Step Into (F11)** at an `attention()` or `mlp()` call to enter its
definition above. The active line advances through the five internal operations
with **F10**, while the outer call remains the parent stack frame. Use
**Shift+F11** to finish the submodule and return to the next outer operation.
Breakpoints can be placed directly on those internal operation lines; Continue
enters the corresponding call and stops on the selected line.

Set `token = 3` in the source to make that zero-based position the default for
ordinary breakpoints. A conditional breakpoint such as `token=4` overrides the
default for one submodule. In a `.llmdbg.py` program, comment out the `token = N`
assignment to let ordinary breakpoints stop at every token position; the existing
`with debug(..., token=token)` call may remain unchanged. llmdbg prefills earlier tokens into the KV cache and
executes the selected token as a one-token cached forward pass. The **Token**
scope shows its position, token ID, decoded text, and prompt prefix.

Watch one zero-based dimension with `residual_stream[42]`, `output[42]`,
`input[42]`, or `delta[42]`. Run **llmdbg: Plot Current Vector** to open a
Matplotlib workbench. Supply a vector expression or pasted list, select an
editable named plotting snippet, and render a PNG. Snippets are persisted in
VS Code extension storage as `*.plot.py` files. Each defines
`plot(vector, vector_name, plt, np)` and returns a Matplotlib figure, so the
file remains valid Python without unresolved injected globals. Compact edit/new/trash icons open the files in
VS Code, create them, or move them to the system trash; the plot panel itself
does not embed a code editor.
Use the **Lens** button beside the plot icon in the debug toolbar after an MLP
GELU activation. Its dedicated Neuron Lens panel shows the five most active
neurons and, for each neuron, the top tokens from its
`W_down → W_unembed` output direction with token IDs and direction logits.
Qwen3.5 0.8B, 2B, 9B, and 27B additionally show each raw activation relative
to a packaged per-layer calibration scale, including its measured sign side.
GPT-2 has the same calibrated relative-activation display for its post-GELU
MLP hidden activations.
Choose **J-Lens** to transport each direction through an ordinary averaged
Jacobian, **R-Lens** to use the RelP transport map, or **All lenses** to compare
both with the direct Logit Lens. Lens artifacts are downloaded lazily from
`sida/llmdbg` on Hugging Face and cached locally. The catalog covers Qwen3.5
2B, 9B, and 27B with both J-Lens and R-Lens maps (and also retains the earlier
0.8B maps). Large 9B/27B files resolve to the R-Lens authors' pinned,
checksum-verified release rather than being duplicated.

The methods are from [Verbalizable Representations Form a Global Workspace in
Language Models](https://transformer-circuits.pub/2026/workspace/index.html)
(J-Lens) and [R-Lens: Making J-Lens More Faithful on Early
Layers](https://www.lesswrong.com/posts/nv8oedrnLXKRzNEL9/r-lens-making-j-lens-more-faithful-on-early-layers).
The debugger also exposes explicit **Residual stream (output)** and **Delta** scopes.

Attention stops expose their 768D LayerNorm output, combined 2304D QKV
projection, 768D attention-mixing result, and 768D output projection. Watch
these as `attention_qkv[42]`, `attention_mixing[42]`, and
`attention_projection[42]`. MLP stops additionally expose the 3072D linear
expansion before activation, the 3072D expansion after GELU, and the 768D
projection. These appear as debugger scopes and plot choices and support
indexed Watches: `layernorm[42]`,
`mlp_expansion_pre_activation[42]`, `mlp_expansion[42]`, and
`mlp_projection[42]`.

At an MLP stop, expand **Top activated neurons** to inspect the five largest
post-GELU activations. Every neuron includes its output-direction norm, five
direct vocabulary projections through `W_down @ W_unembed`, and five scores
scaled by the current activation.
Add `top_neurons` to **Watch** to keep the expandable neuron and token results
updated while stepping between MLP stops.

Stops occur before the highlighted submodule. After **Step Over**, its values
appear as the previous operation's output. Use **Set Value** on Watch entries
such as `output[42]`, `mlp_expansion[1843]`, or
`top_neurons[0].activation` to replay the token with that intervention and
observe changed downstream layers.

The prototype loads GPT-2 through `gb10-load-llm`, captures embedding,
attention, and MLP outputs, and shows a logit-lens interpretation at every
stop. The Debug Console supports `layer`, `operation`, `meaning`, `input`, and
`output`.
The Variables pane exposes a frame-aware **Local variables** scope with source
names such as `model`, `prompt`, `tokens`, `token_pos`, `x`, `layer`, and
completed function locals. Values that do not exist yet—such as `x` before
token embedding—are omitted. The same names work in Watch, hover, and the
Debug Console.

The model is lazy-loaded on the first F5 and kept in a persistent host owned by
the extension. Ending a debug session does not unload its GPU tensors. Later F5
runs with the same model, revision, device, and dtype reuse the resident model
and perform only the prompt forward pass. Changing any of those model settings
replaces the resident model; closing or reloading VS Code releases it.

During extension development, set `llmdbg.developmentSourcePath` in workspace
settings to the directory containing the `llmdbg` Python package. Python
changes are then used on the next debug session without rebuilding the VSIX.
