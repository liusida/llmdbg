# llmdbg

`llmdbg` is an early-stage debugger for stepping through language-model layers
and presenting human-readable interpretations of internal vectors. It speaks
the [Debug Adapter Protocol](https://microsoft.github.io/debug-adapter-protocol/)
so editor integrations can use a standard debugger UI.

The current prototype includes real GPT-2 Small loading, layer-state capture,
logit-lens interpretations, and a VS Code debug extension. A deterministic
built-in trace remains available when no model is configured.

```console
pip install --pre llmdbg
llmdbg --version
```

## VS Code prototype

Build and install the extension:

```console
cd vscode-extension
npm install
npm run package
code --install-extension llmdbg-0.0.1.vsix
```

The recommended format is now a real Python program named `gpt2.llmdbg.py`:

The compound suffix is intentional: VS Code recognizes it as **llmdbg Python**,
so F5 starts llmdbg rather than the ordinary Python debugger, while the editor
still uses Python syntax highlighting.

```python
from llmdbg import debug, gpt2

model = gpt2("openai-community/gpt2", device="cuda", dtype="bfloat16")
prompt = "The capital of France is"
token = 3
tokens = model.tokenize(prompt)

def attention(x, layer):
    normalized = layer.layernorm(x)
    qkv = layer.qkv_projection(normalized)
    mixed = layer.attention_mixing(qkv)
    update = layer.output_projection(mixed)
    return x + update

def mlp(x, layer):
    normalized = layer.layernorm(x)
    hidden = layer.expansion(normalized)
    hidden = layer.gelu(hidden)
    update = layer.projection(hidden)
    return x + update

with debug(model, tokens, token=token):
    x = model.token_embedding(tokens)
    for layer in model.layers:
        x = attention(x, layer)
        x = mlp(x, layer)
    x = model.final_layernorm(x)
    logits = model.vocabulary_projection(x)
```

This is executable Python. Model and tensor objects are handles over llmdbg's
runtime; `hidden[1843] = 0` records an activation intervention and recomputes
the selected token. Python indexing and slicing such as `hidden[:10]` work in
source and Watch. The original compact `gpt2.llmdbg` format remains supported:

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

Open either file and press **F5**. If VS Code asks for a debugger, choose
**llmdbg**. No `launch.json` is required. If a workspace launch configuration
is preferred, the complete minimal version is:

F5 runs directly to the first matching breakpoint. Assign
`stop_on_entry = True` in a `.llmdbg.py` program only when you explicitly want
to stop at token embedding first.

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "type": "llmdbg",
      "request": "launch",
      "name": "llmdbg: Active file",
      "config": "${file}"
    }
  ]
}
```

On first use, the extension creates a private Python environment in VS Code's
extension storage. It does not install anything into the workspace. The first
GPT-2 run downloads and loads the model, then keeps its tensors resident in a
persistent model-host process. Stopping and restarting the debugger reuses the
resident model and runs only a new forward pass. Closing/reloading VS Code
stops the host and releases the model. After `layer = N`, `attention()` and
`mlp()` resolve to that transformer block and become authored execution stops.
Set breakpoints on these calls or use
**Step Over** to advance through them. Runtime vectors appear in debugger
scopes; the source file is never rewritten.

Press **Step Into (F11)** on `attention()` or `mlp()` to enter its authored
definition in the same file. Attention exposes LayerNorm, the combined QKV
projection, the concatenated per-head attention result, output projection, and
residual add. MLP exposes LayerNorm, expansion, GELU, projection, and residual
add. **F10** advances inside a definition; **Shift+F11** returns to the next
model operation. The call site remains visible as the parent stack frame.

`token = 3` makes token position 3 the default for every ordinary source
breakpoint. To override it for one breakpoint, use the shorter conditional
expression `token=4`. Positions are zero-based. The debugger prefills tokens
before the selected position into the KV cache, then captures that token with
a one-token forward pass. Stepping beyond the last selected layer advances to
the next token using the updated cache.

Watch one zero-based dimension with `residual_stream[42]`, `input[42]`,
`output[42]`, or `delta[42]`. The **Residual stream (output)** debugger scope
exposes the current hidden representation explicitly. Run **llmdbg: Plot
Current Vector** to open the Matplotlib workbench. Enter a vector expression
such as `hidden`, `qkv`, `residual_stream`, or `hidden[:256]`, then choose or
edit a named plotting snippet. Snippets are saved in VS Code extension storage;
the included **Line plot** receives `vector`, `vector_name`, `plt`, and `np`.
Pasted Python lists and tuples are also accepted as vectors.
Plot snippets are ordinary `*.plot.py` files under the extension's global
storage. Use the edit icon to open the selected file in VS Code, the plus icon
to create one, and the trash icon to move one to the system trash. The panel
contains no embedded code editor.
Source variables are discovered from the Python assignments and shared by
Watch and Plot. Renaming `hidden` to `features`, for example, makes `features`
and `features[:256]` available automatically after that assignment executes.
`x` is the residual stream entering the current outer call;
`residual_stream` is its debugger-friendly alias.

Attention stops expose the 768D LayerNorm output, combined 2304D QKV
projection, 768D concatenated attention-head result, and 768D output
projection. MLP stops expose the 768D LayerNorm output, the 3072D linear
expansion before activation, the 3072D expansion after GELU, and the 768D
projection. Use indexed Watches such as `layernorm[42]`,
`mlp_expansion_pre_activation[42]`, `mlp_expansion[42]`, and
`mlp_projection[42]`; any full vector or slice can be entered in the plot workbench.

Expanding **Top activated neurons** at an MLP stop lazily ranks the five
largest post-GELU activations. Each neuron shows its output-direction norm,
the top five tokens from `W_down[neuron] @ W_unembed`, and the top five direct
token contributions after scaling by the current activation. Token IDs and
decoded text are both displayed. Results are cached for the current token and
layer.
For Qwen3.5 0.8B, 2B, 9B, and 27B, the activation is also shown relative to a
packaged, model- and layer-specific scale. Each scale is the median across a
fixed 672-token calibration prompt of the per-token p99 absolute post-SwiGLU
activation. The displayed positive/negative side records the sign of the
calibration activation nearest that p99 magnitude; unsupported checkpoints
show only the raw value instead of silently borrowing another model's scale.
GPT-2 is calibrated the same way on its post-GELU MLP hidden activation. Its
1,024-token context limit leaves 1,021 analyzed tokens after the calibration
prefix; that explicit truncation is recorded in the JSON provenance.
The corresponding calibration plots live beside the JSON data in
`src/llmdbg/data/` as `*_mlp_activation_scales.png`. They show every
token p99 measurement as a scatter point, continuous p10–p90 and p1–p99
percentile ribbons, and the packaged median scale as a marked guide line. Run
`scripts/measure_mlp_activation_scales.py` with `--plot-output` to reproduce
them from the calibration prompt.
The toolbar's **Lens** view has a selector for **Logit Lens**, **J-Lens**,
**R-Lens**, or **All lenses**. Logit Lens projects the neuron's down-projection
direction directly through the unembedding. J-Lens first transports it through
an averaged ordinary Jacobian. R-Lens uses the same transport construction
with value-preserving RelP backward rules. Both transported views then apply
the final norm and unembedding. A missing artifact is reported explicitly;
the debugger never silently substitutes the direct projection.

These implementations follow the methods introduced in:

- [Verbalizable Representations Form a Global Workspace in Language Models
  (J-Lens)](https://transformer-circuits.pub/2026/workspace/index.html): for
  each residual layer, average the Jacobian to the final residual layer over
  prompts and token positions, then read an activation with
  `W_unembed · final_norm(J_layer · activation)`.
- [R-Lens: Making J-Lens More Faithful on Early
  Layers](https://www.lesswrong.com/posts/nv8oedrnLXKRzNEL9/r-lens-making-j-lens-more-faithful-on-early-layers):
  retain the J-Lens construction while replacing ordinary backward behavior
  with the LN, identity, and half RelP rules for dense gated models.

The current Neuron Lens UI applies these maps to individual MLP output
directions. (The maps themselves also support residual-stream activations, but
that readout is not yet exposed in this UI.) It displays the resulting
pre-softmax logits; their vocabulary ranking is the same as after softmax. The
catalog currently covers Qwen3.5 0.8B, 2B, 9B, and 27B. The 9B and 27B entries
use checkpoint-matched J-Lens and R-Lens maps from the R-Lens authors' pinned
[`workspace-lenses`](https://huggingface.co/camilablank/workspace-lenses)
release. The 0.8B and 2B Base maps are fitted locally with the same reference
implementations. These maps average 25 pretraining-like sequences of up to 128
tokens, excluding the first four and final token positions, rather than the
J-Lens paper's default 1,000 sequences. They therefore implement the same
estimator and readout, but use a smaller fitting recipe and are not full-data
reproductions of the paper's maps.

Qwen 3.5 J-Lens and R-Lens data can be reproduced independently:

```console
uv pip install datasets "git+https://github.com/anthropics/jacobian-lens.git"
uv run python scripts/fit_qwen35_j_lens.py \
  --model Qwen/Qwen3.5-0.8B-Base --prompts 25
uv run python scripts/fit_qwen35_r_lens.py \
  --model Qwen/Qwen3.5-0.8B-Base --prompts 25
```

Replace the model ID with `Qwen/Qwen3.5-2B-Base` to reproduce the published
2B maps. The 9B and 27B catalog entries point to immutable, checksum-verified
files in the authors' release rather than duplicating those multi-gigabyte
artifacts.

Both scripts use `gb10-load-llm` and checkpoint after every prompt. The R-Lens
fitter additionally verifies that its RelP substitutions preserve the ordinary
forward pass exactly. Fitted artifacts are written below `build/j_lens/` and
`build/r_lens/` for validation and Hub publication; they are not bundled into
the wheel or extension. Neuron Lens lazily downloads only the selected maps
from `sida/llmdbg` into the standard Hugging Face cache. A Base artifact may be
used by the matching Instruct model only when its hidden width agrees and the
artifact manifest opts into that transfer. This Base-to-Instruct reuse is a
convenience approximation, not a guarantee made by either lens method; fitting
a map on the exact checkpoint is preferable for rigorous analysis.

Add `top_neurons` to **Watch** to keep this tree visible while stepping; VS
Code reevaluates it at every stop and preserves the expandable neuron/token
structure. At non-MLP stops it reads `available at MLP stops`.

The yellow execution arrow identifies the next submodule to execute. **Step
Over** executes that submodule, moves to the next statement, and exposes the
completed operation under **Previous output** and its detail scopes. Watch
dimensions are writable through VS Code's **Set Value** action. For example,
set `output[42]`, `mlp_expansion[1843]`, or
`top_neurons[0].activation`; llmdbg records the intervention, replays the
selected token from its KV-cache prefix, and refreshes all downstream values.
Rank-based `top_neurons[0]` may identify a different neuron after mutation;
use `mlp_expansion[neuron_id]` when targeting a fixed neuron.

## License

MIT
