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

Create a `gpt2.llmdbg` file in the workspace:

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

Open that file and press **F5**. If VS Code asks for a debugger, choose
**llmdbg**. No `launch.json` is required. If a workspace launch configuration
is preferred, the complete minimal version is:

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
Current Vector** from the Command Palette or debug toolbar to inspect the full
vector as a dimension plot, histogram, or top-activation chart. An open plot
updates automatically after each debugger stop.

Attention stops expose the 768D LayerNorm output, combined 2304D QKV
projection, 768D concatenated attention-head result, and 768D output
projection. MLP stops expose the 768D LayerNorm output, the 3072D linear
expansion before activation, the 3072D expansion after GELU, and the 768D
projection. Use indexed Watches such as `layernorm[42]`,
`mlp_expansion_pre_activation[42]`, `mlp_expansion[42]`, and
`mlp_projection[42]`; all available vectors also appear in the plot selector.

Expanding **Top activated neurons** at an MLP stop lazily ranks the five
largest post-GELU activations. Each neuron shows its output-direction norm,
the top five tokens from `W_down[neuron] @ W_unembed`, and the top five direct
token contributions after scaling by the current activation. Token IDs and
decoded text are both displayed. Results are cached for the current token and
layer.
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
