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

## Terminal debugger

Use the same DAP backend without VS Code. From a source checkout:

```console
.venv/bin/python -m llmdbg debug starters/Qwen35-Thinking.llmdbg.py
```

It starts a private model host, launches the file, and pauses at entry. Type:

```text
break mlp token=45,layer=10
continue
where
print residual_stream
step
vars
out
lens residual_stream
lens normalized - x
lens j residual-post
tokens
quit
```

`break` accepts a source line number or an authored call name such as `mlp` or
`attention`. `next` steps over, `step` enters, and `out` steps out. `help` lists
commands. `lens EXPRESSION` uses the same decoder as the VS Code `abc` action,
showing all configured, available lenses for a model-width tensor expression.
It also accepts supported slices and arithmetic; non-vector tensors must be
explicitly sliced or projected, not flattened automatically. `lens j residual-post`
retains the separate site-based readout syntax. This is a DAP client, not an
arbitrary Python shell.

To use an already running host and keep its model resident across CLI sessions:

```console
.venv/bin/python -m llmdbg debug --connect 127.0.0.1:5678 starters/Qwen35-Thinking.llmdbg.py
```

For Codex or other automation, use repeated `-c` commands with `--json`:

```console
.venv/bin/python -m llmdbg debug starters/Qwen35-Thinking.llmdbg.py --json \
  -c 'break mlp token=45,layer=10' -c continue -c vector
```

Each command (including automatic launch) writes one JSON result to stdout;
model progress goes to stderr. `-c` executes in order and then disconnects.
`--batch` reads commands from stdin, keeps one debug session for the stream,
and exits nonzero on the first error. An interactive process can likewise stay
open while an agent sends commands. `request COMMAND JSON` exposes arbitrary
DAP and `llmdbg/*` inspection requests; raw requests do not automatically wait
for a stop, so use the dedicated stepping commands for execution control.

The client only terminates hosts it starts itself; disconnecting from an
external `--connect` host leaves that host running. Use `--demo` for a fast,
model-free trial. Model/config execution, downloads, and GPU allocation happen
only when launching a real configuration.

## VS Code extension

Install `SidaLiu.llmdbg` from the VS Code Marketplace. To build and install a
local copy instead:

```console
cd vscode-extension
npm install
npm run package
code --install-extension ../build/releases/vsix/llmdbg-0.1.0.vsix
```

Marketplace publishing and pre-release instructions are in
[`PUBLISHING.md`](PUBLISHING.md).

## Fitting J++ lenses

[`scripts/fit_jpp_lenses.py`](scripts/fit_jpp_lenses.py) is the reproducible,
resumable fitting driver for GPT-2 and all supported Qwen3.5 sizes. It pins the
reference J++ implementation, loads models through `gb10-load-llm`, fits every
residual source layer, and evaluates on a held-out readout split. The Qwen3.5-9B
recipe additionally compares the local refit with the published lens on the
same readouts and writes exact paired significance statistics.

```console
uv pip install datasets pandas pyarrow scikit-learn tqdm
uv pip install --no-deps \
  "jpp-lens @ git+https://github.com/safety-research/jpp_lens@d7efc0de372e20589e899188c0cfe8434acc63ec"

# Inspect the commands first, then run one model or the whole suite.
uv run python scripts/fit_jpp_lenses.py --model all --num-shards 4 --dry-run \
  --bootstrap-reference
uv run python scripts/fit_jpp_lenses.py --model qwen9b --num-shards 4 \
  --bootstrap-reference
```

Outputs live under `build/jpp_lens/<model>/`. For 9B,
`official_comparison_pass_at_k.csv` contains the reference recall table and
`official_comparison_paired.csv` contains local-vs-official recall differences
and exact McNemar p-values. GPT-2 uses llmdbg's value-preserving LayerNorm/GELU
RelP rules because upstream J++ does not currently provide GPT-2 rules; this is
recorded explicitly in each fit manifest.

The recommended format is now a real Python program named `gpt2.llmdbg.py`:

For a new workspace, run **llmdbg: Create Starter Files** from the VS Code
Command Palette. It copies five curated demos from `starters/`: GPT-2 Text,
Qwen3 Chat, Qwen3.5 Vision, Qwen3.5 Thinking, and Qwen3 MultiTurn. It also copies
the vision demo's image and creates a `.vscode/launch.json` entry that runs the
currently active `.llmdbg.py` file. Existing files are left untouched. The GPT-2
demo opens by default; model IDs can be edited near the top of each demo.

`Qwen3-VL-2B-Instruct.llmdbg.py` supports image+text prompts with full attention
at every decoder layer. Use `qwen3vl("Qwen/Qwen3-VL-2B-Instruct")` and the
checkpoint's chat processor. The example exposes Q/K/V, KV-cache tensors,
attention weights, MLP steps, and early `block.visual_residual(x)` DeepStack
injections. The vision encoder runs once during preparation; its embeddings and
DeepStack features are retained for cached stepping and generation. Attention
channel token contributions and image patch maps work at every decoder layer.
Logit Lens is available immediately; J/R/J++ and ICA require artifacts fitted
for this exact checkpoint and are not borrowed from Qwen3.5.

`Gemma4-E2B.llmdbg.py` uses `gemma4("google/gemma-4-E2B-it")` for
image+text debugging. Its 35 decoder layers expose sliding/global softmax
attention, shared KV tensors, GeGLU MLPs, per-layer token-identity embedding
updates, and final layer scaling. Image tokens support attention contribution
and value patch maps. The vision encoder runs during preparation, not as
step-through Python. Logit Lens works without fitted artifacts; other lenses
must match the exact checkpoint (the base E2B and E2B-it are distinct).

The curated chat demos pass literal `messages` through the checkpoint's own
chat template with `model.chat(messages, add_generation_prompt=True)`; the
multi-turn demo instead continues a final assistant prefill.
Set `max_new_tokens` on `model.chat(...)` to continue greedily generating after
the prompt. Each generated token enters the same token-by-token, layer-by-layer
debug loop, so stepping proceeds through its complete forward pass before the
next token is selected. Generation stops at EOS or at the configured limit.

The compound suffix is intentional: VS Code recognizes it as **llmdbg Python**,
so F5 starts llmdbg rather than the ordinary Python debugger, while the editor
still uses Python syntax highlighting.

```python
from llmdbg import debug, gpt2, nop

model = gpt2("openai-community/gpt2", device="auto", dtype="auto")
prompt = "The capital of France is"
tokens = model.tokenize(prompt)

def attention(x, block):
    normalized = block.layernorm(x)
    qkv = block.qkv_projection(normalized)
    mixed = block.attention_mixing(qkv)
    update = block.output_projection(mixed)
    return x + update

def mlp(x, block):
    normalized = block.layernorm(x)
    hidden = block.expansion(normalized)
    hidden = block.gelu(hidden)
    update = block.projection(hidden)
    return x + update

with debug(model, tokens) as run:
    for token in run.tokens:
        x = model.token_embedding(tokens)
        for layer, block in enumerate(model.layers):
            nop()  # resid-pre
            x = attention(x, block)
            nop()  # resid-mid
            x = mlp(x, block)
            nop()  # resid-post

        x = model.final_layernorm(x)
        logits = model.vocabulary_projection(x)
        predictions = model.top_tokens(logits, k=10)
        nop()  # Stable breakpoint with predictions available.
```

This is executable Python. Model and tensor objects are handles over llmdbg's
runtime; `hidden[1843] = 0` records an activation intervention and recomputes
the selected token. Python indexing and slicing such as `hidden[:10]` work in
source and Watch. `nop()` is a repeatable no-operation statement for placing a
breakpoint after an assignment; the older `end()` spelling remains supported as
an alias. The original compact `gpt2.llmdbg` format remains supported:

```text
model = "openai-community/gpt2"
revision = "main"
device = "auto"
dtype = "auto"
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

Dependency environments are isolated into **stable** (GPT-2, Gemma 2, Qwen)
and **gemma4-preview** (Gemma 4) profiles. Selection is automatic from the
source model; `"runtimeProfile": "stable"` or `"gemma4-preview"` in a launch
configuration overrides it. Both pin Python 3.12 and Transformers 5.20.2.
On GB10/CUDA 13, stable pins torch 2.10.0 / torchvision 0.25.0; Gemma preview
pins torch 2.14.1 / torchvision 0.29.1. Other GPU backends have their own
recipes, defined in `vscode-extension/runtime-profiles.js`.

Profiles live in extension storage under `runtimes/<profile-platform-backend-hash>`.
Changing a recipe creates a different directory, leaving other profiles and
the legacy `runtime/` untouched. The resolved dependency lock is saved under
`runtimes/locks/` and reused on rebuild. Model downloads still share the normal
Hugging Face cache. **llmdbg: Show Runtime Diagnostics** shows the selected
profile, interpreter, versions, and CUDA operation checks. Rebuild prompts for
one profile and preserves its old environment as a backup. Switching profiles
restarts the model host, so tensors are not reused across interpreters.
The GB10 profiles have been smoke-tested with Qwen 3.5 2B vision and Gemma 4
E2B-it respectively; other hardware/model combinations require their own tests.

The Qwen 3.5 2B Vision example uses nested `attention(normalized, block)` and
`gated_delta(normalized, block)` cores. Step Into enters either core; Step Over
skips it (unless a breakpoint is hit), and Step Out returns to the outer
output projection. DeltaNet exposes convolved/normalized Q/K/V, decay/beta/output
gates, previous and updated recurrent memory, and the readout as inspectable
tensors. Memory/gate snapshots are read-only diagnostic values; existing
mixed-channel interventions remain available. Extra recurrent snapshots are
captured only for programs defining the expanded DeltaNet core.
When a nested mixer assigns its final output and then uses `return mixed`,
stepping pauses on that return line with `mixed` and the other core locals
still available to Watch, Tensor Plot, and Tensor Decoder. Inline returns such
as `return block.gated_output_norm(readout, gate)` remain supported.

Python helper stepping is derived from the authored source rather than a fixed
operation/line sequence. Ordinary assignments, renamed locals, aliases,
reassignment, tuple unpacking, safe tensor expressions, helper calls and returns
automatically become source locations. Each helper has separate locals and a
stack frame; breakpoints and Step Into/Over/Out follow that same source plan.
Watch, Tensor Plot and Tensor Decoder resolve the locals available at the paused
line. For example, these inspection-only additions need no adapter changes:

```python
mixed = block.gated_output_norm(readout, gate)
copy = mixed
difference = mixed - normalized
return mixed
```

Model computations still use captured native tensors through an explicit
operation registry. Alias arguments are supported; changing an operation's
inputs or the model's residual return is rejected instead of showing an
unrelated captured result. New model operations (or repeated cache-mutating
operations) require backend capture support. Helper bodies are straight-line:
arbitrary Python calls, mutation, recursion, and helper-local loops/branches are
not executed and produce an explicit validation error. Existing outer token and
layer loops remain supported. No runtime dependency changes are needed.

**Decode Tensor Expression** opens a separate toolbar panel with a blank input for expressions
such as `normalized - x`. It displays Logit Lens and every enabled configured
J/R/J++ vocabulary lens, including custom/Neuronpedia entries, independently.
Lenses without fitted maps are hidden; other loading failures remain visible without stopping debugging. The
expression must squeeze to one model-width vector; matrices are not silently
flattened. Variables, layer, and lens context come from the paused line;
there are no separate site/layer controls. These arbitrary-expression readouts are
exploratory labels, not calibrated probabilities or measured causal effects.
Results are marked stale when the debug position changes; press Enter or
Decode to read the current tensors.

`W_o` exposes the paused layer's attention output weight in input-by-output
orientation (including for linear-attention layers). For example,
`mixed[0, 0, :64] @ W_o[:64, :]` projects the first 64 channels into residual
space before decoding. This is a detached inspection copy and excludes bias.
The collapsed **Loop** controls can bind an integer variable (default `i`)
over a start-inclusive, stop-exclusive range, with up to 64 results. Enable it
and use, for example, `mixed[0, 0, i*64:(i+1)*64] @ W_o[i*64:(i+1)*64, :]`
to decode several heads at the current paused position.

The Lens panel also includes **Attention Head Lens**. It decodes
each head's `mixed_h @ W_o_h` using the selected vocabulary lenses and the
model's actual head width. Before the channel lens, it shows the top five heads
by calibrated norm (current norm / per-head reference median), with five tokens
per lens. Compact cards use the same pill meters as channels, full at 10×;
click a pill for raw norm, reference, and × measurements. Missing calibration
is an explicit error with no meter, and ranking falls back to raw norms.
Head outputs exclude projection bias and any later
whole-output normalization. For gated attention, the captured mixed values
already include the output gate. This block is available after mixing completes
in conventional attention and DeltaNet layers.
Each of the five heads occupies its own row. Click the head pill
to inspect the top three channels restricted to that head, indented below
its vocabulary readouts. Channel indices remain global; their existing signed
projection-onto-module-output ranking, calibration, image/source-token controls,
and interventions are preserved. The separate global Channel Lens is optional
in the block picker rather than shown by default.

DeltaNet head cards have a **Memory Contributions** icon. Only clicking it
replays the prefix with a separate cache and captures the selected head's keys,
values, decay and beta history. The shared Token Contributions panel then shows
signed support for the current projected head output, source-token filtering,
image-region highlighting, and collapsed patch maps for support and magnitude.
The decomposition includes later overwriting, with the current query, gate and
RMS denominator held fixed; it is not an attention weight or causal deletion
effect. Positions identify writes, not exclusive pixel origins. Normal debugging
does not collect this history. One result is cached until its inputs or
interventions change. Replays that disagree with the paused snapshot are rejected.

`scripts/measure_attention_channel_activation_scales.py` also records
`head_norm_scales` for each conventional attention layer in its output JSON.
Each query head has its own median reference norm of `mixed_h @ W_o_h`, plus
token-norm percentiles. Prefix tokens are excluded as for channel calibration;
projection and norm statistics use fp32, without bias or subsequent output
normalization. Qwen3.5-2B also includes DeltaNet head references and separate
`value_head_norm_scales` for its Value Lens. The artifact
records the corpus hash and text-only modality; these are text-baseline
references, not image-specific calibration. Existing JSON files must be
regenerated to contain head scales; old files are not silently approximated.
GPT-2 Medium includes head calibration for all 24 layers × 16 heads, using
511 analyzed tokens from the diverse 32-sentence corpus (prefix excluded).
The same corpus also supplies checkpoint-specific head references for GPT-2,
Gemma 2 2B IT, Gemma 4 E2B IT, Qwen3-VL 2B Instruct, and Qwen3.5
0.8B/2B/2B-Base/9B/27B. Only conventional attention layers receive head
references; linear-attention layers retain channel-only calibration.

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
Use `layer=6` to stop only in zero-based transformer block 6, or comparisons
such as `layer>=15` to stop in later blocks. Combine filters as
`token=4, layer>=15`. These are llmdbg selectors, not Python assignments.

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
In the Lens panel, activated MLP-neuron and attention-channel rows decode the
signed component `activation * output_direction`, not the bare positive
direction. Negative activations reverse the readout for Logit, J-, R-, and
Neuronpedia lenses; zero activations have no activated readout. Explicit
`top_neurons[i].vocabulary` Watches retain bare-direction scores, while
`.contribution` retains signed, unnormalized dot-product scores.
For GPT-2 and Qwen3.5 0.8B, 2B, 9B, and 27B, MLP neurons and attention output
channels are ranked by the norm of the vector each component adds to the
residual stream: `abs(activation) * norm(projection direction)`. The signed
`(±…×)` value uses a packaged, model- and layer-specific scale for that same
quantity. Each scale is the median across a fixed calibration prompt of the
per-token p99 component contribution magnitude. Unsupported checkpoints show
the raw activation without silently borrowing another model's scale. GPT-2's
1,024-token context limit leaves 1,021 analyzed tokens after the calibration
prefix; that explicit truncation is recorded in the JSON provenance.
The MLP and attention-channel calibration plots live beside the JSON data in
`src/llmdbg/data/`. They show every
token p99 measurement as a scatter point, continuous p10–p90 and p1–p99
percentile ribbons, and the packaged median scale as a marked guide line. Run
`scripts/measure_mlp_activation_scales.py` or
`scripts/measure_attention_channel_activation_scales.py` with `--plot-output`
to reproduce them from the calibration prompt.
Qwen3-8B has MLP and attention-channel references for all 36 layers, plus
separate output-head and value-head references for all 32 query heads per layer.
Both existing measurement scripts use the same 422 non-prefix tokens from
`scripts/data/diverse_32_sentences.txt`, bfloat16 model values, and fp32 norm
statistics. The artifacts record the exact checkpoint commit. These magnitude
references are shared by Logit/J/R/J++ Lens readouts, not fitted separately per lens.
Gemma 4 E2B-it has attention-channel and MLP scales for all 35 layers,
measured in bfloat16 on the 441 non-prefix tokens of
`scripts/data/diverse_32_sentences.txt`. This is a text-only reference, also
used when inspecting image prompts; it is not an image-specific baseline.
The scripts support Gemma's varying channel/MLP widths. Its plots are in
`build/calibration/`. These scales are not used for the base E2B checkpoint.
The toolbar's **Lens** view has a selector for **Logit Lens**, **J-Lens**,
**R-Lens**, **J++ Lens**, or **All lenses**. A panel refresh fetches its selected
sections in one batched debugger request, with one Lens-availability lookup for
the batch. Top head vectors are decoded together through batched vocabulary
readouts; maps at different activation sites remain separate. Unavailable
sections do not suppress healthy results, and stale refresh replies are ignored.
The single-section debugger request remains supported for clients that need it.
Logit Lens projects the neuron's
down-projection direction directly through the unembedding. J-Lens first
transports it through an averaged ordinary Jacobian. R-Lens uses the same
transport construction with value-preserving RelP backward rules. Both
transported views then apply the final norm and unembedding. A missing artifact
is reported explicitly; the debugger never silently substitutes the direct
projection.
J++ Lens is available for Qwen3.5 9B. It uses the authors' pinned official
artifact, combines filtered expert LRP Jacobians into one transport map per
layer, and excludes tokens without letters or digits from its ranking. The
2.1 GB artifact is downloaded lazily from
[`koayon/jpp-lenses`](https://huggingface.co/koayon/jpp-lenses) and cached by
Hugging Face.

Its block selector supports multiple checked sections. GPT-2 Small and
Qwen3.5 2B Base, 2B Chat, and 9B additionally expose ICA decompositions of residual Pre/Mid/Post
and the attention/MLP updates. They show the top three absolute component
scores and reuse the selected Logit/J/R readout. These projections use the
published per-layer `resid_post` bases; module updates omit the fitted center
because they are displacement vectors.
The 2B Base checkpoint uses the Pile-fitted base ICA artifact, while the 2B
Chat checkpoint uses the UltraChat-fitted instruct artifact. J-Lens and R-Lens
may continue to use their explicitly validated base-to-instruct transfer until
chat-specific artifacts are fitted.

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
- [J++ Lens: Jacobian Filtering Enables More Faithful Workspace
  Lenses](https://www.lesswrong.com/posts/nc9dHfcB22JdMzGbr/j-lens-jacobian-filtering-enables-more-faithful-workspace):
  cluster source activations, fit expert R-Lens Jacobians, and learn a signed
  combination that filters noisy experts before producing the fixed map.

The current Lens UI applies these maps to residual streams, module updates,
attention channels, MLP neurons, and ICA components. It displays the resulting
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

Current site-aware format-v2 Qwen 3.5 J-Lens and R-Lens data can be reproduced
independently:

```console
uv run python scripts/fit_lenses/v2/fit_lenses.py \
  --model Qwen/Qwen3.5-0.8B --method j --prompts 25
uv run python scripts/fit_lenses/v2/fit_lenses.py \
  --model Qwen/Qwen3.5-0.8B --method r --prompts 25
```

The legacy format-v1 fitters and packer are isolated under
`scripts/fit_lenses/v1/` and
their command-line entry points are intentionally blocked because they produce
Post-only maps that cannot be repackaged as site-aware v2 artifacts. The 9B
and 27B catalog entries point to immutable,
checksum-verified files in the authors' release rather than duplicating those
multi-gigabyte artifacts.

Local and Hub artifacts use the same version-first layout:
`lenses/v1/<owner--model>/<method>` and
`lenses/v2/<owner--model>/<method>`. Historical Hub paths remain available as
compatibility aliases for extension versions that cached an older index.

The format-v2 fitter uses `gb10-load-llm` and checkpoints after a configurable
number of prompts. Its R-Lens path additionally verifies that its RelP
substitutions preserve the ordinary forward pass. Fitted artifacts are written
below `build/lenses/v2/` for validation and Hub publication; they are
not bundled into the wheel or extension. Neuron Lens lazily downloads only the
selected maps
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

## Automatic intermediate chat replies

Use `AUTO_REPLY` when a follow-up depends on an assistant response that has not
been generated yet:

```python
from llmdbg import AUTO_REPLY

messages = [
    {"role": "user", "content": "Choose a secret concept. Do not name it. Write a paragraph about something unrelated."},
    {"role": "assistant", "content": AUTO_REPLY},
    {"role": "user", "content": "Now reveal the secret concept you chose."},
]
tokens = model.chat(messages, max_new_tokens=4000, enable_thinking=False,
                    add_generation_prompt=True)
```

Session preparation stops at the first assistant generation prefix: no
intermediate reply is generated just to pause at the first prompt token.
Stepping beyond the prompt generates the intermediate reply one token at a time;
only at EOS does llmdbg append the follow-up with the model's chat-template
boundaries. Full-sequence Concept Trace Replay explicitly resolves pending
intermediate turns on a private execution, leaving live stepping unchanged. Previous token
IDs and the native execution cache are retained rather than decoding and
retokenizing the entire conversation. The final assistant reply is generated by
the debugger or Concept Trace as usual. The complete prior conversation is
available as prompt tokens for inspection.

`max_new_tokens` applies separately to each intermediate reply and the final
reply. Intermediate replies must reach EOS before later turns can run. If one
hits its cap, Concept Trace still shows its partial text and lattice with a
notice; later turns are omitted. Increase the budget to extend the cached partial
reply. Concept Trace has no separate token-position cap: it displays the full
sequence, including partial replies. Longer traces require more memory and
capture time. Generated token IDs are cached across sessions using
the same private cache directory and execution-input identity rules described
below. A bounded resident-model cache also retains up to 32 intermediate
continuations, so replay can reuse them even if disk caching is unavailable.
`main` is resolved to an exact Hub commit before loading config, tokenizer and
weights; that commit is used for persistent cache identity. Mutable local or
otherwise unresolved revisions skip persistent caching. Each turn's key includes
only its preceding context: changing the last user message does not regenerate
earlier `AUTO_REPLY` turns. Logs distinguish resident/disk reuse from actual
generation. The Concept Trace
Regenerate button regenerates the **final** reply; intermediate replies are
resolved as prior conversation context when the full snapshot is requested.

`AUTO_REPLY` must be assistant content between user messages. Multiple such turns
are supported. Images may be supplied before the first placeholder; later
follow-ups currently support text only. This feature preserves the conversation,
but does not guarantee that a later stated secret matches an earlier unwritten
choice.

## Continuing an assistant prefill

To continue directly from an authored assistant prefix, leave the final message
open instead of adding a new assistant header:

```python
messages = [
    {"role": "user", "content": "Reveal the secret concept."},
    {"role": "assistant", "content": "The first concept"},
]
tokens = model.chat(messages, max_new_tokens=4000, enable_thinking=False,
                    add_generation_prompt=False, continue_final_message=True)
```

This also works after `AUTO_REPLY` turns. Those turns remain lazy and finish at
EOS as usual; only the final assistant prefill remains open. The prefix is an
input token sequence, and the model generates its continuation. Combining
`continue_final_message=True` with `add_generation_prompt=True`, or using it
without a non-empty final assistant text message, is an error.

## Concept Trace cache

Concept Trace stores generated token IDs (including thinking and special tokens)
in private JSON files under `$XDG_CACHE_HOME/llmdbg/generations`, defaulting to
`~/.cache/llmdbg/generations`. `LLMDBG_GENERATION_CACHE_DIR` overrides the location.
These files survive debug sessions and VS Code restarts. They contain generated
content in token-ID form; treat them as sensitive data. Writes are atomic with
owner-only file permissions. Invalid or unreadable entries are ignored.

The key hashes the checkpoint revision, tokenizer/template, actual prompt IDs,
image-derived inputs, precision, runtime versions, and greedy generation policy.
It does not hash the pseudo-code file: comments, local names and stepping-only
edits do not invalidate generated text. Targets and vocabulary lenses are also
excluded. Larger generation budgets can extend a matching cached continuation.
Unpinned checkpoints and intervention runs do not use the persistent cache.

Across sessions, cached IDs need only a batched activation capture. Within a
session, one matching token-range capture is retained in memory; changing the
target or lens recomputes scores without generation or model replay. Activations
and KV caches are not persisted to disk. **Regenerate** bypasses both caches and
starts a fresh continuation from the authored prompt (unless interventions pin
the live history). The panel reports which cache was reused.

## License

MIT
