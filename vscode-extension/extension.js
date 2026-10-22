"use strict";

const childProcess = require("child_process");
const fs = require("fs");
const os = require("os");
const path = require("path");
const util = require("util");
const vscode = require("vscode");

const execFile = util.promisify(childProcess.execFile);
const RUNTIME_VERSION = "5";
const LEGACY_PLOT_SNIPPETS_KEY = "llmdbg.plotSnippets";
const GPT2_STARTER_FILE = "GPT-2.llmdbg.py";
const GPT2_STARTER_SOURCE = `from llmdbg import debug, gpt2, nop

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
            nop()  # residual input
            x = attention(x, block)
            nop()  # residual after attention
            x = mlp(x, block)
            nop()  # residual after MLP

        x = model.final_layernorm(x)
        logits = model.vocabulary_projection(x)
        predictions = model.top_tokens(logits, k=10)
        nop()  # Stable breakpoint with predictions available.
`;
function qwen35StarterSource(modelId) {
  return `from llmdbg import debug, nop, qwen35

model = qwen35(
    "${modelId}",
    revision="main",
    device="auto",
    dtype="auto",
)
prompt = "The capital of France is"
tokens = model.tokenize(prompt)

def full_attention(x, block):
    normalized = block.input_layernorm(x)
    mixed = block.full_attention(normalized)
    update = block.output_projection(mixed)
    return x + update

def gated_linear_attention(x, block):
    normalized = block.input_layernorm(x)
    mixed = block.gated_delta(normalized)
    update = block.output_projection(mixed)
    return x + update

def mlp(x, block):
    normalized = block.post_attention_layernorm(x)
    gate = block.gate_projection(normalized)
    gate = block.silu(gate)
    up = block.up_projection(normalized)
    hidden = gate * up
    update = block.down_projection(hidden)
    return x + update

with debug(model, tokens) as run:
    for token in run.tokens:
        x = model.token_embedding(tokens)
        for layer, block in enumerate(model.layers):
            nop()  # residual input
            if block.kind == "full_attention":
                x = full_attention(x, block)
            else:
                x = gated_linear_attention(x, block)
            nop()  # residual after attention
            x = mlp(x, block)
            nop()  # residual after MLP

        x = model.final_layernorm(x)
        logits = model.vocabulary_projection(x)
        predictions = model.top_tokens(logits, k=10)
        nop()  # Stable breakpoint with predictions available.
`;
}
const STARTER_FILES = [
  [GPT2_STARTER_FILE, GPT2_STARTER_SOURCE],
  ["Qwen35-0.8B.llmdbg.py", qwen35StarterSource("Qwen/Qwen3.5-0.8B")],
  ["Qwen35-2B.llmdbg.py", qwen35StarterSource("Qwen/Qwen3.5-2B")],
  ["Qwen35-9B.llmdbg.py", qwen35StarterSource("Qwen/Qwen3.5-9B")],
  ["Qwen35-27B.llmdbg.py", qwen35StarterSource("Qwen/Qwen3.5-27B")]
];
const GPT2_LAUNCH_SOURCE = `${JSON.stringify({
  version: "0.2.0",
  configurations: STARTER_FILES.map(([file]) => ({
    type: "llmdbg",
    request: "launch",
    name: `llmdbg: ${file.replace(".llmdbg.py", "")}`,
    config: `\${workspaceFolder}/${file}`
  }))
}, null, 2)}\n`;
const DEFAULT_PLOT_CODE = `def plot(vector, vector_name, plt, np):
    fig, ax = plt.subplots(figsize=(9, 2.6))
    ax.plot(vector, linewidth=1)
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.set_xlabel("dimension")
    ax.set_ylabel("activation")
    ax.set_title(vector_name)
    fig.tight_layout()
    return fig
`;

function plotFunctionCode(code) {
  if (/^\s*def\s+plot\s*\(/m.test(code)) {
    return code;
  }
  const body = code.trimEnd().split("\n").map(line => `    ${line}`).join("\n");
  return `def plot(vector, vector_name, plt, np):\n${body || "    pass"}\n    return locals().get("fig")\n`;
}

function runtimePython(runtimeRoot) {
  return process.platform === "win32"
    ? path.join(runtimeRoot, "Scripts", "python.exe")
    : path.join(runtimeRoot, "bin", "python");
}

function bootstrapPython() {
  const configured = vscode.workspace
    .getConfiguration("llmdbg")
    .get("bootstrapPythonPath", "")
    .trim();
  if (configured) {
    return configured;
  }
  return process.platform === "win32" ? "python" : "python3";
}

function configuredTorchBackend() {
  return vscode.workspace.getConfiguration("llmdbg").get("torchBackend", "auto");
}

async function detectedCudaVersion() {
  if (!['win32', 'linux'].includes(process.platform)) return undefined;
  try {
    const { stdout } = await execFile("nvidia-smi", [], { timeout: 10000, maxBuffer: 1024 * 1024 });
    const match = stdout.match(/CUDA Version:\s*([0-9]+(?:\.[0-9]+)?)/i);
    return match ? Number.parseFloat(match[1]) : undefined;
  } catch {
    return undefined;
  }
}

async function runtimeRecipe() {
  const configured = configuredTorchBackend();
  if (process.platform === "darwin" && process.arch !== "arm64") {
    throw new Error(
      "Current PyTorch releases do not provide Intel macOS wheels. " +
      "llmdbg supports macOS on Apple Silicon; use a manually managed older runtime on Intel Macs."
    );
  }
  let backend = configured;
  if (backend === "auto") {
    if (process.platform === "darwin") {
      backend = "native";
    } else {
      const cuda = await detectedCudaVersion();
      if (cuda >= 12.8) backend = "cu128";
      else if (cuda >= 12.6) backend = "cu126";
      else if (cuda >= 11.8) backend = "cu118";
      else backend = "cpu";
    }
  }
  const recipes = {
    cu128: { id: "cu128", torch: "2.10.0", index: "https://download.pytorch.org/whl/cu128" },
    cu126: { id: "cu126", torch: "2.10.0", index: "https://download.pytorch.org/whl/cu126" },
    cu118: { id: "cu118", torch: "2.7.1", index: "https://download.pytorch.org/whl/cu118" },
    cpu: { id: "cpu", torch: "2.10.0", index: "https://download.pytorch.org/whl/cpu" },
    native: { id: "native", torch: "2.10.0" }
  };
  const recipe = recipes[backend];
  if (!recipe) throw new Error(`Unsupported llmdbg torch backend: ${backend}`);
  if (process.platform === "darwin" && backend.startsWith("cu")) {
    throw new Error("CUDA PyTorch is not available on macOS. Use the auto or cpu backend.");
  }
  return recipe;
}

function runtimeMarker(recipe) {
  return `${RUNTIME_VERSION}:${process.platform}:${process.arch}:${recipe.id}:${recipe.torch}`;
}

const RUNTIME_PROBE = `import json, platform, torch
cuda = bool(torch.cuda.is_available())
mps = bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
gpu = torch.cuda.get_device_name(0) if cuda else None
memory = torch.cuda.get_device_properties(0).total_memory if cuda else None
print(json.dumps({"platform": platform.platform(), "python": platform.python_version(), "torch": torch.__version__, "built_cuda": torch.version.cuda, "cuda_available": cuda, "mps_available": mps, "gpu": gpu, "gpu_memory": memory}))`;

function compareVersions(left, right) {
  return new Intl.Collator("en", { numeric: true, sensitivity: "base" }).compare(left, right);
}

async function newestInstalledVersion(context) {
  if (context.extensionUri.scheme !== "file") {
    return context.extension.packageJSON.version;
  }
  const extensionParent = path.dirname(context.extensionPath);
  const publisher = context.extension.packageJSON.publisher;
  const name = context.extension.packageJSON.name;
  const extensionId = `${publisher}.${name}`.toLowerCase();
  const directoryPrefix = `${extensionId}-`;
  const versions = [];
  for (const entry of await fs.promises.readdir(extensionParent, { withFileTypes: true })) {
    if (!entry.isDirectory() || !entry.name.toLowerCase().startsWith(directoryPrefix)) {
      continue;
    }
    try {
      const manifest = JSON.parse(
        await fs.promises.readFile(path.join(extensionParent, entry.name, "package.json"), "utf8")
      );
      const manifestId = `${manifest.publisher}.${manifest.name}`.toLowerCase();
      if (manifestId === extensionId && typeof manifest.version === "string") {
        versions.push(manifest.version);
      }
    } catch {
      // Other extensions and incomplete installs are expected in this directory.
    }
  }
  versions.sort(compareVersions);
  return versions.at(-1) || context.extension.packageJSON.version;
}

function uvExecutable() {
  return vscode.workspace.getConfiguration("llmdbg").get("uvPath", "uv").trim() || "uv";
}

function activeLlmdbgFile() {
  const editor = vscode.window.activeTextEditor;
  if (!editor || editor.document.uri.scheme !== "file") {
    return undefined;
  }
  return (editor.document.fileName.endsWith(".llmdbg") || editor.document.fileName.endsWith(".llmdbg.py"))
    ? editor.document.fileName
    : undefined;
}

async function initializeProject() {
  const folders = vscode.workspace.workspaceFolders || [];
  if (!folders.length) {
    void vscode.window.showErrorMessage("Open a folder before creating an llmdbg starter project.");
    return;
  }
  let folder = folders[0];
  if (folders.length > 1) {
    const selected = await vscode.window.showQuickPick(
      folders.map(candidate => ({ label: candidate.name, description: candidate.uri.fsPath, folder: candidate })),
      { placeHolder: "Choose the folder for the llmdbg starter files" }
    );
    if (!selected) return;
    folder = selected.folder;
  }

  const vscodeUri = vscode.Uri.joinPath(folder.uri, ".vscode");
  const launchUri = vscode.Uri.joinPath(vscodeUri, "launch.json");
  const created = [];
  const skipped = [];
  for (const [file, source] of STARTER_FILES) {
    const uri = vscode.Uri.joinPath(folder.uri, file);
    try {
      await vscode.workspace.fs.stat(uri);
      skipped.push(file);
    } catch {
      await vscode.workspace.fs.writeFile(uri, Buffer.from(source, "utf8"));
      created.push(file);
    }
  }
  try {
    await vscode.workspace.fs.stat(launchUri);
    skipped.push(".vscode/launch.json");
  } catch {
    await vscode.workspace.fs.createDirectory(vscodeUri);
    await vscode.workspace.fs.writeFile(launchUri, Buffer.from(GPT2_LAUNCH_SOURCE, "utf8"));
    created.push(".vscode/launch.json");
  }

  const document = await vscode.workspace.openTextDocument(vscode.Uri.joinPath(folder.uri, GPT2_STARTER_FILE));
  await vscode.window.showTextDocument(document);
  const summary = created.length ? `Created ${created.length} llmdbg starter file${created.length === 1 ? "" : "s"}.` : "The llmdbg starter files already exist.";
  const existing = skipped.length ? ` Left existing ${skipped.join(" and ")} unchanged.` : "";
  void vscode.window.showInformationMessage(`${summary}${existing} Set a breakpoint and press F5.`);
}

function expandWorkspaceFolder(value, session) {
  if (!value || !session.workspaceFolder) {
    return value;
  }
  return value.replaceAll("${workspaceFolder}", session.workspaceFolder.uri.fsPath);
}

function vectorPlotHtml(webview) {
  const nonce = `${Date.now()}${Math.random().toString(36).slice(2)}`;
  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src ${webview.cspSource} 'unsafe-inline'; script-src 'nonce-${nonce}'; img-src data:;">
  <style>
    * { box-sizing: border-box; }
    body { margin:0; padding:16px; font:13px/1.4 var(--vscode-font-family); color:var(--vscode-foreground); background:var(--vscode-editor-background); }
    .workbench { display:grid; gap:10px; max-width:1100px; }
    .field { display:grid; grid-template-columns:76px minmax(140px, 1fr) auto; gap:8px; align-items:center; }
    .field > label { color:var(--vscode-foreground); white-space:nowrap; }
    .icon-tools { display:flex; gap:4px; }
    input, select { width:100%; color:var(--vscode-input-foreground); background:var(--vscode-input-background); border:1px solid var(--vscode-input-border, transparent); border-radius:2px; outline:none; }
    input, select { height:28px; padding:3px 7px; }
    input:focus, select:focus { border-color:var(--vscode-focusBorder); }
    input::placeholder { color:var(--vscode-input-placeholderForeground); }
    button { min-height:28px; padding:4px 12px; border:1px solid transparent; border-radius:2px; background:var(--vscode-button-background); color:var(--vscode-button-foreground); font:inherit; cursor:pointer; white-space:nowrap; }
    button:hover { background:var(--vscode-button-hoverBackground); }
    button:focus { outline:1px solid var(--vscode-focusBorder); outline-offset:1px; }
    button.secondary { background:var(--vscode-button-secondaryBackground); color:var(--vscode-button-secondaryForeground); }
    button.secondary:hover { background:var(--vscode-button-secondaryHoverBackground); }
    button.icon { width:28px; min-width:28px; padding:3px; display:grid; place-items:center; background:transparent; color:var(--vscode-foreground); }
    button.icon:hover { background:var(--vscode-toolbar-hoverBackground); }
    button.icon.primary { background:var(--vscode-button-background); color:var(--vscode-button-foreground); }
    button.icon.primary:hover { background:var(--vscode-button-hoverBackground); }
    button.icon svg { width:16px; height:16px; fill:currentColor; }
    #status { min-height:19px; color:var(--vscode-descriptionForeground); }
    #status.error { color:var(--vscode-errorForeground); }
    #preview { padding-top:2px; border-top:1px solid var(--vscode-panel-border); }
    #image { display:block; max-width:100%; margin-top:10px; background:white; border:1px solid var(--vscode-panel-border); }
    #image:not([src]) { display:none; }
    @media (max-width:700px) {
      .field { grid-template-columns:68px minmax(0,1fr) auto; }
      .field { grid-template-columns:68px minmax(0,1fr) auto; }
    }
  </style>
</head>
<body>
  <main class="workbench">
    <div class="field"><label for="snippets">Plot code</label><select id="snippets"></select><div class="icon-tools"><button id="open" class="icon" title="Open plot file" aria-label="Open plot file"><svg viewBox="0 0 16 16"><path d="M12.1 1.9l2 2-8.7 8.7-3 .7.7-3 9-9.4zM2 14h12v1H2v-1z"/></svg></button><button id="new" class="icon" title="New plot file" aria-label="New plot file"><svg viewBox="0 0 16 16"><path d="M7.5 1h1v6.5H15v1H8.5V15h-1V8.5H1v-1h6.5V1z"/></svg></button><button id="delete" class="icon" title="Delete plot file" aria-label="Delete plot file"><svg viewBox="0 0 16 16"><path d="M5 1h6l1 2h3v1h-1l-1 11H3L2 4H1V3h3l1-2zm1 .9L5.5 3h5L10 1.9H6zM3 4l1 10h8l1-10H3z"/></svg></button></div></div>
    <div class="field"><label for="expression">Vector</label><input id="expression" value="residual_stream" placeholder="hidden, qkv, residual_stream, hidden[:256]"><button id="plot" class="icon primary" title="Plot vector" aria-label="Plot vector"><svg viewBox="0 0 16 16"><path d="M4 2.5v11l9-5.5-9-5.5z"/></svg></button></div>
    <section id="preview"><div id="status">Enter a vector expression and run a Matplotlib snippet.</div><img id="image" alt="Matplotlib plot preview"></section>
  </main>
  <script nonce="${nonce}">
    const vscode = acquireVsCodeApi();
    const expression = document.getElementById('expression'), snippets = document.getElementById('snippets');
    const status = document.getElementById('status'), image = document.getElementById('image');
    let items = [];
    window.addEventListener('message', event => {
      if (event.data.type === 'snippets') { const previous=snippets.value; items=event.data.items; snippets.replaceChildren(...items.map(item => { const o=document.createElement('option'); o.value=item.file; o.textContent=item.name; return o; })); snippets.value=items.some(item=>item.file===previous)?previous:(event.data.selected||items[0]?.file||''); }
      if (event.data.type === 'result') { image.src='data:image/png;base64,'+event.data.image; status.className=''; status.textContent=event.data.meta; }
      if (event.data.type === 'error') { status.className='error'; status.textContent='Error: '+event.data.message; }
    });
    document.getElementById('plot').onclick=()=>{ status.className=''; status.textContent='Rendering…'; vscode.postMessage({type:'plot', expression:expression.value, file:snippets.value}); };
    document.getElementById('open').onclick=()=>vscode.postMessage({type:'open', file:snippets.value});
    document.getElementById('new').onclick=()=>vscode.postMessage({type:'new'});
    document.getElementById('delete').onclick=()=>vscode.postMessage({type:'delete', file:snippets.value});
  </script>
</body>
</html>`;
}

function neuronLensHtml(webview) {
  const nonce = `${Date.now()}${Math.random().toString(36).slice(2)}`;
  return `<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src ${webview.cspSource} 'unsafe-inline'; script-src 'nonce-${nonce}';">
<style>
*{box-sizing:border-box} body{margin:0;padding:16px;font:13px/1.4 var(--vscode-font-family);color:var(--vscode-foreground);background:var(--vscode-editor-background)}
header{display:flex;align-items:center;justify-content:space-between;gap:12px;margin-bottom:12px}.actions{display:flex;align-items:center;gap:8px}select{color:var(--vscode-dropdown-foreground);background:var(--vscode-dropdown-background);border:1px solid var(--vscode-dropdown-border);padding:3px 24px 3px 7px;font:inherit}button{padding:4px 12px;border:0;border-radius:2px;background:var(--vscode-button-background);color:var(--vscode-button-foreground);font:inherit;cursor:pointer} button:hover{background:var(--vscode-button-hoverBackground)}.view-picker{position:relative}.view-select{min-width:210px;text-align:left;font-size:15px;font-weight:600}.view-menu{position:absolute;z-index:20;top:calc(100% + 4px);left:0;min-width:300px;padding:6px;background:var(--vscode-menu-background,var(--vscode-dropdown-background));border:1px solid var(--vscode-menu-border,var(--vscode-dropdown-border));box-shadow:0 4px 14px rgba(0,0,0,.35)}.view-menu[hidden]{display:none}.view-option{display:flex;align-items:center;gap:8px;padding:4px 7px;white-space:nowrap;cursor:pointer}.view-option:hover{background:var(--vscode-list-hoverBackground)}.view-option.ica{color:var(--vscode-descriptionForeground)}.view-option input{margin:0}
#context{color:var(--vscode-descriptionForeground);margin-bottom:12px}#context:empty{display:none}.error{color:var(--vscode-errorForeground)}
.waiting-indicator{position:fixed;z-index:100;top:12px;left:50%;transform:translateX(-50%);display:flex;align-items:center;gap:8px;padding:7px 11px;border:1px solid var(--vscode-widget-border,var(--vscode-panel-border));border-radius:5px;background:var(--vscode-editorWidget-background,var(--vscode-sideBar-background));color:var(--vscode-editorWidget-foreground,var(--vscode-foreground));box-shadow:0 3px 14px rgba(0,0,0,.35);pointer-events:none}.waiting-indicator[hidden]{display:none}.waiting-spinner{width:14px;height:14px;border:2px solid color-mix(in srgb,var(--vscode-progressBar-background) 35%,transparent);border-top-color:var(--vscode-progressBar-background);border-radius:50%;animation:lens-spin .75s linear infinite}@keyframes lens-spin{to{transform:rotate(360deg)}}
.ica-menu{position:fixed;z-index:110;min-width:180px;padding:4px;background:var(--vscode-menu-background,var(--vscode-dropdown-background));border:1px solid var(--vscode-menu-border,var(--vscode-dropdown-border));box-shadow:0 4px 14px rgba(0,0,0,.4)}.ica-menu[hidden]{display:none}.ica-menu button{width:100%;text-align:left;background:transparent;color:var(--vscode-menu-foreground,var(--vscode-foreground))}.ica-menu button:hover{background:var(--vscode-menu-selectionBackground,var(--vscode-list-hoverBackground));color:var(--vscode-menu-selectionForeground,var(--vscode-foreground))}.card.intervention-removed{opacity:.3;filter:grayscale(1)}
#neurons{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:8px}.lens-group{grid-column:1/-1;display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:8px;min-width:0}.lens-group[hidden]{display:none}.lens-group.submodule{padding-left:24px;border-left:4px solid color-mix(in srgb,var(--vscode-descriptionForeground) 42%,var(--vscode-editor-background))}.card{position:relative;border:1px solid var(--vscode-panel-border);background:var(--vscode-sideBar-background)}.card.full-width{grid-column:1/-1}.card.saved{opacity:.6;border-color:transparent;background:linear-gradient(rgba(0,0,0,.24),rgba(0,0,0,.24)),var(--vscode-sideBar-background)}.card.saved::after{content:"";position:absolute;inset:-1px;pointer-events:none;background:repeating-linear-gradient(90deg,var(--vscode-descriptionForeground) 0 12px,transparent 12px 21px) top left/100% 1px no-repeat,repeating-linear-gradient(90deg,var(--vscode-descriptionForeground) 0 12px,transparent 12px 21px) bottom left/100% 1px no-repeat,repeating-linear-gradient(0deg,var(--vscode-descriptionForeground) 0 12px,transparent 12px 21px) top left/1px 100% no-repeat,repeating-linear-gradient(0deg,var(--vscode-descriptionForeground) 0 12px,transparent 12px 21px) top right/1px 100% no-repeat}.readouts{display:flex;flex-direction:column}.readout{display:flex;align-items:center;flex-wrap:wrap;gap:5px;padding:5px 8px}.readout[hidden]{display:none}.readout+.readout{border-top:1px solid var(--vscode-panel-border)}.readout-title{font-weight:600;color:var(--vscode-descriptionForeground);white-space:nowrap}.empty{color:var(--vscode-descriptionForeground)}
.target-heading{grid-column:1/-1;display:flex;align-items:center;gap:6px;margin:7px 0 0;font-size:13px;color:var(--vscode-descriptionForeground)}.target-magnitude{font-weight:400;font-variant-numeric:tabular-nums}.residual-triangle{display:block;flex:0 0 auto;width:40px;height:28px;overflow:visible}.component-fan{display:block;flex:0 0 auto;width:40px;height:28px;overflow:visible}.component-triangle{display:block;flex:0 0 auto;width:44px;height:26px;margin-left:auto;overflow:visible}.residual-triangle line,.component-fan line,.component-triangle line{stroke-width:1.8;stroke-linecap:round;vector-effect:non-scaling-stroke}.triangle-input{stroke:#5aa7e8;stroke-dasharray:3 2}.triangle-delta,.fan-result{stroke:#e6a15c}.triangle-output{stroke:#58c5ad}.component-delta{stroke:#9aa0a6}.component-potential{stroke:#58c5ad;stroke-dasharray:3 2}.fan-component{stroke:#9aa0a6;stroke-dasharray:3 2}.residual-triangle circle,.component-fan circle,.component-triangle circle{stroke:none}
.card-head{display:flex;align-items:center;flex-wrap:wrap;gap:5px 10px;padding:7px 9px;border-bottom:1px solid var(--vscode-panel-border)}.neuron-badge{display:inline-flex;align-items:center;justify-content:center;box-sizing:border-box;width:62px;flex:0 0 62px;padding:1px 6px;border:1px solid var(--vscode-badge-background);border-radius:10px;background:linear-gradient(to right,var(--meter-color) 0 var(--meter-fill),var(--vscode-badge-background) var(--meter-fill) 100%);color:var(--vscode-badge-foreground);font-weight:600;font-variant-numeric:tabular-nums;cursor:pointer}.neuron-badge:focus{outline:1px solid var(--vscode-focusBorder);outline-offset:2px}.ica-component-id{color:var(--vscode-foreground);font-family:var(--vscode-editor-font-family);font-weight:600;font-variant-numeric:tabular-nums;white-space:nowrap}.component-details{display:none;align-items:center;flex-wrap:wrap;gap:5px 10px;min-width:0}.card-head.expanded .component-details{display:flex}.stats{color:var(--vscode-descriptionForeground);font-variant-numeric:tabular-nums}.report-toggle{display:flex;align-items:center;gap:5px;color:var(--vscode-descriptionForeground);white-space:nowrap}.report-toggle input{margin:0}.unfade-past .card.saved{opacity:1}.dark-report{color-scheme:dark;--vscode-foreground:#e6edf3;--vscode-descriptionForeground:#9ba7b4;--vscode-editor-background:#0f1115;--vscode-sideBar-background:#171a21;--vscode-panel-border:#343b46;--vscode-dropdown-foreground:#e6edf3;--vscode-dropdown-background:#252a33;--vscode-dropdown-border:#4a5360;--vscode-button-background:#0e639c;--vscode-button-foreground:#fff;--vscode-button-hoverBackground:#1177bb;--vscode-errorForeground:#ff7b72;--vscode-badge-background:#3b414b;--vscode-badge-foreground:#e6edf3;--vscode-focusBorder:#58a6ff}.light-report{color-scheme:light;--vscode-foreground:#364152;--vscode-descriptionForeground:#697586;--vscode-editor-background:#fbfcfd;--vscode-sideBar-background:#f7f9fb;--vscode-panel-border:#cbd3dc;--vscode-dropdown-foreground:#46566a;--vscode-dropdown-background:#f7f9fb;--vscode-dropdown-border:#b8c3cf;--vscode-button-background:#e5edf5;--vscode-button-foreground:#40566f;--vscode-button-hoverBackground:#dbe6f0;--vscode-errorForeground:#9f514b;--vscode-badge-background:#e7ecf1;--vscode-badge-foreground:#52606d;--vscode-focusBorder:#7899b8}.light-report .card{background:#fcfdfe}.light-report .card.saved{opacity:.8;background:#f8fafb}.light-report.unfade-past .card.saved{opacity:1}.light-report .readout-logit .token-chip{color:#526b82;background:#edf3f8;border-color:#b8c9d9}.light-report .readout-j-lens .token-chip{color:#715d80;background:#f3eef7;border-color:#cbbbd8}.light-report .readout-r-lens .token-chip{color:#557565;background:#edf5f1;border-color:#b5cfc2}.light-report .readout-jpp-lens .token-chip{color:#806b4f;background:#f7f1e7;border-color:#d6c19d}.light-report .readout-title,.light-report .stats{color:#697586}.light-report .neuron-badge{color:#52606d;border-color:#b8c3cf;background:linear-gradient(to right,color-mix(in srgb,var(--meter-color) 38%,#fff) 0 var(--meter-fill),#e7ecf1 var(--meter-fill) 100%)}.light-report .lens-group.submodule{border-left-color:#b3bec9}.light-report .triangle-input{stroke:#719fc5}.light-report .triangle-delta,.light-report .fan-result{stroke:#bf895c}.light-report .triangle-output,.light-report .component-potential{stroke:#72a598}.light-report .component-delta,.light-report .fan-component{stroke:#8794a3}
.token-chip{flex:0 0 auto;max-width:100%;padding:3px 7px 2px;border:1px solid var(--vscode-panel-border);border-radius:3px;color:var(--vscode-foreground);font-family:var(--vscode-editor-font-family);white-space:pre-wrap;overflow-wrap:anywhere;text-align:center;cursor:help}.readout-logit .token-chip{background:rgba(24,54,86,.78);border-color:rgba(82,148,214,.55)}.readout-j-lens .token-chip{background:rgba(55,35,78,.78);border-color:rgba(164,111,211,.55)}.readout-r-lens .token-chip{background:rgba(20,64,54,.78);border-color:rgba(63,164,139,.55)}.readout-jpp-lens .token-chip{background:rgba(76,57,18,.82);border-color:rgba(222,176,65,.68)}.token-chip:hover,.token-chip:focus{border-color:var(--vscode-focusBorder);outline:none;filter:brightness(1.22)}.light-report .token-chip{color:#374151!important}
.light-report .view-menu{color:#374151;background:#fbfcfd;border-color:#cbd3dc;box-shadow:0 4px 14px rgba(71,85,105,.18)}.light-report .view-option,.light-report .view-option.ica{color:#374151}.light-report .view-option:hover{background:#e9eef3}.light-report .view-option input,.light-report .report-toggle input{accent-color:#7899b8}
.vector-menu button[hidden],.cosine-result[hidden]{display:none}.vector-menu button:disabled{opacity:.72;cursor:default}.cosine-result{position:fixed;z-index:130;right:20px;top:64px;width:min(390px,calc(100vw - 40px));padding:14px;border:1px solid var(--vscode-panel-border);border-radius:5px;background:var(--vscode-sideBar-background);box-shadow:0 8px 28px rgba(0,0,0,.42)}.cosine-result-head{display:flex;align-items:center;justify-content:space-between;gap:12px}.cosine-result-head strong{font-size:14px}.cosine-close{padding:1px 7px;background:transparent;color:var(--vscode-foreground);font-size:18px}.cosine-values{display:grid;grid-template-columns:auto 1fr;gap:5px 10px;margin-top:9px}.cosine-label{overflow-wrap:anywhere;color:var(--vscode-descriptionForeground)}.cosine-angle{display:block;width:100%;height:105px;margin-top:8px}.cosine-ray-a{stroke:#5aa7e8}.cosine-ray-b{stroke:#e6a15c}.cosine-arc{fill:none;stroke:var(--vscode-descriptionForeground);stroke-dasharray:3 2}.cosine-angle text{fill:var(--vscode-foreground);font:12px var(--vscode-font-family)}.light-report .cosine-result{background:#fbfcfd;box-shadow:0 8px 28px rgba(71,85,105,.22)}
</style></head><body><div id="waiting" class="waiting-indicator" role="status" aria-live="polite"><span class="waiting-spinner" aria-hidden="true"></span><span>Updating Lens…</span></div><div id="icaMenu" class="ica-menu vector-menu" hidden><button id="removeIca" type="button">Intervention: Remove</button><button id="measureA" type="button">Measure Cosine: Vector A</button><button id="measureB" type="button">Measure Cosine: Vector B</button></div><section id="cosineResult" class="cosine-result" hidden><div class="cosine-result-head"><strong>Vector cosine</strong><button id="cosineClose" class="cosine-close" type="button" aria-label="Close">×</button></div><svg id="cosineAngle" class="cosine-angle" viewBox="0 0 340 105" role="img"></svg><div id="cosineValues" class="cosine-values"></div></section><header><div class="view-picker"><button id="viewButton" class="view-select" aria-haspopup="true" aria-expanded="false">7 blocks selected ▾</button><div id="viewMenu" class="view-menu" hidden></div></div><div class="actions"><button id="exportMd">Save</button><button id="refresh">Refresh</button><select id="mode" title="Vocabulary readout"><option value="logit">Logit Lens</option><option value="j-lens" selected>J-Lens</option><option value="r-lens">R-Lens</option><option value="jpp-lens">J++ Lens</option><option value="all">All lenses</option></select></div></header><div id="context">Loading…</div><main id="neurons"></main>
<script nonce="${nonce}">
const vscode=acquireVsCodeApi(),context=document.getElementById('context'),neurons=document.getElementById('neurons'),mode=document.getElementById('mode'),viewButton=document.getElementById('viewButton'),viewMenu=document.getElementById('viewMenu'),waiting=document.getElementById('waiting'),icaMenu=document.getElementById('icaMenu'),removeIca=document.getElementById('removeIca'),measureA=document.getElementById('measureA'),measureB=document.getElementById('measureB'),cosineResult=document.getElementById('cosineResult'),cosineClose=document.getElementById('cosineClose'),cosineAngle=document.getElementById('cosineAngle'),cosineValues=document.getElementById('cosineValues');let snapshot,pendingIntervention,pendingSave,pendingVector;const cosineSelection={a:null,b:null},saved={};
const baseViews=[['residual-pre','Residual Lens · Pre'],['attention-neurons','Attention Output Channel Lens'],['attention-output','Attention Output Lens'],['residual-mid','Residual Lens · Mid'],['neurons','MLP Neuron Lens'],['mlp-output','MLP Output Lens'],['residual-post','Residual Lens · Post']],icaViews=[['ica-residual-pre','Residual ICA Lens · Pre'],['ica-attention-output','Attention Output ICA Lens'],['ica-residual-mid','Residual ICA Lens · Mid'],['ica-mlp-output','MLP Output ICA Lens'],['ica-residual-post','Residual ICA Lens · Post']],orderedViews=[baseViews[0],icaViews[0],baseViews[1],baseViews[2],icaViews[1],baseViews[3],icaViews[2],baseViews[4],baseViews[5],icaViews[3],baseViews[6],icaViews[4]];let selectedViews=new Set(baseViews.map(item=>item[0])),icaSupported=false;
function rebuildViewMenu(){viewMenu.replaceChildren();for(const [value,label] of orderedViews){if(value.startsWith('ica-')&&!icaSupported)continue;const row=document.createElement('label');row.className='view-option'+(value.startsWith('ica-')?' ica':'');const box=document.createElement('input');box.type='checkbox';box.value=value;box.checked=selectedViews.has(value);box.onchange=()=>{box.checked?selectedViews.add(value):selectedViews.delete(value);updateViewButton();requestRefresh()};row.append(box,document.createTextNode(label));viewMenu.append(row)}updateViewButton()}
function updateViewButton(){const count=selectedViews.size;viewButton.textContent=(count===baseViews.length&&!icaSupported?'All':count+' block'+(count===1?'':'s')+' selected')+' ▾'}
function tokenLabel(text){const encoded=JSON.stringify(text);return encoded.slice(1,-1).replace(/ /g,'·')}
function tokensFor(kind,title,items){const box=document.createElement('div');box.className='readout readout-'+kind;const heading=document.createElement('span');heading.className='readout-title';heading.textContent=title+':';box.append(heading);if(!items.length){const empty=document.createElement('span');empty.className='empty';empty.textContent='No compatible artifact is available.';box.append(empty);return box}for(const item of items){const chip=document.createElement('span');chip.className='token-chip';chip.tabIndex=0;chip.textContent=tokenLabel(item.text);chip.title='rank '+item.rank+' · token ID '+item.id+' · logit '+item.logit.toFixed(2);chip.setAttribute('aria-label',item.text+', '+chip.title);box.append(chip)}return box}
function addReadouts(card,source){const readouts=document.createElement('div');readouts.className='readouts';if(mode.value==='logit'||mode.value==='all')readouts.append(tokensFor('logit','Logit Lens',source.logitTokens));if(mode.value==='j-lens'||mode.value==='all')readouts.append(tokensFor('j-lens','J-Lens',source.jLensTokens));if(mode.value==='r-lens'||mode.value==='all')readouts.append(tokensFor('r-lens','R-Lens',source.rLensTokens));if(mode.value==='jpp-lens'||mode.value==='all')readouts.append(tokensFor('jpp-lens',source.jppLensLabel||'J++ Lens',source.jppLensTokens));card.append(readouts)}
function styleCard(card,item){if(item.saved)card.classList.add('saved')}
function cosineOptionText(slot){const selected=cosineSelection[slot];return selected?'Measure Cosine: '+slot.toUpperCase()+' '+selected.label:'Measure Cosine: Vector '+slot.toUpperCase()}
function openVectorMenu(event,vector,intervention){event.preventDefault();pendingVector=vector;pendingIntervention=intervention||null;removeIca.hidden=!pendingIntervention;measureA.textContent=cosineOptionText('a');measureB.textContent=cosineOptionText('b');measureA.disabled=Boolean(cosineSelection.a);measureB.disabled=Boolean(cosineSelection.b);icaMenu.style.left=Math.min(event.clientX,window.innerWidth-330)+'px';icaMenu.style.top=Math.min(event.clientY,window.innerHeight-118)+'px';icaMenu.hidden=false}
function chooseCosine(slot){if(!pendingVector)return;cosineSelection[slot]=pendingVector;icaMenu.hidden=true;pendingVector=null;if(cosineSelection.a&&cosineSelection.b){setBusy(true);vscode.postMessage({type:'measureCosine',vectorA:cosineSelection.a.id,vectorB:cosineSelection.b.id})}}
function showCosine(data){cosineAngle.replaceChildren();const ns='http://www.w3.org/2000/svg',origin=[55,78],length=65,theta=Math.acos(Math.max(-1,Math.min(1,data.cosine))),displayTheta=Math.min(theta,Math.PI*.88),endA=[origin[0]+length,origin[1]],endB=[origin[0]+length*Math.cos(displayTheta),origin[1]-length*Math.sin(displayTheta)];function line(end,className){const node=document.createElementNS(ns,'line');node.setAttribute('x1',origin[0]);node.setAttribute('y1',origin[1]);node.setAttribute('x2',end[0]);node.setAttribute('y2',end[1]);node.setAttribute('stroke-width','3');node.setAttribute('stroke-linecap','round');node.setAttribute('class',className);cosineAngle.append(node)}line(endA,'cosine-ray-a');line(endB,'cosine-ray-b');const radius=30,arc=document.createElementNS(ns,'path');arc.setAttribute('class','cosine-arc');arc.setAttribute('d','M '+(origin[0]+radius)+' '+origin[1]+' A '+radius+' '+radius+' 0 '+(displayTheta>Math.PI?'1':'0')+' 0 '+(origin[0]+radius*Math.cos(displayTheta))+' '+(origin[1]-radius*Math.sin(displayTheta)));cosineAngle.append(arc);const text=document.createElementNS(ns,'text');text.setAttribute('x','145');text.setAttribute('y','52');text.textContent=data.angleDegrees.toFixed(2)+'°';cosineAngle.append(text);cosineValues.replaceChildren();for(const [name,value] of [['A',data.labelA],['B',data.labelB],['cos(A, B)',data.cosine.toFixed(6)],['Angle',data.angleDegrees.toFixed(2)+'°'],['Dimensions',String(data.dimensions)]]){const key=document.createElement('strong'),item=document.createElement('span');key.textContent=name;item.textContent=value;item.className='cosine-label';cosineValues.append(key,item)}cosineResult.hidden=false;setBusy(false)}
function renderSnapshot(item,target=neurons){if(item.error){const card=document.createElement('section');card.className='card full-width';const head=document.createElement('div');head.className='card-head error';head.textContent=item.label+': '+item.error;card.append(head);target.append(card);return}if(item.icaComponents){for(const component of item.icaComponents){const card=document.createElement('section');card.className='card';styleCard(card,item);if(component.removed)card.classList.add('intervention-removed');const head=document.createElement('div');head.className='card-head';const componentId=document.createElement('span');componentId.className='ica-component-id';componentId.textContent='L'+item.transformerLayer+':C'+component.index;componentId.title=component.removed?'This ICA component is removed by an active intervention':'layer '+item.transformerLayer+', ICA component '+component.index+' · rank '+component.rank+' by absolute score'+(item.view.startsWith('ica-residual-')&&!item.saved?' · right-click to intervene':'');if(item.view.startsWith('ica-residual-')&&!item.saved&&!component.removed)componentId.oncontextmenu=event=>{event.preventDefault();pendingIntervention={type:'removeIca',view:item.view,component:component.index,card};icaMenu.style.left=Math.min(event.clientX,window.innerWidth-190)+'px';icaMenu.style.top=Math.min(event.clientY,window.innerHeight-42)+'px';icaMenu.hidden=false};const score=document.createElement('span');score.className='stats';score.textContent='s '+component.score.toFixed(2);score.title='signed ICA component score';const energy=document.createElement('span');energy.className='stats';energy.textContent=(component.energy*100).toFixed(1)+'% energy';energy.title='score² divided by total ICA score energy';head.append(componentId,score,energy);card.append(head);addReadouts(card,component);target.append(card)}return}if(!['neurons','attention-neurons'].includes(item.view)){const card=document.createElement('section');card.className='card full-width';styleCard(card,item);addReadouts(card,item);target.append(card);return}for(const neuron of item.neurons){const card=document.createElement('section');card.className='card';styleCard(card,item);if(neuron.removed)card.classList.add('intervention-removed');const head=document.createElement('div');head.className='card-head';const value=neuron.relativeContribution??neuron.relativeActivation;const title=document.createElement('span');title.className='neuron-badge';title.textContent=String(neuron.index);title.tabIndex=0;title.setAttribute('role','button');title.setAttribute('aria-expanded','false');if(!item.saved&&!neuron.removed)title.oncontextmenu=event=>{event.preventDefault();pendingIntervention={type:'removeComponent',view:item.view,component:neuron.index,card};icaMenu.style.left=Math.min(event.clientX,window.innerWidth-190)+'px';icaMenu.style.top=Math.min(event.clientY,window.innerHeight-42)+'px';icaMenu.hidden=false};const meterValue=Number.isFinite(value)?value:0;title.style.setProperty('--meter-fill',(Math.min(Math.abs(meterValue),10)*10)+'%');title.style.setProperty('--meter-color',meterValue>=0?'rgba(126,55,52,.82)':'rgba(43,73,112,.86)');title.title=neuron.removed?'This component is removed by an active intervention':(item.componentLabel||'neuron')+'['+neuron.index+'] · output-projection rank '+neuron.rank+' · click to show measurements'+(!item.saved?' · right-click to intervene':'')+' · meter '+(value==null?'unavailable':((value>=0?'+':'')+value.toFixed(2)+'×'))+' (full width = 10×)';const projectionImpact=document.createElement('span');projectionImpact.className='stats';projectionImpact.textContent='along Δx '+neuron.outputProjection.toFixed(3);projectionImpact.title='signed projection of this component vector onto the complete module output; components are ranked by this value';const relative=value==null?'':(' ('+(value>=0?'+':'')+value.toFixed(2)+'×)');const impact=document.createElement('span');impact.className='stats';impact.textContent='|Δxᵢ| '+neuron.contributionMagnitude.toFixed(2)+relative;impact.title='individual component magnitude |Δxᵢ| = |a| × |d|';const activation=document.createElement('span');activation.className='stats';activation.textContent='a '+neuron.activation.toFixed(2);activation.title='activation coefficient a = '+neuron.activation.toFixed(6);const norm=document.createElement('span');norm.className='stats';norm.textContent='|d| '+neuron.directionNorm.toFixed(2);norm.title='output direction norm |d| = '+neuron.directionNorm.toFixed(6);const details=document.createElement('span');details.className='component-details';details.append(projectionImpact,impact,activation,norm);const toggle=()=>{const expanded=head.classList.toggle('expanded');title.setAttribute('aria-expanded',String(expanded));title.title=(item.componentLabel||'neuron')+'['+neuron.index+'] · output-projection rank '+neuron.rank+' · click to '+(expanded?'hide':'show')+' measurements'};title.onclick=toggle;title.onkeydown=event=>{if(event.key==='Enter'||event.key===' '){event.preventDefault();toggle()}};head.append(title,details,componentTriangle(neuron.componentGeometry));card.append(head);addReadouts(card,neuron);target.append(card)}}
function residualTriangle(view,geometry){const ns='http://www.w3.org/2000/svg',svg=document.createElementNS(ns,'svg');svg.classList.add('residual-triangle');svg.setAttribute('viewBox','0 0 34 24');svg.setAttribute('role','img');const title=document.createElementNS(ns,'title'),origin=[3,12],inputTip=[27,12],legend='Dashed blue: residual input x · Orange: update Δx · Teal: residual output x + Δx';if(view==='residual-pre'||!geometry){title.textContent=view==='residual-pre'?'Dashed blue: residual input x':legend+' · Update geometry is unavailable';svg.append(title);const line=document.createElementNS(ns,'line');line.classList.add('triangle-input');line.setAttribute('x1',origin[0]);line.setAttribute('y1',origin[1]);line.setAttribute('x2',inputTip[0]);line.setAttribute('y2',inputTip[1]);svg.append(line);const tip=document.createElementNS(ns,'circle');tip.setAttribute('cx',inputTip[0]);tip.setAttribute('cy',inputTip[1]);tip.setAttribute('r','1.7');tip.setAttribute('fill','#5aa7e8');svg.append(tip);return svg}const cosine=geometry.cosine==null?0:Math.max(-1,Math.min(1,geometry.cosine)),ratio=Math.max(0,geometry.deltaRatio||0),unitLength=inputTip[0]-origin[0],points=[origin,inputTip,[inputTip[0]+unitLength*ratio*cosine,inputTip[1]-unitLength*ratio*Math.sqrt(Math.max(0,1-cosine*cosine))]];title.textContent=legend+' · '+geometry.transition+' · cos(x, Δx) '+(geometry.cosine==null?'undefined':geometry.cosine.toFixed(3))+' · |Δx|/|x| '+geometry.deltaRatio.toFixed(3);svg.append(title);function segment(from,to,className,color){const line=document.createElementNS(ns,'line');line.classList.add(className);line.setAttribute('x1',points[from][0]);line.setAttribute('y1',points[from][1]);line.setAttribute('x2',points[to][0]);line.setAttribute('y2',points[to][1]);svg.append(line);const tip=document.createElementNS(ns,'circle');tip.setAttribute('cx',points[to][0]);tip.setAttribute('cy',points[to][1]);tip.setAttribute('r','1.7');tip.setAttribute('fill',color);svg.append(tip)}segment(0,2,'triangle-output','#58c5ad');segment(0,1,'triangle-input','#5aa7e8');segment(1,2,'triangle-delta','#e6a15c');return svg}
function componentTriangle(geometry){const ns='http://www.w3.org/2000/svg',svg=document.createElementNS(ns,'svg');svg.classList.add('component-triangle');svg.setAttribute('viewBox','0 0 44 26');svg.setAttribute('role','img');const origin=[3,14],inputTip=[35,14],cosine=geometry.cosine==null?0:Math.max(-1,Math.min(1,geometry.cosine)),ratio=Math.max(0,geometry.deltaRatio||0),length=inputTip[0]-origin[0],sumTip=[inputTip[0]+length*ratio*cosine,inputTip[1]-length*ratio*Math.sqrt(Math.max(0,1-cosine*cosine))],title=document.createElementNS(ns,'title');title.textContent='Dashed blue: current residual x · Solid gray: this component update vᵢ · Dashed teal: hypothetical residual x + vᵢ · cos(x, vᵢ) '+(geometry.cosine==null?'undefined':geometry.cosine.toFixed(3))+' · |vᵢ|/|x| '+geometry.deltaRatio.toFixed(3)+' · |x + vᵢ| '+geometry.outputNorm.toFixed(3);svg.append(title);function segment(from,to,className,color){const line=document.createElementNS(ns,'line');line.classList.add(className);line.setAttribute('x1',from[0]);line.setAttribute('y1',from[1]);line.setAttribute('x2',to[0]);line.setAttribute('y2',to[1]);svg.append(line);const tip=document.createElementNS(ns,'circle');tip.setAttribute('cx',to[0]);tip.setAttribute('cy',to[1]);tip.setAttribute('r','1.5');tip.setAttribute('fill',color);svg.append(tip)}segment(origin,sumTip,'component-potential','#58c5ad');segment(origin,inputTip,'triangle-input','#5aa7e8');segment(inputTip,sumTip,'component-delta','#9aa0a6');return svg}
function componentFan(geometry){const ns='http://www.w3.org/2000/svg',svg=document.createElementNS(ns,'svg');svg.classList.add('component-fan');svg.setAttribute('role','img');const result=geometry.result,top={x:1,y:0},ends=geometry.components.map(component=>({x:1+component.x,y:component.y,index:component.index})),points=[{x:0,y:0},top,result,...ends],xs=points.map(point=>point.x),ys=points.map(point=>-point.y),span=Math.max(Math.max(...xs)-Math.min(...xs),Math.max(...ys)-Math.min(...ys),.5),pad=span*.18;svg.setAttribute('viewBox',[Math.min(...xs)-pad,Math.min(...ys)-pad,Math.max(...xs)-Math.min(...xs)+2*pad,Math.max(...ys)-Math.min(...ys)+2*pad].join(' '));const title=document.createElementNS(ns,'title'),componentText=geometry.components.map(component=>component.index+' ('+component.x.toFixed(2)+', '+component.y.toFixed(2)+')').join(', ');title.textContent='Dashed gray: top-1 component '+geometry.topIndex+' (unit length), with other top components projected from its tip · Yellow: full output vector · plane coordinates are in top-1 norm units · output ('+result.x.toFixed(2)+', '+result.y.toFixed(2)+') · other components: '+componentText;svg.append(title);function segment(x1,y1,x2,y2,className,color){const line=document.createElementNS(ns,'line');line.classList.add(className);line.setAttribute('x1',x1);line.setAttribute('y1',-y1);line.setAttribute('x2',x2);line.setAttribute('y2',-y2);svg.append(line);const tip=document.createElementNS(ns,'circle');tip.setAttribute('cx',x2);tip.setAttribute('cy',-y2);tip.setAttribute('r',span*.045);tip.setAttribute('fill',color);svg.append(tip)}segment(0,0,result.x,result.y,'fan-result','#e6a15c');segment(0,0,1,0,'fan-component','#9aa0a6');for(const component of geometry.components)segment(1,0,1+component.x,component.y,'fan-component','#9aa0a6');return svg}
function targetHeading(item){const heading=document.createElement('h2');heading.className='target-heading';if(item.view.startsWith('residual-'))heading.append(residualTriangle(item.view,item.residualGeometry));else if((item.view==='attention-output'||item.view==='mlp-output')&&item.componentGeometry)heading.append(componentFan(item.componentGeometry));const labels=Object.fromEntries([...baseViews,...icaViews]),label=document.createElement('span'),layer=Number.isInteger(item.transformerLayer)?' [layer '+item.transformerLayer+']':'';label.textContent=(labels[item.view]||item.view)+layer;heading.append(label);if(Number.isFinite(item.vectorNorm)){const magnitude=document.createElement('span');magnitude.className='target-magnitude';const residual=item.view.includes('residual-');magnitude.textContent=(residual?'|x| ':'|Δx| ')+item.vectorNorm.toFixed(2);magnitude.title=(residual?'residual stream norm |x|':'residual update norm |Δx|')+(item.icaBasis?' · decomposed with this layer’s resid_post ICA basis':'');heading.append(magnitude)}return heading}
function render(){if(!snapshot)return;const light=snapshot.lensTheme==='light';document.body.classList.toggle('light-report',light);document.body.classList.toggle('dark-report',!light);context.className='';context.textContent='';neurons.replaceChildren();for(const item of snapshot.items){const residual=item.view.includes('residual-'),group=document.createElement('section');group.className='lens-group'+(residual?'':' submodule');group.dataset.view=item.view;group.append(targetHeading(item));renderSnapshot(item,group);neurons.append(group)}}
function setBusy(value){waiting.hidden=!value}
function requestRefresh(force=false){setBusy(true);context.className='';context.textContent='Loading…';vscode.postMessage({type:'refresh',mode:mode.value,views:[...selectedViews],force})}
function standalonePage(){const clone=document.documentElement.cloneNode(true),computed=getComputedStyle(document.documentElement),variables=['--vscode-font-family','--vscode-foreground','--vscode-descriptionForeground','--vscode-editor-background','--vscode-sideBar-background','--vscode-panel-border','--vscode-dropdown-foreground','--vscode-dropdown-background','--vscode-dropdown-border','--vscode-button-background','--vscode-button-foreground','--vscode-button-hoverBackground','--vscode-errorForeground','--vscode-badge-background','--vscode-badge-foreground','--vscode-focusBorder'];for(const variable of variables){const value=computed.getPropertyValue(variable);if(value)clone.style.setProperty(variable,value)}const clonedMode=clone.querySelector('#mode');if(clonedMode)for(const option of clonedMode.options)option.toggleAttribute('selected',option.value===mode.value);for(const box of clone.querySelectorAll('#viewMenu input'))box.toggleAttribute('checked',selectedViews.has(box.value));clone.querySelectorAll('script,meta[http-equiv="Content-Security-Policy"],#waiting,#icaMenu,#exportMd,#refresh').forEach(node=>node.remove());const fadeLabel=document.createElement('label');fadeLabel.className='report-toggle';fadeLabel.innerHTML='<input id="fadePast" type="checkbox" checked> Fade past blocks';const themeLabel=document.createElement('label');themeLabel.className='report-toggle';themeLabel.innerHTML='<input id="lightTheme" type="checkbox"> Light theme';const lightTheme=themeLabel.querySelector('input');lightTheme.toggleAttribute('checked',snapshot&&snapshot.reportTheme==='light');clone.querySelector('.actions').prepend(fadeLabel,themeLabel);let title=clone.querySelector('title');if(!title){title=document.createElement('title');clone.querySelector('head').prepend(title)}title.textContent='llmdbg Lens Snapshot';const offline=document.createElement('script');offline.textContent="const button=document.getElementById('viewButton'),menu=document.getElementById('viewMenu'),mode=document.getElementById('mode'),fadePast=document.getElementById('fadePast'),lightTheme=document.getElementById('lightTheme'),groups=[...document.querySelectorAll('.lens-group[data-view]')],boxes=[...menu.querySelectorAll('input')];function updateBlocks(){let count=0;for(const box of boxes){const group=groups.find(item=>item.dataset.view===box.value);box.disabled=!group;if(group){group.hidden=!box.checked;if(box.checked)count++}}button.textContent=count+' block'+(count===1?'':'s')+' selected ▾'}function updateLens(){for(const row of document.querySelectorAll('.readout'))row.hidden=mode.value!=='all'&&!row.classList.contains('readout-'+mode.value)}function updateFade(){document.body.classList.toggle('unfade-past',!fadePast.checked)}function updateTheme(){document.body.classList.toggle('light-report',lightTheme.checked);document.body.classList.toggle('dark-report',!lightTheme.checked)}button.onclick=()=>{menu.hidden=!menu.hidden};for(const box of boxes)box.onchange=updateBlocks;mode.onchange=updateLens;fadePast.onchange=updateFade;lightTheme.onchange=updateTheme;document.addEventListener('click',event=>{if(!event.target.closest('.view-picker'))menu.hidden=true});updateBlocks();updateLens();updateFade();updateTheme();";clone.querySelector('body').append(offline);return '<!doctype html>\\n'+clone.outerHTML}
window.addEventListener('message',event=>{const data=event.data;if(data.type==='busy'){setBusy(Boolean(data.busy));return}if(data.type==='reset'){setBusy(true);for(const key of Object.keys(saved))delete saved[key];snapshot=undefined;neurons.replaceChildren();context.textContent='Loading…';return}if(data.type==='error'){setBusy(false);context.className='error';context.textContent=data.message;neurons.replaceChildren();pendingSave=undefined;return}if(data.type==='lens'){icaSupported=Boolean(data.icaSupported);const jppOption=mode.querySelector('option[value="jpp-lens"]');if(jppOption&&data.jppLensLabel)jppOption.textContent=data.jppLensLabel;if(Array.isArray(data.selectedViews))selectedViews=new Set(data.selectedViews);if(!icaSupported)for(const [value] of icaViews)selectedViews.delete(value);rebuildViewMenu();data.items=data.items.map(item=>{const latestKey=data.mode+'|'+item.view;if(!item.error){saved[latestKey]=item;return item}const old=saved[latestKey];return old?{...old,saved:true}:item});const pre=data.items.find(item=>item.view==='residual-pre'&&!item.error);const activeTransformerLayer=pre&&pre.transformerLayer;if(Number.isInteger(activeTransformerLayer)){data.items=data.items.map(item=>!item.error&&Number.isInteger(item.transformerLayer)&&item.transformerLayer!==activeTransformerLayer?{...item,saved:true}:item)}snapshot=data;render();setBusy(false);if(pendingSave&&data.mode==='all'){const first=snapshot.items.find(item=>!item.error),restore=pendingSave;vscode.postMessage({type:'exportPage',html:standalonePage(),layer:first&&first.transformerLayer});pendingSave=undefined;mode.value=restore.mode;selectedViews=new Set(restore.views);rebuildViewMenu();requestRefresh()}}});mode.onchange=()=>requestRefresh();viewButton.onclick=()=>{const open=viewMenu.hidden;viewMenu.hidden=!open;viewButton.setAttribute('aria-expanded',String(open))};document.addEventListener('click',event=>{if(!event.target.closest('.view-picker')){viewMenu.hidden=true;viewButton.setAttribute('aria-expanded','false')}if(!event.target.closest('.ica-menu'))icaMenu.hidden=true});removeIca.onclick=()=>{if(!pendingIntervention)return;pendingIntervention.card.classList.add('intervention-removed');icaMenu.hidden=true;setBusy(true);vscode.postMessage({type:pendingIntervention.type,view:pendingIntervention.view,component:pendingIntervention.component});pendingIntervention=undefined};document.getElementById('exportMd').onclick=()=>{if(snapshot&&!pendingSave){pendingSave={mode:mode.value,views:[...selectedViews]};mode.value='all';selectedViews=new Set(orderedViews.map(item=>item[0]).filter(value=>icaSupported||!value.startsWith('ica-')));rebuildViewMenu();requestRefresh()}};document.getElementById('refresh').onclick=()=>requestRefresh(true);rebuildViewMenu();requestRefresh();
window.addEventListener('message',event=>{if(event.data.type==='cosineResult')showCosine(event.data.result);if(event.data.type==='reset'){cosineSelection.a=null;cosineSelection.b=null;cosineResult.hidden=true}});
neurons.addEventListener('contextmenu',event=>{if(!snapshot)return;const group=event.target.closest('.lens-group'),heading=event.target.closest('.target-heading'),componentTarget=event.target.closest('.ica-component-id,.neuron-badge');if(!group||(!heading&&!componentTarget))return;const item=snapshot.items.find(candidate=>candidate.view===group.dataset.view);if(!item||item.error)return;let vector,intervention;if(heading&&item.measurementId){vector={id:item.measurementId,label:(item.vectorLabel||item.view)+' [layer '+item.transformerLayer+']'}}else if(componentTarget){const cards=[...group.querySelectorAll(':scope > .card')],card=componentTarget.closest('.card'),index=cards.indexOf(card),component=item.icaComponents?item.icaComponents[index]:item.neurons&&item.neurons[index];if(!component||!component.measurementId)return;const isIca=Boolean(item.icaComponents),kind=isIca?'ICA component':(item.componentLabel||'component');vector={id:component.measurementId,label:isIca?'Residual ICA Lens '+item.view.replace('ica-residual-','')+' L'+item.transformerLayer+':C'+component.index:kind+' L'+item.transformerLayer+':'+component.index};if(!item.saved&&!component.removed){if(isIca&&item.view.startsWith('ica-residual-'))intervention={type:'removeIca',view:item.view,component:component.index,card};if(!isIca)intervention={type:'removeComponent',view:item.view,component:component.index,card}}}if(!vector)return;event.stopImmediatePropagation();openVectorMenu(event,vector,intervention)},true);
measureA.onclick=()=>chooseCosine('a');measureB.onclick=()=>chooseCosine('b');cosineClose.onclick=()=>{cosineResult.hidden=true;cosineSelection.a=null;cosineSelection.b=null;cosineAngle.replaceChildren();cosineValues.replaceChildren()};
</script></body></html>`;
}

async function createRuntime(context, runtimeRoot, recipe) {
  await fs.promises.mkdir(context.globalStorageUri.fsPath, { recursive: true });
  await vscode.window.withProgress(
    {
      location: vscode.ProgressLocation.Notification,
      title: "llmdbg: Creating private debugger environment",
      cancellable: false
    },
    async () => {
      const python = runtimePython(runtimeRoot);
      if (!fs.existsSync(python)) {
        await execFile(
          uvExecutable(),
          ["venv", "--python", bootstrapPython(), runtimeRoot],
          { maxBuffer: 10 * 1024 * 1024 }
        );
      }
      await execFile(
        uvExecutable(),
        [
          "pip", "install",
          "--python", python,
          "--upgrade", "--reinstall",
          ...(recipe.index ? ["--index-url", recipe.index] : []),
          `torch==${recipe.torch}`
        ],
        { maxBuffer: 10 * 1024 * 1024 }
      );
      await execFile(
        uvExecutable(),
        [
          "pip", "install",
          "--python", python,
          "--upgrade",
          "transformers>=5,<6",
          "huggingface-hub>=1.0,<3",
          "safetensors>=0.5,<1",
          "matplotlib>=3.10,<4"
        ],
        { maxBuffer: 10 * 1024 * 1024 }
      );
      await execFile(
        uvExecutable(),
        ["pip", "install", "--python", python, "--upgrade", "--no-deps", "gb10-load-llm==0.1.2"],
        { maxBuffer: 10 * 1024 * 1024 }
      );
      const { stdout } = await execFile(python, ["-c", RUNTIME_PROBE], { maxBuffer: 1024 * 1024 });
      const probe = JSON.parse(stdout.trim());
      if (recipe.id.startsWith("cu") && !probe.cuda_available) {
        throw new Error(
          `Installed torch ${probe.torch} (${recipe.id}), but CUDA verification failed. ` +
          "Check the NVIDIA driver or choose another llmdbg.torchBackend."
        );
      }
      await fs.promises.writeFile(
        path.join(runtimeRoot, ".llmdbg-runtime"),
        `${runtimeMarker(recipe)}\n`
      );
    }
  );
}

function activate(context) {
  const runningVersion = context.extension.packageJSON.version;
  let updateReloadStarted = false;
  async function reloadIfExtensionUpdated() {
    if (updateReloadStarted || !vscode.workspace.getConfiguration("llmdbg").get("autoReloadAfterUpdate", true)) {
      return;
    }
    try {
      const installedVersion = await newestInstalledVersion(context);
      if (compareVersions(installedVersion, runningVersion) > 0) {
        updateReloadStarted = true;
        await vscode.commands.executeCommand("workbench.action.reloadWindow");
      }
    } catch {
      // A transient filesystem error should not interfere with extension activation.
    }
  }
  const updatePoll = setInterval(reloadIfExtensionUpdated, 2000);
  context.subscriptions.push(
    vscode.extensions.onDidChange(reloadIfExtensionUpdated),
    { dispose: () => clearInterval(updatePoll) }
  );

  const runtimeRoot = path.join(context.globalStorageUri.fsPath, "runtime");
  const plotRoot = path.join(context.globalStorageUri.fsPath, "plots");
  let runtimePromise;
  let hostProcess;
  let hostPromise;
  let hostSource;
  let vectorPanel;
  let neuronLensPanel;
  let neuronLensMode = "j-lens";
  let neuronLensViews = [
    "residual-pre", "attention-neurons", "attention-output", "residual-mid",
    "neurons", "mlp-output", "residual-post"
  ];
  let neuronLensIcaInitialized = false;
  let neuronLensRevision = 0;
  const neuronLensItemCache = new Map();
  let residentModel;
  const modelStatus = vscode.window.createStatusBarItem(
    vscode.StatusBarAlignment.Right,
    100
  );
  modelStatus.command = "llmdbg.showModelStatus";
  modelStatus.show();

  function updateModelStatus(model) {
    residentModel = model;
    if (model) {
      const label = model.model.replace(/[\\/]+$/, "").split(/[\\/]/).pop() || model.model;
      modelStatus.text = `$(database) llmdbg: ${label} loaded`;
      modelStatus.tooltip = `${model.model} is resident. Click for details and unload options.`;
    } else {
      modelStatus.text = "$(circle-slash) llmdbg: no model loaded";
      modelStatus.tooltip = "No model is resident in the llmdbg model host.";
    }
  }

  function unloadModel() {
    if (vscode.debug.activeDebugSession?.type === "llmdbg") {
      vscode.window.showWarningMessage("Stop the llmdbg session before unloading its model.");
      return;
    }
    if (!hostProcess || hostProcess.exitCode !== null) {
      updateModelStatus(undefined);
      vscode.window.showInformationMessage("llmdbg: no model is loaded.");
      return;
    }
    const unloaded = residentModel?.model;
    const child = hostProcess;
    hostProcess = undefined;
    hostPromise = undefined;
    hostSource = undefined;
    child.kill();
    updateModelStatus(undefined);
    vscode.window.showInformationMessage(
      unloaded ? `llmdbg: unloaded ${unloaded}.` : "llmdbg: model host stopped."
    );
  }

  function formatBytes(bytes) {
    if (!Number.isFinite(bytes) || bytes <= 0) {
      return "not applicable";
    }
    const units = ["B", "KiB", "MiB", "GiB", "TiB"];
    const exponent = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
    return `${(bytes / (1024 ** exponent)).toFixed(exponent >= 3 ? 2 : 1)} ${units[exponent]}`;
  }

  async function showModelStatus() {
    if (!residentModel) {
      await vscode.window.showInformationMessage(
        "llmdbg: no model is loaded. Start a debug session with F5 to load one."
      );
      return;
    }
    const active = vscode.debug.activeDebugSession?.type === "llmdbg";
    const memory = `model ${formatBytes(residentModel.modelBytes)}; CUDA ${formatBytes(residentModel.cudaAllocatedBytes)} allocated / ${formatBytes(residentModel.cudaReservedBytes)} reserved`;
    const state = active ? " Debugger is active; stop it before unloading." : "";
    const message = `llmdbg: ${residentModel.model}@${residentModel.revision}, ${residentModel.dtype} on ${residentModel.device} — ${memory}.${state}`;
    const choice = active
      ? await vscode.window.showInformationMessage(message)
      : await vscode.window.showInformationMessage(message, "Unload Model");
    if (choice === "Unload Model") {
      unloadModel();
    }
  }

  updateModelStatus(undefined);

  function plotFileName(name) {
    const stem = name.trim().replace(/\.plot\.py$/i, "").replace(/[^A-Za-z0-9_-]+/g, "-").replace(/^-+|-+$/g, "");
    return `${stem || "plot"}.plot.py`;
  }

  async function ensurePlotFiles() {
    await fs.promises.mkdir(plotRoot, { recursive: true });
    let files = (await fs.promises.readdir(plotRoot)).filter(file => file.endsWith(".plot.py"));
    if (!files.length) {
      const legacy = context.globalState.get(LEGACY_PLOT_SNIPPETS_KEY, []);
      const snippets = legacy.length ? legacy : [{ name: "Line plot", code: DEFAULT_PLOT_CODE }];
      for (const snippet of snippets) {
        const file = plotFileName(String(snippet.name || "plot"));
        await fs.promises.writeFile(path.join(plotRoot, file), String(snippet.code || DEFAULT_PLOT_CODE), { flag: "wx" }).catch(error => {
          if (error.code !== "EEXIST") throw error;
        });
      }
      await context.globalState.update(LEGACY_PLOT_SNIPPETS_KEY, undefined);
      files = (await fs.promises.readdir(plotRoot)).filter(file => file.endsWith(".plot.py"));
    }
    for (const file of files) {
      const target = path.join(plotRoot, file);
      const code = await fs.promises.readFile(target, "utf8");
      const normalized = plotFunctionCode(code);
      if (normalized !== code) {
        await fs.promises.writeFile(target, normalized);
      }
    }
    return files.sort().map(file => ({
      file,
      name: file.slice(0, -".plot.py".length).replace(/[-_]+/g, " ").replace(/\b\w/g, character => character.toUpperCase())
    }));
  }

  async function plotPath(file) {
    const files = await ensurePlotFiles();
    if (!files.some(item => item.file === file)) {
      throw new Error("Unknown plot file");
    }
    return path.join(plotRoot, file);
  }

  async function openPlotFile(file) {
    const document = await vscode.workspace.openTextDocument(await plotPath(file));
    await vscode.window.showTextDocument(document, {
      viewColumn: vectorPanel?.viewColumn || vscode.ViewColumn.Beside,
      preview: false
    });
  }

  async function sendPlotSnippets(selected) {
    if (vectorPanel) {
      await vectorPanel.webview.postMessage({ type: "snippets", items: await ensurePlotFiles(), selected });
    }
  }

  async function showVectorPlot() {
    const session = vscode.debug.activeDebugSession;
    if (!session || session.type !== "llmdbg") {
      vscode.window.showErrorMessage("Start and pause an llmdbg session before plotting a vector.");
      return;
    }
    if (!vectorPanel) {
      vectorPanel = vscode.window.createWebviewPanel(
        "llmdbg.vectorPlot",
        "llmdbg Vector Plot",
        vscode.ViewColumn.Beside,
        { enableScripts: true, retainContextWhenHidden: true }
      );
      vectorPanel.webview.html = vectorPlotHtml(vectorPanel.webview);
      vectorPanel.webview.onDidReceiveMessage(async (message) => {
        if (message.type === "plot") {
          const active = vscode.debug.activeDebugSession;
          if (!active || active.type !== "llmdbg") {
            await vectorPanel.webview.postMessage({ type: "error", message: "Start and pause an llmdbg session first." });
            return;
          }
          try {
            const expression = String(message.expression || "residual_stream");
            const code = await fs.promises.readFile(await plotPath(String(message.file || "")), "utf8");
            const result = await active.customRequest("llmdbg/plot", { expression, code });
            await vectorPanel.webview.postMessage({ type: "result", ...result });
          } catch (error) {
            await vectorPanel.webview.postMessage({ type: "error", message: error.message });
          }
        } else if (message.type === "open") {
          await openPlotFile(String(message.file || ""));
        } else if (message.type === "new") {
          const name = await vscode.window.showInputBox({ prompt: "Name the new Matplotlib plot", placeHolder: "activation histogram" });
          if (!name) return;
          const file = plotFileName(name);
          const target = path.join(plotRoot, file);
          await fs.promises.mkdir(plotRoot, { recursive: true });
          await fs.promises.writeFile(target, DEFAULT_PLOT_CODE, { flag: "wx" }).catch(error => {
            if (error.code !== "EEXIST") throw error;
          });
          await sendPlotSnippets(file);
          await openPlotFile(file);
        } else if (message.type === "delete") {
          const file = String(message.file || "");
          const choice = await vscode.window.showWarningMessage(`Delete ${file}?`, { modal: true }, "Move to Trash");
          if (choice !== "Move to Trash") return;
          await vscode.workspace.fs.delete(vscode.Uri.file(await plotPath(file)), { useTrash: true });
          await sendPlotSnippets();
        }
      }, null, context.subscriptions);
      vectorPanel.onDidDispose(() => {
        vectorPanel = undefined;
      }, null, context.subscriptions);
    } else {
      vectorPanel.reveal(vscode.ViewColumn.Beside, true);
    }
    await sendPlotSnippets();
  }

  async function refreshNeuronLens(mode = neuronLensMode, views = neuronLensViews, force = false) {
    if (!neuronLensPanel) return;
    await neuronLensPanel.webview.postMessage({ type: "busy", busy: true });
    const active = vscode.debug.activeDebugSession;
    if (!active || active.type !== "llmdbg") {
      await neuronLensPanel.webview.postMessage({ type: "error", message: "Start and pause an llmdbg session first." });
      return;
    }
    try {
      const support = await active.customRequest("llmdbg/icaSupport", {});
      if (support.supported && !neuronLensIcaInitialized) {
        neuronLensViews = [
          ...neuronLensViews,
          "ica-residual-pre", "ica-attention-output", "ica-residual-mid",
          "ica-mlp-output", "ica-residual-post"
        ];
        views = neuronLensViews;
        neuronLensIcaInitialized = true;
      }
      const targets = new Map([
          ["residual-pre", "Residual Lens · Pre"],
          ["ica-residual-pre", "Residual ICA Lens · Pre"],
          ["attention-neurons", "Attention Output Channel Lens"],
          ["attention-output", "Attention Output Lens"],
          ["ica-attention-output", "Attention Output ICA Lens"],
          ["residual-mid", "Residual Lens · Mid"],
          ["ica-residual-mid", "Residual ICA Lens · Mid"],
          ["neurons", "MLP Neuron Lens"],
          ["mlp-output", "MLP Output Lens"],
          ["ica-mlp-output", "MLP Output ICA Lens"],
          ["residual-post", "Residual Lens · Post"],
          ["ica-residual-post", "Residual ICA Lens · Post"]
      ]);
      const allowedViews = [...targets.keys()].filter(view => views.includes(view) && (support.supported || !view.startsWith("ica-")));
      const items = [];
      for (const targetView of allowedViews) {
        const cacheKey = `${neuronLensRevision}|${mode}|${targetView}`;
        if (!force && neuronLensItemCache.has(cacheKey)) {
          items.push(neuronLensItemCache.get(cacheKey));
          continue;
        }
        try {
          const item = await active.customRequest("llmdbg/neuronLens", { mode, view: targetView });
          neuronLensItemCache.set(cacheKey, item);
          items.push(item);
        } catch (error) {
          const item = { view: targetView, label: targets.get(targetView), error: error.message };
          neuronLensItemCache.set(cacheKey, item);
          items.push(item);
        }
      }
      const lensTheme = vscode.workspace.getConfiguration("llmdbg").get("lensTheme", "dark");
      await neuronLensPanel.webview.postMessage({ type: "lens", mode, items, selectedViews: allowedViews, icaSupported: support.supported, jppLensLabel: support.jppLabel, lensTheme, reportTheme: lensTheme });
    } catch (error) {
      await neuronLensPanel.webview.postMessage({ type: "error", message: error.message });
    }
  }

  async function showNeuronLens(session = vscode.debug.activeDebugSession) {
    if (!session || session.type !== "llmdbg") {
      vscode.window.showErrorMessage("Start and pause an llmdbg session before opening Lens.");
      return;
    }
    if (!neuronLensPanel) {
      neuronLensMode = "j-lens";
      neuronLensViews = [
        "residual-pre", "attention-neurons", "attention-output", "residual-mid",
        "neurons", "mlp-output", "residual-post"
      ];
      neuronLensIcaInitialized = false;
      neuronLensPanel = vscode.window.createWebviewPanel(
        "llmdbg.neuronLens", "llmdbg Lens",
        { viewColumn: vscode.ViewColumn.Beside, preserveFocus: true },
        { enableScripts: true, retainContextWhenHidden: true }
      );
      neuronLensPanel.webview.html = neuronLensHtml(neuronLensPanel.webview);
      neuronLensPanel.webview.onDidReceiveMessage(
        async message => {
          if (message.type === "measureCosine") {
            const active = vscode.debug.activeDebugSession;
            if (!active || active.type !== "llmdbg") {
              await neuronLensPanel.webview.postMessage({ type: "error", message: "Start and pause an llmdbg session first." });
              return;
            }
            try {
              const result = await active.customRequest("llmdbg/measureCosine", {
                vectorA: message.vectorA,
                vectorB: message.vectorB
              });
              await neuronLensPanel.webview.postMessage({ type: "cosineResult", result });
            } catch (error) {
              await neuronLensPanel.webview.postMessage({ type: "error", message: error.message });
            }
            return;
          }
          if (message.type === "removeIca" || message.type === "removeComponent") {
            const active = vscode.debug.activeDebugSession;
            if (!active || active.type !== "llmdbg") {
              await neuronLensPanel.webview.postMessage({ type: "error", message: "Start and pause an llmdbg session first." });
              return;
            }
            try {
              const result = await active.customRequest(
                message.type === "removeIca"
                  ? "llmdbg/removeIcaComponent"
                  : "llmdbg/removeProjectedComponent",
                {
                view: message.view,
                component: message.component
                }
              );
              if (result?.log?.note) {
                vscode.window.setStatusBarMessage(`llmdbg: intervention logged to ${result.log.note}`, 7000);
              }
            } catch (error) {
              await neuronLensPanel.webview.postMessage({ type: "error", message: error.message });
            }
            return;
          }
          if (message.type === "exportPage") {
            const layer = Number.isInteger(message.layer) ? message.layer : "snapshot";
            const folder = session.workspaceFolder?.uri || vscode.workspace.workspaceFolders?.[0]?.uri;
            if (!folder) {
              void vscode.window.showErrorMessage("Open a workspace folder before saving a Lens report.");
              return;
            }
            const timestamp = new Date().toISOString();
            const fileTimestamp = timestamp.replace(/[-:]/g, "").replace(/\.\d{3}Z$/, "Z");
            const reports = vscode.Uri.joinPath(folder, "reports");
            const target = vscode.Uri.joinPath(reports, `${fileTimestamp}-llmdbg-lens-layer${layer}.html`);
            const report = String(message.html || "");
            if (!report.startsWith("<!doctype html>")) throw new Error("Lens page snapshot is invalid.");
            await vscode.workspace.fs.createDirectory(reports);
            await vscode.workspace.fs.writeFile(target, Buffer.from(report, "utf8"));
            void vscode.window.showInformationMessage(`Lens report saved in ./reports/${path.basename(target.fsPath)}`);
            return;
          }
          if (message.type !== "refresh") return;
          if (["logit", "j-lens", "r-lens", "jpp-lens", "all"].includes(message.mode)) {
            neuronLensMode = message.mode;
          }
          if (Array.isArray(message.views)) {
            neuronLensViews = message.views.filter(view => typeof view === "string");
          }
          refreshNeuronLens(neuronLensMode, neuronLensViews, Boolean(message.force));
        }, null, context.subscriptions
      );
      neuronLensPanel.onDidDispose(() => { neuronLensPanel = undefined; }, null, context.subscriptions);
    } else {
      neuronLensPanel.reveal(vscode.ViewColumn.Beside, true);
    }
    await refreshNeuronLens();
  }

  async function ensureRuntime() {
    const recipe = await runtimeRecipe();
    const python = runtimePython(runtimeRoot);
    const marker = path.join(runtimeRoot, ".llmdbg-runtime");
    const ready = fs.existsSync(python)
      && fs.existsSync(marker)
      && fs.readFileSync(marker, "utf8").trim() === runtimeMarker(recipe);
    if (ready) {
      return python;
    }
    if (!runtimePromise) {
      runtimePromise = createRuntime(context, runtimeRoot, recipe)
        .then(() => runtimePython(runtimeRoot))
        .catch((error) => {
          runtimePromise = undefined;
          const message = `llmdbg could not create its private Python environment: ${error.message}`;
          vscode.window.showErrorMessage(message);
          throw error;
        });
    }
    return runtimePromise;
  }

  async function showRuntimeDiagnostics() {
    const output = vscode.window.createOutputChannel("llmdbg Runtime Diagnostics");
    context.subscriptions.push(output);
    output.clear();
    output.appendLine(`OS: ${process.platform} ${process.arch}`);
    output.appendLine(`Runtime: ${runtimeRoot}`);
    output.appendLine(`Configured backend: ${configuredTorchBackend()}`);
    try {
      const recipe = await runtimeRecipe();
      output.appendLine(`Resolved backend: ${recipe.id} (torch ${recipe.torch})`);
      const cuda = await detectedCudaVersion();
      output.appendLine(`NVIDIA driver maximum CUDA: ${cuda ?? "not detected"}`);
      const python = await ensureRuntime();
      const { stdout } = await execFile(python, ["-c", RUNTIME_PROBE], { maxBuffer: 1024 * 1024 });
      const probe = JSON.parse(stdout.trim());
      for (const [key, value] of Object.entries(probe)) output.appendLine(`${key}: ${value}`);
    } catch (error) {
      output.appendLine(`ERROR: ${error.message}`);
    }
    output.show(true);
  }

  async function rebuildRuntime() {
    if (vscode.debug.activeDebugSession?.type === "llmdbg") {
      void vscode.window.showWarningMessage("Stop the llmdbg session before rebuilding its runtime.");
      return;
    }
    if (hostProcess && hostProcess.exitCode === null) hostProcess.kill();
    hostProcess = undefined;
    hostPromise = undefined;
    hostSource = undefined;
    runtimePromise = undefined;
    await fs.promises.rm(runtimeRoot, { recursive: true, force: true });
    await ensureRuntime();
    void vscode.window.showInformationMessage("llmdbg rebuilt and verified its private Python runtime.");
  }

  function adapterSource(session) {
    const configuredSource = session.configuration.adapterSource
      || vscode.workspace.getConfiguration("llmdbg").get("developmentSourcePath", "");
    return configuredSource
      ? path.resolve(expandWorkspaceFolder(configuredSource, session))
      : path.join(context.extensionPath, "bundled");
  }

  async function ensureHost(session) {
    const python = await ensureRuntime();
    const bundledRoot = adapterSource(session);
    if (!fs.existsSync(path.join(bundledRoot, "llmdbg", "host.py"))) {
      throw new Error(`llmdbg adapter source not found under ${bundledRoot}`);
    }
    if (hostProcess && hostProcess.exitCode === null && hostSource === bundledRoot) {
      return hostPromise;
    }
    if (hostProcess && hostProcess.exitCode === null) {
      hostProcess.kill();
    }

    const inheritedPythonPath = process.env.PYTHONPATH;
    const pythonPath = inheritedPythonPath
      ? `${bundledRoot}${path.delimiter}${inheritedPythonPath}`
      : bundledRoot;
    const configuredJppLens = vscode.workspace.getConfiguration("llmdbg").get("jppLensPath", "");
    const jppLensPath = configuredJppLens
      ? path.resolve(expandWorkspaceFolder(configuredJppLens, session))
      : process.env.LLMDBG_JPP_LENS_PATH;
    hostSource = bundledRoot;
    const child = childProcess.spawn(
      python,
      ["-m", "llmdbg.host", "--host", "127.0.0.1", "--port", "0"],
      {
        cwd: session.workspaceFolder ? session.workspaceFolder.uri.fsPath : os.homedir(),
        env: {
          ...process.env,
          PYTHONNOUSERSITE: "1",
          PYTHONPATH: pythonPath,
          ...(jppLensPath ? { LLMDBG_JPP_LENS_PATH: jppLensPath } : {})
        },
        stdio: ["ignore", "pipe", "pipe"]
      }
    );
    hostProcess = child;
    hostPromise = new Promise((resolve, reject) => {
      let stdout = "";
      let stderr = "";
      const timeout = setTimeout(() => reject(new Error("llmdbg model host did not start in time")), 30000);
      child.stdout.on("data", (chunk) => {
        stdout += chunk.toString();
        const match = stdout.match(/LLMDBG_HOST_READY (\d+)/);
        if (match) {
          clearTimeout(timeout);
          resolve(Number(match[1]));
        }
      });
      child.stderr.on("data", (chunk) => {
        stderr += chunk.toString();
      });
      child.once("error", (error) => {
        clearTimeout(timeout);
        reject(error);
      });
      child.once("exit", (code) => {
        clearTimeout(timeout);
        if (hostProcess === child) {
          hostProcess = undefined;
          hostPromise = undefined;
          hostSource = undefined;
          updateModelStatus(undefined);
        }
        reject(new Error(`llmdbg model host exited with code ${code}: ${stderr.trim()}`));
      });
    });
    return hostPromise;
  }

  const factory = {
    async createDebugAdapterDescriptor(session) {
      const port = await ensureHost(session);
      return new vscode.DebugAdapterServer(port, "127.0.0.1");
    }
  };
  context.subscriptions.push(
    vscode.debug.registerDebugAdapterDescriptorFactory("llmdbg", factory),
    vscode.debug.registerDebugConfigurationProvider("llmdbg", {
      provideDebugConfigurations() {
        const config = activeLlmdbgFile();
        return config
          ? [{ type: "llmdbg", request: "launch", name: "llmdbg: Active file", config }]
          : [];
      },
      resolveDebugConfiguration(_folder, config) {
        const activeConfig = activeLlmdbgFile();
        if (!config.type && activeConfig) {
          return {
            type: "llmdbg",
            request: "launch",
            name: "llmdbg: Active file",
            config: activeConfig
          };
        }
        if (config.type === "llmdbg" && !config.config && activeConfig) {
          config.config = activeConfig;
        }
        return config;
      }
    }),
    vscode.commands.registerCommand("llmdbg.plotCurrentVector", showVectorPlot),
    vscode.commands.registerCommand("llmdbg.initializeProject", initializeProject),
    vscode.commands.registerCommand("llmdbg.showRuntimeDiagnostics", showRuntimeDiagnostics),
    vscode.commands.registerCommand("llmdbg.rebuildRuntime", rebuildRuntime),
    vscode.commands.registerCommand("llmdbg.showNeuronLens", () => showNeuronLens()),
    vscode.commands.registerCommand("llmdbg.showModelStatus", showModelStatus),
    vscode.commands.registerCommand("llmdbg.unloadModel", unloadModel),
    vscode.debug.onDidReceiveDebugSessionCustomEvent((event) => {
      if (event.session.type !== "llmdbg") {
        return;
      }
      if (event.event === "llmdbg/modelLoaded") {
        updateModelStatus(event.body);
      } else if (event.event === "llmdbg/computationStarted") {
        if (neuronLensPanel) neuronLensPanel.webview.postMessage({ type: "busy", busy: true });
      } else if (event.event === "llmdbg/vectorChanged") {
        neuronLensRevision += 1;
        neuronLensItemCache.clear();
        refreshNeuronLens();
      }
    }),
    vscode.debug.onDidStartDebugSession((session) => {
      if (session.type !== "llmdbg") return;
      neuronLensRevision += 1;
      neuronLensItemCache.clear();
      if (neuronLensPanel) neuronLensPanel.webview.postMessage({ type: "reset" });
      void showNeuronLens(session);
    }),
    vscode.debug.onDidTerminateDebugSession((session) => {
      if (session.type === "llmdbg" && neuronLensPanel) {
        neuronLensPanel.webview.postMessage({ type: "busy", busy: false });
      }
    }),
    modelStatus
  );
  context.subscriptions.push({
    dispose() {
      if (hostProcess && hostProcess.exitCode === null) {
        hostProcess.kill();
      }
    }
  });
}

function deactivate() {}

module.exports = { activate, deactivate };
