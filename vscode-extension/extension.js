"use strict";

const childProcess = require("child_process");
const fs = require("fs");
const os = require("os");
const path = require("path");
const util = require("util");
const vscode = require("vscode");
const { selectProfile, profileRecipe } = require("./runtime-profiles");

const execFile = util.promisify(childProcess.execFile);
const LEGACY_PLOT_SNIPPETS_KEY = "llmdbg.plotSnippets";
const DEFAULT_J_LENS_SOURCE = "hf://datasets/sida/llmdbg";
const DEFAULT_R_LENS_SOURCE = "hf://datasets/sida/llmdbg";
const DEFAULT_JPP_LENS_SOURCE = "hf://datasets/sida/llmdbg";

function residualToVocabularyLensSettings() {
  const config = vscode.workspace.getConfiguration("llmdbg");
  const entries = [];
  const jSource = String(config.get("jLensSource", DEFAULT_J_LENS_SOURCE) || "").trim();
  const rSource = String(config.get("rLensSource", DEFAULT_R_LENS_SOURCE) || "").trim();
  const jppSource = String(config.get("jppLensSource", DEFAULT_JPP_LENS_SOURCE) || "").trim();
  if (jSource) entries.push({ id: "j", label: "J-Lens", method: "j", source: jSource, indexSection: "j_lenses", enabled: true });
  if (rSource) entries.push({ id: "r", label: "R-Lens", method: "r", source: rSource, indexSection: "r_lenses", enabled: true });
  if (jppSource) entries.push({ id: "jpp", label: "J++ Lens", method: "jpp", source: jppSource, indexSection: "workspace_lenses", enabled: true });
  const custom = config.get("customResidualToVocabularyLenses", []);
  if (Array.isArray(custom)) entries.push(...custom);
  return entries;
}

async function promptLensField(prompt, value, options = {}) {
  return vscode.window.showInputBox({
    prompt,
    value: value === undefined || value === null ? "" : String(value),
    ignoreFocusOut: true,
    validateInput: options.validateInput
  });
}

async function editCustomResidualLens(existing = {}) {
  const id = await promptLensField("Unique Lens ID", existing.id, {
    validateInput: (value) => /^[a-zA-Z0-9._-]+$/.test(value) ? undefined : "Use letters, numbers, dot, underscore, or hyphen."
  });
  if (id === undefined) return undefined;
  const label = await promptLensField("Label shown in the Lens selector", existing.label || id);
  if (label === undefined) return undefined;
  const method = await vscode.window.showQuickPick(
    ["j", "r", "jpp"].map((value) => ({ label: value, picked: value === existing.method })),
    { placeHolder: "Residual-to-Vocabulary Lens method", ignoreFocusOut: true }
  );
  if (!method) return undefined;
  const source = await promptLensField("Artifact source (local folder or hf://...)", existing.source, {
    validateInput: (value) => {
      const candidate = value.trim();
      return /^hf:\/\/(datasets|models)\/[^/]+\/[^/]+/.test(candidate)
        || candidate.startsWith("file://")
        || path.isAbsolute(candidate)
        || candidate.startsWith("./")
        || candidate.startsWith("../")
        || candidate.startsWith("~/")
        ? undefined
        : "Enter a local folder or an hf://datasets/... or hf://models/... source.";
    }
  });
  if (source === undefined) return undefined;
  const indexSection = await promptLensField("Index section (optional)", existing.indexSection);
  if (indexSection === undefined) return undefined;
  const revision = await promptLensField("Pinned revision (optional)", existing.revision);
  if (revision === undefined) return undefined;
  const artifactPath = await promptLensField("Artifact path inside repository (optional)", existing.path);
  if (artifactPath === undefined) return undefined;
  const models = await promptLensField(
    "Compatible model IDs, comma-separated (optional; * means all)",
    Array.isArray(existing.models) ? existing.models.join(", ") : ""
  );
  if (models === undefined) return undefined;
  const result = { id: id.trim(), label: label.trim() || id.trim(), method: method.label, source: source.trim(), enabled: existing.enabled !== false };
  if (indexSection.trim()) result.indexSection = indexSection.trim();
  if (revision.trim()) result.revision = revision.trim();
  if (artifactPath.trim()) result.path = artifactPath.trim();
  const modelList = models.split(",").map((value) => value.trim()).filter(Boolean);
  if (modelList.length) result.models = modelList;
  return result;
}

async function manageResidualToVocabularyLenses() {
  const config = vscode.workspace.getConfiguration("llmdbg");
  let lenses = config.get("customResidualToVocabularyLenses", []);
  lenses = Array.isArray(lenses) ? lenses.map((item) => ({ ...item })) : [];
  while (true) {
    const picked = await vscode.window.showQuickPick([
      { label: "$(add) Add custom Lens", action: "add", alwaysShow: true },
      ...lenses.map((lens, index) => ({
        label: `${lens.enabled === false ? "$(circle-slash)" : "$(check)"} ${lens.label || lens.id}`,
        description: `${lens.method || "?"} · ${lens.source || "source missing"}`,
        detail: lens.id,
        index
      }))
    ], { placeHolder: "Manage custom Residual-to-Vocabulary Lenses", ignoreFocusOut: true });
    if (!picked) return;
    if (picked.action === "add") {
      const added = await editCustomResidualLens();
      if (!added) continue;
      if (lenses.some((lens) => lens.id === added.id)) {
        void vscode.window.showErrorMessage(`A custom Lens with ID ${added.id} already exists.`);
        continue;
      }
      lenses.push(added);
    } else {
      const current = lenses[picked.index];
      const action = await vscode.window.showQuickPick([
        { label: "$(edit) Edit", action: "edit" },
        { label: current.enabled === false ? "$(check) Enable" : "$(circle-slash) Disable", action: "toggle" },
        { label: "$(trash) Remove", action: "remove" }
      ], { placeHolder: current.label || current.id, ignoreFocusOut: true });
      if (!action) continue;
      if (action.action === "edit") {
        const edited = await editCustomResidualLens(current);
        if (!edited) continue;
        if (lenses.some((lens, index) => index !== picked.index && lens.id === edited.id)) {
          void vscode.window.showErrorMessage(`A custom Lens with ID ${edited.id} already exists.`);
          continue;
        }
        lenses[picked.index] = edited;
      } else if (action.action === "toggle") {
        lenses[picked.index] = { ...current, enabled: current.enabled === false };
      } else {
        const answer = await vscode.window.showWarningMessage(`Remove ${current.label || current.id}?`, { modal: true }, "Remove");
        if (answer !== "Remove") continue;
        lenses.splice(picked.index, 1);
      }
    }
    await config.update("customResidualToVocabularyLenses", lenses, vscode.ConfigurationTarget.Global);
  }
}
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
const HISTOGRAM_PLOT_CODE = `def plot(vector, vector_name, plt, np):
    fig, ax = plt.subplots(figsize=(9, 3.6))
    values = np.asarray(vector, dtype=float)
    values = values[np.isfinite(values)]
    ax.hist(values, bins="auto", edgecolor="white", linewidth=0.5)
    ax.set_xlabel("activation")
    ax.set_ylabel("dimensions")
    ax.set_title(vector_name)
    fig.tight_layout()
    return fig
`;
const HEATMAP_PLOT_CODE = `def plot(vector, vector_name, plt, np):
    from matplotlib.ticker import MaxNLocator
    matrix = np.asarray(vector, dtype=float).reshape(original_shape).squeeze()
    if matrix.ndim == 1:
        if heatmap_axis is None:
            raise PlotShapeError("Heatmap squeezed shape is " + str(matrix.shape) + ". Choose where to add a dimension:")
        matrix = np.expand_dims(matrix, axis=heatmap_axis)
    if matrix.ndim != 2:
        raise PlotInputError("Heatmap needs one or two non-singleton dimensions after squeeze; got " + str(matrix.shape) + ". Select a tensor slice with one or two dimensions.")
    finite = matrix[np.isfinite(matrix)]
    if not finite.size:
        raise PlotInputError("Heatmap needs at least one finite value.")
    limit = max(float(np.abs(finite).max()), 1e-12)
    fig, ax = plt.subplots(figsize=(10, 4))
    image = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap="RdBu_r", vmin=-limit, vmax=limit)
    fig.colorbar(image, ax=ax, label="value")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("column")
    ax.set_ylabel("row")
    ax.set_title(vector_name + " · " + str(tuple(original_shape)) + " → " + str(matrix.shape))
    fig.tight_layout()
    fig._llmdbg_plot_shape = matrix.shape
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

async function detectedCudaComputeCapability() {
  if (!["win32", "linux"].includes(process.platform)) return undefined;
  try {
    const { stdout } = await execFile(
      "nvidia-smi",
      ["--query-gpu=compute_cap", "--format=csv,noheader,nounits"],
      { timeout: 10000, maxBuffer: 1024 * 1024 }
    );
    const first = stdout.trim().split(/\r?\n/, 1)[0];
    const capability = Number.parseFloat(first);
    return Number.isFinite(capability) ? capability : undefined;
  } catch {
    return undefined;
  }
}

async function runtimeRecipe(profile = "stable") {
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
      const [cuda, capability] = await Promise.all([
        detectedCudaVersion(), detectedCudaComputeCapability()
      ]);
      // PyTorch's CUDA 12.8 binaries omit Maxwell and Pascal kernels. CUDA
      // 12.6 is the newest official wheel that supports those architectures.
      if (capability >= 12.1 && cuda >= 13.0) backend = "cu130";
      else if (capability >= 12.1) throw new Error("This GPU needs the CUDA 13 runtime profile; update the NVIDIA driver.");
      else if (capability !== undefined && capability < 7.0 && cuda >= 12.6) backend = "cu126";
      else if (cuda >= 12.8) backend = "cu128";
      else if (cuda >= 12.6) backend = "cu126";
      else if (cuda >= 11.8) backend = "cu118";
      else backend = "cpu";
    }
  }
  const recipe = profileRecipe(profile, backend, process.platform, process.arch);
  if (process.platform === "darwin" && backend.startsWith("cu")) {
    throw new Error("CUDA PyTorch is not available on macOS. Use the auto or cpu backend.");
  }
  return recipe;
}

function runtimeMarker(recipe) {
  return recipe.key;
}

const RUNTIME_PROBE = `import json, platform, torch
from importlib.metadata import version
cuda = bool(torch.cuda.is_available())
mps = bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
gpu = torch.cuda.get_device_name(0) if cuda else None
memory = torch.cuda.get_device_properties(0).total_memory if cuda else None
if cuda:
    # Exercise runtime compilation as well as importing torch and allocating memory.
    grid = torch.tensor([[1, 39, 60]], device="cuda", dtype=torch.long)
    assert (grid.prod(-1) // 9).item() == 260
    x = torch.ones((8, 8), device="cuda")
    assert (x @ x).sum().item() == 512
    torch.cuda.synchronize()
print(json.dumps({"platform": platform.platform(), "python": platform.python_version(), "torch": torch.__version__, "torchvision": version("torchvision"), "transformers": version("transformers"), "built_cuda": torch.version.cuda, "cuda_available": cuda, "mps_available": mps, "gpu": gpu, "gpu_memory": memory}))`;

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

  const starterRoot = path.join(__dirname, "bundled", "starters");
  const catalog = JSON.parse(fs.readFileSync(path.join(starterRoot, "starters.json"), "utf8"));
  // Check all packaged sources before creating any workspace files.
  for (const file of catalog.files) fs.accessSync(path.join(starterRoot, file));
  const created = [];
  const skipped = [];
  for (const file of catalog.files) {
    const uri = vscode.Uri.joinPath(folder.uri, file);
    try {
      await vscode.workspace.fs.stat(uri);
      skipped.push(file);
    } catch (error) {
      if (error.code !== "FileNotFound") throw error;
      await vscode.workspace.fs.createDirectory(vscode.Uri.joinPath(uri, ".."));
      await vscode.workspace.fs.copy(vscode.Uri.file(path.join(starterRoot, file)), uri, { overwrite: false });
      created.push(file);
    }
  }
  const document = await vscode.workspace.openTextDocument(vscode.Uri.joinPath(folder.uri, catalog.open));
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
    #status .shape-action { padding:2px 0; margin:3px 16px 0 0; min-height:24px; background:transparent; color:var(--vscode-textLink-foreground); text-decoration:underline; }
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
    <div class="field"><label for="expression">Tensor</label><input id="expression" value="residual_stream" placeholder="query, keys, weights[0, 0]"><button id="plot" class="icon primary" title="Plot tensor" aria-label="Plot tensor"><svg viewBox="0 0 16 16"><path d="M4 2.5v11l9-5.5-9-5.5z"/></svg></button></div>
    <section id="preview"><div id="status">Enter a tensor expression and run a Matplotlib snippet.</div><img id="image" alt="Matplotlib plot preview"></section>
  </main>
  <script nonce="${nonce}">
    const vscode = acquireVsCodeApi();
    const expression = document.getElementById('expression'), snippets = document.getElementById('snippets');
    const status = document.getElementById('status'), image = document.getElementById('image');
    let items = [];
    window.addEventListener('message', event => {
      if (event.data.type === 'snippets') { const previous=snippets.value; items=event.data.items; snippets.replaceChildren(...items.map(item => { const o=document.createElement('option'); o.value=item.file; o.textContent=item.name; return o; })); snippets.value=items.some(item=>item.file===previous)?previous:(event.data.selected||items[0]?.file||''); }
      if (event.data.type === 'result') {
        if(event.data.error){
          image.removeAttribute('src');status.className='error';status.textContent=event.data.error;
          const actions=document.createElement('div');
          for(const choice of event.data.shapeActions||[]){const button=document.createElement('button');button.className='shape-action';button.textContent=choice.label;button.onclick=()=>runPlot(choice.action);actions.append(button)}
          status.append(actions);
        }else{image.src='data:image/png;base64,'+event.data.image; status.className=''; status.textContent=event.data.meta;}
      }
      if (event.data.type === 'error') { status.className='error'; status.textContent='Error: '+event.data.message; }
    });
    function runPlot(shapeAction){status.className='';status.textContent='Rendering…';vscode.postMessage({type:'plot',expression:expression.value,file:snippets.value,shapeAction})}
    document.getElementById('plot').onclick=()=>runPlot();
    expression.addEventListener('keydown', event => { if (event.key === 'Enter' && !event.isComposing) { event.preventDefault(); document.getElementById('plot').click(); } });
    document.getElementById('open').onclick=()=>vscode.postMessage({type:'open', file:snippets.value});
    document.getElementById('new').onclick=()=>vscode.postMessage({type:'new'});
    document.getElementById('delete').onclick=()=>vscode.postMessage({type:'delete', file:snippets.value});
    vscode.postMessage({type:'ready'});
  </script>
</body>
</html>`;
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, character => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
  })[character]);
}

function contributionImagesHtml(images, nonce) {
  if (!images?.length) return "";
  const metadata = JSON.stringify(images).replace(/</g, "\\u003c");
  const cards = images.map((image, index) => `<figure class="contribution-image" data-image-index="${index}"><div class="image-stage"><div class="image-region-frame"><img src="${escapeHtml(image.preview || image.source)}" alt="Visual input ${index + 1}"><div class="image-region" hidden></div></div></div><figcaption><strong>Image ${index + 1}</strong> · tokens ${image.startToken}–${image.endToken}<span class="image-region-label">Click an image token to locate its region.</span><code title="${escapeHtml(image.source)}">${escapeHtml(image.source)}</code></figcaption></figure>`).join("");
  return `<style>
.contribution-images{position:sticky;top:0;z-index:3;display:flex;flex-wrap:wrap;gap:12px;padding:12px 0;background:var(--vscode-editor-background);border-bottom:1px solid var(--vscode-panel-border)}.contribution-image{margin:0;flex:0 1 380px;min-width:180px;max-width:100%;border:1px solid var(--vscode-panel-border);border-radius:5px;overflow:hidden}.image-stage{display:flex;justify-content:center;background:var(--vscode-sideBar-background);padding:8px}.image-region-frame{position:relative;max-width:100%;line-height:0}.image-region-frame img{display:block;max-width:100%;max-height:240px;width:auto;height:auto}.image-region{position:absolute;border:2px solid #ffca28;background:#ffca2840;box-shadow:0 0 0 1px #0008;pointer-events:none}.image-region[hidden]{display:none}.contribution-image figcaption{padding:7px 10px;font-size:12px}.image-region-label{display:block;color:var(--vscode-descriptionForeground)}.contribution-image figcaption code{display:block;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-size:11px;color:var(--vscode-descriptionForeground)}.token-card.is-selected{outline:2px solid var(--vscode-focusBorder);outline-offset:-2px}tr.is-selected{background:var(--vscode-list-activeSelectionBackground)}.token-card.is-visual{cursor:pointer}.token-card.is-visual code{color:var(--vscode-charts-purple,#b180d7)}
</style><section class="contribution-images" aria-label="Visual inputs and selected token region">${cards}</section><script nonce="${nonce}">
(()=>{
const images=${metadata};
function selectRegion(position){
  const index=images.findIndex(image=>position>=image.startToken&&position<=image.endToken);
  if(index<0)return;
  document.querySelectorAll('[data-position]').forEach(item=>item.classList.toggle('is-selected',Number(item.dataset.position)===position));
  document.querySelectorAll('.image-region').forEach(region=>region.hidden=true);
  const image=images[index],card=document.querySelector('[data-image-index="'+index+'"]'),label=card.querySelector('.image-region-label');
  if(image.regionMapping!=='merged-row-major'||image.gridFrames!==1||!image.gridRows||!image.gridColumns){label.textContent='Token '+position+' · exact image grid unavailable; restart the debug run to capture it.';card.scrollIntoView({block:'nearest',behavior:'smooth'});return}
  const offset=position-image.startToken,row=Math.floor(offset/image.gridColumns),column=offset%image.gridColumns,region=card.querySelector('.image-region');
  region.style.left=(100*column/image.gridColumns)+'%';region.style.top=(100*row/image.gridRows)+'%';region.style.width=(100/image.gridColumns)+'%';region.style.height=(100/image.gridRows)+'%';region.hidden=false;
  label.textContent='Token '+position+' · row '+(row+1)+' / '+image.gridRows+' · column '+(column+1)+' / '+image.gridColumns;
  region.title='Spatial region represented by this merged image token';card.scrollIntoView({block:'nearest',behavior:'smooth'});
}
document.addEventListener('click',event=>{const token=event.target.closest('[data-position]');if(token)selectRegion(Number(token.dataset.position))});
document.addEventListener('keydown',event=>{if(event.key!=='Enter'&&event.key!==' ')return;const token=event.target.closest('.token-card[data-position]');if(token){event.preventDefault();selectRegion(Number(token.dataset.position))}});
document.addEventListener('DOMContentLoaded',()=>{for(const token of document.querySelectorAll('.token-card[data-position]'))if(images.some(image=>Number(token.dataset.position)>=image.startToken&&Number(token.dataset.position)<=image.endToken)){token.classList.add('is-visual');token.setAttribute('role','button');token.title+=' · Click to highlight its image region'}});
})();</script>`;
}

function contributionImageOverlaysHtml(images, rows, expanded = false, memory = false) {
  if (!images?.length) return "";
  const byPosition = new Map(rows.map(row => [row.position, row]));
  const maps = images.map((image, imageIndex) => {
    if (image.regionMapping !== "merged-row-major" || image.gridFrames !== 1 || !image.gridRows || !image.gridColumns) {
      return `<p class="note">Image ${imageIndex + 1}: patch grid unavailable. Restart debugging to capture image metadata.</p>`;
    }
    const patches = Array.from({ length: image.gridRows * image.gridColumns }, (_, index) => ({
      index, row: byPosition.get(image.startToken + index)
    }));
    const figures = (memory ? [["contribution", "Signed support for current head output"], ["magnitude", "Projected contribution magnitude"]] : [["contribution", "Contribution magnitude · |weight × value|"], ["value", "Value magnitude · |value|"]]).map(([metric, title]) => {
      const maximum = rows.reduce((largest, row) => Math.max(largest, Math.abs(row[metric] || 0)), 0);
      const imageMaximum = patches.reduce((largest, patch) => Math.max(largest, Math.abs(patch.row?.[metric] || 0)), 0);
      const rects = patches.map(({ index, row }) => {
        const value = row?.[metric] || 0, negative = value < 0;
        const opacity = maximum ? (negative ? .86 : .82) * Math.abs(value) / maximum : 0;
        const imageOpacity = imageMaximum ? (negative ? .86 : .82) * Math.abs(value) / imageMaximum : 0;
        const tooltip = `Token ${image.startToken + index} · ${metric}: ${row ? Number(value).toPrecision(8) : 'not computed'}`;
        return `<rect x="${index % image.gridColumns}" y="${Math.floor(index / image.gridColumns)}" width="1" height="1" fill="${negative ? 'rgb(43,73,112)' : 'rgb(126,55,52)'}" fill-opacity="${opacity}" style="--all-source-opacity:${opacity};--image-only-opacity:${imageOpacity}"><title>${escapeHtml(tooltip)}</title></rect>`;
      }).join("");
      return `<figure class="patch-overlay"><figcaption><span>${title}</span><span class="patch-map-switches"><label class="original-image-toggle"><input type="checkbox" role="switch" checked>Show original image</label><label class="image-only-toggle" title="Off: scale against all source tokens, including text and special tokens. On: scale against this image’s patches only."><input type="checkbox" role="switch">Image-only scaling</label></span></figcaption><div class="patch-overlay-stage"><div class="patch-overlay-frame"><img loading="lazy" src="${escapeHtml(image.preview || image.source)}" alt="Image ${imageIndex + 1}, ${escapeHtml(title)}"><svg viewBox="0 0 ${image.gridColumns} ${image.gridRows}" preserveAspectRatio="none" aria-label="${escapeHtml(title)} patch overlay">${rects}</svg></div></div></figure>`;
    }).join("");
    return `<h3>Image ${imageIndex + 1}</h3><div class="patch-overlays">${figures}</div>`;
  }).join("");
  return `<style>
.patch-overlays{display:grid;grid-template-columns:minmax(0,1fr);gap:16px}.patch-overlay{margin:0;border:1px solid var(--vscode-panel-border);border-radius:4px;overflow:hidden}
.patch-overlay figcaption{display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:6px 12px;padding:8px 10px;font-size:12px;font-weight:600}
.patch-map-switches{display:inline-flex;align-items:center;flex-wrap:wrap;gap:6px 14px}
.original-image-toggle,.image-only-toggle{display:inline-flex;align-items:center;gap:5px;cursor:pointer;font-weight:400;font-size:11px;white-space:nowrap;color:var(--vscode-descriptionForeground)}
.original-image-toggle input,.image-only-toggle input{flex:none;min-width:0;margin:0;accent-color:var(--vscode-focusBorder)}
.patch-overlay:has(.original-image-toggle input:not(:checked)) .patch-overlay-frame img{visibility:hidden}
.patch-overlay-stage{display:flex;justify-content:center;padding:8px;background:var(--vscode-sideBar-background)}.patch-overlay-frame{position:relative;max-width:100%;line-height:0;background:var(--vscode-editor-background)}
.patch-overlay-frame img{display:block;max-width:100%;max-height:420px;width:auto;height:auto}.patch-overlay-frame svg{position:absolute;inset:0;width:100%;height:100%}
.patch-overlay-frame rect{pointer-events:all;fill-opacity:var(--all-source-opacity)}
.patch-overlay:has(.image-only-toggle input:checked) .patch-overlay-frame rect{fill-opacity:var(--image-only-opacity)}
.patch-map-note{margin:8px 0 14px;color:var(--vscode-descriptionForeground);font-size:12px}
</style><details class="image-patch-maps"${expanded ? " open" : ""}><summary>Image patch maps · ${memory ? 'signed support and contribution magnitude' : 'contribution and value'}</summary><p class="patch-map-note">Stronger tint means larger magnitude. Muted red: positive · muted blue: negative. By default, each metric is scaled to its largest absolute value across all source tokens, including text and special tokens. Enable Image-only scaling to compare patches within that image instead. Scales are specific to this ${memory ? 'head' : 'channel'} and query, not calibrated across layers. ${memory ? 'Regions identify write positions, not exclusive pixel origins. ' : ''}Uncomputed patches remain untinted. Hover a patch for its value.</p>${maps}</details>`;
}

function attentionImageMapsHtml(webview, data) {
  return `<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src https: data:; style-src ${webview.cspSource} 'unsafe-inline';"><style>body{margin:0;padding:18px;font:13px/1.5 var(--vscode-font-family);color:var(--vscode-foreground);background:var(--vscode-editor-background)}h1{font-size:18px;margin:0 0 8px}.context{color:var(--vscode-descriptionForeground);margin-bottom:18px}summary{cursor:pointer;font-weight:600}details{border-top:1px solid var(--vscode-panel-border);padding-top:12px}</style></head><body><h1>Channel ${data.channel} · Image patch maps</h1><div class="context">Layer ${data.transformerLayer} · query token ${data.tokenPosition} ${escapeHtml(JSON.stringify(data.token))}<br>Query head ${data.queryHead} · KV head ${data.kvHead} · dimension ${data.dimension} / ${data.headDimension}<br>Snapshot of the selected channel and query token.</div>${contributionImageOverlaysHtml(data.images, data.rows, true)}</body></html>`;
}

function attentionTokenContributionsHtml(webview, data) {
  const memory = data.kind === 'delta-memory';
  const title = memory ? `Head ${data.head} · DeltaNet Memory Contributions` : `Channel ${data.channel} · Token contributions`;
  const headContext = memory ? `Value head ${data.head} · ${data.headDimension} dimensions · saved snapshot` : `Query head ${data.queryHead} · KV head ${data.kvHead} · dimension ${data.dimension} / ${data.headDimension}`;
  const note = memory ? 'Signed support for the current projected head output, including subsequent memory decay and overwriting. Current RMS denominator, query and gate are held fixed. This is an additive attribution, not an attention weight or causal deletion effect. Write positions can contain earlier-token information through convolution and globally contextualized image features. Small reconstruction differences reflect model-dtype rounding.' : `Each card shows one source token's contribution to this channel before output projection${data.outputGate === false ? "" : " and the output gate"}. Positive and negative values can cancel. ${data.outputGate === false ? "This model has no output gate; the sum equals the mixed channel value unless intervened." : "Qwen3.5’s output gate and any interventions affect the actual mixed value."} This is a saved snapshot of the query token shown above.`;
  const nonce = `${Date.now()}${Math.random().toString(36).slice(2)}`;
  const number = value => value == null ? "unavailable" : Number(value).toFixed(2).replace(/^-0\.00$/, "0.00");
  const precise = value => value == null ? "unavailable" : Number(value).toPrecision(8);
  const maximum = data.rows.reduce((largest, row) => Math.max(largest, Math.abs(row.contribution)), 1e-30);
  const rows = data.rows.map(row => {
    const width = 50 * Math.abs(row.contribution) / maximum;
    const positive = row.contribution >= 0;
    return `<tr data-position="${row.position}" data-magnitude="${Math.abs(row.contribution)}" data-search="${escapeHtml(`${row.position} ${JSON.stringify(row.token)}`.toLowerCase())}"><td>${row.position}${row.position === data.tokenPosition ? ' <span class="current">current</span>' : ''}</td><td><code>${escapeHtml(JSON.stringify(row.token))}</code></td><td>${number(memory ? row.beta : row.weight)}</td><td>${number(memory ? row.magnitude : row.value)}</td><td class="contribution ${positive ? 'positive' : 'negative'}">${number(row.contribution)}</td><td class="bar-cell"><div class="bar-track"><span class="zero"></span><span class="bar ${positive ? 'positive' : 'negative'}" style="${positive ? 'left' : 'right'}:50%;width:${width}%"></span></div></td></tr>`;
  }).join("");
  const cards = data.rows.map(row => {
    const sign = row.contribution < 0 ? "negative" : "positive";
    const current = row.position === data.tokenPosition ? " is-current" : "";
    const tooltip = memory ? `Write position ${row.position} · signed support ${precise(row.contribution)} · projected magnitude ${precise(row.magnitude)} · beta ${precise(row.beta)} · decay ${precise(row.decay)}` : `Token ${row.position} · weight ${precise(row.weight)} × value ${precise(row.value)} = ${precise(row.contribution)}`;
    const token = JSON.stringify(row.token).slice(1, -1);
    return `<article class="token-card ${sign}${current}" data-position="${row.position}" data-magnitude="${Math.abs(row.contribution)}" data-search="${escapeHtml(`${row.position} ${JSON.stringify(row.token)}`.toLowerCase())}" title="${escapeHtml(tooltip)}" tabindex="0" aria-label="${escapeHtml(`${tooltip} · ${JSON.stringify(row.token)}`)}"><span class="token-index">${row.position}</span><div class="token-body" style="--fill:${100 * Math.abs(row.contribution) / maximum}%"><code>${escapeHtml(token)}</code></div></article>`;
  }).join("");
  return `<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src https: data:; style-src ${webview.cspSource} 'unsafe-inline'; script-src 'nonce-${nonce}';"><style>
*{box-sizing:border-box}body{margin:0;padding:18px;font:13px/1.5 var(--vscode-font-family);color:var(--vscode-foreground);background:var(--vscode-editor-background)}h1{font-size:18px;margin:0 0 8px}.context,.note{color:var(--vscode-descriptionForeground)}.summary{display:flex;flex-wrap:wrap;gap:10px 24px;padding:12px 0}.summary strong{font-variant-numeric:tabular-nums}.tools{display:flex;gap:10px;margin:14px 0}input,select{padding:5px 8px;color:var(--vscode-input-foreground);background:var(--vscode-input-background);border:1px solid var(--vscode-panel-border)}input{flex:1;min-width:100px}table{width:100%;border-collapse:collapse}th{position:sticky;top:0;background:var(--vscode-editor-background);text-align:left}td,th{padding:7px 10px;border-bottom:1px solid var(--vscode-panel-border)}td:nth-child(n+3){font-variant-numeric:tabular-nums}code{white-space:pre-wrap;overflow-wrap:anywhere;font-family:var(--vscode-editor-font-family)}.current{font-size:11px;color:var(--vscode-descriptionForeground)}.bar-cell{width:24%;min-width:120px}.bar-track{position:relative;height:18px}.zero{position:absolute;left:50%;height:100%;border-left:1px solid var(--vscode-descriptionForeground)}.bar{position:absolute;top:3px;height:12px;border-radius:2px}.bar.positive{background:rgba(126,55,52,.82)}.bar.negative{background:rgba(43,73,112,.86)}.contribution.positive{color:rgba(126,55,52,.82)}.contribution.negative{color:rgba(43,73,112,.86)}tr[hidden]{display:none}.table-scroll{overflow:auto}
.token-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));gap:6px;margin-top:10px}.token-card{display:grid;grid-template-columns:34px minmax(0,1fr);min-height:32px;border:1px solid var(--vscode-panel-border);background:var(--vscode-sideBar-background)}.token-index{display:grid;place-items:center;padding:5px;color:var(--vscode-descriptionForeground);background:var(--vscode-badge-background);font-variant-numeric:tabular-nums}.token-body{position:relative;display:grid;align-content:center;padding:4px 7px;min-width:0;isolation:isolate}.token-body:before{content:"";position:absolute;inset:0 auto 0 0;width:var(--fill);background:var(--fill-color);opacity:.65;z-index:-1}.token-card.positive{--fill-color:rgba(126,55,52,.82)}.token-card.negative{--fill-color:rgba(43,73,112,.86)}.token-value{font-size:11px;color:var(--vscode-descriptionForeground);font-variant-numeric:tabular-nums}.token-card.is-current{border-color:var(--vscode-focusBorder)}.token-card:focus{outline:1px solid var(--vscode-focusBorder);outline-offset:2px}.token-card[hidden]{display:none}.legend{font-size:12px;color:var(--vscode-descriptionForeground)}.legend-positive{color:rgba(126,55,52,.82)}.legend-negative{color:rgba(43,73,112,.86)}details{margin-top:24px;border-top:1px solid var(--vscode-panel-border);padding-top:12px}summary{cursor:pointer;font-weight:600;padding:4px 0}summary:focus{outline:1px solid var(--vscode-focusBorder)}
.token-body code{background:transparent;padding:0;white-space:pre;overflow:hidden;text-overflow:ellipsis}
</style></head><body><h1>${title}</h1><div class="context">Layer ${data.transformerLayer} · query token ${data.tokenPosition} <code>${escapeHtml(JSON.stringify(data.token))}</code><br>${headContext}</div><div class="summary"><span>${memory ? 'Σ signed support' : 'Σ weight × value'} <strong>${number(data.sum)}</strong></span><span>Positive <strong>${number(data.positiveSum)}</strong></span><span>Negative <strong>${number(data.negativeSum)}</strong></span><span>${memory ? 'Actual head output norm' : `Actual mixed[${data.channel}]`} <strong>${number(memory ? data.outputNorm : data.mixedValue)}</strong></span>${memory ? `<span>Reconstruction error ${precise(data.reconstructionError)}</span>` : ''}</div><div class="note">${note}</div>${contributionImagesHtml(data.images, nonce)}<div class="tools"><input id="filter" type="search" placeholder="Filter by source token or position" aria-label="Filter source tokens"><select id="order" aria-label="Sort source tokens"><option value="position">Token order</option><option value="magnitude">${memory ? 'Largest |signed support|' : 'Largest |weight × value|'}</option></select></div><div class="legend"><span class="legend-positive">Muted red: ${memory ? 'supports' : 'positive'}</span> · <span class="legend-negative">Muted blue: ${memory ? 'opposes' : 'negative'}</span> · Text background width shows ${memory ? 'absolute signed support. Hover for support, projected magnitude, beta and decay.' : 'contribution magnitude relative to the largest source-token contribution. Hover for weight, value, and product.'}</div><div id="cards" class="token-grid">${cards}</div><details><summary>Detailed table (${data.rows.length} tokens)</summary><div class="table-scroll"><table><thead><tr><th>${memory ? 'Write position' : 'Position'}</th><th>Source token</th><th>${memory ? 'Beta' : 'Weight'}</th><th>${memory ? 'Projected magnitude' : `Value[${data.dimension}]`}</th><th>${memory ? 'Signed support' : 'Weight × value'}</th><th>Signed contribution</th></tr></thead><tbody id="rows">${rows}</tbody></table></div></details>${contributionImageOverlaysHtml(data.images, data.rows, false, memory)}<script nonce="${nonce}">const body=document.getElementById('rows'),cards=document.getElementById('cards'),rows=[...body.rows],tokens=[...cards.children],filter=document.getElementById('filter'),order=document.getElementById('order');filter.oninput=()=>{const query=filter.value.toLowerCase();for(const item of [...rows,...tokens])item.hidden=!item.dataset.search.includes(query)};order.onchange=()=>{const compare=(a,b)=>order.value==='magnitude'?Number(b.dataset.magnitude)-Number(a.dataset.magnitude):Number(a.dataset.position)-Number(b.dataset.position);body.replaceChildren(...[...rows].sort(compare));cards.replaceChildren(...[...tokens].sort(compare))};</script></body></html>`;
}

const LENS_READOUT_CSS = `
.card{position:relative;border:1px solid var(--vscode-panel-border);background:var(--vscode-sideBar-background)}
.empty{color:var(--vscode-descriptionForeground)}
.readouts{display:flex;flex-direction:column}.readout{display:flex;align-items:center;flex-wrap:wrap;gap:5px;padding:5px 8px}.readout[hidden]{display:none}.readout+.readout{border-top:1px solid var(--vscode-panel-border)}.readout-title{font-weight:600;color:var(--vscode-descriptionForeground);white-space:nowrap}
.token-chip{flex:0 0 auto;max-width:100%;padding:3px 7px 2px;border:1px solid var(--vscode-panel-border);border-radius:3px;color:var(--vscode-foreground);font-family:var(--vscode-editor-font-family);white-space:pre-wrap;overflow-wrap:anywhere;text-align:center;cursor:help}.readout-logit .token-chip{background:rgba(24,54,86,.78);border-color:rgba(82,148,214,.55)}.readout-j-lens .token-chip{background:rgba(55,35,78,.78);border-color:rgba(164,111,211,.55)}.readout-r-lens .token-chip{background:rgba(20,64,54,.78);border-color:rgba(63,164,139,.55)}.readout-jpp-lens .token-chip{background:rgba(76,57,18,.82);border-color:rgba(222,176,65,.68)}.token-chip:hover,.token-chip:focus{border-color:var(--vscode-focusBorder);outline:none;filter:brightness(1.22)}.light-report .token-chip{color:#374151!important}
.light-report .card{background:#fcfdfe}.light-report .readout-logit .token-chip{color:#526b82;background:#edf3f8;border-color:#b8c9d9}.light-report .readout-j-lens .token-chip{color:#715d80;background:#f3eef7;border-color:#cbbbd8}.light-report .readout-r-lens .token-chip{color:#557565;background:#edf5f1;border-color:#b5cfc2}.light-report .readout-jpp-lens .token-chip{color:#806b4f;background:#f7f1e7;border-color:#d6c19d}.light-report .readout-title{color:#697586}
`;

function tokenLabel(text){const encoded=JSON.stringify(text);return encoded.slice(1,-1).replace(/ /g,'·')}
function tokensFor(kind,title,items,unavailable,hint){const box=document.createElement('div');box.className='readout readout-'+kind;const heading=document.createElement('span');heading.className='readout-title';heading.textContent=title+':';if(hint){heading.title=hint;heading.setAttribute('aria-label',title+': '+hint)}box.append(heading);if(!items.length){const empty=document.createElement('span');empty.className='empty';empty.textContent=unavailable||'No compatible artifact is available.';box.append(empty);return box}for(const item of items){const chip=document.createElement('span');chip.className='token-chip';chip.tabIndex=0;chip.textContent=tokenLabel(item.text);chip.title='rank '+item.rank+' · token ID '+item.id+' · logit '+item.logit.toFixed(2);chip.setAttribute('aria-label',item.text+', '+chip.title);box.append(chip)}return box}

function expressionDecoderHtml(webview, theme = "dark") {
  const nonce = `${Date.now()}${Math.random().toString(36).slice(2)}`;
  return `<!DOCTYPE html><html><head><meta charset="UTF-8"><meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src ${webview.cspSource} 'unsafe-inline'; script-src 'nonce-${nonce}';"><style>
*{box-sizing:border-box}body{margin:0;padding:16px;font:13px/1.4 var(--vscode-font-family);color:var(--vscode-foreground);background:var(--vscode-editor-background)}
form{display:flex;gap:8px;flex-wrap:wrap;align-items:center}input{padding:7px;color:var(--vscode-input-foreground);background:var(--vscode-input-background);border:1px solid var(--vscode-input-border,transparent)}#expression{flex:1;min-width:160px}button{padding:7px 12px;background:var(--vscode-button-background);color:var(--vscode-button-foreground);border:0;cursor:pointer}.context,.note{color:var(--vscode-descriptionForeground);line-height:1.5;margin:12px 0}.error{color:var(--vscode-errorForeground)}button:disabled{opacity:.5;cursor:wait}
details.loop{margin-top:10px;color:var(--vscode-descriptionForeground)}.loop-controls{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-top:8px}.loop-controls input{width:75px}.loop-controls label{display:flex;align-items:center;gap:6px}.batch-label{font-size:13px;margin:16px 0 5px}
${LENS_READOUT_CSS}
</style></head><body class="${theme === "light" ? "light-report" : "dark-report"}"><form id="form"><label for="expression">Tensor</label><input id="expression" aria-label="Tensor expression" required><button id="run" type="submit">Decode</button></form>
<details class="loop"><summary>Loop</summary><div class="loop-controls"><label><input id="loopEnabled" type="checkbox" style="width:auto"> Enable loop</label><label>Variable <input id="loopVariable" value="i" aria-label="Loop variable"></label><label>From <input id="loopStart" type="number" step="1" value="0" aria-label="Loop start"></label><label>To (exclusive) <input id="loopStop" type="number" step="1" value="8" aria-label="Loop stop exclusive"></label><span>Up to 64 results</span></div></details>
<p class="note">Uses variables and lens context from the paused line. All enabled vocabulary lenses are shown. Readouts are exploratory—not causal effects or calibrated probabilities.</p><div id="context" class="context">Enter an expression and press Enter or Decode.</div><div id="results"></div>
<script nonce="${nonce}">
const vscode=acquireVsCodeApi(),form=document.getElementById('form'),expression=document.getElementById('expression'),run=document.getElementById('run'),context=document.getElementById('context'),results=document.getElementById('results');let requestId=0;
${tokenLabel.toString()}
${tokensFor.toString()}
form.onsubmit=event=>{event.preventDefault();if(!expression.value.trim())return;const loop=document.getElementById('loopEnabled').checked?{variable:document.getElementById('loopVariable').value.trim(),start:Number(document.getElementById('loopStart').value),stop:Number(document.getElementById('loopStop').value)}:undefined;run.disabled=true;context.className='context';context.textContent='Decoding all lenses… first use may download and load artifacts.';results.replaceChildren();vscode.postMessage({type:'decode',expression:expression.value,loop,requestId:++requestId})};
window.addEventListener('message',event=>{
  const message=event.data;
  if(message.type==='stale'){requestId++;run.disabled=false;context.className='context';context.textContent='Debug position changed. Press Decode to read the current tensors.';results.style.opacity='.45';return}
  if(message.requestId!==requestId)return;
  run.disabled=false;results.style.opacity='1';
  if(message.type==='error'){context.className='error';context.textContent=message.message;return}
  const data=message.data;
  document.body.classList.toggle('light-report',data.lensTheme==='light');
  document.body.classList.toggle('dark-report',data.lensTheme!=='light');
  const describe=item=>item.expression+' · shape ['+item.shape.join(', ')+'] · norm '+item.norm.toFixed(2)+' · token '+item.tokenPosition+' '+JSON.stringify(item.token)+' · layer '+item.layer+(item.sourceLine?' · line '+item.sourceLine:'');
  context.textContent=data.results?data.expression+' · '+data.results.length+' results · '+data.loop.variable+' in ['+data.loop.start+', '+data.loop.stop+')':describe(data);
  results.replaceChildren();
  for(const item of data.results||[data]){
  if(data.results){const heading=document.createElement('h3');heading.className='batch-label';heading.textContent=data.loop.variable+' = '+item.loopValue;results.append(heading);const meta=document.createElement('div');meta.className=item.error?'error':'context';meta.textContent=item.error||describe(item);results.append(meta)}
  if(item.error)continue;
  const card=document.createElement('section'),readouts=document.createElement('div');card.className='card full-width';readouts.className='readouts';
  for(const row of item.rows){
    const kind=row.method==='logit'?'logit':row.method+'-lens';
    const items=(row.tokens||[]).map((item,index)=>({id:item.id,text:item.text,logit:item.score,rank:index+1}));
    const hint=row.metadata&&row.metadata.fallback?'Fallback map: '+JSON.stringify(row.metadata):undefined;
    readouts.append(tokensFor(kind,row.label,items,row.error,hint));
  }
  card.append(readouts);results.append(card);
  }
});
</script></body></html>`;
}

function promptTokensHtml(webview, data) {
  const nonce = `${Date.now()}${Math.random().toString(36).slice(2)}`;
  const current = Number(data.currentToken);
  const imageSpans = data.images || [];
  const tokenItems = [];
  const tokenValues = data.tokens || [];
  for (let index = 0; index < tokenValues.length;) {
    const imageIndex = imageSpans.findIndex(image => index === image.startToken && Number.isInteger(image.endToken));
    if (imageIndex >= 0) {
      const image = imageSpans[imageIndex];
      const active = current >= image.startToken && current <= image.endToken ? " current" : "";
      const currentLabel = active ? ` · current ${current}` : "";
      const search = `${image.startToken} ${image.endToken} image image_pad visual ${image.tokenCount}`.toLowerCase();
      tokenItems.push(`<article class="token visual visual-group${active}" data-group="${imageIndex}" role="button" tabindex="0" aria-expanded="false" data-search="${escapeHtml(search)}" title="Click to expand or collapse Image ${imageIndex + 1} patch tokens"><span class="index range">${image.startToken}–${image.endToken}</span><code>${escapeHtml(String(tokenValues[index]))} <span class="repeat">× ${image.tokenCount}${currentLabel}</span></code></article>`);
      for(let position=image.startToken;position<=image.endToken&&position<tokenValues.length;position++){
        const visible=JSON.stringify(String(tokenValues[position])).slice(1,-1);
        tokenItems.push(`<article class="token visual patch${position===current?' current':''}" data-patch-group="${imageIndex}" data-position="${position}" data-search="${escapeHtml(`${position} ${visible}`.toLowerCase())}" role="button" tabindex="0" title="Token ${position} · Click to locate its image patch" hidden><span class="index">${position}</span><code>${escapeHtml(visible)}</code></article>`);
      }
      index = image.endToken + 1;
      continue;
    }
    const token = tokenValues[index];
    const visible = JSON.stringify(String(token)).slice(1, -1);
    const active = index === current ? " current" : "";
    tokenItems.push(`<article class="token${active}" data-search="${escapeHtml(`${index} ${visible}`.toLowerCase())}"><span class="index">${index}</span><code>${escapeHtml(visible)}</code></article>`);
    index += 1;
  }
  const tokens = tokenItems.join("");
  const images = (data.images || []).map((image, index) => {
    const span = Number.isInteger(image.startToken)
      ? `Tokens ${image.startToken}–${image.endToken} · ${image.tokenCount} visual tokens`
      : "Visual-token span unavailable";
    return `<figure class="image-card" data-image-index="${index}"><button class="image-preview" type="button" title="Expand image"><span class="patch-image-frame"><img src="${escapeHtml(image.preview || image.source)}" alt="Visual input ${index + 1}"><span class="patch-region" hidden></span></span></button><figcaption><div><strong>Image ${index + 1}</strong><span class="image-span">${escapeHtml(span)}</span></div><span class="patch-label">Click a patch token to locate its region.</span><code class="image-source" title="${escapeHtml(image.source)}">${escapeHtml(image.source)}</code></figcaption></figure>`;
  }).join("");
  const visualInputs = images ? `<h2>Visual inputs</h2><section class="images">${images}</section>` : "";
  const patchStyles = `<style>.patch-image-frame{display:block;position:relative;max-width:100%;line-height:0}.patch-region{position:absolute;border:2px solid #ffca28;background:#ffca2840;box-shadow:0 0 0 1px #0008;pointer-events:none}.patch-region[hidden]{display:none}.patch-label{font-size:12px;color:var(--vscode-descriptionForeground)}.visual-group,.patch{cursor:pointer}.token.is-selected{outline:2px solid #ffca28;outline-offset:-2px}.token[role=button]:focus-visible{outline:2px solid var(--vscode-focusBorder);outline-offset:1px}</style>`;
  const patchScript = `
const images=${JSON.stringify(data.images || []).replace(/</g,'\\u003c')},expanded=new Set();
function applyFilter(){const query=filter.value.trim().toLowerCase();for(const item of items){const patch=item.dataset.patchGroup;item.hidden=!item.dataset.search.includes(query)||(patch!==undefined&&!expanded.has(patch)&&!query)}}
filter.oninput=applyFilter;
function activateToken(token){
  if(token.dataset.group!==undefined){const key=token.dataset.group;if(expanded.has(key))expanded.delete(key);else expanded.add(key);token.setAttribute('aria-expanded',String(expanded.has(key)));applyFilter();return}
  if(token.dataset.position===undefined)return;
  const position=Number(token.dataset.position),image=images[Number(token.dataset.patchGroup)];
  const card=document.querySelector('[data-image-index="'+token.dataset.patchGroup+'"]');if(!image||!card)return;
  for(const item of items)item.classList.toggle('is-selected',item===token);
  for(const region of document.querySelectorAll('.patch-region'))region.hidden=true;
  const label=card.querySelector('.patch-label');
  if(image.regionMapping!=='merged-row-major'||image.gridFrames!==1||!image.gridRows||!image.gridColumns){label.textContent='Token '+position+' · exact image grid unavailable; restart the debug run to capture it.';card.scrollIntoView({block:'nearest'});return}
  const offset=position-image.startToken,row=Math.floor(offset/image.gridColumns),column=offset%image.gridColumns;
  if(offset<0||row>=image.gridRows){label.textContent='Token '+position+' · image grid mapping unavailable.';return}
  const region=card.querySelector('.patch-region');region.style.left=(100*column/image.gridColumns)+'%';region.style.top=(100*row/image.gridRows)+'%';region.style.width=(100/image.gridColumns)+'%';region.style.height=(100/image.gridRows)+'%';region.hidden=false;
  label.textContent='Token '+position+' · row '+(row+1)+' / '+image.gridRows+' · column '+(column+1)+' / '+image.gridColumns;
  card.scrollIntoView({block:'nearest',behavior:'smooth'});
}
document.querySelector('.tokens').addEventListener('click',event=>{const token=event.target.closest('.token[role=button]');if(token)activateToken(token)});
document.querySelector('.tokens').addEventListener('keydown',event=>{if(event.key!=='Enter'&&event.key!==' ')return;const token=event.target.closest('.token[role=button]');if(token){event.preventDefault();activateToken(token)}});
`;
  return `<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src https: data:; style-src ${webview.cspSource} 'unsafe-inline'; script-src 'nonce-${nonce}';"><style>
*{box-sizing:border-box}body{margin:0;padding:18px;font:13px/1.45 var(--vscode-font-family);color:var(--vscode-foreground);background:var(--vscode-editor-background)}header{position:sticky;top:0;z-index:2;display:flex;align-items:center;gap:12px;padding:0 0 12px;background:var(--vscode-editor-background)}h1{margin:0;font-size:17px}input{margin-left:auto;width:min(360px,45vw);height:28px;padding:4px 8px;color:var(--vscode-input-foreground);background:var(--vscode-input-background);border:1px solid var(--vscode-input-border,transparent);outline:none}input:focus{border-color:var(--vscode-focusBorder)}h2{margin:18px 0 8px;font-size:14px}pre{margin:0;padding:12px;white-space:pre-wrap;overflow-wrap:anywhere;border:1px solid var(--vscode-panel-border);background:var(--vscode-textCodeBlock-background);font-family:var(--vscode-editor-font-family)}.summary{color:var(--vscode-descriptionForeground);white-space:nowrap}.images{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,360px));gap:12px}.image-card{margin:0;overflow:hidden;border:1px solid var(--vscode-panel-border);border-radius:6px;background:var(--vscode-sideBar-background)}.image-preview{display:grid;place-items:center;width:100%;height:190px;padding:0;border:0;background:repeating-conic-gradient(color-mix(in srgb,var(--vscode-editor-background) 80%,transparent) 0 25%,transparent 0 50%) 50%/18px 18px;cursor:zoom-in}.image-preview img{display:block;max-width:100%;max-height:190px;object-fit:contain}.image-card.expanded{grid-column:1/-1;max-width:min(900px,100%)}.image-card.expanded .image-preview{height:min(65vh,680px);cursor:zoom-out}.image-card.expanded img{max-height:min(65vh,680px)}figcaption{display:grid;gap:5px;padding:9px 11px;border-top:1px solid var(--vscode-panel-border)}figcaption>div{display:flex;align-items:center;justify-content:space-between;gap:10px}.image-span{padding:2px 6px;border-radius:10px;color:var(--vscode-badge-foreground);background:var(--vscode-badge-background);font-size:11px;white-space:nowrap}.image-source{display:block;padding:0;overflow:hidden;color:var(--vscode-descriptionForeground);text-overflow:ellipsis;white-space:nowrap}.tokens{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:7px}.token{display:grid;grid-template-columns:auto minmax(0,1fr);min-height:38px;border:1px solid var(--vscode-panel-border);background:var(--vscode-sideBar-background)}.token.visual{border-color:color-mix(in srgb,var(--vscode-charts-purple,#b180d7) 65%,var(--vscode-panel-border))}.token.visual code{color:var(--vscode-charts-purple,#c69ae8)}.token.visual-group{grid-column:span 2;min-width:260px}.token.visual-group .index.range{min-width:76px}.token.visual-group .repeat{color:var(--vscode-descriptionForeground);font-family:var(--vscode-font-family);font-size:11px}.token[hidden]{display:none}.token.current{border:2px solid var(--vscode-focusBorder);background:var(--vscode-list-activeSelectionBackground)}.index{display:grid;place-items:center;min-width:38px;padding:5px;color:var(--vscode-descriptionForeground);background:var(--vscode-badge-background);font-variant-numeric:tabular-nums}.token.current .index{color:var(--vscode-list-activeSelectionForeground)}code{align-self:center;padding:6px 8px;white-space:pre-wrap;overflow-wrap:anywhere;font-family:var(--vscode-editor-font-family);color:var(--vscode-foreground)}
</style>${patchStyles}</head><body><header><h1>Prompt tokens</h1><span class="summary">${(data.tokens || []).length} tokens · current ${Number.isInteger(current) ? current : "—"}</span><input id="filter" type="search" placeholder="Filter by token index or text" aria-label="Filter tokens"></header>${visualInputs}<h2>Rendered prompt</h2><pre>${escapeHtml(data.prompt || "")}</pre><h2>Tokenizer split</h2><main class="tokens">${tokens}</main><script nonce="${nonce}">const filter=document.getElementById('filter'),items=[...document.querySelectorAll('.token')];${patchScript}for(const button of document.querySelectorAll('.image-preview'))button.onclick=()=>button.closest('.image-card').classList.toggle('expanded');document.querySelector('.token.current')?.scrollIntoView({block:'center'});</script></body></html>`;
}

async function promptImagePreviews(images) {
  const mimeTypes = new Map([
    [".png", "image/png"], [".jpg", "image/jpeg"], [".jpeg", "image/jpeg"],
    [".gif", "image/gif"], [".webp", "image/webp"], [".bmp", "image/bmp"]
  ]);
  return Promise.all((images || []).map(async image => {
    if (/^https:\/\//i.test(image.source) || /^data:image\//i.test(image.source)) {
      return { ...image, preview: image.source };
    }
    try {
      const bytes = await fs.promises.readFile(image.source);
      const mime = mimeTypes.get(path.extname(image.source).toLowerCase()) || "image/png";
      return { ...image, preview: `data:${mime};base64,${bytes.toString("base64")}` };
    } catch {
      return image;
    }
  }));
}

function neuronLensHtml(webview) {
  const nonce = `${Date.now()}${Math.random().toString(36).slice(2)}`;
  return `<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src ${webview.cspSource} 'unsafe-inline'; script-src 'nonce-${nonce}';">
<style>
${LENS_READOUT_CSS}
*{box-sizing:border-box} body{margin:0;padding:16px;font:13px/1.4 var(--vscode-font-family);color:var(--vscode-foreground);background:var(--vscode-editor-background)}
header{display:flex;align-items:center;justify-content:space-between;gap:12px;margin-bottom:12px}.header-context,.actions{display:flex;align-items:center;gap:8px}.current-token{padding:3px 8px;border:1px solid var(--vscode-panel-border);border-radius:3px;color:var(--vscode-descriptionForeground);font-weight:600;font-variant-numeric:tabular-nums;white-space:nowrap}button{padding:4px 12px;border:0;border-radius:2px;background:var(--vscode-button-background);color:var(--vscode-button-foreground);font:inherit;cursor:pointer} button:hover{background:var(--vscode-button-hoverBackground)}.view-picker,.lens-picker{position:relative}.view-select{min-width:210px;text-align:left;font-size:15px;font-weight:600}.lens-select{min-width:170px;text-align:left;color:var(--vscode-dropdown-foreground);background:var(--vscode-dropdown-background);border:1px solid var(--vscode-dropdown-border)}.view-menu,.lens-menu{position:absolute;z-index:20;top:calc(100% + 4px);padding:6px;background:var(--vscode-menu-background,var(--vscode-dropdown-background));border:1px solid var(--vscode-menu-border,var(--vscode-dropdown-border));box-shadow:0 4px 14px rgba(0,0,0,.35)}.view-menu{left:0;min-width:300px}.lens-menu{right:0;min-width:230px}.view-menu[hidden],.lens-menu[hidden]{display:none}.view-option,.lens-option{display:flex;align-items:center;gap:8px;padding:4px 7px;white-space:nowrap;cursor:pointer}.view-option:hover,.lens-option:hover{background:var(--vscode-list-hoverBackground)}.view-option.ica{color:var(--vscode-descriptionForeground)}.view-option input,.lens-option input{margin:0}.select-all{margin-bottom:4px;padding-bottom:7px;border-bottom:1px solid var(--vscode-menu-separatorBackground,var(--vscode-panel-border));font-weight:600}
#context{color:var(--vscode-descriptionForeground);margin-bottom:12px}#context:empty{display:none}.error{color:var(--vscode-errorForeground)}
.waiting-indicator{position:fixed;z-index:100;top:12px;left:50%;transform:translateX(-50%);display:flex;align-items:center;gap:8px;padding:7px 11px;border:1px solid var(--vscode-widget-border,var(--vscode-panel-border));border-radius:5px;background:var(--vscode-editorWidget-background,var(--vscode-sideBar-background));color:var(--vscode-editorWidget-foreground,var(--vscode-foreground));box-shadow:0 3px 14px rgba(0,0,0,.35);pointer-events:none}.waiting-indicator[hidden]{display:none}.waiting-spinner{width:14px;height:14px;border:2px solid color-mix(in srgb,var(--vscode-progressBar-background) 35%,transparent);border-top-color:var(--vscode-progressBar-background);border-radius:50%;animation:lens-spin .75s linear infinite}@keyframes lens-spin{to{transform:rotate(360deg)}}
.ica-menu{position:fixed;z-index:110;width:max-content;min-width:180px;max-width:min(290px,calc(100vw - 16px));padding:5px;border:1px solid var(--vscode-menu-border,var(--vscode-dropdown-border));border-radius:6px;background:var(--vscode-menu-background,var(--vscode-dropdown-background));box-shadow:0 8px 24px rgba(0,0,0,.42)}.ica-menu[hidden],.menu-group[hidden]{display:none}.ica-menu button{display:flex;align-items:center;gap:9px;width:100%;min-height:30px;padding:5px 9px;border-radius:3px;text-align:left;white-space:nowrap;background:transparent;color:var(--vscode-menu-foreground,var(--vscode-foreground))}.ica-menu button:hover{background:var(--vscode-menu-selectionBackground,var(--vscode-list-hoverBackground));color:var(--vscode-menu-selectionForeground,var(--vscode-foreground))}.menu-group+.menu-group{margin-top:4px;padding-top:4px;border-top:1px solid var(--vscode-menu-separatorBackground,var(--vscode-panel-border))}.menu-heading{padding:3px 9px 2px;color:var(--vscode-descriptionForeground);font-size:11px;font-weight:600;letter-spacing:.04em;text-transform:uppercase}.menu-icon{display:inline-grid;place-items:center;width:18px;height:18px;flex:0 0 18px;color:var(--vscode-descriptionForeground);font-size:12px;font-weight:700}.menu-remove .menu-icon{color:var(--vscode-errorForeground)}.menu-patch .menu-icon{color:#c58bd8}.menu-cosine .menu-icon{color:#69b99a}.card.intervention-removed{opacity:.3;filter:grayscale(1)}
#neurons{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:8px}.lens-group{grid-column:1/-1;display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:8px;min-width:0}.lens-group[hidden]{display:none}.lens-group.submodule{padding-left:24px;border-left:4px solid color-mix(in srgb,var(--vscode-descriptionForeground) 42%,var(--vscode-editor-background))}.card{position:relative;border:1px solid var(--vscode-panel-border);background:var(--vscode-sideBar-background)}.card.full-width{grid-column:1/-1}.card.saved{opacity:.6;border-color:transparent;background:linear-gradient(rgba(0,0,0,.24),rgba(0,0,0,.24)),var(--vscode-sideBar-background)}.card.saved::after{content:"";position:absolute;inset:-1px;pointer-events:none;background:repeating-linear-gradient(90deg,var(--vscode-descriptionForeground) 0 12px,transparent 12px 21px) top left/100% 1px no-repeat,repeating-linear-gradient(90deg,var(--vscode-descriptionForeground) 0 12px,transparent 12px 21px) bottom left/100% 1px no-repeat,repeating-linear-gradient(0deg,var(--vscode-descriptionForeground) 0 12px,transparent 12px 21px) top left/1px 100% no-repeat,repeating-linear-gradient(0deg,var(--vscode-descriptionForeground) 0 12px,transparent 12px 21px) top right/1px 100% no-repeat}.readouts{display:flex;flex-direction:column}.readout{display:flex;align-items:center;flex-wrap:wrap;gap:5px;padding:5px 8px}.readout[hidden]{display:none}.readout+.readout{border-top:1px solid var(--vscode-panel-border)}.readout-title{font-weight:600;color:var(--vscode-descriptionForeground);white-space:nowrap}.empty{color:var(--vscode-descriptionForeground)}
.target-heading{grid-column:1/-1;display:flex;align-items:center;gap:6px;margin:7px 0 0;font-size:13px;color:var(--vscode-descriptionForeground)}.final-layer-icon{display:inline-grid;place-items:center;width:16px;height:16px;margin-left:-3px;color:var(--vscode-descriptionForeground);font-size:15px;font-weight:700;line-height:1}.target-magnitude{font-weight:400;font-variant-numeric:tabular-nums}.residual-triangle{display:block;flex:0 0 auto;width:40px;height:28px;overflow:visible}.component-fan{display:block;flex:0 0 auto;width:40px;height:28px;overflow:visible}.component-triangle{display:block;flex:0 0 auto;width:44px;height:26px;margin-left:auto;overflow:visible}.residual-triangle line,.component-fan line,.component-triangle line{stroke-width:1.8;stroke-linecap:round;vector-effect:non-scaling-stroke}.triangle-input{stroke:#5aa7e8;stroke-dasharray:3 2}.triangle-delta,.fan-result{stroke:#e6a15c}.triangle-output{stroke:#58c5ad}.component-delta{stroke:#9aa0a6}.component-potential{stroke:#58c5ad;stroke-dasharray:3 2}.fan-component{stroke:#9aa0a6;stroke-dasharray:3 2}.residual-triangle circle,.component-fan circle,.component-triangle circle{stroke:none}
.card-head{display:flex;align-items:center;flex-wrap:wrap;gap:5px 10px;padding:7px 9px;border-bottom:1px solid var(--vscode-panel-border)}.neuron-badge{display:inline-flex;align-items:center;justify-content:center;box-sizing:border-box;width:62px;flex:0 0 62px;padding:1px 6px;border:1px solid var(--vscode-badge-background);border-radius:10px;background:linear-gradient(to right,var(--meter-color) 0 var(--meter-fill),var(--vscode-badge-background) var(--meter-fill) 100%);color:var(--vscode-badge-foreground);font-weight:600;font-variant-numeric:tabular-nums;cursor:pointer}.neuron-badge:focus{outline:1px solid var(--vscode-focusBorder);outline-offset:2px}.ica-component-id{color:var(--vscode-foreground);font-family:var(--vscode-editor-font-family);font-weight:600;font-variant-numeric:tabular-nums;white-space:nowrap}.component-details{display:none;align-items:center;flex-wrap:wrap;gap:5px 10px;min-width:0}.card-head.expanded .component-details{display:flex}.stats{color:var(--vscode-descriptionForeground);font-variant-numeric:tabular-nums}.report-toggle{display:flex;align-items:center;gap:5px;color:var(--vscode-descriptionForeground);white-space:nowrap}.report-toggle input{margin:0}.unfade-past .card.saved{opacity:1}.dark-report{color-scheme:dark;--vscode-foreground:#e6edf3;--vscode-descriptionForeground:#9ba7b4;--vscode-editor-background:#0f1115;--vscode-sideBar-background:#171a21;--vscode-panel-border:#343b46;--vscode-dropdown-foreground:#e6edf3;--vscode-dropdown-background:#252a33;--vscode-dropdown-border:#4a5360;--vscode-button-background:#0e639c;--vscode-button-foreground:#fff;--vscode-button-hoverBackground:#1177bb;--vscode-errorForeground:#ff7b72;--vscode-badge-background:#3b414b;--vscode-badge-foreground:#e6edf3;--vscode-focusBorder:#58a6ff}.light-report{color-scheme:light;--vscode-foreground:#364152;--vscode-descriptionForeground:#697586;--vscode-editor-background:#fbfcfd;--vscode-sideBar-background:#f7f9fb;--vscode-panel-border:#cbd3dc;--vscode-dropdown-foreground:#46566a;--vscode-dropdown-background:#f7f9fb;--vscode-dropdown-border:#b8c3cf;--vscode-button-background:#e5edf5;--vscode-button-foreground:#40566f;--vscode-button-hoverBackground:#dbe6f0;--vscode-errorForeground:#9f514b;--vscode-badge-background:#e7ecf1;--vscode-badge-foreground:#52606d;--vscode-focusBorder:#7899b8}.light-report .card{background:#fcfdfe}.light-report .card.saved{opacity:.8;background:#f8fafb}.light-report.unfade-past .card.saved{opacity:1}.light-report .readout-logit .token-chip{color:#526b82;background:#edf3f8;border-color:#b8c9d9}.light-report .readout-j-lens .token-chip{color:#715d80;background:#f3eef7;border-color:#cbbbd8}.light-report .readout-r-lens .token-chip{color:#557565;background:#edf5f1;border-color:#b5cfc2}.light-report .readout-jpp-lens .token-chip{color:#806b4f;background:#f7f1e7;border-color:#d6c19d}.light-report .readout-title,.light-report .stats{color:#697586}.light-report .neuron-badge{color:#52606d;border-color:#b8c3cf;background:linear-gradient(to right,color-mix(in srgb,var(--meter-color) 38%,#fff) 0 var(--meter-fill),#e7ecf1 var(--meter-fill) 100%)}.light-report .lens-group.submodule{border-left-color:#b3bec9}.light-report .triangle-input{stroke:#719fc5}.light-report .triangle-delta,.light-report .fan-result{stroke:#bf895c}.light-report .triangle-output,.light-report .component-potential{stroke:#72a598}.light-report .component-delta,.light-report .fan-component{stroke:#8794a3}
.token-chip{flex:0 0 auto;max-width:100%;padding:3px 7px 2px;border:1px solid var(--vscode-panel-border);border-radius:3px;color:var(--vscode-foreground);font-family:var(--vscode-editor-font-family);white-space:pre-wrap;overflow-wrap:anywhere;text-align:center;cursor:help}.readout-logit .token-chip{background:rgba(24,54,86,.78);border-color:rgba(82,148,214,.55)}.readout-j-lens .token-chip{background:rgba(55,35,78,.78);border-color:rgba(164,111,211,.55)}.readout-r-lens .token-chip{background:rgba(20,64,54,.78);border-color:rgba(63,164,139,.55)}.readout-jpp-lens .token-chip{background:rgba(76,57,18,.82);border-color:rgba(222,176,65,.68)}.token-chip:hover,.token-chip:focus{border-color:var(--vscode-focusBorder);outline:none;filter:brightness(1.22)}.light-report .token-chip{color:#374151!important}
.light-report .view-menu,.light-report .lens-menu{color:#374151;background:#fbfcfd;border-color:#cbd3dc;box-shadow:0 4px 14px rgba(71,85,105,.18)}.light-report .view-option,.light-report .view-option.ica,.light-report .lens-option{color:#374151}.light-report .view-option:hover,.light-report .lens-option:hover{background:#e9eef3}.light-report .view-option input,.light-report .lens-option input,.light-report .report-toggle input{accent-color:#7899b8}.actions>.compare-reference,.actions>.patch-reference{position:static;right:auto;top:auto;display:flex;align-items:center;max-width:min(520px,48vw);padding:4px 8px;box-shadow:none;flex:0 1 auto}.reference-label{min-width:0;overflow:hidden;text-overflow:ellipsis}.reference-norm{flex:0 0 auto;margin-left:6px}#referenceMenu,#patchMenu{z-index:140}
.vector-menu button[hidden],.cosine-result[hidden],.compare-reference[hidden],.patch-reference[hidden]{display:none}.vector-menu button:disabled{opacity:.72;cursor:default}.cosine-result{position:fixed;z-index:130;right:20px;top:64px;width:min(430px,calc(100vw - 40px));padding:14px;border:1px solid var(--vscode-panel-border);border-radius:5px;background:var(--vscode-sideBar-background);box-shadow:0 8px 28px rgba(0,0,0,.42)}.cosine-result-head{display:flex;align-items:center;justify-content:space-between;gap:12px;cursor:grab;user-select:none;touch-action:none}.cosine-result-head:active{cursor:grabbing}.cosine-result-head strong{font-size:14px}.cosine-close{padding:1px 7px;background:transparent;color:var(--vscode-foreground);font-size:18px}.cosine-values{display:grid;grid-template-columns:auto 1fr;gap:5px 10px;margin-top:9px}.cosine-label{overflow-wrap:anywhere;color:var(--vscode-descriptionForeground)}.cosine-angle{display:block;width:100%;height:145px;margin-top:8px}.cosine-ray-a{stroke:#69b99a}.cosine-ray-b{stroke:#b9c86c}.cosine-arc{fill:none;stroke:var(--vscode-descriptionForeground);stroke-dasharray:3 2}.cosine-angle text{fill:var(--vscode-foreground);font:12px var(--vscode-font-family)}.cosine-angle .cosine-label-a{fill:#69b99a}.cosine-angle .cosine-label-b{fill:#b9c86c}.light-report .cosine-result{background:#fbfcfd;box-shadow:0 8px 28px rgba(71,85,105,.22)}.light-report .cosine-ray-a{stroke:#3f806a}.light-report .cosine-ray-b{stroke:#7f8f35}.light-report .cosine-angle .cosine-label-a{fill:#3f806a}.light-report .cosine-angle .cosine-label-b{fill:#7f8f35}.ui-icon{width:15px;height:15px;flex:0 0 auto;vertical-align:-3px;stroke:currentColor;fill:none;stroke-width:1.7;stroke-linecap:round;stroke-linejoin:round}.actions button .ui-icon{margin-right:5px}.actions button.icon-only{display:inline-flex;align-items:center;justify-content:center;width:28px;height:26px;padding:0}.actions button.icon-only .ui-icon{margin:0}.report-toggle .ui-icon{width:16px;height:16px}.icon-toggle{position:relative;padding:4px;cursor:pointer}.icon-toggle input{position:absolute;opacity:0;pointer-events:none}.icon-toggle input:checked+.ui-icon{color:#e6a15c;filter:drop-shadow(0 0 2px currentColor)}.compare-reference,.patch-reference{max-width:min(460px,48vw);padding:4px 8px;border:1px solid #69b99a;border-radius:4px;background:var(--vscode-sideBar-background);color:var(--vscode-foreground);font-weight:600;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;cursor:context-menu}.compare-reference::before{content:'Reference · ';color:#69b99a}.patch-reference{border-color:#c58bd8}.patch-reference::before{content:'Patch · ';color:#c58bd8}.all-cosine-angle{width:44px;height:28px;margin-left:auto;flex:0 0 auto;overflow:visible}.all-cosine-angle line{stroke-width:2;stroke-linecap:round}.all-cosine-angle .reference-ray{stroke:#69b99a}.all-cosine-angle .target-ray{stroke:#b9c86c}.all-cosine-angle .angle-arc{fill:none;stroke:var(--vscode-descriptionForeground);stroke-dasharray:2 2;stroke-width:1}.light-report .compare-reference,.light-report .patch-reference{background:#fbfcfd}.light-report .compare-reference{border-color:#3f806a}.light-report .compare-reference::before{color:#3f806a}.light-report .patch-reference{border-color:#8b5ca0}.light-report .patch-reference::before{color:#8b5ca0}.light-report .all-cosine-angle .reference-ray{stroke:#3f806a}.light-report .all-cosine-angle .target-ray{stroke:#7f8f35}
</style></head><body><div id="waiting" class="waiting-indicator" role="status" aria-live="polite"><span class="waiting-spinner" aria-hidden="true"></span><span>Updating Lens…</span></div><div id="icaMenu" class="ica-menu vector-menu" hidden><div id="interventionGroup" class="menu-group"><div class="menu-heading">Intervention</div><button id="removeIca" class="menu-remove" type="button"><span class="menu-icon">−</span><span>Remove component</span></button></div><div class="menu-group"><div class="menu-heading">Patch</div><button id="patchCopy" class="menu-patch" type="button"><span class="menu-icon">⧉</span><span>Copy vector</span></button><button id="patchPaste" class="menu-patch" type="button" hidden><span class="menu-icon">↳</span><span>Paste and replace</span></button></div><div class="menu-group"><div class="menu-heading">Cosine</div><button id="measureA" class="menu-cosine" type="button"><span class="menu-icon">A</span><span class="menu-text">Set as Vector A</span></button><button id="measureB" class="menu-cosine" type="button"><span class="menu-icon">B</span><span class="menu-text">Set as Vector B</span></button><button id="compareAll" class="menu-cosine" type="button"><span class="menu-icon">∠</span><span>Compare to all vectors</span></button></div></div><div id="referenceMenu" class="ica-menu" hidden><button id="unsetReference" type="button">Unset</button></div><div id="patchMenu" class="ica-menu" hidden><button id="unsetPatch" type="button">Unset</button></div><aside id="compareReference" class="compare-reference" hidden title="Right-click to unset"></aside><aside id="patchReference" class="patch-reference" hidden title="Right-click to unset"></aside><section id="cosineResult" class="cosine-result" hidden><div class="cosine-result-head"><strong>Vector cosine</strong><button id="cosineClose" class="cosine-close" type="button" aria-label="Close">×</button></div><svg id="cosineAngle" class="cosine-angle" viewBox="0 0 340 105" role="img"></svg><div id="cosineValues" class="cosine-values"></div></section><header><div class="header-context"><div class="view-picker"><button id="viewButton" class="view-select" aria-haspopup="true" aria-expanded="false">7 blocks selected ▾</button><div id="viewMenu" class="view-menu" hidden></div></div><span id="currentToken" class="current-token">Token —</span></div><div class="actions"><label class="report-toggle live-only icon-toggle" title="Fade past blocks" aria-label="Fade past blocks"><input id="liveFadePast" type="checkbox" checked></label><label class="report-toggle live-only icon-toggle" title="Toggle light theme" aria-label="Toggle light theme"><input id="liveLightTheme" type="checkbox"></label><button id="exportMd" class="icon-only" title="Save report" aria-label="Save report"></button><button id="refresh" class="icon-only" title="Refresh" aria-label="Refresh"></button><div class="lens-picker"><button id="lensButton" class="lens-select" aria-haspopup="true" aria-expanded="false">1 Lens selected ▾</button><div id="lensMenu" class="lens-menu" hidden></div></div></div></header><div id="context">Loading…</div><main id="neurons"></main>
<script nonce="${nonce}">
const vscode=acquireVsCodeApi(),context=document.getElementById('context'),neurons=document.getElementById('neurons'),lensButton=document.getElementById('lensButton'),lensMenu=document.getElementById('lensMenu'),viewButton=document.getElementById('viewButton'),viewMenu=document.getElementById('viewMenu'),currentToken=document.getElementById('currentToken'),waiting=document.getElementById('waiting'),icaMenu=document.getElementById('icaMenu'),interventionGroup=document.getElementById('interventionGroup'),removeIca=document.getElementById('removeIca'),patchCopy=document.getElementById('patchCopy'),patchPaste=document.getElementById('patchPaste'),patchReference=document.getElementById('patchReference'),patchMenu=document.getElementById('patchMenu'),unsetPatch=document.getElementById('unsetPatch'),measureA=document.getElementById('measureA'),measureB=document.getElementById('measureB'),compareAll=document.getElementById('compareAll'),compareReference=document.getElementById('compareReference'),referenceMenu=document.getElementById('referenceMenu'),unsetReference=document.getElementById('unsetReference'),cosineResult=document.getElementById('cosineResult'),cosineClose=document.getElementById('cosineClose'),cosineAngle=document.getElementById('cosineAngle'),cosineValues=document.getElementById('cosineValues'),liveFadePast=document.getElementById('liveFadePast'),liveLightTheme=document.getElementById('liveLightTheme');let snapshot,pendingIntervention,pendingSave,pendingVector,pendingPatchTarget,allCosineReference,patchVectorReference;const cosineSelection={a:null,b:null},saved={};let selectedModes=new Set(),availableModes=[];
const uiIcons={save:'<svg class="ui-icon" viewBox="0 0 16 16" aria-hidden="true"><path d="M2.5 2.5h9l2 2v9h-11zM5 2.5v4h5v-4M5 13.5v-4h6v4"/></svg>',refresh:'<svg class="ui-icon" viewBox="0 0 16 16" aria-hidden="true"><path d="M13 5.5A5.5 5.5 0 1 0 13.2 10M13 2.5v3h-3"/></svg>',fade:'<svg class="ui-icon" viewBox="0 0 16 16" aria-hidden="true"><path d="m2 5 6-3 6 3-6 3zM3.5 8.5 8 11l4.5-2.5M5 12l3 2 3-2"/></svg>',theme:'<svg class="ui-icon" viewBox="0 0 16 16" aria-hidden="true"><path d="M10.8 12.8A5.5 5.5 0 0 1 4 4.2 5.5 5.5 0 1 0 10.8 12.8z"/></svg>'};document.getElementById('exportMd').insertAdjacentHTML('afterbegin',uiIcons.save);document.getElementById('refresh').insertAdjacentHTML('afterbegin',uiIcons.refresh);liveFadePast.parentElement.insertAdjacentHTML('beforeend',uiIcons.fade);liveLightTheme.parentElement.insertAdjacentHTML('beforeend',uiIcons.theme);
document.querySelector('.actions').prepend(compareReference);
document.querySelector('.actions').prepend(patchReference);
const tokenMenu=document.createElement('div'),setTokenReference=document.createElement('button'),unsetTokenReference=document.createElement('button');tokenMenu.id='tokenMenu';tokenMenu.className='ica-menu';tokenMenu.hidden=true;setTokenReference.type='button';setTokenReference.textContent='Compare corresponding vectors';unsetTokenReference.type='button';unsetTokenReference.textContent='Unset token reference';tokenMenu.append(setTokenReference,unsetTokenReference);document.body.append(tokenMenu);let tokenCosineReference=null;
const baseViews=[['residual-pre','Residual Lens · Pre'],['attention-neurons','Attention Output Channel Lens'],['attention-output','Attention Output Lens'],['residual-mid','Residual Lens · Mid'],['neurons','MLP Neuron Lens'],['mlp-output','MLP Output Lens'],['residual-post','Residual Lens · Post']],icaViews=[['ica-residual-pre','Residual ICA Lens · Pre'],['ica-attention-output','Attention Output ICA Lens'],['ica-residual-mid','Residual ICA Lens · Mid'],['ica-mlp-output','MLP Output ICA Lens'],['ica-residual-post','Residual ICA Lens · Post']],orderedViews=[baseViews[0],icaViews[0],baseViews[1],baseViews[2],icaViews[1],baseViews[3],icaViews[2],baseViews[4],baseViews[5],icaViews[3],baseViews[6],icaViews[4]];let selectedViews=new Set(baseViews.map(item=>item[0])),icaSupported=false;
function rebuildViewMenu(){viewMenu.replaceChildren();const choices=orderedViews.filter(([value])=>icaSupported||!value.startsWith('ica-')),values=choices.map(item=>item[0]),allRow=document.createElement('label'),allBox=document.createElement('input'),selectedCount=values.filter(value=>selectedViews.has(value)).length;allRow.className='view-option select-all';allBox.type='checkbox';allBox.checked=selectedCount===values.length;allBox.indeterminate=selectedCount>0&&selectedCount<values.length;allBox.onchange=()=>{for(const value of values)allBox.checked?selectedViews.add(value):selectedViews.delete(value);rebuildViewMenu();requestRefresh()};allRow.append(allBox,document.createTextNode('All'));viewMenu.append(allRow);for(const [value,label] of choices){const row=document.createElement('label');row.className='view-option'+(value.startsWith('ica-')?' ica':'');const box=document.createElement('input');box.type='checkbox';box.value=value;box.checked=selectedViews.has(value);box.onchange=()=>{box.checked?selectedViews.add(value):selectedViews.delete(value);rebuildViewMenu();requestRefresh()};row.append(box,document.createTextNode(label));viewMenu.append(row)}updateViewButton()}
const headView=['attention-heads','Attention Head Lens'];baseViews.push(headView);orderedViews.splice(2,0,headView);selectedViews.add(headView[0]);selectedViews.delete('attention-neurons');
const valueHeadView=['attention-head-values','Attention Head Value Lens'];baseViews.push(valueHeadView);orderedViews.splice(2,0,valueHeadView);selectedViews.add(valueHeadView[0]);
const memoryWriteView=['delta-memory-write','DeltaNet Memory Write Lens'];baseViews.push(memoryWriteView);orderedViews.splice(3,0,memoryWriteView);
function updateViewButton(){const count=selectedViews.size;viewButton.textContent=(count===baseViews.length&&!icaSupported?'All':count+' block'+(count===1?'':'s')+' selected')+' ▾'}
${tokenLabel.toString()}
${tokensFor.toString()}
function renderHeadCards(item,target){
  const title=target.querySelector?.(':scope > .target-heading');
  const writeMode=item.view==='delta-memory-write',valueMode=item.view==='attention-head-values'||writeMode,maximum=Math.max(...item.heads.map(h=>h.vectorNorm),1e-30);
  if(title)title.title=writeMode?'Decodes beta × (value − predicted value) through each value head’s output map. Prediction uses the key and decayed previous memory. This is the write content, not the full key-addressed matrix update or query-dependent readout. No output normalization or gate. Ranked by raw projected norm; output-head calibration does not apply.':valueMode?'Current token value projected through each output head. No attention weighting, memory update, output normalization, or gate. Shared KV values are decoded separately through each query head’s output map. Ranked by projected value norm / this head’s own value reference median when all references are available; otherwise raw norms with a missing-calibration error. References use a text-only corpus, not image-specific importance.':'Decodes each head’s mixed output after projection into residual space (OV-projected).';
  if(writeMode){const note=document.createElement('div');note.className='stats';note.textContent='β × (v − predicted value), projected through W_o · write content only; excludes key addressing, query retrieval, output norm and gate · raw norms, not calibrated × scores';target.append(note)}
  for(const head of item.heads){
    const card=document.createElement('section');card.className='card full-width';styleCard(card,item);
    const heading=document.createElement('div');heading.className='card-head';
    const badge=document.createElement('span');badge.className='neuron-badge';badge.style.width='90px';badge.style.flex='0 0 90px';badge.style.whiteSpace='nowrap';badge.textContent='Head '+head.index;badge.tabIndex=0;badge.setAttribute('role','button');badge.setAttribute('aria-expanded','false');
    const calibrated=Number.isFinite(head.relativeNorm);
    badge.style.setProperty('--meter-fill',(calibrated?Math.min(head.relativeNorm,10)*10:valueMode?head.vectorNorm/maximum*100:0)+'%');
    badge.style.setProperty('--meter-color','rgba(126,55,52,.82)');
    badge.title=calibrated?head.relativeNorm.toFixed(2)+'× · projected '+(valueMode?'value ':'')+'norm / this head’s '+(valueMode?'value ':'')+'reference median · full width = 10×':valueMode?'Value head '+head.valueHead+' · raw projected '+(writeMode?'write':'value')+' norm · relative bar within this snapshot'+(head.calibrationError?' · '+head.calibrationError:''):head.calibrationError;
    const details=document.createElement('span');details.className='component-details';
    const stats=document.createElement('span');stats.className='stats';
    stats.textContent=(writeMode?'|write W_o| ':valueMode?'|v W_o| ':'|Δx| ')+head.vectorNorm.toFixed(2)+(calibrated?' ('+head.relativeNorm.toFixed(2)+'×) · ref '+head.normScale.toFixed(2):'')+' · '+head.headDimension+' channels'+(valueMode?' · V head '+head.valueHead:'');details.append(stats);
    const channels=document.createElement('div');channels.className='head-channels';channels.style.cssText='display:none;grid-column:1/-1;margin:0 0 10px 24px;padding-left:14px;border-left:2px solid var(--vscode-panel-border);grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:8px';
    renderSnapshot({...item,view:'attention-neurons',neurons:head.channels||[],componentLabel:'channel'},channels);
    const toggle=()=>{const expanded=heading.classList.toggle('expanded');badge.setAttribute('aria-expanded',String(expanded));channels.style.display=expanded?'grid':'none'};
    badge.onclick=toggle;badge.onkeydown=event=>{if(event.key==='Enter'||event.key===' '){event.preventDefault();toggle()}};
    heading.append(badge,details);
    if(head.memoryTokenContributions){
      const button=document.createElement('button');button.type='button';button.className='channel-token-contributions';
      button.title='Replay prefix to inspect DeltaNet memory contributions';button.setAttribute('aria-label',button.title);
      button.style.cssText='padding:2px 5px;background:transparent;color:var(--vscode-foreground);display:inline-flex;align-items:center';
      button.innerHTML='<svg class="ui-icon" viewBox="0 0 16 16" aria-hidden="true"><path d="M2 3h8M2 8h12M2 13h5" fill="none" stroke="currentColor" stroke-width="2"/></svg>';
      button.onclick=event=>{event.stopPropagation();vscode.postMessage({type:'deltaMemoryContributions',...head.memoryTokenContributions})};
      heading.append(button);
    }
    if(valueMode){badge.removeAttribute('role');badge.removeAttribute('aria-expanded');badge.onclick=null;badge.onkeydown=null;badge.tabIndex=-1;}
    if(!calibrated&&head.calibrationError){const error=document.createElement('span');error.className='calibration-error';error.style.color='var(--vscode-errorForeground)';error.textContent=head.calibrationError;heading.append(error)}
    card.append(heading);addReadouts(card,head);target.append(card,channels);
  }
}
function addReadouts(card,source){
  if(Number.isFinite(source.contributionMagnitude)&&!(Number.isFinite(source.contributionScale)&&source.contributionScale>0)){
    const heading=card.querySelector('.card-head'),pill=heading&&heading.querySelector('.neuron-badge');
    if(pill){
      pill.style.background='var(--vscode-badge-background)';
      pill.style.removeProperty('--meter-fill');
      pill.title='Error: missing calibrated reference scale. The × score is unavailable, not zero. Click to inspect raw measurements.';
    }
    if(heading){
      const error=document.createElement('span');error.className='calibration-error';
      error.textContent='Error: missing calibration (× unavailable)';
      error.style.color='var(--vscode-errorForeground)';
      error.title='No valid calibrated reference scale exists for this model/layer. Raw measurements and rankings remain available.';
      error.setAttribute('role','status');heading.append(error);
    }
  }
  if(source.channelTokenContributions){
    const mapping=source.channelTokenContributions,details=card.querySelector('.component-details');
    if(details&&Number.isInteger(mapping.kvHead)){
      const heads=document.createElement('span');heads.className='stats';
      heads.textContent='Q head '+mapping.queryHead+' · V head '+mapping.kvHead+' · dim '+mapping.dimension;
      heads.title='Zero-based indices. Value head is the shared KV head; dimension '+mapping.dimension+' of '+mapping.headDimension+'.';
      details.prepend(heads);
    }
    const button=document.createElement('button');
    button.type='button';button.className='channel-token-contributions';
    button.title='Inspect source-token contributions (weight × value)';
    button.setAttribute('aria-label',button.title);
    button.style.cssText='padding:2px 5px;background:transparent;color:var(--vscode-foreground);display:inline-flex;align-items:center';
    button.innerHTML='<svg class="ui-icon" viewBox="0 0 16 16" aria-hidden="true"><path d="M2 3h8M2 8h12M2 13h5" fill="none" stroke="currentColor" stroke-width="2"/></svg>';
    button.onclick=event=>{event.stopPropagation();vscode.postMessage({type:'attentionTokenContributions',...source.channelTokenContributions})};
    const heading=card.querySelector('.card-head'),pill=heading&&heading.querySelector('.neuron-badge');
    if(pill)pill.after(button);
    if(pill&&source.channelTokenContributions.hasImages){
      const picture=document.createElement('button');
      picture.type='button';picture.className='channel-image-maps';
      picture.title='Inspect image patch maps (contribution and value)';
      picture.setAttribute('aria-label',picture.title);picture.style.cssText=button.style.cssText;
      picture.innerHTML='<svg class="ui-icon" viewBox="0 0 16 16" aria-hidden="true"><rect x="1.5" y="2.5" width="13" height="11" rx="1" fill="none" stroke="currentColor"/><circle cx="5" cy="6" r="1" fill="currentColor"/><path d="M2 12l4-4 3 3 2-2 3 3" fill="none" stroke="currentColor"/></svg>';
      picture.onclick=event=>{event.stopPropagation();vscode.postMessage({type:'attentionImageMaps',...source.channelTokenContributions})};
      button.after(picture);
    }
  }
  const readouts=document.createElement('div');readouts.className='readouts';for(const row of source.residualVocabularyReadouts||[])readouts.append(tokensFor(row.kind,row.label,row.tokens||[],row.unavailable,row.hint));card.append(readouts)
}
function updateLensButton(){const count=selectedModes.size;lensButton.textContent=(count?count+' Lens'+(count===1?'':'es')+' selected':'No Lenses selected')+' ▾'}
function rebuildLensOptions(lenses,selected){availableModes=[['logit','Logit Lens'],...(lenses||[]).map(lens=>['lens:'+lens.id,lens.label])];if(Array.isArray(selected))selectedModes=new Set(selected.filter(value=>availableModes.some(item=>item[0]===value)));if(!selectedModes.size&&!Array.isArray(selected)&&availableModes.length)selectedModes.add(availableModes[0][0]);lensMenu.replaceChildren();const values=availableModes.map(item=>item[0]),allRow=document.createElement('label'),allBox=document.createElement('input'),selectedCount=values.filter(value=>selectedModes.has(value)).length;allRow.className='lens-option select-all';allBox.type='checkbox';allBox.checked=selectedCount===values.length;allBox.indeterminate=selectedCount>0&&selectedCount<values.length;allBox.onchange=()=>{selectedModes=allBox.checked?new Set(values):new Set();rebuildLensOptions(lenses,[...selectedModes]);requestRefresh()};allRow.append(allBox,document.createTextNode('All'));lensMenu.append(allRow);for(const [value,label] of availableModes){const row=document.createElement('label');row.className='lens-option';const box=document.createElement('input');box.type='checkbox';box.value=value;box.checked=selectedModes.has(value);box.onchange=()=>{box.checked?selectedModes.add(value):selectedModes.delete(value);rebuildLensOptions(lenses,[...selectedModes]);requestRefresh()};row.append(box,document.createTextNode(label));lensMenu.append(row)}updateLensButton()}
function styleCard(card,item){if(item.saved)card.classList.add('saved')}
function cosineOptionText(slot){const selected=cosineSelection[slot];return selected?slot.toUpperCase()+' · '+selected.label:'Set as Vector '+slot.toUpperCase()}
function placeMenu(menu,event){menu.hidden=false;const bounds=menu.getBoundingClientRect(),margin=8;menu.style.left=Math.max(margin,Math.min(event.clientX,window.innerWidth-bounds.width-margin))+'px';menu.style.top=Math.max(margin,Math.min(event.clientY,window.innerHeight-bounds.height-margin))+'px'}
function openVectorMenu(event,vector,intervention,patchTarget){event.preventDefault();pendingVector=vector;pendingIntervention=intervention||null;pendingPatchTarget=patchTarget||null;interventionGroup.hidden=!pendingIntervention;patchPaste.hidden=!(patchVectorReference&&pendingPatchTarget);measureA.querySelector('.menu-text').textContent=cosineOptionText('a');measureB.querySelector('.menu-text').textContent=cosineOptionText('b');measureA.disabled=Boolean(cosineSelection.a);measureB.disabled=Boolean(cosineSelection.b);placeMenu(icaMenu,event)}
function chooseCosine(slot){if(!pendingVector)return;cosineSelection[slot]=pendingVector;icaMenu.hidden=true;pendingVector=null;if(cosineSelection.a&&cosineSelection.b){setBusy(true);vscode.postMessage({type:'measureCosine',vectorA:cosineSelection.a.id,vectorB:cosineSelection.b.id})}}
function displayedMeasurementTargets(){const targets=[];if(!snapshot)return targets;for(const group of neurons.querySelectorAll('.lens-group')){const item=snapshot.items.find(candidate=>candidate.view===group.dataset.view);if(!item||item.error)continue;if(item.measurementId)targets.push({id:item.measurementId,node:group.querySelector(':scope > .target-heading')});const values=item.icaComponents||item.neurons||[],cards=[...group.querySelectorAll(':scope > .card')];for(let index=0;index<values.length;index++)if(values[index].measurementId&&cards[index])targets.push({id:values[index].measurementId,node:cards[index].querySelector('.card-head')})}return targets}
function correspondingTokenTargets(){const targets=[];if(!snapshot)return targets;for(const group of neurons.querySelectorAll('.lens-group')){const item=snapshot.items.find(candidate=>candidate.view===group.dataset.view);if(!item||item.error||item.view.startsWith('ica-')||!item.measurementId||['neurons','attention-neurons'].includes(item.view))continue;targets.push({key:item.view+'|'+item.transformerLayer,id:item.measurementId,node:group.querySelector(':scope > .target-heading')})}return targets}
function requestAllCosines(){if(!allCosineReference||!snapshot)return;const targets=displayedMeasurementTargets().map(target=>target.id);if(targets.length)vscode.postMessage({type:'measureCosineAll',targets})}
function requestTokenCosines(){if(!tokenCosineReference||!snapshot)return;const targets=correspondingTokenTargets().map(({key,id})=>({key,id}));if(targets.length)vscode.postMessage({type:'measureTokenCosines',targets})}
function showCosineReference(reference){const label=document.createElement('span');label.className='reference-label';label.textContent=reference.label+(Number.isInteger(reference.tokenPosition)?' · token '+reference.tokenPosition:'');compareReference.replaceChildren(label);if(Number.isFinite(reference.norm)){const norm=document.createElement('span');norm.className='reference-norm';norm.textContent='|x| '+reference.norm.toFixed(2);compareReference.append(norm)}compareReference.hidden=false}
function showPatchReference(reference){patchVectorReference=reference;const label=document.createElement('span');label.className='reference-label';label.textContent=reference.label+(Number.isInteger(reference.tokenPosition)?' · token '+reference.tokenPosition:'');patchReference.replaceChildren(label);if(Number.isFinite(reference.norm)){const norm=document.createElement('span');norm.className='reference-norm';norm.textContent='|x| '+reference.norm.toFixed(2);patchReference.append(norm)}patchReference.hidden=false}
function clearPatchReference(){patchVectorReference=null;patchReference.hidden=true;patchMenu.hidden=true;vscode.postMessage({type:'unsetPatchReference'})}
function copyPatchVector(){if(!pendingVector)return;vscode.postMessage({type:'setPatchReference',vector:pendingVector.id});icaMenu.hidden=true;pendingVector=null}
function pastePatchVector(){if(!patchVectorReference||!pendingPatchTarget)return;setBusy(true);vscode.postMessage({type:'pastePatch',view:pendingPatchTarget.view});icaMenu.hidden=true;pendingPatchTarget=null}
function compactCosineAngle(cosine,title){const ns='http://www.w3.org/2000/svg',svg=document.createElementNS(ns,'svg'),origin=[4,24],length=20,theta=Math.acos(Math.max(-1,Math.min(1,cosine))),end=[origin[0]+length*Math.cos(theta),origin[1]-length*Math.sin(theta)];svg.classList.add('all-cosine-angle');svg.setAttribute('viewBox','0 0 44 28');svg.setAttribute('role','img');const tooltip=document.createElementNS(ns,'title');tooltip.textContent=title;svg.append(tooltip);for(const [point,className] of [[[origin[0]+length,origin[1]],'reference-ray'],[end,'target-ray']]){const line=document.createElementNS(ns,'line');line.setAttribute('x1',origin[0]);line.setAttribute('y1',origin[1]);line.setAttribute('x2',point[0]);line.setAttribute('y2',point[1]);line.setAttribute('class',className);svg.append(line)}const radius=8,arc=document.createElementNS(ns,'path');arc.setAttribute('class','angle-arc');arc.setAttribute('d','M '+(origin[0]+radius)+' '+origin[1]+' A '+radius+' '+radius+' 0 0 0 '+(origin[0]+radius*Math.cos(theta))+' '+(origin[1]-radius*Math.sin(theta)));svg.append(arc);return svg}
function showAllCosines(data){document.querySelectorAll('.all-cosine-angle').forEach(node=>node.remove());const targets=new Map(displayedMeasurementTargets().map(target=>[target.id,target.node]));for(const result of data.results||[]){const node=targets.get(result.target);if(node)node.append(compactCosineAngle(result.cosine,'Compared with '+result.labelA+' · cos '+result.cosine.toFixed(4)+' · '+result.angleDegrees.toFixed(2)+'°'))}setBusy(false)}
function showTokenCosines(data){document.querySelectorAll('.all-cosine-angle').forEach(node=>node.remove());const targets=new Map(correspondingTokenTargets().map(target=>[target.key,target.node]));for(const result of data.results||[]){const node=targets.get(result.key);if(node)node.append(compactCosineAngle(result.cosine,'Same site at reference token '+data.tokenPosition+' · cos '+result.cosine.toFixed(4)+' · '+result.angleDegrees.toFixed(2)+'°'))}setBusy(false)}
function chooseCompareAll(){if(!pendingVector)return;allCosineReference=pendingVector;showCosineReference(allCosineReference);icaMenu.hidden=true;setBusy(true);vscode.postMessage({type:'setCosineReference',vector:pendingVector.id});pendingVector=null}
function clearCompareAll(){allCosineReference=null;tokenCosineReference=null;compareReference.hidden=true;referenceMenu.hidden=true;document.querySelectorAll('.all-cosine-angle').forEach(node=>node.remove());vscode.postMessage({type:'unsetCosineReference'});vscode.postMessage({type:'unsetTokenReference'})}
function showCosine(data){cosineAngle.replaceChildren();const ns='http://www.w3.org/2000/svg',origin=[55,78],length=65,theta=Math.acos(Math.max(-1,Math.min(1,data.cosine))),displayTheta=Math.min(theta,Math.PI*.88),endA=[origin[0]+length,origin[1]],endB=[origin[0]+length*Math.cos(displayTheta),origin[1]-length*Math.sin(displayTheta)];function line(end,className){const node=document.createElementNS(ns,'line');node.setAttribute('x1',origin[0]);node.setAttribute('y1',origin[1]);node.setAttribute('x2',end[0]);node.setAttribute('y2',end[1]);node.setAttribute('stroke-width','3');node.setAttribute('stroke-linecap','round');node.setAttribute('class',className);cosineAngle.append(node)}line(endA,'cosine-ray-a');line(endB,'cosine-ray-b');const radius=30,arc=document.createElementNS(ns,'path');arc.setAttribute('class','cosine-arc');arc.setAttribute('d','M '+(origin[0]+radius)+' '+origin[1]+' A '+radius+' '+radius+' 0 '+(displayTheta>Math.PI?'1':'0')+' 0 '+(origin[0]+radius*Math.cos(displayTheta))+' '+(origin[1]-radius*Math.sin(displayTheta)));cosineAngle.append(arc);const text=document.createElementNS(ns,'text');text.setAttribute('x','145');text.setAttribute('y','52');text.textContent=data.angleDegrees.toFixed(2)+'°';cosineAngle.append(text);cosineValues.replaceChildren();for(const [name,value] of [['A',data.labelA],['B',data.labelB],['cos(A, B)',data.cosine.toFixed(6)],['Angle',data.angleDegrees.toFixed(2)+'°'],['Dimensions',String(data.dimensions)]]){const key=document.createElement('strong'),item=document.createElement('span');key.textContent=name;item.textContent=value;item.className='cosine-label';cosineValues.append(key,item)}cosineResult.hidden=false;setBusy(false)}
function showCosineRelative(data){cosineAngle.replaceChildren();const ns='http://www.w3.org/2000/svg',origin=[220,84],theta=Math.acos(Math.max(-1,Math.min(1,data.cosine))),ratio=data.normB/data.normA,scale=72/Math.max(1,ratio),lengthA=scale,lengthB=scale*ratio,endA=[origin[0]-lengthA,origin[1]],endB=[origin[0]-lengthB*Math.cos(theta),origin[1]-lengthB*Math.sin(theta)];function line(end,className){const node=document.createElementNS(ns,'line');node.setAttribute('x1',origin[0]);node.setAttribute('y1',origin[1]);node.setAttribute('x2',end[0]);node.setAttribute('y2',end[1]);node.setAttribute('stroke-width','3');node.setAttribute('stroke-linecap','round');node.setAttribute('class',className);cosineAngle.append(node)}function label(point,text,className){const node=document.createElementNS(ns,'text');node.setAttribute('x',point[0]-4);node.setAttribute('y',point[1]-7);node.setAttribute('text-anchor','end');node.setAttribute('class',className);node.textContent=text;cosineAngle.append(node)}line(endA,'cosine-ray-a');line(endB,'cosine-ray-b');label(endA,'A · 1.00×','cosine-label-a');label(endB,'B · '+ratio.toFixed(2)+'×','cosine-label-b');const radius=Math.min(29,lengthA*.48,lengthB*.48),arc=document.createElementNS(ns,'path'),arcEnd=[origin[0]-radius*Math.cos(theta),origin[1]-radius*Math.sin(theta)];arc.setAttribute('class','cosine-arc');arc.setAttribute('d','M '+(origin[0]-radius)+' '+origin[1]+' A '+radius+' '+radius+' 0 '+(theta>Math.PI?'1':'0')+' 1 '+arcEnd[0]+' '+arcEnd[1]);cosineAngle.append(arc);const angleText=document.createElementNS(ns,'text');angleText.setAttribute('x','258');angleText.setAttribute('y','48');angleText.textContent=data.angleDegrees.toFixed(2)+'°';cosineAngle.append(angleText);cosineValues.replaceChildren();for(const [name,value] of [['A',data.labelA],['|A|',data.normA.toFixed(4)+' (1.00× reference)'],['B',data.labelB],['|B|',data.normB.toFixed(4)+' ('+ratio.toFixed(4)+'× |A|)'],['cos(A, B)',data.cosine.toFixed(6)],['Angle',data.angleDegrees.toFixed(2)+'°'],['Dimensions',String(data.dimensions)]]){const key=document.createElement('strong'),item=document.createElement('span');key.textContent=name;item.textContent=value;item.className='cosine-label';cosineValues.append(key,item)}cosineResult.hidden=false;setBusy(false)}
function showCosineRelativeRight(data){cosineAngle.replaceChildren();cosineAngle.setAttribute('viewBox','0 0 390 145');const ns='http://www.w3.org/2000/svg',origin=[82,118],theta=Math.acos(Math.max(-1,Math.min(1,data.cosine))),ratio=data.normB/data.normA,scale=88/Math.max(1,ratio),lengthA=scale,lengthB=scale*ratio,endA=[origin[0]+lengthA,origin[1]],endB=[origin[0]+lengthB*Math.cos(theta),origin[1]-lengthB*Math.sin(theta)];function line(end,className){const node=document.createElementNS(ns,'line');node.setAttribute('x1',origin[0]);node.setAttribute('y1',origin[1]);node.setAttribute('x2',end[0]);node.setAttribute('y2',end[1]);node.setAttribute('stroke-width','3');node.setAttribute('stroke-linecap','round');node.setAttribute('class',className);cosineAngle.append(node)}function label(point,text,className,anchor,dx,dy){const node=document.createElementNS(ns,'text');node.setAttribute('x',point[0]+dx);node.setAttribute('y',point[1]+dy);node.setAttribute('text-anchor',anchor);node.setAttribute('class',className);node.textContent=text;cosineAngle.append(node)}line(endA,'cosine-ray-a');line(endB,'cosine-ray-b');label(endA,'A · 1.00×','cosine-label-a','start',5,16);label(endB,'B · '+ratio.toFixed(2)+'×','cosine-label-b','start',5,-6);const radius=Math.min(32,lengthA*.48,lengthB*.48),arc=document.createElementNS(ns,'path'),arcEnd=[origin[0]+radius*Math.cos(theta),origin[1]-radius*Math.sin(theta)];arc.setAttribute('class','cosine-arc');arc.setAttribute('d','M '+(origin[0]+radius)+' '+origin[1]+' A '+radius+' '+radius+' 0 '+(theta>Math.PI?'1':'0')+' 0 '+arcEnd[0]+' '+arcEnd[1]);cosineAngle.append(arc);const angleText=document.createElementNS(ns,'text');angleText.setAttribute('x','238');angleText.setAttribute('y','62');angleText.textContent=data.angleDegrees.toFixed(2)+'°';cosineAngle.append(angleText);cosineValues.replaceChildren();for(const [name,value] of [['A',data.labelA],['|A|',data.normA.toFixed(4)+' (1.00× reference)'],['B',data.labelB],['|B|',data.normB.toFixed(4)+' ('+ratio.toFixed(4)+'× |A|)'],['cos(A, B)',data.cosine.toFixed(6)],['Angle',data.angleDegrees.toFixed(2)+'°'],['Dimensions',String(data.dimensions)]]){const key=document.createElement('strong'),item=document.createElement('span');key.textContent=name;item.textContent=value;item.className='cosine-label';cosineValues.append(key,item)}cosineResult.hidden=false;setBusy(false)}
function renderSnapshot(item,target=neurons){if(item.error){const card=document.createElement('section');card.className='card full-width';const head=document.createElement('div');head.className='card-head error';head.textContent=item.label+': '+item.error;card.append(head);target.append(card);return}if(item.icaComponents){for(const component of item.icaComponents){const card=document.createElement('section');card.className='card';styleCard(card,item);if(component.removed)card.classList.add('intervention-removed');const head=document.createElement('div');head.className='card-head';const componentId=document.createElement('span');componentId.className='ica-component-id';componentId.textContent='L'+item.transformerLayer+':C'+component.index;componentId.title=component.removed?'This ICA component is removed by an active intervention':'layer '+item.transformerLayer+', ICA component '+component.index+' · rank '+component.rank+' by absolute score'+(item.view.startsWith('ica-residual-')&&!item.saved?' · right-click to intervene':'');if(item.view.startsWith('ica-residual-')&&!item.saved&&!component.removed)componentId.oncontextmenu=event=>{event.preventDefault();pendingIntervention={type:'removeIca',view:item.view,component:component.index,card};icaMenu.style.left=Math.min(event.clientX,window.innerWidth-190)+'px';icaMenu.style.top=Math.min(event.clientY,window.innerHeight-42)+'px';icaMenu.hidden=false};const score=document.createElement('span');score.className='stats';score.textContent='s '+component.score.toFixed(2);score.title='signed ICA component score';const energy=document.createElement('span');energy.className='stats';energy.textContent=(component.energy*100).toFixed(1)+'% energy';energy.title='score² divided by total ICA score energy';head.append(componentId,score,energy);card.append(head);addReadouts(card,component);target.append(card)}return}if(!['neurons','attention-neurons'].includes(item.view)){const card=document.createElement('section');card.className='card full-width';styleCard(card,item);addReadouts(card,item);target.append(card);return}for(const neuron of item.neurons){const card=document.createElement('section');card.className='card';styleCard(card,item);if(neuron.removed)card.classList.add('intervention-removed');const head=document.createElement('div');head.className='card-head';const value=neuron.relativeContribution??neuron.relativeActivation;const title=document.createElement('span');title.className='neuron-badge';title.textContent=String(neuron.index);title.tabIndex=0;title.setAttribute('role','button');title.setAttribute('aria-expanded','false');if(!item.saved&&!neuron.removed)title.oncontextmenu=event=>{event.preventDefault();pendingIntervention={type:'removeComponent',view:item.view,component:neuron.index,card};icaMenu.style.left=Math.min(event.clientX,window.innerWidth-190)+'px';icaMenu.style.top=Math.min(event.clientY,window.innerHeight-42)+'px';icaMenu.hidden=false};const meterValue=Number.isFinite(value)?value:0;title.style.setProperty('--meter-fill',(Math.min(Math.abs(meterValue),10)*10)+'%');title.style.setProperty('--meter-color',meterValue>=0?'rgba(126,55,52,.82)':'rgba(43,73,112,.86)');title.title=neuron.removed?'This component is removed by an active intervention':(item.componentLabel||'neuron')+'['+neuron.index+'] · output-projection rank '+neuron.rank+' · click to show measurements'+(!item.saved?' · right-click to intervene':'')+' · meter '+(value==null?'unavailable':((value>=0?'+':'')+value.toFixed(2)+'×'))+' (full width = 10×)';const projectionImpact=document.createElement('span');projectionImpact.className='stats';projectionImpact.textContent='along Δx '+neuron.outputProjection.toFixed(3);projectionImpact.title='signed projection of this component vector onto the complete module output; components are ranked by this value';const relative=value==null?'':(' ('+(value>=0?'+':'')+value.toFixed(2)+'×)');const impact=document.createElement('span');impact.className='stats';impact.textContent='|Δxᵢ| '+neuron.contributionMagnitude.toFixed(2)+relative;impact.title='individual component magnitude |Δxᵢ| = |a| × |d|';const activation=document.createElement('span');activation.className='stats';activation.textContent='a '+neuron.activation.toFixed(2);activation.title='activation coefficient a = '+neuron.activation.toFixed(6);const norm=document.createElement('span');norm.className='stats';norm.textContent='|d| '+neuron.directionNorm.toFixed(2);norm.title='output direction norm |d| = '+neuron.directionNorm.toFixed(6);const details=document.createElement('span');details.className='component-details';details.append(projectionImpact,impact,activation,norm);const toggle=()=>{const expanded=head.classList.toggle('expanded');title.setAttribute('aria-expanded',String(expanded));title.title=(item.componentLabel||'neuron')+'['+neuron.index+'] · output-projection rank '+neuron.rank+' · click to '+(expanded?'hide':'show')+' measurements'};title.onclick=toggle;title.onkeydown=event=>{if(event.key==='Enter'||event.key===' '){event.preventDefault();toggle()}};head.append(title,details,componentTriangle(neuron.componentGeometry));card.append(head);addReadouts(card,neuron);target.append(card)}}
function residualTriangle(view,geometry){const ns='http://www.w3.org/2000/svg',svg=document.createElementNS(ns,'svg');svg.classList.add('residual-triangle');svg.setAttribute('viewBox','0 0 34 24');svg.setAttribute('role','img');const title=document.createElementNS(ns,'title'),origin=[3,12],inputTip=[27,12],legend='Dashed blue: residual input x · Orange: update Δx · Teal: residual output x + Δx';if(view==='residual-pre'||!geometry){title.textContent=view==='residual-pre'?'Dashed blue: residual input x':legend+' · Update geometry is unavailable';svg.append(title);const line=document.createElementNS(ns,'line');line.classList.add('triangle-input');line.setAttribute('x1',origin[0]);line.setAttribute('y1',origin[1]);line.setAttribute('x2',inputTip[0]);line.setAttribute('y2',inputTip[1]);svg.append(line);const tip=document.createElementNS(ns,'circle');tip.setAttribute('cx',inputTip[0]);tip.setAttribute('cy',inputTip[1]);tip.setAttribute('r','1.7');tip.setAttribute('fill','#5aa7e8');svg.append(tip);return svg}const cosine=geometry.cosine==null?0:Math.max(-1,Math.min(1,geometry.cosine)),ratio=Math.max(0,geometry.deltaRatio||0),unitLength=inputTip[0]-origin[0],points=[origin,inputTip,[inputTip[0]+unitLength*ratio*cosine,inputTip[1]-unitLength*ratio*Math.sqrt(Math.max(0,1-cosine*cosine))]];title.textContent=legend+' · '+geometry.transition+' · cos(x, Δx) '+(geometry.cosine==null?'undefined':geometry.cosine.toFixed(3))+' · |Δx|/|x| '+geometry.deltaRatio.toFixed(3);svg.append(title);function segment(from,to,className,color){const line=document.createElementNS(ns,'line');line.classList.add(className);line.setAttribute('x1',points[from][0]);line.setAttribute('y1',points[from][1]);line.setAttribute('x2',points[to][0]);line.setAttribute('y2',points[to][1]);svg.append(line);const tip=document.createElementNS(ns,'circle');tip.setAttribute('cx',points[to][0]);tip.setAttribute('cy',points[to][1]);tip.setAttribute('r','1.7');tip.setAttribute('fill',color);svg.append(tip)}segment(0,2,'triangle-output','#58c5ad');segment(0,1,'triangle-input','#5aa7e8');segment(1,2,'triangle-delta','#e6a15c');return svg}
function componentTriangle(geometry){const ns='http://www.w3.org/2000/svg',svg=document.createElementNS(ns,'svg');svg.classList.add('component-triangle');svg.setAttribute('viewBox','0 0 44 26');svg.setAttribute('role','img');const origin=[3,14],inputTip=[35,14],cosine=geometry.cosine==null?0:Math.max(-1,Math.min(1,geometry.cosine)),ratio=Math.max(0,geometry.deltaRatio||0),length=inputTip[0]-origin[0],sumTip=[inputTip[0]+length*ratio*cosine,inputTip[1]-length*ratio*Math.sqrt(Math.max(0,1-cosine*cosine))],title=document.createElementNS(ns,'title');title.textContent='Dashed blue: current residual x · Solid gray: this component update vᵢ · Dashed teal: hypothetical residual x + vᵢ · cos(x, vᵢ) '+(geometry.cosine==null?'undefined':geometry.cosine.toFixed(3))+' · |vᵢ|/|x| '+geometry.deltaRatio.toFixed(3)+' · |x + vᵢ| '+geometry.outputNorm.toFixed(3);svg.append(title);function segment(from,to,className,color){const line=document.createElementNS(ns,'line');line.classList.add(className);line.setAttribute('x1',from[0]);line.setAttribute('y1',from[1]);line.setAttribute('x2',to[0]);line.setAttribute('y2',to[1]);svg.append(line);const tip=document.createElementNS(ns,'circle');tip.setAttribute('cx',to[0]);tip.setAttribute('cy',to[1]);tip.setAttribute('r','1.5');tip.setAttribute('fill',color);svg.append(tip)}segment(origin,sumTip,'component-potential','#58c5ad');segment(origin,inputTip,'triangle-input','#5aa7e8');segment(inputTip,sumTip,'component-delta','#9aa0a6');return svg}
function componentFan(geometry){const ns='http://www.w3.org/2000/svg',svg=document.createElementNS(ns,'svg');svg.classList.add('component-fan');svg.setAttribute('role','img');const result=geometry.result,top={x:1,y:0},ends=geometry.components.map(component=>({x:1+component.x,y:component.y,index:component.index})),points=[{x:0,y:0},top,result,...ends],xs=points.map(point=>point.x),ys=points.map(point=>-point.y),span=Math.max(Math.max(...xs)-Math.min(...xs),Math.max(...ys)-Math.min(...ys),.5),pad=span*.18;svg.setAttribute('viewBox',[Math.min(...xs)-pad,Math.min(...ys)-pad,Math.max(...xs)-Math.min(...xs)+2*pad,Math.max(...ys)-Math.min(...ys)+2*pad].join(' '));const title=document.createElementNS(ns,'title'),componentText=geometry.components.map(component=>component.index+' ('+component.x.toFixed(2)+', '+component.y.toFixed(2)+')').join(', ');title.textContent='Dashed gray: top-1 component '+geometry.topIndex+' (unit length), with other top components projected from its tip · Yellow: full output vector · plane coordinates are in top-1 norm units · output ('+result.x.toFixed(2)+', '+result.y.toFixed(2)+') · other components: '+componentText;svg.append(title);function segment(x1,y1,x2,y2,className,color){const line=document.createElementNS(ns,'line');line.classList.add(className);line.setAttribute('x1',x1);line.setAttribute('y1',-y1);line.setAttribute('x2',x2);line.setAttribute('y2',-y2);svg.append(line);const tip=document.createElementNS(ns,'circle');tip.setAttribute('cx',x2);tip.setAttribute('cy',-y2);tip.setAttribute('r',span*.045);tip.setAttribute('fill',color);svg.append(tip)}segment(0,0,result.x,result.y,'fan-result','#e6a15c');segment(0,0,1,0,'fan-component','#9aa0a6');for(const component of geometry.components)segment(1,0,1+component.x,component.y,'fan-component','#9aa0a6');return svg}
function targetHeading(item){const heading=document.createElement('h2');heading.className='target-heading';if(item.view.startsWith('residual-'))heading.append(residualTriangle(item.view,item.residualGeometry));else if((item.view==='attention-output'||item.view==='mlp-output')&&item.componentGeometry)heading.append(componentFan(item.componentGeometry));const labels=Object.fromEntries([...baseViews,...icaViews]),label=document.createElement('span'),layer=Number.isInteger(item.transformerLayer)?' [layer '+item.transformerLayer+']':'';label.textContent=(labels[item.view]||item.view)+layer;heading.append(label);if(item.isFinalLayer){const finalIcon=document.createElement('span');finalIcon.className='final-layer-icon';finalIcon.textContent='⊥';finalIcon.title='Final transformer layer';finalIcon.setAttribute('aria-label','Final transformer layer');heading.append(finalIcon)}if(Number.isFinite(item.vectorNorm)){const magnitude=document.createElement('span');magnitude.className='target-magnitude';const residual=item.view.includes('residual-');magnitude.textContent=(residual?'|x| ':'|Δx| ')+item.vectorNorm.toFixed(2);magnitude.title=(residual?'residual stream norm |x|':'residual update norm |Δx|')+(item.icaBasis?' · decomposed with this layer’s resid_post ICA basis':'');heading.append(magnitude)}return heading}
function render(){if(!snapshot)return;const active=snapshot.items.find(item=>!item.error&&Number.isInteger(item.tokenPosition));currentToken.textContent=active?'Token '+active.tokenPosition:'Token —';currentToken.title=active&&typeof active.token==='string'?'Current token: '+JSON.stringify(active.token):'Current token position';const light=snapshot.lensTheme==='light';liveLightTheme.checked=light;document.body.classList.toggle('light-report',light);document.body.classList.toggle('dark-report',!light);context.className='';context.textContent='';neurons.replaceChildren();for(const item of snapshot.items){const residual=item.view.includes('residual-'),group=document.createElement('section');group.className='lens-group'+(residual?'':' submodule');group.dataset.view=item.view;group.append(targetHeading(item));if(item.heads&&!item.error)renderHeadCards(item,group);else renderSnapshot(item,group);neurons.append(group)}}
function setBusy(value){waiting.hidden=!value}
function requestRefresh(force=false){setBusy(true);context.className='';context.textContent='Loading…';vscode.postMessage({type:'refresh',modes:availableModes.length?[...selectedModes]:null,views:[...selectedViews],force})}
function standalonePage(){const clone=document.documentElement.cloneNode(true),computed=getComputedStyle(document.documentElement),variables=['--vscode-font-family','--vscode-foreground','--vscode-descriptionForeground','--vscode-editor-background','--vscode-sideBar-background','--vscode-panel-border','--vscode-dropdown-foreground','--vscode-dropdown-background','--vscode-dropdown-border','--vscode-button-background','--vscode-button-foreground','--vscode-button-hoverBackground','--vscode-errorForeground','--vscode-badge-background','--vscode-badge-foreground','--vscode-focusBorder'];for(const variable of variables){const value=computed.getPropertyValue(variable);if(value)clone.style.setProperty(variable,value)}const clonedMode=clone.querySelector('#mode');if(clonedMode)for(const option of clonedMode.options)option.toggleAttribute('selected',option.value===mode.value);for(const box of clone.querySelectorAll('#viewMenu input'))box.toggleAttribute('checked',selectedViews.has(box.value));clone.querySelectorAll('script,meta[http-equiv="Content-Security-Policy"],#waiting,#icaMenu,#exportMd,#refresh').forEach(node=>node.remove());const fadeLabel=document.createElement('label');fadeLabel.className='report-toggle';fadeLabel.innerHTML='<input id="fadePast" type="checkbox" checked> Fade past blocks';const themeLabel=document.createElement('label');themeLabel.className='report-toggle';themeLabel.innerHTML='<input id="lightTheme" type="checkbox"> Light theme';const lightTheme=themeLabel.querySelector('input');lightTheme.toggleAttribute('checked',snapshot&&snapshot.reportTheme==='light');clone.querySelector('.actions').prepend(fadeLabel,themeLabel);let title=clone.querySelector('title');if(!title){title=document.createElement('title');clone.querySelector('head').prepend(title)}title.textContent='llmdbg Lens Snapshot';const offline=document.createElement('script');offline.textContent="const button=document.getElementById('viewButton'),menu=document.getElementById('viewMenu'),mode=document.getElementById('mode'),fadePast=document.getElementById('fadePast'),lightTheme=document.getElementById('lightTheme'),groups=[...document.querySelectorAll('.lens-group[data-view]')],boxes=[...menu.querySelectorAll('input')];function updateBlocks(){let count=0;for(const box of boxes){const group=groups.find(item=>item.dataset.view===box.value);box.disabled=!group;if(group){group.hidden=!box.checked;if(box.checked)count++}}button.textContent=count+' block'+(count===1?'':'s')+' selected ▾'}function updateLens(){for(const row of document.querySelectorAll('.readout'))row.hidden=mode.value!=='all'&&!row.classList.contains('readout-'+mode.value)}function updateFade(){document.body.classList.toggle('unfade-past',!fadePast.checked)}function updateTheme(){document.body.classList.toggle('light-report',lightTheme.checked);document.body.classList.toggle('dark-report',!lightTheme.checked)}button.onclick=()=>{menu.hidden=!menu.hidden};for(const box of boxes)box.onchange=updateBlocks;mode.onchange=updateLens;fadePast.onchange=updateFade;lightTheme.onchange=updateTheme;document.addEventListener('click',event=>{if(!event.target.closest('.view-picker'))menu.hidden=true});updateBlocks();updateLens();updateFade();updateTheme();";clone.querySelector('body').append(offline);return '<!doctype html>\\n'+clone.outerHTML}
const legacyStandalonePage=standalonePage;standalonePage=()=>{const clone=document.documentElement.cloneNode(true),computed=getComputedStyle(document.documentElement),variables=['--vscode-font-family','--vscode-foreground','--vscode-descriptionForeground','--vscode-editor-background','--vscode-sideBar-background','--vscode-panel-border','--vscode-dropdown-foreground','--vscode-dropdown-background','--vscode-dropdown-border','--vscode-button-background','--vscode-button-foreground','--vscode-button-hoverBackground','--vscode-errorForeground','--vscode-badge-background','--vscode-badge-foreground','--vscode-focusBorder'];for(const variable of variables){const value=computed.getPropertyValue(variable);if(value)clone.style.setProperty(variable,value)}clone.querySelectorAll('script,meta[http-equiv="Content-Security-Policy"],#waiting,#icaMenu,#exportMd,#refresh,.lens-picker').forEach(node=>node.remove());const fadeLabel=document.createElement('label');fadeLabel.className='report-toggle';fadeLabel.innerHTML='<input id="fadePast" type="checkbox" checked> Fade past blocks';const themeLabel=document.createElement('label');themeLabel.className='report-toggle';themeLabel.innerHTML='<input id="lightTheme" type="checkbox"> Light theme';clone.querySelector('.actions').prepend(fadeLabel,themeLabel);let title=clone.querySelector('title');if(!title){title=document.createElement('title');clone.querySelector('head').prepend(title)}title.textContent='llmdbg Lens Snapshot';return '<!doctype html>\\n'+clone.outerHTML};const standalonePageBase=standalonePage;standalonePage=()=>standalonePageBase().replace(/<label class="report-toggle live-only">[\\s\\S]*?<\\/label>/g,'');
window.addEventListener('message',event=>{const data=event.data;if(data.type==='busy'){setBusy(Boolean(data.busy));return}if(data.type==='reset'){setBusy(true);for(const key of Object.keys(saved))delete saved[key];snapshot=undefined;neurons.replaceChildren();context.textContent='Loading…';return}if(data.type==='error'){setBusy(false);context.className='error';context.textContent=data.message;neurons.replaceChildren();pendingSave=undefined;return}if(data.type==='lens'){icaSupported=Boolean(data.icaSupported);rebuildLensOptions(data.residualToVocabularyLenses,data.selectedModes);if(Array.isArray(data.selectedViews))selectedViews=new Set(data.selectedViews);if(!icaSupported)for(const [value] of icaViews)selectedViews.delete(value);rebuildViewMenu();const modeKey=(data.selectedModes||[]).join(',')||'none';data.items=data.items.map(item=>{const latestKey=modeKey+'|'+item.view;if(!item.error){saved[latestKey]=item;return item}const old=saved[latestKey];return old?{...old,saved:true}:item});const pre=data.items.find(item=>item.view==='residual-pre'&&!item.error);const activeTransformerLayer=pre&&pre.transformerLayer;if(Number.isInteger(activeTransformerLayer)){data.items=data.items.map(item=>!item.error&&Number.isInteger(item.transformerLayer)&&item.transformerLayer!==activeTransformerLayer?{...item,saved:true}:item)}snapshot=data;render();setBusy(false);if(pendingSave&&selectedModes.size===availableModes.length){const first=snapshot.items.find(item=>!item.error),restore=pendingSave;vscode.postMessage({type:'exportPage',html:standalonePage(),layer:first&&first.transformerLayer});pendingSave=undefined;selectedModes=new Set(restore.modes);selectedViews=new Set(restore.views);rebuildLensOptions(data.residualToVocabularyLenses,[...selectedModes]);rebuildViewMenu();requestRefresh()}}});lensButton.onclick=()=>{const open=lensMenu.hidden;lensMenu.hidden=!open;lensButton.setAttribute('aria-expanded',String(open))};viewButton.onclick=()=>{const open=viewMenu.hidden;viewMenu.hidden=!open;viewButton.setAttribute('aria-expanded',String(open))};document.addEventListener('click',event=>{if(!event.target.closest('.view-picker')){viewMenu.hidden=true;viewButton.setAttribute('aria-expanded','false')}if(!event.target.closest('.lens-picker')){lensMenu.hidden=true;lensButton.setAttribute('aria-expanded','false')}if(!event.target.closest('.ica-menu'))icaMenu.hidden=true});removeIca.onclick=()=>{if(!pendingIntervention)return;pendingIntervention.card.classList.add('intervention-removed');icaMenu.hidden=true;setBusy(true);vscode.postMessage({type:pendingIntervention.type,view:pendingIntervention.view,component:pendingIntervention.component});pendingIntervention=undefined};document.getElementById('exportMd').onclick=()=>{if(snapshot&&!pendingSave){pendingSave={modes:[...selectedModes],views:[...selectedViews]};selectedModes=new Set(availableModes.map(item=>item[0]));selectedViews=new Set(orderedViews.map(item=>item[0]).filter(value=>icaSupported||!value.startsWith('ica-')));rebuildLensOptions(snapshot.residualToVocabularyLenses,[...selectedModes]);rebuildViewMenu();requestRefresh()}};document.getElementById('refresh').onclick=()=>requestRefresh(true);rebuildViewMenu();requestRefresh();
window.addEventListener('message',event=>{if(event.data.type==='cosineResult')showCosineRelativeRight(event.data.result);if(event.data.type==='allCosineResult')showAllCosines(event.data.result);if(event.data.type==='tokenCosineResult')showTokenCosines(event.data.result);if(event.data.type==='patchReferenceSet')showPatchReference(event.data);if(event.data.type==='cosineReferenceSet'){tokenCosineReference=null;allCosineReference={id:'pinned',label:event.data.label,norm:event.data.norm,tokenPosition:event.data.tokenPosition};showCosineReference(allCosineReference);requestAllCosines()}if(event.data.type==='tokenReferenceSet'){allCosineReference=null;tokenCosineReference={tokenPosition:event.data.tokenPosition};showCosineReference({label:'Token reference',tokenPosition:event.data.tokenPosition});requestTokenCosines()}if(event.data.type==='lens'){if(event.data.patchReference)showPatchReference(event.data.patchReference);if(event.data.tokenReference){tokenCosineReference=event.data.tokenReference;allCosineReference=null;showCosineReference({label:'Token reference',tokenPosition:event.data.tokenReference.tokenPosition})}else if(event.data.cosineReference&&!allCosineReference){allCosineReference={id:'pinned',...event.data.cosineReference};showCosineReference(allCosineReference)}if(tokenCosineReference)requestTokenCosines();else if(allCosineReference)requestAllCosines()}if(event.data.type==='reset'){cosineSelection.a=null;cosineSelection.b=null;cosineResult.hidden=true;document.querySelectorAll('.all-cosine-angle').forEach(node=>node.remove())}});
neurons.addEventListener('contextmenu',event=>{if(!snapshot)return;const group=event.target.closest('.lens-group'),heading=event.target.closest('.target-heading'),componentTarget=event.target.closest('.ica-component-id,.neuron-badge');if(!group||(!heading&&!componentTarget))return;const item=snapshot.items.find(candidate=>candidate.view===group.dataset.view);if(!item||item.error)return;let vector,intervention,patchTarget;if(heading&&item.measurementId){vector={id:item.measurementId,label:(item.vectorLabel||item.view)+' [layer '+item.transformerLayer+']'};if(!item.saved&&['residual-pre','residual-mid','residual-post','attention-output','mlp-output'].includes(item.view))patchTarget={view:item.view}}else if(componentTarget){const cards=[...group.querySelectorAll(':scope > .card')],card=componentTarget.closest('.card'),index=cards.indexOf(card),component=item.icaComponents?item.icaComponents[index]:item.neurons&&item.neurons[index];if(!component||!component.measurementId)return;const isIca=Boolean(item.icaComponents),kind=isIca?'ICA component':(item.componentLabel||'component');vector={id:component.measurementId,label:isIca?'Residual ICA Lens '+item.view.replace('ica-residual-','')+' L'+item.transformerLayer+':C'+component.index:kind+' L'+item.transformerLayer+':'+component.index};if(!item.saved&&!component.removed){if(isIca&&item.view.startsWith('ica-residual-'))intervention={type:'removeIca',view:item.view,component:component.index,card};if(!isIca)intervention={type:'removeComponent',view:item.view,component:component.index,card}}}if(!vector)return;event.stopImmediatePropagation();openVectorMenu(event,vector,intervention,patchTarget)},true);
patchCopy.onclick=copyPatchVector;patchPaste.onclick=pastePatchVector;measureA.onclick=()=>chooseCosine('a');measureB.onclick=()=>chooseCosine('b');compareAll.onclick=chooseCompareAll;cosineClose.onclick=()=>{cosineResult.hidden=true;cosineSelection.a=null;cosineSelection.b=null;cosineAngle.replaceChildren();cosineValues.replaceChildren()};patchReference.oncontextmenu=event=>{event.preventDefault();patchMenu.style.left=Math.min(event.clientX,window.innerWidth-100)+'px';patchMenu.style.top=Math.min(event.clientY,window.innerHeight-40)+'px';patchMenu.hidden=false};unsetPatch.onclick=clearPatchReference;compareReference.oncontextmenu=event=>{event.preventDefault();referenceMenu.style.left=Math.min(event.clientX,window.innerWidth-100)+'px';referenceMenu.style.top=Math.min(event.clientY,window.innerHeight-40)+'px';referenceMenu.hidden=false};unsetReference.onclick=clearCompareAll;document.addEventListener('click',event=>{if(!event.target.closest('#referenceMenu'))referenceMenu.hidden=true;if(!event.target.closest('#patchMenu'))patchMenu.hidden=true});liveFadePast.onchange=()=>document.body.classList.toggle('unfade-past',!liveFadePast.checked);liveLightTheme.onchange=()=>{const light=liveLightTheme.checked;document.body.classList.toggle('light-report',light);document.body.classList.toggle('dark-report',!light);if(snapshot){snapshot.lensTheme=light?'light':'dark';snapshot.reportTheme=snapshot.lensTheme}vscode.postMessage({type:'setLensTheme',theme:light?'light':'dark'})};
currentToken.oncontextmenu=event=>{event.preventDefault();setTokenReference.hidden=Boolean(tokenCosineReference);unsetTokenReference.hidden=!tokenCosineReference;tokenMenu.style.left=Math.min(event.clientX,window.innerWidth-230)+'px';tokenMenu.style.top=Math.min(event.clientY,window.innerHeight-42)+'px';tokenMenu.hidden=false};setTokenReference.onclick=()=>{const active=snapshot&&snapshot.items.find(item=>!item.error&&Number.isInteger(item.tokenPosition)),targets=correspondingTokenTargets().map(({key,id})=>({key,id}));if(active&&targets.length){setBusy(true);vscode.postMessage({type:'setTokenReference',tokenPosition:active.tokenPosition,targets})}tokenMenu.hidden=true};unsetTokenReference.onclick=()=>{tokenCosineReference=null;compareReference.hidden=true;document.querySelectorAll('.all-cosine-angle').forEach(node=>node.remove());vscode.postMessage({type:'unsetTokenReference'});tokenMenu.hidden=true};
const cosineDragHandle=cosineResult.querySelector('.cosine-result-head');let cosineDrag=null;cosineDragHandle.onpointerdown=event=>{if(event.button!==0||event.target.closest('button'))return;const rect=cosineResult.getBoundingClientRect();cosineDrag={pointerId:event.pointerId,offsetX:event.clientX-rect.left,offsetY:event.clientY-rect.top};cosineResult.style.left=rect.left+'px';cosineResult.style.top=rect.top+'px';cosineResult.style.right='auto';cosineDragHandle.setPointerCapture(event.pointerId);event.preventDefault()};cosineDragHandle.onpointermove=event=>{if(!cosineDrag||event.pointerId!==cosineDrag.pointerId)return;const rect=cosineResult.getBoundingClientRect(),left=Math.max(0,Math.min(window.innerWidth-rect.width,event.clientX-cosineDrag.offsetX)),top=Math.max(0,Math.min(window.innerHeight-rect.height,event.clientY-cosineDrag.offsetY));cosineResult.style.left=left+'px';cosineResult.style.top=top+'px'};function stopCosineDrag(event){if(!cosineDrag||event.pointerId!==cosineDrag.pointerId)return;cosineDragHandle.releasePointerCapture(event.pointerId);cosineDrag=null}cosineDragHandle.onpointerup=stopCosineDrag;cosineDragHandle.onpointercancel=stopCosineDrag;
</script></body></html>`;
}

function runLoggedCommand(output, executable, args, options = {}) {
  output.appendLine(`\n> ${executable} ${args.join(" ")}`);
  return new Promise((resolve, reject) => {
    const child = childProcess.spawn(executable, args, {
      ...options,
      stdio: ["ignore", "pipe", "pipe"]
    });
    child.stdout.on("data", (data) => output.append(data.toString()));
    child.stderr.on("data", (data) => output.append(data.toString()));
    child.on("error", reject);
    child.on("close", (code) => {
      if (code === 0) resolve();
      else reject(new Error(`${executable} exited with code ${code}`));
    });
  });
}

async function createRuntime(context, runtimeRoot, recipe, output) {
  await fs.promises.mkdir(path.dirname(runtimeRoot), { recursive: true });
  output.clear();
  output.appendLine(`Creating llmdbg private runtime at ${runtimeRoot}`);
  output.appendLine(`Profile: ${recipe.profile}; backend: ${recipe.id}; torch ${recipe.torch}; Transformers ${recipe.transformers}`);
  output.show(true);
  await vscode.window.withProgress(
    {
      location: vscode.ProgressLocation.Notification,
      title: "llmdbg: Creating private debugger environment",
      cancellable: false
    },
    async () => {
      // CUDA distributions can install overlapping NVIDIA library paths. Never
      // repair a changed recipe in place: preserve it and build a clean venv.
      if (fs.existsSync(runtimeRoot)) {
        const backup = `${runtimeRoot}.backup-${Date.now()}`;
        await fs.promises.rename(runtimeRoot, backup);
        output.appendLine(`Previous runtime preserved at ${backup}`);
      }
      const python = runtimePython(runtimeRoot);
      if (!fs.existsSync(python)) {
        await runLoggedCommand(
          output,
          uvExecutable(),
          ["venv", "--python", recipe.python, runtimeRoot],
          {}
        );
      }
      const lock = path.join(path.dirname(runtimeRoot), "locks", `${recipe.key}.txt`);
      if (fs.existsSync(lock)) {
        await runLoggedCommand(output, uvExecutable(), ["pip", "sync", "--python", python, lock,
          ...(recipe.index ? ["--extra-index-url", recipe.index, "--index-strategy", "unsafe-best-match"] : [])]);
      } else {
      await runLoggedCommand(
        output,
        uvExecutable(),
        [
          "pip", "install",
          "--python", python,
          "--upgrade", "--reinstall",
          ...(recipe.index ? ["--index-url", recipe.index] : []),
          `torch==${recipe.torch}`,
          `torchvision==${recipe.torchvision}`
        ],
        {}
      );
      await runLoggedCommand(
        output,
        uvExecutable(),
        [
          "pip", "install",
          "--python", python,
          "--upgrade",
          `transformers==${recipe.transformers}`,
          `huggingface-hub==${recipe.hub}`,
          `safetensors==${recipe.safetensors}`,
          `matplotlib==${recipe.matplotlib}`
        ],
        {}
      );
      await runLoggedCommand(
        output,
        uvExecutable(),
        ["pip", "install", "--python", python, "--upgrade", "--no-deps", `gb10-load-llm==${recipe.loader}`],
        {}
      );
      }
      output.appendLine("\nVerifying the installed runtime...");
      const { stdout } = await execFile(python, ["-c", RUNTIME_PROBE], { maxBuffer: 1024 * 1024 });
      const probe = JSON.parse(stdout.trim());
      if (!probe.python.startsWith(`${recipe.python}.`) || probe.torch.split("+")[0] !== recipe.torch
          || probe.torchvision.split("+")[0] !== recipe.torchvision || probe.transformers !== recipe.transformers) {
        throw new Error(`Runtime dependency verification failed: ${JSON.stringify(probe)}`);
      }
      if (recipe.id.startsWith("cu") && !probe.cuda_available) {
        throw new Error(
          `Installed torch ${probe.torch} (${recipe.id}), but CUDA verification failed. ` +
          "Check the NVIDIA driver or choose another llmdbg.torchBackend."
        );
      }
      const { stdout: frozen } = await execFile(uvExecutable(), ["pip", "freeze", "--python", python]);
      await fs.promises.writeFile(path.join(runtimeRoot, "requirements.resolved.txt"), frozen);
      await fs.promises.mkdir(path.dirname(lock), { recursive: true });
      await fs.promises.writeFile(lock, frozen, { flag: "wx" }).catch(error => { if (error.code !== "EEXIST") throw error; });
      await fs.promises.writeFile(path.join(runtimeRoot, "profile.json"), JSON.stringify(recipe, null, 2));
      await fs.promises.writeFile(path.join(runtimeRoot, ".llmdbg-runtime"), `${runtimeMarker(recipe)}\n`);
      output.appendLine("llmdbg private runtime is ready.");
    }
  );
}

function activate(context) {
  require('./concept-trace').registerConceptTrace(context, vscode);
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

  const runtimesRoot = path.join(context.globalStorageUri.fsPath, "runtimes");
  const plotRoot = path.join(context.globalStorageUri.fsPath, "plots");
  const runtimeOutput = vscode.window.createOutputChannel("llmdbg");
  context.subscriptions.push(runtimeOutput);
  const runtimePromises = new Map();
  let hostProcess;
  let hostPromise;
  let hostSource;
  let hostPython;
  let hostLensConfiguration;
  let vectorPanel;
  let decoderPanel;
  let promptTokensPanel;
  let neuronLensPanel;
  let attentionContributionsPanel;
  let attentionImagesPanel;
  let pinnedCosineReference;
  let pinnedTokenReference;
  let pinnedPatchVector;
  let neuronLensModes;
  let neuronLensViews = [
    "residual-pre", "attention-head-values", "attention-heads", "attention-output", "residual-mid",
    "neurons", "mlp-output", "residual-post"
  ];
  let neuronLensIcaInitialized = false;
  let neuronLensRevision = 0;
  let neuronLensRefreshRequest = 0;
  let neuronLensExecutionReady = false;
  const neuronLensItemCache = new Map();
  let residentModel;

  const toolPanelContextKeys = {
    vector: "llmdbg.vectorPanelOpen",
    decoder: "llmdbg.decoderPanelOpen",
    promptTokens: "llmdbg.promptTokensPanelOpen",
    neuronLens: "llmdbg.neuronLensPanelOpen",
  };

  function setToolPanelOpen(kind, open) {
    void vscode.commands.executeCommand("setContext", toolPanelContextKeys[kind], open);
  }

  function closeToolPanel(kind) {
    let panel;
    if (kind === "vector") {
      panel = vectorPanel;
      vectorPanel = undefined;
    } else if (kind === "decoder") {
      panel = decoderPanel;
      decoderPanel = undefined;
    } else if (kind === "promptTokens") {
      panel = promptTokensPanel;
      promptTokensPanel = undefined;
    } else if (kind === "neuronLens") {
      panel = neuronLensPanel;
      neuronLensPanel = undefined;
    }
    setToolPanelOpen(kind, false);
    panel?.dispose();
  }

  for (const kind of Object.keys(toolPanelContextKeys)) setToolPanelOpen(kind, false);

  function ownToolPanelColumn(currentPanel, peerPanels) {
    const occupied = new Set(
      peerPanels.map(panel => panel?.viewColumn).filter(Number.isInteger)
    );
    if (Number.isInteger(currentPanel?.viewColumn) && !occupied.has(currentPanel.viewColumn)) {
      return currentPanel.viewColumn;
    }
    if (!occupied.size) return vscode.ViewColumn.Beside;
    const highest = Math.max(...occupied);
    if (highest < vscode.ViewColumn.Nine) return highest + 1;
    for (let column = vscode.ViewColumn.One; column <= vscode.ViewColumn.Nine; column += 1) {
      if (!occupied.has(column)) return column;
    }
    return vscode.ViewColumn.Beside;
  }

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
    // Add built-ins on upgrades too, while preserving user-edited plot files.
    for (const [name, code] of [["Line plot", DEFAULT_PLOT_CODE], ["Histogram", HISTOGRAM_PLOT_CODE], ["Heatmap", HEATMAP_PLOT_CODE]]) {
      const file = plotFileName(name);
      if (!files.some(existing => existing.toLowerCase() === file.toLowerCase())) {
        await fs.promises.writeFile(path.join(plotRoot, file), code, { flag: "wx" }).catch(error => {
          if (error.code !== "EEXIST") throw error;
        });
        files.push(file);
      }
    }
    for (const file of files) {
      const target = path.join(plotRoot, file);
      const code = await fs.promises.readFile(target, "utf8");
      let normalized = plotFunctionCode(code);
      if (file.toLowerCase() === "heatmap.plot.py") {
        if (!normalized.includes("MaxNLocator")) {
          normalized = normalized.replace(
            "def plot(vector, vector_name, plt, np):\n",
            "def plot(vector, vector_name, plt, np):\n    from matplotlib.ticker import MaxNLocator\n"
          ).replace(
            '    ax.set_xlabel("column")',
            '    ax.xaxis.set_major_locator(MaxNLocator(integer=True))\n    ax.yaxis.set_major_locator(MaxNLocator(integer=True))\n    ax.set_xlabel("column")'
          );
        }
        normalized = normalized.replace(
          '    matrix = np.asarray(vector, dtype=float).reshape(original_shape).squeeze()\n    if matrix.ndim != 2:',
          '    matrix = np.asarray(vector, dtype=float).reshape(original_shape).squeeze()\n    if matrix.ndim == 1:\n        matrix = np.expand_dims(matrix, axis=0)\n    if matrix.ndim != 2:'
        ).replace("Heatmap needs two non-singleton dimensions after squeeze; got ", "Heatmap needs one or two non-singleton dimensions after squeeze; got ")
          .replace("Select a tensor slice with two dimensions.", "Select a tensor slice with one or two dimensions.");
        normalized = normalized.replace(
          '    if matrix.ndim == 1:\n        matrix = np.expand_dims(matrix, axis=0)',
          '    if matrix.ndim == 1:\n        if heatmap_axis is None:\n            raise PlotShapeError("Heatmap squeezed shape is " + str(matrix.shape) + ". Choose where to add a dimension:")\n        matrix = np.expand_dims(matrix, axis=heatmap_axis)'
        );
      }
      if (normalized !== code) {
        await fs.promises.writeFile(target, normalized);
      }
    }
    return files.sort().map(file => ({
      file,
      name: file.toLowerCase() === "histogram.plot.py" ? "Histogram (all values)" : file.toLowerCase() === "heatmap.plot.py" ? "Heatmap (squeezed)" : file.slice(0, -".plot.py".length).replace(/[-_]+/g, " ").replace(/\b\w/g, character => character.toUpperCase())
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

  async function showExpressionDecoder() {
    if (!vscode.debug.activeDebugSession || vscode.debug.activeDebugSession.type !== "llmdbg") {
      void vscode.window.showErrorMessage("Start and pause an llmdbg session before decoding a tensor.");
      return;
    }
    if (decoderPanel) { decoderPanel.reveal(decoderPanel.viewColumn || vscode.ViewColumn.Beside, true); return; }
    const panel = decoderPanel = vscode.window.createWebviewPanel(
      "llmdbg.expressionDecoder", "llmdbg Tensor Decoder",
      { viewColumn: ownToolPanelColumn(undefined, [neuronLensPanel, vectorPanel, promptTokensPanel, attentionContributionsPanel, attentionImagesPanel]), preserveFocus: true },
      { enableScripts: true, retainContextWhenHidden: true }
    );
    setToolPanelOpen("decoder", true);
    panel.webview.html = expressionDecoderHtml(panel.webview, vscode.workspace.getConfiguration("llmdbg").get("lensTheme", "dark"));
    panel.webview.onDidReceiveMessage(async message => {
      if (message.type !== "decode") return;
      try {
        const session = vscode.debug.activeDebugSession;
        if (!session || session.type !== "llmdbg") throw new Error("Start and pause an llmdbg session first.");
        const data = await session.customRequest("llmdbg/decodeExpression", {
          expression: String(message.expression),
          loop: message.loop,
        });
        data.lensTheme = vscode.workspace.getConfiguration("llmdbg").get("lensTheme", "dark");
        await panel.webview.postMessage({ type: "result", requestId: message.requestId, data });
      } catch (error) { await panel.webview.postMessage({ type: "error", requestId: message.requestId, message: error.message }); }
    }, null, context.subscriptions);
    panel.onDidDispose(() => {
      if (decoderPanel !== panel) return;
      decoderPanel = undefined;
      setToolPanelOpen("decoder", false);
    }, null, context.subscriptions);
  }

  async function showVectorPlot() {
    const session = vscode.debug.activeDebugSession;
    if (!session || session.type !== "llmdbg") {
      vscode.window.showErrorMessage("Start and pause an llmdbg session before plotting a vector.");
      return;
    }
    if (!vectorPanel) {
      const targetColumn = ownToolPanelColumn(vectorPanel, [neuronLensPanel, promptTokensPanel, decoderPanel]);
      vectorPanel = vscode.window.createWebviewPanel(
        "llmdbg.vectorPlot",
        "llmdbg Tensor Plot",
        { viewColumn: targetColumn, preserveFocus: true },
        { enableScripts: true, retainContextWhenHidden: true }
      );
      setToolPanelOpen("vector", true);
      vectorPanel.webview.html = vectorPlotHtml(vectorPanel.webview);
      vectorPanel.webview.onDidReceiveMessage(async (message) => {
        if (message.type === "ready") {
          await sendPlotSnippets();
        } else if (message.type === "plot") {
          const active = vscode.debug.activeDebugSession;
          if (!active || active.type !== "llmdbg") {
            await vectorPanel.webview.postMessage({ type: "error", message: "Start and pause an llmdbg session first." });
            return;
          }
          try {
            const expression = String(message.expression || "residual_stream");
            const code = await fs.promises.readFile(await plotPath(String(message.file || "")), "utf8");
            const result = await active.customRequest("llmdbg/plot", { expression, code, shapeAction: message.shapeAction });
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
      const panel = vectorPanel;
      panel.onDidDispose(() => {
        if (vectorPanel !== panel) return;
        vectorPanel = undefined;
        setToolPanelOpen("vector", false);
      }, null, context.subscriptions);
    } else {
      vectorPanel.reveal(vectorPanel.viewColumn || vscode.ViewColumn.Beside, true);
    }
    await sendPlotSnippets();
  }

  async function refreshPromptTokens() {
    if (!promptTokensPanel) return;
    const active = vscode.debug.activeDebugSession;
    if (!active || active.type !== "llmdbg") return;
    try {
      const data = await active.customRequest("llmdbg/promptTokens", {});
      data.images = await promptImagePreviews(data.images);
      promptTokensPanel.webview.html = promptTokensHtml(promptTokensPanel.webview, data);
    } catch (error) {
      vscode.window.showErrorMessage(`llmdbg: ${error.message}`);
    }
  }

  async function showPromptTokens() {
    const session = vscode.debug.activeDebugSession;
    if (!session || session.type !== "llmdbg") {
      vscode.window.showErrorMessage("Start and pause an llmdbg session before viewing prompt tokens.");
      return;
    }
    const promptColumn = ownToolPanelColumn(
      promptTokensPanel, [neuronLensPanel, vectorPanel, decoderPanel]
    );
    if (!promptTokensPanel) {
      promptTokensPanel = vscode.window.createWebviewPanel(
        "llmdbg.promptTokens", "llmdbg Prompt Tokens",
        { viewColumn: promptColumn, preserveFocus: true },
        { enableScripts: true, retainContextWhenHidden: true }
      );
      setToolPanelOpen("promptTokens", true);
      const panel = promptTokensPanel;
      panel.onDidDispose(() => {
        if (promptTokensPanel !== panel) return;
        promptTokensPanel = undefined;
        setToolPanelOpen("promptTokens", false);
      }, null, context.subscriptions);
    } else {
      promptTokensPanel.reveal(promptColumn, true);
    }
    await refreshPromptTokens();
  }

  function mergeLensSnapshots(entries, residualLenses) {
    const successful = entries.filter(entry => entry.item && !entry.item.error);
    if (!successful.length) return entries[0]?.item;
    const base = structuredClone(successful[0].item);
    const labels = new Map(residualLenses.map(lens => [`lens:${lens.id}`, lens]));
    function readoutHint(view, layer, method, fallback, isFinalLayer) {
      if(view === 'attention-heads' && method !== 'logit') return 'Mid layer '+layer+' map for this bias-free projected head output'+(fallback?' (Post fallback)':'')+'.';
      if (method === "logit") return "Direct vocabulary projection of this vector; no learned transport map is used.";
      const name = method === "j" ? "J-Lens" : method === "r" ? "R-Lens" : "J++ Lens";
      const plainView = String(view || "").replace(/^ica-/, "");
      let site = "Post";
      let siteLayer = layer;
      if (plainView === "residual-pre") {
        if (layer === 0) return `${name} uses the fitted embedding-output map.`;
        siteLayer = layer - 1;
      } else if (["attention-neurons", "attention-output", "residual-mid"].includes(plainView)) {
        site = "Mid";
      }
      if (fallback) {
        if (isFinalLayer) return `${name} uses the exact final Post identity as a fallback for Mid layer ${layer}; this approximation ignores the current layer's MLP response.`;
        return `${name} uses the same-layer Post map as a fallback for Mid layer ${layer}; this approximation ignores the current layer's MLP response.`;
      }
      return `${name} uses the fitted ${site} layer ${siteLayer} map for this vector's downstream starting site.`;
    }
    function mergeNode(target, sources) {
      target.residualVocabularyReadouts = [];
      for (const entry of entries) {
        const source = entry.item && !entry.item.error ? sources.get(entry.mode) : undefined;
        if (entry.mode === "logit") {
          target.residualVocabularyReadouts.push({
            kind: "logit", label: "Logit Lens", tokens: source?.logitTokens || [],
            unavailable: entry.error,
            hint: readoutHint(base.view, base.transformerLayer, "logit", false, base.isFinalLayer)
          });
          continue;
        }
        const lens = labels.get(entry.mode);
        if (!lens) continue;
        const method = lens.method;
        const tokenKey = method === "j" ? "jLensTokens" : method === "r" ? "rLensTokens" : "jppLensTokens";
        const unavailableKey = method === "j" ? "jLensUnavailable" : method === "r" ? "rLensUnavailable" : "jppLensUnavailable";
        const fallbackKey = method === "j" ? "jLensFallback" : method === "r" ? "rLensFallback" : "jppLensFallback";
        const fallback = Boolean(source?.[fallbackKey]);
        target.residualVocabularyReadouts.push({
          kind: `${method}-lens`,
          label: fallback ? `${lens.label} · Post fallback` : lens.label,
          tokens: source?.[tokenKey] || [],
          unavailable: entry.error || source?.[unavailableKey],
          hint: readoutHint(base.view, base.transformerLayer, method, fallback, base.isFinalLayer)
        });
      }
      for (const childKey of ["neurons", "icaComponents", "heads", "channels"]) {
        if (!Array.isArray(target[childKey])) continue;
        target[childKey].forEach((child, index) => {
          const childSources = new Map(successful.map(entry => [entry.mode, sources.get(entry.mode)?.[childKey]?.[index]]));
          mergeNode(child, childSources);
        });
      }
      delete target.activeResidualVocabularyLens;
    }
    mergeNode(base, new Map(successful.map(entry => [entry.mode, entry.item])));
    return base;
  }

  async function refreshNeuronLens(modes = neuronLensModes, views = neuronLensViews, force = false) {
    if (!neuronLensPanel) return;
    const revision = neuronLensRevision;
    const refreshRequest = ++neuronLensRefreshRequest;
    const stale = () => revision !== neuronLensRevision || refreshRequest !== neuronLensRefreshRequest || !neuronLensPanel || !neuronLensExecutionReady;
    await neuronLensPanel.webview.postMessage({ type: "busy", busy: true });
    if (!neuronLensExecutionReady) return;
    const active = vscode.debug.activeDebugSession;
    if (!active || active.type !== "llmdbg") {
      await neuronLensPanel.webview.postMessage({ type: "error", message: "Start and pause an llmdbg session first." });
      return;
    }
    try {
      const support = await active.customRequest("llmdbg/icaSupport", {});
      const residualLensSupport = await active.customRequest(
        "llmdbg/residualToVocabularyLensSupport", {}
      );
      const residualLenses = Array.isArray(residualLensSupport.lenses)
        ? residualLensSupport.lenses
        : [];
      if (stale()) return;
      const availableModes = new Set(["logit", ...residualLenses.map(lens => `lens:${lens.id}`)]);
      if (!Array.isArray(modes)) modes = undefined;
      if (modes === undefined) modes = [residualLenses.length ? `lens:${residualLenses[0].id}` : "logit"];
      modes = modes.filter(mode => availableModes.has(mode));
      neuronLensModes = modes;
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
          ["attention-head-values", "Attention Head Value Lens"],
          ["delta-memory-write", "DeltaNet Memory Write Lens"],
          ["attention-heads", "Attention Head Lens"],
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
      const requestModes = modes.length ? modes : ["none"];
      const missing = [];
      for (const view of allowedViews) for (const mode of requestModes) {
        if (force || !neuronLensItemCache.has(`${revision}|${mode}|${view}`)) missing.push({ mode, view });
      }
      const entriesByKey = new Map();
      if (missing.length) {
        const batch = await active.customRequest("llmdbg/neuronLensBatch", { requests: missing });
        if (stale()) return;
        for (const entry of batch.items) {
          const key = `${revision}|${entry.mode}|${entry.view}`;
          entriesByKey.set(key, entry);
          if (entry.item) neuronLensItemCache.set(key, entry.item);
        }
      }
      for (const targetView of allowedViews) {
        const entries = [];
        for (const mode of requestModes) {
          const key = `${revision}|${mode}|${targetView}`;
          entries.push(entriesByKey.get(key) || { mode, item: neuronLensItemCache.get(key) });
        }
        const merged = mergeLensSnapshots(entries, residualLenses);
        const unavailable = entries.find(entry => entry.item?.unavailable)?.item;
        items.push(merged || (unavailable ? { ...unavailable, label: targets.get(targetView) } : { view: targetView, label: targets.get(targetView), error: entries[0]?.error || "Lens readout failed." }));
      }
      const lensTheme = vscode.workspace.getConfiguration("llmdbg").get("lensTheme", "dark");
      await neuronLensPanel.webview.postMessage({ type: "lens", selectedModes: modes, items, selectedViews: allowedViews, icaSupported: support.supported, residualToVocabularyLenses: residualLenses, lensTheme, reportTheme: lensTheme, cosineReference: pinnedCosineReference ? { label: pinnedCosineReference.label, norm: pinnedCosineReference.norm, tokenPosition: pinnedCosineReference.tokenPosition } : undefined, tokenReference: pinnedTokenReference ? { tokenPosition: pinnedTokenReference.tokenPosition } : undefined, patchReference: pinnedPatchVector ? { label: pinnedPatchVector.label, norm: pinnedPatchVector.norm, tokenPosition: pinnedPatchVector.tokenPosition, dimensions: pinnedPatchVector.dimensions } : undefined });
    } catch (error) {
      if (stale()) return;
      await neuronLensPanel.webview.postMessage({ type: "error", message: error.message });
    }
  }

  async function showAttentionTokenContributions(message, imagesOnly = false) {
    const active = vscode.debug.activeDebugSession;
    if (!active || active.type !== "llmdbg") {
      void vscode.window.showWarningMessage("Start and pause an llmdbg session first.");
      return;
    }
    try {
      const memory = message.type === 'deltaMemoryContributions';
      const request = () => active.customRequest(memory ? 'llmdbg/deltaMemoryContributions' : 'llmdbg/attentionTokenContributions', {
        layer: message.layer, channel: message.channel, head: message.head, tokenPosition: message.tokenPosition
      });
      const data = memory ? await vscode.window.withProgress({location:vscode.ProgressLocation.Notification,title:'Replaying prefix for DeltaNet memory contributions…',cancellable:false},request) : await request();
      data.images = await promptImagePreviews(data.images);
      if (imagesOnly && !data.images?.length) {
        void vscode.window.showWarningMessage("This prompt has no image inputs.");
        return;
      }
      let panel = imagesOnly ? attentionImagesPanel : attentionContributionsPanel;
      if (!panel) {
        panel = vscode.window.createWebviewPanel(
          imagesOnly ? "llmdbg.attentionImageMaps" : "llmdbg.attentionContributions",
          imagesOnly ? "llmdbg Image Patch Maps" : "llmdbg Token Contributions",
          { viewColumn: ownToolPanelColumn(undefined, [neuronLensPanel, vectorPanel, promptTokensPanel, attentionContributionsPanel, attentionImagesPanel]), preserveFocus: false },
          { enableScripts: true, retainContextWhenHidden: true }
        );
        if (imagesOnly) attentionImagesPanel = panel;
        else attentionContributionsPanel = panel;
        panel.onDidDispose(() => {
          if (imagesOnly && attentionImagesPanel === panel) attentionImagesPanel = undefined;
          else if (!imagesOnly && attentionContributionsPanel === panel) attentionContributionsPanel = undefined;
        }, null, context.subscriptions);
      } else {
        panel.reveal(panel.viewColumn);
      }
      panel.title = data.kind === 'delta-memory' ? `Head ${data.head} · DeltaNet Memory Contributions` : `Channel ${data.channel} · ${imagesOnly ? "Image Patch Maps" : "Token Contributions"}`;
      panel.webview.html = imagesOnly ? attentionImageMapsHtml(panel.webview, data) : attentionTokenContributionsHtml(panel.webview, data);
    } catch (error) {
      void vscode.window.showWarningMessage(`llmdbg: ${error.message}`);
    }
  }

  async function showNeuronLens(
    session = vscode.debug.activeDebugSession,
    refreshImmediately = true
  ) {
    if (!session || session.type !== "llmdbg") {
      vscode.window.showErrorMessage("Start and pause an llmdbg session before opening Lens.");
      return;
    }
    if (!neuronLensPanel) {
      neuronLensModes = undefined;
      neuronLensViews = [
        "residual-pre", "attention-head-values", "attention-heads", "attention-output", "residual-mid",
        "neurons", "mlp-output", "residual-post"
      ];
      neuronLensIcaInitialized = false;
      const targetColumn = ownToolPanelColumn(
        neuronLensPanel, [promptTokensPanel, vectorPanel, decoderPanel]
      );
      neuronLensPanel = vscode.window.createWebviewPanel(
        "llmdbg.neuronLens", "llmdbg Lens",
        { viewColumn: targetColumn, preserveFocus: true },
        { enableScripts: true, retainContextWhenHidden: true }
      );
      setToolPanelOpen("neuronLens", true);
      neuronLensPanel.webview.html = neuronLensHtml(neuronLensPanel.webview);
      neuronLensPanel.webview.onDidReceiveMessage(
        async message => {
          if (message.type === "attentionImageMaps") {
            await showAttentionTokenContributions(message, true);
            return;
          }
          if (message.type === "attentionTokenContributions" || message.type === 'deltaMemoryContributions') {
            await showAttentionTokenContributions(message);
            return;
          }
          if (message.type === "setPatchReference") {
            const active = vscode.debug.activeDebugSession;
            if (!active || active.type !== "llmdbg") return;
            try {
              pinnedPatchVector = await active.customRequest(
                "llmdbg/getMeasurementVector", { vector: message.vector }
              );
              await neuronLensPanel.webview.postMessage({
                type: "patchReferenceSet",
                label: pinnedPatchVector.label,
                norm: pinnedPatchVector.norm,
                tokenPosition: pinnedPatchVector.tokenPosition,
                dimensions: pinnedPatchVector.dimensions
              });
            } catch (error) {
              await neuronLensPanel.webview.postMessage({ type: "error", message: error.message });
            }
            return;
          }
          if (message.type === "unsetPatchReference") {
            pinnedPatchVector = undefined;
            return;
          }
          if (message.type === "pastePatch") {
            const active = vscode.debug.activeDebugSession;
            if (!active || active.type !== "llmdbg" || !pinnedPatchVector) return;
            try {
              const result = await active.customRequest("llmdbg/patchVector", {
                view: message.view,
                vector: pinnedPatchVector.vector,
                sourceLabel: pinnedPatchVector.label
              });
              if (result.log?.note) {
                void vscode.window.showInformationMessage(`Vector patched. Log saved to ${result.log.note}`);
              }
            } catch (error) {
              await neuronLensPanel.webview.postMessage({ type: "error", message: error.message });
            }
            return;
          }
          if (message.type === "setCosineReference") {
            const active = vscode.debug.activeDebugSession;
            if (!active || active.type !== "llmdbg") {
              await neuronLensPanel.webview.postMessage({ type: "error", message: "Start and pause an llmdbg session first." });
              return;
            }
            try {
              pinnedCosineReference = await active.customRequest(
                "llmdbg/getMeasurementVector", { vector: message.vector }
              );
              pinnedTokenReference = undefined;
              await neuronLensPanel.webview.postMessage({
                type: "cosineReferenceSet",
                label: pinnedCosineReference.label,
                norm: pinnedCosineReference.norm,
                tokenPosition: pinnedCosineReference.tokenPosition
              });
            } catch (error) {
              await neuronLensPanel.webview.postMessage({ type: "error", message: error.message });
            }
            return;
          }
          if (message.type === "unsetCosineReference") {
            pinnedCosineReference = undefined;
            return;
          }
          if (message.type === "setTokenReference") {
            const active = vscode.debug.activeDebugSession;
            if (!active || active.type !== "llmdbg") return;
            try {
              pinnedTokenReference = await active.customRequest(
                "llmdbg/getTokenReferenceVectors", {}
              );
              pinnedCosineReference = undefined;
              await neuronLensPanel.webview.postMessage({ type: "tokenReferenceSet", tokenPosition: pinnedTokenReference.tokenPosition });
            } catch (error) {
              await neuronLensPanel.webview.postMessage({ type: "error", message: error.message });
            }
            return;
          }
          if (message.type === "unsetTokenReference") {
            pinnedTokenReference = undefined;
            return;
          }
          if (message.type === "setLensTheme") {
            const theme = message.theme === "light" ? "light" : "dark";
            await vscode.workspace.getConfiguration("llmdbg").update(
              "lensTheme", theme, vscode.ConfigurationTarget.Global
            );
            return;
          }
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
          if (message.type === "measureCosineAll") {
            const active = vscode.debug.activeDebugSession;
            if (!active || active.type !== "llmdbg") {
              await neuronLensPanel.webview.postMessage({ type: "error", message: "Start and pause an llmdbg session first." });
              return;
            }
            if (!pinnedCosineReference) {
              await neuronLensPanel.webview.postMessage({ type: "error", message: "Choose a cosine reference vector first." });
              return;
            }
            try {
              const result = await active.customRequest("llmdbg/measureCosineAll", {
                referenceVector: pinnedCosineReference.vector,
                referenceLabel: pinnedCosineReference.label,
                targets: message.targets
              });
              await neuronLensPanel.webview.postMessage({ type: "allCosineResult", result });
            } catch (error) {
              await neuronLensPanel.webview.postMessage({ type: "error", message: error.message });
            }
            return;
          }
          if (message.type === "measureTokenCosines") {
            const active = vscode.debug.activeDebugSession;
            if (!active || active.type !== "llmdbg" || !pinnedTokenReference) return;
            try {
              const results = [];
              for (const target of message.targets || []) {
                const reference = pinnedTokenReference.vectors[target.key];
                if (!reference) continue;
                const measured = await active.customRequest("llmdbg/measureCosineAll", {
                  referenceVector: reference.vector,
                  referenceLabel: reference.label,
                  targets: [target.id]
                });
                if (measured.results?.[0]) results.push({ key: target.key, ...measured.results[0] });
              }
              await neuronLensPanel.webview.postMessage({
                type: "tokenCosineResult",
                result: { tokenPosition: pinnedTokenReference.tokenPosition, results }
              });
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
          if (Array.isArray(message.modes)) {
            neuronLensModes = message.modes.filter(mode => typeof mode === "string");
          }
          if (Array.isArray(message.views)) {
            neuronLensViews = message.views.filter(view => typeof view === "string");
          }
          refreshNeuronLens(neuronLensModes, neuronLensViews, Boolean(message.force));
        }, null, context.subscriptions
      );
      const panel = neuronLensPanel;
      panel.onDidDispose(() => {
        if (neuronLensPanel !== panel) return;
        neuronLensPanel = undefined;
        setToolPanelOpen("neuronLens", false);
      }, null, context.subscriptions);
    } else {
      neuronLensPanel.reveal(neuronLensPanel.viewColumn || vscode.ViewColumn.Beside, true);
    }
    if (refreshImmediately) await refreshNeuronLens();
  }

  async function profileForSession(session) {
    const config = session?.configuration || {};
    if (config.runtimeProfile && config.runtimeProfile !== "auto") return selectProfile("", "", config.runtimeProfile);
    let model = config.model || "", kind = "";
    if (config.config) {
      const source = path.resolve(expandWorkspaceFolder(config.config, session));
      const parser = path.join(adapterSource(session), "llmdbg", "config.py");
      const code = "import json,pathlib,runpy,sys; sys.path.insert(0,str(pathlib.Path(sys.argv[1]).resolve().parent.parent)); c=runpy.run_path(sys.argv[1])['load_debug_config'](sys.argv[2]); print(json.dumps({'model':c.get('model',''),'kind':c.get('modelKind','')}))";
      const { stdout } = await execFile(uvExecutable(), ["run", "--no-project", "--python", "3.12", "python", "-c", code, parser, source], { maxBuffer: 1024 * 1024 });
      const parsed = JSON.parse(stdout.trim());
      model = config.model || parsed.model; kind = config.model ? "" : parsed.kind;
    }
    if (model && fs.existsSync(path.join(model, "config.json"))) {
      kind = JSON.parse(await fs.promises.readFile(path.join(model, "config.json"), "utf8")).model_type === "gemma4" ? "gemma4" : kind;
    }
    return selectProfile(model, kind, config.runtimeProfile || "auto");
  }

  async function ensureRuntime(profile = "stable") {
    const recipe = await runtimeRecipe(profile);
    const runtimeRoot = path.join(runtimesRoot, recipe.key);
    const python = runtimePython(runtimeRoot);
    const marker = path.join(runtimeRoot, ".llmdbg-runtime");
    const ready = fs.existsSync(python)
      && fs.existsSync(marker)
      && fs.readFileSync(marker, "utf8").trim() === runtimeMarker(recipe);
    if (ready) {
      return python;
    }
    if (!runtimePromises.has(recipe.key)) {
      const promise = createRuntime(context, runtimeRoot, recipe, runtimeOutput)
        .then(() => runtimePython(runtimeRoot))
        .catch((error) => {
          runtimePromises.delete(recipe.key);
          const message = `llmdbg could not create its private Python environment: ${error.message}`;
          vscode.window.showErrorMessage(message);
          throw error;
        });
      runtimePromises.set(recipe.key, promise);
    }
    return runtimePromises.get(recipe.key);
  }

  async function showRuntimeDiagnostics() {
    const output = vscode.window.createOutputChannel("llmdbg Runtime Diagnostics");
    context.subscriptions.push(output);
    output.clear();
    output.appendLine(`OS: ${process.platform} ${process.arch}`);
    output.appendLine(`Runtime profiles: ${runtimesRoot}`);
    output.appendLine(`Configured backend: ${configuredTorchBackend()}`);
    try {
      const profile = vscode.debug.activeDebugSession?.type === "llmdbg"
        ? await profileForSession(vscode.debug.activeDebugSession)
        : await vscode.window.showQuickPick(["stable", "gemma4-preview"], { placeHolder: "Inspect which runtime profile?" });
      if (!profile) return;
      const recipe = await runtimeRecipe(profile);
      output.appendLine(`Profile: ${profile}; resolved backend: ${recipe.id} (torch ${recipe.torch}, Transformers ${recipe.transformers})`);
      const cuda = await detectedCudaVersion();
      output.appendLine(`NVIDIA driver maximum CUDA: ${cuda ?? "not detected"}`);
      const capability = await detectedCudaComputeCapability();
      output.appendLine(`NVIDIA compute capability: ${capability ?? "not detected"}`);
      const python = await ensureRuntime(profile);
      output.appendLine(`Python: ${python}`);
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
    const profile = await vscode.window.showQuickPick(["stable", "gemma4-preview"], { placeHolder: "Rebuild which runtime profile? Other profiles are unchanged." });
    if (!profile) return;
    if (hostProcess && hostProcess.exitCode === null) hostProcess.kill();
    hostProcess = undefined;
    hostPromise = undefined;
    hostSource = undefined;
    const recipe = await runtimeRecipe(profile);
    runtimePromises.delete(recipe.key);
    await createRuntime(context, path.join(runtimesRoot, recipe.key), recipe, runtimeOutput);
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
    const profile = await profileForSession(session);
    const python = await ensureRuntime(profile);
    runtimeOutput.appendLine(`Launching ${profile} with ${python}`);
    const bundledRoot = adapterSource(session);
    const residualToVocabularyLenses = residualToVocabularyLensSettings();
    const lensConfiguration = JSON.stringify(residualToVocabularyLenses);
    if (!fs.existsSync(path.join(bundledRoot, "llmdbg", "host.py"))) {
      throw new Error(`llmdbg adapter source not found under ${bundledRoot}`);
    }
    if (hostProcess && hostProcess.exitCode === null && hostPython === python && hostSource === bundledRoot && hostLensConfiguration === lensConfiguration) {
      return hostPromise;
    }
    if (hostProcess && hostProcess.exitCode === null) {
      hostProcess.kill();
    }

    const inheritedPythonPath = process.env.PYTHONPATH;
    const pythonPath = inheritedPythonPath
      ? `${bundledRoot}${path.delimiter}${inheritedPythonPath}`
      : bundledRoot;
    hostSource = bundledRoot;
    hostPython = python;
    hostLensConfiguration = lensConfiguration;
    const child = childProcess.spawn(
      python,
      ["-m", "llmdbg.host", "--host", "127.0.0.1", "--port", "0"],
      {
        cwd: session.workspaceFolder ? session.workspaceFolder.uri.fsPath : os.homedir(),
        env: {
          ...process.env,
          PYTHONNOUSERSITE: "1",
          PYTHONPATH: pythonPath,
          LLMDBG_RUNTIME_PROFILE: profile,
          LLMDBG_RESIDUAL_TO_VOCABULARY_LENSES: lensConfiguration
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
          hostLensConfiguration = undefined;
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

  context.subscriptions.push(vscode.debug.registerDebugAdapterTrackerFactory("llmdbg", {
    createDebugAdapterTracker() {
      let stopTimer;
      let stopRequest;
      const clearStopTimer = () => { clearTimeout(stopTimer); stopTimer = undefined; };
      return {
        onWillReceiveMessage(message) {
          if (message.type !== "request" || !["disconnect", "terminate"].includes(message.command)) return;
          stopRequest = message.seq;
          const child = hostProcess;
          stopTimer = setTimeout(() => {
            if (child && hostProcess === child && child.exitCode === null) {
              runtimeOutput.appendLine("llmdbg: Stop timed out; shutting down the unresponsive model host. The next run will reload the model.");
              child.kill("SIGKILL");
            }
          }, 3000);
        },
        onDidSendMessage(message) {
          if ((message.type === "response" && message.request_seq === stopRequest)
            || (message.type === "event" && message.event === "terminated")) clearStopTimer();
        },
        onExit: clearStopTimer,
        onError: clearStopTimer,
      };
    },
  }));
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
    vscode.commands.registerCommand("llmdbg.plotCurrentVector", () => showVectorPlot()),
    vscode.commands.registerCommand("llmdbg.closeVectorPlot", () => closeToolPanel("vector")),
    vscode.commands.registerCommand("llmdbg.decodeExpression", () => showExpressionDecoder()),
    vscode.commands.registerCommand("llmdbg.closeExpressionDecoder", () => closeToolPanel("decoder")),
    vscode.commands.registerCommand("llmdbg.showPromptTokens", () => showPromptTokens()),
    vscode.commands.registerCommand("llmdbg.closePromptTokens", () => closeToolPanel("promptTokens")),
    vscode.commands.registerCommand("llmdbg.initializeProject", initializeProject),
    vscode.commands.registerCommand("llmdbg.showRuntimeDiagnostics", showRuntimeDiagnostics),
    vscode.commands.registerCommand("llmdbg.rebuildRuntime", rebuildRuntime),
    vscode.commands.registerCommand("llmdbg.showNeuronLens", () => showNeuronLens()),
    vscode.commands.registerCommand("llmdbg.closeNeuronLens", () => closeToolPanel("neuronLens")),
    vscode.commands.registerCommand("llmdbg.showModelStatus", showModelStatus),
    vscode.commands.registerCommand("llmdbg.unloadModel", unloadModel),
    vscode.commands.registerCommand("llmdbg.manageResidualToVocabularyLenses", manageResidualToVocabularyLenses),
    vscode.workspace.onDidChangeConfiguration((event) => {
      if (!event.affectsConfiguration("llmdbg.jLensSource")
        && !event.affectsConfiguration("llmdbg.rLensSource")
        && !event.affectsConfiguration("llmdbg.jppLensSource")
        && !event.affectsConfiguration("llmdbg.customResidualToVocabularyLenses")) return;
      neuronLensRevision += 1;
      neuronLensItemCache.clear();
      void vscode.window.showInformationMessage("Residual-to-Vocabulary Lens sources updated. They will be used by the next debug session.");
    }),
    vscode.debug.onDidReceiveDebugSessionCustomEvent((event) => {
      if (event.session.type !== "llmdbg") {
        return;
      }
      if (event.event === "llmdbg/modelLoaded") {
        updateModelStatus(event.body);
      } else if (event.event === "llmdbg/computationStarted") {
        if (neuronLensPanel) neuronLensPanel.webview.postMessage({ type: "busy", busy: true });
      } else if (event.event === "llmdbg/computationFailed") {
        if (neuronLensPanel) neuronLensPanel.webview.postMessage({ type: "busy", busy: false });
      } else if (event.event === "llmdbg/vectorChanged") {
        if (decoderPanel) decoderPanel.webview.postMessage({ type: "stale" });
        neuronLensExecutionReady = true;
        neuronLensRevision += 1;
        neuronLensItemCache.clear();
        refreshNeuronLens();
        refreshPromptTokens();
      }
    }),
    vscode.debug.onDidStartDebugSession((session) => {
      if (session.type !== "llmdbg") return;
      if (decoderPanel) decoderPanel.webview.postMessage({ type: "stale" });
      neuronLensExecutionReady = false;
      neuronLensRevision += 1;
      neuronLensItemCache.clear();
      if (neuronLensPanel) neuronLensPanel.webview.postMessage({ type: "reset" });
      // The adapter has loaded the model, but no token trace exists yet. The
      // first stopped/vectorChanged event refreshes the panel with valid data.
      void showNeuronLens(session, false);
    }),
    vscode.debug.onDidTerminateDebugSession((session) => {
      if (session.type !== "llmdbg") return;
      if (decoderPanel) decoderPanel.webview.postMessage({ type: "stale" });
      neuronLensExecutionReady = false;
      if (neuronLensPanel) neuronLensPanel.webview.postMessage({ type: "busy", busy: false });
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
