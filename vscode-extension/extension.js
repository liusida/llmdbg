"use strict";

const childProcess = require("child_process");
const fs = require("fs");
const os = require("os");
const path = require("path");
const util = require("util");
const vscode = require("vscode");

const execFile = util.promisify(childProcess.execFile);
const RUNTIME_VERSION = "3";
const LEGACY_PLOT_SNIPPETS_KEY = "llmdbg.plotSnippets";
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
  const directoryPrefix = `${publisher}.${name}-`;
  const versions = [];
  for (const entry of await fs.promises.readdir(extensionParent, { withFileTypes: true })) {
    if (!entry.isDirectory() || !entry.name.startsWith(directoryPrefix)) {
      continue;
    }
    try {
      const manifest = JSON.parse(
        await fs.promises.readFile(path.join(extensionParent, entry.name, "package.json"), "utf8")
      );
      if (manifest.publisher === publisher && manifest.name === name && typeof manifest.version === "string") {
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
    .action-tools { display:flex; gap:6px; }
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
    #lens-result[hidden] { display:none; }
    .lens-summary { display:flex; flex-wrap:wrap; gap:6px 14px; margin:10px 0; color:var(--vscode-descriptionForeground); }
    .lens-method { color:var(--vscode-foreground); }
    .lens-table { width:100%; border-collapse:collapse; table-layout:fixed; }
    .lens-table th { color:var(--vscode-descriptionForeground); font-weight:normal; text-align:left; border-bottom:1px solid var(--vscode-panel-border); padding:5px 7px; }
    .lens-table td { border-bottom:1px solid var(--vscode-panel-border); padding:5px 7px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
    .lens-table .rank { width:42px; text-align:right; }
    .lens-table .token-id { width:72px; color:var(--vscode-descriptionForeground); }
    .lens-table .token { width:34%; font-family:var(--vscode-editor-font-family); white-space:pre; }
    .lens-table .number { width:100px; text-align:right; font-variant-numeric:tabular-nums; }
    .probability { position:relative; }
    .probability > span { position:relative; z-index:1; }
    .probability > i { position:absolute; inset:3px auto 3px 0; background:var(--vscode-charts-blue); opacity:.22; }
    @media (max-width:700px) {
      .field { grid-template-columns:68px minmax(0,1fr) auto; }
      .field { grid-template-columns:68px minmax(0,1fr) auto; }
    }
  </style>
</head>
<body>
  <main class="workbench">
    <div class="field"><label for="expression">Vector</label><input id="expression" value="residual_stream" placeholder="hidden, qkv, residual_stream, hidden[:256]"><div class="action-tools"><button id="plot" title="Plot vector">Plot</button><button id="lens" class="secondary" title="Apply final LayerNorm and vocabulary projection">Lens</button></div></div>
    <div class="field"><label for="snippets">Plot code</label><select id="snippets"></select><div class="icon-tools"><button id="open" class="icon" title="Open plot file" aria-label="Open plot file"><svg viewBox="0 0 16 16"><path d="M12.1 1.9l2 2-8.7 8.7-3 .7.7-3 9-9.4zM2 14h12v1H2v-1z"/></svg></button><button id="new" class="icon" title="New plot file" aria-label="New plot file"><svg viewBox="0 0 16 16"><path d="M7.5 1h1v6.5H15v1H8.5V15h-1V8.5H1v-1h6.5V1z"/></svg></button><button id="delete" class="icon" title="Delete plot file" aria-label="Delete plot file"><svg viewBox="0 0 16 16"><path d="M5 1h6l1 2h3v1h-1l-1 11H3L2 4H1V3h3l1-2zm1 .9L5.5 3h5L10 1.9H6zM3 4l1 10h8l1-10H3z"/></svg></button></div></div>
    <section id="preview"><div id="status">Enter a vector expression, then choose Plot or Lens.</div><img id="image" alt="Matplotlib plot preview"><div id="lens-result" hidden></div></section>
  </main>
  <script nonce="${nonce}">
    const vscode = acquireVsCodeApi();
    const expression = document.getElementById('expression'), snippets = document.getElementById('snippets');
    const status = document.getElementById('status'), image = document.getElementById('image'), lensResult = document.getElementById('lens-result');
    let items = [];
    window.addEventListener('message', event => {
      if (event.data.type === 'snippets') { const previous=snippets.value; items=event.data.items; snippets.replaceChildren(...items.map(item => { const o=document.createElement('option'); o.value=item.file; o.textContent=item.name; return o; })); snippets.value=items.some(item=>item.file===previous)?previous:(event.data.selected||items[0]?.file||''); }
      if (event.data.type === 'result') { lensResult.hidden=true; image.src='data:image/png;base64,'+event.data.image; status.className=''; status.textContent=event.data.meta; }
      if (event.data.type === 'lens') { image.removeAttribute('src'); lensResult.hidden=false; lensResult.replaceChildren(); const summary=document.createElement('div'); summary.className='lens-summary'; const method=document.createElement('span'); method.className='lens-method'; method.textContent='final LayerNorm → unembedding'; const context=document.createElement('span'); context.textContent=event.data.expression+' · '+event.data.dimensions+'D · token '+event.data.tokenPosition+' '+JSON.stringify(event.data.token)+' · '+event.data.layer; summary.append(method,context); const table=document.createElement('table'); table.className='lens-table'; table.innerHTML='<thead><tr><th class="rank">#</th><th class="token-id">ID</th><th class="token">Token</th><th class="number">Logit</th><th class="number">Probability</th></tr></thead><tbody></tbody>'; const body=table.querySelector('tbody'); const maxProbability=Math.max(...event.data.tokens.map(item=>item.probability),1e-12); for(const item of event.data.tokens){ const row=document.createElement('tr'); const rank=document.createElement('td'); rank.className='rank'; rank.textContent=item.rank; const id=document.createElement('td'); id.className='token-id'; id.textContent=item.id; const token=document.createElement('td'); token.className='token'; token.textContent=JSON.stringify(item.text); token.title=item.text; const logit=document.createElement('td'); logit.className='number'; logit.textContent=item.logit.toFixed(4); const probability=document.createElement('td'); probability.className='number probability'; const bar=document.createElement('i'); bar.style.width=(100*item.probability/maxProbability).toFixed(2)+'%'; const value=document.createElement('span'); value.textContent=(100*item.probability).toFixed(3)+'%'; probability.append(bar,value); row.append(rank,id,token,logit,probability); body.append(row); } lensResult.append(summary,table); status.className=''; status.textContent='Top '+event.data.tokens.length+' vocabulary predictions'; }
      if (event.data.type === 'error') { status.className='error'; status.textContent='Error: '+event.data.message; }
    });
    document.getElementById('plot').onclick=()=>{ status.className=''; status.textContent='Rendering…'; vscode.postMessage({type:'plot', expression:expression.value, file:snippets.value}); };
    document.getElementById('lens').onclick=()=>{ status.className=''; status.textContent='Computing logit lens…'; vscode.postMessage({type:'lens', expression:expression.value}); };
    document.getElementById('open').onclick=()=>vscode.postMessage({type:'open', file:snippets.value});
    document.getElementById('new').onclick=()=>vscode.postMessage({type:'new'});
    document.getElementById('delete').onclick=()=>vscode.postMessage({type:'delete', file:snippets.value});
  </script>
</body>
</html>`;
}

async function createRuntime(context, runtimeRoot) {
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
          "--upgrade",
          "gb10-load-llm==0.1.2",
          "matplotlib>=3.10,<4"
        ],
        { maxBuffer: 10 * 1024 * 1024 }
      );
      await fs.promises.writeFile(
        path.join(runtimeRoot, ".llmdbg-runtime"),
        `${RUNTIME_VERSION}\n`
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
        if (message.type === "plot" || message.type === "lens") {
          const active = vscode.debug.activeDebugSession;
          if (!active || active.type !== "llmdbg") {
            await vectorPanel.webview.postMessage({ type: "error", message: "Start and pause an llmdbg session first." });
            return;
          }
          try {
            const expression = String(message.expression || "residual_stream");
            if (message.type === "lens") {
              const result = await active.customRequest("llmdbg/lens", { expression });
              await vectorPanel.webview.postMessage({ type: "lens", ...result });
            } else {
              const code = await fs.promises.readFile(await plotPath(String(message.file || "")), "utf8");
              const result = await active.customRequest("llmdbg/plot", { expression, code });
              await vectorPanel.webview.postMessage({ type: "result", ...result });
            }
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

  async function ensureRuntime() {
    const python = runtimePython(runtimeRoot);
    const marker = path.join(runtimeRoot, ".llmdbg-runtime");
    const ready = fs.existsSync(python)
      && fs.existsSync(marker)
      && fs.readFileSync(marker, "utf8").trim() === RUNTIME_VERSION;
    if (ready) {
      return python;
    }
    if (!runtimePromise) {
      runtimePromise = createRuntime(context, runtimeRoot)
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
    hostSource = bundledRoot;
    const child = childProcess.spawn(
      python,
      ["-m", "llmdbg.host", "--host", "127.0.0.1", "--port", "0"],
      {
        cwd: session.workspaceFolder ? session.workspaceFolder.uri.fsPath : os.homedir(),
        env: {
          ...process.env,
          PYTHONNOUSERSITE: "1",
          PYTHONPATH: pythonPath
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
    vscode.commands.registerCommand("llmdbg.showModelStatus", showModelStatus),
    vscode.commands.registerCommand("llmdbg.unloadModel", unloadModel),
    vscode.debug.onDidReceiveDebugSessionCustomEvent((event) => {
      if (event.session.type !== "llmdbg") {
        return;
      }
      if (event.event === "llmdbg/modelLoaded") {
        updateModelStatus(event.body);
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
