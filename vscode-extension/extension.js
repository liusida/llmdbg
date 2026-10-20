"use strict";

const childProcess = require("child_process");
const fs = require("fs");
const os = require("os");
const path = require("path");
const util = require("util");
const vscode = require("vscode");

const execFile = util.promisify(childProcess.execFile);
const RUNTIME_VERSION = "3";
const PLOT_SNIPPETS_KEY = "llmdbg.plotSnippets";
const DEFAULT_PLOT_CODE = `fig, ax = plt.subplots(figsize=(9, 2.6))
ax.plot(vector, linewidth=1)
ax.axhline(0, color="gray", linewidth=0.5)
ax.set_xlabel("dimension")
ax.set_ylabel("activation")
ax.set_title(vector_name)
fig.tight_layout()`;

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
    .snippet-tools { display:grid; grid-template-columns:minmax(120px, 180px) minmax(110px, 1fr) auto auto; gap:6px; }
    input, select, textarea { width:100%; color:var(--vscode-input-foreground); background:var(--vscode-input-background); border:1px solid var(--vscode-input-border, transparent); border-radius:2px; outline:none; }
    input, select { height:28px; padding:3px 7px; }
    input:focus, select:focus, textarea:focus { border-color:var(--vscode-focusBorder); }
    input::placeholder { color:var(--vscode-input-placeholderForeground); }
    textarea { min-height:168px; padding:8px 10px; resize:vertical; color:var(--vscode-editor-foreground); background:var(--vscode-editor-background); border-color:var(--vscode-panel-border); font:var(--vscode-editor-font-size) var(--vscode-editor-font-family); line-height:1.45; tab-size:4; }
    button { min-height:28px; padding:4px 12px; border:1px solid transparent; border-radius:2px; background:var(--vscode-button-background); color:var(--vscode-button-foreground); font:inherit; cursor:pointer; white-space:nowrap; }
    button:hover { background:var(--vscode-button-hoverBackground); }
    button:focus { outline:1px solid var(--vscode-focusBorder); outline-offset:1px; }
    button.secondary { background:var(--vscode-button-secondaryBackground); color:var(--vscode-button-secondaryForeground); }
    button.secondary:hover { background:var(--vscode-button-secondaryHoverBackground); }
    #status { min-height:19px; color:var(--vscode-descriptionForeground); }
    #status.error { color:var(--vscode-errorForeground); }
    #preview { padding-top:2px; border-top:1px solid var(--vscode-panel-border); }
    #image { display:block; max-width:100%; margin-top:10px; background:white; border:1px solid var(--vscode-panel-border); }
    #image:not([src]) { display:none; }
    @media (max-width:700px) {
      .field { grid-template-columns:68px minmax(0,1fr) auto; }
      .snippet-tools { grid-template-columns:1fr 1fr; }
    }
  </style>
</head>
<body>
  <main class="workbench">
    <div class="field"><label for="expression">Vector</label><input id="expression" value="residual_stream" placeholder="hidden, qkv, residual_stream, hidden[:256]"><button id="plot">Plot</button></div>
    <div class="field"><label for="snippets">Plot code</label><div class="snippet-tools"><select id="snippets"></select><input id="name" placeholder="New snippet name"><button id="save" class="secondary">Save as New</button><button id="delete" class="secondary">Delete</button></div><span></span></div>
    <textarea id="code" aria-label="Matplotlib plotting code" spellcheck="false"></textarea>
    <section id="preview"><div id="status">Enter a vector expression and run a Matplotlib snippet.</div><img id="image" alt="Matplotlib plot preview"></section>
  </main>
  <script nonce="${nonce}">
    const vscode = acquireVsCodeApi();
    const expression = document.getElementById('expression'), snippets = document.getElementById('snippets');
    const code = document.getElementById('code'), name = document.getElementById('name');
    const status = document.getElementById('status'), image = document.getElementById('image');
    let items = [];
    function selectSnippet() { const item = items.find(x => x.name === snippets.value); if (item) code.value = item.code; }
    window.addEventListener('message', event => {
      if (event.data.type === 'snippets') { items = event.data.items; snippets.replaceChildren(...items.map(item => { const o=document.createElement('option'); o.value=item.name; o.textContent=item.name; return o; })); selectSnippet(); }
      if (event.data.type === 'result') { image.src='data:image/png;base64,'+event.data.image; status.className=''; status.textContent=event.data.meta; }
      if (event.data.type === 'error') { status.className='error'; status.textContent='Error: '+event.data.message; }
    });
    snippets.addEventListener('change', selectSnippet);
    document.getElementById('plot').onclick=()=>{ status.className=''; status.textContent='Rendering…'; vscode.postMessage({type:'plot', expression:expression.value, code:code.value}); };
    document.getElementById('save').onclick=()=>vscode.postMessage({type:'save', name:name.value, code:code.value});
    document.getElementById('delete').onclick=()=>vscode.postMessage({type:'delete', name:snippets.value});
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
  const runtimeRoot = path.join(context.globalStorageUri.fsPath, "runtime");
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

  function plotSnippets() {
    return context.globalState.get(PLOT_SNIPPETS_KEY, [
      { name: "Line plot", code: DEFAULT_PLOT_CODE }
    ]);
  }

  async function sendPlotSnippets() {
    if (vectorPanel) {
      await vectorPanel.webview.postMessage({ type: "snippets", items: plotSnippets() });
    }
  }

  function showVectorPlot() {
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
            const result = await active.customRequest("llmdbg/plot", {
              expression: String(message.expression || "residual_stream"),
              code: String(message.code || DEFAULT_PLOT_CODE)
            });
            await vectorPanel.webview.postMessage({ type: "result", ...result });
          } catch (error) {
            await vectorPanel.webview.postMessage({ type: "error", message: error.message });
          }
        } else if (message.type === "save") {
          const snippetName = String(message.name || "").trim();
          if (!snippetName) {
            await vectorPanel.webview.postMessage({ type: "error", message: "Enter a snippet name." });
            return;
          }
          const snippets = plotSnippets().filter(item => item.name !== snippetName);
          snippets.push({ name: snippetName, code: String(message.code || "") });
          await context.globalState.update(PLOT_SNIPPETS_KEY, snippets);
          await sendPlotSnippets();
        } else if (message.type === "delete") {
          const snippets = plotSnippets().filter(item => item.name !== message.name);
          await context.globalState.update(
            PLOT_SNIPPETS_KEY,
            snippets.length ? snippets : [{ name: "Line plot", code: DEFAULT_PLOT_CODE }]
          );
          await sendPlotSnippets();
        }
      }, null, context.subscriptions);
      vectorPanel.onDidDispose(() => {
        vectorPanel = undefined;
      }, null, context.subscriptions);
    } else {
      vectorPanel.reveal(vscode.ViewColumn.Beside, true);
    }
    sendPlotSnippets();
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
