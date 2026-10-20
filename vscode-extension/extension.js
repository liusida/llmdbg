"use strict";

const childProcess = require("child_process");
const fs = require("fs");
const os = require("os");
const path = require("path");
const util = require("util");
const vscode = require("vscode");

const execFile = util.promisify(childProcess.execFile);
const RUNTIME_VERSION = "2";

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
  return editor.document.fileName.endsWith(".llmdbg")
    ? editor.document.fileName
    : undefined;
}

function expandWorkspaceFolder(value, session) {
  if (!value || !session.workspaceFolder) {
    return value;
  }
  return value.replaceAll("${workspaceFolder}", session.workspaceFolder.uri.fsPath);
}

function vectorPlotHtml(webview, extensionUri) {
  const nonce = `${Date.now()}${Math.random().toString(36).slice(2)}`;
  const plotlyUri = webview.asWebviewUri(
    vscode.Uri.joinPath(extensionUri, "bundled", "vendor", "plotly.min.js")
  );
  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src ${webview.cspSource} 'unsafe-inline'; script-src ${webview.cspSource} 'nonce-${nonce}'; img-src data: blob:;">
  <style>
    body { font-family: var(--vscode-font-family); color: var(--vscode-foreground); padding: 8px; }
    header { display: flex; gap: 8px 12px; align-items: center; flex-wrap: wrap; }
    select { color: var(--vscode-dropdown-foreground); background: var(--vscode-dropdown-background); border: 1px solid var(--vscode-dropdown-border); padding: 4px 8px; }
    #meta { margin: 6px 0; color: var(--vscode-descriptionForeground); }
    #plot { width: 100%; height: 220px; border: 1px solid var(--vscode-panel-border); }
  </style>
</head>
<body>
  <header>
    <label>Vector <select id="vector"></select></label>
    <label>View <select id="view"><option value="line">Dimensions</option><option value="histogram">Histogram</option><option value="top">Top |activation|</option></select></label>
  </header>
  <div id="meta">Waiting for a stopped llmdbg session…</div>
  <div id="plot"></div>
  <script src="${plotlyUri}"></script>
  <script nonce="${nonce}">
    const plot = document.getElementById('plot');
    const vectorSelect = document.getElementById('vector');
    const viewSelect = document.getElementById('view');
    const meta = document.getElementById('meta');
    let snapshot;
    const labels = {
      residual_stream: 'Residual stream (output)', input: 'Residual stream (input)', delta: 'Residual delta',
      normalized_input: 'LayerNorm output', normalized_output: 'Normalized output',
      mlp_expansion_pre_activation: 'MLP expansion (pre-activation)',
      mlp_expansion: 'MLP expansion (after GELU)', mlp_projection: 'MLP projection'
    };

    function theme() {
      const css = getComputedStyle(document.documentElement);
      return {
        foreground: css.getPropertyValue('--vscode-foreground').trim() || '#cccccc',
        grid: css.getPropertyValue('--vscode-panel-border').trim() || '#555555',
        positive: css.getPropertyValue('--vscode-charts-blue').trim() || '#3794ff',
        negative: css.getPropertyValue('--vscode-charts-red').trim() || '#f14c4c'
      };
    }

    function draw() {
      if (!snapshot) return;
      const values = snapshot.vectors[vectorSelect.value];
      const colors = theme();
      const indices = values.map((_item, index) => index);
      meta.textContent = 'token ' + snapshot.tokenPosition + ' ' + JSON.stringify(snapshot.token) + ' · ' + snapshot.layer + ' · ' + values.length + 'D';
      let trace;
      if (viewSelect.value === 'histogram') {
        trace = { type: 'histogram', x: values, nbinsx: 40, marker: { color: colors.positive }, hovertemplate: 'value %{x}<br>count %{y}<extra></extra>' };
      } else if (viewSelect.value === 'top') {
        const top = values.map((item, index) => ({ index, value: item }))
          .sort((a, b) => Math.abs(b.value) - Math.abs(a.value)).slice(0, 20).reverse();
        trace = {
          type: 'bar', orientation: 'h', x: top.map(item => Math.abs(item.value)),
          y: top.map(item => String(item.index)), customdata: top.map(item => item.value),
          marker: { color: top.map(item => item.value >= 0 ? colors.positive : colors.negative) },
          hovertemplate: 'dimension %{y}<br>value %{customdata:.8f}<extra></extra>'
        };
      } else {
        trace = {
          type: 'scatter', mode: 'lines', x: indices, y: values,
          line: { color: colors.positive, width: 1.5 },
          hovertemplate: 'dimension %{x}<br>value %{y:.8f}<extra></extra>'
        };
      }
      const axis = { color: colors.foreground, gridcolor: colors.grid, zerolinecolor: colors.grid, automargin: true };
      Plotly.react(plot, [trace], {
        height: 220, margin: { l: 48, r: 12, t: 8, b: 34 },
        paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
        font: { color: colors.foreground }, xaxis: axis, yaxis: axis,
        bargap: 0.06, showlegend: false, hovermode: 'closest'
      }, {
        responsive: true, displaylogo: false, scrollZoom: true,
        toImageButtonOptions: { format: 'png', filename: 'llmdbg-' + vectorSelect.value }
      });
    }

    window.addEventListener('message', event => {
      if (event.data.type !== 'update') return;
      snapshot = event.data.data;
      const previous = vectorSelect.value;
      vectorSelect.replaceChildren(...Object.keys(snapshot.vectors).map(name => {
        const option = document.createElement('option'); option.value = name; option.textContent = labels[name] || name; return option;
      }));
      vectorSelect.value = Object.hasOwn(snapshot.vectors, previous) ? previous : 'residual_stream';
      draw();
    });
    vectorSelect.addEventListener('change', draw); viewSelect.addEventListener('change', draw);
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
          "gb10-load-llm==0.1.2"
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

  async function refreshVectorPlot(session = vscode.debug.activeDebugSession) {
    if (!vectorPanel || !session || session.type !== "llmdbg") {
      return;
    }
    try {
      const data = await session.customRequest("llmdbg/vector");
      await vectorPanel.webview.postMessage({ type: "update", data });
    } catch (error) {
      // A session can terminate while an asynchronous plot refresh is pending.
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
      vectorPanel.webview.html = vectorPlotHtml(vectorPanel.webview, context.extensionUri);
      vectorPanel.onDidDispose(() => {
        vectorPanel = undefined;
      }, null, context.subscriptions);
    } else {
      vectorPanel.reveal(vscode.ViewColumn.Beside, true);
    }
    refreshVectorPlot(session);
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
    vscode.debug.onDidReceiveDebugSessionCustomEvent((event) => {
      if (event.session.type === "llmdbg" && event.event === "llmdbg/vectorChanged") {
        refreshVectorPlot(event.session);
      }
    })
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
