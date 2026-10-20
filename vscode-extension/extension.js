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

function vectorPlotHtml(webview) {
  const nonce = `${Date.now()}${Math.random().toString(36).slice(2)}`;
  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src ${webview.cspSource} 'unsafe-inline'; script-src 'nonce-${nonce}';">
  <style>
    body { font-family: var(--vscode-font-family); color: var(--vscode-foreground); padding: 12px; }
    header { display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }
    select { color: var(--vscode-dropdown-foreground); background: var(--vscode-dropdown-background); border: 1px solid var(--vscode-dropdown-border); padding: 4px 8px; }
    #meta { margin: 10px 0; color: var(--vscode-descriptionForeground); }
    canvas { width: 100%; height: 440px; border: 1px solid var(--vscode-panel-border); }
    #value { min-width: 210px; font-family: var(--vscode-editor-font-family); }
  </style>
</head>
<body>
  <header>
    <label>Vector <select id="vector"><option value="residual_stream">Residual stream</option><option value="input">Input</option><option value="delta">Delta</option></select></label>
    <label>View <select id="view"><option value="line">Dimensions</option><option value="histogram">Histogram</option><option value="top">Top |activation|</option></select></label>
    <span id="value">Hover a dimension</span>
  </header>
  <div id="meta">Waiting for a stopped llmdbg session…</div>
  <canvas id="plot"></canvas>
  <script nonce="${nonce}">
    const canvas = document.getElementById('plot');
    const context = canvas.getContext('2d');
    const vectorSelect = document.getElementById('vector');
    const viewSelect = document.getElementById('view');
    const meta = document.getElementById('meta');
    const value = document.getElementById('value');
    let snapshot;
    let plotted = [];

    function colors() {
      const css = getComputedStyle(document.documentElement);
      return {
        foreground: css.getPropertyValue('--vscode-foreground').trim() || '#cccccc',
        grid: css.getPropertyValue('--vscode-panel-border').trim() || '#555555',
        positive: css.getPropertyValue('--vscode-charts-blue').trim() || '#3794ff',
        negative: css.getPropertyValue('--vscode-charts-red').trim() || '#f14c4c'
      };
    }

    function prepare() {
      const ratio = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();
      canvas.width = Math.max(1, Math.floor(rect.width * ratio));
      canvas.height = Math.max(1, Math.floor(rect.height * ratio));
      context.setTransform(ratio, 0, 0, ratio, 0, 0);
      context.clearRect(0, 0, rect.width, rect.height);
      return { width: rect.width, height: rect.height, color: colors() };
    }

    function drawLine(values, surface) {
      const pad = 36;
      const width = surface.width - pad * 2;
      const height = surface.height - pad * 2;
      const maximum = Math.max(...values.map(item => Math.abs(item)), 1e-9);
      const zero = pad + height / 2;
      context.strokeStyle = surface.color.grid;
      context.beginPath(); context.moveTo(pad, zero); context.lineTo(pad + width, zero); context.stroke();
      context.strokeStyle = surface.color.positive;
      context.beginPath();
      values.forEach((item, index) => {
        const x = pad + index * width / Math.max(1, values.length - 1);
        const y = zero - item / maximum * height / 2;
        if (index === 0) context.moveTo(x, y); else context.lineTo(x, y);
      });
      context.stroke();
      plotted = values.map((item, index) => ({ index, value: item, x: pad + index * width / Math.max(1, values.length - 1) }));
    }

    function drawHistogram(values, surface) {
      const bins = 40; const pad = 36;
      const minimum = Math.min(...values); const maximum = Math.max(...values);
      const span = Math.max(maximum - minimum, 1e-9); const counts = Array(bins).fill(0);
      values.forEach(item => counts[Math.min(bins - 1, Math.floor((item - minimum) / span * bins))]++);
      const peak = Math.max(...counts, 1); const width = surface.width - pad * 2; const height = surface.height - pad * 2;
      counts.forEach((count, index) => {
        const x = pad + index * width / bins; const barWidth = Math.max(1, width / bins - 1);
        const barHeight = count / peak * height;
        context.fillStyle = surface.color.positive; context.fillRect(x, pad + height - barHeight, barWidth, barHeight);
      });
      plotted = [];
    }

    function drawTop(values, surface) {
      const top = values.map((item, index) => ({ index, value: item })).sort((a, b) => Math.abs(b.value) - Math.abs(a.value)).slice(0, 20);
      const pad = 56; const width = surface.width - pad - 12; const row = (surface.height - 20) / top.length;
      const maximum = Math.max(...top.map(item => Math.abs(item.value)), 1e-9);
      context.font = '12px ' + getComputedStyle(document.documentElement).getPropertyValue('--vscode-editor-font-family');
      top.forEach((item, rank) => {
        const y = 8 + rank * row; context.fillStyle = surface.color.foreground; context.fillText(String(item.index), 4, y + row * 0.7);
        context.fillStyle = item.value >= 0 ? surface.color.positive : surface.color.negative;
        context.fillRect(pad, y, Math.abs(item.value) / maximum * width, Math.max(2, row - 3));
      });
      plotted = [];
    }

    function draw() {
      const surface = prepare();
      if (!snapshot) return;
      const values = snapshot.vectors[vectorSelect.value];
      if (viewSelect.value === 'histogram') drawHistogram(values, surface);
      else if (viewSelect.value === 'top') drawTop(values, surface);
      else drawLine(values, surface);
    }

    window.addEventListener('message', event => {
      if (event.data.type !== 'update') return;
      snapshot = event.data.data;
      meta.textContent = 'token ' + snapshot.tokenPosition + ' ' + JSON.stringify(snapshot.token) + ' · ' + snapshot.layer + ' · ' + snapshot.dimensions + 'D';
      draw();
    });
    vectorSelect.addEventListener('change', draw); viewSelect.addEventListener('change', draw);
    window.addEventListener('resize', draw);
    canvas.addEventListener('mousemove', event => {
      if (!plotted.length) return;
      const x = event.clientX - canvas.getBoundingClientRect().left;
      const nearest = plotted.reduce((best, item) => Math.abs(item.x - x) < Math.abs(best.x - x) ? item : best);
      value.textContent = vectorSelect.value + '[' + nearest.index + '] = ' + nearest.value.toFixed(8);
    });
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
      vectorPanel.webview.html = vectorPlotHtml(vectorPanel.webview);
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
