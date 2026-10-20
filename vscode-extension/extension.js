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

  const factory = {
    async createDebugAdapterDescriptor(session) {
      const python = await ensureRuntime();
      const configuredSource = session.configuration.adapterSource
        || vscode.workspace.getConfiguration("llmdbg").get("developmentSourcePath", "");
      const bundledRoot = configuredSource
        ? path.resolve(expandWorkspaceFolder(configuredSource, session))
        : path.join(context.extensionPath, "bundled");
      if (!fs.existsSync(path.join(bundledRoot, "llmdbg", "dap.py"))) {
        throw new Error(`llmdbg adapter source not found under ${bundledRoot}`);
      }
      const inheritedPythonPath = process.env.PYTHONPATH;
      const pythonPath = inheritedPythonPath
        ? `${bundledRoot}${path.delimiter}${inheritedPythonPath}`
        : bundledRoot;
      return new vscode.DebugAdapterExecutable(
        python,
        ["-m", "llmdbg.dap"],
        {
          cwd: session.workspaceFolder ? session.workspaceFolder.uri.fsPath : os.homedir(),
          env: {
            ...process.env,
            PYTHONNOUSERSITE: "1",
            PYTHONPATH: pythonPath
          }
        }
      );
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
    })
  );
}

function deactivate() {}

module.exports = { activate, deactivate };
