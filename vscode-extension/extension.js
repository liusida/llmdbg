"use strict";

const childProcess = require("child_process");
const fs = require("fs");
const os = require("os");
const path = require("path");
const util = require("util");
const vscode = require("vscode");

const execFile = util.promisify(childProcess.execFile);

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

async function createRuntime(context, runtimeRoot) {
  await fs.promises.mkdir(context.globalStorageUri.fsPath, { recursive: true });
  await vscode.window.withProgress(
    {
      location: vscode.ProgressLocation.Notification,
      title: "llmdbg: Creating private debugger environment",
      cancellable: false
    },
    async () => {
      await execFile(bootstrapPython(), ["-m", "venv", runtimeRoot]);
    }
  );
}

function activate(context) {
  const runtimeRoot = path.join(context.globalStorageUri.fsPath, "runtime");
  let runtimePromise;

  async function ensureRuntime() {
    const python = runtimePython(runtimeRoot);
    if (fs.existsSync(python)) {
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
      const bundledRoot = path.join(context.extensionPath, "bundled");
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
    vscode.debug.registerDebugAdapterDescriptorFactory("llmdbg", factory)
  );
}

function deactivate() {}

module.exports = { activate, deactivate };
