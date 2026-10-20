"use strict";

const fs = require("fs");
const path = require("path");
const vscode = require("vscode");

function workspacePython(session) {
  const configured = session.configuration.python
    || vscode.workspace.getConfiguration("llmdbg").get("pythonPath");
  if (configured) {
    return configured;
  }

  const folder = session.workspaceFolder;
  if (folder) {
    const relative = process.platform === "win32"
      ? path.join(".venv", "Scripts", "python.exe")
      : path.join(".venv", "bin", "python");
    const candidate = path.join(folder.uri.fsPath, relative);
    if (fs.existsSync(candidate)) {
      return candidate;
    }
  }
  return process.platform === "win32" ? "python" : "python3";
}

function activate(context) {
  const factory = {
    createDebugAdapterDescriptor(session) {
      return new vscode.DebugAdapterExecutable(
        workspacePython(session),
        ["-m", "llmdbg.dap"],
        { cwd: session.workspaceFolder && session.workspaceFolder.uri.fsPath }
      );
    }
  };
  context.subscriptions.push(
    vscode.debug.registerDebugAdapterDescriptorFactory("llmdbg", factory)
  );
}

function deactivate() {}

module.exports = { activate, deactivate };
