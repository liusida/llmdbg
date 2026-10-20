"use strict";

const fs = require("fs");
const path = require("path");

const extensionRoot = path.resolve(__dirname, "..");
const projectRoot = path.resolve(extensionRoot, "..");
const source = path.join(projectRoot, "src", "llmdbg");
const bundled = path.join(extensionRoot, "bundled", "llmdbg");

if (!fs.existsSync(path.join(source, "dap.py"))) {
  throw new Error(`llmdbg adapter source was not found at ${source}`);
}

fs.rmSync(path.dirname(bundled), { recursive: true, force: true });
fs.mkdirSync(path.dirname(bundled), { recursive: true });
fs.cpSync(source, bundled, {
  recursive: true,
  filter: (entry) => !entry.includes("__pycache__") && !entry.endsWith(".pyc")
});

console.log(`Staged adapter from ${source} to ${bundled}`);
