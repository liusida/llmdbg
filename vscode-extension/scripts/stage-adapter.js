"use strict";

const fs = require("fs");
const path = require("path");

const extensionRoot = path.resolve(__dirname, "..");
const projectRoot = path.resolve(extensionRoot, "..");
const source = path.join(projectRoot, "src", "llmdbg");
const bundled = path.join(extensionRoot, "bundled", "llmdbg");
const plotlySource = path.join(extensionRoot, "node_modules", "plotly.js-dist-min", "plotly.min.js");
const plotlyTarget = path.join(extensionRoot, "bundled", "vendor", "plotly.min.js");

if (!fs.existsSync(path.join(source, "dap.py"))) {
  throw new Error(`llmdbg adapter source was not found at ${source}`);
}

fs.rmSync(path.dirname(bundled), { recursive: true, force: true });
fs.mkdirSync(path.dirname(bundled), { recursive: true });
fs.cpSync(source, bundled, {
  recursive: true,
  filter: (entry) => !entry.includes("__pycache__") && !entry.endsWith(".pyc")
});
fs.mkdirSync(path.dirname(plotlyTarget), { recursive: true });
fs.copyFileSync(plotlySource, plotlyTarget);

console.log(`Staged adapter from ${source} to ${bundled}`);
console.log(`Staged Plotly from ${plotlySource} to ${plotlyTarget}`);
