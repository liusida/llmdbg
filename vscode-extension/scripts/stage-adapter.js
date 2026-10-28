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

// Only copy curated entries, including the shared launch config, not local settings.
const playground = path.join(projectRoot, "playground");
const catalog = JSON.parse(fs.readFileSync(path.join(playground, "starters.json"), "utf8"));
const starterDestination = path.join(path.dirname(bundled), "playground");
for (const file of ["starters.json", ...catalog.files]) {
  const destination = path.join(starterDestination, file);
  fs.mkdirSync(path.dirname(destination), { recursive: true });
  fs.copyFileSync(path.join(playground, file), destination);
}
console.log(`Staged ${catalog.files.length} curated starter files from ${playground}`);
