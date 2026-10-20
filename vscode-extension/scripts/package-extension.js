"use strict";

const fs = require("fs");
const path = require("path");
const { spawnSync } = require("child_process");

const extensionRoot = path.resolve(__dirname, "..");
const packageJson = require(path.join(extensionRoot, "package.json"));
const outputDirectory = path.resolve(extensionRoot, "..", "build", "releases", "vsix");
const output = path.join(outputDirectory, `llmdbg-${packageJson.version}.vsix`);
const executable = path.join(
  extensionRoot,
  "node_modules",
  ".bin",
  process.platform === "win32" ? "vsce.cmd" : "vsce"
);

fs.mkdirSync(outputDirectory, { recursive: true });
const result = spawnSync(executable, ["package", "--no-dependencies", "--out", output], {
  cwd: extensionRoot,
  stdio: "inherit"
});
if (result.error) throw result.error;
process.exitCode = result.status ?? 1;

