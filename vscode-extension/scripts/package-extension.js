"use strict";

const fs = require("fs");
const path = require("path");
const { spawnSync } = require("child_process");

const extensionRoot = path.resolve(__dirname, "..");
const packageJson = require(path.join(extensionRoot, "package.json"));
const outputDirectory = path.resolve(extensionRoot, "..", "build", "releases", "vsix");
const preRelease = process.argv.includes("--pre-release");
const suffix = preRelease ? "-pre-release" : "";
const output = path.join(outputDirectory, `llmdbg-${packageJson.version}${suffix}.vsix`);
const executable = path.join(
  extensionRoot,
  "node_modules",
  ".bin",
  process.platform === "win32" ? "vsce.cmd" : "vsce"
);

fs.mkdirSync(outputDirectory, { recursive: true });
const cliArguments = ["package", "--no-dependencies"];
if (preRelease) cliArguments.push("--pre-release");
cliArguments.push("--out", output);
const result = spawnSync(executable, cliArguments, {
  cwd: extensionRoot,
  stdio: "inherit"
});
if (result.error) throw result.error;
process.exitCode = result.status ?? 1;

