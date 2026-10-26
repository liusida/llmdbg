"use strict";
const test = require("node:test"), assert = require("node:assert/strict");
const { selectProfile, profileRecipe } = require("../runtime-profiles");

test("runtime config probe resolves bundled imports without an installed package", () => {
  const fs = require("node:fs"), path = require("node:path");
  const { execFileSync } = require("node:child_process");
  const root = path.resolve(__dirname, "../..");
  const extension = fs.readFileSync(path.join(root, "vscode-extension/extension.js"), "utf8");
  const probe = JSON.parse(extension.match(/const code = ("import json,pathlib,runpy,sys;[^\n]+");/)[1]);
  const result = JSON.parse(execFileSync("python3", ["-I", "-S", "-c", probe,
    path.join(root, "src/llmdbg/config.py"),
    path.join(root, "../llmdbg-usecase/Qwen35-2B-Vision.llmdbg.py")], { cwd: "/tmp", encoding: "utf8" }));
  assert.equal(result.model, "Qwen/Qwen3.5-2B");
  assert.equal(result.kind, "qwen35");
});

test("Gemma 4 routes to preview; existing architectures stay stable", () => {
  for (const model of ["openai-community/gpt2", "google/gemma-2-2b-it", "Qwen/Qwen3.5-2B", "Qwen/Qwen3.5-27B", "Qwen/Qwen3-VL-2B-Instruct"]) assert.equal(selectProfile(model), "stable");
  assert.equal(selectProfile("google/gemma-4-E2B-it"), "gemma4-preview");
  assert.equal(selectProfile("/models/local", "gemma4"), "gemma4-preview");
  assert.equal(selectProfile("google/gemma-4-E2B-it", "", "stable"), "stable");
  assert.throws(() => selectProfile("", "", "typo"));
});
test("profile/platform/backend identities are isolated and deterministic", () => {
  const stable = profileRecipe("stable", "cu130", "linux", "arm64");
  assert.equal(stable.key, profileRecipe("stable", "cu130", "linux", "arm64").key);
  assert.notEqual(stable.key, profileRecipe("gemma4-preview", "cu130", "linux", "arm64").key);
  assert.notEqual(stable.key, profileRecipe("stable", "cu128", "linux", "arm64").key);
  assert.notEqual(stable.key, profileRecipe("stable", "cu130", "win32", "x64").key);
  assert.equal(stable.torch, "2.10.0");
  assert.equal(profileRecipe("gemma4-preview", "cu130", "linux", "arm64").torch, "2.14.1");
  assert.throws(() => profileRecipe("stable", "bad", "linux", "arm64"));
});
