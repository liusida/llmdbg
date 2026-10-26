"use strict";

// Exercise the actual extension installer outside VS Code. Prints every command
// and installer output, and creates the same profile paths the extension uses.
const fs = require("fs"), path = require("path"), vm = require("vm"), os = require("os");
const { createRequire } = require("module");
const extension = path.resolve(__dirname, "../extension.js"), nativeRequire = createRequire(extension);
const root = process.env.LLMDBG_TEST_STORAGE || path.join(os.homedir(), ".config/Code/User/globalStorage/sidaliu.llmdbg");
const output = { clear() {}, show() {}, append: s => process.stdout.write(s), appendLine: s => console.log(s) };
const vscode = { workspace: { getConfiguration: () => ({ get: (_key, fallback) => fallback }) }, window: { createOutputChannel: () => output, withProgress: (_options, fn) => fn() }, ProgressLocation: { Notification: 1 } };
const sandbox = { require: name => name === "vscode" ? vscode : nativeRequire(name), module: { exports: {} }, process, console, setTimeout, clearTimeout, setInterval, clearInterval, __dirname: path.dirname(extension) };
vm.runInNewContext(fs.readFileSync(extension, "utf8") + "\nglobalThis.testInstaller = createRuntime;", sandbox);
const { profileRecipe } = nativeRequire("./runtime-profiles");
const recipe = profileRecipe(process.argv[2] || "stable", process.argv[3] || "cu130", process.platform, process.arch);
const runtimeRoot = path.join(root, "runtimes", recipe.key);
sandbox.testInstaller({ globalStorageUri: { fsPath: root } }, runtimeRoot, recipe, output).then(() => console.log(`VERIFIED_RUNTIME=${runtimeRoot}`)).catch(error => { console.error(error); process.exitCode = 1; });
