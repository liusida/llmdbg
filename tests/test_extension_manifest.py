from __future__ import annotations

import json
import unittest
from pathlib import Path


class ExtensionManifestTest(unittest.TestCase):
    def test_debugger_activation_and_contribution_match(self) -> None:
        root = Path(__file__).resolve().parents[1]
        manifest = json.loads((root / "vscode-extension" / "package.json").read_text())

        self.assertIn("onDebug", manifest["activationEvents"])
        debugger_types = {debugger["type"] for debugger in manifest["contributes"]["debuggers"]}
        self.assertIn("llmdbg", debugger_types)
        self.assertEqual(manifest["main"], "./extension.js")
        self.assertIn("bundled/**", manifest["files"])
        self.assertIn("stage-adapter", manifest["scripts"])
        self.assertEqual(manifest["contributes"]["languages"][0]["extensions"], [".llmdbg"])
        self.assertEqual(manifest["contributes"]["breakpoints"][0]["language"], "llmdbg")

    def test_extension_uses_a_private_runtime(self) -> None:
        root = Path(__file__).resolve().parents[1]
        source = (root / "vscode-extension" / "extension.js").read_text()

        self.assertIn("context.globalStorageUri.fsPath", source)
        self.assertIn('["-m", "llmdbg.host"', source)
        self.assertIn('new vscode.DebugAdapterServer(port, "127.0.0.1")', source)
        self.assertIn('"gb10-load-llm==0.1.2"', source)
        self.assertIn('["venv", "--python", bootstrapPython(), runtimeRoot]', source)
        self.assertIn('"pip", "install"', source)
        self.assertIn('get("uvPath", "uv")', source)
        self.assertIn("session.configuration.adapterSource", source)
        self.assertIn("registerDebugConfigurationProvider", source)
        self.assertIn("activeLlmdbgFile", source)
        self.assertNotIn('path.join(".venv"', source)


if __name__ == "__main__":
    unittest.main()
