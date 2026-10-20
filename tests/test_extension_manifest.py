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


if __name__ == "__main__":
    unittest.main()
