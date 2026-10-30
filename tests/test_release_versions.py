"""Keep the bundled adapter and extension on one release version."""
import ast
import json
from pathlib import Path
import re
import unittest


class ReleaseVersionsTest(unittest.TestCase):
    def test_python_extension_and_lockfiles_agree(self):
        root=Path(__file__).resolve().parents[1]
        package=json.loads((root/'vscode-extension/package.json').read_text())
        lock=json.loads((root/'vscode-extension/package-lock.json').read_text())
        version=package['version']
        self.assertRegex(version,r'^\d+\.\d+\.\d+$')
        self.assertEqual(lock['version'],version)
        self.assertEqual(lock['packages']['']['version'],version)
        project=(root/'pyproject.toml').read_text().split('[project]',1)[1].split('\n[',1)[0]
        self.assertEqual(re.search(r'^version = "([^"]+)"',project,re.M)[1],version)
        locked=re.search(r'\[\[package\]\]\nname = "llmdbg"\nversion = "([^"]+)"',(root/'uv.lock').read_text())
        self.assertEqual(locked[1],version)
        tree=ast.parse((root/'src/llmdbg/__init__.py').read_text())
        python_version=next(ast.literal_eval(node.value) for node in tree.body
            if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='__version__' for t in node.targets))
        self.assertEqual(python_version,version)


if __name__=='__main__':unittest.main()
