from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "package_jpp_workspace_lens.py"


def load_script():
    spec = importlib.util.spec_from_file_location("package_jpp_workspace_lens_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class PackageJppWorkspaceLensTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_script()

    def test_virtual_indices_translate_to_explicit_sites(self) -> None:
        self.assertEqual(self.module.site_key(0, 8), "embed")
        self.assertEqual(self.module.site_key(1, 8), "mid.0")
        self.assertEqual(self.module.site_key(2, 8), "post.0")
        self.assertEqual(self.module.site_key(7, 8), "mid.3")
        with self.assertRaises(ValueError):
            self.module.site_key(8, 8)


if __name__ == "__main__":
    unittest.main()
