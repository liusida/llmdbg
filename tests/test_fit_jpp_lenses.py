from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "fit_jpp_lenses.py"


def load_script():
    spec = importlib.util.spec_from_file_location("fit_jpp_lenses_test_module", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FitJppLensesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_script()

    def test_supported_profiles_cover_extension_models(self) -> None:
        self.assertEqual(
            {profile.model_id for profile in self.module.PROFILES.values()},
            {
                "openai-community/gpt2",
                "Qwen/Qwen3.5-0.8B",
                "Qwen/Qwen3.5-2B",
                "Qwen/Qwen3.5-9B",
                "Qwen/Qwen3.5-27B",
            },
        )
        self.assertTrue(self.module.PROFILES["gpt2"].gpt2_relp)
        self.assertTrue(
            all(
                not profile.gpt2_relp
                for key, profile in self.module.PROFILES.items()
                if key != "gpt2"
            )
        )

    def test_exact_mcnemar_p_is_symmetric_and_bounded(self) -> None:
        self.assertEqual(self.module.exact_mcnemar_p(0, 0), 1.0)
        self.assertEqual(
            self.module.exact_mcnemar_p(2, 7), self.module.exact_mcnemar_p(7, 2)
        )
        self.assertLessEqual(self.module.exact_mcnemar_p(0, 12), 1.0)

    def test_gpt_manifest_discloses_external_relp_rules(self) -> None:
        args = self.module.build_parser().parse_args(["--model", "gpt2"])
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "manifest.json"
            self.module.write_manifest(output, self.module.PROFILES["gpt2"], args, [0, 1, 2])
            payload = json.loads(output.read_text())
        self.assertEqual(payload["target_layer"], 3)
        self.assertEqual(payload["lrp"]["mode"], "external-gpt2-relp")
        self.assertIn("not upstream", payload["lrp"]["note"])


if __name__ == "__main__":
    unittest.main()
