"""Standard-library checks that keep the interactive CLI guide in sync."""

import ast
import unittest
from html.parser import HTMLParser
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RUN_SCRIPT = ROOT / "run_accuracy_check.py"
GUIDE = ROOT / "cli_params_guide.html"


class _GuideParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = set()

    def handle_starttag(self, tag, attrs):
        del tag
        attrs = dict(attrs)
        if attrs.get("id"):
            self.ids.add(attrs["id"])


def _cli_flags():
    tree = ast.parse(RUN_SCRIPT.read_text(encoding="utf-8"))
    flags = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute) or node.func.attr != "add_argument":
            continue
        for arg in node.args:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                if arg.value.startswith("--"):
                    flags.add(arg.value)
    return flags


class TestCliGuide(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.html = GUIDE.read_text(encoding="utf-8")
        cls.parser = _GuideParser()
        cls.parser.feed(cls.html)

    def test_guide_exists_and_has_command_builder_controls(self):
        required = {
            "refModel", "quantModel", "modelType", "visibleDevices",
            "refDevices", "quantDevices", "commandOutput", "copyCommand",
            "modelSizeB", "perDeviceMemory", "rotationMatrix",
            "preflightSummary", "applyRecommendation",
            "dsparkSample", "paramSearch", "paramRows",
        }
        self.assertTrue(required.issubset(self.parser.ids))

    def test_every_cli_flag_is_documented(self):
        missing = sorted(flag for flag in _cli_flags() if flag not in self.html)
        self.assertEqual(missing, [], f"CLI guide missing flags: {missing}")

    def test_special_model_recommendations_are_present(self):
        for marker in ("qwen3_6_moe", "kimi_k3", "dspark", "grouped_dual"):
            self.assertIn(marker, self.html)

    def test_preflight_blocks_invalid_commands_and_recommends_cards(self):
        for marker in (
            "validateParameters", "copyCommand.disabled",
            "recommendedConfig", "应用推荐", "isAbsoluteLinuxPath",
        ):
            self.assertIn(marker, self.html)

    def test_l1_standalone_run_writes_product_report(self):
        source = RUN_SCRIPT.read_text(encoding="utf-8")
        self.assertIn("_write_stage_report_artifacts(args, report, mode)", source)
        self.assertIn('"product_report.html"', source)
        self.assertIn('"report_data.json"', source)


if __name__ == "__main__":
    unittest.main()
