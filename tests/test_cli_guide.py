"""Standard-library checks that keep the interactive CLI guide in sync."""

import ast
import unittest
from html.parser import HTMLParser
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RUN_SCRIPT = ROOT / "run_accuracy_check.py"
GUIDE = ROOT / "cli_params_guide.html"
HTML_REPORT = ROOT / "accuracy_checker" / "html_report.py"


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

    def test_report_runs_are_archived_and_latest_opens_history(self):
        run_source = RUN_SCRIPT.read_text(encoding="utf-8")
        html_source = HTML_REPORT.read_text(encoding="utf-8")
        self.assertIn("def _resolve_report_run_dir(args, mode):", run_source)
        self.assertIn('_resolve_report_run_dir(args, "full")', run_source)
        self.assertIn("def _update_latest_report_link(target_path: str)", html_source)
        self.assertGreaterEqual(
            html_source.count("_update_latest_report_link(output_path)"), 3
        )
        self.assertIn("latest.html 左侧可切换当前与历史报告", self.html)

    def test_report_visual_style_matches_cli_guide(self):
        html_source = HTML_REPORT.read_text(encoding="utf-8")
        for marker in (
            "--paper:#F4F1EA", "--navy:#112B3A", "--mint:#96E6C3",
            "report-kicker", "sidebar-brand", "ACC BENCH / ALIGNMENT REPORT",
        ):
            self.assertIn(marker, html_source)

if __name__ == "__main__":
    unittest.main()
