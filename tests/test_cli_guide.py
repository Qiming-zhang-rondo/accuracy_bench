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
            "dsparkSample", "kimiKdaField", "activationQuantType",
            "targetLayersField", "cacheDir", "mlaFine", "l2WorkflowNote",
            "paramSearch", "paramRows",
        }
        self.assertTrue(required.issubset(self.parser.ids))

    def test_every_cli_flag_is_documented(self):
        missing = sorted(flag for flag in _cli_flags() if flag not in self.html)
        self.assertEqual(missing, [], f"CLI guide missing flags: {missing}")

    def test_special_model_recommendations_are_present(self):
        for marker in ("qwen3_6_moe", "kimi_k3", "dspark", "grouped_dual"):
            self.assertIn(marker, self.html)
        self.assertIn('setFieldVisibility("kimiKdaField", kimiL1)', self.html)
        self.assertIn("模型目录通常不自带", self.html)

    def test_preflight_blocks_invalid_commands_and_recommends_cards(self):
        for marker in (
            "validateParameters", "copyCommand.disabled",
            "recommendedConfig", "应用推荐", "isAbsoluteLinuxPath",
        ):
            self.assertIn(marker, self.html)

    def test_activation_quant_type_is_selectable_and_generates_both_flags(self):
        for quant_type in (
            "W8A8_MXFP8", "W4A8_MXFP", "W4A4_MXFP4",
            "W4A4_DYNAMIC",
        ):
            self.assertIn(f'value="{quant_type}"', self.html)
        self.assertNotIn('value="W4A4_LAOS"', self.html)
        self.assertIn('args.push(line("--activation_quant"))', self.html)
        self.assertIn(
            'args.push(line("--activation_quant_type", el.activationQuantType.value))',
            self.html,
        )

    def test_l2_form_uses_cache_and_single_device_contract(self):
        l2_block = self.html.split('} else if (mode === "l2") {', 1)[1]
        l2_block = l2_block.split('} else if (mode === "boundary") {', 1)[0]
        self.assertIn('line("--ref_device"', l2_block)
        self.assertIn('line("--quant_device"', l2_block)
        self.assertIn('line("--target_layers"', l2_block)
        self.assertIn('line("--no_mla_fine"', l2_block)
        for l1_only_flag in (
            "--compare_mode", "--ref_devices", "--quant_devices",
            "--layers_per_shard", "--cache_top_k", "--activation_quant",
        ):
            self.assertNotIn(l1_only_flag, l2_block)
        self.assertIn('line("--cache_dir"', self.html)
        self.assertIn('setFieldVisibility("l2WorkflowNote", l2Flow)', self.html)
        self.assertIn('data-preset="l2"', self.html)
        self.assertIn("留空时自动发现当前模型与 Prompt 可用的缓存层", self.html)

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
