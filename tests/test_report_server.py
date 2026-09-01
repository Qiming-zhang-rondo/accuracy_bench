"""Safe local report deletion tests."""

import json
from pathlib import Path

import pytest

from accuracy_checker.report_server import delete_report_directory


def test_delete_report_directory_and_rebuild_index(tmp_path, monkeypatch):
    reports = tmp_path / "reports"
    run = reports / "demo_l1_20260901"
    run.mkdir(parents=True)
    (run / "report_data.json").write_text(json.dumps({
        "overview": {"model_name": "demo"},
        "run_status": "SUCCESS",
        "run_mode": "l1",
    }), encoding="utf-8")
    (run / "large-output.bin").write_bytes(b"x" * 32)

    # Avoid changing the repository-level latest.html during a temp-dir test.
    import accuracy_checker.html_report as html_report
    monkeypatch.setattr(html_report, "_update_latest_report_link", lambda path: path)

    deleted = delete_report_directory(str(reports), "demo_l1_20260901")

    assert Path(deleted).resolve() == run.resolve()
    assert not run.exists()
    assert (reports / "index.html").exists()


@pytest.mark.parametrize("path", [".", "..", "../outside", "/tmp/outside"])
def test_delete_report_directory_rejects_broad_or_outside_paths(tmp_path, path):
    reports = tmp_path / "reports"
    reports.mkdir()
    with pytest.raises(ValueError):
        delete_report_directory(str(reports), path)


def test_delete_report_directory_requires_report_marker(tmp_path):
    reports = tmp_path / "reports"
    target = reports / "not-a-report"
    target.mkdir(parents=True)
    (target / "file.txt").write_text("keep", encoding="utf-8")

    with pytest.raises(ValueError, match="not an acc_bench report"):
        delete_report_directory(str(reports), "not-a-report")

    assert target.exists()
