"""Safe local report deletion tests."""

import json
import threading
from functools import partial
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.request import Request, urlopen

import pytest

from accuracy_checker.report_server import (
    HEALTH_ENDPOINT,
    ReportRequestHandler,
    delete_report_directory,
)


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


def test_report_server_health_and_delete_post(tmp_path, monkeypatch):
    reports = tmp_path / "reports"
    run = reports / "run_boundary"
    run.mkdir(parents=True)
    (run / "report_data.json").write_text(json.dumps({
        "overview": {"model_name": "demo"}, "run_mode": "boundary",
    }), encoding="utf-8")
    import accuracy_checker.html_report as html_report
    monkeypatch.setattr(html_report, "_update_latest_report_link", lambda path: path)

    handler_type = type(
        "TestReportRequestHandler", (ReportRequestHandler,),
        {"reports_root": str(reports)},
    )
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0), partial(handler_type, directory=str(tmp_path)),
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with urlopen(base + HEALTH_ENDPOINT, timeout=3) as response:
            health = json.load(response)
        assert health["service"] == "accuracy_bench_report_server"

        request = Request(
            base + "/__accuracy_bench__/delete-report",
            data=json.dumps({"path": "run_boundary"}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=3) as response:
            deleted = json.load(response)
        assert deleted["ok"] is True
        assert not run.exists()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
