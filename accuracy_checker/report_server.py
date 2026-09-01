"""Local report browser with a narrowly scoped report-delete endpoint.

Generated reports are otherwise static HTML, so a browser opened with
``file://`` cannot remove files from disk.  This server exposes the repository
read-only through ``SimpleHTTPRequestHandler`` plus one guarded POST endpoint
that can delete exactly one archived run directory under ``reports/``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional
from urllib.parse import parse_qs, unquote, urlparse


logger = logging.getLogger(__name__)
DELETE_ENDPOINT = "/__accuracy_bench__/delete-report"
HEALTH_ENDPOINT = "/__accuracy_bench__/health"
LOGITS_DETAIL_ENDPOINT = "/__accuracy_bench__/logits-detail"


def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def delete_report_directory(reports_root: str, relative_path: str) -> str:
    """Delete one validated archived run directory and rebuild the index."""
    if not isinstance(relative_path, str) or not relative_path.strip():
        raise ValueError("missing report path")
    relative_path = relative_path.strip().replace("\\", "/")
    if os.path.isabs(relative_path):
        raise ValueError("report path must be relative to reports/")

    normalized = os.path.normpath(relative_path)
    if normalized in ("", ".") or normalized == ".." or normalized.startswith(".." + os.sep):
        raise ValueError("refusing to delete the reports root or an outside path")

    reports_root = os.path.realpath(reports_root)
    target = os.path.realpath(os.path.join(reports_root, normalized))
    try:
        inside = os.path.commonpath([reports_root, target]) == reports_root
    except ValueError as exc:
        raise ValueError("report path is outside reports/") from exc
    if not inside or target == reports_root:
        raise ValueError("report path is outside reports/")
    if not os.path.isdir(target):
        raise FileNotFoundError(f"report directory not found: {relative_path}")
    if not os.path.isfile(os.path.join(target, "report_data.json")):
        raise ValueError("target is not an acc_bench report directory")

    shutil.rmtree(target)

    # Rebuild the self-contained history and repository-level latest.html so
    # the browser can immediately reload a consistent remaining archive.
    from .html_report import generate_index_html

    generate_index_html(reports_root)
    return target


class ReportRequestHandler(SimpleHTTPRequestHandler):
    """Serve report files and handle the same-origin delete request."""

    reports_root: str = ""

    def _json_response(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
        parsed = urlparse(self.path)
        if parsed.path == HEALTH_ENDPOINT:
            self._json_response(200, {
                "ok": True,
                "service": "accuracy_bench_report_server",
                "delete_endpoint": DELETE_ENDPOINT,
            })
            return
        if parsed.path == LOGITS_DETAIL_ENDPOINT:
            try:
                query = parse_qs(parsed.query)
                relative = unquote((query.get("report") or [""])[0]).replace("\\", "/")
                position = int((query.get("position") or [""])[0])
                if not relative or os.path.isabs(relative) or ".." in relative.split("/"):
                    raise ValueError("invalid report path")
                target = os.path.realpath(os.path.join(self.reports_root, relative))
                reports_root = os.path.realpath(self.reports_root)
                if os.path.commonpath([reports_root, target]) != reports_root:
                    raise ValueError("report path is outside reports/")
                if os.path.basename(target) != "boundary_result.json":
                    raise ValueError("detail source must be boundary_result.json")
                with open(target, encoding="utf-8") as f:
                    raw = json.load(f)
                data = ((raw.get("evidence") or {}).get("captured_logits_replay") or {}).get("logits_data")
                if not isinstance(data, dict):
                    raise ValueError("captured logits detail is unavailable")
                positions = [int(x) for x in (data.get("token_positions") or [])]
                try:
                    index = positions.index(position)
                except ValueError as exc:
                    raise ValueError(f"position {position} is not in captured logits") from exc
                row = {"position": position}
                for key in ("ref_topk", "quant_topk", "ref_logits", "quant_logits",
                            "token_wise_cos", "token_wise_kl", "token_wise_topk_overlap",
                            "token_wise_top1_match", "ref_top1_margin", "quant_top1_margin"):
                    values = data.get(key)
                    if isinstance(values, list) and index < len(values):
                        row[key] = values[index]
                self._json_response(200, {"ok": True, "detail": row})
            except (ValueError, json.JSONDecodeError, OSError) as exc:
                self._json_response(404, {"error": str(exc)})
            return
        super().do_GET()

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
        if urlparse(self.path).path != DELETE_ENDPOINT:
            self._json_response(404, {"error": "unknown endpoint"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 64 * 1024:
                raise ValueError("invalid request body length")
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            deleted = delete_report_directory(
                self.reports_root, payload.get("path") if isinstance(payload, dict) else None
            )
            self._json_response(200, {"ok": True, "deleted": deleted})
        except FileNotFoundError as exc:
            self._json_response(404, {"error": str(exc)})
        except (ValueError, json.JSONDecodeError) as exc:
            self._json_response(400, {"error": str(exc)})
        except Exception as exc:  # noqa: BLE001
            logger.exception("report deletion failed")
            self._json_response(500, {"error": str(exc)})

    def end_headers(self) -> None:
        # The index is regenerated after deletion; avoid stale browser copies.
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


def serve_reports(
    host: str = "127.0.0.1",
    port: int = 8765,
    repo_root: Optional[str] = None,
    reports_root: Optional[str] = None,
) -> None:
    repo_root = os.path.abspath(repo_root or _repo_root())
    reports_root = os.path.abspath(reports_root or os.path.join(repo_root, "reports"))
    os.makedirs(reports_root, exist_ok=True)

    # Keep latest.html useful even before the first request reaches the server.
    from .html_report import generate_index_html

    generate_index_html(reports_root)
    bound_handler = type(
        "BoundReportRequestHandler",
        (ReportRequestHandler,),
        {"reports_root": reports_root},
    )
    handler = partial(bound_handler, directory=repo_root)
    server = ThreadingHTTPServer((host, int(port)), handler)
    logger.info("报告目录: %s", reports_root)
    logger.info("打开: http://%s:%d/latest.html", host, int(port))
    logger.info("历史记录可右键删除；Ctrl+C 停止服务")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("\n报告服务已停止")
    finally:
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(description="acc_bench 本地报告浏览与历史删除服务")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--repo_root", default=None)
    parser.add_argument("--reports_dir", default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    serve_reports(
        host=args.host,
        port=args.port,
        repo_root=args.repo_root,
        reports_root=args.reports_dir,
    )


__all__ = [
    "DELETE_ENDPOINT", "HEALTH_ENDPOINT", "LOGITS_DETAIL_ENDPOINT", "ReportRequestHandler",
    "delete_report_directory", "serve_reports",
]
