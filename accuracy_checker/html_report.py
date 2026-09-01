"""
HTML 报告生成

两个版本共存:

  v1  generate_html_report(boundary_results, l2_results, model_name, output_path)
      inline CSS, 无 JS。banner+表格。供旧入口/UT 使用 (backward-compat)。

  v2  generate_product_html_report(report_data, output_path)
      自包含 (inline CSS+JS, 无外网依赖)。用 vanilla JS+SVG 画图:
        * Overview 一屏结论 (定界/首次发散/误差边界/根因算子/可信度条/ground-truth 命中)
        * L1 逐层 cos_sim 对比 (默认显示首个发散层, 点击展开全部层)
        * L2 subgraph recovery / self_rot_err 柱状图 (绿框=修复点, 红框=嫌疑)
        * Logits: scatter + top-k 并排柱 + token-wise 折线 + 分布直方图 (4 类可视化)
        * Inference Compare: ref vs quant 生成结果逐 token 对比
      每个指标附 ❓ 弹窗 (含义+公式+典型值+解读); INVALID_RUN 红色告警不误导。
      配色: ref=#2563eb (蓝) quant=#ea580c (橙红)。
"""

from __future__ import annotations

import html
import json
import logging
import os
import shutil
import time
from typing import Any, Dict, List, Optional, Tuple


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 配色阈值 (hardcode, 来自 l2_html_output.md §4)
# ---------------------------------------------------------------------------

# 误差类 (高=坏): (green<, yellow<, red>=)
_ERR_THRESH = {"selfroterr": (0.02, 0.10), "rotberr": (0.05, 0.15),
               "baseline_l2": (0.05, 0.20)}
# 翻转类 (高=坏)
_FLIP_THRESH = (0.01, 0.10)
# 恢复类 (高=好): (green>=, yellow>=, red<)
_REC_THRESH = (0.80, 0.30)

_CSS = """
body { font-family: -apple-system, "PingFang SC", "Microsoft YaHei", sans-serif; font-size: 13px; margin: 20px; color: #222; }
h1 { font-size: 18px; margin-bottom: 4px; }
h2 { font-size: 15px; margin-top: 22px; border-bottom: 2px solid #1a6b1a; padding-bottom: 3px; }
.meta { color: #555; font-size: 12px; margin: 4px 0 16px 0; }
table { border-collapse: collapse; margin: 8px 0; }
th, td { border: 1px solid #ccc; padding: 4px 8px; white-space: nowrap; }
th { background: #f0f0f0; cursor: help; }
td.text-cell { white-space: normal; max-width: 600px; }
.good { background: #d6f5d6; }
.warn { background: #fff3b0; }
.bad  { background: #ffd5d0; }
.na   { background: #eee; color: #888; }
.root-suspect    { border: 2px solid #b30000; font-weight: 700; }
.impact-boundary { border: 2px solid #1a6b1a; }
.formula { background: #f7f7f7; padding: 8px 12px; border-left: 3px solid #1a6b1a; margin: 8px 0; }
.banner { padding: 10px 14px; border-radius: 4px; margin: 8px 0; }
.banner-ok   { background: #d6f5d6; border-left: 4px solid #1a6b1a; }
.banner-bad  { background: #ffd5d0; border-left: 4px solid #b30000; }
.banner-warn { background: #fff3b0; border-left: 4px solid #b8860b; }
.legend { font-size: 12px; color: #555; margin-top: 20px; padding-top: 10px; border-top: 1px solid #ccc; }
pre.gen { white-space: pre-wrap; word-break: break-word; background: #fafafa; padding: 8px; border: 1px solid #ddd; max-height: 200px; overflow: auto; }
"""


# ---------------------------------------------------------------------------
# 工具: 颜色 class 判定 (v1, 保留供 UT/report_data 复用)
# ---------------------------------------------------------------------------

def _err_class(val: Optional[float], key: str) -> str:
    """误差类 (高=坏): selfroterr / rotberr / baseline_l2"""
    if val is None:
        return "na"
    g, y = _ERR_THRESH.get(key, (0.02, 0.10))
    if val < g:
        return "good"
    if val < y:
        return "warn"
    return "bad"


def _flip_class(val: Optional[float]) -> str:
    if val is None:
        return "na"
    g, y = _FLIP_THRESH
    if val < g:
        return "good"
    if val < y:
        return "warn"
    return "bad"


def _rec_class(val: Optional[float]) -> str:
    """恢复类 (高=好): recovery / input_recovery / branch_patch"""
    if val is None:
        return "na"
    g, r = _REC_THRESH  # 0.80, 0.30
    if val < 0:           # 负值 = softmax 耦合
        return "bad"
    if val >= g:
        return "good"
    if val >= r:
        return "warn"
    return "bad"


def _fmt_pct(val: Optional[float]) -> str:
    if val is None:
        return "—"
    return f"{val * 100:.1f}%"


def _fmt_err(val: Optional[float]) -> str:
    if val is None:
        return "—"
    return f"{val:.4f}"


def _esc(s: Any) -> str:
    return html.escape(str(s) if s is not None else "")


# ---------------------------------------------------------------------------
# 定界 (inference_check) — 简单重复启发式判定 (v1 + report_data 复用)
# ---------------------------------------------------------------------------

def _detect_repetition(text: str) -> Tuple[bool, str]:
    """简单重复检测: 4-gram 连续重复 ≥3 次 → 乱码嫌疑。
    返回 (is_suspect, reason)"""
    if not text or len(text) < 12:
        return False, ""
    # 按字符滑窗 4-gram
    n = 4
    for i in range(len(text) - n * 3):
        gram = text[i:i + n]
        # 连续重复次数
        rep = 1
        j = i + n
        while j + n <= len(text) and text[j:j + n] == gram:
            rep += 1
            j += n
        if rep >= 3:
            return True, f"4-gram '{gram}' 连续重复 {rep} 次"
    return False, ""


def _verdict_banner(results: List[Dict]) -> str:
    """根据生成结果给一个初步判定 banner (需人工确认)"""
    if not results:
        return ""
    suspects = []
    for i, r in enumerate(results):
        gen = r.get("generated") or ""
        is_sus, reason = _detect_repetition(gen)
        if is_sus:
            suspects.append((i + 1, reason))
    if suspects and len(suspects) >= len(results) / 2:
        reason_txt = "; ".join(f"#{i}:{html.escape(rs)}" for i, rs in suspects[:3])
        return (f'<div class="banner banner-bad"><b>初步判定: 量化本身有问题 (乱码嫌疑)</b><br>'
                f'检测到重复: {reason_txt}<br>'
                f'<span style="color:#666">→ 需要用 L2 逐层定位坏权重；生成结果乱码/重复 = 量化本身有问题</span></div>')
    elif suspects:
        return (f'<div class="banner banner-warn"><b>初步判定: 部分结果异常 (需人工确认)</b><br>'
                f'检测到重复: {"; ".join(f"#{i}:{rs}" for i, rs in suspects[:3])}<br>'
                f'<span style="color:#666">→ 请人工检查生成文本是否通顺</span></div>')
    else:
        return (f'<div class="banner banner-ok"><b>初步判定: 量化本身没问题 (未检测到重复)</b><br>'
                f'<span style="color:#666">→ 若生成通顺合理，精度问题在推理框架；若仍异常请人工复核</span></div>')


def _render_boundary(results: List[Dict]) -> str:
    """Section 1: 定界 (inference_check 结果)"""
    if not results:
        return ""
    parts = ["<h2>① 定界 — 排除框架影响 (inference_check)</h2>"]
    parts.append(_verdict_banner(results))
    parts.append(
        '<p class="meta">判定原则: 生成通顺合理 → 量化本身没问题，精度问题在推理框架；'
        '生成乱码/重复 → 量化本身有问题，需 L2 逐层定位。</p>'
    )
    parts.append("<table><thead><tr>"
                 "<th>#</th><th>问题 (前 40 字)</th><th>生成 tokens</th>"
                 "<th>耗时</th><th>思维链截断</th></tr></thead><tbody>")
    for i, r in enumerate(results):
        msgs = r.get("messages", [])
        last_user = ""
        for m in msgs:
            if m.get("role") == "user":
                last_user = m.get("content", "")
        q = _esc((last_user[:40] + "…") if len(last_user) > 40 else last_user)
        trunc = "是" if r.get("thinking_truncated") else "否"
        parts.append(
            f"<tr><td>{i+1}</td><td>{q}</td>"
            f"<td>{r.get('output_tokens', '?')}</td>"
            f"<td>{r.get('time', 0):.1f}s</td>"
            f"<td>{trunc}</td></tr>"
        )
    parts.append("</tbody></table>")
    # 生成文本预览
    parts.append("<h3>生成文本预览</h3>")
    for i, r in enumerate(results):
        gen = r.get("generated") or "(无正文 / 思维链截断)"
        thinking = r.get("thinking") or ""
        parts.append(f"<p><b>#{i+1} 生成:</b></p>")
        # Keep the complete answer available in the report; the terminal
        # logger intentionally prints only a short preview.
        parts.append(f'<pre class="gen">{_esc(gen)}</pre>')
        if thinking:
            parts.append(f"<p style='color:#666'><b>#{i+1} 思维链:</b></p>")
            parts.append(f'<pre class="gen" style="color:#666">{_esc(thinking)}</pre>')
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# subgraph 级别定位 (L2) — v1
# ---------------------------------------------------------------------------

def _render_l2(results: List[Dict]) -> str:
    """Section 2: subgraph 级别定位 (L2 root_suspect + impact_boundary)"""
    if not results:
        return ""
    parts = ["<h2>② subgraph 级别定位 — 坏权重定位 (L2 反事实诊断)</h2>"]

    # Recovery 公式静态块
    parts.append(
        '<div class="formula">'
        '<b>Recovery 公式:</b> (baseline_l2 − patched_l2) / baseline_l2<br>'
        '70% = 把该子图改回正确值后整层误差能消除 70%；0% = 没效果；'
        '负值 = softmax 耦合 (Q/KV 单独替换打破 Q-K 相对关系)。<br>'
        '基线 baseline_l2 = rel_l2(quant_out, ref_out) = ‖a−b‖₂ / ‖b‖₂ (整层量化相对误差)。'
        '</div>'
    )

    # 收集所有子图名 (保持顺序)
    all_subgraphs: List[str] = []
    for r in results:
        for name in r.get("subgraphs", {}):
            if name not in all_subgraphs:
                all_subgraphs.append(name)

    # 表头
    parts.append("<table><thead><tr>")
    parts.append('<th title="层号">Layer</th>')
    parts.append('<th title="整层量化相对 L2 误差，越低越好↓">基线误差↓</th>')
    parts.append('<th title="输入 patch 恢复率，越高越好↑；高=上游污染严重">输入恢复↑</th>')
    parts.append('<th title="主要嫌疑算子 (红框=本表已高亮)">主要嫌疑</th>')
    parts.append('<th title="Recovery 最大的可 patch 子图，绿框=往后 OK">关键边界</th>')
    for name in all_subgraphs:
        parts.append(f"<th>{_esc(name)}</th>")
    parts.append("</tr></thead><tbody>")

    for r in results:
        layer = r.get("layer_idx", "?")
        base_l2 = r.get("baseline_l2")
        input_rec = r.get("input_recovery")
        root = r.get("root_suspect")
        impact_bnd = r.get("impact_boundary")
        subgraphs = r.get("subgraphs", {})
        qtypes = r.get("subgraph_quant_types", {})

        base_cls = _err_class(base_l2, "baseline_l2")
        rec_cls = _rec_class(input_rec)
        root_cell = f'<td class="root-suspect">{_esc(root or "—")}</td>'
        impact_cls = "impact-boundary" if impact_bnd else ""
        impact_cell = f'<td class="{impact_cls}">{_esc(impact_bnd or "—")}</td>'

        parts.append(
            f"<tr><td>{layer}</td>"
            f'<td class="{base_cls}">{_fmt_err(base_l2)}</td>'
            f'<td class="{rec_cls}">{_fmt_pct(input_rec)}</td>'
            f"{root_cell}{impact_cell}"
        )
        for name in all_subgraphs:
            val = subgraphs.get(name)
            qt = qtypes.get(name, "")
            if qt == "FLOAT":
                parts.append('<td class="na" title="未量化">FLOAT†</td>')
            elif qt == "UNPATCHABLE":
                parts.append('<td class="na" title="旋转对齐失败，不可 patch">SKIP</td>')
            elif val is None:
                parts.append('<td class="na">—</td>')
            else:
                cls = _rec_class(val)
                # root_suspect 命中单元格加红框
                if root and name == root:
                    cls = f"{cls} root-suspect"
                # impact_boundary 命中加绿框
                if impact_bnd and name == impact_bnd:
                    cls = f"{cls} impact-boundary"
                parts.append(f'<td class="{cls}" title="{_esc(qt)}">{_fmt_pct(val)}</td>')
        parts.append("</tr>")
    parts.append("</tbody></table>")

    # 诊断文字 (root_suspect 汇总)
    parts.append("<h3>诊断汇总</h3>")
    parts.append("<ul>")
    for r in results:
        layer = r.get("layer_idx", "?")
        root = r.get("root_suspect") or "未定位"
        impact = r.get("impact_boundary") or "无"
        base = r.get("baseline_l2")
        parts.append(
            f"<li><b>Layer {layer}</b>: 主要嫌疑 = "
            f'<span style="color:#b30000;font-weight:700">{_esc(root)}</span>'
            f" · 关键边界 = <span style='color:#1a6b1a'>{_esc(impact)}</span>"
            f" · 基线误差 = {_fmt_err(base)}</li>"
        )
    parts.append("</ul>")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# v1 主入口 (backward-compat)
# ---------------------------------------------------------------------------

def generate_html_report(
    boundary_results: Optional[List[Dict]] = None,
    l2_results: Optional[List[Dict]] = None,
    model_name: str = "",
    output_path: Optional[str] = None,
) -> str:
    """生成 HTML 报告 (v1: 定界 + subgraph 级别定位)

    Args:
        boundary_results: hf_inference_check 返回的 list[dict]，None=未做定界
        l2_results: diagnose_layers 返回的 list[dict]，None=未做 L2
        model_name: 模型名 (用于标题和文件名)
        output_path: 输出路径，None=自动生成 reports/html_report_<model>_<ts>.html

    Returns:
        HTML 文件路径
    """
    ts = time.strftime("%Y%m%d_%H%M%S")
    safe_model = model_name.replace("/", "_").replace(" ", "_") if model_name else "model"
    if output_path is None:
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        reports_dir = os.path.join(repo_root, "reports")
        os.makedirs(reports_dir, exist_ok=True)
        output_path = os.path.join(reports_dir, f"html_report_{safe_model}_{ts}.html")

    body_parts = [
        f"<h1>acc_bench 精度对齐报告</h1>",
        f'<p class="meta">模型: <b>{_esc(model_name or "—")}</b> · 生成: {ts}</p>',
    ]

    if boundary_results is None and l2_results is None:
        body_parts.append('<p class="banner banner-warn">未提供任何诊断数据。</p>')
    else:
        if boundary_results is not None:
            body_parts.append(_render_boundary(boundary_results))
        if l2_results is not None:
            body_parts.append(_render_l2(l2_results))

    body_parts.append(
        '<div class="legend">'
        '<b>图例:</b> '
        '<span class="good">绿=好</span> / '
        '<span class="warn">黄=警告</span> / '
        '<span class="bad">红=坏</span> / '
        '<span class="na">灰=未量化(FLOAT†)/不可替换(SKIP)/无数据</span>；'
        '恢复类↑越高越好 / 误差类↓越低越好 / 翻转类↓越低越好；'
        '<b style="color:#b30000">红框=主要嫌疑</b> (root_suspect)；'
        '<b style="color:#1a6b1a">绿框=关键边界</b> (impact_boundary, 往后 OK)。'
        '</div>'
    )

    html_doc = (
        "<!doctype html><html lang='zh'><head><meta charset='utf-8'>"
        f"<title>acc_bench 报告 - {_esc(model_name)}</title>"
        f"<style>{_CSS}</style></head><body>"
        + "\n".join(body_parts)
        + "</body></html>"
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_doc)
    return output_path


# ===========================================================================
# v2: 产品化 HTML (ReportData -> 自包含交互报告)
# ===========================================================================

from .report_schema import ReportData

# 配色 (用户指定)
_COL_REF = "#2563eb"      # ref 蓝色系
_COL_QUANT = "#ea580c"    # quant 橙红
_COL_GOOD = "#16a34a"
_COL_WARN = "#f59e0b"
_COL_BAD = "#dc2626"
_COL_NA = "#9ca3af"
_COL_BG = "#f8fafc"
_COL_CARD = "#ffffff"
_COL_BORDER = "#e2e8f0"
_COL_INK = "#0f172a"
_COL_MUTED = "#64748b"

_V2_CSS = """
:root { --ref:#176B55; --quant:#C2705C; --good:#238164; --warn:#B87710; --bad:#A53939; --na:#8A929C; --deg:#D97706;
  --paper:#F4F1EA; --bg:#F4F1EA; --card:#FFFDF8; --border:#D9D4C9; --border-strong:#B8C4C5;
  --ink:#17202B; --muted:#66717F; --faint:#87909B; --navy:#112B3A; --navy-2:#173F51;
  --mint:#96E6C3; --accent:#176B55; --accent-soft:#E2F7EC; --amber:#FFCB74; --code-bg:#F0EDE5;
  --shadow:0 18px 45px rgba(17,43,58,.09); --mono:'SFMono-Regular',Consolas,'Liberation Mono',monospace; }
* { box-sizing: border-box; margin:0; padding:0; }
body { font-family: -apple-system,'Inter','Helvetica Neue','Segoe UI','PingFang SC','Microsoft YaHei',sans-serif;
  margin:0; color:var(--ink); font-size:15px; line-height:1.65; -webkit-font-smoothing:antialiased;
  background:linear-gradient(rgba(17,43,58,.035) 1px,transparent 1px),
    linear-gradient(90deg,rgba(17,43,58,.035) 1px,transparent 1px),var(--paper);
  background-size:32px 32px; }
.wrap { max-width:1120px; margin:0 auto; padding:42px 34px 140px; }
a { color:var(--accent); text-decoration:none; }
/* Hero header */
.topbar { position:relative; overflow:hidden; margin-bottom:22px; padding:30px 34px 32px; color:#fff;
  background:linear-gradient(135deg,var(--navy),var(--navy-2)); border:1px solid rgba(255,255,255,.08);
  border-radius:22px; box-shadow:var(--shadow); }
.topbar::after { content:'AB'; position:absolute; right:-14px; bottom:-74px; color:rgba(150,230,195,.07);
  font-family:var(--mono); font-size:190px; font-weight:900; line-height:1; pointer-events:none; }
.report-kicker { position:relative; z-index:1; display:flex; align-items:center; gap:10px; margin-bottom:18px;
  color:var(--mint); font-family:var(--mono); font-size:11px; font-weight:800; letter-spacing:.11em; text-transform:uppercase; }
.brand-mark { display:inline-grid; place-items:center; width:30px; height:30px; color:var(--navy); background:var(--mint);
  border-radius:9px; font-family:var(--mono); font-size:11px; font-weight:900; letter-spacing:0;
  box-shadow:inset 0 0 0 1px rgba(255,255,255,.45); }
.topbar h1 { position:relative; z-index:1; max-width:760px; font-size:clamp(30px,4vw,46px); font-weight:800;
  line-height:1.08; letter-spacing:-.045em; margin-bottom:10px; }
.topbar .ts { position:relative; z-index:1; color:#C9D8DE; font-size:13px; font-weight:450; }
.topbar .accent-line { position:relative; z-index:1; width:44px; height:3px; background:var(--mint); border-radius:2px; margin-top:18px; }
/* Sticky nav */
.nav { position:sticky; top:10px; z-index:20; display:flex; gap:5px; margin-bottom:24px; padding:7px; flex-wrap:wrap;
  background:rgba(255,253,248,.94); border:1px solid rgba(17,43,58,.12); border-radius:12px;
  box-shadow:0 8px 24px rgba(17,43,58,.07); backdrop-filter:blur(12px); }
.nav a { padding:7px 13px; font-size:12px; font-weight:700; color:var(--muted); border-radius:8px;
  border:1px solid transparent; transition:all .16s ease; }
.nav a:hover { color:var(--navy); background:#F5F2EB; }
.nav a.active { color:var(--navy); background:var(--accent-soft); border-color:rgba(23,107,85,.2); }
/* Sections */
section { background:var(--card); border:1px solid rgba(17,43,58,.12); border-radius:18px; padding:30px 32px; margin-bottom:24px;
  box-shadow:0 10px 30px rgba(17,43,58,.055); scroll-margin-top:88px; }
section > h2 { margin:0 0 6px 0; color:var(--navy); font-size:20px; font-weight:800; letter-spacing:-.025em; border:none; padding:0; }
section > h2 .section-hint { font-size:13px; color:var(--muted); font-weight:400; margin-left:8px; }
section > .sec-desc { color:var(--muted); font-size:14px; margin:0 0 24px 0; line-height:1.5; }
/* Grid + cards */
.grid { display:grid; gap:12px; }
.grid > * { min-width:0; }
.cols-2 { grid-template-columns:1fr 1fr; }
.cols-3 { grid-template-columns:repeat(3,1fr); }
@media(max-width:900px){ .cols-2,.cols-3{ grid-template-columns:1fr; } }
.card { background:#FAF8F3; border:1px solid var(--border); border-radius:12px; padding:17px 20px;
  transition:all .2s cubic-bezier(.4,0,.2,1); }
.card:hover { border-color:var(--border-strong); box-shadow:0 8px 18px rgba(17,43,58,.07); transform:translateY(-1px); }
.card h3 { margin:0 0 8px 0; color:var(--muted); font-family:var(--mono); font-size:11px; font-weight:700; letter-spacing:.035em; }
.card .val { font-size:22px; font-weight:750; color:var(--navy); letter-spacing:-.02em; }
.card .sub { font-size:12px; color:var(--muted); margin-top:4px; word-break:break-all; }
.kv { display:grid; grid-template-columns:100px 1fr; gap:4px 12px; font-size:13px; }
.kv .k { color:var(--faint); }
.kv .v { color:var(--ink); word-break:break-all; }
.kv .v.input-preview { max-height:180px; overflow:auto; white-space:pre-wrap; overflow-wrap:anywhere;
  padding:6px 8px; background:#fff; border:1px solid var(--border); border-radius:7px; }
/* Pills */
.pill { display:inline-block; padding:4px 12px; border-radius:999px; font-size:12px; font-weight:500; letter-spacing:.01em; }
.pill.ok { background:#DFF4E9; color:#176B55; }
.pill.bad { background:#F8E3E0; color:#8D2F2F; }
.pill.warn { background:#FFF0D2; color:#855600; }
.pill.muted { background:#ECE9E2; color:#66717F; }
/* Hit badge — prominent, satisfying */
.hit-badge { display:inline-flex; align-items:center; gap:6px; padding:6px 14px; border-radius:999px;
  font-size:13px; font-weight:600; letter-spacing:.01em; }
.hit-badge.hit { background:linear-gradient(135deg,#E8F5E9,#D4EDDA); color:#2E6B2E; border:1px solid #C3E6CB; }
.hit-badge.miss { background:linear-gradient(135deg,#FBEAEA,#F8D7DA); color:#8B3838; border:1px solid #F5C2C7; }
.verdict { font-size:24px; font-weight:700; }
.conf-wrap { background:var(--border); border-radius:6px; height:8px; overflow:hidden; margin-top:6px; }
.conf-bar { height:100%; background:var(--accent); border-radius:6px; transition:width .4s cubic-bezier(.4,0,.2,1); }
/* Help icon */
.help { display:inline-block; width:17px; height:17px; border-radius:50%; background:var(--accent-soft); color:var(--accent);
  text-align:center; line-height:16px; font-size:10px; cursor:help; margin-left:4px; padding:0; border:0;
  font-family:inherit; vertical-align:middle; user-select:none;
  transition:all .15s; }
.help:hover { background:var(--accent); color:#fff; }
/* Charts */
.chart { width:100%; overflow:visible; }
.chart-scroll { width:100%; overflow-x:auto; overscroll-behavior-inline:contain; scrollbar-width:thin; }
.axis text { font-size:10px; fill:var(--muted); }
.axis line,.axis path { stroke:var(--border); }
.gridline { stroke:#E8E3D9; }
.bar { cursor:pointer; transition:opacity .15s; }
.bar:hover { opacity:.75; }
/* Legend */
.legend-row { display:flex; align-items:center; gap:16px; font-size:12px; color:var(--muted); flex-wrap:wrap;
  padding-top:14px; border-top:1px solid var(--border); }
.legend-chip { display:inline-block; width:10px; height:10px; border-radius:2px; vertical-align:middle; margin-right:4px; }
/* Tables */
table.grid-tbl { width:100%; border-collapse:collapse; font-size:12px; }
table.grid-tbl th, table.grid-tbl td { border:0; border-bottom:1px solid var(--border); padding:9px 11px; }
table.grid-tbl th { background:var(--navy); text-align:left; font-weight:700; color:#D7E4E9; position:sticky; top:0; z-index:1; }
table.grid-tbl td.repair { outline:2px solid var(--good); outline-offset:-2px; }
table.grid-tbl td.source { outline:2px solid var(--bad); outline-offset:-2px; }
table.grid-tbl tr:hover td { background:var(--accent-soft); }
.table-scroll { overflow:auto; max-height:min(68vh,720px); border:1px solid var(--border); border-radius:11px; background:#fff; }
.table-scroll table.grid-tbl { min-width:720px; margin:-1px; width:calc(100% + 2px); }
.toggle-btn { display:inline-flex; align-items:center; gap:6px; padding:8px 14px; font-size:12px; font-weight:800;
  color:var(--navy); background:var(--mint); border:1px solid rgba(17,43,58,.14); border-radius:8px; cursor:pointer; margin-bottom:8px; }
.toggle-btn:hover { background:var(--accent-soft); }
.l1-metrics { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:10px 20px; align-items:start; }
/* L2 layers */
.l2layer { margin-bottom:18px; padding:18px 18px 18px 20px; background:#FAF8F3; border:1px solid var(--border);
  border-left:4px solid #B8C4C5; border-radius:0 12px 12px 0; }
.l2layer.fd { border-left-color:var(--accent); box-shadow:0 8px 20px rgba(23,107,85,.07); }
.l2layer.jump-target { background:var(--accent-soft); }
.l2layer h3 { margin:0 0 12px 0; color:var(--navy); font-size:15px; font-weight:800; }
.tag { font-size:10px; font-weight:600; padding:2px 7px; border-radius:4px; margin-right:6px; vertical-align:middle; letter-spacing:.02em; }
.tag.fd { background:#DFF4E9; color:#176B55; }
.tag.max { background:#F8E3E0; color:#8D2F2F; }
/* Modal */
.modal-ov { position:fixed; inset:0; background:rgba(0,0,0,.25); display:none; align-items:center; justify-content:center; z-index:50;
  backdrop-filter:blur(2px); }
.modal-ov.show { display:flex; }
.modal { background:var(--card); border:1px solid rgba(17,43,58,.12); border-radius:16px; max-width:520px; width:90%; padding:24px 28px;
  box-shadow:var(--shadow); }
.modal h3 { margin-top:0; font-size:16px; font-weight:600; }
.modal .formula { background:var(--code-bg); border-left:3px solid var(--accent); padding:8px 12px;
  font-family:'SF Mono','Fira Code',ui-monospace,monospace; margin:8px 0; font-size:12px; border-radius:0 4px 4px 0; }
.modal .close { float:right; cursor:pointer; color:var(--faint); font-size:16px; transition:color .15s;
  border:0; background:transparent; padding:2px 4px; line-height:1; }
.modal .close:hover { color:var(--ink); }
/* Alert */
.alert-invalid { background:#F8E3E0; border:1px solid var(--bad); color:#8B3838; padding:12px 16px; border-radius:10px;
  font-weight:500; margin-bottom:12px; }
/* Comparison semantics: make the run contract impossible to miss. */
.scope-banner { position:relative; overflow:hidden; margin-bottom:14px; padding:18px 20px; border:1px solid var(--border);
  border-left:5px solid var(--na); border-radius:12px; background:#FAF8F3; }
.scope-banner.weight { border-left-color:var(--accent); background:linear-gradient(135deg,#F4FBF7,#FAF8F3); }
.scope-banner.joint { border-left-color:var(--warn); background:linear-gradient(135deg,#FFF7E7,#FAF8F3); }
.scope-banner.unknown { border-left-color:var(--na); }
.scope-kicker { color:var(--muted); font-family:var(--mono); font-size:10px; font-weight:800; letter-spacing:.09em;
  text-transform:uppercase; }
.scope-title { margin:3px 0 4px; color:var(--navy); font-size:19px; font-weight:800; letter-spacing:-.02em; }
.scope-copy { max-width:900px; color:var(--muted); font-size:13px; line-height:1.55; }
.scope-meta { display:flex; gap:7px; flex-wrap:wrap; margin-top:10px; }
.scope-meta .pill { padding:3px 9px; font-family:var(--mono); font-size:10px; font-weight:750; }
.scope-note, .context-note { margin-top:10px; padding:9px 12px; color:var(--muted); background:rgba(255,255,255,.58);
  border:1px dashed var(--border-strong); border-radius:8px; font-size:12px; line-height:1.5; }
.context-note { margin:0 0 12px; background:#F8F6F0; }
.l2-metrics { margin-top:12px; max-height:360px; }
.l2-metrics table.grid-tbl { width:100%; margin:0; }
.metric-good { color:var(--good); font-weight:750; }
.metric-warn { color:var(--warn); font-weight:750; }
.metric-bad { color:var(--bad); font-weight:750; }
.metric-na { color:var(--na); }
.position-picker { display:flex; gap:6px; flex-wrap:wrap; margin-top:8px; }
.position-btn { min-width:34px; padding:4px 8px; color:var(--muted); background:#fff; border:1px solid var(--border);
  border-radius:7px; font-family:var(--mono); font-size:10px; font-weight:750; cursor:pointer; }
.position-btn:hover { color:var(--navy); border-color:var(--border-strong); }
.position-btn.active { color:#fff; background:var(--accent); border-color:var(--accent); }
.position-btn.topk-match { color:#176B55; border-color:#B8DCCB; background:#F1FAF5; }
.position-btn.topk-partial { color:#8A5A08; border-color:#E7C783; background:#FFF7E3; }
.position-btn.topk-mismatch { color:#fff; border-color:var(--bad); background:var(--bad); }
.position-btn.topk-match.active { color:#fff; background:var(--accent); border-color:var(--accent); }
.position-btn.topk-partial.active { color:#fff; background:var(--warn); border-color:var(--warn); box-shadow:0 0 0 2px rgba(184,119,16,.20); }
.position-btn.topk-mismatch:hover { color:#fff; border-color:#7F2929; background:#7F2929; }
.position-btn.topk-mismatch.active { color:#fff; background:#7F2929; border-color:#7F2929; box-shadow:0 0 0 2px rgba(165,57,57,.22); }
#logits_topk .chart-scroll { max-height:520px; overflow:auto; }
/* End-to-end generation comparison */
.output-box { margin:0; max-height:320px; overflow:auto; padding:13px 15px; color:var(--ink); background:#fff;
  border:1px solid var(--border); border-radius:9px; white-space:pre-wrap; overflow-wrap:anywhere;
  font-family:var(--mono); font-size:12px; line-height:1.65; }
.health-row { display:flex; gap:7px; flex-wrap:wrap; margin:10px 0 0; }
.token-cell { max-width:360px; white-space:pre-wrap; overflow-wrap:anywhere; font-family:var(--mono); }
.token-match { color:var(--good); }
.token-mismatch { color:var(--bad); font-weight:700; }
/* Error samples */
.err-sample { background:var(--card); border:1px solid var(--border); border-radius:8px; padding:10px 14px; margin:8px 0; font-size:12px; }
.err-sample pre { white-space:pre-wrap; word-break:break-word; margin:6px 0 0 0; background:var(--code-bg); padding:8px; border-radius:4px; font-size:11px; }
/* Empty state — intentional, not broken */
.empty { display:flex; flex-direction:column; align-items:center; justify-content:center; gap:8px;
  padding:48px 24px; text-align:center; }
.empty .empty-icon { width:40px; height:40px; border-radius:50%; background:var(--code-bg); display:flex; align-items:center; justify-content:center;
  font-size:18px; color:var(--faint); }
.empty .empty-text { color:var(--faint); font-size:14px; font-weight:400; }
.empty .empty-hint { color:var(--faint); font-size:12px; }
/* Tip */
.tip { font-size:12px; color:var(--muted); margin-top:8px; line-height:1.5; }
svg { font-family:-apple-system,'Inter','PingFang SC',sans-serif; }
/* Smooth scroll */
html { scroll-behavior:smooth; }
@media (max-width:680px) {
  body { font-size:14px; line-height:1.55; }
  .wrap { max-width:none; padding:28px 14px 80px; }
  .topbar { margin-bottom:20px; padding:24px 22px 26px; border-radius:17px; }
  .topbar h1 { font-size:24px; line-height:1.22; overflow-wrap:anywhere; }
  .topbar .ts { display:block; font-size:12px; overflow-wrap:anywhere; }
  .nav { margin:0 -2px 20px; padding-top:4px; flex-wrap:nowrap; overflow-x:auto; scrollbar-width:none; }
  .nav::-webkit-scrollbar { display:none; }
  .nav a { flex:0 0 auto; padding:9px 12px; }
  section { padding:20px 16px; margin-bottom:18px; border-radius:14px; scroll-margin-top:62px; }
  section > h2 { font-size:17px; line-height:1.4; }
  section > h2 .section-hint { display:block; margin:4px 0 0; font-size:12px; }
  section > .sec-desc { font-size:13px; margin-bottom:18px; }
  .card { padding:14px 16px; }
  .card:hover { transform:none; }
  .card .val { font-size:19px; overflow-wrap:anywhere; }
  .kv { grid-template-columns:72px minmax(0,1fr); }
  .l1-metrics { grid-template-columns:1fr 1fr; gap:12px; }
  .l2layer { padding-left:12px; }
  #l2, #logits_scatter, #logits_topk, #logits_lines, #logits_hist { min-width:0; overflow-x:auto; }
  .chart { min-width:680px; }
  .modal { width:calc(100% - 28px); padding:20px; max-height:80vh; overflow:auto; }
}
@media print {
  :root { --bg:#fff; --card:#fff; }
  body { background:#fff; color:#000; font-size:11pt; }
  .wrap, .main-area .wrap { max-width:none !important; padding:0 !important; }
  .nav, .sidebar, .toggle-btn, .help, .modal-ov { display:none !important; }
  .main-area { margin-left:0 !important; }
  .topbar { margin-bottom:18px; }
  section { box-shadow:none; break-inside:avoid; padding:18px; margin-bottom:14px; }
  .card { break-inside:avoid; box-shadow:none; }
  .chart-scroll, .table-scroll { overflow:visible; max-height:none; }
  .chart { min-width:0 !important; }
  #l1fulltbl { display:block !important; }
}
"""

# ---- v2 JS (vanilla, SVG 图表 + 指标解释 modal) ----
_V2_JS = r"""
(function(){
"use strict";
const R = window.__REPORT__;
const C={ref:"#176B55",quant:"#C2705C",topk:"#4F46E5",good:"#238164",warn:"#B87710",bad:"#A53939",na:"#8A929C",deg:"#D97706",ink:"#17202B",muted:"#66717F",border:"#D9D4C9"};
const el=(id)=>document.getElementById(id);
const ns="http://www.w3.org/2000/svg";
function E(tag,attrs){const e=document.createElementNS(ns,tag);if(attrs)for(const k in attrs)e.setAttribute(k,attrs[k]);return e;}
function num(v){return (v===null||v===undefined||Number.isNaN(v))?null:Number(v);}
function pct(v,d=1){const n=num(v);return n===null?"—":(n*100).toFixed(d)+"%";}
function fix(v,d=4){const n=num(v);return n===null?"—":n.toFixed(d);}
// cos_sim 颜色档
function cosTier(v){const n=num(v);if(n===null)return C.na;if(n>=0.99)return C.good;if(n>=0.95)return C.warn;return C.bad;}
// recovery 颜色档 (高=好)
function recTier(v){const n=num(v);if(n===null)return C.na;if(n<0)return C.bad;if(n>=0.8)return C.good;if(n>=0.3)return C.warn;return C.bad;}
// "degraded" 量化类型: W4 系列 (W4A8_DYNAMIC / W4A4_DYNAMIC 等) 比 W8 更激进, 用橙色区分
function isDegradedQuantType(qt){if(!qt)return false;const s=String(qt).toUpperCase();return s.indexOf("W4")>=0;}
function clamp(x,a,b){return Math.max(a,Math.min(b,x));}
function esc(s){return String(s==null?"":s).replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;"}[c]));}

// ---- metric 指标解释 ----
const HELP={
  cos_sim:{t:"Cosine Similarity (逐 token 余弦)",f:"cos = (a·b)/(‖a‖·‖b‖)",r:"0~1, ≥0.99 视为对齐",i:"<0.99 即发散: 该层 ref/quant 输出方向偏离; 越接近 1 量化越无损。"},
  rel_l2:{t:"相对 L2 误差",f:"rel_l2 = ‖a−b‖₂ / ‖b‖₂",r:"越低越好 <0.05 OK",i:"整层输出的相对误差; 越低量化精度越接近参考。"},
  snr:{t:"信噪比 (dB)",f:"snr = 10·log₁₀(P_signal / P_noise)",r:">20dB 优, <10dB 差",i:"信号能量与噪声能量之比; dB 越高误差越被信号压制。"},
  patch_recovery:{t:"Patch Recovery (子图恢复率)",f:"(baseline_l2 − patched_l2) / baseline_l2",r:"0~1 高=好; 负=耦合",i:"把该子图改回 ref 输出后整层误差的消除比例; 70% 即消除 70%; 负值=softmax 耦合 (Q/KV 单独替换打破相对关系)。"},
  self_rot_err:{t:"SelfRotErr (自量化误差)",f:"‖quant_op(ref_in) − ref_op(ref_in)‖ / ‖ref_op(ref_in)‖",r:"<0.02 好, >=0.10 坏",i:"喂同样 ref 输入, 该算子自身量化的相对误差; 隔离了上游污染, 是最强的根因证据。"},
  rot_b_err:{t:"RotBErr (边界误差 含上游)",f:"‖quant_op(quant_in) − ref_op(ref_in)‖ / ‖ref_op(ref_in)‖",r:"<0.05 好, >=0.15 坏",i:"喂量化输入测的边界误差, 包含上游累计污染; 比 SelfRotErr 高则说明上游也在贡献误差。"},
  input_recovery:{t:"Input Recovery (上游输入恢复)",f:"(baseline_l2 − patched_input_l2) / baseline_l2",r:"0~1 高=上游污染重",i:"把上游输入改回 ref 后整层误差的消除比例; 高即上游污染严重, 问题在前面层。"},
  baseline_l2:{t:"整层基线误差",f:"rel_l2(quant_out, ref_out) = ‖a−b‖₂/‖b‖₂",r:"<0.05 OK, >=0.20 坏",i:"整层量化输出的相对 L2 误差, L2 诊断的基线 (反事实的对照)。"},
  top1_match:{t:"Top-1 Token Match",f:"argmax(ref_logits)==argmax(quant_logits)",r:"True/False",i:"该位置 ref 和 quant 选中的下一个 token 是否一致; False 即量化改变了贪心解码路径。"},
  token_wise_kl:{t:"Token-wise KL(ref‖quant)",f:"Σ p·log(p/q), p=ref, q=quant 概率",r:"越低越好 ~0",i:"每位置参考分布相对量化分布的 KL; 高则两分布差异大, 选词路径可能分叉。"},
  token_wise_cos:{t:"Token-wise Logits Cos",f:"cos(ref_logits, quant_logits) per position",r:"0~1 高=好",i:"每位置全词表 logits 的余弦相似度; 高即分布形状一致。"},
  topk_overlap:{t:"Top-K Overlap",f:"|ref_topk ∩ quant_topk| / k",r:"0~1 高=好",i:"候选词集合重合度; 低即候选集都不同, 解码分歧大。"},
  first_divergence:{t:"L1 首个诊断候选层",f:"Δᵢ=cosᵢ−cosᵢ₋₁；baseline=median(最近 10 个 Δ)；MAD=median(|Δ−baseline|)",r:"优先: Δ < baseline−5×MAD 且 Δ<−0.005，并检查后续 3 层是否持续；回退: 首个 cos<0.99",i:"优先寻找相邻层之间异常且持续的局部突降，避免把逐层缓慢累积误差的首次阈值越界误当成根因。未找到显著突降时，才使用首个 cos_sim<0.99 的层作为辅助候选。"},
  logits_scatter:{t:"Ref vs Quant Logits 散点图",f:"每个点=(同一 position、同一 Top-K token 的 Ref logit, Quant logit)",r:"横轴=Ref logit；纵轴=Quant logit；虚线 y=x 表示完全一致",i:"每个已采集 position 分别取 generation config 指定的 Ref/Quant Top-K 并集，只展示真正可能参与解码的候选。点越贴近 y=x 越一致；虚线上方表示 Quant logit 更高，下方表示更低。候选超过 2000 点时在 Top-K 候选池内做确定性均匀下采样。"},
  logits_hist_overlay:{t:"Logits 分布直方图 Overlay",f:"横轴=logit 数值区间；纵轴=落入该区间的 Ref/Quant logit 数量",r:"两条分布越重合越好；整体平移、变宽或长尾增多表示量化改变了分布",i:"把所有已采集 position × vocab 的 logits 汇总分箱，并叠加显示 Ref 与 Quant。它适合观察整体偏置、尺度变化和异常值，但不保留 token 对应关系；即使直方图重合，也仍需结合逐位置 cos、KL、Top-K overlap 判断具体 token 是否对齐。"},
  confidence:{t:"定位可信度",f:"启发式: 命中 source+repair=0.9; 仅一项=0.6; 都无=0.2",r:"0.2/0.6/0.9",i:"L2 在首次发散层是否同时给出 source_candidate 和 best_repair_point。"},
  flip_rate:{t:"Flip Rate (离散选择翻转率)",f:"|ref_topk ∩ quant_topk| 不同的比例",r:"<0.01 好, >=0.10 坏",i:"indexer/gate 等离散算子 top-k 选择发生改变的比例; 高则路由/选择被量化颠覆。"},
  chain_delta:{t:"串联子链 Delta (误差传播链)",f:"Δ_i = baseline_l2 − patched_l2(after replacing module i back to ref)",r:"0~1, 高=该算子贡献大",i:"按 attention (q→k→v→o) 或 MLP (gate→up→down) 子图内部顺序, 逐步替换回 ref 权重, 每步的误差消除量; 高 delta 说明该算子是误差传播链上的关键节点。负值=耦合效应 (单独替换打破相对关系)。"},
};
function modal(key){const h=HELP[key];if(!h)return;const m=el("helpModal");const body=el("helpBody");
  body.innerHTML="<h3>"+esc(h.t)+"</h3><div class='formula'>"+esc(h.f)+"</div>"
    +"<p><b>典型范围:</b> "+esc(h.r)+"</p><p><b>解读:</b> "+esc(h.i)+"</p>";
  m.classList.add("show");m.setAttribute("aria-hidden","false");}
function closeModal(){const m=el("helpModal");m.classList.remove("show");m.setAttribute("aria-hidden","true");}
window.__modal=modal;window.__closeModal=closeModal;
function hIcon(key){return " <button type='button' class='help' onclick=\"__modal('"+key+"')\" title='"+esc(key)+"' aria-label='查看 "+esc(key)+" 指标说明'>?</button>";}
// ---- SVG 基础 ----
function svgBox(w,h){const s=E("svg",{viewBox:"0 0 "+w+" "+h,width:"100%",class:"chart"});return s;}
function axisX(s,x0,x1,y,w){const g=E("g",{class:"axis"});for(const v of [x0,x1]){}const ln=E("line",{x1:30,y1:y,x2:w,y2:y,stroke:C.border});g.appendChild(ln);return g;}
function axisLabels(){/*placeholder*/}

// ====================================================================
// Overview
// ====================================================================
function renderScopeBanner(o){
  const scope=o.comparison_scope||"unknown";
  const hasL1=(R.l1_layers||[]).length>0;
  const hasL2=(R.l2_results||[]).length>0;
  let cls="unknown",title="对比口径未记录",copy="这是历史报告或外部构造报告，无法确认运行时是否启用了 Activation QDQ。不要把它自动解释为仅权重对比。";
  let activationBadge='<span class="pill muted">ACT QDQ UNKNOWN</span>';
  if(scope==="weight_plus_activation_qdq"||o.activation_quant_enabled===true){
    cls="joint";title="权重 + 激活 QDQ 联合仿真";
    copy="Activation QDQ 只作用于 Quant 侧 descriptor 匹配算子。L1 的 block output 同时包含权重误差与激活 QDQ 误差的累计传播，单次结果不能拆分两者贡献。";
    activationBadge='<span class="pill warn">ACT QDQ ON · QUANT SIDE</span>';
  }else if(scope==="weight_only"||o.activation_quant_enabled===false){
    cls="weight";title="仅权重误差定位";
    copy="Activation QDQ 未启用。L1 比较 Ref 与 Quant checkpoint 权重路径的逐层 block output，用于观察权重映射、反量化及量化权重误差的累计传播。";
    activationBadge='<span class="pill ok">ACT QDQ OFF</span>';
  }
  const tags=[activationBadge];
  if(o.quant_method)tags.push('<span class="pill muted">WEIGHT · '+esc(o.quant_method)+'</span>');
  if(o.activation_quant_enabled===true&&o.activation_quant_type)tags.push('<span class="pill warn">TYPE · '+esc(o.activation_quant_type)+'</span>');
  if(o.activation_quant_enabled===true&&o.activation_quant_backend)tags.push('<span class="pill muted">BACKEND · '+esc(o.activation_quant_backend)+'</span>');
  if(o.activation_quant_enabled===true&&o.activation_quant_group_size!=null)tags.push('<span class="pill muted">GROUP · '+esc(o.activation_quant_group_size)+'</span>');
  let l2Note="";
  if(hasL2)l2Note='<div class="scope-note"><b>L2 口径独立：</b>子图反事实诊断基于 checkpoint 权重路径，不重放 L1 的 Activation QDQ hooks。</div>';
  const prefix=hasL1?"L1 COMPARISON SCOPE":"DIAGNOSTIC SCOPE";
  return '<div class="scope-banner '+cls+'"><div class="scope-kicker">'+prefix+'</div><div class="scope-title">'+title+'</div>'+
    '<div class="scope-copy">'+copy+'</div><div class="scope-meta">'+tags.join("")+'</div>'+l2Note+'</div>';
}
function renderOverview(){
  const o=R.overview||{};const root=el("overview");
  const st=R.run_status||"PARTIAL";
  const statusPill = st==="SUCCESS"?"ok":(st==="PARTIAL"?"warn":(st==="INVALID_RUN"||st==="FAILED"?"bad":"warn"));
  const statusLabel={SUCCESS:"成功",PARTIAL:"部分完成",INVALID_RUN:"输入无效",INCONCLUSIVE:"结论存疑",FAILED:"失败"}[st]||st;
  let bndClass="muted",bndTxt="未做定界";
  if(o.boundary_result==="CLEAN"){bndClass="ok";bndTxt="生成通顺 (量化本身可能 OK)";}
  else if(o.boundary_result==="GARBLED"){bndClass="bad";bndTxt="生成乱码 (量化本身有问题)";}
  else if(o.boundary_result==="TRUNCATED"){bndClass="warn";bndTxt="思维链截断";}
  else if(o.boundary_result==="INTERMITTENT_LOGITS_ALIGNED"){bndClass="ok";bndTxt="当前 logits boundary 未发现明显偏差";}
  else if(o.boundary_result==="INTERMITTENT_RANKING_SENSITIVE"){bndClass="warn";bndTxt="Top-1 低 margin 敏感分叉";}
  else if(o.boundary_result==="INTERMITTENT_LOGITS_MISMATCH"){bndClass="bad";bndTxt="sampler 前 logits 已出现明显偏差";}
  let gtHit="";
  if(o.ground_truth_hit===true)gtHit='<span class="pill ok">GT 命中</span>';
  else if(o.ground_truth_hit===false)gtHit='<span class="pill bad">GT 未命中</span>';
  const conf=num(o.confidence);
  const confBar = conf===null?'<div class="val">—</div><div class="tip">无对应 L2 定位证据</div>':
    ('<div class="conf-wrap"><div class="conf-bar" style="width:'+clamp(conf*100,0,100).toFixed(0)+'%"></div></div>'
    +'<div class="tip">可信度 '+pct(conf,0)+'</div>');
  const inputMode=o.input_mode==="messages"?"Chat messages · apply_chat_template":(o.input_mode==="captured_logits"?"Captured vLLM input_ids · replay":(o.input_mode==="prompt"?"Raw prompt":(o.input_mode||"—")));
  const activationGroup=o.activation_quant_group_size==null?"":(" · group "+o.activation_quant_group_size);
  const activationText=o.activation_quant_enabled===true?("ON · "+(o.activation_quant_type||"AUTO")+" · "+(o.activation_quant_backend||"auto")+activationGroup):
    (o.activation_quant_enabled===false?"OFF":"未记录");
  let alert="";
  if(st==="INVALID_RUN"){alert='<div class="alert-invalid">输入无效 (模型加载/forward 失败或全 NaN)。以下排名仅供参考, 不可作为定论。'+hIcon("baseline_l2")+'</div>';}
  if(st==="INCONCLUSIVE"){alert='<div class="alert-invalid">结论存疑: L1 逐层对齐结果与生成定界不一致, 需人工复核 (可能 framework 层误差/截断/采样差异)。'+hIcon("cos_sim")+'</div>';}
  let capturedPanel="";
  const cr=o.captured_replay;
  if(o.boundary_issue_mode==="intermittent"&&cr){
    const verdict=cr.verdict||"—";
    const verdictCls=verdict==="INTERMITTENT_LOGITS_ALIGNED"?"ok":(verdict==="INTERMITTENT_RANKING_SENSITIVE"?"warn":"bad");
    const cosVal=cr.mean_cosine==null?"—":fix(cr.mean_cosine,6);
    const klVal=cr.max_kl==null?"—":fix(cr.max_kl,6);
    const overlapVal=cr.min_topk_overlap==null?"—":pct(cr.min_topk_overlap,1);
    capturedPanel='<div class="card" style="margin-top:12px;border-left:3px solid '+(verdictCls==="ok"?C.good:(verdictCls==="warn"?C.warn:C.bad))+'">'
      +'<h3>Captured vLLM vs Transformers Replay</h3>'
      +'<div class="scope-meta"><span class="pill muted">'+esc(cr.input_token_count||0)+' input tokens</span>'
      +'<span class="pill muted">'+esc(cr.compared_position_count==null?(cr.compared_positions||[]).length:cr.compared_position_count)+' positions</span>'
      +'<span class="pill '+verdictCls+'">'+esc(verdict)+'</span></div>'
      +'<div class="kv" style="margin-top:8px"><span class="k">Top-1 match</span><span class="v">'+esc(cr.top1_match_count||0)+' / '+esc(cr.top1_total||0)+'</span>'
      +'<span class="k">Mean cosine</span><span class="v">'+cosVal+'</span>'
      +'<span class="k">Max KL</span><span class="v">'+klVal+'</span>'
      +'<span class="k">Min Top-K overlap</span><span class="v">'+overlapVal+'</span>'
      +'<span class="k">First mismatch</span><span class="v">'+esc(cr.first_mismatch_position==null?"—":cr.first_mismatch_position)+'</span>'
      +'<span class="k">First Top-1 flip</span><span class="v">'+esc(cr.first_top1_flip_position==null?"—":cr.first_top1_flip_position)+'</span></div>'
      +(cr.missing_metrics?'<div class="tip">'+esc(cr.missing_metrics)+'</div>':"")+'</div>';
  }
  root.innerHTML = alert+renderScopeBanner(o)+
    '<div class="grid cols-3">'+
      '<div class="card"><h3>定界结论'+hIcon("baseline_l2")+'</h3><div class="pill '+bndClass+'">'+bndTxt+'</div><div class="sub">'+esc(o.boundary_result||"—")+'</div></div>'+
      '<div class="card"><h3>L1 首个诊断候选层'+hIcon("first_divergence")+'</h3><div class="val">'+(o.first_divergence_layer==null?"—":("layer "+o.first_divergence_layer))+'</div><div class="sub">delta/MAD 优先，阈值回退</div></div>'+
      '<div class="card"><h3>运行状态</h3><div class="pill '+statusPill+'">'+statusLabel+'</div><div class="sub">'+st+'</div></div>'+
    '</div>'+
    '<div class="grid cols-3" style="margin-top:12px">'+
      '<div class="card"><h3>误差边界 (关键子图)'+hIcon("patch_recovery")+'</h3><div class="val" style="color:'+C.good+'">'+esc(o.best_repair_point||"—")+'</div><div class="sub">误差峰值出现的 coarse 子图边界</div></div>'+
      '<div class="card"><h3>根因算子 (主要嫌疑)'+hIcon("self_rot_err")+'</h3><div class="val" style="color:'+C.bad+'">'+esc(o.source_candidate||"—")+'</div><div class="sub">子图内最可能的根因算子</div></div>'+
      '<div class="card"><h3>定位可信度'+hIcon("confidence")+'</h3>'+confBar+'</div>'+
    '</div>'+
    '<div class="grid cols-2" style="margin-top:12px">'+
      '<div class="card"><h3>问题链路</h3><div class="val" style="font-size:15px">'+esc(o.problem_path||"—")+'</div>'+
        (gtHit?('<div style="margin-top:6px">'+gtHit+'</div>'):"")+'</div>'+
      '<div class="card"><h3>模型 / 运行口径</h3><div class="kv">'+
        '<span class="k">模型</span><span class="v">'+esc(o.model_name||"—")+'</span>'+
        '<span class="k">量化格式</span><span class="v">'+esc(o.quant_format||"—")+'</span>'+
        '<span class="k">权重执行</span><span class="v">'+esc(o.quant_method||"—")+'</span>'+
        '<span class="k">Activation</span><span class="v">'+esc(activationText)+'</span>'+
        '<span class="k">输入方式</span><span class="v">'+esc(inputMode)+'</span>'+
        '<span class="k">设备</span><span class="v">'+esc(o.device_mode||"—")+'</span>'+
        '<span class="k">Ref</span><span class="v" style="font-size:11px">'+esc(o.ref_model_path||"—")+'</span>'+
        '<span class="k">Quant</span><span class="v" style="font-size:11px">'+esc(o.quant_model_path||"—")+'</span>'+
        '<span class="k">输入</span><span class="v input-preview" style="font-size:11px">'+esc(o.prompt||"")+'</span>'+
      '</div></div>'+
    '</div>'+capturedPanel;
}

// ====================================================================
// L1: 首个发散层卡片 + 可折叠全量表 (无柱状图)
// ====================================================================
function renderL1(){
  const root=el("l1");const Ls=R.l1_layers||[];
  if(!Ls.length){root.innerHTML='<div class="empty"><div class="empty-icon">—</div><div class="empty-text">未运行 L1 逐层对比</div><div class="empty-hint">运行 --mode l1 采集逐层 cos_sim</div></div>';el("l1legend").innerHTML="";return;}
  const BAD_THRESHOLD=0.99;
  // 首个发散层: 优先 delta 检测 (is_first_divergence), 回退到 cos_sim < 0.99
  let divIdx=-1,badIdx=-1,badCnt=0;
  for(let i=0;i<Ls.length;i++){
    const v=num(Ls[i].cos_sim);
    const isBad=(v!==null&&v<BAD_THRESHOLD)||Ls[i].is_first_divergence===true;
    if(isBad){badCnt++;if(badIdx<0)badIdx=i;}
    if(Ls[i].is_first_divergence===true&&divIdx<0)divIdx=i;
  }
  // 优先用 delta 检测的层, 没有才回退到 threshold
  const showIdx=divIdx>=0?divIdx:badIdx;
  if(showIdx<0){
    root.innerHTML='<div class="card" style="border-left:3px solid '+C.good+';background:rgba(91,140,90,0.06);padding:16px 18px">'
      +'<div style="display:flex;align-items:center;gap:10px">'
      +'<span style="font-size:22px;color:'+C.good+';line-height:1">✓</span>'
      +'<div><div style="font-size:16px;font-weight:700;color:'+C.good+'">全部对齐</div>'
      +'<div style="font-size:13px;color:var(--muted)">所有 '+Ls.length+' 层 cos_sim ≥ 0.99, 未发现发散层。</div></div>'
      +'</div></div>';
    el("l1legend").innerHTML="";
    return;
  }
  const L=Ls[showIdx];
  const v=num(L.cos_sim);
  const tier=v===null?C.na:cosTier(v);
  const tags=[];
  if(L.is_first_divergence)tags.push('<span class="tag fd">首次发散 (delta检测)</span>');
  if(L.is_max_error)tags.push('<span class="tag max">最大误差</span>');
  // 如果 delta 检测层和 threshold 层不同, 显示辅助告警
  let auxAlert="";
  if(divIdx>=0&&badIdx>=0&&divIdx!==badIdx){
    const tL=Ls[badIdx];
    auxAlert='<div style="margin-top:6px;font-size:12px;color:var(--warn)">'
      +'⚠ 辅助告警: layer '+tL.layer_idx+' 首次 cos_sim &lt; 0.99 (累计缓慢下降, 非根因)</div>';
  }
  let html='<div class="card" style="border-left:3px solid '+C.bad+';background:rgba(194,85,85,0.05);padding:16px 18px">'
    +'<h3 style="margin:0 0 8px 0;color:'+C.bad+'">L1 诊断候选层: Layer '+L.layer_idx+'</h3>'
    +'<div class="l1-metrics">'
    +'<div><div style="font-size:11px;color:var(--muted);margin-bottom:2px">cos_sim'+hIcon("cos_sim")+'</div><div style="font-size:18px;font-weight:700;color:'+tier+';display:flex;align-items:center;gap:6px"><span class="legend-chip" style="background:'+tier+'"></span>'+fix(L.cos_sim,8)+'</div></div>'
    +'<div><div style="font-size:11px;color:var(--muted);margin-bottom:2px">rel_l2'+hIcon("rel_l2")+'</div><div style="font-size:16px;font-weight:600">'+fix(L.rel_l2,8)+'</div></div>'
    +'<div><div style="font-size:11px;color:var(--muted);margin-bottom:2px">SNR(dB)'+hIcon("snr")+'</div><div style="font-size:16px;font-weight:600">'+fix(L.snr,2)+'</div></div>'
    +'<div><div style="font-size:11px;color:var(--muted);margin-bottom:2px">name</div><div style="font-size:12px;font-family:monospace;word-break:break-all">'+esc(L.layer_name||"")+'</div></div>'
    +'</div>'
    +(tags.length?('<div style="margin-top:8px">'+tags.join("")+'</div>'):"")
    +auxAlert
    +'<div style="margin-top:10px;font-size:12px;color:var(--muted)">共 '+Ls.length+' 层检查 · 累计 '+badCnt+' 层 cos_sim &lt; 0.99。</div>'
    +'</div>';
  root.innerHTML=html;
  el("l1legend").innerHTML="";
}
function renderL1Table(){
  const root=el("l1table");const Ls=R.l1_layers||[];
  if(!Ls.length){root.innerHTML="";return;}
  let badCnt=0;
  for(let i=0;i<Ls.length;i++){const v=num(Ls[i].cos_sim);if(v!==null&&v<0.99)badCnt++;}
  const btnId="l1toggle",tblId="l1fulltbl";
  const btnTxt="展开全部 "+Ls.length+" 层详细表 ("+badCnt+" 发散) ▾";
  let h='<button type="button" id="'+btnId+'" class="toggle-btn" aria-expanded="false" aria-controls="'+tblId+'">'+btnTxt+'</button>';
  h+='<div id="'+tblId+'" class="table-scroll" style="display:none;margin-top:8px">';
  h+='<table class="grid-tbl"><thead><tr><th>Layer</th><th>name</th><th>cos_sim'+hIcon("cos_sim")+'</th><th>rel_l2'+hIcon("rel_l2")+'</th><th>SNR(dB)'+hIcon("snr")+'</th><th>标记</th></tr></thead><tbody>';
  Ls.forEach(l=>{let tags=[];if(l.is_first_divergence)tags.push('<span class="tag fd">首次发散</span>');if(l.is_max_error)tags.push('<span class="tag max">最大误差</span>');
    h+='<tr style="cursor:pointer" onclick="__selectL2('+(l.layer_idx==null?0:l.layer_idx)+')"><td>'+l.layer_idx+'</td><td>'+esc(l.layer_name)+'</td>'
      +'<td><span class="legend-chip" style="background:'+cosTier(l.cos_sim)+'"></span>'+fix(l.cos_sim,8)+'</td>'
      +'<td>'+fix(l.rel_l2,8)+'</td><td>'+fix(l.snr,2)+'</td><td>'+tags.join("")+'</td></tr>';});
  h+='</tbody></table>';
  h+='</div>';
  root.innerHTML=h;
  const btn=document.getElementById(btnId);const tbl=document.getElementById(tblId);
  if(btn&&tbl){
    btn.onclick=function(){
      const hidden=tbl.style.display==="none";
      tbl.style.display=hidden?"block":"none";
      btn.setAttribute("aria-expanded",hidden?"true":"false");
      btn.textContent=hidden?"收起 ▴":btnTxt;
    };
  }
}

// ====================================================================
// L2 diagnosis
// ====================================================================
function recColor(v){return recTier(v);}
function metricClass(v,kind){
  const n=num(v);if(n===null)return "metric-na";
  if(kind==="recovery")return n>=0.8?"metric-good":(n>=0.3?"metric-warn":"metric-bad");
  if(kind==="self")return n<0.02?"metric-good":(n<0.10?"metric-warn":"metric-bad");
  return n<0.05?"metric-good":(n<0.15?"metric-warn":"metric-bad");
}
function l2MetricsTable(subs){
  let h='<div class="table-scroll l2-metrics"><table class="grid-tbl"><thead><tr>'+
    '<th>Subgraph</th><th>Quant type</th><th>Patch Recovery'+hIcon("patch_recovery")+'</th>'+
    '<th>SelfRotErr'+hIcon("self_rot_err")+'</th><th>RotBErr'+hIcon("rot_b_err")+'</th><th>定位标记</th></tr></thead><tbody>';
  subs.forEach(sg=>{
    const roles=[];
    if(sg.is_repair_point)roles.push('<span class="tag fd">误差边界</span>');
    if(sg.is_source_candidate)roles.push('<span class="tag max">根因嫌疑</span>');
    const nameCls=(sg.is_repair_point?" repair":"")+(sg.is_source_candidate?" source":"");
    const pr=num(sg.patch_recovery),se=num(sg.self_rot_err),rb=num(sg.rot_b_err);
    h+='<tr><td class="'+nameCls.trim()+'">'+esc(sg.name||"—")+'</td><td>'+esc(sg.quant_type||"—")+'</td>'+
      '<td class="'+metricClass(pr,"recovery")+'">'+(pr===null?"—":pct(pr))+'</td>'+
      '<td class="'+metricClass(se,"self")+'">'+(se===null?"—":fix(se,4))+'</td>'+
      '<td class="'+metricClass(rb,"rot")+'">'+(rb===null?"—":fix(rb,4))+'</td>'+
      '<td>'+(roles.join("")||"—")+'</td></tr>';
  });
  return h+'</tbody></table></div>';
}
function renderL2(){
  const root=el("l2");const Ls=R.l2_results||[];
  if(!Ls.length){root.innerHTML='<div class="empty"><div class="empty-icon">—</div><div class="empty-text">未运行 L2 subgraph 反事实诊断</div><div class="empty-hint">运行 --mode l2 采集子图 patch_recovery</div></div>';return;}
  let html="";
  Ls.forEach(L=>{
    const fd=R.overview&&R.overview.first_divergence_layer!=null&&L.layer_idx===R.overview.first_divergence_layer;
    html+='<div class="l2layer'+(fd?" fd":"")+'" data-layer="'+L.layer_idx+'"><h3>Layer '+L.layer_idx
      +(fd?'<span class="tag fd">首次发散</span>':'')
      +'<span style="font-size:12px;color:'+C.muted+';margin-left:8px">基线误差 '+fix(L.base_l2,4)+hIcon("baseline_l2")
      +' · 输入恢复 '+pct(L.input_recovery)+hIcon("input_recovery")+'</span></h3>';
    const subs=L.subgraphs||[];
    if(!subs.length){html+='<div class="empty">无子图</div></div>';return;}
    // subgraph recovery bar chart
    const W=1000,H=Math.max(120,subs.length*26+30),mL=220,mR=14,mT=10,mB=24,pw=W-mL-mR,ph=H-mT-mB;
    const bw=18;const gap=(ph- subs.length*bw)/(subs.length>1?subs.length-1:1)||0;
    const X=v=>mL+pw*clamp(v,-0.1,1.1)/1.2;
    const Y=i=>mT+i*(bw+gap);
    const s=svgBox(W,H);
    // x: 0..1 (recovery), negatives shown as small
    for(let g=0;g<=5;g++){const v=g/5;const x=X(v);const ln=E("line",{x1:x,x2:x,y1:mT,y2:mT+ph,class:"gridline"});s.appendChild(ln);
      const t=E("text",{x:x,y:mT+ph+14,"text-anchor":"middle",class:"axis"});t.textContent=(v*100).toFixed(0)+"%";s.appendChild(t);}
    subs.forEach((sg,i)=>{
      const pr=num(sg.patch_recovery);
      const y=Y(i);
      const shortName=sg.name.length>28?sg.name.slice(0,26)+"…":sg.name;
      const lbl=E("text",{x:mL-8,y:y+bw/2+3,"text-anchor":"end",class:"axis"});lbl.textContent=shortName;s.appendChild(lbl);
      const qt=E("text",{x:mL-8,y:y+bw/2+13,"text-anchor":"end","font-size":"8",fill:C.muted});qt.textContent=sg.quant_type;s.appendChild(qt);
      // The chart contains recovery only.  Error metrics use a separate table
      // because their direction and scale are different.
      if(pr!==null){
        const rcol=isDegradedQuantType(sg.quant_type)?C.deg:recColor(pr);
        const xv=X(pr);
        if(pr>=0){s.appendChild(E("rect",{x:mL,y:y,width:Math.max(xv-mL,1),height:bw,fill:rcol,opacity:0.85,rx:2}));}
        else{s.appendChild(E("rect",{x:xv,y:y,width:Math.max(mL-xv,1),height:bw,fill:rcol,opacity:0.85,rx:2}));}
        const tx=E("text",{x:(pr>=0?xv+4:mL+4),y:y+bw/2+3,"text-anchor":"start",class:"axis"});tx.textContent="rec "+pct(pr);s.appendChild(tx);
      } else {
        const tx=E("text",{x:mL+4,y:y+bw/2+3,class:"axis",fill:C.na});tx.textContent="未执行 patch";s.appendChild(tx);
      }
      // border for repair/source: concentric (green outer, red inner) so both visible
      if(sg.is_repair_point){s.appendChild(E("rect",{x:mL-4,y:y-4,width:pw+8,height:bw+8,fill:"none",stroke:C.good,"stroke-width":2,rx:3}));}
      if(sg.is_source_candidate){s.appendChild(E("rect",{x:mL-1,y:y-1,width:pw+2,height:bw+2,fill:"none",stroke:C.bad,"stroke-width":2,rx:2}));}
    });
    const ax=E("line",{x1:mL,x2:mL,y1:mT,y2:mT+ph,stroke:C.border});s.appendChild(ax);
    html+='<div class="tip" style="margin-bottom:4px"><span style="color:'+C.good+'">■</span> Patch Recovery（高=好）· 橙色=W4 降级量化 · 空值=未执行/不可 patch</div>';
    html+=svgOuter(s);
    html+=l2MetricsTable(subs);
    // interpretation line
    const interp=interpretRotBErr(subs);
    if(interp){html+='<div class="tip" style="margin-top:4px;color:'+C.muted+';font-style:italic">'+interp+'</div>'}
    // chain_delta + flip_rates
    if(L.chain_delta){html+='<details style="margin-top:6px"><summary style="cursor:pointer;color:'+C.muted+'">串联子链 delta <span class="help" onclick="__modal(\'chain_delta\')" title="chain_delta">?</span></summary><pre style="background:#f8fafc;padding:8px;border-radius:6px;font-size:11px">'+esc(JSON.stringify(L.chain_delta,null,1))+'</pre></details>';}
    if(L.flip_rates){html+='<details style="margin-top:4px"><summary style="cursor:pointer;color:'+C.muted+'">flip rates '+hIcon("flip_rate")+'</summary><pre style="background:#f8fafc;padding:8px;border-radius:6px;font-size:11px">'+esc(JSON.stringify(L.flip_rates,null,1))+'</pre></details>';}
    html+='</div>';
  });
  root.innerHTML=html;
}
function interpretRotBErr(subs){
  const vals=subs.filter(s=>s.rot_b_err!==null&&s.rot_b_err!==undefined);
  if(!vals.length)return "";
  vals.sort((a,b)=>b.rot_b_err-a.rot_b_err);
  const top=vals[0];
  const tn=top.name.replace("self_attn.","sa.").replace("mlp.","m.");
  const tv=top.rot_b_err;
  let parts=[];
  if(tv>=0.15){parts.push(tn+"="+fix(tv,3)+"（极高，主要误差源）");}
  else if(tv>=0.05){parts.push(tn+"="+fix(tv,3)+"（偏高）");}
  else{parts.push("所有子图 rot_b_err < 0.05，边界误差可控");}
  // mention 2nd highest if significantly different
  if(vals.length>1){
    const s2=vals[1];
    const s2n=s2.name.replace("self_attn.","sa.").replace("mlp.","m.");
    if(s2.rot_b_err>=0.05&&s2.rot_b_err<tv*0.8){
      parts.push(s2n+"="+fix(s2.rot_b_err,3)+"（次高）");
    } else if(s2.rot_b_err>=0.05){
      parts.push(s2n+"="+fix(s2.rot_b_err,3)+"（同量级）");
    }
  }
  return "解读: "+parts.join("；")+"。";
}
function svgOuter(s){const w=document.createElement("div");w.className="chart-scroll";w.appendChild(s.cloneNode(true));return w.outerHTML;}
function appendChart(root,s){const w=document.createElement("div");w.className="chart-scroll";w.appendChild(s);root.appendChild(w);}

// ====================================================================
// LOGITS 可视化
// ====================================================================
let curPos=-1;
function renderLogits(){
  const root=el("logits");const L=R.logits;
  if(!L||!L.token_positions||!L.token_positions.length){
    const reason=(R.overview&&R.overview.logits_error)||"未采集 logits 对比";
    root.innerHTML='<div class="empty"><div class="empty-icon">—</div><div class="empty-text">未采集 logits 对比</div><div class="empty-hint">'+esc(reason)+'</div></div>';return;
  }
  if(curPos<0||curPos>=L.token_positions.length){
    const cr=(R.overview||{}).captured_replay||{};
    const diagnosticPos=cr.first_mismatch_position==null?cr.first_top1_flip_position:cr.first_mismatch_position;
    const diagnosticIndex=diagnosticPos==null?-1:L.token_positions.indexOf(Number(diagnosticPos));
    curPos=diagnosticIndex>=0?diagnosticIndex:(L.position_mode==="prompt_prefill"?L.token_positions.length-1:0);
  }
  let modeNote="采集位置语义未记录；请结合生成方式人工确认 position 含义。";
  if(L.position_mode==="prompt_prefill")modeNote="这里展示 Prompt prefill 各位置的 next-token 预测；只有最后一行对应首个 Decode Token，页面已默认选中最后一行。前面的 position 不是生成序列。";
  else if(L.position_mode==="generation")modeNote="这里展示自回归生成的 Decode step；Position 0 对应首个 Decode Token。";
  if(L.display_sampled){
    modeNote += " 后端已比较全部 "+(L.total_positions||L.token_positions.length)+
      " 个 position；为避免浏览器卡死，页面仅展示其中 "+L.token_positions.length+
      " 个诊断位置（优先异常并均匀覆盖全序列），完整定界结果保留在 boundary_result.json。";
  } else if(L.token_positions.length>1 && Number(L.token_positions[0])>0){
    modeNote += " 当前报告为长序列安全采样，仅采集最后 "+L.token_positions.length+
      " 个 position（"+L.token_positions[0]+"–"+
      L.token_positions[L.token_positions.length-1]+"）；可用 --logits_max_positions 0 采集全部。";
  }
  root.innerHTML='<div class="context-note"><b>Position 口径：</b>'+modeNote+' 散点与直方图为已采集位置的聚合视图。</div>'+
    '<div class="grid" style="grid-template-columns:1fr"><div id="logits_scatter"></div>'
    +'<div id="logits_topk"></div><div id="logits_lines"></div><div id="logits_hist"></div></div>';
  renderScatter();renderLines();renderHist();renderTopK(curPos);
}
function renderScatter(){
  const root=el("logits_scatter");if(!root)return;
  const rf=R.logits.scatter_ref||[],qf=R.logits.scatter_quant||[];
  if(!rf.length){root.innerHTML='<div class="empty">无散点样本</div>';return;}
  const leftName=R.logits.position_mode==="captured_replay"?"vLLM capture":"Ref";
  const rightName=R.logits.position_mode==="captured_replay"?"Transformers replay":"Quant";
  const W=900,H=300,m=40;const xs=rf,qx=qf;
  let lo=Math.min(...rf,...qf),hi=Math.max(...rf,...qf);if(hi-lo<1)hi=lo+1;
  const X=v=>m+(W-2*m)*(v-lo)/(hi-lo),Y=v=>(H-m)-(H-2*m)*(v-lo)/(hi-lo);
  const s=svgBox(W,H);
  for(let g=0;g<=4;g++){const v=lo+(hi-lo)*g/4;const y=Y(v);s.appendChild(E("line",{x1:m,x2:W-m,y1:y,y2:y,class:"gridline"}));
    const t=E("text",{x:m-6,y:y+3,"text-anchor":"end",class:"axis"});t.textContent=v.toFixed(1);s.appendChild(t);
    const x=v;const xx=X(x);const tx=E("text",{x:xx,y:H-m+14,"text-anchor":"middle",class:"axis"});tx.textContent=x.toFixed(1);s.appendChild(tx);}
  s.appendChild(E("line",{x1:X(0),y1:Y(hi),x2:X(0),y2:Y(0),stroke:C.border,opacity:0.0})); // axis lines below
  s.appendChild(E("line",{x1:m,y1:H-m,x2:W-m,y2:H-m,stroke:C.border}));s.appendChild(E("line",{x1:m,y1:m,x2:m,y2:H-m,stroke:C.border}));
  // y=x 参考线
  s.appendChild(E("line",{x1:X(lo),y1:Y(lo),x2:X(hi),y2:Y(hi),stroke:C.muted,"stroke-dasharray":"4 3"}));
  for(let i=0;i<rf.length;i++){s.appendChild(E("circle",{cx:X(rf[i]),cy:Y(qf[i]),r:1.6,fill:C.quant,opacity:0.35}));}
  const t1=E("text",{x:W/2,y:H-4,"text-anchor":"middle",class:"axis"});t1.textContent=leftName+" logit";s.appendChild(t1);
  const t2=E("text",{x:12,y:H/2,"text-anchor":"middle",class:"axis",transform:"rotate(-90 12 "+(H/2)+")"});t2.textContent=rightName+" logit";s.appendChild(t2);
  root.innerHTML='<div class="card"><h3>'+leftName+' vs '+rightName+' logits 散点 '+hIcon("logits_scatter")+'</h3></div>';
  appendChart(root,s);
  const lg=document.createElement("div");lg.className="tip";lg.innerHTML='<span class="legend-chip" style="background:'+C.quant+'"></span>每个点=某 position 的 '+leftName+'/'+rightName+' Top-K 并集候选 ('+leftName+'_x, '+rightName+'_y) · 虚线=y=x 完全吻合 · 离线越远=该 token 偏离越大';
  root.appendChild(lg);
}
function renderLines(){
  const root=el("logits_lines");if(!root)return;
  const L=R.logits;const ps=L.token_positions||[];
  const top1Arr=L.token_wise_top1_match||[];
  const ser=[["cos",L.token_wise_cos,C.ref],["topk_overlap",L.token_wise_topk_overlap,C.topk]];
  const W=900,H=240,m=36;const yl=[0,1];
  const X=i=>m+(W-2*m)*(ps.length<=1?0.5:i/(ps.length-1||1));
  const Y=v=>(H-m)-(H-2*m)*(v-yl[0])/(yl[1]-yl[0]);
  const s=svgBox(W,H);
  for(let g=0;g<=4;g++){const v=g/4;const y=Y(v);s.appendChild(E("line",{x1:m,x2:W-m,y1:y,y2:y,class:"gridline"}));
    const t=E("text",{x:m-6,y:y+3,"text-anchor":"end",class:"axis"});t.textContent=v.toFixed(2);s.appendChild(t);}
  ser.forEach(([name,arr,col])=>{
    if(!arr||!arr.length)return;
    let d="";arr.forEach((v,i)=>{const n=num(v);if(n===null)return;d+=(d?" L":"M")+X(i)+" "+Y(n);});
    s.appendChild(E("path",{d:d,stroke:col,"stroke-width":1.8,fill:"none"}));
  });
  // clickable position markers
  ps.forEach((p,i)=>{const overlap=num((L.token_wise_topk_overlap||[])[i]);
    const top1Match=top1Arr[i];
    const markerColor=top1Match===false?C.bad:(overlap!==null&&overlap<1?C.warn:(overlap!==null?C.good:C.muted));
    const mk=E("circle",{cx:X(i),cy:Y(0.5),r:curPos===i?6:5,fill:markerColor,opacity:0.72,cursor:"pointer",
      stroke:curPos===i?C.ink:"none","stroke-width":curPos===i?1.5:0,
      role:"button",tabindex:"0","aria-label":"选择 Position "+p});
    const choose=()=>window.__selPos(i);
    mk.addEventListener("click",choose);
    mk.addEventListener("keydown",e=>{if(e.key==="Enter"||e.key===" "){e.preventDefault();choose();}});
    s.appendChild(mk);
    const tickEvery=Math.max(1,Math.ceil(ps.length/10));
    if(i===0||i===ps.length-1||i%tickEvery===0){
      const tx=E("text",{x:X(i),y:H-m+14,"text-anchor":"middle",class:"axis","font-size":"8"});tx.textContent=p;s.appendChild(tx);
    }});
  s.appendChild(E("line",{x1:m,y1:H-m,x2:W-m,y2:H-m,stroke:C.border}));s.appendChild(E("line",{x1:m,y1:m,x2:m,y2:H-m,stroke:C.border}));
  const tt=E("text",{x:W/2,y:H-2,"text-anchor":"middle",class:"axis"});tt.textContent=L.position_mode==="prompt_prefill"?"Prompt token position":"Decode step";s.appendChild(tt);
  root.innerHTML="";
  const header=document.createElement("div");header.className="card";
  header.innerHTML='<h3>Token-wise 指标折线 '+hIcon("token_wise_cos")+hIcon("topk_overlap")+hIcon("token_wise_kl")+'</h3>';
  const overlapArr=L.token_wise_topk_overlap||[];
  const top1MismatchCount=ps.reduce((n,p,i)=>n+(top1Arr[i]===false?1:0),0);
  const topkPartialCount=ps.reduce((n,p,i)=>n+(top1Arr[i]===true&&num(overlapArr[i])!==null&&num(overlapArr[i])<1?1:0),0);
  const topkExactCount=ps.reduce((n,p,i)=>n+(num(overlapArr[i])!==null&&num(overlapArr[i])>=1?1:0),0);
  const topkTotal=overlapArr.length||ps.length;
  const topkNote=document.createElement("div");topkNote.className="tip";
  topkNote.innerHTML='<span class="legend-chip" style="background:#A53939"></span>Top-1 不一致 '+top1MismatchCount+'/'+topkTotal
    +' · <span class="legend-chip" style="background:#B87710"></span>Top-1 一致、Top-K 部分重合 '+topkPartialCount
    +' · <span style="color:#176B55">Top-K 完全一致 '+topkExactCount+'</span>'
    +' · Position 选择器已放在上方的 Top-K 概率卡片中';
  header.appendChild(topkNote);root.appendChild(header);
  appendChart(root,s);
  const lg=document.createElement("div");lg.className="tip";lg.innerHTML=
    '<span class="legend-chip" style="background:'+C.ref+'"></span>cos(ref,quant logits) '
    +'<span class="legend-chip" style="background:'+C.topk+'"></span>top-k overlap '
    +'KL(平均)='+avgKL();
  root.appendChild(lg);
  // KL mini chart
  const Lk=L.token_wise_kl||[];
  if(Lk.length){
    const kw=W,kh=120,m2=30;const x2=i=>m2+(kw-2*m2)*(ps.length<=1?0.5:i/(ps.length-1||1));let maxk=Math.max(...Lk.map(x=>num(x)||0),0.01);
    const Y2=v=>kh-m2-(kh-2*m2)*(v/maxk);const sk=svgBox(kw,kh);
    Lk.forEach((v,i)=>{const n=num(v);if(n===null)return;const x=x2(i);const y=Y2(n);
      const bar=E("rect",{x:x-2,y:y,width:4,height:kh-m2-y,fill:C.quant,opacity:0.7,cursor:"pointer",
        role:"button",tabindex:"0","aria-label":"选择 Position "+ps[i]});
      const choose=()=>window.__selPos(i);
      bar.addEventListener("click",choose);
      bar.addEventListener("keydown",e=>{if(e.key==="Enter"||e.key===" "){e.preventDefault();choose();}});
      sk.appendChild(bar);});
    for(let g=0;g<=2;g++){const v=maxk*g/2;const y=Y2(v);const t=E("text",{x:m2-6,y:y+3,"text-anchor":"end",class:"axis"});t.textContent=v.toFixed(3);sk.appendChild(t);sk.appendChild(E("line",{x1:m2,x2:kw-m2,y1:y,y2:y,class:"gridline"}));}
    sk.appendChild(E("line",{x1:m2,y1:kh-m2,x2:kw-m2,y2:kh-m2,stroke:C.border}));
    const tk=E("text",{x:kw/2,y:kh-2,"text-anchor":"middle",class:"axis"});tk.textContent="KL(ref‖quant) per position";sk.appendChild(tk);
    appendChart(root,sk);
  }
}
function avgKL(){const k=R.logits.token_wise_kl||[];const f=k.map(x=>num(x)).filter(x=>x!==null);if(!f.length)return"—";return (f.reduce((a,b)=>a+b,0)/f.length).toFixed(4);}
function renderTopK(i){
  const root=el("logits_topk");if(!root)return;const L=R.logits;if(!L.ref_topk||!L.ref_topk[i]){root.innerHTML="";return;}
  curPos=i;
  const pos=L.token_positions[i]===undefined?i:L.token_positions[i];
  const isFirstDecode=L.position_mode==="prompt_prefill"&&i===L.token_positions.length-1;
  const role=L.position_mode==="prompt_prefill"?(isFirstDecode?"首个 Decode Token":"Prompt 内 next-token 预测"):(L.position_mode==="generation"?"Decode step":"Token position");
  const inputRows=L.input_ids||[];const inputRow=inputRows[0]||[];
  const inputToken=(pos>=0&&pos<inputRow.length)?("#"+inputRow[pos]):"—";
  const leftName=L.position_mode==="captured_replay"?"vLLM capture":"Ref";
  const rightName=L.position_mode==="captured_replay"?"Transformers replay":"Quant";
  const rf=L.ref_topk[i]||[];const qf=L.quant_topk[i]||[];
  // merge by token_id: 同一 token 取 ref_prob 与 quant_prob 并排
  const map={};const order=[];
  rf.forEach(t=>{if(!map[t.token_id]){map[t.token_id]={token_id:t.token_id,token_str:t.token_str,ref_prob:t.ref_prob,quant_prob:null};order.push(map[t.token_id]);}});
  qf.forEach(t=>{const ex=map[t.token_id];if(ex){ex.quant_prob=t.quant_prob;}else{map[t.token_id]={token_id:t.token_id,token_str:t.token_str,ref_prob:null,quant_prob:t.quant_prob};order.push(map[t.token_id]);}});
  const rows=order;
  const W=900,h=22,mL=130,mR=14,mT=10,mB=20;const H=Math.max(rows.length*h+mT+mB,80);
  const maxp=Math.max(...rows.map(t=>Math.max(num(t.ref_prob)||0,num(t.quant_prob)||0)),0.001);
  const X=v=>mL+(W-mL-mR)*v/maxp;
  const s=svgBox(W,H);
  rows.forEach((t,r)=>{const y=mT+r*h;
    const rp=num(t.ref_prob),qp=num(t.quant_prob);
    const lblClean=(t.token_str||'').replace(/[\u200b-\u200f\u202a-\u202e\u2060-\u206f\ufeff\x00-\x1f\x7f]/g,'').trim();
    const lblText=lblClean?lblClean:("#"+t.token_id);
    const lbl=E("text",{x:mL-8,y:y+11,"text-anchor":"end",class:"axis","font-size":"11"});lbl.textContent=lblText.slice(0,14);s.appendChild(lbl);
    // ref 粗 (上) 蓝
    if(rp!==null){const w=Math.max(X(rp)-mL,2);const rr=E("rect",{x:mL,y:y,width:w,height:h/2-1,fill:C.ref,opacity:0.85,rx:2});s.appendChild(rr);}
    // quant 梓 (下) 橙
    if(qp!==null){const w=Math.max(X(qp)-mL,2);const rr=E("rect",{x:mL,y:y+h/2,width:w,height:h/2-1,fill:C.quant,opacity:0.85,rx:2});s.appendChild(rr);}
  });
  // legend
  const lg=E("text",{x:W-mR,y:H-4,"text-anchor":"end",class:"axis","font-size":"9"});lg.textContent="绿(粗)="+leftName+" 概率 · 橙(细)="+rightName+" 概率";s.appendChild(lg);
  const top1Arr=R.logits.token_wise_top1_match||[];
  const recordedTotal=Number(L.full_top1_total||0);
  const matchCnt=recordedTotal>0?Number(L.full_top1_match_count||0):top1Arr.filter(x=>x===true).length;
  const tot=recordedTotal>0?recordedTotal:top1Arr.length;
  const top1=R.logits.token_wise_top1_match&&R.logits.token_wise_top1_match[i];
  const refMargin=num((R.logits.ref_top1_margin||[])[i]);
  const quantMargin=num((R.logits.quant_top1_margin||[])[i]);
  const matchBadge=top1?'<span class="pill ok">Top-1 一致</span>':'<span class="pill bad">Top-1 不一致</span>';
  const summaryBadge=matchCnt+'/'+tot+' positions match';
  const summaryCls=matchCnt===tot?"pill ok":(matchCnt>tot/2?"pill warn":"pill bad");
  const card=document.createElement("div");
  card.className="card";
  card.innerHTML='<h3>Top-K 概率并排 '+hIcon("top1_match")+' <span class="'+summaryCls+'">'+summaryBadge+'</span></h3>'
    +'<div class="tip"><b>Position '+pos+' · '+role+'</b> · 输入 token '+inputToken+' — 当前: '+matchBadge+' · 绿(粗)='+leftName+' 概率 · 橙(细)='+rightName+' 概率'
    +' · Top-1 margin ('+leftName+'/'+rightName+')='+(refMargin===null?"—":refMargin.toFixed(4))+'/'+(quantMargin===null?"—":quantMargin.toFixed(4))
    +' · 使用下方 Position 选择器或折线节点切换</div>';
  const picker=document.createElement("div");picker.className="position-picker";picker.setAttribute("aria-label","Top-K position selector");
  const overlapArr=L.token_wise_topk_overlap||[];
  L.token_positions.forEach((p,j)=>{const b=document.createElement("button");b.type="button";
    const overlap=num(overlapArr[j]);
    const top1Match=top1Arr[j];
    const stateClass=overlap===null?"":(top1Match===false?" topk-mismatch":(overlap<1?" topk-partial":" topk-match"));
    b.className="position-btn"+stateClass+(curPos===j?" active":"");
    b.textContent=String(p)+(L.position_mode==="prompt_prefill"&&j===L.token_positions.length-1?" · decode":"");
    const stateText=overlap===null?"Top-K overlap 未记录":(top1Match===false?"Top-1 不一致":(overlap<1?"Top-1 一致，Top-K 部分重合":"Top-K 完全一致"));
    b.setAttribute("aria-label","Position "+p+"，"+stateText);
    b.title=stateText+(overlap!==null?"（overlap="+overlap.toFixed(3)+"）":"");
    b.addEventListener("click",()=>window.__selPos(j));picker.appendChild(b);});
  card.appendChild(picker);
  const pickerNote=document.createElement("div");pickerNote.className="tip";
  pickerNote.innerHTML='<span class="legend-chip" style="background:#A53939"></span>红色=Top-1 不一致 · '
    +'<span class="legend-chip" style="background:#B87710"></span>黄色=Top-1 一致但 Top-K 未完全重合 · '
    +'<span style="color:#176B55">绿色=Top-K 完全一致</span>'
    +(L.display_sampled?' · 当前选择器展示 '+L.token_positions.length+'/'+L.total_positions+' 个诊断 position':'');
  card.appendChild(pickerNote);
  root.innerHTML="";
  root.appendChild(card);
  appendChart(card,s);
}
function bw_h(h){return Math.max(8,h-4);}
function renderHist(){
  const root=el("logits_hist");if(!root)return;
  const L=R.logits;const bins=L.hist_bins||[];if(!bins.length){root.innerHTML="<div class='empty'>无直方图</div>";return;}
  const rc=L.hist_ref_counts||[],qc=L.hist_quant_counts||[];
  const W=900,H=220,m=36;const maxc=Math.max(...rc,...qc,1);
  const n=rc.length;const bw=(W-2*m)/n;
  const X=i=>m+i*bw;const Yc=v=>H-m-(H-2*m)*v/maxc;
  const s=svgBox(W,H);
  for(let g=0;g<=4;g++){const v=maxc*g/4;const y=Yc(v);s.appendChild(E("line",{x1:m,x2:W-m,y1:y,y2:y,class:"gridline"}));
    const t=E("text",{x:m-6,y:y+3,"text-anchor":"end",class:"axis"});t.textContent=Math.round(v);s.appendChild(t);}
  for(let i=0;i<n;i++){const x=X(i);const yrc=Yc(rc[i]);const yqc=Yc(qc[i]);
    s.appendChild(E("rect",{x:x+2,y:yrc,width:bw/2-2,height:H-m-yrc,fill:C.ref,opacity:0.6}));
    s.appendChild(E("rect",{x:x+bw/2,y:yqc,width:bw/2-2,height:H-m-yqc,fill:C.quant,opacity:0.6}));
  }
  s.appendChild(E("line",{x1:m,y1:H-m,x2:W-m,y2:H-m,stroke:C.border}));s.appendChild(E("line",{x1:m,y1:m,x2:m,y2:H-m,stroke:C.border}));
  for(let i=0;i<n;i+=Math.max(1,Math.floor(n/8))){const x=X(i);const t=E("text",{x:x+bw/2,y:H-m+14,"text-anchor":"middle",class:"axis","font-size":"9"});t.textContent=bins[i]!=null?(+bins[i]).toFixed(1):"";s.appendChild(t);}
  const t1=E("text",{x:W/2,y:H-2,"text-anchor":"middle",class:"axis"});t1.textContent="logit 值";s.appendChild(t1);
  root.innerHTML='<div class="card"><h3>Logits 分布直方图 overlay '+hIcon("logits_hist_overlay")+'</h3></div>';
  appendChart(root,s);
  const lg=document.createElement("div");lg.className="tip";lg.innerHTML='<span class="legend-chip" style="background:'+C.ref+'"></span>ref '+
    '<span class="legend-chip" style="background:'+C.quant+'"></span>quant · 分布形状一致=量化保留整体分布';
  root.appendChild(lg);
}

// ====================================================================
// End-to-end generation / bad-case comparison
// ====================================================================
function renderBadcase(){
  const root=el("badcase");if(!root)return;
  const section=el("badcase-section");
  const navLink=document.querySelector('a[data-target="badcase-section"]');
  const I=R.inference_compare;
  if(!I){
    if(section)section.style.display="none";
    if(navLink)navLink.style.display="none";
    root.innerHTML='<div class="empty"><div class="empty-icon">—</div><div class="empty-text">未运行生成输出对比</div>'+
      '<div class="empty-hint">full / boundary 流程可记录 Ref 与 Quant 的实际生成结果</div></div>';
    return;
  }
  if(section)section.style.display="";
  if(navLink)navLink.style.display="";
  const rate=num(I.token_match_rate);
  const exact=I.exact_match===true;
  const first=I.first_divergence_pos;
  const flags=[];
  if(I.ref_garbled)flags.push('<span class="pill bad">REF 乱码</span>');
  if(I.quant_garbled)flags.push('<span class="pill bad">QUANT 乱码</span>');
  if(I.ref_repeat)flags.push('<span class="pill warn">REF 复读</span>');
  if(I.quant_repeat)flags.push('<span class="pill warn">QUANT 复读</span>');
  if(I.logits_nan_inf)flags.push('<span class="pill bad">LOGITS NaN/Inf</span>');
  if(!flags.length)flags.push('<span class="pill ok">未发现退化信号</span>');
  const boundarySingle = (!I.ref_output || !I.quant_output);
  let h=boundarySingle ?
    '<div class="card"><h3>Boundary 原生 Transformers 输出</h3><div class="pill ok">已生成</div>'+
    '<div class="sub">当前为 Quant-only 或 Ref-only 定界，暂无另一侧输出，未计算 Token 匹配率。</div></div>' :
    '<div class="grid cols-3">'+
      '<div class="card"><h3>生成文本完全一致</h3><div class="pill '+(exact?'ok':'warn')+'">'+(exact?'YES':'NO')+'</div></div>'+
      '<div class="card"><h3>逐 Token 匹配率</h3><div class="val">'+pct(rate,1)+'</div></div>'+
      '<div class="card"><h3>首个分歧位置</h3><div class="val">'+(first==null?'—':esc(first))+'</div><div class="sub">max_new_tokens '+esc(I.max_new_tokens||0)+'</div></div>'+
      '</div>';
  h+='<div class="health-row">'+flags.join('')+'</div>';
  h+='<div class="card" style="margin-top:12px"><h3>输入</h3><pre class="output-box">'+esc(I.prompt||R.overview?.prompt||'')+'</pre></div>';
  if(boundarySingle){
    h+='<div class="card" style="margin-top:12px"><h3>'+(I.quant_output?'QUANT OUTPUT':'REF OUTPUT')+'</h3>'+
      '<pre class="output-box">'+esc(I.quant_output||I.ref_output||'')+'</pre></div>';
  } else {
    h+='<div class="grid cols-2" style="margin-top:12px">'+
      '<div class="card"><h3>REF OUTPUT</h3><pre class="output-box">'+esc(I.ref_output||'')+'</pre></div>'+
      '<div class="card"><h3>QUANT OUTPUT</h3><pre class="output-box">'+esc(I.quant_output||'')+'</pre></div></div>';
  }

  const rt=I.ref_tokens||[],qt=I.quant_tokens||[];
  if(rt.length||qt.length){
    const total=Math.max(rt.length,qt.length),limit=Math.min(total,160);
    h+='<div class="table-scroll" style="margin-top:12px"><table class="grid-tbl"><thead><tr><th>Pos</th><th>Ref token</th><th>Quant token</th><th>状态</th></tr></thead><tbody>';
    for(let i=0;i<limit;i++){
      const a=rt[i],b=qt[i],same=(a!==undefined&&b!==undefined&&a===b);
      h+='<tr><td>'+i+'</td><td class="token-cell">'+esc(a===undefined?'—':a)+'</td><td class="token-cell">'+esc(b===undefined?'—':b)+'</td>'+
        '<td class="'+(same?'token-match':'token-mismatch')+'">'+(same?'MATCH':'DIFF')+'</td></tr>';
    }
    h+='</tbody></table></div>';
    if(total>limit)h+='<div class="tip">Token 明细仅展示前 '+limit+' / '+total+' 个位置。</div>';
  }
  root.innerHTML=h;
}

// ---- nav helpers ----
function goto(id){const e=el(id);if(e){e.scrollIntoView({behavior:"smooth",block:"start"});}}
window.__selectL2=function(idx){const root=el("l2");if(!root)return;const c=root.querySelector('.l2layer[data-layer="'+idx+'"]');
  if(!c){goto("l2-section");return;}c.scrollIntoView({behavior:"smooth",block:"start"});c.classList.add("jump-target");
  window.setTimeout(()=>c.classList.remove("jump-target"),1400);}
window.__selPos=function(i){renderTopK(i);renderLines();}
// scroll-spy: highlight nav based on visible section
function initScrollSpy(){
  const links=document.querySelectorAll('.nav a');
  const sections=[];
  links.forEach(a=>{const id=a.getAttribute('data-target');const s=el(id);if(s)sections.push({id,el:s,link:a});});
  if(!sections.length)return;
  function update(){const scrollY=window.scrollY+120;let active=sections[0];
    sections.forEach(s=>{if(s.el.offsetTop<=scrollY)active=s;});
    links.forEach(a=>a.classList.remove('active'));
    active.link.classList.add('active');}
  window.addEventListener('scroll',update,{passive:true});
  update();
}

// ---- boot ----
// __accInitDone guard: scroll-spy + modal listener register ONLY once per page
// (multi-report index.html re-runs boot() on each sidebar switch)
function boot(){
  renderOverview();renderL1();renderL1Table();renderL2();renderLogits();renderBadcase();
  if(!window.__accInitDone){
    window.__accInitDone=true;
    initScrollSpy();
    el("helpModal").addEventListener("click",function(e){if(e.target===this)closeModal();});
    document.addEventListener("keydown",function(e){if(e.key==="Escape")closeModal();});
  }
}
if(document.readyState!=="loading")boot();else document.addEventListener("DOMContentLoaded",boot);
})();
"""


def _embed_json_safe(d: Dict[str, Any]) -> str:
    """把 dict 嵌入 <script> 安全: 转义 < 防止 </script> 注入。"""
    return json.dumps(d, ensure_ascii=False).replace("<", "\\u003c")


_HTML_LOGITS_POSITION_LIMIT = 256
_LOGITS_POSITION_FIELDS = (
    "token_positions", "ref_topk", "quant_topk", "ref_logits", "quant_logits",
    "token_wise_cos", "token_wise_kl", "token_wise_topk_overlap",
    "token_wise_top1_match", "ref_top1_margin", "quant_top1_margin",
)


def _evenly_sample_indices(indices: List[int], count: int) -> List[int]:
    """Return deterministic, order-preserving samples including both ends."""
    if count <= 0 or not indices:
        return []
    if len(indices) <= count:
        return list(indices)
    if count == 1:
        return [indices[0]]
    return list(dict.fromkeys(
        indices[round(i * (len(indices) - 1) / (count - 1))]
        for i in range(count)
    ))


def _html_logits_indices(logits: Dict[str, Any], limit: int) -> List[int]:
    """Prefer anomaly positions, then fill with uniform sequence coverage."""
    positions = list(logits.get("token_positions") or [])
    n = len(positions)
    if n <= limit:
        return list(range(n))
    top1 = list(logits.get("token_wise_top1_match") or [])
    overlaps = list(logits.get("token_wise_topk_overlap") or [])
    cosines = list(logits.get("token_wise_cos") or [])
    kls = list(logits.get("token_wise_kl") or [])
    anomalies = []
    for index in range(n):
        overlap = overlaps[index] if index < len(overlaps) else None
        cosine = cosines[index] if index < len(cosines) else None
        kl = kls[index] if index < len(kls) else None
        if (
            (index < len(top1) and top1[index] is False)
            or (overlap is not None and overlap < 1.0)
            or (cosine is not None and cosine < 0.99)
            or (kl is not None and kl > 0.05)
        ):
            anomalies.append(index)

    selected = {0, n - 1}
    anomaly_budget = min(len(anomalies), max(16, limit * 2 // 3))
    selected.update(_evenly_sample_indices(anomalies, anomaly_budget))
    remaining = limit - len(selected)
    if remaining > 0:
        selected.update(_evenly_sample_indices(list(range(n)), remaining))
    if len(selected) < limit:
        unselected = [index for index in range(n) if index not in selected]
        selected.update(_evenly_sample_indices(unselected, limit - len(selected)))
    return sorted(selected)[:limit]


def _compact_report_for_html(
    raw: Dict[str, Any], limit: int = _HTML_LOGITS_POSITION_LIMIT,
) -> Dict[str, Any]:
    """Bound browser payload/DOM cost while preserving full-run aggregates.

    Boundary's raw JSON remains untouched. Product-report payloads may use
    this compact representation, so all positions are still compared and
    archived in ``boundary_result.json`` while the browser renders a
    diagnostic subset.
    """
    if not isinstance(raw, dict):
        return raw
    compact = dict(raw)
    logits = raw.get("logits")
    if isinstance(logits, dict):
        positions = list(logits.get("token_positions") or [])
        n = len(positions)
        if limit > 0 and n > limit:
            indices = _html_logits_indices(logits, limit)
            bounded = dict(logits)
            for key in _LOGITS_POSITION_FIELDS:
                values = logits.get(key)
                if isinstance(values, list) and len(values) == n:
                    bounded[key] = [values[index] for index in indices]
            top1 = list(logits.get("token_wise_top1_match") or [])
            bounded["total_positions"] = n
            bounded["display_sampled"] = True
            bounded["full_top1_total"] = len(top1)
            bounded["full_top1_match_count"] = sum(value is True for value in top1)
            compact["logits"] = bounded

    overview = raw.get("overview")
    if isinstance(overview, dict) and isinstance(overview.get("captured_replay"), dict):
        overview_copy = dict(overview)
        captured = dict(overview["captured_replay"])
        compared = list(captured.get("compared_positions") or [])
        captured["compared_position_count"] = int(
            captured.get("compared_position_count") or len(compared)
        )
        if compared:
            captured["compared_position_range"] = [compared[0], compared[-1]]
            captured["compared_positions"] = _evenly_sample_indices(compared, 64)
        for key in ("top1_flip_positions", "low_margin_flip_positions", "nonfinite_positions"):
            values = list(captured.get(key) or [])
            captured[key.replace("_positions", "_count")] = len(values)
            if len(values) > 128:
                captured[key] = _evenly_sample_indices(values, 128)
        overview_copy["captured_replay"] = captured
        compact["overview"] = overview_copy
    return compact


def compact_report_for_html(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Public wrapper used by report writers before persisting UI payloads."""
    return _compact_report_for_html(raw)


# 侧边栏 CSS (复用于 product_report.html 的左侧历史栏)
_SIDEBAR_CSS = """
.layout { display:flex; min-height:100vh; }
.sidebar { width:280px; flex-shrink:0; color:#D7E4E9; background:linear-gradient(180deg,var(--navy),#0B202B);
           border-right:1px solid rgba(255,255,255,.08); position:fixed; top:0; left:0; bottom:0;
           overflow-y:auto; z-index:100; padding:18px 0 28px; box-shadow:10px 0 30px rgba(17,43,58,.08); }
.sidebar-brand { display:flex; align-items:center; gap:11px; margin:0 16px 24px; padding-bottom:17px;
                 border-bottom:1px solid rgba(255,255,255,.1); color:#fff; font-weight:800; letter-spacing:-.02em; }
.sidebar-brand small { display:block; margin-top:1px; color:#91A8B2; font-family:var(--mono); font-size:9px;
                       font-weight:600; letter-spacing:.09em; text-transform:uppercase; }
.sidebar h2 { margin:0 0 9px; padding:0 17px; color:#91A8B2; font-family:var(--mono); font-size:10px;
              font-weight:800; letter-spacing:.11em; text-transform:uppercase; }
.sidebar-item { margin:3px 9px; padding:11px 13px; cursor:pointer; border:1px solid transparent;
                border-radius:10px; transition:all .15s ease; }
.sidebar-item:hover { background:rgba(255,255,255,.06); border-color:rgba(255,255,255,.08); }
.sidebar-item.active { background:rgba(150,230,195,.11); border-color:rgba(150,230,195,.27);
                       box-shadow:inset 3px 0 0 var(--mint); }
.sidebar-item .si-ts { color:#91A8B2; font-family:var(--mono); font-size:10px; font-weight:500; }
.sidebar-item .si-model { margin:3px 0 5px; color:#F4F8F9; font-size:13px; font-weight:750; overflow-wrap:anywhere; }
.sidebar-item .si-meta { display:flex; gap:7px; flex-wrap:wrap; color:#9FB2BA; font-size:10px; }
.sidebar-item .si-badge { padding:2px 6px; border-radius:999px; font-size:9px; font-weight:800; }
.sidebar-item .si-badge.good { background:rgba(150,230,195,.15); color:var(--mint); }
.sidebar-item .si-badge.bad { background:rgba(255,139,139,.13); color:#FFB3B3; }
.sidebar-item .si-badge.warn { background:rgba(255,203,116,.14); color:var(--amber); }
.sidebar-item .si-badge.na { background:rgba(255,255,255,.08); color:#B5C4CA; }
.sidebar-item .si-badge.mode { letter-spacing:.05em; }
.sidebar-item .si-badge.mode-l1 { background:rgba(105,185,255,.16); color:#A9D8FF; }
.sidebar-item .si-badge.mode-l2 { background:rgba(180,144,255,.17); color:#D5BEFF; }
.sidebar-item .si-badge.mode-boundary { background:rgba(255,203,116,.16); color:#FFD998; }
.sidebar-item .si-badge.mode-full { background:rgba(150,230,195,.17); color:var(--mint); }
.sidebar-item .si-badge.mode-report { background:rgba(255,255,255,.08); color:#C6D2D7; }
.sidebar-delete-hint { float:right; color:#6F8A96; font-size:9px; font-weight:600; letter-spacing:0; text-transform:none; }
.history-context-menu { display:none; position:fixed; z-index:1000; width:240px; padding:8px;
                        color:#D7E4E9; background:#102B38; border:1px solid rgba(255,255,255,.14);
                        border-radius:10px; box-shadow:0 14px 40px rgba(0,0,0,.32); }
.history-context-menu.open { display:block; }
.history-context-menu .hm-title { padding:4px 6px 2px; color:#F4F8F9; font-size:12px; font-weight:750;
                                  overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.history-context-menu .hm-path { padding:0 6px 7px; color:#91A8B2; font-family:var(--mono); font-size:9px;
                                 overflow-wrap:anywhere; }
.history-context-menu button { width:100%; padding:8px 9px; color:#FFB3B3; background:rgba(255,139,139,.08);
                               border:1px solid rgba(255,139,139,.22); border-radius:7px; cursor:pointer;
                               font:700 11px var(--sans); text-align:left; }
.history-context-menu button:hover { background:rgba(255,139,139,.15); }
.history-context-menu button:disabled { opacity:.55; cursor:wait; }
.history-context-menu .hm-tip { padding:7px 6px 2px; color:#78919B; font-size:9px; line-height:1.4; }
.main-area { flex:1; min-width:0; margin-left:280px; }
.main-area .wrap { max-width:1120px; margin:0 auto; padding:42px 34px 140px; }
@media (max-width:720px){
  .layout { display:block; }
  .sidebar { width:100%; height:auto; position:relative; top:auto; bottom:auto; border-right:none;
             border-bottom:1px solid rgba(255,255,255,.08); padding:9px 0 11px; max-height:none; overflow:visible; }
  .sidebar-brand { display:none; }
  .sidebar h2 { padding:0 12px; margin:0 0 5px; }
  #sidebar-list { display:flex; gap:4px; padding:0 5px 2px; overflow-x:auto; scrollbar-width:thin; }
  .sidebar-item { flex:0 0 min(78vw,280px); padding:8px 11px; margin:2px; }
  .main-area { margin-left:0; }
  .main-area .wrap { padding:24px 16px 80px; }
  .nav { flex-wrap:wrap; gap:8px; }
}
"""


_HISTORY_MENU_HTML = (
    '<div id="history-menu" class="history-context-menu" role="menu">'
    '<div id="history-menu-title" class="hm-title">历史报告</div>'
    '<div id="history-menu-path" class="hm-path"></div>'
    '<button id="history-delete-button" type="button" onclick="__deleteHistoryReport()">删除并释放本地文件</button>'
    '<div class="hm-tip">删除整个运行目录，并自动重建 latest.html 历史索引。</div>'
    '</div>'
)


_HISTORY_MENU_JS = (
    'window.__closeHistoryMenu=function(){var m=document.getElementById("history-menu");if(m)m.classList.remove("open");};'
    'window.__openHistoryMenu=function(ev,idx){ev.preventDefault();ev.stopPropagation();'
    'window.__HISTORY_MENU_IDX=idx;var r=window.__SIDEBAR__[idx]||{};var m=document.getElementById("history-menu");'
    'document.getElementById("history-menu-title").textContent=r.model_name||"历史报告";'
    'document.getElementById("history-menu-path").textContent="reports/"+(r.report_relpath||r.dir||"?")+"/";'
    'm.classList.add("open");var x=Math.min(ev.clientX,window.innerWidth-m.offsetWidth-8);'
    'var y=Math.min(ev.clientY,window.innerHeight-m.offsetHeight-8);m.style.left=Math.max(8,x)+"px";m.style.top=Math.max(8,y)+"px";return false;};'
    'window.__deleteHistoryReport=async function(){var idx=window.__HISTORY_MENU_IDX;var r=window.__SIDEBAR__[idx]||{};'
    'var rel=r.report_relpath;if(!rel||rel==="."){alert("该条目不是可删除的独立报告目录");return;}'
    'if(!confirm("确认删除本地报告目录 reports/"+rel+"/？\\n此操作会释放该目录占用的磁盘空间。"))return;'
    'if(location.protocol==="file:"){alert("静态 file:// 页面没有删除本地文件的权限。\\n请运行 python3 serve_reports.py，再从 http://127.0.0.1:8765/latest.html 打开。");return;}'
    'var b=document.getElementById("history-delete-button");b.disabled=true;b.textContent="正在删除…";'
    'try{var resp=await fetch("/__accuracy_bench__/delete-report",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({path:rel})});'
    'var body=await resp.json().catch(function(){return {};});if(!resp.ok)throw new Error(body.error||("HTTP "+resp.status));'
    'location.assign("/latest.html?refresh="+Date.now());}catch(err){alert("删除失败："+err.message+"\\n请确认页面由 serve_reports.py 提供服务。");'
    'b.disabled=false;b.textContent="删除并释放本地文件";}};'
    'document.addEventListener("click",window.__closeHistoryMenu);window.addEventListener("blur",window.__closeHistoryMenu);'
)


def _infer_report_mode(rd, dir_hint: str = "") -> str:
    """Return a stable sidebar mode, including for pre-run_mode reports."""
    explicit = str(getattr(rd, "run_mode", "") or "").strip().lower()
    if explicit in {"l1", "l2", "boundary", "full"}:
        return explicit.upper()

    padded_hint = f"_{str(dir_hint or '').lower()}_"
    for mode in ("boundary", "full", "l2", "l1"):
        if f"_{mode}_" in padded_hint:
            return mode.upper()

    has_l1 = bool(rd.l1_layers)
    has_l2 = bool(rd.l2_results)
    has_boundary = rd.overview.boundary_result is not None
    if has_boundary and (has_l1 or has_l2):
        return "FULL"
    if has_boundary:
        return "BOUNDARY"
    if has_l1 and has_l2:
        return "FULL"
    if has_l2:
        return "L2"
    if has_l1:
        return "L1"
    return "REPORT"


def _extract_report_meta(rd, raw: dict, dir_hint: str = "") -> dict:
    """Extract metadata from a ReportData object for sidebar display."""
    ic = rd.inference_compare
    lg = rd.logits
    return {
        "model_name": rd.overview.model_name,
        "run_mode": _infer_report_mode(rd, dir_hint),
        "run_status": rd.run_status or "—",
        "first_div": rd.overview.first_divergence_layer,
        "boundary": rd.overview.boundary_result or "—",
        "scope": rd.overview.comparison_scope or "unknown",
        "token_rate": ic.token_match_rate if ic else None,
        "exact": ic.exact_match if ic else None,
        "logits_positions": int(
            ((raw.get("logits") or {}).get("total_positions") or len(lg.token_positions))
            if lg else 0
        ),
        "full_data": raw,
    }


def _scan_sibling_reports(output_path: str, current_data: ReportData):
    """扫描 output_path 的上级目录树下所有 report_data.json, 返回 sidebar entries + all_data。

    用于在 product_report.html 左侧显示历史报告列表。
    当前报告 (output_path 对应目录) 排在第一位 (active)。

    Returns:
        (sidebar_entries, all_data, current_idx) 或 None (无兄弟报告)
    """
    import glob as _glob

    # output_path = .../reports/<subdir>/product_report.html
    # reports_root = .../reports/
    out_dir = os.path.dirname(os.path.abspath(output_path))
    reports_root = os.path.dirname(out_dir)
    if os.path.basename(reports_root) != "reports":
        # output 不在 reports/ 子目录下, 不扫
        return None

    json_files = _glob.glob(os.path.join(reports_root, "**", "report_data.json"), recursive=True)
    root_json = os.path.join(reports_root, "report_data.json")
    if root_json not in json_files and os.path.exists(root_json):
        json_files.append(root_json)

    current_dir_name = os.path.basename(out_dir) or "root"

    # 过滤掉当前目录的 JSON (当前报告用内存数据, 避免重复)
    sibling_files = [jf for jf in json_files
                     if (os.path.basename(os.path.dirname(jf)) or "root") != current_dir_name]
    if not sibling_files:
        return None  # 无兄弟报告

    runs = []
    # 当前报告先放进去 (内存数据, 可能 JSON 还没写)
    current_raw = _compact_report_for_html(current_data.to_dict())
    current_meta = _extract_report_meta(current_data, current_raw, current_dir_name)
    current_meta.update({
        "dir": current_dir_name,
        "report_relpath": os.path.relpath(out_dir, reports_root).replace(os.sep, "/"),
        "ts": time.strftime("%m-%d %H:%M"),
        "mtime": time.time(),
        "_is_current": True,
    })
    runs.append(current_meta)

    # 扫描磁盘上的兄弟报告 (已过滤掉当前目录)
    for jf in sorted(sibling_files, key=os.path.getmtime, reverse=True):
        jf_dir = os.path.basename(os.path.dirname(jf)) or "root"
        try:
            with open(jf, encoding="utf-8") as f:
                raw = json.load(f)
            from .report_schema import ReportData as _RD
            html_raw = _compact_report_for_html(raw)
            rd = _RD.from_dict(html_raw)
            mtime = os.path.getmtime(jf)
            meta = _extract_report_meta(rd, html_raw, jf_dir)
            meta.update({
                "dir": jf_dir,
                "report_relpath": os.path.relpath(
                    os.path.dirname(jf), reports_root
                ).replace(os.sep, "/"),
                "ts": time.strftime("%m-%d %H:%M", time.localtime(mtime)),
                "mtime": mtime,
                "_is_current": False,
            })
            runs.append(meta)
        except Exception as e:
            logger.debug(f"_scan_sibling_reports parse failed: {e}")
            continue

    # 按时间倒序 (当前报告强制排第一)
    current = [r for r in runs if r.get("_is_current")]
    others = sorted([r for r in runs if not r.get("_is_current")],
                     key=lambda x: x.get("mtime", 0), reverse=True)
    ordered = current + others

    all_data = [r["full_data"] for r in ordered]
    sidebar_entries = [{k: v for k, v in r.items() if k not in ("full_data", "_is_current")} for r in ordered]
    return sidebar_entries, all_data, 0  # current_idx=0


def _update_latest_report_link(target_path: str) -> str:
    """Point the repository-level ``latest.html`` at a generated report page.

    Linux uses a relative symlink so refreshing the browser always sees the
    regenerated target.  Environments that cannot create symlinks (notably a
    default Windows setup) receive a plain copy at the same stable path.
    """
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    link_path = os.path.join(repo_root, "latest.html")
    target_path = os.path.abspath(target_path)
    try:
        if os.path.lexists(link_path):
            os.remove(link_path)
        os.symlink(os.path.relpath(target_path, repo_root), link_path)
    except OSError as exc:
        try:
            shutil.copyfile(target_path, link_path)
            logger.debug("latest.html copied because symlink creation failed: %s", exc)
        except OSError as copy_error:
            logger.debug("Failed to update latest.html: %s", copy_error)
    return link_path


def generate_product_html_report(
    report_data: ReportData,
    output_path: Optional[str] = None,
) -> str:
    """生成 v2 产品化 HTML 报告 (自包含, 交互, 左侧带历史报告侧边栏)。

    自动扫描同目录树的兄弟 report_data.json, 有多个就显示左侧边栏
    (可点击切换), 没有就单列布局。

    Args:
        report_data: ``ReportData`` (由 :func:`report_data.assemble_report` 组装)
        output_path: 输出路径, None=自动 reports/product_report_<model>_<ts>.html

    Returns:
        HTML 文件路径
    """
    ts = time.strftime("%Y%m%d_%H%M%S")
    ov = report_data.overview
    safe_model = (ov.model_name or "model").replace("/", "_").replace(" ", "_")
    if output_path is None:
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        reports_dir = os.path.join(repo_root, "reports")
        os.makedirs(reports_dir, exist_ok=True)
        output_path = os.path.join(reports_dir, f"product_report_{safe_model}_{ts}.html")

    json_blob = _embed_json_safe(_compact_report_for_html(report_data.to_dict()))
    title = f"acc_bench 精度对齐报告 - {ov.model_name or '—'}"

    # --- 侧边栏: 扫描兄弟报告 ---
    sidebar_info = _scan_sibling_reports(output_path, report_data)

    body_sections = (
        '<div class="topbar"><div class="report-kicker"><span class="brand-mark">AB</span>'
        '<span>ACC BENCH / ALIGNMENT REPORT</span></div><h1>精度对齐报告</h1>'
        f'<span class="ts">{html.escape(ov.model_name or "")} · {ts}</span>'
        '<div class="accent-line"></div></div>'
        '<div class="nav">'
        '<a href="#overview-section" data-target="overview-section">概览</a>'
        '<a href="#l1-section" data-target="l1-section">L1 逐层</a>'
        '<a href="#l2-section" data-target="l2-section">L2 子图</a>'
        '<a href="#logits-section" data-target="logits-section">Logits</a>'
        '<a href="#badcase-section" data-target="badcase-section">生成对比</a>'
        '</div>'
        '<section id="overview-section"><h2>① 概览 — 一屏结论<span class="section-hint">点击 ❓ 图标看指标解释/公式</span></h2>'
        '<p class="sec-desc">先确认本次比较口径，再查看模型/设备、定界结论、L1 候选层、L2 误差边界与根因嫌疑。</p>'
        '<div id="overview"></div></section>'
        '<section id="l1-section"><h2>② L1 逐层对比</h2>'
        '<p class="sec-desc">比较每个 decoder block 的输出 hidden state；数值包含截至该层的累计误差，并不等于该层权重自身误差。默认显示首个诊断候选层；点击 "展开全部" 查看逐层 cos_sim / rel_l2 / SNR。'
        '绿≥0.99 / 黄0.95~0.99 / 红&lt;0.95, 行点击跳转 L2 该层。'
        '⚠ 黄色辅助告警 = 首次 cos_sim &lt; 0.99 的层 (累计缓慢下降, 非根因)。</p><div id="l1"></div><div id="l1legend"></div>'
        '<div id="l1table" style="margin-top:12px"></div></section>'
        '<section id="l2-section"><h2>③ L2 subgraph 反事实诊断</h2>'
        '<p class="sec-desc">柱状图只展示 Patch Recovery（高=好）；SelfRotErr / RotBErr 使用独立表格与各自阈值，避免把方向不同的指标混在同一比例尺。<b style="color:var(--good)">绿框=误差边界 (best_repair_point, coarse 子图)</b>'
        ' · <b style="color:var(--bad)">红框=根因算子 (source_candidate, fine 定位)</b>'
        ' · <span class="legend-chip" style="background:var(--deg)"></span>橙色=W4 降级量化 (W4A8_DYNAMIC/W4A4_DYNAMIC 等);'
        ' 可点击 ❓ 查看公式与解读。</p>'
        '<div id="l2"></div></section>'
        '<section id="logits-section"><h2>④ Logits 对比可视化</h2>'
        '<p class="sec-desc">散点 (ref vs quant) · top-k 并排柱 (Position 选择器切换) · token-wise 折线+KL · 分布直方图。Prompt prefill 与真实 decode step 会明确分开标注。</p>'
        '<div id="logits"></div></section>'
        '<section id="badcase-section"><h2>⑤ 生成输出对比</h2>'
        '<p class="sec-desc">同一输入下 Ref / Quant 的端到端生成结果、逐 Token 匹配率、首个分歧位置以及乱码/复读/NaN 信号。该区块回答最终行为是否已经分叉。</p>'
        '<div id="badcase"></div></section>'
        '<div class="legend-row">'
        '<span><span class="legend-chip" style="background:var(--ref)"></span>ref</span>'
        '<span><span class="legend-chip" style="background:var(--quant)"></span>quant</span>'
        '<span><span class="legend-chip" style="background:var(--good)"></span>对齐/好</span>'
        '<span><span class="legend-chip" style="background:var(--bad)"></span>发散/坏</span>'
        '<span><span class="legend-chip" style="background:var(--deg)"></span>DYNAMIC 降级量化</span>'
        '<span style="color:var(--muted)">INVALID_RUN 下排名仅供参考</span></div>'
        '<div id="helpModal" class="modal-ov" role="dialog" aria-modal="true" aria-hidden="true" aria-label="指标说明" onclick="__closeModal()"><div class="modal" '
        'onclick="event.stopPropagation()"><button type="button" class="close" aria-label="关闭指标说明" onclick="__closeModal()">✕</button>'
        '<div id="helpBody"></div></div></div>'
    )

    if sidebar_info is not None:
        # --- 有兄弟报告: 双列布局 (左侧边栏 + 右侧主内容) ---
        sidebar_entries, all_data, current_idx = sidebar_info
        all_data_blob = _embed_json_safe(all_data)
        sidebar_blob = _embed_json_safe(sidebar_entries)

        # JS: let R (可变) + switchReport + renderSidebar
        index_js = _V2_JS.replace("const R = window.__REPORT__;", "let R;")
        index_js = index_js.replace(
            'if(document.readyState!=="loading")boot();else document.addEventListener("DOMContentLoaded",boot);\n})();',
            'window.__switchReport=function(idx){'
            'R=window.__REPORTS__[idx];'
            'document.querySelectorAll(".sidebar-item").forEach(function(e,i){'
            'e.classList.toggle("active",i===idx);});'
            'boot();};'
            'function renderSidebar(){'
            'var sb=document.getElementById("sidebar-list");'
            'var h="";'
            'window.__SIDEBAR__.forEach(function(r,i){'
            'var st=r.run_status||"—";'
            'var bc=st.indexOf("SUCCESS")>=0?"good":st.indexOf("INVALID")>=0?"bad":(st.indexOf("PARTIAL")>=0||st.indexOf("INCONCLUSIVE")>=0)?"warn":"na";'
            'var mode=(r.run_mode||"REPORT").toUpperCase();var mc="mode-"+mode.toLowerCase();'
            'var m="";'
            'if(r.first_div!==null&&r.first_div!==undefined)m+="<span>候选 L"+r.first_div+"</span>";'
            'if(r.scope==="weight_plus_activation_qdq")m+="<span>W + ACT</span>";'
            'else if(r.scope==="weight_only")m+="<span>WEIGHT</span>";'
            'if(r.token_rate!==null&&r.token_rate!==undefined)m+="<span>"+(r.token_rate*100).toFixed(0)+"% match</span>";'
            'if(r.logits_positions>0)m+="<span>"+r.logits_positions+" logits</span>";'
            'h+="<div class=\\"sidebar-item\\" data-idx=\\""+i+"\\" onclick=\\"__switchReport("+i+")\\" oncontextmenu=\\"return __openHistoryMenu(event,"+i+")\\">"'
            '+"<div class=\\"si-ts\\">"+r.ts+"</div>"'
            '+"<div class=\\"si-model\\">"+esc(r.model_name)+"</div>"'
            '+"<div class=\\"si-meta\\"><span class=\\"si-badge mode "+mc+"\\">"+mode+"</span><span class=\\"si-badge "+bc+"\\">"+st+"</span>"+m+"</div>"'
            '+"</div>";});'
            'sb.innerHTML=h;}'
            'renderSidebar();'
            + _HISTORY_MENU_JS +
            'if(document.readyState!=="loading")window.__switchReport(' + str(current_idx) + ');'
            'else document.addEventListener("DOMContentLoaded",function(){window.__switchReport(' + str(current_idx) + ');});\n})();'
        )

        body = (
            '<div class="layout">'
            '<div class="sidebar"><div class="sidebar-brand"><span class="brand-mark">AB</span>'
            '<div>acc_bench<small>report archive</small></div></div>'
            '<h2>历史报告<span class="sidebar-delete-hint">右键删除</span></h2><div id="sidebar-list"></div></div>'
            '<div class="main-area"><div class="wrap">'
            + body_sections
            + '</div></div></div>' + _HISTORY_MENU_HTML
        )

        html_doc = (
            "<!doctype html><html lang='zh'><head><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>{html.escape(title)}</title>"
            f"<style>{_V2_CSS}{_SIDEBAR_CSS}</style></head><body>"
            + body
            + f"<script type='application/json' id='reports-data'>{all_data_blob}</script>"
            + f"<script type='application/json' id='sidebar-data'>{sidebar_blob}</script>"
            + "<script>window.__REPORTS__=JSON.parse(document.getElementById('reports-data').textContent);"
            + "window.__SIDEBAR__=JSON.parse(document.getElementById('sidebar-data').textContent);"
            + "window.__REPORT__=window.__REPORTS__[" + str(current_idx) + "];</script>"
            + f"<script>{index_js}</script></body></html>"
        )
    else:
        # --- 无兄弟报告: 单列布局 (原样) ---
        html_doc = (
            "<!doctype html><html lang='zh'><head><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>{html.escape(title)}</title>"
            f"<style>{_V2_CSS}</style></head><body><div class='wrap'>"
            + body_sections
            + f"</div><script type='application/json' id='report-data'>{json_blob}</script>"
            f"<script>window.__REPORT__=JSON.parse(document.getElementById('report-data').textContent);</script>"
            f"<script>{_V2_JS}</script></body></html>"
        )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_doc)

    # Direct API callers still receive a useful latest.html.  CLI workflows
    # generate index.html immediately afterwards, replacing this with the
    # current + historical report dashboard.
    _update_latest_report_link(output_path)

    return output_path


def generate_index_html(reports_dir: str, output_path: Optional[str] = None) -> str:
    """生成汇总 index.html: 左侧边栏历史记录 + 主区域完整报告。

    扫描 reports_dir 下所有 report_data.json, 按时间倒序排列,
    内嵌全部报告数据, 自包含 HTML 无需 web server。

    Args:
        reports_dir: reports 目录路径
        output_path: 输出路径, None=reports_dir/index.html

    Returns:
        HTML 文件路径
    """
    import glob

    if output_path is None:
        output_path = os.path.join(reports_dir, "index.html")

    # Scan for all report_data.json files
    json_files = glob.glob(os.path.join(reports_dir, "**", "report_data.json"), recursive=True)
    # Also check root level
    root_json = os.path.join(reports_dir, "report_data.json")
    if root_json not in json_files and os.path.exists(root_json):
        json_files.append(root_json)

    # Load each and extract summary
    from .report_schema import ReportData
    runs = []
    for jf in sorted(json_files, key=os.path.getmtime, reverse=True):
        try:
            with open(jf, encoding="utf-8") as f:
                raw = json.load(f)
            html_raw = _compact_report_for_html(raw)
            rd = ReportData.from_dict(html_raw)
            mtime = os.path.getmtime(jf)
            ts_str = time.strftime("%m-%d %H:%M", time.localtime(mtime))
            run_dir = os.path.basename(os.path.dirname(jf)) or "root"
            meta = _extract_report_meta(rd, html_raw, run_dir)
            meta.update({
                "dir": run_dir,
                "report_relpath": os.path.relpath(
                    os.path.dirname(jf), reports_dir
                ).replace(os.sep, "/"),
                "ts": ts_str,
                "mtime": mtime,
            })
            if not meta["model_name"]:
                meta["model_name"] = run_dir
            runs.append(meta)
        except Exception as e:
            logger.debug(f"_scan_sibling_reports parse failed: {e}")
            continue

    if not runs:
        # No reports found, generate minimal HTML
        html_doc = (
            "<!doctype html><html lang='zh'><head><meta charset='utf-8'>"
            "<title>acc_bench 历史报告</title>"
            f"<style>{_V2_CSS}</style></head><body><div class='wrap'>"
            "<h1>暂无历史报告</h1><p>运行 acc_bench 后此处会显示所有历史报告。</p>"
            "</div></body></html>"
        )
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_doc)
        _update_latest_report_link(output_path)
        return output_path

    # Build sidebar + main area HTML
    # Embed all reports' full data as JSON array
    all_data = [r["full_data"] for r in runs]
    # Remove full_data from sidebar entries (keep only summary)
    sidebar_entries = [{k: v for k, v in r.items() if k != "full_data"} for r in runs]
    json_blob = _embed_json_safe(all_data)
    sidebar_blob = _embed_json_safe(sidebar_entries)

    # Product report and history index share one visual/sidebar system.
    sidebar_css = _SIDEBAR_CSS

    # Modified JS: let R instead of const, switchReport function
    # Replace 'const R = window.__REPORT__;' with 'let R;'
    index_js = _V2_JS.replace("const R = window.__REPORT__;", "let R;")
    # Replace the auto-boot with switchReport
    index_js = index_js.replace(
        'if(document.readyState!=="loading")boot();else document.addEventListener("DOMContentLoaded",boot);\n})();',
        'window.__switchReport=function(idx){'
        'R=window.__REPORTS__[idx];'
        'document.getElementById("run-title").textContent=R.overview?.model_name||"—";'
        'document.getElementById("run-ts").textContent=window.__SIDEBAR__[idx]?.ts||"";'
        'document.querySelectorAll(".sidebar-item").forEach(function(e,i){'
        'e.classList.toggle("active",i===idx);});'
        'boot();};'
        'function renderSidebar(){'
        'var sb=document.getElementById("sidebar-list");'
        'var html="";'
        'window.__SIDEBAR__.forEach(function(r,i){'
        'var st=r.run_status||"—";'
        'var bc=st.indexOf("SUCCESS")>=0?"good":st.indexOf("INVALID")>=0?"bad":(st.indexOf("PARTIAL")>=0||st.indexOf("INCONCLUSIVE")>=0)?"warn":"na";'
        'var mode=(r.run_mode||"REPORT").toUpperCase();var mc="mode-"+mode.toLowerCase();'
        'var metrics="";'
        'if(r.first_div!==null&&r.first_div!==undefined)metrics+="<span>候选 L"+esc(r.first_div)+"</span>";'
        'if(r.scope==="weight_plus_activation_qdq")metrics+="<span>W + ACT</span>";'
        'else if(r.scope==="weight_only")metrics+="<span>WEIGHT</span>";'
        'if(r.token_rate!==null&&r.token_rate!==undefined)metrics+="<span>"+esc((r.token_rate*100).toFixed(0))+"% match</span>";'
        'if(r.logits_positions>0)metrics+="<span>"+esc(r.logits_positions)+" logits</span>";'
        'html+="<div class=\\"sidebar-item\\" data-idx=\\""+i+"\\" onclick=\\"__switchReport("+i+")\\" oncontextmenu=\\"return __openHistoryMenu(event,"+i+")\\">"'
        '+"<div class=\\"si-ts\\">"+r.ts+"</div>"'
        '+"<div class=\\"si-model\\">"+esc(r.model_name)+"</div>"'
        '+"<div class=\\"si-meta\\"><span class=\\"si-badge mode "+mc+"\\">"+mode+"</span><span class=\\"si-badge "+bc+"\\">"+st+"</span>"+metrics+"</div>"'
        '+"</div>";});'
        'sb.innerHTML=html;}'
        'renderSidebar();'
        + _HISTORY_MENU_JS +
        'if(document.readyState!=="loading")window.__switchReport(0);'
        'else document.addEventListener("DOMContentLoaded",function(){window.__switchReport(0);});\n})();'
    )

    body = (
        '<div class="layout">'
        '<div class="sidebar"><div class="sidebar-brand"><span class="brand-mark">AB</span>'
        '<div>acc_bench<small>report archive</small></div></div>'
        '<h2>历史报告<span class="sidebar-delete-hint">右键删除</span></h2><div id="sidebar-list"></div></div>'
        '<div class="main-area"><div class="wrap">'
        '<div class="topbar"><div class="report-kicker"><span class="brand-mark">AB</span>'
        '<span>ACC BENCH / ALIGNMENT REPORT</span></div><h1>精度对齐报告</h1>'
        '<span class="ts" id="run-title">—</span>'
        '<span class="ts" id="run-ts" style="margin-left:8px"></span>'
        '<div class="accent-line"></div></div>'
        '<div class="nav">'
        '<a href="#overview-section" data-target="overview-section">概览</a>'
        '<a href="#l1-section" data-target="l1-section">L1 逐层</a>'
        '<a href="#l2-section" data-target="l2-section">L2 子图</a>'
        '<a href="#logits-section" data-target="logits-section">Logits</a>'
        '<a href="#badcase-section" data-target="badcase-section">生成对比</a>'
        '</div>'
        '<section id="overview-section"><h2>① 概览 — 一屏结论</h2>'
        '<p class="sec-desc">先确认本次比较口径，再查看模型/设备、定界结论、L1 候选层、L2 误差边界与根因嫌疑。</p>'
        '<div id="overview"></div></section>'
        '<section id="l1-section"><h2>② L1 逐层对比</h2>'
        '<p class="sec-desc">比较每个 decoder block 的输出 hidden state；数值包含截至该层的累计误差，并不等于该层权重自身误差。默认显示首个诊断候选层。</p>'
        '<div id="l1"></div><div id="l1legend"></div>'
        '<div id="l1table" style="margin-top:12px"></div></section>'
        '<section id="l2-section"><h2>③ L2 subgraph 反事实诊断</h2>'
        '<p class="sec-desc">柱状图仅展示 Patch Recovery；SelfRotErr / RotBErr 使用独立指标表。</p>'
        '<div id="l2"></div></section>'
        '<section id="logits-section"><h2>④ Logits 对比可视化</h2>'
        '<p class="sec-desc">散点 · top-k 并排柱 · token-wise 折线+KL · 分布直方图；Prompt prefill 与 decode step 分开标注。</p>'
        '<div id="logits"></div></section>'
        '<section id="badcase-section"><h2>⑤ 生成输出对比</h2>'
        '<p class="sec-desc">同一输入下 Ref / Quant 的端到端生成结果、Token 分歧和退化信号。</p>'
        '<div id="badcase"></div></section>'
        '<div class="legend-row">'
        '<span><span class="legend-chip" style="background:var(--ref)"></span>ref</span>'
        '<span><span class="legend-chip" style="background:var(--quant)"></span>quant</span>'
        '<span><span class="legend-chip" style="background:var(--good)"></span>对齐</span>'
        '<span><span class="legend-chip" style="background:var(--bad)"></span>发散</span>'
        '</div>'
        '<div id="helpModal" class="modal-ov" role="dialog" aria-modal="true" aria-hidden="true" aria-label="指标说明" onclick="__closeModal()"><div class="modal" '
        'onclick="event.stopPropagation()"><button type="button" class="close" aria-label="关闭指标说明" onclick="__closeModal()">✕</button>'
        '<div id="helpBody"></div></div></div>'
        '</div></div></div>' + _HISTORY_MENU_HTML
    )

    html_doc = (
        "<!doctype html><html lang='zh'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>acc_bench 历史报告</title>"
        f"<style>{_V2_CSS}{sidebar_css}</style></head><body>"
        + body
        + f"<script type='application/json' id='reports-data'>{json_blob}</script>"
        + f"<script type='application/json' id='sidebar-data'>{sidebar_blob}</script>"
        + "<script>window.__REPORTS__=JSON.parse(document.getElementById('reports-data').textContent);"
        + "window.__SIDEBAR__=JSON.parse(document.getElementById('sidebar-data').textContent);"
        + "window.__REPORT__=window.__REPORTS__[0];</script>"
        + f"<script>{index_js}</script></body></html>"
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_doc)
    _update_latest_report_link(output_path)
    return output_path


__all__ = [
    # v1 (backward-compat)
    "generate_html_report", "_detect_repetition", "_verdict_banner",
    "_err_class", "_flip_class", "_rec_class", "_fmt_pct", "_fmt_err",
    # v2
    "generate_product_html_report", "generate_index_html",
]
