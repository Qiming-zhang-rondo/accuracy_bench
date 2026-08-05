"""独立推理结果 HTML 报告生成器。

1 个模型 → 单独展示生成结果
2 个模型 → 并排对比 + token 级 diff
"""
from __future__ import annotations
import html
import os
import time
from typing import Dict, List, Optional


def _build_token_diff(models):
    """构建两模型 token 级对比 HTML + 指标条; 非对比场景返回 ("", "")。"""
    if len(models) < 2:
        return "", ""
    m1, m2 = models[0], models[1]
    t1 = m1.get("token_strs", [])
    t2 = m2.get("token_strs", [])
    max_len = max(len(t1), len(t2))
    match_count = 0
    spans1, spans2 = "", ""
    for i in range(max_len):
        a = t1[i] if i < len(t1) else None
        b = t2[i] if i < len(t2) else None
        match = a is not None and b is not None and a == b
        if match:
            match_count += 1
        bg = "rgba(76,175,80,0.12)" if match else "rgba(244,67,54,0.12)"
        spans1 += f'<span style="background:{bg};padding:1px 3px;border-radius:3px;margin:1px">{html.escape(a or "∅")}</span> '
        spans2 += f'<span style="background:{bg};padding:1px 3px;border-radius:3px;margin:1px">{html.escape(b or "∅")}</span> '
    rate = (match_count / max_len * 100) if max_len else 0
    exact = m1.get("output", "") == m2.get("output", "")
    badge = "完全一致" if exact else f"token 匹配率 {rate:.1f}%"
    badge_cls = "badge-ok" if (exact or rate >= 80) else "badge-bad"
    diff_html = f'''
        <div class="card">
          <h3>Token 级对比 <span class="badge {badge_cls}">{badge}</span></h3>
          <div class="grid-2">
            <div><div class="label" style="color:var(--c1)">{html.escape(m1.get("name","ref"))}</div>
              <div class="tok-stream">{spans1}</div></div>
            <div><div class="label" style="color:var(--c2)">{html.escape(m2.get("name","quant"))}</div>
              <div class="tok-stream">{spans2}</div></div>
          </div>
        </div>'''
    metrics_html = f'<div class="metrics"><span>匹配 token: {match_count}/{max_len}</span><span>匹配率: {rate:.1f}%</span></div>'
    return diff_html, metrics_html


def _build_model_card(m, is_first):
    """构建单个模型的卡片 HTML。"""
    name = html.escape(m.get("name", "?"))
    out = html.escape(m.get("output", "—"))
    n_tok = m.get("num_tokens", 0)
    prefill_t = m.get("prefill_time")
    decode_t = m.get("decode_times", [])
    extras = []
    if n_tok:
        extras.append(f"{n_tok} tokens")
    if prefill_t:
        extras.append(f"prefill {prefill_t:.1f}s")
    if decode_t:
        avg = sum(decode_t) / len(decode_t) if decode_t else 0
        extras.append(f"decode avg {avg:.1f}s/tok")
    meta = " · ".join(extras)
    color = "var(--c1)" if is_first else "var(--c2)"
    return f'''
        <div class="card">
          <h3 style="color:{color}">{name}</h3>
          {f'<div class="meta">{meta}</div>' if meta else ''}
          <div class="output-text">{out}</div>
        </div>'''


def _build_model_cards(models):
    """构建所有模型卡片 HTML。"""
    return "".join(_build_model_card(m, m == models[0]) for m in models)


def _build_html_doc(title, ts, prompt, metrics_html, grid_cls, cards, diff_html):
    """组装完整 HTML 文档字符串。"""
    return f"""<!doctype html>
<html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>推理结果 — {html.escape(title)}</title>
<style>
  :root {{
    --bg:#fafafa; --card:#fff; --border:#e0e0e0; --muted:#888;
    --c1:#2563eb; --c2:#ea580c; --good:#4caf50; --bad:#f44336;
  }}
  * {{ box-sizing:border-box; margin:0; padding:0; }}
  body {{ font-family:-apple-system,'Segoe UI',Roboto,sans-serif; background:var(--bg); color:#1a1a1a; line-height:1.6; }}
  .wrap {{ max-width:1100px; margin:0 auto; padding:24px 16px; }}
  h1 {{ font-size:22px; font-weight:600; margin-bottom:4px; }}
  .ts {{ font-size:12px; color:var(--muted); margin-bottom:20px; }}
  .card {{ background:var(--card); border:1px solid var(--border); border-radius:8px; padding:16px; margin-bottom:16px; }}
  .card h3 {{ font-size:15px; font-weight:600; margin-bottom:8px; }}
  .meta {{ font-size:12px; color:var(--muted); margin-bottom:8px; }}
  .output-text {{ font-size:14px; white-space:pre-wrap; word-break:break-all; line-height:1.8; }}
  .prompt-box {{ background:#f5f5f5; border-radius:6px; padding:12px; font-size:14px; color:var(--muted); margin-bottom:16px; }}
  .prompt-box b {{ color:#333; }}
  .grid-2 {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; }}
  .tok-stream {{ font-size:13px; line-height:2.2; word-break:break-all; }}
  .label {{ font-size:13px; font-weight:600; margin-bottom:4px; }}
  .badge {{ font-size:12px; padding:2px 8px; border-radius:10px; font-weight:500; }}
  .badge-ok {{ background:rgba(76,175,80,0.15); color:var(--good); }}
  .badge-bad {{ background:rgba(244,67,54,0.15); color:var(--bad); }}
  .metrics {{ display:flex; gap:16px; font-size:13px; color:var(--muted); margin-bottom:16px; }}
  @media(max-width:700px) {{ .grid-2 {{ grid-template-columns:1fr; }} }}
</style></head><body><div class="wrap">
  <h1>推理结果 — {html.escape(title)}</h1>
  <div class="ts">{ts}</div>
  <div class="prompt-box"><b>Prompt:</b> {html.escape(prompt)}</div>
  {metrics_html}
  <div class="{grid_cls}">{cards}</div>
  {diff_html}
</div></body></html>"""


def _update_latest_symlink(output_path):
    """更新 latest_inference.html 软链指向 output_path。失败静默。"""
    try:
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        link = os.path.join(repo_root, "latest_inference.html")
        rel = os.path.relpath(output_path, repo_root)
        if os.path.islink(link) or os.path.exists(link):
            os.remove(link)
        os.symlink(rel, link)
    except OSError:
        pass


def generate_inference_html(
    data: dict,
    output_path: Optional[str] = None,
) -> str:
    """生成独立推理 HTML。

    Args:
        data: {"prompt": str, "models": [{"name", "output", "token_strs", ...}]}
              models 长度 1=单模型, 2=对比
        output_path: 输出路径

    Returns:
        HTML 文件路径
    """
    prompt = data.get("prompt", "")
    models = data.get("models", [])
    is_compare = len(models) >= 2

    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    title_parts = [m.get("name", "?") for m in models]
    title = " vs ".join(title_parts) if is_compare else (title_parts[0] if title_parts else "推理结果")

    diff_html, metrics_html = _build_token_diff(models)
    cards = _build_model_cards(models)
    grid_cls = "grid-2" if is_compare else ""

    html_doc = _build_html_doc(title, ts, prompt, metrics_html, grid_cls, cards, diff_html)

    if output_path is None:
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_path = os.path.join(repo_root, "inference_report.html")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_doc)

    _update_latest_symlink(output_path)

    return output_path
