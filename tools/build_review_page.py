"""
Build a manual-verification page.

Pairs each candidate's parsed record with the Chinese translation of their
resume, side by side, so a reviewer can check field by field without moving
between a spreadsheet and a text file.

    python tools/build_review_page.py

Output: data/review_zh.html (self-contained, opens in any browser)

This is a working tool, not a deliverable. Its output is how the ground-truth
labels for the accuracy evaluation get produced.
"""

from __future__ import annotations

import html
import json
import re
import sys
import unicodedata
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parent.parent
TRANSLATION = ROOT / "data" / "resumes_zh.md"
CANDIDATES = ROOT / "data" / "candidates.json"
OUTPUT = ROOT / "data" / "review_zh.html"


def split_sections(text: str) -> list[tuple[str, str]]:
    """Split the translation on H2 headings into (title, markdown body)."""
    parts = re.split(r"^## +(.+?)$", text, flags=re.M)
    return [
        (parts[i].strip(), parts[i + 1].strip()) for i in range(1, len(parts) - 1, 2)
    ]


def norm(text: str) -> str:
    """Fold case, accents and punctuation for loose name matching."""
    folded = "".join(
        c
        for c in unicodedata.normalize("NFKD", text)
        if not unicodedata.combining(c)
    )
    return re.sub(r"[^a-z0-9]+", "", folded.lower())


def match_candidate(title: str, body: str, candidates: list[dict]) -> dict | None:
    """Find the parsed record for a translated section.

    Matches on the source filename quoted in the section body first, since
    that is unambiguous; falls back to a normalised name comparison.
    """
    if file_match := re.search(r"`([^`]+\.(?:docx|pdf))`", body):
        stem = file_match.group(1)
        for c in candidates:
            if c["source_file"] == stem:
                return c

    key = norm(title)
    for c in candidates:
        name = norm(c["display_name"])
        if key and name and (key in name or name in key):
            return c
    return None


def field_rows(c: dict) -> list[tuple[str, str, str]]:
    """(label, value, extra-css-class) rows for the parsed-fields panel."""
    e = c["extraction"]
    approach = e["investment_approach"]
    side = e["market_side"]

    def join(values: list[str]) -> str:
        return ", ".join(values) if values else "—"

    rows: list[tuple[str, str, str]] = [
        ("姓名 name", c["display_name"], ""),
        (
            "姓名来源 name_source",
            c["name_source"],
            "warn" if c["name_source"] == "filename" else "",
        ),
        ("地区 region", c.get("region") or "—", ""),
        ("地点 location", c.get("location") or "—", ""),
        ("总年限 years_experience", str(c.get("years_experience") or "—"), ""),
        (
            "投研年限 years_investment",
            str(c.get("years_investment_experience") or "—"),
            "",
        ),
        ("初级区间 in_junior_range", str(c.get("is_junior_range")), ""),
        ("资历 seniority_band", c.get("seniority_band") or "—", ""),
        ("方法 approach", f"{approach['value']}  ({approach['confidence']})", ""),
        ("市场侧 market_side", f"{side['value']}  ({side['confidence']})", ""),
        ("方法族 approach_family", c.get("approach_family") or "—", ""),
        ("行业 sectors", join(c.get("sectors", [])), ""),
        ("资产类别 asset_classes", join(c.get("asset_classes", [])), ""),
        (
            "覆盖市场 coverage_markets",
            join(c.get("coverage_markets", []))
            + ("  (推断)" if c.get("coverage_markets_source") == "inferred" else ""),
            "warn" if c.get("coverage_markets_source") == "inferred" else "",
        ),
        ("当前机构 current_firm", c.get("current_firm") or "—", ""),
        ("机构类型 current_firm_type", c.get("current_firm_type") or "—", ""),
        ("买方经历 has_buy_side", str(c.get("has_buy_side_experience")), ""),
        ("卖方经历 has_sell_side", str(c.get("has_sell_side_experience")), ""),
        (
            "平台校友 platform_alum_of",
            join(c.get("platform_alum_of", [])),
            "flag" if c.get("platform_alum_of") else "",
        ),
        ("雇主 employers", join(c.get("employers", [])), ""),
        ("非职业经历 non_professional", join(c.get("non_professional_affiliations", [])), ""),
        ("资质 credentials", join(c.get("credentials_summary", [])), ""),
        ("语言 languages", join(c.get("languages", [])), ""),
        ("软件 software_tools", join(c.get("software_tools", [])), ""),
        ("方法 methods", join(c.get("methods", [])), ""),
        ("覆盖只数 stocks_covered", str(e["coverage"].get("stocks_covered") or "—"), ""),
        ("职位条数 n_positions", str(len(e["positions"])), ""),
    ]
    return rows


def flags_block(c: dict) -> str:
    """Merged problem list, model-found and computed, with provenance."""
    if not c.get("flags"):
        return "<p style='color:var(--muted);font-size:13px'>未发现问题</p>"
    items = []
    for f in c["flags"]:
        quote = (
            f"<blockquote>{html.escape(f['quote'])}</blockquote>" if f.get("quote") else ""
        )
        items.append(
            f"<li><span class='tag {f['source']}'>{f['source']}</span>"
            f"<span class='cat'>{html.escape(f['category'])}</span><br>"
            f"<b>{html.escape(f['summary'])}</b><br>"
            f"<span class='fd'>{html.escape(f['detail'])}</span>{quote}</li>"
        )
    return f"<ul class='flags'>{''.join(items)}</ul>"


def positions_table(c: dict) -> str:
    """Render parsed positions -- the densest source of extraction errors."""
    head = (
        "<tr><th>机构 firm</th><th>解析为 canonical</th><th>职位 title</th>"
        "<th>起止</th><th>类型</th><th>投研?</th></tr>"
    )
    firms = {f["raw"]: f for f in c.get("firms", [])}
    body = []
    for p in c["extraction"]["positions"]:
        link = firms.get(p["firm"], {})
        resolution = link.get("resolution", "unresolved")
        canonical = link.get("canonical") or "—"
        cls = "bad" if resolution in {"unresolved", "ambiguous"} else ""
        dates = f"{p.get('start_date') or '?'} → {'现在' if p.get('is_current') else (p.get('end_date') or '?')}"
        if not p.get("start_date") and p.get("duration_months"):
            dates = f"时长 {p['duration_months']} 个月"
        body.append(
            "<tr>"
            f"<td>{html.escape(p['firm'])}</td>"
            f"<td class='{cls}'>{html.escape(canonical)}"
            f"<span class='res'>{html.escape(resolution)}</span></td>"
            f"<td>{html.escape(p['title'])}</td>"
            f"<td class='mono'>{html.escape(dates)}</td>"
            f"<td>{html.escape(p.get('employment_type', '—'))}</td>"
            f"<td>{'✓' if p.get('is_investment_role') else '·'}</td>"
            "</tr>"
        )
    return f"<table class='pos'>{head}{''.join(body)}</table>"


def evidence_block(c: dict) -> str:
    """Show each judged field beside the quote that justifies it."""
    e = c["extraction"]
    items = [("approach", e["investment_approach"]), ("market_side", e["market_side"])]
    items += [(f"sector · {s['value']}", s) for s in e["primary_sectors"]]
    out = []
    for label, inferred in items:
        quote = inferred.get("evidence") or "（无证据）"
        kws = "".join(
            f"<span class='kw'>{html.escape(k)}</span>"
            for k in inferred.get("keywords", [])
        )
        out.append(
            f"<div class='ev'><div class='ev-h'>{html.escape(label)} "
            f"<b>{html.escape(inferred['value'])}</b> "
            f"<span class='conf {inferred['confidence']}'>"
            f"{inferred['confidence']}</span></div>"
            f"<div class='kws'>{kws}</div>"
            f"<blockquote>{html.escape(quote)}</blockquote></div>"
        )
    return "".join(out)


CSS = """
:root{--bg:#fbfaf8;--panel:#fff;--ink:#1b1a18;--muted:#6b675f;--line:#e5e1da;
--accent:#7a4d1d;--warn:#b45309;--bad:#b91c1c;--good:#15803d;--flag:#7c2d92;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC",
"Hiragino Sans GB","Microsoft YaHei",sans-serif;}
.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.86em}
header{position:sticky;top:0;z-index:20;background:var(--panel);
border-bottom:1px solid var(--line);padding:14px 22px}
header h1{margin:0;font-size:17px;letter-spacing:-.01em}
header p{margin:3px 0 0;color:var(--muted);font-size:13px}
nav{display:flex;flex-wrap:wrap;gap:6px;padding:10px 22px;background:var(--panel);
border-bottom:1px solid var(--line);position:sticky;top:63px;z-index:19}
nav a{font-size:12.5px;padding:4px 10px;border:1px solid var(--line);
border-radius:20px;text-decoration:none;color:var(--ink);background:var(--bg)}
nav a:hover{border-color:var(--accent);color:var(--accent)}
main{padding:22px;max-width:1580px;margin:0 auto}
section{margin-bottom:34px;scroll-margin-top:120px}
h2{font-size:20px;margin:0 0 4px;letter-spacing:-.01em}
.src{color:var(--muted);font-size:12.5px;margin-bottom:12px}
.grid{display:grid;grid-template-columns:minmax(390px,1fr) minmax(430px,1.15fr);
gap:18px;align-items:start}
@media(max-width:1080px){.grid{grid-template-columns:1fr}}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;
padding:16px 18px}
.card h3{margin:0 0 12px;font-size:12px;text-transform:uppercase;
letter-spacing:.08em;color:var(--muted);font-weight:600}
table{border-collapse:collapse;width:100%;font-size:13px}
.kv td{padding:5px 6px;border-bottom:1px solid var(--line);vertical-align:top}
.kv td:first-child{color:var(--muted);width:44%;white-space:nowrap}
.kv .warn td:last-child{color:var(--warn);font-weight:600}
.kv .flag td:last-child{color:var(--flag);font-weight:600}
.pos{margin-top:6px;font-size:12.5px}
.pos th{text-align:left;color:var(--muted);font-weight:600;padding:5px 6px;
border-bottom:1px solid var(--line);white-space:nowrap}
.pos td{padding:5px 6px;border-bottom:1px solid var(--line);vertical-align:top}
.pos td.bad{color:var(--bad)}
.res{display:block;font-size:10.5px;color:var(--muted);text-transform:uppercase;
letter-spacing:.05em}
.ev{margin-bottom:10px}
.ev-h{font-size:12px;color:var(--muted)}
.ev-h b{color:var(--ink)}
.conf{font-size:10.5px;padding:1px 6px;border-radius:9px;margin-left:4px}
.conf.high{background:#dcfce7;color:var(--good)}
.conf.medium{background:#fef3c7;color:var(--warn)}
.conf.low{background:#fee2e2;color:var(--bad)}
blockquote{margin:3px 0 0;padding:6px 10px;border-left:3px solid var(--line);
background:var(--bg);font-size:12.5px;color:#3f3b35}
.quality{margin-top:10px;font-size:12.5px}
.badge{display:inline-block;padding:2px 9px;border-radius:9px;font-weight:600;
font-size:11.5px}
.badge.high{background:#dcfce7;color:var(--good)}
.badge.medium{background:#fef3c7;color:var(--warn)}
.badge.low{background:#fee2e2;color:var(--bad)}
.kws{margin:4px 0 2px}
.kw{display:inline-block;background:#f0ece4;color:var(--accent);font-size:11.5px;
padding:1px 8px;border-radius:9px;margin:0 4px 3px 0;font-weight:600}
.flags{list-style:none;margin:0;padding:0}
.flags li{border-left:3px solid var(--line);padding:6px 0 6px 10px;
margin-bottom:9px;font-size:12.5px}
.tag{font-size:10px;text-transform:uppercase;letter-spacing:.06em;
padding:1px 6px;border-radius:8px;font-weight:700;margin-right:5px}
.tag.model{background:#ede9fe;color:#5b21b6}
.tag.computed{background:#dbeafe;color:#1e40af}
.cat{font-size:10.5px;color:var(--muted);text-transform:uppercase;
letter-spacing:.05em}
.fd{color:var(--muted)}
.zh h3{font-size:13.5px;text-transform:none;letter-spacing:0;color:var(--accent);
margin:16px 0 6px}
.zh h3:first-child{margin-top:0}
.zh p{margin:6px 0}
.zh ul{margin:6px 0;padding-left:19px}
.zh li{margin-bottom:4px}
.zh blockquote{border-left:3px solid var(--warn);background:#fffbeb;
color:#78350f;font-size:12.5px}
.zh strong{font-weight:650}
.zh hr{border:0;border-top:1px solid var(--line);margin:14px 0}
footer{padding:22px;text-align:center;color:var(--muted);font-size:12px}
"""


def main() -> int:
    if not TRANSLATION.exists() or not CANDIDATES.exists():
        print("Run build_dataset.py and produce resumes_zh.md first.", file=sys.stderr)
        return 1

    candidates = json.loads(CANDIDATES.read_text(encoding="utf-8"))
    sections = split_sections(TRANSLATION.read_text(encoding="utf-8"))
    md = markdown.Markdown(extensions=["extra", "sane_lists"])

    nav, blocks, unmatched = [], [], []
    for index, (title, body) in enumerate(sections):
        candidate = match_candidate(title, body, candidates)
        anchor = f"c{index}"
        nav.append(f"<a href='#{anchor}'>{html.escape(title)}</a>")

        if candidate is None:
            unmatched.append(title)
            left = "<div class='card'><h3>系统解析结果</h3><p>未能匹配到解析记录。</p></div>"
        else:
            rows = "".join(
                f"<tr class='{cls}'><td>{html.escape(label)}</td>"
                f"<td>{html.escape(value)}</td></tr>"
                for label, value, cls in field_rows(candidate)
            )
            q = candidate["quality"]
            left = (
                "<div class='card'><h3>系统解析结果 parsed record</h3>"
                f"<table class='kv'>{rows}</table>"
                f"{positions_table(candidate)}"
                "<h3 style='margin-top:16px'>判断依据 evidence</h3>"
                f"{evidence_block(candidate)}"
                "<h3 style='margin-top:16px'>发现的问题 flags "
                f"<span class='badge {q['band']}'>{q['band']} · {q['score']}"
                "</span></h3>"
                f"{flags_block(candidate)}"
                "</div>"
            )

        md.reset()
        right = (
            "<div class='card zh'><h3 style='text-transform:uppercase;"
            "letter-spacing:.08em;color:var(--muted)'>简历原文（中文）</h3>"
            f"{md.convert(body)}</div>"
        )
        blocks.append(
            f"<section id='{anchor}'><h2>{html.escape(title)}</h2>"
            f"<div class='grid'>{left}{right}</div></section>"
        )

    page = f"""<!doctype html>
<html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>简历解析核对页</title><style>{CSS}</style></head><body>
<header><h1>简历解析人工核对</h1>
<p>左：系统解析结果　右：简历中文原文　·　共 {len(sections)} 位候选人　·
　橙色标注＝需留意，红色＝机构未解析</p></header>
<nav>{''.join(nav)}</nav>
<main>{''.join(blocks)}</main>
<footer>Millennium BD 案例 · 解析核对工具 · 由 tools/build_review_page.py 生成</footer>
</body></html>"""

    OUTPUT.write_text(page, encoding="utf-8")
    print(f"Wrote {OUTPUT} ({OUTPUT.stat().st_size/1024:.0f} KB)")
    if unmatched:
        print(f"Unmatched sections: {', '.join(unmatched)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
