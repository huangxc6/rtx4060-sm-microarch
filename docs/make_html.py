"""把 FINAL-REPORT.md 转成自包含 HTML（SVG 内联，无任何外部依赖）。

只支持本报告实际用到的 Markdown 子集：标题、段落、列表、表格、
粗体、行内代码、围栏代码、图片、分隔线、引用块。
"""
from __future__ import annotations

import html
import re
from pathlib import Path

DOCS = Path(__file__).resolve().parent
SRC = DOCS / "FINAL-REPORT.md"
OUT = DOCS / "final-report.html"

CSS = """
:root{--ink:#111827;--muted:#6b7280;--line:#e5e7eb;--bg:#ffffff;
      --accent:#2563eb;--soft:#f9fafb;--amber:#d97706}
*{box-sizing:border-box}
body{margin:0;background:#f3f4f6;color:var(--ink);
     font-family:"Segoe UI","Microsoft YaHei","PingFang SC",sans-serif;
     line-height:1.75;font-size:15px}
.page{max-width:1020px;margin:0 auto;background:var(--bg);
      padding:56px 64px 80px;box-shadow:0 1px 3px rgba(0,0,0,.08)}
h1{font-size:32px;line-height:1.3;margin:0 0 6px;letter-spacing:-.3px}
h2{font-size:22px;margin:52px 0 14px;padding-top:14px;border-top:1px solid var(--line)}
h3{font-size:17px;margin:30px 0 10px;color:#1f2937}
p{margin:12px 0}
a{color:var(--accent)}
code{font-family:Consolas,"Cascadia Mono",monospace;font-size:.9em;
     background:var(--soft);border:1px solid var(--line);border-radius:4px;
     padding:1px 5px}
pre{background:#0f172a;color:#e5e7eb;padding:16px 18px;border-radius:8px;
    overflow:auto;font-size:13px;line-height:1.6}
pre code{background:none;border:0;color:inherit;padding:0}
table{border-collapse:collapse;width:100%;margin:18px 0;font-size:14px}
th,td{border:1px solid var(--line);padding:8px 11px;vertical-align:top}
th{background:var(--soft);font-weight:600;text-align:left}
tr:nth-child(even) td{background:#fcfcfd}
blockquote{margin:16px 0;padding:10px 16px;border-left:3px solid var(--accent);
           background:#eff6ff;color:#1e3a8a}
blockquote p{margin:6px 0}
hr{border:0;border-top:1px solid var(--line);margin:36px 0}
ul,ol{margin:12px 0;padding-left:26px}
li{margin:5px 0}
figure{margin:26px 0;padding:8px;border:1px solid var(--line);border-radius:10px;
       background:#fff}
figure svg{display:block;width:100%;height:auto}
figcaption{font-size:12.5px;color:var(--muted);margin-top:6px;padding:0 6px}
.table-wrap{overflow-x:auto}
@media print{
  body{background:#fff}
  .page{box-shadow:none;padding:0;max-width:100%}
  h2{page-break-after:avoid}
  figure{page-break-inside:avoid}
}
"""


def inline(text: str) -> str:
    """处理 **粗体** 与 `代码`，并转义其余字符。

    粗体内部允许嵌套行内代码（``**`wgmma` 不存在**``），所以递归处理内层，
    否则反引号会原样漏到页面上（踩过）。
    """
    parts: list[str] = []
    for seg in re.split(r"(`[^`]*`|\*\*[^*]+\*\*)", text):
        if seg.startswith("`") and seg.endswith("`") and len(seg) > 1:
            parts.append(f"<code>{html.escape(seg[1:-1])}</code>")
        elif seg.startswith("**") and seg.endswith("**") and len(seg) > 4:
            parts.append(f"<strong>{inline(seg[2:-2])}</strong>")
        else:
            parts.append(html.escape(seg))
    return "".join(parts)


def figure(path: Path, alt: str) -> str:
    svg = path.read_text(encoding="utf-8")
    svg = re.sub(r'<\?xml[^>]*\?>', "", svg)
    svg = svg.replace('width="', 'data-w="', 1).replace('height="', 'data-h="', 1)
    return (f'<figure>{svg}'
            f'<figcaption>{html.escape(alt)}</figcaption></figure>')


def convert(md: str) -> str:
    lines = md.splitlines()
    out: list[str] = []
    i = 0
    n = len(lines)
    while i < n:
        ln = lines[i]
        s = ln.rstrip()

        if not s.strip():
            i += 1
            continue

        # 围栏代码
        if s.startswith("```"):
            i += 1
            buf = []
            while i < n and not lines[i].startswith("```"):
                buf.append(html.escape(lines[i]))
                i += 1
            i += 1
            out.append("<pre><code>" + "\n".join(buf) + "</code></pre>")
            continue

        # 分隔线（紧跟标题时不输出，避免和 h2 的上边框叠成双线）
        if s.strip() == "---":
            j = i + 1
            while j < n and not lines[j].strip():
                j += 1
            if j < n and not lines[j].lstrip().startswith("#"):
                out.append("<hr/>")
            i += 1
            continue

        # 标题
        m = re.match(r"^(#{1,3})\s+(.*)$", s)
        if m:
            lvl = len(m.group(1))
            out.append(f"<h{lvl}>{inline(m.group(2))}</h{lvl}>")
            i += 1
            continue

        # 图片
        m = re.match(r"^!\[(.*?)\]\((.*?)\)\s*$", s)
        if m:
            alt, rel = m.group(1), m.group(2)
            p = (DOCS / rel).resolve()
            if p.exists():
                out.append(figure(p, alt))
            else:
                out.append(f'<p><em>缺图：{html.escape(rel)}</em></p>')
            i += 1
            continue

        # 表格
        if s.startswith("|") and i + 1 < n and re.match(r"^\|[\s:\-|]+\|$", lines[i + 1].strip()):
            header = [c.strip() for c in s.strip().strip("|").split("|")]
            i += 2
            rows = []
            while i < n and lines[i].strip().startswith("|"):
                rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            th = "".join(f"<th>{inline(c)}</th>" for c in header)
            body = "".join(
                "<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>"
                for r in rows)
            out.append(f'<div class="table-wrap"><table><thead><tr>{th}</tr></thead>'
                       f"<tbody>{body}</tbody></table></div>")
            continue

        # 引用块
        if s.startswith(">"):
            buf = []
            while i < n and lines[i].startswith(">"):
                buf.append(lines[i].lstrip(">").strip())
                i += 1
            out.append("<blockquote><p>" + inline(" ".join(buf)) + "</p></blockquote>")
            continue

        # 列表
        if re.match(r"^[-*]\s+", s) or re.match(r"^\d+\.\s+", s):
            ordered = bool(re.match(r"^\d+\.\s+", s))
            tag = "ol" if ordered else "ul"
            items: list[str] = []
            while i < n:
                line = lines[i]
                stripped = line.strip()
                if not stripped:
                    break
                m2 = re.match(r"^([-*]|\d+\.)\s+(.*)$", stripped)
                if m2:
                    items.append(m2.group(2))
                    i += 1
                elif line[:1] in (" ", "\t") and items:
                    # 续行：接到上一项（否则每项都会被拆成独立列表，编号全变 1）
                    items[-1] += stripped
                    i += 1
                else:
                    break
            out.append(f"<{tag}>" + "".join(f"<li>{inline(x)}</li>" for x in items)
                       + f"</{tag}>")
            continue

        # 段落（合并连续行）
        buf = [s]
        i += 1
        while (i < n and lines[i].strip() and
               not re.match(r"^(#{1,3}\s|```|\||>|[-*]\s|\d+\.\s|---\s*$|!\[)",
                            lines[i].strip())):
            buf.append(lines[i].strip())
            i += 1
        out.append("<p>" + inline("".join(buf)) + "</p>")

    return "\n".join(out)


def main():
    md = SRC.read_text(encoding="utf-8")
    body = convert(md)
    doc = (f'<!DOCTYPE html>\n<html lang="zh-CN"><head><meta charset="utf-8"/>'
           f'<meta name="viewport" content="width=device-width,initial-scale=1"/>'
           f"<title>RTX 4060 SM 微架构逆向（发布版）</title>"
           f"<style>{CSS}</style></head><body><div class=\"page\">{body}</div></body></html>")
    OUT.write_text(doc, encoding="utf-8")
    n_svg = doc.count("<svg")
    n_tab = doc.count("<table")
    n_h2 = doc.count("<h2")
    print(f"写出 {OUT.name}: {len(doc.encode('utf-8'))/1024:.1f} KB, "
          f"{n_svg} 张内联 SVG, {n_tab} 个表格, {n_h2} 个二级标题")


if __name__ == "__main__":
    main()
