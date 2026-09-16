"""Render a markdown report to a self-contained HTML file (images embedded as data URIs).

Usage: python render_html.py report/findings.md report/findings.html
"""

import base64
import mimetypes
import re
import sys
from pathlib import Path


def _embed(md: str, base: Path) -> str:
    def repl(m):
        alt, src = m.group(1), m.group(2)
        p = (base / src).resolve()
        if p.exists():
            mime = mimetypes.guess_type(p.name)[0] or "image/png"
            data = base64.b64encode(p.read_bytes()).decode()
            return f"![{alt}](data:{mime};base64,{data})"
        return m.group(0)

    return re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", repl, md)


def main(src: str, dst: str) -> None:
    try:
        import markdown  # type: ignore
    except ImportError:
        sys.exit("pip install markdown")
    p = Path(src)
    body = markdown.markdown(_embed(p.read_text(), p.parent), extensions=["tables", "fenced_code"])
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>{p.stem}</title>
<style>body{{max-width:880px;margin:40px auto;font:16px/1.55 system-ui,sans-serif;color:#1a1a1a;padding:0 16px}}
img{{max-width:100%}}table{{border-collapse:collapse}}td,th{{border:1px solid #ddd;padding:6px 10px}}</style>
</head><body>{body}</body></html>"""
    Path(dst).write_text(html)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
