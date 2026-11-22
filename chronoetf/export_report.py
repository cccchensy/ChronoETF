"""Export the Markdown report using Python Markdown and Edge's print renderer.

Optional publishing dependencies: pip install markdown playwright
Usage: python chronoetf/export_report.py --output /path/to/ChronoETF.pdf
"""
import argparse
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import quote

import markdown
from playwright.sync_api import sync_playwright


def export(source, output, channel):
    root = source.parent
    css = (Path(__file__).with_name("report_print.css")).read_text(encoding="utf-8")
    content = markdown.markdown(source.read_text(encoding="utf-8"), extensions=["tables", "fenced_code"])
    html = (f'<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">'
            f'<base href="{root.as_uri()}/"><style>{css}</style></head><body>{content}</body></html>')
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="chronoetf-print-") as tmp:
        path = Path(tmp) / "report.html"
        path.write_text(html, encoding="utf-8")
        with sync_playwright() as p:
            browser = p.chromium.launch(channel=channel, headless=True)
            try:
                page = browser.new_page()
                page.goto(path.as_uri())
                page.evaluate("document.fonts.ready")
                if not page.locator("img").evaluate_all("xs=>xs.every(x=>x.complete&&x.naturalWidth>0)"):
                    raise RuntimeError("A report image failed to load")
                # Published source links remain useful when the PDF is shared.
                for link in page.locator("a").all():
                    href = link.get_attribute("href")
                    if href and not href.startswith(("http:", "https:", "#")):
                        url = "https://github.com/cccchensy/ChronoETF/blob/main/" + quote(href, safe="/#")
                        link.evaluate("(a,url)=>a.href=url", url)
                page.pdf(path=str(output), print_background=True, prefer_css_page_size=True,
                         display_header_footer=True,
                         header_template='<div style="font:9px SimSun,serif;color:#526174;margin:0 17mm">ChronoETF 研究报告</div>',
                         footer_template='<div style="font:9px SimSun,serif;color:#526174;width:100%;margin:0 17mm;display:flex;justify-content:space-between"><span>样本截至 2026-10-30</span><span><span class="pageNumber"></span> / <span class="totalPages"></span></span></div>')
            finally:
                browser.close()
    print(f"Exported {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[1] / "ChronoETF.md")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--browser-channel", default="msedge")
    args = parser.parse_args()
    export(args.source.resolve(), args.output.resolve(), args.browser_channel)
