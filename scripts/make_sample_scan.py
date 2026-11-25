"""Render a sample invoice page (same values and errors as the challenge invoice) to data/sample_scan.png.

    python scripts/make_sample_scan.py [path-to-chrome-or-edge]

Used by the dashboard's "Try the sample scan" button, so the scan feature can be tried
without a real document. Values come from data/invoice_WB-2026-11-0457.json.
"""

from __future__ import annotations

import json
import shutil
import subprocess  # nosec B404
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BROWSERS = (
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    "chromium",
    "google-chrome",
)


def page(inv: dict) -> str:  # type: ignore[type-arg]
    rows = "".join(
        f"<tr><td>{ln['no']}</td><td>{ln['desc_en']}<br><span dir='rtl'>{ln['desc_ar']}</span></td>"
        f"<td>{ln['unit']}</td><td>{ln['ordered']}</td><td>{ln['delivered']}</td><td>{ln['unit_price']}</td>"
        f"<td>{ln['amount']}</td></tr>"
        for group in inv["groups"]
        for ln in group["lines"]
    )
    t = inv["totals"]
    return f"""<!doctype html><meta charset="utf-8"><style>
body{{font:20px Arial,sans-serif;margin:40px;color:#111;background:#fff}} h1{{font-size:30px;margin:0}}
table{{border-collapse:collapse;width:100%;margin:18px 0}} td,th{{border:1px solid #333;padding:8px;text-align:left}}
.r{{text-align:right}} .box{{display:flex;justify-content:space-between}}</style>
<div class="box"><div><h1>{inv["supplier"]["name_en"]}</h1>{inv["supplier"]["address"]}<br>TRN {inv["supplier"]["trn"]}</div>
<div dir="rtl"><h1>{inv["supplier"]["name_ar"]}</h1>فاتورة ضريبية</div></div>
<p>Tax Invoice No. {inv["invoice_no"]} &nbsp; Date {inv["date"]} &nbsp; PO {inv["po_no"]}</p>
<table><tr><th>#</th><th>Description</th><th>Unit</th><th>Ordered</th><th>Delivered</th><th>Unit price</th><th>Amount AED</th></tr>{rows}</table>
<p>Subtotal Site A {inv["groups"][0]["subtotal"]}<br>Subtotal Site B {inv["groups"][1]["subtotal"]}<br>
Total before discount {t["total_before_discount"]}<br>{t["discount_label"]} {t["discount"]}<br>
Taxable amount {t["taxable_amount"]}<br>VAT 5% {t["vat"]}<br><b>Grand total {t["grand_total"]}</b></p>
<p>D3 delivery {inv["deliveries"][2]["planned_date"]} &nbsp; {inv["terms_en"]}</p>"""


def main() -> None:
    browser = next((b for b in (sys.argv[1:] or BROWSERS) if Path(b).exists() or shutil.which(b)), None)
    if not browser:
        raise SystemExit("no Chrome/Edge found: pass its path")
    inv = json.loads((ROOT / "data" / "invoice_WB-2026-11-0457.json").read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory() as tmp:
        html = Path(tmp) / "invoice.html"
        html.write_text(page(inv), encoding="utf-8")
        out = ROOT / "data" / "sample_scan.png"
        subprocess.run(  # noqa: S603  # nosec B603
            [
                browser,
                "--headless=new",
                "--disable-gpu",
                "--hide-scrollbars",
                "--window-size=1300,1150",
                f"--screenshot={out}",
                html.as_uri(),
            ],
            check=True,
            capture_output=True,
        )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
