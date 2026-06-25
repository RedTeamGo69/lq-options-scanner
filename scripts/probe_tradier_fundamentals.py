"""
Probe Tradier's beta fundamentals API to confirm the live JSON schema.

The app derives dividend yield, ex-dividend date, and next earnings date from
Tradier's Morningstar-sourced fundamentals endpoints. Those endpoints are
entitlement-gated (a funded Tradier Brokerage account) and the payload is deeply
nested, so this script lets you confirm — with your own key — that real data
comes back and that the field paths the parser expects are present.

Usage:
    TRADIER_API_KEY=your_token python scripts/probe_tradier_fundamentals.py
    # optional: pass symbols to probe (default: AAPL KO TSLA SPY)
    TRADIER_API_KEY=... python scripts/probe_tradier_fundamentals.py AAPL MSFT

It prints, per symbol:
  - whether cash_dividends / corporate_calendars rows were found
  - the parsed result the app would use (ex-div, trailing dividend, earnings)
  - a trimmed raw sample so you can eyeball the actual field names

Exit code is non-zero if no fundamentals data is returned for any symbol, which
usually means the token lacks the fundamentals entitlement.
"""

import json
import os
import sys

import requests

BASE = os.getenv("TRADIER_FUNDAMENTALS_BASE_URL", "https://api.tradier.com/beta").rstrip("/")
TOKEN = os.environ.get("TRADIER_API_KEY")

if not TOKEN:
    print("ERROR: set TRADIER_API_KEY in the environment first.", file=sys.stderr)
    sys.exit(2)

HEADERS = {"Authorization": f"Bearer {TOKEN}", "Accept": "application/json"}


def get(path: str, symbols: str):
    resp = requests.get(f"{BASE}{path}", headers=HEADERS, params={"symbols": symbols}, timeout=15)
    resp.raise_for_status()
    return resp.json()


def collect_rows(payload, table_key):
    """Recursively gather rows under table_key anywhere in the nested payload."""
    rows = []

    def walk(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k == table_key and isinstance(v, list):
                    rows.extend(x for x in v if isinstance(x, dict))
                else:
                    walk(v)
        elif isinstance(obj, list):
            for item in obj:
                walk(item)

    walk(payload)
    return rows


def main():
    symbols = sys.argv[1:] or ["AAPL", "KO", "TSLA", "SPY"]
    any_data = False

    for sym in symbols:
        print(f"\n{'=' * 60}\n{sym}\n{'=' * 60}")

        try:
            div_payload = get("/markets/fundamentals/dividends", sym)
            div_rows = collect_rows(div_payload, "cash_dividends")
            print(f"cash_dividends rows: {len(div_rows)}")
            if div_rows:
                any_data = True
                print("  sample:", json.dumps(div_rows[0], indent=2)[:500])
                ex_dates = sorted(
                    {r.get("ex_date") for r in div_rows if r.get("ex_date")}
                )
                print(f"  ex_date values (deduped): {ex_dates[-6:]}")
        except Exception as e:  # noqa: BLE001
            print(f"  dividends error: {e}")

        try:
            cal_payload = get("/markets/fundamentals/calendars", sym)
            cal_rows = collect_rows(cal_payload, "corporate_calendars")
            print(f"corporate_calendars rows: {len(cal_rows)}")
            if cal_rows:
                any_data = True
                earnings = [
                    r for r in cal_rows
                    if isinstance(r.get("event"), str) and "earnings" in r["event"].lower()
                ]
                print(f"  earnings rows: {len(earnings)}")
                if earnings:
                    print("  sample:", json.dumps(earnings[0], indent=2)[:500])
        except Exception as e:  # noqa: BLE001
            print(f"  calendars error: {e}")

    if not any_data:
        print(
            "\nNo fundamentals data returned for any symbol — your token likely "
            "lacks the fundamentals entitlement (needs a funded brokerage account).",
            file=sys.stderr,
        )
        sys.exit(1)

    print("\nOK: fundamentals data is available for at least one symbol.")


if __name__ == "__main__":
    main()
