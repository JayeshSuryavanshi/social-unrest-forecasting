"""Fetch ACLED event data via the 2026 OAuth2 API.

ACLED changed its access model in 2025. The legacy `?email=...&key=...` query
scheme (what the 2022 project used) was shut off after 2025-09-15 and now 401s.
The current flow is OAuth2: exchange your myACLED login for a bearer token, then
call the data endpoint with an Authorization header.

Register (free) at https://acleddata.com/ ; an institutional email gets a higher
access tier than a personal one (raw-event API access is gated to Research tier+).

Credentials are read from the environment -- never hardcode them, and never
commit ACLED data (their terms forbid redistribution; keep it gitignored):
    export ACLED_EMAIL='you@university.edu'
    export ACLED_PASSWORD='...'

Usage:
    python scripts/fetch_acled.py --country Nigeria --start 2018-01-01 \
        --end 2026-06-30 --out data/raw/acled_nigeria.csv

If this script's endpoints ever drift, the official `acled` PyPI package and the
R `acledR` package wrap the same OAuth dance and are the robust fallback.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import pandas as pd
import requests

TOKEN_URL = "https://acleddata.com/oauth/token"
READ_URL = "https://acleddata.com/api/acled/read"
PAGE_SIZE = 5000  # ACLED returns ~5000 rows/page; paging doesn't count against limits

# The fields Task 1 (classification) and a forecasting revival actually need.
DEFAULT_FIELDS = "|".join(
    [
        "event_id_cnty",
        "event_date",
        "year",
        "time_precision",
        "disorder_type",
        "event_type",
        "sub_event_type",
        "actor1",
        "actor2",
        "inter1",
        "inter2",
        "interaction",
        "civilian_targeting",
        "iso",
        "region",
        "country",
        "admin1",
        "admin2",
        "location",
        "latitude",
        "longitude",
        "geo_precision",
        "source",
        "notes",
        "fatalities",
        "tags",
        "timestamp",
    ]
)


def get_token(email: str, password: str) -> str:
    resp = requests.post(
        TOKEN_URL,
        data={
            "username": email,
            "password": password,
            "grant_type": "password",
            "client_id": "acled",
            "scope": "authenticated",
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=60,
    )
    if resp.status_code != 200:
        sys.exit(
            f"Token request failed ({resp.status_code}): {resp.text[:300]}\n"
            "Check ACLED_EMAIL/ACLED_PASSWORD and that your account has API access."
        )
    return resp.json()["access_token"]


def fetch(token: str, country: str, start: str, end: str) -> pd.DataFrame:
    headers = {"Authorization": f"Bearer {token}"}
    frames, page = [], 1
    while True:
        params = {
            "_format": "csv",
            "country": country,
            "event_date": f"{start}|{end}",
            "event_date_where": "BETWEEN",
            "fields": DEFAULT_FIELDS,
            "limit": PAGE_SIZE,
            "page": page,
        }
        r = requests.get(READ_URL, headers=headers, params=params, timeout=120)
        if r.status_code != 200:
            sys.exit(f"Read failed on page {page} ({r.status_code}): {r.text[:300]}")
        text = r.text.strip()
        if not text or text.count("\n") < 1:  # header-only or empty => done
            break
        from io import StringIO

        chunk = pd.read_csv(StringIO(text), dtype=str)
        if chunk.empty:
            break
        frames.append(chunk)
        print(f"  page {page}: {len(chunk)} rows (total {sum(map(len, frames)):,})")
        if len(chunk) < PAGE_SIZE:
            break
        page += 1
        time.sleep(1)  # be polite
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--country", required=True, help="ACLED country name, e.g. Nigeria")
    ap.add_argument("--start", required=True, help="YYYY-MM-DD")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD")
    ap.add_argument("--out", required=True, help="output .csv path")
    args = ap.parse_args()

    email = os.environ.get("ACLED_EMAIL")
    password = os.environ.get("ACLED_PASSWORD")
    if not email or not password:
        sys.exit(
            "Set ACLED_EMAIL and ACLED_PASSWORD env vars first (see module docstring)."
        )

    print(f"Authenticating as {email} ...")
    token = get_token(email, password)
    print(f"Fetching ACLED: {args.country} {args.start}..{args.end}")
    df = fetch(token, args.country, args.start, args.end)
    if df.empty:
        sys.exit("No rows returned. Check country name / date range / access tier.")
    df.to_csv(args.out, index=False)
    print(
        f"\nwrote {len(df):,} rows -> {args.out}  (DO NOT commit this file — ACLED terms)"
    )


if __name__ == "__main__":
    main()
