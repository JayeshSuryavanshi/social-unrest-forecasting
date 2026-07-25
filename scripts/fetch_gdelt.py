"""Fetch GDELT 2.0 event data (no API key required) for a date range.

GDELT publishes a new Events file every 15 minutes at
http://data.gdeltproject.org/gdeltv2/ . The master file list enumerates every
file since Feb 2015. We stream the ones in [start, end], keep unrest-relevant
CAMEO root codes, and write a tidy parquet/csv.

Usage:
    python scripts/fetch_gdelt.py --start 2026-07-01 --end 2026-07-24 \
        --out data/raw/gdelt_events_jul2026.parquet

This is the modern, no-registration replacement for the retired Bing News API
that the 2022 project used. It is NOT a substitute for ACLED's hand-coded
ground truth -- treat GDELT as a high-recall/low-precision news signal.
"""

from __future__ import annotations

import argparse
import io
import zipfile
from datetime import datetime, timezone

import pandas as pd
import requests

GDELT_COLS = [
    "GlobalEventID",
    "Day",
    "MonthYear",
    "Year",
    "FractionDate",
    "Actor1Code",
    "Actor1Name",
    "Actor1CountryCode",
    "Actor1KnownGroupCode",
    "Actor1EthnicCode",
    "Actor1Religion1Code",
    "Actor1Religion2Code",
    "Actor1Type1Code",
    "Actor1Type2Code",
    "Actor1Type3Code",
    "Actor2Code",
    "Actor2Name",
    "Actor2CountryCode",
    "Actor2KnownGroupCode",
    "Actor2EthnicCode",
    "Actor2Religion1Code",
    "Actor2Religion2Code",
    "Actor2Type1Code",
    "Actor2Type2Code",
    "Actor2Type3Code",
    "IsRootEvent",
    "EventCode",
    "EventBaseCode",
    "EventRootCode",
    "QuadClass",
    "GoldsteinScale",
    "NumMentions",
    "NumSources",
    "NumArticles",
    "AvgTone",
    "Actor1Geo_Type",
    "Actor1Geo_Fullname",
    "Actor1Geo_CountryCode",
    "Actor1Geo_ADM1Code",
    "Actor1Geo_ADM2Code",
    "Actor1Geo_Lat",
    "Actor1Geo_Long",
    "Actor1Geo_FeatureID",
    "Actor2Geo_Type",
    "Actor2Geo_Fullname",
    "Actor2Geo_CountryCode",
    "Actor2Geo_ADM1Code",
    "Actor2Geo_ADM2Code",
    "Actor2Geo_Lat",
    "Actor2Geo_Long",
    "Actor2Geo_FeatureID",
    "ActionGeo_Type",
    "ActionGeo_Fullname",
    "ActionGeo_CountryCode",
    "ActionGeo_ADM1Code",
    "ActionGeo_ADM2Code",
    "ActionGeo_Lat",
    "ActionGeo_Long",
    "ActionGeo_FeatureID",
    "DATEADDED",
    "SOURCEURL",
]
assert len(GDELT_COLS) == 61

# CAMEO root codes that map to social unrest / political violence.
UNREST_ROOT_CODES = {
    "14",
    "17",
    "18",
    "19",
    "20",
}  # protest, coerce, assault, fight, mass violence

MASTERLIST = "http://data.gdeltproject.org/gdeltv2/masterfilelist.txt"
KEEP = [
    "GlobalEventID",
    "Day",
    "Actor1Name",
    "Actor1CountryCode",
    "Actor2Name",
    "EventCode",
    "EventRootCode",
    "QuadClass",
    "GoldsteinScale",
    "NumMentions",
    "NumSources",
    "NumArticles",
    "AvgTone",
    "ActionGeo_Fullname",
    "ActionGeo_CountryCode",
    "ActionGeo_ADM1Code",
    "ActionGeo_Lat",
    "ActionGeo_Long",
    "SOURCEURL",
]


def export_urls_in_range(start: datetime, end: datetime) -> list[str]:
    lines = requests.get(MASTERLIST, timeout=120).text.splitlines()
    urls = []
    for line in lines:
        parts = line.split()
        if len(parts) != 3 or not parts[2].endswith("export.CSV.zip"):
            continue
        stamp = parts[2].split("/")[-1][:14]  # YYYYMMDDHHMMSS
        try:
            ts = datetime.strptime(stamp, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        if start <= ts <= end:
            urls.append(parts[2])
    return urls


def fetch_one(url: str, unrest_only: bool) -> pd.DataFrame | None:
    try:
        blob = requests.get(url, timeout=90).content
        z = zipfile.ZipFile(io.BytesIO(blob))
        df = pd.read_csv(
            z.open(z.namelist()[0]), sep="\t", header=None, names=GDELT_COLS, dtype=str
        )
    except Exception as exc:  # noqa: BLE001 - one bad 15-min file shouldn't kill the run
        print(f"  ! skip {url.split('/')[-1]}: {exc}")
        return None
    if unrest_only:
        df = df[df.EventRootCode.isin(UNREST_ROOT_CODES)]
    return df[KEEP]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", required=True, help="UTC date YYYY-MM-DD (inclusive)")
    ap.add_argument("--end", required=True, help="UTC date YYYY-MM-DD (inclusive)")
    ap.add_argument("--out", required=True, help="output .parquet or .csv path")
    ap.add_argument(
        "--all-events",
        action="store_true",
        help="keep every event, not just unrest-relevant CAMEO codes",
    )
    args = ap.parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end = datetime.strptime(args.end, "%Y-%m-%d").replace(
        hour=23, minute=59, second=59, tzinfo=timezone.utc
    )

    urls = export_urls_in_range(start, end)
    print(
        f"{len(urls)} GDELT 15-min export files in range "
        f"({len(urls) * 15 / 60 / 24:.1f} days of coverage)"
    )

    frames, n = [], len(urls)
    for i, url in enumerate(urls, 1):
        df = fetch_one(url, unrest_only=not args.all_events)
        if df is not None and len(df):
            frames.append(df)
        if i % 50 == 0 or i == n:
            print(f"  {i}/{n} files, {sum(map(len, frames)):,} rows kept")

    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=KEEP)
    out["Day"] = pd.to_datetime(out["Day"], format="%Y%m%d", errors="coerce")
    if args.out.endswith(".parquet"):
        out.to_parquet(args.out, index=False)
    else:
        out.to_csv(args.out, index=False)
    print(f"\nwrote {len(out):,} rows -> {args.out}")


if __name__ == "__main__":
    main()
