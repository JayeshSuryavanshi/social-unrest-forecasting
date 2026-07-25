"""Stream GDELT 1.0 daily event files, filter to one country, and build a
compact (date x admin1) event-count panel. No API key. Disk-light: raw daily
files are streamed and discarded; only the filtered aggregate is persisted.

GDELT 1.0 daily export = one file per day (whole world), ~3-6 MB zipped, at
http://data.gdeltproject.org/events/YYYYMMDD.export.CSV.zip . We keep only the
target country's rows, dedupe by GlobalEventID (the same event recurs across
daily files as it accrues mentions), and aggregate by (event_date, admin1).

Ground-truth caveat: GDELT is machine-coded news, high-recall/low-precision and
NOT ACLED. It has no fatality count, so the forecasting label built downstream
is a GDELT PROXY for unrest (material-conflict event volume), used to prove the
pipeline + evaluation are honest. Swap in ACLED labels for the real thing.

Usage:
    python scripts/build_gdelt_panel.py --country NI \
        --start 2025-01-01 --end 2026-06-30 \
        --out data/interim/gdelt_NI_daily_panel.parquet
"""

from __future__ import annotations

import argparse
import io
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta

import pandas as pd
import requests

BASE = "http://data.gdeltproject.org/events/{ymd}.export.CSV.zip"

# GDELT 1.0 Events: 58 columns. Indices we use:
I_ID, I_DATE, I_ROOT, I_QUAD = 0, 1, 28, 29
I_GOLD, I_NARTS, I_TONE = 30, 33, 34
I_ACTION_CTRY, I_ACTION_ADM1 = 51, 52

# CAMEO root codes for unrest / political violence.
PROTEST, COERCE, ASSAULT, FIGHT, MASSVIO = "14", "17", "18", "19", "20"
USE_COLS = [
    I_ID,
    I_DATE,
    I_ROOT,
    I_QUAD,
    I_GOLD,
    I_NARTS,
    I_TONE,
    I_ACTION_CTRY,
    I_ACTION_ADM1,
]
NAMES = ["id", "date", "root", "quad", "gold", "narts", "tone", "ctry", "adm1"]


def daterange(start: date, end: date):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def fetch_day(d: date, countries: set[str]) -> pd.DataFrame | None:
    url = BASE.format(ymd=d.strftime("%Y%m%d"))
    try:
        r = requests.get(url, timeout=90)
        if r.status_code != 200:
            return None  # some early days are missing; skip quietly
        z = zipfile.ZipFile(io.BytesIO(r.content))
        df = pd.read_csv(
            z.open(z.namelist()[0]),
            sep="\t",
            header=None,
            usecols=USE_COLS,
            names=NAMES,
            dtype=str,
            on_bad_lines="skip",
        )
    except Exception:
        return None
    return df[df["ctry"].isin(countries)].copy()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--country",
        default="NI",
        help="GDELT FIPS country code(s), comma-separated (Nigeria=NI, Kenya=KE)",
    )
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    countries = {c.strip() for c in args.country.split(",")}
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    days = list(daterange(start, end))
    print(
        f"streaming {len(days)} GDELT daily files for countries={sorted(countries)} "
        f"({start}..{end}) with {args.workers} workers"
    )

    frames, done, missing = [], 0, 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(fetch_day, d, countries): d for d in days}
        for fut in as_completed(futs):
            df = fut.result()
            done += 1
            if df is None or df.empty:
                missing += 1
            else:
                frames.append(df)
            if done % 60 == 0 or done == len(days):
                kept = sum(map(len, frames))
                print(
                    f"  {done}/{len(days)} files | {kept:,} rows kept | {missing} missing/empty"
                )

    raw = pd.concat(frames, ignore_index=True)
    before = len(raw)
    raw = raw.drop_duplicates(subset="id")  # same event recurs across daily files
    print(f"deduped {before:,} -> {len(raw):,} unique events")

    raw["date"] = pd.to_datetime(raw["date"], format="%Y%m%d", errors="coerce")
    for c in ("gold", "narts", "tone"):
        raw[c] = pd.to_numeric(raw[c], errors="coerce")
    raw = raw.dropna(subset=["date", "adm1"])
    raw = raw[raw["adm1"].str.len() >= 4]  # keep province-coded rows only
    raw["country"] = raw["adm1"].str[:2]  # FIPS country prefix, for per-country slicing

    g = raw.groupby(["country", "date", "adm1"])
    panel = g.agg(
        n_events=("id", "size"),
        n_protest=("root", lambda s: (s == PROTEST).sum()),
        n_assault=("root", lambda s: (s == ASSAULT).sum()),
        n_fight=("root", lambda s: (s == FIGHT).sum()),
        n_massvio=("root", lambda s: (s == MASSVIO).sum()),
        n_matconf=(
            "quad",
            lambda s: (s == "4").sum(),
        ),  # QuadClass 4 = material conflict
        sum_articles=("narts", "sum"),
        mean_tone=("tone", "mean"),
        mean_goldstein=("gold", "mean"),
    ).reset_index()
    panel["n_violence"] = panel[["n_assault", "n_fight", "n_massvio"]].sum(axis=1)

    panel.to_parquet(args.out, index=False)
    print(
        f"\nwrote daily panel: {len(panel):,} (date,admin1) rows, "
        f"{panel['adm1'].nunique()} admin1 units -> {args.out}"
    )
    print("\nper-country: admin1 units | total events | protest | violence")
    for cc, sub in panel.groupby("country"):
        print(
            f"  {cc}: {sub['adm1'].nunique():3d} units | {int(sub['n_events'].sum()):>8,} "
            f"| {int(sub['n_protest'].sum()):>6,} | {int(sub['n_violence'].sum()):>6,}"
        )


if __name__ == "__main__":
    main()
