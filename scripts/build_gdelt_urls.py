"""Stream GDELT 1.0 daily events keeping article SOURCEURLs per UCDP unit-month
— the raw-text bridge for the modern-representation arm (and Phase-2 full-text
fetching).

Identical geographic join to build_gdelt_ucdp_join.py (nearest UCDP-unit
centroid by haversine, 200 km cap). Per (unit, month) we keep the top-K URLs by
NumArticles (deduped), which carry human-written headline slugs.

Output: data/interim/gdelt_ucdp_urls.parquet with columns
(unit, month, url, narts). Monthly checkpoints in _gdelt_urls_cache/ — safe to
kill and rerun.

Usage:
    python scripts/build_gdelt_urls.py --start 2013-04 --end 2024-12 --topk 12
"""

from __future__ import annotations

import argparse
import io
import os
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta

import numpy as np
import pandas as pd
import requests
from sklearn.neighbors import BallTree

from build_gdelt_ucdp_join import unit_centroids  # same centroid logic

BASE = "http://data.gdeltproject.org/events/{ymd}.export.CSV.zip"
CACHE = "data/interim/_gdelt_urls_cache"
# GDELT 1.0 column indices (58-col layout, validated)
I_ID, I_DATE, I_NARTS, I_LAT, I_LON, I_URL = 0, 1, 33, 53, 54, 57
USECOLS = [I_ID, I_DATE, I_NARTS, I_LAT, I_LON, I_URL]
NAMES = ["id", "date", "narts", "lat", "lon", "url"]
EARTH_KM = 6371.0


def fetch_day(d: date) -> pd.DataFrame | None:
    url = BASE.format(ymd=d.strftime("%Y%m%d"))
    try:
        r = requests.get(url, timeout=90)
        if r.status_code != 200:
            return None
        z = zipfile.ZipFile(io.BytesIO(r.content))
        df = pd.read_csv(
            z.open(z.namelist()[0]),
            sep="\t",
            header=None,
            usecols=USECOLS,
            names=NAMES,
            dtype=str,
            on_bad_lines="skip",
        )
    except Exception:  # noqa: BLE001
        return None
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    return df.dropna(subset=["lat", "lon", "url"])


def process_month(
    ym: str, days: list[date], tree, units, radius, workers, topk: int
) -> pd.DataFrame:
    frames = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fetch_day, d): d for d in days}
        for fut in as_completed(futs):
            df = fut.result()
            if df is None or df.empty:
                continue
            df["month"] = pd.to_datetime(
                df["date"], format="%Y%m%d", errors="coerce"
            ).dt.to_period("M")
            df = df.dropna(subset=["month"])
            ref = pd.Period(ym, freq="M")
            df = df[(df["month"] >= ref - 2) & (df["month"] <= ref)]
            if df.empty:
                continue
            rad = np.radians(df[["lat", "lon"]].values)
            dist, idx = tree.query(rad, k=1)
            df["unit"] = units[idx[:, 0]]
            df = df[dist[:, 0] * EARTH_KM <= radius]
            frames.append(df[["unit", "month", "url", "narts"]])
    if not frames:
        return pd.DataFrame()
    m = pd.concat(frames, ignore_index=True)
    m["narts"] = pd.to_numeric(m["narts"], errors="coerce").fillna(0)
    # dedupe URLs within unit-month (keep max narts), then top-K per unit-month
    m = (
        m.sort_values("narts", ascending=False)
        .drop_duplicates(subset=["unit", "month", "url"])
        .groupby(["unit", "month"], as_index=False)
        .head(topk)
    )
    m["month"] = m["month"].dt.to_timestamp()
    return m


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ged", default="data/raw/ucdp_ged_251.parquet")
    ap.add_argument("--start", default="2013-04")
    ap.add_argument("--end", default="2024-12")
    ap.add_argument("--radius", type=float, default=200.0)
    ap.add_argument("--topk", type=int, default=12)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--out", default="data/interim/gdelt_ucdp_urls.parquet")
    args = ap.parse_args()

    cent = unit_centroids(args.ged)
    tree = BallTree(np.radians(cent[["lat", "lon"]].values), metric="haversine")
    units = cent["unit"].values
    os.makedirs(CACHE, exist_ok=True)
    months = pd.period_range(args.start, args.end, freq="M")
    print(
        f"{len(cent)} centroids | top-{args.topk} URLs/unit-month | "
        f"{len(months)} months",
        flush=True,
    )

    for k, per in enumerate(months, 1):
        cpath = f"{CACHE}/{per.strftime('%Y%m')}.parquet"
        if os.path.exists(cpath):
            continue
        d0 = per.to_timestamp().date()
        d1 = (per + 1).to_timestamp().date() - timedelta(days=1)
        days = [d0 + timedelta(days=i) for i in range((d1 - d0).days + 1)]
        out = process_month(
            str(per), days, tree, units, args.radius, args.workers, args.topk
        )
        out.to_parquet(cpath, index=False)
        if k % 6 == 0 or k == len(months):
            print(f"  {k}/{len(months)} months (through {per})", flush=True)

    parts = [pd.read_parquet(f"{CACHE}/{p.strftime('%Y%m')}.parquet") for p in months]
    full = pd.concat([p for p in parts if len(p)], ignore_index=True)
    # re-apply top-K where a unit-month spans two caches
    full = (
        full.sort_values("narts", ascending=False)
        .drop_duplicates(subset=["unit", "month", "url"])
        .groupby(["unit", "month"], as_index=False)
        .head(args.topk)
    )
    full.to_parquet(args.out, index=False)
    print(
        f"\nwrote {len(full):,} URL rows | {full['unit'].nunique()} units "
        f"-> {args.out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
