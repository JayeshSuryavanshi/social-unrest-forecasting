"""Stream GDELT 1.0 daily events (2013-04 onward) and aggregate machine-coded
news signals per UCDP unit x month — the text family for the multi-country
keystone experiment.

Join design (no crosswalks): each UCDP unit's centroid = median lat/lon of its
own GED events; each GDELT event (ActionGeo lat/lon) is assigned to the nearest
centroid by haversine BallTree, dropped if farther than --radius km. Border
mis-assignments are possible and noted as a limitation.

Signals per (unit, month), prefixed txt_ so the harness lags them as the text
family: txt_n_events, txt_n_protest, txt_n_matconf, txt_n_verbconf,
txt_sum_articles, txt_mean_tone, txt_mean_goldstein.

Robustness: processes files month-by-month with a parquet checkpoint per month
(data/interim/_gdelt_ucdp_cache/YYYYMM.parquet); safe to kill and rerun —
completed months are skipped. Within-month dedup by GlobalEventID.

Usage:
    python scripts/build_gdelt_ucdp_join.py --start 2013-04 --end 2024-12 \
        --out data/interim/gdelt_ucdp_month.parquet
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

BASE = "http://data.gdeltproject.org/events/{ymd}.export.CSV.zip"
CACHE = "data/interim/_gdelt_ucdp_cache"
# GDELT 1.0 export column indices (58-col layout, validated previously)
I_ID, I_DATE, I_ROOT, I_QUAD, I_GOLD = 0, 1, 28, 29, 30
I_NARTS, I_TONE, I_LAT, I_LON = 33, 34, 53, 54
USECOLS = [I_ID, I_DATE, I_ROOT, I_QUAD, I_GOLD, I_NARTS, I_TONE, I_LAT, I_LON]
NAMES = ["id", "date", "root", "quad", "gold", "narts", "tone", "lat", "lon"]
EARTH_KM = 6371.0


def unit_centroids(ged_path: str) -> pd.DataFrame:
    g = pd.read_parquet(ged_path).dropna(subset=["adm_1", "latitude", "longitude"])
    g["unit"] = g["country"] + " | " + g["adm_1"].str.strip()
    c = (
        g.groupby("unit")
        .agg(lat=("latitude", "median"), lon=("longitude", "median"))
        .reset_index()
    )
    return c


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
    except Exception:  # noqa: BLE001 - one bad day must not kill an overnight run
        return None
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    return df.dropna(subset=["lat", "lon"])


def process_month(
    ym: str,
    days: list[date],
    tree: BallTree,
    units: np.ndarray,
    radius_km: float,
    workers: int,
) -> pd.DataFrame:
    frames = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fetch_day, d): d for d in days}
        for fut in as_completed(futs):
            df = fut.result()
            if df is None or df.empty:
                continue
            # sane event-date window: keep SQLDATE within ~2 months of the file month
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
            df["km"] = dist[:, 0] * EARTH_KM
            frames.append(df[df["km"] <= radius_km])
    if not frames:
        return pd.DataFrame()
    m = pd.concat(frames, ignore_index=True).drop_duplicates(subset="id")
    for c in ("gold", "narts", "tone"):
        m[c] = pd.to_numeric(m[c], errors="coerce")
    g = m.groupby(["unit", "month"])
    out = g.agg(
        txt_n_events=("id", "size"),
        txt_n_protest=("root", lambda s: (s == "14").sum()),
        txt_n_matconf=("quad", lambda s: (s == "4").sum()),
        txt_n_verbconf=("quad", lambda s: (s == "3").sum()),
        txt_sum_articles=("narts", "sum"),
        txt_mean_tone=("tone", "mean"),
        txt_mean_goldstein=("gold", "mean"),
    ).reset_index()
    out["month"] = out["month"].dt.to_timestamp()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ged", default="data/raw/ucdp_ged_251.parquet")
    ap.add_argument("--start", default="2013-04")
    ap.add_argument("--end", default="2024-12")
    ap.add_argument("--radius", type=float, default=200.0, help="max km to centroid")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="data/interim/gdelt_ucdp_month.parquet")
    args = ap.parse_args()

    cent = unit_centroids(args.ged)
    tree = BallTree(np.radians(cent[["lat", "lon"]].values), metric="haversine")
    units = cent["unit"].values
    print(f"{len(cent)} UCDP unit centroids | radius cap {args.radius:.0f} km")

    os.makedirs(CACHE, exist_ok=True)
    months = pd.period_range(args.start, args.end, freq="M")
    print(
        f"streaming {len(months)} months of GDELT daily files "
        f"({args.start}..{args.end}) with {args.workers} workers"
    )

    for k, per in enumerate(months, 1):
        cpath = f"{CACHE}/{per.strftime('%Y%m')}.parquet"
        if os.path.exists(cpath):
            continue
        d0 = per.to_timestamp().date()
        d1 = (per + 1).to_timestamp().date() - timedelta(days=1)
        days = [d0 + timedelta(days=i) for i in range((d1 - d0).days + 1)]
        out = process_month(str(per), days, tree, units, args.radius, args.workers)
        out.to_parquet(cpath, index=False)
        if k % 6 == 0 or k == len(months):
            print(f"  {k}/{len(months)} months done (through {per})")

    parts = [pd.read_parquet(f"{CACHE}/{p.strftime('%Y%m')}.parquet") for p in months]
    full = pd.concat([p for p in parts if len(p)], ignore_index=True)
    # a unit-month may span two monthly caches (late-arriving rows): re-aggregate
    sums = [c for c in full.columns if c.startswith("txt_") and "mean" not in c]
    means = [c for c in full.columns if "mean" in c]
    g = full.groupby(["unit", "month"])
    full = pd.concat([g[sums].sum(), g[means].mean()], axis=1).reset_index()
    full.to_parquet(args.out, index=False)
    print(
        f"\nwrote {len(full):,} (unit, month) news-signal rows | "
        f"{full['unit'].nunique()} units -> {args.out}"
    )


if __name__ == "__main__":
    main()
