"""Structural (non-news) covariates: weekly climate anomalies per admin1.

Builds the genuinely non-news baseline arm. State centroids are computed from
ACLED's own event coordinates (median lat/lon per admin1 — no shapefiles), then
daily precipitation + temperature come from the free, no-key Open-Meteo
historical archive. Weekly anomalies are computed against each state's own
same-week-of-year climatology using PRIOR years only (expanding, leakage-free).

Output columns: week, adm1, str_precip_anom, str_temp_anom, str_precip_13wk
(13-week trailing rainfall deficit — an agricultural-stress proxy).

Usage:
    python scripts/build_climate_panel.py --csv data/raw/acled_india.csv \
        --start 2017-01-01 --end 2025-07-31 \
        --out data/interim/climate_IN_weekly.parquet
"""

from __future__ import annotations

import argparse
import time

import pandas as pd
import requests

API = "https://archive-api.open-meteo.com/v1/archive"


def fetch_daily(lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
    r = requests.get(
        API,
        params={
            "latitude": lat,
            "longitude": lon,
            "start_date": start,
            "end_date": end,
            "daily": "precipitation_sum,temperature_2m_mean",
            "timezone": "UTC",
        },
        timeout=120,
    )
    r.raise_for_status()
    d = r.json()["daily"]
    return pd.DataFrame(
        {
            "date": pd.to_datetime(d["time"]),
            "precip": d["precipitation_sum"],
            "temp": d["temperature_2m_mean"],
        }
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default="data/raw/acled_india.csv")
    ap.add_argument("--start", default="2017-01-01")
    ap.add_argument("--end", default="2025-07-31")
    ap.add_argument("--out", default="data/interim/climate_IN_weekly.parquet")
    args = ap.parse_args()

    ac = pd.read_csv(args.csv, dtype=str, low_memory=False)
    ac = ac[ac["country"] == "India"].dropna(subset=["latitude", "longitude", "admin1"])
    cent = (
        ac.groupby("admin1")
        .agg(
            lat=("latitude", lambda s: pd.to_numeric(s).median()),
            lon=("longitude", lambda s: pd.to_numeric(s).median()),
        )
        .reset_index()
    )
    print(f"{len(cent)} state centroids from ACLED event coordinates")

    import os

    cache = "data/interim/_climate_cache"
    os.makedirs(cache, exist_ok=True)
    frames = []
    for i, row in cent.iterrows():
        safe = row["admin1"].replace("/", "_")
        cpath = f"{cache}/{safe}.parquet"
        if os.path.exists(cpath):  # resume support after rate-limit aborts
            d = pd.read_parquet(cpath)
        else:
            d = None
            for attempt in range(6):
                try:
                    d = fetch_daily(row["lat"], row["lon"], args.start, args.end)
                    break
                except requests.HTTPError as e:
                    wait = (
                        90
                        if e.response is not None and e.response.status_code == 429
                        else 10
                    )
                    print(f"  {row['admin1']}: HTTP retry in {wait}s")
                    time.sleep(wait)
                except Exception:  # noqa: BLE001
                    time.sleep(10)
            if d is None:
                raise RuntimeError(f"failed to fetch {row['admin1']}")
            d.to_parquet(cpath, index=False)
            time.sleep(3)  # polite cadence for the free API
        d["adm1"] = row["admin1"]
        frames.append(d)
        if (i + 1) % 10 == 0 or i == len(cent) - 1:
            print(f"  {i + 1}/{len(cent)} states fetched")

    daily = pd.concat(frames, ignore_index=True)
    daily["week"] = daily["date"].dt.to_period("W").dt.start_time
    wk = daily.groupby(["adm1", "week"], as_index=False).agg(
        precip=("precip", "sum"), temp=("temp", "mean")
    )
    wk["woy"] = wk["week"].dt.isocalendar().week.astype(int)
    wk = wk.sort_values(["adm1", "week"]).reset_index(drop=True)

    # leakage-free anomalies: same-week-of-year expanding mean over PRIOR years
    def anom(g: pd.DataFrame, col: str) -> pd.Series:
        base = g.groupby("woy")[col].transform(
            lambda s: s.shift(1).expanding(min_periods=1).mean()
        )
        return g[col] - base

    out = []
    for _, g in wk.groupby("adm1"):
        g = g.copy()
        g["str_precip_anom"] = anom(g, "precip")
        g["str_temp_anom"] = anom(g, "temp")
        g["str_precip_13wk"] = g["precip"].rolling(13, min_periods=4).sum() - g[
            "precip"
        ].rolling(13, min_periods=4).sum().shift(52)
        out.append(g)
    res = pd.concat(out, ignore_index=True)
    res = res.dropna(subset=["str_precip_anom"])[
        ["week", "adm1", "str_precip_anom", "str_temp_anom", "str_precip_13wk"]
    ]
    res["str_precip_13wk"] = res["str_precip_13wk"].fillna(0.0)
    res.to_parquet(args.out, index=False)
    print(f"wrote {len(res):,} (week, adm1) climate rows -> {args.out}")


if __name__ == "__main__":
    main()
