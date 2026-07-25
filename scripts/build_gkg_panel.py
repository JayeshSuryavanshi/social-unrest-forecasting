"""Stream GDELT 1.0 GKG (Global Knowledge Graph) daily files and build a
(date x admin1) DEEP-TEXT theme panel for one country. No API key.

This is the "deep text" signal — GDELT's per-article thematic tags (PROTEST,
UNREST_*, political-violence, terror, ...) parsed from full article text — as
opposed to the shallow coverage-tone / article-volume signal. It is the closest
open analog to the LDA news-topic features Mueller & Rauh used for conflict
onset. GKG geolocates sparsely at province level for small countries, so this is
run for a country with dense sub-national coverage (India: ~5k GKG records/day).

Disk-light: global daily GKG files (~19 MB each) are streamed, filtered to the
target country, and discarded; only the aggregated theme panel is persisted.

Usage:
    python scripts/build_gkg_panel.py --cc IN --adm1-prefix IN \
        --start 2025-09-01 --end 2026-06-30 \
        --out data/interim/gkg_IN_daily_themes.parquet
"""

from __future__ import annotations

import argparse
import io
import re
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta

import pandas as pd
import requests

BASE = "http://data.gdeltproject.org/gkg/{ymd}.gkg.csv.zip"

# Curated unrest-relevant GKG theme flags (substring match on the THEMES field).
THEME_FLAGS = {
    "th_protest": ("PROTEST",),
    "th_unrest": ("UNREST_",),
    "th_conflict": (
        "WB_2433_CONFLICT_AND_VIOLENCE",
        "ARMEDCONFLICT",
        "WB_2462_POLITICAL_VIOLENCE_AND_WAR",
    ),
    "th_terror": ("TERROR",),
    "th_kill": ("KILL",),
    "th_arrest": ("ARREST", "SECURITY_SERVICES"),
    "th_crisis": ("CRISISLEX",),
}


def daterange(s: date, e: date):
    d = s
    while d <= e:
        yield d
        d += timedelta(days=1)


def fetch_day(d: date, cc: str, adm1_re: re.Pattern) -> pd.DataFrame | None:
    url = BASE.format(ymd=d.strftime("%Y%m%d"))
    try:
        r = requests.get(url, timeout=120)
        if r.status_code != 200:
            return None
        z = zipfile.ZipFile(io.BytesIO(r.content))
        df = pd.read_csv(
            z.open(z.namelist()[0]), sep="\t", dtype=str, on_bad_lines="skip"
        )
    except Exception:
        return None
    if "LOCATIONS" not in df.columns:
        return None
    # cheap prefilter: keep only records mentioning the target country
    df = df[df["LOCATIONS"].fillna("").str.contains(f"#{cc}#", regex=False)]
    if df.empty:
        return None

    themes = df["THEMES"].fillna("")
    out = pd.DataFrame({"date": d.strftime("%Y%m%d")}, index=df.index)
    for flag, subs in THEME_FLAGS.items():
        pat = "|".join(re.escape(s) for s in subs)
        out[flag] = themes.str.contains(pat, regex=True).astype("int8")
    out["tone"] = pd.to_numeric(
        df["TONE"].fillna("").str.split(",").str[0], errors="coerce"
    )
    # each record -> its distinct target-country admin1 codes (record counts once per adm1)
    out["adm1s"] = (
        df["LOCATIONS"].fillna("").apply(lambda s: sorted(set(adm1_re.findall(s))))
    )
    out = (
        out[out["adm1s"].map(len) > 0]
        .explode("adm1s")
        .rename(columns={"adm1s": "adm1"})
    )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cc", default="IN", help="GKG FIPS country code (India=IN)")
    ap.add_argument(
        "--adm1-prefix", default="IN", help="admin1 code prefix (India=IN -> IN01..)"
    )
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    adm1_re = re.compile(rf"#({re.escape(args.adm1_prefix)}\d{{2}})#")
    days = list(daterange(date.fromisoformat(args.start), date.fromisoformat(args.end)))
    print(
        f"streaming {len(days)} GKG daily files for cc={args.cc} "
        f"({args.start}..{args.end}) with {args.workers} workers"
    )

    frames, done, missing = [], 0, 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(fetch_day, d, args.cc, adm1_re): d for d in days}
        for fut in as_completed(futs):
            r = fut.result()
            done += 1
            if r is None or r.empty:
                missing += 1
            else:
                frames.append(r)
            if done % 30 == 0 or done == len(days):
                kept = sum(map(len, frames))
                print(
                    f"  {done}/{len(days)} files | {kept:,} record-admin1 rows | {missing} empty"
                )

    raw = pd.concat(frames, ignore_index=True)
    raw["date"] = pd.to_datetime(raw["date"], format="%Y%m%d")
    flags = list(THEME_FLAGS)
    g = raw.groupby(["date", "adm1"])
    panel = g.agg(
        gkg_n=("tone", "size"),
        gkg_tone=("tone", "mean"),
        **{f"gkg_{f.replace('th_', '')}": (f, "sum") for f in flags},
    ).reset_index()
    panel.to_parquet(args.out, index=False)
    print(
        f"\nwrote GKG theme panel: {len(panel):,} (date,admin1) rows, "
        f"{panel['adm1'].nunique()} admin1 units -> {args.out}"
    )
    tot = panel[
        [c for c in panel.columns if c.startswith("gkg_") and c != "gkg_tone"]
    ].sum()
    print("theme totals:\n" + tot.to_string())


if __name__ == "__main__":
    main()
