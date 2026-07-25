"""Forecast unrest on ACLED ground truth (India), admin1 x week, out-of-time.

ACLED is human-curated and INDEPENDENT of the GDELT news stream, so it is the
right baseline for the faithful test: does news text add skill beyond a history
that was NOT itself derived from news? This script builds the ACLED panel and
runs the same leakage-free rolling-origin backtest; with --gkg-panel it merges
GKG news-theme features and runs the deep-text ablation against ACLED history.

Usage:
    python scripts/forecast_acled.py --csv data/raw/acled_india.csv --target n_unrest
    python scripts/forecast_acled.py --csv data/raw/acled_india.csv \
        --gkg-panel data/interim/gkg_IN_named_themes.parquet --gkg-ablation \
        --start 2025-01-06 --end 2025-07-20
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from forecast_unrest import (  # noqa: E402
    build_features,
    evaluate,
    gkg_ablation,
    merge_gkg_weekly,
    rolling_origin,
    text_ablation,
)

VIOLENT = [
    "Riots",
    "Violence against civilians",
    "Battles",
    "Explosions/Remote violence",
]


def acled_daily(csv: str) -> pd.DataFrame:
    df = pd.read_csv(csv, dtype=str, low_memory=False)
    df = df[df["country"] == "India"].copy()
    df["date"] = pd.to_datetime(df["event_date"], errors="coerce")
    df["fat"] = pd.to_numeric(df["fatalities"], errors="coerce").fillna(0)
    df = df.dropna(subset=["date", "admin1"])
    et = df["event_type"]
    df["is_protest"] = (et == "Protests").astype(int)
    df["is_violence"] = et.isin(VIOLENT).astype(int)
    df["is_matconf"] = et.isin(
        ["Battles", "Explosions/Remote violence", "Violence against civilians"]
    ).astype(int)
    g = df.groupby(["date", "admin1"])
    panel = (
        g.agg(
            n_events=("event_type", "size"),
            n_protest=("is_protest", "sum"),
            n_violence=("is_violence", "sum"),
            n_matconf=("is_matconf", "sum"),
            sum_fatalities=("fat", "sum"),
        )
        .reset_index()
        .rename(columns={"admin1": "adm1"})
    )
    # columns build_features expects but ACLED lacks (news-derived) -> neutral 0
    panel["sum_articles"] = 0.0
    panel["mean_tone"] = 0.0
    panel["mean_goldstein"] = 0.0
    return panel


def acled_weekly(daily: pd.DataFrame) -> pd.DataFrame:
    d = daily.copy()
    d["week"] = pd.to_datetime(d["date"]).dt.to_period("W").dt.start_time
    cols = [
        "n_events",
        "n_protest",
        "n_violence",
        "n_matconf",
        "sum_articles",
        "sum_fatalities",
    ]
    agg = {c: "sum" for c in cols}
    agg["mean_tone"] = "mean"
    agg["mean_goldstein"] = "mean"
    wk = d.groupby(["week", "adm1"], as_index=False).agg(agg)
    weeks = pd.date_range(wk["week"].min(), wk["week"].max(), freq="W-MON")
    units = sorted(wk["adm1"].unique())
    grid = pd.MultiIndex.from_product([weeks, units], names=["week", "adm1"]).to_frame(
        index=False
    )
    wk = grid.merge(wk, on=["week", "adm1"], how="left")
    for c in cols + ["mean_tone", "mean_goldstein"]:
        wk[c] = wk[c].fillna(0.0)
    wk["n_unrest"] = wk["n_protest"] + wk["n_violence"]
    return wk.sort_values(["adm1", "week"]).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", required=True)
    ap.add_argument(
        "--target", default="n_unrest", choices=["n_unrest", "n_violence", "n_protest"]
    )
    ap.add_argument("--gkg-panel", default=None)
    ap.add_argument("--gkg-ablation", action="store_true")
    ap.add_argument("--ablation", action="store_true")
    ap.add_argument("--start", default="2018-01-08")
    ap.add_argument("--end", default="2025-07-20")
    ap.add_argument("--min-train-weeks", type=int, default=52)
    args = ap.parse_args()

    daily = acled_daily(args.csv)
    daily = daily[(daily["date"] >= args.start) & (daily["date"] <= args.end)]
    print(
        f"ACLED India daily rows: {len(daily):,} | "
        f"{daily['date'].min().date()}..{daily['date'].max().date()} | "
        f"{daily['adm1'].nunique()} states"
    )
    wk = acled_weekly(daily)
    print(
        f"weekly grid: {wk['week'].nunique()} weeks x {wk['adm1'].nunique()} states "
        f"= {len(wk):,} state-weeks"
    )
    if args.gkg_panel:
        wk = merge_gkg_weekly(wk, args.gkg_panel)
        print(
            f"merged GKG: {len([c for c in wk.columns if c.startswith('gkg_')])} theme cols"
        )

    print(f"target = {args.target}")
    if args.gkg_ablation:
        gkg_ablation(wk, args.target, args.min_train_weeks)
        return
    if args.ablation:
        text_ablation(wk, args.target, args.min_train_weeks)
        return
    for kind in ("occurrence", "escalation"):
        print(f"\n\n########## ACLED | LABEL = {kind.upper()} ##########")
        for h in (1, 3):
            df, feats, _ = build_features(
                wk, horizon=h, label_kind=kind, target=args.target
            )
            pred = rolling_origin(df, feats, args.min_train_weeks)
            evaluate(
                pred, f"ACLED | {args.target} | {kind} | t+{h} wk | {len(feats)} feats"
            )


if __name__ == "__main__":
    main()
