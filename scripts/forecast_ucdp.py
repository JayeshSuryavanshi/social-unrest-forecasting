"""Multi-country forecasting on UCDP GED (admin1 x month, 55 countries).

Establishes the benchmark's multi-country backbone on redistributable ground
truth: occurrence, escalation, and hard-problem ONSET (violence in a unit calm
for >= 24 months) with pooled rolling-origin backtests. The 'week' column name
is reused for months — the harness machinery is period-agnostic.

Usage:
    python scripts/forecast_ucdp.py --kinds occurrence,escalation,onset
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backtest_lib import blocked_backtest, report  # noqa: E402
from forecast_unrest import build_features  # noqa: E402

PANEL = "data/interim/ucdp_adm1_month.parquet"


def monthly_grid(panel: pd.DataFrame) -> pd.DataFrame:
    """Complete (unit x month) grid per country, zero-filled, mapped onto the
    column schema build_features expects."""
    months = pd.date_range(panel["month"].min(), panel["month"].max(), freq="MS")
    units = panel[["country", "unit"]].drop_duplicates()
    grid = units.merge(pd.DataFrame({"month": months}), how="cross")
    wk = grid.merge(panel, on=["country", "unit", "month"], how="left")
    for c in ("n_events", "n_sb", "n_ns", "n_os", "deaths"):
        wk[c] = wk[c].fillna(0.0)
    # schema mapping for build_features
    wk = wk.rename(columns={"month": "week", "unit": "adm1"})
    wk["n_unrest"] = wk["n_events"]
    wk["n_violence"] = wk["n_events"]
    wk["n_protest"] = 0.0
    wk["n_matconf"] = wk["n_sb"] + wk["n_os"]
    wk["sum_articles"] = 0.0
    wk["mean_tone"] = 0.0
    wk["mean_goldstein"] = 0.0
    return wk.sort_values(["adm1", "week"]).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kinds", default="occurrence,escalation,onset")
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--min-train", type=int, default=48, help="months")
    ap.add_argument("--every", type=int, default=3, help="refit cadence (months)")
    ap.add_argument("--calm", type=int, default=24, help="onset calm window (months)")
    ap.add_argument(
        "--news-panel",
        default=None,
        help="GDELT news-signal panel (unit x month, txt_* cols) to merge",
    )
    ap.add_argument(
        "--grid-start",
        default=None,
        help="restrict grid to months >= this (YYYY-MM), e.g. news coverage era",
    )
    ap.add_argument(
        "--families", default="history", help="comma-separated feature families to run"
    )
    ap.add_argument(
        "--save-preds",
        action="store_true",
        help="save pooled per-family predictions for bootstrap CIs",
    )
    ap.add_argument("--out", default="results/ucdp_multicountry.csv")
    args = ap.parse_args()

    wk = monthly_grid(pd.read_parquet(PANEL))
    if args.news_panel:
        news = pd.read_parquet(args.news_panel).rename(
            columns={"unit": "adm1", "month": "week"}
        )
        tcols = [c for c in news.columns if c.startswith("txt_")]
        wk = wk.merge(news, on=["week", "adm1"], how="left")
        wk[tcols] = wk[tcols].fillna(0.0)
        print(f"merged news panel: {len(tcols)} txt_ signal cols")
    if args.grid_start:
        wk = wk[wk["week"] >= args.grid_start].reset_index(drop=True)
    print(
        f"grid: {wk['week'].nunique()} months x {wk['adm1'].nunique():,} adm1 "
        f"units ({wk['country'].nunique()} countries) = {len(wk):,} unit-months"
    )

    rows = []
    for kind in args.kinds.split(","):
        df, _, fam = build_features(
            wk,
            horizon=args.horizon,
            label_kind=kind,
            target="n_unrest",
            calm_periods=args.calm,
        )
        print(
            f"\n########## {kind.upper()} t+{args.horizon} mo | rows "
            f"{len(df):,} | positives {int(df['y'].sum()):,} ##########"
        )
        for model in ("rf", "l2"):
            for family in args.families.split(","):
                if not fam.get(family):
                    continue
                pred = blocked_backtest(
                    df, fam[family], args.min_train, model=model, every=args.every
                )
                r = report(
                    pred,
                    f"UCDP {kind} t+{args.horizon} | {model.upper()} | {family}",
                )
                if args.save_preds:
                    safe = family.replace("+", "-")
                    pred.to_parquet(
                        f"results/preds_ucdp_{kind}_{model}_{safe}.parquet", index=False
                    )
                rows.append(
                    {
                        "kind": kind,
                        "model": model,
                        "family": family,
                        # note: scorer "model" renamed to avoid clobbering the
                        # rf/l2 "model" column (bug found in the first run)
                        **{
                            ("auprc" if x["scorer"] == "model" else x["scorer"]): x[
                                "AUPRC"
                            ]
                            for x in r["table"]
                        },
                        "base": r["base"],
                        "n": r["n"],
                    }
                )

    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
