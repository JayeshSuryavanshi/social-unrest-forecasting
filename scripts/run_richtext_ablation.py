"""Rich-text ablation on ACLED India: do narrative TOPIC features add skill
beyond event-history counts — on escalation AND on hard-problem onset?

Families: history / richtext (topics only) / history+richtext, each under BOTH
RandomForest and L2 logistic regression (marginals must be model-family robust).

Usage:
    python scripts/run_richtext_ablation.py
"""

from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backtest_lib import blocked_backtest, score  # noqa: E402
from forecast_acled import acled_daily, acled_weekly  # noqa: E402
from forecast_unrest import build_features  # noqa: E402

TOPICS = "data/interim/acled_notes_topics_IN.parquet"
CSV = "data/raw/acled_india.csv"
MIN_TRAIN = 52
EVERY = 4


def main() -> None:
    daily = acled_daily(CSV)
    wk = acled_weekly(daily)
    topics = pd.read_parquet(TOPICS)
    tcols = [c for c in topics.columns if c.startswith("txt_")]
    wk = wk.merge(topics, on=["week", "adm1"], how="left")
    wk[tcols] = wk[tcols].fillna(0.0)
    print(
        f"panel: {wk['week'].nunique()} weeks x {wk['adm1'].nunique()} states "
        f"| {len(tcols)} topic cols merged"
    )

    results = []
    for kind, h in (("escalation", 1), ("escalation", 3), ("onset", 1)):
        df, _, fam = build_features(
            wk, horizon=h, label_kind=kind, target="n_unrest", calm_periods=26
        )
        if kind == "onset":
            print(
                f"\nONSET risk set: {len(df):,} calm state-weeks, "
                f"{int(df['y'].sum())} onsets "
                f"(base rate {df['y'].mean():.4f})"
                if len(df)
                else "\nONSET risk set EMPTY"
            )
            if df["y"].sum() < 10:
                print(
                    "  -> too few onset positives in India (states rarely calm "
                    "26+ wks); onset needs the multi-country UCDP panel"
                )
                continue
        for model in ("rf", "l2"):
            preds = {}
            for name in ("history", "richtext", "history+richtext"):
                preds[name] = blocked_backtest(
                    df, fam[name], MIN_TRAIN, model=model, every=EVERY
                )
            y = preds["history"]["y"].values
            ap_h = score(y, preds["history"]["p"].values)[0]
            ap_r = score(y, preds["richtext"]["p"].values)[0]
            ap_hr = score(y, preds["history+richtext"]["p"].values)[0]
            ap_p = score(y, preds["history"]["persist"].values.astype(float))[0]
            print(
                f"\n--- {kind} t+{h} | {model.upper()} | base "
                f"{y.mean():.3f} n={len(y):,} ---"
            )
            print(f"  persistence      AUPRC {ap_p:.3f}")
            print(f"  history          AUPRC {ap_h:.3f}")
            print(f"  topics only      AUPRC {ap_r:.3f}  (vs history {ap_r-ap_h:+.3f})")
            print(f"  history+topics   AUPRC {ap_hr:.3f}  MARGINAL {ap_hr-ap_h:+.3f}")
            results.append(
                {
                    "kind": kind,
                    "h": h,
                    "model": model,
                    "persist": ap_p,
                    "history": ap_h,
                    "richtext": ap_r,
                    "combined": ap_hr,
                }
            )

    pd.DataFrame(results).to_csv("results/richtext_ablation_IN.csv", index=False)
    print("\nsaved -> results/richtext_ablation_IN.csv")


if __name__ == "__main__":
    main()
