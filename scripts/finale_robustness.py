"""Robustness + bootstrap for the finale: is 'GKG text adds nothing over ACLED
history' stable across configs and statistically distinguishable from zero?"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from forecast_acled import acled_daily, acled_weekly  # noqa: E402
from forecast_unrest import (  # noqa: E402
    RNG,
    RandomForestClassifier,
    build_features,
    merge_gkg_weekly,
)

CSV = "data/raw/acled_india.csv"
GKG = "data/interim/gkg_IN_long_named.parquet"


def backtest(df: pd.DataFrame, feats: list[str], min_train: int) -> pd.DataFrame:
    weeks = np.array(sorted(df["week"].unique()))
    rows = []
    for w in weeks[min_train:]:
        tr, te = df[df["week"] < w], df[df["week"] == w]
        if tr["y"].nunique() < 2 or len(te) == 0:
            continue
        m = RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=5,
            class_weight="balanced",
            n_jobs=-1,
            random_state=RNG,
        )
        m.fit(tr[feats].values, tr["y"].values)
        out = te[["week", "y"]].copy()
        out["p"] = m.predict_proba(te[feats].values)[:, 1]
        rows.append(out)
    return pd.concat(rows, ignore_index=True)


def load_wk(start: str, end: str) -> pd.DataFrame:
    d = acled_daily(CSV)
    d = d[(d["date"] >= start) & (d["date"] <= end)]
    return merge_gkg_weekly(acled_weekly(d), GKG)


def run(wk, kind, h, target, min_train):
    df, _, fam = build_features(wk, horizon=h, label_kind=kind, target=target)
    ph = backtest(df, fam["history"], min_train)
    pg = backtest(df, fam["history+gkg"], min_train)
    return ph, pg


def main() -> None:
    wk = load_wk("2024-10-01", "2025-07-20")
    print("=== PRIMARY: escalation, n_unrest, min_train=16, block-bootstrap 90% CI ===")
    rng = np.random.default_rng(0)
    for h in (1, 3):
        ph, pg = run(wk, "escalation", h, "n_unrest", 16)
        y, weeks = ph["y"].values, ph["week"].values
        ap_h = average_precision_score(y, ph["p"].values)
        ap_g = average_precision_score(y, pg["p"].values)
        uw = np.unique(weeks)
        diffs = []
        for _ in range(500):
            samp = rng.choice(uw, size=len(uw), replace=True)
            idx = np.concatenate([np.where(weeks == w)[0] for w in samp])
            yb = y[idx]
            if 1 < yb.sum() < len(yb):
                diffs.append(
                    average_precision_score(yb, pg["p"].values[idx])
                    - average_precision_score(yb, ph["p"].values[idx])
                )
        lo, hi = np.percentile(diffs, [5, 95])
        p_pos = float(np.mean(np.array(diffs) > 0))
        print(
            f"  t+{h}: history AUPRC {ap_h:.3f} | +GKG {ap_g:.3f} | "
            f"marginal {ap_g - ap_h:+.3f} | 90% CI [{lo:+.3f}, {hi:+.3f}] | P(text helps)={p_pos:.2f}"
        )

    print(
        "\n=== ROBUSTNESS: escalation marginal (AUPRC of history+GKG minus history) ==="
    )
    print("  target       min_train  t+1      t+3")
    for target in ("n_unrest", "n_violence"):
        for mt in (12, 16, 20):
            m1 = m3 = None
            for h in (1, 3):
                ph, pg = run(wk, "escalation", h, target, mt)
                y = ph["y"].values
                d = average_precision_score(
                    y, pg["p"].values
                ) - average_precision_score(y, ph["p"].values)
                if h == 1:
                    m1 = d
                else:
                    m3 = d
            print(f"  {target:12s} {mt:^9d} {m1:+.3f}   {m3:+.3f}")


if __name__ == "__main__":
    main()
