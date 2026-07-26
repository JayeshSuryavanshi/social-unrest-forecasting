"""Label-sensitivity robustness for the keystone findings.

Q1: does the escalation news-text marginal (+0.003..+0.006, both families)
survive alternative escalation-label definitions?
Q2: does the onset null survive alternative calm windows?

Variants (frozen v1.0 config first as the same-seed reference):
  escalation: (window=8,  sigma=1, floor=2)   <- frozen
              (window=12, sigma=1, floor=2)
              (window=8,  sigma=2, floor=2)
              (window=8,  sigma=1, floor=3)
  onset calm: 24 months <- frozen; plus 12 and 36

Protocol identical to the keystone: multi-country UCDP grid + GDELT news
panel, grid-start 2013-04, min-train 48, refit every 3, RF + L2,
history vs history+news, block-bootstrap (500 reps) CI on the marginal.

Usage:  python scripts/label_sensitivity.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backtest_lib import blocked_backtest  # noqa: E402
from forecast_ucdp import monthly_grid  # noqa: E402
from forecast_unrest import build_features  # noqa: E402

NEWS = "data/interim/gdelt_ucdp_month.parquet"
PANEL = "data/interim/ucdp_adm1_month.parquet"
MIN_TRAIN, EVERY, REPS = 48, 3, 500
RNG = np.random.default_rng(7)


def boot_ci(m: pd.DataFrame) -> tuple[float, float, float]:
    y, ph, po, weeks = (m["y"].values, m["ph"].values, m["po"].values, m["week"].values)
    point = average_precision_score(y, po) - average_precision_score(y, ph)
    uw = np.unique(weeks)
    idx_by = {w: np.where(weeks == w)[0] for w in uw}
    diffs = []
    for _ in range(REPS):
        idx = np.concatenate([idx_by[w] for w in RNG.choice(uw, len(uw))])
        yb = y[idx]
        if 0 < yb.sum() < len(yb):
            diffs.append(
                average_precision_score(yb, po[idx])
                - average_precision_score(yb, ph[idx])
            )
    lo, hi = np.percentile(diffs, [5, 95])
    return point, lo, hi


def run_variant(wk, name, kind, model, **kw):
    df, _, fam = build_features(wk, horizon=1, label_kind=kind, target="n_unrest", **kw)
    ph = blocked_backtest(
        df, fam["history"], MIN_TRAIN, model=model, every=EVERY
    ).rename(columns={"p": "ph"})
    po = blocked_backtest(
        df, fam["history+richtext"], MIN_TRAIN, model=model, every=EVERY
    )[["week", "adm1", "p"]]
    m = ph.merge(po.rename(columns={"p": "po"}), on=["week", "adm1"])
    pt, lo, hi = boot_ci(m)
    base, n = float(m["y"].mean()), len(m)
    line = (
        f"{name:28s} | {model.upper():2s} | base {base:.4f} n={n:,} | "
        f"marginal {pt:+.4f} | 90% CI [{lo:+.4f}, {hi:+.4f}]"
    )
    print(line, flush=True)
    return {
        "variant": name,
        "model": model,
        "base": base,
        "n": n,
        "marginal": pt,
        "lo": lo,
        "hi": hi,
    }


def main() -> None:
    wk = monthly_grid(pd.read_parquet(PANEL))
    news = pd.read_parquet(NEWS).rename(columns={"unit": "adm1", "month": "week"})
    tcols = [c for c in news.columns if c.startswith("txt_")]
    wk = wk.merge(news, on=["week", "adm1"], how="left")
    wk[tcols] = wk[tcols].fillna(0.0)
    wk = wk[wk["week"] >= "2013-04-01"].reset_index(drop=True)
    print(
        f"grid: {wk['week'].nunique()} months x {wk['adm1'].nunique():,} units",
        flush=True,
    )

    variants = [
        ("escal frozen (8,1s,+2)", "escalation", {}),
        ("escal window=12", "escalation", {"escal_window": 12}),
        ("escal sigma=2", "escalation", {"escal_sigma": 2.0}),
        ("escal floor=3", "escalation", {"escal_floor": 3.0}),
        ("onset calm=24 (frozen)", "onset", {"calm_periods": 24}),
        ("onset calm=12", "onset", {"calm_periods": 12}),
        ("onset calm=36", "onset", {"calm_periods": 36}),
    ]
    rows = []
    for name, kind, kw in variants:
        for model in ("rf", "l2"):
            rows.append(run_variant(wk, name, kind, model, **kw))
    pd.DataFrame(rows).to_csv("results/label_sensitivity.csv", index=False)
    print("\nsaved -> results/label_sensitivity.csv", flush=True)


if __name__ == "__main__":
    main()
