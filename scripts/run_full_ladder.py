"""The full India ablation ladder on one panel: event-history, rich-text
narrative topics, and non-news structural climate covariates — every family and
the combinations that answer the paper's questions:

  history vs history+X        -> does X add over news-derived history?
  structural vs structural+richtext -> does TEXT add over a NON-news baseline?

Both RandomForest and L2 logistic per family (model-family robustness).

Usage:
    python scripts/run_full_ladder.py
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
CLIMATE = "data/interim/climate_IN_weekly.parquet"
CSV = "data/raw/acled_india.csv"
MIN_TRAIN, EVERY = 52, 4

FAMILIES = [
    "history",
    "richtext",
    "structural",
    "history+richtext",
    "history+structural",
    "structural+richtext",
    "all",
]


def main() -> None:
    wk = acled_weekly(acled_daily(CSV))
    for path in (TOPICS, CLIMATE):
        aux = pd.read_parquet(path)
        cols = [c for c in aux.columns if c.startswith(("txt_", "str_"))]
        wk = wk.merge(aux, on=["week", "adm1"], how="left")
        wk[cols] = wk[cols].fillna(0.0)
    n_txt = len([c for c in wk.columns if c.startswith("txt_")])
    n_str = len([c for c in wk.columns if c.startswith("str_")])
    print(
        f"panel: {wk['week'].nunique()} weeks x {wk['adm1'].nunique()} states "
        f"| {n_txt} topic cols + {n_str} structural cols"
    )

    rows = []
    for h in (1, 3):
        df, _, fam = build_features(
            wk, horizon=h, label_kind="escalation", target="n_unrest"
        )
        for model in ("rf", "l2"):
            res = {}
            for name in FAMILIES:
                pred = blocked_backtest(
                    df, fam[name], MIN_TRAIN, model=model, every=EVERY
                )
                res[name] = (pred, score(pred["y"].values, pred["p"].values)[0])
            y = res["history"][0]["y"].values
            ap_pers = score(y, res["history"][0]["persist"].values.astype(float))[0]
            print(
                f"\n=== escalation t+{h} | {model.upper()} | base {y.mean():.3f} "
                f"n={len(y):,} ==="
            )
            print(f"  persistence            {ap_pers:.3f}")
            for name in FAMILIES:
                print(f"  {name:22s} {res[name][1]:.3f}")
            print(
                f"  -> text over news-history:      "
                f"{res['history+richtext'][1] - res['history'][1]:+.3f}"
            )
            print(
                f"  -> structural over history:     "
                f"{res['history+structural'][1] - res['history'][1]:+.3f}"
            )
            print(
                f"  -> TEXT over NON-NEWS baseline: "
                f"{res['structural+richtext'][1] - res['structural'][1]:+.3f}"
            )
            rows.append(
                {
                    "h": h,
                    "model": model,
                    "persistence": ap_pers,
                    **{k: v[1] for k, v in res.items()},
                }
            )

    pd.DataFrame(rows).to_csv("results/full_ladder_IN.csv", index=False)
    print("\nsaved -> results/full_ladder_IN.csv")


if __name__ == "__main__":
    main()
