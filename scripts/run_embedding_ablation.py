"""Phase-1 modern-text ablation: do headline EMBEDDINGS add skill beyond
(a) event history and (b) history + shallow news counts?

Embeddings enter the harness under the gkg_ prefix (their own lagged family on
the multi-country track, where GKG themes were never used). Families:

  history            event-history baseline
  gkg                embeddings only        (representation quality)
  history+gkg        emb over history       <- key comparison (a)
  history+text       shallow news over history (keystone reference)
  all                history + shallow news + embeddings  <- key comparison (b)

Kinds: escalation + onset; models RF + L2; keystone protocol; block-bootstrap
CIs on the two key marginals; predictions saved.

Usage:  python scripts/run_embedding_ablation.py
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

PANEL = "data/interim/ucdp_adm1_month.parquet"
NEWS = "data/interim/gdelt_ucdp_month.parquet"
EMB = "data/interim/headline_emb_ucdp.parquet"
MIN_TRAIN, EVERY, REPS = 48, 3, 1000
FAMILIES = ["history", "gkg", "history+gkg", "history+text", "all"]
RNG = np.random.default_rng(11)


def boot_ci(base: pd.DataFrame, other: pd.DataFrame):
    m = base.rename(columns={"p": "ph"}).merge(
        other[["week", "adm1", "p"]].rename(columns={"p": "po"}), on=["week", "adm1"]
    )
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
    pos = float(np.mean(np.array(diffs) > 0))
    return point, lo, hi, pos


def main() -> None:
    wk = monthly_grid(pd.read_parquet(PANEL))
    news = pd.read_parquet(NEWS).rename(columns={"unit": "adm1", "month": "week"})
    tcols = [c for c in news.columns if c.startswith("txt_")]
    wk = wk.merge(news, on=["week", "adm1"], how="left")
    wk[tcols] = wk[tcols].fillna(0.0)

    emb = pd.read_parquet(EMB).rename(columns={"unit": "adm1", "month": "week"})
    ecols = [c for c in emb.columns if c.startswith("emb_")]
    emb = emb.rename(columns={c: "gkg_" + c for c in ecols})
    gcols = ["gkg_" + c for c in ecols]
    wk = wk.merge(emb, on=["week", "adm1"], how="left")
    wk[gcols] = wk[gcols].fillna(0.0)
    wk = wk[wk["week"] >= "2013-04-01"].reset_index(drop=True)
    print(
        f"grid: {wk['week'].nunique()} months x {wk['adm1'].nunique():,} units "
        f"| {len(tcols)} news cols + {len(gcols)} emb cols",
        flush=True,
    )

    rows, preds = [], {}
    for kind in ("escalation", "onset"):
        df, _, fam = build_features(
            wk, horizon=1, label_kind=kind, target="n_unrest", calm_periods=24
        )
        print(
            f"\n### {kind}: {len(df):,} rows, {int(df['y'].sum()):,} positives",
            flush=True,
        )
        for model in ("rf", "l2"):
            for family in FAMILIES:
                pred = blocked_backtest(
                    df, fam[family], MIN_TRAIN, model=model, every=EVERY
                )
                ap = average_precision_score(pred["y"], pred["p"])
                print(
                    f"  {kind:10s} {model.upper():2s} {family:14s} " f"AUPRC {ap:.4f}",
                    flush=True,
                )
                preds[(kind, model, family)] = pred
                pred.to_parquet(
                    f"results/preds_emb_{kind}_{model}_"
                    f"{family.replace('+', '-')}.parquet",
                    index=False,
                )
                rows.append(
                    {
                        "kind": kind,
                        "model": model,
                        "family": family,
                        "auprc": ap,
                        "base": float(pred["y"].mean()),
                        "n": len(pred),
                    }
                )
            for name, a, b in [
                ("emb_over_history", "history", "history+gkg"),
                ("emb_over_hist+news", "history+text", "all"),
            ]:
                pt, lo, hi, pos = boot_ci(
                    preds[(kind, model, a)], preds[(kind, model, b)]
                )
                print(
                    f"  -> {name} [{model.upper()}]: {pt:+.4f} "
                    f"90% CI [{lo:+.4f}, {hi:+.4f}] P(+)={pos:.2f}",
                    flush=True,
                )
                rows.append(
                    {
                        "kind": kind,
                        "model": model,
                        "family": name,
                        "auprc": pt,
                        "lo": lo,
                        "hi": hi,
                        "ppos": pos,
                    }
                )

    pd.DataFrame(rows).to_csv("results/embedding_ablation.csv", index=False)
    print("\nsaved -> results/embedding_ablation.csv", flush=True)


if __name__ == "__main__":
    main()
