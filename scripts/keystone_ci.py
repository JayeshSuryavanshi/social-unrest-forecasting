"""Block-bootstrap CIs for the keystone marginals from saved predictions.

Reads results/preds_ucdp_{kind}_{model}_{family}.parquet (written by
forecast_ucdp.py --save-preds), aligns rows across families, and bootstraps the
AUPRC marginal of history+richtext over history by resampling test months
(blocks) — the same protocol the adversarial review validated.

Usage:
    python scripts/keystone_ci.py --kind onset
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


def load(kind: str, model: str, family: str) -> pd.DataFrame:
    safe = family.replace("+", "-")
    p = pd.read_parquet(f"results/preds_ucdp_{kind}_{model}_{safe}.parquet")
    return p.rename(columns={"p": f"p_{safe}"})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kind", default="onset")
    ap.add_argument("--boot", type=int, default=1000)
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    for model in ("rf", "l2"):
        base = load(args.kind, model, "history")
        for fam2 in ("history+richtext", "richtext"):
            other = load(args.kind, model, fam2)[
                ["week", "adm1", f"p_{fam2.replace('+', '-')}"]
            ]
            m = base.merge(other, on=["week", "adm1"], how="inner")
            y = m["y"].values
            ph = m["p_history"].values
            po = m[f"p_{fam2.replace('+', '-')}"].values
            ap_h = average_precision_score(y, ph)
            ap_o = average_precision_score(y, po)
            weeks = m["week"].values
            uw = np.unique(weeks)
            idx_by_week = {w: np.where(weeks == w)[0] for w in uw}
            diffs = []
            for _ in range(args.boot):
                samp = rng.choice(uw, size=len(uw), replace=True)
                idx = np.concatenate([idx_by_week[w] for w in samp])
                yb = y[idx]
                if 0 < yb.sum() < len(yb):
                    diffs.append(
                        average_precision_score(yb, po[idx])
                        - average_precision_score(yb, ph[idx])
                    )
            lo, hi = np.percentile(diffs, [5, 95])
            p_pos = float(np.mean(np.array(diffs) > 0))
            print(
                f"{args.kind} | {model.upper()} | {fam2:18s} vs history: "
                f"{ap_o:.4f} vs {ap_h:.4f} | marginal {ap_o - ap_h:+.4f} "
                f"| 90% CI [{lo:+.4f}, {hi:+.4f}] | P(text helps)={p_pos:.2f}"
            )


if __name__ == "__main__":
    main()
