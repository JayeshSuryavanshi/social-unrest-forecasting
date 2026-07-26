"""95% CI sensitivity for the keystone escalation text marginals.

Same protocol as scripts/keystone_ci.py (block bootstrap by test month,
1000 reps, seed 0, history+richtext vs history), reporting 95% intervals
alongside the 90% ones. Reads the shipped per-row predictions only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

RES = "/Users/jayeshsuryavanshi/Desktop/social-unrest-forecasting/results"


def load(kind: str, model: str, family: str) -> pd.DataFrame:
    safe = family.replace("+", "-")
    p = pd.read_parquet(f"{RES}/preds_ucdp_{kind}_{model}_{safe}.parquet")
    return p.rename(columns={"p": f"p_{safe}"})


def main() -> None:
    kind = "escalation"
    for model in ("rf", "l2"):
        rng = np.random.default_rng(0)
        base = load(kind, model, "history")
        fam2 = "history+richtext"
        safe2 = fam2.replace("+", "-")
        other = load(kind, model, fam2)[["week", "adm1", f"p_{safe2}"]]
        m = base.merge(other, on=["week", "adm1"], how="inner")
        y = m["y"].values
        ph = m["p_history"].values
        po = m[f"p_{safe2}"].values
        ap_h = average_precision_score(y, ph)
        ap_o = average_precision_score(y, po)
        weeks = m["week"].values
        uw = np.unique(weeks)
        idx_by_week = {w: np.where(weeks == w)[0] for w in uw}
        diffs = []
        for _ in range(1000):
            samp = rng.choice(uw, size=len(uw), replace=True)
            idx = np.concatenate([idx_by_week[w] for w in samp])
            yb = y[idx]
            if 0 < yb.sum() < len(yb):
                diffs.append(
                    average_precision_score(yb, po[idx])
                    - average_precision_score(yb, ph[idx])
                )
        d = np.array(diffs)
        lo90, hi90 = np.percentile(d, [5, 95])
        lo95, hi95 = np.percentile(d, [2.5, 97.5])
        pos = int((d > 0).sum())
        print(
            f"{kind} | {model.upper()} | n={len(m)} | marginal {ap_o - ap_h:+.4f} "
            f"| 90% CI [{lo90:+.4f}, {hi90:+.4f}] "
            f"| 95% CI [{lo95:+.4f}, {hi95:+.4f}] "
            f"| {pos}/{len(d)} replicates positive"
        )


if __name__ == "__main__":
    main()
