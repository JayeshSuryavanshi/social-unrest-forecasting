"""Paper figures, computed from the shipped per-row predictions (authoritative)
— no hand-typed numbers.

Fig 1  regime_map.pdf   : AUPRC by regime x scorer (log scale, base-rate ticks)
Fig 2  forest.pdf       : text-marginal forest plot with block-bootstrap CIs;
                          multi-country cells recomputed from preds; India
                          single-country cells from the recorded bootstrap
                          (results/finale_robustness.txt) to show CI width.

Usage:  python scripts/make_figures.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update(
    {
        "font.size": 8.5,
        "font.family": "serif",
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "pdf.fonttype": 42,
    }
)

KINDS = ["occurrence", "escalation", "onset"]
RNG = np.random.default_rng(0)


def load(kind: str, model: str, family: str) -> pd.DataFrame:
    safe = family.replace("+", "-")
    return pd.read_parquet(f"results/preds_ucdp_{kind}_{model}_{safe}.parquet")


def boot_marginal(kind: str, model: str, reps: int = 1000):
    base = load(kind, model, "history").rename(columns={"p": "ph"})
    oth = load(kind, model, "history+richtext")[["week", "adm1", "p"]]
    m = base.merge(oth.rename(columns={"p": "po"}), on=["week", "adm1"])
    y, ph, po, weeks = (m["y"].values, m["ph"].values, m["po"].values, m["week"].values)
    point = average_precision_score(y, po) - average_precision_score(y, ph)
    uw = np.unique(weeks)
    idx_by = {w: np.where(weeks == w)[0] for w in uw}
    diffs = []
    for _ in range(reps):
        idx = np.concatenate([idx_by[w] for w in RNG.choice(uw, len(uw))])
        yb = y[idx]
        if 0 < yb.sum() < len(yb):
            diffs.append(
                average_precision_score(yb, po[idx])
                - average_precision_score(yb, ph[idx])
            )
    lo, hi = np.percentile(diffs, [5, 95])
    return point, lo, hi


def fig_regime_map() -> None:
    scorers = [
        ("persistence", "persist", None),
        ("climatology", "clim", None),
        ("history", "p", "history"),
        ("history+news", "p", "history+richtext"),
    ]
    fig, ax = plt.subplots(figsize=(3.3, 2.5))
    xs = np.arange(len(KINDS))
    markers = ["o", "s", "D", "^"]
    for j, (label, col, fam) in enumerate(scorers):
        vals = []
        for kind in KINDS:
            pr = load(kind, "rf", fam or "history")
            s = pr[col].values.astype(float)
            vals.append(average_precision_score(pr["y"].values, s))
        ax.plot(
            xs + (j - 1.5) * 0.13,
            vals,
            markers[j],
            ms=4.5,
            mew=0.8,
            mfc="white" if j < 2 else None,
            label=label,
        )
    for i, kind in enumerate(KINDS):
        b = load(kind, "rf", "history")["y"].mean()
        ax.hlines(b, i - 0.32, i + 0.32, color="0.45", lw=0.9, ls=":", zorder=0)
        ax.annotate(
            f"base {b:.3f}",
            (i + 0.02, b),
            textcoords="offset points",
            xytext=(0, -9),
            fontsize=6.5,
            color="0.35",
            ha="center",
        )
    ax.set_yscale("log")
    ax.set_xticks(xs, ["occurrence", "escalation", "onset\n(calm risk set)"])
    ax.set_ylabel("AUPRC (log scale)")
    ax.legend(
        frameon=False,
        fontsize=7,
        loc="upper right",
        handletextpad=0.2,
        borderaxespad=0.1,
    )
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig("paper/figures/regime_map.pdf")
    print("wrote paper/figures/regime_map.pdf")


def fig_forest() -> None:
    rows = []
    for kind in KINDS:
        for model in ("rf", "l2"):
            pt, lo, hi = boot_marginal(kind, model)
            rows.append((f"multi-country {kind} ({model.upper()})", pt, lo, hi, False))
            print(f"  {kind} {model}: {pt:+.4f} [{lo:+.4f}, {hi:+.4f}]")
    # single-country India cells: recorded block-bootstrap, finale_robustness.txt
    rows.append(("India escalation h=1 (RF)", -0.022, -0.056, 0.023, True))
    rows.append(("India escalation h=3 (RF)$^\\dagger$", -0.043, -0.071, -0.011, True))

    fig, ax = plt.subplots(figsize=(3.3, 2.7))
    ys = np.arange(len(rows))[::-1]
    for y0, (label, pt, lo, hi, india) in zip(ys, rows):
        c = "0.45" if india else ("#1a6b3c" if lo > 0 else "#333333")
        ax.hlines(y0, lo, hi, color=c, lw=1.4)
        ax.plot(pt, y0, "o", ms=3.6, color=c)
    ax.axvline(0, color="0.2", lw=0.7, ls="--")
    ax.set_yticks(ys, [r[0] for r in rows], fontsize=7)
    ax.set_xlabel("news-signal marginal AUPRC (90% CI)")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(left=False)
    fig.tight_layout()
    fig.savefig("paper/figures/forest.pdf")
    print("wrote paper/figures/forest.pdf")


if __name__ == "__main__":
    fig_regime_map()
    fig_forest()
