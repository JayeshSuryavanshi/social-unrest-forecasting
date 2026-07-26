"""Shared rolling-origin backtest utilities for the benchmark runs.

Blocked refits: retrain at every `every`-th period and score all periods in the
block with that (<= every-1 periods stale) model — every test row remains
strictly out-of-time. Both a tree model and a regularized linear model are run,
per the adversarial-review lesson that signed marginals can be model-family
artifacts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

RNG = 42


def fit_predict(model: str, Xtr, ytr, Xte) -> np.ndarray:
    if model == "rf":
        m = RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=5,
            class_weight="balanced",
            n_jobs=-1,
            random_state=RNG,
        )
        m.fit(Xtr, ytr)
        return m.predict_proba(Xte)[:, 1]
    sc = StandardScaler().fit(Xtr)
    m = LogisticRegression(
        C=0.1, penalty="l2", solver="lbfgs", class_weight="balanced", max_iter=2000
    )
    m.fit(sc.transform(Xtr), ytr)
    return m.predict_proba(sc.transform(Xte))[:, 1]


def blocked_backtest(
    df: pd.DataFrame,
    feats: list[str],
    min_train: int,
    model: str = "rf",
    every: int = 4,
) -> pd.DataFrame:
    """Expanding-window backtest with blocked refits. Returns rows
    (week, adm1?, y, p, persist, clim) pooled over all test periods."""
    periods = np.array(sorted(df["week"].unique()))
    out = []
    for i in range(min_train, len(periods), every):
        block = periods[i : i + every]
        train = df[df["week"] < block[0]]
        test = df[df["week"].isin(block)]
        if train["y"].nunique() < 2 or len(test) == 0:
            continue
        p = fit_predict(
            model, train[feats].values, train["y"].values, test[feats].values
        )
        keep = [c for c in ("week", "adm1", "y", "persist", "clim") if c in test]
        o = test[keep].copy()
        o["p"] = p
        out.append(o)
    return pd.concat(out, ignore_index=True)


def score(y: np.ndarray, s: np.ndarray) -> tuple[float, float, float]:
    s = np.asarray(s, dtype=float)
    ap = average_precision_score(y, s)
    auc = roc_auc_score(y, s) if len(np.unique(s)) > 1 else 0.5
    br = brier_score_loss(y, np.clip(s, 0, 1))
    return ap, auc, br


def report(pred: pd.DataFrame, label: str) -> dict:
    y = pred["y"].values
    rows = []
    for name, col in [
        ("persistence", "persist"),
        ("climatology", "clim"),
        ("model", "p"),
    ]:
        if col in pred:
            ap, auc, br = score(y, pred[col].values)
            rows.append({"scorer": name, "AUPRC": ap, "AUROC": auc, "Brier": br})
    tbl = pd.DataFrame(rows)
    print(
        f"\n=== {label} | base rate {y.mean():.4f} | n={len(y):,} "
        f"({int(y.sum())} positives) ==="
    )
    print(tbl.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    return {"label": label, "base": float(y.mean()), "n": int(len(y)), "table": rows}
