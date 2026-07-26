"""Honest out-of-time unrest forecasting demo on the GDELT Nigeria panel.

This is the reframed version of the 2022 project's Task 3, done without the
label leakage. The label is a FUTURE outcome defined from event data alone
(not from the text we predict from), and evaluation is a strict rolling-origin
backtest (train on the past, test on the future) with rare-event metrics and
mandatory baselines.

Task: for each admin1 unit u and week w, predict whether >=1 violent event
(GDELT CAMEO assault/fight/mass-violence) occurs in u during week w+h.
GROUND-TRUTH CAVEAT: GDELT is a noisy news proxy, not ACLED. This demonstrates
that the pipeline and evaluation are sound; point it at ACLED for the real task.

The bar is not accuracy -- it is beating PERSISTENCE ("violence next week iff
violence this week") on AUPRC/Brier out-of-time. Conflict is very sticky, so
that is a hard bar, and clearing it is the only honest evidence of skill.

Usage:
    python scripts/forecast_unrest.py --panel data/interim/gdelt_NI_daily_panel.parquet
"""

from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
RNG = 42

COUNT_COLS = [
    "n_events",
    "n_protest",
    "n_assault",
    "n_fight",
    "n_massvio",
    "n_matconf",
    "n_violence",
    "sum_articles",
]


def weekly_panel(daily: pd.DataFrame) -> pd.DataFrame:
    """Daily (date,admin1) -> complete (week,admin1) grid, zero-filled."""
    d = daily.copy()
    # drop bare country-level codes (e.g. "NI"): keep only province units ("NI01"..)
    d = d[d["adm1"].str.len() >= 4].copy()
    d["week"] = pd.to_datetime(d["date"]).dt.to_period("W").dt.start_time
    agg = {c: "sum" for c in COUNT_COLS if c in d.columns}
    agg["mean_tone"] = "mean"
    agg["mean_goldstein"] = "mean"
    wk = d.groupby(["week", "adm1"], as_index=False).agg(agg)

    # complete grid: every unit x every week in range (absence is a real observation)
    weeks = pd.date_range(wk["week"].min(), wk["week"].max(), freq="W-MON")
    units = sorted(wk["adm1"].unique())
    grid = pd.MultiIndex.from_product([weeks, units], names=["week", "adm1"]).to_frame(
        index=False
    )
    wk = grid.merge(wk, on=["week", "adm1"], how="left")
    for c in COUNT_COLS:
        if c in wk.columns:
            wk[c] = wk[c].fillna(0.0)
    wk[["mean_tone", "mean_goldstein"]] = wk[["mean_tone", "mean_goldstein"]].fillna(
        0.0
    )
    # social-unrest target = protests + violent events (India is protest-heavy)
    wk["n_unrest"] = wk["n_protest"] + wk["n_violence"]
    return wk.sort_values(["adm1", "week"]).reset_index(drop=True)


def build_features(
    wk: pd.DataFrame,
    horizon: int,
    label_kind: str = "occurrence",
    target: str = "n_unrest",
    calm_periods: int = 26,
) -> tuple[pd.DataFrame, list[str]]:
    df = wk.copy()
    tgt = target if target in df.columns else "n_violence"
    df["viol"] = (df[tgt] > 0).astype(int)

    # ESCALATION indicator (leakage-free): this week's target count spikes above
    # the unit's own recent norm -- threshold uses only PAST weeks (shift(1)).
    ge = df.groupby("adm1", group_keys=False)[tgt]
    tmean = ge.apply(lambda s: s.shift(1).rolling(8, min_periods=3).mean())
    tstd = ge.apply(lambda s: s.shift(1).rolling(8, min_periods=3).std())
    df["escal"] = (
        (df[tgt] > (tmean + tstd.fillna(0))) & (df[tgt] >= tmean + 2)
    ).astype(int)

    # the "event" series this run predicts (occurrence vs escalation vs onset)
    df["evt"] = df["escal"] if label_kind == "escalation" else df["viol"]

    # ONSET (hard-problem) risk set: units with ZERO target events in the last
    # `calm_periods` periods (inclusive of t). Predicting eruption among the calm
    # -- the regime where persistence is blind and text is claimed to help.
    if label_kind == "onset":
        gc = df.groupby("adm1", group_keys=False)["viol"]
        recent = gc.apply(
            lambda s: s.rolling(calm_periods, min_periods=calm_periods).sum()
        )
        df["_risk"] = (recent == 0).astype(int)

    # national "rest-of-country" activity that week (crude spatial diffusion signal)
    nat = df.groupby("week")[tgt].transform("sum")
    df["roc_violence"] = nat - df[tgt]

    feats: list[str] = []
    g = df.groupby("adm1", group_keys=False)
    for col in [
        "n_unrest",
        "n_events",
        "n_violence",
        "n_protest",
        "n_matconf",
        "sum_articles",
        "mean_tone",
        "mean_goldstein",
        "roc_violence",
    ]:
        for lag in (1, 2, 3, 4):
            name = f"{col}_lag{lag}"
            df[name] = g[col].shift(lag)
            feats.append(name)
        for win in (4, 12):
            name = f"{col}_roll{win}"
            df[name] = (
                g[col]
                .shift(1)
                .rolling(win, min_periods=1)
                .mean()
                .reset_index(level=0, drop=True)
            )
            feats.append(name)

    # recency: weeks since last violent week
    def weeks_since(s: pd.Series) -> pd.Series:
        out, cnt = [], 999
        for v in s:
            cnt = 0 if v > 0 else cnt + 1
            out.append(cnt)
        return pd.Series(out, index=s.index).shift(1)

    df["weeks_since_viol"] = g["viol"].apply(weeks_since)
    feats.append("weeks_since_viol")

    # per-unit climatology (expanding past positive rate of the target event)
    df["clim_rate"] = g["evt"].apply(
        lambda s: s.shift(1).expanding(min_periods=1).mean()
    )
    feats.append("clim_rate")

    # trend
    df["viol_trend4"] = df["n_violence_lag1"] - df["n_violence_lag4"]
    feats.append("viol_trend4")

    # seasonality
    woy = df["week"].dt.isocalendar().week.astype(int)
    df["woy_sin"] = np.sin(2 * np.pi * woy / 52.0)
    df["woy_cos"] = np.cos(2 * np.pi * woy / 52.0)
    feats += ["woy_sin", "woy_cos"]

    # AUXILIARY SIGNAL FAMILIES, lagged identically if merged into wk:
    #   gkg_* = GKG deep-text themes; txt_* = rich-text narrative topics;
    #   str_* = non-news STRUCTURAL covariates (climate anomalies, elections...)
    gkg_base = [c for c in df.columns if c.startswith(("gkg_", "txt_", "str_"))]
    for col in gkg_base:
        for lag in (1, 2, 3, 4):
            nm = f"{col}_lag{lag}"
            df[nm] = g[col].shift(lag)
            feats.append(nm)
        nm = f"{col}_roll4"
        df[nm] = (
            g[col]
            .shift(1)
            .rolling(4, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
        feats.append(nm)

    # LABEL: target event in week w+h (strictly future)
    df["y"] = g["evt"].shift(-horizon)

    # baseline scores (available at prediction time)
    df["persist"] = df["evt"]  # this week's event state -> predict next
    df["clim"] = df["clim_rate"].fillna(0)  # trailing positive rate

    df = df.dropna(subset=feats + ["y"]).reset_index(drop=True)
    df["y"] = df["y"].astype(int)
    if label_kind == "onset":  # evaluate only among currently-calm units
        df = df[df["_risk"] == 1].reset_index(drop=True)

    # feature families for ablations:
    #  text     = shallow news signals (coverage sentiment + article volume)
    #  gkg      = deep-text GKG theme features
    #  richtext = topic-model features over event narratives (txt_*)
    #  history  = structural event-history counts (everything else)
    text_feats = [f for f in feats if f.startswith(("mean_tone", "sum_articles"))]
    gkg_feats = [f for f in feats if f.startswith("gkg_")]
    rich_feats = [f for f in feats if f.startswith("txt_")]
    struct_feats = [f for f in feats if f.startswith("str_")]
    aux = set(text_feats) | set(gkg_feats) | set(rich_feats) | set(struct_feats)
    hist_feats = [f for f in feats if f not in aux]
    fams = {
        "history": hist_feats,
        "text": text_feats,
        "gkg": gkg_feats,
        "richtext": rich_feats,
        "structural": struct_feats,
        "all": feats,
        "history+text": hist_feats + text_feats,
        "history+gkg": hist_feats + gkg_feats,
        "history+richtext": hist_feats + rich_feats,
        "history+structural": hist_feats + struct_feats,
        "structural+richtext": struct_feats + rich_feats,
    }
    return df, feats, fams


def rolling_origin(df: pd.DataFrame, feats: list[str], min_train_weeks: int = 24):
    """Expanding-window backtest: for each test week, train on all rows whose
    label week is strictly before the test week's label week. Refit each step."""
    weeks = np.array(sorted(df["week"].unique()))
    test_weeks = weeks[min_train_weeks:]
    rows = []
    for w in test_weeks:
        train = df[df["week"] < w]  # label weeks all strictly earlier
        test = df[df["week"] == w]
        if train["y"].nunique() < 2 or len(test) == 0:
            continue
        Xtr, ytr = train[feats].values, train["y"].values
        Xte = test[feats].values

        rf = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=5,
            class_weight="balanced",
            n_jobs=-1,
            random_state=RNG,
        )
        rf.fit(Xtr, ytr)
        p_rf = rf.predict_proba(Xte)[:, 1]

        sc = StandardScaler().fit(Xtr)
        lr = LogisticRegression(max_iter=1000, class_weight="balanced", C=0.5)
        lr.fit(sc.transform(Xtr), ytr)
        p_lr = lr.predict_proba(sc.transform(Xte))[:, 1]

        out = test[["week", "adm1", "y", "persist", "clim"]].copy()
        out["p_rf"], out["p_lr"] = p_rf, p_lr
        rows.append(out)
    return pd.concat(rows, ignore_index=True)


def evaluate(pred: pd.DataFrame, label: str) -> dict:
    y = pred["y"].values
    base = y.mean()
    res = {"model": [], "AUPRC": [], "AUROC": [], "Brier": []}
    for name, col in [
        ("Persistence", "persist"),
        ("Climatology", "clim"),
        ("LogReg", "p_lr"),
        ("RandomForest", "p_rf"),
    ]:
        s = pred[col].values.astype(float)
        res["model"].append(name)
        res["AUPRC"].append(average_precision_score(y, s))
        res["AUROC"].append(roc_auc_score(y, s))
        # brier needs [0,1]; persistence is already 0/1, clim in [0,1]
        sb = np.clip(s, 0, 1)
        res["Brier"].append(brier_score_loss(y, sb))
    tbl = pd.DataFrame(res)
    tbl["AUPRC_lift_vs_persist"] = tbl["AUPRC"] - tbl.loc[0, "AUPRC"]
    print(
        f"\n{'='*68}\n  {label}   (base rate = {base:.3f}, n_test = {len(y):,})\n{'='*68}"
    )
    print(tbl.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    return {
        "label": label,
        "base_rate": float(base),
        "n": int(len(y)),
        "table": tbl.to_dict(orient="records"),
    }


def _rf_backtest(df: pd.DataFrame, feats: list[str], min_train_weeks: int):
    """Lean rolling-origin backtest, RandomForest only, for the ablation.
    Returns pooled (y, prob) across all out-of-time test weeks."""
    weeks = np.array(sorted(df["week"].unique()))
    ys, ps = [], []
    for w in weeks[min_train_weeks:]:
        train = df[df["week"] < w]
        test = df[df["week"] == w]
        if train["y"].nunique() < 2 or len(test) == 0:
            continue
        ytr = train["y"].values
        m = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=5,
            class_weight="balanced",
            n_jobs=-1,
            random_state=RNG,
        )
        m.fit(train[feats].values, ytr)
        ps.append(m.predict_proba(test[feats].values)[:, 1])
        ys.append(test["y"].values)
    return np.concatenate(ys), np.concatenate(ps)


def text_ablation(wk: pd.DataFrame, target: str, min_train_weeks: int) -> None:
    """Marginal-value study: does a news-text signal (coverage tone + article
    volume) add forecasting skill BEYOND the structural event-history features?
    The honest, leakage-free version of the 2022 project's core question."""
    print(
        "\n\n########## TEXT-SIGNAL ABLATION "
        "(does news tone + article volume add skill beyond event history?) ##########"
    )
    for kind in ("occurrence", "escalation"):
        for h in (1, 3):
            df, _, fam = build_features(wk, horizon=h, label_kind=kind, target=target)
            scores = {}
            for setname in ("history", "text", "all"):
                y, p = _rf_backtest(df, fam[setname], min_train_weeks)
                scores[setname] = (average_precision_score(y, p), roc_auc_score(y, p))
            base = float(y.mean())
            # persistence baseline pooled over the same out-of-time test weeks
            pers_y, pers_s = [], []
            weeks = np.array(sorted(df["week"].unique()))
            for w in weeks[min_train_weeks:]:
                t = df[df["week"] == w]
                if len(t):
                    pers_y.append(t["y"].values)
                    pers_s.append(t["persist"].values.astype(float))
            pers_y, pers_s = np.concatenate(pers_y), np.concatenate(pers_s)
            p_ap, p_auc = (
                average_precision_score(pers_y, pers_s),
                roc_auc_score(pers_y, pers_s),
            )

            tbl = pd.DataFrame(
                {
                    "feature set": [
                        "persistence",
                        "history only",
                        "text only",
                        "history + text",
                    ],
                    "AUPRC": [
                        p_ap,
                        scores["history"][0],
                        scores["text"][0],
                        scores["all"][0],
                    ],
                    "AUROC": [
                        p_auc,
                        scores["history"][1],
                        scores["text"][1],
                        scores["all"][1],
                    ],
                }
            )
            tbl["AUPRC vs history"] = tbl["AUPRC"] - scores["history"][0]
            print(
                f"\n=== {kind} | t+{h} wk | base rate {base:.3f} "
                f"| {len(fam['history'])} history + {len(fam['text'])} text feats ==="
            )
            print(tbl.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
            print(
                f"  -> MARGINAL AUPRC of news text over history alone: "
                f"{scores['all'][0] - scores['history'][0]:+.3f}"
            )


def merge_gkg_weekly(wk: pd.DataFrame, gkg_path: str) -> pd.DataFrame:
    """Merge a GKG daily theme panel into the weekly (week, adm1) grid."""
    g = pd.read_parquet(gkg_path)
    g["week"] = pd.to_datetime(g["date"]).dt.to_period("W").dt.start_time
    count_cols = [c for c in g.columns if c.startswith("gkg_") and c != "gkg_tone"]
    agg = {c: "sum" for c in count_cols}
    agg["gkg_tone"] = "mean"
    gw = g.groupby(["week", "adm1"], as_index=False).agg(agg)
    out = wk.merge(gw, on=["week", "adm1"], how="left")
    for c in count_cols + ["gkg_tone"]:
        out[c] = out[c].fillna(0.0)
    return out


def gkg_ablation(wk: pd.DataFrame, target: str, min_train_weeks: int) -> None:
    """Deep-text marginal-value study: do GKG THEME features add forecasting
    skill BEYOND event history? The faithful version of the 2022 question."""
    print(
        "\n\n########## GKG DEEP-TEXT ABLATION "
        "(do GKG theme signals add skill beyond event history?) ##########"
    )
    for kind in ("occurrence", "escalation"):
        for h in (1, 3):
            df, _, fam = build_features(wk, horizon=h, label_kind=kind, target=target)
            scores, pred = {}, None
            for setname in ("history", "gkg", "history+gkg"):
                y, p = _rf_backtest(df, fam[setname], min_train_weeks)
                scores[setname] = (average_precision_score(y, p), roc_auc_score(y, p))
            base = float(y.mean())
            pers_y, pers_s = [], []
            for w in np.array(sorted(df["week"].unique()))[min_train_weeks:]:
                t = df[df["week"] == w]
                if len(t):
                    pers_y.append(t["y"].values)
                    pers_s.append(t["persist"].values.astype(float))
            pers_y, pers_s = np.concatenate(pers_y), np.concatenate(pers_s)
            tbl = pd.DataFrame(
                {
                    "feature set": [
                        "persistence",
                        "history only",
                        "GKG themes only",
                        "history + GKG",
                    ],
                    "AUPRC": [
                        average_precision_score(pers_y, pers_s),
                        scores["history"][0],
                        scores["gkg"][0],
                        scores["history+gkg"][0],
                    ],
                    "AUROC": [
                        roc_auc_score(pers_y, pers_s),
                        scores["history"][1],
                        scores["gkg"][1],
                        scores["history+gkg"][1],
                    ],
                }
            )
            tbl["AUPRC vs history"] = tbl["AUPRC"] - scores["history"][0]
            print(
                f"\n=== {kind} | t+{h} wk | base rate {base:.3f} "
                f"| {len(fam['history'])} history + {len(fam['gkg'])} GKG feats ==="
            )
            print(tbl.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
            print(
                f"  -> MARGINAL AUPRC of GKG deep-text over history alone: "
                f"{scores['history+gkg'][0] - scores['history'][0]:+.3f}"
            )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--panel", required=True)
    ap.add_argument(
        "--ablation",
        action="store_true",
        help="run the news-text marginal-value ablation instead of the standard eval",
    )
    ap.add_argument(
        "--gkg-panel",
        default=None,
        help="path to a GKG daily theme panel to merge as deep-text features",
    )
    ap.add_argument(
        "--gkg-ablation",
        action="store_true",
        help="run the GKG deep-text marginal-value ablation",
    )
    ap.add_argument("--min-train-weeks", type=int, default=24)
    ap.add_argument(
        "--start", default="2025-01-06", help="first event-date week to keep"
    )
    ap.add_argument("--end", default="2026-06-22", help="last event-date week to keep")
    ap.add_argument(
        "--cc",
        default=None,
        help="FIPS country code to slice from a multi-country panel (e.g. KE)",
    )
    ap.add_argument(
        "--target",
        default="n_unrest",
        choices=["n_unrest", "n_violence", "n_protest"],
        help="event type to forecast (n_unrest = protests + violence)",
    )
    args = ap.parse_args()

    daily = pd.read_parquet(args.panel)
    daily["date"] = pd.to_datetime(daily["date"])
    if args.cc:  # slice one country out of a multi-country panel
        daily = daily[daily["adm1"].str[:2] == args.cc]
    # GDELT daily files are keyed by when an event was ADDED; SQLDATE (event date)
    # can be historical. Keep only the window with complete daily-file coverage,
    # trimmed a week at each edge to avoid left/right censoring.
    n0 = len(daily)
    daily = daily[(daily["date"] >= args.start) & (daily["date"] <= args.end)]
    print(
        f"loaded daily panel: {n0:,} rows -> {len(daily):,} in window "
        f"[{args.start}..{args.end}], {daily['adm1'].nunique()} admin1 units"
    )
    wk = weekly_panel(daily)
    print(
        f"weekly grid: {wk['week'].nunique()} weeks x {wk['adm1'].nunique()} units "
        f"= {len(wk):,} unit-weeks"
    )
    if args.gkg_panel:
        wk = merge_gkg_weekly(wk, args.gkg_panel)
        ncol = len([c for c in wk.columns if c.startswith("gkg_")])
        print(f"merged GKG deep-text panel: {ncol} theme columns")

    print(f"target = {args.target}")
    if args.gkg_ablation:
        gkg_ablation(wk, args.target, args.min_train_weeks)
        return
    if args.ablation:
        text_ablation(wk, args.target, args.min_train_weeks)
        return

    for kind in ("occurrence", "escalation"):
        desc = (
            ">=1 unrest event next week"
            if kind == "occurrence"
            else "unrest spikes above the state's recent norm"
        )
        print(f"\n\n########## LABEL = {kind.upper()} ({desc}) ##########")
        for h in (1, 3):
            df, feats, _ = build_features(
                wk, horizon=h, label_kind=kind, target=args.target
            )
            pred = rolling_origin(df, feats, args.min_train_weeks)
            evaluate(pred, f"{args.target} | {kind} | t+{h} wk | {len(feats)} features")

    print("\nRead this as: does any model beat PERSISTENCE out-of-time, and how does")
    print("the lift decay from t+1 to t+3? For OCCURRENCE (base rate ~0.96) AUROC is")
    print("the honest lens; for ESCALATION (low base rate) AUPRC is. Accuracy is never")
    print(
        "reported -- with imbalanced classes it just rewards predicting the majority."
    )


if __name__ == "__main__":
    main()
