"""Rich-text arm: topic-model features over ACLED event narratives.

Addresses the central reviewer confound in our negative text result: GDELT
tone/volume and GKG themes are SHALLOW machine-coded signals, while the
literature's positive text results (Mueller & Rauh) come from topic models over
full text. Here we build the richest text signal available in our data without
proprietary news access: NMF topics over ACLED's human-written event `notes`
(272k narratives for India), aggregated per (state, week).

Leakage discipline: the TF-IDF vocabulary and NMF topic model are fitted ONLY on
notes from the first `--fit-weeks` weeks of the panel (default 52 — identical to
the backtest's minimum training window), then frozen and applied to all later
notes. No test-era text influences the representation.

Output: parquet with (week, adm1, txt_t00..txt_tNN topic shares, txt_n_notes),
mergeable onto the weekly panel; `txt_*` columns are lagged by build_features.

Usage:
    python scripts/build_notes_topics.py --csv data/raw/acled_india.csv \
        --topics 15 --out data/interim/acled_notes_topics_IN.parquet
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default="data/raw/acled_india.csv")
    ap.add_argument("--topics", type=int, default=15)
    ap.add_argument("--fit-weeks", type=int, default=52)
    ap.add_argument("--out", default="data/interim/acled_notes_topics_IN.parquet")
    args = ap.parse_args()

    df = pd.read_csv(args.csv, dtype=str, low_memory=False)
    df = df[df["country"] == "India"].dropna(subset=["notes", "admin1", "event_date"])
    df["date"] = pd.to_datetime(df["event_date"])
    df["week"] = df["date"].dt.to_period("W").dt.start_time
    print(f"{len(df):,} India events with notes")

    fit_cutoff = df["week"].min() + pd.Timedelta(weeks=args.fit_weeks)
    fit_notes = df.loc[df["week"] < fit_cutoff, "notes"]
    print(
        f"fitting vocabulary + NMF on {len(fit_notes):,} notes before "
        f"{fit_cutoff.date()} (inside the min training window)"
    )

    vec = TfidfVectorizer(
        max_features=20000, stop_words="english", min_df=5, sublinear_tf=True
    )
    X_fit = vec.fit_transform(fit_notes)
    nmf = NMF(n_components=args.topics, init="nndsvda", random_state=42, max_iter=400)
    nmf.fit(X_fit)

    terms = np.array(vec.get_feature_names_out())
    print("\ntopics (top-8 terms):")
    for k, row in enumerate(nmf.components_):
        print(f"  t{k:02d}: {' '.join(terms[row.argsort()[::-1][:8]])}")

    # transform ALL notes with the frozen model
    W = nmf.transform(vec.transform(df["notes"]))
    W = W / np.maximum(W.sum(axis=1, keepdims=True), 1e-9)  # doc topic shares
    tcols = [f"txt_t{k:02d}" for k in range(args.topics)]
    tw = pd.DataFrame(W, columns=tcols, index=df.index)
    tw["week"], tw["adm1"] = df["week"].values, df["admin1"].values

    agg = tw.groupby(["week", "adm1"], as_index=False)[tcols].mean()
    counts = tw.groupby(["week", "adm1"], as_index=False).size()
    agg = agg.merge(counts.rename(columns={"size": "txt_n_notes"}), on=["week", "adm1"])
    agg.to_parquet(args.out, index=False)
    print(
        f"\nwrote {len(agg):,} (week, adm1) rows x {args.topics} topics -> {args.out}"
    )


if __name__ == "__main__":
    main()
