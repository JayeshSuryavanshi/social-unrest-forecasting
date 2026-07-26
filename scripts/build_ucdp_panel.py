"""Build a multi-country (admin1 x month) panel from UCDP GED.

UCDP GED is human-coded organized violence (state-based / non-state / one-sided),
1989-2024, globally, with ~95% of events carrying an adm_1 name. Unlike ACLED it
is REDISTRIBUTABLE for research with citation — which is what makes a public
benchmark shippable. Scope note: GED has no protest/riot category, so the target
here is organized violence, not protest-unrest.

Usage:
    python scripts/build_ucdp_panel.py --raw data/raw/ucdp_ged_251.parquet \
        --start 2010-01 --out data/interim/ucdp_adm1_month.parquet
"""

from __future__ import annotations

import argparse

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", default="data/raw/ucdp_ged_251.parquet")
    ap.add_argument("--start", default="2010-01", help="first month to keep (YYYY-MM)")
    ap.add_argument(
        "--min-events",
        type=int,
        default=50,
        help="keep countries with at least this many events in-window",
    )
    ap.add_argument("--out", default="data/interim/ucdp_adm1_month.parquet")
    args = ap.parse_args()

    df = pd.read_parquet(args.raw)
    df = df.dropna(subset=["adm_1"])
    df["month"] = pd.to_datetime(df["date_start"]).dt.to_period("M").dt.to_timestamp()
    df = df[df["month"] >= args.start]

    keep = df.groupby("country")["id"].count()
    countries = keep[keep >= args.min_events].index
    df = df[df["country"].isin(countries)]
    print(
        f"{len(df):,} events | {df['country'].nunique()} countries with >= "
        f"{args.min_events} events since {args.start}"
    )

    df["unit"] = df["country"] + " | " + df["adm_1"].str.strip()
    g = df.groupby(["country", "unit", "month"])
    panel = g.agg(
        n_events=("id", "size"),
        n_sb=("type_of_violence", lambda s: (s == 1).sum()),
        n_ns=("type_of_violence", lambda s: (s == 2).sum()),
        n_os=("type_of_violence", lambda s: (s == 3).sum()),
        deaths=("best", "sum"),
    ).reset_index()

    panel.to_parquet(args.out, index=False)
    print(
        f"wrote {len(panel):,} (country, adm1, month) rows | "
        f"{panel['unit'].nunique():,} adm1 units -> {args.out}"
    )
    top = panel.groupby("country")["unit"].nunique().sort_values(ascending=False)
    print("\nadm1 units per country (top 12):")
    print(top.head(12).to_string())


if __name__ == "__main__":
    main()
