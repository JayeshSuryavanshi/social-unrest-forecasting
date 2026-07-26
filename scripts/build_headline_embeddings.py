"""Modern-text arm, Phase 1: headline embeddings per UCDP unit-month.

Pipeline: URL slugs -> cleaned pseudo-headlines -> local sentence-transformer
embeddings (MPS) -> narts-weighted mean pool per (unit, month) -> PCA to K dims
(PCA fitted on TRAIN-ERA months only, <= 2017-03, mirroring the benchmark's
strictly-past representation discipline) -> panel with emb_* columns.

Slug parsing keeps the last URL path segment when it looks like editorial text
(>= 4 tokens, >= 60% alphabetic) after stripping ids/extensions; junk URLs
(numeric ids, feed endpoints) are dropped.

Usage:
    python scripts/build_headline_embeddings.py \
        --urls data/interim/gdelt_ucdp_urls.parquet \
        --out data/interim/headline_emb_ucdp.parquet
"""

from __future__ import annotations

import argparse
import re
from urllib.parse import urlparse

import numpy as np
import pandas as pd

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TRAIN_ERA_END = "2017-03-31"  # PCA fit window: strictly before first test month

_ID = re.compile(r"^[\d\-_.]+$")
_CLEAN = re.compile(r"\.(html?|php|aspx?|cms|ece|amp)$", re.I)
_TOKEN = re.compile(r"[a-z]{2,}", re.I)


def slug_to_headline(url: str) -> str | None:
    try:
        path = urlparse(url).path
    except Exception:  # noqa: BLE001
        return None
    segs = [s for s in path.split("/") if s]
    if not segs:
        return None
    # pick the longest segment that looks like text (usually the last)
    cand = max(segs, key=len)
    cand = _CLEAN.sub("", cand)
    if _ID.match(cand):
        return None
    words = re.split(r"[-_+]", cand)
    words = [w for w in words if w and not w.isdigit()]
    if len(words) < 4:
        return None
    alpha = sum(1 for w in words if _TOKEN.fullmatch(w))
    if alpha / len(words) < 0.6:
        return None
    return " ".join(words).lower()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--urls", default="data/interim/gdelt_ucdp_urls.parquet")
    ap.add_argument("--dims", type=int, default=32)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--out", default="data/interim/headline_emb_ucdp.parquet")
    args = ap.parse_args()

    d = pd.read_parquet(args.urls)
    print(f"{len(d):,} URL rows")
    d["headline"] = d["url"].map(slug_to_headline)
    d = d.dropna(subset=["headline"])
    print(
        f"{len(d):,} rows with parseable headlines "
        f"({d['headline'].nunique():,} unique)"
    )

    uniq = d["headline"].drop_duplicates().reset_index(drop=True)
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(MODEL, device="mps")
    emb = model.encode(
        uniq.tolist(),
        batch_size=args.batch,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    print(f"embedded {emb.shape[0]:,} unique headlines -> {emb.shape[1]} dims")
    lookup = {h: i for i, h in enumerate(uniq)}

    # narts-weighted mean pool per (unit, month)
    d["ei"] = d["headline"].map(lookup)
    d["w"] = np.clip(d["narts"].values, 1, None)
    rows, units, months = [], [], []
    for (u, m), g in d.groupby(["unit", "month"]):
        w = g["w"].values[:, None]
        rows.append((emb[g["ei"].values] * w).sum(0) / w.sum())
        units.append(u)
        months.append(m)
    X = np.vstack(rows)
    panel = pd.DataFrame({"unit": units, "month": months})
    print(f"pooled: {len(panel):,} unit-months")

    # PCA fitted on train-era months only (leakage discipline)
    from sklearn.decomposition import PCA

    train_mask = (panel["month"] <= TRAIN_ERA_END).values
    pca = PCA(n_components=args.dims, random_state=0).fit(X[train_mask])
    Z = pca.transform(X)
    print(
        f"PCA fit on {train_mask.sum():,} train-era unit-months | "
        f"explained var {pca.explained_variance_ratio_.sum():.2f}"
    )
    for k in range(args.dims):
        panel[f"emb_{k:02d}"] = Z[:, k]
    panel["emb_n_headlines"] = d.groupby(["unit", "month"]).size().values

    panel.to_parquet(args.out, index=False)
    print(f"wrote {len(panel):,} rows x {args.dims} emb dims -> {args.out}")


if __name__ == "__main__":
    main()
