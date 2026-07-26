# SUB-Forecast v1.0 — Subnational Unrest Benchmark (frozen spec)

A leakage-controlled benchmark for subnational unrest/violence forecasting with an
explicit **signal-family ablation ladder**. Frozen 2026-07-26; changes require a
version bump.

## Units, periods, ground truth

| Track | Ground truth | Unit | Period | Coverage |
|---|---|---|---|---|
| **Multi-country (primary)** | UCDP GED 25.1 (organized violence; redistributable) | admin1 ("Country \| Adm1", 994 units, 55 countries with ≥50 events since 2010) | calendar month | 2010-01 → 2024-12 |
| India high-resolution | ACLED (protests + violence; **bring your own export** — not redistributable) | state/UT (36) | ISO week | 2018 → free-tier lag |

Panels are **complete grids** — every unit × every period, zeros included.
Multi-country panel ships in `data/interim/ucdp_adm1_month.parquet`.

## Targets (label definitions — frozen)

Let `x(u,t)` = event count for unit `u` in period `t` (UCDP: all GED events;
ACLED: protests + riots + violence).

1. **occurrence**: `y = 1[x(u, t+h) > 0]`
2. **escalation**: `y = 1[x(u,t+h) > μ₈(u,t) + σ₈(u,t) AND x(u,t+h) ≥ μ₈(u,t)+2]`
   where μ₈/σ₈ are the trailing 8-period mean/std through `t` (strictly past).
3. **onset** (hard problem): occurrence, evaluated **only on the calm risk set**
   `Σ x(u, t−23..t) = 0` (24 calm months; 26 weeks on the India track).

Horizons: h = 1 (primary), h = 3 (lead-time decay).

## Signal families (the ablation ladder)

| Family | Content | Source |
|---|---|---|
| `history` | lags 1–4 + rolling 4/12 means of event counts by type, recency, per-unit expanding climatology, rest-of-country diffusion, seasonality | ground-truth event stream |
| `text` (shallow) | coverage tone, article volume | GDELT events |
| `gkg` (deep themes) | PROTEST/UNREST/conflict/terror/kill/arrest/crisis theme counts + tone | GDELT GKG |
| `richtext` | NMF-15 topic shares over event narratives (topic model fitted on the first 52 weeks ONLY) | ACLED notes (India track) |
| `structural` | precipitation/temperature anomalies vs same-week-of-year prior-years mean, 13-week rainfall deficit | Open-Meteo (CC-BY) |

Multi-country news signals (`txt_*` in `data/interim/gdelt_ucdp_month.parquet`):
336M GDELT events (2013-04 → 2024-12) assigned to UCDP units by
**nearest-centroid haversine join** (unit centroid = median GED event lat/lon;
200 km cap; no name crosswalks). Known limitation: border mis-assignment.

## Evaluation protocol (frozen)

- **Rolling-origin, out-of-time only.** Expanding window; refit every 3 months
  (multi-country; every 4 weeks India) and score the block with the frozen
  model. Minimum training: 48 months / 52 weeks. No random splits, ever.
- **Models**: report BOTH a tree ensemble (RandomForest 300, class-balanced,
  min_samples_leaf 5) and a regularized linear model (L2 logistic, C=0.1,
  standardized). Signed marginals must agree in direction across families or be
  reported as model-conditional.
- **Metrics**: AUPRC (headline; always with base rate), AUROC (secondary),
  Brier. Marginals between feature families get a **block bootstrap by test
  period** (1000 reps, 90% CI) computed from saved per-row predictions.
- **Baselines that must be reported**: persistence (current state), per-unit
  expanding climatology.

## Reference results (v1.0 keystone, tests 2017-04 → 2024-12)

| Regime | base | n test | persistence | history (RF) | +news marginal (RF / L2) |
|---|--:|--:|--:|--:|---|
| occurrence | 0.200 | 87,472 | 0.531 | 0.801 | +0.002 ✓ / −0.000 ∅ |
| escalation | 0.037 | 87,472 | 0.107 | 0.175 | **+0.006 [+0.002,+0.011] / +0.003 [+0.001,+0.005]** |
| onset | 0.018 | 27,565 | 0.018 (blind) | 0.025 | −0.000 ∅ / +0.003 ∅ (P=0.92) |

Per-row predictions for all 18 reference backtests ship in
`results/preds_ucdp_*.parquet`; recompute CIs with `scripts/keystone_ci.py`.

## Reproduce

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate && uv pip install -r requirements.txt
# keystone from shipped panels (~3 h):
python scripts/forecast_ucdp.py --news-panel data/interim/gdelt_ucdp_month.parquet \
    --grid-start 2013-04-01 --kinds onset,escalation,occurrence \
    --families history,richtext,history+richtext --save-preds --min-train 48 --every 3
python scripts/keystone_ci.py --kind escalation   # bootstrap CIs
# rebuild panels from sources (large downloads): see script docstrings
```

## Citations required by data licenses
UCDP GED (Sundberg & Melander 2013; Davies et al. 2025 version 25.1) · The GDELT
Project (gdeltproject.org) · Open-Meteo (CC-BY 4.0) · ACLED (acleddata.com; India
track only — obtain your own export, do not redistribute).
