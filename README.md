# Social Unrest Forecasting

A rigorous, honest revival of a 2022 UB NLP course project
([`Social-Unrest-Prediction-NLP`](https://github.com/JayeshSuryavanshi/Social-Unrest-Prediction-NLP)),
rebuilt from scratch in 2026 as a real **out-of-time spatiotemporal forecasting** study on
current conflict data (ACLED + GDELT/GKG).

📄 **Feasibility brief (full write-up):** [`docs/feasibility-brief.html`](docs/feasibility-brief.html)
· also published at https://claude.ai/code/artifact/354de06a-0975-4ba4-b010-9580d47d1e4e

---

## ⚡ SUB-Forecast v1.0 — the benchmark
This repo now ships a frozen, leakage-controlled **subnational unrest forecasting benchmark**:
994 admin1 units × 55 countries × 15 years on redistributable UCDP ground truth, a news-signal
panel built from **336M GDELT events**, three targets (occurrence / escalation / hard-problem
onset), a signal-family ablation ladder, and saved reference predictions with bootstrap-CI
tooling. **Spec: [`BENCHMARK.md`](BENCHMARK.md).**

## TL;DR — what we found

**1. Can we predict social unrest? Yes — modestly, and it's real.**
Occurrence is well forecastable at multi-country scale (AUPRC 0.80 vs persistence 0.53);
escalation moderately (0.18 vs 0.11); and clean human-curated ground truth beats machine-coded
news data on identical tasks. But **onset in long-calm regions stays near-unforecastable for
every signal family tested** (best model 1.35× a 1.8% base rate; persistence is fully blind).

**2. Does news *text* add forecasting value? A little — far less than naive evaluations imply.**
Across shallow tone/volume, deep GKG themes, and narrative topic models, in 55+ countries, under
two model families: text is **informative but nearly subsumed** by the event record. Its genuine
complement is **+0.3–0.6 AUPRC points on escalation** (block-bootstrap 90% CIs clear of zero in
BOTH model families) — an order of magnitude smaller than leaky pipelines suggest, and detectable
only at ~90,000-observation scale. Single-country studies lack the power to see it, text-only
models collapse at scale, and text provides no rescue in the onset regime.

**3. The original 2022 "85% prediction" was label leakage** — confirmed in the original notebooks. The
model was trained to recover a label built from the same text it was predicting from.

### Honesty note — corrections from adversarial review
A 4-agent adversarial review (re-running our own numbers) caught two overclaims we had made, and we
corrected them (see [`RESULTS.md`](RESULTS.md) → "THE FINALE"):
- ❌ *"News text significantly degrades the forecast"* was **wrong** — an unregularized-RandomForest
  overfitting artifact. Under regularized logistic regression the marginal flips slightly **positive**
  (+0.02–0.03). Correct claim: *text fails to improve the best model*, not *text hurts*.
- ❌ *"ACLED is independent of the news stream"* was **false** — ~77% of ACLED India events are
  newspaper-sourced. ACLED is human-*curated* news, not non-news ground truth. A truly clean test of
  the text-vs-independent-history question needs **structural/administrative** data (police, NGO field
  records) — **never run here, and the real open frontier.**

---

## Repository map

```
├── README.md                     # this file
├── RESULTS.md                    # full results, every experiment, with the corrections
├── requirements.txt              # Python deps (Python 3.12)
├── docs/
│   └── feasibility-brief.html    # the polished write-up (open in a browser)
├── scripts/                      # all pipeline + modeling code (see below)
├── data/
│   ├── raw/                      # source data — DO NOT redistribute (see data/raw/README.md)
│   │   ├── acled_india.csv        # ACLED India export 2018-01 → 2025-07 (272k events)
│   │   └── gdelt_latest_export.csv
│   └── interim/                  # built (date × admin1) panels + the FIPS→state crosswalk
│       ├── gdelt_IN_daily_panel.parquet          # India GDELT events
│       ├── gdelt_NI_daily_panel.parquet          # Nigeria GDELT events
│       ├── gdelt_lowhistory_daily_panel.parquet  # 6 low-history countries
│       ├── gkg_IN_daily_themes.parquet           # India GKG themes 2025-09→2026-06
│       ├── gkg_IN_2024h2/2025_themes.parquet     # GKG themes for the ACLED-overlap window
│       ├── gkg_IN_long_named.parquet             # combined 2024-10→2025-07, keyed by ACLED state
│       └── gkg_fips_to_acled.json                # FIPS admin1 → ACLED state-name crosswalk
└── results/                      # saved backtest outputs (.txt) and download logs
```

## Scripts

| Script | What it does |
|---|---|
| `fetch_gdelt.py` | Download GDELT 2.0 events (no key) for a date range |
| `fetch_acled.py` | Download ACLED via the 2026 OAuth2 API (needs your myACLED login) |
| `build_gdelt_panel.py` | Stream GDELT 1.0 daily events → (date × admin1) panel for one or more countries |
| `build_gkg_panel.py` | Stream GDELT 1.0 GKG daily → (date × admin1) deep-text **theme** panel |
| `forecast_unrest.py` | Weekly panel, baselines, RandomForest, rolling-origin backtest; `--ablation` (shallow text) and `--gkg-ablation` (deep text) modes |
| `forecast_acled.py` | Same forecaster on ACLED ground truth (state names, fatalities) |
| `finale_robustness.py` | Bootstrap CI + robustness grid for the finale |

---

## Setup

```bash
cd ~/Desktop/social-unrest-forecasting
# uv (recommended) — or use python -m venv + pip
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -r requirements.txt
```

The built panels are already in `data/interim/`, so the **forecasting/ablation commands below run
immediately** — you only need to re-download if you want fresh data.

## Reproduce the key results

```bash
# --- Forecasting (uses prebuilt panels) ---
# India GDELT forecaster (occurrence + escalation, t+1 & t+3):
python scripts/forecast_unrest.py --panel data/interim/gdelt_IN_daily_panel.parquet --target n_unrest
# ACLED ground-truth forecaster (the strongest result):
python scripts/forecast_acled.py --csv data/raw/acled_india.csv --target n_unrest

# --- Does news text add skill? (ablations) ---
# shallow text (tone + volume) on India GDELT:
python scripts/forecast_unrest.py --panel data/interim/gdelt_IN_daily_panel.parquet --ablation
# low-history country (Ecuador) shallow-text test:
python scripts/forecast_unrest.py --panel data/interim/gdelt_lowhistory_daily_panel.parquet --cc EC --ablation --min-train-weeks 20
# deep GKG themes vs GDELT history:
python scripts/forecast_unrest.py --panel data/interim/gdelt_IN_daily_panel.parquet \
    --gkg-panel data/interim/gkg_IN_daily_themes.parquet --gkg-ablation --start 2025-09-08 --end 2026-06-22 --min-train-weeks 16

# --- The finale: news text vs ACLED history (+ robustness) ---
python scripts/forecast_acled.py --csv data/raw/acled_india.csv \
    --gkg-panel data/interim/gkg_IN_long_named.parquet --gkg-ablation \
    --start 2024-10-01 --end 2025-07-20 --min-train-weeks 16
python scripts/finale_robustness.py        # bootstrap CI + model-family robustness

# --- Rebuild panels from scratch (optional; large downloads) ---
python scripts/build_gdelt_panel.py --country IN --start 2025-01-01 --end 2026-06-30 --out data/interim/gdelt_IN_daily_panel.parquet
python scripts/build_gkg_panel.py --cc IN --adm1-prefix IN --start 2024-10-01 --end 2025-07-20 --out data/interim/gkg_IN_2024h2_themes.parquet
```

## Data provenance & licensing
See [`data/raw/README.md`](data/raw/README.md). **Short version:** GDELT is free/redistributable
(cite it); **ACLED data must NOT be redistributed** (free to access after registration, but
redistribution is forbidden by their terms) — keep `data/` out of any public repo.
