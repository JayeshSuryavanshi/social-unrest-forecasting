# Demo results — GDELT out-of-time backtest

Run: `python scripts/forecast_unrest.py --panel data/interim/gdelt_<CC>_daily_panel.parquet --target n_unrest`
Ablation: add `--ablation`. Data: GDELT 1.0, event-weeks 2025-01-06 → 2026-06-22 (77 weeks).
Model: **RandomForest** (300 trees, class-balanced). Evaluation: expanding-window rolling-origin
backtest, refit each week (~50 out-of-time test weeks). **No random split; the label is a future
outcome defined from events, never from the text/features.** Target `n_unrest` = protests + violence.

> ⚠️ **GDELT is a noisy news proxy, not ACLED.** These numbers validate the *pipeline and
> evaluation*, not a deployable forecaster. Point the same code at ACLED for ground truth.
>
> _Model note: switched from sklearn HistGradientBoosting to RandomForest — HGB had a thread-
> oversubscription pathology on this machine (~5 s/fit → 15+ min runs); RF fits in ~0.3 s and the
> qualitative story is unchanged._

## ⭐⭐⭐⭐⭐⭐ KEYSTONE — text vs history across regimes, MULTI-COUNTRY at scale (2026-07-25)
The campaign's decisive experiment. 336M GDELT events (2013–2024) joined to 1,682 UCDP units by
nearest-centroid haversine (200 km cap, no crosswalks) → machine-coded news signals (tone, volume,
protest/conflict counts) per unit-month, merged onto the 55-country UCDP panel. 18 rolling-origin
backtests (3 regimes × RF/L2 × {history, text-only, history+text}), tests 2017→2024, block-bootstrap
CIs (1000 reps, by test month) from saved predictions (`results/preds_ucdp_*.parquet`,
`keystone_ucdp.csv` — note: first run's CSV `model` column was clobbered by a scorer-key collision,
fixed in code; rows are ordered rf then l2 per kind, and the preds parquets are the authoritative record).

| Regime | base | n test | history (RF) | text marginal (RF) | 90% CI | text marginal (L2) | 90% CI |
|---|--:|--:|--:|--:|---|--:|---|
| occurrence | 0.200 | 87,472 | 0.801 | +0.0020 | [+0.001, +0.003] ✓ | −0.0001 | [−0.001, +0.000] ∅ |
| **escalation** | 0.037 | 87,472 | 0.175 | **+0.0064** | **[+0.002, +0.011] ✓ P=0.99** | **+0.0029** | **[+0.001, +0.005] ✓ P=1.00** |
| onset (calm≥24mo) | 0.018 | 27,565 | 0.025 | −0.0002 | [−0.006, +0.004] ∅ | +0.0029 | [−0.001, +0.009] P=0.92 ∅ |

**Findings:**
1. **First statistically significant positive text marginal in the project — escalation, BOTH model
   families** (+0.3–0.6 AUPRC points, ~2–4% relative). Small, real, robust.
2. **Consistent with the India nulls, not contradicting them**: India's CIs were ±0.03 wide — a true
   +0.005 effect was below its detection floor. The 87k-observation panel had the power to resolve it.
   Text's real complement is an order of magnitude smaller than naive evaluations imply.
3. **Onset stays unforecastable for everyone**: best model 0.025 vs base 0.018 (1.35×); text adds
   nothing significant even in the regime the literature reserves for it (L2 hint P=0.92, ns).
4. **Text-only collapses at scale** (occurrence 0.40–0.46 vs history 0.78–0.80; escalation 0.085 vs
   0.175): the India near-parity of text-only was a hot-region/small-sample exception, not a rule.

**Unified thesis (final): news-derived signals are informative but nearly subsumed — they carry real
signal (beat non-news baselines; substitute partially for history), complement event history only
slightly (+0.3–0.6 AUPRC pts on escalation, detectable only at ~90k-obs scale), and offer no rescue at
the onset frontier, where forecasting fails for every signal family tested.**

### Label-sensitivity robustness (2026-07-26; `label_sensitivity.{py,csv,log}`)
Both headline claims are **label-robust** (identical keystone protocol, 500-rep block-bootstrap CIs):

| Escalation variant | RF marginal [90% CI] | L2 marginal [90% CI] |
|---|---|---|
| frozen (8-mo, 1σ, +2) | +0.0064 [+0.002, +0.011] ✓ | +0.0029 [+0.001, +0.005] ✓ |
| window = 12 | **+0.0112 [+0.006, +0.016] ✓** | +0.0030 [+0.001, +0.005] ✓ |
| σ = 2 | +0.0057 [+0.002, +0.009] ✓ | +0.0012 [−0.000, +0.002] ⚠ grazes 0 |
| floor = +3 | +0.0062 [+0.001, +0.012] ✓ | +0.0034 [+0.001, +0.006] ✓ |

→ **Positive in 8/8 point estimates, CI excludes zero in 7/8**; strongest under the 12-month window.
The one soft cell (σ=2 × L2) is the rarest label (base 0.024) under the weaker model — reported, not hidden.

**Onset** across calm ∈ {12, 24, 36} months: null under the dual-family rule in every window (RF and L2
never agree on a nonzero sign; calm-36 L2 alone shows +0.0022 [+0.000, +0.006] while RF is −0.0006 —
consistent with scattered 90%-level noise across 6 cells and with the earlier L2 P=0.92 hint).

### ⭐ Modern-representation arm, Phase 1 (2026-07-26): headline EMBEDDINGS
The "your text was shallow" objection, attacked with real editorial language: 2.1M article URLs
streamed per unit-month (top-12 by coverage, `build_gdelt_urls.py`), slugs parsed → **1.37M unique
human-written headlines**, embedded locally (MiniLM, MPS), narts-weighted mean pool per unit-month,
PCA→32 dims **fitted on pre-2017 months only** (strictly-past representation discipline). Keystone
protocol, 5 families, CIs from saved preds (`run_embedding_ablation.py`, `preds_emb_*.parquet`).

| Comparison (escalation) | RF | L2 |
|---|---|---|
| embeddings-only vs news-counts-only (standalone) | **0.105 vs 0.085** ✓ better | **0.099 vs 0.085** ✓ better |
| embeddings over history | +0.0003 [−0.007,+0.008] ∅ | +0.0021 [−0.001,+0.005] ∅ |
| embeddings beyond history+news-counts | −0.0073 [−0.013,−0.002] (RF dilution) | +0.0006 [−0.002,+0.003] ∅ |

Onset: null in all embedding cells (RF −0.0017 ∅; L2 +0.0004 ∅; "all" L2 +0.0036 P=0.92 ns — the
familiar hint, still ns).

**Phase-1 verdict — subsumption survives its strongest challenger, with a mechanistic refinement:**
1. Neural embeddings are the clearly better *standalone* text representation (+0.015–0.020 over
   machine-coded counts) — the representation ladder points the right way.
2. Yet over event history they add **nothing** (both families null), and beyond history+news-counts
   they add nothing (L2) or actively dilute trees (RF; the known RF-dilution pattern → dual-family
   rule: no claim).
3. **The compositional insight:** the keystone's small significant escalation marginal came from
   GDELT's *count-like* news signals, not semantics. What complements event history is the
   quantitative event-flow content of news; the *semantic* content is what history subsumes.
   "Informative but subsumed" is really a claim about news semantics.

(Analysis note: the runner's printed "emb_over_hist+news" contrast used a zero-filled placeholder
family; the true contrast above was recomputed from saved keystone `history+richtext` predictions —
same protocol, test sets verified identical via matching history AUPRCs.)

## ⭐⭐⭐⭐ ACLED ground truth (India, 2018–2025) — the real deal
Human-*curated* (higher fidelity than machine-coded GDELT; note ~77% still newspaper-sourced, so it is
curated news, not non-news ground truth). 272k events → India only, admin1 (state) × week,
~12,000 out-of-time test state-weeks (rolling origin, min-train 52 wk). This is the strongest and
most robust result — clean ground truth is far more learnable than machine-coded GDELT.

| Label · horizon | Persistence | RandomForest | AUROC (persist→RF) | AUPRC lift |
|---|--:|--:|--:|--:|
| occurrence · t+1 (base 0.82) | 0.925 | 0.988 | 0.816 → 0.951 | +0.063 |
| escalation · t+1 (base 0.16) | 0.182 | **0.239** | 0.559 → **0.648** | **+0.057** |
| escalation · t+3 | 0.161 | **0.244** | 0.516 → **0.664** | **+0.083** |

→ On clean ACLED data the **escalation** forecaster is markedly stronger than on GDELT (GDELT India
escalation lift was +0.026/+0.030, AUROC ~0.585). Here: AUROC 0.65–0.66, +0.06–0.08 AUPRC over
persistence, on ~12k test points — statistically solid, genuine skill on the hard task. Data source
quality matters more than model choice. (ACLED free-tier lags ~1 yr → data ends 2025-07-25.)

## India GDELT (machine-coded news) — 36 states/UTs, 3.1M events

### Occurrence: ≥1 unrest event in a state next week  (base rate ≈ 0.82 → AUROC is the lens)
| Horizon | Model | AUPRC | AUROC |
|---|---|--:|--:|
| t+1 | Persistence | 0.927 | 0.812 |
| t+1 | **RandomForest** | 0.991 | **0.958** |
| t+3 | Persistence | 0.918 | 0.788 |
| t+3 | **RandomForest** | 0.991 | **0.958** |

→ Models **clearly beat persistence** (AUROC 0.81 → 0.96). India isn't saturated, so there's real
signal in *which* states flare.

### Escalation: unrest spikes above the state's own recent norm  (base rate ≈ 0.14 → AUPRC is the lens)
| Horizon | Model | AUPRC | AUROC | AUPRC lift vs persistence |
|---|---|--:|--:|--:|
| t+1 | Persistence | 0.163 | 0.550 | 0.000 |
| t+1 | **RandomForest** | 0.188 | 0.585 | **+0.026** |
| t+3 | Persistence | 0.143 | 0.506 | 0.000 |
| t+3 | **RandomForest** | 0.173 | 0.586 | **+0.030** |

→ The model **beats persistence** on the hard task — modestly (~+0.03 AUPRC), AUROC ~0.585.
Genuine skill above the naive baseline, honestly in the range serious onset models report.

## ⭐ The headline experiment — does news TEXT add skill beyond event history?
Ablation (`--ablation`): history-only vs text-only vs history+text (text = coverage tone +
article-volume signals). **The honest, leakage-free version of the 2022 project's core question.**

| Label · horizon | history only (AUPRC) | history + text (AUPRC) | **marginal of text** |
|---|--:|--:|--:|
| occurrence · t+1 | 0.991 | 0.991 | **−0.000** |
| occurrence · t+3 | 0.991 | 0.991 | **−0.000** |
| escalation · t+1 | 0.184 | 0.188 | **+0.005** |
| escalation · t+3 | 0.172 | 0.173 | **+0.001** |

**Finding: the news-text signal adds essentially nothing once conflict history is controlled.**
Text-*only* is consistently *worse* than history-only (e.g. escalation t+1: 0.159 vs 0.184). This
is the exact inverse of the inflated 0.85 — and it matches the literature: Mueller & Rauh show
newspaper text helps mainly for **onset in *low-history* regions**; every Indian state has rich
event history, so a shallow news signal has nothing to add. **Scope caveat:** this is a *shallow*
text signal (GDELT tone + volume), not deep topic/LLM features, and India is a *high-history*
setting. The deeper LLM/GKG-theme version, and testing in low-history regions, remain open — that's
the honest frontier.

## ⭐⭐ Low-history test — does text help where history is SPARSE? (the Mueller–Rauh hypothesis)
The India ablation was in a high-history setting. The literature says text should help most in
*low-history* regions. So I pulled 6 normally-calmer countries in one GDELT pass
(`gdelt_lowhistory_daily_panel.parquet`, `--cc` slices one out) and re-ran the ablation across the
base-rate spectrum. Occurrence · t+1, target = n_unrest:

| Country | occ. base rate | history-only AUPRC | text-**only** AUPRC | marginal (text on top of history) |
|---|--:|--:|--:|--:|
| **Ecuador (EC)** | **0.14** | 0.653 | **0.661** ← text wins | **+0.009** |
| Georgia (GG) | 0.30 | 0.857 | 0.838 | −0.015 |
| Sri Lanka (CE) | 0.72 | 0.911 | 0.886 | +0.001 |
| India (IN) | 0.82 | 0.991 | 0.990 | −0.000 |
| Kenya (0.96), Bangladesh (0.91) | saturated | — | — | ~0 (like India) |

**Nuanced honest finding — the hypothesis is only *faintly* supported:**
1. Text-only becomes **competitive with / better than** history-only ONLY in the single lowest-history
   country (Ecuador, base rate 0.14). Everywhere else history-only wins. So there IS a directional
   signal: as history gets sparse, text stops being clearly inferior.
2. But the **marginal value of text ON TOP of history is ≈0 across the entire spectrum**
   (−0.03 to +0.01). The two signals are largely redundant — text is a proxy for the same
   "is this place becoming active" that event history already encodes.
3. **Scope caveat (important):** this is a *shallow* text signal (GDELT coverage tone + article
   volume), NOT the rich LDA topic models over full article text that Mueller & Rauh used to get
   onset AUC 0.73–0.82. So the honest claim is narrow: *shallow* news signals add ~no marginal value
   beyond event history, even in low-history settings. Whether *deep* text (GKG themes / LLM-extracted
   risk cues) would clear that bar is the genuinely open question — and the natural next experiment.

## ⭐⭐⭐ Deep text — GKG THEMES (the faithful Mueller–Rauh signal)
Shallow text was tone + article volume. This is the real deep-text signal: GDELT GKG per-article
**thematic tags** (PROTEST, UNREST_*, political-violence, terror, kill, arrest, crisis), parsed from
full article text — the open analog to Mueller & Rauh's LDA news topics. GKG geolocates too sparsely
at province level for small countries (Ecuador ~3 records/day), so tested on **India** (dense: ~5k
GKG records/day, 2.9M article-mentions over 10 months → `gkg_IN_daily_themes.parquet`). 45 lagged
GKG theme features. Ablation `--gkg-ablation`:

| Label · horizon | history only | **GKG themes only** | history + GKG | marginal of GKG |
|---|--:|--:|--:|--:|
| occurrence · t+1 | 0.990 | 0.991 | 0.991 | +0.001 |
| occurrence · t+3 | 0.991 | 0.992 | 0.991 | +0.000 |
| escalation · t+1 | 0.173 | 0.173 | 0.171 | −0.002 |
| escalation · t+3 | 0.177 | **0.196** | 0.181 | +0.004 |

**Finding — two parts:**
1. **Deep text closes the gap to history as a *standalone* predictor** — GKG-only matches history on
   occurrence and BEATS it on escalation t+3 (0.196 vs 0.177). A clear step up from shallow text,
   which was always worse than history. Richer text carries as much signal as the coded event stream.
2. **But GKG themes STILL add ≈0 marginal value on top of history** (−0.002 to +0.004).

**⭐ The deeper insight (why every text experiment nets out at ~zero marginal):** in a **GDELT-only**
pipeline, the "event history" is *itself a news-derived signal* — GDELT events are CAMEO-coded from
the same news articles that GKG themes tag. So "event history" and "news text" are two views of the
**same source**, hence largely redundant: text adds little orthogonal information. Mueller & Rauh got a
positive text result because their *history* came from a curated dataset that is genuinely *empty* in
low-history countries — leaving room for text to contribute. **The faithful test therefore needs a
genuinely non-news history source.** NOTE (per adversarial review): ACLED is only *partly* that — it's
human-*curated* but ~77% newspaper-sourced, so it is higher-fidelity news, not non-news ground truth.
A truly clean test needs structural/administrative (police, NGO field) history — never run here. So the
loop is only *partially* closed; the finale (below) tests text vs curated-news history, which is a step
toward but not the full independent test.

## ⭐⭐⭐⭐⭐ THE FINALE — news text vs ACLED history (tightened + adversarially reviewed)
Does news text add skill over ACLED event history? GKG themes (pulled 2024-10→2025-07 to overlap
ACLED, FIPS→state-name crosswalk, 33 of 36 states) merged into the ACLED panel; rolling-origin
backtest over **~40 weeks / ~24 out-of-time test weeks**. A 4-agent adversarial review then re-ran the
numbers and forced three corrections (below) — this section reflects the corrected reading.

| Label · horizon | history only | GKG text only | history + GKG (RF) | marginal | 90% CI |
|---|--:|--:|--:|--:|--:|
| occurrence · t+1 | 0.991 | 0.983 | 0.991 | −0.000 | — |
| escalation · t+1 | 0.288 | 0.221 | 0.266 | −0.022 | [−0.056, +0.023] |
| escalation · t+3 | 0.291 | 0.204 | 0.248 | −0.043 | [−0.071, −0.011] |

**What survives (robust):** No history+text configuration beats the **best** history-only forecaster
(RandomForest AUPRC ~0.29 on escalation). Text does not improve the best available model. GKG-text-only
is clearly worse than history (0.221 vs 0.288).

**What does NOT survive — corrections from the adversarial review (I verified each):**
1. ❌ **"Text significantly degrades the forecast" was WRONG — an unregularized-RF artifact.** The
   −0.043 / CI-below-zero comes from adding 45 GKG features to a 92-feature RF with only ~136 positives
   (1.48 positives/feature — underpowered). Re-run with **regularized logistic regression**, the t+3
   marginal **flips POSITIVE**: L2 **+0.029**, L1 **+0.024**. So text carries *a little* signal (it helps
   a weaker model); it just can't beat the best history model. Honest claim = "text fails to improve the
   best model," NOT "text degrades forecasting."
2. ❌ **"ACLED is INDEPENDENT of news" was FALSE.** ~77% of ACLED India events are newspaper-sourced
   (Times of India 20.9k, The Hindu 17.7k, …). ACLED is human-*curated* news, not non-news ground truth.
   So the "confirms GDELT circularity / independent history" framing is overstated — the flip vs GDELT
   reflects ACLED's higher *curation fidelity*, not information independence. A clean test of the
   redundancy claim needs **truly non-news history** (police/administrative records, NGO field data),
   which this project never had. That remains genuinely open.
3. ⚠️ **Coverage/selection caveat:** the finale covers 33/36 states; Telangana (rank-5, 6.5% of window
   unrest), Ladakh, Sikkim were systematically dropped (no FIPS code). Worst-case reweighting keeps the
   RF marginal negative (−0.034), but the population was non-randomly selected — scope the claim to the
   33-state subset.

**Corrected conclusion across all text experiments (shallow tone/volume, deep GKG themes, 8 countries
0.14→0.96 base rate, GDELT + ACLED history): news text does not improve on the best history-only
forecaster of unrest. It carries at most weak signal, redundant with the event record for the best
model. The stronger claims ("text hurts", "ACLED proves independence/circularity") do NOT hold. Whether
text helps over a TRULY non-news structural baseline is untested — the real open question.**

## Nigeria (initial method-validation) — 38 provinces, 1.8M events (target = violence)
- **Occurrence** base rate **0.96** (saturated): RF AUROC 0.58 → 0.82, AUPRC near-ceiling.
- **Escalation** t+1: RF **does NOT beat** persistence (AUPRC 0.217 vs 0.224, lift −0.008) —
  reproduces the ViEWS "no model beats no-change" finding. India differs because its cross-state
  heterogeneity (calm south/west vs hot J&K, Chhattisgarh, Manipur) gives history features signal.

## Takeaways
1. Reframed task is a **real, leakage-free forecast** — unlike the 2022 pipeline's circular 0.85.
2. Honest skill is **modest and label-dependent**: strong on occurrence, small-but-real on
   escalation in India, at-baseline on escalation in saturated Nigeria.
3. **Text adds ~nothing beyond history here** — a credible negative result, not a failure.

## Credible next steps
- Swap GDELT → **ACLED** labels (needs the user's key); rerun the identical harness.
- Test the text question in **low-history regions** (where the literature says text *should* help).
- Deeper text: **GDELT GKG themes** or **LLM-extracted** risk cues vs the history-only baseline.
- Real **admin1 adjacency graph** (India state shapefiles) for true spatial-diffusion features.
