---
name: mmm
description: Marketing mix modeling with Google Meridian — input schema, model spec checklist (adstock, saturation, priors), fitting and convergence checks, optimizer interpretation. Use for any MMM specification, fit or budget question.
---

# MMM with Google Meridian

Meridian (`pip install google-meridian`, Python 3.11/3.12, JAX backend; GPU strongly
recommended for fits, CPU fine for small models and for the optimizer/report stages).

## Modeling table (`data/mmm_input.parquet`)

Long format, one row per `geo × time`, weekly unless the workflow says daily:

| column group | examples | notes |
|---|---|---|
| keys | `geo`, `time` | `time` is the period start date; no gaps |
| kpi | `kpi`, `revenue_per_kpi` | the thing we explain |
| media | `<channel>_impressions`, `<channel>_spend` | one pair per paid channel |
| organic | `<channel>_organic` | optional |
| controls | `price`, `promo`, `holiday`, `weather` | non-media drivers |
| population | `population` | for per-capita scaling |

`references/input_schema.md` has the full mapping to `meridian.data.InputData`.

## Model spec checklist (write `artifacts/model-spec.md` and `model/spec.py`)

1. KPI type (revenue vs non-revenue) and whether `revenue_per_kpi` is available.
2. Media channels in, channels dropped and why (spend share < 1 %, missing data).
3. Adstock: max lag per channel (default 8 weeks), geometric decay.
4. Saturation: Hill; note channels where you expect early saturation.
5. Priors: ROI priors per channel from past tests or benchmarks — write the source
   next to every prior. This is where business knowledge enters; be explicit.
6. Controls and their expected sign.
7. Sampling plan: chains, draws, warmup (start 4 × 1000 draws, 500 warmup).

## Fit and diagnostics

- `sample_prior` first; check prior predictive is sane before `sample_posterior`.
- Convergence: R-hat < 1.1 on every parameter, no divergences. Write a table to
  `artifacts/diagnostics.md`; `scripts/check_rhat.py` fails the workflow auto-gate
  if R-hat ≥ 1.1.
- Save `model/posterior.pkl` with `meridian.model.save_mmm`.

## Optimizer and reporting

- Run the `BudgetOptimizer` for at least two scenarios (fixed budget; +10 %).
- Report contribution and ROI **with credible intervals** — `Analyzer.roi()`,
  `Analyzer.incremental_outcome()`.
- Response curves per channel go to `artifacts/figures/`.
