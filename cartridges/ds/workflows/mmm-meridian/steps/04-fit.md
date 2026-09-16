Fit the model defined in `model/spec.py`.

1. `sample_prior` and sanity-check the prior predictive against the observed KPI
   range; note anything absurd in `artifacts/diagnostics.md`.
2. `sample_posterior` with the sampling plan from the spec. If R-hat is poor,
   increase draws once (×2) and refit; if still poor, stop and explain.
3. Save the model with `meridian.model.save_mmm` to `model/posterior.pkl`.
4. Write `artifacts/diagnostics.md` (R-hat table, divergences, effective sample
   sizes, trace notes) and `artifacts/diagnostics.json` with the shape
   `{"rhat_max": float, "divergences": int, "params": {name: rhat}}` — the
   workflow's auto gate reads it.
