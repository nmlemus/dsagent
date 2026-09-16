---
name: ml
description: Supervised ML practice for this team — problem framing, split strategy, leakage checks, baseline-first modeling, metrics with uncertainty and the model card. Use for any predictive modeling task.
---

# ML

## Before training

1. Write down: target, unit of analysis, prediction time (what is known when),
   and how the model will be used. Put it at the top of `artifacts/model-card.md`.
2. Split strategy follows the use: time-based for anything forecast-like, grouped
   by entity when rows share an entity, plain stratified otherwise. Never random
   split on time series.
3. Leakage sweep: any feature computed with information after prediction time,
   any target-derived feature, any ID that encodes the label.

## Modeling

- Baseline first (mean/mode, last value, or a linear model) — every later model is
  reported as a delta against it.
- Prefer gradient-boosted trees or regularised linear models; deep models only when
  the data size and modality justify it.
- Cross-validate with the split strategy above; report mean ± std of the metric.
- Save the fitted model under `model/`, metrics under `artifacts/metrics.json`.

## Model card (`artifacts/model-card.md`)

Problem framing · data (source, period, rows, split) · features and exclusions
(with leakage reasoning) · models compared with metrics ± uncertainty · chosen
model and why · known failure modes · how to monitor.
