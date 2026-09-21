# Test fixtures

## `seattle-weather.csv`

Daily Seattle weather, 2012-01-01 → 2015-12-31.

- **Source**: [vega-datasets](https://github.com/vega/vega-datasets) —
  `data/seattle-weather.csv`, fetched 2026-09-16.
- **License**: BSD-3-Clause (vega-datasets). Derived from NOAA GHCN-Daily,
  which is public domain.
- **sha256**: `0845078a290b48e3149ab8639966824110a251db4e06fc144c06ebb534af23be`
- **Shape**: 1461 rows × 6 columns —
  `date, precipitation, temp_max, temp_min, wind, weather`.

Vendored rather than downloaded so the integration run is deterministic and
offline: a network failure should never look like a harness failure.

## `brief-lanzamiento-ejemplo.md`

A campaign brief for the `marketing` cartridge's `campaign-brief` workflow, to
upload as that workflow's `brief_path` when trying the UI by hand.

- **Source**: written for this repository. Cuadra, its numbers and its team are
  invented; no real company, client or engagement is described.
- **Shape**: the six things the `briefs` skill extracts, plus the three things
  it is meant to catch — an objective nobody has a baseline for, a claim legal
  forbids, and channels that were decided before the strategy existed.

It is deliberately imperfect: a brief with no holes would let the research step
report "nothing missing" and prove nothing about the workflow.

Why this dataset for `eda-to-report`: it is the smallest public CSV that
exercises all four gate checks in the `eda` skill at once — a `date` column
unique at the declared grain (key uniqueness), four complete years with no
missing days (time coverage), no empty cells (null share) and no repeated rows
(duplicates) — plus a categorical (`weather`, 5 values) and four numerics with
real outliers for the analysis step. It is clean by construction, so the
expected gate verdict is `GATE: PASS`; anything else is a prompt problem, not a
data problem.
