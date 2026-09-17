# Data gate — `seattle-weather.csv`

Source: `artifacts/data-profile.json` (1,461 rows × 6 columns, `date` at daily
grain, 2012-01-01 to 2015-12-31). No values were recomputed here; every number
below is read directly from the profile.

## Checks

| Check | Value | Threshold | Result |
|---|---|---|---|
| Null share — date | 0.0% (0/1461) | > 20% fails | PASS |
| Null share — precipitation | 0.0% (0/1461) | > 20% fails | PASS |
| Null share — temp_max | 0.0% (0/1461) | > 20% fails | PASS |
| Null share — temp_min | 0.0% (0/1461) | > 20% fails | PASS |
| Null share — wind | 0.0% (0/1461) | > 20% fails | PASS |
| Null share — weather | 0.0% (0/1461) | > 20% fails | PASS |
| Exact duplicate rows | 0.0% (0/1461) | > 1% fails | PASS |
| Key uniqueness (`date`) | `unique: true`, 0 duplicates (0/1461) | any violation fails | PASS |
| Time coverage / missing periods | 0 missing days — 1,461/1,461 expected daily periods observed (2012-01-01 to 2015-12-31) | any missing period inside the requested range fails | PASS |

No key column ambiguity here: `key_uniqueness.column` is `date` (not `null`),
so the key check above is a real check, not a "not applicable" placeholder.

## Flags (do not breach a threshold, but affect how the data should be read)

- **`precipitation` is heavily right-skewed**: 206 of 1,461 rows (14.1%) sit
  beyond 1.5×IQR. This is typical for daily rainfall (many dry days, a few
  heavy ones) rather than a data error, but any headline "average
  precipitation" number should be reported with the median or a distribution
  alongside the mean, not the mean alone.
- **`wind` has moderate outliers**: 34 of 1,461 rows (2.3%) beyond 1.5×IQR —
  consistent with occasional gustier days, not flagged as an error.
- **`precipitation` and `weather` disagree on some rows**: all `drizzle`,
  `fog`, and `sun` days (794/1,461, 54.3%) have `precipitation == 0.0`, and 44
  of the 641 `rain` days (6.9% of rain days) also show `precipitation == 0.0`.
  A reader combining the two columns should not assume they always agree —
  this looks like `weather` came from a separate label source, not a
  precipitation error.
- **No units documented** for `wind`, `temp_max`/`temp_min`, or
  `precipitation` in the source file; values are consistent with m/s, °C and
  mm respectively but this is inferred, not stated.
- **No station/location field** — presumed single Seattle station, not
  confirmed in the file.

None of the above breaches a gate threshold; they are carried forward as
caveats for the analysis step, not reasons to fail this gate.

## Verdict

All 9 checks pass: no column exceeds the 20% null-share threshold, exact
duplicate rows are 0.0% (threshold > 1%), the declared key `date` is unique
with zero duplicates, and the daily time series has zero missing periods
across its full 2012–2015 span.

GATE: PASS
