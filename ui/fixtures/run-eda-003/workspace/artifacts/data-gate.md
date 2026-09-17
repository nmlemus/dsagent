# Data gate — `seattle-weather.csv`

Source: `artifacts/data-profile.json` (rows = 1,461, columns = 6). No recomputation was
performed; all values below are read directly from the profile.

## Gate checks

| Check | Value | Threshold (fail when) | Denominator | Result |
|---|---|---|---|---|
| Null share — `date` | 0.0 % | > 20 % | 1,461 rows | PASS |
| Null share — `precipitation` | 0.0 % | > 20 % | 1,461 rows | PASS |
| Null share — `temp_max` | 0.0 % | > 20 % | 1,461 rows | PASS |
| Null share — `temp_min` | 0.0 % | > 20 % | 1,461 rows | PASS |
| Null share — `wind` | 0.0 % | > 20 % | 1,461 rows | PASS |
| Null share — `weather` | 0.0 % | > 20 % | 1,461 rows | PASS |
| Exact duplicate rows | 0.0 % (0 rows) | > 1 % | 1,461 rows | PASS |
| Key uniqueness — declared key `date` | `unique: true`, 0 duplicates | any violation fails | 1,461 rows / 1,461 distinct dates | PASS |
| Missing time periods (daily grain, 2012-01-01 to 2015-12-31) | 0 missing days (1,461 expected, 1,461 observed) | any missing period inside requested range fails | 1,461 expected calendar days | PASS |

No columns are exempt: all six columns in the profile have a reported null share, the
one declared key (`date`) is unique, and the daily calendar is complete over the full
2012–2015 span, so every applicable threshold in the `eda` skill was evaluated.

## Flags (non-blocking, carry forward as caveats)

- **`precipitation` right-skew / outliers**: 206 of 1,461 rows (14.1 %) sit beyond the
  1.5×IQR fence (upper bound ≈ 7.0 mm; max 55.9 mm). Not a threshold breach (outliers
  are not a gate check), but means will overstate a "typical" day — prefer median or a
  log scale when summarizing.
- **`wind` outliers**: 34 of 1,461 rows (2.3 %) exceed the IQR fence (upper bound ≈ 6.7;
  max 9.5). Milder tail than precipitation but still worth noting on any wind chart.
- **`weather` category drift over time**: the 5-category mix shifts year over year (e.g.
  `snow` 21 days in 2012 → 0 in 2015; `drizzle` 31 → 7 of ~365 days/year), which may
  reflect a change in observer/instrument classification rather than a real climate
  shift. Any year-over-year comparison of weather categories should note this.
- **No station/location metadata**: the file does not document which Seattle station
  produced the readings, so findings should be scoped to "this station's record," not
  generalized to "Seattle."
- **Single categorical label per day**: `weather` cannot represent mixed conditions
  (e.g., rain and fog on the same day), so secondary conditions are undercounted by
  construction.
- **Row-level consistency not checked here**: the profile does not confirm `temp_max ≥
  temp_min` per row; this should be verified before any temperature-range analysis.

## Verdict

All four applicable gate thresholds (null share, exact duplicates, key uniqueness,
missing time periods) pass against the full 1,461-row / 6-column dataset with no
exemptions. The flags above do not block the gate but should travel with the data into
the analysis step as caveats.

GATE: PASS
