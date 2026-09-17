# Data profile — `seattle-weather.csv`

Rows: 1,461 · Columns: 6 · Exact duplicates: 0.0 % · Key `date`: 0 duplicate(s)

| column | dtype | null % | distinct | notes |
|---|---|---|---|---|
| date | str | 0.0 | 1461 | top: 2012-01-01 (1), 2012-01-02 (1), 2012-01-03 (1), 2012-01-04 (1), 2012-01-05 (1) |
| precipitation | float64 | 0.0 | 111 | min 0, max 55.9, IQR outliers 206 |
| temp_max | float64 | 0.0 | 67 | min -1.6, max 35.6, IQR outliers 0 |
| temp_min | float64 | 0.0 | 55 | min -7.1, max 18.3, IQR outliers 0 |
| wind | float64 | 0.0 | 79 | min 0.4, max 9.5, IQR outliers 34 |
| weather | str | 0.0 | 5 | top: rain (641), sun (640), fog (101), drizzle (53), snow (26) |

## Time coverage

- `date` spans **2012-01-01 to 2015-12-31** (4 full calendar years, including the
  2012 leap day), daily granularity.
- Expected daily periods over that span: 1,461. Observed: 1,461. **No missing
  days** and no duplicate dates — the key column is unique at daily grain.
- Per-year row counts: 2012 = 366, 2013 = 365, 2014 = 365, 2015 = 365 — matches
  the calendar exactly, so there is no partial year at either end to caveat.

## Distribution notes

- **precipitation** (mm/day): heavily right-skewed — median is far below the
  mean of 3.03, and 206 of 1,461 rows (14.1 %) sit beyond 1.5×IQR. This is
  expected for daily rainfall (many dry days, a few heavy ones), not a data
  error, but any average precipitation figure should be reported with the
  median or a distribution alongside it, not the mean alone.
- **wind** (m/s, presumably — units not stated in the source): 34 rows (2.3 %)
  beyond 1.5×IQR, consistent with occasional gustier days.
- **temp_max / temp_min** (°C): no IQR outliers; both look like clean,
  well-behaved seasonal series with max always ≥ min (checked: 0 violations).
- **weather** is a 5-level categorical (`rain`, `sun`, `fog`, `drizzle`,
  `snow`) with no nulls or unexpected labels.

## Cross-column consistency (worth flagging, not a defect)

- `precipitation` is **exactly 0.0** for every `drizzle`, `fog`, and `sun` day
  (all 794 of them) — those categories carry no measured rainfall in this
  dataset even though "drizzle" and "fog" sound like they should. Only `rain`
  and `snow` days ever have non-zero precipitation.
- 44 of the 641 `rain` days (6.9 %) have `precipitation == 0.0` — a `weather`
  label of "rain" with no measured rainfall that day. This may reflect how the
  categorical label was assigned (e.g. from an external source rather than
  the precipitation column itself) rather than a measurement error, but a
  reader combining the two columns should not assume they always agree.
- No exact duplicate rows and no duplicate `date` values, so the key column
  is trustworthy as a daily grain identifier.

## What this data cannot answer

- No station/location field — this is presumably a single Seattle station,
  but that is not confirmed in the file itself.
- No units are documented in the source for `wind`, `temp_max`/`temp_min`, or
  `precipitation`; values are consistent with m/s, °C and mm respectively
  based on typical ranges, but this is an inference, not a stated fact.
- No missing-data mechanism to investigate (there are no nulls), so this
  profile cannot say anything about imputation or data collection gaps beyond
  "there are none observed."
