# Precipitation and Temperature by Weather Category, 2012–2015

## Headline findings

1. Precipitation is effectively a rain/snow signal: `rain` and `snow` days carry almost
   all recorded precipitation, while `sun`, `fog` and `drizzle` days show a median of
   0 mm — treat `weather` as a strong proxy for "wet vs. dry," not a graded scale.
2. Temperature separates the categories cleanly: `snow` days average far colder
   (mean max 5.6 °C, mean min 0.1 °C) than `sun` days (mean max 19.9 °C, mean min
   9.3 °C), with `fog`, `drizzle` and `rain` clustered in between.
3. Both daily temperature extremes rose from 2012 to 2015 (season-adjusted trend:
   +0.73 °C/year for `temp_max`, +0.51 °C/year for `temp_min`, both 95% CIs exclude
   zero); precipitation shows no clear trend over the same period (+0.04 mm/year,
   95% CI includes zero).

## Context

**Question asked:** how do precipitation and temperature vary by the recorded
`weather` category, and did either variable trend over 2012–2015?

**Data used:** `seattle-weather.csv`, a single daily record with 6 columns —
`date`, `precipitation` (mm), `temp_max` (°C), `temp_min` (°C), `wind` (m/s, presumed),
`weather` (5-category label: rain, sun, fog, drizzle, snow) — covering 1,461 daily
rows from 2012-01-01 to 2015-12-31 (4 full calendar years, no gaps, no duplicate
dates). The data quality gate passed on all four applicable checks (null share,
exact duplicates, key uniqueness, missing time periods) with no exemptions; see
`artifacts/data-gate.md` for the full check table.

### What this data cannot answer

- **Whether this generalizes beyond one station.** There is no station or location
  field in the file. All findings are scoped to "this station's daily record," not
  to "Seattle" as a whole.
- **Sub-daily patterns.** Precipitation, wind and temperature are daily aggregates,
  so a short, heavy downpour and a day of steady light rain can look identical, and
  within-day extremes are invisible.
- **Mixed weather conditions.** `weather` is one label per day. A day coded `sun` or
  `fog` may still have had some precipitation that its category framing hides — the
  precipitation *column* is unaffected by this, but the *category comparison* in
  Finding 1 undercounts secondary conditions.
- **Whether the category mix itself is a climate signal.** The `weather` label
  distribution drifts across years in a way that looks like classification drift —
  e.g. `snow` falls from 21 days in 2012 to 2 in 2015, `drizzle` from 31 to 7 of
  ~365 days/year — with no metadata documenting instrument or observer changes. This
  report does not use year-over-year category counts as evidence of anything; it
  reads categories only in aggregate across the full 2012–2015 window.
- **Causality for the temperature trend.** The +0.73 °C/year and +0.51 °C/year
  season-adjusted slopes describe this station's 2012–2015 record only; four years of
  daily data cannot establish a long-run climate trend, only that these years moved in
  this direction.

## Evidence

### 1. Precipitation differs sharply by weather category

![Precipitation by weather category](../artifacts/figures/fig1_precip_by_category.png)

Median daily precipitation is 0.0 mm for `sun` (n=640/1,461), `fog` (n=101/1,461) and
`drizzle` (n=53/1,461) — even the 75th percentile is 0.0 mm for all three, so
three-quarters of days in those categories record no measurable rain. `rain` days
(n=641/1,461) have a median of 3.3 mm (mean 6.6 mm, IQR 1.0–8.6 mm), and `snow` days
(n=26/1,461) have the highest median of any category, 5.45 mm (mean 8.6 mm, IQR
3.6–13.5 mm; this is water-equivalent precipitation recorded on a snow day, not snow
depth). Together `rain` and `snow` are 45.7 % of days (667 of 1,461) but account for
essentially all recorded precipitation volume; the other 54.3 % (794 of 1,461) are
effectively dry by this measure.

### 2. Temperature differs sharply by weather category

![Temperature by weather category](../artifacts/figures/fig2_temp_by_category.png)

Mean `temp_max` by category (± 95% CI): `sun` 19.9 ± 0.6 °C (n=640), `fog` 16.8 ±
1.3 °C (n=101), `drizzle` 15.9 ± 2.4 °C (n=53), `rain` 13.5 ± 0.4 °C (n=641), `snow`
5.6 ± 1.2 °C (n=26). Mean `temp_min` follows the same order: `sun` 9.3 ± 0.4 °C, `fog`
8.0 ± 1.0 °C, `drizzle` 7.1 ± 1.7 °C, `rain` 7.6 ± 0.3 °C, `snow` 0.1 ± 0.9 °C. The
`sun`–`snow` gap is about 14 °C on both series and the 95% CIs do not overlap, so this
separation is robust, not noise. `rain`, `fog` and `drizzle` overlap each other within
their CIs and should be read as "similar, intermediate," not individually distinguishable.

### 3. Temperature rose 2012–2015; precipitation did not

![Temperature trend, season-adjusted](../artifacts/figures/fig3_temp_trend.png)
![Precipitation trend, season-adjusted](../artifacts/figures/fig4_precip_trend.png)

Trend model: daily OLS of each variable on year (centered at 2012) plus a day-of-year
sine/cosine pair to remove the seasonal cycle, HC3 robust standard errors, n=1,461
daily observations. `temp_max` rose +0.732 °C/year (95% CI [0.565, 0.899], p < 0.001)
— a cumulative season-adjusted shift of about +2.2 °C from 2012 to 2015, matching the
raw annual-mean gap (15.28 °C in 2012 → 17.43 °C in 2015, +2.15 °C, n=366 and n=365
days respectively). `temp_min` rose +0.510 °C/year (95% CI [0.393, 0.627], p < 0.001)
— cumulative shift about +1.53 °C (raw annual means: 7.29 °C → 8.84 °C, +1.55 °C).
`precipitation` moved +0.044 mm/year (95% CI [-0.267, 0.355], p = 0.78) — the interval
straddles zero, so this reads as no clear trend, not a small increase; raw annual
means bounce between years (3.35, 2.27, 3.38, 3.12 mm in 2012–2015, n≈365/year)
without a monotonic direction.

## Data quality notes

The gate passed all four applicable checks against the full 1,461-row / 6-column
dataset (see `artifacts/data-gate.md`): 0.0 % nulls in every column, 0.0 % exact
duplicate rows, a unique `date` key with 0 duplicates, and a complete daily calendar
from 2012-01-01 to 2015-12-31 with 0 missing days out of 1,461 expected. No columns
were exempted from these checks.

Non-blocking flags carried forward from the gate:

- **`precipitation` right-skew**: 206 of 1,461 rows (14.1 %) sit beyond the 1.5×IQR
  fence (upper bound ≈ 7.0 mm, max 55.9 mm). This is treated as genuine rain-event
  variability, not an error, but means overstate a "typical" day — the analysis above
  uses medians or CI-bounded means for this reason.
- **`wind` outliers**: 34 of 1,461 rows (2.3 %) exceed the IQR fence (upper bound ≈
  6.7, max 9.5) — a milder tail than precipitation; `wind` was not used in the
  headline findings but would need the same care if analyzed.
- **`weather` category drift over time**: the 5-category mix shifts year over year
  (e.g. `snow` 21 days in 2012 → 2 in 2015; `drizzle` 31 → 7 of ~365 days/year),
  consistent with classification drift rather than a real climate shift. Category-level
  comparisons above (Findings 1–2) pool all four years and describe categories "as
  recorded"; no year-over-year category comparison is used as evidence.
- **No station/location metadata**: findings are scoped to this single station's
  record, not generalized to "Seattle."
- **Single categorical label per day**: `weather` cannot represent mixed conditions,
  so secondary conditions (e.g. fog with light rain) are undercounted by construction.
- **Row-level `temp_max ≥ temp_min` consistency was not verified** in the profile or
  gate; this should be confirmed before any temperature-range (max − min) analysis.

## Next steps

With one more week, in priority order:

1. **Verify `temp_max ≥ temp_min` row-by-row** — this was flagged but not checked, and
   any temperature-range analysis depends on it.
2. **Get station/location metadata** (or confirm there is only one station) so
   findings can be scoped or generalized correctly, and check whether an
   instrument/observer change coincides with the `weather`-category drift
   (`snow`/`drizzle` shares falling since 2012) before treating that drift as
   meaningless.
3. **Re-run the temperature trend on more years of data if available** — four years is
   enough to detect a directional slope with a tight CI here, but not enough to call
   it a durable climate trend; extending the series (or sourcing a second nearby
   station) would test whether +0.7 °C/year for `temp_max` persists or was specific to
   2012–2015.
4. **Model precipitation on a log or zero-inflated scale** given the 14.1 % right-skew
   flag, if any future work needs a predictive (not just descriptive) precipitation
   model.
5. **Source sub-daily or hourly data**, if available, to check whether the daily
   `weather` label is hiding meaningful within-day variation (e.g., a "sun" day with a
   short afternoon shower).

## Appendix

- **Row/column counts**: 1,461 rows × 6 columns. Time coverage: 2012-01-01 to
  2015-12-31 (1,461 expected daily periods, 1,461 observed, 0 missing, 0 duplicate
  dates).
- **Category counts** (n / 1,461, share of total): `rain` 641 (43.9 %), `sun` 640
  (43.8 %), `fog` 101 (6.9 %), `drizzle` 53 (3.6 %), `snow` 26 (1.8 %).
- **Precipitation by category** (mm/day): `sun` median 0.0, `fog` median 0.0,
  `drizzle` median 0.0 (all three: 75th pctile 0.0); `rain` median 3.3, mean 6.6,
  IQR 1.0–8.6; `snow` median 5.45, mean 8.6, IQR 3.6–13.5.
- **Temperature by category** (°C, mean ± 95% CI): see Finding 2 for the full table.
- **Trend model**: daily OLS, year (centered 2012) + sine/cosine day-of-year terms,
  HC3 robust SEs, n=1,461. `temp_max` +0.732 [0.565, 0.899] °C/yr, p<0.001; `temp_min`
  +0.510 [0.393, 0.627] °C/yr, p<0.001; `precipitation` +0.044 [-0.267, 0.355] mm/yr,
  p=0.78.
- **Full methodology and caveats**: `artifacts/findings.md`, `artifacts/data-profile.md`,
  `artifacts/data-gate.md`.
</content>
