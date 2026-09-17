# Findings — Precipitation and temperature by weather category, 2012–2015

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

## Evidence

### 1. Precipitation differs sharply by weather category

Chart: `artifacts/figures/fig1_precip_by_category.png`

- Median daily precipitation is 0.0 mm for `sun` (n=640/1,461), `fog` (n=101/1,461)
  and `drizzle` (n=53/1,461) — the 75th percentile is also 0.0 mm for all three, so
  even three-quarters of days in those categories record no measurable rain.
- `rain` days (n=641/1,461) have a median of 3.3 mm and a mean of 6.6 mm (IQR 1.0–8.6 mm).
- `snow` days (n=26/1,461) have the highest median precipitation of any category,
  5.45 mm (mean 8.6 mm, IQR 3.6–13.5 mm) — this is water-equivalent precipitation
  recorded on a snow day, not snow depth.
- Together, `rain` and `snow` are 45.7 % of days (667 of 1,461) but account for
  essentially all recorded precipitation volume; the other 54.3 % (794 of 1,461)
  are effectively dry by this measure.

### 2. Temperature differs sharply by weather category

Chart: `artifacts/figures/fig2_temp_by_category.png`

- Mean `temp_max` by category (± 95% CI, n as above): `sun` 19.9 ± 0.6 °C (n=640),
  `fog` 16.8 ± 1.3 °C (n=101), `drizzle` 15.9 ± 2.4 °C (n=53), `rain` 13.5 ± 0.4 °C
  (n=641), `snow` 5.6 ± 1.2 °C (n=26).
- Mean `temp_min` follows the same order: `sun` 9.3 ± 0.4 °C, `fog` 8.0 ± 1.0 °C,
  `drizzle` 7.1 ± 1.7 °C, `rain` 7.6 ± 0.3 °C, `snow` 0.1 ± 0.9 °C.
- The `sun`–`snow` gap is about 14 °C on both the max and min series and the 95% CIs
  do not overlap, so this is a robust separation, not noise. `rain`, `fog` and
  `drizzle` overlap each other within their CIs and should be read as "similar,
  intermediate," not individually distinguishable from one another.

### 3. Temperature rose 2012–2015; precipitation did not

Charts: `artifacts/figures/fig3_temp_trend.png`, `artifacts/figures/fig4_precip_trend.png`

- Trend model: daily OLS of each variable on year (centered at 2012) plus a
  day-of-year sine/cosine pair to remove the seasonal cycle, HC3 robust SEs, n=1,461
  daily observations.
- `temp_max`: +0.732 °C/year, 95% CI [0.565, 0.899], p < 0.001 — a cumulative
  season-adjusted shift of about +2.2 °C over the 3 years from 2012 to 2015, matching
  the raw annual-mean gap (15.28 °C in 2012 → 17.43 °C in 2015, +2.15 °C, n=366 and
  365 days respectively).
- `temp_min`: +0.510 °C/year, 95% CI [0.393, 0.627], p < 0.001 — cumulative
  season-adjusted shift about +1.53 °C (raw annual means: 7.29 °C in 2012 → 8.84 °C
  in 2015, +1.55 °C).
- `precipitation`: +0.044 mm/year, 95% CI [-0.267, 0.355], p = 0.78 — the interval
  straddles zero, so this reads as no clear trend, not a small increase. Raw annual
  means bounce between years (3.35, 2.27, 3.38, 3.12 mm in 2012–2015, n≈365/year)
  without a monotonic direction.

## Caveats

- `precipitation` is right-skewed (206 of 1,461 rows, 14.1 %, beyond the 1.5×IQR
  fence); category and trend comparisons above use medians or CI-bounded means, not
  raw means alone, for this reason.
- The mix of `weather` categories shifts year over year in ways that look like
  classification drift rather than climate signal — e.g. `snow` falls from 21 days
  in 2012 to 2 in 2015, and `drizzle` from 31 to 7 of ~365 days/year — so the
  category-level comparisons in Finding 1–2 describe the categories as recorded, and
  any year-over-year read of category counts specifically (not attempted as a
  headline finding here) should not be taken as a real climate shift.
- There is no station/location metadata in the file, so all findings are scoped to
  this single station's daily record, not "Seattle" generally.
- `weather` is one label per day and cannot capture mixed conditions, so a day coded
  `fog` or `sun` may still have had some precipitation not reflected in that day's
  category framing (though the precipitation column itself is unaffected by this).
