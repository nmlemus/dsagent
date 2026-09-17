# Seattle weather, 2012–2015 — precipitation and temperature by category

## Headline findings

1. Weather categories differ sharply on both axes: snow days are the coldest and (when it snows) the wettest, sun days are the warmest and record zero precipitation, and rain sits in between — despite the name, drizzle and fog days show no measured rainfall at all.
2. Average temperature rose measurably every year from 2012 to 2015, while average daily precipitation stayed flat — treat this as a warming signal, not a change in rainfall.
3. Fog's share of days grew measurably from 2012 to 2015; the apparent declines in rain and snow days are not statistically confirmed with only four years of data, so do not act on them yet.

## Context

**Question asked:** how do precipitation and temperature vary by weather category, and how have they trended over time, in Seattle's daily weather record?

**Data used:** `seattle-weather.csv`, 1,461 rows × 6 columns (`date`, `precipitation`, `temp_max`, `temp_min`, `wind`, `weather`), daily grain, single unbroken series from 2012-01-01 to 2015-12-31 (4 full calendar years, 366/365/365/365 days). No station or location field is present; the file is presumed to be one Seattle station but that is not confirmed. Units for `precipitation`, `temp_max`/`temp_min` and `wind` are inferred (mm, °C, m/s) from typical value ranges, not documented in the source.

**Data quality gate:** PASS — all 9 checks (null share per column, exact duplicates, key uniqueness on `date`, and time-coverage/missing-periods) passed. No column has any nulls, there are no exact duplicate rows, `date` is unique at daily grain, and all 1,461 expected daily periods are present with none missing. The gate carried forward five flags as reading caveats (none breach a threshold) — see "Data quality notes" below.

### What this data cannot answer

- **Whether the warming trend is a real climate signal or noise beyond this window.** The temperature trend is fit on only 4 annual points (2 degrees of freedom); the confidence intervals are correspondingly wide, and a longer series would be needed to say anything about causes or to extrapolate past 2015.
- **Whether rain and snow are actually becoming less frequent.** Both declines are directionally suggestive but their confidence intervals cross zero (rain: 95% CI −9.6 to +1.5 pp/year; snow: −5.1 to +1.6 pp/year) — this dataset cannot distinguish a real decline from four-year sampling noise.
- **Why `weather` and `precipitation` disagree on some days.** 44 of 641 `rain`-labelled days show zero measured precipitation, and no `drizzle`, `fog` or `sun` day ever shows non-zero precipitation. The file gives no way to tell whether `weather` came from a separate label source, a different measurement window, or a coding convention — only that the two columns should not be assumed to agree.
- **Whether this is one station or several, and what the exact units are.** There is no location field and no documented units; the analysis assumes a single Seattle station and mm/°C/m/s based on plausible ranges, not a stated fact.
- **Anything about missing-data mechanisms.** There are zero nulls and zero missing calendar days in this file, so there is nothing to diagnose here — but that also means this profile cannot speak to how the underlying collection process handles gaps, if any exist upstream of this file.

## Evidence

### 1. Weather categories differ sharply in precipitation and temperature

Charts: *"Only rain and snow days record any precipitation; snow is the wettest per day"* and *"Snow days are far colder and sun days far warmer than any other category"*.

- Precipitation is non-zero only on `rain` (641/1,461 days, 43.9%) and `snow` (26/1,461 days, 1.8%) days. Snow days have a higher median (5.45 mm, IQR 3.6–13.5, n=26) than rain days (median 3.3 mm, IQR 1.0–8.6, n=641); `drizzle` (n=53), `fog` (n=101) and `sun` (n=640) days are 0.0 mm on every single day.
- Mean daily max temperature ranges from 5.6 °C on snow days (n=26) to 19.9 °C on sun days (n=640); mean daily min temperature ranges from 0.1 °C (snow) to 9.3 °C (sun). Rain (13.5 °C max / 7.6 °C min, n=641), drizzle (15.9 °C / 7.1 °C, n=53) and fog (16.8 °C / 8.0 °C, n=101) fall in between, all warmer than snow and cooler than sun.

### 2. Temperature rose from 2012 to 2015; precipitation did not

Charts: *"Mean annual temperature rose every year from 2012 to 2015"* and *"Mean daily precipitation shows no clear trend from 2012 to 2015"*.

- Fitting a linear trend on the four annual means (n=4 years, 365–366 days each): mean daily max temperature rose 0.74 °C/year (95% CI 0.43 to 1.05 °C/year, p=0.009), from 15.3 °C in 2012 to 17.4 °C in 2015. Mean daily min temperature rose 0.52 °C/year (95% CI 0.05 to 0.99 °C/year, p=0.042), from 7.3 °C in 2012 to 8.8 °C in 2015. Both trends are directionally consistent — every year is warmer than the last on both series.
- Mean daily precipitation shows no such pattern: 3.35 mm/day (2012, n=366) → 2.27 (2013, n=365) → 3.38 (2014, n=365) → 3.12 mm/day (2015, n=365). The fitted slope is 0.04 mm/day per year (95% CI −1.18 to +1.26, p=0.895) — indistinguishable from flat. Median daily precipitation is 0.0 mm in all four years (precipitation is heavily right-skewed; see Data quality notes).

### 3. Fog days became more common; rain/snow declines are not statistically confirmed

Chart: *"Fog's share of days rose every year while snow nearly disappeared"*.

- Fog's share of days rose from 1.4% (5/366 days, 2012) to 14.2% (52/365 days, 2015). The fitted linear trend is +4.2 percentage points/year (95% CI +1.6 to +6.8 pp/year, p=0.020) — the only category-share trend that clears conventional significance with 4 data points.
- Rain's share fell from 52.2% (191/366, 2012) to 39.5% (144/365, 2015), a fitted slope of −4.1 pp/year, but the 95% CI (−9.6 to +1.5 pp/year, p=0.085) includes zero — no clear trend by this test.
- Snow's share fell from 5.7% (21/366, 2012) to 0.0% (0/365, 2015), a fitted slope of −1.8 pp/year, 95% CI (−5.1 to +1.6 pp/year, p=0.152) — also not statistically confirmed, and snow's per-year counts (21, 3, 2, 0) are too small to trust a rate estimate at yearly grain regardless.

## Data quality notes

- **Gate verdict: PASS.** No column exceeds the 20% null-share threshold (all columns are 0.0% null), exact duplicate rows are 0.0% against a >1% failure threshold, the key column `date` is unique with zero duplicates, and the daily series has zero missing periods across the full 2012–2015 span (1,461/1,461 expected days observed).
- **`precipitation` is heavily right-skewed** (flag, not a failure): 206 of 1,461 rows (14.1%) sit beyond 1.5×IQR. This is typical for daily rainfall — many dry days, a few heavy ones — not a data error, but every average precipitation figure in this report is reported with a median or year-by-year breakdown alongside the mean, never the mean alone.
- **`wind` has moderate outliers**: 34 of 1,461 rows (2.3%) beyond 1.5×IQR, consistent with occasional gustier days; not used as a headline metric in this report.
- **`precipitation` and `weather` disagree on some rows**: all `drizzle`, `fog` and `sun` days (794/1,461, 54.3%) show `precipitation == 0.0`, and 44 of the 641 `rain` days (6.9% of rain days) also show `precipitation == 0.0`. Finding 1's precipitation-by-category numbers use the measured `precipitation` column, not the `weather` label; the two should not be assumed to agree.
- **No units documented** in the source for `wind`, `temp_max`/`temp_min` or `precipitation`; values are consistent with m/s, °C and mm respectively based on typical ranges, but this is inferred, not stated, and is worth confirming before citing these numbers externally.
- **No station/location field** — presumed single Seattle station, not confirmed in the file.
- All time-trend statistics (Findings 2 and 3) are fit on only 4 annual points, leaving 2 degrees of freedom in each regression. Confidence intervals are correspondingly wide, so "not statistically confirmed" above means genuinely inconclusive, not "no effect" — a longer series could resolve these either way.

## Next steps

- **Pull a longer time series** (ideally 10+ years) for the same station before acting on the rain/snow decline or extrapolating the warming trend — four annual points cannot distinguish a real shift from noise for any series with p > 0.05 above.
- **Reconcile `weather` against `precipitation`** with whoever produced the file: find out whether `weather` is sourced from a separate station, a different time window (e.g. a forecast label vs. observed total), or a coding rule, so the 6.9% of "rain" days with zero measured precipitation and the fully-dry drizzle/fog days can be explained rather than caveated.
- **Confirm the units and station identity** with the data source before this file is cited outside the team — the mm/°C/m/s assumption and single-station assumption are both inferences from value ranges, not documented facts.
- **If a rainfall headline number is needed**, report it as median plus IQR or as a year-by-year table, not a single overall mean — the 14.1% of days beyond 1.5×IQR pull the mean well above what a typical day looks like.
