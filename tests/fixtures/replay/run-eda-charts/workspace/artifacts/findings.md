# Findings — precipitation and temperature by weather category, 2012–2015

## Headline findings

1. Weather categories differ sharply on both axes: snow days are the coldest and (when it snows) the wettest, sun days are the warmest and record zero precipitation, and rain sits in between — despite the name, drizzle and fog days show no measured rainfall at all.
2. Average temperature rose measurably every year from 2012 to 2015, while average daily precipitation stayed flat — treat this as a warming signal, not a change in rainfall.
3. Fog's share of days grew measurably from 2012 to 2015; the apparent declines in rain and snow days are not statistically confirmed with only four years of data, so do not act on them yet.

## Evidence

### 1. Weather categories differ sharply in precipitation and temperature

Charts: *"Only rain and snow days record any precipitation; snow is the wettest per day"* and *"Snow days are far colder and sun days far warmer than any other category"*.

- Precipitation is non-zero only on `rain` (641/1,461 days, 43.9%) and `snow` (26/1,461 days, 1.8%) days. Snow days have a higher median (5.45 mm, IQR 3.6–13.5, n=26) than rain days (median 3.3 mm, IQR 1.0–8.6, n=641); `drizzle` (n=53), `fog` (n=101) and `sun` (n=640) days are 0.0 mm on every single day.
- Mean daily max temperature ranges from 5.6 °C on snow days (n=26) to 19.9 °C on sun days (n=640); mean daily min temperature ranges from 0.1 °C (snow) to 9.3 °C (sun). Rain (13.5 °C max / 7.6 °C min, n=641), drizzle (15.9 °C / 7.1 °C, n=53) and fog (16.8 °C / 8.0 °C, n=101) fall in between, all warmer than snow and cooler than sun.

### 2. Temperature rose from 2012 to 2015; precipitation did not

Charts: *"Mean annual temperature rose every year from 2012 to 2015"* and *"Mean daily precipitation shows no clear trend from 2012 to 2015"*.

- Fitting a linear trend on the four annual means (n=4 years, 365–366 days each): mean daily max temperature rose 0.74 °C/year (95% CI 0.43 to 1.05 °C/year, p=0.009), from 15.3 °C in 2012 to 17.4 °C in 2015. Mean daily min temperature rose 0.52 °C/year (95% CI 0.05 to 0.99 °C/year, p=0.042), from 7.3 °C in 2012 to 8.8 °C in 2015. Both trends are directionally consistent — every year is warmer than the last on both series.
- Mean daily precipitation shows no such pattern: 3.35 mm/day (2012, n=366) → 2.27 (2013, n=365) → 3.38 (2014, n=365) → 3.12 mm/day (2015, n=365). The fitted slope is 0.04 mm/day per year (95% CI −1.18 to +1.26, p=0.895) — indistinguishable from flat. Median daily precipitation is 0.0 mm in all four years (precipitation is heavily right-skewed; see caveats).

### 3. Fog days became more common; rain/snow declines are not statistically confirmed

Chart: *"Fog's share of days rose every year while snow nearly disappeared"*.

- Fog's share of days rose from 1.4% (5/366 days, 2012) to 14.2% (52/365 days, 2015). The fitted linear trend is +4.2 percentage points/year (95% CI +1.6 to +6.8 pp/year, p=0.020) — the only category-share trend that clears conventional significance with 4 data points.
- Rain's share fell from 52.2% (191/366, 2012) to 39.5% (144/365, 2015), a fitted slope of −4.1 pp/year, but the 95% CI (−9.6 to +1.5 pp/year, p=0.085) includes zero — no clear trend by this test.
- Snow's share fell from 5.7% (21/366, 2012) to 0.0% (0/365, 2015), a fitted slope of −1.8 pp/year, 95% CI (−5.1 to +1.6 pp/year, p=0.152) — also not statistically confirmed, and snow's per-year counts (21, 3, 2, 0) are too small to trust a rate estimate at yearly grain regardless.

## Caveats

- All time trends above are fit on only 4 annual points (2012–2015); the regression has 2 degrees of freedom, so confidence intervals are wide and the rain/snow trend tests correctly come back inconclusive rather than "no effect" — a longer series could resolve them either way.
- Precipitation is heavily right-skewed (14.1% of all days sit beyond 1.5×IQR, per the gate report): the mean figures quoted above (e.g. 3.03 mm/day overall) are pulled up by a few heavy-rain days, which is why medians are reported alongside them and why every category's median away from rain/snow is 0.0 even where the mean is not.
- `precipitation` and `weather` disagree on some rows: 44 of 641 `rain` days (6.9% of rain days) have `precipitation == 0.0`, and no `drizzle`/`fog`/`sun` day ever has non-zero precipitation, even though "drizzle" and "fog" sound wet. Finding 1's precipitation-by-category numbers reflect the measured `precipitation` column, not the `weather` label, and the two should not be assumed to agree.
- Units for temperature and precipitation are inferred (°C, mm) from typical ranges, not documented in the source file; the direction and magnitude of the trends above hold regardless of the exact unit, but this is worth confirming before citing externally.
