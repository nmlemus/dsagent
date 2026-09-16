# Findings — Precipitation and temperature by weather category, Seattle 2012–2015

## 1. Headline findings

1. **Weather category is a reliable temperature proxy** — mean daily max temperature falls in a step-wise ladder from sun (19.9°C) through fog (16.8°C) and drizzle (15.9°C) to rain (13.5°C) and snow (5.6°C), so `weather` can substitute for temperature in quick screens.
2. **Precipitation-sensitive planning only needs two labels** — rain (6.6 mm/day, n=641) and snow (8.6 mm/day, n=26) are the sole categories with any recorded rainfall, while sun, fog and drizzle (794 days combined, out of 1,461) read exactly 0 mm every day; treat the other three labels as dry by default.
3. **Budget for continued warming, not more rain** — daily max and min temperature rose about 0.7°C/year and 0.5°C/year respectively from 2012 to 2015, with 95% confidence intervals that exclude zero, while daily precipitation showed no such trend (slope 0.04 mm/year, 95% CI crosses zero).

## 2. Context

**Question asked:** does the recorded `weather` category track temperature and precipitation closely enough to use as a shorthand, and has Seattle's climate shifted over 2012–2015?

**Data used:** `data/seattle-weather.csv`, a single daily Seattle weather series — 1,461 rows × 6 columns (`date`, `precipitation`, `temp_max`, `temp_min`, `wind`, `weather`), covering 2012-01-01 to 2015-12-31 with zero missing calendar days. Full profiling detail is in `artifacts/data-profile.md` and the pass/fail checks are in `artifacts/data-gate.md` (verdict: **GATE PASS**, no blocking issues).

**What this data cannot answer** (see Section 4 for full detail):
- Whether these patterns hold anywhere other than Seattle — there is no station/location column, so this is one city's climate only.
- Anything at sub-daily resolution — the data is daily aggregates only, so within-day precipitation timing or temperature swings are invisible.
- Whether `weather` was assigned by a human observer or an automated system, or whether that method changed across the four years — no labeling metadata is provided, so apparent category "reliability" cannot be separated from labeling-process quirks.
- A strict rain/no-rain split from `precipitation` alone that matches the `weather` label exactly — the two fields disagree on 70 of 1,461 rows (4.8%).
- Anything about `snow` or `drizzle` with tight confidence — these classes have only 26 and 53 days respectively, so single extreme days move their means noticeably.

## 3. Evidence

### Finding 1 — temperature ladder across weather categories
![Mean daily max temperature by weather category](../artifacts/figures/01_temp_by_category.png)

Mean daily max temperature by category, with 95% bootstrap CI, out of 1,461 total days:

| Category | Mean max temp | 95% CI | n / 1,461 |
|---|---|---|---|
| sun | 19.9°C | [19.3, 20.5] | 640 |
| fog | 16.8°C | [15.4, 18.1] | 101 |
| drizzle | 15.9°C | [13.6, 18.2] | 53 |
| rain | 13.5°C | [13.1, 13.8] | 641 |
| snow | 5.6°C | [4.4, 6.7] | 26 |

The ordering sun > fog > drizzle > rain > snow holds on point estimates across all five categories. Confidence intervals are non-overlapping between sun/fog, fog/rain, and rain/snow, so those gaps are distinguishable from noise; fog and drizzle intervals overlap ([15.4, 18.1] vs [13.6, 18.2]), so that adjacent pair is not statistically distinguishable at n=101 and n=53. Minimum temperature follows the same ordering (sun 9.3°C down to snow 0.2°C).

### Finding 2 — precipitation is concentrated in two categories
![Mean daily precipitation by weather category](../artifacts/figures/02_precip_by_category.png)

Mean daily precipitation by category, with 95% bootstrap CI, out of 1,461 total days:

| Category | Mean precip | 95% CI | n / 1,461 |
|---|---|---|---|
| sun | 0.00 mm | [0.00, 0.00] | 640 |
| fog | 0.00 mm | [0.00, 0.00] | 101 |
| drizzle | 0.00 mm | [0.00, 0.00] | 53 |
| rain | 6.56 mm | [5.91, 7.26] | 641 |
| snow | 8.55 mm | [5.90, 11.27] | 26 |

All 794 days labeled sun, fog or drizzle (out of 1,461) record precipitation of exactly 0.00 mm — no partial or trace amounts are logged under these labels. Rain and snow are the only categories with nonzero precipitation, and rain's wide spread (min 0, max 55.9 mm, mean 6.56 mm vs median 3.30 mm) reflects the right-skewed tail flagged in the data quality notes below.

### Finding 3 — a real warming trend, no precipitation trend
![Yearly mean max and min temperature, 2012-2015](../artifacts/figures/03_temp_trend_2012_2015.png)
![Yearly mean precipitation, 2012-2015](../artifacts/figures/04_precip_trend_2012_2015.png)

Yearly means (pooled across all weather categories), with 95% bootstrap CI, n = 366 days in 2012 (leap year) or 365 days in 2013–2015:

| Metric | 2012 | 2015 | OLS slope (daily data, n=1,461) | 95% CI on slope |
|---|---|---|---|---|
| Max temp | 15.3°C [14.6, 16.0] | 17.4°C [16.7, 18.2] | +0.74°C/year | [0.41, 1.09] — excludes zero |
| Min temp | 7.3°C [6.8, 7.8] | 8.8°C [8.4, 9.3] | +0.52°C/year | [0.30, 0.73] — excludes zero |
| Precipitation | 3.35 mm [2.72, 4.03] | 3.12 mm [2.37, 3.92] | +0.04 mm/year | [-0.28, 0.37] — crosses zero |

2012 and 2015 confidence intervals do not overlap for either max or min temperature, and the OLS slope confidence intervals exclude zero for both — this is a real warming signal over the four years, not noise. Precipitation shows the opposite pattern: 2012 and 2015 intervals overlap heavily, yearly means bounce without a consistent direction (2013 dips to 2.27 mm), and the slope's confidence interval spans zero — there is no detectable precipitation trend in this window.

A secondary check restricting to the two large, per-year-stable categories (rain n=144–191/year, sun n=118–187/year) shows the same qualitative picture at the category level: rain and sun mean max temperature move within overlapping year-over-year CIs (e.g., rain 2012 12.8°C [12.1, 13.6] vs 2015 13.4°C [12.6, 14.1]). The pooled warming trend in Finding 3 is a population-level signal built from four years of daily data (n=1,461); it is not reliably visible within a single category's smaller yearly samples, and should not be re-derived that way.

## 4. Data quality notes

Gate verdict: **PASS** (`artifacts/data-gate.md`) — no null-share, duplicate, key-uniqueness, or time-coverage threshold was breached, out of 1,461 rows × 6 columns. The following are non-blocking flags carried into the findings above as caveats:

- **Precipitation is right-skewed with flagged outliers.** 206 of 1,461 days (14.1%) sit above the upper IQR fence on precipitation, all on the high side (up to 55.9 mm). These are real wet days in a wet climate, not data-entry errors, but they pull the `rain` and `snow` category means above their medians (rain: mean 6.56 mm vs median 3.30 mm) — the category comparisons in Finding 2 describe the mean, which is more tail-sensitive than the median.
- **Wind outliers, not used above.** 34 of 1,461 rows (2.3%) exceed the upper IQR fence on wind (~6.7 m/s, max 9.5 m/s); plausible gust days, flagged but not investigated in this report since wind was not part of the question asked.
- **`weather` label and `precipitation` disagree on 70 of 1,461 rows (4.8%).** 44 rows labeled `rain` record 0.00 mm precipitation, and 26 rows labeled something other than `rain` (mostly `snow`) record precipitation > 0. Finding 2's "rain and snow are the only wet categories" is a categorical pattern, not a perfect precipitation-value cutoff.
- **Class imbalance.** Of 1,461 rows: rain 43.9% (641), sun 43.8% (640), fog 6.9% (101), drizzle 3.6% (53), snow 1.8% (26). `snow` and `drizzle` are thin classes — their confidence intervals in Findings 1 and 2 are correspondingly wide, and any single extreme day moves their mean noticeably.
- **Scope limits.** Single Seattle station, daily grain, no documentation of how `weather` was labeled (observer vs. automated). No nulls (0.0% across all 6 columns) and no duplicate rows or keys (0.0%), so within its scope the series is complete and clean.

## 5. Next steps

With one more week, in priority order:

1. **Get labeling metadata for `weather`.** Ask the data source whether categories are observer-assigned or automated, and whether the method changed over 2012–2015. This directly affects how much to trust Finding 1 and the rain/precipitation mismatch in Finding 2.
2. **Resolve the 70-row rain/precipitation mismatch** by checking whether it stems from measurement timing (e.g., precipitation recorded for a different 24h window than the category label) rather than treating it as unexplained noise.
3. **Add more stations or years** if the goal is anything beyond "what happened in Seattle 2012–2015" — one city and four years cannot support claims about the region or about climate normals.
4. **Model precipitation on a log1p or outlier-aware scale** before any regression work, since the raw right-skewed distribution (206 flagged high-side outliers, 14.1% of rows) will otherwise dominate a linear fit.
5. **If snow/drizzle matter to the business question, collect more of them** — at n=26 and n=53 respectively, any further slicing of these categories (e.g., by month) will be too noisy to act on.

## 6. Appendix

**Method.** All means and slopes use the full 1,461-row daily series; no rows were dropped for outliers (flagged, not removed — see Data Quality Notes). Confidence intervals are 95% bootstrap intervals unless stated as an OLS slope CI. Category comparisons (Findings 1–2) group by the `weather` label; trend comparisons (Finding 3) group by calendar year and are pooled across all weather categories unless noted as the "secondary check" restricted to rain/sun only.

**Full category table** (n and share out of 1,461 rows):

| Category | n | Share | Mean max temp | Mean min temp | Mean precipitation |
|---|---|---|---|---|---|
| rain | 641 | 43.9% | 13.5°C | — | 6.56 mm |
| sun | 640 | 43.8% | 19.9°C | 9.3°C | 0.00 mm |
| fog | 101 | 6.9% | 16.8°C | — | 0.00 mm |
| drizzle | 53 | 3.6% | 15.9°C | — | 0.00 mm |
| snow | 26 | 1.8% | 5.6°C | 0.2°C | 8.55 mm |

**Sources.** `artifacts/data-profile.md` (full column profile), `artifacts/data-gate.md` (gate checks and flags), `artifacts/findings.md` (underlying analysis this report summarizes).
