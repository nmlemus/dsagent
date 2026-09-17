# Precipitation and temperature by weather category — Seattle, 2012–2015

## 1. Headline findings

1. **Precipitation only separates rain and snow days from the rest** — drizzle,
   fog and sun days all record 0.0 mm, so precipitation alone cannot flag them; use
   the `weather` label, not the precipitation value, for anything that needs to
   distinguish dry-but-overcast (fog, drizzle) from clear (sun) days.
2. **Daily temperatures rose over 2012–2015 and the trend is real, not noise** —
   season-adjusted max temperature climbed ~0.62 °C/year (95% CI 0.16–1.07) and min
   temperature ~0.43 °C/year (95% CI 0.14–0.73); plan for warmer baselines each
   successive year rather than treating 2012 norms as current.
3. **Fog displaced drizzle and snow, but rain's and sun's shares did not move by a
   provable amount, and precipitation itself showed no trend** — fog's share of
   days rose from 1.4% (2012) to 14.2% (2015) while drizzle and snow nearly
   disappeared; treat any "it's raining/sunny more now" claim as unsupported until
   more years of data narrow the interval on rain and sun.

## 2. Context

**Question asked:** how do precipitation and temperature vary by weather category
in Seattle, and has either shifted over 2012–2015?

**Data used:** `data/seattle-weather.csv`, one row per calendar day, 1,461 rows
(2012-01-01 to 2015-12-31, 4 full calendar years including the 2012 leap day), 6
columns — `date`, `precipitation` (mm/day), `temp_max` / `temp_min` (°C), `wind`
(m/s), `weather` (5 categories: rain, sun, fog, drizzle, snow). 0 nulls, 0 duplicate
rows, `date` unique at the daily grain, 0 missing calendar days in range.

**What this data cannot answer:**

- No time-of-day, station location, humidity or pressure fields — this is a single
  daily city-level summary, not hourly or spatially resolved. It cannot say
  whether conditions changed within a day (e.g., rain in the morning, sun in the
  afternoon), because `weather` is one label per day.
- `precipitation` cannot stand in for `weather`: drizzle, fog and sun days all
  carry exactly 0.0 mm recorded precipitation (a labeling/instrument-resolution
  quirk, not missing data), so the two columns are complementary, not redundant —
  the precipitation figures in Finding 1 describe rain/snow days only.
- Four years (2012–2015) is enough to detect within-period movement and is long
  enough to see the seasonal cycle, but it is too short to support a long-term
  trend or climate claim. The category-share and temperature slopes below describe
  what happened between 2012 and 2015 specifically and should not be extrapolated
  beyond that window.
- For the rarer categories with wide confidence intervals — rain and sun in
  Finding 3 — the data cannot currently distinguish a real shift from noise; more
  years would be needed to narrow those intervals.

## 3. Evidence

### Finding 1 — precipitation and temperature both split cleanly by category, but not the same way

Chart: *Only rain and snow days record measurable precipitation; drizzle, fog and
sun show 0 mm* — and *Snow days are far colder than any other category; sun days
are warmest*.

- Precipitation: rain days average 6.56 mm/day (n = 641/1,461, 43.9% of days), snow
  days average 8.55 mm/day (n = 26/1,461, 1.8%); drizzle (n = 53), fog (n = 101)
  and sun (n = 640) days all average exactly 0.0 mm.
- Temperature runs on a different axis than precipitation: snow days are by far
  the coldest (mean max 5.57 °C, mean min 0.15 °C, n = 26) and sun days the
  warmest (mean max 19.86 °C, mean min 9.34 °C, n = 640/1,461, 43.8%). Fog (mean
  max 16.76 °C, n = 101) and drizzle (mean max 15.93 °C, n = 53) sit in between,
  both above rain's 13.45 °C (n = 641) despite rain carrying all the measured wet
  days except snow.
- Net: precipitation tells you "rain/snow vs. everything else"; temperature tells
  a finer story (snow ≪ rain < drizzle < fog < sun) that precipitation cannot.

### Finding 2 — temperature trend, tested

Chart: *Daily max temperature rose about 0.6 °C/year, 2012-2015, after removing the
seasonal cycle (95% CI 0.16-1.07)*.

- Model: OLS on all 1,461 days, `temp ~ year + sin(day-of-year) + cos(day-of-year)`,
  Newey-West (HAC, 30-lag) standard errors to account for day-to-day
  autocorrelation.
- Max temperature: slope = +0.62 °C/year, 95% CI [0.16, 1.07], p = 0.008 (n =
  1,461) — the interval excludes zero, so this is a real within-period trend, not
  noise.
- Min temperature: slope = +0.43 °C/year, 95% CI [0.14, 0.73], p = 0.004 (n =
  1,461) — same conclusion, smaller magnitude.
- Raw annual means move the same direction: mean max temp was 15.28 °C in 2012
  (n = 366) vs. 17.43 °C in 2015 (n = 365); mean min temp was 7.29 °C in 2012 vs.
  8.84 °C in 2015.
- Precipitation, tested the same way: slope = +0.10 mm/day per year, 95% CI
  [-0.35, 0.55], p = 0.663 (n = 1,461) — CI straddles zero, **no clear trend**,
  even though annual mean precipitation bounced between 2.27 and 3.38 mm/day
  across the four years.

### Finding 3 — category mix shifted for the rare categories, not proven for the common ones

Chart: *Fog's share nearly tripled and drizzle/snow nearly vanished from 2012 to
2015 (labels = n days/year); rain and sun show no clear trend*.

- Model: linear probability model on the daily category indicator vs. year, HAC
  (30-lag) standard errors, n = 1,461 days per category.
- Fog: +4.16 percentage points/year, 95% CI [2.30, 6.02], p < 0.001 — share rose
  from 1.4% (5/366 days, 2012) to 14.2% (52/365 days, 2015). Clear increasing
  trend.
- Drizzle: -2.34 pp/year, 95% CI [-3.80, -0.88], p = 0.002 — share fell from 8.5%
  (31/366, 2012) to 1.9% (7/365, 2015). Clear decreasing trend.
- Snow: -1.84 pp/year, 95% CI [-3.42, -0.26], p = 0.022 — share fell from 5.7%
  (21/366, 2012) to 0.0% (0/365, 2015). Clear decreasing trend.
- Rain: -4.10 pp/year, 95% CI [-9.62, 1.43], p = 0.146 (52.2%, 191/366 in 2012 →
  39.5%, 144/365 in 2015) — moved in the raw numbers but the interval crosses
  zero: **no clear trend**, say so rather than "rain is declining."
- Sun: +4.12 pp/year, 95% CI [-1.75, 9.99], p = 0.169 (32.2%, 118/366 in 2012 →
  44.4%, 162/365 in 2015) — same caveat: **no clear trend** by this test.

## 4. Data quality notes

The gate report (`artifacts/data-gate.md`) verdict: **GATE: PASS**, 9/9 checks
passed — no null share exceeds 20% on any column, 0 exact duplicate rows, the
declared key (`date`) is unique with 0 duplicates, and the daily series has 0
missing periods across the full requested range. Read-with-caution flags, none of
which breach a threshold:

- **Precipitation is zero-inflated**: 838 / 1,461 days (57%) record exactly 0 mm.
  The profiler's 206 IQR-flagged "outliers" are ordinary rainy days above a low
  (~7 mm) upper fence, not data errors — the mean and fitted trend in Finding 2 are
  driven by the minority of wet days.
- **Wind** has 34 IQR-flagged high values, all physically plausible (max 9.5 m/s,
  no negatives) — mild right skew, not an error.
- **`weather` label vs. `precipitation` mismatch**: drizzle, fog and sun days all
  show exactly 0.0 mm recorded precipitation, while rain and snow days always have
  positive precipitation. This is a labeling/instrument-resolution quirk, not a
  data-quality defect — `weather` carries information `precipitation` cannot
  recover on its own.
- **Scope limits**: single daily station summary — no time-of-day, location,
  humidity or pressure fields; 4 years is enough to see seasonality but too short
  to support long-term trend or climate claims.
- `temp_max >= temp_min` holds for all 1,461 rows; no logical violations; no
  negative values in `precipitation` or `wind`.

## 5. Next steps

- Pull a longer history (10+ years) for the same station to test whether the
  fog-up/drizzle-down/snow-down category shift and the temperature trend continue,
  and to narrow the currently inconclusive rain and sun intervals.
- Get an hourly or sub-daily feed, or at least a secondary station, to check
  whether the single daily `weather` label is hiding within-day category changes
  (e.g., fog burning off to sun) that this table cannot see.
- Investigate the drizzle/fog/sun-always-0.0mm pattern with the data source or
  instrument owner to confirm it is a genuine measurement floor and not a
  systematic gap that would also affect rain/snow readings below the threshold.
- If a climate or planning decision hinges on the temperature trend, re-run the
  same seasonally-adjusted regression with the extended history before committing
  to a specific °C/year planning figure.
