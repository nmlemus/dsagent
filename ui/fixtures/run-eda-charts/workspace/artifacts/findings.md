# Findings — precipitation and temperature by weather category, 2012–2015

Unit of analysis: one calendar day (Seattle, `data/seattle-weather.csv`, 1,461 days,
2012-01-01 to 2015-12-31, 0 nulls, 0 duplicates — see `artifacts/data-gate.md`,
GATE: PASS).

## 1. Headline findings

1. **Precipitation only separates rain and snow days from the rest** — drizzle, fog
   and sun days all record 0.0 mm, so precipitation alone cannot flag them; use the
   `weather` label, not the precipitation value, for anything that needs to
   distinguish dry-but-overcast (fog, drizzle) from clear (sun) days.
2. **Daily temperatures rose over 2012–2015 and the trend is real, not noise** —
   season-adjusted max temperature climbed ~0.62 °C/year (95% CI 0.16–1.07) and min
   temperature ~0.43 °C/year (95% CI 0.14–0.73); plan for warmer baselines each
   successive year rather than treating 2012 norms as current.
3. **Fog displaced drizzle and snow, but rain's and sun's shares did not move by a
   provable amount, and precipitation itself showed no trend** — fog's share of days
   rose from 1.4% (2012) to 14.2% (2015) while drizzle and snow nearly disappeared;
   treat any "it's raining/sunny more now" claim as unsupported until more years of
   data narrow the interval on rain and sun.

## 2. Evidence

### Finding 1 — precipitation and temperature both split cleanly by category, but not the same way

Chart: *Only rain and snow days record measurable precipitation; drizzle, fog and
sun show 0 mm* — and *Snow days are far colder than any other category; sun days are
warmest*.

- Precipitation: rain days average 6.56 mm/day (n = 641/1,461, 43.9% of days), snow
  days average 8.55 mm/day (n = 26/1,461, 1.8%); drizzle (n = 53), fog (n = 101) and
  sun (n = 640) days all average exactly 0.0 mm.
- Temperature runs on a different axis than precipitation: snow days are by far the
  coldest (mean max 5.57 °C, mean min 0.15 °C, n = 26) and sun days the warmest
  (mean max 19.86 °C, mean min 9.34 °C, n = 640/1,461, 43.8%). Fog (mean max
  16.76 °C, n = 101) and drizzle (mean max 15.93 °C, n = 53) sit in between, both
  above rain's 13.45 °C (n = 641) despite rain carrying all the measured wet days
  except snow.
- Net: precipitation tells you "rain/snow vs. everything else"; temperature tells
  you a finer story (snow ≪ rain < drizzle < fog < sun) that precipitation cannot.

### Finding 2 — temperature trend, tested

Chart: *Daily max temperature rose about 0.6 °C/year, 2012-2015, after removing the
seasonal cycle (95% CI 0.16-1.07)*.

- Model: OLS on all 1,461 days, `temp ~ year + sin(day-of-year) + cos(day-of-year)`,
  Newey-West (HAC, 30-lag) standard errors to account for day-to-day autocorrelation.
- Max temperature: slope = +0.62 °C/year, 95% CI [0.16, 1.07], p = 0.008 (n =
  1,461) — the interval excludes zero, so this is a real within-period trend, not
  noise.
- Min temperature: slope = +0.43 °C/year, 95% CI [0.14, 0.73], p = 0.004 (n =
  1,461) — same conclusion, smaller magnitude.
- Raw annual means move the same direction: mean max temp was 15.28 °C in 2012 (n =
  366) vs. 17.43 °C in 2015 (n = 365); mean min temp was 7.29 °C in 2012 vs.
  8.84 °C in 2015.
- Precipitation, tested the same way: slope = +0.10 mm/day per year, 95% CI
  [-0.35, 0.55], p = 0.663 (n = 1,461) — CI straddles zero, **no clear trend**, even
  though annual mean precipitation bounced between 2.27 and 3.38 mm/day across the
  four years.

### Finding 3 — category mix shifted for the rare categories, not proven for the common ones

Chart: *Fog's share nearly tripled and drizzle/snow nearly vanished from 2012 to
2015 (labels = n days/year); rain and sun show no clear trend*.

- Model: linear probability model on the daily category indicator vs. year, HAC
  (30-lag) standard errors, n = 1,461 days per category.
- Fog: +4.16 percentage points/year, 95% CI [2.30, 6.02], p < 0.001 — share rose
  from 1.4% (5/366 days, 2012) to 14.2% (52/365 days, 2015). Clear increasing trend.
- Drizzle: -2.34 pp/year, 95% CI [-3.80, -0.88], p = 0.002 — share fell from 8.5%
  (31/366, 2012) to 1.9% (7/365, 2015). Clear decreasing trend.
- Snow: -1.84 pp/year, 95% CI [-3.42, -0.26], p = 0.022 — share fell from 5.7%
  (21/366, 2012) to 0.0% (0/365, 2015). Clear decreasing trend.
- Rain: -4.10 pp/year, 95% CI [-9.62, 1.43], p = 0.146 (52.2%, 191/366 in 2012 →
  39.5%, 144/365 in 2015) — moved in the raw numbers but the interval crosses zero:
  **no clear trend**, say so rather than "rain is declining."
- Sun: +4.12 pp/year, 95% CI [-1.75, 9.99], p = 0.169 (32.2%, 118/366 in 2012 →
  44.4%, 162/365 in 2015) — same caveat: **no clear trend** by this test.

## 3. Caveats

- **`weather` and `precipitation` are complementary, not redundant** (flagged in the
  gate report): drizzle, fog and sun days all carry 0.0 mm recorded precipitation,
  so precipitation cannot be used on its own to validate or reconstruct the
  `weather` label for those three categories — Finding 1's precipitation numbers
  describe rain/snow only.
- **Precipitation is zero-inflated** (57% of all 1,461 days record 0 mm) — its mean
  and the fitted trend in Finding 2 are driven by the minority of wet days; the
  "no clear trend" conclusion is about the daily mean, not about the intensity of
  individual storms.
- **Four years is enough to detect within-period movement, not a climate trend**
  (gate scope-limit flag): the temperature and category-share slopes above describe
  what happened between 2012 and 2015 specifically; they should not be extrapolated
  beyond this window.
- **Rain and snow days always have positive precipitation, but drizzle/fog/sun do
  not have zero severity** — the 0.0 mm figure for those categories reflects
  instrument/labeling resolution per the gate report, not necessarily "no
  moisture."
