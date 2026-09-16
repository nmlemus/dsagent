# Seattle Weather 2012–2015: Precipitation & Temperature by Category

**Report prepared by:** Marie, Data Analyst  
**Data source:** `data/seattle-weather.csv` · 1,461 daily rows · 2012-01-01 → 2015-12-31 · 5 weather categories · Gate: **PASS** (9/9 checks)  
**Question answered:** How do precipitation and temperature differ across the weather categories, and how did they change between 2012 and 2015?

---

## Headline Findings

1. **Precipitation is confined to rain and snow days — drizzle, fog, and sun record exactly 0 mm by construction; cross-category precipitation comparisons must treat only rain and snow as distinct signals.**
2. **Rain days fell by 25 % from 2012 to 2015 (191 → 144 days/year), but individual rain events became heavier — median daily precipitation on rain days rose from 2.5 mm to 3.8 mm, with fog days absorbing much of the slack.**
3. **Sun days are 6.4 °C warmer (max) than rain days and reached their highest mean max temperature in 2015 at 21.4 °C — a +1.2 °C gain over 2012 — though four years of data are insufficient to confirm a statistically significant trend.**

---

## 1. Context

### Data

| Attribute | Value |
|---|---|
| File | `data/seattle-weather.csv` |
| Period | 2012-01-01 → 2015-12-31 (inclusive) |
| Grain | One row per calendar day |
| Rows | 1,461 (0 missing dates, 0 exact duplicates) |
| Columns | `date`, `precipitation` (mm), `temp_max` (°C), `temp_min` (°C), `wind` (m/s), `weather` (categorical: rain, sun, fog, drizzle, snow) |
| Key column | `date` — unique, no gaps |
| Data gate | **PASS** — all 9 checks (null %, duplicate rows, key uniqueness, missing dates) |

### What this data cannot answer

- Whether drizzle or fog days carry any measurable precipitation (the column is zero by construction for those labels — see Data Quality section).
- A reliable year-on-year snow trend — only 2012 has a meaningful sample (21 days; 2015 has zero).
- Whether the observed temperature signal reflects genuine warming or a shift in the mix of weather types within each year.
- Sub-daily patterns — the dataset is a single daily observation per day.
- Spatial variation within Seattle — one station or area average, no neighbourhood breakdown.

---

## 2. Evidence

### Finding 1 — Precipitation splits into two groups: wet labels (rain, snow) and dry labels (drizzle, fog, sun)

![Precipitation by weather category — rain and snow are the only categories with non-zero values](../artifacts/figures/fig1_precip_by_category.png)

*Figure 1: Daily precipitation (mm) by weather category. Box = IQR; whiskers = 1.5 × IQR; points = individual days beyond fence. n shown per category of 1,461 total days.*

Every one of the 53 drizzle days, 101 fog days, and 640 sun days (794 days, 54.3 % of the dataset) records exactly **0.00 mm** precipitation. The precipitation column therefore cannot distinguish drizzle from fog or sun on any given day.

Among the two wet categories:

| Category | n (of 1,461) | Precip median (mm) | Precip mean (mm) | Precip max (mm) |
|---|---:|---:|---:|---:|
| rain | 641 (43.9 %) | 3.30 | 6.56 | 55.9 |
| snow | 26 (1.8 %) | 5.45 | 8.55 | 23.9 |

Rain-day precipitation is highly right-skewed (mean 6.56 mm is nearly twice the median 3.30 mm); 206 rain days (14.1 % of all days) exceed the IQR upper fence of 7.0 mm. The **median is the more reliable central-tendency measure** for precipitation; means are reported alongside for completeness.

---

### Finding 2 — Fewer rain days, heavier individual events; fog absorbed what rain lost

![Weather category day counts per year, 2012–2015](../artifacts/figures/fig3_day_counts_by_year.png)

*Figure 2: Stacked bar — number of days per weather category per year (total = days in year: 366 in 2012, 365 in 2013–2015).*

| Year | Rain days (% of year) | Sun days (% of year) | Fog days | Snow days | Drizzle days |
|------|----------------------:|---------------------:|---------:|----------:|-------------:|
| 2012 | 191 (52.2 %) | 118 (32.2 %) | 5 (1.4 %) | 21 (5.7 %) | 31 (8.5 %) |
| 2013 | 158 (43.3 %) | 173 (47.4 %) | 16 (4.4 %) | 3 (0.8 %) | 15 (4.1 %) |
| 2014 | 148 (40.5 %) | 187 (51.2 %) | 28 (7.7 %) | 2 (0.5 %) | 0 (0.0 %) |
| 2015 | 144 (39.5 %) | 162 (44.4 %) | 52 (14.2 %) | 0 (0.0 %) | 7 (1.9 %) |
| **Change** | **−47 days (−24.6 %)** | **+44 days (+37.3 %)** | **+47 days (+940 %)** | **−21 days** | — |

Rain days declined every year. Fog days grew ten-fold; given that drizzle simultaneously collapsed from 31 to 0 days in 2014 and partially recovered in 2015, **some of the fog-day increase likely reflects reclassification of drizzle, not a genuine atmospheric shift** (see Data Quality section).

![Median and mean daily precipitation on rain days by year; error bars = IQR](../artifacts/figures/fig4_rain_intensity_by_year.png)

*Figure 3: Rain-day precipitation (mm) per year. Centre marks = median (solid) and mean (dashed). Error bars span Q1–Q3. n = rain-day count for that year.*

Despite fewer rain days, **individual rain events intensified after 2012**:

| Year | Rain days (n) | Median precip (mm) | Mean precip (mm) | Q1 (mm) | Q3 (mm) |
|------|---:|---:|---:|---:|---:|
| 2012 | 191 | 2.50 | 5.37 | 0.50 | 7.10 |
| 2013 | 158 | 3.05 | 5.15 | 0.80 | 7.60 |
| 2014 | 148 | 4.95 | 8.27 | 1.60 | 11.50 |
| 2015 | 144 | 3.80 | 7.91 | 1.20 | 11.10 |

Median rain-day precipitation nearly doubled from 2012 to 2014 (+98 %) before easing in 2015. Fewer but wetter rain events is the dominant signal in Seattle's precipitation record over this period.

---

### Finding 3 — Temperature separates clearly by weather type; sun and rain categories show a modest upward signal

![Mean daily temp_max and temp_min by weather category; error bars = 95 % CI](../artifacts/figures/fig2_temperature_by_category.png)

*Figure 4: Mean daily maximum and minimum temperature (°C) by weather category across all four years (1,461 days). Error bars = 95 % CI of the mean.*

| Category | n | Mean temp\_max (°C) | Mean temp\_min (°C) | Diurnal range (°C) |
|---|---:|---:|---:|---:|
| sun | 640 | 19.9 | 9.3 | 10.5 |
| fog | 101 | 16.8 | 8.0 | 8.8 |
| drizzle | 53 | 15.9 | 7.1 | 8.8 |
| rain | 641 | 13.5 | 7.6 | 5.9 |
| snow | 26 | 5.6 | 0.1 | 5.4 |

Sun days are **6.4 °C warmer** (max) than rain days and **14.3 °C warmer** than snow days. Precipitation categories (rain, snow) also compress the diurnal range to 5.4–5.9 °C versus 10.5 °C on sun days — consistent with cloud-cover insulating the surface overnight.

![Mean daily max temperature by year for rain, sun, and fog; shading = 95 % CI](../artifacts/figures/fig5_tempmax_trend_by_category.png)

*Figure 5: Mean daily max temperature (°C) per year for rain, sun, and fog categories. Shading = 95 % CI. Snow and drizzle omitted: sample sizes too small for reliable year-level estimates.*

| Year | Sun mean temp\_max (°C) | n | Rain mean temp\_max (°C) | n | Fog mean temp\_max (°C) | n |
|------|------------------------:|--:|-------------------------:|--:|------------------------:|--:|
| 2012 | 20.2 | 118 | 12.8 | 191 | 21.1 | 5 |
| 2013 | 18.9 | 173 | 13.6 | 158 | 19.4 | 16 |
| 2014 | 19.2 | 187 | 14.2 | 148 | 17.9 | 28 |
| 2015 | 21.4 | 162 | 13.4 | 144 | 14.9 | 52 |
| **Change 2012→2015** | **+1.2 °C** | — | **+0.6 °C** | — | **−6.2 °C** | — |

Sun-day max temperature rose +1.2 °C and rain-day max rose +0.6 °C from 2012 to 2015. Fog-day max fell −6.2 °C, but the fog-day sample grew from 5 to 52 days over the same period — the apparent cooling is almost certainly **composition bias**: in 2012 the five fog days happened to fall during a warm spell, while by 2015 fog was a year-round phenomenon. These signals should not be interpreted as statistically confirmed trends without formal time-series modelling over a longer horizon.

---

## 3. Data Quality Notes

| Flag | Severity | Impact on this analysis |
|---|---|---|
| **Precipitation is 0 mm by construction for drizzle, fog, and sun** (794 days, 54.3 % of dataset) | High — affects cross-category comparison | Precipitation analysis is restricted to rain vs. snow only. Do not compare precipitation levels across all five categories. |
| **Snow is a thin stratum: 26 days total, 21 in 2012** | High — affects trend analysis | Year-over-year snow trends are unreliable. The "snow vanished after 2012" observation may reflect sparse sampling, not climate. |
| **Drizzle absent from 2014 entirely** | Medium — affects category-level year trends | The ten-fold fog increase in 2014–2015 may partly reflect reclassification of drizzle. Fog trends are directionally plausible but not confirmed. |
| **Precipitation is right-skewed with 206 IQR-outlier days (14.1 % of all rows)** | Medium — affects summary statistics | Arithmetic means overstate typical rain-day intensity. Medians are reported throughout and are preferred for central-tendency statements. |
| **Four-year window is short for trend claims** | Medium — affects temperature trend interpretation | +1.2 °C on sun days is directionally interesting but not statistically established without formal modelling. |

---

## 4. Next Steps

1. **Verify the 0 mm precipitation anomaly** — request raw hourly station data for days labelled drizzle and fog. If those days carry measurable precipitation, the dataset needs correction before any precipitation modelling.
2. **Formalise the temperature trend test** — apply a seasonal decomposition (STL) or Mann-Kendall test to the daily max temperature series, stratified by weather category, to establish whether the observed +1.2 °C signal on sun days is statistically distinguishable from natural variability.
3. **Clarify the fog/drizzle reclassification** — review the classification algorithm's changelog or documentation for 2013–2014 to determine whether the drizzle→fog shift is methodological or meteorological.

---

## Appendix

### A. Full gate results

| # | Check | Observed | Threshold | Denominator | Result |
|---|---|---|---|---|---|
| 1 | Null % — `date` | 0.0 % | ≤ 20 % | 1,461 rows | ✅ PASS |
| 2 | Null % — `precipitation` | 0.0 % | ≤ 20 % | 1,461 rows | ✅ PASS |
| 3 | Null % — `temp_max` | 0.0 % | ≤ 20 % | 1,461 rows | ✅ PASS |
| 4 | Null % — `temp_min` | 0.0 % | ≤ 20 % | 1,461 rows | ✅ PASS |
| 5 | Null % — `wind` | 0.0 % | ≤ 20 % | 1,461 rows | ✅ PASS |
| 6 | Null % — `weather` | 0.0 % | ≤ 20 % | 1,461 rows | ✅ PASS |
| 7 | Exact duplicate rows | 0.0 % | ≤ 1 % | 1,461 rows | ✅ PASS |
| 8 | Key uniqueness — `date` | 0 violations | 0 | 1,461 rows | ✅ PASS |
| 9 | Missing daily dates | 0 gaps | 0 | 1,461 expected | ✅ PASS |

**`GATE: PASS`**

### B. Column-level profile summary

| Column | Dtype | Null % | Distinct | Min | Max | Mean | Median | Std |
|---|---|---|---|---|---|---|---|---|
| date | datetime | 0.0 % | 1,461 | 2012-01-01 | 2015-12-31 | — | — | — |
| precipitation | float64 | 0.0 % | 111 | 0.0 mm | 55.9 mm | 3.03 mm | 0.0 mm | 6.68 |
| temp\_max | float64 | 0.0 % | 67 | −1.6 °C | 35.6 °C | 16.44 °C | 15.6 °C | 7.35 |
| temp\_min | float64 | 0.0 % | 55 | −7.1 °C | 18.3 °C | 8.23 °C | 8.3 °C | 5.02 |
| wind | float64 | 0.0 % | 79 | 0.4 m/s | 9.5 m/s | 3.24 m/s | 3.0 m/s | 1.44 |
| weather | string | 0.0 % | 5 | — | — | — | — | — |
