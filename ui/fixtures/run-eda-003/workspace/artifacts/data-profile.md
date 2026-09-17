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

- Range: 2012-01-01 to 2015-12-31 (4 full calendar years).
- Expected daily periods over that range: 1,461. Observed distinct dates: 1,461. **No missing days** and no duplicate dates (`date` is unique at daily grain, consistent with the key-uniqueness check above).
- Row counts per year: 2012 = 366 (leap year), 2013 = 365, 2014 = 365, 2015 = 365 — all match the calendar exactly.

## Distribution notes

- `precipitation` (mm/day): median is much lower than the mean (3.03 mm), right-skewed — most days are dry or lightly wet, with a long tail up to 55.9 mm. 206 rows (14.1 % of 1,461) sit beyond the 1.5×IQR fence (upper bound ≈ 7.0 mm). This looks like genuine rain-event variability rather than a data error, but it means simple means/std-dev summaries will overstate a "typical" day — a median or log scale is more representative.
- `wind` (m/s, presumed): 34 rows (2.3 %) exceed the IQR fence (upper bound ≈ 6.7), max 9.5 — a shorter, milder tail than precipitation.
- `temp_max` / `temp_min` (°C): no IQR outliers; ranges (-1.6 to 35.6 °C and -7.1 to 18.3 °C) are physically plausible for Seattle and internally consistent (need to confirm temp_max ≥ temp_min row-by-row before modeling).
- `weather` (5 categories): `rain` (43.9 %) and `sun` (43.8 %) dominate, together 87.7 % of days; `fog` 6.9 %, `drizzle` 3.6 %, `snow` 1.8 %. The category mix shifts across years — e.g., `snow` drops from 21 days in 2012 to 0 in 2015, and `drizzle` from 31 to 7 — worth flagging since it may reflect a change in how observers/instruments classified conditions rather than an actual climate shift; the report should not assume the categories are recorded consistently year over year.

## What this data cannot answer

- No station/location field — this is presumably a single Seattle station, but that isn't documented in the file itself, so any generalization beyond that station is unsupported.
- No hourly or sub-daily detail — precipitation, wind and temp are daily aggregates, so within-day extremes (e.g., short bursts of heavy rain) are invisible.
- `weather` is a single categorical label per day; days with mixed conditions (e.g., rain and fog) can only show one class, so it undercounts secondary conditions.
- No metadata on measurement units or instrument changes over the 2012–2015 span, which matters given the category-mix drift noted above.
