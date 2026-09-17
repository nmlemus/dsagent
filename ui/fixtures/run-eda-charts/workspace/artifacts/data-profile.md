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

- `date` spans **2012-01-01 to 2015-12-31**, daily grain, 4 full calendar years.
- Expected 1,461 calendar days for that span; **1,461 observed, 0 missing** — no gaps,
  including the 2012 leap day (366 rows in 2012, 365 in each of 2013–2015).
- `date` is unique at daily grain (0 duplicate keys), so it is a valid primary key for
  this table.
- 0 exact duplicate rows across all columns.

## Distribution notes

- **precipitation** (mm/day) is zero-inflated and right-skewed: 838 of 1,461 days
  (57 %) record 0 mm, median is 0 mm, but the max is 55.9 mm. The 206 "IQR outliers"
  flagged by the profiler are mostly ordinary rainy days above the (low) upper fence
  of ~7 mm, not data errors — expected shape for daily rainfall, not a quality issue.
- **wind** (m/s, presumably) is mildly right-skewed (mean 3.24 vs. median 3.0, max 9.5);
  the 34 flagged outliers sit at the high end of the range with no negative or
  implausible values.
- **temp_max** and **temp_min** show no IQR outliers and no logical violations —
  `temp_max >= temp_min` holds for all 1,461 rows. Ranges (-1.6–35.6 °C for max,
  -7.1–18.3 °C for min) are physically plausible for Seattle.
- **weather** (5 categories) cross-checked against `precipitation`: `rain` and `snow`
  days always carry positive precipitation (means 6.6 mm and 8.6 mm respectively),
  while `drizzle`, `fog` and `sun` days all show exactly 0.0 mm recorded precipitation.
  That is a labeling quirk worth flagging, not a null/duplicate problem: "drizzle" is
  presumably too light to register in this instrument's precision, so precipitation
  alone cannot distinguish `drizzle`/`fog`/`sun` days — the `weather` label carries
  information precipitation does not.
- No nulls anywhere; no negative values in `precipitation` or `wind`.

## What this data cannot answer

- No time-of-day, station location, or humidity/pressure fields — this is a daily
  city-level summary only, not hourly or spatially resolved.
- `weather` is a single categorical label per day; it cannot say whether conditions
  changed within a day (e.g., rain in the morning, sun in the afternoon).
- Four years of data (2012–2015) is enough to see a seasonal cycle but too short to
  say much about long-term trend or climate change with confidence.

