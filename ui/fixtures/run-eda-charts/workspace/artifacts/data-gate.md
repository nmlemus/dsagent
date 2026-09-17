# Data gate — `seattle-weather.csv`

Source: `data/seattle-weather.csv` · 1,461 rows × 6 columns (from `artifacts/data-profile.json`).
Gate thresholds per the `eda` skill: null share on a required column > 20 % fails; exact
duplicate rows > 1 % fails; any missing time period inside the requested range fails; any
key-uniqueness violation at the declared grain fails.

## Checks

| Check | Value | Threshold | Result |
|---|---|---|---|
| Null share — date | 0.0 % (0 / 1,461) | > 20 % | PASS |
| Null share — precipitation | 0.0 % (0 / 1,461) | > 20 % | PASS |
| Null share — temp_max | 0.0 % (0 / 1,461) | > 20 % | PASS |
| Null share — temp_min | 0.0 % (0 / 1,461) | > 20 % | PASS |
| Null share — wind | 0.0 % (0 / 1,461) | > 20 % | PASS |
| Null share — weather | 0.0 % (0 / 1,461) | > 20 % | PASS |
| Exact duplicate rows | 0.0 % (0 / 1,461) | > 1 % | PASS |
| Key uniqueness — `date` (declared grain: one row per calendar day) | `unique: true`, 0 duplicates (0 / 1,461) | any violation | PASS |
| Time coverage — missing daily periods, 2012-01-01 to 2015-12-31 | 0 missing (0 / 1,461 expected calendar days) | any missing, inside requested range | PASS |

No column is missing from this table: every column in the profile has a null-share row,
the declared key (`date`) has a uniqueness row, and the profile's time-coverage section
(1,461 expected vs. 1,461 observed days, including the 2012 leap day) covers the
missing-periods check. `key_uniqueness.column` was not `null`, so the key check applies.

## Flags (do not breach a threshold, but affect how the data should be read)

- **precipitation is zero-inflated**: 838 / 1,461 days (57 %) record exactly 0 mm.
  The profiler's 206 IQR-flagged "outliers" are ordinary rainy days above a low
  (~7 mm) upper fence, not data errors — do not drop them as anomalies.
- **wind** has 34 IQR-flagged high values, all physically plausible (max 9.5, no
  negatives) — mild right skew, not an error.
- **weather label vs. precipitation mismatch**: `drizzle`, `fog`, and `sun` days all
  show exactly 0.0 mm recorded precipitation, while `rain` and `snow` days always
  have positive precipitation. `weather` therefore carries information the
  `precipitation` column cannot recover on its own — treat them as complementary,
  not redundant, in any downstream analysis.
- **Scope limits**: single daily station summary — no time-of-day, location, humidity,
  or pressure fields; 4 years (2012–2015) is enough to see seasonality but too short
  to support trend/climate claims.

## Verdict

9 / 9 checks pass. No null share exceeds 20 % on any column, there are no exact
duplicate rows, the declared key (`date`) is unique with 0 duplicates, and the daily
time series has 0 missing periods across the full requested range (2012-01-01 to
2015-12-31, including the leap day). The flags above are read-with-caution notes for
the analysis step, not gate failures.

GATE: PASS
