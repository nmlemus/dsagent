# Mapping the modeling table to `meridian.data.InputData`

Use `meridian.data.load.DataFrameDataLoader` (or `CsvDataLoader`) with a
`CoordToColumns` mapping:

| `CoordToColumns` field | modeling-table column(s) |
|---|---|
| `time` | `time` (ISO date string) |
| `geo` | `geo` |
| `kpi` | `kpi` |
| `revenue_per_kpi` | `revenue_per_kpi` (omit for revenue KPI) |
| `controls` | `[price, promo, holiday, ...]` |
| `population` | `population` |
| `media` | `[<ch>_impressions, ...]` |
| `media_spend` | `[<ch>_spend, ...]` |
| `organic_media` | `[<ch>_organic, ...]` (optional) |

`media_to_channel` / `media_spend_to_channel` map the column names to channel
labels — keep labels short and stable; they appear in every report.

Do not scale spend by population; Meridian does per-capita scaling internally when
`population` is provided.
