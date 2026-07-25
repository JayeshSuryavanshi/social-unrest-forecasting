# Source data — provenance & licensing

⚠️ **Do not redistribute the files in this folder or commit them to a public repo.**

## `acled_india.csv`
- **Source:** ACLED (Armed Conflict Location & Event Data), downloaded via the myACLED Data Export
  Tool. India, all event types, **2018-01-01 → 2025-07-25** (272,462 events). Note: ACLED's free/Open
  tier serves data lagged ~1 year, which is why coverage ends mid-2025.
- **License:** Free to access after registration, but **redistribution is prohibited** by ACLED's
  Terms of Use. This copy is for personal/research use only. Attribution required in any output:
  cite ACLED and link https://acleddata.com/ . ~77% of India events are newspaper-sourced (relevant
  to the "is history independent of news?" question — see RESULTS.md finale).
- **Refresh:** re-export from the myACLED Data Export Tool, or use `scripts/fetch_acled.py` with a
  Research-tier API account.

## `gdelt_latest_export.csv`
- **Source:** GDELT 2.0 Event export (one 15-minute slice), fetched with no API key.
- **License:** GDELT is free and **redistributable** with attribution (cite the GDELT Project,
  https://www.gdeltproject.org/). This single-slice sample is illustrative; the modeling uses the
  GDELT 1.0 daily panels in `../interim/` (rebuilt via `scripts/build_gdelt_panel.py`).

## Built panels (`../interim/*.parquet`)
Derived aggregates (per date × admin1 counts / theme sums). Because they are derived from ACLED/GDELT,
treat them under the same non-redistribution stance as the raw sources.
