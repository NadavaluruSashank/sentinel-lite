# Free2Move Charging Threshold Case Study — Deliverables

## Files
- **Free2Move_Charging_Analysis.pptx** — final business presentation (13 slides)
- **01_data_prep.py** — data loading, validation, cleaning, feature engineering (run this first)
- **02_sql_analysis.sql** — organized, validated SQL queries (SQLite syntax, portable to Postgres/BigQuery)
- **03_python_eda.py** — EDA + all charts used in the presentation
- **data/clean_rentals.csv** — cleaned + feature-engineered rental table (output of step 1)
- **data/charging_events.csv** — derived "charging interruption" events used for the threshold analysis
- **data/tableau_rentals_extract.csv** — Tableau-ready rentals extract (friendly column names, flags spelled out)
- **data/tableau_threshold_extract.csv** — Tableau-ready threshold-analysis extract
- **charts/** — all PNG charts used in the deck

## Reproducing the analysis
```
python3 01_data_prep.py       # -> data/clean_rentals.csv, data/charging_events.csv
python3 03_python_eda.py      # -> charts/*.png
node build_deck.js            # -> Free2Move_Charging_Analysis.pptx
```
SQL queries can be run against `data/free2move.db` (SQLite) or against the two
CSV extracts loaded into any SQL engine.

## Tableau dashboard — how to build it
An actual Tableau Desktop session isn't available in this environment, so
instead of a .twbx file, here is everything needed to build the dashboard in
Tableau in a few minutes:

1. Connect Tableau to `data/tableau_rentals_extract.csv` and
   `data/tableau_threshold_extract.csv`.
2. **Sheet 1 — Executive KPIs**: BAN tiles (Number of Records, COUNTD(VEHICLE_ID),
   COUNTD([SESSION_TYPE]='Service Agent')) — mirrors slide 3 of the deck.
3. **Sheet 2 — Rental behaviour**: line chart of COUNT(rows) by WEEK_NUM; bar
   chart by HOUR; bar chart by DAY_OF_WEEK — mirrors slides 5–6.
4. **Sheet 3 — Charging & service behaviour**: bar chart of SESSION_TYPE x
   WAS_CHARGED; histogram of IDLE_GAP_MIN (from the threshold extract, filtered
   to INTERRUPTED_BY = 'agent' and INTERRUPT_LEVEL_PCT = 100) — mirrors slide 7.
5. **Sheet 4 — Charging threshold**: bar chart of LEVEL_BUCKET (sorted) with a
   calculated field `pct_customer = COUNTD(IF [INTERRUPTED_BY]='customer' THEN
   [VEHICLE_ID] END) / COUNTD([VEHICLE_ID])` — mirrors slide 9 (the key chart).
6. Combine into one Dashboard/Story with a filter action on VEHICLE_ID, and a
   text box repeating the "send agent at 90%" recommendation.

## Key numbers (all reproducible from the scripts above)
- 730 sessions, 52 vehicles, single MODEL_ID, Jan 3 – Feb 24 2023
- 615 customer rentals, 115 service-agent actions
- Vehicles found at 100% by an agent: median 59h idle, max 305h idle
- Charging-interruption events: 293 total, customer-claim rate 96% at 50-60%
  battery down to 67% at 95-100%, and 51.6% at exactly 100%
- Recommended dispatch threshold: 90% battery
