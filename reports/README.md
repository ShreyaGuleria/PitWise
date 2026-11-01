# Current cached dataset reports

Generated from the 2018–2025 cache on 2026-11-01. All 3,458 driver rows and
173 races are retained. Stop counts, classes, weather, and existing fields
other than exclude_reason were preserved exactly when saving the updated CSV.

The first matching exclusion wins: dns, dnf, rule_forced_stops,
red_flag_race, chaotic_race, suspect_stop_data, zero_stop_finisher,
outlier_stops, then empty (usable). Disqualified counts as a finisher.
Race flags remain true even where a higher-priority exclusion wins.

There are 2,507 usable rows: class 1 = 1,239; class 2 = 1,039; class 3+ = 229.
The suspect-race report includes all 13 flagged races, including those excluded
by other rules. Sixteen usable rows in 14 races have consecutive-lap records.
These records are retained without automatic correction.

See full_report.txt for the complete printed report, exclusions_by_season.csv
and usable_classes_by_season.csv for season totals, and verification.json for
retention, UTF-8, and zero-network verification. Previous reports are under
historical/ and describe the OLD exclusions.

From the project directory, run the existing and new offline checks:

```powershell
.venv/Scripts/python.exe src/test_build_dataset.py
```

To rebuild exclusively from cache and print refreshed reports:

```powershell
.venv/Scripts/python.exe src/build_dataset.py --all-seasons --cache-only
```

The rebuild saves the dataset under data/processed/ and diagnostic CSVs here.
Capture its console output as reports/full_report.txt when regenerating that log.
