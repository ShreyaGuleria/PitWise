# f1-strategy

One race only: the 2023 Bahrain Grand Prix. All drivers in the session results
are retained, including retirements.

Run from this folder in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe src/build_dataset.py
```

FastF1 downloads are stored in this project's `./cache`, regardless of the
working directory used to launch the script. The first run needs internet access.

FastF1 3.9.0 supports Formula 1's October 2026 authentication requirement for
archived timing data. Before the first download, sign in locally with an eligible
F1TV account if prompted:

```powershell
.\.venv\Scripts\python.exe -m fastf1 auth f1tv --authenticate
```

Enter your credentials only in your local terminal. FastF1 manages the login
token; no credentials belong in this project's source files. See the
[FastF1 3.9.0 release notes](https://github.com/theOehrly/Fast-F1/releases/tag/v3.9.0).

Run the small offline check with:

```powershell
.\.venv\Scripts\python.exe src/test_build_dataset.py
```

`num_stops` is the number of distinct non-missing `Stint` values in each driver's
lap data minus one. No usable stints means an unknown count (`<NA>`), not -1 or
an assumed zero. This assumes each observed stint transition represents a stop;
missing laps, stops without a recorded following stint, or red-flag tyre changes
can make this differ from a count of physical pit-lane visits.

Weather temperatures are unweighted means of all available race-session weather
samples (including any pre/post-race samples supplied by FastF1), in Celsius.
Missing temperatures are ignored by pandas; entirely missing temperatures stay
unknown. `rain` is true if any non-missing `Rainfall` observation is true, false
if observations exist and none are true, and unknown if there are no observations.
The same session weather values appear for every driver, including retirees.

`circuit` uses the event name, as requested. `race_date` uses the race session's
scheduled UTC date. Finish statuses are kept exactly as supplied by FastF1;
grid positions retain FastF1's values (including 0 if supplied for a pit-lane start).

Field reference: [FastF1 getting started documentation](https://github.com/theOehrly/Fast-F1/blob/main/docs/getting_started/basics.rst).

## Season dataset

The default command now runs **2023 only** and saves `data/processed/stops_dataset.csv`.
Run `python src/build_dataset.py --all-seasons` with the project interpreter only
when ready to download 2018 through 2025. Existing cached JSON is reused.
The Desktop copy is the current working project.

All result rows are retained. `is_finisher` accepts Finished, +... and Lapped.
Exclusion reasons are semicolon-separated, so a row can have multiple reasons:
retired, no_pit_data (zero records), outlier_stops (>=5), possible_red_flag.
Zero records do not prove missing coverage: an early retirement or a genuine
zero-stop drive also gets no_pit_data. Failed pit downloads produce unknown stop
counts rather than fabricated zeros. Weather failures retain rows with missing
weather. `data_complete` identifies incomplete downloads; usable summaries require
both an empty exclusion reason and complete data.

The possible-red-flag heuristic requires at least three distinct drivers with
pit durations of at least ten minutes on the same lap. Jolpica's result/schedule
responses lack scheduled lap counts and a red-flag indicator, so this is a limited
signal, not a verified stoppage list. It can miss red flags without such pit records
and can flag other prolonged pit events. All race rows share the signal.

A failed results request cannot yield driver rows; it is explicitly reported in
`data/processed/failed_races.json`. Reruns reuse good cached responses and retry
failed requests. The console prints samples, season coverage, reason counts, usable
stop distributions, and failures. Reason counts overlap when multiple flags apply.
