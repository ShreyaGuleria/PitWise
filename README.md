# PitWise

PitWise aims to be a what-if advisor predicting **1, 2 or 3+ pit stops** per
F1 driver from circuit, date, grid, team and conditions. The current project
provides a historical classification dataset and validation baselines, not
proof of which strategy optimises race results or a causal strategy simulator.

## Data and sources

`data/processed/features.csv` contains 3,458 driver-race rows across 173 races
in 2018–2025. Join on `season`, `round`, `driver`. Jolpica's Ergast-compatible
results and pit-stop records supply circuit, driver, constructor, grid, outcome
and stop information. Open-Meteo supplies historical air temperature,
precipitation and model elevation. Other fields are derived.

`stops_dataset.csv` is the processed source; `features.csv` is the enriched
model table. All original rows are retained. Train/evaluate only rows with
`is_usable == True` and `stops_class` in `1`, `2`, `3+`: 2,507 rows.
Select inputs explicitly from `src/features.py`, never all numeric CSV fields.
See [DATA_DICTIONARY.md](DATA_DICTIONARY.md) for every column and
[reports/README.md](reports/README.md) for the proposed extra fields' availability.

`pit_lane_start` is true for original grid zero; `grid_position_clean` maps
zero to 20 and leaves other positions unchanged. Original `grid_position` is
preserved. Grid zero denotes a pit-lane start. Only the cleaned grid and pit-lane flag
are inputs; raw grid_position is reference metadata and is not allowlisted.

`circuit_pit_duration_prior` is median cached pit duration below 60 seconds from
strictly earlier usable races at the same circuit. A usable race has at least
one usable driver; all its drivers' stops contribute once per race. Same-date
and future races cannot contribute. This is pit-lane duration, not net time
lost. No history means NaN (640 rows). Fit any imputation on training only.

## Exclusion priority

First match wins; flags remain available even when a higher-priority rule wins.
Blank `exclude_reason` means usable.

| Priority | Rule | Reason |
|---|---|---|
| 1 | dns | Did not start, so no completed race strategy to model. |
| 2 | dnf | Final status does not satisfy the project's finisher rule. |
| 3 | rule_forced_stops | A listed mandatory stop/tyre-limit rule constrains ordinary strategy. |
| 4 | red_flag_race | Three or more drivers have 600+ second stops on a common lap, suggesting a stoppage. |
| 5 | chaotic_race | Three or more drivers have five or more stops, an atypical strategy context. |
| 6 | suspect_stop_data | Eight or more drivers have consecutive-lap stop records, suggesting questionable data. |
| 7 | zero_stop_finisher | Zero-stop finishers are outside the 1/2/3+ target definition. |
| 8 | outlier_stops | Individual count is five or more stops, outside the retained strategy range. |
| 9 | usable | None of the exclusions applies; retain for labelled modelling. |

The finisher rule accepts Finished, statuses starting with `+`, Lapped and
Disqualified. These are existing dataset choices, not universal definitions.
The 217 blank classes are zero-stop rows. Selection depends on outcomes, so
reported performance describes this filtered population, not every starter.

## Fixed season split

| Split | Seasons | Usable labelled rows |
|---|---|---:|
| Train | 2018–2023 | 1,784 |
| Validation | 2024 | 363 |
| Test | 2025 | 360 |

Never randomly split driver rows: all drivers in a race must stay together.
Assignments are saved in `data/processed/splits.csv`. **2025 remains untouched
for performance evaluation until the explicitly authorised final run.** Only
counts and data checks have been inspected; no 2025 performance metrics have
been computed. `evaluate_split` locks test evaluation by default. Tune using
validation only, and fit preprocessing on training only.

## Validation baselines (2024)

| Baseline | Accuracy | Balanced accuracy | Macro F1 |
|---|---:|---:|---:|
| B0: fixed training majority (1 stop) | 0.4215 | 0.3333 | 0.1977 |
| B1: all earlier usable circuit history | 0.6088 | 0.4560 | 0.4242 |
| B2: last three earlier usable circuit races | 0.5730 | 0.4807 | 0.4728 |

B1 pools driver class counts at the circuit; unseen circuits fall back to B0.
B2 pools the last three races; with fewer than three, fallback is B1, then B0.
Ties favour 1, then 2, then 3+. History includes strictly earlier race dates
only and can update from earlier completed validation races. No validation
rows have circuits unseen in usable training. B1 leads accuracy; B2 leads
balanced accuracy/macro F1 and recalls 20% of 3+ cases (B0/B1 recall none).
Models should beat these benchmarks, not merely exploit class imbalance.
Full metrics are in `reports/baselines_validation.txt`.

## Known limitations

- `scheduled_laps` is actual maximum completed distance, not planned distance.
  Weather inputs are race-window reanalysis, not pre-start forecasts. These
  retained retrospective proxies must be excluded or replaced before claiming
  a strict pre-race model; the dictionary records their actual availability.
- Rain is weak: a few hourly precipitation samples at circuit coordinates
  cannot reliably capture showers, wet track conditions or changing grip.
- Only 173 races provide independent contexts. Driver rows share conditions,
  some circuits have little history and 3+ stops are relatively rare.
- The 2022 regulation/tyre era shift changes strategy. `is_2022_plus` is a
  coarse indicator; older circuit history may not transfer to recent years.
- Pit records count visits, not guaranteed tyre changes. Red flags, suspect
  records, outcome filtering and the grid-zero convention limit interpretation.
- This cached route lacks tyre compounds/life, track temperature, degradation,
  accurate lap series and SC/VSC timelines. Do not invent those fields.

## Rebuild and test offline

Use the existing environment; these commands install and download nothing.
Rebuilding requires the local cache. A fresh clone can use saved CSVs directly;
ignored cache files may not be present in the clone.

```powershell
.venv/Scripts/python.exe src/build_dataset.py --all-seasons --cache-only
.venv/Scripts/python.exe src/build_features.py --cache-only
.venv/Scripts/python.exe src/evaluate.py

.venv/Scripts/python.exe src/test_build_dataset.py
.venv/Scripts/python.exe src/test_features.py
.venv/Scripts/python.exe src/test_evaluate.py
.venv/Scripts/python.exe src/test_history.py
```

The dataset builder refreshes source/exclusion reports; the feature builder
refreshes `features.csv` and feature diagnostics. `evaluate.py` writes splits
and evaluates **2024 only**. Cache misses fail without downloading when
`--cache-only` is supplied. Tests cover original-field preservation, grid
encodings, cached reconstruction, historical cutoffs and split/leakage guards.
Historical diagnostics and logs belong under `reports/historical/` and do not
define the current dataset. Local `.venv`, `cache`, `__pycache__` and `.tmp`
folders are ignored.

## Expanding-window first model

Run `.venv/Scripts/python.exe src/train.py` for four folds: validate on 2021,
2022, 2023 and 2024, each time training on 2018 through the previous season.
It excludes 2025 before any fitting, prediction or metric calculation.
Each race remains wholly in one fold. Earlier completed validation races may
update history, matching B1/B2's online information policy. Fitted model weights
and preprocessing stay fixed within each validation season.

Four new INPUT histories use strictly earlier usable races, excluding every
same-date and future race:

- `circuit_avg_stops_prior`: pooled usable-driver mean stops at that circuit.
- `circuit_races_prior`: count of distinct earlier usable races there (zero at cold start).
- `circuit_avg_stops_last3`: pooled usable-driver mean over the last three usable circuit races, or all available if fewer.
- `recent_avg_stops_prior`: pooled usable-driver mean over the last three usable races globally, or all available if fewer.

Means remain NaN without history. Windows count races, not driver rows, and
race stop means are weighted by usable-driver counts. The renamed
`circuit_pit_duration_prior` replaces `circuit_pit_loss_prior`; the old column
has been removed. Its numeric definition is unchanged.

The first model is a fixed random forest (200 trees, depth 8, minimum leaf 5,
balanced class weights, seed 42), without hyperparameter search. Each fold fits
median imputation, missingness indicators and one-hot encoding on training
only. Unseen categories are ignored. Requirements now record scikit-learn,
matplotlib and joblib; these were installed into the existing .venv.

Both variants use the same model and history features. The retrospective
variant retains the documented current-race proxies. The strict pre-race
variant replaces actual distance with `circuit_laps_last_year` (latest usable
visit in exactly the preceding season), replaces current air temperature with
`circuit_month_air_temp_prior` (equal-race mean over earlier usable circuit
races in the same calendar month), and drops current precipitation entirely.
These replacements are historical estimates, not known current planned laps
or forecasts. Missing histories are imputed using that fold's training only.
`STRICT_FEATURE_COLUMNS` lists this variant; `ALL_FEATURE_COLUMNS` describes
the union of both feature schemas.

Reports: `reports/expanding_validation.txt`, `expanding_validation.csv` and
`expanding_validation_summary.csv` include B0/B1/B2, both models, per-season
metrics, per-class results, confusion matrices and mean/sample standard
deviation/min/max across four seasons. Spread describes between-season
variation, not a confidence interval. Driver rows are correlated within races;
small differences should not be interpreted as reliable improvements.

Initial four-fold results (unweighted seasonal mean +/- sample standard deviation):

| Method | Accuracy | Balanced accuracy | Macro F1 |
|---|---:|---:|---:|
| B0 | 0.4211 +/- 0.0833 | 0.3333 +/- 0.0000 | 0.1964 +/- 0.0265 |
| B1 | 0.5247 +/- 0.0840 | 0.3974 +/- 0.0405 | 0.3574 +/- 0.0515 |
| B2 | 0.5224 +/- 0.0715 | 0.4086 +/- 0.0519 | 0.3744 +/- 0.0725 |
| RF retrospective | 0.4889 +/- 0.0867 | 0.4039 +/- 0.0410 | 0.3868 +/- 0.0655 |
| RF strict | 0.4589 +/- 0.0675 | 0.3860 +/- 0.0201 | 0.3734 +/- 0.0361 |

No clear model win: the retrospective model's small mean macro-F1 advantage
comes with unavailable current-race information and large seasonal variation.
B1/B2 retain higher mean accuracy. The strict model is a pre-race starting
point, not a demonstrated improvement. Fold race counts are 14, 19, 17 and 20
for 2021 through 2024; they are not independent driver-level sample sizes.
