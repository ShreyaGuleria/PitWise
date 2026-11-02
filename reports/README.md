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

## Enriched features

`data/processed/features.csv` retains all original columns and 3,458 rows,
with circuit, calendar, constructor, lap-count, era and usability fields.
All 31 circuit elevations were present in cached weather; no requests or
imputations were needed. The original 13 model inputs have no missing
values. The additional `circuit_pit_duration_prior` deliberately has missing values
where no eligible earlier history exists. The 217 blank classes are zero-stop rows; the 2,507 blank exclusion
reasons are usable rows, not missing exclusion information.

`features_report.txt` prints every null/blank count, circuit summary and observed
constructor mapping. Corresponding CSVs are `feature_null_counts.csv`,
`feature_circuits.csv`, and `constructor_mapping.csv`. A usable race in the
circuit summary means a race with at least one usable driver row.

`src/features.py` explicitly separates model inputs from outcomes, filters and
metadata. As requested, the input list includes three retrospective proxies:
`scheduled_laps` is the maximum actual completed laps (not planned distance),
and temperature/precipitation use race-window reanalysis (not pre-start
forecasts). This is documented in the code; replace those sources before a
strictly pre-race evaluation. Elevation is Open-Meteo's model elevation at the
requested location, rather than surveyed track elevation.

Rebuild and test without downloads:

```powershell
.venv/Scripts/python.exe src/build_features.py --cache-only
.venv/Scripts/python.exe src/test_features.py
.venv/Scripts/python.exe src/test_build_dataset.py
```

The cached enrichment test blocks
network requests and compares every saved value with the rebuilt table;
existing source fields are checked exactly. Tests also cover constructor
aliases, date/era flags, requested street circuits, actual maximum laps,
cached elevations, and one fallback elevation lookup per circuit.

## Historical pit-duration feature (2026-11-02)

The feature table also includes `pit_lane_start` (original grid equals zero)
and `grid_position_clean` (zero maps to 20). Original `grid_position` remains
unchanged. Both encodings are allowlisted; grid zero is treated as a pit-lane
start by convention. There are now 19 allowlisted inputs, including the three
documented retrospective proxies. See the root DATA_DICTIONARY.md and README.md.

`circuit_pit_duration_prior` is an **F** input in `src/features.py` and the model
table `data/processed/features.csv`. For each row it is the median of individual
cached pit-stop durations, in seconds, at the same `circuit_id` in usable races
with `race_date` **strictly less than** that row's date. A usable race has at
least one usable driver, matching the circuit-summary convention above. All
drivers' cached stops from such a race contribute; races with no usable drivers
do not. Records are collected once per race, not repeated per driver row.

Blank, malformed, nonfinite and nonpositive durations are ignored. Durations
of **60 seconds or more** are excluded using the requested red-flag-stop cutoff;
this threshold is a filtering heuristic, not proof of a red flag. Both decimal
seconds and colon-separated durations are parsed. Same-date races and future
races never contribute. Earlier races in the same season may contribute: this
feature uses the requested strict date cutoff, not an earlier-season cutoff.
The median pools individual stops, rather than giving each race equal weight.

No history means **NaN**, saved as an empty CSV cell; no global fallback or
imputation is applied. The rebuilt table retains all 3,458 rows; 640 have no
eligible historical duration. The feature builder and evaluation loader preserve this
as a numeric missing value. Other CSV readers using `keep_default_na=False`
should also pass `na_values={'circuit_pit_duration_prior': ['']}`. Any eventual
imputation must be fitted on training data only.

Despite its requested name, this feature measures **pit-lane duration**, not
net time lost against staying on track. The latter needs suitable lap or sector
timing and a normal-pace reference. It does not implement the proposed
in-lap/out-lap `pit_loss_median` calculation.

The history helper forces cache-only reads and fails on a cache miss, restoring
the caller's cache setting afterward. Rebuild with `--cache-only` to keep the
rest of enrichment offline too. Tests cover the strict cutoff, different races
on the same date, future-record changes, row ordering, circuit isolation,
unusable races, duration boundaries, no-history NaNs, and blocked downloads.
No 2025 test performance metrics were computed for this addition.

## Proposed extra fields: availability and roles

**F** means available before the start; **S** means a proposed strategy or a
strategy target; **T** means an outcome target; **X** means a post-start
observation for analysis/filtering. The current target remains `stops_class`.
Realised S fields are not inputs to this stop-count classifier: they can reveal
its answer. A separate what-if task could use proposed S values, but prediction
alone would not establish the causal benefit of a strategy.

The route currently cached here contains Jolpica/Ergast-style race results and
pit-stop records plus Open-Meteo historical air temperature and precipitation.
There are no FastF1 lap, telemetry, tyre, track-status or session-weather files
in the local session cache. The distinctions below concern this data route and
cache, not a claim that these fields require paid access everywhere. FastF1 may
provide additional timing data where available, but obtaining it would require
data not cached here and is outside this offline task. Only the pit-duration
prior above was added; the other proposed additions were not generated.

| Proposed field(s) | Correct role | Availability or limitation on this route |
|---|---|---|
| `total_laps` | F for scheduled distance; X for actual distance | Planned pre-start distance is not supplied by the cached result rows. Existing `scheduled_laps` is actual maximum completed laps and remains a retrospective proxy. |
| `track_temp_mean`, `track_temp_max`, `humidity_mean`, `wind_speed_mean`, `rain_share` | X for observed race-session summaries; F only from pre-start forecasts | Cached weather has historical air temperature and precipitation, not track temperature, humidity, wind or FastF1 boolean Rainfall samples. Rainfall sample share cannot be replaced silently with precipitation totals. Archived pre-start forecasts are absent. |
| `sc_count`, `sc_laps`, `vsc_count`, `vsc_laps` | X | No track-status timeline in this route/cache. Pit-stop duration cannot reliably reconstruct SC/VSC deployments or laps. |
| `circuit_sc_rate_prior` | F | Needs historical SC counts, unavailable here. If later built, use earlier seasons only as specified. |
| `circuit_avg_stops_prior` | F | Now added using strictly earlier usable race dates (the latest requested definition supersedes the earlier-season-only proposal). |
| `pit_loss_median` | X measured in current race; F as a separate historical estimate | Net in-lap/out-lap loss needs lap timing and a normal-lap reference absent from the current cache. `circuit_pit_duration_prior` is only the documented pit-duration proxy. |
| `deg_soft`, `deg_medium`, `deg_hard` | X measured in current race; F as separate earlier-season estimates | No compound, tyre-life or suitable lap-time series, so neither race degradation nor its historical estimates can be built here. |
| `team` | F | Already represented by `constructor` and model input `constructor_group`; no extra team field added. |
| `quali_position` | F | Qualifying results are not cached. The free results route can provide qualifying separately, but downloading is forbidden. Race grid position is not an interchangeable substitute. |
| `finish_position`, `positions_gained`, `gap_to_winner_s` | T for a separate outcome task; excluded from current inputs | Result position and available result times exist in cached results; not added. Gaps need care for lapped/nonfinishers; positions gained needs care for grid zero/pit-lane starts. |
| `classified_position`, `points`, `laps_completed` | X | Position/classification, points and reported completed laps are available in results, but FastF1-specific classification formatting and count-of-lap-records semantics are not identical. Not added. |
| `starting_compound`, `compound_sequence`, `num_compounds_used`, `used_two_dry_compounds`, `used_inter_or_wet` | S when proposed; realised values excluded from classifier inputs | No tyre-compound or stint data in the results/pit-stop route. Stop records cannot establish tyre changes or mandatory-compound compliance. |
| `stint_1_compound` through `stint_5_compound`, `stint_1_length` through `stint_5_length` | S when proposed; realised values excluded from classifier inputs | No compound or stint boundaries. Pit intervals alone are not reliable tyre stints, especially across red flags. |
| `pit_lap_1` through `pit_lap_4`, `first_stop_pct` | S when proposed; realised values excluded from classifier inputs | Pit laps exist in cached stop records, but were not added. Pre-race stop percentages would need the planned distance; actual completed distance leaks information. |
| `started_on_used_tyres` | S when proposed; realised values excluded from classifier inputs | No `FreshTyre` or tyre-age observations. |
| `stops_under_sc` | X | Stop laps alone are insufficient; the SC/VSC timeline is absent. |
| `median_lap_time`, `best_lap_time`, `pace_vs_field` | X | No driver lap-time series, `IsAccurate` flags or green-track status. A fastest-lap result, where present, cannot reproduce these complete lap-based definitions. |
| `deg_rate_soft`, `deg_rate_medium`, `deg_rate_hard` | X | No compound, tyre life or lap-time series. |
| `avg_pit_loss` | X | Cached pit durations are not net lap-time loss; the specified driver-level timing calculation is unavailable. |
| `position_lap_1`, `lap1_positions_gained`, `position_at_half` | X | Race lap-position charts are not in this cache. The free route has separate lap endpoints, but these were not downloaded; results only give grid and final positions. |

SC/VSC lap counts also require a precise rule for mapping status changes to
laps; they should not be estimated from long stops. Existing race-weather and
actual-lap proxies remain documented exceptions in `features.py`; this change
does not reclassify them as information genuinely known before the race.

## Expanding-window validation update

Raw grid_position is reference only, excluded from model inputs. Grid zero is
a pit-lane start; grid_position_clean and pit_lane_start are inputs. The pit
feature is now circuit_pit_duration_prior (old pit_loss name removed).
Strictly earlier stop-history features and two strict-variant proxies are
specified in the root DATA_DICTIONARY.md. Source columns and row identifiers
are preserved. Current validation reports from src/train.py are
expanding_validation.txt, expanding_validation.csv and
expanding_validation_summary.csv. These use validation seasons 2021–2024,
training only on preceding seasons. They include all three historical
baselines, both random-forest variants, actual race counts and season spread.
2025 remains excluded from model fitting, prediction and evaluation.
