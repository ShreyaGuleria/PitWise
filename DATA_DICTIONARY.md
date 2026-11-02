# Data dictionary

One row in `data/processed/features.csv` is one driver in one race. Join keys:
`season`, `round`, `driver`. Model only usable rows labelled 1, 2 or 3+.

**INPUT** means known before the start; **TARGET** is the quantity to predict;
**NOT_FOR_MODEL** excludes post-start observations and outcomes. Identifiers
and duplicate display fields also use NOT_FOR_MODEL despite being known
beforehand; their exclusions follow the explicit feature specification.

Three legacy allowlisted inputs (`scheduled_laps`, `air_temp_mean`,
`precipitation_mm_total`) are retrospective proxies. Their dictionary role is
NOT_FOR_MODEL because the current source is not pre-race information, despite
their retained membership in FEATURE_COLUMNS. Remove or replace them for a
strict pre-race model. This documents the exception without changing that list.

| Column | Plain-English meaning | Source | Role |
|---|---|---|---|
| season | Championship year; join/split metadata. | Jolpica | NOT_FOR_MODEL |
| round | Race number in the season; join metadata. | Jolpica | NOT_FOR_MODEL |
| circuit | Display name; use circuit_id instead. | Jolpica | NOT_FOR_MODEL |
| race_date | Scheduled date; metadata, with explicit derived calendar inputs. | Jolpica | NOT_FOR_MODEL |
| driver | Driver code; join identifier excluded from inputs. | Jolpica | NOT_FOR_MODEL |
| grid_position | Original supplied grid, zero denotes pit-lane start; reference only, use cleaned column. | Jolpica | NOT_FOR_MODEL |
| finish_status | Final status, such as Finished, lapped or retirement. | Jolpica | NOT_FOR_MODEL |
| num_stops | Observed pit-record count that directly determines the target class. | derived (Jolpica) | NOT_FOR_MODEL |
| air_temp_mean | Mean historical air temperature in degrees C from scheduled start to +2h, not a forecast. | Open-Meteo | NOT_FOR_MODEL |
| rain | Whether any sampled race-window precipitation is positive; weak observed rain proxy. | derived (Open-Meteo) | NOT_FOR_MODEL |
| precipitation_mm_total | Sum of race-window hourly precipitation in mm; not a forecast. | derived (Open-Meteo) | NOT_FOR_MODEL |
| precipitation_mm_max | Maximum race-window hourly precipitation in mm. | derived (Open-Meteo) | NOT_FOR_MODEL |
| is_finisher | Whether final status meets the project's Finished/lapped/Disqualified rule. | derived | NOT_FOR_MODEL |
| possible_red_flag | At least three drivers have 600+ second pit durations on a common lap; stoppage heuristic. | derived (Jolpica) | NOT_FOR_MODEL |
| exclude_reason | First matching exclusion rule; blank means usable. | derived | NOT_FOR_MODEL |
| stops_class | Target: 1 stop, 2 stops or 3+; blank for zero/unknown counts. | derived | TARGET |
| race_name | Display name of the Grand Prix. | Jolpica | NOT_FOR_MODEL |
| red_flag_evidence | Mapping from long-stop laps to affected-driver counts supporting the stoppage flag. | derived (Jolpica) | NOT_FOR_MODEL |
| data_complete | Whether collection obtained required data; known after collection. | derived | NOT_FOR_MODEL |
| rule_forced_stops | Hand-coded mandatory stop/tyre-limit races; known pre-race but reserved for filtering. | derived | NOT_FOR_MODEL |
| chaotic_race | At least three drivers have five or more observed stops. | derived | NOT_FOR_MODEL |
| suspect_stop_data | At least eight drivers have consecutive-lap pit records. | derived (Jolpica) | NOT_FOR_MODEL |
| circuit_id | Stable circuit identifier for location and history. | Jolpica | INPUT |
| latitude | Circuit latitude in degrees. | Jolpica | INPUT |
| longitude | Circuit longitude in degrees. | Jolpica | INPUT |
| constructor | Raw team name; use constructor_group instead. | Jolpica | NOT_FOR_MODEL |
| scheduled_laps | Despite its name, maximum actual completed laps, not scheduled distance. | derived (Jolpica) | NOT_FOR_MODEL |
| altitude_m | Model elevation at circuit coordinates in metres, not surveyed track altitude. | Open-Meteo | INPUT |
| month | Month from race_date. | derived | INPUT |
| day_of_year | Calendar day number from race_date. | derived | INPUT |
| is_street_circuit | Hand-coded flag for Monaco, Baku, Singapore, Jeddah, Las Vegas and Miami. | derived | INPUT |
| constructor_group | Team name grouped across supported historical aliases. | derived | INPUT |
| is_2022_plus | Whether season is 2022 or later; coarse regulation-era indicator. | derived | INPUT |
| is_usable | Whether exclusion reason is blank; outcome-dependent selection. | derived | NOT_FOR_MODEL |
| circuit_pit_duration_prior | Median pit duration below 60s from strictly earlier usable circuit races; seconds, NaN without history. | derived (Jolpica) | INPUT |
| pit_lane_start | True exactly when original grid_position is zero; the pit-lane-start flag. | derived | INPUT |
| grid_position_clean | Original grid position with zero changed to 20; 20 for pit-lane starts, distinguished by pit_lane_start. | derived | INPUT |

| circuit_avg_stops_prior | Pooled usable-driver mean stops over strictly earlier usable circuit races; NaN without history. | derived | INPUT |
| circuit_races_prior | Number of distinct strictly earlier usable races at the circuit; zero without history. | derived | INPUT |
| circuit_avg_stops_last3 | Pooled usable-driver mean stops over the last three earlier usable circuit races, or fewer if unavailable. | derived | INPUT |
| recent_avg_stops_prior | Pooled usable-driver mean stops over the last three earlier usable races globally, or fewer if unavailable. | derived | INPUT |
| circuit_laps_last_year | Actual laps from the latest earlier usable circuit visit exactly in the preceding season; strict variant proxy, NaN if absent. | derived (Jolpica) | INPUT |
| circuit_month_air_temp_prior | Equal-race mean historical air temperature at the circuit in the same month, using strictly earlier usable races; strict variant proxy. | derived (Open-Meteo) | INPUT |

`num_stops` would give away the answer. `finish_status`, final finisher flags,
observed anomalies and exclusion/usability fields use race outcomes. Including
them leaks information unavailable at prediction time. `stops_class` is passed
as the label, never as a feature. The code's NOT_FOR_MODEL list includes the
target as a guard against accidental feature selection.

The pit-duration prior pools all drivers' stops in races with at least one
usable driver, once per race. Same-date/future races are excluded; earlier
same-season races may contribute. Its 60-second duration cutoff is separate
from the 600-second red-flag race heuristic. It measures pit-lane duration,
not net race time lost. No history stays NaN, saved as an empty CSV cell.
With `keep_default_na=False`, also pass
`na_values={'circuit_pit_duration_prior': ['']}`. Fit imputation on training only.
Grid 20 and recoded pit-lane starts remain distinguishable via pit_lane_start.

For `splits.csv`, season/round/driver retain the meanings above. Its additional
`split` column is derived NOT_FOR_MODEL metadata assigning train, validation or
test by season. Original grid_position is unchanged and excluded from inputs.

The four stop-history features and two strict-variant proxies exclude same-date
and later races. Stop means pool usable drivers; the race count is distinct
season/round pairs. The strict model removes current weather and actual laps,
using only historical distance/temperature proxies. No history means NaN,
except the race count which is zero. CSV readers with keep_default_na=False
should pass empty-cell NA values for all ALL_FEATURE_COLUMNS, as the bundled
loader does. `circuit_pit_duration_prior` is the new name for the previous
`circuit_pit_loss_prior`; no duplicate compatibility column is retained.
