"""Explicit model inputs; select these by name rather than using all columns."""

# Requested inputs, with three accepted retrospective proxies:
# scheduled_laps is MAX ACTUAL completed laps, not the planned race distance.
# Shortened races (e.g. Belgium 2021) therefore expose post-start information.
# air_temp_mean and precipitation_mm_total are race-window reanalysis, not
# forecasts available at the start. Replace these three with planned laps and
# archived pre-start forecasts before claiming a strictly pre-race evaluation.
FEATURE_COLUMNS = [
    'circuit_id', 'latitude', 'longitude', 'altitude_m', 'month', 'day_of_year',
    'is_street_circuit', 'constructor_group', 'scheduled_laps', 'is_2022_plus',
    'grid_position', 'air_temp_mean', 'precipitation_mm_total',
    # F: median cached pit duration (<60s) from strictly earlier usable circuit races;
    # NaN when no history exists. Pit-lane duration is a proxy, not net time lost.
    'circuit_pit_loss_prior',
]

NOT_FOR_MODEL = [
    'num_stops',           # Observed target count; directly reveals the label.
    'stops_class',         # Target label to predict, never an input.
    'finish_status',       # Outcome determined after the race starts.
    'is_finisher',         # Derived from the final finish status.
    'exclude_reason',      # Selection rule uses outcomes and target counts.
    'possible_red_flag',   # Inferred from pit durations recorded during the race.
    'red_flag_evidence',   # Actual long-stop laps/drivers observed during the race.
    'chaotic_race',        # Computed from observed driver stop counts.
    'suspect_stop_data',   # Computed from pit records observed during the race.
    'is_usable',           # Derived from outcome-dependent exclusions.
    'data_complete',       # Post-collection availability, not a race input.
    'rain',                # Observed race-window weather; not a pre-start forecast.
    'precipitation_mm_max', # Observed race-window maximum; not a pre-start forecast.
    'rule_forced_stops',   # Known rule, but reserved for filtering per this feature specification.
    'season',              # Metadata; use the explicit era input instead.
    'round',               # Race identifier, excluded from this feature specification.
    'driver',              # Join identifier, not an input in this specification.
    'race_date',           # Date metadata; month/day_of_year are explicit inputs.
    'circuit',             # Display name; use the stable circuit_id input.
    'race_name',           # Display name, not an additional model input.
    'constructor',         # Raw team name; use constructor_group instead.
]
