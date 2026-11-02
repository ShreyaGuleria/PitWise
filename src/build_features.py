"""Enrich the saved driver dataset without rebuilding or changing its columns."""

import argparse
import contextlib
import io

import pandas as pd

import build_dataset as dataset
from features import FEATURE_COLUMNS, ALL_FEATURE_COLUMNS, NOT_FOR_MODEL
from history import history_features

KEYS = ['season', 'round', 'driver']
# Requested hand-coded street circuits: Monaco, Baku, Singapore, Jeddah,
# Las Vegas and Miami. All others (including Albert Park) are False.
STREET_CIRCUITS = {'monaco', 'baku', 'marina_bay', 'jeddah', 'vegas', 'miami'}
CONSTRUCTOR_GROUPS = {
    'Force India': 'Aston Martin', 'Racing Point': 'Aston Martin',
    'Aston Martin': 'Aston Martin',
    'Toro Rosso': 'RB', 'AlphaTauri': 'RB', 'RB': 'RB', 'RB F1 Team': 'RB',
    'Alfa Romeo': 'Sauber', 'Sauber': 'Sauber', 'Kick Sauber': 'Sauber',
    'Renault': 'Alpine', 'Alpine': 'Alpine', 'Alpine F1 Team': 'Alpine',
}


def circuit_pit_duration_prior(table):
    """Median seconds per stop, pooled over strictly earlier usable circuit races.

    A usable race has at least one usable driver. Include all cached stops in
    those races, not only stops by usable drivers. Cache misses raise without
    downloading. Exclude invalid durations and durations >=60s; retain NaN when
    there are no eligible historical stops. This is not net pit-stop time loss.
    """
    races = table.groupby(['season', 'round', 'circuit_id', 'race_date'], as_index=False).agg(
        usable=('is_usable', 'any'))
    durations = []
    previous_cache_only = dataset.CACHE_ONLY
    dataset.CACHE_ONLY = True
    try:
        for race in races.loc[races.usable].itertuples(index=False):
            _, stops = dataset.fetch_pit_stops(race.season, race.round, include_records=True)
            for stop in stops:
                try:
                    seconds = 0.0
                    for part in str(stop.get('duration', '')).split(':'):
                        seconds = seconds * 60 + float(part)
                except (TypeError, ValueError):
                    continue
                if 0 < seconds < 60:
                    durations.append((race.circuit_id, race.race_date, seconds))
    finally:
        dataset.CACHE_ONLY = previous_cache_only
    history = pd.DataFrame(durations, columns=['circuit_id', 'race_date', 'duration'])
    history['race_date'] = pd.to_datetime(history.race_date, errors='raise')
    dates = pd.to_datetime(table.race_date, errors='raise')
    prior = pd.Series(float('nan'), index=table.index, name='circuit_pit_duration_prior')
    for (circuit, date), rows in table.groupby(['circuit_id', dates]):
        earlier = history.loc[history.circuit_id.eq(circuit) & history.race_date.lt(date)]
        prior.loc[rows.index] = earlier.duration.median()
    return prior


def build_features(source):
    if source.duplicated(KEYS).any():
        raise ValueError('Duplicate input driver/race keys')
    metadata, elevations = [], {}
    for season, number in source[['season', 'round']].drop_duplicates().itertuples(index=False, name=None):
        race, results = dataset.fetch_race_results(season, number)
        circuit = race['Circuit']
        circuit_id = circuit['circuitId']
        location = circuit['Location']
        weather = dataset.fetch_weather(race)
        if pd.notna(weather.get('elevation')):
            elevation = float(weather['elevation'])
            if circuit_id in elevations and elevations[circuit_id] != elevation:
                raise ValueError(f'Conflicting cached elevations for {circuit_id}')
            elevations[circuit_id] = elevation
        # Requested retrospective proxy, deliberately not called planned distance.
        laps = max(int(result['laps']) for result in results)
        for result in results:
            metadata.append({
                'season': season, 'round': number, 'driver': result['Driver']['code'],
                'circuit_id': circuit_id, 'latitude': float(location['lat']),
                'longitude': float(location['long']),
                'constructor': result['Constructor']['name'], 'scheduled_laps': laps,
            })
    metadata = pd.DataFrame(metadata)
    # Only if no cached race weather supplied this circuit's elevation: one
    # cached, politely paced request per circuit through the existing fetcher.
    for row in metadata.drop_duplicates('circuit_id').itertuples(index=False):
        if row.circuit_id not in elevations:
            response = dataset.fetch_json('https://api.open-meteo.com/v1/elevation', {
                'latitude': row.latitude, 'longitude': row.longitude,
            })
            elevations[row.circuit_id] = response['elevation'][0]
    table = source.merge(metadata, on=KEYS, how='left', sort=False,
                         validate='one_to_one', indicator=True)
    if not table['_merge'].eq('both').all():
        raise ValueError('Some input driver/race keys have no cached race result')
    table = table.drop(columns='_merge')
    table['altitude_m'] = table['circuit_id'].map(elevations)
    date = pd.to_datetime(table['race_date'], format='%Y-%m-%d', errors='raise')
    table['month'] = date.dt.month
    table['day_of_year'] = date.dt.dayofyear
    table['is_street_circuit'] = table['circuit_id'].isin(STREET_CIRCUITS)
    table['constructor_group'] = table['constructor'].map(CONSTRUCTOR_GROUPS).fillna(table['constructor'])
    table['is_2022_plus'] = table['season'].ge(2022)
    table['is_usable'] = table['exclude_reason'].eq('')
    table['pit_lane_start'] = table.grid_position.eq(0)
    table['grid_position_clean'] = table.grid_position.replace(0, 20)
    table['circuit_pit_duration_prior'] = circuit_pit_duration_prior(table)
    table = table.join(history_features(table))
    pd.testing.assert_frame_equal(table[source.columns], source.reset_index(drop=True), check_exact=True)
    if set(table.columns) != set(ALL_FEATURE_COLUMNS) | set(NOT_FOR_MODEL):
        raise ValueError('Every column must be explicitly classified in features.py')
    return table


def print_report(table):
    output = dataset.ROOT / 'reports'
    output.mkdir(exist_ok=True)
    # Count blank CSV fields as nulls, but distinguish their intentional meanings.
    nulls = table.replace('', pd.NA).isna().sum().rename('null_count')
    print('Null/blank count per column:\n' + nulls.to_string())
    print('\nEmpty exclude_reason means usable; empty stops_class means zero/unknown stops.')
    missing_inputs = table[FEATURE_COLUMNS].replace('', pd.NA).isna().sum()
    print('\nUnresolved missing model inputs:\n' + missing_inputs[missing_inputs.gt(0)].to_string())
    races = table.groupby(['circuit_id', 'season', 'round'], as_index=False).agg(
        latitude=('latitude', 'first'), longitude=('longitude', 'first'),
        altitude_m=('altitude_m', 'first'), is_street_circuit=('is_street_circuit', 'first'),
        usable=('is_usable', 'any'))
    circuits = races.groupby('circuit_id', as_index=False).agg(
        latitude=('latitude', 'first'), longitude=('longitude', 'first'),
        altitude_m=('altitude_m', 'first'), is_street_circuit=('is_street_circuit', 'first'),
        number_of_races=('round', 'size'), number_of_usable_races=('usable', 'sum'))
    print('\nCircuits (usable race = at least one usable driver):\n' + circuits.to_string(index=False))
    mapping = table[['constructor', 'constructor_group']].drop_duplicates().sort_values('constructor')
    print('\nFinal constructor mapping:\n' + mapping.to_string(index=False))
    print(f'\nRetained {len(table)} rows; usable: {table.is_usable.sum()}.')
    print('\nLimitations: scheduled_laps = actual maximum completed laps; weather = race-window reanalysis.\n'
          'These requested inputs are retrospective proxies, not strictly pre-race observations.\n'
          'altitude_m uses Open-Meteo elevation, not a surveyed track elevation.')
    if not missing_inputs.any():
        print('No missing feature values were imputed; no unresolved missing feature values remain.')
    else:
        print('Some model inputs remain missing; see counts above.')
    nulls.to_csv(output / 'feature_null_counts.csv', encoding='utf-8')
    circuits.to_csv(output / 'feature_circuits.csv', index=False, encoding='utf-8')
    mapping.to_csv(output / 'constructor_mapping.csv', index=False, encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-only', action='store_true', help='Refuse downloads on cache misses.')
    args = parser.parse_args()
    dataset.CACHE_ONLY = args.cache_only
    output = dataset.ROOT / 'data' / 'processed'
    source = pd.read_csv(output / 'stops_dataset.csv', keep_default_na=False, float_precision='round_trip')
    table = build_features(source)
    candidate = output / 'features.candidate.csv'
    table.to_csv(candidate, index=False, encoding='utf-8')
    candidate.read_bytes().decode('utf-8', errors='strict')
    saved = pd.read_csv(candidate, keep_default_na=False, float_precision='round_trip',
                        na_values={column: [''] for column in ALL_FEATURE_COLUMNS})
    pd.testing.assert_frame_equal(saved[source.columns], source, check_exact=True)
    pd.testing.assert_frame_equal(saved, table, check_exact=True, check_dtype=False)
    candidate.replace(output / 'features.csv')
    report = io.StringIO()
    with contextlib.redirect_stdout(report):
        print_report(table)
    (dataset.ROOT / 'reports' / 'features_report.txt').write_text(report.getvalue(), encoding='utf-8')
    print(report.getvalue(), end='')


if __name__ == '__main__':
    main()
