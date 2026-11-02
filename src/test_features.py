from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd

import build_dataset as dataset
from build_features import build_features, circuit_pit_loss_prior, CONSTRUCTOR_GROUPS, STREET_CIRCUITS
from features import FEATURE_COLUMNS, NOT_FOR_MODEL


def load_source():
    return pd.read_csv(dataset.ROOT / 'data' / 'processed' / 'stops_dataset.csv',
                       keep_default_na=False, float_precision='round_trip')


def test_cached_features():
    source = load_source()
    original_bytes = (dataset.ROOT / 'data' / 'processed' / 'stops_dataset.csv').read_bytes()
    with patch.object(dataset, 'CACHE_ONLY', True), \
         patch('requests.sessions.Session.request', side_effect=AssertionError('Network forbidden')) as request:
        table = build_features(source)
        for (season, number), rows in table.groupby(['season', 'round']):
            race, results = dataset.fetch_race_results(season, number)
            assert rows.scheduled_laps.eq(max(int(r['laps']) for r in results)).all()
            assert rows.altitude_m.eq(dataset.fetch_weather(race)['elevation']).all()
            constructors = {r['Driver']['code']: r['Constructor']['name'] for r in results}
            assert rows.constructor.tolist() == rows.driver.map(constructors).tolist()
    request.assert_not_called()
    assert len(table) == 3458 and not table.duplicated(['season', 'round', 'driver']).any()
    assert table.groupby(['season', 'round']).ngroups == 173
    assert table.circuit_id.nunique() == 31
    pd.testing.assert_frame_equal(table[source.columns], source, check_exact=True)
    assert (dataset.ROOT / 'data' / 'processed' / 'stops_dataset.csv').read_bytes() == original_bytes
    other_inputs = [column for column in FEATURE_COLUMNS if column != 'circuit_pit_loss_prior']
    assert not table[other_inputs].replace('', pd.NA).isna().any().any()
    assert table.circuit_pit_loss_prior.isna().any()  # Deliberate: no historical fallback.
    assert table.circuit_pit_loss_prior.dropna().between(0, 60, inclusive='neither').all()
    assert table.is_usable.equals(table.exclude_reason.eq('')) and table.is_usable.sum() == 2507
    assert table.is_2022_plus.equals(table.season.ge(2022))
    date = pd.to_datetime(table.race_date)
    assert table.month.equals(date.dt.month) and table.day_of_year.equals(date.dt.dayofyear)
    assert table.is_street_circuit.equals(table.circuit_id.isin(STREET_CIRCUITS))
    assert table.loc[table.is_street_circuit, 'circuit_id'].nunique() == 6
    assert table.loc[table.circuit_id.eq('baku'), 'altitude_m'].eq(-18).all()
    assert table.loc[table.season.eq(2021) & table.circuit_id.eq('spa'), 'scheduled_laps'].eq(1).all()
    assert set(FEATURE_COLUMNS).isdisjoint(NOT_FOR_MODEL)
    assert len(FEATURE_COLUMNS) == len(set(FEATURE_COLUMNS)) == 14
    assert len(NOT_FOR_MODEL) == len(set(NOT_FOR_MODEL))
    assert set(table.columns) == set(FEATURE_COLUMNS) | set(NOT_FOR_MODEL)
    assert table.loc[table.stops_class.eq(''), 'num_stops'].eq(0).all()
    # The saved UTF-8 artifact must reproduce the in-memory enrichment exactly.
    path = dataset.ROOT / 'data' / 'processed' / 'features.csv'
    if path.exists():
        path.read_bytes().decode('utf-8', errors='strict')
        saved = pd.read_csv(path, keep_default_na=False, float_precision='round_trip',
                            na_values={'circuit_pit_loss_prior': ['']})
        # CSV inference widens the calendar int32 fields to int64.
        pd.testing.assert_frame_equal(saved, table, check_exact=True, check_dtype=False)


def test_elevation_fallback_once_per_circuit():
    source = load_source()
    bahrain = source.loc[source.circuit.eq('Bahrain International Circuit')]
    races = bahrain[['season', 'round']].drop_duplicates().iloc[:2]
    source = bahrain.merge(races, on=['season', 'round'], how='inner').reset_index(drop=True)
    with patch.object(dataset, 'CACHE_ONLY', True):
        results = {(s, r): dataset.fetch_race_results(s, r)
                   for s, r in races.itertuples(index=False, name=None)}
    assert len(results) == 2
    with patch.object(dataset, 'fetch_race_results', side_effect=lambda s, r: results[(s, r)]), \
         patch.object(dataset, 'fetch_weather', return_value={}), \
         patch.object(dataset, 'fetch_json', return_value={'elevation': [8.0]}) as fetch, \
         patch.object(dataset, 'fetch_pit_stops', return_value=({}, [])):
        table = build_features(source)
    fetch.assert_called_once_with('https://api.open-meteo.com/v1/elevation', {
        'latitude': 26.0325, 'longitude': 50.5106,
    })
    assert table.altitude_m.eq(8).all()
    # A later cached race's elevation should avoid even that fallback request.
    with patch.object(dataset, 'fetch_race_results', side_effect=lambda s, r: results[(s, r)]), \
         patch.object(dataset, 'fetch_weather', side_effect=[{}, {'elevation': 8.0}]), \
         patch.object(dataset, 'fetch_json') as fetch, \
         patch.object(dataset, 'fetch_pit_stops', return_value=({}, [])):
        assert build_features(source).altitude_m.eq(8).all()
    fetch.assert_not_called()


def test_constructor_aliases():
    for name in ['Force India', 'Racing Point', 'Aston Martin']:
        assert CONSTRUCTOR_GROUPS[name] == 'Aston Martin'
    for name in ['Toro Rosso', 'AlphaTauri', 'RB', 'RB F1 Team']:
        assert CONSTRUCTOR_GROUPS[name] == 'RB'
    for name in ['Alfa Romeo', 'Sauber', 'Kick Sauber']:
        assert CONSTRUCTOR_GROUPS[name] == 'Sauber'
    for name in ['Renault', 'Alpine', 'Alpine F1 Team']:
        assert CONSTRUCTOR_GROUPS[name] == 'Alpine'


def test_pit_loss_history():
    table = pd.DataFrame([
        (2023, 1, 'a', '2023-01-01', True),
        (2023, 1, 'a', '2023-01-01', False),  # Same race: do not count stops twice.
        (2023, 2, 'a', '2023-02-01', False),  # Entirely unusable race.
        (2023, 3, 'a', '2023-03-01', True),
        (2023, 4, 'a', '2023-03-01', True),   # Different race, same date.
        (2023, 5, 'a', '2023-04-01', True),
        (2023, 6, 'b', '2023-01-01', True),   # Other circuit cannot contribute.
        (2023, 7, 'c', '2023-01-01', True),   # No qualifying durations.
        (2023, 8, 'c', '2023-02-01', True),
    ], columns=['season', 'round', 'circuit_id', 'race_date', 'is_usable'])
    raw = {1: ['20', '30', '59.9', '60', '1:00.000', '12:00', '', None, 'NaN', '-1', 'inf'],
           2: ['1'], 3: ['40'], 4: ['50'], 5: ['2'], 6: ['3'], 7: ['60'], 8: []}

    def cached_stops(season, number, include_records):
        assert dataset.CACHE_ONLY and include_records
        return {}, [{'duration': duration} for duration in raw[number]]

    with patch.object(dataset, 'fetch_pit_stops', side_effect=cached_stops) as fetch, \
         patch('requests.sessions.Session.request', side_effect=AssertionError('Network forbidden')):
        before = dataset.CACHE_ONLY
        prior = circuit_pit_loss_prior(table)
        assert dataset.CACHE_ONLY == before
        assert fetch.call_count == 7
        assert all(call.args[1] != 2 for call in fetch.call_args_list)
        assert prior.iloc[:2].isna().all()
        assert prior.iloc[2] == 30  # Excluded rows still receive a prior.
        assert prior.iloc[3] == prior.iloc[4] == 30  # Neither same-date race is visible.
        assert prior.iloc[5] == 40  # Pooled stop median, not mean of race medians.
        assert prior.iloc[6:].isna().all()
        raw[3], raw[4], raw[5] = ['1'], ['1'], ['59']
        changed = circuit_pit_loss_prior(table)
        pd.testing.assert_series_equal(prior.iloc[:5], changed.iloc[:5])
        # Row order cannot change which history is available.
        shuffled = table.sample(frac=1, random_state=7)
        pd.testing.assert_series_equal(circuit_pit_loss_prior(shuffled).sort_index(), changed)


def test_pit_loss_cache_miss():
    table = pd.DataFrame([(2023, 1, 'missing', '2023-01-01', True)],
                         columns=['season', 'round', 'circuit_id', 'race_date', 'is_usable'])
    with TemporaryDirectory(dir=dataset.ROOT / '.tmp') as directory, \
         patch.object(dataset, 'ROOT', Path(directory)), \
         patch('requests.sessions.Session.request', side_effect=AssertionError('Network forbidden')) as request:
        before = dataset.CACHE_ONLY
        try:
            circuit_pit_loss_prior(table)
        except RuntimeError as error:
            assert 'downloads disabled' in str(error)
        else:
            raise AssertionError('A missing cache must fail without downloading')
        assert dataset.CACHE_ONLY == before
        request.assert_not_called()


if __name__ == '__main__':
    for test in (test_cached_features, test_elevation_fallback_once_per_circuit, test_constructor_aliases,
                 test_pit_loss_history, test_pit_loss_cache_miss):
        test()
        print(f'{test.__name__}: passed')
