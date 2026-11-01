from collections import Counter
from build_dataset import build_race_table

# A driver with no pit records must survive with zero stops.
def test_no_stops():
    race = {'date':'2023-03-05', 'time':'15:00:00Z', 'season':'2023', 'round':'1', 'Circuit':{'circuitName':'Bahrain'}}
    results = [{'Driver':{'code':'AAA','driverId':'a'},'grid':'1','status':'Finished'}]
    weather = {'hourly':{'time':['2023-03-05T14:00','2023-03-05T15:00','2023-03-05T16:00','2023-03-05T17:00'], 'temperature_2m':[99,20,22,24], 'precipitation':[5,0,0,0]}}
    table = build_race_table(race, results, Counter(), weather)
    assert len(table) == 1 and table.num_stops.iloc[0] == 0
    assert table.air_temp_mean.iloc[0] == 22 and not table.rain.iloc[0]
    assert table.precipitation_mm_total.iloc[0] == 0 and table.precipitation_mm_max.iloc[0] == 0
    weather['hourly']['precipitation'] = [99, 0.5, 1.0, 0.2]
    wet = build_race_table(race, results, Counter(), weather)
    assert wet.precipitation_mm_total.iloc[0] == 1.7 and wet.precipitation_mm_max.iloc[0] == 1
    assert wet.rain.iloc[0]

# Check combined flags and the shared long-stop signal without API requests.
def test_flags():
    import pandas as pd
    from build_dataset import flag_rows, possible_red_flag
    table = pd.DataFrame({'finish_status': ['Finished', '+1 Lap', 'Lapped', 'Retired'], 'num_stops': [2, 5, 0, 0]})
    marked = flag_rows(table)
    assert len(marked) == 4
    assert marked.is_finisher.tolist() == [True, True, True, False]
    assert marked.exclude_reason.tolist() == ['', 'outlier_stops', 'zero_stop_finisher', 'dnf']
    records = [{'driverId': str(i), 'lap': '10', 'duration': '12:00.0'} for i in range(3)]
    assert possible_red_flag(records)
    assert not possible_red_flag(records[:2])
    assert not possible_red_flag([{'driverId':'a', 'lap':'1', 'duration':''}])
    assert flag_rows(table, True).exclude_reason.tolist() == ['red_flag_race', 'red_flag_race', 'red_flag_race', 'dnf']
    assert marked.stops_class.iloc[:2].tolist() == ['2', '3+']
    assert marked.stops_class.iloc[2:].isna().all()
    dns = pd.DataFrame({'finish_status':['Did not start', 'Disqualified'], 'num_stops':[5, 5]})
    assert flag_rows(dns, True).exclude_reason.tolist() == ['dns', 'red_flag_race']

def test_new_rules():
    import pandas as pd
    from build_dataset import flag_rows, consecutive_lap_drivers, RULE_FORCED_STOPS

    records = [{'driverId': str(i), 'lap': str(lap)}
               for i in range(8) for lap in (10, 11, 11)]
    assert len(consecutive_lap_drivers(records)) == 8
    duplicates = [{'driverId': str(i), 'lap': '10'} for i in range(8) for _ in range(2)]
    assert not consecutive_lap_drivers(duplicates)
    assert not consecutive_lap_drivers([{'driverId': 'norris', 'lap': str(lap)}
                                       for lap in (10, 17, 27, 37, 47, 54)])
    table = pd.DataFrame({
        'season': [2025] * 6, 'round': [8] * 6,
        'finish_status': ['Did not start', 'Retired', 'Finished', 'Disqualified', 'Lapped', '+2 Laps'],
        'num_stops': [5, 5, 5, 0, 2, 1],
    })
    for season, number in RULE_FORCED_STOPS:
        forced = flag_rows(table.assign(season=season, round=number), True, records)
        assert forced.rule_forced_stops.all()
        assert forced.exclude_reason.tolist() == ['dns', 'dnf'] + ['rule_forced_stops'] * 4
        assert forced[['chaotic_race', 'suspect_stop_data', 'possible_red_flag']].all().all()
    ordinary = table.assign(round=7)
    assert flag_rows(ordinary, True, records).exclude_reason.tolist() == ['dns', 'dnf'] + ['red_flag_race'] * 4
    assert flag_rows(ordinary, False, records).exclude_reason.tolist() == ['dns', 'dnf'] + ['chaotic_race'] * 4
    ordinary.loc[0, 'num_stops'] = 2
    suspect = flag_rows(ordinary, False, records)
    assert suspect.exclude_reason.tolist() == ['dns', 'dnf'] + ['suspect_stop_data'] * 4
    assert not flag_rows(ordinary, False, records[:-3]).suspect_stop_data.any()
    clean = flag_rows(ordinary, False, duplicates)
    assert clean.exclude_reason.tolist() == ['dns', 'dnf', 'outlier_stops', 'zero_stop_finisher', '', '']
    assert clean.is_finisher.tolist() == [False, False, True, True, True, True]
    assert clean.num_stops.tolist() == ordinary.num_stops.tolist()
    assert clean.stops_class.fillna('').tolist() == ['2', '3+', '3+', '', '2', '1']


def test_cached_rebuild():
    import contextlib
    import io
    import socket
    from unittest.mock import patch
    import pandas as pd
    import build_dataset as builder

    path = builder.ROOT / 'data' / 'processed' / 'stops_dataset.csv'
    path.read_bytes().decode('utf-8', errors='strict')
    saved = pd.read_csv(path, keep_default_na=False, float_precision='round_trip')
    failures = []
    with patch.object(builder, 'CACHE_ONLY', True), \
         patch('requests.sessions.Session.request', side_effect=AssertionError('Network forbidden')) as request, \
         patch.object(socket, 'create_connection', side_effect=AssertionError('Network forbidden')) as connection, \
         contextlib.redirect_stdout(io.StringIO()):
        rebuilt = pd.concat([builder.build_season_table(s, failures) for s in range(2018, 2026)], ignore_index=True)
        # Also prove cache misses fail without invoking a network request.
        try:
            builder.fetch_json('https://invalid.example/offline-test')
        except RuntimeError as error:
            assert 'downloads disabled' in str(error)
        else:
            raise AssertionError('Cache miss must fail closed')
    request.assert_not_called()
    connection.assert_not_called()
    assert failures == []
    assert len(rebuilt) == len(saved) == 3458
    keys = ['season', 'round', 'driver']
    assert not rebuilt.duplicated(keys).any() and not saved.duplicated(keys).any()
    assert rebuilt.groupby(['season', 'round']).ngroups == 173
    unchanged = [c for c in saved.columns if c not in (
        'exclude_reason', 'rule_forced_stops', 'chaotic_race', 'suspect_stop_data')]
    pd.testing.assert_frame_equal(
        rebuilt[unchanged].fillna('').reset_index(drop=True), saved[unchanged],
        check_exact=False, rtol=1e-12, atol=1e-12)
    usable = rebuilt.exclude_reason.eq('')
    assert usable.sum() == 2507
    assert rebuilt.loc[usable, 'stops_class'].value_counts().to_dict() == {'1': 1239, '2': 1039, '3+': 229}
    assert rebuilt.rule_forced_stops.sum() == 60
    flags = ['rule_forced_stops', 'chaotic_race', 'suspect_stop_data']
    assert all(rebuilt[c].dtype == bool for c in flags)
    if all(c in saved for c in flags):
        pd.testing.assert_frame_equal(rebuilt[flags + ['exclude_reason']], saved[flags + ['exclude_reason']])
    assert rebuilt.loc[rebuilt.suspect_stop_data, ['season', 'round']].drop_duplicates().shape[0] == 13
    # Verify the audited 16 usable drivers in 14 races directly against raw cache.
    remaining_rows, remaining_races = 0, 0
    with patch.object(builder, 'CACHE_ONLY', True), \
         patch('requests.sessions.Session.request', side_effect=AssertionError('Network forbidden')):
        for (season, number), rows in rebuilt.groupby(['season', 'round']):
            _, records = builder.fetch_pit_stops(season, number, include_records=True)
            drivers = builder.consecutive_lap_drivers(records)
            _, results = builder.fetch_race_results(season, number)
            codes = {r['Driver']['code'] for r in results if r['Driver']['driverId'] in drivers}
            count = (rows.exclude_reason.eq('') & rows.driver.isin(codes)).sum()
            remaining_rows += int(count)
            remaining_races += int(count > 0)
    assert (remaining_rows, remaining_races) == (16, 14)


if __name__ == '__main__':
    for test in (test_no_stops, test_flags, test_new_rules, test_cached_rebuild):
        test()
        print(f'{test.__name__}: passed')
