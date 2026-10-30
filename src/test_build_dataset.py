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

if __name__ == '__main__':
    test_no_stops()
    print('Checks passed.')

# Check combined flags and the shared long-stop signal without API requests.
def test_flags():
    import pandas as pd
    from build_dataset import flag_rows, possible_red_flag
    table = pd.DataFrame({'finish_status': ['Finished', '+1 Lap', 'Lapped', 'Retired'], 'num_stops': [2, 5, 0, 0]})
    marked = flag_rows(table)
    assert len(marked) == 4
    assert marked.is_finisher.tolist() == [True, True, True, False]
    assert marked.exclude_reason.tolist() == ['', 'outlier_stops', 'no_pit_data', 'retired;no_pit_data']
    records = [{'driverId': str(i), 'lap': '10', 'duration': '12:00.0'} for i in range(3)]
    assert possible_red_flag(records)
    assert not possible_red_flag(records[:2])
    assert flag_rows(table, True).exclude_reason.str.contains('possible_red_flag').all()

if __name__ == '__main__':
    test_flags()
    print('Flag checks passed.')
