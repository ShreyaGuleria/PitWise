"""Pre-race historical features. No network access or model fitting."""

import pandas as pd

HISTORY_COLUMNS = ['circuit_avg_stops_prior', 'circuit_races_prior',
                   'circuit_avg_stops_last3', 'recent_avg_stops_prior',
                   'circuit_laps_last_year', 'circuit_month_air_temp_prior']


def history_features(table):
    """Stop means pool usable driver rows; windows count distinct usable races.

    Global recent history is the last three races at any circuit. Cold starts
    keep NaN means and a zero race count. Strict proxy weather averages race
    means (one vote per usable race); previous-year laps use the latest usable
    visit in exactly the preceding calendar season. These are historical
    proxies, not the current race's scheduled distance or a weather forecast.
    """
    dates = pd.to_datetime(table.race_date, errors='raise')
    usable = table.loc[table.is_usable.eq(True)].copy()
    usable['race_date'] = dates.loc[usable.index]
    races = usable.groupby(['season', 'round', 'circuit_id', 'race_date'], as_index=False).agg(
        stop_sum=('num_stops', 'sum'), drivers=('num_stops', 'count'),
        laps=('scheduled_laps', 'first'), air_temp=('air_temp_mean', 'first'))
    races = races.sort_values(['race_date', 'season', 'round'])
    result = pd.DataFrame(float('nan'), index=table.index, columns=HISTORY_COLUMNS)
    for (circuit, date, season), rows in table.groupby(['circuit_id', dates, 'season']):
        earlier = races.loc[races.race_date.lt(date)]
        circuit_history = earlier.loc[earlier.circuit_id.eq(circuit)]
        recent = circuit_history.tail(3)
        global_recent = earlier.tail(3)
        last_year = circuit_history.loc[circuit_history.season.eq(season - 1)].tail(1)
        month_history = circuit_history.loc[circuit_history.race_date.dt.month.eq(date.month)]
        def stop_mean(history):
            return history.stop_sum.sum() / history.drivers.sum() if len(history) else float('nan')
        result.loc[rows.index] = [
            stop_mean(circuit_history), len(circuit_history), stop_mean(recent),
            stop_mean(global_recent), last_year.laps.iloc[0] if len(last_year) else float('nan'),
            month_history.air_temp.mean(),
        ]
    result['circuit_races_prior'] = result.circuit_races_prior.astype(int)
    return result
