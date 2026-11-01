import argparse
import hashlib
from email.utils import parsedate_to_datetime
from datetime import datetime, timezone, timedelta
import json
import time
from collections import Counter
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
BASE = 'https://api.jolpi.ca/ergast/f1/'
CACHE_ONLY = False
RULE_FORCED_STOPS = {
    (2023, 17),  # Qatar: 18-lap tyre cap; https://www.fia.com/news/fia-statement-treatment-tyres-2023-fia-formula-1-qatar-grand-prix-update
    (2025, 8),  # Monaco: two mandatory stops; https://www.fia.com/news/f1-lando-norris-wins-monaco-gp-ahead-charles-leclerc-and-oscar-piastri
    (2025, 23),  # Qatar: 25-lap tyre cap; https://press.pirelli.com/a-maximum-of-25-laps-per-tyre-set-in-qatar/
}


# Read cached JSON first; pace downloads and retry rate limits without hammering.
def fetch_json(url, params=None):
    cache = ROOT / 'cache'
    cache.mkdir(exist_ok=True)
    key = hashlib.sha256((url + json.dumps(params, sort_keys=True)).encode()).hexdigest()
    path = cache / (key + '.json')
    if path.exists():
        return json.loads(path.read_text(encoding='utf-8'))
    if CACHE_ONLY:
        raise RuntimeError(f'Cache miss (downloads disabled): {url}')
    for attempt in range(4):
        time.sleep(2)
        try:
            response = requests.get(url, params=params, timeout=30)
        except (requests.ConnectionError, requests.Timeout):
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)
            continue
        if response.status_code == 429:
            retry_after = response.headers.get('Retry-After', '5')
            try:
                delay = float(retry_after)
            except ValueError:
                delay = (parsedate_to_datetime(retry_after) - datetime.now(timezone.utc)).total_seconds()
            wait = max(5 * 2 ** attempt, delay)
            if wait > 60 or attempt == 3:
                raise RuntimeError('API rate limit reached. Please try again later.')
            print(f'Rate limited; waiting {wait:.0f} seconds.')
            time.sleep(wait)
            continue
        response.raise_for_status()
        data = response.json()
        path.write_text(json.dumps(data, indent=2), encoding='utf-8')
        return data


# Retrieve all pages for this one race, checking the API's total record count.
def fetch_race_records(endpoint, field, season=2023, round_number=1):
    records = []
    offset = 0
    race = None
    while True:
        data = fetch_json(f'{BASE}{season}/{round_number}/{endpoint}/', {'limit': 100, 'offset': offset})['MRData']
        races = data['RaceTable']['Races']
        page = races[0].get(field, []) if races else []
        race = races[0] if races else race
        records.extend(page)
        offset += len(page)
        if offset >= int(data['total']):
            return race, records
        if not page:
            raise RuntimeError('Incomplete API pagination; refusing to undercount.')


# Results include the schedule, start time, circuit coordinates, and all drivers.
def fetch_race_results(season=2023, round_number=1):
    return fetch_race_records('results', 'Results', season, round_number)


# Count pit-stop records rather than inferred tyre stints.
def fetch_pit_stops(season=2023, round_number=1, include_records=False):
    _, stops = fetch_race_records('pitstops', 'PitStops', season, round_number)
    counts = Counter(stop['driverId'] for stop in stops)
    return (counts, stops) if include_records else counts


# Fetch hourly historical weather for the circuit on the scheduled race date.
def fetch_weather(race):
    location = race['Circuit']['Location']
    return fetch_json('https://archive-api.open-meteo.com/v1/archive', {
        'latitude': location['lat'], 'longitude': location['long'],
        'start_date': race['date'],
        'end_date': (datetime.fromisoformat(race['date'] + 'T' + race['time']) + timedelta(hours=2)).date().isoformat(),
        'hourly': 'temperature_2m,precipitation', 'timezone': 'UTC',
    })


# Keep every result and use inclusive UTC hourly timestamps from start to +2h.
def build_race_table(race, results, stops, weather):
    start = datetime.fromisoformat(race['date'] + 'T' + race['time'])
    hourly = pd.DataFrame(weather['hourly'])
    timestamps = pd.to_datetime(hourly['time'], utc=True)
    end = start + timedelta(hours=2)
    window = hourly.loc[(timestamps >= start) & (timestamps <= end)]
    if window.empty or window[['temperature_2m', 'precipitation']].isna().any().any():
        raise ValueError('Race-window weather is missing; cannot summarise reliably.')
    return pd.DataFrame([{
        'season': int(race['season']), 'round': int(race['round']),
        'circuit': race['Circuit']['circuitName'], 'race_date': race['date'],
        'driver': result['Driver']['code'], 'grid_position': int(result['grid']),
        'finish_status': result['status'],
        'num_stops': stops[result['Driver']['driverId']],
        'air_temp_mean': window['temperature_2m'].mean(),
        'rain': bool((window['precipitation'] > 0).any()),
        'precipitation_mm_total': window['precipitation'].sum(),
        'precipitation_mm_max': window['precipitation'].max(),
    } for result in results])


# Identify a possible stoppage from long pit durations shared on the same lap.
def red_flag_evidence(records):
    long_stops = {}
    for stop in records:
        duration = stop.get('duration')
        if not duration:
            continue
        parts = str(duration).split(':')
        seconds = 0.0
        for part in parts:
            seconds = seconds * 60 + float(part)
        if seconds >= 600:
            long_stops.setdefault(stop['lap'], set()).add(stop['driverId'])
    return {lap: len(drivers) for lap, drivers in long_stops.items() if len(drivers) >= 3}


# Preserve the boolean helper while exposing the supporting lap/driver counts.
def possible_red_flag(records):
    return bool(red_flag_evidence(records))


# Same-lap duplicates do not count; keep the raw records and stop totals intact.
def consecutive_lap_drivers(records):
    laps = {}
    for stop in records:
        laps.setdefault(stop['driverId'], set()).add(int(stop['lap']))
    return {driver for driver, values in laps.items()
            if any(lap + 1 in values for lap in values)}


# Apply to one race at a time; zero stops have no defined class.
def flag_rows(table, red_flag=False, records=()):
    table = table.copy()
    status = table['finish_status'].fillna('')
    table['is_finisher'] = status.eq('Finished') | status.str.startswith('+') | status.eq('Lapped') | status.eq('Disqualified')
    table['possible_red_flag'] = bool(red_flag)
    table['rule_forced_stops'] = [
        (season, number) in RULE_FORCED_STOPS
        for season, number in zip(table.get('season', [None] * len(table)),
                                  table.get('round', [None] * len(table)))
    ]
    table['chaotic_race'] = bool(table['num_stops'].ge(5).sum() >= 3)
    table['suspect_stop_data'] = len(consecutive_lap_drivers(records)) >= 8
    table['exclude_reason'] = [
        'dns' if state == 'Did not start' else
        'dnf' if not finisher else
        'rule_forced_stops' if forced else
        'red_flag_race' if red_flag else
        'chaotic_race' if chaotic else
        'suspect_stop_data' if suspect else
        'zero_stop_finisher' if pd.notna(stops) and stops == 0 else
        'outlier_stops' if pd.notna(stops) and stops >= 5 else ''
        for state, finisher, stops, forced, chaotic, suspect in zip(
            status, table['is_finisher'], table['num_stops'],
            table['rule_forced_stops'], table['chaotic_race'], table['suspect_stop_data'])
    ]
    table['stops_class'] = table['num_stops'].map(
        lambda value: pd.NA if pd.isna(value) or value < 1 else
        '1' if value == 1 else '2' if value == 2 else '3+')
    return table


# Fetch the season calendar with pagination, keeping each race in API order.
def fetch_season_races(season):
    races = []
    while True:
        data = fetch_json(f'{BASE}{season}/', {'limit': 100, 'offset': len(races)})['MRData']
        page = data['RaceTable']['Races']
        races.extend(page)
        if len(races) >= int(data['total']):
            return races
        if not page:
            raise RuntimeError('Incomplete season calendar.')


# Build every race of one season and retain result rows when auxiliary data fails.
def build_season_table(season, failures=None):
    failures = failures if failures is not None else []
    tables = []
    try:
        calendar = fetch_season_races(season)
    except (requests.RequestException, ValueError, KeyError, RuntimeError) as error:
        failures.append({'season': season, 'round': None, 'stage': 'calendar', 'error': str(error)})
        return pd.DataFrame()
    for scheduled in calendar:
        number = int(scheduled['round'])
        print(f"Loading {season} round {number}: {scheduled['raceName']}", flush=True)
        try:
            race, results = fetch_race_results(season, number)
            if not results:
                raise ValueError('No race results returned.')
        except (requests.RequestException, ValueError, KeyError, RuntimeError) as error:
            failures.append({'season': season, 'round': number, 'stage': 'results', 'error': str(error)})
            continue
        counts, records = Counter(), []
        pit_failed = False
        try:
            counts, records = fetch_pit_stops(season, number, include_records=True)
        except (requests.RequestException, ValueError, KeyError, RuntimeError) as error:
            pit_failed = True
            failures.append({'season': season, 'round': number, 'stage': 'pit_stops', 'error': str(error)})
        try:
            table = build_race_table(race, results, counts, fetch_weather(race))
        except (requests.RequestException, ValueError, KeyError, RuntimeError) as error:
            failures.append({'season': season, 'round': number, 'stage': 'weather', 'error': str(error)})
            # Preserve all known drivers even when weather is unavailable.
            table = pd.DataFrame([{
                'season': season, 'round': number, 'circuit': race['Circuit']['circuitName'],
                'race_date': race['date'], 'driver': r['Driver']['code'],
                'grid_position': int(r['grid']), 'finish_status': r['status'],
                'num_stops': counts[r['Driver']['driverId']], 'air_temp_mean': float('nan'),
                'rain': pd.NA, 'precipitation_mm_total': float('nan'),
                'precipitation_mm_max': float('nan'),
            } for r in results])
        if pit_failed:
            table['num_stops'] = pd.NA
        table = flag_rows(table, possible_red_flag(records), records)
        table['race_name'] = race['raceName']
        table['red_flag_evidence'] = json.dumps(red_flag_evidence(records), sort_keys=True)
        # Missing weather stays visible and cannot silently qualify as usable.
        table['data_complete'] = table['air_temp_mean'].notna() & table['rain'].notna() & ~table['num_stops'].isna()
        tables.append(table)
    result = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
    print(f'Completed season {season}: {len(result)} rows; {sum(f["season"] == season for f in failures)} failures', flush=True)
    return result


# Build the requested eight seasons only when this function is explicitly called.
def build_all_seasons(failures=None):
    failures = failures if failures is not None else []
    tables = []
    for season in range(2018, 2026):
        print(f'\nStarting season {season}', flush=True)
        tables.append(build_season_table(season, failures))
        output = ROOT / 'data' / 'processed'
        output.mkdir(parents=True, exist_ok=True)
        pd.concat(tables, ignore_index=True).to_csv(output / 'stops_dataset.csv', index=False, encoding='utf-8')
        (output / 'failed_races.json').write_text(json.dumps(failures, indent=2), encoding='utf-8')
    return pd.concat(tables, ignore_index=True)


# Print and save race-level evidence, class counts, coverage, and failures.
def print_summary(table, failures):
    output = ROOT / 'reports'
    output.mkdir(parents=True, exist_ok=True)
    print(f'\nTotal rows: {len(table)}')
    if table.empty:
        print('No results available. Failures:', failures)
        return
    keys = ['season', 'round', 'race_name']
    usable = table['exclude_reason'].eq('')
    print('\nRows per season:\n', table.groupby('season').size().to_string())
    exclusions = pd.crosstab(table['season'], table['exclude_reason'].replace('', 'usable'))
    print('\nExclusions per season:\n', exclusions.to_string())
    exclusions.to_csv(output / 'exclusions_by_season.csv', encoding='utf-8')
    races = table.drop_duplicates(['season', 'round'])
    evidence = races.loc[races['possible_red_flag'], keys + ['red_flag_evidence']]
    print('\nPossible red flag races (lap: distinct drivers with >=600s pit duration):\n', evidence.to_string(index=False))
    evidence.to_csv(output / 'red_flag_races.csv', index=False, encoding='utf-8')
    suspect = races.loc[races['suspect_stop_data'], keys].copy()
    suspect_counts = []
    remaining_rows, remaining_races = 0, 0
    for (season, number), rows in table.groupby(['season', 'round']):
        _, records = fetch_pit_stops(int(season), int(number), include_records=True)
        drivers = consecutive_lap_drivers(records)
        if rows['suspect_stop_data'].iloc[0]:
            suspect_counts.append((season, number, len(drivers)))
        _, results = fetch_race_results(int(season), int(number))
        codes = {r['Driver']['code'] for r in results if r['Driver']['driverId'] in drivers}
        count = (rows['exclude_reason'].eq('') & rows['driver'].isin(codes)).sum()
        remaining_rows += int(count)
        remaining_races += int(count > 0)
    counts_by_race = {(season, number): count for season, number, count in suspect_counts}
    suspect['consecutive_lap_drivers'] = [counts_by_race[(s, r)] for s, r in zip(suspect['season'], suspect['round'])]
    print('\nAll suspect_stop_data races (including higher-priority exclusions):\n', suspect.to_string(index=False))
    suspect.to_csv(output / 'suspect_stop_data_races.csv', index=False, encoding='utf-8')
    print(f'\nUsable rows with consecutive-lap records: {remaining_rows}; races involved: {remaining_races}')
    counts = table.loc[usable].groupby(keys + ['stops_class']).size().unstack(fill_value=0)
    counts = counts.reindex(columns=['1', '2', '3+'], fill_value=0)
    counts = races[keys].merge(counts.reset_index(), on=keys, how='left').fillna(0)
    counts[['1', '2', '3+']] = counts[['1', '2', '3+']].astype(int)
    print('\nUsable classes per race:\n', counts.to_string(index=False))
    counts.to_csv(output / 'usable_classes_by_race.csv', index=False, encoding='utf-8')
    wet = races.assign(wet=races['precipitation_mm_total'].ge(1)).groupby('season')['wet'].sum()
    print('\nRACES with precipitation total >=1 mm per season:\n', wet.to_string())
    wet.to_csv(output / 'wet_races_by_season.csv', encoding='utf-8')
    print('\nOverall usable stops_class:\n', table.loc[usable, 'stops_class'].value_counts().sort_index().to_string())
    seasonal_classes = pd.crosstab(table.loc[usable, 'season'], table.loc[usable, 'stops_class']).reindex(columns=['1', '2', '3+'], fill_value=0)
    print('\nUsable stops_class per season:\n', seasonal_classes.to_string())
    seasonal_classes.to_csv(output / 'usable_classes_by_season.csv', encoding='utf-8')
    sizes = table.groupby(keys).size()
    print('\nRaces with fewer than 18 driver rows:\n', sizes[sizes < 18].to_string())
    print('\nIncomplete rows:', (~table['data_complete']).sum())
    print('\nFailed downloads:', len(failures))
    for failure in failures:
        print(failure)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--all-seasons', action='store_true', help='Run 2018–2025; use only after approval.')
    parser.add_argument('--cache-only', action='store_true', help='Refuse all downloads on cache misses.')
    args = parser.parse_args()
    CACHE_ONLY = args.cache_only
    failures = []
    table = build_all_seasons(failures) if args.all_seasons else build_season_table(2023, failures)
    output = ROOT / 'data' / 'processed'
    output.mkdir(parents=True, exist_ok=True)
    (output / 'failed_races.json').write_text(json.dumps(failures, indent=2), encoding='utf-8')
    if not table.empty:
        table.to_csv(output / 'stops_dataset.csv', index=False, encoding='utf-8')
    print_summary(table, failures)
