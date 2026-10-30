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


# Read cached JSON first; pace downloads and retry rate limits without hammering.
def fetch_json(url, params=None):
    cache = ROOT / 'cache'
    cache.mkdir(exist_ok=True)
    key = hashlib.sha256((url + json.dumps(params, sort_keys=True)).encode()).hexdigest()
    path = cache / (key + '.json')
    if path.exists():
        return json.loads(path.read_text(encoding='utf-8'))
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
        'start_date': race['date'], 'end_date': race['date'],
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
    } for result in results])


# Identify a possible stoppage from long pit durations shared on the same lap.
def possible_red_flag(records):
    long_stops = {}
    for stop in records:
        parts = str(stop.get('duration', '0')).split(':')
        seconds = 0.0
        for part in parts:
            seconds = seconds * 60 + float(part)
        if seconds >= 600:
            long_stops.setdefault(stop['lap'], set()).add(stop['driverId'])
    return any(len(drivers) >= 3 for drivers in long_stops.values())


# Add flags without deleting rows; multiple reasons are separated by semicolons.
def flag_rows(table, red_flag=False):
    table = table.copy()
    status = table['finish_status'].fillna('')
    table['is_finisher'] = status.eq('Finished') | status.str.startswith('+') | status.eq('Lapped')
    table['possible_red_flag'] = bool(red_flag)
    table['exclude_reason'] = [
        ';'.join(reason for condition, reason in [
            (not finisher, 'retired'), (stops == 0, 'no_pit_data'),
            (stops >= 5, 'outlier_stops'), (red_flag, 'possible_red_flag'),
        ] if condition)
        for finisher, stops in zip(table['is_finisher'], table['num_stops'])
    ]
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
                'rain': pd.NA,
            } for r in results])
        table = flag_rows(table, possible_red_flag(records))
        if pit_failed:
            table['num_stops'] = pd.NA
            table['exclude_reason'] = table['exclude_reason'].apply(
                lambda reason: reason if 'no_pit_data' in reason else ';'.join(filter(None, [reason, 'no_pit_data'])))
        # Missing weather stays visible and cannot silently qualify as usable.
        table['data_complete'] = table['air_temp_mean'].notna() & table['rain'].notna() & ~table['num_stops'].isna()
        tables.append(table)
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()


# Build the requested eight seasons only when this function is explicitly called.
def build_all_seasons(failures=None):
    failures = failures if failures is not None else []
    tables = [build_season_table(season, failures) for season in range(2018, 2026)]
    return pd.concat(tables, ignore_index=True)


# Print coverage, individual reason counts, and usable-row stop frequencies.
def print_summary(table, failures):
    print(f"\nRows: {len(table)}")
    print(table.head(10).to_string(index=False))
    if not table.empty:
        print('\nRows per season:\n', table.groupby('season').size().to_string())
        reasons = table['exclude_reason'].str.split(';').explode()
        print('\nExclusion reason counts (a row can count more than once):\n', reasons[reasons.ne('')].value_counts().to_string())
        usable = table['exclude_reason'].eq('') & table['data_complete']
        print(f"\nUsable rows: {usable.sum()}")
        print('Usable num_stops distribution:\n', table.loc[usable, 'num_stops'].value_counts().sort_index().to_string())
        print('Incomplete rows:', (~table['data_complete']).sum())
    print('\nFailed downloads:', len(failures))
    for failure in failures:
        print(failure)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--all-seasons', action='store_true', help='Run 2018–2025; use only after approval.')
    args = parser.parse_args()
    failures = []
    table = build_all_seasons(failures) if args.all_seasons else build_season_table(2023, failures)
    output = ROOT / 'data' / 'processed'
    output.mkdir(parents=True, exist_ok=True)
    (output / 'failed_races.json').write_text(json.dumps(failures, indent=2), encoding='utf-8')
    if not table.empty:
        table.to_csv(output / 'stops_dataset.csv', index=False)
    print_summary(table, failures)
