"""Strict date cutoffs and race-level windows, including shuffled input."""

import pandas as pd

from history import history_features
from features import FEATURE_COLUMNS, STRICT_FEATURE_COLUMNS, NOT_FOR_MODEL
from train import season_fold, model_pipeline
from evaluate import load_features


def test_history():
    table = pd.DataFrame([
        (2020, 1, 'a', '2020-03-01', True, 1, 50, 10),
        (2020, 1, 'a', '2020-03-01', True, 3, 50, 10),
        (2020, 2, 'a', '2020-04-01', False, 99, 99, 99),
        (2021, 1, 'a', '2021-03-01', True, 4, 60, 20),
        (2021, 2, 'a', '2021-03-01', True, 4, 60, 20),
        (2021, 3, 'b', '2021-04-01', True, 3, 70, 25),
        (2021, 4, 'a', '2021-05-01', True, 1, 60, 30),
        (2025, 1, 'a', '2025-03-01', True, 99, 99, 99),
    ], columns=['season', 'round', 'circuit_id', 'race_date', 'is_usable',
                'num_stops', 'scheduled_laps', 'air_temp_mean'])
    result = history_features(table)
    assert result.iloc[0].circuit_races_prior == 0
    assert pd.isna(result.iloc[0].circuit_avg_stops_prior)
    for index in (3, 4):
        assert result.iloc[index].circuit_avg_stops_prior == 2
        assert result.iloc[index].circuit_races_prior == 1
        assert result.iloc[index].circuit_laps_last_year == 50
        assert result.iloc[index].circuit_month_air_temp_prior == 10
    assert result.iloc[6].circuit_races_prior == 3  # Distinct races, not driver rows.
    assert result.iloc[6].recent_avg_stops_prior == 11 / 3  # Last 3 races globally.
    assert result.iloc[6].circuit_avg_stops_last3 == 3  # Four drivers, three races.
    changed = table.copy()
    changed.loc[changed.index >= 3, ['num_stops', 'scheduled_laps', 'air_temp_mean']] = 1000
    updated = history_features(changed)
    pd.testing.assert_frame_equal(result.iloc[:5], updated.iloc[:5])
    pd.testing.assert_frame_equal(history_features(table.sample(frac=1, random_state=9)).sort_index(), result)


def test_folds_and_inputs():
    assert 'grid_position' not in FEATURE_COLUMNS
    assert 'grid_position_clean' in FEATURE_COLUMNS
    assert not set(STRICT_FEATURE_COLUMNS) & set(NOT_FOR_MODEL)
    assert not {'scheduled_laps', 'air_temp_mean', 'precipitation_mm_total'} & set(STRICT_FEATURE_COLUMNS)
    rows = pd.DataFrame({'season': [2018, 2020, 2021, 2025],
                         'race_date': pd.to_datetime(['2018-01-01','2020-01-01','2021-01-01','2025-01-01'])})
    train, validation = season_fold(rows, 2021)
    assert train.season.tolist() == [2018, 2020] and validation.season.tolist() == [2021]
    try:
        season_fold(rows, 2025)
    except ValueError:
        pass
    else:
        raise AssertionError('2025 must remain locked')
    actual = load_features()
    for year in (2021, 2022, 2023, 2024):
        training_fold, validation_fold = season_fold(actual, year)
        train_races = set(zip(training_fold.season, training_fold['round']))
        valid_races = set(zip(validation_fold.season, validation_fold['round']))
        assert train_races.isdisjoint(valid_races)
        assert training_fold.season.max() == year - 1
        assert validation_fold.season.eq(year).all()
    # Actual pipeline check: unseen categories and cold-start NaNs are supported.
    columns = ['circuit_id', 'constructor_group', 'grid_position_clean', 'history']
    training = pd.DataFrame({'circuit_id': ['a'] * 6, 'constructor_group': ['t'] * 6,
                             'grid_position_clean': [1, 2, 3, 4, 5, 6], 'history': [float('nan')] * 6})
    pipeline = model_pipeline(columns)
    pipeline.fit(training, [0, 1, 2, 0, 1, 2])
    assert pipeline.named_steps['columntransformer'].named_transformers_['numeric'].statistics_[0] == 3.5
    unseen = training.iloc[:1].copy()
    unseen['circuit_id'] = 'new'
    assert len(pipeline.predict(unseen)) == 1


if __name__ == '__main__':
    for test in (test_history, test_folds_and_inputs):
        test()
        print(f'{test.__name__}: passed')
