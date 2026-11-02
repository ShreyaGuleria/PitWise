"""Offline checks: run with python src/test_evaluate.py."""

import pandas as pd

from evaluate import (ROOT, CLASSES, SPLIT_YEARS, assign_splits, classification_metrics,
                      evaluate_split, load_features, predict_baselines)
from features import FEATURE_COLUMNS, NOT_FOR_MODEL


def test_splits_and_features():
    rows = assign_splits(load_features())
    assert rows.is_usable.all() and rows.stops_class.isin(CLASSES).all()
    assert pd.api.types.is_numeric_dtype(rows.circuit_pit_duration_prior)
    assert rows.circuit_pit_duration_prior.isna().any()
    assert rows.groupby(['season', 'round']).split.nunique().eq(1).all()
    for split, years in SPLIT_YEARS.items():
        assert set(rows.loc[rows.split.eq(split), 'season']) == set(years)
    assert set(FEATURE_COLUMNS).isdisjoint(NOT_FOR_MODEL)
    saved = pd.read_csv(ROOT / 'data' / 'processed' / 'splits.csv')
    pd.testing.assert_frame_equal(saved, rows[['season', 'round', 'driver', 'split']])


def test_history_and_test_lock():
    # Four historical races: B1 favors 1, last three pooled races favor 2.
    records = []
    for year, number, date, labels in [
        (2023, 1, '2023-01-01', ['1'] * 8),
        (2023, 2, '2023-02-01', ['2', '2']),
        (2023, 3, '2023-03-01', ['2', '2']),
        (2023, 4, '2023-04-01', ['2', '2']),
        (2024, 1, '2024-01-01', ['3+'] * 20),
        (2024, 2, '2024-02-01', ['1']),
        (2025, 1, '2025-01-01', ['3+'] * 30),
    ]:
        for driver, label in enumerate(labels):
            records.append((year, number, str(driver), pd.Timestamp(date), 'a', label))
    rows = assign_splits(pd.DataFrame(records, columns=[
        'season', 'round', 'driver', 'race_date', 'circuit_id', 'stops_class']))
    target = rows.loc[rows.season.eq(2024)]
    predictions = predict_baselines(rows, target)
    first = predictions.loc[target['round'].eq(1)]
    assert first.B0.eq('1').all() and first.B1.eq('1').all() and first.B2.eq('2').all()
    assert predictions.loc[target['round'].eq(2), ['B1', 'B2']].eq('3+').all().all()
    changed = rows.copy()
    changed.loc[changed.season.eq(2025), 'stops_class'] = '1'
    pd.testing.assert_frame_equal(predictions, predict_baselines(changed, target))
    unseen = target.iloc[:1].copy()
    unseen['circuit_id'] = 'unseen'
    assert predict_baselines(rows, unseen).eq('1').all().all()
    short_history = rows.loc[~(rows.season.eq(2023) & rows['round'].isin([1, 2]))]
    assert predict_baselines(short_history, target.iloc[:1]).eq('2').all().all()
    try:
        evaluate_split(rows, 'test')
    except ValueError as error:
        assert 'locked' in str(error)
    else:
        raise AssertionError('Test metrics must be locked by default')


def test_metrics():
    metrics = classification_metrics(['1', '2', '3+'], ['1', '1', '1'])
    assert metrics['accuracy'] == metrics['balanced_accuracy'] == 1 / 3
    assert metrics['macro_f1'] == 1 / 6
    assert metrics['per_class'].loc['2', 'precision'] == 0
    assert metrics['confusion_matrix'].values.tolist() == [[1, 0, 0], [1, 0, 0], [1, 0, 0]]


if __name__ == '__main__':
    for test in (test_splits_and_features, test_history_and_test_lock, test_metrics):
        test()
        print(f'{test.__name__}: passed')
