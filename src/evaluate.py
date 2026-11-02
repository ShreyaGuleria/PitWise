"""Offline season splits and historical baselines; main evaluates 2024 only."""

from io import StringIO
from pathlib import Path

import pandas as pd

from features import FEATURE_COLUMNS, ALL_FEATURE_COLUMNS, NOT_FOR_MODEL

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ('1', '2', '3+')
SPLIT_YEARS = {'train': range(2018, 2024), 'validation': (2024,), 'test': (2025,)}


def load_features(path=ROOT / 'data' / 'processed' / 'features.csv'):
    """Keep usable, labelled rows only, preserving identifiers and feature data."""
    rows = pd.read_csv(path, keep_default_na=False, dtype={'stops_class': str},
                       na_values={column: [''] for column in ALL_FEATURE_COLUMNS})
    rows = rows.loc[rows.is_usable.eq(True) & rows.stops_class.isin(CLASSES)].copy()
    rows['race_date'] = pd.to_datetime(rows.race_date, errors='raise')
    return rows.reset_index(drop=True)


def assign_splits(rows):
    rows = rows.copy()
    rows['split'] = rows.season.map({year: name for name, years in SPLIT_YEARS.items()
                                    for year in years})
    if rows.split.isna().any():
        raise ValueError('Usable rows outside the specified 2018-2025 seasons')
    if rows.duplicated(['season', 'round', 'driver']).any():
        raise ValueError('Duplicate driver/race identifiers')
    if rows.groupby(['season', 'round']).split.nunique().gt(1).any():
        raise ValueError('A race appears in multiple splits')
    if set(FEATURE_COLUMNS) & set(NOT_FOR_MODEL):
        raise ValueError('Model features overlap excluded columns')
    return rows


def majority(labels):
    """Ties favor 1, then 2, then 3+, independently of input order."""
    counts = labels.value_counts().reindex(CLASSES, fill_value=0)
    if counts.sum() == 0:
        raise ValueError('Cannot determine a majority from empty history')
    return counts.idxmax()


def predict_baselines(rows, targets):
    """Use strictly earlier usable races, including earlier validation races.

    B0 is fixed from training. Targets must follow all training race dates.
    B2 pools driver labels from the last three distinct usable circuit races;
    when fewer than three exist, it uses B1 (then B0 for no history).
    """
    train = rows.loc[rows.split.eq('train')]
    b0 = majority(train.stops_class)
    if targets.empty or targets.race_date.min() <= train.race_date.max():
        raise ValueError('Targets must be nonempty and strictly after training')
    predictions = pd.DataFrame(index=targets.index, columns=['B0', 'B1', 'B2'])
    predictions['B0'] = b0
    for (date, circuit), group in targets.groupby(['race_date', 'circuit_id']):
        history = rows.loc[rows.race_date.lt(date) & rows.circuit_id.eq(circuit)]
        b1 = majority(history.stops_class) if not history.empty else b0
        races = history[['season', 'round', 'race_date']].drop_duplicates()
        races = races.sort_values(['race_date', 'season', 'round']).tail(3)
        recent = history.merge(races, on=['season', 'round', 'race_date'])
        b2 = majority(recent.stops_class) if len(races) == 3 else b1
        predictions.loc[group.index, ['B1', 'B2']] = [b1, b2]
    return predictions


def classification_metrics(actual, predicted):
    """Three-class metrics with zero precision/recall for zero denominators."""
    matrix = pd.DataFrame(0, index=CLASSES, columns=CLASSES)
    if len(actual) != len(predicted) or not len(actual):
        raise ValueError('Expected equal, nonempty label sequences')
    for truth, guess in zip(actual, predicted):
        if truth not in CLASSES or guess not in CLASSES:
            raise ValueError('Unexpected class label')
        matrix.loc[truth, guess] += 1
    per_class = pd.DataFrame(index=CLASSES, columns=['precision', 'recall', 'f1'], dtype=float)
    for label in CLASSES:
        tp = matrix.loc[label, label]
        support, guesses = matrix.loc[label].sum(), matrix[label].sum()
        precision = tp / guesses if guesses else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class.loc[label] = [precision, recall, f1]
    return {'accuracy': sum(matrix.loc[c, c] for c in CLASSES) / len(actual),
            'balanced_accuracy': per_class.recall.mean(),
            'macro_f1': per_class.f1.mean(), 'per_class': per_class,
            'confusion_matrix': matrix}


def evaluate_split(rows, split='validation', *, allow_test=False):
    """Test evaluation is available only through an explicit future opt-in."""
    if split == 'test' and not allow_test:
        raise ValueError('2025 test metrics are locked; explicit authorization is required')
    if split not in ('validation', 'test'):
        raise ValueError('Only validation or test evaluation is supported')
    targets = rows.loc[rows.split.eq(split)]
    predictions = predict_baselines(rows, targets)
    return {name: classification_metrics(targets.stops_class, predictions[name])
            for name in predictions}


def validation_report(rows, results):
    report = StringIO()
    def line(value=''):
        print(value, file=report)

    line('Pit-stop baselines: 2024 validation only')
    line('Usable rows with stops_class in 1, 2, 3+; season split, never row split.')
    line('Train: 2018-2023; validation: 2024; test: 2025 (counts only, no metrics).')
    line('\nClass counts:')
    counts = pd.crosstab(rows.split, rows.stops_class).reindex(
        index=SPLIT_YEARS, columns=CLASSES, fill_value=0)
    line(counts.to_string())
    train = rows.loc[rows.split.eq('train')]
    validation = rows.loc[rows.split.eq('validation')]
    unseen = ~validation.circuit_id.isin(train.circuit_id)
    line(f'\nValidation rows at circuits absent from usable training: {unseen.sum()}')
    line(f'B0: always predict training majority ({majority(train.stops_class)}).')
    line('B1: majority of driver labels at the circuit in all strictly earlier usable races; fallback B0.')
    line('B2: majority of driver labels pooled across the last 3 earlier usable circuit races;')
    line('    with fewer than 3 races, fallback B1, then B0.')
    line('History updates after each race; earlier validation labels may inform later races.')
    line('Same-date and future labels never enter history. Ties favor 1, then 2, then 3+.')
    line('Metrics use all three classes; zero denominators produce 0.')
    for name, metrics in results.items():
        line(f'\n{name}')
        for key in ('accuracy', 'balanced_accuracy', 'macro_f1'):
            line(f'{key}: {metrics[key]:.4f}')
        line(metrics['per_class'].to_string(float_format=lambda value: f'{value:.4f}'))
        line('Confusion matrix (rows = actual, columns = predicted; order 1, 2, 3+):')
        line(metrics['confusion_matrix'].to_string())
    line('\ngrid_position diagnostics (filtered usable rows, all splits):')
    grid = rows.grid_position
    line(f'min={grid.min()}, max={grid.max()}, zeros={grid.eq(0).sum()}')
    line('By split:')
    for split in SPLIT_YEARS:
        grid = rows.loc[rows.split.eq(split), 'grid_position']
        line(f'{split}: min={grid.min()}, max={grid.max()}, zeros={grid.eq(0).sum()}')
    if rows.grid_position.eq(0).any():
        line('Grid zero denotes a pit-lane start. Use grid_position_clean (0 -> 20) and pit_lane_start.')
        line('Raw grid_position is preserved for reference and excluded from model inputs.')
    line('\nInterpretation:')
    for metric in ('accuracy', 'balanced_accuracy', 'macro_f1'):
        best = max(result[metric] for result in results.values())
        winners = ', '.join(name for name, result in results.items() if result[metric] == best)
        line(f'Best {metric}: {winners} ({best:.4f}).')
    line('B0 measures how well a constant guess exploits class imbalance; B1 adds circuit history;')
    line('B2 emphasizes recent circuit strategy. Balanced accuracy averages class recalls; macro F1')
    line('gives each class equal weight, so common-class accuracy alone is not enough.')
    for name, result in results.items():
        line(f'{name} correctly identifies {result["per_class"].loc["3+", "recall"]:.1%} of 3+ stop cases.')
    line('Prefer the macro-F1/balanced-accuracy leader for equal attention to all classes;')
    line('also retain the accuracy leader as the benchmark for overall correct predictions.')
    line('The model should beat these validation benchmarks, especially macro F1 and balanced accuracy,')
    line('while recovering the rarer classes. A zero class recall means that class is never identified correctly.')
    line('These results cover outcome-filtered usable drivers, not every starter. Future model inputs')
    line('scheduled_laps and weather retain the retrospective-proxy limitations documented in features.py.')
    line('2025 test performance has not been computed. Keep it held out until explicitly authorized.')
    return report.getvalue()


def main():
    rows = assign_splits(load_features())
    rows[['season', 'round', 'driver', 'split']].to_csv(
        ROOT / 'data' / 'processed' / 'splits.csv', index=False)
    results = evaluate_split(rows)  # Validation only. Do not opt in to test here.
    report = validation_report(rows, results)
    (ROOT / 'reports').mkdir(exist_ok=True)
    (ROOT / 'reports' / 'baselines_validation.txt').write_text(report, encoding='utf-8')
    print(report, end='')


if __name__ == '__main__':
    main()
