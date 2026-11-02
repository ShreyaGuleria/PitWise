"""Fixed random-forest first model; expanding-window validation, never 2025."""

from io import StringIO

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

from evaluate import ROOT, CLASSES, classification_metrics, load_features, predict_baselines
from features import FEATURE_COLUMNS, STRICT_FEATURE_COLUMNS, NOT_FOR_MODEL

VALIDATION_YEARS = (2021, 2022, 2023, 2024)
METRICS = ('accuracy', 'balanced_accuracy', 'macro_f1')


def season_fold(rows, year):
    if year not in VALIDATION_YEARS:
        raise ValueError('Only 2021-2024 validation is allowed; 2025 is held out')
    train = rows.loc[rows.season.between(2018, year - 1)]
    validation = rows.loc[rows.season.eq(year)]
    if train.empty or validation.empty or train.race_date.max() >= validation.race_date.min():
        raise ValueError('Expected nonempty, chronologically separated season folds')
    return train, validation


def model_pipeline(columns):
    if set(columns) & set(NOT_FOR_MODEL):
        raise ValueError('Excluded fields cannot be model inputs')
    categorical = ['circuit_id', 'constructor_group']
    numeric = [column for column in columns if column not in categorical]
    preprocess = ColumnTransformer([
        ('categorical', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical),
        ('numeric', SimpleImputer(strategy='median', add_indicator=True,
                                 keep_empty_features=True), numeric),
    ])
    return make_pipeline(preprocess, RandomForestClassifier(
        n_estimators=200, max_depth=8, min_samples_leaf=5,
        class_weight='balanced', random_state=42, n_jobs=1))


def main():
    # Drop the held-out season before any fold, fitting, prediction or metrics.
    rows = load_features()
    rows = rows.loc[rows.season.lt(2025)].copy()
    records, details = [], StringIO()
    for year in VALIDATION_YEARS:
        train, validation = season_fold(rows, year)
        available = pd.concat([train, validation]).copy()
        available['split'] = available.season.map(lambda s: 'validation' if s == year else 'train')
        predictions = predict_baselines(available, validation)
        for variant, columns in [('RF_retrospective', FEATURE_COLUMNS),
                                 ('RF_strict', STRICT_FEATURE_COLUMNS)]:
            pipeline = model_pipeline(columns)
            # Numeric class IDs avoid mixed numeric-looking/string labels in
            # sklearn's class-weight handling; decode before reporting metrics.
            pipeline.fit(train[columns], train.stops_class.map(dict(zip(CLASSES, range(3)))))
            predictions[variant] = [CLASSES[int(value)] for value in pipeline.predict(validation[columns])]
        for name in predictions:
            metrics = classification_metrics(validation.stops_class, predictions[name])
            records.append({'validation_season': year, 'method': name,
                            'train_rows': len(train), 'validation_rows': len(validation),
                            'validation_races': validation.groupby(['season', 'round']).ngroups,
                            **{key: metrics[key] for key in METRICS}})
            print(f'\n{year} {name}', file=details)
            print(metrics['per_class'].to_string(float_format=lambda x: f'{x:.4f}'), file=details)
            print('Confusion matrix: rows actual, columns predicted (1, 2, 3+)', file=details)
            print(metrics['confusion_matrix'].to_string(), file=details)
    folds = pd.DataFrame(records)
    summary = folds.groupby('method')[list(METRICS)].agg(['mean', 'std', 'min', 'max'])
    report = '\n'.join([
        'Expanding-window validation: 2021, 2022, 2023, 2024; train = 2018 through previous season.',
        '2025 excluded before fitting, predicting or evaluating; no test metrics computed.',
        'All driver rows of a race stay in one fold. No hyperparameter search performed.',
        'Fixed RF: 200 trees, depth 8, minimum leaf 5, balanced weights, seed 42.',
        'Imputation and category encoding are fitted separately on each training fold only.',
        'Strict model removes ALL current-race weather (including precipitation) and actual laps.',
        'Replacements: latest usable circuit visit exactly last year (actual laps),',
        'and equal-race mean historical circuit air temperature in the same calendar month.',
        'These are historical proxies, not planned laps or a weather forecast. No history stays NaN.',
        'History may update from earlier completed validation races, equally for models and B1/B2.',
        'Stop means pool usable driver counts. Recent windows contain three distinct usable races.',
        'Mean/std are unweighted across FOUR seasons; std is sample std, not a confidence interval.',
        'Driver rows share race conditions; small differences are noise and do not establish superiority.',
        '\nPer-season metrics:', folds.to_string(index=False, float_format=lambda x: f'{x:.4f}'),
        '\nMean, sample standard deviation and range:', summary.to_string(float_format=lambda x: f'{x:.4f}'),
        '\nInterpretation: compare macro F1, balanced accuracy and accuracy together.',
        'Small mean differences relative to season spread do not establish a model win.',
        'The retrospective variant uses unavailable current-race information; prefer the strict',
        'variant for a deployable pre-race starting point, while retaining all proxy/selection caveats.',
        '\nPer-class metrics and confusion matrices:', details.getvalue(),
    ])
    folds.to_csv(ROOT / 'reports' / 'expanding_validation.csv', index=False)
    summary.to_csv(ROOT / 'reports' / 'expanding_validation_summary.csv')
    (ROOT / 'reports' / 'expanding_validation.txt').write_text(report + '\n', encoding='utf-8')
    print(report)


if __name__ == '__main__':
    main()
