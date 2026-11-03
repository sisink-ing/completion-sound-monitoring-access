"""Generate and evaluate independent synthetic stream realizations."""
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['BLIS_NUM_THREADS'] = '1'
import hashlib
import importlib
import time
import warnings
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import wilcoxon, rankdata
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.matshow = lambda *args, **kwargs: None
import gen_data
engine = importlib.import_module('cohort_protocol')
SCRIPT_NAME = 'unseen_stream_pipeline'
OUTPUT_DIR = os.path.join('result', SCRIPT_NAME)
STREAM_OUTPUT_DIR = os.path.join(OUTPUT_DIR, 'streams')
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(STREAM_OUTPUT_DIR, exist_ok=True)
DATASETS = ['SEA', 'Agrawal', 'Mixed', 'RandomTree', 'Sine']
PATTERN = '__v__r__v__'
BATCH_SIZE = 200
EXPECTED_STREAM_LENGTH = len(PATTERN) * BATCH_SIZE
STREAM_SEEDS = list(range(6001, 6011))
DELAY_SEEDS = list(range(7001, 7006))
LEARNERS = ['LR', 'DT']
DELAY_SCENARIOS = engine.DELAY_SCENARIOS
STRATEGIES = engine.STRATEGIES
EXPECTED_STRATEGIES = ['STATIC', 'FULL_SOFT_Q30', 'FULL_COMPLETE_COHORT', 'WAIT_ALL_COMPLETE_COHORT']
assert STRATEGIES == EXPECTED_STRATEGIES
assert engine.WINDOW_SIZE == 100
assert abs(engine.EPSILON - 0.02) <= 1e-15
assert engine.UPDATE_TRAIN_SIZE == 100
assert engine.MIN_CLASS_COVERAGE == 30
assert engine.POST_UPDATE_CAL_SIZE == 100
INITIAL_TRAIN_SIZE = 200
GENERATION_JOBS = 2
EVALUATION_JOBS = 20
SMOKE_TEST = os.environ.get('STREAM_SMOKE', '0') == '1'

def normalize_binary_labels(y):
    y = np.asarray(y).reshape(-1)
    labels = np.unique(y)
    if len(labels) != 2:
        raise RuntimeError(f'Expected binary stream, got labels={labels}')
    mapping = {label: index for index, label in enumerate(labels)}
    return np.asarray([mapping[value] for value in y], dtype=int)

def stream_hash(X, y):
    hasher = hashlib.sha256()
    X_contiguous = np.ascontiguousarray(X)
    y_contiguous = np.ascontiguousarray(y)
    hasher.update(X_contiguous.tobytes())
    hasher.update(y_contiguous.tobytes())
    return hasher.hexdigest()

def reset_official_generator_cache():
    defaults = gen_data.gen_window_matrix.__defaults__
    if defaults is not None and len(defaults) >= 1 and isinstance(defaults[0], dict):
        defaults[0].clear()

def generate_one_stream(dataset, seed):
    start = time.perf_counter()
    try:
        np.random.seed(int(seed))
        reset_official_generator_cache()
        import matplotlib.pyplot as local_plt
        local_plt.matshow = lambda *args, **kwargs: None
        if dataset not in gen_data.datasources:
            raise KeyError(f'Dataset {dataset} not found in gen_data.datasources')
        source = gen_data.datasources[dataset]()
        source.take(100)
        source.generate_drift()
        source.take(100)
        stream_segments = []
        drift_annotations = []
        for segment_type in PATTERN.replace('_', '-'):
            virtual_drift, real_drift = source.generate_drift(segment_type)
            drift_code = int(virtual_drift) + 2 * int(real_drift)
            drift_annotations.append(drift_code * np.ones(BATCH_SIZE, dtype=int))
            X_segment, y_segment = source.take(BATCH_SIZE)
            stream_segments.append((X_segment, y_segment))
        X = np.concatenate([item[0] for item in stream_segments], axis=0)
        y = np.concatenate([item[1] for item in stream_segments], axis=0)
        dd = np.concatenate(drift_annotations, axis=0)
        X = np.asarray(X, dtype=float)
        y = normalize_binary_labels(y)
        dd = np.asarray(dd, dtype=int)
        if X.shape[0] != EXPECTED_STREAM_LENGTH:
            raise RuntimeError(f'{dataset}/seed={seed}: unexpected stream length {X.shape[0]} != {EXPECTED_STREAM_LENGTH}')
        if len(y) != EXPECTED_STREAM_LENGTH:
            raise RuntimeError(f'{dataset}/seed={seed}: X/y length mismatch')
        if np.isnan(X).any():
            raise RuntimeError(f'{dataset}/seed={seed}: NaN in X')
        if np.isinf(X).any():
            raise RuntimeError(f'{dataset}/seed={seed}: Inf in X')
        train_y = y[:INITIAL_TRAIN_SIZE]
        train_classes = np.unique(train_y)
        initial_train_valid = len(train_classes) == 2
        digest = stream_hash(X, y)
        elapsed = time.perf_counter() - start
        return {'generation_ok': True, 'dataset': dataset, 'stream_seed': int(seed), 'stream_hash': digest, 'X': X, 'y': y, 'dd': dd, 'n_samples': int(len(y)), 'n_features': int(X.shape[1]), 'class0': int(np.sum(y == 0)), 'class1': int(np.sum(y == 1)), 'train_class0': int(np.sum(train_y == 0)), 'train_class1': int(np.sum(train_y == 1)), 'initial_train_valid': bool(initial_train_valid), 'dd_none': int(np.sum(dd == 0)), 'dd_virtual': int(np.sum(dd == 1)), 'dd_real': int(np.sum(dd == 2)), 'dd_combined': int(np.sum(dd == 3)), 'generation_seconds': float(elapsed), 'error': ''}
    except Exception as exc:
        elapsed = time.perf_counter() - start
        return {'generation_ok': False, 'dataset': dataset, 'stream_seed': int(seed), 'stream_hash': '', 'X': None, 'y': None, 'dd': None, 'n_samples': 0, 'n_features': 0, 'class0': 0, 'class1': 0, 'train_class0': 0, 'train_class1': 0, 'initial_train_valid': False, 'dd_none': 0, 'dd_virtual': 0, 'dd_real': 0, 'dd_combined': 0, 'generation_seconds': float(elapsed), 'error': repr(exc)}

def save_generated_stream(record):
    if not record['generation_ok']:
        return
    path = os.path.join(STREAM_OUTPUT_DIR, f"{record['dataset']}_streamseed_{record['stream_seed']}.npz")
    np.savez_compressed(path, X=record['X'], y=record['y'], dd=record['dd'], dataset=record['dataset'], stream_seed=record['stream_seed'], stream_hash=record['stream_hash'])

def run_evaluation_task(dataset, stream_seed, stream_digest, X, y, learner, scenario, lambda0, lambda1, delay_seed):
    stream_id = f'{dataset}_s{stream_seed}'
    runs, decisions, updates, skipped = engine.run_one(stream_id, X, y, learner, scenario, lambda0, lambda1, delay_seed)
    collections = [runs, decisions, updates, skipped]
    for collection in collections:
        for row in collection:
            row['stream_id'] = stream_id
            row['dataset_family'] = dataset
            row['stream_seed'] = int(stream_seed)
            row['stream_hash'] = stream_digest
            row['dataset'] = dataset
    return (runs, decisions, updates, skipped)

def summarize(raw_df, groups):
    return raw_df.groupby(groups).agg(runs=('accuracy', 'count'), accuracy_mean=('accuracy', 'mean'), accuracy_std=('accuracy', 'std'), macro_f1_mean=('macro_f1', 'mean'), macro_f1_std=('macro_f1', 'std'), mcc_mean=('mcc', 'mean'), mcc_std=('mcc', 'std'), retrain_mean=('retrain_count', 'mean'), decision_accuracy=('decision_accuracy', 'mean'), decision_delay=('decision_delay', 'mean'), false_positive=('false_positive', 'sum'), false_negative=('false_negative', 'sum'), alarm_to_update=('mean_alarm_to_update', 'mean'), monitor_coverage=('monitor_coverage', 'mean'), skipped_single_class_updates=('skipped_single_class_updates', 'sum')).reset_index()

def holm_adjust(p_values):
    p_values = np.asarray(p_values, dtype=float)
    m = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(m, dtype=float)
    running = 0.0
    for rank_index, original_index in enumerate(order):
        candidate = min(1.0, (m - rank_index) * p_values[original_index])
        running = max(running, candidate)
        adjusted[original_index] = running
    return adjusted

def paired_tests(cell_df):
    final_strategy = 'FULL_COMPLETE_COHORT'
    baselines = ['STATIC', 'FULL_SOFT_Q30', 'WAIT_ALL_COMPLETE_COHORT']
    metrics = ['accuracy', 'macro_f1', 'mcc']
    key_columns = ['dataset', 'stream_seed', 'learner', 'scenario']
    final_df = cell_df[cell_df['strategy'] == final_strategy]
    records = []
    for baseline in baselines:
        baseline_df = cell_df[cell_df['strategy'] == baseline]
        merged = final_df.merge(baseline_df, on=key_columns, suffixes=('_final', '_baseline'))
        for metric in metrics:
            final_values = merged[f'{metric}_final'].to_numpy(dtype=float)
            baseline_values = merged[f'{metric}_baseline'].to_numpy(dtype=float)
            difference = final_values - baseline_values
            wins = int(np.sum(difference > 1e-12))
            losses = int(np.sum(difference < -1e-12))
            ties = int(len(difference) - wins - losses)
            nonzero = difference[np.abs(difference) > 1e-12]
            if len(nonzero) == 0:
                statistic = 0.0
                p_raw = 1.0
                rank_biserial = 0.0
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    test = wilcoxon(nonzero, zero_method='wilcox', alternative='two-sided', method='auto')
                statistic = float(test.statistic)
                p_raw = float(test.pvalue)
                ranks = rankdata(np.abs(nonzero))
                w_positive = float(np.sum(ranks[nonzero > 0]))
                w_negative = float(np.sum(ranks[nonzero < 0]))
                denominator = w_positive + w_negative
                rank_biserial = (w_positive - w_negative) / denominator if denominator > 0 else 0.0
            records.append({'baseline': baseline, 'metric': metric, 'paired_stream_cells': len(difference), 'final_mean': float(np.mean(final_values)), 'baseline_mean': float(np.mean(baseline_values)), 'mean_difference': float(np.mean(difference)), 'median_difference': float(np.median(difference)), 'wins_final': wins, 'ties': ties, 'losses_final': losses, 'wilcoxon_statistic': statistic, 'p_raw': p_raw, 'rank_biserial_effect': rank_biserial})
    result = pd.DataFrame(records)
    result['p_holm'] = holm_adjust(result['p_raw'].to_numpy())
    result['significant_0_05_holm'] = result['p_holm'] < 0.05
    return result

def main():
    global_start = time.perf_counter()
    print('=' * 155)
    print('UNSEEN SYNTHETIC STREAM-REALIZATION HOLDOUT')
    print('=' * 155)
    print('Smoke test:', SMOKE_TEST)
    print('Pattern:', PATTERN)
    print('Batch size:', BATCH_SIZE)
    print('Expected stream length:', EXPECTED_STREAM_LENGTH)
    print('Window:', engine.WINDOW_SIZE)
    print('Epsilon:', engine.EPSILON)
    print('Update size:', engine.UPDATE_TRAIN_SIZE)
    print('Calibration size:', engine.POST_UPDATE_CAL_SIZE)
    if SMOKE_TEST:
        selected_datasets = ['SEA']
        selected_stream_seeds = [6001]
        selected_learners = ['LR']
        selected_delay_seeds = [7001]
        if 'UNIFORM_100_100' in DELAY_SCENARIOS:
            selected_scenarios = {'UNIFORM_100_100': DELAY_SCENARIOS['UNIFORM_100_100']}
        else:
            first_key = next(iter(DELAY_SCENARIOS))
            selected_scenarios = {first_key: DELAY_SCENARIOS[first_key]}
        generation_jobs = 1
        evaluation_jobs = 1
    else:
        selected_datasets = DATASETS
        selected_stream_seeds = STREAM_SEEDS
        selected_learners = LEARNERS
        selected_delay_seeds = DELAY_SEEDS
        selected_scenarios = DELAY_SCENARIOS
        generation_jobs = GENERATION_JOBS
        evaluation_jobs = EVALUATION_JOBS
    generation_tasks = [(dataset, stream_seed) for dataset in selected_datasets for stream_seed in selected_stream_seeds]
    print()
    print('=' * 155)
    print('PHASE 1 — GENERATE COMPLETELY NEW STREAM REALIZATIONS')
    print('=' * 155)
    print('Number of new streams:', len(generation_tasks))
    generated = Parallel(n_jobs=generation_jobs, backend='loky', verbose=10, batch_size=1, pre_dispatch=generation_jobs)((delayed(generate_one_stream)(dataset, seed) for dataset, seed in generation_tasks))
    for record in generated:
        save_generated_stream(record)
    feasibility_columns = ['generation_ok', 'dataset', 'stream_seed', 'stream_hash', 'n_samples', 'n_features', 'class0', 'class1', 'train_class0', 'train_class1', 'initial_train_valid', 'dd_none', 'dd_virtual', 'dd_real', 'dd_combined', 'generation_seconds', 'error']
    feasibility_df = pd.DataFrame([{column: record[column] for column in feasibility_columns} for record in generated])
    feasibility_path = os.path.join(OUTPUT_DIR, 'stream_feasibility.csv')
    feasibility_df.to_csv(feasibility_path, index=False, encoding='utf-8-sig')
    print()
    print(feasibility_df[['dataset', 'stream_seed', 'generation_ok', 'initial_train_valid', 'n_features', 'class0', 'class1', 'train_class0', 'train_class1', 'generation_seconds']].to_string(index=False))
    successful_df = feasibility_df[feasibility_df['generation_ok'] == True].copy()
    duplicated = successful_df[successful_df.duplicated(subset=['dataset', 'stream_hash'], keep=False)]
    if not duplicated.empty:
        print()
        print('DUPLICATE STREAMS DETECTED:')
        print(duplicated.to_string(index=False))
        raise RuntimeError('Different stream seeds produced duplicate stream realizations. Stop: stream-generation seed control must be checked.')
    valid_records = [record for record in generated if record['generation_ok'] and record['initial_train_valid']]
    excluded_records = [record for record in generated if not (record['generation_ok'] and record['initial_train_valid'])]
    print()
    print('Generated streams:', len(generated))
    print('Valid streams:', len(valid_records))
    print('Excluded streams:', len(excluded_records))
    if len(valid_records) == 0:
        raise RuntimeError('No valid streams available.')
    evaluation_tasks = []
    for record in valid_records:
        dataset = record['dataset']
        stream_seed = record['stream_seed']
        digest = record['stream_hash']
        X = record['X']
        y = record['y']
        for learner in selected_learners:
            for scenario, delay_config in selected_scenarios.items():
                lambda0 = delay_config[0]
                lambda1 = delay_config[1]
                for delay_seed in selected_delay_seeds:
                    evaluation_tasks.append((dataset, stream_seed, digest, X, y, learner, scenario, lambda0, lambda1, delay_seed))
    expected_raw_rows = len(evaluation_tasks) * len(STRATEGIES)
    print()
    print('=' * 155)
    print('PHASE 2 — FINAL METHOD EVALUATION')
    print('=' * 155)
    print('Independent stream/delay tasks:', len(evaluation_tasks))
    print('Strategies:', len(STRATEGIES))
    print('Expected raw rows:', expected_raw_rows)
    evaluation_results = Parallel(n_jobs=evaluation_jobs, backend='loky', verbose=10, batch_size=1, pre_dispatch='2*n_jobs')((delayed(run_evaluation_task)(*task) for task in evaluation_tasks))
    all_runs = []
    all_decisions = []
    all_updates = []
    all_skipped = []
    for runs, decisions, updates, skipped in evaluation_results:
        all_runs.extend(runs)
        all_decisions.extend(decisions)
        all_updates.extend(updates)
        all_skipped.extend(skipped)
    raw_df = pd.DataFrame(all_runs)
    decision_df = pd.DataFrame(all_decisions)
    update_df = pd.DataFrame(all_updates)
    skipped_df = pd.DataFrame(all_skipped)
    print()
    print('=' * 155)
    print('COMPLETENESS CHECK')
    print('=' * 155)
    print('Expected rows:', expected_raw_rows)
    print('Actual rows:', len(raw_df))
    if len(raw_df) != expected_raw_rows:
        raise RuntimeError('Evaluation result table is incomplete.')
    overall_df = summarize(raw_df, ['strategy'])
    dataset_df = summarize(raw_df, ['dataset', 'strategy'])
    scenario_df = summarize(raw_df, ['scenario', 'strategy'])
    dataset_learner_df = summarize(raw_df, ['dataset', 'learner', 'strategy'])
    cell_group_columns = ['dataset', 'stream_seed', 'stream_hash', 'learner', 'scenario', 'strategy']
    cell_df = raw_df.groupby(cell_group_columns).agg(delay_seed_runs=('accuracy', 'count'), accuracy=('accuracy', 'mean'), macro_f1=('macro_f1', 'mean'), mcc=('mcc', 'mean'), retrain_count=('retrain_count', 'mean'), decision_accuracy=('decision_accuracy', 'mean'), decision_delay=('decision_delay', 'mean'), false_positive=('false_positive', 'sum'), false_negative=('false_negative', 'sum'), alarm_to_update=('mean_alarm_to_update', 'mean'), monitor_coverage=('monitor_coverage', 'mean')).reset_index()
    paired_df = paired_tests(cell_df)
    safety_records = []
    for strategy in ['FULL_SOFT_Q30', 'FULL_COMPLETE_COHORT', 'WAIT_ALL_COMPLETE_COHORT']:
        subset = raw_df[raw_df['strategy'] == strategy]
        valid_decision = subset[subset['decision_accuracy'].notna()]
        safety_records.append({'strategy': strategy, 'runs': len(subset), 'valid_decision_runs': len(valid_decision), 'no_decision_runs': int(len(subset) - len(valid_decision)), 'nonperfect_runs': int((valid_decision['decision_accuracy'] < 1.0 - 1e-12).sum()), 'false_positive': int(subset['false_positive'].sum()), 'false_negative': int(subset['false_negative'].sum()), 'decision_accuracy': float(subset['decision_accuracy'].mean()), 'decision_delay': float(subset['decision_delay'].mean()), 'retrain_mean': float(subset['retrain_count'].mean())})
    safety_df = pd.DataFrame(safety_records)
    complete = overall_df[overall_df['strategy'] == 'FULL_COMPLETE_COHORT'].iloc[0]
    comparison_records = []
    for baseline_name in ['STATIC', 'FULL_SOFT_Q30', 'WAIT_ALL_COMPLETE_COHORT']:
        baseline = overall_df[overall_df['strategy'] == baseline_name].iloc[0]
        complete_delay = complete['decision_delay']
        baseline_delay = baseline['decision_delay']
        if pd.notna(complete_delay) and pd.notna(baseline_delay) and (baseline_delay > 0):
            delay_saved = baseline_delay - complete_delay
            delay_reduction_percent = delay_saved / baseline_delay * 100.0
        else:
            delay_saved = np.nan
            delay_reduction_percent = np.nan
        comparison_records.append({'baseline': baseline_name, 'complete_accuracy': complete['accuracy_mean'], 'baseline_accuracy': baseline['accuracy_mean'], 'accuracy_difference': complete['accuracy_mean'] - baseline['accuracy_mean'], 'complete_macro_f1': complete['macro_f1_mean'], 'baseline_macro_f1': baseline['macro_f1_mean'], 'macro_f1_difference': complete['macro_f1_mean'] - baseline['macro_f1_mean'], 'complete_mcc': complete['mcc_mean'], 'baseline_mcc': baseline['mcc_mean'], 'mcc_difference': complete['mcc_mean'] - baseline['mcc_mean'], 'complete_confirmation_delay': complete_delay, 'baseline_confirmation_delay': baseline_delay, 'delay_saved': delay_saved, 'delay_reduction_percent': delay_reduction_percent, 'complete_retrain': complete['retrain_mean'], 'baseline_retrain': baseline['retrain_mean']})
    comparison_df = pd.DataFrame(comparison_records)
    paths = {'feasibility': feasibility_path, 'raw': os.path.join(OUTPUT_DIR, 'raw.csv'), 'decision': os.path.join(OUTPUT_DIR, 'decision_raw.csv'), 'update': os.path.join(OUTPUT_DIR, 'update_raw.csv'), 'skipped': os.path.join(OUTPUT_DIR, 'skipped_updates.csv'), 'overall': os.path.join(OUTPUT_DIR, 'overall_summary.csv'), 'dataset': os.path.join(OUTPUT_DIR, 'dataset_summary.csv'), 'scenario': os.path.join(OUTPUT_DIR, 'scenario_summary.csv'), 'dataset_learner': os.path.join(OUTPUT_DIR, 'dataset_learner_summary.csv'), 'cell': os.path.join(OUTPUT_DIR, 'stream_realization_cells.csv'), 'paired': os.path.join(OUTPUT_DIR, 'stream_cell_paired_tests.csv'), 'safety': os.path.join(OUTPUT_DIR, 'safety.csv'), 'comparison': os.path.join(OUTPUT_DIR, 'key_comparisons.csv')}
    raw_df.to_csv(paths['raw'], index=False, encoding='utf-8-sig')
    decision_df.to_csv(paths['decision'], index=False, encoding='utf-8-sig')
    update_df.to_csv(paths['update'], index=False, encoding='utf-8-sig')
    skipped_df.to_csv(paths['skipped'], index=False, encoding='utf-8-sig')
    overall_df.to_csv(paths['overall'], index=False, encoding='utf-8-sig')
    dataset_df.to_csv(paths['dataset'], index=False, encoding='utf-8-sig')
    scenario_df.to_csv(paths['scenario'], index=False, encoding='utf-8-sig')
    dataset_learner_df.to_csv(paths['dataset_learner'], index=False, encoding='utf-8-sig')
    cell_df.to_csv(paths['cell'], index=False, encoding='utf-8-sig')
    paired_df.to_csv(paths['paired'], index=False, encoding='utf-8-sig')
    safety_df.to_csv(paths['safety'], index=False, encoding='utf-8-sig')
    comparison_df.to_csv(paths['comparison'], index=False, encoding='utf-8-sig')
    print()
    print('=' * 155)
    print('OVERALL RESULTS')
    print('=' * 155)
    print(overall_df.to_string(index=False))
    print()
    print('=' * 155)
    print('SAFETY')
    print('=' * 155)
    print(safety_df.to_string(index=False))
    print()
    print('=' * 155)
    print('KEY COMPARISONS')
    print('=' * 155)
    print(comparison_df.to_string(index=False))
    print()
    print('=' * 155)
    print('STREAM-REALIZATION CELL PAIRED TESTS')
    print('=' * 155)
    print(paired_df.to_string(index=False))
    final_safety = safety_df[safety_df['strategy'] == 'FULL_COMPLETE_COHORT'].iloc[0]
    safety_pass = int(final_safety['nonperfect_runs']) == 0 and int(final_safety['false_positive']) == 0 and (int(final_safety['false_negative']) == 0)
    elapsed = time.perf_counter() - global_start
    print()
    print('=' * 155)
    print('FINAL STATUS')
    print('=' * 155)
    print('New stream-generation seeds:', selected_stream_seeds)
    print('New delay seeds:', selected_delay_seeds)
    print('Valid new stream realizations:', len(valid_records))
    print('Excluded pre-specified realizations:', len(excluded_records))
    print('Unique stream hashes:', successful_df['stream_hash'].nunique())
    print('FULL_COMPLETE_COHORT safety:', 'PASS ' if safety_pass else 'CHECK REQUIRED ')
    print(f'Total runtime: {elapsed:.2f} s = {elapsed / 60:.2f} min')
    print()
    print('Fixed protocol: W, epsilon, q, update size and calibration size.')
    print()
    print('Primary analysis unit:')
    print('dataset × stream realization × learner × delay scenario')
    print('Delay seeds are averaged inside each cell before paired tests.')
    print()
    print('Evaluation scope:')
    print('Unseen synthetic stream realizations evaluated under multiple delay seeds.')
    print()
    print('Output directory:')
    print(OUTPUT_DIR)
    print('=' * 155)
if __name__ == '__main__':
    main()
