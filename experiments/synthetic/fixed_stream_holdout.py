"""Evaluate fixed synthetic streams under fresh delay schedules."""
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['BLIS_NUM_THREADS'] = '1'
import time
import importlib
import warnings
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import wilcoxon, rankdata
engine = importlib.import_module('cohort_protocol')
SCRIPT_NAME = 'fixed_stream_holdout'
OUTPUT_DIR = os.path.join('result', SCRIPT_NAME)
os.makedirs(OUTPUT_DIR, exist_ok=True)
CPU_JOBS = 20
DATASETS = ['SEA', 'Agrawal', 'Mixed', 'RandomTree', 'Sine']
STREAM_DIR = os.path.join('result', 'fixed_stream_inputs', 'streams')
LEARNERS = ['LR', 'DT']
FINAL_SEEDS = list(range(2056, 2076))
DELAY_SCENARIOS = engine.DELAY_SCENARIOS
STRATEGIES = engine.STRATEGIES
EXPECTED_STRATEGIES = ['STATIC', 'FULL_SOFT_Q30', 'FULL_COMPLETE_COHORT', 'WAIT_ALL_COMPLETE_COHORT']
assert STRATEGIES == EXPECTED_STRATEGIES
assert engine.WINDOW_SIZE == 100
assert abs(engine.EPSILON - 0.02) <= 1e-15
assert engine.UPDATE_TRAIN_SIZE == 100
assert engine.MIN_CLASS_COVERAGE == 30
assert engine.POST_UPDATE_CAL_SIZE == 100

def normalize_labels(y):
    y = np.asarray(y)
    labels = np.unique(y)
    mapping = {label: idx for idx, label in enumerate(labels)}
    encoded = np.asarray([mapping[value] for value in y], dtype=int)
    if len(np.unique(encoded)) != 2:
        raise RuntimeError(f'Expected binary labels, got {labels}')
    return encoded

def summarize(raw_df, group_cols):
    return raw_df.groupby(group_cols).agg(runs=('accuracy', 'count'), accuracy_mean=('accuracy', 'mean'), accuracy_std=('accuracy', 'std'), macro_f1_mean=('macro_f1', 'mean'), macro_f1_std=('macro_f1', 'std'), mcc_mean=('mcc', 'mean'), mcc_std=('mcc', 'std'), retrain_mean=('retrain_count', 'mean'), decision_accuracy=('decision_accuracy', 'mean'), decision_delay=('decision_delay', 'mean'), false_positive=('false_positive', 'sum'), false_negative=('false_negative', 'sum'), alarm_to_update=('mean_alarm_to_update', 'mean'), monitor_coverage=('monitor_coverage', 'mean'), skipped_single_class_updates=('skipped_single_class_updates', 'sum')).reset_index()

def holm_adjust(p_values):
    p_values = np.asarray(p_values, dtype=float)
    m = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(m, dtype=float)
    running = 0.0
    for rank_idx, original_idx in enumerate(order):
        candidate = min(1.0, (m - rank_idx) * p_values[original_idx])
        running = max(running, candidate)
        adjusted[original_idx] = running
    return adjusted

def paired_tests(df, key_cols, analysis_level):
    final_name = 'FULL_COMPLETE_COHORT'
    baselines = ['STATIC', 'FULL_SOFT_Q30', 'WAIT_ALL_COMPLETE_COHORT']
    metrics = ['accuracy', 'macro_f1', 'mcc']
    final_df = df[df['strategy'] == final_name]
    records = []
    for baseline in baselines:
        baseline_df = df[df['strategy'] == baseline]
        merged = final_df.merge(baseline_df, on=key_cols, suffixes=('_final', '_baseline'))
        for metric in metrics:
            final_values = merged[f'{metric}_final'].to_numpy(dtype=float)
            baseline_values = merged[f'{metric}_baseline'].to_numpy(dtype=float)
            differences = final_values - baseline_values
            wins = int(np.sum(differences > 1e-12))
            losses = int(np.sum(differences < -1e-12))
            ties = int(len(differences) - wins - losses)
            nonzero = differences[np.abs(differences) > 1e-12]
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
                w_plus = float(np.sum(ranks[nonzero > 0]))
                w_minus = float(np.sum(ranks[nonzero < 0]))
                denominator = w_plus + w_minus
                rank_biserial = (w_plus - w_minus) / denominator if denominator > 0 else 0.0
            records.append({'analysis_level': analysis_level, 'baseline': baseline, 'metric': metric, 'pairs': len(differences), 'final_mean': float(np.mean(final_values)), 'baseline_mean': float(np.mean(baseline_values)), 'mean_difference_final_minus_baseline': float(np.mean(differences)), 'median_difference': float(np.median(differences)), 'wins_final': wins, 'ties': ties, 'losses_final': losses, 'wilcoxon_statistic': statistic, 'p_raw': p_raw, 'rank_biserial_effect': rank_biserial})
    result = pd.DataFrame(records)
    result['p_holm'] = holm_adjust(result['p_raw'].to_numpy())
    result['significant_0_05_holm'] = result['p_holm'] < 0.05
    return result

def main():
    start = time.perf_counter()
    print('=' * 150)
    print('FINAL COMPLETE-COHORT SYNTHETIC HOLDOUT')
    print('=' * 150)
    print('Exact method engine:', 'cohort_protocol.py')
    print('Fresh seeds:', FINAL_SEEDS)
    print('Window:', engine.WINDOW_SIZE)
    print('Epsilon:', engine.EPSILON)
    print('Update size:', engine.UPDATE_TRAIN_SIZE)
    print('Soft q (precursor only):', engine.MIN_CLASS_COVERAGE)
    print('Calibration size:', engine.POST_UPDATE_CAL_SIZE)
    print('Strategies:', STRATEGIES)
    datasets = {}
    print()
    print('=' * 150)
    print('LOAD FROZEN SYNTHETIC STREAMS')
    print('=' * 150)
    for dataset in DATASETS:
        path = os.path.join(STREAM_DIR, f'{dataset}.npz')
        if not os.path.exists(path):
            raise FileNotFoundError(f'Missing {path}\nPlease confirm Step-11 synthetic streams still exist.')
        data = np.load(path, allow_pickle=True)
        X = np.asarray(data['X'], dtype=float)
        y = normalize_labels(data['y'])
        if len(X) != len(y):
            raise RuntimeError(f'{dataset}: X/y length mismatch')
        if np.isnan(X).any():
            raise RuntimeError(f'{dataset}: X contains NaN')
        if np.isinf(X).any():
            raise RuntimeError(f'{dataset}: X contains Inf')
        datasets[dataset] = (X, y)
        print(f'{dataset:<12} X={X.shape} class0={int(np.sum(y == 0))} class1={int(np.sum(y == 1))}')
    tasks = []
    for dataset in DATASETS:
        X, y = datasets[dataset]
        for learner in LEARNERS:
            for scenario, config in DELAY_SCENARIOS.items():
                for seed in FINAL_SEEDS:
                    tasks.append((dataset, X, y, learner, scenario, config[0], config[1], seed))
    expected_tasks = len(DATASETS) * len(LEARNERS) * len(DELAY_SCENARIOS) * len(FINAL_SEEDS)
    expected_rows = expected_tasks * len(STRATEGIES)
    print()
    print('=' * 150)
    print('START FINAL HOLDOUT')
    print('=' * 150)
    print('Tasks:', len(tasks))
    print('Strategies:', len(STRATEGIES))
    print('Expected result rows:', expected_rows)
    if len(tasks) != expected_tasks:
        raise RuntimeError('Unexpected task count')
    results = Parallel(n_jobs=CPU_JOBS, backend='loky', verbose=10, batch_size=1, pre_dispatch='2*n_jobs')((delayed(engine.run_one)(*task) for task in tasks))
    all_runs = []
    all_decisions = []
    all_updates = []
    all_skipped = []
    for runs, decisions, updates, skipped in results:
        all_runs.extend(runs)
        all_decisions.extend(decisions)
        all_updates.extend(updates)
        all_skipped.extend(skipped)
    raw_df = pd.DataFrame(all_runs)
    decision_df = pd.DataFrame(all_decisions)
    update_df = pd.DataFrame(all_updates)
    skipped_df = pd.DataFrame(all_skipped)
    print()
    print('=' * 150)
    print('COMPLETENESS CHECK')
    print('=' * 150)
    print('Expected:', expected_rows)
    print('Actual:', len(raw_df))
    if len(raw_df) != expected_rows:
        raise RuntimeError('Final holdout results are incomplete')
    overall_df = summarize(raw_df, ['strategy'])
    scenario_df = summarize(raw_df, ['scenario', 'strategy'])
    dataset_df = summarize(raw_df, ['dataset', 'strategy'])
    dataset_learner_df = summarize(raw_df, ['dataset', 'learner', 'strategy'])
    cell_summary_df = summarize(raw_df, ['dataset', 'learner', 'scenario', 'strategy'])
    task_tests_df = paired_tests(raw_df, key_cols=['dataset', 'learner', 'scenario', 'delay_seed'], analysis_level='task_level_1000_delay_realizations')
    cell_means_df = raw_df.groupby(['dataset', 'learner', 'scenario', 'strategy'])[['accuracy', 'macro_f1', 'mcc']].mean().reset_index()
    cell_tests_df = paired_tests(cell_means_df, key_cols=['dataset', 'learner', 'scenario'], analysis_level='cell_level_50_dataset_learner_scenario_cells')
    safety_records = []
    for strategy in ['FULL_SOFT_Q30', 'FULL_COMPLETE_COHORT', 'WAIT_ALL_COMPLETE_COHORT']:
        subset = raw_df[raw_df['strategy'] == strategy]
        valid = subset[subset['decision_accuracy'].notna()]
        safety_records.append({'strategy': strategy, 'runs': len(subset), 'nonperfect_runs': int((valid['decision_accuracy'] < 1.0 - 1e-12).sum()), 'false_positive': int(subset['false_positive'].sum()), 'false_negative': int(subset['false_negative'].sum()), 'decision_accuracy': float(subset['decision_accuracy'].mean()), 'decision_delay': float(subset['decision_delay'].mean()), 'alarm_to_update': float(subset['mean_alarm_to_update'].mean()), 'retrain_mean': float(subset['retrain_count'].mean()), 'skipped_single_class_updates': int(subset['skipped_single_class_updates'].sum())})
    safety_df = pd.DataFrame(safety_records)
    full = overall_df[overall_df['strategy'] == 'FULL_COMPLETE_COHORT'].iloc[0]
    comparison_records = []
    for baseline_name in ['STATIC', 'FULL_SOFT_Q30', 'WAIT_ALL_COMPLETE_COHORT']:
        baseline = overall_df[overall_df['strategy'] == baseline_name].iloc[0]
        comparison_records.append({'baseline': baseline_name, 'full_accuracy': full['accuracy_mean'], 'baseline_accuracy': baseline['accuracy_mean'], 'accuracy_difference': full['accuracy_mean'] - baseline['accuracy_mean'], 'full_macro_f1': full['macro_f1_mean'], 'baseline_macro_f1': baseline['macro_f1_mean'], 'macro_f1_difference': full['macro_f1_mean'] - baseline['macro_f1_mean'], 'full_mcc': full['mcc_mean'], 'baseline_mcc': baseline['mcc_mean'], 'mcc_difference': full['mcc_mean'] - baseline['mcc_mean'], 'full_retrain': full['retrain_mean'], 'baseline_retrain': baseline['retrain_mean'], 'full_decision_delay': full['decision_delay'], 'baseline_decision_delay': baseline['decision_delay'], 'full_alarm_to_update': full['alarm_to_update'], 'baseline_alarm_to_update': baseline['alarm_to_update']})
    comparison_df = pd.DataFrame(comparison_records)
    paper_table_df = overall_df[['strategy', 'runs', 'accuracy_mean', 'accuracy_std', 'macro_f1_mean', 'mcc_mean', 'retrain_mean', 'decision_accuracy', 'decision_delay', 'false_positive', 'false_negative', 'alarm_to_update', 'monitor_coverage', 'skipped_single_class_updates']].copy()
    print()
    print('=' * 150)
    print('FINAL PAPER TABLE')
    print('=' * 150)
    print(paper_table_df.to_string(index=False))
    print()
    print('=' * 150)
    print('FINAL METHOD vs COMPARATORS')
    print('=' * 150)
    print(comparison_df.to_string(index=False))
    print()
    print('=' * 150)
    print('SAFETY')
    print('=' * 150)
    print(safety_df.to_string(index=False))
    print()
    print('=' * 150)
    print('PRIMARY STATISTICS: 50-CELL PAIRED TESTS')
    print('=' * 150)
    print(cell_tests_df.to_string(index=False))
    print()
    print('=' * 150)
    print('SUPPORTIVE: 1000 TASK-LEVEL PAIRED TESTS')
    print('=' * 150)
    print(task_tests_df.to_string(index=False))
    paths = {'raw': os.path.join(OUTPUT_DIR, 'raw.csv'), 'decision': os.path.join(OUTPUT_DIR, 'decision_raw.csv'), 'update': os.path.join(OUTPUT_DIR, 'update_raw.csv'), 'skipped': os.path.join(OUTPUT_DIR, 'skipped_updates.csv'), 'overall': os.path.join(OUTPUT_DIR, 'overall_summary.csv'), 'scenario': os.path.join(OUTPUT_DIR, 'scenario_summary.csv'), 'dataset': os.path.join(OUTPUT_DIR, 'dataset_summary.csv'), 'dataset_learner': os.path.join(OUTPUT_DIR, 'dataset_learner_summary.csv'), 'cell_summary': os.path.join(OUTPUT_DIR, 'dataset_learner_scenario_summary.csv'), 'cell_means': os.path.join(OUTPUT_DIR, 'cell_means.csv'), 'task_tests': os.path.join(OUTPUT_DIR, 'task_level_paired_tests.csv'), 'cell_tests': os.path.join(OUTPUT_DIR, 'cell_level_paired_tests.csv'), 'safety': os.path.join(OUTPUT_DIR, 'safety.csv'), 'comparison': os.path.join(OUTPUT_DIR, 'final_complete_vs_comparators.csv'), 'paper': os.path.join(OUTPUT_DIR, 'final_paper_table.csv')}
    raw_df.to_csv(paths['raw'], index=False, encoding='utf-8-sig')
    decision_df.to_csv(paths['decision'], index=False, encoding='utf-8-sig')
    update_df.to_csv(paths['update'], index=False, encoding='utf-8-sig')
    skipped_df.to_csv(paths['skipped'], index=False, encoding='utf-8-sig')
    overall_df.to_csv(paths['overall'], index=False, encoding='utf-8-sig')
    scenario_df.to_csv(paths['scenario'], index=False, encoding='utf-8-sig')
    dataset_df.to_csv(paths['dataset'], index=False, encoding='utf-8-sig')
    dataset_learner_df.to_csv(paths['dataset_learner'], index=False, encoding='utf-8-sig')
    cell_summary_df.to_csv(paths['cell_summary'], index=False, encoding='utf-8-sig')
    cell_means_df.to_csv(paths['cell_means'], index=False, encoding='utf-8-sig')
    task_tests_df.to_csv(paths['task_tests'], index=False, encoding='utf-8-sig')
    cell_tests_df.to_csv(paths['cell_tests'], index=False, encoding='utf-8-sig')
    safety_df.to_csv(paths['safety'], index=False, encoding='utf-8-sig')
    comparison_df.to_csv(paths['comparison'], index=False, encoding='utf-8-sig')
    paper_table_df.to_csv(paths['paper'], index=False, encoding='utf-8-sig')
    elapsed = time.perf_counter() - start
    final_safety = safety_df[safety_df['strategy'] == 'FULL_COMPLETE_COHORT'].iloc[0]
    safety_pass = int(final_safety['nonperfect_runs']) == 0 and int(final_safety['false_positive']) == 0 and (int(final_safety['false_negative']) == 0)
    print()
    print('=' * 150)
    print('OUTPUT FILES')
    print('=' * 150)
    for path in paths.values():
        print(path)
    print()
    print(f'Runtime: {elapsed:.2f} s = {elapsed / 60:.2f} min')
    print('FULL_COMPLETE_COHORT safety:', 'PASS ' if safety_pass else 'CHECK REQUIRED ')
    print('Fresh holdout seeds: 2056-2075')
    print('No method parameter was retuned.')
    print('For formal inference, use the 50-cell paired table as primary; the 1000 task-level table is supportive because delay seeds share the same underlying streams.')
    print('=' * 150)
if __name__ == '__main__':
    main()
