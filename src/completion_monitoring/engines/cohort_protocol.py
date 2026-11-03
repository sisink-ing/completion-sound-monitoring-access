"""Completion-sound monitoring and complete event-time cohort adaptation."""
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['BLIS_NUM_THREADS'] = '1'
import time
import importlib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
base = importlib.import_module('chronological_protocol')
SCRIPT_NAME = 'cohort_protocol'
OUTPUT_DIR = os.path.join('result', SCRIPT_NAME)
os.makedirs(OUTPUT_DIR, exist_ok=True)
CPU_JOBS = 20
STRATEGIES = ['STATIC', 'FULL_SOFT_Q30', 'FULL_COMPLETE_COHORT', 'WAIT_ALL_COMPLETE_COHORT']
ADAPTIVE_STRATEGIES = ['FULL_SOFT_Q30', 'FULL_COMPLETE_COHORT', 'WAIT_ALL_COMPLETE_COHORT']
COMPLETE_STRATEGIES = ['FULL_COMPLETE_COHORT', 'WAIT_ALL_COMPLETE_COHORT']
RAW_DATASETS = base.RAW_DATASETS
LEARNERS = base.LEARNERS
DELAY_SCENARIOS = base.DELAY_SCENARIOS
DELAY_SEEDS = base.DELAY_SEEDS
MAX_SAMPLES = base.MAX_SAMPLES
MONITOR_START = base.MONITOR_START
WINDOW_SIZE = base.WINDOW_SIZE
EPSILON = base.EPSILON
FLOAT_TOL = base.FLOAT_TOL
UPDATE_TRAIN_SIZE = base.UPDATE_TRAIN_SIZE
MIN_CLASS_COVERAGE = base.MIN_CLASS_COVERAGE
POST_UPDATE_CAL_SIZE = base.POST_UPDATE_CAL_SIZE
assert MAX_SAMPLES == 5000
assert WINDOW_SIZE == 100
assert abs(EPSILON - 0.02) <= 1e-15
assert UPDATE_TRAIN_SIZE == 100
assert MIN_CLASS_COVERAGE == 30
assert POST_UPDATE_CAL_SIZE == 100
assert DELAY_SEEDS == list(range(5001, 5021))

def create_state(X, y, learner):
    model, reference, train_acc, cal_acc = base.create_initial_model(X, y, learner)
    return {'model': model, 'model_version': 0, 'reference_loss': reference, 'predictions': {}, 'feedback': {}, 'phase': 'MONITORING', 'open_block': [], 'closed_blocks': [], 'next_block_id': 0, 'alarm_anchor': None, 'alarm_time': None, 'adaptation_cohort_ids': None, 'cohort_missing_at_alarm': None, 'calibration_ids': [], 'retrain_count': 0, 'decision_records': [], 'update_records': [], 'skipped_update_records': [], 'monitor_samples': 0, 'waiting_samples': 0, 'calibration_samples': 0, 'train_acc': train_acc, 'cal_acc': cal_acc, 'initial_reference': reference}

def get_complete_cohort_training_ids(state):
    cohort = state['adaptation_cohort_ids']
    if cohort is None:
        return None
    all_returned = all((sample_id in state['feedback'] for sample_id in cohort))
    if not all_returned:
        return None
    return np.asarray(cohort, dtype=int)

def reset_after_skipped_update(state):
    state['phase'] = 'MONITORING'
    state['open_block'] = []
    state['alarm_anchor'] = None
    state['alarm_time'] = None
    state['adaptation_cohort_ids'] = None
    state['cohort_missing_at_alarm'] = None

def perform_update(strategy, state, train_ids, current_time, X, y, learner):
    train_ids = np.asarray(train_ids, dtype=int)
    unique_classes = np.unique(y[train_ids])
    cohort_ids = state['adaptation_cohort_ids']
    if cohort_ids is not None:
        cohort_array = np.asarray(cohort_ids, dtype=int)
        cohort_class1_rate = float(np.mean(y[cohort_array] == 1))
    else:
        cohort_class1_rate = np.nan
    if len(unique_classes) < 2:
        state['skipped_update_records'].append({'strategy': strategy, 'alarm_time': state['alarm_time'], 'skip_time': current_time, 'reason': 'single_class_complete_cohort', 'cohort_start': int(train_ids[0]), 'cohort_end': int(train_ids[-1]), 'cohort_class1_rate': cohort_class1_rate})
        reset_after_skipped_update(state)
        return False
    new_model = base.create_model(learner)
    new_model.fit(X[train_ids], y[train_ids])
    old_version = int(state['model_version'])
    old_reference = float(state['reference_loss'])
    count1 = int(np.sum(y[train_ids] == 1))
    state['update_records'].append({'update_time': int(current_time), 'alarm_time': int(state['alarm_time']), 'alarm_to_update': int(current_time - state['alarm_time']), 'training_mode': 'SOFT_Q30' if strategy == 'FULL_SOFT_Q30' else 'COMPLETE_COHORT', 'old_version': old_version, 'new_version': old_version + 1, 'old_reference': old_reference, 'train_size': int(len(train_ids)), 'train_class1_rate': float(count1 / len(train_ids)), 'train_min_event': int(train_ids.min()), 'train_max_event': int(train_ids.max()), 'cohort_start': int(cohort_ids[0]) if cohort_ids is not None else np.nan, 'cohort_end': int(cohort_ids[-1]) if cohort_ids is not None else np.nan, 'cohort_class1_rate': cohort_class1_rate, 'cohort_missing_at_alarm': state['cohort_missing_at_alarm']})
    state['model'] = new_model
    state['model_version'] += 1
    state['retrain_count'] += 1
    state['reference_loss'] = np.nan
    state['phase'] = 'CALIBRATING'
    state['calibration_ids'] = []
    state['open_block'] = []
    state['alarm_anchor'] = None
    state['alarm_time'] = None
    state['adaptation_cohort_ids'] = None
    state['cohort_missing_at_alarm'] = None
    return True

def run_one(dataset, X, y, learner, scenario, lambda0, lambda1, delay_seed):
    start_clock = time.perf_counter()
    arrival_map, arrival_groups = base.generate_delay(y, delay_seed, lambda0, lambda1)
    last_event = len(y) - 1
    simulation_end = max(last_event, max(arrival_map.values()))
    states = {strategy: create_state(X, y, learner) for strategy in STRATEGIES}
    for current_time in range(MONITOR_START, simulation_end + 1):
        for strategy in ADAPTIVE_STRATEGIES:
            state = states[strategy]
            if state['phase'] != 'WAITING_READY':
                continue
            if current_time > last_event:
                continue
            if strategy == 'FULL_SOFT_Q30':
                train_ids = base.get_selective_training_ids(current_time, state['alarm_anchor'], arrival_map, y)
            else:
                train_ids = get_complete_cohort_training_ids(state)
            if train_ids is None:
                continue
            perform_update(strategy, state, train_ids, current_time, X, y, learner)
        if current_time <= last_event:
            sample_id = int(current_time)
            for strategy in STRATEGIES:
                state = states[strategy]
                prediction = int(state['model'].predict(X[sample_id].reshape(1, -1))[0])
                state['predictions'][sample_id] = prediction
                if strategy == 'STATIC':
                    continue
                if state['phase'] == 'MONITORING':
                    state['monitor_samples'] += 1
                    state['open_block'].append(sample_id)
                    if len(state['open_block']) == WINDOW_SIZE:
                        ids = list(state['open_block'])
                        block = {'block_id': int(state['next_block_id']), 'model_version': int(state['model_version']), 'reference_loss': float(state['reference_loss']), 'sample_ids': ids, 'start': int(ids[0]), 'end': int(ids[-1]), 'close_time': int(ids[-1]), 'status': 'PENDING'}
                        state['closed_blocks'].append(block)
                        state['next_block_id'] += 1
                        state['open_block'] = []
                elif state['phase'] == 'CALIBRATING':
                    state['calibration_samples'] += 1
                    if len(state['calibration_ids']) < POST_UPDATE_CAL_SIZE:
                        state['calibration_ids'].append(sample_id)
                elif state['phase'] == 'WAITING_READY':
                    state['waiting_samples'] += 1
        arriving = arrival_groups.get(current_time, [])
        for strategy in STRATEGIES:
            state = states[strategy]
            for sample_id in arriving:
                if sample_id not in state['predictions']:
                    raise RuntimeError(f'{dataset}/{learner}/{scenario}/{strategy}: missing prediction {sample_id}')
                state['feedback'][sample_id] = int(state['predictions'][sample_id] != y[sample_id])
        for strategy in ADAPTIVE_STRATEGIES:
            state = states[strategy]
            if state['phase'] != 'CALIBRATING':
                continue
            if len(state['calibration_ids']) != POST_UPDATE_CAL_SIZE:
                continue
            cal_ids = state['calibration_ids']
            all_returned = all((sample_id in state['feedback'] for sample_id in cal_ids))
            if not all_returned:
                continue
            losses = [state['feedback'][sample_id] for sample_id in cal_ids]
            state['reference_loss'] = float(np.mean(losses))
            state['phase'] = 'MONITORING'
            state['calibration_ids'] = []
            state['open_block'] = []
        for strategy in ADAPTIVE_STRATEGIES:
            state = states[strategy]
            if state['phase'] != 'MONITORING':
                continue
            for block in state['closed_blocks']:
                if block['status'] != 'PENDING':
                    continue
                if strategy in ['FULL_SOFT_Q30', 'FULL_COMPLETE_COHORT']:
                    result = base.evaluate_bounds(state, block)
                else:
                    result = base.evaluate_wait_all(state, block)
                if result is None:
                    continue
                block['status'] = 'DONE'
                oracle, final_loss, final_delta = base.calculate_oracle(y, state, block)
                decision = result['decision']
                correct = int(decision == oracle)
                cohort_ids = list(block['sample_ids'])
                missing_now = int(sum((sample_id not in state['feedback'] for sample_id in cohort_ids)))
                cohort_class1_rate = float(np.mean(y[np.asarray(cohort_ids, dtype=int)] == 1))
                state['decision_records'].append({'block_id': int(block['block_id']), 'model_version': int(block['model_version']), 'decision': decision, 'oracle': oracle, 'correct': correct, 'decision_time': int(current_time), 'decision_delay': int(current_time - block['close_time']), 'labels_at_decision': int(result['m']), 'unknown_at_decision': int(result['u']), 'reference_loss': float(block['reference_loss']), 'final_loss': float(final_loss), 'final_delta': float(final_delta), 'cohort_missing_at_decision': missing_now, 'cohort_class1_rate': cohort_class1_rate})
                if decision == 'DEGRADATION':
                    state['phase'] = 'WAITING_READY'
                    state['alarm_anchor'] = int(block['start'])
                    state['alarm_time'] = int(current_time)
                    state['adaptation_cohort_ids'] = cohort_ids
                    state['cohort_missing_at_alarm'] = missing_now
                    state['open_block'] = []
                    for other in state['closed_blocks']:
                        if other['status'] == 'PENDING':
                            other['status'] = 'STALE'
                    break
    monitor_ids = np.arange(MONITOR_START, len(y))
    y_true = y[MONITOR_START:]
    run_records = []
    decision_records = []
    update_records = []
    skipped_records = []
    for strategy in STRATEGIES:
        state = states[strategy]
        predictions = np.asarray([state['predictions'][int(sample_id)] for sample_id in monitor_ids], dtype=int)
        accuracy = float(accuracy_score(y_true, predictions))
        macro_f1 = float(f1_score(y_true, predictions, average='macro'))
        mcc = float(matthews_corrcoef(y_true, predictions))
        decisions = state['decision_records']
        if decisions:
            decision_accuracy = float(np.mean([d['correct'] for d in decisions]))
            decision_delay = float(np.mean([d['decision_delay'] for d in decisions]))
            labels_at_decision = float(np.mean([d['labels_at_decision'] for d in decisions]))
            fp = int(sum((d['decision'] == 'DEGRADATION' and d['oracle'] == 'TOLERABLE' for d in decisions)))
            fn = int(sum((d['decision'] == 'TOLERABLE' and d['oracle'] == 'DEGRADATION' for d in decisions)))
            degradation_count = int(sum((d['decision'] == 'DEGRADATION' for d in decisions)))
            mean_missing_at_decision = float(np.mean([d['cohort_missing_at_decision'] for d in decisions]))
        else:
            decision_accuracy = np.nan
            decision_delay = np.nan
            labels_at_decision = np.nan
            fp = 0
            fn = 0
            degradation_count = 0
            mean_missing_at_decision = np.nan
        updates = state['update_records']
        if updates:
            mean_alarm_to_update = float(np.mean([update['alarm_to_update'] for update in updates]))
            mean_train_class1_rate = float(np.mean([update['train_class1_rate'] for update in updates]))
            mean_cohort_class1_rate = float(np.mean([update['cohort_class1_rate'] for update in updates]))
            mean_cohort_missing_at_alarm = float(np.mean([update['cohort_missing_at_alarm'] for update in updates]))
        else:
            mean_alarm_to_update = np.nan
            mean_train_class1_rate = np.nan
            mean_cohort_class1_rate = np.nan
            mean_cohort_missing_at_alarm = np.nan
        if strategy == 'STATIC':
            monitor_coverage = np.nan
            waiting_coverage = np.nan
            calibration_coverage = np.nan
        else:
            denominator = len(monitor_ids)
            monitor_coverage = state['monitor_samples'] / denominator
            waiting_coverage = state['waiting_samples'] / denominator
            calibration_coverage = state['calibration_samples'] / denominator
        run_records.append({'dataset': dataset, 'learner': learner, 'scenario': scenario, 'lambda_class0': lambda0, 'lambda_class1': lambda1, 'delay_seed': delay_seed, 'strategy': strategy, 'accuracy': accuracy, 'macro_f1': macro_f1, 'mcc': mcc, 'retrain_count': int(state['retrain_count']), 'degradation_count': degradation_count, 'decision_count': len(decisions), 'decision_accuracy': decision_accuracy, 'decision_delay': decision_delay, 'mean_labels_at_decision': labels_at_decision, 'false_positive': fp, 'false_negative': fn, 'mean_alarm_to_update': mean_alarm_to_update, 'mean_train_class1_rate': mean_train_class1_rate, 'mean_cohort_class1_rate': mean_cohort_class1_rate, 'mean_cohort_missing_at_alarm': mean_cohort_missing_at_alarm, 'mean_cohort_missing_at_decision': mean_missing_at_decision, 'monitor_coverage': monitor_coverage, 'waiting_coverage': waiting_coverage, 'calibration_coverage': calibration_coverage, 'skipped_single_class_updates': len(state['skipped_update_records']), 'initial_train_accuracy': state['train_acc'], 'initial_cal_accuracy': state['cal_acc'], 'initial_reference_loss': state['initial_reference'], 'task_elapsed_seconds': time.perf_counter() - start_clock})
        for record in decisions:
            decision_records.append({'dataset': dataset, 'learner': learner, 'scenario': scenario, 'delay_seed': delay_seed, 'strategy': strategy, **record})
        for record in updates:
            update_records.append({'dataset': dataset, 'learner': learner, 'scenario': scenario, 'delay_seed': delay_seed, 'strategy': strategy, **record})
        for record in state['skipped_update_records']:
            skipped_records.append({'dataset': dataset, 'learner': learner, 'scenario': scenario, 'delay_seed': delay_seed, 'strategy': strategy, **record})
    return (run_records, decision_records, update_records, skipped_records)

def run_task(*args):
    return run_one(*args)

def summarize(raw_df, group_columns):
    return raw_df.groupby(group_columns).agg(runs=('accuracy', 'count'), accuracy_mean=('accuracy', 'mean'), accuracy_std=('accuracy', 'std'), macro_f1_mean=('macro_f1', 'mean'), mcc_mean=('mcc', 'mean'), retrain_mean=('retrain_count', 'mean'), decision_accuracy=('decision_accuracy', 'mean'), decision_delay=('decision_delay', 'mean'), false_positive=('false_positive', 'sum'), false_negative=('false_negative', 'sum'), alarm_to_update=('mean_alarm_to_update', 'mean'), train_class1_rate=('mean_train_class1_rate', 'mean'), cohort_class1_rate=('mean_cohort_class1_rate', 'mean'), cohort_missing_at_alarm=('mean_cohort_missing_at_alarm', 'mean'), monitor_coverage=('monitor_coverage', 'mean'), skipped_single_class_updates=('skipped_single_class_updates', 'sum')).reset_index()

def build_pairwise(raw_df):
    key_columns = ['dataset', 'learner', 'scenario', 'delay_seed']
    soft = raw_df[raw_df['strategy'] == 'FULL_SOFT_Q30']
    complete = raw_df[raw_df['strategy'] == 'FULL_COMPLETE_COHORT']
    wait_complete = raw_df[raw_df['strategy'] == 'WAIT_ALL_COMPLETE_COHORT']
    comparisons = [('COMPLETE_vs_SOFT', complete, soft), ('FULL_COMPLETE_vs_WAIT_COMPLETE', complete, wait_complete)]
    records = []
    for comparison_name, method_a, method_b in comparisons:
        merged = method_a.merge(method_b, on=key_columns, suffixes=('_a', '_b'))
        for metric in ['accuracy', 'macro_f1', 'mcc']:
            differences = merged[f'{metric}_a'].to_numpy() - merged[f'{metric}_b'].to_numpy()
            wins = int(np.sum(differences > FLOAT_TOL))
            losses = int(np.sum(differences < -FLOAT_TOL))
            ties = int(len(differences) - wins - losses)
            records.append({'comparison': comparison_name, 'metric': metric, 'pairs': len(differences), 'mean_a': float(merged[f'{metric}_a'].mean()), 'mean_b': float(merged[f'{metric}_b'].mean()), 'mean_difference_a_minus_b': float(np.mean(differences)), 'median_difference': float(np.median(differences)), 'wins_a': wins, 'ties': ties, 'losses_a': losses})
    return pd.DataFrame(records)

def main():
    total_start = time.perf_counter()
    print('=' * 155)
    print('COMPLETE EVENT-TIME COHORT COMPARISON')
    print('=' * 155)
    print('Reusing the chronological benchmark inputs')
    print('MAX_SAMPLES：', MAX_SAMPLES)
    print('Window：', WINDOW_SIZE)
    print('Epsilon：', EPSILON)
    print('Soft q：', MIN_CLASS_COVERAGE)
    print('Calibration：', POST_UPDATE_CAL_SIZE)
    print('Delay Seeds：', DELAY_SEEDS)
    print('Strategies: ', STRATEGIES)
    datasets = {}
    feasibility_records = []
    print()
    print('=' * 155)
    print('Loading the same chronological benchmark inputs')
    print('=' * 155)
    for dataset, config in RAW_DATASETS.items():
        X, y, feasibility = base.load_natural_dataset(dataset, config)
        datasets[dataset] = (X, y)
        feasibility_records.append(feasibility)
    feasibility_df = pd.DataFrame(feasibility_records)
    tasks = []
    for dataset, (X, y) in datasets.items():
        for learner in LEARNERS:
            for scenario, config in DELAY_SCENARIOS.items():
                for seed in DELAY_SEEDS:
                    tasks.append((dataset, X, y, learner, scenario, config[0], config[1], seed))
    expected_rows = len(tasks) * len(STRATEGIES)
    print()
    print('=' * 155)
    print('Starting the cohort-selection comparison')
    print('=' * 155)
    print('Tasks: ', len(tasks))
    print('Number of strategies: ', len(STRATEGIES))
    print('Expected result rows: ', expected_rows)
    results = Parallel(n_jobs=CPU_JOBS, backend='loky', verbose=10, batch_size=1, pre_dispatch='2*n_jobs')((delayed(run_task)(*task) for task in tasks))
    all_runs = []
    all_decisions = []
    all_updates = []
    all_skipped = []
    for run_records, decision_records, update_records, skipped_records in results:
        all_runs.extend(run_records)
        all_decisions.extend(decision_records)
        all_updates.extend(update_records)
        all_skipped.extend(skipped_records)
    raw_df = pd.DataFrame(all_runs)
    decision_df = pd.DataFrame(all_decisions)
    update_df = pd.DataFrame(all_updates)
    skipped_df = pd.DataFrame(all_skipped)
    print('Actual result rows: ', len(raw_df))
    if len(raw_df) != expected_rows:
        raise RuntimeError('The cohort-selection comparison results are incomplete.')
    overall_df = summarize(raw_df, ['strategy'])
    dataset_df = summarize(raw_df, ['dataset', 'strategy'])
    dataset_learner_df = summarize(raw_df, ['dataset', 'learner', 'strategy'])
    scenario_df = summarize(raw_df, ['scenario', 'strategy'])
    pairwise_df = build_pairwise(raw_df)
    safety_records = []
    for strategy in ADAPTIVE_STRATEGIES:
        subset = raw_df[raw_df['strategy'] == strategy]
        valid = subset[subset['decision_accuracy'].notna()]
        nonperfect_runs = int((valid['decision_accuracy'] < 1.0 - FLOAT_TOL).sum())
        safety_records.append({'strategy': strategy, 'runs': len(subset), 'nonperfect_runs': nonperfect_runs, 'false_positive': int(subset['false_positive'].sum()), 'false_negative': int(subset['false_negative'].sum()), 'decision_accuracy': float(subset['decision_accuracy'].mean()), 'decision_delay': float(subset['decision_delay'].mean()), 'alarm_to_update': float(subset['mean_alarm_to_update'].mean()), 'retrain_mean': float(subset['retrain_count'].mean())})
    safety_df = pd.DataFrame(safety_records)
    full_complete = overall_df[overall_df['strategy'] == 'FULL_COMPLETE_COHORT'].iloc[0]
    wait_complete = overall_df[overall_df['strategy'] == 'WAIT_ALL_COMPLETE_COHORT'].iloc[0]
    soft = overall_df[overall_df['strategy'] == 'FULL_SOFT_Q30'].iloc[0]
    key_comparison_df = pd.DataFrame([{'comparison': 'FULL_COMPLETE_vs_WAIT_ALL_COMPLETE', 'accuracy_A': full_complete['accuracy_mean'], 'accuracy_B': wait_complete['accuracy_mean'], 'accuracy_difference': full_complete['accuracy_mean'] - wait_complete['accuracy_mean'], 'decision_delay_A': full_complete['decision_delay'], 'decision_delay_B': wait_complete['decision_delay'], 'decision_time_saved': wait_complete['decision_delay'] - full_complete['decision_delay'], 'alarm_to_update_A': full_complete['alarm_to_update'], 'alarm_to_update_B': wait_complete['alarm_to_update'], 'retrain_A': full_complete['retrain_mean'], 'retrain_B': wait_complete['retrain_mean']}, {'comparison': 'FULL_COMPLETE_vs_SOFT_Q30', 'accuracy_A': full_complete['accuracy_mean'], 'accuracy_B': soft['accuracy_mean'], 'accuracy_difference': full_complete['accuracy_mean'] - soft['accuracy_mean'], 'decision_delay_A': full_complete['decision_delay'], 'decision_delay_B': soft['decision_delay'], 'decision_time_saved': soft['decision_delay'] - full_complete['decision_delay'], 'alarm_to_update_A': full_complete['alarm_to_update'], 'alarm_to_update_B': soft['alarm_to_update'], 'retrain_A': full_complete['retrain_mean'], 'retrain_B': soft['retrain_mean']}])
    print()
    print('=' * 155)
    print('OVERALL')
    print('=' * 155)
    print(overall_df.to_string(index=False))
    print()
    print('=' * 155)
    print('DATASET × STRATEGY')
    print('=' * 155)
    print(dataset_df.to_string(index=False))
    print()
    print('=' * 155)
    print('KEY COMPARISONS')
    print('=' * 155)
    print(key_comparison_df.to_string(index=False))
    print()
    print('=' * 155)
    print('PAIRED WIN/TIE/LOSS')
    print('=' * 155)
    print(pairwise_df.to_string(index=False))
    print()
    print('=' * 155)
    print('SAFETY')
    print('=' * 155)
    print(safety_df.to_string(index=False))
    paths = {'feasibility': os.path.join(OUTPUT_DIR, 'natural_stream_feasibility.csv'), 'raw': os.path.join(OUTPUT_DIR, 'raw.csv'), 'decision': os.path.join(OUTPUT_DIR, 'decision_raw.csv'), 'update': os.path.join(OUTPUT_DIR, 'update_raw.csv'), 'skipped': os.path.join(OUTPUT_DIR, 'skipped_updates.csv'), 'overall': os.path.join(OUTPUT_DIR, 'overall.csv'), 'dataset': os.path.join(OUTPUT_DIR, 'dataset_summary.csv'), 'dataset_learner': os.path.join(OUTPUT_DIR, 'dataset_learner_summary.csv'), 'scenario': os.path.join(OUTPUT_DIR, 'scenario_summary.csv'), 'pairwise': os.path.join(OUTPUT_DIR, 'pairwise.csv'), 'key': os.path.join(OUTPUT_DIR, 'key_comparisons.csv'), 'safety': os.path.join(OUTPUT_DIR, 'safety.csv')}
    feasibility_df.to_csv(paths['feasibility'], index=False, encoding='utf-8-sig')
    raw_df.to_csv(paths['raw'], index=False, encoding='utf-8-sig')
    decision_df.to_csv(paths['decision'], index=False, encoding='utf-8-sig')
    update_df.to_csv(paths['update'], index=False, encoding='utf-8-sig')
    skipped_df.to_csv(paths['skipped'], index=False, encoding='utf-8-sig')
    overall_df.to_csv(paths['overall'], index=False, encoding='utf-8-sig')
    dataset_df.to_csv(paths['dataset'], index=False, encoding='utf-8-sig')
    dataset_learner_df.to_csv(paths['dataset_learner'], index=False, encoding='utf-8-sig')
    scenario_df.to_csv(paths['scenario'], index=False, encoding='utf-8-sig')
    pairwise_df.to_csv(paths['pairwise'], index=False, encoding='utf-8-sig')
    key_comparison_df.to_csv(paths['key'], index=False, encoding='utf-8-sig')
    safety_df.to_csv(paths['safety'], index=False, encoding='utf-8-sig')
    elapsed = time.perf_counter() - total_start
    print()
    print('=' * 155)
    print('Output files')
    print('=' * 155)
    for path in paths.values():
        print(path)
    print()
    print(f'运行时间：{elapsed:.2f} 秒')
    print(f'约：{elapsed / 60:.2f} 分钟')
    print()
    print('=' * 155)
    for strategy in ['FULL_SOFT_Q30', 'FULL_COMPLETE_COHORT']:
        row = safety_df[safety_df['strategy'] == strategy].iloc[0]
        passed = int(row['nonperfect_runs']) == 0 and int(row['false_positive']) == 0 and (int(row['false_negative']) == 0)
        print(f"{strategy} 安全检查：{('Passed: ' if passed else 'Failed: ')}")
    print()
    print('This experiment compares training-sample selection mechanisms.')
    print('Window size, epsilon, q, update size, calibration size, the 5,000-observation stream length, and seeds retain their fixed settings.')
    print('=' * 155)
if __name__ == '__main__':
    main()
