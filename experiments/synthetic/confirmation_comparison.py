"""Compare returned-label, completion-bound and complete-feedback decisions."""
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['BLIS_NUM_THREADS'] = '1'
import time
import hashlib
import importlib
import warnings
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from scipy.stats import wilcoxon
stream_v2 = importlib.import_module('unseen_stream_holdout')
legacy = importlib.import_module('chronological_protocol')
SCRIPT_NAME = 'confirmation_comparison'
OUTPUT_DIR = os.path.join('result', SCRIPT_NAME)
os.makedirs(OUTPUT_DIR, exist_ok=True)
INITIAL_TRAIN_END = 200
INITIAL_CAL_END = 400
MONITOR_START = 400
WINDOW_SIZE = 100
EPSILON = 0.02
FLOAT_TOL = 1e-12
POST_UPDATE_CAL_SIZE = 100
MIN_ARRIVED = 20
assert legacy.WINDOW_SIZE == WINDOW_SIZE
assert abs(legacy.EPSILON - EPSILON) <= 1e-15
assert legacy.POST_UPDATE_CAL_SIZE == POST_UPDATE_CAL_SIZE
assert legacy.MIN_ARRIVED == MIN_ARRIVED
DATASETS = ['SEA', 'Agrawal', 'Mixed', 'RandomTree', 'Sine']
STREAM_SEEDS = list(range(6001, 6011))
DELAY_SEEDS = list(range(7001, 7006))
LEARNERS = ['LR', 'DT']
DELAY_SCENARIOS = {'UNIFORM_100_100': {0: 100, 1: 100}, 'MILD_50_100': {0: 50, 1: 100}, 'MEDIUM_30_150': {0: 30, 1: 150}, 'SEVERE_20_200': {0: 20, 1: 200}, 'REVERSED_200_20': {0: 200, 1: 20}}
STRATEGIES = ['ARRIVED_ONLY_COMPLETE_COHORT', 'FULL_COMPLETE_COHORT', 'WAIT_ALL_COMPLETE_COHORT']
CPU_JOBS = 20
SMOKE_TEST = os.environ.get('STRESS_SMOKE', '0') == '1'

def stream_hash(X, y):
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(X).tobytes())
    h.update(np.ascontiguousarray(y).tobytes())
    return h.hexdigest()

def load_or_generate_stream(dataset, stream_seed):
    stream_path = os.path.join('result', 'unseen_stream_pipeline', 'streams', f'{dataset}_streamseed_{stream_seed}.npz')
    if os.path.exists(stream_path):
        data = np.load(stream_path, allow_pickle=True)
        X = np.asarray(data['X'], dtype=float)
        y = np.asarray(data['y'], dtype=int)
        digest = stream_hash(X, y)
        return {'dataset': dataset, 'stream_seed': int(stream_seed), 'X': X, 'y': y, 'stream_hash': digest, 'source': 'STEP23_SAVED_NPZ', 'ok': True}
    result = stream_v2.generate_one_stream_v2(dataset, stream_seed)
    if not result['generation_ok']:
        return {'dataset': dataset, 'stream_seed': int(stream_seed), 'X': None, 'y': None, 'stream_hash': '', 'source': 'STEP23_V2_REGENERATED', 'ok': False, 'error': result['error']}
    return {'dataset': dataset, 'stream_seed': int(stream_seed), 'X': np.asarray(result['X'], dtype=float), 'y': np.asarray(result['y'], dtype=int), 'stream_hash': result['stream_hash'], 'source': 'STEP23_V2_REGENERATED', 'ok': True}

def create_state(X, y, learner):
    model, reference_loss, train_accuracy, calibration_accuracy = legacy.create_initial_model(X, y, learner)
    return {'model': model, 'model_version': 0, 'reference_loss': float(reference_loss), 'predictions': {}, 'feedback': {}, 'phase': 'MONITORING', 'open_block': [], 'closed_blocks': [], 'next_block_id': 0, 'locked_cohort': None, 'alarm_time': None, 'alarm_block_id': None, 'calibration_ids': [], 'retrain_count': 0, 'decision_records': [], 'update_records': [], 'skipped_updates': [], 'monitor_samples': 0, 'waiting_samples': 0, 'calibration_samples': 0, 'initial_train_accuracy': float(train_accuracy), 'initial_calibration_accuracy': float(calibration_accuracy), 'initial_reference_loss': float(reference_loss)}

def calculate_oracle(y, state, block):
    losses = [int(state['predictions'][sample_id] != y[sample_id]) for sample_id in block['sample_ids']]
    final_loss = float(np.mean(losses))
    threshold = float(block['reference_loss']) + EPSILON
    oracle = 'DEGRADATION' if final_loss > threshold + FLOAT_TOL else 'TOLERABLE'
    return (oracle, final_loss, final_loss - block['reference_loss'])

def evaluate_complete_bounds(state, block):
    ids = block['sample_ids']
    n = len(ids)
    S = 0
    m = 0
    for sample_id in ids:
        if sample_id in state['feedback']:
            m += 1
            S += int(state['feedback'][sample_id])
    u = n - m
    lower = S / n
    upper = (S + u) / n
    threshold = block['reference_loss'] + EPSILON
    if lower > threshold + FLOAT_TOL:
        decision = 'DEGRADATION'
    elif upper <= threshold + FLOAT_TOL:
        decision = 'TOLERABLE'
    else:
        return None
    return {'decision': decision, 'm': int(m), 'u': int(u), 'observed_loss': S / m if m > 0 else np.nan, 'lower': float(lower), 'upper': float(upper), 'threshold': float(threshold)}

def evaluate_arrived_only(state, block):
    ids = block['sample_ids']
    arrived = [sample_id for sample_id in ids if sample_id in state['feedback']]
    m = len(arrived)
    if m < MIN_ARRIVED:
        return None
    arrived_loss = float(np.mean([state['feedback'][sample_id] for sample_id in arrived]))
    threshold = block['reference_loss'] + EPSILON
    decision = 'DEGRADATION' if arrived_loss > threshold + FLOAT_TOL else 'TOLERABLE'
    return {'decision': decision, 'm': int(m), 'u': int(len(ids) - m), 'observed_loss': arrived_loss, 'lower': np.nan, 'upper': np.nan, 'threshold': float(threshold)}

def evaluate_wait_all(state, block):
    ids = block['sample_ids']
    if not all((sample_id in state['feedback'] for sample_id in ids)):
        return None
    loss = float(np.mean([state['feedback'][sample_id] for sample_id in ids]))
    threshold = block['reference_loss'] + EPSILON
    decision = 'DEGRADATION' if loss > threshold + FLOAT_TOL else 'TOLERABLE'
    return {'decision': decision, 'm': int(len(ids)), 'u': 0, 'observed_loss': loss, 'lower': loss, 'upper': loss, 'threshold': float(threshold)}

def run_one(dataset, stream_seed, stream_digest, X, y, learner, scenario, lambda0, lambda1, delay_seed):
    task_start = time.perf_counter()
    arrival_map, arrival_groups = legacy.generate_delay(y, delay_seed, lambda0, lambda1)
    last_event = len(y) - 1
    simulation_end = max(last_event, max(arrival_map.values()))
    states = {strategy: create_state(X, y, learner) for strategy in STRATEGIES}
    for current_time in range(MONITOR_START, simulation_end + 1):
        for strategy in STRATEGIES:
            state = states[strategy]
            if state['phase'] != 'WAITING_READY':
                continue
            cohort = state['locked_cohort']
            if cohort is None:
                raise RuntimeError('WAITING_READY without locked cohort.')
            if current_time > last_event:
                continue
            cohort_ready = all((sample_id in state['feedback'] for sample_id in cohort))
            if not cohort_ready:
                continue
            train_ids = np.asarray(cohort, dtype=int)
            train_labels = y[train_ids]
            unique_classes = np.unique(train_labels)
            if len(unique_classes) < 2:
                state['skipped_updates'].append({'time': int(current_time), 'alarm_time': int(state['alarm_time']), 'block_id': int(state['alarm_block_id']), 'reason': 'SINGLE_CLASS_LOCKED_COHORT', 'class': int(unique_classes[0])})
                state['phase'] = 'MONITORING'
                state['locked_cohort'] = None
                state['alarm_time'] = None
                state['alarm_block_id'] = None
                state['open_block'] = []
                continue
            new_model = legacy.create_model(learner)
            new_model.fit(X[train_ids], y[train_ids])
            old_version = int(state['model_version'])
            old_reference = float(state['reference_loss'])
            state['model'] = new_model
            state['model_version'] += 1
            state['retrain_count'] += 1
            state['reference_loss'] = np.nan
            state['phase'] = 'CALIBRATING'
            state['calibration_ids'] = []
            state['open_block'] = []
            state['update_records'].append({'update_time': int(current_time), 'alarm_time': int(state['alarm_time']), 'alarm_to_update': int(current_time - state['alarm_time']), 'block_id': int(state['alarm_block_id']), 'old_model_version': old_version, 'new_model_version': int(state['model_version']), 'old_reference_loss': old_reference, 'train_size': int(len(train_ids)), 'train_class0': int(np.sum(train_labels == 0)), 'train_class1': int(np.sum(train_labels == 1)), 'train_class1_rate': float(np.mean(train_labels == 1))})
            state['locked_cohort'] = None
            state['alarm_time'] = None
            state['alarm_block_id'] = None
        if current_time <= last_event:
            sample_id = int(current_time)
            for strategy in STRATEGIES:
                state = states[strategy]
                prediction = int(state['model'].predict(X[sample_id].reshape(1, -1))[0])
                state['predictions'][sample_id] = prediction
                if state['phase'] == 'MONITORING':
                    state['monitor_samples'] += 1
                    state['open_block'].append(sample_id)
                    if len(state['open_block']) == WINDOW_SIZE:
                        block_ids = list(state['open_block'])
                        state['closed_blocks'].append({'block_id': int(state['next_block_id']), 'model_version': int(state['model_version']), 'reference_loss': float(state['reference_loss']), 'sample_ids': block_ids, 'start': int(block_ids[0]), 'close_time': int(block_ids[-1]), 'status': 'PENDING'})
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
                    raise RuntimeError(f'{dataset} / {strategy}: missing original prediction for sample {sample_id}')
                state['feedback'][sample_id] = int(state['predictions'][sample_id] != y[sample_id])
        for strategy in STRATEGIES:
            state = states[strategy]
            if state['phase'] != 'CALIBRATING':
                continue
            if len(state['calibration_ids']) != POST_UPDATE_CAL_SIZE:
                continue
            cal_ids = state['calibration_ids']
            if not all((sample_id in state['feedback'] for sample_id in cal_ids)):
                continue
            state['reference_loss'] = float(np.mean([state['feedback'][sample_id] for sample_id in cal_ids]))
            state['phase'] = 'MONITORING'
            state['calibration_ids'] = []
            state['open_block'] = []
        for strategy in STRATEGIES:
            state = states[strategy]
            if state['phase'] != 'MONITORING':
                continue
            for block in state['closed_blocks']:
                if block['status'] != 'PENDING':
                    continue
                if strategy == 'ARRIVED_ONLY_COMPLETE_COHORT':
                    result = evaluate_arrived_only(state, block)
                elif strategy == 'FULL_COMPLETE_COHORT':
                    result = evaluate_complete_bounds(state, block)
                elif strategy == 'WAIT_ALL_COMPLETE_COHORT':
                    result = evaluate_wait_all(state, block)
                else:
                    raise RuntimeError(strategy)
                if result is None:
                    continue
                block['status'] = 'DONE'
                oracle, final_loss, final_delta = calculate_oracle(y, state, block)
                decision = result['decision']
                correct = int(decision == oracle)
                labels_at_decision = int(result['m'])
                unknown_at_decision = int(result['u'])
                returned_fraction = labels_at_decision / WINDOW_SIZE
                state['decision_records'].append({'block_id': int(block['block_id']), 'model_version': int(block['model_version']), 'decision': decision, 'oracle': oracle, 'correct': correct, 'decision_time': int(current_time), 'block_close_time': int(block['close_time']), 'decision_delay': int(current_time - block['close_time']), 'labels_at_decision': labels_at_decision, 'unknown_at_decision': unknown_at_decision, 'returned_fraction': float(returned_fraction), 'early_confirmation': int(labels_at_decision < WINDOW_SIZE), 'reference_loss': float(block['reference_loss']), 'threshold': float(block['reference_loss'] + EPSILON), 'observed_loss': result['observed_loss'], 'lower_bound': result['lower'], 'upper_bound': result['upper'], 'final_loss': float(final_loss), 'final_delta': float(final_delta)})
                if decision == 'DEGRADATION':
                    state['phase'] = 'WAITING_READY'
                    state['locked_cohort'] = list(block['sample_ids'])
                    state['alarm_time'] = int(current_time)
                    state['alarm_block_id'] = int(block['block_id'])
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
        pred = np.asarray([state['predictions'][int(sample_id)] for sample_id in monitor_ids], dtype=int)
        accuracy = float(accuracy_score(y_true, pred))
        macro_f1 = float(f1_score(y_true, pred, average='macro'))
        mcc = float(matthews_corrcoef(y_true, pred))
        decisions = state['decision_records']
        if decisions:
            oracle_agreement = float(np.mean([x['correct'] for x in decisions]))
            decision_delay = float(np.mean([x['decision_delay'] for x in decisions]))
            median_decision_delay = float(np.median([x['decision_delay'] for x in decisions]))
            mean_labels = float(np.mean([x['labels_at_decision'] for x in decisions]))
            median_labels = float(np.median([x['labels_at_decision'] for x in decisions]))
            returned_fraction = float(np.mean([x['returned_fraction'] for x in decisions]))
            early_confirmation_rate = float(np.mean([x['early_confirmation'] for x in decisions]))
            fp = int(sum((x['decision'] == 'DEGRADATION' and x['oracle'] == 'TOLERABLE' for x in decisions)))
            fn = int(sum((x['decision'] == 'TOLERABLE' and x['oracle'] == 'DEGRADATION' for x in decisions)))
        else:
            oracle_agreement = np.nan
            decision_delay = np.nan
            median_decision_delay = np.nan
            mean_labels = np.nan
            median_labels = np.nan
            returned_fraction = np.nan
            early_confirmation_rate = np.nan
            fp = 0
            fn = 0
        updates = state['update_records']
        if updates:
            mean_alarm_to_update = float(np.mean([x['alarm_to_update'] for x in updates]))
        else:
            mean_alarm_to_update = np.nan
        monitor_coverage = state['monitor_samples'] / len(monitor_ids)
        run_records.append({'dataset': dataset, 'stream_seed': int(stream_seed), 'stream_hash': stream_digest, 'learner': learner, 'scenario': scenario, 'lambda0': int(lambda0), 'lambda1': int(lambda1), 'delay_seed': int(delay_seed), 'strategy': strategy, 'accuracy': accuracy, 'macro_f1': macro_f1, 'mcc': mcc, 'retrain_count': int(state['retrain_count']), 'decision_count': int(len(decisions)), 'oracle_agreement': oracle_agreement, 'decision_delay_mean': decision_delay, 'decision_delay_median': median_decision_delay, 'mean_labels_at_decision': mean_labels, 'median_labels_at_decision': median_labels, 'mean_returned_fraction': returned_fraction, 'early_confirmation_rate': early_confirmation_rate, 'false_positive': fp, 'false_negative': fn, 'alarm_to_update': mean_alarm_to_update, 'monitor_coverage': float(monitor_coverage), 'skipped_single_class_updates': int(len(state['skipped_updates'])), 'initial_train_accuracy': state['initial_train_accuracy'], 'initial_calibration_accuracy': state['initial_calibration_accuracy'], 'initial_reference_loss': state['initial_reference_loss'], 'elapsed_seconds': time.perf_counter() - task_start})
        for record in decisions:
            decision_records.append({'dataset': dataset, 'stream_seed': int(stream_seed), 'stream_hash': stream_digest, 'learner': learner, 'scenario': scenario, 'lambda0': int(lambda0), 'lambda1': int(lambda1), 'delay_seed': int(delay_seed), 'strategy': strategy, **record})
        for record in updates:
            update_records.append({'dataset': dataset, 'stream_seed': int(stream_seed), 'learner': learner, 'scenario': scenario, 'delay_seed': int(delay_seed), 'strategy': strategy, **record})
        for record in state['skipped_updates']:
            skipped_records.append({'dataset': dataset, 'stream_seed': int(stream_seed), 'learner': learner, 'scenario': scenario, 'delay_seed': int(delay_seed), 'strategy': strategy, **record})
    return (run_records, decision_records, update_records, skipped_records)

def summarize_runs(raw_df, group_cols):
    return raw_df.groupby(group_cols).agg(runs=('accuracy', 'count'), accuracy_mean=('accuracy', 'mean'), macro_f1_mean=('macro_f1', 'mean'), mcc_mean=('mcc', 'mean'), retrain_mean=('retrain_count', 'mean'), oracle_agreement=('oracle_agreement', 'mean'), confirmation_delay_mean=('decision_delay_mean', 'mean'), confirmation_delay_median=('decision_delay_median', 'mean'), mean_labels_at_decision=('mean_labels_at_decision', 'mean'), mean_returned_fraction=('mean_returned_fraction', 'mean'), early_confirmation_rate=('early_confirmation_rate', 'mean'), false_positive=('false_positive', 'sum'), false_negative=('false_negative', 'sum'), alarm_to_update=('alarm_to_update', 'mean'), monitor_coverage=('monitor_coverage', 'mean'), skipped_single_class_updates=('skipped_single_class_updates', 'sum')).reset_index()

def main():
    total_start = time.perf_counter()
    print('=' * 155)
    print('FINAL ARRIVED-ONLY STRESS TEST')
    print('=' * 155)
    print('Purpose:')
    print('Compare fast Arrived-Only confirmation against completion-sound bounds and Wait-All.')
    print()
    print('IMPORTANT:')
    print('All degradation alarms use the SAME exact triggering event-time block for adaptation.')
    print()
    print('Window:', WINDOW_SIZE)
    print('Epsilon:', EPSILON)
    print('Arrived-Only minimum returned labels:', MIN_ARRIVED)
    print('Calibration size:', POST_UPDATE_CAL_SIZE)
    print('Smoke test:', SMOKE_TEST)
    if SMOKE_TEST:
        selected_datasets = ['SEA']
        selected_stream_seeds = [6001]
        selected_learners = ['LR']
        selected_scenarios = {'UNIFORM_100_100': DELAY_SCENARIOS['UNIFORM_100_100']}
        selected_delay_seeds = [7001]
        cpu_jobs = 1
    else:
        selected_datasets = DATASETS
        selected_stream_seeds = STREAM_SEEDS
        selected_learners = LEARNERS
        selected_scenarios = DELAY_SCENARIOS
        selected_delay_seeds = DELAY_SEEDS
        cpu_jobs = CPU_JOBS
    print()
    print('=' * 155)
    print('PHASE 1 — LOAD VERIFIED synthetic-stream pipeline STREAMS')
    print('=' * 155)
    streams = []
    for dataset in selected_datasets:
        for stream_seed in selected_stream_seeds:
            record = load_or_generate_stream(dataset, stream_seed)
            if not record['ok']:
                raise RuntimeError(f"Stream generation failed: {dataset}, {stream_seed}\n{record.get('error', '')}")
            streams.append(record)
            print(f"{dataset:12s} seed={stream_seed} N={len(record['y'])} d={record['X'].shape[1]} hash={record['stream_hash'][:12]} source={record['source']}")
    hashes = [(x['dataset'], x['stream_hash']) for x in streams]
    if len(hashes) != len(set(hashes)):
        raise RuntimeError('Duplicate stream hash detected.')
    tasks = []
    for stream in streams:
        for learner in selected_learners:
            for scenario, config in selected_scenarios.items():
                for delay_seed in selected_delay_seeds:
                    tasks.append((stream['dataset'], stream['stream_seed'], stream['stream_hash'], stream['X'], stream['y'], learner, scenario, config[0], config[1], delay_seed))
    expected_rows = len(tasks) * len(STRATEGIES)
    print()
    print('=' * 155)
    print('PHASE 2 — STRESS TEST')
    print('=' * 155)
    print('Independent stream/delay tasks:', len(tasks))
    print('Strategies:', STRATEGIES)
    print('Expected run-level rows:', expected_rows)
    results = Parallel(n_jobs=cpu_jobs, backend='loky', verbose=10, batch_size=1, pre_dispatch='2*n_jobs')((delayed(run_one)(*task) for task in tasks))
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
    print('=' * 155)
    print('COMPLETENESS CHECK')
    print('=' * 155)
    print('Expected:', expected_rows)
    print('Actual:', len(raw_df))
    if len(raw_df) != expected_rows:
        raise RuntimeError('Incomplete confirmation comparison results.')
    overall_df = summarize_runs(raw_df, ['strategy'])
    scenario_df = summarize_runs(raw_df, ['scenario', 'strategy'])
    dataset_df = summarize_runs(raw_df, ['dataset', 'strategy'])
    dataset_learner_df = summarize_runs(raw_df, ['dataset', 'learner', 'strategy'])
    if not decision_df.empty:
        decision_summary_df = decision_df.groupby('strategy').agg(decisions=('correct', 'count'), oracle_agreement=('correct', 'mean'), mean_confirmation_delay=('decision_delay', 'mean'), median_confirmation_delay=('decision_delay', 'median'), mean_labels_at_decision=('labels_at_decision', 'mean'), median_labels_at_decision=('labels_at_decision', 'median'), mean_returned_fraction=('returned_fraction', 'mean'), median_returned_fraction=('returned_fraction', 'median'), early_confirmation_rate=('early_confirmation', 'mean')).reset_index()
        fp_records = []
        for strategy in STRATEGIES:
            subset = decision_df[decision_df['strategy'] == strategy]
            fp = int(((subset['decision'] == 'DEGRADATION') & (subset['oracle'] == 'TOLERABLE')).sum())
            fn = int(((subset['decision'] == 'TOLERABLE') & (subset['oracle'] == 'DEGRADATION')).sum())
            fp_records.append({'strategy': strategy, 'false_positive': fp, 'false_negative': fn})
        error_df = pd.DataFrame(fp_records)
        decision_summary_df = decision_summary_df.merge(error_df, on='strategy', how='left')
    else:
        decision_summary_df = pd.DataFrame()
    main_table_df = overall_df[['strategy', 'runs', 'oracle_agreement', 'false_positive', 'false_negative', 'confirmation_delay_mean', 'mean_labels_at_decision', 'mean_returned_fraction', 'early_confirmation_rate', 'accuracy_mean', 'macro_f1_mean', 'mcc_mean', 'retrain_mean']].copy()
    cell_df = raw_df.groupby(['dataset', 'stream_seed', 'learner', 'scenario', 'strategy']).agg(delay_seed_runs=('accuracy', 'count'), oracle_agreement=('oracle_agreement', 'mean'), confirmation_delay=('decision_delay_mean', 'mean'), false_positive=('false_positive', 'sum'), false_negative=('false_negative', 'sum'), returned_fraction=('mean_returned_fraction', 'mean'), early_confirmation_rate=('early_confirmation_rate', 'mean'), accuracy=('accuracy', 'mean'), macro_f1=('macro_f1', 'mean'), mcc=('mcc', 'mean')).reset_index()
    pareto_df = cell_df.groupby('strategy').agg(cells=('oracle_agreement', 'count'), oracle_agreement=('oracle_agreement', 'mean'), mean_confirmation_delay=('confirmation_delay', 'mean'), mean_returned_fraction=('returned_fraction', 'mean'), early_confirmation_rate=('early_confirmation_rate', 'mean'), total_fp=('false_positive', 'sum'), total_fn=('false_negative', 'sum')).reset_index()
    outputs = {'raw': os.path.join(OUTPUT_DIR, 'raw.csv'), 'decisions': os.path.join(OUTPUT_DIR, 'decision_raw.csv'), 'updates': os.path.join(OUTPUT_DIR, 'update_raw.csv'), 'skipped': os.path.join(OUTPUT_DIR, 'skipped_updates.csv'), 'overall': os.path.join(OUTPUT_DIR, 'overall_summary.csv'), 'scenario': os.path.join(OUTPUT_DIR, 'scenario_summary.csv'), 'dataset': os.path.join(OUTPUT_DIR, 'dataset_summary.csv'), 'dataset_learner': os.path.join(OUTPUT_DIR, 'dataset_learner_summary.csv'), 'decision_summary': os.path.join(OUTPUT_DIR, 'decision_summary.csv'), 'main_table': os.path.join(OUTPUT_DIR, 'main_paper_table.csv'), 'cells': os.path.join(OUTPUT_DIR, 'cell_summary.csv'), 'pareto': os.path.join(OUTPUT_DIR, 'safety_latency_pareto_table.csv')}
    raw_df.to_csv(outputs['raw'], index=False, encoding='utf-8-sig')
    decision_df.to_csv(outputs['decisions'], index=False, encoding='utf-8-sig')
    update_df.to_csv(outputs['updates'], index=False, encoding='utf-8-sig')
    skipped_df.to_csv(outputs['skipped'], index=False, encoding='utf-8-sig')
    overall_df.to_csv(outputs['overall'], index=False, encoding='utf-8-sig')
    scenario_df.to_csv(outputs['scenario'], index=False, encoding='utf-8-sig')
    dataset_df.to_csv(outputs['dataset'], index=False, encoding='utf-8-sig')
    dataset_learner_df.to_csv(outputs['dataset_learner'], index=False, encoding='utf-8-sig')
    decision_summary_df.to_csv(outputs['decision_summary'], index=False, encoding='utf-8-sig')
    main_table_df.to_csv(outputs['main_table'], index=False, encoding='utf-8-sig')
    cell_df.to_csv(outputs['cells'], index=False, encoding='utf-8-sig')
    pareto_df.to_csv(outputs['pareto'], index=False, encoding='utf-8-sig')
    print()
    print('=' * 155)
    print('MAIN PAPER TABLE')
    print('=' * 155)
    print(main_table_df.to_string(index=False))
    print()
    print('=' * 155)
    print('DECISION-LEVEL SAFETY')
    print('=' * 155)
    print(decision_summary_df.to_string(index=False))
    print()
    print('=' * 155)
    print('SAFETY-LATENCY PARETO TABLE')
    print('=' * 155)
    print(pareto_df.to_string(index=False))
    print()
    print('=' * 155)
    print('SCENARIO BREAKDOWN')
    print('=' * 155)
    print(scenario_df[['scenario', 'strategy', 'oracle_agreement', 'confirmation_delay_mean', 'mean_labels_at_decision', 'mean_returned_fraction', 'false_positive', 'false_negative']].to_string(index=False))
    complete_row = main_table_df[main_table_df['strategy'] == 'FULL_COMPLETE_COHORT'].iloc[0]
    wait_row = main_table_df[main_table_df['strategy'] == 'WAIT_ALL_COMPLETE_COHORT'].iloc[0]
    arrived_row = main_table_df[main_table_df['strategy'] == 'ARRIVED_ONLY_COMPLETE_COHORT'].iloc[0]
    complete_safety_pass = abs(complete_row['oracle_agreement'] - 1.0) <= 1e-12 and int(complete_row['false_positive']) == 0 and (int(complete_row['false_negative']) == 0)
    wait_safety_pass = abs(wait_row['oracle_agreement'] - 1.0) <= 1e-12 and int(wait_row['false_positive']) == 0 and (int(wait_row['false_negative']) == 0)
    delay_reduction = (wait_row['confirmation_delay_mean'] - complete_row['confirmation_delay_mean']) / wait_row['confirmation_delay_mean'] * 100.0
    print()
    print('=' * 155)
    print('FINAL SCIENTIFIC CHECK')
    print('=' * 155)
    print('Complete Cohort safety:', 'PASS ' if complete_safety_pass else 'CHECK ')
    print('Wait-All safety:', 'PASS ' if wait_safety_pass else 'CHECK ')
    print(f'Complete vs Wait-All delay reduction: {delay_reduction:.2f}%')
    print('Arrived-Only Oracle Agreement:', f"{arrived_row['oracle_agreement']:.6f}")
    print('Arrived-Only FP:', int(arrived_row['false_positive']))
    print('Arrived-Only FN:', int(arrived_row['false_negative']))
    elapsed = time.perf_counter() - total_start
    print()
    print(f'Total runtime: {elapsed:.2f} s = {elapsed / 60:.2f} min')
    print()
    print('Output directory:')
    print(OUTPUT_DIR)
    print('=' * 155)
if __name__ == '__main__':
    main()
