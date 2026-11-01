"""Chronological prediction, delayed feedback and reference-window monitoring."""
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['BLIS_NUM_THREADS'] = '1'
import time
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
SCRIPT_NAME = 'chronological_protocol'
OUTPUT_DIR = os.path.join('result', SCRIPT_NAME)
os.makedirs(OUTPUT_DIR, exist_ok=True)
CPU_JOBS = 20
RAW_DATASETS = {'Electricity': {'path': os.path.join('stream_ds', 'electricity.csv.xz'), 'target': 'class', 'drop': ['date', 'Unnamed: 0']}, 'Weather': {'path': os.path.join('stream_ds', 'neweather.csv.xz'), 'target': 'y', 'drop': ['Unnamed: 0']}}
MAX_SAMPLES = 5000
LEARNERS = ['LR', 'DT']
MODEL_RANDOM_SEED = 2026
INITIAL_TRAIN_START = 0
INITIAL_TRAIN_END = 200
INITIAL_CAL_START = 200
INITIAL_CAL_END = 400
MONITOR_START = 400
WINDOW_SIZE = 100
EPSILON = 0.02
FLOAT_TOL = 1e-12
UPDATE_TRAIN_SIZE = 100
MIN_CLASS_COVERAGE = 30
POST_UPDATE_CAL_SIZE = 100
MIN_ARRIVED = 20
DELAY_SEEDS = list(range(5001, 5021))
DELAY_SCENARIOS = {'UNIFORM_100_100': {0: 100, 1: 100}, 'MILD_50_100': {0: 50, 1: 100}, 'MEDIUM_30_150': {0: 30, 1: 150}, 'SEVERE_20_200': {0: 20, 1: 200}, 'REVERSED_200_20': {0: 200, 1: 20}}
STRATEGIES = ['STATIC', 'PERIODIC_100', 'PERIODIC_300', 'PERIODIC_600', 'ARRIVED_ONLY', 'WAIT_ALL', 'FULL']
PERIODIC_INTERVALS = {'PERIODIC_100': 100, 'PERIODIC_300': 300, 'PERIODIC_600': 600}
SELECTIVE_STRATEGIES = ['ARRIVED_ONLY', 'WAIT_ALL', 'FULL']

def create_model(learner):
    if learner == 'LR':
        return Pipeline([('scaler', StandardScaler()), ('classifier', LogisticRegression(max_iter=3000, random_state=MODEL_RANDOM_SEED))])
    if learner == 'DT':
        return DecisionTreeClassifier(min_samples_leaf=2, random_state=MODEL_RANDOM_SEED)
    raise ValueError(learner)

def load_natural_dataset(dataset, config):
    path = config['path']
    target = config['target']
    drop_columns = config['drop']
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    print()
    print(f'{dataset} 原始shape：', df.shape)
    print(f'{dataset} 原始列：', list(df.columns))
    if target not in df.columns:
        raise RuntimeError(f'{dataset}: 找不到target={target}')
    for column in drop_columns:
        if column in df.columns:
            df = df.drop(columns=[column])
    if len(df) < MAX_SAMPLES:
        raise RuntimeError(f'{dataset}: 总样本只有{len(df)}，不足{MAX_SAMPLES}')
    df = df.iloc[:MAX_SAMPLES].copy()
    raw_y = df[target].to_numpy()
    encoder = LabelEncoder()
    y = encoder.fit_transform(raw_y).astype(int)
    if len(np.unique(y)) != 2:
        raise RuntimeError(f'{dataset}: 前{MAX_SAMPLES}条不是二分类。')
    X_df = df.drop(columns=[target]).copy()
    for column in X_df.columns:
        X_df[column] = pd.to_numeric(X_df[column], errors='coerce')
    if X_df.isna().any().any():
        bad_columns = X_df.columns[X_df.isna().any()].tolist()
        raise RuntimeError(f'{dataset}: 存在非数值/缺失特征：{bad_columns}')
    X = X_df.to_numpy(dtype=float)
    if np.isinf(X).any():
        raise RuntimeError(f'{dataset}: X存在Inf')
    train_y = y[0:INITIAL_TRAIN_END]
    cal_y = y[INITIAL_TRAIN_END:INITIAL_CAL_END]
    monitor_y = y[MONITOR_START:]
    if len(np.unique(train_y)) != 2:
        raise RuntimeError(f'{dataset}: 前200训练样本只有一个类别，不兼容冻结协议。')
    monitor_counts = np.bincount(monitor_y, minlength=2)
    if np.min(monitor_counts) < MIN_CLASS_COVERAGE:
        raise RuntimeError(f'{dataset}: monitor少数类总数不足q=30。')
    block_minority = []
    for start in range(MONITOR_START, MAX_SAMPLES, WINDOW_SIZE):
        block_y = y[start:start + WINDOW_SIZE]
        counts = np.bincount(block_y, minlength=2)
        block_minority.append(int(np.min(counts)))
    print(f'{dataset} 使用shape：', X.shape)
    print(f'{dataset} 总类别：', np.bincount(y, minlength=2))
    print(f'{dataset} Train类别：', np.bincount(train_y, minlength=2))
    print(f'{dataset} Calibration类别：', np.bincount(cal_y, minlength=2))
    print(f'{dataset} Monitor类别：', monitor_counts)
    print(f'{dataset} 100窗口平均少数类：', f'{np.mean(block_minority):.2f}')
    print(f'{dataset} 100窗口最小少数类：', int(np.min(block_minority)))
    return (X, y, {'dataset': dataset, 'n_samples': len(y), 'n_features': X.shape[1], 'class0_total': int(np.sum(y == 0)), 'class1_total': int(np.sum(y == 1)), 'train_class0': int(np.sum(train_y == 0)), 'train_class1': int(np.sum(train_y == 1)), 'monitor_class0': int(monitor_counts[0]), 'monitor_class1': int(monitor_counts[1]), 'mean_block_minority': float(np.mean(block_minority)), 'min_block_minority': int(np.min(block_minority))})

def create_initial_model(X, y, learner):
    train_ids = np.arange(INITIAL_TRAIN_START, INITIAL_TRAIN_END)
    cal_ids = np.arange(INITIAL_CAL_START, INITIAL_CAL_END)
    model = create_model(learner)
    model.fit(X[train_ids], y[train_ids])
    train_pred = model.predict(X[train_ids])
    cal_pred = model.predict(X[cal_ids])
    train_acc = float(accuracy_score(y[train_ids], train_pred))
    cal_acc = float(accuracy_score(y[cal_ids], cal_pred))
    reference_loss = float(np.mean(cal_pred != y[cal_ids]))
    return (model, reference_loss, train_acc, cal_acc)

def generate_delay(y, seed, lambda0, lambda1):
    rng = np.random.default_rng(seed)
    sample_ids = np.arange(MONITOR_START, len(y))
    delays = np.zeros(len(sample_ids), dtype=int)
    for index, sample_id in enumerate(sample_ids):
        label = int(y[sample_id])
        lam = lambda0 if label == 0 else lambda1
        delays[index] = int(rng.poisson(lam))
    arrivals = sample_ids + delays
    arrival_map = {int(sample_id): int(arrival) for sample_id, arrival in zip(sample_ids, arrivals)}
    arrival_groups = {}
    for sample_id, arrival in arrival_map.items():
        arrival_groups.setdefault(arrival, [])
        arrival_groups[arrival].append(sample_id)
    return (arrival_map, arrival_groups)

def get_all_available_ids(current_time, arrival_map, y):
    available = list(range(MONITOR_START))
    upper = min(current_time, len(y))
    for sample_id in range(MONITOR_START, upper):
        if arrival_map[sample_id] < current_time:
            available.append(sample_id)
    return available

def get_returned_after_anchor(current_time, alarm_anchor, arrival_map, y):
    returned = []
    upper = min(current_time, len(y))
    for sample_id in range(alarm_anchor, upper):
        if sample_id not in arrival_map:
            continue
        if arrival_map[sample_id] >= current_time:
            continue
        returned.append(sample_id)
    return returned

def soft_coverage_from_list(available, y):
    if len(available) < UPDATE_TRAIN_SIZE:
        return None
    class0 = [sample_id for sample_id in available if y[sample_id] == 0]
    class1 = [sample_id for sample_id in available if y[sample_id] == 1]
    if len(class0) < MIN_CLASS_COVERAGE or len(class1) < MIN_CLASS_COVERAGE:
        return None
    selected0 = class0[-MIN_CLASS_COVERAGE:]
    selected1 = class1[-MIN_CLASS_COVERAGE:]
    selected_set = set(selected0 + selected1)
    remaining_needed = UPDATE_TRAIN_SIZE - 2 * MIN_CLASS_COVERAGE
    candidates = [sample_id for sample_id in reversed(available) if sample_id not in selected_set]
    if len(candidates) < remaining_needed:
        return None
    extra = candidates[:remaining_needed]
    train_ids = np.asarray(sorted(selected0 + selected1 + extra), dtype=int)
    if len(train_ids) != UPDATE_TRAIN_SIZE:
        raise RuntimeError('训练集数量错误。')
    return train_ids

def get_selective_training_ids(current_time, alarm_anchor, arrival_map, y):
    returned = get_returned_after_anchor(current_time, alarm_anchor, arrival_map, y)
    return soft_coverage_from_list(returned, y)

def get_periodic_training_ids(current_time, arrival_map, y):
    available = get_all_available_ids(current_time, arrival_map, y)
    return soft_coverage_from_list(available, y)

def evaluate_bounds(state, block):
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
    ref = float(block['reference_loss'])
    lower_delta = lower - ref
    upper_delta = upper - ref
    if lower_delta > EPSILON + FLOAT_TOL:
        decision = 'DEGRADATION'
    elif upper_delta <= EPSILON + FLOAT_TOL:
        decision = 'TOLERABLE'
    else:
        return None
    return {'decision': decision, 'm': m, 'u': u}

def evaluate_arrived_only(state, block):
    ids = block['sample_ids']
    arrived = [sample_id for sample_id in ids if sample_id in state['feedback']]
    m = len(arrived)
    if m < MIN_ARRIVED:
        return None
    loss = float(np.mean([state['feedback'][sample_id] for sample_id in arrived]))
    decision = 'DEGRADATION' if loss - block['reference_loss'] > EPSILON + FLOAT_TOL else 'TOLERABLE'
    return {'decision': decision, 'm': m, 'u': len(ids) - m}

def evaluate_wait_all(state, block):
    ids = block['sample_ids']
    if not all((sample_id in state['feedback'] for sample_id in ids)):
        return None
    loss = float(np.mean([state['feedback'][sample_id] for sample_id in ids]))
    decision = 'DEGRADATION' if loss - block['reference_loss'] > EPSILON + FLOAT_TOL else 'TOLERABLE'
    return {'decision': decision, 'm': len(ids), 'u': 0}

def calculate_oracle(y, state, block):
    losses = [int(state['predictions'][sample_id] != y[sample_id]) for sample_id in block['sample_ids']]
    final_loss = float(np.mean(losses))
    final_delta = final_loss - block['reference_loss']
    oracle = 'DEGRADATION' if final_delta > EPSILON + FLOAT_TOL else 'TOLERABLE'
    return (oracle, final_loss, final_delta)

def create_state(X, y, learner):
    model, reference, train_acc, cal_acc = create_initial_model(X, y, learner)
    return {'model': model, 'model_version': 0, 'reference_loss': reference, 'predictions': {}, 'feedback': {}, 'phase': 'MONITORING', 'open_block': [], 'closed_blocks': [], 'next_block_id': 0, 'alarm_anchor': None, 'alarm_time': None, 'calibration_ids': [], 'retrain_count': 0, 'decision_records': [], 'update_records': [], 'monitor_samples': 0, 'waiting_samples': 0, 'calibration_samples': 0, 'train_acc': train_acc, 'cal_acc': cal_acc, 'initial_reference': reference}

def run_one(dataset, X, y, learner, scenario, lambda0, lambda1, delay_seed):
    start_clock = time.perf_counter()
    arrival_map, arrival_groups = generate_delay(y, delay_seed, lambda0, lambda1)
    last_event = len(y) - 1
    simulation_end = max(last_event, max(arrival_map.values()))
    states = {strategy: create_state(X, y, learner) for strategy in STRATEGIES}
    for current_time in range(MONITOR_START, simulation_end + 1):
        for strategy in SELECTIVE_STRATEGIES:
            state = states[strategy]
            if state['phase'] != 'WAITING_READY':
                continue
            if current_time > last_event:
                continue
            train_ids = get_selective_training_ids(current_time, state['alarm_anchor'], arrival_map, y)
            if train_ids is None:
                continue
            new_model = create_model(learner)
            new_model.fit(X[train_ids], y[train_ids])
            count1 = int(np.sum(y[train_ids] == 1))
            old_version = int(state['model_version'])
            old_reference = float(state['reference_loss'])
            state['model'] = new_model
            state['model_version'] += 1
            state['retrain_count'] += 1
            state['reference_loss'] = np.nan
            state['phase'] = 'CALIBRATING'
            state['calibration_ids'] = []
            state['open_block'] = []
            state['update_records'].append({'update_time': current_time, 'alarm_time': state['alarm_time'], 'alarm_to_update': current_time - state['alarm_time'], 'old_version': old_version, 'new_version': state['model_version'], 'old_reference': old_reference, 'train_size': len(train_ids), 'train_class1_rate': count1 / len(train_ids)})
            state['alarm_anchor'] = None
            state['alarm_time'] = None
        if current_time <= last_event:
            for strategy, interval in PERIODIC_INTERVALS.items():
                if current_time <= MONITOR_START:
                    continue
                if (current_time - MONITOR_START) % interval != 0:
                    continue
                state = states[strategy]
                train_ids = get_periodic_training_ids(current_time, arrival_map, y)
                if train_ids is None:
                    continue
                model = create_model(learner)
                model.fit(X[train_ids], y[train_ids])
                count1 = int(np.sum(y[train_ids] == 1))
                state['model'] = model
                state['model_version'] += 1
                state['retrain_count'] += 1
                state['update_records'].append({'update_time': current_time, 'alarm_time': np.nan, 'alarm_to_update': np.nan, 'old_version': state['model_version'] - 1, 'new_version': state['model_version'], 'old_reference': np.nan, 'train_size': len(train_ids), 'train_class1_rate': count1 / len(train_ids)})
        if current_time <= last_event:
            sample_id = int(current_time)
            for strategy in STRATEGIES:
                state = states[strategy]
                prediction = int(state['model'].predict(X[sample_id].reshape(1, -1))[0])
                state['predictions'][sample_id] = prediction
                if strategy not in SELECTIVE_STRATEGIES:
                    continue
                if state['phase'] == 'MONITORING':
                    state['monitor_samples'] += 1
                    state['open_block'].append(sample_id)
                    if len(state['open_block']) == WINDOW_SIZE:
                        ids = list(state['open_block'])
                        state['closed_blocks'].append({'block_id': state['next_block_id'], 'model_version': state['model_version'], 'reference_loss': float(state['reference_loss']), 'sample_ids': ids, 'start': ids[0], 'close_time': ids[-1], 'status': 'PENDING'})
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
                    raise RuntimeError(f'prediction missing {sample_id}')
                state['feedback'][sample_id] = int(state['predictions'][sample_id] != y[sample_id])
        for strategy in SELECTIVE_STRATEGIES:
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
        for strategy in SELECTIVE_STRATEGIES:
            state = states[strategy]
            if state['phase'] != 'MONITORING':
                continue
            for block in state['closed_blocks']:
                if block['status'] != 'PENDING':
                    continue
                if strategy == 'FULL':
                    result = evaluate_bounds(state, block)
                elif strategy == 'ARRIVED_ONLY':
                    result = evaluate_arrived_only(state, block)
                else:
                    result = evaluate_wait_all(state, block)
                if result is None:
                    continue
                block['status'] = 'DONE'
                oracle, final_loss, final_delta = calculate_oracle(y, state, block)
                decision = result['decision']
                correct = int(decision == oracle)
                state['decision_records'].append({'block_id': block['block_id'], 'decision': decision, 'oracle': oracle, 'correct': correct, 'decision_time': current_time, 'decision_delay': current_time - block['close_time'], 'labels_at_decision': result['m'], 'unknown_at_decision': result['u'], 'reference_loss': block['reference_loss'], 'final_loss': final_loss, 'final_delta': final_delta})
                if decision == 'DEGRADATION':
                    state['phase'] = 'WAITING_READY'
                    state['alarm_anchor'] = int(block['start'])
                    state['alarm_time'] = int(current_time)
                    state['open_block'] = []
                    for other in state['closed_blocks']:
                        if other['status'] == 'PENDING':
                            other['status'] = 'STALE'
                    break
    ids = np.arange(MONITOR_START, len(y))
    y_true = y[MONITOR_START:]
    run_records = []
    decision_records = []
    update_records = []
    for strategy in STRATEGIES:
        state = states[strategy]
        pred = np.asarray([state['predictions'][int(sample_id)] for sample_id in ids], dtype=int)
        accuracy = float(accuracy_score(y_true, pred))
        macro_f1 = float(f1_score(y_true, pred, average='macro'))
        mcc = float(matthews_corrcoef(y_true, pred))
        decisions = state['decision_records']
        if decisions:
            decision_accuracy = float(np.mean([x['correct'] for x in decisions]))
            decision_delay = float(np.mean([x['decision_delay'] for x in decisions]))
            fp = int(sum((x['decision'] == 'DEGRADATION' and x['oracle'] == 'TOLERABLE' for x in decisions)))
            fn = int(sum((x['decision'] == 'TOLERABLE' and x['oracle'] == 'DEGRADATION' for x in decisions)))
        else:
            decision_accuracy = np.nan
            decision_delay = np.nan
            fp = 0
            fn = 0
        if strategy in SELECTIVE_STRATEGIES:
            monitor_coverage = state['monitor_samples'] / len(ids)
        else:
            monitor_coverage = np.nan
        run_records.append({'dataset': dataset, 'learner': learner, 'scenario': scenario, 'delay_seed': delay_seed, 'strategy': strategy, 'accuracy': accuracy, 'macro_f1': macro_f1, 'mcc': mcc, 'retrain_count': state['retrain_count'], 'decision_count': len(decisions), 'decision_accuracy': decision_accuracy, 'decision_delay': decision_delay, 'false_positive': fp, 'false_negative': fn, 'monitor_coverage': monitor_coverage, 'initial_train_accuracy': state['train_acc'], 'initial_cal_accuracy': state['cal_acc'], 'initial_reference_loss': state['initial_reference'], 'elapsed_seconds': time.perf_counter() - start_clock})
        for record in decisions:
            decision_records.append({'dataset': dataset, 'learner': learner, 'scenario': scenario, 'delay_seed': delay_seed, 'strategy': strategy, **record})
        for record in state['update_records']:
            update_records.append({'dataset': dataset, 'learner': learner, 'scenario': scenario, 'delay_seed': delay_seed, 'strategy': strategy, **record})
    return (run_records, decision_records, update_records)

def run_task(dataset, X, y, learner, scenario, lambda0, lambda1, seed):
    return run_one(dataset, X, y, learner, scenario, lambda0, lambda1, seed)

def main():
    total_start = time.perf_counter()
    print('=' * 150)
    print('NATURAL CHRONOLOGICAL REAL-WORLD VALIDATION')
    print('=' * 150)
    print('固定样本数：', MAX_SAMPLES)
    print('Window：', WINDOW_SIZE)
    print('Epsilon：', EPSILON)
    print('q：', MIN_CLASS_COVERAGE)
    print('Delay Seeds：', DELAY_SEEDS)
    datasets = {}
    feasibility_records = []
    for dataset, config in RAW_DATASETS.items():
        X, y, feasibility = load_natural_dataset(dataset, config)
        datasets[dataset] = (X, y)
        feasibility_records.append(feasibility)
    feasibility_df = pd.DataFrame(feasibility_records)
    tasks = []
    for dataset, (X, y) in datasets.items():
        for learner in LEARNERS:
            for scenario, config in DELAY_SCENARIOS.items():
                for seed in DELAY_SEEDS:
                    tasks.append((dataset, X, y, learner, scenario, config[0], config[1], seed))
    expected = len(tasks) * len(STRATEGIES)
    print()
    print('=' * 150)
    print('开始20C')
    print('=' * 150)
    print('独立任务：', len(tasks))
    print('理论结果：', expected)
    results = Parallel(n_jobs=CPU_JOBS, backend='loky', verbose=10, batch_size=1, pre_dispatch='2*n_jobs')((delayed(run_task)(*task) for task in tasks))
    all_runs = []
    all_decisions = []
    all_updates = []
    for run_records, decision_records, update_records in results:
        all_runs.extend(run_records)
        all_decisions.extend(decision_records)
        all_updates.extend(update_records)
    raw_df = pd.DataFrame(all_runs)
    decision_df = pd.DataFrame(all_decisions)
    update_df = pd.DataFrame(all_updates)
    print()
    print('实际结果：', len(raw_df))
    if len(raw_df) != expected:
        raise RuntimeError('20C结果不完整。')
    overall_df = raw_df.groupby('strategy').agg(runs=('accuracy', 'count'), accuracy_mean=('accuracy', 'mean'), accuracy_std=('accuracy', 'std'), macro_f1_mean=('macro_f1', 'mean'), mcc_mean=('mcc', 'mean'), retrain_mean=('retrain_count', 'mean'), decision_accuracy=('decision_accuracy', 'mean'), decision_delay=('decision_delay', 'mean'), false_positive=('false_positive', 'sum'), false_negative=('false_negative', 'sum'), monitor_coverage=('monitor_coverage', 'mean')).reset_index()
    dataset_df = raw_df.groupby(['dataset', 'strategy']).agg(runs=('accuracy', 'count'), accuracy_mean=('accuracy', 'mean'), macro_f1_mean=('macro_f1', 'mean'), mcc_mean=('mcc', 'mean'), retrain_mean=('retrain_count', 'mean'), decision_accuracy=('decision_accuracy', 'mean'), decision_delay=('decision_delay', 'mean'), false_positive=('false_positive', 'sum'), false_negative=('false_negative', 'sum')).reset_index()
    dataset_learner_df = raw_df.groupby(['dataset', 'learner', 'strategy']).agg(runs=('accuracy', 'count'), accuracy_mean=('accuracy', 'mean'), macro_f1_mean=('macro_f1', 'mean'), mcc_mean=('mcc', 'mean'), retrain_mean=('retrain_count', 'mean'), decision_accuracy=('decision_accuracy', 'mean'), decision_delay=('decision_delay', 'mean'), false_positive=('false_positive', 'sum'), false_negative=('false_negative', 'sum')).reset_index()
    scenario_df = raw_df.groupby(['scenario', 'strategy']).agg(runs=('accuracy', 'count'), accuracy_mean=('accuracy', 'mean'), macro_f1_mean=('macro_f1', 'mean'), mcc_mean=('mcc', 'mean'), retrain_mean=('retrain_count', 'mean'), decision_accuracy=('decision_accuracy', 'mean'), decision_delay=('decision_delay', 'mean'), false_positive=('false_positive', 'sum'), false_negative=('false_negative', 'sum')).reset_index()
    full = overall_df[overall_df['strategy'] == 'FULL'].iloc[0]
    comparison_records = []
    for baseline_name in ['STATIC', 'PERIODIC_100', 'PERIODIC_300', 'PERIODIC_600', 'ARRIVED_ONLY', 'WAIT_ALL']:
        baseline = overall_df[overall_df['strategy'] == baseline_name].iloc[0]
        comparison_records.append({'baseline': baseline_name, 'full_accuracy': full['accuracy_mean'], 'baseline_accuracy': baseline['accuracy_mean'], 'accuracy_difference': full['accuracy_mean'] - baseline['accuracy_mean'], 'full_mcc': full['mcc_mean'], 'baseline_mcc': baseline['mcc_mean'], 'mcc_difference': full['mcc_mean'] - baseline['mcc_mean'], 'full_retrain': full['retrain_mean'], 'baseline_retrain': baseline['retrain_mean']})
    comparison_df = pd.DataFrame(comparison_records)
    safety_records = []
    for strategy in SELECTIVE_STRATEGIES:
        subset = raw_df[raw_df['strategy'] == strategy]
        valid = subset[subset['decision_accuracy'].notna()]
        nonperfect = int((valid['decision_accuracy'] < 1.0 - FLOAT_TOL).sum())
        safety_records.append({'strategy': strategy, 'runs': len(subset), 'nonperfect_runs': nonperfect, 'false_positive': int(subset['false_positive'].sum()), 'false_negative': int(subset['false_negative'].sum()), 'decision_accuracy': float(subset['decision_accuracy'].mean()), 'decision_delay': float(subset['decision_delay'].mean())})
    safety_df = pd.DataFrame(safety_records)
    wait = overall_df[overall_df['strategy'] == 'WAIT_ALL'].iloc[0]
    wait_df = pd.DataFrame([{'full_accuracy': full['accuracy_mean'], 'wait_accuracy': wait['accuracy_mean'], 'accuracy_difference': full['accuracy_mean'] - wait['accuracy_mean'], 'full_delay': full['decision_delay'], 'wait_delay': wait['decision_delay'], 'time_saved': wait['decision_delay'] - full['decision_delay'], 'full_retrain': full['retrain_mean'], 'wait_retrain': wait['retrain_mean']}])
    print()
    print('=' * 150)
    print('OVERALL')
    print('=' * 150)
    print(overall_df.to_string(index=False))
    print()
    print('=' * 150)
    print('DATASET × STRATEGY')
    print('=' * 150)
    print(dataset_df.to_string(index=False))
    print()
    print('=' * 150)
    print('FULL COMPARISONS')
    print('=' * 150)
    print(comparison_df.to_string(index=False))
    print()
    print('=' * 150)
    print('SAFETY')
    print('=' * 150)
    print(safety_df.to_string(index=False))
    print()
    print('=' * 150)
    print('FULL vs WAIT-ALL')
    print('=' * 150)
    print(wait_df.to_string(index=False))
    paths = {'feasibility': os.path.join(OUTPUT_DIR, 'natural_stream_feasibility.csv'), 'raw': os.path.join(OUTPUT_DIR, 'raw.csv'), 'decision': os.path.join(OUTPUT_DIR, 'decision_raw.csv'), 'update': os.path.join(OUTPUT_DIR, 'update_raw.csv'), 'overall': os.path.join(OUTPUT_DIR, 'overall.csv'), 'dataset': os.path.join(OUTPUT_DIR, 'dataset_summary.csv'), 'dataset_learner': os.path.join(OUTPUT_DIR, 'dataset_learner_summary.csv'), 'scenario': os.path.join(OUTPUT_DIR, 'scenario_summary.csv'), 'comparison': os.path.join(OUTPUT_DIR, 'full_comparisons.csv'), 'safety': os.path.join(OUTPUT_DIR, 'safety.csv'), 'wait': os.path.join(OUTPUT_DIR, 'full_vs_waitall.csv')}
    feasibility_df.to_csv(paths['feasibility'], index=False, encoding='utf-8-sig')
    raw_df.to_csv(paths['raw'], index=False, encoding='utf-8-sig')
    decision_df.to_csv(paths['decision'], index=False, encoding='utf-8-sig')
    update_df.to_csv(paths['update'], index=False, encoding='utf-8-sig')
    overall_df.to_csv(paths['overall'], index=False, encoding='utf-8-sig')
    dataset_df.to_csv(paths['dataset'], index=False, encoding='utf-8-sig')
    dataset_learner_df.to_csv(paths['dataset_learner'], index=False, encoding='utf-8-sig')
    scenario_df.to_csv(paths['scenario'], index=False, encoding='utf-8-sig')
    comparison_df.to_csv(paths['comparison'], index=False, encoding='utf-8-sig')
    safety_df.to_csv(paths['safety'], index=False, encoding='utf-8-sig')
    wait_df.to_csv(paths['wait'], index=False, encoding='utf-8-sig')
    elapsed = time.perf_counter() - total_start
    print()
    print('=' * 150)
    print('输出文件')
    print('=' * 150)
    for path in paths.values():
        print(path)
    print()
    print(f'运行时间：{elapsed:.2f} 秒')
    print(f'约：{elapsed / 60:.2f} 分钟')
    full_safety = safety_df[safety_df['strategy'] == 'FULL'].iloc[0]
    print()
    print('=' * 150)
    if int(full_safety['nonperfect_runs']) == 0 and int(full_safety['false_positive']) == 0 and (int(full_safety['false_negative']) == 0):
        print('NATURAL REAL STREAM：FULL安全检查通过 ')
    else:
        print('NATURAL REAL STREAM：FULL存在异常 ')
    print()
    print('重要：20C使用原始时间顺序，不使用人工漂移生成。')
    print('禁止根据20C结果重新修改：W / epsilon / q / Update Size / Calibration Size / 5000样本长度。')
    print('=' * 150)
if __name__ == '__main__':
    main()
