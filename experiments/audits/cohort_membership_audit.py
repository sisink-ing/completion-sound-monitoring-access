"""Evaluate training membership under fixed-trigger feedback permutations."""
from completion_monitoring.paths import workspace_root, external_data_root, source_file
import os
import hashlib
import importlib.util
from itertools import combinations
import numpy as np
import pandas as pd
SCRIPT_NAME = 'cohort_membership_audit'
OUTPUT_DIR = os.path.join('result', SCRIPT_NAME)
os.makedirs(OUTPUT_DIR, exist_ok=True)
STEP24_DECISION_FILE = os.path.join('result', 'confirmation_comparison', 'decision_raw.csv')
WINDOW_SIZE = 100
SOFT_Q = 30
SOFT_TRAIN_SIZE = 100
MONITOR_START = 400
CASES_PER_SCENARIO = 20
N_PERMUTATIONS = 100
MASTER_SEED = 26026
SMOKE_TEST = os.environ.get('RETURN_ORDER_SMOKE', '0') == '1'
if SMOKE_TEST:
    CASES_PER_SCENARIO = 2
    N_PERMUTATIONS = 10

def stable_seed(*items):
    text = '|'.join((str(x) for x in items))
    digest = hashlib.sha256(text.encode('utf-8')).hexdigest()
    return int(digest[:16], 16) % 2 ** 32

def membership_hash(ids):
    ids = np.asarray(ids, dtype=np.int64)
    h = hashlib.sha256()
    h.update(ids.tobytes())
    return h.hexdigest()

def ids_to_string(ids):
    return ';'.join((str(int(x)) for x in ids))

def jaccard_similarity(a, b):
    a = set(a)
    b = set(b)
    union = a | b
    if len(union) == 0:
        return np.nan
    return len(a & b) / len(union)

def mean_pairwise_jaccard(memberships):
    if len(memberships) < 2:
        return np.nan
    values = []
    for a, b in combinations(memberships, 2):
        values.append(jaccard_similarity(a, b))
    return float(np.mean(values))

def median_pairwise_jaccard(memberships):
    if len(memberships) < 2:
        return np.nan
    values = []
    for a, b in combinations(memberships, 2):
        values.append(jaccard_similarity(a, b))
    return float(np.median(values))
STREAM_CACHE = {}
STEP24_MODULE = None

def load_step24_module():
    global STEP24_MODULE
    if STEP24_MODULE is not None:
        return STEP24_MODULE
    step24_path = str(source_file('confirmation_comparison.py'))
    if not os.path.exists(step24_path):
        raise FileNotFoundError('找不到 confirmation_comparison.py。\n请确认当前目录是 the project directory。')
    spec = importlib.util.spec_from_file_location('step24_for_step26', step24_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    STEP24_MODULE = module
    return STEP24_MODULE

def load_stream(dataset, stream_seed):
    key = (dataset, int(stream_seed))
    if key in STREAM_CACHE:
        return STREAM_CACHE[key]
    filename = f'{dataset}_streamseed_{int(stream_seed)}.npz'
    candidate_paths = [os.path.join('result', 'unseen_stream_pipeline', 'streams', filename), os.path.join('result', 'unseen_stream_holdout', 'streams', filename)]
    for path in candidate_paths:
        if not os.path.exists(path):
            continue
        data = np.load(path, allow_pickle=True)
        X = np.asarray(data['X'], dtype=float)
        y = np.asarray(data['y'], dtype=int)
        result = {'X': X, 'y': y, 'source': path}
        STREAM_CACHE[key] = result
        return result
    module = load_step24_module()
    result24 = module.load_or_generate_stream(dataset, int(stream_seed))
    if not result24['ok']:
        raise RuntimeError(f'无法加载/生成流：{dataset}, seed={stream_seed}')
    result = {'X': np.asarray(result24['X'], dtype=float), 'y': np.asarray(result24['y'], dtype=int), 'source': result24.get('source', 'STEP24_LOADER')}
    STREAM_CACHE[key] = result
    return result

def generate_arrival_map(y, delay_seed, lambda0, lambda1):
    rng = np.random.default_rng(int(delay_seed))
    sample_ids = np.arange(MONITOR_START, len(y), dtype=int)
    delays = np.zeros(len(sample_ids), dtype=int)
    for index, sample_id in enumerate(sample_ids):
        label = int(y[sample_id])
        lam = int(lambda0) if label == 0 else int(lambda1)
        delays[index] = int(rng.poisson(lam))
    arrivals = sample_ids + delays
    return {int(sample_id): int(arrival) for sample_id, arrival in zip(sample_ids, arrivals)}

def soft_q30_from_schedule(candidate_ids, arrival_times, y):
    candidate_ids = np.asarray(candidate_ids, dtype=int)
    arrival_times = np.asarray(arrival_times, dtype=int)
    if len(candidate_ids) < SOFT_TRAIN_SIZE:
        return (None, None)
    unique_arrivals = np.unique(arrival_times)
    unique_arrivals.sort()
    for arrival_time in unique_arrivals:
        available_mask = arrival_times <= arrival_time
        available = candidate_ids[available_mask]
        available = np.sort(available)
        if len(available) < SOFT_TRAIN_SIZE:
            continue
        class0 = [int(sample_id) for sample_id in available if int(y[sample_id]) == 0]
        class1 = [int(sample_id) for sample_id in available if int(y[sample_id]) == 1]
        if len(class0) < SOFT_Q:
            continue
        if len(class1) < SOFT_Q:
            continue
        selected0 = class0[-SOFT_Q:]
        selected1 = class1[-SOFT_Q:]
        selected_set = set(selected0 + selected1)
        remaining_needed = SOFT_TRAIN_SIZE - 2 * SOFT_Q
        remaining_candidates = [int(sample_id) for sample_id in reversed(available.tolist()) if int(sample_id) not in selected_set]
        if len(remaining_candidates) < remaining_needed:
            continue
        additional = remaining_candidates[:remaining_needed]
        train_ids = np.asarray(sorted(selected0 + selected1 + additional), dtype=int)
        if len(train_ids) != SOFT_TRAIN_SIZE:
            raise RuntimeError('Soft-q30 training size != 100.')
        if len(np.unique(train_ids)) != SOFT_TRAIN_SIZE:
            raise RuntimeError('Soft-q30 contains duplicated sample IDs.')
        readiness_time = int(arrival_time + 1)
        return (train_ids, readiness_time)
    return (None, None)

def perturb_schedule(y, original_arrival_map, wk_ids, future_ids, alarm_time, permutation_index, case_seed):
    rng = np.random.default_rng(stable_seed(MASTER_SEED, case_seed, permutation_index))
    wk_ids = np.asarray(wk_ids, dtype=int)
    wk_arrivals = np.asarray([original_arrival_map[int(sample_id)] for sample_id in wk_ids], dtype=int)
    perturbed_wk_arrivals = wk_arrivals.copy()
    if permutation_index > 0:
        unresolved_mask = wk_arrivals > alarm_time
        for label in [0, 1]:
            class_mask = unresolved_mask & (y[wk_ids] == label)
            indices = np.where(class_mask)[0]
            if len(indices) <= 1:
                continue
            residual_waits = (wk_arrivals[indices] - alarm_time).copy()
            rng.shuffle(residual_waits)
            perturbed_wk_arrivals[indices] = alarm_time + residual_waits
    future_ids = np.asarray(future_ids, dtype=int)
    future_delays = np.asarray([original_arrival_map[int(sample_id)] - int(sample_id) for sample_id in future_ids], dtype=int)
    perturbed_future_delays = future_delays.copy()
    if permutation_index > 0:
        for label in [0, 1]:
            indices = np.where(y[future_ids] == label)[0]
            if len(indices) <= 1:
                continue
            values = future_delays[indices].copy()
            rng.shuffle(values)
            perturbed_future_delays[indices] = values
    perturbed_future_arrivals = future_ids + perturbed_future_delays
    if np.any(perturbed_future_arrivals < future_ids):
        raise RuntimeError('Illegal future schedule: arrival < event.')
    original_unresolved = wk_arrivals > alarm_time
    if np.any(perturbed_wk_arrivals[original_unresolved] <= alarm_time):
        raise RuntimeError('Perturbation moved unresolved W_k label to before the fixed alarm.')
    return (perturbed_wk_arrivals, perturbed_future_arrivals)

def build_case(row):
    dataset = str(row['dataset'])
    stream_seed = int(row['stream_seed'])
    stream = load_stream(dataset, stream_seed)
    y = stream['y']
    block_close = int(row['block_close_time'])
    block_start = block_close - WINDOW_SIZE + 1
    alarm_time = int(row['decision_time'])
    if block_start < MONITOR_START:
        return None
    if block_close >= len(y):
        return None
    wk_ids = np.arange(block_start, block_close + 1, dtype=int)
    if len(wk_ids) != WINDOW_SIZE:
        return None
    future_start = alarm_time + 1
    if future_start >= len(y):
        return None
    future_ids = np.arange(future_start, len(y), dtype=int)
    if len(future_ids) < SOFT_TRAIN_SIZE:
        return None
    future_y = y[future_ids]
    future_count0 = int(np.sum(future_y == 0))
    future_count1 = int(np.sum(future_y == 1))
    if future_count0 < SOFT_Q:
        return None
    if future_count1 < SOFT_Q:
        return None
    lambda0 = int(row['lambda0'])
    lambda1 = int(row['lambda1'])
    delay_seed = int(row['delay_seed'])
    arrival_map = generate_arrival_map(y, delay_seed, lambda0, lambda1)
    reconstructed_labels_at_alarm = int(np.sum([arrival_map[int(sample_id)] <= alarm_time for sample_id in wk_ids]))
    logged_labels = int(row['labels_at_decision'])
    if reconstructed_labels_at_alarm != logged_labels:
        raise RuntimeError(f"\ncohort-membership audit reconstruction check FAILED.\nDataset={dataset}\nstream_seed={stream_seed}\ndelay_seed={delay_seed}\nblock_id={row['block_id']}\nreconstructed labels={reconstructed_labels_at_alarm}\nconfirmation comparison logged labels={logged_labels}\n\n这说明不能安全地用 close_time-99 重建 W_k，请不要继续解释结果。")
    case_key = f"{dataset}|{stream_seed}|{row['learner']}|{row['scenario']}|{delay_seed}|{int(row['block_id'])}|{alarm_time}"
    case_seed = stable_seed(case_key)
    return {'case_id': case_key, 'case_seed': case_seed, 'dataset': dataset, 'stream_seed': stream_seed, 'learner': str(row['learner']), 'scenario': str(row['scenario']), 'lambda0': lambda0, 'lambda1': lambda1, 'delay_seed': delay_seed, 'block_id': int(row['block_id']), 'block_start': block_start, 'block_close': block_close, 'alarm_time': alarm_time, 'labels_at_alarm': reconstructed_labels_at_alarm, 'unknown_at_alarm': WINDOW_SIZE - reconstructed_labels_at_alarm, 'wk_ids': wk_ids, 'future_ids': future_ids, 'arrival_map': arrival_map, 'y': y, 'stream_source': stream['source']}

def select_cases():
    if not os.path.exists(STEP24_DECISION_FILE):
        raise FileNotFoundError(f'找不到：\n{STEP24_DECISION_FILE}\n\n请先确认 confirmation comparison 已完成。')
    decision_df = pd.read_csv(STEP24_DECISION_FILE)
    required_columns = ['dataset', 'stream_seed', 'learner', 'scenario', 'lambda0', 'lambda1', 'delay_seed', 'strategy', 'block_id', 'decision', 'correct', 'decision_time', 'block_close_time', 'labels_at_decision']
    missing = [column for column in required_columns if column not in decision_df.columns]
    if missing:
        raise RuntimeError('confirmation comparison decision CSV 缺少字段：' + str(missing))
    candidates = decision_df[(decision_df['strategy'] == 'FULL_COMPLETE_COHORT') & (decision_df['decision'] == 'DEGRADATION') & (decision_df['correct'] == 1)].copy()
    if len(candidates) == 0:
        raise RuntimeError('confirmation comparison 中没有找到 FULL_COMPLETE_COHORT 的正确 DEGRADATION decisions。')
    task_columns = ['dataset', 'stream_seed', 'learner', 'scenario', 'delay_seed']
    candidates = candidates.sort_values(task_columns + ['decision_time']).drop_duplicates(subset=task_columns, keep='first').reset_index(drop=True)
    selected_cases = []
    rng = np.random.default_rng(MASTER_SEED)
    scenarios = sorted(candidates['scenario'].unique())
    print()
    print('Candidate first-degradation cases:', len(candidates))
    print('Scenarios:', scenarios)
    print()
    for scenario in scenarios:
        scenario_df = candidates[candidates['scenario'] == scenario].copy()
        indices = np.arange(len(scenario_df))
        rng.shuffle(indices)
        accepted = 0
        for index in indices:
            row = scenario_df.iloc[int(index)]
            case = build_case(row)
            if case is None:
                continue
            selected_cases.append(case)
            accepted += 1
            if accepted >= CASES_PER_SCENARIO:
                break
        print(f'{scenario}: selected {accepted} valid cases')
    if len(selected_cases) == 0:
        raise RuntimeError('没有可用于 cohort-membership audit 的有效 alarm case。')
    return selected_cases

def run_case(case):
    y = case['y']
    wk_ids = case['wk_ids']
    future_ids = case['future_ids']
    alarm_time = case['alarm_time']
    original_arrival_map = case['arrival_map']
    case_seed = case['case_seed']
    permutation_records = []
    method_memberships = {'COMPLETE_COHORT': [], 'SOFT_Q30': []}
    method_class1_rates = {'COMPLETE_COHORT': [], 'SOFT_Q30': []}
    method_readiness_delays = {'COMPLETE_COHORT': [], 'SOFT_Q30': []}
    method_valid = {'COMPLETE_COHORT': 0, 'SOFT_Q30': 0}
    complete_membership = tuple((int(x) for x in wk_ids))
    complete_class1_rate = float(np.mean(y[wk_ids] == 1))
    for permutation_index in range(N_PERMUTATIONS):
        wk_arrivals, future_arrivals = perturb_schedule(y=y, original_arrival_map=original_arrival_map, wk_ids=wk_ids, future_ids=future_ids, alarm_time=alarm_time, permutation_index=permutation_index, case_seed=case_seed)
        complete_ready_time = max(alarm_time, int(np.max(wk_arrivals))) + 1
        complete_readiness_delay = complete_ready_time - alarm_time
        method_memberships['COMPLETE_COHORT'].append(complete_membership)
        method_class1_rates['COMPLETE_COHORT'].append(complete_class1_rate)
        method_readiness_delays['COMPLETE_COHORT'].append(complete_readiness_delay)
        method_valid['COMPLETE_COHORT'] += 1
        permutation_records.append({'case_id': case['case_id'], 'dataset': case['dataset'], 'stream_seed': case['stream_seed'], 'learner': case['learner'], 'scenario': case['scenario'], 'lambda0': case['lambda0'], 'lambda1': case['lambda1'], 'delay_seed': case['delay_seed'], 'block_id': case['block_id'], 'alarm_time': alarm_time, 'permutation': permutation_index, 'method': 'COMPLETE_COHORT', 'valid': 1, 'training_size': WINDOW_SIZE, 'training_class1_rate': complete_class1_rate, 'readiness_time': complete_ready_time, 'readiness_delay': complete_readiness_delay, 'membership_hash': membership_hash(wk_ids), 'training_ids': ids_to_string(wk_ids)})
        soft_ids, soft_ready_time = soft_q30_from_schedule(future_ids, future_arrivals, y)
        if soft_ids is None:
            permutation_records.append({'case_id': case['case_id'], 'dataset': case['dataset'], 'stream_seed': case['stream_seed'], 'learner': case['learner'], 'scenario': case['scenario'], 'lambda0': case['lambda0'], 'lambda1': case['lambda1'], 'delay_seed': case['delay_seed'], 'block_id': case['block_id'], 'alarm_time': alarm_time, 'permutation': permutation_index, 'method': 'SOFT_Q30', 'valid': 0, 'training_size': 0, 'training_class1_rate': np.nan, 'readiness_time': np.nan, 'readiness_delay': np.nan, 'membership_hash': '', 'training_ids': ''})
            continue
        soft_membership = tuple((int(x) for x in soft_ids))
        soft_class1_rate = float(np.mean(y[soft_ids] == 1))
        soft_readiness_delay = int(soft_ready_time) - alarm_time
        method_memberships['SOFT_Q30'].append(soft_membership)
        method_class1_rates['SOFT_Q30'].append(soft_class1_rate)
        method_readiness_delays['SOFT_Q30'].append(soft_readiness_delay)
        method_valid['SOFT_Q30'] += 1
        permutation_records.append({'case_id': case['case_id'], 'dataset': case['dataset'], 'stream_seed': case['stream_seed'], 'learner': case['learner'], 'scenario': case['scenario'], 'lambda0': case['lambda0'], 'lambda1': case['lambda1'], 'delay_seed': case['delay_seed'], 'block_id': case['block_id'], 'alarm_time': alarm_time, 'permutation': permutation_index, 'method': 'SOFT_Q30', 'valid': 1, 'training_size': len(soft_ids), 'training_class1_rate': soft_class1_rate, 'readiness_time': int(soft_ready_time), 'readiness_delay': soft_readiness_delay, 'membership_hash': membership_hash(soft_ids), 'training_ids': ids_to_string(soft_ids)})
    case_summary_records = []
    for method in ['COMPLETE_COHORT', 'SOFT_Q30']:
        memberships = method_memberships[method]
        class1_rates = np.asarray(method_class1_rates[method], dtype=float)
        readiness_delays = np.asarray(method_readiness_delays[method], dtype=float)
        valid_count = int(method_valid[method])
        failed_count = N_PERMUTATIONS - valid_count
        if valid_count == 0:
            mean_jaccard = np.nan
            median_jaccard = np.nan
            unique_sets = 0
            change_rate = np.nan
            class1_sd = np.nan
            readiness_mean = np.nan
            readiness_sd = np.nan
        else:
            mean_jaccard = mean_pairwise_jaccard(memberships)
            median_jaccard = median_pairwise_jaccard(memberships)
            unique_sets = len(set(memberships))
            baseline_membership = memberships[0]
            change_rate = float(np.mean([membership != baseline_membership for membership in memberships]))
            class1_sd = float(np.std(class1_rates, ddof=0))
            readiness_mean = float(np.mean(readiness_delays))
            readiness_sd = float(np.std(readiness_delays, ddof=0))
        case_summary_records.append({'case_id': case['case_id'], 'dataset': case['dataset'], 'stream_seed': case['stream_seed'], 'learner': case['learner'], 'scenario': case['scenario'], 'delay_seed': case['delay_seed'], 'block_id': case['block_id'], 'alarm_time': alarm_time, 'labels_at_alarm': case['labels_at_alarm'], 'unknown_at_alarm': case['unknown_at_alarm'], 'method': method, 'permutations': N_PERMUTATIONS, 'valid_permutations': valid_count, 'failed_permutations': failed_count, 'failed_permutation_rate': failed_count / N_PERMUTATIONS, 'mean_pairwise_membership_jaccard': mean_jaccard, 'median_pairwise_membership_jaccard': median_jaccard, 'unique_training_sets': unique_sets, 'membership_change_rate_vs_original': change_rate, 'training_class1_fraction_mean': float(np.mean(class1_rates)) if valid_count > 0 else np.nan, 'training_class1_fraction_sd': class1_sd, 'readiness_delay_mean': readiness_mean, 'readiness_delay_sd': readiness_sd})
    return (permutation_records, case_summary_records)

def main():
    print('=' * 155)
    print('RETURN-ORDER INVARIANCE TEST')
    print('=' * 155)
    print()
    print('Input:', STEP24_DECISION_FILE)
    print('Cases per scenario:', CASES_PER_SCENARIO)
    print('Permutations per case:', N_PERMUTATIONS)
    print('Soft-q:', SOFT_Q)
    print('Training size:', SOFT_TRAIN_SIZE)
    print('Smoke test:', SMOKE_TEST)
    print()
    cases = select_cases()
    print()
    print('Total selected fixed alarm cases:', len(cases))
    print()
    selected_case_records = []
    for case in cases:
        selected_case_records.append({'case_id': case['case_id'], 'dataset': case['dataset'], 'stream_seed': case['stream_seed'], 'learner': case['learner'], 'scenario': case['scenario'], 'lambda0': case['lambda0'], 'lambda1': case['lambda1'], 'delay_seed': case['delay_seed'], 'block_id': case['block_id'], 'block_start': case['block_start'], 'block_close': case['block_close'], 'alarm_time': case['alarm_time'], 'labels_at_alarm': case['labels_at_alarm'], 'unknown_at_alarm': case['unknown_at_alarm'], 'future_candidate_count': len(case['future_ids']), 'stream_source': case['stream_source']})
    selected_cases_df = pd.DataFrame(selected_case_records)
    selected_cases_df.to_csv(os.path.join(OUTPUT_DIR, 'selected_cases.csv'), index=False)
    permutation_records = []
    case_summary_records = []
    for case_index, case in enumerate(cases, start=1):
        print(f"[{case_index:03d}/{len(cases):03d}] {case['dataset']} | {case['learner']} | {case['scenario']} | stream={case['stream_seed']} | delay={case['delay_seed']} | block={case['block_id']}")
        raw_records, summary_records = run_case(case)
        permutation_records.extend(raw_records)
        case_summary_records.extend(summary_records)
    raw_df = pd.DataFrame(permutation_records)
    case_df = pd.DataFrame(case_summary_records)
    raw_df.to_csv(os.path.join(OUTPUT_DIR, 'permutation_raw.csv'), index=False)
    case_df.to_csv(os.path.join(OUTPUT_DIR, 'case_summary.csv'), index=False)
    main_records = []
    for method in ['COMPLETE_COHORT', 'SOFT_Q30']:
        temp = case_df[case_df['method'] == method].copy()
        main_records.append({'method': method, 'cases': len(temp), 'permutations_per_case': N_PERMUTATIONS, 'mean_pairwise_membership_jaccard': float(temp['mean_pairwise_membership_jaccard'].mean()), 'median_case_pairwise_jaccard': float(temp['mean_pairwise_membership_jaccard'].median()), 'mean_unique_training_sets': float(temp['unique_training_sets'].mean()), 'median_unique_training_sets': float(temp['unique_training_sets'].median()), 'max_unique_training_sets': int(temp['unique_training_sets'].max()), 'invariant_case_rate': float(np.mean(temp['unique_training_sets'] == 1)), 'mean_membership_change_rate': float(temp['membership_change_rate_vs_original'].mean()), 'mean_training_class1_fraction_sd': float(temp['training_class1_fraction_sd'].mean()), 'mean_readiness_delay': float(temp['readiness_delay_mean'].mean()), 'mean_readiness_time_sd': float(temp['readiness_delay_sd'].mean()), 'failed_permutation_rate': float(temp['failed_permutation_rate'].mean())})
    main_df = pd.DataFrame(main_records)
    main_df.to_csv(os.path.join(OUTPUT_DIR, 'main_paper_table.csv'), index=False)
    scenario_df = case_df.groupby(['scenario', 'method'], as_index=False).agg(cases=('case_id', 'nunique'), mean_pairwise_membership_jaccard=('mean_pairwise_membership_jaccard', 'mean'), mean_unique_training_sets=('unique_training_sets', 'mean'), invariant_case_rate=('unique_training_sets', lambda x: float(np.mean(np.asarray(x) == 1))), mean_membership_change_rate=('membership_change_rate_vs_original', 'mean'), mean_class1_fraction_sd=('training_class1_fraction_sd', 'mean'), mean_readiness_delay=('readiness_delay_mean', 'mean'), mean_readiness_time_sd=('readiness_delay_sd', 'mean'), failed_permutation_rate=('failed_permutation_rate', 'mean'))
    scenario_df.to_csv(os.path.join(OUTPUT_DIR, 'scenario_summary.csv'), index=False)
    complete = main_df[main_df['method'] == 'COMPLETE_COHORT'].iloc[0]
    soft = main_df[main_df['method'] == 'SOFT_Q30'].iloc[0]
    complete_jaccard_pass = bool(abs(complete['mean_pairwise_membership_jaccard'] - 1.0) < 1e-12)
    complete_unique_pass = bool((case_df[case_df['method'] == 'COMPLETE_COHORT']['unique_training_sets'] == 1).all())
    complete_change_pass = bool((case_df[case_df['method'] == 'COMPLETE_COHORT']['membership_change_rate_vs_original'] == 0).all())
    complete_class_sd_pass = bool((np.abs(case_df[case_df['method'] == 'COMPLETE_COHORT']['training_class1_fraction_sd']) < 1e-12).all())
    soft_variability_detected = bool(soft['mean_pairwise_membership_jaccard'] < 0.999999 or soft['mean_unique_training_sets'] > 1.0)
    checks_df = pd.DataFrame([{'check': 'Complete mean pairwise Jaccard == 1', 'pass': complete_jaccard_pass}, {'check': 'Complete has exactly one membership set per case', 'pass': complete_unique_pass}, {'check': 'Complete membership change rate == 0', 'pass': complete_change_pass}, {'check': 'Complete class-ratio SD == 0', 'pass': complete_class_sd_pass}, {'check': 'Soft-q30 membership variability detected', 'pass': soft_variability_detected}])
    checks_df.to_csv(os.path.join(OUTPUT_DIR, 'scientific_checks.csv'), index=False)
    print()
    print('=' * 155)
    print('MAIN PAPER TABLE')
    print('=' * 155)
    print(main_df.to_string(index=False))
    print()
    print('=' * 155)
    print('SCENARIO SUMMARY')
    print('=' * 155)
    print(scenario_df.to_string(index=False))
    print()
    print('=' * 155)
    print('SCIENTIFIC CHECKS')
    print('=' * 155)
    print(checks_df.to_string(index=False))
    print()
    print('=' * 155)
    print('KEY INTERPRETATION')
    print('=' * 155)
    print('Complete Cohort mean pairwise Jaccard:', f"{complete['mean_pairwise_membership_jaccard']:.6f}")
    print('Complete Cohort mean unique sets:', f"{complete['mean_unique_training_sets']:.3f}")
    print('Complete Cohort invariant-case rate:', f"{complete['invariant_case_rate']:.6f}")
    print('Complete Cohort class-ratio SD:', f"{complete['mean_training_class1_fraction_sd']:.6f}")
    print()
    print('Soft-q30 mean pairwise Jaccard:', f"{soft['mean_pairwise_membership_jaccard']:.6f}")
    print('Soft-q30 mean unique sets:', f"{soft['mean_unique_training_sets']:.3f}")
    print('Soft-q30 invariant-case rate:', f"{soft['invariant_case_rate']:.6f}")
    print('Soft-q30 membership-change rate:', f"{soft['mean_membership_change_rate']:.6f}")
    print('Soft-q30 class-ratio SD:', f"{soft['mean_training_class1_fraction_sd']:.6f}")
    print()
    print('IMPORTANT:')
    print('Complete Cohort Jaccard=1 is an implementation-level verification of the design property, not an unexpected statistical discovery.')
    print()
    print('Output directory:')
    print(OUTPUT_DIR)
    print('=' * 155)
if __name__ == '__main__':
    main()
