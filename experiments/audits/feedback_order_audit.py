"""Evaluate complete adaptive trajectories under perturbed feedback schedules."""
from completion_monitoring.paths import workspace_root, external_data_root, source_file
import os
for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS']:
    os.environ[key] = '1'
import argparse
import hashlib
import importlib
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config
SCRIPT_NAME = 'feedback_order_audit'
IMPLEMENTATION_VERSION = 'v2_case_parallel_high_performance'
PROJECT_ROOT = workspace_root()
OUTPUT_DIR = PROJECT_ROOT / 'result' / SCRIPT_NAME
SMOKE_DIR = OUTPUT_DIR / 'smoke'
CACHE_DIR = OUTPUT_DIR / 'case_cache'
SMOKE_CACHE_DIR = SMOKE_DIR / 'case_cache'
STEP27_DIR = PROJECT_ROOT / 'result' / 'arf_holdout'
STEP27_STREAM_DIR = STEP27_DIR / 'streams'
STEP27_MANIFEST = STEP27_DIR / 'stream_manifest.csv'
for directory in [OUTPUT_DIR, SMOKE_DIR, CACHE_DIR, SMOKE_CACHE_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
DATASETS = ['SEA', 'Agrawal', 'Mixed', 'RandomTree', 'Sine']
STREAM_SEEDS = list(range(6101, 6111))
BASE_DELAY_SEEDS = list(range(9001, 9006))
DELAY_SCENARIOS = {'UNIFORM_100_100': (100, 100), 'MILD_50_100': (50, 100), 'MEDIUM_30_150': (30, 150), 'SEVERE_20_200': (20, 200), 'REVERSED_200_20': (200, 20)}
MONITOR_START = 400
WINDOW_SIZE = 100
EPSILON = 0.02
ARF_RANDOM_SEED = 2026
MASTER_SEED = 28028
CASES_PER_DATASET_SCENARIO = 4
N_PERTURBATIONS = 30
CPU_COUNT = os.cpu_count() or 1
DEFAULT_JOBS = min(16, max(1, CPU_COUNT - 3))
CORE_STRATEGIES = ['FULL_SOFT_Q30', 'FULL_COMPLETE_COHORT', 'WAIT_ALL_COMPLETE_COHORT']
ALL_STRATEGIES = ['STATIC', 'FULL_SOFT_Q30', 'FULL_COMPLETE_COHORT', 'WAIT_ALL_COMPLETE_COHORT']
STREAM_CACHE = {}
_WORKER_CONTEXT = None

def stable_seed(*items):
    text = '|'.join((str(x) for x in items))
    digest = hashlib.sha256(text.encode('utf-8')).hexdigest()
    return int(digest[:16], 16) % 2 ** 32

def ids_hash(ids):
    if ids is None:
        return 'NONE'
    ids = np.asarray(ids, dtype=np.int64)
    return hashlib.sha256(ids.tobytes()).hexdigest()

def ids_to_string(ids):
    if ids is None:
        return ''
    return ';'.join((str(int(x)) for x in np.asarray(ids, dtype=int)))

def string_to_ids(text):
    if text is None:
        return None
    if isinstance(text, float) and np.isnan(text):
        return None
    text = str(text).strip()
    if text in {'', 'NONE', 'nan'}:
        return None
    return np.asarray([int(x) for x in text.split(';') if x], dtype=int)

def jaccard_similarity(a, b):
    if a is None and b is None:
        return 1.0
    if a is None or b is None:
        return 0.0
    a = set(np.asarray(a, dtype=int).tolist())
    b = set(np.asarray(b, dtype=int).tolist())
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)

def population_std(values):
    values = np.asarray([x for x in values if pd.notna(x)], dtype=float)
    if len(values) <= 1:
        return 0.0
    return float(np.std(values, ddof=0))

def stream_content_sha256(X, y):
    X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
    y = np.ascontiguousarray(np.asarray(y, dtype=np.int64).reshape(-1))
    h = hashlib.sha256()
    h.update(str(X.shape).encode('utf-8'))
    h.update(X.tobytes())
    h.update(str(y.shape).encode('utf-8'))
    h.update(y.tobytes())
    return h.hexdigest()

def return_order_hash(arrival_map):
    ordered = sorted(arrival_map.items(), key=lambda x: (x[1], x[0]))
    return ids_hash([sample_id for sample_id, _ in ordered])

def stream_path(dataset, stream_seed):
    return STEP27_STREAM_DIR / f'{dataset}_streamseed_{int(stream_seed)}.npz'

def load_frozen_stream(dataset, stream_seed):
    key = (str(dataset), int(stream_seed))
    if key in STREAM_CACHE:
        return STREAM_CACHE[key]
    path = stream_path(dataset, stream_seed)
    if not path.exists():
        raise FileNotFoundError(f'\nMissing ARF holdout frozen stream:\n{path}\n')
    with np.load(path, allow_pickle=False) as data:
        X = np.asarray(data['X'], dtype=float)
        y = np.asarray(data['y'], dtype=int).reshape(-1)
    if len(y) != 2200:
        raise RuntimeError(f'{path}: expected 2200 samples, got {len(y)}')
    if X.ndim != 2:
        raise RuntimeError(f'{path}: X must be 2-D')
    if len(X) != len(y):
        raise RuntimeError(f'{path}: X/y length mismatch')
    if set(np.unique(y).tolist()) != {0, 1}:
        raise RuntimeError(f'{path}: binary labels expected')
    result = {'X': X, 'y': y, 'hash': stream_content_sha256(X, y), 'path': str(path)}
    STREAM_CACHE[key] = result
    return result

def validate_step27_streams():
    if not STEP27_MANIFEST.exists():
        raise FileNotFoundError(f'Cannot find {STEP27_MANIFEST}')
    manifest = pd.read_csv(STEP27_MANIFEST)
    required = {'dataset', 'stream_seed', 'frozen_content_sha256'}
    missing = required - set(manifest.columns)
    if missing:
        raise RuntimeError('Manifest missing columns: ' + str(sorted(missing)))
    rows = []
    for dataset in DATASETS:
        for stream_seed in STREAM_SEEDS:
            match = manifest[(manifest['dataset'].astype(str) == dataset) & (manifest['stream_seed'].astype(int) == int(stream_seed))]
            if len(match) != 1:
                raise RuntimeError(f'Manifest mismatch for {dataset}/{stream_seed}')
            stream = load_frozen_stream(dataset, stream_seed)
            expected_hash = str(match.iloc[0]['frozen_content_sha256'])
            actual_hash = str(stream['hash'])
            passed = expected_hash == actual_hash
            rows.append({'dataset': dataset, 'stream_seed': stream_seed, 'expected_hash': expected_hash, 'actual_hash': actual_hash, 'hash_match': passed})
            if not passed:
                raise RuntimeError(f'ARF holdout frozen stream changed: {dataset}/{stream_seed}')
    result = pd.DataFrame(rows)
    if len(result) != 50:
        raise RuntimeError('Expected exactly 50 streams.')
    return result

def select_cases(smoke=False):
    if smoke:
        lambda0, lambda1 = DELAY_SCENARIOS['MEDIUM_30_150']
        return pd.DataFrame([{'case_id': 'SMOKE|SEA|6101|MEDIUM_30_150|9001', 'dataset': 'SEA', 'stream_seed': 6101, 'scenario': 'MEDIUM_30_150', 'lambda0': lambda0, 'lambda1': lambda1, 'base_delay_seed': 9001}])
    rows = []
    for scenario, (lambda0, lambda1) in DELAY_SCENARIOS.items():
        for dataset in DATASETS:
            candidates = [(stream_seed, delay_seed) for stream_seed in STREAM_SEEDS for delay_seed in BASE_DELAY_SEEDS]
            rng = np.random.default_rng(stable_seed(MASTER_SEED, 'CASE_SELECTION', dataset, scenario))
            selected_indices = rng.choice(len(candidates), size=CASES_PER_DATASET_SCENARIO, replace=False)
            for index in sorted(selected_indices.tolist()):
                stream_seed, delay_seed = candidates[index]
                rows.append({'case_id': f'{dataset}|{stream_seed}|{scenario}|{delay_seed}', 'dataset': dataset, 'stream_seed': int(stream_seed), 'scenario': scenario, 'lambda0': int(lambda0), 'lambda1': int(lambda1), 'base_delay_seed': int(delay_seed)})
    result = pd.DataFrame(rows).sort_values(['scenario', 'dataset', 'stream_seed', 'base_delay_seed']).reset_index(drop=True)
    if len(result) != 100:
        raise RuntimeError('Formal feedback-order audit must contain exactly 100 cases.')
    return result

def generate_base_delays(y, delay_seed, lambda0, lambda1):
    rng = np.random.default_rng(int(delay_seed))
    sample_ids = np.arange(MONITOR_START, len(y), dtype=int)
    labels = y[sample_ids]
    delays = np.empty(len(sample_ids), dtype=int)
    for label, lam in [(0, lambda0), (1, lambda1)]:
        mask = labels == label
        delays[mask] = rng.poisson(int(lam), size=int(np.sum(mask)))
    return (sample_ids, delays)

def perturb_global_schedule(y, sample_ids, base_delays, case_id, schedule_index):
    sample_ids = np.asarray(sample_ids, dtype=int)
    base_delays = np.asarray(base_delays, dtype=int)
    perturbed = base_delays.copy()
    if schedule_index > 0:
        rng = np.random.default_rng(stable_seed(MASTER_SEED, case_id, 'GLOBAL_ORDER', schedule_index))
        labels = y[sample_ids]
        for label in [0, 1]:
            indices = np.flatnonzero(labels == label)
            values = base_delays[indices].copy()
            rng.shuffle(values)
            perturbed[indices] = values
    arrivals = sample_ids + perturbed
    arrival_map = dict(zip(sample_ids.tolist(), arrivals.tolist()))
    arrival_groups = {}
    for sample_id, arrival in arrival_map.items():
        arrival_groups.setdefault(int(arrival), []).append(int(sample_id))
    for arrival in arrival_groups:
        arrival_groups[arrival].sort()
    return (arrival_map, arrival_groups, return_order_hash(arrival_map))

def get_worker_context():
    global _WORKER_CONTEXT
    if _WORKER_CONTEXT is not None:
        return _WORKER_CONTEXT
    base = importlib.import_module('chronological_protocol')
    engine = importlib.import_module('cohort_protocol')
    context = {'engine': engine, 'original_create_state': engine.create_state, 'original_calculate_oracle': engine.base.calculate_oracle, 'original_perform_update': engine.perform_update, 'original_generate_delay': engine.base.generate_delay, 'original_create_model': engine.base.create_model, 'original_strategies': list(engine.STRATEGIES)}
    _WORKER_CONTEXT = context
    return context

def restore_engine(context):
    engine = context['engine']
    engine.create_state = context['original_create_state']
    engine.base.calculate_oracle = context['original_calculate_oracle']
    engine.perform_update = context['original_perform_update']
    engine.base.generate_delay = context['original_generate_delay']
    engine.base.create_model = context['original_create_model']
    engine.STRATEGIES = list(context['original_strategies'])
    return engine

def prepare_engine(arrival_map, arrival_groups, strategies):
    context = get_worker_context()
    engine = restore_engine(context)
    from skmultiflow.meta import AdaptiveRandomForestClassifier

    def create_arf_model(_learner):
        return AdaptiveRandomForestClassifier(split_criterion='gini', random_state=ARF_RANDOM_SEED)
    engine.base.create_model = create_arf_model
    engine.STRATEGIES = list(strategies)
    frozen_arrival_map = dict(arrival_map)
    frozen_arrival_groups = {int(k): list(v) for k, v in arrival_groups.items()}

    def custom_generate_delay(_y, _seed, _lambda0, _lambda1):
        return (dict(frozen_arrival_map), {int(k): list(v) for k, v in frozen_arrival_groups.items()})
    engine.base.generate_delay = custom_generate_delay
    original_create_state = context['original_create_state']
    original_calculate_oracle = context['original_calculate_oracle']
    original_perform_update = context['original_perform_update']

    class DecisionAuditList(list):

        def __init__(self, state):
            super().__init__()
            self.state = state

        def append(self, record):
            enriched = dict(record)
            ids = self.state.get('_audit_current_block_ids')
            enriched['block_start'] = self.state.get('_audit_current_block_start', np.nan)
            enriched['block_end'] = self.state.get('_audit_current_block_end', np.nan)
            enriched['block_close_time'] = self.state.get('_audit_current_block_close', np.nan)
            enriched['trigger_cohort_hash'] = ids_hash(ids)
            enriched['trigger_cohort_ids'] = ids_to_string(ids)
            super().append(enriched)

    def create_state_audited(*args, **kwargs):
        state = original_create_state(*args, **kwargs)
        state['decision_records'] = DecisionAuditList(state)
        return state

    def calculate_oracle_audited(y, state, block):
        ids = np.asarray(block['sample_ids'], dtype=int)
        state['_audit_current_block_ids'] = ids.copy()
        state['_audit_current_block_start'] = int(block['start'])
        state['_audit_current_block_end'] = int(block['end'])
        state['_audit_current_block_close'] = int(block['close_time'])
        return original_calculate_oracle(y, state, block)

    def perform_update_audited(strategy, state, train_ids, current_time, X, y, learner):
        train_ids = np.asarray(train_ids, dtype=int).copy()
        cohort_ids = state.get('adaptation_cohort_ids')
        if cohort_ids is not None:
            cohort_ids = np.asarray(cohort_ids, dtype=int).copy()
        n_update_before = len(state['update_records'])
        n_skip_before = len(state['skipped_update_records'])
        result = original_perform_update(strategy, state, train_ids, current_time, X, y, learner)
        target = None
        if len(state['update_records']) > n_update_before:
            target = state['update_records'][-1]
        elif len(state['skipped_update_records']) > n_skip_before:
            target = state['skipped_update_records'][-1]
        if target is not None:
            target['train_ids_hash'] = ids_hash(train_ids)
            target['train_ids'] = ids_to_string(train_ids)
            target['trigger_cohort_hash'] = ids_hash(cohort_ids)
            target['trigger_cohort_ids'] = ids_to_string(cohort_ids)
            target['train_equals_trigger_cohort'] = int(cohort_ids is not None and np.array_equal(np.sort(train_ids), np.sort(cohort_ids)))
        return result
    engine.create_state = create_state_audited
    engine.base.calculate_oracle = calculate_oracle_audited
    engine.perform_update = perform_update_audited
    if int(engine.WINDOW_SIZE) != WINDOW_SIZE:
        raise RuntimeError('WINDOW_SIZE changed.')
    if abs(float(engine.EPSILON) - EPSILON) > 1e-15:
        raise RuntimeError('EPSILON changed.')
    return engine

def extract_first_trigger(decision_df, strategy):
    if len(decision_df) == 0 or 'strategy' not in decision_df.columns:
        return None
    sub = decision_df[(decision_df['strategy'] == strategy) & (decision_df['decision'] == 'DEGRADATION')].copy()
    if len(sub) == 0:
        return None
    sort_cols = [col for col in ['decision_time', 'block_start', 'block_id'] if col in sub.columns]
    sub = sub.sort_values(sort_cols, kind='mergesort')
    return sub.iloc[0].to_dict()

def find_matching_record(df, strategy, alarm_time, time_column):
    if alarm_time is None or pd.isna(alarm_time) or len(df) == 0 or ('strategy' not in df.columns) or ('alarm_time' not in df.columns):
        return None
    sub = df[(df['strategy'] == strategy) & (df['alarm_time'].astype(float) == float(alarm_time))].copy()
    if len(sub) == 0:
        return None
    if time_column in sub.columns:
        sub = sub.sort_values(time_column, kind='mergesort')
    return sub.iloc[0].to_dict()

def extract_schedule_rows(run_records, decision_records, update_records, skipped_records, case, schedule_index, schedule_hash, stream_hash, strategies):
    run_df = pd.DataFrame(run_records)
    decision_df = pd.DataFrame(decision_records)
    update_df = pd.DataFrame(update_records)
    skipped_df = pd.DataFrame(skipped_records)
    output = []
    for strategy in strategies:
        run_match = run_df[run_df['strategy'] == strategy]
        if len(run_match) != 1:
            raise RuntimeError(f"Expected exactly one run row for {case['case_id']} / {schedule_index} / {strategy}")
        run_row = run_match.iloc[0]
        trigger = None if strategy == 'STATIC' else extract_first_trigger(decision_df, strategy)
        if trigger is None:
            trigger_exists = 0
            alarm_time = np.nan
            block_id = np.nan
            model_version = np.nan
            trigger_start = np.nan
            trigger_end = np.nan
            trigger_hash = 'NONE'
            trigger_ids = ''
            trigger_delay = np.nan
        else:
            trigger_exists = 1
            alarm_time = int(trigger['decision_time'])
            block_id = int(trigger['block_id'])
            model_version = int(trigger['model_version'])
            trigger_start = int(trigger['block_start'])
            trigger_end = int(trigger['block_end'])
            trigger_hash = str(trigger['trigger_cohort_hash'])
            trigger_ids = str(trigger['trigger_cohort_ids'])
            trigger_delay = float(trigger['decision_delay'])
        if trigger_exists:
            update = find_matching_record(update_df, strategy, alarm_time, 'update_time')
            skipped = find_matching_record(skipped_df, strategy, alarm_time, 'skip_time')
        else:
            update = None
            skipped = None
        if update is not None:
            update_ready = 1
            update_skipped = 0
            update_time = float(update['update_time'])
            train_hash = str(update.get('train_ids_hash', 'NONE'))
            train_ids = str(update.get('train_ids', ''))
            train_equals_trigger = int(update.get('train_equals_trigger_cohort', 0))
        elif skipped is not None:
            update_ready = 0
            update_skipped = 1
            update_time = float(skipped['skip_time'])
            train_hash = str(skipped.get('train_ids_hash', 'NONE'))
            train_ids = str(skipped.get('train_ids', ''))
            train_equals_trigger = int(skipped.get('train_equals_trigger_cohort', 0))
        else:
            update_ready = 0
            update_skipped = 0
            update_time = np.nan
            train_hash = 'NONE'
            train_ids = ''
            train_equals_trigger = np.nan
        output.append({'case_id': case['case_id'], 'dataset': case['dataset'], 'stream_seed': int(case['stream_seed']), 'scenario': case['scenario'], 'lambda0': int(case['lambda0']), 'lambda1': int(case['lambda1']), 'base_delay_seed': int(case['base_delay_seed']), 'schedule_index': int(schedule_index), 'is_original': int(schedule_index == 0), 'return_order_hash': schedule_hash, 'stream_hash': stream_hash, 'strategy': strategy, 'accuracy': float(run_row['accuracy']), 'macro_f1': float(run_row['macro_f1']), 'mcc': float(run_row['mcc']), 'retrain_count': int(run_row['retrain_count']), 'decision_count': int(run_row['decision_count']), 'decision_accuracy': float(run_row['decision_accuracy']) if pd.notna(run_row['decision_accuracy']) else np.nan, 'false_positive': int(run_row['false_positive']), 'false_negative': int(run_row['false_negative']), 'mean_decision_delay': float(run_row['decision_delay']) if pd.notna(run_row['decision_delay']) else np.nan, 'trigger_exists': trigger_exists, 'first_alarm_time': alarm_time, 'first_trigger_block_id': block_id, 'first_trigger_model_version': model_version, 'first_trigger_start': trigger_start, 'first_trigger_end': trigger_end, 'first_trigger_hash': trigger_hash, 'first_trigger_ids': trigger_ids, 'first_trigger_decision_delay': trigger_delay, 'first_trigger_update_ready': update_ready, 'first_trigger_skipped': update_skipped, 'first_update_time': update_time, 'first_train_hash': train_hash, 'first_train_ids': train_ids, 'first_train_equals_trigger': train_equals_trigger})
    return output

def run_one_case(case, schedule_indices, cache_path):
    if cache_path.exists():
        cached = pd.read_csv(cache_path)
        expected_core_rows = len(schedule_indices) * len(CORE_STRATEGIES)
        expected_static_rows = min(2, len(schedule_indices))
        if len(cached) == expected_core_rows + expected_static_rows:
            return cached.to_dict(orient='records')
    dataset = str(case['dataset'])
    stream_seed = int(case['stream_seed'])
    lambda0 = int(case['lambda0'])
    lambda1 = int(case['lambda1'])
    delay_seed = int(case['base_delay_seed'])
    case_id = str(case['case_id'])
    stream = load_frozen_stream(dataset, stream_seed)
    X = stream['X']
    y = stream['y']
    sample_ids, base_delays = generate_base_delays(y, delay_seed, lambda0, lambda1)
    rows = []
    for schedule_index in schedule_indices:
        arrival_map, arrival_groups, schedule_hash = perturb_global_schedule(y, sample_ids, base_delays, case_id, schedule_index)
        if schedule_index in {0, 1}:
            strategies = ALL_STRATEGIES
        else:
            strategies = CORE_STRATEGIES
        engine = prepare_engine(arrival_map, arrival_groups, strategies)
        run_records, decision_records, update_records, skipped_records = engine.run_one(dataset, X, y, 'ARF', case['scenario'], lambda0, lambda1, delay_seed)
        rows.extend(extract_schedule_rows(run_records, decision_records, update_records, skipped_records, case, schedule_index, schedule_hash, stream['hash'], strategies))
    df = pd.DataFrame(rows)
    temp_path = cache_path.with_suffix('.tmp.csv.gz')
    df.to_csv(temp_path, index=False, compression='gzip')
    temp_path.replace(cache_path)
    return rows

def summarize_case_strategy(group):
    group = group.sort_values('schedule_index').reset_index(drop=True)
    original_rows = group[group['schedule_index'] == 0]
    if len(original_rows) != 1:
        raise RuntimeError('Missing original schedule.')
    original = original_rows.iloc[0]
    perturbed = group[group['schedule_index'] > 0]
    original_trigger_hash = str(original['first_trigger_hash'])
    original_train_hash = str(original['first_train_hash'])
    original_trigger_ids = string_to_ids(original['first_trigger_ids'])
    original_train_ids = string_to_ids(original['first_train_ids'])
    trigger_agreement = []
    trigger_jaccard = []
    train_agreement = []
    train_jaccard = []
    conditional_train_agreement = []
    conditional_train_jaccard = []
    for _, row in perturbed.iterrows():
        this_trigger_hash = str(row['first_trigger_hash'])
        this_train_hash = str(row['first_train_hash'])
        this_trigger_ids = string_to_ids(row['first_trigger_ids'])
        this_train_ids = string_to_ids(row['first_train_ids'])
        trigger_agreement.append(int(this_trigger_hash == original_trigger_hash))
        trigger_jaccard.append(jaccard_similarity(original_trigger_ids, this_trigger_ids))
        train_agreement.append(int(this_train_hash == original_train_hash))
        train_jaccard.append(jaccard_similarity(original_train_ids, this_train_ids))
        if this_trigger_hash == original_trigger_hash and original_trigger_hash != 'NONE':
            conditional_train_agreement.append(int(this_train_hash == original_train_hash))
            if original_train_ids is not None and this_train_ids is not None:
                conditional_train_jaccard.append(jaccard_similarity(original_train_ids, this_train_ids))
    alarm_times = group.loc[group['trigger_exists'] == 1, 'first_alarm_time'].dropna().tolist()
    return pd.Series({'dataset': original['dataset'], 'scenario': original['scenario'], 'strategy': original['strategy'], 'n_schedules': len(group), 'first_trigger_agreement_rate': float(np.mean(trigger_agreement)), 'mean_trigger_jaccard_vs_original': float(np.mean(trigger_jaccard)), 'distinct_trigger_states': int(group['first_trigger_hash'].astype(str).nunique()), 'alarm_time_sd': population_std(alarm_times), 'first_training_state_agreement_rate': float(np.mean(train_agreement)), 'mean_training_jaccard_vs_original': float(np.mean(train_jaccard)), 'distinct_training_states': int(group['first_train_hash'].astype(str).nunique()), 'training_agreement_given_same_trigger': float(np.mean(conditional_train_agreement)) if conditional_train_agreement else np.nan, 'training_jaccard_given_same_trigger': float(np.mean(conditional_train_jaccard)) if conditional_train_jaccard else np.nan, 'final_accuracy_sd': population_std(group['accuracy']), 'final_macro_f1_sd': population_std(group['macro_f1']), 'final_mcc_sd': population_std(group['mcc']), 'accuracy_range': float(group['accuracy'].max() - group['accuracy'].min()), 'unique_return_orders': int(group['return_order_hash'].nunique())})

def build_case_summary(raw_df):
    rows = []
    adaptive = raw_df[raw_df['strategy'].isin(CORE_STRATEGIES)]
    for (case_id, strategy), group in adaptive.groupby(['case_id', 'strategy'], sort=False):
        summary = summarize_case_strategy(group)
        summary['case_id'] = case_id
        rows.append(summary)
    return pd.DataFrame(rows)

def aggregate_case_summaries(case_df, grouping):
    metrics = ['first_trigger_agreement_rate', 'mean_trigger_jaccard_vs_original', 'distinct_trigger_states', 'alarm_time_sd', 'first_training_state_agreement_rate', 'mean_training_jaccard_vs_original', 'distinct_training_states', 'training_agreement_given_same_trigger', 'training_jaccard_given_same_trigger', 'final_accuracy_sd', 'final_macro_f1_sd', 'final_mcc_sd', 'accuracy_range', 'unique_return_orders']
    agg_args = {f'mean_{metric}': (metric, 'mean') for metric in metrics}
    return case_df.groupby(grouping, as_index=False).agg(cases=('case_id', 'count'), **agg_args)

def build_main_table(case_df):
    overall = aggregate_case_summaries(case_df, ['strategy'])
    order = {'FULL_SOFT_Q30': 0, 'FULL_COMPLETE_COHORT': 1, 'WAIT_ALL_COMPLETE_COHORT': 2}
    overall['_order'] = overall['strategy'].map(order)
    overall = overall.sort_values('_order').drop(columns=['_order']).reset_index(drop=True)
    main_table = overall[['strategy', 'cases', 'mean_first_trigger_agreement_rate', 'mean_mean_trigger_jaccard_vs_original', 'mean_training_agreement_given_same_trigger', 'mean_training_jaccard_given_same_trigger', 'mean_distinct_trigger_states', 'mean_alarm_time_sd', 'mean_final_accuracy_sd']].copy()
    return (main_table, overall)

def build_checks(raw_df, selected_cases, schedule_indices, smoke):
    expected_cases = 1 if smoke else 100
    expected_core = expected_cases * len(schedule_indices) * len(CORE_STRATEGIES)
    expected_static = expected_cases * min(2, len(schedule_indices))
    expected_rows = expected_core + expected_static
    checks = []
    checks.append({'check': 'expected base cases', 'pass': len(selected_cases) == expected_cases, 'value': len(selected_cases)})
    checks.append({'check': 'expected raw rows', 'pass': len(raw_df) == expected_rows, 'value': f'{len(raw_df)} / {expected_rows}'})
    for strategy in CORE_STRATEGIES:
        subset = raw_df[raw_df['strategy'] == strategy]
        counts = subset.groupby('case_id')['schedule_index'].nunique()
        checks.append({'check': f'{strategy}: all schedules present', 'pass': bool((counts == len(schedule_indices)).all()), 'value': f'min={counts.min()}, max={counts.max()}'})
    static_df = raw_df[raw_df['strategy'] == 'STATIC']
    static_range = static_df.groupby('case_id')['accuracy'].agg(lambda x: float(x.max() - x.min()))
    checks.append({'check': 'STATIC prediction invariant under original vs perturbation', 'pass': bool((static_range <= 1e-12).all()), 'value': float(static_range.max())})
    for strategy in ['FULL_COMPLETE_COHORT', 'WAIT_ALL_COMPLETE_COHORT']:
        subset = raw_df[raw_df['strategy'] == strategy]
        agreement = subset['decision_accuracy'].dropna()
        checks.append({'check': f'{strategy}: oracle agreement == 1', 'pass': bool((np.abs(agreement - 1.0) <= 1e-12).all()), 'value': float(agreement.min()) if len(agreement) else np.nan})
        fp = int(subset['false_positive'].sum())
        fn = int(subset['false_negative'].sum())
        checks.append({'check': f'{strategy}: total FP == 0', 'pass': fp == 0, 'value': fp})
        checks.append({'check': f'{strategy}: total FN == 0', 'pass': fn == 0, 'value': fn})
    adaptive = raw_df[raw_df['strategy'] == 'FULL_COMPLETE_COHORT']
    unique_orders = adaptive.groupby('case_id')['return_order_hash'].nunique()
    checks.append({'check': 'multiple legal return orders per case', 'pass': bool((unique_orders >= 2).all()), 'value': f'min={unique_orders.min()}, mean={unique_orders.mean():.2f}'})
    return pd.DataFrame(checks)

def protocol_dict():
    return {'script': SCRIPT_NAME, 'scientific_design': 'Full-process return-order stability; triggering block not fixed.', 'stream_seeds': STREAM_SEEDS, 'delay_seeds': BASE_DELAY_SEEDS, 'delay_scenarios': {k: list(v) for k, v in DELAY_SCENARIOS.items()}, 'window_size': WINDOW_SIZE, 'epsilon': EPSILON, 'arf_random_seed': ARF_RANDOM_SEED, 'cases_per_dataset_scenario': CASES_PER_DATASET_SCENARIO, 'base_cases': 100, 'perturbations': N_PERTURBATIONS, 'schedules_per_case': N_PERTURBATIONS + 1, 'core_strategies': CORE_STRATEGIES, 'return_order_perturbation': 'Exact class-conditioned delay multiset permutation.', 'master_seed': MASTER_SEED, 'scientific_rule': 'Formal feedback-order audit settings are frozen before inspecting formal feedback-order audit results.'}

def validate_or_write_protocol(path):
    new_protocol = protocol_dict()
    if path.exists():
        try:
            with open(path, 'r', encoding='utf-8') as f:
                existing = json.load(f)
            for key, value in new_protocol.items():
                if key in existing and existing[key] != value:
                    raise RuntimeError(f'\nScientific feedback-order audit protocol changed after freeze:\n{key}\n')
        except json.JSONDecodeError:
            raise RuntimeError('Existing protocol file is invalid.')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump({**new_protocol, 'implementation_version': IMPLEMENTATION_VERSION}, f, indent=2, ensure_ascii=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--jobs', type=int, default=DEFAULT_JOBS)
    parser.add_argument('--fresh', action='store_true', help='Ignore and overwrite case checkpoints. Normally DO NOT use.')
    args = parser.parse_args()
    smoke = bool(args.smoke)
    jobs = 1 if smoke else int(args.jobs)
    output_dir = SMOKE_DIR if smoke else OUTPUT_DIR
    cache_dir = SMOKE_CACHE_DIR if smoke else CACHE_DIR
    print('=' * 120)
    print('GLOBAL RETURN-ORDER STABILITY')
    print('HIGH-PERFORMANCE CASE-PARALLEL VERSION')
    print('=' * 120)
    print('Implementation:', IMPLEMENTATION_VERSION)
    print('Visible CPU cores:', CPU_COUNT)
    print('Parallel workers:', jobs)
    print('ARF random seed:', ARF_RANDOM_SEED)
    print('feedback-order audit delay seeds:', BASE_DELAY_SEEDS)
    print('Smoke:', smoke)
    print()
    try:
        from skmultiflow.meta import AdaptiveRandomForestClassifier
        _ = AdaptiveRandomForestClassifier(random_state=ARF_RANDOM_SEED)
    except Exception as exc:
        raise ImportError('AdaptiveRandomForestClassifier requires the scikit-multiflow backend.') from exc
    print('ARF import: OK')
    audit = validate_step27_streams()
    audit.to_csv(output_dir / 'frozen_stream_audit.csv', index=False)
    print('ARF holdout stream hashes: 50 / 50 matched')
    selected_cases = select_cases(smoke=smoke)
    selected_cases.to_csv(output_dir / 'selected_cases.csv', index=False)
    validate_or_write_protocol(output_dir / 'protocol_freeze.json')
    if smoke:
        schedule_indices = [0, 1, 2]
    else:
        schedule_indices = list(range(0, N_PERTURBATIONS + 1))
    print('Base cases:', len(selected_cases))
    print('Schedules / case:', len(schedule_indices))
    print('Core adaptive strategy-runs:', len(selected_cases) * len(schedule_indices) * 3)
    print('STATIC negative-control runs:', len(selected_cases) * min(2, len(schedule_indices)))
    print()
    if args.fresh:
        print('WARNING: --fresh selected.')
        print('Existing case cache will be removed.')
        for path in cache_dir.glob('*.csv.gz'):
            path.unlink()
    case_jobs = []
    for _, row in selected_cases.iterrows():
        case = row.to_dict()
        safe_name = str(case['case_id']).replace('|', '__')
        cache_path = cache_dir / f'{safe_name}.csv.gz'
        case_jobs.append((case, cache_path))
    start = time.perf_counter()
    with parallel_config(backend='loky', inner_max_num_threads=1):
        results = Parallel(n_jobs=jobs, verbose=10, batch_size=1, pre_dispatch=2 * jobs, prefer='processes')((delayed(run_one_case)(case, schedule_indices, cache_path) for case, cache_path in case_jobs))
    rows = [row for case_rows in results for row in case_rows]
    raw_df = pd.DataFrame(rows)
    raw_df = raw_df.sort_values(['case_id', 'schedule_index', 'strategy'], kind='mergesort').reset_index(drop=True)
    raw_df.to_csv(output_dir / 'schedule_raw.csv', index=False)
    case_df = build_case_summary(raw_df)
    case_df.to_csv(output_dir / 'case_strategy_summary.csv', index=False)
    main_table, overall = build_main_table(case_df)
    main_table.to_csv(output_dir / 'main_paper_table.csv', index=False)
    overall.to_csv(output_dir / 'overall_summary.csv', index=False)
    scenario_summary = aggregate_case_summaries(case_df, ['scenario', 'strategy'])
    scenario_summary.to_csv(output_dir / 'scenario_summary.csv', index=False)
    checks = build_checks(raw_df, selected_cases, schedule_indices, smoke)
    checks.to_csv(output_dir / 'scientific_checks.csv', index=False)
    print()
    print('=' * 120)
    print('MAIN PAPER TABLE')
    print('=' * 120)
    print(main_table.to_string(index=False))
    print()
    print('=' * 120)
    print('SCIENTIFIC CHECKS')
    print('=' * 120)
    print(checks.to_string(index=False))
    passed = bool(checks['pass'].astype(bool).all())
    print()
    print('ALL SCIENTIFIC CHECKS PASSED:', passed)
    print()
    print('=' * 120)
    print('RUN COMPLETE')
    print('=' * 120)
    elapsed = time.perf_counter() - start
    print('Elapsed seconds:', round(elapsed, 2))
    print('Elapsed minutes:', round(elapsed / 60.0, 2))
    print('Output:', output_dir)
if __name__ == '__main__':
    main()
