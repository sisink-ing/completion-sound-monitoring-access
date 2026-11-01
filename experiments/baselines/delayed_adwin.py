"""Evaluate delayed-feedback ADWIN on the common ARF holdout."""
from completion_monitoring.paths import workspace_root, external_data_root, source_file
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[name] = '1'
import argparse
import importlib
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import wilcoxon
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
ROOT = Path(os.environ.get('CREATING_ROOT', str(workspace_root())))
STEP27 = ROOT / 'result' / 'arf_holdout'
OUT = ROOT / 'result' / 'delayed_adwin'
KEY = ['dataset', 'stream_seed', 'scenario', 'delay_seed']
OLD = ['STATIC', 'FULL_SOFT_Q30', 'FULL_COMPLETE_COHORT', 'WAIT_ALL_COMPLETE_COHORT']
NEW = 'ADWIN_ARRIVAL_ARF'
ORDER = OLD + [NEW]
NAMES = {'STATIC': 'Static', 'FULL_SOFT_Q30': 'Soft-q30', 'FULL_COMPLETE_COHORT': 'Complete Cohort (Ours)', 'WAIT_ALL_COMPLETE_COHORT': 'Wait-All Complete', NEW: 'Delayed-ADWIN-Arrival'}
SEED = 2026

def load_modules():
    s27 = importlib.import_module('arf_holdout')
    engine = importlib.import_module('cohort_protocol')
    if list(engine.STRATEGIES) != OLD:
        raise RuntimeError('cohort protocol strategies have changed.')
    if list(s27.STREAM_SEEDS) != list(range(6101, 6111)):
        raise RuntimeError('ARF holdout stream seeds changed.')
    if list(s27.DELAY_SEEDS) != list(range(8001, 8006)):
        raise RuntimeError('ARF holdout delay seeds changed.')
    actual = (engine.WINDOW_SIZE, engine.UPDATE_TRAIN_SIZE, engine.POST_UPDATE_CAL_SIZE, engine.MONITOR_START)
    if actual != (100, 100, 100, 400):
        raise RuntimeError('cohort protocol protocol changed.')
    return (s27, engine)

def tasks_from_protocol(s27, engine):
    tasks = [(family, int(stream_seed), scenario, int(lambda0), int(lambda1), int(delay_seed)) for family in s27.DATASETS for stream_seed in s27.STREAM_SEEDS for scenario, lambda0, lambda1 in s27.scenario_items(engine) for delay_seed in s27.DELAY_SEEDS]
    assert len(tasks) == 1250
    return tasks

def ensure_old_results(s27, tasks):
    path = STEP27 / 'raw.csv'
    if not path.is_file():
        raise FileNotFoundError(path)
    old = pd.read_csv(path)
    if len(old) != 5000:
        raise RuntimeError('ARF holdout must contain exactly 5000 rows.')
    if set(old.strategy.unique()) != set(OLD):
        raise RuntimeError('Unexpected ARF holdout strategies.')
    if old.duplicated(KEY + ['strategy']).any():
        raise RuntimeError('Duplicate ARF holdout tasks.')
    counts = old.groupby('strategy').size()
    if not (counts == 1250).all():
        raise RuntimeError('Each ARF holdout strategy must have 1250 tasks.')
    expected = {(t[0], t[1], t[2], t[5]) for t in tasks}
    actual = set(zip(old.dataset, old.stream_seed.astype(int), old.scenario, old.delay_seed.astype(int)))
    if expected != actual:
        raise RuntimeError('Frozen task keys do not match.')
    for family in s27.DATASETS:
        for seed in s27.STREAM_SEEDS:
            path = s27.stream_path(family, seed)
            _, _, _, sha = s27.read_stream_file(path)
            sub = old[(old.dataset == family) & (old.stream_seed == seed)]
            if sub.empty:
                raise RuntimeError('Missing frozen stream.')
            if not (sub.frozen_content_sha256.astype(str) == sha).all():
                raise RuntimeError(f'Stream hash mismatch: {family}/{seed}')
    return old

def adwin_one(task, delta):
    from skmultiflow.drift_detection import ADWIN
    s27, engine = load_modules()
    engine = s27.patch_engine_to_arf(engine)
    base = engine.base
    family, stream_seed, scenario, lambda0, lambda1, delay_seed = task
    X, y, stored_hash, sha = s27.read_stream_file(s27.stream_path(family, stream_seed))
    model, reference, train_acc, cal_acc = base.create_initial_model(X, y, 'ARF')
    initial_reference = float(reference)
    arrival_map, arrival_groups = base.generate_delay(y, delay_seed, lambda0, lambda1)
    last_event = len(y) - 1
    simulation_end = max(last_event, max(arrival_map.values()))
    detector = ADWIN(delta=float(delta))
    started = time.perf_counter()
    phase = 'MONITORING'
    version = 0
    episode = 0
    predictions = {}
    prediction_version = {}
    prediction_episode = {}
    prediction_phase = {}
    feedback = {}
    open_block = []
    closed_blocks = []
    block_id = 0
    cohort = None
    alarm_time = None
    alarm_origin = None
    alarm_latched = False
    calibration_ids = []
    monitor_events = 0
    waiting_events = 0
    calibration_events = 0
    detector_inputs = 0
    excluded_feedback = 0
    alarm_count = 0
    deferred_count = 0
    retrain_count = 0
    skipped_count = 0
    alarm_rows = []
    update_rows = []

    def lock_cohort(current_time, alert_time):
        nonlocal phase
        nonlocal cohort
        nonlocal alarm_latched
        nonlocal alarm_time
        nonlocal alarm_origin
        nonlocal open_block
        nonlocal closed_blocks
        if not closed_blocks:
            return False
        block = closed_blocks[-1]
        if block['version'] != version or block['episode'] != episode:
            raise RuntimeError('Incompatible cohort version.')
        cohort = tuple(block['ids'])
        if len(cohort) != 100:
            raise RuntimeError('Locked cohort size must be 100.')
        alarm_time = current_time
        alarm_origin = alert_time
        phase = 'WAITING_READY'
        alarm_latched = False
        open_block = []
        closed_blocks = []
        alarm_rows.append({'action': 'LOCKED', 'time': int(current_time), 'detector_alarm_time': int(alert_time), 'version': int(version), 'episode': int(episode), 'cohort_start': int(cohort[0]), 'cohort_end': int(cohort[-1]), 'returned_at_lock': int(sum((i in feedback for i in cohort))), 'adwin_inputs': int(detector_inputs), 'adwin_width': int(detector.width), 'adwin_mean': float(detector.estimation)})
        return True
    for t in range(base.MONITOR_START, simulation_end + 1):
        if phase == 'WAITING_READY' and cohort is not None and (t <= last_event):
            ready = all((i in feedback for i in cohort))
            if ready:
                ids = np.asarray(cohort, dtype=int)
                unique_classes = np.unique(y[ids])
                if len(unique_classes) < 2:
                    skipped_count += 1
                    update_rows.append({'alarm_time': alarm_time, 'update_time': t, 'old_version': version, 'result': 'SKIPPED_SINGLE_CLASS', 'train_size': 100, 'cohort_start': int(ids[0]), 'cohort_end': int(ids[-1])})
                    phase = 'MONITORING'
                    episode += 1
                    detector = ADWIN(delta=float(delta))
                    open_block = []
                    closed_blocks = []
                    cohort = None
                    alarm_latched = False
                    alarm_time = None
                    alarm_origin = None
                else:
                    new_model = base.create_model('ARF')
                    new_model.fit(X[ids], y[ids])
                    update_rows.append({'alarm_time': int(alarm_time), 'detector_alarm_time': int(alarm_origin), 'update_time': int(t), 'alarm_to_update': int(t - alarm_time), 'old_version': int(version), 'new_version': int(version + 1), 'result': 'UPDATED', 'train_size': 100, 'cohort_start': int(ids[0]), 'cohort_end': int(ids[-1]), 'train_class1_rate': float(np.mean(y[ids] == 1))})
                    model = new_model
                    version += 1
                    episode += 1
                    retrain_count += 1
                    reference = np.nan
                    phase = 'CALIBRATING'
                    detector = ADWIN(delta=float(delta))
                    calibration_ids = []
                    open_block = []
                    closed_blocks = []
                    cohort = None
                    alarm_time = None
                    alarm_origin = None
                    alarm_latched = False
        if t <= last_event:
            prediction = int(model.predict(X[t].reshape(1, -1))[0])
            predictions[t] = prediction
            prediction_version[t] = version
            prediction_episode[t] = episode
            prediction_phase[t] = phase
            if phase == 'MONITORING':
                monitor_events += 1
                open_block.append(t)
                if len(open_block) == 100:
                    closed_blocks.append({'ids': tuple(open_block), 'version': int(version), 'episode': int(episode), 'close_time': int(t), 'block_id': int(block_id)})
                    block_id += 1
                    open_block = []
            elif phase == 'WAITING_READY':
                waiting_events += 1
            elif phase == 'CALIBRATING':
                calibration_events += 1
                if len(calibration_ids) < 100:
                    calibration_ids.append(t)
        arrived = arrival_groups.get(t, ())
        for sample_id in arrived:
            if sample_id not in predictions:
                raise RuntimeError('Feedback without historical prediction.')
            error = int(predictions[sample_id] != int(y[sample_id]))
            feedback[sample_id] = error
            eligible = phase == 'MONITORING' and (not alarm_latched) and (prediction_phase[sample_id] == 'MONITORING') and (prediction_version[sample_id] == version) and (prediction_episode[sample_id] == episode)
            if not eligible:
                excluded_feedback += 1
                continue
            detector.add_element(error)
            detector_inputs += 1
            if detector.detected_change():
                alarm_count += 1
                alarm_latched = True
                if not closed_blocks:
                    deferred_count += 1
                    alarm_rows.append({'action': 'DEFERRED_NO_CLOSED_BLOCK', 'time': int(t), 'detector_alarm_time': int(t), 'version': int(version), 'episode': int(episode), 'adwin_inputs': int(detector_inputs), 'adwin_width': int(detector.width), 'adwin_mean': float(detector.estimation)})
                continue
        if phase == 'CALIBRATING' and len(calibration_ids) == 100:
            all_returned = all((i in feedback for i in calibration_ids))
            if all_returned:
                reference = float(np.mean([feedback[i] for i in calibration_ids]))
                phase = 'MONITORING'
                detector = ADWIN(delta=float(delta))
                open_block = []
                closed_blocks = []
                alarm_latched = False
        if phase == 'MONITORING' and alarm_latched:
            original_alert = t
            for event in reversed(alarm_rows):
                if event.get('action') == 'DEFERRED_NO_CLOSED_BLOCK' and event.get('version') == version and (event.get('episode') == episode):
                    original_alert = event['detector_alarm_time']
                    break
            lock_cohort(t, original_alert)
    online_ids = np.arange(base.MONITOR_START, len(y), dtype=int)
    y_true = y[online_ids]
    y_pred = np.asarray([predictions[int(i)] for i in online_ids], dtype=int)
    online_count = len(online_ids)
    total_phase_events = monitor_events + waiting_events + calibration_events
    if total_phase_events != online_count:
        raise RuntimeError('Phase coverage accounting failed.')
    row = {'dataset': family, 'dataset_family': family, 'stream_seed': int(stream_seed), 'stream_id': f'{family}_s{stream_seed}', 'scenario': scenario, 'lambda_class0': int(lambda0), 'lambda_class1': int(lambda1), 'delay_seed': int(delay_seed), 'learner': 'ARF', 'strategy': NEW, 'accuracy': float(accuracy_score(y_true, y_pred)), 'macro_f1': float(f1_score(y_true, y_pred, average='macro', zero_division=0)), 'mcc': float(matthews_corrcoef(y_true, y_pred)), 'retrain_count': int(retrain_count), 'alarm_count': int(alarm_count), 'update_count': int(retrain_count), 'skipped_single_class_updates': int(skipped_count), 'deferred_alarms': int(deferred_count), 'detector_feedback_used': int(detector_inputs), 'detector_feedback_excluded': int(excluded_feedback), 'monitor_coverage': float(monitor_events / online_count), 'waiting_coverage': float(waiting_events / online_count), 'calibration_coverage': float(calibration_events / online_count), 'initial_train_accuracy': float(train_acc), 'initial_cal_accuracy': float(cal_acc), 'initial_reference_loss': initial_reference, 'decision_accuracy': np.nan, 'decision_delay': np.nan, 'false_positive': np.nan, 'false_negative': np.nan, 'delta': float(delta), 'frozen_content_sha256': sha, 'stored_generator_hash': stored_hash, 'arf_random_seed': SEED, 'arf_backend': 'skmultiflow.AdaptiveRandomForestClassifier', 'task_elapsed_seconds': float(time.perf_counter() - started)}
    common = {k: row[k] for k in KEY}
    for r in alarm_rows:
        r.update(common)
    for r in update_rows:
        r.update(common)
    return (row, alarm_rows, update_rows)

def read_optional_csv(path):
    if not path.is_file() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()

def save_csv_atomic(df, filename):
    filename.parent.mkdir(parents=True, exist_ok=True)
    tmp = filename.with_suffix(filename.suffix + '.tmp')
    df.to_csv(tmp, index=False)
    os.replace(str(tmp), str(filename))

def holm_adjust(pvalues):
    p = np.asarray(pvalues, dtype=float)
    adjusted = np.zeros(len(p))
    running = 0.0
    for rank, pos in enumerate(np.argsort(p)):
        running = max(running, min(1.0, (len(p) - rank) * p[pos]))
        adjusted[pos] = running
    return adjusted

def analysis(old, new, tasks, delta):
    expected = {(t[0], t[1], t[2], t[5]) for t in tasks}
    observed = set(zip(new.dataset, new.stream_seed.astype(int), new.scenario, new.delay_seed.astype(int)))
    if len(new) != 1250 or new.duplicated(KEY).any() or observed != expected:
        raise RuntimeError('ADWIN output is incomplete.')
    merged = pd.concat([old, new], ignore_index=True, sort=False)
    if len(merged) != 6250 or merged.duplicated(KEY + ['strategy']).any():
        raise RuntimeError('Five-method merge failed.')
    merged['method_name'] = merged.strategy.map(NAMES)
    merged['backbone'] = 'ARF'
    save_csv_atomic(merged.sort_values(KEY + ['strategy']), OUT / 'five_methods_raw.csv')
    summary = merged.groupby('strategy', as_index=False).agg(tasks=('accuracy', 'size'), accuracy=('accuracy', 'mean'), macro_f1=('macro_f1', 'mean'), mcc=('mcc', 'mean'), updates=('retrain_count', 'mean'))
    summary['method'] = summary.strategy.map(NAMES)
    summary['sort'] = summary.strategy.map({m: i for i, m in enumerate(ORDER)})
    summary = summary.sort_values('sort').drop(columns='sort')
    save_csv_atomic(summary, OUT / 'main_paper_table.csv')
    stream = merged.groupby(['dataset', 'stream_seed', 'strategy'], as_index=False)[['accuracy', 'macro_f1', 'mcc']].mean()
    save_csv_atomic(stream, OUT / 'stream_level_mean.csv')
    records = []
    for metric in ('accuracy', 'macro_f1', 'mcc'):
        pivot = stream.pivot(index=['dataset', 'stream_seed'], columns='strategy', values=metric)
        for comparator in ORDER:
            if comparator == 'FULL_COMPLETE_COHORT':
                continue
            diff_series = (pivot['FULL_COMPLETE_COHORT'] - pivot[comparator]).dropna()
            if len(diff_series) != 50:
                raise RuntimeError('Expected 50 paired streams.')
            diff = diff_series.to_numpy(dtype=float)
            nonzero = diff[np.abs(diff) > 1e-12]
            p = float(wilcoxon(nonzero).pvalue) if len(nonzero) else 1.0
            rng = np.random.default_rng(30300)
            families = [diff_series.xs(family, level='dataset').to_numpy(dtype=float) for family in sorted(diff_series.index.get_level_values('dataset').unique())]
            bootstrap = np.empty(5000)
            for b in range(5000):
                sample = np.concatenate([arr[rng.integers(0, len(arr), len(arr))] for arr in families])
                bootstrap[b] = np.mean(sample)
            low, high = np.quantile(bootstrap, [0.025, 0.975])
            records.append({'metric': metric, 'comparator': NAMES[comparator], 'mean_difference': float(diff.mean()), 'ci95_low': float(low), 'ci95_high': float(high), 'wins': int(np.sum(diff > 1e-12)), 'ties': int(np.sum(np.abs(diff) <= 1e-12)), 'losses': int(np.sum(diff < -1e-12)), 'p_wilcoxon': p})
    tests = pd.DataFrame(records)
    tests['p_holm_12'] = holm_adjust(tests.p_wilcoxon.to_numpy())
    save_csv_atomic(tests, OUT / 'paired_tests_50_streams.csv')
    scenarios = new.groupby('scenario', as_index=False)[['accuracy', 'macro_f1', 'mcc', 'retrain_count', 'alarm_count']].mean()
    save_csv_atomic(scenarios, OUT / 'adwin_scenario_summary.csv')
    protocol = {'tasks_per_method': 1250, 'total_method_runs': 6250, 'independent_streams': 50, 'delay_seed_count': 5, 'detector': 'scikit-multiflow ADWIN', 'delta': float(delta), 'new_method': 'Delayed-ADWIN-Arrival', 'detector_input': 'Historical 0-1 prediction errors in label-arrival order. Only current monitoring episode and model version.', 'update_rule': 'Lock latest closed 100-event window, wait for all 100 labels, then refit ARF.', 'calibration': '100 new-version predictions, all labels must return.', 'oracle_limitation': 'ADWIN change detection is not fixed-window oracle classification.', 'evaluation_limitation': 'This external baseline was added after inspection of previous holdout results.'}
    (OUT / 'protocol.json').write_text(json.dumps(protocol, indent=2, ensure_ascii=False), encoding='utf-8')
    print()
    print('=' * 90)
    print('FIVE-STRATEGY COMMON-ARF COMPARISON')
    print('=' * 90)
    print(summary.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    print()
    print('Saved to:', OUT)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--jobs', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--delta', type=float, default=0.002)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--merge-only', action='store_true')
    args = parser.parse_args()
    if args.jobs < 1 or args.batch_size < 1 or (not 0 < args.delta < 1):
        parser.error('Invalid jobs, batch-size or delta.')
    OUT.mkdir(parents=True, exist_ok=True)
    s27, engine = load_modules()
    tasks = tasks_from_protocol(s27, engine)
    old = ensure_old_results(s27, tasks)
    raw_file = OUT / 'adwin_raw.csv'
    alarm_file = OUT / 'alarm_log.csv'
    update_file = OUT / 'update_log.csv'
    if args.smoke:
        row, alarms, updates = adwin_one(tasks[0], args.delta)
        smoke_dir = OUT / 'smoke'
        save_csv_atomic(pd.DataFrame([row]), smoke_dir / 'adwin_raw.csv')
        save_csv_atomic(pd.DataFrame(alarms), smoke_dir / 'alarm_log.csv')
        save_csv_atomic(pd.DataFrame(updates), smoke_dir / 'update_log.csv')
        print('SMOKE TEST OK')
        print(row)
        return
    if not args.merge_only:
        existing = read_optional_csv(raw_file)
        if len(existing):
            if existing.duplicated(KEY).any():
                raise RuntimeError('Duplicate checkpoint tasks.')
            if set(existing.strategy) != {NEW}:
                raise RuntimeError('Unexpected checkpoint strategy.')
            old_delta = existing.delta.astype(float)
            if (old_delta - args.delta).abs().max() > 1e-12:
                raise RuntimeError('Checkpoint delta differs. Do not mix experiments.')
            completed = set(zip(existing.dataset, existing.stream_seed.astype(int), existing.scenario, existing.delay_seed.astype(int)))
        else:
            completed = set()
        pending = [t for t in tasks if (t[0], t[1], t[2], t[5]) not in completed]
        print(f'Completed: {len(completed)}/1250')
        print(f'Remaining: {len(pending)}')
        old_alarms = read_optional_csv(alarm_file)
        old_updates = read_optional_csv(update_file)
        if len(old_alarms):
            old_alarms = old_alarms[old_alarms[KEY].apply(tuple, axis=1).isin(completed)].copy()
        if len(old_updates):
            old_updates = old_updates[old_updates[KEY].apply(tuple, axis=1).isin(completed)].copy()
        start = time.perf_counter()
        for offset in range(0, len(pending), args.batch_size):
            batch = pending[offset:offset + args.batch_size]
            results = Parallel(n_jobs=args.jobs, backend='loky', verbose=5)((delayed(adwin_one)(task, args.delta) for task in batch))
            new_rows = pd.DataFrame([result[0] for result in results])
            new_alarms = pd.DataFrame([entry for result in results for entry in result[1]])
            new_updates = pd.DataFrame([entry for result in results for entry in result[2]])
            existing = pd.concat([existing, new_rows], ignore_index=True)
            old_alarms = pd.concat([old_alarms, new_alarms], ignore_index=True)
            old_updates = pd.concat([old_updates, new_updates], ignore_index=True)
            save_csv_atomic(old_alarms, alarm_file)
            save_csv_atomic(old_updates, update_file)
            save_csv_atomic(existing.sort_values(KEY), raw_file)
            elapsed = time.perf_counter() - start
            print(f'Saved {len(existing)}/1250 ADWIN tasks; elapsed {elapsed:.1f}s', flush=True)
    if not raw_file.is_file():
        raise FileNotFoundError(raw_file)
    new = pd.read_csv(raw_file)
    analysis(old, new, tasks, args.delta)
if __name__ == '__main__':
    main()
