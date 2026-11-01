"""Partition monitored event mass by terminal monitoring state."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
SRC = Path('result/confirmation_comparison')
STREAMS = Path('result/unseen_stream_pipeline/streams')
OUT = Path('result/coverage_accounting')
STRATEGY = 'FULL_COMPLETE_COHORT'
START = 400
W = 100
CAL = 100
MON_OPEN = 0
WAIT = 1
CALIB = 2
RES_ONLINE = 3
RES_POST = 4
STALE = 5
PENDING = 6
PART_ALARM = 7
PART_END = 8
NAMES = {MON_OPEN: 'UNASSIGNED_MONITORING', WAIT: 'WAITING_COHORT', CALIB: 'CALIBRATING', RES_ONLINE: 'RESOLVED_ONLINE', RES_POST: 'RESOLVED_POST_STREAM', STALE: 'CLOSED_STALE', PENDING: 'CLOSED_UNRESOLVED_END', PART_ALARM: 'PARTIAL_DISCARDED_ALARM', PART_END: 'PARTIAL_STREAM_END'}
MON_TYPES = (RES_ONLINE, RES_POST, STALE, PENDING, PART_ALARM, PART_END)
KEYS = ['dataset', 'stream_seed', 'learner', 'scenario', 'delay_seed']

def check_columns(df, required, name):
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f'{name}: missing columns {missing}')

def read_input():
    required_files = ['raw.csv', 'decision_raw.csv', 'update_raw.csv']
    for name in required_files:
        if not (SRC / name).is_file():
            raise FileNotFoundError(f'Missing {SRC / name}')
    runs = pd.read_csv(SRC / 'raw.csv', encoding='utf-8-sig')
    decisions = pd.read_csv(SRC / 'decision_raw.csv', encoding='utf-8-sig')
    updates = pd.read_csv(SRC / 'update_raw.csv', encoding='utf-8-sig')
    runs = runs[runs.strategy == STRATEGY].copy()
    decisions = decisions[decisions.strategy == STRATEGY].copy()
    updates = updates[updates.strategy == STRATEGY].copy()
    check_columns(runs, KEYS + ['decision_count', 'monitor_coverage', 'skipped_single_class_updates', 'lambda0', 'lambda1', 'stream_hash'], 'confirmation comparison runs')
    check_columns(decisions, KEYS + ['decision_time', 'block_id', 'block_close_time', 'model_version', 'decision'], 'confirmation comparison decisions')
    check_columns(updates, KEYS + ['update_time', 'block_id', 'alarm_time'], 'confirmation comparison updates')
    if runs.duplicated(KEYS).any():
        raise AssertionError('Duplicate confirmation comparison task keys')
    return (runs, decisions, updates)

def get_labels_and_arrivals(run):
    path = STREAMS / f"{run['dataset']}_streamseed_{int(run['stream_seed'])}.npz"
    if not path.is_file():
        raise FileNotFoundError(f'Frozen stream missing: {path}')
    with np.load(path, allow_pickle=False) as data:
        y = np.asarray(data['y'], dtype=np.int8)
        X = np.asarray(data['X'])
    digest = hashlib.sha256(np.ascontiguousarray(X).tobytes() + np.ascontiguousarray(y.astype(int)).tobytes()).hexdigest()
    if digest != str(run['stream_hash']):
        digest = hashlib.sha256(np.ascontiguousarray(np.asarray(X, dtype=float)).tobytes() + np.ascontiguousarray(np.asarray(y, dtype=int)).tobytes()).hexdigest()
    if digest != str(run['stream_hash']):
        raise AssertionError(f'Stream SHA256 mismatch: {path}')
    if len(y) <= START:
        raise ValueError('Frozen stream has no online events')
    rng = np.random.default_rng(int(run['delay_seed']))
    lam0 = int(run['lambda0'])
    lam1 = int(run['lambda1'])
    delays = np.fromiter((int(rng.poisson(lam0 if int(label) == 0 else lam1)) for label in y[START:]), dtype=np.int64, count=len(y) - START)
    arrival = np.full(len(y), -1, dtype=np.int64)
    arrival[START:] = np.arange(START, len(y), dtype=np.int64) + delays
    return (y, arrival)

def audit_one(run, decision_rows, update_rows):
    y, arrival = get_labels_and_arrivals(run)
    last = len(y) - 1
    simulation_end = max(last, int(arrival[START:].max()))
    categories = np.full(len(y), -1, dtype=np.int8)
    phase = 'MONITORING'
    version = 0
    open_buffer = []
    blocks = []
    calibration_ids = []
    decision_by_time = {}
    update_by_time = {}
    for record in decision_rows:
        t = int(record['decision_time'])
        decision_by_time.setdefault(t, []).append(record)
    for t in decision_by_time:
        decision_by_time[t].sort(key=lambda x: int(x['block_id']))
    for record in update_rows:
        t = int(record['update_time'])
        update_by_time.setdefault(t, []).append(record)
    for t, records in update_by_time.items():
        if len(records) != 1:
            raise AssertionError(f'Duplicate updates at time {t}')
    if int(run['skipped_single_class_updates']) > 0:
        raise RuntimeError('Unlogged skipped updates detected. An explicit skip-event log is required.')
    decisions_seen = 0
    updates_seen = 0
    invalid_decisions = 0
    for t in range(START, simulation_end + 1):
        if t in update_by_time:
            if t > last or phase != 'WAITING_READY':
                raise AssertionError(f'Unexpected update at {t}, phase={phase}')
            version += 1
            updates_seen += 1
            phase = 'CALIBRATING'
            calibration_ids = []
            open_buffer = []
        if t <= last:
            if phase == 'MONITORING':
                categories[t] = MON_OPEN
                open_buffer.append(t)
                if len(open_buffer) == W:
                    blocks.append({'id': len(blocks), 'ids': list(open_buffer), 'close': t, 'version': version, 'status': 'PENDING'})
                    open_buffer = []
            elif phase == 'WAITING_READY':
                categories[t] = WAIT
            elif phase == 'CALIBRATING':
                categories[t] = CALIB
                if len(calibration_ids) < CAL:
                    calibration_ids.append(t)
            else:
                raise RuntimeError(f'Unknown phase: {phase}')
        if phase == 'CALIBRATING' and len(calibration_ids) == CAL:
            cal_array = np.asarray(calibration_ids, dtype=int)
            if np.all(arrival[cal_array] <= t):
                phase = 'MONITORING'
                calibration_ids = []
                open_buffer = []
        current_decisions = decision_by_time.get(t, [])
        if current_decisions and phase != 'MONITORING':
            raise AssertionError(f'Decision in invalid phase: {t}')
        alarm_seen = False
        for record in current_decisions:
            block_id = int(record['block_id'])
            if block_id < 0 or block_id >= len(blocks):
                raise AssertionError(f'Nonexistent block {block_id}')
            block = blocks[block_id]
            if block['status'] != 'PENDING' or block['close'] != int(record['block_close_time']):
                raise AssertionError(f'Block mismatch: {block_id}')
            if block['version'] != int(record['model_version']):
                raise AssertionError(f'Block version mismatch: {block_id}')
            if block['version'] != version:
                invalid_decisions += 1
            if alarm_seen:
                raise AssertionError('Decision after alarm at same time')
            block['status'] = 'DECIDED'
            if t <= last:
                categories[block['ids']] = RES_ONLINE
            else:
                categories[block['ids']] = RES_POST
            decisions_seen += 1
            if record['decision'] == 'DEGRADATION':
                alarm_seen = True
                phase = 'WAITING_READY'
                if open_buffer:
                    categories[open_buffer] = PART_ALARM
                    open_buffer = []
                for other in blocks:
                    if other['status'] == 'PENDING':
                        other['status'] = 'STALE'
                        categories[other['ids']] = STALE
    if open_buffer:
        categories[open_buffer] = PART_END
    for block in blocks:
        if block['status'] == 'PENDING':
            block['status'] = 'UNRESOLVED_END'
            categories[block['ids']] = PENDING
    online = categories[START:]
    if np.any(online < 0) or np.any(online == MON_OPEN):
        raise AssertionError('At least one online event was unaccounted for')
    counts = np.bincount(online.astype(np.int64), minlength=len(NAMES))
    n_online = len(online)
    monitoring = int(sum((counts[c] for c in MON_TYPES)))
    resolved = int(counts[RES_ONLINE] + counts[RES_POST])
    if monitoring + int(counts[WAIT]) + int(counts[CALIB]) != n_online:
        raise AssertionError('Phase partition does not sum to total')
    if decisions_seen != int(run['decision_count']):
        raise AssertionError('confirmation comparison decision-count mismatch')
    if abs(monitoring / n_online - float(run['monitor_coverage'])) > 1e-10:
        raise AssertionError('confirmation comparison monitoring-coverage mismatch')
    if updates_seen != int(run['retrain_count']):
        raise AssertionError('confirmation comparison update-count mismatch')
    if invalid_decisions:
        raise AssertionError('Invalid-version decision in full protocol')
    if resolved != decisions_seen * W:
        raise AssertionError('Resolved windows overlap or have wrong size')
    result = {key: run[key] for key in KEYS}
    result.update({'n_online': n_online, 'simulation_end': simulation_end, 'monitor_samples': monitoring, 'decision_count': decisions_seen, 'closed_blocks': len(blocks), 'updates': updates_seen, 'monitor_coverage': monitoring / n_online, 'resolved_coverage': resolved / n_online, 'resolved_online_coverage': int(counts[RES_ONLINE]) / n_online, 'resolved_post_stream_coverage': int(counts[RES_POST]) / n_online, 'unresolved_collected_coverage': (monitoring - resolved) / n_online})
    for category, name in NAMES.items():
        if category != MON_OPEN:
            result[f'N_{name}'] = int(counts[category])
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true', help='Audit the first five runs')
    parser.add_argument('--jobs', type=int, default=16)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error('--jobs must be >= 1')
    run_df, decision_df, update_df = read_input()
    if args.smoke:
        run_df = run_df.head(5).copy()
    decision_groups = {key: group.to_dict('records') for key, group in decision_df.groupby(KEYS, sort=False)}
    update_groups = {key: group.to_dict('records') for key, group in update_df.groupby(KEYS, sort=False)}
    tasks = []
    for run in run_df.to_dict('records'):
        key = tuple((run[k] for k in KEYS))
        tasks.append((run, decision_groups.get(key, []), update_groups.get(key, [])))
    print(f'coverage accounting: auditing {len(tasks)} original confirmation comparison runs, no model training', flush=True)
    results = Parallel(n_jobs=min(args.jobs, len(tasks)), prefer='processes', verbose=5)((delayed(audit_one)(*task) for task in tasks))
    out = pd.DataFrame(results)
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT / 'run_audit.csv', index=False)
    total = int(out['n_online'].sum())
    category_rows = []
    for category, name in NAMES.items():
        if category == MON_OPEN:
            continue
        count = int(out[f'N_{name}'].sum())
        category_rows.append({'category': name, 'events': count, 'fraction': count / total, 'percentage': 100 * count / total})
    category_df = pd.DataFrame(category_rows)
    if int(category_df.events.sum()) != total:
        raise AssertionError('Event categories do not sum to total')
    category_df.to_csv(OUT / 'event_mass_decomposition.csv', index=False)
    summary = {'runs': len(out), 'online_events': total, 'collected_coverage_pct': 100 * out.monitor_samples.sum() / total, 'resolved_coverage_pct': 100 * out.decision_count.sum() * W / total, 'resolved_online_pct': 100 * out.N_RESOLVED_ONLINE.sum() / total, 'resolved_post_stream_pct': 100 * out.N_RESOLVED_POST_STREAM.sum() / total, 'collected_unresolved_pct': 100 * (out.monitor_samples.sum() - out.decision_count.sum() * W) / total, 'not_collected_pct': 100 * (total - out.monitor_samples.sum()) / total, 'decisions': int(out.decision_count.sum()), 'updates': int(out.updates.sum())}
    pd.DataFrame([summary]).to_csv(OUT / 'main_si_table.csv', index=False)
    scenario = out.groupby(['dataset', 'learner', 'scenario'], as_index=False).agg(runs=('n_online', 'size'), online_events=('n_online', 'sum'), monitored=('monitor_samples', 'sum'), decisions=('decision_count', 'sum'), resolved_online=('N_RESOLVED_ONLINE', 'sum'), resolved_post_stream=('N_RESOLVED_POST_STREAM', 'sum'), stale=('N_CLOSED_STALE', 'sum'), waiting=('N_WAITING_COHORT', 'sum'), calibrating=('N_CALIBRATING', 'sum'))
    for new_column, source_column in [('monitor_coverage_pct', 'monitored'), ('resolved_online_pct', 'resolved_online'), ('post_stream_pct', 'resolved_post_stream'), ('stale_pct', 'stale'), ('waiting_pct', 'waiting'), ('calibrating_pct', 'calibrating')]:
        scenario[new_column] = 100 * scenario[source_column] / scenario.online_events
    scenario.to_csv(OUT / 'scenario_decomposition.csv', index=False)
    protocol = {**summary, 'unit': 'Event mass: each online event is assigned exactly one mutually exclusive terminal state.', 'post_stream': 'Decision time exceeds last input-event time; delayed labels may still be released.', 'limitations': 'No full-stream drift recall is inferred. Unresolved status is not itself evidence of an actual degradation.'}
    (OUT / 'protocol.json').write_text(json.dumps(protocol, indent=2, ensure_ascii=False), encoding='utf-8')
    print('\ncoverage accounting MAIN SI TABLE\n')
    print(pd.DataFrame([summary]).to_string(index=False, float_format=lambda v: f'{v:.4f}'))
    print('\nEVENT MASS DECOMPOSITION\n')
    print(category_df.to_string(index=False, float_format=lambda v: f'{v:.4f}'))
    print('\nResults saved in:', OUT)
if __name__ == '__main__':
    main()
