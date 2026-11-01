"""Evaluate protocol mechanisms under frequent synthetic changes."""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import importlib
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
step31 = importlib.import_module('protocol_ablation')
OUT = Path('result/dense_drift_audit')
VARIANTS = ('FULL_PROTOCOL', 'SINGLE_PENDING', 'NO_STALE_GUARD')
DELAYS = ('POISSON_150', 'ASYMMETRIC_50_300', 'BATCHED_200')

def dense_stream(y, period):
    """
    Controlled synthetic stress intervention.

    Keep X unchanged.
    Keep initial training/calibration labels unchanged.
    Alternately invert the binary concept every 'period'
    online events.
    """
    result = np.array(y, copy=True, dtype=np.int64)
    start = step31.START
    online = np.arange(start, len(result), dtype=np.int64)
    flip = ((online - start) // period % 2).astype(np.int64)
    result[start:] ^= flip
    return result

def delay_schedule(y, seed, mode):
    if mode == 'POISSON_150':
        return step31.legacy.generate_delay(y, seed, 150, 150)
    if mode == 'ASYMMETRIC_50_300':
        return step31.legacy.generate_delay(y, seed, 50, 300)
    if mode == 'BATCHED_200':
        base, _ = step31.legacy.generate_delay(y, seed, 100, 100)
        size = 200
        start = step31.START
        arrivals = {int(i): int(start + size * ((int(a) - start + size - 1) // size)) for i, a in base.items()}
        groups = {}
        for i, a in arrivals.items():
            if a < i:
                raise AssertionError('Noncausal label arrival')
            groups.setdefault(a, []).append(i)
        for a in groups:
            groups[a].sort()
        return (arrivals, groups)
    raise ValueError(mode)

def first_alarm(events):
    alarms = [e for e in events if e['event'] == 'ALARM']
    if not alarms:
        return None
    a = min(alarms, key=lambda x: (x['time'], x['block_id']))
    return (int(a['block_id']), int(a['time']))

def work(task):
    dataset, stream_seed, learner, period, delay_mode, delay_seed = task
    X, original_y, stream_hash = step31.read_stream(dataset, stream_seed)
    y = dense_stream(original_y, period)
    arrivals, groups = delay_schedule(y, delay_seed, delay_mode)
    meta = {'dataset': dataset, 'stream_seed': stream_seed, 'learner': learner, 'flip_period': period, 'delay_mode': delay_mode, 'delay_seed': delay_seed, 'base_stream_hash': stream_hash}
    runs = []
    decisions_all = []
    events_all = []
    first_alarms = {}
    for variant in VARIANTS:
        run, decisions, events = step31.run_variant(X, y, arrivals, groups, learner, variant, meta)
        valid = sum((int(d['block_version'] == d['current_version']) for d in decisions))
        invalid = len(decisions) - valid
        valid_correct = sum((int(d['correct']) for d in decisions if d['block_version'] == d['current_version']))
        run.update({'version_valid_decisions': valid, 'version_invalid_decisions': invalid, 'version_valid_coverage': valid * step31.W / run['n_online'], 'version_valid_oracle_agreement': valid_correct / valid if valid else float('nan'), 'all_decision_coverage': len(decisions) * step31.W / run['n_online'], 'alarm_events': sum((e['event'] == 'ALARM' for e in events)), 'update_events': sum((e['event'] == 'UPDATE' for e in events))})
        first_alarms[variant] = first_alarm(events)
        runs.append(run)
        decisions_all.extend(decisions)
        events_all.extend(events)
    baseline = runs[0]
    paired = []
    for x in runs[1:]:
        variant = x['variant']
        alarm_a = first_alarms[variant]
        alarm_b = first_alarms['FULL_PROTOCOL']
        first_block_same = alarm_a is None and alarm_b is None or (alarm_a is not None and alarm_b is not None and (alarm_a[0] == alarm_b[0]))
        paired.append({**meta, 'variant': variant, 'first_alarm_block_agreement': int(first_block_same), 'first_alarm_time_agreement': int(alarm_a == alarm_b), 'delta_accuracy': x['accuracy'] - baseline['accuracy'], 'delta_mcc': x['mcc'] - baseline['mcc'], 'delta_valid_coverage': x['version_valid_coverage'] - baseline['version_valid_coverage'], 'delta_decisions': x['decisions'] - baseline['decisions'], 'delta_updates': x['retrain_count'] - baseline['retrain_count']})
    if baseline['fp'] != 0 or baseline['fn'] != 0 or baseline['stale_update_violations'] != 0 or (baseline['historical_loss_mismatch'] != 0) or (baseline['version_invalid_decisions'] != 0):
        raise AssertionError(f'Full protocol invariant failed: {meta}')
    return (runs, decisions_all, events_all, paired)

def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--smoke', action='store_true')
    mode.add_argument('--full', action='store_true')
    parser.add_argument('--jobs', type=int, default=16)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error('--jobs must be at least 1')
    if args.smoke:
        datasets = ['SEA']
        stream_seeds = [6001]
        learners = ['LR']
        periods = [100]
        delays = ['POISSON_150']
        delay_seeds = [7001]
    else:
        datasets = ['SEA', 'Agrawal', 'Mixed', 'RandomTree', 'Sine']
        stream_seeds = list(range(6001, 6006)) if args.full else [6001, 6002]
        learners = ['LR', 'DT']
        periods = [100, 200, 300]
        delays = list(DELAYS)
        delay_seeds = [7001, 7002, 7003] if args.full else [7001, 7002]
    tasks = [(d, s, l, p, m, k) for d in datasets for s in stream_seeds for l in learners for p in periods for m in delays for k in delay_seeds]
    OUT.mkdir(parents=True, exist_ok=True)
    print(f'dense-change audit: {len(tasks)} matched tasks, {len(tasks) * len(VARIANTS)} runs', flush=True)
    start_time = time.perf_counter()
    result = Parallel(n_jobs=min(args.jobs, len(tasks)), verbose=5)((delayed(work)(task) for task in tasks))
    run_df = pd.DataFrame([row for runs, _, _, _ in result for row in runs])
    decision_df = pd.DataFrame([row for _, decisions, _, _ in result for row in decisions])
    event_df = pd.DataFrame([row for _, _, events, _ in result for row in events])
    paired_df = pd.DataFrame([row for _, _, _, paired in result for row in paired])
    run_df.to_csv(OUT / 'run_raw.csv', index=False)
    decision_df.to_csv(OUT / 'decision_raw.csv', index=False)
    event_df.to_csv(OUT / 'event_log.csv', index=False)
    paired_df.to_csv(OUT / 'paired_vs_full.csv', index=False)
    records = []
    for variant in VARIANTS:
        df = run_df[run_df['variant'] == variant]
        n_decisions = int(df['decisions'].sum())
        n_valid = int(df['version_valid_decisions'].sum())
        n_online = int(df['n_online'].sum())
        records.append({'variant': variant, 'runs': len(df), 'decisions': n_decisions, 'version_valid_decisions': n_valid, 'version_invalid_decisions': int(df['version_invalid_decisions'].sum()), 'valid_coverage_pct': 100 * n_valid * step31.W / n_online, 'all_decision_coverage_pct': 100 * n_decisions * step31.W / n_online, 'mean_accuracy': float(df['accuracy'].mean()), 'mean_mcc': float(df['mcc'].mean()), 'mean_confirmation_delay': float(df['mean_confirmation_delay'].mean()), 'max_pending_observed': int(df['max_pending'].max()), 'runs_with_multi_pending': int((df['max_pending'] > 1).sum()), 'mean_multi_pending_event_steps': float(df['multi_pending_event_steps'].mean()), 'total_stale_trigger_violations': int(df['stale_trigger_violations'].sum()), 'total_stale_update_violations': int(df['stale_update_violations'].sum()), 'total_single_pending_drops': int(df['dropped_single_pending_windows'].sum()), 'total_updates': int(df['retrain_count'].sum()), 'total_fp': int(df['fp'].sum()), 'total_fn': int(df['fn'].sum())})
    summary = pd.DataFrame(records)
    summary.to_csv(OUT / 'main_si_table.csv', index=False)
    scenario_df = run_df.groupby(['flip_period', 'delay_mode', 'variant'], as_index=False).agg(runs=('accuracy', 'size'), mean_accuracy=('accuracy', 'mean'), mean_valid_coverage=('version_valid_coverage', 'mean'), mean_max_pending=('max_pending', 'mean'), total_stale_updates=('stale_update_violations', 'sum'))
    scenario_df.to_csv(OUT / 'scenario_table.csv', index=False)
    protocol = {'source': 'protocol ablation original execution protocol and synthetic-stream pipeline frozen streams', 'experiment': 'Controlled synthetic dense concept inversions with delayed feedback', 'flip_periods': periods, 'delay_patterns': delays, 'matched_tasks': len(tasks), 'variant_runs': len(run_df), 'window_size': step31.W, 'epsilon': step31.EPS, 'runtime_seconds': round(time.perf_counter() - start_time, 3), 'notes': 'Fault-injection controls are not normal competing algorithms. Version-valid coverage excludes decisions made under mismatched model versions.'}
    (OUT / 'protocol.json').write_text(json.dumps(protocol, indent=2, ensure_ascii=False), encoding='utf-8')
    print('\ndense-change audit MAIN SI TABLE\n')
    print(summary.to_string(index=False, float_format=lambda v: f'{v:.5f}'))
    print('\nResults saved in:', OUT)
    print('Elapsed:', protocol['runtime_seconds'], 'seconds')
if __name__ == '__main__':
    main()
