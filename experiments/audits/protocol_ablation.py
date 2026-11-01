"""Evaluate pending-window, version-authority and prediction-history mechanisms."""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import hashlib
import importlib
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import accuracy_score, matthews_corrcoef
legacy = importlib.import_module('chronological_protocol')
W = 100
EPS = 0.02
TOL = 1e-12
START = 400
CAL = 100
assert legacy.WINDOW_SIZE == W
assert legacy.POST_UPDATE_CAL_SIZE == CAL
assert abs(legacy.EPSILON - EPS) < 1e-14
OUT = Path('result/protocol_ablation')
STREAM_DIR = Path('result/unseen_stream_pipeline/streams')
VARIANTS = ('FULL_PROTOCOL', 'SINGLE_PENDING', 'NO_STALE_GUARD', 'RECOMPUTED_HISTORY')
SCENARIOS = {'UNIFORM_100_100': (100, 100), 'MILD_50_100': (50, 100), 'MEDIUM_30_150': (30, 150), 'SEVERE_20_200': (20, 200), 'REVERSED_200_20': (200, 20)}

def read_stream(dataset, stream_seed):
    path = STREAM_DIR / f'{dataset}_streamseed_{stream_seed}.npz'
    if not path.is_file():
        raise FileNotFoundError(f'Missing frozen stream: {path}\nRun synthetic-stream pipeline first, or restore the saved synthetic-stream pipeline streams from your project ZIP.')
    with np.load(path, allow_pickle=False) as data:
        X = np.asarray(data['X'], dtype=float)
        y = np.asarray(data['y'], dtype=int)
    if len(X) != len(y) or len(y) <= START or set(np.unique(y)) != {0, 1}:
        raise ValueError(f'Invalid stream: {path}')
    digest = hashlib.sha256(np.ascontiguousarray(X).tobytes() + np.ascontiguousarray(y).tobytes()).hexdigest()
    return (X, y, digest)

def decide(block, feedback):
    ids = block['ids']
    losses = feedback[ids]
    observed = losses[losses >= 0]
    m = len(observed)
    S = int(observed.sum())
    lo = S / W
    hi = (S + W - m) / W
    tau = block['reference'] + EPS
    if lo > tau + TOL:
        label = 'DEGRADATION'
    elif hi <= tau + TOL:
        label = 'TOLERABLE'
    else:
        return None
    return (label, m, S, lo, hi, tau)

def run_variant(X, y, arrivals, groups, learner, variant, meta):
    model, ref, _, _ = legacy.create_initial_model(X, y, learner)
    end = len(y) - 1
    simulation_end = max(end, max(arrivals.values()))
    original_pred = np.full(len(y), -1, dtype=np.int16)
    returned_loss = np.full(len(y), -1, dtype=np.int16)
    phase = 'MONITORING'
    version = 0
    updates = 0
    blocks = []
    open_ids = []
    cal_ids = []
    locked = None
    alarm = None
    trigger_block = None
    decisions = []
    events = []
    block_serial = 0
    mismatch = 0
    stale_triggers = 0
    stale_updates = 0
    single_dropped = 0
    stale_revocations = 0
    max_pending = 0
    multi_pending_events = 0
    monitor_samples = 0
    wait_samples = 0
    calibration_samples = 0
    skipped_updates = 0
    for t in range(START, simulation_end + 1):
        if phase == 'WAITING_READY' and t <= end and all((returned_loss[j] >= 0 for j in locked)):
            ids = np.asarray(locked, dtype=int)
            if len(np.unique(y[ids])) < 2:
                skipped_updates += 1
                events.append({**meta, 'variant': variant, 'time': t, 'event': 'SKIP_SINGLE_CLASS', 'block_id': trigger_block, 'old_version': version, 'new_version': version})
                phase = 'MONITORING'
                locked = None
                alarm = None
                trigger_block = None
                open_ids = []
            else:
                old_version = version
                model = legacy.create_model(learner)
                model.fit(X[ids], y[ids])
                version += 1
                updates += 1
                if trigger_block is not None:
                    origin = next((b for b in blocks if b['id'] == trigger_block))
                    if origin['version'] < old_version:
                        stale_updates += 1
                events.append({**meta, 'variant': variant, 'time': t, 'event': 'UPDATE', 'block_id': trigger_block, 'old_version': old_version, 'new_version': version})
                ref = float('nan')
                phase = 'CALIBRATING'
                locked = None
                alarm = None
                trigger_block = None
                cal_ids = []
                open_ids = []
        if t <= end:
            prediction = int(model.predict(X[t:t + 1])[0])
            original_pred[t] = prediction
            if phase == 'MONITORING':
                monitor_samples += 1
                open_ids.append(t)
                if len(open_ids) == W:
                    pending_before = sum((b['status'] == 'PENDING' for b in blocks))
                    block = {'id': block_serial, 'ids': np.array(open_ids, dtype=int), 'version': version, 'reference': float(ref), 'close': t, 'status': 'PENDING'}
                    block_serial += 1
                    if variant == 'SINGLE_PENDING' and pending_before:
                        block['status'] = 'DROPPED_SINGLE_PENDING'
                        single_dropped += 1
                    blocks.append(block)
                    open_ids = []
            elif phase == 'CALIBRATING':
                calibration_samples += 1
                if len(cal_ids) < CAL:
                    cal_ids.append(t)
            elif phase == 'WAITING_READY':
                wait_samples += 1
        arriving = groups.get(t, [])
        if arriving:
            ids = np.array(arriving, dtype=int)
            if np.any(original_pred[ids] < 0):
                raise RuntimeError('Arrival preceded historical prediction.')
            if variant == 'RECOMPUTED_HISTORY':
                used_pred = np.asarray(model.predict(X[ids]), dtype=int)
                mismatch += int(np.count_nonzero(used_pred != original_pred[ids]))
            else:
                used_pred = original_pred[ids]
            returned_loss[ids] = (used_pred != y[ids]).astype(np.int16)
        if phase == 'CALIBRATING' and len(cal_ids) == CAL:
            ci = np.array(cal_ids, dtype=int)
            if np.all(returned_loss[ci] >= 0):
                ref = float(returned_loss[ci].mean())
                phase = 'MONITORING'
                cal_ids = []
                open_ids = []
        if phase == 'MONITORING':
            for block in blocks:
                if block['status'] != 'PENDING':
                    continue
                value = decide(block, returned_loss)
                if value is None:
                    continue
                label, m, S, lo, hi, tau = value
                true_loss = float(np.mean(original_pred[block['ids']] != y[block['ids']]))
                oracle = 'DEGRADATION' if true_loss > tau + TOL else 'TOLERABLE'
                is_stale = block['version'] != version
                decisions.append({**meta, 'variant': variant, 'block_id': block['id'], 'block_version': block['version'], 'current_version': version, 'block_close': block['close'], 'decision_time': t, 'decision_delay': t - block['close'], 'decision': label, 'oracle': oracle, 'correct': int(label == oracle), 'returned_labels': m, 'S': S, 'lower': lo, 'upper': hi, 'threshold': tau, 'stale_window': int(is_stale)})
                block['status'] = 'DONE'
                if label == 'DEGRADATION':
                    if is_stale:
                        stale_triggers += 1
                    locked = list(map(int, block['ids']))
                    alarm = t
                    trigger_block = block['id']
                    phase = 'WAITING_READY'
                    open_ids = []
                    events.append({**meta, 'variant': variant, 'time': t, 'event': 'ALARM', 'block_id': block['id'], 'old_version': block['version'], 'new_version': version})
                    if variant != 'NO_STALE_GUARD':
                        for other in blocks:
                            if other['status'] == 'PENDING':
                                other['status'] = 'STALE'
                                stale_revocations += 1
                    break
                if variant == 'SINGLE_PENDING':
                    break
        if t <= end:
            n_pending = sum((b['status'] == 'PENDING' for b in blocks))
            max_pending = max(max_pending, n_pending)
            multi_pending_events += int(n_pending >= 2)
    target = np.arange(START, len(y), dtype=int)
    if np.any(original_pred[target] < 0):
        raise RuntimeError('Incomplete chronological prediction log.')
    n_decisions = len(decisions)
    errors = sum((1 - d['correct'] for d in decisions))
    run = {**meta, 'variant': variant, 'stream_length': len(y), 'n_online': len(target), 'accuracy': float(accuracy_score(y[target], original_pred[target])), 'mcc': float(matthews_corrcoef(y[target], original_pred[target])), 'decisions': n_decisions, 'correct_decisions': n_decisions - errors, 'fp': sum((d['decision'] == 'DEGRADATION' and d['oracle'] == 'TOLERABLE' for d in decisions)), 'fn': sum((d['decision'] == 'TOLERABLE' and d['oracle'] == 'DEGRADATION' for d in decisions)), 'oracle_agreement': (n_decisions - errors) / n_decisions if n_decisions else float('nan'), 'mean_confirmation_delay': float(np.mean([d['decision_delay'] for d in decisions])) if n_decisions else float('nan'), 'retrain_count': updates, 'skipped_updates': skipped_updates, 'collected_fraction': monitor_samples / len(target), 'resolved_fraction': n_decisions * W / len(target), 'closed_windows': len(blocks), 'dropped_single_pending_windows': single_dropped, 'max_pending': max_pending, 'multi_pending_event_steps': multi_pending_events, 'stale_revoked_windows': stale_revocations, 'stale_trigger_violations': stale_triggers, 'stale_update_violations': stale_updates, 'historical_loss_mismatch': mismatch, 'waiting_event_steps': wait_samples, 'calibration_event_steps': calibration_samples}
    if variant == 'FULL_PROTOCOL':
        assert errors == 0
        assert stale_triggers == 0
        assert stale_updates == 0
        assert mismatch == 0
    if variant == 'SINGLE_PENDING':
        assert max_pending <= 1
    if variant == 'NO_STALE_GUARD':
        assert mismatch == 0
    return (run, decisions, events)

def work(task):
    dataset, stream_seed, learner, scenario, delay_seed = task
    X, y, digest = read_stream(dataset, stream_seed)
    lam0, lam1 = SCENARIOS[scenario]
    arrivals, groups = legacy.generate_delay(y, delay_seed, lam0, lam1)
    meta = {'dataset': dataset, 'stream_seed': stream_seed, 'learner': learner, 'scenario': scenario, 'delay_seed': delay_seed, 'stream_hash': digest}
    all_rows = []
    all_decisions = []
    all_events = []
    for variant in VARIANTS:
        a, b, c = run_variant(X, y, arrivals, groups, learner, variant, meta)
        all_rows.append(a)
        all_decisions.extend(b)
        all_events.extend(c)
    return (all_rows, all_decisions, all_events)

def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--smoke', action='store_true')
    group.add_argument('--full', action='store_true')
    parser.add_argument('--jobs', type=int, default=8)
    args = parser.parse_args()
    datasets = ['SEA'] if args.smoke else ['SEA', 'Agrawal', 'Mixed', 'RandomTree', 'Sine']
    stream_seeds = [6001] if args.smoke else list(range(6001, 6011 if args.full else 6003))
    learners = ['LR'] if args.smoke else ['LR', 'DT']
    scenarios = ['UNIFORM_100_100'] if args.smoke else list(SCENARIOS)
    delay_seeds = [7001] if args.smoke else list(range(7001, 7006 if args.full else 7004))
    tasks = [(d, s, l, c, k) for d in datasets for s in stream_seeds for l in learners for c in scenarios for k in delay_seeds]
    for d, s, _, _, _ in tasks:
        path = STREAM_DIR / f'{d}_streamseed_{s}.npz'
        if not path.is_file():
            raise FileNotFoundError(f'Frozen stream missing: {path}')
    OUT.mkdir(parents=True, exist_ok=True)
    print(f'protocol ablation: {len(tasks)} matched tasks x {len(VARIANTS)} variants = {len(tasks) * len(VARIANTS)} runs', flush=True)
    start = time.perf_counter()
    result = Parallel(n_jobs=min(args.jobs, len(tasks)), verbose=5)((delayed(work)(task) for task in tasks))
    run_df = pd.DataFrame([x for runs, _, _ in result for x in runs])
    dec_df = pd.DataFrame([x for _, ds, _ in result for x in ds])
    evt_df = pd.DataFrame([x for _, _, es in result for x in es])
    run_df.to_csv(OUT / 'run_raw.csv', index=False)
    dec_df.to_csv(OUT / 'decision_raw.csv', index=False)
    evt_df.to_csv(OUT / 'event_log.csv', index=False)
    records = []
    for variant, group_df in run_df.groupby('variant', sort=False):
        n_decisions = int(group_df.decisions.sum())
        n_correct = int(group_df.correct_decisions.sum())
        records.append({'variant': variant, 'runs': len(group_df), 'decisions': n_decisions, 'oracle_agreement_pooled': n_correct / n_decisions if n_decisions else float('nan'), 'oracle_agreement_run_macro': float(group_df.oracle_agreement.mean()), 'mean_accuracy': float(group_df.accuracy.mean()), 'mean_mcc': float(group_df.mcc.mean()), 'mean_confirmation_delay': float(group_df.mean_confirmation_delay.mean()), 'mean_collected_fraction': float(group_df.collected_fraction.mean()), 'mean_resolved_fraction': float(group_df.resolved_fraction.mean()), 'total_fp': int(group_df.fp.sum()), 'total_fn': int(group_df.fn.sum()), 'total_stale_trigger_violations': int(group_df.stale_trigger_violations.sum()), 'total_stale_update_violations': int(group_df.stale_update_violations.sum()), 'total_history_mismatches': int(group_df.historical_loss_mismatch.sum()), 'total_single_pending_drops': int(group_df.dropped_single_pending_windows.sum()), 'max_pending_observed': int(group_df.max_pending.max())})
    main_df = pd.DataFrame(records)
    main_df.to_csv(OUT / 'main_paper_table.csv', index=False)
    keys = ['dataset', 'stream_seed', 'learner', 'scenario', 'delay_seed']
    baseline = run_df[run_df.variant == 'FULL_PROTOCOL'].set_index(keys)
    paired = []
    for variant in VARIANTS[1:]:
        other = run_df[run_df.variant == variant].set_index(keys)
        for idx in baseline.index:
            b = baseline.loc[idx]
            x = other.loc[idx]
            paired.append({**dict(zip(keys, idx)), 'variant': variant, 'delta_accuracy': float(x.accuracy - b.accuracy), 'delta_resolved_fraction': float(x.resolved_fraction - b.resolved_fraction), 'delta_retrains': int(x.retrain_count - b.retrain_count), 'delta_decisions': int(x.decisions - b.decisions)})
    pd.DataFrame(paired).to_csv(OUT / 'paired_vs_full.csv', index=False)
    info = {'method': 'confirmation comparison-aligned event loop; matched streams and delay schedules', 'modes': {'FULL_PROTOCOL': 'Multiple pending blocks, immutable predictions, stale authority revoked', 'SINGLE_PENDING': 'Drop a newly closed monitoring block if an older pending block exists', 'NO_STALE_GUARD': 'Retain other pending windows after alarm; old windows may trigger later', 'RECOMPUTED_HISTORY': 'Fault injection: recompute returned losses using current model'}, 'event_order': ['Prior cohort update', 'Current event prediction', 'Release delayed labels', 'Calibration completion', 'Event-time window evaluation'], 'windows': W, 'epsilon': EPS, 'post_update_calibration_size': CAL, 'units': 'Run macro = mean across nonempty runs; pooled = total correct / total decisions', 'interpretation': 'Fault-injection controls are not valid competing algorithms', 'total_tasks': len(tasks), 'total_runs': len(run_df), 'runtime_seconds': round(time.perf_counter() - start, 3)}
    (OUT / 'protocol.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
    print('\nMAIN TABLE\n')
    print(main_df.to_string(index=False))
    print(f'\nAll results saved in: {OUT}')
    print(f"Elapsed: {info['runtime_seconds']:.1f} s")
if __name__ == '__main__':
    main()
