"""Replay BTS predictions and selective adaptation."""
import os
for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[k] = '1'
import argparse
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, recall_score
from importlib import import_module
core = import_module('matched_window_evaluation')
ROOT = Path(__file__).resolve().parent
FEATS = core.FEATURES

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--full', action='store_true', help='run all chronological online rows')
    ap.add_argument('--window', type=int, default=100)
    ap.add_argument('--calibration', type=int, default=100)
    args = ap.parse_args()
    if args.window < 2 or args.calibration < 2:
        raise ValueError('window and calibration must be >= 2')
    src = ROOT / 'result' / 'prepare_bts'
    train = pd.read_csv(src / 'train.csv')
    cal = pd.read_csv(src / 'calibration.csv')
    data = pd.read_csv(src / ('online_full.csv' if args.full else 'online_4600.csv'))
    data = data.sort_values('event_id').reset_index(drop=True)
    n = len(data)
    if not np.array_equal(data.event_id.to_numpy(), np.arange(1, n + 1)):
        raise ValueError('Expected sequential event_id 1..N')
    predtime = pd.to_datetime(data.prediction_time_utc, utc=True).astype('int64').to_numpy()
    labtime = pd.to_datetime(data.label_available_utc, utc=True).astype('int64').to_numpy()
    labels = data.label.to_numpy(dtype=int)
    if np.any(np.diff(predtime) < 0) or np.any(labtime <= predtime):
        raise ValueError('Invalid chronological or feedback timestamps')
    slots = np.searchsorted(predtime, labtime, side='left')
    release_ns = np.where(slots < n, predtime[np.minimum(slots, n - 1)], labtime)
    arrivals = [[] for _ in range(n)]
    for j, slot in enumerate(slots):
        if slot < n:
            arrivals[int(slot)].append(j)
    for group in arrivals:
        group.sort(key=lambda j: (int(labtime[j]), j))
    out = ROOT / 'result' / 'adaptive_replay' / ('full' if args.full else 'online_4600')
    out.mkdir(parents=True, exist_ok=True)
    all_summaries = []
    all_windows = []
    all_updates = []
    all_predictions = []
    for learner in ('LR', 'DT'):
        for method in ('Static', 'CompleteCohort', 'WaitAllComplete'):
            model = core.get_predictor(learner)
            model.fit(train[FEATS], train.label.astype(int))
            prediction_cache = model.predict(data[FEATS]).astype(int)
            ref = float(np.mean(model.predict(cal[FEATS]) != cal.label.to_numpy(dtype=int)))
            version = 0
            phase = 'Monitoring'
            saved_pred = np.full(n, -1, dtype=int)
            pred_version = np.full(n, -1, dtype=int)
            observed_loss = np.full(n, -1, dtype=int)
            block = []
            pending = []
            windows = []
            cohort = []
            calids = []
            status_counter = defaultdict(int)
            updates = []

            def bounds(w):
                e = observed_loss[w['members']]
                known = e >= 0
                s = int(e[known].sum())
                m = int(known.sum())
                return (s, m, len(e) - m)

            def check_windows(now_event, now_ns, post_stream=False):
                nonlocal phase, block, pending, cohort
                if phase != 'Monitoring':
                    return
                pending.sort(key=lambda w: w['window_id'])
                waiting = []
                for pos, w in enumerate(pending):
                    s, m, u = bounds(w)
                    nn = len(w['members'])
                    tau = w['threshold']
                    if method == 'WaitAllComplete' and u > 0:
                        waiting.append(w)
                        continue
                    if s / nn > tau + 1e-12:
                        ans = 'Degradation'
                    elif (s + u) / nn <= tau + 1e-12:
                        ans = 'Tolerable'
                    else:
                        waiting.append(w)
                        continue
                    w['decision'] = ans
                    w['decision_event'] = now_event
                    w['decision_utc'] = pd.Timestamp(now_ns, unit='ns', tz='UTC').isoformat()
                    w['labels_at_decision'] = m
                    w['confirmation_min'] = (now_ns - w['close_ns']) / 60000000000.0
                    w['advance_min'] = (max(w['close_ns'], int(release_ns[w['members']].max())) - now_ns) / 60000000000.0
                    w['oracle'] = 'Degradation' if w['final_loss'] > tau + 1e-12 else 'Tolerable'
                    assert ans == w['oracle'], (learner, method, w['window_id'], s, m, u)
                    if ans == 'Tolerable':
                        w['status'] = 'Tolerable'
                        status_counter['tolerable'] += 1
                        continue
                    w['status'] = 'Degradation'
                    status_counter['degradation'] += 1
                    cohort = w['members'].copy()
                    w['trigger'] = True
                    for old in waiting + pending[pos + 1:]:
                        old['status'] = 'Stale'
                        old['update_authority'] = False
                        status_counter['stale'] += 1
                    pending = []
                    block = []
                    phase = 'WaitingReady'
                    status_counter['alarm'] += 1
                    return
                pending = waiting
            for i in range(n):
                if method != 'Static' and phase == 'WaitingReady' and (len(cohort) > 0) and np.all(observed_loss[cohort] >= 0):
                    cohort_y = labels[cohort]
                    if np.unique(cohort_y).size == 2:
                        newmodel = core.get_predictor(learner)
                        newmodel.fit(data.iloc[cohort][FEATS], cohort_y)
                        model = newmodel
                        prediction_cache = model.predict(data[FEATS]).astype(int)
                        version += 1
                        phase = 'Calibrating'
                        calids = []
                        status_counter['updates'] += 1
                        updates.append({'learner': learner, 'method': method, 'update_event': i + 1, 'update_time_utc': str(pd.Timestamp(predtime[i], unit='ns', tz='UTC')), 'new_version': version, 'cohort_start': int(cohort[0] + 1), 'cohort_end': int(cohort[-1] + 1), 'training_count': len(cohort), 'training_class_0': int(np.sum(cohort_y == 0)), 'training_class_1': int(np.sum(cohort_y == 1)), 'status': 'updated'})
                    else:
                        phase = 'Monitoring'
                        block = []
                        status_counter['skipped_updates'] += 1
                        updates.append({'learner': learner, 'method': method, 'update_event': i + 1, 'update_time_utc': str(pd.Timestamp(predtime[i], unit='ns', tz='UTC')), 'new_version': version, 'cohort_start': int(cohort[0] + 1), 'cohort_end': int(cohort[-1] + 1), 'training_count': len(cohort), 'training_class_0': int(np.sum(cohort_y == 0)), 'training_class_1': int(np.sum(cohort_y == 1)), 'status': 'skipped_one_class'})
                    cohort = []
                yhat = int(prediction_cache[i])
                saved_pred[i] = yhat
                pred_version[i] = version
                if method != 'Static' and phase == 'Monitoring':
                    block.append(i)
                    status_counter['monitoring_collection_events'] += 1
                    if len(block) == args.window:
                        members = np.array(block, dtype=int)
                        w = {'learner': learner, 'method': method, 'window_id': len(windows) + 1, 'start_event': int(members[0] + 1), 'end_event': int(members[-1] + 1), 'version': version, 'threshold': ref + 0.02, 'reference_loss': ref, 'close_ns': int(predtime[i]), 'final_loss': float(np.mean(saved_pred[members] != labels[members])), 'members': members, 'status': 'Pending', 'trigger': False, 'update_authority': True, 'decision': None, 'oracle': None, 'labels_at_decision': None, 'decision_event': None, 'decision_utc': None, 'confirmation_min': None, 'advance_min': None}
                        windows.append(w)
                        pending.append(w)
                        block = []
                elif method != 'Static' and phase == 'Calibrating':
                    if len(calids) < args.calibration:
                        calids.append(i)
                status_counter[f'prediction_phase_{phase}'] += 1
                for j in arrivals[i]:
                    if saved_pred[j] < 0:
                        raise AssertionError('Feedback before prediction')
                    observed_loss[j] = int(saved_pred[j] != labels[j])
                if method != 'Static':
                    if phase == 'Calibrating' and len(calids) == args.calibration and np.all(observed_loss[calids] >= 0):
                        ref = float(np.mean(observed_loss[calids]))
                        calids = []
                        phase = 'Monitoring'
                        block = []
                        status_counter['calibration_completed'] += 1
                    if phase == 'Monitoring':
                        check_windows(i + 1, int(predtime[i]))
            late = sorted(np.flatnonzero(slots >= n), key=lambda j: (labtime[j], j))
            for j in late:
                observed_loss[j] = int(saved_pred[j] != labels[j])
                if method != 'Static' and phase == 'Monitoring':
                    check_windows(n, int(labtime[j]), post_stream=True)
            for w in windows:
                if w['status'] == 'Pending':
                    status_counter['pending_at_end'] += 1
                if w['status'] == 'Stale':
                    w['oracle'] = 'Degradation' if w['final_loss'] > w['threshold'] + 1e-12 else 'Tolerable'
                w['waitall_min'] = float((max(w['close_ns'], int(release_ns[w['members']].max())) - w['close_ns']) / 60000000000.0)
                w['max_label_available_utc'] = pd.Timestamp(int(labtime[w['members']].max()), unit='ns', tz='UTC').isoformat()
                all_windows.append({k: v for k, v in w.items() if k not in ('members', 'close_ns')})
            all_updates.extend(updates)
            for i in range(n):
                all_predictions.append({'learner': learner, 'method': method, 'event_id': i + 1, 'historical_prediction': int(saved_pred[i]), 'historical_version': int(pred_version[i]), 'actual_label_for_offline_scoring': int(labels[i]), 'prediction_time_utc': str(data.prediction_time_utc.iloc[i]), 'label_available_utc': str(data.label_available_utc.iloc[i])})
            confirmed = [w for w in windows if w['decision'] is not None]
            all_summaries.append({'learner': learner, 'method': method, 'online_rows': n, 'accuracy': float(accuracy_score(labels, saved_pred)), 'macro_f1': float(f1_score(labels, saved_pred, average='macro', zero_division=0)), 'mcc': float(matthews_corrcoef(labels, saved_pred)), 'delay_recall': float(recall_score(labels, saved_pred, zero_division=0)), 'predicted_positive_rate': float(np.mean(saved_pred == 1)), 'updates': int(status_counter['updates']), 'skipped_updates': int(status_counter['skipped_updates']), 'n_closed_windows': len(windows), 'resolved_windows': len(confirmed), 'degradation_decisions': int(status_counter['degradation']), 'tolerable_decisions': int(status_counter['tolerable']), 'stale_windows': int(status_counter['stale']), 'oracle_agreement_resolved': float(np.mean([w['decision'] == w['oracle'] for w in confirmed])) if confirmed else None, 'mean_confirmation_min': float(np.mean([w['confirmation_min'] for w in confirmed])) if confirmed else None, 'mean_advance_min': float(np.mean([w['advance_min'] for w in confirmed])) if confirmed else None, 'monitoring_collected_fraction': status_counter['monitoring_collection_events'] / n, 'eventual_resolved_fraction': len(confirmed) * args.window / n, 'calibration_prediction_fraction': status_counter['prediction_phase_Calibrating'] / n, 'waiting_prediction_fraction': status_counter['prediction_phase_WaitingReady'] / n, 'phase_end': phase, 'threshold_initial': float(np.mean(core.get_predictor(learner).fit(train[FEATS], train.label.astype(int)).predict(cal[FEATS]) != cal.label.to_numpy(dtype=int)) + 0.02)})
            print(learner, method, 'accuracy', round(all_summaries[-1]['accuracy'], 4), 'mcc', round(all_summaries[-1]['mcc'], 4), 'updates', status_counter['updates'], 'resolved', len(confirmed), 'windows', len(windows))
    pd.DataFrame(all_summaries).to_csv(out / 'summary.csv', index=False)
    pd.DataFrame(all_windows).to_csv(out / 'windows.csv', index=False)
    pd.DataFrame(all_updates).to_csv(out / 'updates.csv', index=False)
    pd.DataFrame(all_predictions).to_csv(out / 'historical_predictions.csv', index=False)
    metadata = {'scope': 'BTS adaptation pilot: static, early-confirm Complete Cohort, Wait-All Complete', 'time_axis': 'UTC minute reconstructed labels released at next prediction event; prediction-before-feedback at ties', 'feedback_proxy': 'actual flight completion plus an ASSUMED 10-minute processing buffer', 'training': 'fixed triggering window with all labels; skip update if only one class', 'model_preprocessing': 'Step 04 LR/DT pipelines; identical settings across methods', 'calibration': f'{args.calibration} fresh predictions of each successful new version, wait for all labels', 'window_size': args.window, 'n_online': n, 'post_stream': 'remaining labels are flushed for historical decisions; no new model training is run without future inputs', 'limitations': ['BTS replay with timestamp-derived feedback.', 'Same-window matched differences are NOT estimated from independently adapted strategy windows', 'Only 3 methods here. Soft-q30, Arrived-Only, Stale ablation and fixed-trigger audit need separate controlled tests', 'Real flight labels, but NOT observed label-verification log times', 'Retrospectively excludes canceled/diverted flights, not a true operational prospective sample']}
    (out / 'protocol.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Outputs:', out)
if __name__ == '__main__':
    main()
