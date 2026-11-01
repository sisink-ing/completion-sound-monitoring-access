"""Replay mortgage prediction and verification on a calendar clock."""
from completion_monitoring.paths import workspace_root, external_data_root, source_file
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, recall_score, precision_score, confusion_matrix, average_precision_score
ROOT = workspace_root()
SOURCE = ROOT / 'result/reconstruct_outcomes/all_loan_outcomes.csv'
BASE_OUT = ROOT / 'result/calendar_replay'
CUTOFF = 2020 * 12
FEATURES = ['fico', 'cltv', 'dti', 'orig_upb', 'ltv', 'orig_int_rt', 'orig_loan_term', 'cnt_borr']
BAD_ANCHOR = ('not_current_at_anchor', 'already_closed_at_anchor', 'no_performance', 'no_valid_report')
STRATEGIES = ('STATIC', 'ARRIVED_ONLY', 'FULL_SOFT_Q30', 'FULL_COMPLETE_COHORT', 'WAIT_ALL_COMPLETE_COHORT')
ADAPTIVE = STRATEGIES[1:]

def month_name(t):
    return f'{int(t) // 12:04d}-{int(t) % 12 + 1:02d}'

def prepare(task, limit):
    if not SOURCE.is_file():
        raise FileNotFoundError(f'outcome reconstruction file missing: {SOURCE}')
    df = pd.read_csv(SOURCE, dtype={'id_loan': str})
    df = df[(df.event_month >= 0) & ~df.reason1.isin(BAD_ANCHOR)].copy()
    df = df.sort_values(['event_month', 'id_loan']).reset_index(drop=True)
    yc, rc = (f'y{task}', f'ready{task}')
    hist = df[(df.event_month < CUTOFF) & df[yc].isin([0, 1]) & (df[rc] < CUTOFF) & (df.event_month + 12 < CUTOFF)].head(400)
    if len(hist) != 400:
        raise RuntimeError('Historical 200+200 complete labels unavailable.')
    if hist.iloc[:200][yc].nunique() != 2:
        raise RuntimeError('200 initial training events contain one class only.')
    online = df[df.event_month >= CUTOFF].copy()
    if limit > 0:
        online = online.head(limit)
    if len(online) < 100:
        raise RuntimeError('At least 100 online loans required.')
    rows = pd.concat([hist, online], ignore_index=True)
    vals = rows[FEATURES].apply(pd.to_numeric, errors='coerce')
    bounds = {'fico': (300, 850), 'cltv': (0, 200), 'dti': (0, 100), 'orig_upb': (1000, 10000000), 'ltv': (0, 200), 'orig_int_rt': (0.01, 30), 'orig_loan_term': (1, 600), 'cnt_borr': (1, 10)}
    for col, (lo, hi) in bounds.items():
        vals[col] = vals[col].where(vals[col].between(lo, hi))
    medians = vals.iloc[:200].median().fillna(0)
    X = vals.fillna(medians).to_numpy(dtype=np.float64)
    y = rows[yc].astype(int).to_numpy()
    ev = rows.event_month.astype(int).to_numpy()
    ready = rows[rc].astype(int).to_numpy()
    if any((y[i] >= 0 and ready[i] <= ev[i] for i in range(400, len(y)))):
        raise RuntimeError('Maturity not strictly later than online prediction.')
    return (rows, X, y, ev, ready, medians)

def attach_original_modules(learner):
    base = importlib.import_module('chronological_protocol')
    engine = importlib.import_module('cohort_protocol')
    if learner == 'ARF':
        module = importlib.import_module('arf_holdout')
        module.patch_engine_to_arf(engine)
    assert base.MONITOR_START == 400
    assert base.WINDOW_SIZE == 100
    assert abs(base.EPSILON - 0.02) < 1e-14
    assert base.UPDATE_TRAIN_SIZE == 100
    assert base.MIN_CLASS_COVERAGE == 30
    assert base.POST_UPDATE_CAL_SIZE == 100
    return (base, engine)

def window_oracle(y, state, block, epsilon, tol):
    ids = block['sample_ids']
    if np.any(y[np.asarray(ids, dtype=int)] == -1):
        return ('UNOBSERVABLE', np.nan)
    errors = [state['predictions'][i] != int(y[i]) for i in ids]
    loss = float(np.mean(errors))
    status = 'DEGRADATION' if loss - block['reference_loss'] > epsilon + tol else 'TOLERABLE'
    return (status, loss)

def run(task, learner, limit=0, out=None):
    out = Path(out) if out else BASE_OUT / f'delq{task}_{learner.lower()}'
    out.mkdir(parents=True, exist_ok=True)
    rows, X, y, events, ready, medians = prepare(task, limit)
    base, engine = attach_original_modules(learner)
    states = {s: engine.create_state(X, y, learner) for s in STRATEGIES}
    last_new = int(events[-1])
    known = ready[400:][y[400:] >= 0]
    last_maturity = int(max(known)) if len(known) else last_new
    final_month = max(last_new, last_maturity)
    event_groups = {}
    feedback_groups = {}
    for idx in range(400, len(y)):
        event_groups.setdefault(int(events[idx]), []).append(idx)
        if y[idx] in (0, 1):
            feedback_groups.setdefault(int(ready[idx]), []).append(idx)
    prediction_scores = {}
    prediction_versions = {}
    stop_state = []
    for month in range(CUTOFF, final_month + 1):
        new_events = event_groups.get(month, [])
        for strategy in ADAPTIVE:
            st = states[strategy]
            if st['phase'] != 'WAITING_READY' or month > last_new:
                continue
            if strategy in ('ARRIVED_ONLY', 'FULL_SOFT_Q30'):
                ids = [i for i in range(st['alarm_anchor'], len(y)) if i in st['feedback']]
                train_ids = base.soft_coverage_from_list(ids, y)
            else:
                train_ids = engine.get_complete_cohort_training_ids(st)
            if train_ids is None:
                continue
            if np.any(y[np.asarray(train_ids, dtype=int)] == -1):
                raise RuntimeError('Training attempted on an unknown label.')
            if any((i not in st['feedback'] for i in train_ids)):
                raise RuntimeError('Training attempted before feedback receipt.')
            prev_updates = len(st['update_records'])
            engine.perform_update(strategy, st, train_ids, month, X, y, learner)
            if len(st['update_records']) > prev_updates:
                r = st['update_records'][-1]
                if strategy == 'ARRIVED_ONLY':
                    r['training_mode'] = 'ARRIVED_ONLY_Q30'
                training = np.asarray(train_ids, dtype=int)
                r['training_id_sha256'] = hashlib.sha256(','.join(map(str, training)).encode()).hexdigest()
                r['latest_training_label_month'] = month_name(int(np.max(ready[training])))
                if int(np.max(ready[training])) >= month:
                    raise RuntimeError('Future label used to update model.')
                if st['adaptation_cohort_ids'] is not None:
                    frozen = st['adaptation_cohort_ids']
                    r['frozen_cohort_sha256'] = hashlib.sha256(','.join(map(str, frozen)).encode()).hexdigest()
        for i in new_events:
            for strategy in STRATEGIES:
                st = states[strategy]
                pred = int(st['model'].predict(X[i:i + 1])[0])
                st['predictions'][i] = pred
                score = float('nan')
                if hasattr(st['model'], 'predict_proba'):
                    try:
                        prob = np.asarray(st['model'].predict_proba(X[i:i + 1]))
                        if prob.ndim == 2 and prob.shape[1] == 2:
                            score = float(prob[0, 1])
                    except (ValueError, TypeError, AttributeError):
                        pass
                prediction_scores[strategy, i] = score
                prediction_versions[strategy, i] = int(st['model_version'])
                if strategy == 'STATIC':
                    continue
                phase = st['phase']
                if phase == 'MONITORING':
                    st['monitor_samples'] += 1
                    st['open_block'].append(i)
                    if len(st['open_block']) == base.WINDOW_SIZE:
                        ids = list(st['open_block'])
                        block = {'block_id': int(st['next_block_id']), 'model_version': int(st['model_version']), 'reference_loss': float(st['reference_loss']), 'sample_ids': ids, 'start': ids[0], 'end': ids[-1], 'close_time': month, 'status': 'PENDING'}
                        st['closed_blocks'].append(block)
                        st['next_block_id'] += 1
                        st['open_block'] = []
                elif phase == 'CALIBRATING':
                    st['calibration_samples'] += 1
                    if len(st['calibration_ids']) < base.POST_UPDATE_CAL_SIZE:
                        st['calibration_ids'].append(i)
                elif phase == 'WAITING_READY':
                    st['waiting_samples'] += 1
                else:
                    raise RuntimeError(f'Unknown model phase {phase}')
        for i in feedback_groups.get(month, []):
            if not (int(ready[i]) == month and y[i] in (0, 1)):
                raise RuntimeError('Invalid receipt schedule.')
            for strategy in STRATEGIES:
                st = states[strategy]
                if i not in st['predictions']:
                    raise RuntimeError(f'Feedback arrived before prediction: {i}')
                st['feedback'][i] = int(st['predictions'][i] != y[i])
        for strategy in ADAPTIVE:
            st = states[strategy]
            if st['phase'] != 'CALIBRATING' or len(st['calibration_ids']) != base.POST_UPDATE_CAL_SIZE:
                continue
            cal = st['calibration_ids']
            if all((i in st['feedback'] for i in cal)):
                st['reference_loss'] = float(np.mean([st['feedback'][i] for i in cal]))
                st['phase'] = 'MONITORING'
                st['calibration_ids'] = []
                st['open_block'] = []
        for strategy in ADAPTIVE:
            st = states[strategy]
            if st['phase'] != 'MONITORING':
                continue
            for block in st['closed_blocks']:
                if block['status'] != 'PENDING':
                    continue
                if block['model_version'] != st['model_version']:
                    raise RuntimeError('Stale version block was not revoked.')
                if strategy == 'ARRIVED_ONLY':
                    result = base.evaluate_arrived_only(st, block)
                elif strategy == 'WAIT_ALL_COMPLETE_COHORT':
                    result = base.evaluate_wait_all(st, block)
                else:
                    result = base.evaluate_bounds(st, block)
                if result is None:
                    continue
                block['status'] = 'DONE'
                oracle, final_loss = window_oracle(y, st, block, base.EPSILON, base.FLOAT_TOL)
                decision = result['decision']
                if strategy in ('FULL_SOFT_Q30', 'FULL_COMPLETE_COHORT', 'WAIT_ALL_COMPLETE_COHORT') and oracle != 'UNOBSERVABLE' and (decision != oracle):
                    raise RuntimeError('Certified decision disagrees with completed oracle.')
                ids = block['sample_ids']
                stable_hash = hashlib.sha256(','.join(map(str, ids)).encode()).hexdigest()
                st['decision_records'].append({'block_id': block['block_id'], 'model_version': block['model_version'], 'block_start_event': int(ids[0]), 'block_end_event': int(ids[-1]), 'block_hash': stable_hash, 'close_month_index': int(block['close_time']), 'close_month': month_name(block['close_time']), 'decision_month_index': month, 'decision_month': month_name(month), 'delay_months': month - int(block['close_time']), 'decision': decision, 'oracle': oracle, 'correct': int(decision == oracle) if oracle != 'UNOBSERVABLE' else np.nan, 'labels_at_decision': int(result['m']), 'unknown_at_decision': int(result['u']), 'reference_loss': float(block['reference_loss']), 'final_loss': final_loss})
                if decision == 'DEGRADATION':
                    st['phase'] = 'WAITING_READY'
                    st['alarm_anchor'] = int(ids[0])
                    st['alarm_time'] = month
                    st['adaptation_cohort_ids'] = list(ids)
                    st['cohort_missing_at_alarm'] = int(sum((i not in st['feedback'] for i in ids)))
                    st['open_block'] = []
                    for other in st['closed_blocks']:
                        if other['status'] == 'PENDING':
                            other['status'] = 'STALE'
                    break
        stop_state.append({'month': month_name(month), 'month_index': month, 'n_new_events': len(new_events), 'n_feedback': len(feedback_groups.get(month, [])), **{f'{s}_phase': states[s]['phase'] for s in ADAPTIVE}})
    pred_rows = []
    metric_rows = []
    decision_rows = []
    update_rows = []
    skipped_rows = []
    mask = np.isin(y[400:], [0, 1])
    actual = y[400:][mask]
    for strategy in STRATEGIES:
        st = states[strategy]
        preds = np.array([st['predictions'][i] for i in range(400, len(y))], dtype=int)
        yp = preds[mask]
        scores = np.array([prediction_scores[strategy, i] for i in range(400, len(y))], dtype=float)
        observed_scores = scores[mask]
        ap = float(average_precision_score(actual, observed_scores)) if np.isfinite(observed_scores).all() and len(np.unique(actual)) == 2 else np.nan
        tn, fp, fn, tp = confusion_matrix(actual, yp, labels=[0, 1]).ravel()
        metric_rows.append({'task': f'delq{task}', 'learner': learner, 'strategy': strategy, 'n_online': len(y) - 400, 'n_evaluable': int(mask.sum()), 'n_unresolved': int((~mask).sum()), 'accuracy': accuracy_score(actual, yp), 'macro_f1': f1_score(actual, yp, average='macro', zero_division=0), 'mcc': matthews_corrcoef(actual, yp), 'average_precision': ap, 'recall_positive': recall_score(actual, yp, zero_division=0), 'precision_positive': precision_score(actual, yp, zero_division=0), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp), 'predicted_positives': int(yp.sum()), 'retrain_count': int(st['retrain_count']), 'decision_count': len(st['decision_records']), 'oracle_evaluable_decisions': int(sum((d['oracle'] != 'UNOBSERVABLE' for d in st['decision_records']))), 'oracle_agreement_evaluable': float(np.mean([d['correct'] for d in st['decision_records'] if d['oracle'] != 'UNOBSERVABLE'])) if any((d['oracle'] != 'UNOBSERVABLE' for d in st['decision_records'])) else np.nan, 'monitor_coverage': st['monitor_samples'] / (len(y) - 400) if strategy != 'STATIC' else np.nan, 'waiting_coverage': st['waiting_samples'] / (len(y) - 400) if strategy != 'STATIC' else np.nan, 'calibration_coverage': st['calibration_samples'] / (len(y) - 400) if strategy != 'STATIC' else np.nan, 'final_phase': st['phase']})
        for i in range(400, len(y)):
            pred_rows.append({'task': task, 'learner': learner, 'strategy': strategy, 'id_loan': rows.at[i, 'id_loan'], 'event_id': i, 'event_month': month_name(events[i]), 'label': int(y[i]), 'label_available_month': month_name(ready[i]) if y[i] >= 0 else '', 'prediction': int(st['predictions'][i]), 'score_positive': prediction_scores[strategy, i], 'model_version': prediction_versions[strategy, i], 'evaluable': int(y[i] in (0, 1))})
        for d in st['decision_records']:
            decision_rows.append({'task': task, 'learner': learner, 'strategy': strategy, **d})
        for u in st['update_records']:
            update_rows.append({'task': task, 'learner': learner, 'strategy': strategy, **u, 'update_month': month_name(u['update_time']), 'alarm_month': month_name(u['alarm_time']) if u.get('alarm_time') is not None else ''})
        for u in st['skipped_update_records']:
            skipped_rows.append({'task': task, 'learner': learner, 'strategy': strategy, **u})
    raw_decisions = pd.DataFrame(decision_rows)
    pair = []
    if not raw_decisions.empty:
        left = raw_decisions[raw_decisions.strategy == 'FULL_COMPLETE_COHORT'].copy()
        right = raw_decisions[raw_decisions.strategy == 'WAIT_ALL_COMPLETE_COHORT'].copy()
        keys = ['task', 'learner', 'model_version', 'block_hash']
        match = left.merge(right, on=keys, suffixes=('_complete', '_wait'), validate='one_to_one')
        for r in match.itertuples():
            pair.append({'task': task, 'learner': learner, 'model_version': r.model_version, 'block_hash': r.block_hash, 'block_start_event': r.block_start_event_complete, 'block_end_event': r.block_end_event_complete, 'complete_decision_month': r.decision_month_complete, 'wait_decision_month': r.decision_month_wait, 'complete_decision': r.decision_complete, 'wait_decision': r.decision_wait, 'same_decision': int(r.decision_complete == r.decision_wait), 'complete_labels': r.labels_at_decision_complete, 'wait_labels': r.labels_at_decision_wait, 'advance_months': int(r.decision_month_index_wait - r.decision_month_index_complete), 'oracle_evaluable': int(r.oracle_complete != 'UNOBSERVABLE' and r.oracle_wait != 'UNOBSERVABLE')})
    pd.DataFrame(metric_rows).to_csv(out / 'main_table.csv', index=False)
    pd.DataFrame(pred_rows).to_csv(out / 'predictions.csv', index=False)
    raw_decisions.to_csv(out / 'decisions.csv', index=False)
    pd.DataFrame(pair).to_csv(out / 'paired_windows.csv', index=False)
    pd.DataFrame(update_rows).to_csv(out / 'updates.csv', index=False)
    pd.DataFrame(skipped_rows).to_csv(out / 'skipped.csv', index=False)
    pd.DataFrame(stop_state).to_csv(out / 'month_clock.csv', index=False)
    protocol = {'dataset': 'Freddie Mac 2017-2022 sample; labels reconstructed in outcome reconstruction', 'task': f'delq{task}', 'learner': learner, 'training_n': 200, 'calibration_n': 200, 'window_n': base.WINDOW_SIZE, 'epsilon': base.EPSILON, 'complete_training_n': base.UPDATE_TRAIN_SIZE, 'soft_q': base.MIN_CLASS_COVERAGE, 'post_update_calibration_n': base.POST_UPDATE_CAL_SIZE, 'month_clock_start': month_name(CUTOFF), 'month_clock_last': month_name(final_month), 'new_event_last_month': month_name(last_new), 'simulated_feedback_rule': 'predictions in same month before maturity at month-end; empty months explicit', 'unknown_labels': 'never released or trained on; oracle of affected windows UNOBSERVABLE', 'simulator': 'new calendar scheduler calling original 20C/20D primitives; not identical to original 20D.run_one', 'source_lineage': 'outcome reconstruction redefines loan outcome as delinquency before voluntary payoff competing exit', 'actual_bank_receipt_times': 'not available', 'publication_ready': False, 'remaining_gaps': ['Real bank receipt timestamps not observed', 'End-of-report-month chronology assumed', 'ADWIN calendar-consistent baseline not yet implemented', 'Initial training cohort selected from 400 known pre-cutoff loans', 'Training and feature preprocessing sensitivity/independent seeds needed']}
    (out / 'protocol.json').write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding='utf-8')
    print(pd.DataFrame(metric_rows)[['task', 'learner', 'strategy', 'accuracy', 'macro_f1', 'mcc', 'recall_positive', 'retrain_count', 'oracle_evaluable_decisions', 'monitor_coverage']].to_string(index=False), flush=True)
    if len(pair):
        print('Matched complete-vs-wait windows:', len(pair), 'early:', sum((x['advance_months'] > 0 for x in pair)), 'mean advance months:', round(float(np.mean([x['advance_months'] for x in pair])), 3), flush=True)
    else:
        print('No strictly paired windows. Do not fabricate confirmation advantage.', flush=True)
    print('Saved:', out, flush=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=int, choices=[1, 2, 3], required=True)
    parser.add_argument('--learner', choices=['LR', 'DT', 'ARF'], required=True)
    parser.add_argument('--limit-online', type=int, default=0)
    parser.add_argument('--output', default='')
    a = parser.parse_args()
    run(a.task, a.learner, a.limit_online, a.output or None)
if __name__ == '__main__':
    main()
