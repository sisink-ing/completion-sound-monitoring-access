"""Replay the monitoring protocol on mortgage observations."""
from completion_monitoring.paths import workspace_root, external_data_root, source_file
import argparse
import importlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score as original_accuracy, f1_score as original_f1, matthews_corrcoef as original_mcc
ROOT = workspace_root()
SOURCE = ROOT / 'result' / 'reconstruct_outcomes' / 'all_loan_outcomes.csv'
OUT = ROOT / 'result' / 'protocol_replay'
FEATURES = ['fico', 'cltv', 'dti', 'orig_upb', 'ltv', 'orig_int_rt', 'orig_loan_term', 'cnt_borr']
CUTOFF = 2020 * 12
BAD_ANCHOR = ['not_current_at_anchor', 'already_closed_at_anchor', 'no_performance', 'no_valid_report']

def prepare(task, limit):
    df = pd.read_csv(SOURCE, dtype={'id_loan': str})
    df = df[(df.event_month >= 0) & ~df.reason1.isin(BAD_ANCHOR)].copy()
    df = df.sort_values(['event_month', 'id_loan']).reset_index(drop=True)
    ycol = f'y{task}'
    rcol = f'ready{task}'
    history = df[(df.event_month < CUTOFF) & df[ycol].isin([0, 1]) & (df[rcol] < CUTOFF) & (df.event_month + 12 < CUTOFF)].head(400)
    if len(history) != 400:
        raise RuntimeError('At least 400 historical observations are required.')
    if history.iloc[:200][ycol].nunique() != 2:
        raise RuntimeError('The initial training set contains only one class.')
    online = df[df.event_month >= CUTOFF]
    if limit > 0:
        online = online.head(limit)
    if len(online) < 100:
        raise RuntimeError('At least 100 online observations are required.')
    data = pd.concat([history, online], ignore_index=True)
    values = data[FEATURES].apply(pd.to_numeric, errors='coerce')
    bounds = {'fico': (300, 850), 'cltv': (0, 200), 'dti': (0, 100), 'orig_upb': (1000, 10000000), 'ltv': (0, 200), 'orig_int_rt': (0.01, 30), 'orig_loan_term': (1, 600), 'cnt_borr': (1, 10)}
    for col, (lo, hi) in bounds.items():
        values[col] = values[col].where(values[col].between(lo, hi))
    medians = values.iloc[:200].median().fillna(0)
    X = values.fillna(medians).to_numpy(dtype=np.float64)
    y = data[ycol].astype(int).to_numpy()
    assert np.all(np.isin(y[:400], [0, 1]))
    months = data.event_month.to_numpy(dtype=int)[400:]
    last_month = int(months[-1])
    last_id = len(y) - 1
    arrivals = {}
    groups = {}
    for i in range(400, len(y)):
        label = int(y[i])
        ready = int(data.at[i, rcol])
        event = int(data.at[i, 'event_month'])
        if label == -1:
            continue
        if ready <= event:
            raise RuntimeError(f'不合法反馈时间: {i}')
        if ready <= last_month:
            j = int(np.searchsorted(months, ready, side='right'))
            if j > 0 and months[j - 1] == ready:
                tick = 400 + j - 1
            else:
                tick = 400 + j
        else:
            tick = last_id + ready - last_month
        tick = max(int(tick), i)
        arrivals[i] = tick
        groups.setdefault(tick, []).append(i)
    return (data, X, y, arrivals, groups)

def masked_metric(func):

    def metric(truth, pred, *args, **kwargs):
        a = np.asarray(truth)
        b = np.asarray(pred)
        mask = np.isin(a, [0, 1])
        if not mask.any():
            return float('nan')
        return float(func(a[mask], b[mask], *args, **kwargs))
    return metric

def run(task, limit, learners, enable_adwin):
    OUT.mkdir(parents=True, exist_ok=True)
    data, X, y, arrivals, groups = prepare(task, limit)
    base = importlib.import_module('chronological_protocol')
    engine = importlib.import_module('cohort_protocol')

    def real_feedback(_y, *_):
        if len(_y) != len(y):
            raise RuntimeError('The feedback data lengths do not match.')
        return (arrivals, groups)
    base.generate_delay = real_feedback
    engine.base.generate_delay = real_feedback
    base.STRATEGIES = ['ARRIVED_ONLY']
    base.SELECTIVE_STRATEGIES = ['ARRIVED_ONLY']
    base.PERIODIC_INTERVALS = {}

    def available_ids(t, arrival_map, labels):
        upper = min(t, len(labels))
        returned = [i for i in range(400, upper) if arrival_map.get(i, float('inf')) < t]
        return list(range(400)) + returned
    base.get_all_available_ids = available_ids
    old_oracle = base.calculate_oracle

    def safe_oracle(labels, state, block):
        ids = np.asarray(block['sample_ids'], dtype=int)
        if np.any(labels[ids] == -1):
            return ('UNOBSERVABLE', float('nan'), float('nan'))
        return old_oracle(labels, state, block)
    base.calculate_oracle = safe_oracle
    for module in (base, engine):
        module.accuracy_score = masked_metric(original_accuracy)
        module.f1_score = masked_metric(original_f1)
        module.matthews_corrcoef = masked_metric(original_mcc)
    runs = []
    decisions = []
    updates = []
    skipped = []
    for learner in learners:
        if learner == 'ARF':
            step27 = importlib.import_module('arf_holdout')
            step27.patch_engine_to_arf(engine)
        r, d, u, s = engine.run_one('FreddieMac', X, y, learner, 'REAL_MONTHLY_REPLAY', 0, 0, 2026)
        runs.extend(r)
        decisions.extend(d)
        updates.extend(u)
        skipped.extend(s)
        if learner != 'ARF':
            r2, d2, u2 = base.run_one('FreddieMac', X, y, learner, 'REAL_MONTHLY_REPLAY', 0, 0, 2026)
            runs.extend((z for z in r2 if z['strategy'] == 'ARRIVED_ONLY'))
            decisions.extend((z for z in d2 if z['strategy'] == 'ARRIVED_ONLY'))
            updates.extend((z for z in u2 if z['strategy'] == 'ARRIVED_ONLY'))
        if learner == 'ARF' and enable_adwin:
            step30 = importlib.import_module('delayed_adwin')
            step30.load_modules = lambda: (step27, engine)
            step27.stream_path = lambda *_: 'memory://freddie'
            step27.read_stream_file = lambda *_: (X, y, 'real_mortgage', 'not_applicable_replay')
            for name, func in [('accuracy_score', original_accuracy), ('f1_score', original_f1), ('matthews_corrcoef', original_mcc)]:
                setattr(step30, name, masked_metric(func))
            ar, alarms, au = step30.adwin_one(('FreddieMac', 2026, 'REAL_MONTHLY_REPLAY', 0, 0, 2026), 0.002)
            runs.append(ar)
            updates.extend(au)
            pd.DataFrame(alarms).to_csv(OUT / f'adwin_alarms_delq{task}.csv', index=False)
    report = pd.DataFrame(runs)
    detail = pd.DataFrame(decisions)
    if not detail.empty:
        detail['oracle_evaluable'] = detail.oracle.isin(['DEGRADATION', 'TOLERABLE'])
        valid = detail[detail.oracle_evaluable]
        if not valid.empty:
            summary = valid.groupby(['strategy', 'learner'], as_index=False).agg(oracle_agreement=('correct', 'mean'), n_oracle_windows=('correct', 'size'), mean_confirm_event_steps=('decision_delay', 'mean'))
            report = report.merge(summary, on=['strategy', 'learner'], how='left')
    report['n_evaluable_online'] = int(np.isin(y[400:], [0, 1]).sum())
    report['n_unknown_online'] = int((y[400:] == -1).sum())
    prefix = f'delq{task}'
    report.to_csv(OUT / f'{prefix}_main_table.csv', index=False)
    detail.to_csv(OUT / f'{prefix}_decisions.csv', index=False)
    pd.DataFrame(updates).to_csv(OUT / f'{prefix}_updates.csv', index=False)
    pd.DataFrame(skipped).to_csv(OUT / f'{prefix}_skipped.csv', index=False)
    pd.DataFrame({'id_loan': data.id_loan, 'event_month': data.event_month, 'label': y, 'label_ready_month': data[f'ready{task}'], 'arrival_event_step': [arrivals.get(i, np.nan) for i in range(len(y))]}).to_csv(OUT / f'{prefix}_event_schedule.csv', index=False)
    protocol = {'task': f'delq{task}', 'dataset': 'Freddie Mac 2017-2022', 'online_cutoff': '2020-01', 'train': 200, 'initial_calibration': 200, 'window': 100, 'epsilon': 0.02, 'soft_q': 30, 'post_update_calibration': 100, 'online_events': len(y) - 400, 'unknown_labels': int((y[400:] == -1).sum()), 'time_mode': 'event-step month-end approximation', 'publication_ready': False, 'limitations': ['calendar-month exact scheduling pending', 'unknown outcome oracle excluded', 'real-world class imbalance', 'no actual bank model deployment logs']}
    (OUT / f'{prefix}_protocol.json').write_text(json.dumps(protocol, indent=2, ensure_ascii=False), encoding='utf-8')
    columns = ['strategy', 'learner', 'accuracy', 'macro_f1', 'mcc', 'retrain_count', 'oracle_agreement', 'n_oracle_windows', 'monitor_coverage', 'waiting_coverage', 'calibration_coverage']
    present = [col for col in columns if col in report.columns]
    print('\n' + '=' * 90)
    print('protocol replay REAL MORTGAGE PILOT')
    print('=' * 90)
    print(report[present].to_string(index=False))
    print('\nResults:', OUT)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=int, choices=[1, 2, 3], default=1)
    parser.add_argument('--limit-online', type=int, default=1000)
    parser.add_argument('--models', nargs='+', choices=['LR', 'DT', 'ARF'], default=['LR', 'DT'])
    parser.add_argument('--adwin', action='store_true')
    args = parser.parse_args()
    if args.adwin and 'ARF' not in args.models:
        parser.error('--adwin requires the ARF learner.')
    run(args.task, args.limit_online, args.models, args.adwin)
