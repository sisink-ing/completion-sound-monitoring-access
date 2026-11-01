"""Reconstruct mortgage outcomes and competing exits."""
from completion_monitoring.paths import workspace_root, external_data_root, source_file
from pathlib import Path
import json
import numpy as np
import pandas as pd
ROOT = workspace_root()
DATA = external_data_root('freddie')
SOURCE = ROOT / 'result/task_feasibility' / 'events_delq1.csv'
OUT = ROOT / 'result/reconstruct_outcomes'
OUT.mkdir(parents=True, exist_ok=True)
HORIZON = 12
CUTOFF = 2020 * 12
FEATURE_POS = {0: 'fico', 8: 'cltv', 9: 'dti', 10: 'orig_upb', 11: 'ltv', 12: 'orig_int_rt', 21: 'orig_loan_term', 22: 'cnt_borr'}

def month_index(values):
    x = pd.to_numeric(values, errors='coerce')
    y = x // 100
    m = x % 100
    good = y.between(1990, 2035) & m.between(1, 12)
    return (y * 12 + m - 1).where(good)

def classify_loan(group, threshold):
    if group is None or group.empty:
        return (-1, -1, 'no_performance')
    group = group.sort_values('month').drop_duplicates('month', keep='last').dropna(subset=['month'])
    if group.empty:
        return (-1, -1, 'no_valid_report')
    t0 = int(group.month.iloc[0])
    first = group.iloc[0]
    if pd.isna(first.delq) or first.delq != 0:
        return (-1, -1, 'not_current_at_anchor')
    if first.zero != '':
        return (-1, -1, 'already_closed_at_anchor')
    future = group[(group.month > t0) & (group.month <= t0 + HORIZON)]
    expected_month = t0 + 1
    has_gap = False
    for r in future.itertuples(index=False):
        current_month = int(r.month)
        if current_month > expected_month:
            has_gap = True
        expected_month = current_month + 1
        if pd.notna(r.delq) and r.delq >= threshold:
            if r.zero != '':
                return (-1, -1, 'same_month_event_exit_ambiguous')
            return (1, current_month, 'delinquency_event')
        if r.zero == '01':
            if has_gap or pd.isna(r.delq):
                return (-1, -1, 'gap_before_payoff')
            return (0, current_month, 'voluntary_payoff_competing_event')
        if r.zero != '':
            return (-1, -1, 'other_exit_' + str(r.zero))
        if pd.isna(r.delq):
            has_gap = True
    if not has_gap and expected_month == t0 + HORIZON + 1:
        return (0, t0 + HORIZON, 'matured_no_event')
    return (-1, -1, 'missing_followup')

def main():
    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)
    ids = pd.read_csv(SOURCE, usecols=['id_loan', 'year'], dtype={'id_loan': str})
    if ids.id_loan.duplicated().any():
        raise RuntimeError('Step37存在重复贷款ID')
    events = []
    origins = []
    for year, group in ids.groupby('year'):
        year = int(year)
        selected_ids = set(group.id_loan)
        orig_file = DATA / f'sample_orig_{year}.txt'
        perf_file = DATA / f'sample_perf_{year}.txt'
        if not orig_file.exists():
            raise FileNotFoundError(orig_file)
        if not perf_file.exists():
            raise FileNotFoundError(perf_file)
        columns = sorted(list(FEATURE_POS.keys()) + [19])
        original = pd.read_csv(orig_file, sep='|', header=None, usecols=columns, dtype=str)
        original = original.rename(columns={**FEATURE_POS, 19: 'id_loan'})
        original = original[original.id_loan.isin(selected_ids)].drop_duplicates('id_loan')
        origins.append(original)
        pieces = []
        reader = pd.read_csv(perf_file, sep='|', header=None, usecols=[0, 1, 3, 8], dtype=str, chunksize=200000)
        for part in reader:
            part = part[part[0].isin(selected_ids)]
            if len(part):
                pieces.append(part)
        if pieces:
            perf = pd.concat(pieces, ignore_index=True)
            perf = perf.rename(columns={0: 'id_loan', 1: 'report_period', 3: 'delq', 8: 'zero'})
            perf['month'] = month_index(perf.report_period)
            perf['delq'] = pd.to_numeric(perf.delq, errors='coerce')
            perf['zero'] = perf.zero.fillna('').astype(str).str.strip().str.zfill(2)
            perf.loc[perf.zero == '00', 'zero'] = ''
            perf = perf.dropna(subset=['month'])
            grouped = dict(tuple(perf.groupby('id_loan', sort=False)))
        else:
            grouped = {}
        for loan_id in sorted(selected_ids):
            loan = grouped.get(loan_id)
            start = int(loan.month.min()) if loan is not None and len(loan) else -1
            record = {'id_loan': loan_id, 'origin_year': year, 'event_month': start}
            for threshold in (1, 2, 3):
                y, ready, reason = classify_loan(loan, threshold)
                record[f'y{threshold}'] = y
                record[f'ready{threshold}'] = ready
                record[f'reason{threshold}'] = reason
            events.append(record)
        print(f'YEAR {year}: {len(selected_ids)} loans loaded', flush=True)
    df = pd.DataFrame(events)
    df = df.merge(pd.concat(origins, ignore_index=True), on='id_loan', how='left')
    df = df.sort_values(['event_month', 'id_loan']).reset_index(drop=True)
    df.to_csv(OUT / 'all_loan_outcomes.csv', index=False)
    summary = []
    for threshold in (1, 2, 3):
        y_col = f'y{threshold}'
        ready_col = f'ready{threshold}'
        reason_col = f'reason{threshold}'
        reasons = df[reason_col].value_counts().rename_axis('reason').reset_index(name='count')
        reasons.to_csv(OUT / f'reasons_delq{threshold}.csv', index=False)
        online = df[df.event_month >= CUTOFF].copy()
        online = online.sort_values(['event_month', 'id_loan']).reset_index(drop=True)
        windows = []
        for start in range(0, len(online) - 99, 100):
            w = online.iloc[start:start + 100]
            windows.append({'window_id': len(windows), 'unknown': int((w[y_col] == -1).sum()), 'positives': int((w[y_col] == 1).sum())})
        window_df = pd.DataFrame(windows)
        window_df.to_csv(OUT / f'windows_delq{threshold}.csv', index=False)
        historical = df[(df.event_month >= 0) & (df.event_month + HORIZON < CUTOFF) & df[y_col].isin([0, 1]) & (df[ready_col] < CUTOFF)]
        warmup = historical.sort_values(['event_month', 'id_loan']).head(400)
        train = warmup.iloc[:200]
        cal = warmup.iloc[200:400]
        result = {'task': f'delq{threshold}', 'total': len(df), 'online_events': len(online), 'observed_labels': int(df[y_col].isin([0, 1]).sum()), 'unresolved_labels': int((df[y_col] == -1).sum()), 'payoff_labels': int((df[reason_col] == 'voluntary_payoff_competing_event').sum()), 'positives': int((df[y_col] == 1).sum()), 'warmup_train': len(train), 'train_positives': int((train[y_col] == 1).sum()), 'warmup_cal': len(cal), 'cal_positives': int((cal[y_col] == 1).sum()), 'online_windows': len(window_df), 'complete_windows': int((window_df.unknown == 0).sum()) if len(window_df) else 0, 'online_unknown': int((online[y_col] == -1).sum())}
        summary.append(result)
        print('\nTASK:', result, flush=True)
    pd.DataFrame(summary).to_csv(OUT / 'outcome_summary.csv', index=False)
    protocol = {'prediction_time': 'end of first monthly performance report', 'baseline': 'current loan only', 'horizon_months': 12, 'payoff_01': 'competing event, not unconditional negative', 'other_termination': 'unresolved', 'unknown_label': -1, 'online_cutoff': '2020-01', 'time_resolution': 'monthly report, not bank receipt timestamp'}
    (OUT / 'protocol.json').write_text(json.dumps(protocol, indent=2, ensure_ascii=False), encoding='utf-8')
    print('\noutcome reconstruction COMPLETED')
    print('Output:', OUT)
if __name__ == '__main__':
    main()
