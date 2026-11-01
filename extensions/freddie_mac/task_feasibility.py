"""Evaluate label availability for mortgage prediction tasks."""
from completion_monitoring.paths import workspace_root, external_data_root, source_file
from pathlib import Path
import numpy as np
import pandas as pd
DATA = external_data_root('freddie')
OUT = workspace_root() / 'result/task_feasibility'
OUT.mkdir(parents=True, exist_ok=True)
SAMPLE_PER_YEAR = 3000
SEED = 2026

def to_month(series):
    n = pd.to_numeric(series, errors='coerce')
    year, month = (n // 100, n % 100)
    good = year.between(1990, 2035) & month.between(1, 12)
    return (year * 12 + month - 1).where(good)
all_origin, all_perf = ([], [])
for year in range(2017, 2023):
    o_path = DATA / f'sample_orig_{year}.txt'
    p_path = DATA / f'sample_perf_{year}.txt'
    for p in (o_path, p_path):
        if not p.is_file():
            raise FileNotFoundError(p)
    o = pd.read_csv(o_path, sep='|', header=None, usecols=[1, 19], dtype=str)
    o = o.rename(columns={1: 'first_payment', 19: 'id_loan'})
    o['year'] = year
    o = o.drop_duplicates('id_loan').sample(n=min(SAMPLE_PER_YEAR, len(o)), random_state=SEED)
    ids = set(o.id_loan)
    all_origin.append(o)
    chunks = []
    for part in pd.read_csv(p_path, sep='|', header=None, usecols=[0, 1, 3], dtype=str, chunksize=200000):
        part = part.rename(columns={0: 'id_loan', 1: 'period', 3: 'status'})
        part = part[part.id_loan.isin(ids)]
        if len(part):
            chunks.append(part)
    all_perf.extend(chunks)
    print(f'Loaded {year}: origin={len(o)}, performance={sum(map(len, chunks))}', flush=True)
orig = pd.concat(all_origin, ignore_index=True)
orig['t0'] = to_month(orig.first_payment)
perf = pd.concat(all_perf, ignore_index=True)
perf['month'] = to_month(perf.period)
perf['status'] = pd.to_numeric(perf.status, errors='coerce')
perf = perf.dropna(subset=['month']).sort_values(['id_loan', 'month']).drop_duplicates(['id_loan', 'month'], keep='last')
joined = perf.merge(orig[['id_loan', 't0']], on='id_loan', how='inner')
joined = joined[(joined.month >= joined.t0) & (joined.month < joined.t0 + 12)].copy()
joined['month'] = joined.month.astype('int64')
records = []
for loan_id, g in joined.groupby('id_loan', sort=False):
    t0 = int(g.t0.iloc[0])
    observed_months = set(g.month)
    for severity in (1, 2, 3):
        hits = g[g.status >= severity]
        if len(hits):
            label = 1
            ready = int(hits.month.min())
        elif observed_months == set(range(t0, t0 + 12)) and g.status.notna().all() and (g.status < severity).all():
            label = 0
            ready = t0 + 11
        else:
            label = -1
            ready = -1
        records.append((loan_id, severity, label, ready))
labels = pd.DataFrame(records, columns=['id_loan', 'threshold', 'label', 'ready'])
results = []
for severity in (1, 2, 3):
    one = orig.merge(labels[labels.threshold == severity], on='id_loan', how='left')
    one['label'] = one.label.fillna(-1).astype(int)
    one['ready'] = one.ready.fillna(-1).astype(int)
    one = one.dropna(subset=['t0']).sort_values(['t0', 'id_loan']).reset_index(drop=True)
    usable = one[one.label >= 0]
    positives = usable[usable.label == 1]
    delays = usable.ready - usable.t0
    windows = []
    for i in range(0, len(usable) - 99, 100):
        windows.append(int(usable.iloc[i:i + 100].label.sum()))
    possible = 0
    for t in sorted(one.t0.unique()):
        past = one[(one.t0 < t) & (one.ready >= 0) & (one.ready < t)]
        future = one[one.t0 >= t]
        if len(past) >= 400 and len(future) >= 1000 and (past.tail(400).iloc[:200].label.nunique() == 2):
            possible += 1
    result = {'delinquency_threshold_months': severity, 'all_events': len(one), 'observable_labels': len(usable), 'unresolved_or_censored': int((one.label == -1).sum()), 'positives': len(positives), 'positive_rate': float(usable.label.mean()) if len(usable) else np.nan, 'mean_maturity_delay_months': float(delays.mean()) if len(delays) else np.nan, 'windows_100': len(windows), 'windows_without_positive': sum((w == 0 for w in windows)), 'windows_with_3plus_positive': sum((w >= 3 for w in windows)), 'eligible_temporal_splits': possible}
    results.append(result)
    one[['id_loan', 'year', 'first_payment', 't0', 'label', 'ready']].to_csv(OUT / f'events_delq{severity}.csv', index=False)
    print(f'\nThreshold >= {severity} delinquency months:\n{result}', flush=True)
pd.DataFrame(results).to_csv(OUT / 'feasibility_summary.csv', index=False)
print(f'\nCompleted. Reports: {OUT}', flush=True)
