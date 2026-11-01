"""Summarize incomplete mortgage label histories."""
from completion_monitoring.paths import workspace_root, external_data_root, source_file
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = workspace_root()
DATA = external_data_root('freddie')
SOURCE = ROOT / 'result/task_feasibility'
OUT = ROOT / 'result/missing_label_audit'
OUT.mkdir(parents=True, exist_ok=True)

def month_index(col):
    value = pd.to_numeric(col, errors='coerce')
    yy = np.floor(value / 100)
    mm = value % 100
    good = (yy >= 1990) & (yy <= 2035) & (mm >= 1) & (mm <= 12)
    return (yy * 12 + mm - 1).where(good)
base = pd.read_csv(SOURCE / 'events_delq1.csv', dtype={'id_loan': str})
if base.id_loan.duplicated().any():
    raise ValueError('存在重复贷款ID')
base['t0'] = pd.to_numeric(base.t0, errors='coerce')
base['year'] = pd.to_numeric(base.year, errors='coerce')
years = sorted(set(base.year.dropna().astype(int)))
missing = base[base.label == -1].copy()
print('=' * 75)
print('missing-label audit FREDDIE MAC MISSINGNESS AUDIT')
print('=' * 75)
print('Total loans:', len(base))
print('Incomplete delq1 loans:', len(missing))
parts = []
for year in years:
    path = DATA / f'sample_perf_{year}.txt'
    if not path.exists():
        raise FileNotFoundError(path)
    ids = set(missing.loc[missing.year == year, 'id_loan'])
    count = 0
    reader = pd.read_csv(path, sep='|', header=None, usecols=[0, 1, 3, 8, 9], dtype=str, chunksize=200000)
    for part in reader:
        part = part[part[0].isin(ids)].copy()
        if not len(part):
            continue
        part = part.rename(columns={0: 'id_loan', 1: 'period', 3: 'delq', 8: 'zero_code', 9: 'zero_date'})
        parts.append(part)
        count += len(part)
    print(f'Year {year}: incomplete_loans={len(ids)}, monthly_records={count}', flush=True)
raw = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=['id_loan', 'period', 'delq', 'zero_code', 'zero_date'])
raw['month'] = month_index(raw['period'])
raw['numeric_delq'] = pd.to_numeric(raw['delq'], errors='coerce')
raw['zero_code'] = raw.zero_code.fillna('').astype(str).str.strip().str.zfill(2)
raw.loc[raw.zero_code.isin(['00', '0', 'nan', 'NaN', '']), 'zero_code'] = ''
known_codes = {'01', '02', '03', '09', '15', '16', '96'}
unexpected = set(raw.loc[(raw.zero_code != '') & ~raw.zero_code.isin(known_codes), 'zero_code'].unique())
if unexpected:
    print('WARNING: Unexpected zero balance codes:', sorted(unexpected))
by_loan = {loan_id: group.sort_values('month') for loan_id, group in raw.groupby('id_loan', sort=False)}
rows = []
for r in missing.itertuples(index=False):
    t0 = int(r.t0)
    end = t0 + 11
    g = by_loan.get(r.id_loan)
    if g is None:
        g = raw.iloc[:0]
    g = g.dropna(subset=['month'])
    months = set(g.month.astype(int))
    first = min(months) if months else None
    last = max(months) if months else None
    expected = set(range(t0, t0 + 12))
    observed = months & expected
    leading = min(12, max(0, first - t0)) if first is not None else 12
    trailing = min(12, max(0, end - last)) if last is not None else 12
    if first is not None and last is not None and (max(t0, first) <= min(end, last)):
        interior = sum((month not in observed for month in range(max(t0, first), min(end, last) + 1)))
    else:
        interior = 0
    within = g[g.month.between(t0, end)]
    invalid_status = int(within.numeric_delq.isna().sum())
    terminated = g[(g.zero_code != '') & (g.month <= end)]
    payoff = bool((terminated.zero_code == '01').any())
    other_termination = bool((terminated.zero_code != '01').any())
    termination_codes = '|'.join(sorted(set(terminated.zero_code.tolist())))
    rows.append({'id_loan': r.id_loan, 'year': int(r.year), 'first_month': t0, 'observed_months': len(observed), 'leading_gap': leading, 'interior_gap': interior, 'trailing_gap': trailing, 'invalid_status': invalid_status, 'early_payoff_01': payoff, 'other_termination': other_termination, 'termination_codes': termination_codes})
audit = pd.DataFrame(rows)
audit.to_csv(OUT / 'incomplete_loan_reasons.csv', index=False)
print('\n[1] MISSING LABEL REASONS')
checks = {'Leading data gap': audit.leading_gap > 0, 'Interior gap': audit.interior_gap > 0, 'Trailing gap': audit.trailing_gap > 0, 'Invalid delinquency status': audit.invalid_status > 0, 'Voluntary payoff code 01': audit.early_payoff_01, 'Other termination': audit.other_termination}
for name, condition in checks.items():
    print(f'{name}: {int(condition.sum())} ({condition.mean():.2%})')
print('\n[2] TERMINATION CODES')
print(audit.termination_codes.value_counts().head(20).to_string())
print('\n[3] TRAINING/CALIBRATION BIAS')
cutoff = 2020 * 12
historical = base[(base.t0 < cutoff) & (base.ready >= 0) & (base.ready < cutoff)].sort_values(['t0', 'id_loan'])
print('Available historical labels:', len(historical))
print('Historical positive rate:', round((historical.label == 1).mean(), 6))
if len(historical) >= 400:
    warmup = historical.tail(400)
    train = warmup.iloc[:200]
    calibration = warmup.iloc[200:]
    print('Training positives:', int((train.label == 1).sum()))
    print('Calibration positives:', int((calibration.label == 1).sum()))
print('\n[4] HISTORICAL YEAR DISTRIBUTION')
print(historical.groupby('year').label.agg(samples='size', positives='sum', positive_rate='mean').to_string())
print('\nmissing-label audit COMPLETED')
print('Results:', OUT)
