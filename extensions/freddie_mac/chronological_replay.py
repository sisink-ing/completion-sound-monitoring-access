"""Replay mortgage outcomes under delayed verification."""
from completion_monitoring.paths import workspace_root, external_data_root, source_file
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import glob
import hashlib
import importlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
ROOT = workspace_root()
OUT = ROOT / 'result' / 'chronological_replay'
FEATURES = ['fico', 'dti', 'cltv', 'ltv', 'orig_upb', 'orig_int_rt', 'orig_loan_term', 'cnt_borr']
ORIG_POS = {'fico': 0, 'dt_first_pi': 1, 'cltv': 8, 'dti': 9, 'orig_upb': 10, 'ltv': 11, 'orig_int_rt': 12, 'id_loan': 19, 'orig_loan_term': 21, 'cnt_borr': 22}
PERF_POS = {'id_loan': 0, 'period': 1, 'delq_sts': 3}
KNOWN_IDS = None

def month_number(values):
    """将YYYYMM转换为整数月份索引。"""
    x = pd.to_numeric(pd.Series(values), errors='coerce')
    year = (x // 100).astype('Int64')
    month = (x % 100).astype('Int64')
    valid = (year.between(1990, 2035) & month.between(1, 12)).fillna(False)
    result = pd.Series(np.nan, index=x.index, dtype='float64')
    result.loc[valid] = year[valid].astype(float) * 12 + month[valid].astype(float) - 1
    return result

def read_table(path, kind):
    """
    支持：
    1. 有字段名的CSV
    2. 较早版Freddie Mac管道分隔TXT

    若官方数据布局改变，须先修正字段映射。
    """
    global KNOWN_IDS
    path = Path(path)
    positions = ORIG_POS if kind == 'orig' else PERF_POS
    if path.suffix.lower() == '.txt':
        with path.open(encoding='utf-8-sig', errors='replace') as f:
            first = f.readline().strip()
        if len(first.split('|')) <= max(positions.values()):
            raise ValueError(f'{path}: 原始字段数量不足，请核对官方File Layout。')
        usecols = sorted(set(positions.values()))
        chunks = []
        reader = pd.read_csv(path, sep='|', header=None, usecols=usecols, dtype=str, chunksize=300000, low_memory=False)
        rename = {index: name for name, index in positions.items()}
        for chunk in reader:
            chunk = chunk.rename(columns=rename)
            if kind == 'perf' and KNOWN_IDS is not None:
                chunk = chunk[chunk['id_loan'].isin(KNOWN_IDS)]
            chunks.append(chunk)
        return pd.concat(chunks, ignore_index=True)
    df = pd.read_csv(path, dtype=str, low_memory=False)
    df.columns = [c.strip().lower() for c in df.columns]
    aliases = {'loan_sequence_number': 'id_loan', 'loan_id': 'id_loan', 'monthly_reporting_period': 'period', 'current_loan_delinquency_status': 'delq_sts', 'original_credit_score': 'fico', 'original_dti': 'dti', 'original_ltv': 'ltv', 'original_cltv': 'cltv', 'original_upb': 'orig_upb', 'original_interest_rate': 'orig_int_rt'}
    df = df.rename(columns=aliases)
    required = ['id_loan', 'period', 'delq_sts'] if kind == 'perf' else ['id_loan'] + FEATURES
    missing = [name for name in required if name not in df.columns]
    if missing:
        raise ValueError(f'{path} 缺少字段: {missing}')
    if kind == 'perf' and KNOWN_IDS is not None:
        df = df[df.id_loan.isin(KNOWN_IDS)]
    return df

def build_events(orig_files, perf_files, per_year, seed):
    """
    构建二分类任务：

    正类：
        12个月内首次观测到逾期状态 >= 3

    负类：
        12个月逐月记录完整，
        且始终没有达到该逾期状态。

    观察期不完整的贷款暂时排除，
    并记录排除数量。
    """
    global KNOWN_IDS
    origins = pd.concat([read_table(p, 'orig') for p in orig_files], ignore_index=True).drop_duplicates('id_loan')
    origins['vintage'] = origins.id_loan.str.extract('F(\\d{2})Q', expand=False).fillna('unknown')
    pieces = []
    for vintage, group in origins.groupby('vintage', sort=True):
        pieces.append(group.sample(n=min(per_year, len(group)), random_state=seed))
    origins = pd.concat(pieces, ignore_index=True)
    KNOWN_IDS = set(origins.id_loan)
    performance = pd.concat([read_table(p, 'perf') for p in perf_files], ignore_index=True)
    performance = performance[performance.id_loan.isin(KNOWN_IDS)].copy()
    performance['month'] = month_number(performance['period'])
    performance['status'] = pd.to_numeric(performance['delq_sts'], errors='coerce')
    performance = performance.dropna(subset=['month'])
    performance = performance.sort_values(['id_loan', 'month']).drop_duplicates(['id_loan', 'month'], keep='last')
    records = []
    for loan_id, group in performance.groupby('id_loan', sort=False):
        group = group.sort_values('month')
        t0 = int(group.month.iloc[0])
        horizon = group[(group.month >= t0) & (group.month < t0 + 12)]
        positive = horizon[horizon.status >= 3]
        if len(positive) > 0:
            label = 1
            available = int(positive.month.iloc[0])
            label_kind = 'event'
        else:
            expected = set(range(t0, t0 + 12))
            valid = horizon.dropna(subset=['status'])
            observed = set(valid.month.astype(int))
            if observed != expected:
                continue
            if not valid.status.between(0, 2).all():
                continue
            label = 0
            available = t0 + 11
            label_kind = 'matured_non_event'
        records.append((loan_id, t0, available, label, label_kind))
    labels = pd.DataFrame(records, columns=['id_loan', 'event_month', 'label_month', 'label', 'label_kind'])
    df = origins.merge(labels, on='id_loan', how='inner')
    df = df.sort_values(['event_month', 'id_loan']).reset_index(drop=True)
    counts = {'origin_rows': int(len(origins)), 'performance_rows': int(len(performance)), 'eligible_loans': int(len(df)), 'excluded_loans': int(len(origins) - len(df)), 'positive_loans': int(df.label.sum()), 'negative_loans': int((df.label == 0).sum())}
    return (df, counts)

def choose_split(df, min_online):
    """
    严格时间切分。

    初始200训练+200校准的真实标签，
    必须在在线测试开始前就已成熟。

    防止使用未来贷款结果训练过去模型。
    """
    for start in sorted(df.event_month.unique()):
        historical = df[(df.event_month < start) & (df.label_month < start)]
        online = df[df.event_month >= start]
        if len(historical) < 400:
            continue
        if len(online) < min_online:
            continue
        warmup = historical.tail(400)
        if warmup.iloc[:200].label.nunique() < 2:
            continue
        output = pd.concat([warmup, online], ignore_index=True)
        return (output, int(start))
    raise ValueError('无法形成无时间泄漏的200/200训练校准集。请使用多个年份的数据，不能只用2020Q1。')

def make_arrivals(df):
    """
    从真实贷款标签月份生成反馈日程。

    同一月份中，先处理该月预测，
    再释放当月月末可确定的标签。

    输出时间单位仍是原代码的事件步。
    """
    online = df.iloc[400:]
    months = online.event_month.to_numpy(dtype=np.int64)
    arrival_map = {}
    arrival_groups = {}
    for index, row in online.iterrows():
        release = 400 + int(np.searchsorted(months, int(row.label_month), side='right'))
        release = max(int(index) + 1, release)
        arrival_map[int(index)] = release
        arrival_groups.setdefault(release, []).append(int(index))
    return (arrival_map, arrival_groups)

def evaluate(df, counts, split_month, models, enable_adwin):
    base = importlib.import_module('chronological_protocol')
    engine = importlib.import_module('cohort_protocol')
    X = df[FEATURES].apply(pd.to_numeric, errors='coerce')
    medians = X.iloc[:200].median().fillna(0)
    X = X.fillna(medians).to_numpy(dtype='float64')
    y = df.label.to_numpy(dtype=np.int64)
    arrival_map, arrival_groups = make_arrivals(df)

    def real_feedback(_y, _seed, _lambda0, _lambda1):
        if len(_y) != len(y):
            raise RuntimeError('回放数据长度不一致')
        return (arrival_map, arrival_groups)
    base.generate_delay = real_feedback
    engine.base.generate_delay = real_feedback
    results = []
    decisions = []
    updates = []
    skipped = []
    for learner in models:
        if learner == 'ARF':
            s27 = importlib.import_module('arf_holdout')
            s27.patch_engine_to_arf(engine)
        run, decision, update, skip = engine.run_one('FreddieMac', X, y, learner, 'REAL_MONTHLY', 0, 0, 2026)
        results.extend(run)
        decisions.extend(decision)
        updates.extend(update)
        skipped.extend(skip)
        if learner != 'ARF':
            run2, decision2, update2 = base.run_one('FreddieMac', X, y, learner, 'REAL_MONTHLY', 0, 0, 2026)
            results.extend((r for r in run2 if r['strategy'] == 'ARRIVED_ONLY'))
            decisions.extend((r for r in decision2 if r['strategy'] == 'ARRIVED_ONLY'))
            updates.extend((r for r in update2 if r['strategy'] == 'ARRIVED_ONLY'))
        if learner == 'ARF' and enable_adwin:
            m30 = importlib.import_module('delayed_adwin')
            proxy = SimpleNamespace(patch_engine_to_arf=s27.patch_engine_to_arf, stream_path=lambda family, seed: 'in_memory', read_stream_file=lambda path: (X, y, 'real_monthly', hashlib.sha256(X.tobytes() + y.tobytes()).hexdigest()))
            m30.load_modules = lambda: (proxy, engine)
            row, alarms, adwin_updates = m30.adwin_one(('FreddieMac', 2026, 'REAL_MONTHLY', 0, 0, 2026), 0.002)
            results.append(row)
            updates.extend(adwin_updates)
            pd.DataFrame(alarms).to_csv(OUT / 'adwin_alarm_log.csv', index=False)
    run_df = pd.DataFrame(results)
    decision_df = pd.DataFrame(decisions)
    run_df.to_csv(OUT / 'strategy_raw.csv', index=False)
    decision_df.to_csv(OUT / 'decision_raw.csv', index=False)
    pd.DataFrame(updates).to_csv(OUT / 'update_raw.csv', index=False)
    pd.DataFrame(skipped).to_csv(OUT / 'skipped_updates.csv', index=False)
    columns = ['strategy', 'learner', 'accuracy', 'macro_f1', 'mcc', 'retrain_count', 'decision_accuracy', 'decision_delay', 'false_positive', 'false_negative', 'monitor_coverage', 'waiting_coverage', 'calibration_coverage', 'skipped_single_class_updates']
    columns = [c for c in columns if c in run_df.columns]
    main_table = run_df[columns]
    main_table.to_csv(OUT / 'main_paper_table.csv', index=False)
    print('\n=== REAL MORTGAGE RESULTS ===')
    print(main_table.to_string(index=False))
    paired = []
    if not decision_df.empty:
        ours = decision_df[decision_df.strategy == 'FULL_COMPLETE_COHORT']
        for _, record in ours.iterrows():
            close = int(record.decision_time - record.decision_delay)
            ids = list(range(close - 99, close + 1))
            if ids[0] < 400:
                raise RuntimeError('窗口索引异常')
            if any((i not in arrival_map for i in ids)):
                raise RuntimeError('窗口反馈映射不完整')
            wait_time = max(close, max((arrival_map[i] for i in ids)))
            advance = wait_time - int(record.decision_time)
            if advance < 0:
                raise RuntimeError('反事实确认时间错误')
            paired.append({'learner': record.learner, 'oracle': record.oracle, 'window_end': close, 'complete_delay': int(record.decision_delay), 'waitall_delay': wait_time - close, 'advance_steps': advance, 'early': int(advance > 0)})
    paired_df = pd.DataFrame(paired)
    paired_df.to_csv(OUT / 'matched_windows.csv', index=False)
    if not paired_df.empty:
        print('\n=== SAME-WINDOW CHECK ===')
        print('Matched windows:', len(paired_df))
        print('Mean advance (event steps):', paired_df.advance_steps.mean())
        print('Early confirmation fraction:', paired_df.early.mean())
    protocol = {'dataset': 'Freddie Mac', 'experiment': 'retrospective_monthly_replay', 'label_definition': '3+ delinquency in 12 months', 'label_timing': 'derived_from_monthly_reports', 'actual_internal_bank_feedback': False, 'initial_train': 200, 'initial_calibration': 200, 'window_size': 100, 'epsilon': 0.02, 'training_cohort': 100, 'post_update_calibration': 100, 'soft_q': 30, 'split_month': split_month, 'online_events': len(df) - 400, 'data_counts': counts}
    (OUT / 'protocol.json').write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding='utf-8')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--orig', required=True)
    parser.add_argument('--perf', required=True)
    parser.add_argument('--max-per-year', type=int, default=3000)
    parser.add_argument('--min-online', type=int, default=1000)
    parser.add_argument('--models', nargs='+', choices=['LR', 'DT', 'ARF'], default=['LR', 'DT'])
    parser.add_argument('--adwin', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    orig_files = sorted(glob.glob(args.orig))
    perf_files = sorted(glob.glob(args.perf))
    if not orig_files or not perf_files:
        raise FileNotFoundError('没有找到贷款数据文件，检查--orig和--perf')
    df, counts = build_events(orig_files, perf_files, args.max_per_year, 2026)
    print('DATA QUALITY:')
    print(counts)
    (OUT / 'data_eligibility.json').write_text(json.dumps(counts, ensure_ascii=False, indent=2), encoding='utf-8')
    df, split_month = choose_split(df, args.min_online)
    df[['id_loan', 'event_month', 'label_month', 'label', 'label_kind']].to_csv(OUT / 'event_schedule.csv', index=False)
    print('Initial train/cal:', 400, 'Online:', len(df) - 400)
    if args.adwin and 'ARF' not in args.models:
        raise ValueError('--adwin必须同时指定ARF')
    models = sorted(set(args.models), key=lambda name: {'LR': 0, 'DT': 1, 'ARF': 2}[name])
    evaluate(df, counts, split_month, models, args.adwin)
if __name__ == '__main__':
    main()
