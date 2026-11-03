"""Evaluate complete-cohort readiness for mortgage prediction tasks."""
from completion_monitoring.paths import workspace_root, external_data_root, source_file
from pathlib import Path
import argparse
import pandas as pd
ROOT = workspace_root()

def month_index(value):
    year, month = divmod(int(value), 100)
    if not (1990 <= year <= 2035 and 1 <= month <= 12):
        raise ValueError(f'错误的日期：{value}')
    return year * 12 + month - 1

def run(args):
    source = Path(args.input)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    cutoff = month_index(args.cutoff)
    reports = []
    for severity in (1, 2, 3):
        path = source / f'events_delq{severity}.csv'
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_csv(path, dtype={'id_loan': str})
        required = {'id_loan', 't0', 'label', 'ready'}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f'缺少字段：{missing}')
        for name in ('t0', 'label', 'ready'):
            df[name] = pd.to_numeric(df[name], errors='coerce')
        df = df.dropna(subset=['t0']).sort_values(['t0', 'id_loan']).reset_index(drop=True)
        if df.id_loan.duplicated().any():
            raise ValueError('Duplicate loan IDs.')
        matured = (df.t0 < cutoff) & df.label.isin([0, 1]) & (df.ready >= df.t0) & (df.ready < cutoff)
        historical = df[matured].copy()
        required_history = args.train + args.cal
        if len(historical) >= required_history:
            warmup = historical.tail(required_history)
            train = warmup.iloc[:args.train]
            calibration = warmup.iloc[args.train:]
        else:
            train = historical.iloc[:0]
            calibration = historical.iloc[:0]
        online = df[df.t0 >= cutoff].copy().reset_index(drop=True)
        blocks = []
        for start in range(0, len(online) - args.window + 1, args.window):
            block = online.iloc[start:start + args.window]
            unresolved = int((~block.label.isin([0, 1])).sum())
            blocks.append({'window_id': len(blocks), 'first_event_month': int(block.t0.iloc[0]), 'last_event_month': int(block.t0.iloc[-1]), 'window_size': len(block), 'unknown_labels': unresolved, 'known_positives': int((block.label == 1).sum()), 'fully_observable': int(unresolved == 0)})
        windows = pd.DataFrame(blocks, columns=['window_id', 'first_event_month', 'last_event_month', 'window_size', 'unknown_labels', 'known_positives', 'fully_observable'])
        windows.to_csv(output / f'window_audit_delq{severity}.csv', index=False)
        online[['id_loan', 't0', 'label', 'ready']].to_csv(output / f'online_delq{severity}.csv', index=False)
        unknown = int((~online.label.isin([0, 1])).sum())
        total_windows = len(windows)
        complete_windows = int(windows.fully_observable.sum())
        result = {'task': f'delq{severity}', 'historical_matured': len(historical), 'train_size': len(train), 'train_positives': int((train.label == 1).sum()), 'calibration_size': len(calibration), 'calibration_positives': int((calibration.label == 1).sum()), 'online_events': len(online), 'online_positive': int((online.label == 1).sum()), 'online_unknown': unknown, 'online_unknown_rate': unknown / len(online) if len(online) else 0, 'windows': total_windows, 'fully_observable_windows': complete_windows, 'incomplete_windows': total_windows - complete_windows, 'complete_window_rate': complete_windows / total_windows if total_windows else 0, 'train_two_classes': train.label.nunique() == 2, 'calibration_two_classes': calibration.label.nunique() == 2}
        reports.append(result)
        print('\n' + '=' * 70)
        print(f'DELINQUENCY TASK >= {severity}')
        print('=' * 70)
        for key, value in result.items():
            print(f'{key}: {value}')
    summary = pd.DataFrame(reports)
    summary.to_csv(output / 'cohort_feasibility_summary.csv', index=False)
    print('\n' + '=' * 70)
    print('cohort-feasibility analysis COMPLETED')
    print('Output:', output)
    print('=' * 70)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default=str(ROOT / 'result/task_feasibility'))
    parser.add_argument('--output', default=str(ROOT / 'result/cohort_feasibility'))
    parser.add_argument('--cutoff', type=int, default=202001)
    parser.add_argument('--train', type=int, default=200)
    parser.add_argument('--cal', type=int, default=200)
    parser.add_argument('--window', type=int, default=100)
    run(parser.parse_args())
if __name__ == '__main__':
    main()
