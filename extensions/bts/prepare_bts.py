"""Construct chronological BTS prediction and feedback records."""
from completion_monitoring.paths import workspace_root, external_data_root, source_file
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
ROOT = workspace_root()
RAW = ROOT / 'data' / 'raw_bts'
OUT = ROOT / 'result' / 'prepare_bts'
FEATURES = ['Reporting_Airline', 'Dest', 'DayOfWeek', 'CRSDepTime', 'CRSElapsedTime', 'Distance']

def main():
    parser = argparse.ArgumentParser(description='Prepare BTS ATL historical flight arrival-delay stream')
    parser.add_argument('--online-limit', type=int, default=4600, help='Primary online subset size, default=4600')
    parser.add_argument('--feedback-buffer-minutes', type=float, default=10.0)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    inputs = sorted(RAW.glob('bts_2024_*.csv'))
    if not inputs:
        raise FileNotFoundError(f'No monthly CSVs in {RAW}')
    dfs = []
    for path in inputs:
        d = pd.read_csv(path, low_memory=False)
        d['source_file'] = path.name
        d['source_row'] = np.arange(len(d))
        dfs.append(d)
    df = pd.concat(dfs, ignore_index=True)
    raw_rows = len(df)
    if not df.Origin.astype(str).eq('ATL').all():
        raise ValueError('Non-ATL departures found. Check the source files.')
    for col in ['Cancelled', 'Diverted', 'ArrDel15', 'CRSDepTime', 'DepDelay', 'ActualElapsedTime', 'CRSElapsedTime', 'Distance', 'DayOfWeek']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    incomplete = int((~df['Cancelled'].eq(0) | ~df['Diverted'].eq(0) | ~df['ArrDel15'].isin([0, 1])).sum())
    df = df[df['Cancelled'].eq(0) & df['Diverted'].eq(0) & df['ArrDel15'].isin([0, 1])].copy()
    critical = FEATURES + ['FlightDate', 'DepDelay', 'ActualElapsedTime']
    df = df.dropna(subset=critical).copy()
    df = df[(df['ActualElapsedTime'] > 0) & (df['CRSElapsedTime'] > 0) & (df['Distance'] > 0)].copy()
    hhmm = df['CRSDepTime'].astype(int)
    hr = hhmm // 100
    mi = hhmm % 100
    good = mi.between(0, 59) & (hr.between(0, 23) | (hr == 24) & (mi == 0))
    df = df[good].copy()
    hr = hr.loc[df.index]
    mi = mi.loc[df.index]
    base = pd.to_datetime(df['FlightDate'], format='%Y-%m-%d', errors='coerce')
    local = base + pd.to_timedelta(hr * 60 + mi, unit='m')
    df['prediction_time_utc'] = local.dt.tz_localize('America/New_York', ambiguous='NaT', nonexistent='NaT').dt.tz_convert('UTC')
    df = df.dropna(subset=['prediction_time_utc']).copy()
    df['label_available_utc'] = df['prediction_time_utc'] + pd.to_timedelta(df['DepDelay'] + df['ActualElapsedTime'] + args.feedback_buffer_minutes, unit='m')
    df = df[df['label_available_utc'] > df['prediction_time_utc']].copy()
    df = df.sort_values(['prediction_time_utc', 'Reporting_Airline', 'Flight_Number_Reporting_Airline', 'source_file', 'source_row'], kind='stable').reset_index(drop=True)
    df['label'] = df['ArrDel15'].astype(int)
    train = df.iloc[:200].copy()
    if len(train) != 200 or train.label.nunique() != 2:
        raise ValueError('Initial train cohort does not contain 200 labeled flights from both classes.')
    train_ready = train['label_available_utc'].max()
    calibration = df[df['prediction_time_utc'] > train_ready].head(200).copy()
    if len(calibration) != 200:
        raise ValueError('Insufficient calibration rows after train cohort is fully labeled.')
    cal_ready = calibration['label_available_utc'].max()
    online_full = df[df['prediction_time_utc'] > cal_ready].copy()
    if len(online_full) < args.online_limit:
        raise ValueError('Not enough online flights after calibration is fully labeled.')
    online_full = online_full.reset_index(drop=True)
    online_full['event_id'] = np.arange(1, len(online_full) + 1)
    pts = online_full['prediction_time_utc'].astype('int64').to_numpy()
    fts = online_full['label_available_utc'].astype('int64').to_numpy()
    slots = np.searchsorted(pts, fts, side='left') + 1
    online_full['release_event_id'] = np.where(slots <= len(pts), slots, pd.NA)
    online_full['feedback_delay_minutes'] = (online_full.label_available_utc - online_full.prediction_time_utc).dt.total_seconds() / 60
    online_full['feedback_delay_event_steps'] = np.where(slots <= len(pts), slots - online_full.event_id, np.nan)
    online = online_full.head(args.online_limit).copy()
    p_small = online['prediction_time_utc'].astype('int64').to_numpy()
    a_small = online['label_available_utc'].astype('int64').to_numpy()
    small_slots = np.searchsorted(p_small, a_small, side='left') + 1
    online['release_event_id'] = pd.Series(small_slots).where(small_slots <= len(online), pd.NA)
    online['feedback_delay_event_steps'] = pd.Series(np.where(small_slots <= len(online), small_slots - online.event_id, np.nan))
    train.to_csv(OUT / 'train.csv', index=False)
    calibration.to_csv(OUT / 'calibration.csv', index=False)
    online.to_csv(OUT / 'online_4600.csv', index=False)
    online_full.to_csv(OUT / 'online_full.csv', index=False)

    def nclass(x):
        return {str(int(k)): int(v) for k, v in x.label.value_counts().sort_index().items()}
    arrival = online['label_available_utc'].astype('int64').to_numpy()
    adjacent_inversion = int(np.sum(np.diff(arrival) < 0))
    ever_overtaken = int(np.sum(arrival[1:] < np.maximum.accumulate(arrival)[:-1]))
    audit = {'source': 'BTS 2024-01 to 2024-03, Origin ATL, completed nondiverted flights', 'raw_atl_rows': raw_rows, 'excluded_cancelled_diverted_or_missing_label': incomplete, 'cleaned_rows': len(df), 'all_classes': nclass(df), 'train_rows': len(train), 'train_classes': nclass(train), 'train_max_label_time_utc': str(train_ready), 'calibration_rows': len(calibration), 'calibration_classes': nclass(calibration), 'calibration_max_label_time_utc': str(cal_ready), 'online_4600_rows': len(online), 'online_4600_classes': nclass(online), 'online_full_rows': len(online_full), 'online_full_classes': nclass(online_full), 'feedback_buffer_minutes_assumed': args.feedback_buffer_minutes, 'online_4600_mean_delay_minutes': round(float(online.feedback_delay_minutes.mean()), 2), 'online_4600_median_delay_minutes': round(float(online.feedback_delay_minutes.median()), 2), 'online_4600_adjacent_feedback_inversions': adjacent_inversion, 'online_4600_overtaken_later_by_any_prior_flight': ever_overtaken, 'online_4600_labels_arriving_after_last_prediction': int((small_slots > len(online)).sum()), 'prediction_feature_columns': FEATURES, 'caveats': ['Flight outcomes and labels are real BTS historical fields; feedback times are reconstructed proxies, NOT recorded label-processing timestamps.', 'Post hoc exclusion of cancelled/diverted/incomplete flights defines a retrospective completed-flight subpopulation.', 'Train/calibration selection waits until preceding labels are available, to prevent chronological leakage.', 'Release event id is the first subsequent prediction-event slot, with end-of-stream labels requiring time-ordered feedback flush.', 'CSV contains future/outcome fields for delayed-feedback simulation; ONLY prediction_feature_columns may enter the predictor.']}
    (OUT / 'audit.json').write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    print(f'\n[OK] Prepared files: {OUT}')
if __name__ == '__main__':
    main()
