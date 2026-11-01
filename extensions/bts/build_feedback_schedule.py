"""Build a delayed-feedback schedule for the BTS replay."""
from pathlib import Path
import argparse, json
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'result' / 'build_feedback_schedule'

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--full', action='store_true', help='use all 77802 online flights')
    args = ap.parse_args()
    path = ROOT / 'result' / 'prepare_bts' / ('online_full.csv' if args.full else 'online_4600.csv')
    df = pd.read_csv(path)
    df['prediction_time_utc'] = pd.to_datetime(df['prediction_time_utc'], utc=True)
    df['label_available_utc'] = pd.to_datetime(df['label_available_utc'], utc=True)
    df = df.sort_values('event_id', kind='stable').reset_index(drop=True)
    ids = df.event_id.to_numpy(dtype=int)
    if not np.array_equal(ids, np.arange(1, len(df) + 1)):
        raise ValueError('event_id must be sequential 1,2,...')
    p = df.prediction_time_utc.astype('int64').to_numpy()
    a = df.label_available_utc.astype('int64').to_numpy()
    if not np.all(np.diff(p) >= 0) or not np.all(a > p):
        raise ValueError('Chronological or post-prediction availability check failed.')
    computed = np.searchsorted(p, a, side='left') + 1
    release = np.where(computed <= len(df), computed, -1)
    df['computed_release_event_id'] = release
    df['needs_post_stream_flush'] = release == -1
    df['delay_minutes'] = (df.label_available_utc - df.prediction_time_utc).dt.total_seconds() / 60
    O = OUT / ('full' if args.full else 'online_4600')
    O.mkdir(parents=True, exist_ok=True)
    cols = ['event_id', 'prediction_time_utc', 'label_available_utc', 'computed_release_event_id', 'needs_post_stream_flush', 'delay_minutes', 'label']
    df[cols].sort_values(['label_available_utc', 'event_id'], kind='stable').to_csv(O / 'feedback_arrival_order.csv', index=False)
    inversions = int(np.count_nonzero(np.diff(a) < 0))
    overtaken = int(np.count_nonzero(a[1:] < np.maximum.accumulate(a)[:-1]))
    stats = {'source': 'BTS ATL 2024 Jan-Mar, arrival-driven proxy feedback (not observed human verification timestamps)', 'n_predictions': int(len(df)), 'mean_delay_minutes': float(np.mean((a - p) / 60000000000.0)), 'median_delay_minutes': float(np.median((a - p) / 60000000000.0)), 'adjacent_inversions': inversions, 'later_predictions_overtaking_any_earlier_prediction': overtaken, 'labels_after_last_prediction': int(np.count_nonzero(release == -1)), 'prestream_at_prediction_ties': 'prediction before feedback on identical timestamp'}
    (O / 'feedback_audit.json').write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    print('Output:', O)
if __name__ == '__main__':
    main()
