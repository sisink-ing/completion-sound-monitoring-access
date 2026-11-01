"""Compare confirmation rules on matched BTS event-time windows."""
from pathlib import Path
import argparse, json
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
ROOT = Path(__file__).resolve().parent
IN = ROOT / 'result' / 'prepare_bts'
OUT = ROOT / 'result' / 'matched_window_evaluation'
NUM = ['DayOfWeek', 'CRSDepTime', 'CRSElapsedTime', 'Distance']
CAT = ['Reporting_Airline', 'Dest']
FEATURES = NUM + CAT

def get_predictor(name):
    prep = ColumnTransformer([('numeric', Pipeline([('impute', SimpleImputer(strategy='median')), ('scale', StandardScaler())]), NUM), ('categorical', Pipeline([('impute', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))]), CAT)])
    if name == 'LR':
        model = LogisticRegression(max_iter=1500, random_state=2026)
    else:
        model = DecisionTreeClassifier(max_depth=10, min_samples_leaf=5, random_state=2026)
    return Pipeline([('prep', prep), ('model', model)])

def certify(losses, arrive_ns, closed_ns, tau):
    """First safe decision at closure or after an actual feedback-arrival timestamp."""
    n = len(losses)
    order = np.argsort(arrive_ns, kind='stable')
    arrived = arrive_ns[order]
    observed = losses[order]
    s = int(observed[arrived <= closed_ns].sum())
    m = int(np.count_nonzero(arrived <= closed_ns))

    def outcome(s, m):
        if s / n > tau + 1e-12:
            return 'Degradation'
        if (s + n - m) / n <= tau + 1e-12:
            return 'Tolerable'
        return 'Pending'
    ans = outcome(s, m)
    decision_ns = int(closed_ns)
    if ans == 'Pending':
        j = m
        while j < n:
            t = arrived[j]
            while j < n and arrived[j] == t:
                s += int(observed[j])
                j += 1
            ans = outcome(s, j)
            if ans != 'Pending':
                decision_ns = int(t)
                break
    if ans == 'Pending':
        raise AssertionError('All labels returned, but result unresolved')
    wait_ns = max(int(closed_ns), int(arrived[-1]))
    oracle = 'Degradation' if losses.mean() > tau + 1e-12 else 'Tolerable'
    if ans != oracle:
        raise AssertionError(f'Completion soundness failed: {ans} vs {oracle}')
    return {'decision': ans, 'oracle': oracle, 'labels_at_decision': int(j if m < n and int(decision_ns) > int(closed_ns) else m), 'confirmation_min': (decision_ns - int(closed_ns)) / 60000000000.0, 'waitall_min': (wait_ns - int(closed_ns)) / 60000000000.0, 'advance_min': (wait_ns - decision_ns) / 60000000000.0, 'confirmed_before_all_labels': bool(decision_ns < wait_ns)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--full', action='store_true')
    ap.add_argument('--window', type=int, default=100)
    args = ap.parse_args()
    if args.window < 1:
        raise ValueError('--window must be positive')
    train = pd.read_csv(IN / 'train.csv')
    cal = pd.read_csv(IN / 'calibration.csv')
    online = pd.read_csv(IN / ('online_full.csv' if args.full else 'online_4600.csv'))
    online['prediction_time_utc'] = pd.to_datetime(online.prediction_time_utc, utc=True)
    online['label_available_utc'] = pd.to_datetime(online.label_available_utc, utc=True)
    online = online.sort_values('event_id').reset_index(drop=True)
    rows = []
    for learner in ('LR', 'DT'):
        p = get_predictor(learner)
        p.fit(train[FEATURES], train.label.astype(int))
        ref_errors = p.predict(cal[FEATURES]) != cal.label.to_numpy(dtype=int)
        ref_loss = float(ref_errors.mean())
        tau = ref_loss + 0.02
        all_errors = (p.predict(online[FEATURES]) != online.label.to_numpy(dtype=int)).astype(int)
        for start in range(0, len(online) - args.window + 1, args.window):
            part = online.iloc[start:start + args.window]
            errors = all_errors[start:start + args.window]
            close = int(part.prediction_time_utc.iloc[-1].value)
            arrivals = part.label_available_utc.astype('int64').to_numpy()
            r = certify(errors, arrivals, close, tau)
            rows.append({'learner': learner, 'window_id': start // args.window + 1, 'start_event_id': int(part.event_id.iloc[0]), 'end_event_id': int(part.event_id.iloc[-1]), 'ref_loss': ref_loss, 'threshold': tau, 'final_loss': float(errors.mean()), **r})
    out = OUT / ('full' if args.full else 'online_4600')
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / 'matched_windows.csv', index=False)
    summary = []
    for name, grp in df.groupby('learner'):
        summary.append({'learner': name, 'n_windows': len(grp), 'degradation_windows': int((grp.decision == 'Degradation').sum()), 'oracle_agreement': float((grp.decision == grp.oracle).mean()), 'mean_confirmation_min': float(grp.confirmation_min.mean()), 'mean_waitall_min': float(grp.waitall_min.mean()), 'mean_advance_min': float(grp.advance_min.mean()), 'early_fraction': float(grp.confirmed_before_all_labels.mean()), 'reference_loss': float(grp.ref_loss.iloc[0]), 'threshold': float(grp.threshold.iloc[0])})
    pd.DataFrame(summary).to_csv(out / 'summary.csv', index=False)
    metadata = {'scope': 'STATIC LR/DT predictions, fixed nonoverlapping windows; matched stopping rules only', 'time_unit': 'real UTC minutes (not event-time steps)', 'no_cohort_training': True, 'no_retraining': True, 'no_stale_guard_audit': True, 'feedback_times': 'reconstructed from BTS arrival, with assumed 10-minute buffer'}
    (out / 'protocol.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(pd.DataFrame(summary).to_string(index=False))
    print('Outputs:', out)
if __name__ == '__main__':
    main()
