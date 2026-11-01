"""Check mortgage replay metrics and update records."""
import sys
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parent
BASE = ROOT / 'result/calendar_replay/full_20core'
status = BASE / 'experiment_status.csv'
if not status.exists():
    sys.exit(f'Not found: {status}. Run run_20core.py first.')
s = pd.read_csv(status)
if len(s) != 9 or (s.status != 'SUCCESS').any():
    sys.exit('Not all 9 task-learner combinations completed successfully; inspect run.log.')
checks = 0
for t in (1, 2, 3):
    for learner in ('LR', 'DT', 'ARF'):
        d = BASE / f'delq{t}_{learner.lower()}'
        m = pd.read_csv(d / 'main_table.csv')
        assert len(m) == 5 and set(m.strategy) == {'STATIC', 'ARRIVED_ONLY', 'FULL_SOFT_Q30', 'FULL_COMPLETE_COHORT', 'WAIT_ALL_COMPLETE_COHORT'}
        assert (m.n_online >= 9000).all()
        assert ((m.monitor_coverage.fillna(0) + m.waiting_coverage.fillna(0) + m.calibration_coverage.fillna(0))[m.strategy != 'STATIC'] - 1).abs().max() < 1e-10
        predicted = pd.read_csv(d / 'predictions.csv')
        assert len(predicted) == len(m) * int(m.n_online.iloc[0])
        p = d / 'paired_windows.csv'
        if p.exists() and p.stat().st_size:
            try:
                paired = pd.read_csv(p)
            except pd.errors.EmptyDataError:
                paired = pd.DataFrame()
            if len(paired):
                assert (paired.advance_months >= 0).all()
                assert (paired.same_decision == 1).all()
                assert (paired.complete_labels <= paired.wait_labels).all()
        updates = d / 'updates.csv'
        if updates.exists() and updates.stat().st_size:
            try:
                ud = pd.read_csv(updates)
            except pd.errors.EmptyDataError:
                ud = pd.DataFrame()
            if len(ud):
                assert (ud.latest_training_label_month < ud.update_month).all()
        checks += 1
print('PASS: 9 tasks; valid calendar-month pairs, no future training labels, phase coverages sum to 1.')
print('Verified experiments:', checks)
