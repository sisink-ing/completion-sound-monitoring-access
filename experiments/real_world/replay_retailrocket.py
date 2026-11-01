from __future__ import annotations
import argparse
from collections import defaultdict
import importlib
import json
import os
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from completion_monitoring.paths import activate_sources, prepare_workspace, write_environment

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'results/retailrocket')
    args = parser.parse_args()
    out = args.output.resolve()
    activate_sources()
    work = prepare_workspace(out, references=False)
    os.chdir(work)
    import numpy as np
    import pandas as pd
    data = ROOT / 'data/processed/retailrocket'
    protocol = json.loads((data / 'retailrocket_protocol.json').read_text())
    frame = pd.read_csv(data / 'retailrocket_model_input.csv', float_precision='round_trip')
    feedback = pd.read_csv(data / 'retailrocket_feedback_schedule.csv')
    assert len(frame) == 10594 and len(feedback) == 10194
    assert frame.processed_event_id.tolist() == list(range(len(frame)))
    features = protocol['features']
    assert len(features) == 24
    X = frame[features].to_numpy(dtype=float)
    y = frame.label.to_numpy(dtype=int)
    arrivals = dict(zip(feedback.processed_event_id.astype(int), feedback.release_tick.astype(int)))
    assert set(arrivals) == set(range(400, len(frame)))
    assert all((t >= i for i, t in arrivals.items()))
    groups = defaultdict(list)
    for event_id, tick in arrivals.items():
        groups[tick].append(event_id)
    engine = importlib.import_module('cohort_protocol')
    engine.base.generate_delay = lambda *_args: (arrivals, dict(groups))
    runs, decisions, updates, skipped = engine.run_one('Retailrocket', X, y, 'DT', 'NATIVE', 0, 0, 0)
    metrics = pd.DataFrame(runs)
    reference = pd.read_csv(ROOT / 'data/reference/retailrocket/retailrocket_metrics.csv', float_precision='round_trip')
    joined = metrics.merge(reference, on='strategy', suffixes=('', '_reference'))
    differences = {metric: float((joined[metric] - joined[metric + '_reference']).abs().dropna().max()) for metric in ['accuracy', 'macro_f1', 'mcc', 'retrain_count', 'decision_delay', 'decision_accuracy', 'decision_count']}
    report = {'rows': len(frame), 'online_queries': len(feedback), 'features': len(features), 'input': 'archived processed holdout and release schedule', 'strategies': len(metrics), 'maximum_absolute_difference_from_archived_results': differences, 'matches_reference': all((value <= 1e-12 for value in differences.values()))}
    metrics.to_csv(out / 'metrics.csv', index=False)
    pd.DataFrame(decisions).to_csv(out / 'decisions.csv', index=False)
    pd.DataFrame(updates).to_csv(out / 'updates.csv', index=False)
    pd.DataFrame(skipped).to_csv(out / 'skipped_updates.csv', index=False)
    (out / 'verification.json').write_text(json.dumps(report, indent=2))
    write_environment(out / 'environment.json')
    print(json.dumps(report, indent=2))
if __name__ == '__main__':
    main()
