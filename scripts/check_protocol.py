import argparse
import importlib
import json
import os
from pathlib import Path
from _bootstrap import ROOT
from completion_monitoring.paths import activate_sources, prepare_workspace, write_environment

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'results/smoke')
    args = parser.parse_args()
    out = args.output.resolve()
    activate_sources()
    work = prepare_workspace(out, references=False)
    os.chdir(work)
    import numpy as np
    import pandas as pd
    engine = importlib.import_module('cohort_protocol')
    with np.load(ROOT / 'data/frozen/lrdt/SEA_streamseed_6001.npz', allow_pickle=False) as stream:
        X, y = (stream['X'], stream['y'])
    runs, decisions, updates, skipped = ([], [], [], [])
    for learner in ('LR', 'DT'):
        records = engine.run_one('SEA', X, y, learner, 'UNIFORM_100_100', 100, 100, 7001)
        for target, values in zip((runs, decisions, updates, skipped), records):
            target.extend(values)
    metrics = pd.DataFrame(runs)
    confirmed = pd.DataFrame(decisions)
    sound = metrics[metrics.strategy.isin(['FULL_COMPLETE_COHORT', 'WAIT_ALL_COMPLETE_COHORT'])]
    assert len(sound) == 4
    assert (sound.decision_count > 0).all()
    assert (sound.decision_accuracy == 1).all()
    assert (sound.false_positive == 0).all() and (sound.false_negative == 0).all()
    metrics.to_csv(out / 'metrics.csv', index=False)
    confirmed.to_csv(out / 'decisions.csv', index=False)
    pd.DataFrame(updates).to_csv(out / 'updates.csv', index=False)
    report = {'passed': True, 'dataset': 'SEA', 'stream_seed': 6001, 'delay_seed': 7001, 'learners': ['LR', 'DT'], 'strategy_runs': len(metrics), 'decisions': len(confirmed), 'completion_sound_decision_accuracy': 1.0, 'scope': 'protocol integration test'}
    (out / 'verification.json').write_text(json.dumps(report, indent=2))
    write_environment(out / 'environment.json')
    print(json.dumps(report, indent=2))
if __name__ == '__main__':
    main()
