"""Compare selected ARF and Delayed-ADWIN replays with archived results."""
import argparse
import importlib
import importlib.metadata
import json
import os
import platform
from pathlib import Path
from _bootstrap import ROOT
from completion_monitoring.paths import activate_sources, prepare_workspace, write_environment


def compare(actual, reference, metrics):
    keys = ['dataset', 'stream_seed', 'scenario', 'delay_seed', 'strategy']
    joined = actual.merge(reference, on=keys, suffixes=('', '_reference'), validate='one_to_one')
    if len(joined) != len(actual):
        raise AssertionError('Some replay rows have no archived counterpart.')
    errors = {}
    for name in metrics:
        left, right = joined[name], joined[name + '_reference']
        if not left.isna().equals(right.isna()):
            raise AssertionError(f'Missing-value pattern differs for {name}.')
        difference = (left - right).abs().dropna()
        errors[name] = float(difference.max()) if len(difference) else 0.0
    if max(errors.values()) > 1e-12:
        raise AssertionError(f'Replay differs from the reference: {errors}')
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'results/arf_environment_check')
    args = parser.parse_args()
    out = args.output.resolve()
    activate_sources()
    work = prepare_workspace(out, 'arf_holdout')
    os.chdir(work)
    import pandas as pd
    arf = importlib.import_module('arf_holdout')
    rows = []
    for family in arf.DATASETS:
        rows.extend(arf.run_one_arf_task(family, 6101, 'UNIFORM_100_100', 100, 100, 8001)[0])
    actual = pd.DataFrame(rows)
    if len(actual) != 20:
        raise AssertionError('Expected 20 ARF strategy rows.')
    reference = pd.read_csv(ROOT / 'data/reference/arf_holdout/raw.csv', float_precision='round_trip')
    metrics = ['accuracy', 'macro_f1', 'mcc', 'retrain_count', 'degradation_count', 'decision_count', 'decision_accuracy', 'decision_delay', 'mean_labels_at_decision', 'false_positive', 'false_negative', 'mean_alarm_to_update', 'monitor_coverage', 'waiting_coverage', 'calibration_coverage', 'skipped_single_class_updates', 'initial_train_accuracy', 'initial_cal_accuracy', 'initial_reference_loss']
    errors = compare(actual, reference, metrics)
    adwin = importlib.import_module('delayed_adwin')
    row, _, _ = adwin.adwin_one(('SEA', 6101, 'UNIFORM_100_100', 100, 100, 8001), 0.002)
    adwin_actual = pd.DataFrame([row])
    adwin_reference = pd.read_csv(ROOT / 'data/reference/delayed_adwin/adwin_raw.csv', float_precision='round_trip')
    adwin_metrics = ['accuracy', 'macro_f1', 'mcc', 'retrain_count', 'alarm_count', 'update_count', 'detector_feedback_used', 'detector_feedback_excluded', 'monitor_coverage', 'waiting_coverage', 'calibration_coverage']
    adwin_errors = compare(adwin_actual, adwin_reference, adwin_metrics)
    actual.to_csv(out / 'selected_task_metrics.csv', index=False)
    adwin_actual.to_csv(out / 'adwin_metrics.csv', index=False)
    report = {
        'environment_role': 'Validated ARF replay environment',
        'platform': platform.system(), 'machine': platform.machine(),
        'python': platform.python_version(),
        'packages': {name: importlib.metadata.version(name) for name in ['numpy', 'pandas', 'scipy', 'scikit-learn', 'joblib', 'scikit-multiflow']},
        'comparison': {'families': list(arf.DATASETS), 'stream_seed': 6101, 'scenario': 'UNIFORM_100_100', 'delay_seed': 8001, 'strategy_rows': len(actual), 'reference': 'data/reference/arf_holdout/raw.csv', 'maximum_absolute_errors': errors, 'matches_reference_at_tolerance': True, 'tolerance': 1e-12},
        'delayed_adwin': {'strategy_rows': 1, 'family': 'SEA', 'stream_seed': 6101, 'scenario': 'UNIFORM_100_100', 'delay_seed': 8001, 'delta': 0.002, 'reference': 'data/reference/delayed_adwin/adwin_raw.csv', 'maximum_absolute_errors': adwin_errors, 'matches_reference_at_tolerance': True, 'tolerance': 1e-12},
        'original_training_environment': 'Exact historical dependency versions were not recorded.',
    }
    write_environment(out / 'environment.json')
    (out / 'verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
