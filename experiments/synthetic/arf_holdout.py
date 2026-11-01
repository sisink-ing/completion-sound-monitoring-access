"""Evaluate monitoring strategies with Adaptive Random Forest."""
from completion_monitoring.paths import workspace_root, external_data_root, source_file
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')
os.environ.setdefault('BLIS_NUM_THREADS', '1')
import argparse
import hashlib
import importlib
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
SCRIPT_NAME = 'arf_holdout'
PROJECT_ROOT = workspace_root()
OUTPUT_DIR = PROJECT_ROOT / 'result' / SCRIPT_NAME
STREAM_DIR = OUTPUT_DIR / 'streams'
SMOKE_DIR = OUTPUT_DIR / 'smoke'
SMOKE_STREAM_DIR = SMOKE_DIR / 'streams'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
STREAM_DIR.mkdir(parents=True, exist_ok=True)
SMOKE_DIR.mkdir(parents=True, exist_ok=True)
SMOKE_STREAM_DIR.mkdir(parents=True, exist_ok=True)
DATASETS = ['SEA', 'Agrawal', 'Mixed', 'RandomTree', 'Sine']
STREAM_SEEDS = list(range(6101, 6111))
DELAY_SEEDS = list(range(8001, 8006))
LEARNER_NAME = 'ARF'
ARF_RANDOM_SEED = 2026
DEFAULT_ARF_JOBS = 6
DEFAULT_GENERATION_JOBS = 2
PAIR_TOL = 1e-12
BOOTSTRAP_REPS = 10000
BOOTSTRAP_SEED = 27027

def content_sha256(X, y):
    """
    Deterministic SHA-256 over the numerical stream contents.

    This hash is independent of npz compression metadata.
    """
    X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
    y = np.ascontiguousarray(np.asarray(y, dtype=np.int64).reshape(-1))
    h = hashlib.sha256()
    h.update(str(X.shape).encode('utf-8'))
    h.update(X.tobytes())
    h.update(str(y.shape).encode('utf-8'))
    h.update(y.tobytes())
    return h.hexdigest()

def flatten_nested(results, index):
    output = []
    for item in results:
        output.extend(item[index])
    return output

def scenario_items(engine):
    """
    Convert engine.DELAY_SCENARIOS to a stable list:
        [(name, lambda0, lambda1), ...]
    """
    items = []
    for name, config in engine.DELAY_SCENARIOS.items():
        items.append((str(name), int(config[0]), int(config[1])))
    return items

def stream_path(dataset, stream_seed, smoke_generation=False):
    base = SMOKE_STREAM_DIR if smoke_generation else STREAM_DIR
    return base / f'{dataset}_streamseed_{stream_seed}.npz'

def read_stream_file(path):
    """
    Read and validate one frozen stream.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'Missing stream file: {path}')
    with np.load(path, allow_pickle=False) as data:
        X = np.asarray(data['X'], dtype=float)
        y = np.asarray(data['y'], dtype=int).reshape(-1)
        stored_hash = ''
        if 'stream_hash' in data.files:
            try:
                stored_hash = str(data['stream_hash'].item())
            except Exception:
                stored_hash = str(data['stream_hash'])
    if X.ndim != 2:
        raise RuntimeError(f'{path}: X is not 2-D.')
    if len(X) != len(y):
        raise RuntimeError(f'{path}: X/y length mismatch.')
    if len(y) != 2200:
        raise RuntimeError(f'{path}: expected 2200 samples, found {len(y)}.')
    labels = set(np.unique(y).tolist())
    if labels != {0, 1}:
        raise RuntimeError(f'{path}: labels must be binary 0/1, found {labels}.')
    if len(np.unique(y[:200])) != 2:
        raise RuntimeError(f'{path}: first 200 training samples contain only one class.')
    frozen_hash = content_sha256(X, y)
    return (X, y, stored_hash, frozen_hash)

def build_stream_manifest():
    """
    Build a manifest from the CURRENTLY FROZEN stream files.

    This does not regenerate anything.
    """
    rows = []
    missing = []
    for dataset in DATASETS:
        for seed in STREAM_SEEDS:
            path = stream_path(dataset, seed, smoke_generation=False)
            if not path.exists():
                missing.append(str(path))
                continue
            X, y, stored_hash, frozen_hash = read_stream_file(path)
            rows.append({'dataset': dataset, 'stream_seed': int(seed), 'n_samples': int(len(y)), 'n_features': int(X.shape[1]), 'class0': int(np.sum(y == 0)), 'class1': int(np.sum(y == 1)), 'train_class0': int(np.sum(y[:200] == 0)), 'train_class1': int(np.sum(y[:200] == 1)), 'stored_generator_hash': stored_hash, 'frozen_content_sha256': frozen_hash, 'path': str(path)})
    df = pd.DataFrame(rows)
    return (df, missing)

def save_protocol_freeze():
    """
    Save formal experiment protocol before evaluation.
    """
    protocol = {'script': SCRIPT_NAME, 'datasets': DATASETS, 'stream_seeds': STREAM_SEEDS, 'delay_seeds': DELAY_SEEDS, 'learner': 'scikit-multiflow AdaptiveRandomForestClassifier', 'arf_random_seed': ARF_RANDOM_SEED, 'window_size': 100, 'epsilon': 0.02, 'update_cohort_size': 100, 'post_update_calibration_size': 100, 'strategies': ['STATIC', 'FULL_SOFT_Q30', 'FULL_COMPLETE_COHORT', 'WAIT_ALL_COMPLETE_COHORT'], 'scientific_rule': 'Formal stream and delay seeds are frozen before inspecting formal ARF results.'}
    path = OUTPUT_DIR / 'protocol_freeze.json'
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(protocol, f, indent=2, ensure_ascii=False)

def generate_one_missing_stream(dataset, seed, smoke_generation=False):
    """
    Generate ONE stream using the already verified synthetic-stream pipeline
    generator.

    Existing files are not overwritten.
    """
    output_path = stream_path(dataset, seed, smoke_generation=smoke_generation)
    if output_path.exists():
        X, y, stored_hash, frozen_hash = read_stream_file(output_path)
        return {'dataset': dataset, 'stream_seed': int(seed), 'status': 'REUSED_EXISTING', 'n_samples': int(len(y)), 'n_features': int(X.shape[1]), 'stored_generator_hash': stored_hash, 'frozen_content_sha256': frozen_hash, 'path': str(output_path), 'generation_seconds': 0.0}
    start = time.perf_counter()
    step23 = importlib.import_module('unseen_stream_pipeline')
    record = step23.generate_one_stream(dataset, seed)
    if not record['generation_ok']:
        raise RuntimeError(f'Stream generation failed: {dataset}, seed={seed}')
    X = np.asarray(record['X'], dtype=float)
    y = np.asarray(record['y'], dtype=int)
    dd = np.asarray(record.get('dd', np.zeros(len(y), dtype=int)), dtype=int)
    stored_hash = str(record.get('stream_hash', ''))
    frozen_hash = content_sha256(X, y)
    np.savez_compressed(output_path, X=X, y=y, dd=dd, dataset=str(dataset), stream_seed=int(seed), stream_hash=stored_hash, frozen_content_sha256=frozen_hash)
    X_check, y_check, stored_hash_check, frozen_hash_check = read_stream_file(output_path)
    if frozen_hash_check != frozen_hash:
        raise RuntimeError(f'Frozen stream changed immediately after saving: {dataset}, seed={seed}')
    return {'dataset': dataset, 'stream_seed': int(seed), 'status': 'GENERATED_NEW', 'n_samples': int(len(y_check)), 'n_features': int(X_check.shape[1]), 'stored_generator_hash': stored_hash_check, 'frozen_content_sha256': frozen_hash_check, 'path': str(output_path), 'generation_seconds': float(time.perf_counter() - start)}

def run_generation(smoke=False):
    """
    Formal generation will NOT overwrite already existing
    formal stream files.

    Therefore your currently generated 50 streams remain frozen.
    """
    print('=' * 120)
    print('STRONG-BACKBONE HOLDOUT -- STREAM FREEZE / GENERATION')
    print('=' * 120)
    print('Frozen stream seeds:', STREAM_SEEDS)
    print('Smoke generation:', smoke)
    print()
    if smoke:
        jobs = [('SEA', 6101)]
        n_jobs = 1
    else:
        jobs = [(dataset, seed) for dataset in DATASETS for seed in STREAM_SEEDS]
        n_jobs = int(os.environ.get('ARF_GENERATION_JOBS', DEFAULT_GENERATION_JOBS))
    start = time.perf_counter()
    records = Parallel(n_jobs=n_jobs, backend='loky', verbose=10)((delayed(generate_one_missing_stream)(dataset, seed, smoke_generation=smoke) for dataset, seed in jobs))
    records_df = pd.DataFrame(records)
    if smoke:
        records_df.to_csv(SMOKE_DIR / 'smoke_stream_manifest.csv', index=False)
        print(records_df.to_string(index=False))
        print()
        print('Smoke generation elapsed:', round(time.perf_counter() - start, 2), 's')
        return
    manifest_df, missing = build_stream_manifest()
    manifest_df.to_csv(OUTPUT_DIR / 'stream_manifest.csv', index=False)
    if missing:
        print()
        print('MISSING STREAM FILES:')
        for x in missing:
            print(x)
        raise RuntimeError(f'{len(missing)} formal streams are missing.')
    checks = []
    checks.append({'check': 'exactly 50 frozen streams available', 'pass': len(manifest_df) == 50, 'value': len(manifest_df)})
    checks.append({'check': 'all 50 frozen content hashes unique', 'pass': manifest_df['frozen_content_sha256'].nunique() == 50, 'value': manifest_df['frozen_content_sha256'].nunique()})
    per_family = manifest_df.groupby('dataset')['frozen_content_sha256'].nunique()
    checks.append({'check': '10 unique frozen streams within every family', 'pass': bool((per_family == 10).all()), 'value': '; '.join((f'{k}:{v}' for k, v in per_family.items()))})
    checks_df = pd.DataFrame(checks)
    checks_df.to_csv(OUTPUT_DIR / 'stream_generation_checks.csv', index=False)
    save_protocol_freeze()
    print()
    print('SCIENTIFIC CHECKS')
    print(checks_df.to_string(index=False))
    if not checks_df['pass'].all():
        raise RuntimeError('Formal stream checks failed.')
    print()
    print('Formal frozen stream set is valid.')
    print('Generation/freeze elapsed:', round(time.perf_counter() - start, 2), 's')

def patch_engine_to_arf(engine):
    """
    IMPORTANT:
    This function is called INSIDE each joblib worker.

    That prevents loky subprocesses from silently re-importing
    the original LR/DT create_model() function.
    """
    try:
        from skmultiflow.meta import AdaptiveRandomForestClassifier
    except Exception as exc:
        raise ImportError('AdaptiveRandomForestClassifier requires the scikit-multiflow backend.') from exc

    def create_arf_model(_learner_name):
        return AdaptiveRandomForestClassifier(split_criterion='gini', random_state=ARF_RANDOM_SEED)
    engine.base.create_model = create_arf_model
    test_model = engine.base.create_model('ARF')
    if test_model.__class__.__name__ != 'AdaptiveRandomForestClassifier':
        raise RuntimeError('ARF model patch failed.')
    return engine

def run_one_arf_task(dataset, stream_seed, scenario, lambda0, lambda1, delay_seed):
    """
    One combination:
        one frozen stream
        one delay scenario
        one delay seed

    engine.run_one internally evaluates the four strategies.
    """
    engine = importlib.import_module('cohort_protocol')
    engine = patch_engine_to_arf(engine)
    if engine.WINDOW_SIZE != 100:
        raise RuntimeError('WINDOW_SIZE changed.')
    if abs(float(engine.EPSILON) - 0.02) > 1e-15:
        raise RuntimeError('EPSILON changed.')
    if engine.UPDATE_TRAIN_SIZE != 100:
        raise RuntimeError('UPDATE_TRAIN_SIZE changed.')
    if engine.POST_UPDATE_CAL_SIZE != 100:
        raise RuntimeError('POST_UPDATE_CAL_SIZE changed.')
    expected_strategies = ['STATIC', 'FULL_SOFT_Q30', 'FULL_COMPLETE_COHORT', 'WAIT_ALL_COMPLETE_COHORT']
    if list(engine.STRATEGIES) != expected_strategies:
        raise RuntimeError('Strategy list changed.')
    path = stream_path(dataset, stream_seed, smoke_generation=False)
    X, y, stored_hash, frozen_hash = read_stream_file(path)
    stream_id = f'{dataset}_s{stream_seed}'
    runs, decisions, updates, skipped = engine.run_one(stream_id, X, y, LEARNER_NAME, scenario, lambda0, lambda1, delay_seed)
    for collection in [runs, decisions, updates, skipped]:
        for row in collection:
            row['dataset'] = dataset
            row['dataset_family'] = dataset
            row['stream_id'] = stream_id
            row['stream_seed'] = int(stream_seed)
            row['stored_generator_hash'] = stored_hash
            row['frozen_content_sha256'] = frozen_hash
            row['learner'] = LEARNER_NAME
            row['arf_random_seed'] = ARF_RANDOM_SEED
            row['arf_backend'] = 'skmultiflow.AdaptiveRandomForestClassifier'
    return (runs, decisions, updates, skipped)

def stratified_bootstrap_ci(diff_df, reps=BOOTSTRAP_REPS):
    """
    Family-stratified bootstrap.

    FIX:
    Iterate ONLY over families actually present in diff_df.

    This allows smoke tests containing SEA only.
    """
    if len(diff_df) == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    family_values = {family: group['diff'].to_numpy(dtype=float) for family, group in diff_df.groupby('dataset_family')}
    if not family_values:
        return (np.nan, np.nan)
    observed_families = sorted(family_values.keys())
    means = np.empty(reps, dtype=float)
    for rep in range(reps):
        sampled = []
        for family in observed_families:
            values = family_values[family]
            indices = rng.integers(0, len(values), size=len(values))
            sampled.extend(values[indices].tolist())
        means[rep] = float(np.mean(sampled))
    return (float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975)))

def make_stream_level_comparison(stream_df):
    from scipy.stats import wilcoxon
    complete_df = stream_df[stream_df['strategy'] == 'FULL_COMPLETE_COHORT'].copy()
    comparator_list = ['STATIC', 'FULL_SOFT_Q30', 'WAIT_ALL_COMPLETE_COHORT']
    metrics = ['accuracy', 'macro_f1', 'mcc']
    comparison_rows = []
    for comparator in comparator_list:
        comparator_df = stream_df[stream_df['strategy'] == comparator].copy()
        merged = complete_df.merge(comparator_df, on=['dataset_family', 'stream_seed'], suffixes=('_complete', '_comparator'), validate='one_to_one')
        if len(merged) == 0:
            continue
        for metric in metrics:
            complete_values = merged[f'{metric}_complete'].to_numpy(dtype=float)
            comparator_values = merged[f'{metric}_comparator'].to_numpy(dtype=float)
            diff = complete_values - comparator_values
            wins = int(np.sum(diff > PAIR_TOL))
            ties = int(np.sum(np.abs(diff) <= PAIR_TOL))
            losses = int(np.sum(diff < -PAIR_TOL))
            if len(diff) < 2 or np.all(np.abs(diff) <= PAIR_TOL):
                p_value = np.nan
            else:
                try:
                    p_value = float(wilcoxon(diff, alternative='two-sided', zero_method='wilcox').pvalue)
                except ValueError:
                    p_value = np.nan
            bootstrap_df = merged[['dataset_family', 'stream_seed']].copy()
            bootstrap_df['diff'] = diff
            ci_low, ci_high = stratified_bootstrap_ci(bootstrap_df)
            comparison_rows.append({'comparator': comparator, 'metric': metric, 'n_streams': int(len(diff)), 'n_families': int(bootstrap_df['dataset_family'].nunique()), 'mean_diff_complete_minus_comparator': float(np.mean(diff)), 'median_diff': float(np.median(diff)), 'bootstrap_95_low': ci_low, 'bootstrap_95_high': ci_high, 'wins': wins, 'ties': ties, 'losses': losses, 'wilcoxon_p': p_value})
    return pd.DataFrame(comparison_rows)

def run_evaluation(smoke=False, jobs=None):
    print('=' * 120)
    print('STRONG-BACKBONE INDEPENDENT HOLDOUT')
    print('ADAPTIVE RANDOM FOREST')
    print('=' * 120)
    print('Frozen stream seeds:', STREAM_SEEDS)
    print('Frozen delay seeds:', DELAY_SEEDS)
    print('ARF random seed:', ARF_RANDOM_SEED)
    print('Smoke test:', smoke)
    print()
    try:
        from skmultiflow.meta import AdaptiveRandomForestClassifier
    except Exception as exc:
        raise ImportError('AdaptiveRandomForestClassifier requires the scikit-multiflow backend.') from exc
    test_arf = AdaptiveRandomForestClassifier(random_state=ARF_RANDOM_SEED)
    print('ARF import:', test_arf.__class__.__name__, 'OK')
    manifest_df, missing = build_stream_manifest()
    if missing:
        print()
        print('Missing formal streams:')
        for item in missing:
            print(item)
        raise RuntimeError('Formal stream set is incomplete. Do not run ARF evaluation.')
    if len(manifest_df) != 50:
        raise RuntimeError(f'Expected 50 frozen streams, found {len(manifest_df)}.')
    if manifest_df['frozen_content_sha256'].nunique() != 50:
        raise RuntimeError('Frozen stream content hashes are not all unique.')
    manifest_df.to_csv(OUTPUT_DIR / 'stream_manifest.csv', index=False)
    save_protocol_freeze()
    engine = importlib.import_module('cohort_protocol')
    scenarios = scenario_items(engine)
    if smoke:
        selected_streams = [('SEA', 6101)]
        selected_scenarios = [scenarios[0]]
        selected_delay_seeds = [8001]
        n_jobs = 1
        eval_output_dir = SMOKE_DIR
    else:
        selected_streams = [(dataset, seed) for dataset in DATASETS for seed in STREAM_SEEDS]
        selected_scenarios = scenarios
        selected_delay_seeds = DELAY_SEEDS
        if jobs is not None:
            n_jobs = int(jobs)
        else:
            n_jobs = int(os.environ.get('ARF_JOBS', DEFAULT_ARF_JOBS))
        eval_output_dir = OUTPUT_DIR
    eval_output_dir.mkdir(parents=True, exist_ok=True)
    tasks = [(dataset, stream_seed, scenario, lambda0, lambda1, delay_seed) for dataset, stream_seed in selected_streams for scenario, lambda0, lambda1 in selected_scenarios for delay_seed in selected_delay_seeds]
    print('Evaluation task combinations:', len(tasks))
    print('Each combination runs 4 strategies.')
    print('Expected strategy-level rows:', len(tasks) * 4)
    print('Parallel jobs:', n_jobs)
    print('Output directory:', eval_output_dir)
    print()
    start = time.perf_counter()
    results = Parallel(n_jobs=n_jobs, backend='loky', verbose=10)((delayed(run_one_arf_task)(*task) for task in tasks))
    run_df = pd.DataFrame(flatten_nested(results, 0))
    decision_df = pd.DataFrame(flatten_nested(results, 1))
    update_df = pd.DataFrame(flatten_nested(results, 2))
    skipped_df = pd.DataFrame(flatten_nested(results, 3))
    run_df.to_csv(eval_output_dir / 'raw.csv', index=False)
    decision_df.to_csv(eval_output_dir / 'decision_raw.csv', index=False)
    update_df.to_csv(eval_output_dir / 'update_raw.csv', index=False)
    skipped_df.to_csv(eval_output_dir / 'skipped_update_raw.csv', index=False)
    overall_df = run_df.groupby('strategy', as_index=False).agg(runs=('accuracy', 'count'), accuracy=('accuracy', 'mean'), macro_f1=('macro_f1', 'mean'), mcc=('mcc', 'mean'), retrain_count=('retrain_count', 'mean'), oracle_agreement=('decision_accuracy', 'mean'), confirmation_delay=('decision_delay', 'mean'), false_positive=('false_positive', 'sum'), false_negative=('false_negative', 'sum'), monitor_coverage=('monitor_coverage', 'mean'), skipped_single_class_updates=('skipped_single_class_updates', 'sum'))
    strategy_order = ['STATIC', 'FULL_SOFT_Q30', 'FULL_COMPLETE_COHORT', 'WAIT_ALL_COMPLETE_COHORT']
    order_map = {strategy: index for index, strategy in enumerate(strategy_order)}
    overall_df['_order'] = overall_df['strategy'].map(order_map)
    overall_df = overall_df.sort_values('_order').drop(columns=['_order']).reset_index(drop=True)
    overall_df.to_csv(eval_output_dir / 'overall_summary.csv', index=False)
    main_table = overall_df[['strategy', 'accuracy', 'macro_f1', 'mcc', 'retrain_count', 'oracle_agreement', 'confirmation_delay', 'false_positive', 'false_negative']].copy()
    main_table.to_csv(eval_output_dir / 'main_paper_table.csv', index=False)
    scenario_df = run_df.groupby(['scenario', 'strategy'], as_index=False).agg(runs=('accuracy', 'count'), accuracy=('accuracy', 'mean'), macro_f1=('macro_f1', 'mean'), mcc=('mcc', 'mean'), oracle_agreement=('decision_accuracy', 'mean'), confirmation_delay=('decision_delay', 'mean'), fp=('false_positive', 'sum'), fn=('false_negative', 'sum'))
    scenario_df.to_csv(eval_output_dir / 'scenario_summary.csv', index=False)
    stream_df = run_df.groupby(['dataset_family', 'stream_seed', 'strategy'], as_index=False).agg(runs=('accuracy', 'count'), accuracy=('accuracy', 'mean'), macro_f1=('macro_f1', 'mean'), mcc=('mcc', 'mean'), retrain_count=('retrain_count', 'mean'), oracle_agreement=('decision_accuracy', 'mean'), confirmation_delay=('decision_delay', 'mean'))
    stream_df.to_csv(eval_output_dir / 'stream_strategy_summary.csv', index=False)
    paired_df = make_stream_level_comparison(stream_df)
    paired_df.to_csv(eval_output_dir / 'stream_level_paired_comparison.csv', index=False)
    taskmatched_df = overall_df[overall_df['strategy'].isin(['FULL_COMPLETE_COHORT', 'WAIT_ALL_COMPLETE_COHORT'])].copy()
    taskmatched_df.to_csv(eval_output_dir / 'complete_vs_waitall_arf.csv', index=False)
    safety_rows = []
    for strategy in ['FULL_COMPLETE_COHORT', 'WAIT_ALL_COMPLETE_COHORT']:
        selected = overall_df[overall_df['strategy'] == strategy]
        if len(selected) != 1:
            safety_rows.append({'strategy': strategy, 'check': 'strategy row exists exactly once', 'pass': False, 'value': len(selected)})
            continue
        row = selected.iloc[0]
        oracle = float(row['oracle_agreement'])
        fp = int(row['false_positive'])
        fn = int(row['false_negative'])
        safety_rows.extend([{'strategy': strategy, 'check': 'oracle agreement == 1', 'pass': bool(abs(oracle - 1.0) < 1e-12), 'value': oracle}, {'strategy': strategy, 'check': 'false positives == 0', 'pass': fp == 0, 'value': fp}, {'strategy': strategy, 'check': 'false negatives == 0', 'pass': fn == 0, 'value': fn}])
    checks_df = pd.DataFrame(safety_rows)
    checks_df.to_csv(eval_output_dir / 'safety_checks.csv', index=False)
    print()
    print('=' * 120)
    print('MAIN PAPER TABLE')
    print('=' * 120)
    print(main_table.to_string(index=False))
    print()
    print('=' * 120)
    print('STREAM-LEVEL PAIRED COMPARISON')
    print('=' * 120)
    if len(paired_df):
        print(paired_df.to_string(index=False))
    else:
        print('No paired comparison rows.')
    print()
    print('=' * 120)
    print('SAFETY CHECKS')
    print('=' * 120)
    print(checks_df.to_string(index=False))
    ours_rows = overall_df[overall_df['strategy'] == 'FULL_COMPLETE_COHORT']
    wait_rows = overall_df[overall_df['strategy'] == 'WAIT_ALL_COMPLETE_COHORT']
    if len(ours_rows) == 1 and len(wait_rows) == 1:
        ours = ours_rows.iloc[0]
        wait = wait_rows.iloc[0]
        ours_delay = float(ours['confirmation_delay'])
        wait_delay = float(wait['confirmation_delay'])
        if np.isfinite(ours_delay) and np.isfinite(wait_delay) and (wait_delay > 0):
            delay_reduction = (wait_delay - ours_delay) / wait_delay
        else:
            delay_reduction = np.nan
        print()
        print('=' * 120)
        print('KEY COMPLETE vs WAIT-ALL')
        print('SAME ARF BACKBONE')
        print('=' * 120)
        print('Accuracy:', f"{float(ours['accuracy']):.6f}", 'vs', f"{float(wait['accuracy']):.6f}")
        print('Macro-F1:', f"{float(ours['macro_f1']):.6f}", 'vs', f"{float(wait['macro_f1']):.6f}")
        print('MCC:', f"{float(ours['mcc']):.6f}", 'vs', f"{float(wait['mcc']):.6f}")
        print('Oracle agreement:', f"{float(ours['oracle_agreement']):.6f}", 'vs', f"{float(wait['oracle_agreement']):.6f}")
        print('Confirmation delay:', f'{ours_delay:.3f}', 'vs', f'{wait_delay:.3f}')
        if np.isfinite(delay_reduction):
            print('Delay reduction:', f'{100.0 * delay_reduction:.2f}%')
    print()
    print('=' * 120)
    print('RUN COMPLETE')
    print('=' * 120)
    print('Elapsed:', round(time.perf_counter() - start, 2), 'seconds')
    print('Output:', eval_output_dir)

def main():
    parser = argparse.ArgumentParser(description='ARF holdout ARF strong-backbone independent holdout.')
    parser.add_argument('--stage', choices=['generate', 'evaluate'], required=True)
    parser.add_argument('--smoke', action='store_true', help='Run the small smoke-test configuration.')
    parser.add_argument('--jobs', type=int, default=None, help='Formal ARF evaluation parallel jobs. Recommended: 6 on your 20-core machine.')
    args = parser.parse_args()
    if args.stage == 'generate':
        run_generation(smoke=args.smoke)
    elif args.stage == 'evaluate':
        run_evaluation(smoke=args.smoke, jobs=args.jobs)
if __name__ == '__main__':
    main()
