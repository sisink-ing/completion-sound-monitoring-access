"""Check synthetic stream identity across generation seeds."""
import os
import importlib
import pandas as pd
v2 = importlib.import_module('unseen_stream_holdout')
DATASETS = ['SEA', 'Agrawal', 'Mixed', 'RandomTree', 'Sine']
SEED_A = 6001
SEED_B = 6002
OUTPUT_DIR = os.path.join('result', 'unseen_stream_pipeline')
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_CSV = os.path.join(OUTPUT_DIR, 'stream_seed_checks.csv')

def short_hash(value):
    if not value:
        return ''
    return value[:16]

def main():
    print('=' * 125)
    print('V2 STREAM-SEED REPRODUCIBILITY CHECK')
    print('=' * 125)
    rows = []
    for dataset in DATASETS:
        print()
        print(f'[{dataset}]')
        run1 = v2.generate_one_stream_v2(dataset, SEED_A)
        run2 = v2.generate_one_stream_v2(dataset, SEED_A)
        run3 = v2.generate_one_stream_v2(dataset, SEED_B)
        for name, result in [('seed_A_run1', run1), ('seed_A_run2', run2), ('seed_B', run3)]:
            if not result['generation_ok']:
                raise RuntimeError(f"{dataset} / {name} generation failed:\n{result['error']}")
        same_seed_same_hash = run1['stream_hash'] == run2['stream_hash']
        different_seed_different_hash = run1['stream_hash'] != run3['stream_hash']
        passed = same_seed_same_hash and different_seed_different_hash
        print(f"seed {SEED_A} run1: {short_hash(run1['stream_hash'])}")
        print(f"seed {SEED_A} run2: {short_hash(run2['stream_hash'])}")
        print(f"seed {SEED_B}:      {short_hash(run3['stream_hash'])}")
        print('same seed -> same hash:', same_seed_same_hash)
        print('different seed -> different hash:', different_seed_different_hash)
        print('STATUS:', 'PASS ' if passed else 'FAIL ❌')
        rows.append({'dataset': dataset, 'seed_A': SEED_A, 'seed_B': SEED_B, 'hash_A_run1': run1['stream_hash'], 'hash_A_run2': run2['stream_hash'], 'hash_B': run3['stream_hash'], 'same_seed_same_hash': same_seed_same_hash, 'different_seed_different_hash': different_seed_different_hash, 'PASS': passed, 'samples': run1['n_samples'], 'features': run1['n_features'], 'class0': run1['class0'], 'class1': run1['class1'], 'train_class0': run1['train_class0'], 'train_class1': run1['train_class1'], 'generation_seconds_run1': run1['generation_seconds'], 'generation_seconds_run2': run2['generation_seconds'], 'generation_seconds_seedB': run3['generation_seconds']})
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    print()
    print('=' * 125)
    print(df[['dataset', 'same_seed_same_hash', 'different_seed_different_hash', 'PASS']].to_string(index=False))
    print('=' * 125)
    all_pass = bool(df['PASS'].all())
    if all_pass:
        print()
        print('ALL DATASETS PASS ')
        print()
        print('Same seed -> identical stream.')
        print('Different seed -> different stream.')
        print()
        print('synthetic-stream pipeline reproducibility check PASSED.')
        print('The final unseen stream-realization experiment can now be treated as reproducible.')
    else:
        print()
        print('REPRODUCIBILITY CHECK FAILED ❌')
        print()
        failed = df[df['PASS'] == False]['dataset'].tolist()
        print('Failed datasets:', failed)
        print()
        print('Synthetic stream reproducibility check failed.')
    print()
    print('Saved:')
    print(OUTPUT_CSV)
    print('=' * 125)
if __name__ == '__main__':
    main()
