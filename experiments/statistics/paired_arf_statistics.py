"""Compute paired-stream ARF statistics and exact signed-rank tests."""
from pathlib import Path
import argparse
import hashlib
import json
import platform
import sys
import numpy as np
import pandas as pd
import scipy
from scipy.stats import rankdata, wilcoxon
METRICS = ['accuracy', 'macro_f1', 'mcc']
COMPARATORS = [('STATIC', 'Static'), ('FULL_SOFT_Q30', 'Soft-q30'), ('WAIT_ALL_COMPLETE_COHORT', 'Wait-All Complete'), ('ADWIN_ARRIVAL_ARF', 'Delayed-ADWIN')]
TOL = 1e-12
BOOTSTRAP_REPS = 5000
BOOTSTRAP_SEED = 30300

def holm_adjust(p):
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    adjusted[order] = np.minimum(1.0, np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order]))
    return adjusted

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, default=Path(__file__).parent / 'source_stream_scores.csv')
    parser.add_argument('--archive', type=Path, default=Path(__file__).parent / 'archived_table8_statistics.csv')
    parser.add_argument('--output', type=Path, default=Path(__file__).parent / 'result' / 'paired_arf_statistics')
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    source = pd.read_csv(args.input, float_precision='round_trip')
    archived = pd.read_csv(args.archive, float_precision='round_trip')
    assert len(source) == 250
    assert not source.duplicated(['dataset', 'stream_seed', 'strategy']).any()
    counts = source.groupby(['dataset', 'stream_seed']).size()
    assert len(counts) == 50 and (counts == 5).all()
    families = sorted(source.dataset.unique())
    assert len(families) == 5
    for family in families:
        assert source.loc[source.dataset == family, 'stream_seed'].nunique() == 10
    wide = None
    rank_rows = []
    summaries = []
    wait_accuracy = None
    for metric in METRICS:
        pivot = source.pivot(index=['dataset', 'stream_seed'], columns='strategy', values=metric).sort_index()
        assert pivot.shape == (50, 5) and (not pivot.isna().any().any())
        if wide is None:
            wide = pd.DataFrame(index=pivot.index)
        for comparator, display in COMPARATORS:
            ours = pivot['FULL_COMPLETE_COHORT'].to_numpy()
            other = pivot[comparator].to_numpy()
            d = ours - other
            zero = np.abs(d) <= TOL
            z = d[~zero]
            assert len(z) > 0
            _, multiplicities = np.unique(np.abs(z), return_counts=True)
            absolute_tie_groups = int(np.sum(multiplicities > 1))
            assert absolute_tie_groups == 0, 'Input has absolute-rank ties; use a tie-aware exact permutation test.'
            ranks = rankdata(np.abs(z), method='average')
            r_plus = float(ranks[z > 0].sum())
            r_minus = float(ranks[z < 0].sum())
            test = wilcoxon(z, zero_method='wilcox', alternative='two-sided', correction=False, method='exact')
            assert test.statistic == min(r_plus, r_minus)
            full_ranks = np.full(50, np.nan)
            full_ranks[~zero] = ranks
            wide[f'{comparator}__{metric}'] = d
            for k, (family, seed) in enumerate(pivot.index):
                rank_rows.append({'dataset': family, 'stream_seed': int(seed), 'comparator': display, 'metric': metric, 'complete_score': float(ours[k]), 'comparator_score': float(other[k]), 'difference': float(d[k]), 'zero_difference': bool(zero[k]), 'absolute_rank': None if zero[k] else float(full_ranks[k]), 'positive_rank_contribution': float(full_ranks[k]) if d[k] > TOL else 0.0, 'negative_rank_contribution': float(full_ranks[k]) if d[k] < -TOL else 0.0})
            if comparator == 'WAIT_ALL_COMPLETE_COHORT' and metric == 'accuracy':
                wait_accuracy = pd.DataFrame(rank_rows[-50:])
            family_d = [d[pivot.index.get_level_values('dataset') == f] for f in families]
            rng = np.random.default_rng(BOOTSTRAP_SEED)
            bootstrap = np.empty(BOOTSTRAP_REPS)
            for b in range(BOOTSTRAP_REPS):
                sample = np.concatenate([arr[rng.integers(0, len(arr), len(arr))] for arr in family_d])
                bootstrap[b] = np.mean(sample)
            low, high = np.quantile(bootstrap, [0.025, 0.975], method='linear')
            summaries.append({'comparator': display, 'metric': metric, 'n_total': 50, 'zero_differences': int(zero.sum()), 'n_effective': len(z), 'absolute_rank_tie_groups': absolute_tie_groups, 'wins': int((d > TOL).sum()), 'ties': int(zero.sum()), 'losses': int((d < -TOL).sum()), 'R_plus': r_plus, 'R_minus': r_minus, 'W': float(test.statistic), 'mean_difference': float(d.mean()), 'ci95_low': float(low), 'ci95_high': float(high), 'p_exact': float(test.pvalue), 'method': 'exact', 'alternative': 'two-sided', 'zero_method': 'wilcox', 'continuity_correction': False, 'zero_tolerance': TOL, 'bootstrap_repetitions': BOOTSTRAP_REPS, 'bootstrap_seed': BOOTSTRAP_SEED, 'bootstrap_method': 'family-stratified paired percentile', 'quantile_method': 'linear', 'rng': 'PCG64'})
    result = pd.DataFrame(summaries)
    result['p_holm_12'] = holm_adjust(result.p_exact.to_numpy())
    comparator_names = {'Delayed-ADWIN-Arrival': 'Delayed-ADWIN'}
    archived['comparator'] = archived.comparator.replace(comparator_names)
    check = result.merge(archived, on=['comparator', 'metric'], suffixes=('_recomputed', '_archived'), validate='one_to_one')
    maximum_errors = {}
    for original, new in [('mean_difference', 'mean_difference'), ('ci95_low', 'ci95_low'), ('ci95_high', 'ci95_high'), ('p_wilcoxon', 'p_exact'), ('p_holm_12', 'p_holm_12')]:
        left = new + '_recomputed' if new in ['mean_difference', 'ci95_low', 'ci95_high', 'p_holm_12'] else new
        right = original + '_archived' if original in ['mean_difference', 'ci95_low', 'ci95_high', 'p_holm_12'] else original
        err = np.abs(check[left] - check[right]).to_numpy()
        maximum_errors[original] = float(err.max())
        assert np.all(err < 1e-12), (original, err.max())
    for field in ['wins', 'ties', 'losses']:
        assert (check[field + '_recomputed'] == check[field + '_archived']).all()
    result['matches_archived_table8_to_displayed_precision'] = True
    wide.reset_index().to_csv(out / 'table8_50_stream_paired_differences.csv', index=False, float_format='%.17g')
    result.to_csv(out / 'table8_effective_n_and_rank_statistics.csv', index=False, float_format='%.17g')
    pd.DataFrame(rank_rows).to_csv(out / 'table8_all_600_pairwise_rank_records.csv', index=False, float_format='%.17g')
    wait_accuracy.to_csv(out / 'waitall_accuracy_50_stream_differences.csv', index=False, float_format='%.17g')
    wait_accuracy.loc[~wait_accuracy.zero_difference].sort_values('absolute_rank').to_csv(out / 'waitall_accuracy_14_nonzero_ranks.csv', index=False, float_format='%.17g')
    w = result[(result.comparator == 'Wait-All Complete') & (result.metric == 'accuracy')].iloc[0]
    subset_counts = np.array([1], dtype=np.int64)
    for rank in range(1, 15):
        new = np.zeros(len(subset_counts) + rank, dtype=np.int64)
        new[:len(subset_counts)] += subset_counts
        new[rank:] += subset_counts
        subset_counts = new
    extreme_patterns = int(2 * subset_counts[:int(w.W) + 1].sum())
    assert extreme_patterns == 3174 and extreme_patterns / 2 ** 14 == w.p_exact
    environment = {'python': platform.python_version(), 'python_executable': sys.executable, 'numpy': np.__version__, 'pandas': pd.__version__, 'scipy': scipy.__version__, 'role': 'Explicit-method revision reanalysis; not a claim about the unrecorded original runtime.', 'input_sha256': hashlib.sha256(args.input.read_bytes()).hexdigest(), 'archive_sha256': hashlib.sha256(args.archive.read_bytes()).hexdigest(), 'n_streams': 50, 'n_families': 5, 'streams_per_family': 10, 'tests': 12, 'maximum_absolute_errors_vs_archive': maximum_errors, 'all_p_values_match_archived_table8_to_displayed_precision': True, 'all_bootstrap_intervals_match_archived_table8_to_displayed_precision': True, 'wait_all_accuracy': {'n_total': 50, 'n_zero': 36, 'n_effective': 14, 'R_plus': 74, 'R_minus': 31, 'W': 31, 'two_sided_extreme_sign_patterns': extreme_patterns, 'all_sign_patterns': 2 ** 14, 'p_exact': float(w.p_exact), 'p_holm_12': float(w.p_holm_12)}}
    (out / 'reanalysis_environment_and_verification.json').write_text(json.dumps(environment, ensure_ascii=False, indent=2))
    (out / 'requirements_reanalysis.txt').write_text(f'numpy=={np.__version__}\npandas=={pd.__version__}\nscipy=={scipy.__version__}\n')
    print(json.dumps(environment, ensure_ascii=False, indent=2))
    print(result[['comparator', 'metric', 'zero_differences', 'n_effective', 'R_plus', 'R_minus', 'W', 'p_exact', 'p_holm_12']].to_string(index=False))
if __name__ == '__main__':
    main()
