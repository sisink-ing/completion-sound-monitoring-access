"""Compute paired-stream LR/DT statistics and exact signed-rank tests."""
from pathlib import Path
import argparse
import hashlib
import json
import platform
import numpy as np
import pandas as pd
import scipy
from scipy.stats import rankdata, wilcoxon
METRICS = ['accuracy', 'macro_f1', 'mcc']
COMPARATORS = [('STATIC', 'Static'), ('FULL_SOFT_Q30', 'Soft-q30'), ('WAIT_ALL_COMPLETE_COHORT', 'Wait-All Complete')]
TOL = 1e-12
REPS = 5000
SEED = 30300
ARCHIVED = [('Static', 'accuracy', 0.1494, 0.085, 0.1321, 0.1664, 47, 0, 3, 2.7e-13), ('Static', 'macro_f1', 0.1606, 0.1341, 0.1426, 0.1784, 48, 0, 2, 2.24e-13), ('Static', 'mcc', 0.2962, 0.1956, 0.2656, 0.3268, 47, 0, 3, 3.11e-13), ('Soft-q30', 'accuracy', -0.0106, -0.0034, -0.0186, -0.0008, 21, 0, 29, 0.1578), ('Soft-q30', 'macro_f1', -0.0091, 0.0002, -0.0184, 0.0013, 25, 0, 25, 0.208), ('Soft-q30', 'mcc', -0.0307, -0.0098, -0.0455, -0.0148, 18, 0, 32, 0.0357), ('Wait-All Complete', 'accuracy', 0.0041, 0.0, 0.0018, 0.0067, 20, 24, 6, 0.0102), ('Wait-All Complete', 'macro_f1', 0.0041, 0.0, 0.0019, 0.0067, 19, 24, 7, 0.0102), ('Wait-All Complete', 'mcc', 0.0076, 0.0, 0.0037, 0.0121, 19, 24, 7, 0.0102)]

def holm_adjust(values):
    p = np.asarray(values, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    adjusted[order] = np.minimum(1.0, np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order]))
    return adjusted

def p_tex(value):
    if value < 0.0001:
        mantissa, exponent = f'{value:.2e}'.split('e')
        return '$' + mantissa + '\\times10^{' + str(int(exponent)) + '}$'
    return f'{value:.4f}'

def number_tex(value, signed=False):
    if abs(value) < 5e-05:
        return '0.0000'
    if value < 0:
        return '$-$' + f'{abs(value):.4f}'
    return ('+' if signed else '') + f'{value:.4f}'

def main():
    parser = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    parser.add_argument('--input', type=Path, default=here / 'source_stream_cells.csv')
    parser.add_argument('--output', type=Path, default=here / 'result' / 'paired_lrdt_statistics')
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    cells = pd.read_csv(args.input, float_precision='round_trip')
    keys = ['dataset', 'stream_seed', 'learner', 'scenario', 'strategy']
    assert len(cells) == 2000 and (not cells.duplicated(keys).any())
    assert set(cells.learner) == {'LR', 'DT'}
    assert (cells.delay_seed_runs == 5).all()
    assert cells.groupby(['dataset', 'stream_seed', 'strategy']).size().eq(10).all()
    assert cells.groupby(['dataset', 'stream_seed']).stream_hash.nunique().eq(1).all()
    scores = cells.groupby(['dataset', 'stream_seed', 'strategy'], sort=True)[METRICS].mean().reset_index()
    assert len(scores) == 200
    families = sorted(scores.dataset.unique())
    assert len(families) == 5
    assert scores.groupby('dataset').stream_seed.nunique().eq(10).all()
    wide = None
    rows, rank_rows = ([], [])
    for comparator, name in COMPARATORS:
        for metric in METRICS:
            pivot = scores.pivot(index=['dataset', 'stream_seed'], columns='strategy', values=metric).sort_index()
            assert pivot.shape == (50, 4) and (not pivot.isna().any().any())
            if wide is None:
                wide = pd.DataFrame(index=pivot.index)
            ours = pivot['FULL_COMPLETE_COHORT'].to_numpy()
            other = pivot[comparator].to_numpy()
            difference = ours - other
            zero = np.abs(difference) <= TOL
            nonzero = difference[~zero]
            magnitudes = np.sort(np.abs(nonzero))
            assert len(nonzero) > 0
            assert np.all(np.diff(magnitudes) > TOL), 'Nonzero magnitudes have rank ties.'
            ranks = rankdata(np.abs(nonzero), method='average')
            r_plus = float(ranks[nonzero > 0].sum())
            r_minus = float(ranks[nonzero < 0].sum())
            test = wilcoxon(nonzero, zero_method='wilcox', alternative='two-sided', correction=False, method='exact')
            assert test.statistic == min(r_plus, r_minus)
            full_ranks = np.full(50, np.nan)
            full_ranks[~zero] = ranks
            wide[f'{comparator}__{metric}'] = difference
            for j, (family, stream_seed) in enumerate(pivot.index):
                rank_rows.append(dict(dataset=family, stream_seed=int(stream_seed), comparator=name, metric=metric, complete_score=float(ours[j]), comparator_score=float(other[j]), difference=float(difference[j]), zero_difference=bool(zero[j]), absolute_rank=None if zero[j] else float(full_ranks[j]), positive_rank_contribution=float(full_ranks[j]) if difference[j] > TOL else 0.0, negative_rank_contribution=float(full_ranks[j]) if difference[j] < -TOL else 0.0))
            family_values = [difference[pivot.index.get_level_values('dataset') == f] for f in families]
            rng = np.random.default_rng(SEED)
            bootstrap = np.empty(REPS)
            for b in range(REPS):
                draw = np.concatenate([v[rng.integers(0, len(v), len(v))] for v in family_values])
                bootstrap[b] = draw.mean()
            low, high = np.quantile(bootstrap, [0.025, 0.975], method='linear')
            rows.append(dict(comparator=name, metric=metric, n_total=50, zero_differences=int(zero.sum()), n_effective=len(nonzero), absolute_rank_tie_groups=0, R_plus=r_plus, R_minus=r_minus, W=float(test.statistic), mean_difference=float(difference.mean()), median_difference=float(np.median(difference)), ci95_low=float(low), ci95_high=float(high), wins=int((difference > TOL).sum()), ties=int(zero.sum()), losses=int((difference < -TOL).sum()), p_exact=float(test.pvalue), method='exact', alternative='two-sided', zero_method='wilcox', zero_tolerance=TOL, continuity_correction=False, bootstrap_repetitions=REPS, bootstrap_seed=SEED, bootstrap_method='family-stratified paired percentile', quantile_method='linear', rng='PCG64'))
    result = pd.DataFrame(rows)
    result['p_holm_9'] = holm_adjust(result.p_exact.to_numpy())
    previous = pd.DataFrame(ARCHIVED, columns=['comparator', 'metric', 'mean_difference', 'median_difference', 'ci95_low', 'ci95_high', 'wins', 'ties', 'losses', 'p_holm_9'])
    check = previous.merge(result, on=['comparator', 'metric'], suffixes=('_archived_display', '_revision'), validate='one_to_one')
    for field in ['mean_difference', 'median_difference']:
        assert np.all(np.abs(check[field + '_archived_display'] - check[field + '_revision']) < 5.00001e-05)
    for field in ['wins', 'ties', 'losses']:
        assert (check[field + '_archived_display'] == check[field + '_revision']).all()
    assert ((check.p_holm_9_archived_display < 0.05) == (check.p_holm_9_revision < 0.05)).all()
    previous.to_csv(out / 'tables1_archived_display_values.csv', index=False, float_format='%.17g')
    check.to_csv(out / 'tables1_archived_vs_revision.csv', index=False, float_format='%.17g')
    scores.to_csv(out / 'tables1_50_stream_scores.csv', index=False, float_format='%.17g')
    wide.reset_index().to_csv(out / 'tables1_50_stream_paired_differences.csv', index=False, float_format='%.17g')
    result.to_csv(out / 'tables1_statistics.csv', index=False, float_format='%.17g')
    result.to_csv(out / 'tables1_effective_n_and_rank_statistics.csv', index=False, float_format='%.17g')
    pd.DataFrame(rank_rows).to_csv(out / 'tables1_all_450_pairwise_rank_records.csv', index=False, float_format='%.17g')
    metric_names = {'accuracy': 'Accuracy', 'macro_f1': 'Macro-F1', 'mcc': 'MCC'}
    table_rows, rank_table_rows = ([], [])
    for j, row in result.iterrows():
        comparator_cell = ''
        if j % 3 == 0:
            name = row.comparator
            if name == 'Wait-All Complete':
                name = '\\shortstack[l]{Wait-All\\\\Complete}'
            comparator_cell = '\\multirow{3}{*}{' + name + '}'
        values = [comparator_cell, metric_names[row.metric], number_tex(row.mean_difference, True), number_tex(row.median_difference, True), '[' + number_tex(row.ci95_low) + ',' + number_tex(row.ci95_high) + ']', f'{row.wins}/{row.ties}/{row.losses}', p_tex(row.p_holm_9)]
        suffix = '\\\\\\midrule' if j in [2, 5] else '\\\\'
        if j == 8:
            suffix = '\\\\\\bottomrule'
        table_rows.append(' & '.join(values) + suffix)
        ranks = [row.comparator, metric_names[row.metric], str(int(row.zero_differences)), str(int(row.n_effective)), str(int(row.R_plus)), str(int(row.R_minus)), str(int(row.W)), p_tex(row.p_exact)]
        rank_table_rows.append(' & '.join(ranks) + ('\\\\\\bottomrule' if j == 8 else '\\\\'))
    (out / 'TableS1_rows.tex').write_text('\n'.join(table_rows) + '\n')
    (out / 'TableS11_rows.tex').write_text('\n'.join(rank_table_rows) + '\n')
    environment = dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__, role='Explicit-method revision reanalysis of archived LR/DT scores; not an original training environment.', input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(), independent_streams=50, families=5, streams_per_family=10, learner_count=2, delay_regimes=5, delay_seeds_per_cell=5, input_cells=2000, tests=9, bootstrap_repetitions=REPS, bootstrap_seed=SEED, archived_means_medians_and_WTL_preserved_at_displayed_precision=True, all_nine_Holm_005_decisions_unchanged=True, changed_p_values_are_revision_estimates_not_changes_to_predictive_scores=True)
    (out / 'reanalysis_environment_and_verification.json').write_text(json.dumps(environment, indent=2))
    (out / 'requirements_reanalysis.txt').write_text(f'numpy=={np.__version__}\npandas=={pd.__version__}\nscipy=={scipy.__version__}\n')
    print(result[['comparator', 'metric', 'n_effective', 'W', 'ci95_low', 'ci95_high', 'p_exact', 'p_holm_9']].to_string(index=False))
    print(json.dumps(environment, indent=2))
if __name__ == '__main__':
    main()
