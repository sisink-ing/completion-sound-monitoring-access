# Statistical analysis

```bash
python scripts/reproduce_statistics.py
```

The ARF analysis uses 50 independent stream pairs and a family of 12 comparisons.
The LR/DT analysis aggregates archived score cells into 50 independent stream
pairs and a separate family of nine comparisons. Differences are defined as
Complete Cohort minus the comparator.

## Signed-rank tests

Tests are two-sided exact Wilcoxon signed-rank tests. Differences with magnitude
at most 1e-12 are omitted from the ranks. The implementation uses
`zero_method='wilcox'`, `correction=False` and `method='exact'`.
Retained absolute differences are distinct in the distributed comparisons.

Each output reports the total pair count, zero differences, effective sample
size, positive and negative rank sums, W=min(R+,R−), raw probability and
Holm-adjusted probability. For the Wait-All accuracy comparison in Table 8,
the 50 pairs contain 36 zeros and 14 nonzero differences. R+=74, R−=31 and
W=31; the exact two-sided probability is 0.1937255859375 and the adjusted
probability is 0.77490234375. The individual differences and ranks are included.

## Bootstrap intervals

Intervals use 5,000 paired, family-stratified resamples with seed 30300. Each
resample draws ten streams with replacement within each of the five families.
Zero differences remain in this analysis. The random generator is reset for
each comparison. Percentile limits use quantiles 0.025 and 0.975 with linear
interpolation. Intervals describe mean differences; signed-rank probabilities
are computed independently.

## Files

Source scores are stored in `data/statistics/`. The command writes paired
differences, full rank records, summary statistics and environment metadata to
`results/statistics/table8/` and `results/statistics/tableS1/`. Reference outputs
are under `data/reference/statistics/`.
