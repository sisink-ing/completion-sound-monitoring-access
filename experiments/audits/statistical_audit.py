"""Verify aggregation and paired-window confirmation statistics."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
KEYS = ['dataset', 'stream_seed', 'learner', 'scenario', 'delay_seed', 'strategy']
CELL_KEYS = ['dataset', 'stream_seed', 'learner', 'scenario', 'strategy']
S24 = Path('result/confirmation_comparison')
S25 = [Path('result/25'), Path('result/confirmation_efficiency'), Path('confirmation_efficiency')]
OUT = Path('result/statistical_audit')
ORDER = ['ARRIVED_ONLY_COMPLETE_COHORT', 'FULL_COMPLETE_COHORT', 'WAIT_ALL_COMPLETE_COHORT']
SHORT = {'ARRIVED_ONLY_COMPLETE_COHORT': 'Arrived-Only', 'FULL_COMPLETE_COHORT': 'Complete Cohort', 'WAIT_ALL_COMPLETE_COHORT': 'Wait-All Complete'}
TABLE4 = {'ARRIVED_ONLY_COMPLETE_COHORT': (0.6805, 1634, 5789, 5.79), 'FULL_COMPLETE_COHORT': (1.0, 0, 0, 102.69), 'WAIT_ALL_COMPLETE_COHORT': (1.0, 0, 0, 160.98)}

def require(condition, message):
    if not bool(condition):
        raise AssertionError(message)

def near(actual, expected, name, tol=1e-10):
    a = np.asarray(actual, dtype=float)
    b = np.asarray(expected, dtype=float)
    valid = a.shape == b.shape and np.all(np.isclose(a, b, atol=tol, rtol=0, equal_nan=True))
    require(valid, f'Statistical mismatch: {name}')

def read(path):
    if not path.is_file():
        raise FileNotFoundError(f'Required CSV is missing: {path}')
    return pd.read_csv(path, encoding='utf-8-sig')

def check_fields(df, fields, label):
    missing = sorted(set(fields).difference(df.columns))
    require(not missing, f'{label}: missing fields {missing}')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src24', default=str(S24))
    parser.add_argument('--src25', default=None)
    parser.add_argument('--output', default=str(OUT))
    args = parser.parse_args()
    src24 = Path(args.src24)
    if args.src25:
        src25 = Path(args.src25)
    else:
        src25 = next((p for p in S25 if (p / 'main_paper_table.csv').exists()), None)
        if src25 is None:
            raise FileNotFoundError('confirmation analysis results missing. Use --src25 to specify their folder.')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    run = read(src24 / 'raw.csv')
    dec = read(src24 / 'decision_raw.csv')
    upd = read(src24 / 'update_raw.csv')
    main25 = read(src25 / 'main_paper_table.csv')
    cell25 = read(src25 / 'cell_summary.csv')
    check_fields(run, KEYS + ['decision_count', 'oracle_agreement', 'false_positive', 'false_negative', 'decision_delay_mean', 'mean_returned_fraction', 'early_confirmation_rate', 'retrain_count'], 'confirmation comparison runs')
    check_fields(dec, KEYS + ['block_id', 'decision', 'oracle', 'correct', 'decision_time', 'block_close_time', 'decision_delay', 'labels_at_decision', 'returned_fraction', 'early_confirmation'], 'confirmation comparison decisions')
    check_fields(upd, KEYS, 'confirmation comparison updates')
    check_fields(main25, ['strategy', 'decisions', 'oracle_agreement', 'mean_confirmation_delay', 'mean_returned_fraction', 'early_confirmation_rate'], 'confirmation analysis main')
    check_fields(cell25, ['strategy', 'cells', 'oracle_agreement', 'mean_confirmation_delay', 'mean_returned_fraction'], 'confirmation analysis cell')
    require(set(run.strategy) == set(ORDER), 'Unexpected strategies in confirmation comparison')
    require(set(dec.strategy) == set(ORDER), 'Unexpected strategies in decision records')
    require(not run.duplicated(KEYS).any(), 'Duplicate confirmation comparison task keys')
    require(not dec.duplicated(KEYS + ['block_id']).any(), 'Duplicate decision block IDs')
    require(not run[KEYS].isna().any().any(), 'Missing task keys')
    require(not dec[KEYS].isna().any().any(), 'Missing decision keys')
    computed_correct = (dec['decision'].to_numpy() == dec['oracle'].to_numpy()).astype(int)
    near(dec['correct'].to_numpy(), computed_correct, 'Oracle correctness', tol=0)
    near(dec['decision_delay'].to_numpy(), (dec['decision_time'] - dec['block_close_time']).to_numpy(), 'Decision delay', tol=0)
    require(dec['decision'].isin(['DEGRADATION', 'TOLERABLE']).all(), 'Unexpected decision value')
    require(dec['oracle'].isin(['DEGRADATION', 'TOLERABLE']).all(), 'Unexpected Oracle value')
    dec['FP_audit'] = ((dec.decision == 'DEGRADATION') & (dec.oracle == 'TOLERABLE')).astype(int)
    dec['FN_audit'] = ((dec.decision == 'TOLERABLE') & (dec.oracle == 'DEGRADATION')).astype(int)
    dec['correct_audit'] = computed_correct
    require((dec.FP_audit + dec.FN_audit + dec.correct_audit == 1).all(), 'FP/FN/correct partition failed')
    by_run = dec.groupby(KEYS, dropna=False, sort=False).agg(audited_decisions=('correct_audit', 'size'), audited_correct=('correct_audit', 'sum'), audited_FP=('FP_audit', 'sum'), audited_FN=('FN_audit', 'sum'), audited_delay=('decision_delay', 'mean'), audited_fraction=('returned_fraction', 'mean'), audited_early=('early_confirmation', 'mean')).reset_index()
    by_run['audited_agreement'] = by_run.audited_correct / by_run.audited_decisions
    audit = run.merge(by_run, on=KEYS, how='outer', validate='one_to_one', indicator=True)
    require(audit._merge.eq('both').all(), 'Missing run or decision record')
    audit = audit.drop(columns='_merge')
    update_counts = upd.groupby(KEYS, dropna=False).size().rename('updates_from_log').reset_index()
    audit = audit.merge(update_counts, on=KEYS, how='left', validate='one_to_one')
    audit['updates_from_log'] = audit['updates_from_log'].fillna(0).astype(int)
    tests = [('decision_count', 'audited_decisions', 0), ('false_positive', 'audited_FP', 0), ('false_negative', 'audited_FN', 0), ('oracle_agreement', 'audited_agreement', 1e-10), ('decision_delay_mean', 'audited_delay', 1e-09), ('mean_returned_fraction', 'audited_fraction', 1e-10), ('early_confirmation_rate', 'audited_early', 1e-10), ('retrain_count', 'updates_from_log', 0)]
    for original, rebuilt, tolerance in tests:
        near(audit[original].to_numpy(), audit[rebuilt].to_numpy(), f'Run audit: {original}', tolerance)
    by_cell = dec.groupby(CELL_KEYS, dropna=False, sort=False).agg(decisions=('correct_audit', 'size'), correct=('correct_audit', 'sum'), delay=('decision_delay', 'mean'), returned=('returned_fraction', 'mean'), early=('early_confirmation', 'mean')).reset_index()
    by_cell['agreement'] = by_cell.correct / by_cell.decisions
    rows = []
    for strategy in ORDER:
        dr = dec[dec.strategy == strategy]
        rr = audit[audit.strategy == strategy]
        cc = by_cell[by_cell.strategy == strategy]
        require(len(rr) > 0 and len(dr) > 0, f'No data for {strategy}')
        rows.append({'strategy': strategy, 'display': SHORT[strategy], 'runs': len(rr), 'cells': len(cc), 'decisions': len(dr), 'correct_windows': int(dr.correct_audit.sum()), 'FP': int(dr.FP_audit.sum()), 'FN': int(dr.FN_audit.sum()), 'agreement_run_macro': float(rr.audited_agreement.mean()), 'agreement_pooled': float(dr.correct_audit.mean()), 'agreement_cell_macro': float(cc.agreement.mean()), 'delay_run_macro': float(rr.audited_delay.mean()), 'delay_pooled': float(dr.decision_delay.mean()), 'delay_cell_macro': float(cc.delay.mean()), 'fraction_run_macro': float(rr.audited_fraction.mean()), 'fraction_pooled': float(dr.returned_fraction.mean()), 'fraction_cell_macro': float(cc.returned.mean()), 'early_run_macro': float(rr.audited_early.mean()), 'early_pooled': float(dr.early_confirmation.mean()), 'total_updates': int(rr.retrain_count.sum())})
    summary = pd.DataFrame(rows)
    checks = []
    for _, v in summary.iterrows():
        comparisons = [('MAIN_PAPER_TABLE', main25[main25.strategy == v.strategy], {'decisions': 'decisions', 'oracle_agreement': 'agreement_pooled', 'mean_confirmation_delay': 'delay_pooled', 'mean_returned_fraction': 'fraction_pooled', 'early_confirmation_rate': 'early_pooled'}), ('25_500cell_summary', cell25[cell25.strategy == v.strategy], {'cells': 'cells', 'oracle_agreement': 'agreement_cell_macro', 'mean_confirmation_delay': 'delay_cell_macro', 'mean_returned_fraction': 'fraction_cell_macro'})]
        for label, target, fields in comparisons:
            require(len(target) == 1, f'{label}: expected one row')
            target = target.iloc[0]
            for source_field, computed_field in fields.items():
                original = float(target[source_field])
                rebuilt = float(v[computed_field])
                near([original], [rebuilt], f'{label}/{v.strategy}/{source_field}', 1e-09)
                checks.append({'file': label, 'strategy': v.strategy, 'field': source_field, 'original': original, 'reconstructed': rebuilt, 'difference': rebuilt - original, 'status': 'PASS'})
    manuscript = []
    for _, v in summary.iterrows():
        old = TABLE4[v.strategy]
        table_checks = [('oracle_agreement_run_macro', old[0], round(v.agreement_run_macro, 4)), ('FP_total', old[1], int(v.FP)), ('FN_total', old[2], int(v.FN)), ('delay_run_macro', old[3], round(v.delay_run_macro, 2))]
        for field, original, rebuilt in table_checks:
            require(original == rebuilt, f'V1.3 Table4 mismatch: {field}')
            manuscript.append({'strategy': v.strategy, 'field': field, 'V1_3_value': original, 'recomputed': rebuilt, 'status': 'MATCH'})
    run_counts = summary.set_index('strategy').runs
    require(run_counts.nunique() == 1, 'Unequal task counts')
    tasks = audit[KEYS[:-1]].drop_duplicates()
    require(len(tasks) == int(run_counts.iloc[0]), 'Unmatched task sets')
    audit.to_csv(output / 'run_audit.csv', index=False)
    by_cell.to_csv(output / 'cell_audit.csv', index=False)
    summary.to_csv(output / 'main_si_table.csv', index=False)
    pd.DataFrame(checks).to_csv(output / 'confirmation_reference_check.csv', index=False)
    pd.DataFrame(manuscript).to_csv(output / 'reliability_reference_check.csv', index=False)
    protocol = {'status': 'PASS', 'source24': str(src24), 'source25': str(src25), 'n_strategies': len(summary), 'task_count_per_strategy': int(run_counts.iloc[0]), 'decision_counts': {r.strategy: int(r.decisions) for _, r in summary.iterrows()}, 'definitions': {'run_macro': 'Equal-weight mean of individual-run resolved-window agreement', 'pooled': 'Total correct resolved decisions divided by all resolved decisions', 'cell_macro': 'Equal-weight mean of 500 cells defined by dataset, stream seed, learner and scenario; delay seeds pooled within cells', 'FP': 'DEGRADATION versus TOLERABLE Oracle', 'FN': 'TOLERABLE versus DEGRADATION Oracle', 'delay_run_macro': 'Mean of run-level average delays', 'delay_pooled': 'Mean delay over all resolved windows'}, 'note': 'Table 4 uses run-macro agreement and delay, but pooled FP/FN counts. confirmation analysis main table uses decision-pooled aggregation. Different strategies can have different resolved-window sets.'}
    (output / 'protocol.json').write_text(json.dumps(protocol, indent=2, ensure_ascii=False), encoding='utf-8')
    print('\nstatistical audit - ALL CHECKS PASS\n')
    columns = ['display', 'runs', 'cells', 'decisions', 'FP', 'FN', 'agreement_run_macro', 'agreement_pooled', 'agreement_cell_macro', 'delay_run_macro', 'delay_pooled']
    print(summary[columns].to_string(index=False, float_format=lambda x: f'{x:.6f}'))
    print('\nconfirmation comparison and confirmation analysis independently verified.')
    print('V1.3 Table 4 values verified.')
    print('\nResults saved to:', output)
if __name__ == '__main__':
    main()
