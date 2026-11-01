"""Summarize confirmation reliability and label availability."""
import os
import numpy as np
import pandas as pd
INPUT_CANDIDATES = [os.path.join('result', 'confirmation_comparison', 'decision_raw.csv'), 'decision_raw.csv']
INPUT_FILE = None
for candidate in INPUT_CANDIDATES:
    if os.path.exists(candidate):
        INPUT_FILE = candidate
        break
if INPUT_FILE is None:
    raise FileNotFoundError('Cannot find decision_raw.csv.\nExpected:\nresult/confirmation_comparison/decision_raw.csv')
OUTPUT_DIR = os.path.join('result', 'confirmation_efficiency')
os.makedirs(OUTPUT_DIR, exist_ok=True)
ARRIVED = 'ARRIVED_ONLY_COMPLETE_COHORT'
OURS = 'FULL_COMPLETE_COHORT'
WAIT = 'WAIT_ALL_COMPLETE_COHORT'
STRATEGIES = [ARRIVED, OURS, WAIT]
df = pd.read_csv(INPUT_FILE)
print('=' * 145)
print('CONFIRMATION EFFICIENCY ANALYSIS')
print('=' * 145)
print('Input:', INPUT_FILE)
print('Decision rows:', len(df))
REQUIRED_COLUMNS = ['dataset', 'stream_seed', 'learner', 'scenario', 'delay_seed', 'strategy', 'correct', 'decision_delay', 'labels_at_decision', 'unknown_at_decision', 'returned_fraction', 'early_confirmation']
missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
if missing:
    raise RuntimeError(f'Missing required columns: {missing}')
df = df[df['strategy'].isin(STRATEGIES)].copy()
numeric_columns = ['correct', 'decision_delay', 'labels_at_decision', 'unknown_at_decision', 'returned_fraction', 'early_confirmation']
for column in numeric_columns:
    df[column] = pd.to_numeric(df[column], errors='coerce')
WINDOW_SIZE = 100
df['labels_saved_vs_waitall'] = WINDOW_SIZE - df['labels_at_decision']
df['label_saving_fraction'] = 1.0 - df['returned_fraction']
df['returned_percent'] = df['returned_fraction'] * 100.0
df['saved_percent'] = df['label_saving_fraction'] * 100.0
df['confirmed_by_50pct'] = (df['returned_fraction'] <= 0.5).astype(int)
df['confirmed_by_75pct'] = (df['returned_fraction'] <= 0.75).astype(int)
df['confirmed_by_90pct'] = (df['returned_fraction'] <= 0.9).astype(int)
df['confirmed_before_all'] = (df['returned_fraction'] < 1.0 - 1e-12).astype(int)

def summarize_decisions(data, group_columns):
    rows = []
    grouped = data.groupby(group_columns, dropna=False)
    for keys, group in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        record = {column: value for column, value in zip(group_columns, keys)}
        returned = group['returned_fraction'].dropna()
        labels = group['labels_at_decision'].dropna()
        delays = group['decision_delay'].dropna()
        if len(returned) > 0:
            q25_fraction = float(returned.quantile(0.25))
            median_fraction = float(returned.median())
            q75_fraction = float(returned.quantile(0.75))
        else:
            q25_fraction = np.nan
            median_fraction = np.nan
            q75_fraction = np.nan
        if len(labels) > 0:
            q25_labels = float(labels.quantile(0.25))
            median_labels = float(labels.median())
            q75_labels = float(labels.quantile(0.75))
        else:
            q25_labels = np.nan
            median_labels = np.nan
            q75_labels = np.nan
        record.update({'decisions': int(len(group)), 'oracle_agreement': float(group['correct'].mean()), 'mean_confirmation_delay': float(delays.mean()), 'median_confirmation_delay': float(delays.median()), 'mean_labels_at_decision': float(labels.mean()), 'q25_labels_at_decision': q25_labels, 'median_labels_at_decision': median_labels, 'q75_labels_at_decision': q75_labels, 'iqr_labels_at_decision': q75_labels - q25_labels, 'mean_returned_fraction': float(returned.mean()), 'q25_returned_fraction': q25_fraction, 'median_returned_fraction': median_fraction, 'q75_returned_fraction': q75_fraction, 'iqr_returned_fraction': q75_fraction - q25_fraction, 'mean_labels_saved': float(group['labels_saved_vs_waitall'].mean()), 'mean_label_saving_fraction': float(group['label_saving_fraction'].mean()), 'early_confirmation_rate': float(group['confirmed_before_all'].mean()), 'confirmed_by_50pct_rate': float(group['confirmed_by_50pct'].mean()), 'confirmed_by_75pct_rate': float(group['confirmed_by_75pct'].mean()), 'confirmed_by_90pct_rate': float(group['confirmed_by_90pct'].mean())})
        rows.append(record)
    return pd.DataFrame(rows)
overall_df = summarize_decisions(df, ['strategy'])
scenario_df = summarize_decisions(df, ['scenario', 'strategy'])
dataset_df = summarize_decisions(df, ['dataset', 'strategy'])
learner_df = summarize_decisions(df, ['learner', 'strategy'])
cell_group = ['dataset', 'stream_seed', 'learner', 'scenario', 'strategy']
cell_df = df.groupby(cell_group).agg(decisions=('correct', 'count'), oracle_agreement=('correct', 'mean'), mean_confirmation_delay=('decision_delay', 'mean'), mean_labels_at_decision=('labels_at_decision', 'mean'), mean_returned_fraction=('returned_fraction', 'mean'), mean_labels_saved=('labels_saved_vs_waitall', 'mean'), mean_label_saving_fraction=('label_saving_fraction', 'mean'), early_confirmation_rate=('confirmed_before_all', 'mean'), confirmed_by_50pct_rate=('confirmed_by_50pct', 'mean'), confirmed_by_75pct_rate=('confirmed_by_75pct', 'mean'), confirmed_by_90pct_rate=('confirmed_by_90pct', 'mean')).reset_index()
cell_summary_df = cell_df.groupby('strategy').agg(cells=('mean_returned_fraction', 'count'), oracle_agreement=('oracle_agreement', 'mean'), mean_confirmation_delay=('mean_confirmation_delay', 'mean'), mean_labels_at_decision=('mean_labels_at_decision', 'mean'), mean_returned_fraction=('mean_returned_fraction', 'mean'), mean_labels_saved=('mean_labels_saved', 'mean'), mean_label_saving_fraction=('mean_label_saving_fraction', 'mean'), early_confirmation_rate=('early_confirmation_rate', 'mean'), confirmed_by_50pct_rate=('confirmed_by_50pct_rate', 'mean'), confirmed_by_75pct_rate=('confirmed_by_75pct_rate', 'mean'), confirmed_by_90pct_rate=('confirmed_by_90pct_rate', 'mean')).reset_index()

def fraction_bin(value):
    if pd.isna(value):
        return 'NA'
    if value <= 0.5:
        return '<=50%'
    if value <= 0.75:
        return '50-75%'
    if value <= 0.9:
        return '75-90%'
    if value < 1.0 - 1e-12:
        return '90-<100%'
    return '100%'
df['returned_fraction_bin'] = df['returned_fraction'].apply(fraction_bin)
bin_order = ['<=50%', '50-75%', '75-90%', '90-<100%', '100%']
distribution_df = df.groupby(['strategy', 'returned_fraction_bin']).size().reset_index(name='decision_count')
totals = distribution_df.groupby('strategy')['decision_count'].transform('sum')
distribution_df['decision_fraction'] = distribution_df['decision_count'] / totals
distribution_df['returned_fraction_bin'] = pd.Categorical(distribution_df['returned_fraction_bin'], categories=bin_order, ordered=True)
distribution_df = distribution_df.sort_values(['strategy', 'returned_fraction_bin'])
main_rows = []
for strategy in [ARRIVED, OURS, WAIT]:
    row = overall_df[overall_df['strategy'] == strategy].iloc[0]
    main_rows.append({'strategy': strategy, 'decisions': int(row['decisions']), 'oracle_agreement': row['oracle_agreement'], 'mean_confirmation_delay': row['mean_confirmation_delay'], 'mean_labels_at_decision': row['mean_labels_at_decision'], 'median_labels_at_decision': row['median_labels_at_decision'], 'mean_returned_fraction': row['mean_returned_fraction'], 'median_returned_fraction': row['median_returned_fraction'], 'mean_labels_saved_vs_waitall': row['mean_labels_saved'], 'mean_label_saving_fraction': row['mean_label_saving_fraction'], 'early_confirmation_rate': row['early_confirmation_rate'], 'confirmed_by_50pct_rate': row['confirmed_by_50pct_rate'], 'confirmed_by_75pct_rate': row['confirmed_by_75pct_rate'], 'confirmed_by_90pct_rate': row['confirmed_by_90pct_rate']})
main_paper_df = pd.DataFrame(main_rows)
ours_row = overall_df[overall_df['strategy'] == OURS].iloc[0]
wait_row = overall_df[overall_df['strategy'] == WAIT].iloc[0]
comparison_df = pd.DataFrame([{'comparison': 'Complete Cohort vs Wait-All', 'ours_oracle_agreement': ours_row['oracle_agreement'], 'waitall_oracle_agreement': wait_row['oracle_agreement'], 'ours_mean_returned_fraction': ours_row['mean_returned_fraction'], 'waitall_mean_returned_fraction': wait_row['mean_returned_fraction'], 'returned_fraction_reduction': wait_row['mean_returned_fraction'] - ours_row['mean_returned_fraction'], 'ours_mean_labels_at_decision': ours_row['mean_labels_at_decision'], 'waitall_mean_labels_at_decision': wait_row['mean_labels_at_decision'], 'mean_labels_saved': wait_row['mean_labels_at_decision'] - ours_row['mean_labels_at_decision'], 'ours_early_confirmation_rate': ours_row['early_confirmation_rate'], 'waitall_early_confirmation_rate': wait_row['early_confirmation_rate'], 'ours_mean_confirmation_delay': ours_row['mean_confirmation_delay'], 'waitall_mean_confirmation_delay': wait_row['mean_confirmation_delay'], 'delay_reduction_percent': (wait_row['mean_confirmation_delay'] - ours_row['mean_confirmation_delay']) / wait_row['mean_confirmation_delay'] * 100.0}])
outputs = {'overall': 'overall_confirmation_efficiency.csv', 'scenario': 'scenario_confirmation_efficiency.csv', 'dataset': 'dataset_confirmation_efficiency.csv', 'learner': 'learner_confirmation_efficiency.csv', 'cells': 'cell_confirmation_efficiency.csv', 'cell_summary': 'cell_summary.csv', 'distribution': 'returned_fraction_distribution.csv', 'main_paper': 'main_paper_table.csv', 'comparison': 'ours_vs_waitall.csv'}
tables = {'overall': overall_df, 'scenario': scenario_df, 'dataset': dataset_df, 'learner': learner_df, 'cells': cell_df, 'cell_summary': cell_summary_df, 'distribution': distribution_df, 'main_paper': main_paper_df, 'comparison': comparison_df}
for key, filename in outputs.items():
    path = os.path.join(OUTPUT_DIR, filename)
    tables[key].to_csv(path, index=False, encoding='utf-8-sig')
print()
print('=' * 145)
print('MAIN PAPER TABLE')
print('=' * 145)
print(main_paper_df.to_string(index=False))
print()
print('=' * 145)
print('OURS VS WAIT-ALL')
print('=' * 145)
print(comparison_df.to_string(index=False))
print()
print('=' * 145)
print('RETURNED-FRACTION DISTRIBUTION')
print('=' * 145)
print(distribution_df.to_string(index=False))
print()
print('=' * 145)
print('FINAL SCIENTIFIC CHECK')
print('=' * 145)
ours_agreement = float(ours_row['oracle_agreement'])
wait_agreement = float(wait_row['oracle_agreement'])
ours_fraction = float(ours_row['mean_returned_fraction'])
wait_fraction = float(wait_row['mean_returned_fraction'])
ours_early = float(ours_row['early_confirmation_rate'])
print(f'Ours oracle agreement: {ours_agreement:.6f}')
print(f'Wait-All oracle agreement: {wait_agreement:.6f}')
print(f'Ours mean returned-label fraction: {ours_fraction:.6f} ({ours_fraction * 100:.2f}%)')
print(f'Wait-All mean returned-label fraction: {wait_fraction:.6f} ({wait_fraction * 100:.2f}%)')
print(f'Mean label fraction saved by Ours: {wait_fraction - ours_fraction:.6f} ({(wait_fraction - ours_fraction) * 100:.2f} percentage points)')
print(f'Ours early-confirmation rate: {ours_early:.6f} ({ours_early * 100:.2f}%)')
if abs(ours_agreement - 1.0) < 1e-12:
    print('Complete Cohort safety: PASS ')
else:
    print('Complete Cohort safety: CHECK REQUIRED ')
if abs(wait_agreement - 1.0) < 1e-12:
    print('Wait-All safety: PASS ')
else:
    print('Wait-All safety: CHECK REQUIRED ')
print()
print('Output directory:')
print(OUTPUT_DIR)
print('=' * 145)
