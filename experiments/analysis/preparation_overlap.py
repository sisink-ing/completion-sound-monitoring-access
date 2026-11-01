"""Compute matched-window confirmation leads and preparation overlap."""
from completion_monitoring.paths import workspace_root, external_data_root, source_file
import json
import importlib
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = workspace_root()
INPUT_DECISIONS = ROOT / 'result' / 'confirmation_comparison' / 'decision_raw.csv'
STREAM_DIR = ROOT / 'result' / 'unseen_stream_pipeline' / 'streams'
OUT = ROOT / 'result' / 'preparation_overlap'
OUT.mkdir(parents=True, exist_ok=True)
WINDOW_SIZE = 100
WINDOW_COLLECTION_STEPS = WINDOW_SIZE - 1
PREPARATION_HORIZONS = [10, 25, 50, 100]
STRATEGY = 'FULL_COMPLETE_COHORT'
ANCHORS = {'ALL': {'n': 19979, 'complete_mean': 103.17, 'waitall_mean': 152.48, 'lead_mean': 49.31, 'lead_median': 30.0, 'lead_q1': 14.0, 'lead_q3': 61.0}, 'DEGRADATION': {'n': 6188, 'complete_mean': 77.66, 'waitall_mean': 158.81, 'lead_mean': 81.15, 'lead_median': 56.0, 'lead_q1': 25.0, 'lead_q3': 118.0}, 'TOLERABLE': {'n': 13791, 'complete_mean': 114.62, 'waitall_mean': 149.64, 'lead_mean': 35.02, 'lead_median': 24.0, 'lead_q1': 12.0, 'lead_q3': 44.0}}
MEAN_TOL = 0.03
QUANTILE_TOL = 0.01
base = importlib.import_module('chronological_protocol')
if not hasattr(base, 'generate_delay'):
    raise RuntimeError('chronological_protocol does not contain generate_delay().')

def load_stream(dataset, stream_seed):
    path = STREAM_DIR / f'{dataset}_streamseed_{int(stream_seed)}.npz'
    if not path.exists():
        raise FileNotFoundError(f'Frozen stream not found:\n{path}')
    with np.load(path, allow_pickle=False) as data:
        if 'y' in data.files:
            y = data['y']
        elif 'labels' in data.files:
            y = data['labels']
        elif 'target' in data.files:
            y = data['target']
        else:
            raise RuntimeError(f'No label array found in {path}; keys={data.files}')
    return np.asarray(y, dtype=int).reshape(-1)

def normalize_arrival_map(arrival_map):
    if isinstance(arrival_map, dict):
        return {int(k): int(v) for k, v in arrival_map.items()}
    raise TypeError('generate_delay() did not return arrival_map as a dictionary.')

def select_group(df, name):
    if name == 'ALL':
        return df
    return df[df['decision'] == name]

def q1(x):
    return float(np.quantile(np.asarray(x, dtype=float), 0.25))

def q3(x):
    return float(np.quantile(np.asarray(x, dtype=float), 0.75))
print('=' * 120)
print('PREPARATION-OVERLAP UTILITY — v2')
print('EXACT SAME-WINDOW COUNTERFACTUAL RECONSTRUCTION')
print('=' * 120)
if not INPUT_DECISIONS.exists():
    raise FileNotFoundError(INPUT_DECISIONS)
raw = pd.read_csv(INPUT_DECISIONS)
cc = raw[raw['strategy'] == STRATEGY].copy()
cc = cc.reset_index(drop=True)
print('FULL_COMPLETE_COHORT decisions:', len(cc))
if len(cc) != 19979:
    raise RuntimeError(f'Expected 19,979 Complete decisions; found {len(cc)}.')
if not (cc['correct'].astype(int) == 1).all():
    raise RuntimeError('Not every Complete decision matches its oracle.')
result_rows = []
group_cols = ['dataset', 'stream_seed', 'stream_hash', 'learner', 'scenario', 'lambda0', 'lambda1', 'delay_seed']
groups = cc.groupby(group_cols, sort=False)
print('Unique run conditions:', groups.ngroups)
print()
print('Reconstructing exact same-window counterfactual Wait-All times...')
for group_index, (key, group) in enumerate(groups, start=1):
    dataset, stream_seed, stream_hash, learner, scenario, lambda0, lambda1, delay_seed = key
    y = load_stream(dataset, stream_seed)
    generated = base.generate_delay(y, int(delay_seed), int(lambda0), int(lambda1))
    if not isinstance(generated, tuple) or len(generated) < 1:
        raise RuntimeError('Unexpected generate_delay() return value.')
    arrival_map = normalize_arrival_map(generated[0])
    for _, row in group.iterrows():
        close_time = int(round(float(row['block_close_time'])))
        block_start = close_time - WINDOW_SIZE + 1
        block_ids = range(block_start, close_time + 1)
        missing = [sample_id for sample_id in block_ids if sample_id not in arrival_map]
        if missing:
            raise RuntimeError(f"\nMissing event IDs in arrival map.\ndataset={dataset}\nstream_seed={stream_seed}\ndelay_seed={delay_seed}\nblock_id={row['block_id']}\nclose={close_time}\nmissing(first 10)={missing[:10]}")
        last_arrival = max((arrival_map[sample_id] for sample_id in block_ids))
        waitall_delay = last_arrival - close_time
        complete_delay = float(row['decision_delay'])
        lead = waitall_delay - complete_delay
        result_rows.append({'dataset': dataset, 'stream_seed': int(stream_seed), 'stream_hash': stream_hash, 'learner': learner, 'scenario': scenario, 'lambda0': int(lambda0), 'lambda1': int(lambda1), 'delay_seed': int(delay_seed), 'block_id': int(row['block_id']), 'model_version': int(row['model_version']), 'decision': str(row['decision']), 'oracle': str(row['oracle']), 'block_start_time': int(block_start), 'block_close_time': int(close_time), 'decision_time': int(row['decision_time']), 'complete_delay': float(complete_delay), 'same_window_waitall_delay': float(waitall_delay), 'confirmation_lead': float(lead), 'labels_at_decision': int(row['labels_at_decision']), 'returned_fraction': float(row['returned_fraction']), 'early_confirmation': int(row['early_confirmation']), 'last_label_arrival_time': int(last_arrival)})
    if group_index % 100 == 0:
        print(f'  reconstructed {group_index}/{groups.ngroups} run conditions')
same = pd.DataFrame(result_rows)
if len(same) != 19979:
    raise RuntimeError('Reconstruction row count mismatch.')
min_lead = float(same['confirmation_lead'].min())
negative_count = int((same['confirmation_lead'] < -1e-12).sum())
if negative_count != 0:
    bad = same[same['confirmation_lead'] < -1e-12].head(20)
    print(bad.to_string(index=False))
    raise RuntimeError(f'Found {negative_count} negative same-window confirmation leads.')
validation_rows = []
for name in ['ALL', 'DEGRADATION', 'TOLERABLE']:
    sub = select_group(same, name)
    leads = sub['confirmation_lead'].to_numpy(dtype=float)
    observed = {'group': name, 'n': int(len(sub)), 'complete_mean': float(sub['complete_delay'].mean()), 'waitall_mean': float(sub['same_window_waitall_delay'].mean()), 'lead_mean': float(np.mean(leads)), 'lead_median': float(np.median(leads)), 'lead_q1': q1(leads), 'lead_q3': q3(leads), 'min_lead': float(np.min(leads))}
    target = ANCHORS[name]
    checks = {'n_pass': observed['n'] == target['n'], 'complete_mean_pass': abs(observed['complete_mean'] - target['complete_mean']) <= MEAN_TOL, 'waitall_mean_pass': abs(observed['waitall_mean'] - target['waitall_mean']) <= MEAN_TOL, 'lead_mean_pass': abs(observed['lead_mean'] - target['lead_mean']) <= MEAN_TOL, 'lead_median_pass': abs(observed['lead_median'] - target['lead_median']) <= QUANTILE_TOL, 'lead_q1_pass': abs(observed['lead_q1'] - target['lead_q1']) <= QUANTILE_TOL, 'lead_q3_pass': abs(observed['lead_q3'] - target['lead_q3']) <= QUANTILE_TOL}
    validation_rows.append({**observed, 'expected_n': target['n'], 'expected_complete_mean': target['complete_mean'], 'expected_waitall_mean': target['waitall_mean'], 'expected_lead_mean': target['lead_mean'], 'expected_lead_median': target['lead_median'], 'expected_lead_q1': target['lead_q1'], 'expected_lead_q3': target['lead_q3'], **checks, 'all_pass': all(checks.values())})
validation = pd.DataFrame(validation_rows)
print()
print('=' * 120)
print('EXACT SAME-WINDOW ANCHOR VALIDATION')
print('=' * 120)
print(validation.to_string(index=False))
if not validation['all_pass'].astype(bool).all():
    same.to_csv(OUT / 'debug_same_window_reconstruction.csv', index=False)
    validation.to_csv(OUT / 'debug_anchor_validation.csv', index=False)
    raise RuntimeError('\nExact reconstruction did not reproduce the frozen Table-5 anchors.\nDo NOT use preparation analysis results.')
for h in PREPARATION_HORIZONS:
    gain = np.minimum(same['confirmation_lead'].to_numpy(dtype=float), float(h))
    same[f'gain_h{h}'] = gain
    same[f'fraction_hidden_h{h}'] = gain / float(h)
    same[f'fully_hidden_h{h}'] = (same['confirmation_lead'] >= h).astype(int)
    same[f'waitall_ready_h{h}'] = same['same_window_waitall_delay'] + h
    same[f'complete_ready_h{h}'] = np.maximum(same['same_window_waitall_delay'].to_numpy(dtype=float), same['complete_delay'].to_numpy(dtype=float) + h)

def utility_table(df, decision_group):
    sub = select_group(df, decision_group)
    rows = []
    for h in PREPARATION_HORIZONS:
        gain = sub[f'gain_h{h}'].to_numpy(dtype=float)
        wa_ready = sub[f'waitall_ready_h{h}'].to_numpy(dtype=float)
        cc_ready = sub[f'complete_ready_h{h}'].to_numpy(dtype=float)
        rows.append({'decision_group': decision_group, 'preparation_steps': int(h), 'decisions': int(len(sub)), 'mean_gain': float(np.mean(gain)), 'median_gain': float(np.median(gain)), 'q1_gain': q1(gain), 'q3_gain': q3(gain), 'positive_gain_rate': float(np.mean(gain > 0)), 'fully_hidden_rate': float(sub[f'fully_hidden_h{h}'].mean()), 'mean_fraction_hidden': float(sub[f'fraction_hidden_h{h}'].mean()), 'mean_waitall_readiness': float(np.mean(wa_ready)), 'mean_complete_readiness': float(np.mean(cc_ready)), 'mean_readiness_reduction_pct': float(100.0 * (np.mean(wa_ready) - np.mean(cc_ready)) / np.mean(wa_ready))})
    return pd.DataFrame(rows)
main_table = utility_table(same, 'DEGRADATION')
all_table = utility_table(same, 'ALL')
tolerable_table = utility_table(same, 'TOLERABLE')
supp_table = pd.concat([all_table, main_table, tolerable_table], ignore_index=True)
e2e_rows = []
for group_name in ['ALL', 'DEGRADATION', 'TOLERABLE']:
    sub = select_group(same, group_name)
    cc_post = float(sub['complete_delay'].mean())
    wa_post = float(sub['same_window_waitall_delay'].mean())
    lead = wa_post - cc_post
    cc_total = WINDOW_COLLECTION_STEPS + cc_post
    wa_total = WINDOW_COLLECTION_STEPS + wa_post
    e2e_rows.append({'decision_group': group_name, 'decisions': len(sub), 'post_closure_complete': cc_post, 'post_closure_waitall': wa_post, 'post_closure_lead': lead, 'post_closure_reduction_pct': 100.0 * lead / wa_post, 'from_window_start_complete': cc_total, 'from_window_start_waitall': wa_total, 'from_window_start_lead': wa_total - cc_total, 'from_window_start_reduction_pct': 100.0 * (wa_total - cc_total) / wa_total})
e2e = pd.DataFrame(e2e_rows)
checks = []
checks.append({'check': 'exact 19,979 Complete decisions', 'pass': len(same) == 19979, 'value': len(same)})
checks.append({'check': 'exact 6,188 degradation decisions', 'pass': int((same['decision'] == 'DEGRADATION').sum()) == 6188, 'value': int((same['decision'] == 'DEGRADATION').sum())})
checks.append({'check': 'exact 13,791 tolerable decisions', 'pass': int((same['decision'] == 'TOLERABLE').sum()) == 13791, 'value': int((same['decision'] == 'TOLERABLE').sum())})
checks.append({'check': 'all frozen Table-5 anchors reproduced', 'pass': bool(validation['all_pass'].all()), 'value': f"{int(validation['all_pass'].sum())}/{len(validation)}"})
checks.append({'check': 'zero negative same-window leads', 'pass': negative_count == 0, 'value': negative_count})
checks.append({'check': 'minimum lead >= 0', 'pass': min_lead >= -1e-12, 'value': min_lead})
gain_formula_pass = True
for h in PREPARATION_HORIZONS:
    expected = np.minimum(same['confirmation_lead'].to_numpy(dtype=float), h)
    observed = same[f'gain_h{h}'].to_numpy(dtype=float)
    if not np.allclose(expected, observed, atol=1e-12, rtol=0):
        gain_formula_pass = False
checks.append({'check': 'gain(h) == min(h, confirmation lead)', 'pass': gain_formula_pass, 'value': gain_formula_pass})
readiness_pass = True
for h in PREPARATION_HORIZONS:
    diff = same[f'waitall_ready_h{h}'] - same[f'complete_ready_h{h}']
    if not np.allclose(diff.to_numpy(dtype=float), same[f'gain_h{h}'].to_numpy(dtype=float), atol=1e-12, rtol=0):
        readiness_pass = False
checks.append({'check': 'readiness difference exactly equals gain(h)', 'pass': readiness_pass, 'value': readiness_pass})
checks_df = pd.DataFrame(checks)
protocol = {'script': 'preparation_overlap', 'scientific_question': 'Can completion-sound early degradation confirmation overlap downstream preparation with unresolved-label acquisition?', 'counterfactual_definition': 'For each exact Complete-Cohort window, same-window Wait-All is the final label arrival time in that identical event-time block.', 'main_paper_population': 'DEGRADATION decisions', 'same_window_complete_decisions': 19979, 'degradation_decisions': 6188, 'preparation_horizons': PREPARATION_HORIZONS, 'window_size': WINDOW_SIZE, 'window_collection_steps': WINDOW_COLLECTION_STEPS, 'utility': 'gain(h)=min(T_waitall-T_complete, h)', 'important_scope': 'This quantifies preparation overlap, not earlier label-complete retraining.', 'scientific_rule': 'h values 10,25,50,100 were fixed before successful preparation analysis reconstruction.'}
with open(OUT / 'protocol_freeze.json', 'w', encoding='utf-8') as f:
    json.dump(protocol, f, indent=2, ensure_ascii=False)
same.to_csv(OUT / 'same_window_decision_level.csv', index=False)
validation.to_csv(OUT / 'anchor_validation.csv', index=False)
main_table.to_csv(OUT / 'main_paper_table.csv', index=False)
supp_table.to_csv(OUT / 'all_decision_types_table.csv', index=False)
e2e.to_csv(OUT / 'end_to_end_time_table.csv', index=False)
checks_df.to_csv(OUT / 'scientific_checks.csv', index=False)
print()
print('=' * 120)
print('MAIN PAPER TABLE — DEGRADATION RESPONSE PREPARATION')
print('=' * 120)
print(main_table.to_string(index=False))
print()
print('=' * 120)
print('ALL DECISION TYPES')
print('=' * 120)
print(supp_table.to_string(index=False))
print()
print('=' * 120)
print('END-TO-END TIME ACCOUNTING')
print('=' * 120)
print(e2e.to_string(index=False))
print()
print('=' * 120)
print('SCIENTIFIC CHECKS')
print('=' * 120)
print(checks_df.to_string(index=False))
all_pass = bool(checks_df['pass'].astype(bool).all())
print()
print('ALL SCIENTIFIC CHECKS PASSED:', all_pass)
if not all_pass:
    raise RuntimeError('preparation analysis checks failed.')
print()
print('=' * 120)
print('RUN COMPLETE')
print('=' * 120)
print('Output:', OUT)
