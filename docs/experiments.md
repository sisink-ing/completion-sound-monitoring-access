# Experiments

| Study | Entry | Reference data |
| --- | --- | --- |
| Chronological Electricity and Weather | `natural` | `data/reference/cohort_protocol/` |
| Fixed streams with fresh delays | `fresh_delay` | `data/reference/fixed_stream_holdout/` |
| Independent LR/DT streams | `unseen` | `data/reference/unseen_stream_pipeline/` |
| Confirmation reliability | `confirmation` | `data/reference/confirmation_comparison/` |
| Confirmation summaries | `confirmation_summary` | `data/reference/confirmation_efficiency/` |
| Fixed-trigger training membership | `fixed_trigger` | `data/reference/cohort_membership_audit/` |
| Common ARF holdout | `arf` | `data/reference/arf_holdout/` |
| Full-process feedback permutations | `full_process` | `data/reference/feedback_order_audit/` |
| Matched-window preparation overlap | `preparation` | `data/reference/preparation_overlap/` |
| Delayed-ADWIN with ARF | `adwin` | `data/reference/delayed_adwin/` |
| Protocol ablations | `protocol_ablation` | `data/reference/protocol_ablation/` |
| Dense changes | `dense_change` | `data/reference/dense_drift_audit/` |
| Event-mass coverage | `coverage` | `data/reference/coverage_accounting/` |
| Aggregation checks | `statistical_audit` | `data/reference/statistical_audit/` |

Run an entry with `python scripts/run_experiment.py <entry>`. Commands create
private working directories with input copies and record their execution
environment. `configs/protocol.json` specifies the documented protocol values;
the corresponding drivers contain the executable constants.

## Experimental units

The independent LR/DT design comprises five stream families, ten stream seeds,
two learners, five delay regimes and five delay seeds: 2,500 tasks per strategy.
The ARF design uses 50 independent stream realizations with five regimes and
five delay seeds: 1,250 tasks per strategy. Stream-level inference uses the
50 streams as independent units.

The fixed-trigger audit holds the triggering event-time window constant across
100 schedules for each of 100 cases. The full-process audit reruns the complete
monitoring and update sequence under each perturbed schedule.

Protocol-ablation and dense-change drivers use default designs of 300 and
360 matched tasks, respectively. Their optional `--full` switches specify
larger designs. Preparation analysis uses 19,979 matched decisions, including
6,188 degradation decisions. Coverage accounting uses 4,500,000 online events
from 2,500 runs.

## Retailrocket

```bash
python experiments/real_world/replay_retailrocket.py
```

The selected chronological holdout contains 200 training observations,
200 calibration observations and 10,194 online queries. The replay reads the
24-feature matrix and integer label-release schedule from
`data/processed/retailrocket/` and runs the four primary DT strategies.

The causal query builder is `task_construction/query_builder.py`. For an
external Retailrocket archive containing `events.csv`, run:

```bash
python experiments/real_world/prepare_retailrocket.py --input /path/to/RetailRocket.zip
```

Task construction and selected-holdout replay are separate operations. The
selected task and feature specification are stored in
`data/processed/retailrocket/retailrocket_protocol.json`.

The task-selection script reads `catalog.json` and dataset pickle files from
`results/retailrocket_preparation/processed/`. Set `PAA_TASK_ROOT` to another
preparation directory when using an external catalog.
