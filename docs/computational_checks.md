# Computational checks

The following commands were executed with the environment specified in
`environment.md`.

| Computation | Executed scope | Result |
| --- | --- | --- |
| File integrity | 576 data files and 105 frozen streams | SHA-256 checks pass |
| Completion certificates | 1,092 partial-observation patterns; 5,460 admissible completions | All certificates agree with every compatible completion |
| LR/DT protocol | Two learners and four strategies on a frozen SEA stream | Eight strategy runs; certified decisions agree with the oracle |
| Parallel execution | Two workers, two tasks | Eight strategy runs complete |
| ARF score analysis | 12 paired comparisons over 50 streams | Reference statistics reproduced |
| LR/DT score analysis | Nine paired comparisons over 50 streams | Reference statistics reproduced |
| Retailrocket replay | Four strategies; 10,194 online queries | Prediction metrics, update counts and confirmation delays match reference results to floating-point tolerance |
| Fixed-trigger check | Ten cases, ten schedules per case | Complete Cohort membership remains unchanged |
| Matched-window analysis | 19,979 decisions | Reference confirmation-time and preparation checks pass |
| Coverage accounting | 2,500 runs and 4,500,000 events | Reference event-mass partition reproduced |
| Protocol ablation | One task, four variants | Completes |
| BTS preparation | Three monthly input files | Completes |

Execution commands record their package versions and parameters in the selected
output directory. Reference records are stored separately under `data/`.
