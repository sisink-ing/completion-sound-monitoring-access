# Data

| Location | Contents |
| --- | --- |
| `data/frozen/fixed/` | Five fixed synthetic streams |
| `data/frozen/lrdt/` | 50 LR/DT streams; seeds 6001–6010 within each family |
| `data/frozen/arf/` | 50 ARF streams; seeds 6101–6110 within each family |
| `data/raw/benchmarks/` | Electricity and Nebraska Weather inputs |
| `data/raw/bts/` | Three monthly ATL-filtered BTS input files |
| `data/processed/retailrocket/` | Model inputs, label-release ticks and protocol |
| `data/statistics/` | Source score records for paired-stream inference |
| `data/reference/` | Reference monitoring, prediction and statistical results |
| `data/extensions/` | BTS and Freddie Mac result records |

The NPZ files contain frozen feature and label arrays. The Retailrocket input
contains 10,594 observations and 24 predictive features; identifiers and outcome
fields are excluded from the feature matrix by the stored feature specification.
The release schedule contains the 10,194 online query identifiers and their
label-release ticks. Positive release times derive from purchase events;
negative release times derive from the 24-hour target deadline.

The 50-stream statistical inputs are in `data/statistics/`. Paired differences,
rank records and statistical summaries are in
`data/reference/statistics/table8/` and `data/reference/statistics/tableS1/`.

## Sources

- Benchmark generator and benchmark data distribution:
  https://github.com/FabianHinder/Creating-Drift-Benchmarks
- Retailrocket:
  https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset
- BTS:
  https://www.transtats.bts.gov/ONTIME/
- Freddie Mac:
  https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset

The full Retailrocket event log and Freddie Mac origination/performance files
are obtained from their providers. Dataset licenses and access terms are
separate from the software license.

`data/manifest.csv` records each distributed data file, its size and SHA-256
digest. `scripts/verify_integrity.py` verifies these values.
