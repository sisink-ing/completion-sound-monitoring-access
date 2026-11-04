# Extended studies

## BTS

```bash
python scripts/run_extension.py bts_prepare
python scripts/run_extension.py bts_static
python scripts/run_extension.py bts_schedule
python scripts/run_extension.py bts_matched
python scripts/run_extension.py bts_adaptive
```

Inputs are stored in `data/raw/bts/`; each command uses a separate working
directory below `results/extensions/`. The replay constructs label-release times from flight
arrival timestamps and a specified buffer. Predictive features are selected
explicitly by the preparation pipeline.

## Freddie Mac

The mortgage studies include task feasibility, cohort readiness, incomplete
labels, outcome reconstruction and chronological/calendar replay. Provide the
origination and performance files under the provider's access terms and set
`PAA_FREDDIE_DATA` to their directory.

```bash
python scripts/run_extension.py freddie_feasibility
python scripts/run_extension.py freddie_protocol
python scripts/run_extension.py freddie_calendar
```

Reference extension results are stored in `data/extensions/`. These studies
are separate from the manuscript's synthetic, Electricity and Retailrocket
evaluations.
