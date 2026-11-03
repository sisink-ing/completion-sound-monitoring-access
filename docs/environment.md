# Computational environment

## Core checks and statistical reconstruction

The protocol checks, statistical reconstruction and selected Retailrocket
replay were executed with Python 3.12.14 and the following packages:

| Package | Version |
| --- | --- |
| NumPy | 2.3.5 |
| pandas | 2.2.3 |
| SciPy | 1.17.0 |
| scikit-learn | 1.8.0 |
| joblib | 1.5.3 |

Install this environment with `python -m pip install -r requirements.txt`.
These computations use CPU. Execution commands record package versions in
their output directories.

## ARF training and replay

The separate environment in `environments/arf.txt` was validated on Linux
x86_64 with Python 3.9.25 and scikit-multiflow 0.5.3. The experiment drivers use
`skmultiflow.meta.AdaptiveRandomForestClassifier` with `split_criterion='gini'`
and `random_state=2026`; Delayed-ADWIN uses the detector from the same library.

From the repository root, create the environment and install its build and
runtime dependencies. Source installation requires a C/C++ compiler.

```bash
python3.9 -m venv .venv-arf
source .venv-arf/bin/activate
python -m pip install pip==24.0
python -m pip install -r environments/arf-bootstrap.txt
python -m pip install --no-build-isolation -r environments/arf.txt
python -m pip check
```

Check numerical agreement on selected frozen-input tasks before a full evaluation:

```bash
python scripts/check_arf_environment.py
```

Full frozen-input evaluations use `python scripts/run_experiment.py arf --jobs 1`
and `python scripts/run_experiment.py adwin --jobs 1`. Each command records the
installed package versions and run settings under its `results/<entry>/`
directory. Choose a fresh `--output` directory when repeating a run.

This check compares 20 ARF strategy rows across the five stream families and
one SEA Delayed-ADWIN row with the distributed reference records. It writes
the replay scores, maximum absolute errors and environment under
`results/arf_environment_check/`. The supplied record is
`environments/arf-verification.json`. Exact historical training
versions were not recorded; this file identifies the environment validated
for the present replay.

## Stream generation

Synthetic stream generation imports River, Matplotlib and tqdm, listed in
`environments/generation.txt`. The frozen-input ARF replay reads the supplied
streams directly.

`--jobs` controls process-level parallelism. Drivers restrict numerical-library
threads to limit nested parallelism. Use a fresh output directory for each run.
