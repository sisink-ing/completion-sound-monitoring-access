# Completion-Sound Performance Monitoring

Research code for **Completion-Sound Performance Monitoring under Out-of-Order Delayed Labels: Finite-Window Confirmation and Complete Event-Time Cohorts**.

English | [简体中文](README.zh-CN.md)

The framework combines finite-window completion decisions, model-version authority and Complete Event-Time Cohorts for performance monitoring and selective adaptation under delayed, out-of-order feedback.

## 1. Set up the environment

Use **Python 3.12**. The reference CPU environment uses Python 3.12.14, NumPy 2.3.5, pandas 2.2.3, SciPy 1.17.0, scikit-learn 1.8.0 and joblib 1.5.3.

Open the repository root, where `README.md`, `requirements.txt`, `scripts/` and `data/` are located together. On Windows, type `cmd` in that folder's address bar and press Enter. Run these commands in Command Prompt, one line at a time:

```bat
py -3.12 -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
```

Linux/macOS:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
```

For a new terminal session, return to the repository root and reactivate `.venv`.

Additional dependencies are grouped in `environments/`: `generation.txt` for River, Matplotlib and tqdm; `bts.txt` for tzdata. ARF training uses a separate environment compatible with [scikit-multiflow](https://scikit-multiflow.github.io/). The core reproduction commands below use the pinned CPU packages. See [environment details](docs/environment.md).

## 2. Repository layout

| Path | Contents |
| --- | --- |
| `src/completion_monitoring/` | Monitoring engines, delayed feedback, version authority and cohort updates. |
| `scripts/` | File checks, statistical reconstruction and experiment launchers. |
| `experiments/` | Synthetic, baseline, protocol-audit, statistical and Retailrocket studies. |
| `data/frozen/` | 105 synthetic realizations: 5 fixed, 50 LR/DT and 50 ARF. |
| `data/raw/` | Electricity, Weather and filtered BTS inputs. |
| `data/processed/retailrocket/` | Selected model-input matrix, label-release schedule and task specification. |
| `data/statistics/`, `data/reference/` | Source scores and archived experiment results. |
| `configs/`, `environments/` | Experiment registry, parameter documentation and optional dependencies. |
| `extensions/`, `data/extensions/` | BTS/Freddie Mac study scripts and archived records. |
| `tests/`, `third_party/`, `docs/` | Completion-rule tests, attributed generator code and detailed guides. |
| `results/` | New results and experiment working copies. |

Keep these paths intact. `configs/protocol.json` documents the settings; executable constants remain in the drivers.

## 3. Data sources

The studies use five real-world sources and five synthetic families. BTS and Freddie Mac are extension studies.

| Real-world source | Acquisition links | Included material |
| --- | --- | --- |
| Electricity | [Benchmark repository](https://github.com/FabianHinder/Creating-Drift-Benchmarks) · [CSV download](https://raw.githubusercontent.com/FabianHinder/Creating-Drift-Benchmarks/main/stream_ds/electricity.csv.xz) | Compressed CSV under `data/raw/benchmarks/`. |
| Nebraska Weather | [Original benchmark page](https://users.rowan.edu/~polikar/nse.html) · [CSV download](https://raw.githubusercontent.com/FabianHinder/Creating-Drift-Benchmarks/main/stream_ds/NEweather.csv.xz) | Local file `data/raw/benchmarks/neweather.csv.xz`. |
| Retailrocket | [Kaggle dataset](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset) | Processed holdout: 10,594 rows, 24 features and 10,194 online label-release records. |
| BTS Airline On-Time Performance | [Official portal](https://www.transtats.bts.gov/ONTIME/) · [Download table](https://www.transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=FGJ&QO_fu146_anzr=Nv4%20Pn44vr45) | Three ATL-filtered monthly CSVs, January–March 2024. |
| Freddie Mac Single-Family Loan-Level Dataset | [Provider and documentation](https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset) · [Download portal](https://claritydownload.fmapps.freddiemac.com/) | Derived study records; original TXT inputs are obtained through the provider. |

| Synthetic family | Official generator |
| --- | --- |
| Agrawal | <https://riverml.xyz/latest/api/datasets/synth/Agrawal/> |
| Mixed | <https://riverml.xyz/latest/api/datasets/synth/Mixed/> |
| RandomTree | <https://riverml.xyz/latest/api/datasets/synth/RandomTree/> |
| SEA | <https://riverml.xyz/latest/api/datasets/synth/SEA/> |
| Sine | <https://riverml.xyz/latest/api/datasets/synth/Sine/> |

Drift construction uses [Creating-Drift-Benchmarks](https://github.com/FabianHinder/Creating-Drift-Benchmarks). The 105 frozen files are realizations of these five families. The core commands in Section 4 read the distributed inputs directly.

For original-data reconstruction, obtain the Retailrocket ZIP containing `events.csv`, or the Freddie Mac 2017–2022 sample origination/performance files. Acquisition and preprocessing details are in [docs/data.md](docs/data.md) and [docs/experiments.md](docs/experiments.md).

## 4. Reproduction and outputs

Run the following commands from the repository root in the activated environment, in order:

```bash
python scripts/verify_integrity.py
python -m unittest discover -s tests -v
python scripts/check_protocol.py
python scripts/reproduce_statistics.py
python experiments/real_world/replay_retailrocket.py
```

| Command/script | What it produces | Where to look |
| --- | --- | --- |
| `verify_integrity.py` | Checks 576 data files, 47 Python files and frozen-stream counts. | Terminal: `"passed": true`. |
| `unittest` | Exhaustive completion-rule test for window sizes 1–6. | Terminal: `OK`. |
| `check_protocol.py` | Eight runs: LR/DT with four strategies each on a frozen SEA stream. | `results/smoke/metrics.csv` and `verification.json`. |
| `reproduce_statistics.py` | Table 8/Table S1 paired differences, ranks, effective sample sizes and test results. | `results/statistics/table8/`, `tableS1/` and `verification.json`. |
| `replay_retailrocket.py` | Four DT strategies on the selected chronological holdout. | `results/retailrocket/metrics.csv`; verification: `"matches_reference": true`. |

The Table 8 output `waitall_accuracy_50_stream_differences.csv` contains all 50 paired differences; `waitall_accuracy_14_nonzero_ranks.csv` contains the 14 nonzero ranks. Statistical definitions are in [docs/statistics.md](docs/statistics.md).

## 5. Additional experiments

List the manuscript experiments, then select an entry:

```bash
python scripts/run_experiment.py --list
python scripts/run_experiment.py protocol_ablation --smoke --jobs 1
python scripts/run_experiment.py preparation
python scripts/run_experiment.py coverage --smoke --jobs 1
```

`protocol_ablation` examines pending windows and update authority; `preparation` computes overlap from archived matched-window decisions; `coverage` accounts for monitoring coverage. `--smoke` selects a reduced run and `--jobs` sets worker parallelism. Driver outputs are normally under `results/<entry>/work/result/`; [docs/experiments.md](docs/experiments.md) lists the full experiment catalog.

For another run, choose a fresh output directory, for example:

```bash
python scripts/check_protocol.py --output results/smoke_second
```

New computations go to `results/`; distributed reference records stay in `data/reference/`. Extension status: BTS preprocessing is runnable; downstream BTS replay requires adapting its source-relative input paths. Freddie Mac replay requires external provider files.

## 6. Troubleshooting and documentation

| Symptom | Action |
| --- | --- |
| Script or requirements.txt not found | Run from the repository root containing README, scripts/ and data/. |
| Existing work directory | Select a new directory with `--output`. |
| Missing generation or ARF dependency | Configure the relevant task-specific environment listed under `environments/`. |

Detailed guides: [experiments](docs/experiments.md) · [data](docs/data.md) · [statistics](docs/statistics.md) · [environment](docs/environment.md).

## 7. License

Project software uses the [MIT license](LICENSE). Upstream code retains its original license; dataset use follows provider terms. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
