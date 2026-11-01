# Computational environment

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

Synthetic stream generation imports River, Matplotlib and tqdm. ARF training
imports `skmultiflow.meta.AdaptiveRandomForestClassifier`; the ARF Delayed-ADWIN
driver uses the corresponding scikit-multiflow detector. Package names are
listed in `environments/generation.txt` and `environments/arf.txt`.

`--jobs` controls process-level parallelism. Drivers restrict numerical-library
threads to limit nested parallelism. Use a fresh output directory for each run.
