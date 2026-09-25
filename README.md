# Safety-aware federated learning for multilingual Parkinson's disease speech

Manuscript-specific code release for **“Safety-aware federated learning for robust multilingual speech biomarkers of Parkinson’s disease.”**

## Contents
- `fed_proposed.py`: proposed safety-aware federated learning framework.
- `fed_proposed_dp_sgd.py`: preliminary record-level DP-SGD sensitivity analysis.
- `pd_fl_common.py`: shared data/model utilities used by baseline scripts.
- `baselines/`: centralized, local, FedAvg, FedPer, q-FedAvg, Median, and Krum implementations.
- `analysis/compare_all_methods.py`: multi-seed experiment orchestration/comparison utilities.
- `analysis/statistical_analysis.py`: manuscript-matched mixed-effects, parametric-bootstrap, BH-FDR, and seed-level sign-permutation analysis.
- `analysis/analyze_proposed_method_dynamics.py`: dynamics/source-data analysis for the proposed method.

## Installation
```bash
pip install -r requirements.txt
```

## Data
The speech datasets are **not distributed** with this repository. Access is subject to the terms of the original data providers. The scripts expect precomputed time-series log-Mel arrays (`.npy`) and WAV filenames used to associate labels and features.

For public release, the code uses relative placeholders under `data/PD/`. Edit the `CFG` paths in `pd_fl_common.py`, `fed_proposed.py`, `fed_proposed_dp_sgd.py`, `baselines/fed_qfedavg_pd.py`, and `baselines/fed_robust_aggregation_pd.py` to point to your authorized local copies. The `data/` directory is excluded by `.gitignore`.

## Main non-private evaluation
The manuscript uses five prespecified seeds:

`32, 42, 52, 62, 123`

Example proposed-method run:
```bash
python fed_proposed.py --pd_clients EN ZH ES IT CZ --seed 42 --rounds 100 --outdir results/proposed_seed42
```

## DP-SGD sensitivity analysis
The privacy experiment is a preliminary **record-level** DP-SGD analysis. Example:
```bash
python fed_proposed_dp_sgd.py --pd_clients EN ZH ES IT CZ --seed 42 --rounds 100 --dp --noise_multiplier 1.0 --dp_clip_norm 1.0 --delta 1e-5 --outdir results/dp_sigma1
```

## Robust baselines
```bash
python baselines/fed_robust_aggregation_pd.py --pd_clients EN ZH ES IT CZ --seed 42 --rounds 100 --aggregator median --outdir results/median_seed42
python baselines/fed_robust_aggregation_pd.py --pd_clients EN ZH ES IT CZ --seed 42 --rounds 100 --aggregator krum --byzantine_f 1 --outdir results/krum_seed42
```

## Statistical analysis
`analysis/statistical_analysis.py` expects a tidy CSV containing `method`, `seed`, `client`, and metric columns such as `macro_f1`, `accuracy`, `mcc`, `sensitivity`, and `specificity`.
```bash
python analysis/statistical_analysis.py --input analysis_input.csv --proposed proposed --bootstrap 3000 --output statistical_results.csv
```

## Reproducibility notes
The repository is a curated manuscript-specific release. Raw audio, restricted datasets, participant-level manifests, checkpoints, experimental logs, and unrelated development/ongoing experiments are intentionally not included.

## License
MIT License.

## Copyright and reuse

Copyright (c) 2026 Changqin Quan. All rights reserved.

This repository is provided as research code accompanying the manuscript "Safety-aware federated learning for robust multilingual speech biomarkers of Parkinson's disease" for transparency and scholarly evaluation. No open-source license is granted with this repository. Permission for reuse, modification, redistribution, or incorporation of substantial portions of the source code into other software should be obtained from the copyright holder.

The repository intentionally contains only manuscript-specific release code. Datasets, participant-level information, experimental logs, checkpoints, and code for subsequent or ongoing research are not included.
