# scaling-tabular-data : Predicting Neural Scaling Exponents on Tabular Data

Code and data for the paper:  
**"Why Some Tabular Datasets Scale Better Than Others"**

> *to be submitted for review in IEEE ICSIT 2026.*

---
## Authors

| Name | Role | Affiliation | Email |
|---|---|---|---|
| Md M. Zaman | First Author | M.Tech, BITS Pilani | 2025ab05168@wilp.bits-pilani.ac.in |
| A. Zaman | Co-Author | M.Tech, BITS Pilani | 2025ab05175@wilp.bits-pilani.ac.in |
| Dr. Neha Vinayak | Supervisor | Asst. Professor, CSIS, BITS Pilani | neha.vinayak@pilani.bits-pilani.ac.in |


## Overview

Neural test loss follows **L ∝ D^{−α}** as training size D grows, but what determines α
on tabular data is poorly understood. We show that α̂ ranges from near-zero to above 0.96
across **141 OpenML datasets** — and that this variation is largely predictable from
geometric and statistical properties of the dataset, *before any model is trained at scale.*

**Key results:**

| Metric | Value |
|---|---|
| Datasets analysed | 141 |
| XGBoost LODO R² | 0.687 |
| XGBoost LODO Spearman ρ | 0.811 |
| Permutation p-value | < 0.001 |
| Bootstrap 95% CI (ρ) | [0.714, 0.856] |
| Stability mean ρ (20 × 80% subsamples) | 0.765, CI [0.720, 0.814] |

---

## Repository Structure

```
.
├── README.md
├── requirements.txt
│
├── 01_generate_data.py       # Stage 1 + 2: alpha estimation & feature extraction
├── 02_generate_figures.py    # Reproduces all paper figures
├── geometry_image.ipynb      # Exploratory geometry analysis notebook
│
├── data/
│   ├── results.csv               # Per-dataset: α̂, R², CI, features (141 rows)
│   ├── lodo_predictions.csv      # Leave-one-dataset-out predictions
│   ├── feature_importance.csv    # XGBoost Gini importances (16 features)
│   ├── feature_corr.csv          # Spearman ρ of each feature vs α̂
│   ├── bootstrap_stability.csv   # 20 × 80%-subsample stability analysis
│   ├── ablation_results.csv      # Feature ablation R² and ρ
│   └── curve_bank.json           # Raw learning curves per dataset
│
└── validation/
    └── statistical_validation.json   # All reported statistics in one file
```

---

## Reproducing Results

### Requirements

```bash
pip install -r requirements.txt
```

Tested on **Kaggle T4 × 2 GPU** (free tier). CPU-only runs are supported but
`01_generate_data.py` will be significantly slower (~8–12 h vs ~2 h on GPU).

### Step 1 — Generate data

```bash
python 01_generate_data.py
```

Downloads 141 OpenML datasets, fits learning curves for MLP / RF / XGBoost,
extracts 16 geometric and statistical features, runs LODO cross-validation,
and writes all CSV/JSON outputs to `./out/`.

**Outputs:** `results.csv`, `lodo_predictions.csv`, `feature_importance.csv`,
`feature_corr.csv`, `ablation_results.csv`, `bootstrap_stability.csv`,
`statistical_validation.json`

### Step 2 — Generate figures

```bash
python 02_generate_figures.py
```

Reads from `./out/` and writes all paper figures to `./figures/`.

### Skip re-running

All pre-computed outputs are included in `data/` and `validation/`.
To verify reported statistics without re-running, inspect `statistical_validation.json` directly.

---

## Feature Set

16 features extracted per dataset, grouped by type:

| Group | Features |
|---|---|
| **Geometric** | intrinsic dimension, LID, anisotropy, density variation, margin |
| **Cluster** | cluster structure, cluster separation |
| **Statistical** | linear probe difficulty, Fisher ratio, class entropy, class imbalance, label noise |
| **Structural** | n\_samples, n\_features, n\_classes, early\_loss |

**Top predictors by Gini importance:**

| Rank | Feature | Importance |
|---|---|---|
| 1 | linear\_probe\_difficulty | 0.230 |
| 2 | class\_entropy | 0.119 |
| 3 | label\_noise | 0.110 |
| 4 | intrinsic\_dim | 0.069 |
| 5 | n\_features | 0.067 |

---

## Architecture Agnosticism

α̂ estimated independently under MLP, Random Forest, and XGBoost.
Cross-architecture Spearman ρ ranges **0.537–0.777**, confirming the exponent
reflects data geometry rather than model-specific behaviour.

---

## Statistical Validation

All validation statistics are in `validation/statistical_validation.json`:

```json
{
  "n_datasets": 141,
  "lodo_xgb_r2": 0.687,
  "lodo_xgb_spearman": 0.811,
  "permutation_p": 0.0,
  "bootstrap_spearman_ci_lo": 0.714,
  "bootstrap_spearman_ci_hi": 0.856,
  "stability_mean_rho": 0.765,
  "stability_ci_lo": 0.720,
  "stability_ci_hi": 0.814
}
```

---

## Requirements

```
numpy
pandas
scipy
scikit-learn
xgboost
torch
openml
matplotlib
seaborn
tqdm
```

Full pinned versions: see `requirements.txt`

---

## Dataset Sources

All datasets sourced from [OpenML](https://www.openml.org/) via the `openml` Python API.
Dataset IDs are listed in `01_generate_data.py` (DATASETS dict, line 78).

---



---

## License

Code: MIT  
Data: Derived from OpenML — original dataset licenses apply.
