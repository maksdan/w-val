# W-Values: Statistical Significance for Neural Network Weights

This codebase implements **w-values** — a principled statistical significance measure for neural network weights based on the Beta distribution.

**Key idea:** After Frobenius-normalizing a weight matrix, each squared weight follows Beta(0.5, (B-1)/2) under the null hypothesis of random initialization. The w-value is Beta.sf(x) — a p-value. Weights with small w-values are statistically significant (too large to be explained by chance). The codebase uses this to regularize or prune networks.

---

## Quick start

```bash
pip install -r requirements.txt

# Run the broad classification experiment (trains on MNIST by default)
python experiments/run_broad_eval.py

# Run null distribution analysis + pruning (uses California Housing by default)
python experiments/run_null_analysis.py

# Run sample-size study (uses three_way synthetic task by default)
python experiments/run_sample_size.py
```

All scripts save figures to a results directory (e.g. `results_broad_eval/`). To display interactively, change `matplotlib.use('Agg')` to `matplotlib.use('TkAgg')` at the top of the script.

---

## Experiments

**`experiments/run_broad_eval.py`** — Multi-dataset classification experiment. Trains MLP classifiers with baseline, L2 regularization, and w-value filtering on any combination of MNIST, FashionMNIST, CIFAR-10, SVHN, Covertype, 20 Newsgroups, and UCI datasets. Produces training curves, weight distribution heatmaps, Q-Q plots vs. the theoretical null, and near-zero weight counts.

**`experiments/run_null_analysis.py`** — Null distribution analysis and BH pruning. Trains a regression MLP, fits a Beta null to the normalized squared weights, selects an ECDF-deviation threshold, applies Benjamini-Hochberg FDR correction to identify significant weights, prunes insignificant ones, and compares three fine-tuning strategies. Produces Figures 1-5.

**`experiments/run_sample_size.py`** — Sample-size study. Sweeps a range of training set sizes and compares baseline vs. pruned+freeze vs. pruned+full-finetune R² to show when pruning helps under data-limited conditions.

**`experiments/download_uci_data.py`** — Setup helper. Checks which UCI datasets are present and prints instructions for downloading the rest.

---

## The wvalue package

| Module | Description |
|--------|-------------|
| `wvalue.core` | Beta SF computation, BetaSFLookupTable, `compute_w_value`, significance regularizer, w-value filtered training |
| `wvalue.training` | Classification training loops: baseline, L1, L2, significance regularizer |
| `wvalue.utils` | `set_seed` for reproducibility |
| `wvalue.datasets` | Classification datasets (MNIST, UCI, grokking) + regression DATASET_CASES |
| `wvalue.analysis` | Null distribution analysis (Beta fitting, BH pruning, Q-Q) + weight snapshot capture |
| `wvalue.regression` | Regression MLP, training, pruning experiments, threshold sweep, sample-size sweep |
| `wvalue.broad_eval` | High-level runner for multi-dataset classification experiments |

---

## Data

**Image and sklearn datasets** download automatically when first used:
- MNIST, FashionMNIST, CIFAR-10, SVHN — downloaded by torchvision
- Covertype, 20 Newsgroups — downloaded by scikit-learn
- California Housing, Diabetes, etc. — downloaded via OpenML/sklearn

**UCI classification datasets** must be downloaded manually from the KEEL repository:
- URL: https://sci2s.ugr.es/keel/datasets.php
- Place each dataset folder under `UCI_data/` at the repo root
- Run `python experiments/download_uci_data.py` for detailed instructions and to check which are present

---

## Configuration

Every experiment script has a clearly labelled `# ── CONFIG ──` section at the top with all user-editable parameters and comments explaining each option. Edit that section to change datasets, model sizes, training hyperparameters, and output paths. No other changes are needed.
