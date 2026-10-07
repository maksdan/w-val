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

**`experiments/download_uci_data.py`** — Setup helper. Downloads the Fernández-Delgado et al. UCI collection (121 datasets) from a mirror into `UCI_data/` (git-ignored) and checks it with the repo loader.

**`experiments/run_tabular_figs345.py`** — Real tabular data (UCI classification, regression sets): paper Figs 3A/3B, 4, 5 and Table 1, plus a screening step. See the next section.

---

## Tabular-data figures (Figs 3–5, Table 1)

```bash
pip install -r requirements.txt
python experiments/download_uci_data.py          # one-off, writes UCI_data/ (not committed)
python experiments/run_tabular_figs345.py all    # inventory -> screen -> figures -> fig4
```

Subcommands: `inventory`, `screen`, `figures`, `fig4`, `all`. Useful flags: `--uci-dir`, `--out-dir`, `--run-name`, `--device {auto,cuda,cpu}`, `--cases uci:spambase openml:pol` (override the automatic dataset selection), `--seeds`, `--full-seeds`, `--epochs`, `--no-resume`. Defaults live in `TabularConfig` (`wvalue/tabular.py`). A GPU is used when available; on CPU it falls back with a warning (the full pipeline is slow). Everything checkpoints under `results_tabular/<run-name>/` and re-running resumes; changing training/analysis settings on an existing run folder raises an error (use a new `--run-name`).

| Paper | Output (in `results_tabular/<run>/figures/`, each as `.png` and `.pdf`) | Produced by |
|---|---|---|
| Fig 3A | `<dataset>_<regime>_s<seed>_fig3A` | `plots.fig1_weight_distributions` |
| Fig 3B | `<dataset>_<regime>_s<seed>_fig3B` | `plots.fig2_null_comparison` |
| Fig 4 | `<dataset>_<regime>_fig4` | copy of `plots.plot_sample_size_r2` + layer-1 reset control |
| Fig 5 | `<dataset>_<regime>_fig5` | copy of `plots.fig5_threshold_sweep` |
| Table 1 | `tables/table1.csv` | same rows as the paper plus the freeze-mask arm |

Also written: `tables/dataset_inventory.csv`, `tables/screen_rows.csv`, `tables/screen_summary.csv` (transparency table), `tables/selection.json`, and per-run artifacts for redrawing.

Notes: the default selection rule (BH fires in ≥60% of screen seeds, model learned, init KS ok, ranked by tail enrichment) can pick no regression sets — use `--cases`. Datasets that fail to load (e.g. a wrong OpenML fetch) are logged and skipped. Regimes: `fixed` (100 + 10 epochs) and `tuned` (weight decay and epochs chosen on a validation split). Deliberate differences from the paper scripts (epochs, batch size, auto-sized hidden layers, reset control) are listed in `wvalue/tabular.py` and the original notebook.

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

**UCI classification datasets**: run `python experiments/download_uci_data.py` to download them into `UCI_data/` at the repo root (git-ignored; uses `gdown`). Alternatively place each dataset folder (`<name>/<name>_R.dat`, `conxuntos.dat`) under `UCI_data/` manually.

---

## Configuration

Every experiment script has a clearly labelled `# ── CONFIG ──` section at the top with all user-editable parameters and comments explaining each option. Edit that section to change datasets, model sizes, training hyperparameters, and output paths. No other changes are needed.
