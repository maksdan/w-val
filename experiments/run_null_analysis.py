#!/usr/bin/env python3
"""
Null distribution analysis, Beta fitting, Benjamini-Hochberg pruning, and fine-tuning.

Trains a regression MLP on a synthetic or real dataset, captures first-layer weight
snapshots at initialization and after training, then:
  - Fits a Beta null distribution to the Frobenius-normalized squared weights
  - Computes the ECDF-deviation threshold (argmin MSE)
  - Applies Benjamini-Hochberg FDR correction to identify significant weights
  - Prunes insignificant weights and compares three fine-tuning strategies
  - Sweeps pruning thresholds and plots R² vs. threshold

Produces Figures 1-5 from the analysis notebooks.

Run:
    python experiments/run_null_analysis.py

Edit the CONFIG section below to choose the dataset/task and model hyperparameters.
"""

# ── Path setup ────────────────────────────────────────────────────────────────
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# ── CONFIG ────────────────────────────────────────────────────────────────────
SEED         = 42
INPUT_DIM    = 100         # overridden for real datasets
N_TRAIN      = 10_000      # training samples (synthetic); real datasets use their split
N_TEST       = 2_000       # test samples (synthetic); real datasets use their fixed test split
HIDDEN_SIZES = [128, 64]
BATCH_SIZE   = 256
EPOCHS       = 50
LR           = 3e-4

# Weight decay (overridden per-dataset from DATASET_CASES)
WEIGHT_DECAY = 1e-2

# Threshold for the Beta approach (100% of mass in theoretical null CDF)
BETA_THRESHOLD_QUANTILES = [1.00]

QQ_MAX_POINTS = 500
QQ_CI         = 95           # order-statistic band confidence level (%)
NULL_N_DRAWS  = 5            # null draws for w-value erfc null

# Dataset/task selection — key into DATASET_CASES in wvalue/datasets.py
# Synthetic options: 'x1_x2', 'linear_sparse', 'quadratic', 'two_products',
#                    'three_way', 'four_way', 'sin_product', 'five_products'
# Real options:      'california_housing', 'diabetes', 'wine_quality', 'concrete',
#                    'abalone', 'power_plant', 'energy_efficiency', 'auto_mpg'
CASE = 'california_housing'

# Plot controls
SHOW_BETA_FITS   = False   # show fitted Beta PDF overlays
SHOW_NULL_CURVES = False   # show true null / Exp(1) / erfc null curves
LOG_XB_AXIS      = False   # use log x-axis with log-spaced bins for x·B histograms

SAVE_PLOT = True
PLOT_DIR  = 'results_null_analysis'

# Fine-tuning after pruning
FINETUNE_EPOCHS = 10
BH_ALPHA        = 0.05    # FDR level for Benjamini-Hochberg
# ─────────────────────────────────────────────────────────────────────────────

import math, copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import wvalue.core as wvalue_utils
from wvalue.core import BetaSFLookupTable
from wvalue.utils import set_seed
import wvalue.regression as reg
import wvalue.analysis as null_a
from wvalue.datasets import DATASET_CASES
from wvalue import plots

# ── Device & lookup table ─────────────────────────────────────────────────────
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
print(f'Using device: {device}')

LOOKUP_RESOLUTION = 500_000
wvalue_utils.beta_sf_lookup = BetaSFLookupTable(resolution=LOOKUP_RESOLUTION)
print(f'Lookup table initialized  (resolution={LOOKUP_RESOLUTION:,})')

# ── Validate and load dataset case ────────────────────────────────────────────
assert CASE in DATASET_CASES, f'Unknown CASE {CASE!r}. Choose from: {list(DATASET_CASES)}'
CASE_CFG     = DATASET_CASES[CASE]
TASK_DESC    = CASE_CFG['desc']
SIGNAL_COLS  = CASE_CFG['signal_cols']
WEIGHT_DECAY = CASE_CFG.get('weight_decay', WEIGHT_DECAY)
INPUT_DIM    = CASE_CFG.get('input_dim', INPUT_DIM)
N_WEIGHTS    = HIDDEN_SIZES[0] * INPUT_DIM
NULL_STD     = 1.0 / math.sqrt(N_WEIGHTS)
H0           = HIDDEN_SIZES[0]

PLOT_FILE_PREFIX = f'{CASE}_d{INPUT_DIM}_e{EPOCHS}_s{SEED}'

print(f'Case         : {CASE}')
print(f'Task         : {TASK_DESC}')
print(f'Signal cols  : {SIGNAL_COLS}  ({len(SIGNAL_COLS)} features)')
print(f'Weight decay : {WEIGHT_DECAY}')
print(f'Model        : {INPUT_DIM} -> {HIDDEN_SIZES} -> 1  (ReLU)   B = {N_WEIGHTS}')
print(f'Plot prefix  : {PLOT_FILE_PREFIX}')

# ── Data ──────────────────────────────────────────────────────────────────────
if 'load_data' in CASE_CFG:
    X_train, y_train, X_test, y_test = CASE_CFG['load_data'](seed=SEED)
    N_TRAIN   = len(X_train)
    N_TEST    = len(X_test)
    INPUT_DIM = X_train.shape[1]
    N_WEIGHTS = HIDDEN_SIZES[0] * INPUT_DIM
    NULL_STD  = 1.0 / math.sqrt(N_WEIGHTS)
    print(f'Actual INPUT_DIM={INPUT_DIM}  N_WEIGHTS={N_WEIGHTS}')
else:
    X_train, y_train = reg.make_data(N_TRAIN, INPUT_DIM, CASE_CFG['make_y'], seed=SEED)
    X_test,  y_test  = reg.make_data(N_TEST,  INPUT_DIM, CASE_CFG['make_y'], seed=SEED + 1)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE,
                          shuffle=True, generator=torch.Generator().manual_seed(SEED))
test_loader  = DataLoader(TensorDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)
print(f'Train: {N_TRAIN}  |  Test: {N_TEST}')

# ── Model & null ──────────────────────────────────────────────────────────────
model     = reg.make_model(INPUT_DIM, HIDDEN_SIZES, SEED, device)
criterion = nn.MSELoss()
NULL_SORTED = null_a.build_null_sorted(H0, INPUT_DIM, n_draws=NULL_N_DRAWS)
print(f'Null sorted: {len(NULL_SORTED):,} samples  '
      f'(median = {np.median(NULL_SORTED):.4f}, '
      f'99th pct = {np.percentile(NULL_SORTED, 99):.4f})')
print('Ready.')

# ── Train ─────────────────────────────────────────────────────────────────────
print('\nCapturing init snapshot...')
snap_init = reg.capture(model, N_WEIGHTS, device, wvalue_utils.beta_sf_lookup)
print('Init snapshot captured.')

print(f'\nTraining for {EPOCHS} epochs...')
reg.train_regression(model, train_loader, device, EPOCHS, LR, WEIGHT_DECAY,
                     test_loader=test_loader, print_every=25)

snap_trained = reg.capture(model, N_WEIGHTS, device, wvalue_utils.beta_sf_lookup)
mse, r2 = reg.evaluate(model, test_loader, device)
print(f'\nFinal  mse={mse:.4f}  r²={r2:+.4f}')
print('Trained snapshot captured.')

# ── Null analysis: ECDF threshold, Beta fits, Benjamini-Hochberg ──────────────
print('\nRunning null analysis...')

analysis        = null_a.run_null_analysis(
    snap_init, snap_trained, N_WEIGHTS,
    beta_threshold_quantiles=BETA_THRESHOLD_QUANTILES,
    signal_cols=SIGNAL_COLS,
    alpha_bh=BH_ALPHA,
)
ecdf_thresholds = analysis['ecdf_thresholds']
fits            = analysis['fits']
ecdf_fits       = analysis['ecdf_fits']
T_z_bh          = analysis['T_z_bh']
T_z_ecdf_all    = analysis['T_z_ecdf_all']
T_x_ecdf        = analysis['T_x_ecdf']

print(f'BH threshold : T_z = {T_z_bh:.4f}  (FDR alpha = {BH_ALPHA})')
for subset_key, (T_z, mse_star, T_grid_e, mse_grid_e) in ecdf_thresholds.items():
    print(f'ECDF [{subset_key:6s}]: T_z={T_z:.4f}  mse*={mse_star:.6f}  '
          f'(grid {len(T_grid_e)} points)')
for q, fit in fits.items():
    ai, bi = fit['init']
    at, bt = fit['trained']
    print(f'Beta {q:.0%}: init (a={ai:.4f} b={bi:.1f})  trained (a={at:.4f} b={bt:.1f})')

os.makedirs(PLOT_DIR, exist_ok=True)

# ── Pruning + fine-tuning ─────────────────────────────────────────────────────
print(f'\nRunning pruning experiments (finetune_epochs={FINETUNE_EPOCHS})...')
pruning         = reg.run_pruning_experiments(
    model, snap_trained, train_loader, test_loader, device,
    T_z_bh=T_z_bh, N_WEIGHTS=N_WEIGHTS,
    finetune_epochs=FINETUNE_EPOCHS, lr=LR, weight_decay=WEIGHT_DECAY,
    lookup_table=wvalue_utils.beta_sf_lookup,
)
mask_2d         = pruning['mask_2d']
model_pruned    = pruning['model_pruned']
model_masked_ft = pruning['model_masked_ft']
snap_masked_ft  = pruning['snap_masked_ft']
mse_masked_ft   = pruning['mse_masked_ft']
r2_masked_ft    = pruning['r2_masked_ft']
model_finetuned = pruning['model_finetuned']
snap_finetuned  = pruning['snap_finetuned']
mse_finetuned   = pruning['mse_finetuned']
r2_finetuned    = pruning['r2_finetuned']
model_continued = pruning['model_continued']
snap_continued  = pruning['snap_continued']
mse_continued   = pruning['mse_continued']
r2_continued    = pruning['r2_continued']
mse_trained     = pruning['mse_trained']
r2_trained      = pruning['r2_trained']

# ── Performance comparison table ─────────────────────────────────────────────
print('\nPerformance comparison:')
total_ep = EPOCHS + FINETUNE_EPOCHS
rows = [
    (f'Trained            ({EPOCHS} ep)',                        mse_trained,   r2_trained),
    (f'Pruned + freeze-mask train ({FINETUNE_EPOCHS} ep)',      mse_masked_ft, r2_masked_ft),
    (f'Pruned + full fine-tune ({FINETUNE_EPOCHS} ep)',         mse_finetuned, r2_finetuned),
    (f'Continued          ({total_ep} ep, no prune)',           mse_continued, r2_continued),
]
col_w = max(len(r[0]) for r in rows) + 2
print(f'\n  {"Model":<{col_w}}  {"MSE":>10}  {"R2":>10}')
print('  ' + '-' * (col_w + 24))
for name, mse, r2 in rows:
    delta = f'  ({(mse - mse_trained) / mse_trained * 100:+.1f}%)' if mse != mse_trained else ''
    print(f'  {name:<{col_w}}  {mse:>10.4f}  {r2:>+10.4f}{delta}')

# ── Threshold sweep for Figure 5 ──────────────────────────────────────────────
print('\nRunning threshold sweep for Figure 5...')
T_grid_5  = np.arange(0.0, 20, 1)

r2_masked_5, r2_ft_5 = reg.threshold_sweep(
    model, snap_trained, train_loader, test_loader, device,
    T_grid=T_grid_5, finetune_epochs=FINETUNE_EPOCHS, lr=LR, weight_decay=WEIGHT_DECAY,
)
print(f'Sweep complete.  Optimal T_z (BH) = {T_z_bh:.4f}')

# ── Plots ─────────────────────────────────────────────────────────────────────
print('\nGenerating figures...')
plots.fig1_weight_distributions(
    snap_init, snap_trained, T_z_bh, T_z_ecdf_all, N_WEIGHTS, EPOCHS,
    save=SAVE_PLOT, plot_dir=PLOT_DIR, prefix=PLOT_FILE_PREFIX,
    show_null=SHOW_NULL_CURVES, show_beta_fits=SHOW_BETA_FITS, log_xb=LOG_XB_AXIS,
    fits=fits, beta_threshold_quantiles=BETA_THRESHOLD_QUANTILES,
)
plots.fig2_null_comparison(
    snap_init, snap_trained, fits, ecdf_fits, T_z_ecdf_all,
    NULL_SORTED, N_WEIGHTS, EPOCHS,
    is_real_data=('load_data' in CASE_CFG),
    qq_ci=QQ_CI, show_beta_fits=SHOW_BETA_FITS,
    beta_threshold_quantiles=BETA_THRESHOLD_QUANTILES,
    save=SAVE_PLOT, plot_dir=PLOT_DIR, prefix=PLOT_FILE_PREFIX,
)
plots.fig3_ecdf_diagnostic(
    snap_trained, ecdf_thresholds, N_WEIGHTS, SIGNAL_COLS, INPUT_DIM,
    save=SAVE_PLOT, plot_dir=PLOT_DIR, prefix=PLOT_FILE_PREFIX,
)
plots.fig5_threshold_sweep(
    T_grid_5, r2_masked_5, r2_ft_5, r2_trained, r2_continued,
    T_z_bh, snap_trained['z'].ravel(), N_WEIGHTS, EPOCHS, FINETUNE_EPOCHS,
    save=SAVE_PLOT, plot_dir=PLOT_DIR, prefix=PLOT_FILE_PREFIX,
)
plots.fig4_post_pruning_distributions(
    snap_trained, snap_masked_ft, snap_finetuned, snap_continued,
    ecdf_thresholds, NULL_SORTED, N_WEIGHTS, EPOCHS, FINETUNE_EPOCHS, TASK_DESC,
    save=SAVE_PLOT, plot_dir=PLOT_DIR, prefix=PLOT_FILE_PREFIX,
)
print('\nAll plots complete.')
