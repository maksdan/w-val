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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats    import beta as sp_beta, norm as sp_norm
from scipy.special  import erfc

import wvalue.core as wvalue_utils
from wvalue.core import BetaSFLookupTable
from wvalue.utils import set_seed
import wvalue.regression as reg
import wvalue.analysis as null_a
from wvalue.datasets import DATASET_CASES

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
ECDF_COLOR = 'darkorchid'
OBS_COLOR  = 'steelblue'
NULL_COLOR = '#444444'

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

# ── Figure 1: 3 × 3  (heatmap + all-weights histograms) ──────────────────────
print('\nPlotting Figure 1...')
CMAP           = 'RdBu_r'
_THRESH_COLORS = {1.00: 'crimson'}
_SNAP_KEYS     = ['init', 'trained']

snaps  = [snap_init,        snap_trained]
labels = ['initialization', 'Trained']

# Shared axis ranges
vmax_shared = max(np.abs(s['W']).max() for s in snaps)
wn_abs_max  = max(np.abs(s['w_normed']).max() for s in snaps) * 1.05
_xs_wn      = np.linspace(-wn_abs_max, wn_abs_max, 600)
_null_wn    = sp_norm.pdf(_xs_wn, 0, NULL_STD)

z_x_max    = 20
_xs_z   = np.linspace(1e-6, z_x_max, 600)
_null_z = sp_beta.pdf(_xs_z / N_WEIGHTS, 0.5, (N_WEIGHTS - 1) / 2.0) / N_WEIGHTS

# Build figure
fig, axes = plt.subplots(3, 3, figsize=(14, 12))
fig.suptitle('Weight Distributions', fontsize=24, fontweight='bold')

for row, (snap, label, snap_key) in enumerate(zip(snaps, labels, _SNAP_KEYS)):
    # Col 0: heatmap
    ax = axes[row, 0]
    im = ax.imshow(snap['W'], aspect='auto', cmap=CMAP,
                   vmin=-vmax_shared, vmax=vmax_shared, interpolation='nearest')
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    ax.set_xlabel('Input feature  j', fontsize=11)
    ax.set_ylabel('Hidden neuron  i', fontsize=11)
    ax.tick_params(labelsize=10)

    # Col 1: normalized-weight histogram — all weights
    data = snap['w_normed'].ravel()
    ax   = axes[row, 1]
    ax.hist(data, bins=60, range=(-wn_abs_max, wn_abs_max),
            weights=np.ones(len(data)) / N_WEIGHTS,
            color=OBS_COLOR, alpha=0.55, label='observed')
    if SHOW_NULL_CURVES:
        _bin_w = 2 * wn_abs_max / 60
        ax.plot(_xs_wn, _null_wn * _bin_w, color=NULL_COLOR, lw=1.4, ls='--', label='null')
    ax.set_xlim(-wn_abs_max, wn_abs_max)
    ax.set_xlabel('Normalized weight', fontsize=11)
    ax.tick_params(labelsize=10)
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3, linestyle='--')

    # Col 2: x·B histogram — all weights (log y-scale)
    z_data = snap['z'].ravel()
    ax     = axes[row, 2]
    if LOG_XB_AXIS:
        _pos      = z_data[z_data > 0]
        _xb_lo    = max(float(_pos.min()), 1e-3) if len(_pos) else 1e-3
        _xb_bins  = np.logspace(np.log10(_xb_lo), np.log10(z_x_max), 80)
        _xs_z_c   = np.logspace(np.log10(_xb_lo), np.log10(z_x_max), 600)
        ax.hist(_pos, bins=_xb_bins, density=True, color=OBS_COLOR, alpha=0.40, label='observed')
    else:
        _xb_lo    = 0.0
        _xs_z_c   = _xs_z
        ax.hist(z_data, bins=80, range=(0, z_x_max), density=True, color=OBS_COLOR, alpha=0.40, label='observed')
    if SHOW_NULL_CURVES:
        _null_z_c = sp_beta.pdf(_xs_z_c / N_WEIGHTS, 0.5, (N_WEIGHTS - 1) / 2.0) / N_WEIGHTS
        ax.plot(_xs_z_c, _null_z_c, color=NULL_COLOR, lw=1.4, ls='--', label='true null')
    if SHOW_BETA_FITS:
        for q in BETA_THRESHOLD_QUANTILES:
            a_fit, b_fit = fits[q][snap_key]
            _curve_z = sp_beta.pdf(_xs_z_c / N_WEIGHTS, a_fit, b_fit) / N_WEIGHTS
            ax.plot(_xs_z_c, _curve_z, color=_THRESH_COLORS[q], lw=1.5,
                    label=f'Beta {int(q*100)}%: a={a_fit:.2f}, b={b_fit:.0f}')
    if snap_key == 'trained':
        T_z_ecdf, _, _, _ = ecdf_thresholds['all']
        ax.axvline(T_z_ecdf, color=ECDF_COLOR, lw=1.8, ls='-', alpha=0.85, zorder=10,
                   label=f'ECDF thresh ({T_z_ecdf:.2f})')
        ax.axvline(T_z_bh, color='forestgreen', lw=1.8, ls='--', alpha=0.85, zorder=11,
                   label=f'BH thresh ({T_z_bh:.2f})')
    ax.set_yscale('log')
    if LOG_XB_AXIS:
        ax.set_xscale('log')
        ax.set_xlim(_xb_lo, z_x_max)
    else:
        ax.set_xlim(0, z_x_max)
    ax.set_ylim(1e-3, None)
    ax.set_xlabel('Squared normalized weight', fontsize=11)
    ax.tick_params(labelsize=10)
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3, linestyle='--', which='both')

# Row 2: Pruned
T_z_prune   = T_z_bh
mask_2d     = snap_trained['z'] >= T_z_prune
W_pruned    = snap_trained['W'] * mask_2d.astype(float)

# Col 0: heatmap of pruned W
ax = axes[2, 0]
im = ax.imshow(W_pruned, aspect='auto', cmap=CMAP,
               vmin=-vmax_shared, vmax=vmax_shared, interpolation='nearest')
fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
ax.set_xlabel('Input feature  j', fontsize=11)
ax.set_ylabel('Hidden neuron  i', fontsize=11)
ax.tick_params(labelsize=10)

# Col 1: normalized-weight histogram — surviving weights
surviving_wn = snap_trained['w_normed'][mask_2d]
ax = axes[2, 1]
if len(surviving_wn):
    ax.hist(surviving_wn, bins=60, range=(-wn_abs_max, wn_abs_max),
            weights=np.ones(len(surviving_wn)) / N_WEIGHTS,
            color=OBS_COLOR, alpha=0.55, label='surviving')
if SHOW_NULL_CURVES:
    _bin_w = 2 * wn_abs_max / 60
    ax.plot(_xs_wn, _null_wn * _bin_w, color=NULL_COLOR, lw=1.4, ls='--', label='null')
ax.set_xlim(-wn_abs_max, wn_abs_max)
ax.set_xlabel('Normalized weight', fontsize=11)
ax.tick_params(labelsize=10)
ax.legend(fontsize=9, loc='upper right')
ax.grid(True, alpha=0.3, linestyle='--')

# Col 2: x·B histogram — surviving weights
surviving_z = snap_trained['z'][mask_2d]
z_hi        = z_x_max
ax = axes[2, 2]
if len(surviving_z) and LOG_XB_AXIS:
    _xb_lo_p   = max(float(surviving_z.min()), 1e-3)
    _xb_bins_p = np.logspace(np.log10(_xb_lo_p), np.log10(z_hi), 80)
    _xs_z_p    = np.logspace(np.log10(_xb_lo_p), np.log10(z_hi), 600)
    ax.hist(surviving_z, bins=_xb_bins_p, density=True, color=OBS_COLOR, alpha=0.40, label='surviving')
elif len(surviving_z):
    _xb_lo_p = 0.0
    _xs_z_p  = np.linspace(1e-6, z_hi, 600)
    ax.hist(surviving_z, bins=80, range=(0, z_hi),
            density=True, color=OBS_COLOR, alpha=0.40, label='surviving')
else:
    _xb_lo_p = 0.0
    _xs_z_p  = np.linspace(1e-6, z_hi, 600)
if SHOW_NULL_CURVES:
    _null_z_p = sp_beta.pdf(_xs_z_p / N_WEIGHTS, 0.5, (N_WEIGHTS - 1) / 2.0) / N_WEIGHTS
    ax.plot(_xs_z_p, _null_z_p, color=NULL_COLOR, lw=1.4, ls='--', label='true null')
if SHOW_BETA_FITS:
    for q in BETA_THRESHOLD_QUANTILES:
        a_fit, b_fit = fits[q]['trained']
        _curve_z = sp_beta.pdf(_xs_z_p / N_WEIGHTS, a_fit, b_fit) / N_WEIGHTS
        ax.plot(_xs_z_p, _curve_z, color=_THRESH_COLORS[q], lw=1.5,
                label=f'Beta {int(q*100)}%: a={a_fit:.2f}, b={b_fit:.0f}')
T_z_ecdf, _, _, _ = ecdf_thresholds['all']
ax.axvline(T_z_ecdf, color=ECDF_COLOR, lw=1.8, ls='-', alpha=0.85, zorder=10,
           label=f'ECDF thresh ({T_z_ecdf:.2f})')
ax.axvline(T_z_bh, color='forestgreen', lw=1.8, ls='--', alpha=0.85, zorder=11,
           label=f'BH thresh ({T_z_bh:.2f})')
ax.set_yscale('log')
if LOG_XB_AXIS and len(surviving_z):
    ax.set_xscale('log')
    ax.set_xlim(_xb_lo_p, z_hi)
else:
    ax.set_xlim(0, z_hi)
ax.set_ylim(1e-3, None)
ax.set_xlabel('Squared normalized weight', fontsize=11)
ax.tick_params(labelsize=10)
ax.legend(fontsize=9, loc='upper right')
ax.grid(True, alpha=0.3, linestyle='--', which='both')

# Shared y-axes
ymax_wn = max(axes[r, 1].get_ylim()[1] for r in range(2))
for r in range(2):
    axes[r, 1].set_ylim(0, ymax_wn)
for r in range(3):
    axes[r, 1].set_ylabel('Fraction of all weights', fontsize=11)

_xlim_col2 = axes[0, 2].get_xlim()
for r in range(3):
    axes[r, 2].set_xlim(_xlim_col2)

ymin_z = min(axes[r, 2].get_ylim()[0] for r in range(3))
ymax_z = max(axes[r, 2].get_ylim()[1] for r in range(3))
for r in range(3):
    axes[r, 2].set_ylim(ymin_z, ymax_z)
for r in range(3):
    axes[r, 2].set_ylabel('Density', fontsize=11)

plt.tight_layout()

# Row labels
_row_labels = ['Initialization', f'Trained ({EPOCHS} ep)', 'Pruned']
for r, rlabel in enumerate(_row_labels):
    ax = axes[r, 0]
    ax.annotate(
        rlabel,
        xy=(0, 0.5), xycoords=ax.transAxes,
        xytext=(-0.38, 0.5), textcoords=ax.transAxes,
        fontsize=15, fontweight='bold', ha='center', va='center',
        rotation=90,
    )

# Column labels
_col_labels = ['Raw weights', 'Weight distribution', 'Squared normalized weights']
for c, clabel in enumerate(_col_labels):
    ax = axes[0, c]
    ax.annotate(
        clabel,
        xy=(0.5, 1), xycoords=ax.transAxes,
        xytext=(0.5, 1.12), textcoords=ax.transAxes,
        fontsize=15, fontweight='bold', ha='center', va='bottom',
    )

fig.subplots_adjust(left=0.12, top=0.88, hspace=0.45, wspace=0.50)
if SAVE_PLOT:
    fpath = os.path.join(PLOT_DIR, f'{PLOT_FILE_PREFIX}_fig1.pdf')
    fig.savefig(fpath, dpi=150, bbox_inches='tight')
    print(f'Saved: {fpath}')
plt.show()

# ── Figure 2: 2 × 5 ──────────────────────────────────────────────────────────
print('Plotting Figure 2...')
BAND_ALPHA = 0.18
WVAL_COLOR = 'darkorange'

THRESH_COLORS = {1.00: 'crimson'}

snaps     = [snap_init,        snap_trained]
labels    = ['Initialization', 'Trained']
snap_keys = ['init',            'trained']

N_COLS = 5

A_TRUE = 0.5
B_TRUE = (N_WEIGHTS - 1) / 2.0

DIST_X_MAX = 20.0
QQ_MAX     = 20.0

_z_x_max = 8.0
_xs_z2   = np.linspace(1e-6, _z_x_max, 600)

_null_counts, _null_edges = np.histogram(
    NULL_SORTED, bins=80, range=(0, DIST_X_MAX), density=True)
_null_centers = 0.5 * (_null_edges[:-1] + _null_edges[1:])

# ECDF threshold projected into w-value significance space
s_ecdf_wval   = -np.log(np.clip(erfc(np.sqrt(T_z_ecdf_all / 2.0)), 1e-300, 1.0))

fig, axes = plt.subplots(2, N_COLS, figsize=(N_COLS * 3.5, 8))
fig.suptitle(
    f"Feature learning with {'real' if 'load_data' in CASE_CFG else 'simulated'} data",
    fontsize=24, fontweight='bold', y=0.90, x=0.54)

for row, (snap, _, snap_key) in enumerate(zip(snaps, labels, snap_keys)):
    x_data = snap['x'].ravel()

    # Col 0: x·B histogram + 100% Beta (red) + MSE-threshold Beta (purple)
    ax             = axes[row, 0]
    a_100, b_100   = fits[1.00][snap_key]
    a_ecdf_c, b_ecdf_c = ecdf_fits[snap_key]
    z_data_all     = snap['z'].ravel()

    ax.hist(z_data_all, bins=80, range=(0, _z_x_max),
            density=True, color=OBS_COLOR, alpha=0.40, label='observed', zorder=1)

    _curve_100  = sp_beta.pdf(_xs_z2 / N_WEIGHTS, a_100,    b_100)    / N_WEIGHTS
    _curve_ecdf = sp_beta.pdf(_xs_z2 / N_WEIGHTS, a_ecdf_c, b_ecdf_c) / N_WEIGHTS
    ax.plot(_xs_z2, _curve_100,  color=THRESH_COLORS[1.00], lw=2.0, alpha=0.80,
            label=f'Beta 100%: a={a_100:.3f}, b={b_100:.0f}', zorder=3)
    ax.plot(_xs_z2, _curve_ecdf, color=ECDF_COLOR,           lw=2.0, alpha=0.80,
            label=f'Beta MSE:  a={a_ecdf_c:.3f}, b={b_ecdf_c:.0f}', zorder=3)

    ax.set_yscale('log')
    ax.set_xlim(0, _z_x_max)
    ax.set_ylim(1e-3, None)
    ax.set_xlabel('Squared normalized weight', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.tick_params(labelsize=10)
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3, linestyle='--', which='both')

    # Col 1: -ln(w-value) histogram
    ax       = axes[row, 1]
    sig_data = snap['sig'].ravel()

    ax.hist(sig_data, bins=60, range=(0, DIST_X_MAX),
            density=True, color=WVAL_COLOR, alpha=0.55, label='observed')
    ax.plot(_null_centers, _null_counts, color=NULL_COLOR,
            lw=1.0, ls='--', alpha=0.35, label='null')

    ax.set_xlim(0, DIST_X_MAX)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel('-ln(w-value)', fontsize=12)
    ax.tick_params(labelsize=10)
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3, linestyle='--')

    # Col 2: Q-Q for MSE-threshold Beta fit
    ax = axes[row, 2]
    _a_qq, _b_qq = (0.5, B_TRUE) if snap_key == 'init' else ecdf_fits[snap_key]
    snap['beta_sf_ecdf'] = sp_beta.sf(snap['x'].ravel(), _a_qq, _b_qq)
    exp_sub, obs_sub, lo, hi = null_a.qq_beta(snap, _a_qq, _b_qq, N_WEIGHTS)
    ax.fill_between(exp_sub, lo, hi, color=ECDF_COLOR, alpha=BAND_ALPHA,
                    label=f'{QQ_CI}% null band')
    ax.plot(exp_sub, obs_sub, color=ECDF_COLOR, lw=1.3, marker='o', ms=2, label='observed', zorder=3)
    ax.plot([0, QQ_MAX], [0, QQ_MAX], color=NULL_COLOR, ls='--', lw=0.9,
            label='y = x', zorder=2)
    ax.set_xlim(0, QQ_MAX)
    ax.set_ylim(0, QQ_MAX)
    ax.set_xlabel('Expected', fontsize=12)
    ax.set_ylabel('Observed', fontsize=12)
    ax.tick_params(labelsize=10)
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.3, linestyle='--')

    # Col 3: Q-Q in x·B space, 100% threshold
    q            = 1.00
    a_fit, b_fit = fits[q][snap_key]
    snap['beta_sf_100'] = sp_beta.sf(snap['x'].ravel(), a_fit, b_fit)
    ax           = axes[row, 3]
    exp_sub, obs_sub, lo, hi = null_a.qq_beta(snap, a_fit, b_fit, N_WEIGHTS)
    ax.fill_between(exp_sub, lo, hi, color=THRESH_COLORS[q], alpha=BAND_ALPHA,
                    label=f'{QQ_CI}% null band')
    ax.plot(exp_sub, obs_sub, color=THRESH_COLORS[q], lw=1.3, marker='o', ms=2, label='observed', zorder=3)
    ax.plot([0, QQ_MAX], [0, QQ_MAX], color=NULL_COLOR, ls='--', lw=0.9,
            label='y = x', zorder=2)
    ax.set_xlim(0, QQ_MAX)
    ax.set_ylim(0, QQ_MAX)
    ax.set_xlabel('Expected', fontsize=12)
    ax.set_ylabel('Observed', fontsize=12)
    ax.tick_params(labelsize=10)
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.3, linestyle='--')

    # Col 4: Q-Q in -ln(w-value) space
    ax = axes[row, 4]
    null_sub, obs_sub, lo, hi = null_a.qq_wvalue(snap, NULL_SORTED)
    ax.fill_between(null_sub, lo, hi, color=WVAL_COLOR, alpha=BAND_ALPHA,
                    label=f'{QQ_CI}% null band')
    ax.plot(null_sub, obs_sub, color=WVAL_COLOR, lw=1.3, marker='o', ms=2, label='observed', zorder=3)
    ax.plot([0, QQ_MAX], [0, QQ_MAX], color=NULL_COLOR, ls='--', lw=0.9,
            label='y = x', zorder=2)
    ax.set_xlim(0, QQ_MAX)
    ax.set_ylim(0, QQ_MAX)
    ax.set_xlabel('Expected -ln(w-value)', fontsize=12)
    ax.set_ylabel('Observed -ln(w-value)', fontsize=12)
    ax.tick_params(labelsize=10)
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.3, linestyle='--')

COL_TITLES = [
    "Squared normalized weights",
    "-ln(w-value)",
    "Q-Q - ECDF threshold",
    "Q-Q - 100% threshold",
    "Q-Q - w-value null",
]
for c, t in enumerate(COL_TITLES):
    axes[0, c].set_title(t, fontsize=15, fontweight='bold', pad=12)

plt.tight_layout(rect=[0.10, 0.02, 0.98, 0.90])
for row, row_name in enumerate(['Initialization', f'Trained ({EPOCHS} ep)']):
    pos = axes[row, 0].get_position()
    y_center = pos.y0 + 0.5 * pos.height
    x_left = pos.x0 - 0.040
    fig.text(
        x_left, y_center, row_name, transform=fig.transFigure,
        rotation=90, va='center', ha='right', fontsize=15, fontweight='bold', clip_on=False)

if SAVE_PLOT:
    fpath = os.path.join(PLOT_DIR, f'{PLOT_FILE_PREFIX}_fig2.pdf')
    fig.savefig(fpath, dpi=150, bbox_inches='tight')
    print(f'Saved: {fpath}')
plt.show()

# ── Figure 3: ECDF diagnostic + MSE-based threshold (trained only) ────────────
print('Plotting Figure 3...')
_z_grid      = np.linspace(0, 8.0, 800)
_F_null_grid = sp_beta.cdf(_z_grid / N_WEIGHTS, 0.5, (N_WEIGHTS - 1) / 2.0)

fig3, axes3 = plt.subplots(1, 2, figsize=(14, 5))
fig3.suptitle(
    f'Figure 3: ECDF Deviation Threshold  —  Trained\n'
    f'MSE(T) between conditional ECDF and conditional Beta CDF,   T* = argmin MSE',
    fontsize=12, fontweight='bold')

_sig_subtitle = f'signal cols only  ({len(SIGNAL_COLS)} of {INPUT_DIM})'
for col, (subset_key, subtitle) in enumerate(
        [('all', 'all weights'), ('signal', _sig_subtitle)]):

    T_z_ecdf, mse_star, T_grid, mse_grid = ecdf_thresholds[subset_key]
    z_all     = snap_trained['z'][:, SIGNAL_COLS].ravel() if subset_key == 'signal' \
                else snap_trained['z'].ravel()
    z_sorted  = np.sort(z_all)
    n         = len(z_sorted)
    F_emp_obs = np.arange(1, n + 1) / n
    F_null_obs = sp_beta.cdf(z_sorted / N_WEIGHTS, 0.5, (N_WEIGHTS - 1) / 2.0)

    ax  = axes3[col]
    ax2 = ax.twinx()

    # CDF curves on left axis
    ax.plot(_z_grid, _F_null_grid,
            color=NULL_COLOR, lw=1.5, ls='--', label='Null CDF  Beta(0.5, (B-1)/2)')
    ax.plot(z_sorted, F_emp_obs,
            color=OBS_COLOR, lw=1.2, alpha=0.7, label='Empirical CDF')

    # MSE curve on right axis
    valid = ~np.isnan(mse_grid)
    ax2.plot(T_grid[valid], mse_grid[valid], color=ECDF_COLOR, lw=1.6, alpha=0.85,
             label='MSE(T)')
    ax2.axhline(0, color='gray', lw=0.6, ls=':')
    ax2.set_ylabel('MSE  (cond. ECDF vs cond. Beta CDF)', fontsize=8, color=ECDF_COLOR)
    ax2.tick_params(axis='y', labelcolor=ECDF_COLOR, labelsize=7)
    _mse_ymax = float(np.nanmax(mse_grid)) * 1.4 if valid.any() else 0.1
    ax2.set_ylim(-_mse_ymax * 0.05, _mse_ymax)

    # Chosen threshold T*
    ax.axvline(T_z_ecdf, color=ECDF_COLOR, lw=2.0, ls='-', alpha=0.9,
               label=f'T* = {T_z_ecdf:.3f}  (MSE={mse_star:.5f})')

    ax.set_xlim(0, 8.0)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel('x·B  (= W²/SW² x B)', fontsize=8)
    ax.set_ylabel('Cumulative probability', fontsize=8)
    ax.set_title(f'Trained  —  {subtitle}', fontsize=10, fontweight='bold')
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.3, linestyle='--')

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=6, loc='center right')

plt.tight_layout()
if SAVE_PLOT:
    fpath = os.path.join(PLOT_DIR, f'{PLOT_FILE_PREFIX}_fig3_ecdf.pdf')
    fig3.savefig(fpath, dpi=150, bbox_inches='tight')
    print(f'Saved: {fpath}')
plt.show()

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

# ── Figure 5: R² vs pruning threshold sweep ──────────────────────────────────
print('\nRunning threshold sweep for Figure 5...')
T_grid_5  = np.arange(0.0, 20, 1)
T_optimal = T_z_bh

r2_masked_5, r2_ft_5 = reg.threshold_sweep(
    model, snap_trained, train_loader, test_loader, device,
    T_grid=T_grid_5, finetune_epochs=FINETUNE_EPOCHS, lr=LR, weight_decay=WEIGHT_DECAY,
)
print(f'Sweep complete.  Optimal T_z (BH) = {T_optimal:.4f}')

_b_null_5 = (N_WEIGHTS - 1) / 2.0

fig5, ax5 = plt.subplots(figsize=(10, 6))

ax5.plot(T_grid_5, r2_masked_5,
         color='steelblue', lw=2.0, marker='o', ms=4, label=f'Post-pruning + freeze-mask ({FINETUNE_EPOCHS} ep)')
ax5.plot(T_grid_5, r2_ft_5,
         color='darkorange', lw=2.0, marker='s', ms=4,
         label=f'Post-pruning + full fine-tune ({FINETUNE_EPOCHS} ep)')
ax5.axhline(r2_trained, color='#444444', lw=1.2, ls='--',
            label=f'Trained ({EPOCHS} ep)')
ax5.axhline(r2_continued, color='seagreen', lw=1.2, ls='--',
            label=f'Continued ({EPOCHS + FINETUNE_EPOCHS} ep, no prune)')

r2_opt_masked = float(r2_masked_5[np.argmin(np.abs(T_grid_5 - T_optimal))])
r2_opt_ft    = float(r2_ft_5   [np.argmin(np.abs(T_grid_5 - T_optimal))])
ax5.axvline(T_optimal, color=ECDF_COLOR, lw=2.0, ls='-', alpha=0.3, zorder=10,
            label=f'BH pruning threshold  T_z={T_optimal:.3f}')
ax5.plot(T_optimal, r2_opt_masked, marker='*', ms=14, color=ECDF_COLOR, zorder=11)
ax5.plot(T_optimal, r2_opt_ft,    marker='*', ms=14, color=ECDF_COLOR, zorder=11)

ax5.set_xlabel('Threshold', fontsize=14)
ax5.set_ylabel('R²', fontsize=14)
ax5.set_title(f'R² vs Pruning Threshold', fontsize=16, fontweight='bold')
ax5.legend(fontsize=11, loc='lower right')
ax5.tick_params(labelsize=12)
ax5.grid(True, alpha=0.3, linestyle='--')

# W-value axis at the top
ax5_top = ax5.twiny()
ax5_top.set_xlim(ax5.get_xlim())
for _side in ['bottom', 'right', 'left']:
    ax5_top.spines[_side].set_visible(False)
ax5_top.xaxis.set_ticks_position('top')
ax5_top.xaxis.set_label_position('top')

_wval_pcts = np.array([50, 25, 10, 5, 1, 0.5, 0.1])
_wval_z    = np.array([
    sp_beta.isf(np.clip(p / 100, 1e-15, 1 - 1e-15), 0.5, _b_null_5) * N_WEIGHTS
    for p in _wval_pcts
])
_xlim_top = ax5.get_xlim()
_valid_top = (_wval_z >= _xlim_top[0]) & (_wval_z <= _xlim_top[1])
ax5_top.set_xticks(_wval_z[_valid_top])
ax5_top.set_xticklabels(
    [f'{p:.1f}' if p < 1 else f'{int(p)}' for p in _wval_pcts[_valid_top]],
    fontsize=12)
ax5_top.set_xlabel('w-value  (keep weights where w-value is <= this %)', fontsize=14)

# % pruned axis below bottom
_z_all_sorted_5 = np.sort(snap_trained['z'].ravel())

def _pct_pruned_to_z(pct):
    return np.percentile(_z_all_sorted_5, np.clip(np.asarray(pct, dtype=float), 0.0, 100.0))

ax5_bot = ax5.twiny()
for side in ['top', 'right', 'left']:
    ax5_bot.spines[side].set_visible(False)
ax5_bot.xaxis.set_ticks_position('bottom')
ax5_bot.xaxis.set_label_position('bottom')
ax5_bot.spines['bottom'].set_position(('outward', 48))
ax5_bot.set_xlim(ax5.get_xlim())

_nice_pcts = np.array([75, 90, 95, 99, 99.5])
_nice_z    = _pct_pruned_to_z(_nice_pcts)
_xlim      = ax5.get_xlim()
_valid     = (_nice_z >= _xlim[0]) & (_nice_z <= _xlim[1])
ax5_bot.set_xticks(_nice_z[_valid])
ax5_bot.set_xticklabels([f'{p:.1f}' if p > 99 else f'{int(p)}' for p in _nice_pcts[_valid]], fontsize=12)
ax5_bot.set_xlabel('Weights pruned (%)', fontsize=14)

fig5.subplots_adjust(left=0.09, right=0.97, top=0.87, bottom=0.20)
if SAVE_PLOT:
    fpath = os.path.join(PLOT_DIR, f'{PLOT_FILE_PREFIX}_fig5_threshold_sweep.pdf')
    fig5.savefig(fpath, dpi=150)
    print(f'Saved: {fpath}')
plt.show()

# ── Figure 4: Weight distributions — trained vs pruned vs fine-tuned ──────────
print('Plotting Figure 4...')
_fig4_snaps  = [snap_trained,  snap_masked_ft,  snap_finetuned,           snap_continued]
_fig4_labels = [f'Trained ({EPOCHS} ep)',
                f'Pruned + freeze-mask train ({FINETUNE_EPOCHS} ep)',
                f'Pruned + full fine-tune ({FINETUNE_EPOCHS} ep)',
                f'Continued ({EPOCHS + FINETUNE_EPOCHS} ep, no prune)']

_vmax4    = max(np.abs(s['W']).max() for s in _fig4_snaps)
_wn_max4  = max(np.abs(s['w_normed']).max() for s in _fig4_snaps) * 1.05
_w_max4   = _vmax4 * 1.05
_z_max4   = 8.0
_xs_z4    = np.linspace(1e-6, _z_max4, 600)
_null_z4  = sp_beta.pdf(_xs_z4 / N_WEIGHTS, 0.5, (N_WEIGHTS - 1) / 2.0) / N_WEIGHTS

fig4, axes4 = plt.subplots(4, 3, figsize=(14, 16))
fig4.suptitle(
    f'Figure 4: Weight Distribution Comparison  —  {TASK_DESC}',
    fontsize=13, fontweight='bold')

for row, (snap, label) in enumerate(zip(_fig4_snaps, _fig4_labels)):

    # Col 0: heatmap of raw W1
    ax = axes4[row, 0]
    im = ax.imshow(snap['W'], aspect='auto', cmap='RdBu_r',
                   vmin=-_vmax4, vmax=_vmax4, interpolation='nearest')
    fig4.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    ax.set_title(f'{label}  —  W1 (raw)', fontsize=10, fontweight='bold')
    ax.set_xlabel('Input feature  j', fontsize=8)
    ax.set_ylabel('Hidden neuron  i', fontsize=8)
    ax.tick_params(labelsize=7)

    # Col 1: raw weight histogram (fraction of all weights)
    w_data = snap['W'].ravel()
    ax     = axes4[row, 1]
    ax.hist(w_data, bins=80, range=(-_w_max4, _w_max4),
            weights=np.ones(len(w_data)) / N_WEIGHTS,
            color=OBS_COLOR, alpha=0.55, label='observed')
    ax.set_xlim(-_w_max4, _w_max4)
    ax.set_xlabel('Raw weight  Wij', fontsize=8)
    ax.set_title(f'{label}  —  raw W  (all weights)', fontsize=10, fontweight='bold')
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(True, alpha=0.3, linestyle='--')

    # Col 2: -ln(w-value) significance histogram
    sig_data = snap['sig'].ravel()
    _sig_max = max(s['sig'].max() for s in _fig4_snaps) * 1.05
    ax       = axes4[row, 2]
    ax.hist(sig_data, bins=60, range=(0, _sig_max),
            density=True, color=OBS_COLOR, alpha=0.55, label='observed')
    ax.plot(_null_centers, _null_counts, color=NULL_COLOR,
            lw=1.0, ls='--', alpha=0.35, label='null')
    T_z_ecdf = ecdf_thresholds['all'][0]
    s_thresh = -np.log(np.clip(erfc(np.sqrt(T_z_ecdf / 2.0)), 1e-300, 1.0))
    ax.axvline(s_thresh, color=ECDF_COLOR, lw=1.5, ls='-', alpha=0.85, zorder=10,
               label=f'ECDF thresh ({s_thresh:.2f})')
    ax.set_xlabel('-ln(w-value)', fontsize=8)
    ax.set_title(f'{label}  —  significance (all weights)', fontsize=10, fontweight='bold')
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(True, alpha=0.3, linestyle='--')

# Shared y-axes
ymax_w4 = max(axes4[r, 1].get_ylim()[1] for r in [0, 2, 3])
for r in [0, 2, 3]:
    axes4[r, 1].set_ylim(0, ymax_w4)
for r in range(4):
    axes4[r, 1].set_ylabel('Fraction of all weights', fontsize=8)

ymax_sig4 = max(axes4[r, 2].get_ylim()[1] for r in range(4))
for r in range(4):
    axes4[r, 2].set_ylim(0, ymax_sig4)
    axes4[r, 2].set_ylabel('Density', fontsize=8)

plt.tight_layout()
if SAVE_PLOT:
    fpath = os.path.join(PLOT_DIR, f'{PLOT_FILE_PREFIX}_fig4.pdf')
    fig4.savefig(fpath, dpi=150, bbox_inches='tight')
    print(f'Saved: {fpath}')
plt.show()

print('\nAll plots complete.')
