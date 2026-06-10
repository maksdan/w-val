#!/usr/bin/env python3
"""
Sample-size analysis: test R² vs training set size.

Trains regression MLPs across a log-spaced range of training set sizes and
compares three strategies:
  - Baseline MLP   — standard training, no pruning
  - Pruned + freeze fine-tune — BH-threshold pruning, frozen pruned weights, FT epochs
  - Pruned + full fine-tune   — BH-threshold pruning, all weights free, FT epochs

Pruning uses Benjamini-Hochberg FDR correction (alpha = 0.05) applied to per-weight
p-values under a truncated-Beta null fit, matching the pipeline in run_null_analysis.py.

Run:
    python experiments/run_sample_size.py

Edit the CONFIG section below to choose the dataset/task, sample sizes, and model params.
"""

# ── Path setup ────────────────────────────────────────────────────────────────
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# ── CONFIG ────────────────────────────────────────────────────────────────────
SEED            = 42
INPUT_DIM       = 100        # overridden for real datasets
N_TEST          = 2_000      # ignored for real datasets (use their fixed test split)
HIDDEN_SIZES    = [128, 64]
BATCH_SIZE      = 256
EPOCHS          = 300        # full training epochs
FINETUNE_EPOCHS = 10         # fine-tuning epochs after pruning
LR              = 3e-4

# Dataset/task selection — key into DATASET_CASES in wvalue/datasets.py
# Synthetic options: 'x1_x2', 'linear_sparse', 'quadratic', 'two_products',
#                    'three_way', 'four_way', 'sin_product', 'five_products'
# Real options:      'california_housing', 'diabetes', 'wine_quality', 'concrete',
#                    'abalone', 'power_plant', 'energy_efficiency', 'auto_mpg'
CASE = 'three_way'

BH_ALPHA = 0.05    # FDR level for Benjamini-Hochberg pruning

# Sample size range (clipped to dataset size for real data)
SAMPLE_START = 1_000
SAMPLE_END   = 10_000
N_STEPS      = 10    # number of evenly-spaced sizes between START and END

SAVE_PLOT        = True
PLOT_DIR         = 'results_sample_size'
# ─────────────────────────────────────────────────────────────────────────────

import math, copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import beta as sp_beta

import wvalue.core as wvalue_utils
from wvalue.core import BetaSFLookupTable
from wvalue.utils import set_seed
import wvalue.regression as reg
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
WEIGHT_DECAY = CASE_CFG.get('weight_decay', 0.01)
INPUT_DIM    = CASE_CFG.get('input_dim', INPUT_DIM)
N_WEIGHTS    = HIDDEN_SIZES[0] * INPUT_DIM
NULL_STD     = 1.0 / math.sqrt(N_WEIGHTS)
H0           = HIDDEN_SIZES[0]

PLOT_FILE_PREFIX = f'{CASE}_d{INPUT_DIM}_e{EPOCHS}_s{SEED}'

print(f'Case      : {CASE}  |  {TASK_DESC}')
print(f'Model     : {INPUT_DIM} -> {HIDDEN_SIZES} -> 1  (ReLU)   B = {N_WEIGHTS}')
print(f'Weight wd : {WEIGHT_DECAY}')
print(f'Plot prefix: {PLOT_FILE_PREFIX}')

# ── Sample sizes ──────────────────────────────────────────────────────────────
SAMPLE_SIZES = np.unique(np.round(np.linspace(SAMPLE_START, SAMPLE_END, num=N_STEPS)).astype(int))
print(f'Sample sizes (initial): {SAMPLE_SIZES}')

# ── Load / generate full dataset once ─────────────────────────────────────────
if 'load_data' in CASE_CFG:
    X_train_full, y_train_full, X_test, y_test = CASE_CFG['load_data'](seed=SEED)
    INPUT_DIM   = X_train_full.shape[1]
    N_WEIGHTS   = H0 * INPUT_DIM
    NULL_STD    = 1.0 / math.sqrt(N_WEIGHTS)
    N_TRAIN_MAX = len(X_train_full)
    print(f'Real dataset: N_train_max={N_TRAIN_MAX}  N_test={len(X_test)}  INPUT_DIM={INPUT_DIM}')
else:
    def _make_data_synthetic(n, seed=0):
        return reg.make_data(n, INPUT_DIM, CASE_CFG['make_y'], seed=seed)
    X_test,       y_test       = _make_data_synthetic(N_TEST,  seed=SEED + 1)
    X_train_full, y_train_full = None, None
    N_TRAIN_MAX = int(SAMPLE_SIZES.max())
    print(f'Synthetic: N_train_max={N_TRAIN_MAX}  N_test={N_TEST}  INPUT_DIM={INPUT_DIM}')

test_loader = DataLoader(TensorDataset(X_test, y_test),
                         batch_size=BATCH_SIZE, shuffle=False)

SAMPLE_SIZES = SAMPLE_SIZES[SAMPLE_SIZES <= N_TRAIN_MAX]
print(f'Sample sizes (after clamp): {SAMPLE_SIZES}')

# ── Sample-size sweep ─────────────────────────────────────────────────────────
print(f'\nStarting sample-size sweep across {len(SAMPLE_SIZES)} sizes...')
results = reg.sample_size_sweep(
    case_cfg=CASE_CFG,
    sample_sizes=SAMPLE_SIZES,
    input_dim=INPUT_DIM,
    hidden_sizes=HIDDEN_SIZES,
    epochs=EPOCHS,
    finetune_epochs=FINETUNE_EPOCHS,
    lr=LR,
    weight_decay=WEIGHT_DECAY,
    batch_size=BATCH_SIZE,
    bh_alpha=BH_ALPHA,
    test_loader=test_loader,
    device=device,
    N_WEIGHTS=N_WEIGHTS,
    lookup_table=wvalue_utils.beta_sf_lookup,
    x_train_full=X_train_full,
    y_train_full=y_train_full,
    seed=SEED,
)
print('Sweep complete.')

# ── Print results table ───────────────────────────────────────────────────────
print(f'\n{"n":>8}  {"baseline":>10}  {"freeze":>10}  {"finetune":>10}  {"T_z_bh":>10}  {"kept%":>7}')
print('-' * 65)
for i, n in enumerate(results['n']):
    print(f'{n:>8d}  {results["baseline"][i]:>+10.4f}  {results["freeze"][i]:>+10.4f}  '
          f'{results["finetune"][i]:>+10.4f}  {results["T_z_bh"][i]:>10.2f}  '
          f'{100*results["kept_frac"][i]:>6.1f}%')

# ── Plot: test R² vs training sample size ────────────────────────────────────
ns        = np.array(results['n'])
r2_base   = np.array(results['baseline'])
r2_freeze = np.array(results['freeze'])
r2_ft     = np.array(results['finetune'])

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(ns, r2_base,   color='steelblue',   lw=2, marker='o', ms=6,
        label='Baseline MLP  (no pruning)')
ax.plot(ns, r2_freeze, color='forestgreen', lw=2, marker='s', ms=6,
        label=f'Pruned + freeze FT  ({FINETUNE_EPOCHS} ep)')
ax.plot(ns, r2_ft,     color='darkorange',  lw=2, marker='^', ms=6,
        label=f'Pruned + full FT  ({FINETUNE_EPOCHS} ep)')
ax.axhline(0, color='#999999', lw=0.8, ls=':')
ax.set_xscale('log')
ax.set_xlabel('Training samples  (n)', fontsize=13)
ax.set_ylabel('Test R²', fontsize=13)
ax.set_title(
    f'Sample size vs Test R²  —  {TASK_DESC}\n'
    f'({EPOCHS} training epochs + {FINETUNE_EPOCHS} FT epochs,  BH alpha = {BH_ALPHA})',
    fontsize=12)
ax.legend(fontsize=11, loc='lower right')
ax.grid(True, alpha=0.3, linestyle='--', which='both')
plt.tight_layout()

if SAVE_PLOT:
    os.makedirs(PLOT_DIR, exist_ok=True)
    fpath = os.path.join(PLOT_DIR, f'{PLOT_FILE_PREFIX}_fig6.pdf')
    fig.savefig(fpath, dpi=150, bbox_inches='tight')
    print(f'Saved: {fpath}')

plt.show()
print('\nDone.')
