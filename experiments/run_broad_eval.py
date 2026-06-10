#!/usr/bin/env python3
"""
Multi-dataset classification experiment: baseline, L2, and w-value-filtered MLPs.

Trains MLP classifiers on one or more datasets (MNIST, FashionMNIST, CIFAR-10, SVHN,
UCI datasets, etc.) and compares regularization strategies. Captures weight snapshots
at regular intervals and produces training curves, distribution heatmaps, Q-Q plots,
near-zero weight charts, and training speed comparisons.

Run:
    python experiments/run_broad_eval.py

Edit the CONFIG section below to choose datasets, thresholds, and output options.
"""

# ── Path setup ────────────────────────────────────────────────────────────────
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# ── CONFIG ────────────────────────────────────────────────────────────────────
# Single dataset:   DATASETS = ['MNIST']
#                   DATASETS = ['uci:iris']
# Multiple:         DATASETS = ['MNIST', 'uci:iris', 'uci:wine']
# All UCI:          DATASETS = ['uci:all']
# Everything:       DATASETS = ['MNIST', 'FashionMNIST', 'CIFAR10', 'SVHN', 'uci:all']
DATASETS = ['MNIST']

SEED = 42

# Toggle which methods to run:
RUN_BASELINE = True

RUN_L2 = False
L2_STRENGTH = 1e-4

RUN_WVAL_FILTER = True
# Probability-based thresholds: significance = -ln(p)
# Common choices: 0.90 (permissive), 0.50, 0.25, 0.10 (aggressive)
WVAL_PROB_THRESHOLDS = [0.90, 0.50]
WVAL_UPDATE_INTERVAL = 1   # recompute significance every N epochs

# Override dataset defaults (set to None to use per-dataset defaults)
EPOCHS_OVERRIDE = None      # e.g. 5 for a quick test, 50 for full run
LR_OVERRIDE = None          # e.g. 0.0005
BATCH_SIZE_OVERRIDE = None  # e.g. 128 (UCI datasets only)

# Printing & logging
PRINT_INTERVAL = 10   # print every N epochs

# Saving
SAVE_PLOT = True
PLOT_DIR = 'results_broad_eval'

# Snapshot schedule: N intervals -> N+1 snapshots at 0%, 100/N%, ..., 100%
# N=5 -> 0% / 20% / 40% / 60% / 80% / 100%
SNAPSHOT_N = 5

# Plot verbosity
# SHOW_ALL_DIST_PLOTS = True  -> per-snapshot histogram grids (many figures)
# SHOW_ALL_DIST_PLOTS = False -> intensity summary only
SHOW_ALL_DIST_PLOTS = False

# SHOW_ALL_QQ_PLOTS = True  -> full Q-Q grid (first & last snapshot x all layers)
# SHOW_ALL_QQ_PLOTS = False -> Q-Q evolution (init vs final) only
SHOW_ALL_QQ_PLOTS = False
# ─────────────────────────────────────────────────────────────────────────────

import math
import torch

import wvalue.core as wvalue_utils
from wvalue.core import BetaSFLookupTable
from wvalue.broad_eval import run_broad_eval
from wvalue import plots

# ── Device ────────────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
print(f'Using device: {device}')

# ── Lookup table (speeds up Beta SF computation) ──────────────────────────────
if wvalue_utils.beta_sf_lookup is None:
    wvalue_utils.beta_sf_lookup = BetaSFLookupTable(resolution=50_000)
print(f'Lookup table initialized (resolution={wvalue_utils.beta_sf_lookup.resolution:,})')

# ── Resolve datasets (expand 'uci:all') ──────────────────────────────────────
resolved_datasets = []
for ds in DATASETS:
    if ds == 'uci:all':
        from wvalue.datasets import list_uci_datasets
        resolved_datasets.extend([f'uci:{name}' for name in list_uci_datasets()])
    else:
        resolved_datasets.append(ds)

print(f'\nDatasets to run: {len(resolved_datasets)}')
for ds in resolved_datasets[:10]:
    print(f'  {ds}')
if len(resolved_datasets) > 10:
    print(f'  ... and {len(resolved_datasets) - 10} more')

# Compute significance thresholds
wval_thresholds = [-math.log(p) for p in WVAL_PROB_THRESHOLDS] if RUN_WVAL_FILTER else []
wval_labels     = [f'w<{p:g}' for p in WVAL_PROB_THRESHOLDS] if RUN_WVAL_FILTER else []

if RUN_WVAL_FILTER:
    print(f'\nW-value thresholds:')
    for p, t in zip(WVAL_PROB_THRESHOLDS, wval_thresholds):
        print(f'  p={p:.0%} -> significance = {t:.4f}')

# ── Run experiments ───────────────────────────────────────────────────────────
all_dataset_results, failed_datasets = run_broad_eval(
    datasets=resolved_datasets,
    device=device,
    seed=SEED,
    run_baseline=RUN_BASELINE,
    run_l2=RUN_L2,
    l2_strength=L2_STRENGTH,
    run_wval_filter=RUN_WVAL_FILTER,
    wval_prob_thresholds=WVAL_PROB_THRESHOLDS,
    wval_update_interval=WVAL_UPDATE_INTERVAL,
    epochs_override=EPOCHS_OVERRIDE,
    lr_override=LR_OVERRIDE,
    batch_size_override=BATCH_SIZE_OVERRIDE,
    snapshot_n=SNAPSHOT_N,
    print_interval=PRINT_INTERVAL,
    lookup_table=wvalue_utils.beta_sf_lookup,
)

# ── Summary table ─────────────────────────────────────────────────────────────
method_names = [r['name'] for r in list(all_dataset_results.values())[0]] if all_dataset_results else []

print('='*90)
header = f"{'Dataset':<30}"
for m in method_names:
    header += f' {m:>10}'
print(header)
print('-'*90)
for ds_name, results in all_dataset_results.items():
    row = f"{ds_name:<30}"
    for r in results:
        row += f" {r['final_acc']:>9.2f}%"
    print(row)
print('='*90)

# Show best method per dataset
if len(method_names) > 1:
    print(f'\nBest method per dataset:')
    win_counts = {m: 0 for m in method_names}
    for ds_name, results in all_dataset_results.items():
        best = max(results, key=lambda r: r['final_acc'])
        win_counts[best['name']] += 1
        print(f"  {ds_name:<30} -> {best['name']} ({best['final_acc']:.2f}%)")
    print(f'\nWin counts:')
    for m, c in sorted(win_counts.items(), key=lambda x: -x[1]):
        print(f"  {m:<20} {c} wins")

# ── Plots (per dataset) ───────────────────────────────────────────────────────
for ds_name, results in all_dataset_results.items():
    num_epochs_ds = len(results[0]['train_losses'])
    plots.plot_broad_eval_dataset(
        ds_name, results, num_epochs_ds,
        save=SAVE_PLOT, plot_dir=PLOT_DIR,
        show_all_dist=SHOW_ALL_DIST_PLOTS, show_all_qq=SHOW_ALL_QQ_PLOTS,
        lookup_table=wvalue_utils.beta_sf_lookup,
    )

print('\nAll plots complete.')
