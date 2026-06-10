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
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as _cm
import torch

import wvalue.core as wvalue_utils
from wvalue.core import BetaSFLookupTable
from wvalue.broad_eval import run_broad_eval
from wvalue.analysis import generate_null_significance, qq_obs_and_null

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

# ── Plots ─────────────────────────────────────────────────────────────────────
_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
           '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
NEAR_ZERO_THRESH = 1e-3
QQ_MAX_POINTS    = 500   # max scatter/line points per snapshot curve (large layers)
QQ_COLORS        = ['steelblue', 'crimson']   # init -> final

_norm_tag = 'matnorm'   # Frobenius per layer (only scheme)


def _snap_colors(n):
    return [_cm.plasma(i / max(n - 1, 1)) for i in range(n)]


def _count_near_zero(snapshot, thr=NEAR_ZERO_THRESH):
    return int(sum(np.sum(np.abs(d['weights']) <= thr) for d in snapshot.values()))


def _total_weights(snapshot):
    return int(sum(d['weights'].size for d in snapshot.values()))


def _safe_fname(model_name):
    return (model_name
            .replace(' ', '_').replace('=', '').replace('%', 'pct').replace('.', 'p'))


def _ordinal(n):
    """Return ordinal string for integer n, e.g. 0 -> '0th', 1 -> '1st', 22 -> '22nd'."""
    if 11 <= (n % 100) <= 13:
        return f'{n}th'
    suffix = ['th', 'st', 'nd', 'rd']
    idx = n % 10
    return f'{n}{suffix[idx] if idx < 4 else "th"}'


def _snap_epoch_label(snap_key, num_epochs, is_final=False):
    """Convert a snapshot percentage key (e.g. '0%', '100%') to an epoch label."""
    if snap_key == '0%':
        ep = 0
    else:
        pct = int(snap_key.rstrip('%'))
        ep  = min(num_epochs, max(1, round(pct / 100 * num_epochs)))
    label = f'{_ordinal(ep)} epoch'
    if is_final:
        label += ' (final)'
    return label


for ds_name, results in all_dataset_results.items():
    num_epochs_ds = len(results[0]['train_losses'])
    epochs_range  = range(1, num_epochs_ds + 1)
    safe_name     = ds_name.replace(':', '_').replace('/', '_')
    pfx           = f'{safe_name}_{_norm_tag}'   # common filename prefix

    snap_keys = list(next(r['snapshots'] for r in results if r.get('snapshots')).keys())
    n_snaps   = len(snap_keys)
    s_colors  = _snap_colors(n_snaps)

    # ── 1. Training curves (2x2) — train on top, test on bottom ──────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(f'{ds_name}: Training Curves', fontsize=15, fontweight='bold')

    for i, r in enumerate(results):
        c   = _COLORS[i % len(_COLORS)]
        lbl = r['name']
        axes[0, 0].plot(epochs_range, r['train_losses'], label=lbl, color=c, lw=1.8, alpha=0.85)
        axes[0, 1].plot(epochs_range, r['train_accs'],   label=lbl, color=c, lw=1.8, alpha=0.85)
        axes[1, 0].plot(epochs_range, r['test_losses'],  label=lbl, color=c, lw=1.8, alpha=0.85)
        axes[1, 1].plot(epochs_range, r['test_accs'],    label=lbl, color=c, lw=1.8, alpha=0.85)

    for ax, title, ylabel in [
        (axes[0, 0], 'Train Loss',     'Loss'),
        (axes[0, 1], 'Train Accuracy', 'Accuracy (%)'),
        (axes[1, 0], 'Test Loss',      'Loss'),
        (axes[1, 1], 'Test Accuracy',  'Accuracy (%)'),
    ]:
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    if SAVE_PLOT:
        os.makedirs(PLOT_DIR, exist_ok=True)
        fig.savefig(os.path.join(PLOT_DIR, f'{pfx}_curves.png'), dpi=150, bbox_inches='tight')
        print(f'  Saved curves: {pfx}_curves.png')
    plt.show()

    # ── 2. Distribution plots ─────────────────────────────────────────────────
    for r in results:
        snaps = r.get('snapshots')
        if not snaps:
            continue
        r_snap_keys = list(snaps.keys())
        n_cols      = len(r_snap_keys)
        layer_names = list(snaps[r_snap_keys[0]].keys())
        n_layers    = len(layer_names)
        thresh      = r.get('threshold')
        mname       = _safe_fname(r['name'])

        # Pre-compute fixed bin edges per layer (shared by both 2a and 2b)
        layer_w_bins = {}
        layer_s_bins = {}
        for lname in layer_names:
            all_w = np.concatenate([snaps[tk][lname]['weights']      for tk in r_snap_keys])
            all_s = np.concatenate([snaps[tk][lname]['significance'] for tk in r_snap_keys])
            w_pad = max((all_w.max() - all_w.min()) * 0.02, 1e-10)
            s_pad = max((all_s.max() - all_s.min()) * 0.02, 1e-10)
            layer_w_bins[lname] = np.linspace(all_w.min() - w_pad, all_w.max() + w_pad, 62)
            layer_s_bins[lname] = np.linspace(all_s.min() - s_pad, all_s.max() + s_pad, 62)

        # ── 2a. Per-snapshot histogram grid (gated by SHOW_ALL_DIST_PLOTS) ───
        if SHOW_ALL_DIST_PLOTS:
            fig, axes = plt.subplots(
                n_layers * 2, n_cols,
                figsize=(max(4 * n_cols, 8), 4.5 * n_layers),
                squeeze=False,
            )
            title_str = f'{ds_name} — {r["name"]}: Weight & Significance by Epoch'
            if thresh is not None:
                title_str += f'\n(dashed line = significance threshold  τ = {thresh:.4f})'
            fig.suptitle(title_str, fontsize=11, fontweight='bold')

            for ti, tkey in enumerate(r_snap_keys):
                axes[0, ti].set_title(tkey, fontsize=10, fontweight='bold')

            for li, lname in enumerate(layer_names):
                for ti, tkey in enumerate(r_snap_keys):
                    w_vals = snaps[tkey][lname]['weights']
                    s_vals = snaps[tkey][lname]['significance']

                    ax_w = axes[li * 2, ti]
                    ax_w.hist(w_vals, bins=layer_w_bins[lname], color='#4878d0', alpha=0.75,
                              density=True, edgecolor='none')
                    ax_w.set_xlabel('Weight value', fontsize=8)
                    ax_w.grid(True, alpha=0.3, linestyle='--')
                    ax_w.tick_params(labelsize=7)

                    ax_s = axes[li * 2 + 1, ti]
                    ax_s.hist(s_vals, bins=layer_s_bins[lname], color='#ee854a', alpha=0.75,
                              density=True, edgecolor='none')
                    if thresh is not None:
                        ax_s.axvline(thresh, color='#222', linestyle='--', lw=0.9,
                                     label=f'τ = {thresh:.3f}')
                        if ti == 0:
                            ax_s.legend(fontsize=7, loc='upper right')
                    ax_s.set_xlabel('Significance  (−ln w)', fontsize=8)
                    ax_s.grid(True, alpha=0.3, linestyle='--')
                    ax_s.tick_params(labelsize=7)

                axes[li * 2,     0].set_ylabel(f'{lname}\nWeight density', fontsize=8)
                axes[li * 2 + 1, 0].set_ylabel(f'{lname}\nSig. density',   fontsize=8)

                for row_offset in [0, 1]:
                    row_axs = [axes[li * 2 + row_offset, ti] for ti in range(n_cols)]
                    y_max   = max(ax.get_ylim()[1] for ax in row_axs)
                    for ax in row_axs:
                        ax.set_ylim(0, y_max)

            plt.tight_layout()
            if SAVE_PLOT:
                fpath = os.path.join(PLOT_DIR, f'{pfx}_{mname}_snapshots.png')
                fig.savefig(fpath, dpi=150, bbox_inches='tight')
                print(f'  Saved snapshots: {os.path.basename(fpath)}')
            plt.show()

        # ── 2b. Intensity summary plots (always shown) ────────────────────────
        x_edges = np.arange(n_cols + 1) - 0.5

        fig, axes = plt.subplots(
            2, n_layers,
            figsize=(max(3 * n_layers, 6), 6),
            squeeze=False,
        )
        title_str = f'{ds_name} — {r["name"]}: Distribution Summary (intensity)'
        if thresh is not None:
            title_str += f'  [τ = {thresh:.4f}]'
        fig.suptitle(title_str, fontsize=11, fontweight='bold')

        for ci, lname in enumerate(layer_names):
            w_bins = layer_w_bins[lname]
            s_bins = layer_s_bins[lname]

            w_density = np.zeros((len(w_bins) - 1, n_cols))
            s_density = np.zeros((len(s_bins) - 1, n_cols))
            for ti, tkey in enumerate(r_snap_keys):
                w_density[:, ti], _ = np.histogram(
                    snaps[tkey][lname]['weights'], bins=w_bins, density=True)
                s_density[:, ti], _ = np.histogram(
                    snaps[tkey][lname]['significance'], bins=s_bins, density=True)

            ax_w = axes[0, ci]
            mesh_w = ax_w.pcolormesh(x_edges, w_bins, w_density, cmap='Blues', shading='flat')
            fig.colorbar(mesh_w, ax=ax_w, label='Density', pad=0.02)
            ax_w.set_xticks(range(n_cols))
            ax_w.set_xticklabels(r_snap_keys, rotation=45, ha='right', fontsize=7)
            ax_w.set_title(lname, fontsize=9, fontweight='bold')
            ax_w.tick_params(labelsize=7)
            if ci == 0:
                ax_w.set_ylabel('Weight value', fontsize=8)

            ax_s = axes[1, ci]
            mesh_s = ax_s.pcolormesh(x_edges, s_bins, s_density, cmap='Oranges', shading='flat')
            fig.colorbar(mesh_s, ax=ax_s, label='Density', pad=0.02)
            ax_s.set_xticks(range(n_cols))
            ax_s.set_xticklabels(r_snap_keys, rotation=45, ha='right', fontsize=7)
            ax_s.set_xlabel('Training progress', fontsize=8)
            ax_s.tick_params(labelsize=7)
            if ci == 0:
                ax_s.set_ylabel('Significance  (−ln w)', fontsize=8)
            if thresh is not None:
                ax_s.axhline(thresh, color='#cc2222', linestyle='--', lw=0.9,
                             label=f'τ={thresh:.3f}')
                if ci == 0:
                    ax_s.legend(fontsize=7, loc='upper right')

        plt.tight_layout()
        if SAVE_PLOT:
            fpath = os.path.join(PLOT_DIR, f'{pfx}_{mname}_intensity.png')
            fig.savefig(fpath, dpi=150, bbox_inches='tight')
            print(f'  Saved intensity: {os.path.basename(fpath)}')
        plt.show()

    # ── Pre-compute null distributions once per dataset ───────────────────────
    _ref_snaps = next(r['snapshots'] for r in results if r.get('snapshots'))
    _ref_keys  = list(_ref_snaps.keys())
    _layer_names_qq = list(_ref_snaps[_ref_keys[0]].keys())

    _null_cache_ds = {}
    for _ln in _layer_names_qq:
        _shape = _ref_snaps[_ref_keys[0]][_ln]['shape']
        if _shape not in _null_cache_ds:
            _null_cache_ds[_shape] = np.sort(generate_null_significance(
                _shape,
                beta_sf_lookup_table=wvalue_utils.beta_sf_lookup,
            ))

    # ── 3. Q-Q grid: observed vs null significance (gated by SHOW_ALL_QQ_PLOTS)
    if SHOW_ALL_QQ_PLOTS:
        for r in results:
            snaps = r.get('snapshots')
            if not snaps:
                continue
            all_snap_keys = list(snaps.keys())
            r_snap_keys   = [all_snap_keys[0], all_snap_keys[-1]]
            layer_names   = list(snaps[r_snap_keys[0]].keys())
            n_rows        = len(r_snap_keys)
            n_cols        = len(layer_names)
            mname         = _safe_fname(r['name'])

            fig, axes = plt.subplots(
                n_rows, n_cols,
                figsize=(4 * n_cols, 4 * n_rows),
                squeeze=False,
            )
            fig.suptitle(
                f'{ds_name} — {r["name"]}: Q-Q  (observed vs null significance)',
                fontsize=11, fontweight='bold',
            )

            for ri, tkey in enumerate(r_snap_keys):
                is_final  = (tkey == r_snap_keys[-1])
                row_label = _snap_epoch_label(tkey, num_epochs_ds, is_final=is_final)
                color     = QQ_COLORS[ri]

                for ci, lname in enumerate(layer_names):
                    ax = axes[ri, ci]

                    null_sub, obs_sub = qq_obs_and_null(
                        snaps[tkey][lname], _null_cache_ds[snaps[tkey][lname]['shape']]
                    )
                    if len(obs_sub) == 0:
                        ax.text(0.5, 0.5, 'all pruned', ha='center', va='center',
                                transform=ax.transAxes, fontsize=8)
                        continue
                    max_val = float(max(null_sub[-1], obs_sub[-1])) * 1.05

                    ax.scatter(null_sub, obs_sub,
                               s=10, alpha=0.65, color=color, edgecolors='none', zorder=3)
                    ax.plot([0, max_val], [0, max_val],
                            color='#333333', linestyle='--', lw=0.9, zorder=2)
                    ax.set_xlim(0, max_val)
                    ax.set_ylim(0, max_val)
                    ax.grid(True, alpha=0.3, linestyle='--')
                    ax.tick_params(labelsize=7)

                    if ri == 0:
                        ax.set_title(lname, fontsize=9, fontweight='bold')
                    if ci == 0:
                        ax.set_ylabel(f'{row_label}\nObserved sig.', fontsize=8)
                    if ri == n_rows - 1:
                        ax.set_xlabel('Expected sig. (null)', fontsize=8)

            plt.tight_layout()
            if SAVE_PLOT:
                fpath = os.path.join(PLOT_DIR, f'{pfx}_{mname}_qq.png')
                fig.savefig(fpath, dpi=150, bbox_inches='tight')
                print(f'  Saved Q-Q: {os.path.basename(fpath)}')
            plt.show()

    # ── 3b. Q-Q: first & last snapshot overlaid per layer (always shown) ──────
    for r in results:
        snaps = r.get('snapshots')
        if not snaps:
            continue
        all_snap_keys = list(snaps.keys())
        r_snap_keys   = [all_snap_keys[0], all_snap_keys[-1]]
        layer_names   = list(snaps[r_snap_keys[0]].keys())
        n_layers      = len(layer_names)
        mname         = _safe_fname(r['name'])

        fig, axes = plt.subplots(
            1, n_layers,
            figsize=(4 * n_layers, 4),
            squeeze=False,
        )
        fig.suptitle(
            f'{ds_name} — {r["name"]}: Q-Q  (init vs final)',
            fontsize=11, fontweight='bold',
        )

        for ci, lname in enumerate(layer_names):
            ax        = axes[0, ci]
            layer_max = 0.0

            for tkey, color in zip(r_snap_keys, QQ_COLORS):
                is_final = (tkey == r_snap_keys[-1])
                label    = _snap_epoch_label(tkey, num_epochs_ds, is_final=is_final)

                null_sub, obs_sub = qq_obs_and_null(
                    snaps[tkey][lname], _null_cache_ds[snaps[tkey][lname]['shape']]
                )
                if len(obs_sub) == 0:
                    continue
                ax.plot(null_sub, obs_sub, color=color, lw=1.2, alpha=0.85,
                        label=label, zorder=3)
                layer_max = max(layer_max, float(null_sub[-1]), float(obs_sub[-1]))

            max_val = layer_max * 1.05
            ax.plot([0, max_val], [0, max_val],
                    color='#333333', linestyle='--', lw=0.9, zorder=2, label='y = x')
            ax.set_xlim(0, max_val)
            ax.set_ylim(0, max_val)
            ax.set_title(lname, fontsize=9, fontweight='bold')
            ax.set_xlabel('Expected sig. (null)', fontsize=8)
            if ci == 0:
                ax.set_ylabel('Observed significance', fontsize=8)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.tick_params(labelsize=7)
            ax.legend(fontsize=7, loc='upper left')

        plt.tight_layout()
        if SAVE_PLOT:
            fpath = os.path.join(PLOT_DIR, f'{pfx}_{mname}_qq_evo.png')
            fig.savefig(fpath, dpi=150, bbox_inches='tight')
            print(f'  Saved Q-Q evolution: {os.path.basename(fpath)}')
        plt.show()

    # ── 4. Near-zero weight % at each snapshot ────────────────────────────────
    snap_results = [r for r in results if r.get('snapshots')]
    valid_names  = [r['name'] for r in snap_results]
    n_models     = len(valid_names)

    group_w = 0.8
    bar_w   = group_w / n_snaps
    offsets = np.linspace(-group_w / 2 + bar_w / 2, group_w / 2 - bar_w / 2, n_snaps)
    x = np.arange(n_models)

    fig, ax = plt.subplots(figsize=(max(8, 2.5 * n_models), 5))
    fig.suptitle(
        f'{ds_name}: Near-Zero Weights  (|w| ≤ {NEAR_ZERO_THRESH:.0e})',
        fontsize=13, fontweight='bold',
    )

    for si, (skey, color, offset) in enumerate(zip(snap_keys, s_colors, offsets)):
        pct_vals = [
            100.0 * _count_near_zero(r['snapshots'][skey]) / max(_total_weights(r['snapshots'][skey]), 1)
            for r in snap_results
        ]
        bars = ax.bar(x + offset, pct_vals, bar_w, label=skey, color=color, alpha=0.85)
        for bar, val in zip(bars, pct_vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f'{val:.1f}%', ha='center', va='bottom', fontsize=6)

    ax.set_xticks(x)
    ax.set_xticklabels(valid_names, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel(f'% of weights with |w| ≤ {NEAR_ZERO_THRESH:.0e}', fontsize=10)
    ax.yaxis.set_major_formatter(plt.matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:.1f}%'))
    ax.legend(fontsize=9, title='Epoch %', title_fontsize=9)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    plt.tight_layout()
    if SAVE_PLOT:
        fig.savefig(os.path.join(PLOT_DIR, f'{pfx}_near_zero.png'), dpi=150, bbox_inches='tight')
        print(f'  Saved near-zero: {pfx}_near_zero.png')
    plt.show()

    # ── 5. Training speed comparison ──────────────────────────────────────────
    times      = [r['time_taken'] for r in results]
    n_epochs_r = [len(r['train_losses']) for r in results]
    tpe        = [t / max(e, 1) for t, e in zip(times, n_epochs_r)]
    names      = [r['name'] for r in results]

    use_minutes = max(times) > 120
    scale, unit = (60, 'min') if use_minutes else (1, 's')
    times_scaled = [t / scale for t in times]
    tpe_scaled   = [t / scale for t in tpe]

    fig, ax = plt.subplots(figsize=(max(6, 1.8 * len(results)), 5))
    fig.suptitle(f'{ds_name}: Training Speed', fontsize=13, fontweight='bold')

    bars = ax.bar(names, times_scaled,
                  color=[_COLORS[i % len(_COLORS)] for i in range(len(results))],
                  alpha=0.85)
    for bar, t, tp in zip(bars, times_scaled, tpe_scaled):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f'{t:.2f}{unit}\n({tp:.3f}{unit}/epoch)',
                ha='center', va='bottom', fontsize=8)

    ax.set_ylabel(f'Total training time ({unit})', fontsize=10)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha='right', fontsize=9)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    plt.tight_layout()
    if SAVE_PLOT:
        fig.savefig(os.path.join(PLOT_DIR, f'{pfx}_speed.png'), dpi=150, bbox_inches='tight')
        print(f'  Saved speed: {pfx}_speed.png')
    plt.show()

print('\nAll plots complete.')
