"""
Plotting functions for w-value experiments.

Imported by the experiment scripts in experiments/. Each function produces one
figure and optionally saves it. Modify this file to customise colors, sizes, or
layout without touching the experiment scripts.
"""

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as _cm
from scipy.stats import beta as sp_beta, norm as sp_norm
from scipy.special import erfc

# ── Shared colour palette — edit here to restyle all figures at once ──────────
ECDF_COLOR       = 'darkorchid'
OBS_COLOR        = 'steelblue'
NULL_COLOR       = '#444444'
WVAL_COLOR       = 'darkorange'
BAND_ALPHA       = 0.18
NEAR_ZERO_THRESH = 1e-3

# ── Shared palette for broad-eval figures ─────────────────────────────────────
_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
           '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
QQ_COLORS = ['steelblue', 'crimson']


# ── Private helpers ───────────────────────────────────────────────────────────

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


def _snap_colors(n):
    return [_cm.plasma(i / max(n - 1, 1)) for i in range(n)]


def _count_near_zero(snapshot, thr=NEAR_ZERO_THRESH):
    return int(sum(np.sum(np.abs(d['weights']) <= thr) for d in snapshot.values()))


def _total_weights(snapshot):
    return int(sum(d['weights'].size for d in snapshot.values()))


def _null_hist(null_sorted, x_max=20.0, bins=80):
    """Compute null histogram counts and bin centers from sorted null array."""
    counts, edges = np.histogram(null_sorted, bins=bins, range=(0, x_max), density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return counts, centers


# ── Null-analysis figures (used by run_null_analysis.py) ─────────────────────

def fig1_weight_distributions(snap_init, snap_trained, T_z_bh, T_z_ecdf, N_WEIGHTS, epochs, *,
                               save=True, plot_dir='results', prefix='run',
                               show_null=False, show_beta_fits=False, log_xb=False,
                               fits=None, beta_threshold_quantiles=None):
    """3 × 3 grid: heatmap + weight histogram + x·B histogram for init / trained / pruned."""
    CMAP           = 'RdBu_r'
    _THRESH_COLORS = {1.00: 'crimson'}
    _SNAP_KEYS     = ['init', 'trained']

    snaps  = [snap_init,        snap_trained]
    labels = ['initialization', 'Trained']

    NULL_STD = 1.0 / np.sqrt(N_WEIGHTS)

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
        if show_null:
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
        if log_xb:
            _pos      = z_data[z_data > 0]
            _xb_lo    = max(float(_pos.min()), 1e-3) if len(_pos) else 1e-3
            _xb_bins  = np.logspace(np.log10(_xb_lo), np.log10(z_x_max), 80)
            _xs_z_c   = np.logspace(np.log10(_xb_lo), np.log10(z_x_max), 600)
            ax.hist(_pos, bins=_xb_bins, density=True, color=OBS_COLOR, alpha=0.40, label='observed')
        else:
            _xb_lo    = 0.0
            _xs_z_c   = _xs_z
            ax.hist(z_data, bins=80, range=(0, z_x_max), density=True, color=OBS_COLOR, alpha=0.40, label='observed')
        if show_null:
            _null_z_c = sp_beta.pdf(_xs_z_c / N_WEIGHTS, 0.5, (N_WEIGHTS - 1) / 2.0) / N_WEIGHTS
            ax.plot(_xs_z_c, _null_z_c, color=NULL_COLOR, lw=1.4, ls='--', label='true null')
        if show_beta_fits and fits is not None and beta_threshold_quantiles is not None:
            for q in beta_threshold_quantiles:
                a_fit, b_fit = fits[q][snap_key]
                _curve_z = sp_beta.pdf(_xs_z_c / N_WEIGHTS, a_fit, b_fit) / N_WEIGHTS
                ax.plot(_xs_z_c, _curve_z, color=_THRESH_COLORS.get(q, 'crimson'), lw=1.5,
                        label=f'Beta {int(q*100)}%: a={a_fit:.2f}, b={b_fit:.0f}')
        if snap_key == 'trained':
            ax.axvline(T_z_ecdf, color=ECDF_COLOR, lw=1.8, ls='-', alpha=0.85, zorder=10,
                       label=f'ECDF thresh ({T_z_ecdf:.2f})')
            ax.axvline(T_z_bh, color='forestgreen', lw=1.8, ls='--', alpha=0.85, zorder=11,
                       label=f'BH thresh ({T_z_bh:.2f})')
        ax.set_yscale('log')
        if log_xb:
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
    if show_null:
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
    if len(surviving_z) and log_xb:
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
    if show_null:
        _null_z_p = sp_beta.pdf(_xs_z_p / N_WEIGHTS, 0.5, (N_WEIGHTS - 1) / 2.0) / N_WEIGHTS
        ax.plot(_xs_z_p, _null_z_p, color=NULL_COLOR, lw=1.4, ls='--', label='true null')
    if show_beta_fits and fits is not None and beta_threshold_quantiles is not None:
        for q in beta_threshold_quantiles:
            a_fit, b_fit = fits[q]['trained']
            _curve_z = sp_beta.pdf(_xs_z_p / N_WEIGHTS, a_fit, b_fit) / N_WEIGHTS
            ax.plot(_xs_z_p, _curve_z, color=_THRESH_COLORS.get(q, 'crimson'), lw=1.5,
                    label=f'Beta {int(q*100)}%: a={a_fit:.2f}, b={b_fit:.0f}')
    ax.axvline(T_z_ecdf, color=ECDF_COLOR, lw=1.8, ls='-', alpha=0.85, zorder=10,
               label=f'ECDF thresh ({T_z_ecdf:.2f})')
    ax.axvline(T_z_bh, color='forestgreen', lw=1.8, ls='--', alpha=0.85, zorder=11,
               label=f'BH thresh ({T_z_bh:.2f})')
    ax.set_yscale('log')
    if log_xb and len(surviving_z):
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
    _row_labels = ['Initialization', f'Trained ({epochs} ep)', 'Pruned']
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
    if save:
        os.makedirs(plot_dir, exist_ok=True)
        fpath = os.path.join(plot_dir, f'{prefix}_fig1.pdf')
        fig.savefig(fpath, dpi=150, bbox_inches='tight')
        print(f'Saved: {fpath}')
    plt.show()


def fig2_null_comparison(snap_init, snap_trained, fits, ecdf_fits, T_z_ecdf_all,
                          null_sorted, N_WEIGHTS, epochs, *,
                          is_real_data=False, qq_ci=95,
                          show_beta_fits=False, beta_threshold_quantiles=None,
                          save=True, plot_dir='results', prefix='run'):
    """2 × 5 grid: x·B histogram, -ln(w) histogram, and three Q-Q plots for init / trained."""
    from wvalue.analysis import qq_beta, qq_wvalue

    THRESH_COLORS = {1.00: 'crimson'}

    snaps     = [snap_init,        snap_trained]
    labels    = ['Initialization', 'Trained']
    snap_keys = ['init',            'trained']

    N_COLS = 5

    B_TRUE = (N_WEIGHTS - 1) / 2.0

    DIST_X_MAX = 20.0
    QQ_MAX     = 20.0

    _z_x_max = 8.0
    _xs_z2   = np.linspace(1e-6, _z_x_max, 600)

    _null_counts, _null_centers = _null_hist(null_sorted, x_max=DIST_X_MAX)

    # ECDF threshold projected into w-value significance space
    s_ecdf_wval = -np.log(np.clip(erfc(np.sqrt(T_z_ecdf_all / 2.0)), 1e-300, 1.0))

    fig, axes = plt.subplots(2, N_COLS, figsize=(N_COLS * 3.5, 8))
    fig.suptitle(
        f"Feature learning with {'real' if is_real_data else 'simulated'} data",
        fontsize=24, fontweight='bold', y=0.90, x=0.54)

    for row, (snap, _, snap_key) in enumerate(zip(snaps, labels, snap_keys)):

        # Col 0: x·B histogram + 100% Beta (red) + MSE-threshold Beta (purple)
        ax             = axes[row, 0]
        a_100, b_100   = fits[1.00][snap_key]
        a_ecdf_c, b_ecdf_c = ecdf_fits[snap_key]

        ax.hist(snap['z'].ravel(), bins=80, range=(0, _z_x_max),
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
        exp_sub, obs_sub, lo, hi = qq_beta(snap, _a_qq, _b_qq, N_WEIGHTS)
        ax.fill_between(exp_sub, lo, hi, color=ECDF_COLOR, alpha=BAND_ALPHA,
                        label=f'{qq_ci}% null band')
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
        exp_sub, obs_sub, lo, hi = qq_beta(snap, a_fit, b_fit, N_WEIGHTS)
        ax.fill_between(exp_sub, lo, hi, color=THRESH_COLORS[q], alpha=BAND_ALPHA,
                        label=f'{qq_ci}% null band')
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
        null_sub, obs_sub, lo, hi = qq_wvalue(snap, null_sorted)
        ax.fill_between(null_sub, lo, hi, color=WVAL_COLOR, alpha=BAND_ALPHA,
                        label=f'{qq_ci}% null band')
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
    for row, row_name in enumerate(['Initialization', f'Trained ({epochs} ep)']):
        pos = axes[row, 0].get_position()
        y_center = pos.y0 + 0.5 * pos.height
        x_left = pos.x0 - 0.040
        fig.text(
            x_left, y_center, row_name, transform=fig.transFigure,
            rotation=90, va='center', ha='right', fontsize=15, fontweight='bold', clip_on=False)

    if save:
        os.makedirs(plot_dir, exist_ok=True)
        fpath = os.path.join(plot_dir, f'{prefix}_fig2.pdf')
        fig.savefig(fpath, dpi=150, bbox_inches='tight')
        print(f'Saved: {fpath}')
    plt.show()


def fig3_ecdf_diagnostic(snap_trained, ecdf_thresholds, N_WEIGHTS, signal_cols, input_dim, *,
                          save=True, plot_dir='results', prefix='run'):
    """1 × 2: empirical vs null CDF + MSE(T) curve for all weights and signal columns."""
    _z_grid      = np.linspace(0, 8.0, 800)
    _F_null_grid = sp_beta.cdf(_z_grid / N_WEIGHTS, 0.5, (N_WEIGHTS - 1) / 2.0)

    fig3, axes3 = plt.subplots(1, 2, figsize=(14, 5))
    fig3.suptitle(
        f'Figure 3: ECDF Deviation Threshold  —  Trained\n'
        f'MSE(T) between conditional ECDF and conditional Beta CDF,   T* = argmin MSE',
        fontsize=12, fontweight='bold')

    _sig_subtitle = f'signal cols only  ({len(signal_cols)} of {input_dim})'
    for col, (subset_key, subtitle) in enumerate(
            [('all', 'all weights'), ('signal', _sig_subtitle)]):

        T_z_ecdf, mse_star, T_grid, mse_grid = ecdf_thresholds[subset_key]
        z_all     = snap_trained['z'][:, signal_cols].ravel() if subset_key == 'signal' \
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
    if save:
        os.makedirs(plot_dir, exist_ok=True)
        fpath = os.path.join(plot_dir, f'{prefix}_fig3_ecdf.pdf')
        fig3.savefig(fpath, dpi=150, bbox_inches='tight')
        print(f'Saved: {fpath}')
    plt.show()


def fig4_post_pruning_distributions(snap_trained, snap_masked_ft, snap_finetuned, snap_continued,
                                     ecdf_thresholds, null_sorted, N_WEIGHTS,
                                     epochs, finetune_epochs, task_desc, *,
                                     save=True, plot_dir='results', prefix='run'):
    """4 × 3: heatmap + weight histogram + -ln(w) histogram for four model states."""
    _fig4_snaps  = [snap_trained,  snap_masked_ft,  snap_finetuned,           snap_continued]
    _fig4_labels = [f'Trained ({epochs} ep)',
                    f'Pruned + freeze-mask train ({finetune_epochs} ep)',
                    f'Pruned + full fine-tune ({finetune_epochs} ep)',
                    f'Continued ({epochs + finetune_epochs} ep, no prune)']

    _vmax4    = max(np.abs(s['W']).max() for s in _fig4_snaps)
    _w_max4   = _vmax4 * 1.05
    _z_max4   = 8.0
    _xs_z4    = np.linspace(1e-6, _z_max4, 600)

    # Recompute null histogram from null_sorted
    _null_counts, _null_centers = _null_hist(null_sorted, x_max=20.0)

    fig4, axes4 = plt.subplots(4, 3, figsize=(14, 16))
    fig4.suptitle(
        f'Figure 4: Weight Distribution Comparison  —  {task_desc}',
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
    if save:
        os.makedirs(plot_dir, exist_ok=True)
        fpath = os.path.join(plot_dir, f'{prefix}_fig4.pdf')
        fig4.savefig(fpath, dpi=150, bbox_inches='tight')
        print(f'Saved: {fpath}')
    plt.show()


def fig5_threshold_sweep(T_grid, r2_masked, r2_ft, r2_trained, r2_continued,
                          T_z_bh, snap_trained_z, N_WEIGHTS, epochs, finetune_epochs, *,
                          save=True, plot_dir='results', prefix='run'):
    """R² vs pruning threshold with w-value % and % pruned secondary axes."""
    T_optimal    = T_z_bh
    _b_null_5    = (N_WEIGHTS - 1) / 2.0

    fig5, ax5 = plt.subplots(figsize=(10, 6))

    ax5.plot(T_grid, r2_masked,
             color='steelblue', lw=2.0, marker='o', ms=4,
             label=f'Post-pruning + freeze-mask ({finetune_epochs} ep)')
    ax5.plot(T_grid, r2_ft,
             color='darkorange', lw=2.0, marker='s', ms=4,
             label=f'Post-pruning + full fine-tune ({finetune_epochs} ep)')
    ax5.axhline(r2_trained, color='#444444', lw=1.2, ls='--',
                label=f'Trained ({epochs} ep)')
    ax5.axhline(r2_continued, color='seagreen', lw=1.2, ls='--',
                label=f'Continued ({epochs + finetune_epochs} ep, no prune)')

    r2_opt_masked = float(r2_masked[np.argmin(np.abs(T_grid - T_optimal))])
    r2_opt_ft    = float(r2_ft    [np.argmin(np.abs(T_grid - T_optimal))])
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
    _z_all_sorted_5 = np.sort(snap_trained_z)

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
    if save:
        os.makedirs(plot_dir, exist_ok=True)
        fpath = os.path.join(plot_dir, f'{prefix}_fig5_threshold_sweep.pdf')
        fig5.savefig(fpath, dpi=150)
        print(f'Saved: {fpath}')
    plt.show()


# ── Classification figures (used by run_broad_eval.py) ────────────────────────

def plot_broad_eval_dataset(ds_name, results, num_epochs, *,
                             save=True, plot_dir='results',
                             show_all_dist=False, show_all_qq=False,
                             lookup_table=None):
    """All figures for one dataset: training curves, distribution heatmaps, Q-Q plots,
    near-zero weight bars, and training speed."""
    from wvalue.analysis import generate_null_significance, qq_obs_and_null

    _norm_tag = 'matnorm'

    epochs_range = range(1, num_epochs + 1)
    safe_name    = ds_name.replace(':', '_').replace('/', '_')
    pfx          = f'{safe_name}_{_norm_tag}'

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
    if save:
        os.makedirs(plot_dir, exist_ok=True)
        fig.savefig(os.path.join(plot_dir, f'{pfx}_curves.png'), dpi=150, bbox_inches='tight')
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

        # ── 2a. Per-snapshot histogram grid (gated by show_all_dist) ─────────
        if show_all_dist:
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
            if save:
                fpath = os.path.join(plot_dir, f'{pfx}_{mname}_snapshots.png')
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
        if save:
            fpath = os.path.join(plot_dir, f'{pfx}_{mname}_intensity.png')
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
                beta_sf_lookup_table=lookup_table,
            ))

    # ── 3. Q-Q grid: observed vs null significance (gated by show_all_qq) ────
    if show_all_qq:
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
                row_label = _snap_epoch_label(tkey, num_epochs, is_final=is_final)
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
            if save:
                fpath = os.path.join(plot_dir, f'{pfx}_{mname}_qq.png')
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
                label    = _snap_epoch_label(tkey, num_epochs, is_final=is_final)

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
        if save:
            fpath = os.path.join(plot_dir, f'{pfx}_{mname}_qq_evo.png')
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
    if save:
        fig.savefig(os.path.join(plot_dir, f'{pfx}_near_zero.png'), dpi=150, bbox_inches='tight')
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
    if save:
        fig.savefig(os.path.join(plot_dir, f'{pfx}_speed.png'), dpi=150, bbox_inches='tight')
        print(f'  Saved speed: {pfx}_speed.png')
    plt.show()


# ── Sample-size figure (used by run_sample_size.py) ──────────────────────────

def plot_sample_size_r2(results, task_desc, epochs, finetune_epochs, bh_alpha, *,
                         save=True, plot_dir='results', prefix='run'):
    """Test R² vs training set size for baseline, freeze, and full fine-tune."""
    ns        = np.array(results['n'])
    r2_base   = np.array(results['baseline'])
    r2_freeze = np.array(results['freeze'])
    r2_ft     = np.array(results['finetune'])

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(ns, r2_base,   color='steelblue',   lw=2, marker='o', ms=6,
            label='Baseline MLP  (no pruning)')
    ax.plot(ns, r2_freeze, color='forestgreen', lw=2, marker='s', ms=6,
            label=f'Pruned + freeze FT  ({finetune_epochs} ep)')
    ax.plot(ns, r2_ft,     color='darkorange',  lw=2, marker='^', ms=6,
            label=f'Pruned + full FT  ({finetune_epochs} ep)')
    ax.axhline(0, color='#999999', lw=0.8, ls=':')
    ax.set_xscale('log')
    ax.set_xlabel('Training samples  (n)', fontsize=13)
    ax.set_ylabel('Test R²', fontsize=13)
    ax.set_title(
        f'Sample size vs Test R²  —  {task_desc}\n'
        f'({epochs} training epochs + {finetune_epochs} FT epochs,  BH alpha = {bh_alpha})',
        fontsize=12)
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(True, alpha=0.3, linestyle='--', which='both')
    plt.tight_layout()

    if save:
        os.makedirs(plot_dir, exist_ok=True)
        fpath = os.path.join(plot_dir, f'{prefix}_fig6.pdf')
        fig.savefig(fpath, dpi=150, bbox_inches='tight')
        print(f'Saved: {fpath}')
    plt.show()
