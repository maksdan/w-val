"""
Analysis utilities for w-value experiments.

This module merges two analysis sources into one file:

Section 1 — Null distribution analysis (originally from null_analysis.py):
  - build_null_sorted()      — empirical erfc null distribution
  - fit_beta()               — truncated Beta MLE fit
  - select_threshold_mse()   — ECDF-deviation threshold selection
  - bh_threshold()           — Benjamini-Hochberg FDR pruning threshold
  - qq_beta()                — Q-Q data vs fitted Beta null
  - qq_wvalue()              — Q-Q data vs empirical w-value null
  - run_null_analysis()      — complete null analysis pipeline

Section 2 — Weight snapshot capture (originally from snapshots.py):
  - capture_snapshot()           — per-layer weights and significance dict
  - generate_null_significance() — simulate null significance scores
  - qq_obs_and_null()            — Q-Q helper for observed vs null
"""

# ══════════════════════════════════════════════════════════════════════════════
# Section 1 — Null distribution analysis
# ══════════════════════════════════════════════════════════════════════════════

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import beta as sp_beta
from scipy.optimize import minimize
from scipy.special import erfc


def build_null_sorted(h0, input_dim, n_draws=5, seed=0):
    """Build the empirical erfc null distribution of w-value significance scores.

    Draws n_draws random Frobenius-unit weight matrices of shape (h0, input_dim),
    computes -log(erfc(sqrt(B*x/2))) for each weight, and returns the sorted union.
    """
    n_weights = h0 * input_dim
    sigs = []
    for k in range(n_draws):
        gen  = torch.Generator().manual_seed(seed + k)
        rand = torch.randn(h0, input_dim, generator=gen)
        wn   = F.normalize(rand.view(1, -1), p=2, dim=1).numpy().ravel()
        x_   = np.clip(wn ** 2, 1e-300, 1.0).astype(np.float64)
        z_   = np.sqrt(np.clip(n_weights * x_ / 2.0, 0.0, None))
        sigs.append(-np.log(np.clip(erfc(z_), 1e-300, 1.0)))
    return np.sort(np.concatenate(sigs))


def fit_beta(x_data, threshold, n_weights, verbose=False):
    """Fit Beta(a, b) by MLE to weights below threshold (truncated Beta likelihood).

    Log-likelihood: Σ log BetaPDF(xᵢ; a, b) − n · log BetaCDF(T; a, b)
    """
    mask   = (x_data < threshold) if threshold < 1.0 else np.ones(x_data.size, dtype=bool)
    x_bulk = x_data.ravel()[mask.ravel()]
    n      = len(x_bulk)
    a0, b0 = 0.5, (n_weights - 1) / 2.0
    if verbose:
        print(f'  fitting on {n:,} / {x_data.size:,} weights  (threshold={threshold:.6g})')
    if n < max(20, x_data.size // 20):
        if verbose:
            print('  WARNING: too few weights in bulk — returning theoretical null')
        return a0, b0

    def neg_ll(log_params):
        a, b = np.exp(log_params)
        lp   = sp_beta.logpdf(x_bulk, a, b)
        if not np.all(np.isfinite(lp)):
            return 1e15
        ll = lp.sum()
        if threshold < 1.0:
            log_cdf = np.log(np.clip(sp_beta.cdf(threshold, a, b), 1e-300, 1.0))
            if not np.isfinite(log_cdf):
                return 1e15
            ll -= n * log_cdf
        return -ll

    _bounds = [(np.log(0.01), np.log(10.0)),
               (np.log(1.0),  np.log(20.0 * n_weights))]
    res = minimize(neg_ll, [np.log(a0), np.log(b0)], method='L-BFGS-B',
                   bounds=_bounds,
                   options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-8})
    if not res.success and verbose:
        print(f'  NOTE: optimizer did not fully converge ({res.message})')
    return float(np.exp(res.x[0])), float(np.exp(res.x[1]))


def select_threshold_mse(z_data, n_weights, quantiles=None, verbose=False):
    """Choose the ECDF threshold T that minimises MSE between conditional empirical
    and fitted Beta CDFs. Returns (T_z, mse_star, T_grid, mse_grid).
    """
    if quantiles is None:
        quantiles = np.linspace(0.50, 0.99, 30)
    z_arr    = np.asarray(z_data).ravel()
    T_grid   = np.quantile(z_arr, quantiles)
    mse_grid = np.full(len(T_grid), np.nan)

    for i, T_z in enumerate(T_grid):
        z_bulk = z_arr[z_arr <= T_z]
        n      = len(z_bulk)
        if n < max(20, z_arr.size // 20):
            continue
        x_bulk = z_bulk / n_weights
        T_x    = float(T_z / n_weights)
        a_fit, b_fit = fit_beta(np.array(x_bulk), threshold=T_x,
                                n_weights=n_weights, verbose=verbose)
        z_sorted = np.sort(z_bulk)
        F_emp    = np.arange(1, n + 1) / n
        cdf_vals = sp_beta.cdf(z_sorted / n_weights, a_fit, b_fit)
        cdf_at_T = sp_beta.cdf(T_x, a_fit, b_fit)
        if cdf_at_T < 1e-12:
            continue
        mse_grid[i] = float(np.mean((F_emp - cdf_vals / cdf_at_T) ** 2))

    valid = ~np.isnan(mse_grid)
    if not valid.any():
        return float(np.median(z_arr)), np.nan, T_grid, mse_grid
    best_i = int(np.nanargmin(mse_grid))
    return float(T_grid[best_i]), float(mse_grid[best_i]), T_grid, mse_grid


def bh_threshold(snap, n_weights, alpha=0.05):
    """Apply Benjamini-Hochberg FDR correction to per-weight p-values under the
    empirical Beta null. Returns the pruning threshold T_z (inf if no weight passes).
    """
    T_z_ecdf = select_threshold_mse(snap['z'].ravel(), n_weights=n_weights, verbose=False)[0]
    T_x_ecdf = T_z_ecdf / n_weights
    a_fit, b_fit = fit_beta(snap['x'], threshold=T_x_ecdf,
                            n_weights=n_weights, verbose=False)
    pvals    = sp_beta.sf(snap['x'].ravel(), a_fit, b_fit)
    m        = len(pvals)
    sorted_p = pvals[np.argsort(pvals)]
    bh_line  = (np.arange(1, m + 1) / m) * alpha
    passing  = sorted_p <= bh_line
    if passing.any():
        k_bh   = int(np.where(passing)[0][-1])
        T_x_bh = float(sp_beta.isf(np.clip(bh_line[k_bh], 1e-300, 1 - 1e-15), a_fit, b_fit))
        return T_x_bh * n_weights
    return np.inf


def qq_beta(snap, a_fit, b_fit, n_weights, max_pts=500, ci=95):
    """Q-Q data in z = x·B space against a fitted Beta null.
    Returns (exp_sub, obs_sub, lo_band, hi_band).
    """
    obs_all = np.sort(snap['z'].ravel())
    n_obs   = len(obs_all)
    n_pts   = min(n_obs, max_pts)
    idx     = np.round(np.linspace(0, n_obs - 1, n_pts)).astype(int)
    obs_sub = obs_all[idx]
    ks      = idx + 1
    probs   = (ks - 0.5) / n_obs
    exp_sub = sp_beta.ppf(probs, a_fit, b_fit) * n_weights
    alpha   = (100.0 - ci) / 100.0
    lo_prob = sp_beta.ppf(alpha / 2,       ks, n_obs - ks + 1)
    hi_prob = sp_beta.ppf(1 - alpha / 2,   ks, n_obs - ks + 1)
    lo      = sp_beta.ppf(np.clip(lo_prob, 0, 1), a_fit, b_fit) * n_weights
    hi      = sp_beta.ppf(np.clip(hi_prob, 0, 1), a_fit, b_fit) * n_weights
    return exp_sub, obs_sub, lo, hi


def run_null_analysis(
    snap_init,
    snap_trained,
    N_WEIGHTS,
    beta_threshold_quantiles=None,
    signal_cols=None,
    alpha_bh=0.05,
):
    """Run the complete null analysis pipeline on init and trained snapshots.

    Steps:
      1. ECDF-deviation threshold (select_threshold_mse) on trained weights.
      2. Beta MLE fits for each quantile threshold (init + trained).
      3. ECDF-threshold Beta fits + Benjamini-Hochberg pruning threshold.

    Args:
        snap_init:                 Snapshot dict from reg.capture at initialisation.
        snap_trained:              Snapshot dict from reg.capture after training.
        N_WEIGHTS:                 Total first-layer weight count (H0 * INPUT_DIM).
        beta_threshold_quantiles:  List of Beta CDF quantiles to fit; default [1.00].
        signal_cols:               Column indices of known-signal features (optional).
        alpha_bh:                  FDR level for Benjamini-Hochberg; default 0.05.

    Returns dict with keys:
        ecdf_thresholds  – {'all': (T_z, mse, T_grid, mse_grid), 'signal': ...}
        fits             – {q: {'T', 'T_z', 'init': (a,b), 'trained': (a,b)}}
        ecdf_fits        – {'init': (a,b), 'trained': (a,b)}
        T_z_bh           – BH pruning threshold in z = x·B space (np.inf if none pass)
        T_z_ecdf_all     – ECDF threshold for all weights
        T_x_ecdf         – T_z_ecdf_all / N_WEIGHTS
    """
    if beta_threshold_quantiles is None:
        beta_threshold_quantiles = [1.00]

    # ── ECDF-deviation thresholds ────────────────────────────────────────────
    ecdf_thresholds = {}
    subsets = [('all', snap_trained['z'].ravel())]
    if signal_cols is not None:
        subsets.append(('signal', snap_trained['z'][:, signal_cols].ravel()))
    for key, z_data in subsets:
        ecdf_thresholds[key] = select_threshold_mse(z_data, N_WEIGHTS)

    # ── Beta fits for each quantile threshold ────────────────────────────────
    fits = {}
    for q in beta_threshold_quantiles:
        T_q   = 1.0 if q >= 1.0 else float(sp_beta.ppf(q, 0.5, (N_WEIGHTS - 1) / 2))
        T_z_q = T_q * N_WEIGHTS
        a_i, b_i = fit_beta(snap_init['x'],    T_q, N_WEIGHTS)
        a_t, b_t = fit_beta(snap_trained['x'], T_q, N_WEIGHTS)
        fits[q] = {'T': T_q, 'T_z': T_z_q, 'init': (a_i, b_i), 'trained': (a_t, b_t)}

    # ── ECDF-threshold Beta fits ─────────────────────────────────────────────
    T_z_ecdf_all = ecdf_thresholds['all'][0]
    T_x_ecdf     = T_z_ecdf_all / N_WEIGHTS
    ecdf_fits = {}
    for sk, sn in zip(['init', 'trained'], [snap_init, snap_trained]):
        ecdf_fits[sk] = fit_beta(sn['x'], T_x_ecdf, N_WEIGHTS)

    # ── Benjamini-Hochberg correction ────────────────────────────────────────
    _a_bh, _b_bh = ecdf_fits['trained']
    _pvals   = sp_beta.sf(snap_trained['x'].ravel(), _a_bh, _b_bh)
    _m       = len(_pvals)
    _sorted_p = _pvals[np.argsort(_pvals)]
    _bh_line  = (np.arange(1, _m + 1) / _m) * alpha_bh
    _pass     = _sorted_p <= _bh_line
    if _pass.any():
        _k_bh  = int(np.where(_pass)[0][-1])
        T_x_bh = float(sp_beta.isf(
            np.clip(_bh_line[_k_bh], 1e-300, 1 - 1e-15), _a_bh, _b_bh))
        T_z_bh = T_x_bh * N_WEIGHTS
    else:
        T_z_bh = np.inf

    return {
        'ecdf_thresholds': ecdf_thresholds,
        'fits':            fits,
        'ecdf_fits':       ecdf_fits,
        'T_z_bh':          T_z_bh,
        'T_z_ecdf_all':    T_z_ecdf_all,
        'T_x_ecdf':        T_x_ecdf,
    }


def qq_wvalue(snap, null_sorted, max_pts=500, ci=95):
    """Q-Q data in -ln(w-value) space against an empirical null distribution.
    Returns (null_sub, obs_sub, lo_band, hi_band).
    """
    obs_all  = np.sort(snap['sig'].ravel())
    n_obs    = len(obs_all)
    n_pts    = min(n_obs, max_pts)
    idx      = np.round(np.linspace(0, n_obs - 1, n_pts)).astype(int)
    obs_sub  = obs_all[idx]
    null_sub = np.quantile(null_sorted, np.linspace(0, 1, n_pts))
    alpha    = (100.0 - ci) / 100.0
    ks       = np.clip(idx + 1, 1, n_obs)
    lo_prob  = sp_beta.ppf(alpha / 2,       ks, n_obs - ks + 1)
    hi_prob  = sp_beta.ppf(1.0 - alpha / 2, ks, n_obs - ks + 1)
    lo       = np.quantile(null_sorted, np.clip(lo_prob, 0.0, 1.0))
    hi       = np.quantile(null_sorted, np.clip(hi_prob, 0.0, 1.0))
    return null_sub, obs_sub, lo, hi


# ══════════════════════════════════════════════════════════════════════════════
# Section 2 — Weight snapshot capture
# ══════════════════════════════════════════════════════════════════════════════


def capture_snapshot(model, lookup_table=None):
    """Capture per-layer weights and significance for all Linear layers in model.

    Uses Frobenius normalization per layer when computing w-values.

    Returns a dict: layer_label -> {'weights', 'significance', 'shape'}.
    """
    import wvalue.core as _core
    from wvalue.core import compute_w_value

    table     = lookup_table if lookup_table is not None else _core.beta_sf_lookup
    snap      = {}
    layer_idx = 0

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            layer_idx += 1
            w      = module.weight.data.clone().cpu()
            _, sig = compute_w_value(w, use_lookup=True, beta_sf_lookup_table=table)
            snap[f'Layer {layer_idx}'] = {
                'weights':      w.numpy().ravel(),
                'significance': sig.detach().cpu().numpy().ravel(),
                'shape':        tuple(w.shape),
            }
    return snap


def generate_null_significance(layer_shape, beta_sf_lookup_table=None, seed=0):
    """Draw a random weight matrix and return significance scores under the Beta null.

    Uses Frobenius normalization on the simulated layer weight matrix (same as compute_w_value).

    For large B (> 5000) uses the erfc approximation; otherwise uses the lookup table.
    Returns a 1-D numpy array of length out_features * in_features.
    """
    import wvalue.core as _core

    gen = torch.Generator()
    gen.manual_seed(seed)
    out_features, in_features = layer_shape
    n     = out_features * in_features
    table = beta_sf_lookup_table if beta_sf_lookup_table is not None else _core.beta_sf_lookup

    def _sf_approx(x_sq, B):
        x_sq = np.asarray(x_sq, dtype=np.float64)
        z    = np.sqrt(np.clip(B * x_sq / 2.0, 0.0, None))
        return np.clip(erfc(z), 1e-300, 1.0)

    rand_w   = torch.randn(out_features, in_features, generator=gen)
    B        = n
    w_normed = F.normalize(rand_w.view(1, -1), p=2, dim=1).view(out_features, in_features)

    x = np.clip((w_normed ** 2).numpy().astype(np.float64).ravel(), 1e-300, 1.0)

    if B > 5000:
        return -np.log(_sf_approx(x, B))

    beta_param = (B - 1) / 2.0
    if table is not None:
        sf = table.lookup(x, beta_param)
    else:
        from scipy.stats import beta as _beta_dist
        sf = _beta_dist.sf(x, 0.5, beta_param)
    return -np.log(np.clip(sf, 1e-300, 1.0))


def qq_obs_and_null(snap_layer, null_full_sorted, max_pts=500):
    """Return (null_quantiles, obs_sig) subsampled to max_pts, excluding zero weights."""
    w_flat   = snap_layer['weights']
    s_flat   = snap_layer['significance']
    obs_all  = np.sort(s_flat[np.abs(w_flat) > 0])
    n_obs    = len(obs_all)
    if n_obs == 0:
        return np.array([]), np.array([])
    n_pts    = min(n_obs, max_pts)
    idx      = np.round(np.linspace(0, n_obs - 1, n_pts)).astype(int)
    obs_sub  = obs_all[idx]
    null_sub = np.quantile(null_full_sorted, np.linspace(0, 1, n_pts))
    return null_sub, obs_sub
