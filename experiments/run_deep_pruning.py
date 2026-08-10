#!/usr/bin/env python3
"""
Deep pruning with w-values.

Extends the wvalue package (untouched) to prune every hidden Linear layer,
not just the first. The readout layer is never pruned. Three regimes are
compared at a shared per-layer threshold T_z (or each layer's own BH
threshold): simultaneous, sequential (freeze), sequential (unfrozen).

get_prunable_layers is the one seam to swap out for a name-based selector
when moving to transformer / protein-LM MLP sublayers instead of "every
Linear except the last".

Run:
    python experiments/run_deep_pruning.py
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# ── CONFIG ────────────────────────────────────────────────────────────────────
SEED           = 0

OPENML_IDS     = [37, 44, 50, 31, 15, 29, 1462, 1464, 1480, 1494,
                   1063, 1068, 1049, 1050, 1067, 23, 1487, 1492, 1510, 40994,
                   40975, 1497, 1485, 1478, 1496, 1489, 40701, 40670, 1043, 1056]
N_DATASETS     = 12
MAX_ROWS       = 4_000

HIDDEN_SIZES   = [64, 32]
EPOCHS         = 80
FT_EPOCHS      = 10
LR             = 1e-3
BATCH_SIZE     = 256

T_GRID         = tuple(round(t, 2) for t in
                        [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0])
BH_ALPHA       = 0.05

SAVE_RESULTS   = True
RESULTS_DIR    = 'results_deep_pruning'

RUN_SYNTHETIC  = True
RUN_TABULAR    = True

SYNTHETIC_CASES     = ['three_way', 'two_products', 'four_way', 'quadratic',
                       'staircase', 'sumsq3', 'twoblock']
SYNTHETIC_INPUT_DIM = 20
SYNTHETIC_HIDDEN    = [64, 32]
SYNTHETIC_N_TRAIN   = 8_000
SYNTHETIC_N_TEST    = 1_000
SYNTHETIC_EPOCHS    = 150
SYNTHETIC_FT_EPOCHS = 10
SYNTHETIC_LR        = 3e-4
SYNTHETIC_WD        = 1e-2
SYNTHETIC_T_GRID    = (0.5, 1.0, 2.0, 4.0, 8.0)
# ─────────────────────────────────────────────────────────────────────────────

import copy
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import beta as sp_beta

import wvalue.core as wvalue_core
from wvalue.core import BetaSFLookupTable
from wvalue.analysis import select_threshold_mse, fit_beta
from wvalue.datasets import DATASET_CASES
from wvalue.utils import set_seed
import wvalue.regression as reg

if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
print(f'Using device: {device}')

wvalue_core.beta_sf_lookup = BetaSFLookupTable(resolution=50_000)


# ── Deep-pruning primitives ────────────────────────────────────────────────

def get_prunable_layers(model, exclude_last=True):
    """All nn.Linear layers in `model`. Excludes the readout by default."""
    lins = [m for m in model.modules() if isinstance(m, nn.Linear)]
    return lins[:-1] if exclude_last else lins


def layer_significance(W_t):
    """z = (w^2 / sum(w^2)) * num_weights, for an arbitrary Linear layer."""
    W = W_t.detach().cpu().numpy()
    w_sq = W ** 2
    s = w_sq.sum()
    x = w_sq / s if s > 0 else np.full_like(w_sq, 1.0 / w_sq.size)
    return x, x * W.size


def layer_bh_threshold(W_t, alpha=BH_ALPHA):
    """Per-layer Benjamini-Hochberg threshold, in z-space."""
    x, z = layer_significance(W_t)
    B = W_t.numel()
    T_z_ecdf = select_threshold_mse(z.ravel(), n_weights=B, verbose=False)[0]
    a_fit, b_fit = fit_beta(x, threshold=T_z_ecdf / B, n_weights=B, verbose=False)
    pvals = sp_beta.sf(x.ravel(), a_fit, b_fit)
    m = len(pvals)
    sorted_p = np.sort(pvals)
    bh_line = (np.arange(1, m + 1) / m) * alpha
    passing = sorted_p <= bh_line
    if not passing.any():
        return np.inf
    k_bh = int(np.where(passing)[0][-1])
    T_x_bh = float(sp_beta.isf(np.clip(bh_line[k_bh], 1e-300, 1 - 1e-15), a_fit, b_fit))
    return T_x_bh * B


def keep_mask(W_t, T_z):
    _, z = layer_significance(W_t)
    return torch.tensor(z >= T_z, dtype=torch.bool)


def resolve_layer_thresholds(layers, T_z, bh=False, alpha=BH_ALPHA):
    if bh:
        return {lin: layer_bh_threshold(lin.weight.data, alpha) for lin in layers}
    return {lin: T_z for lin in layers}


# ── Generic train / fine-tune loop (shared by regression and classification) ─

def iterate_batches(X, y, batch_size, generator=None):
    n = len(X)
    perm = torch.randperm(n, generator=generator)
    for i in range(0, n, batch_size):
        idx = perm[i:i + batch_size]
        yield X[idx], y[idx]


def finetune(model, X, y, criterion, epochs, lr, weight_decay=0.0,
             batch_size=BATCH_SIZE, frozen_masks=None, locked_layers=None):
    """frozen_masks pins positions at zero; locked_layers zeros a layer's
    gradient entirely (used by the sequential regime)."""
    frozen_masks = frozen_masks or {}
    locked_layers = locked_layers or []
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.train()
    for _ in range(epochs):
        for Xb, yb in iterate_batches(X, y, batch_size):
            Xb, yb = Xb.to(device), yb.to(device)
            opt.zero_grad()
            criterion(model(Xb), yb).backward()
            for lin, mask in frozen_masks.items():
                if lin.weight.grad is not None:
                    lin.weight.grad.masked_fill_(mask.to(lin.weight.device), 0.0)
            for lin in locked_layers:
                if lin.weight.grad is not None:
                    lin.weight.grad.zero_()
            opt.step()
            if frozen_masks:
                with torch.no_grad():
                    for lin, mask in frozen_masks.items():
                        lin.weight.masked_fill_(mask.to(lin.weight.device), 0.0)
    return model


# ── Three deep-pruning regimes ────────────────────────────────────────────

def deep_prune_simultaneous(model, X, y, criterion, T_z, ft_epochs, lr,
                            weight_decay=0.0, bh=False, alpha=BH_ALPHA):
    m = copy.deepcopy(model)
    layers = get_prunable_layers(m)
    thresholds = resolve_layer_thresholds(layers, T_z, bh, alpha)
    frozen = {}
    for lin in layers:
        keep = keep_mask(lin.weight.data, thresholds[lin])
        with torch.no_grad():
            lin.weight.data[~keep.to(lin.weight.device)] = 0.0
        frozen[lin] = ~keep
    finetune(m, X, y, criterion, ft_epochs, lr, weight_decay, frozen_masks=frozen)
    sparsities = [round(float(fr.float().mean()), 4) for fr in frozen.values()]
    return m, sparsities


def deep_prune_sequential(model, X, y, criterion, T_z, ft_epochs, lr,
                          weight_decay=0.0, freeze=True, bh=False, alpha=BH_ALPHA):
    """freeze=True keeps pruned positions dead (sequential-freeze).
    freeze=False lets all weights move once a later layer is tuned, so
    pruned connections may regrow (sequential-unfrozen); prune_sparsities
    is measured at prune time, not final."""
    m = copy.deepcopy(model)
    layers = get_prunable_layers(m)
    thresholds = resolve_layer_thresholds(layers, T_z, bh, alpha)
    frozen = {}
    prune_sparsities = []
    for lin in layers:
        keep = keep_mask(lin.weight.data, thresholds[lin])
        with torch.no_grad():
            lin.weight.data[~keep.to(lin.weight.device)] = 0.0
        prune_sparsities.append(round(float((~keep).float().mean()), 4))
        if freeze:
            frozen[lin] = ~keep
            finetune(m, X, y, criterion, ft_epochs, lr, weight_decay, frozen_masks=frozen)
        else:
            finetune(m, X, y, criterion, ft_epochs, lr, weight_decay)
    return m, prune_sparsities


def final_layer_sparsities(model):
    return [round(float((lin.weight.data == 0).float().mean()), 4)
            for lin in get_prunable_layers(model)]


# ── Magnitude-pruning baseline (matched sparsity) ─────────────────────────

def magnitude_keep_mask(W_t, sparsity):
    w = np.abs(W_t.detach().cpu().numpy())
    k = max(1, int(round((1 - sparsity) * w.size)))
    thresh = np.partition(w.ravel(), -k)[-k]
    return torch.tensor(w >= thresh, dtype=torch.bool)


def magnitude_prune(model, X, y, criterion, sparsities, ft_epochs, lr, weight_decay=0.0):
    m = copy.deepcopy(model)
    frozen = {}
    for lin, sp in zip(get_prunable_layers(m), sparsities):
        keep = magnitude_keep_mask(lin.weight.data, sp)
        with torch.no_grad():
            lin.weight.data[~keep.to(lin.weight.device)] = 0.0
        frozen[lin] = ~keep
    finetune(m, X, y, criterion, ft_epochs, lr, weight_decay, frozen_masks=frozen)
    return m


# ── Tabular data + model ───────────────────────────────────────────────────

class TabMLP(nn.Module):
    def __init__(self, d_in, hidden, d_out):
        super().__init__()
        layers, prev = [], d_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, d_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def load_openml_binary(did, max_rows=MAX_ROWS, seed=0):
    """Returns None if the dataset doesn't fit the shape/class constraints."""
    from sklearn.datasets import fetch_openml
    from sklearn.preprocessing import LabelEncoder

    ds = fetch_openml(data_id=did, as_frame=True)
    df = ds.frame.dropna()
    if not (100 <= df.shape[0] <= 50_000):
        return None
    X = df[ds.feature_names].select_dtypes(include=[np.number])
    if X.shape[1] < 3:
        return None
    y = LabelEncoder().fit_transform(df[ds.target_names[0]].astype(str))
    if len(np.unique(y)) != 2:
        return None
    Xv = X.to_numpy(dtype=np.float32)
    if len(Xv) > max_rows:
        idx = np.random.RandomState(seed).choice(len(Xv), max_rows, replace=False)
        Xv, y = Xv[idx], y[idx]
    return Xv, y.astype(np.int64)


@torch.no_grad()
def accuracy(model, X, y):
    model.eval()
    out = model(torch.tensor(X, device=device)).argmax(1).cpu().numpy()
    return float((out == y).mean())


# ── Synthetic sanity check ─────────────────────────────────────────────────

EXTRA_SYNTHETIC_CASES = {
    'staircase': {
        'desc': 'y = x1 + x1*x2 + x1*x2*x3',
        'make_y': lambda X: X[:, 0] + X[:, 0] * X[:, 1] + X[:, 0] * X[:, 1] * X[:, 2],
        'signal_cols': [0, 1, 2],
    },
    'sumsq3': {
        'desc': 'y = (x1 + x2 + x3)^2',
        'make_y': lambda X: (X[:, 0] + X[:, 1] + X[:, 2]) ** 2,
        'signal_cols': [0, 1, 2],
    },
    'twoblock': {
        'desc': 'y = (x1 + x2)^2 + (x3 + x4)^2',
        'make_y': lambda X: (X[:, 0] + X[:, 1]) ** 2 + (X[:, 2] + X[:, 3]) ** 2,
        'signal_cols': [0, 1, 2, 3],
    },
}


def get_synthetic_case(name):
    if name in EXTRA_SYNTHETIC_CASES:
        return EXTRA_SYNTHETIC_CASES[name]
    return DATASET_CASES[name]


def run_synthetic_case(name, input_dim=SYNTHETIC_INPUT_DIM, hidden_sizes=SYNTHETIC_HIDDEN,
                       n_train=SYNTHETIC_N_TRAIN, n_test=SYNTHETIC_N_TEST,
                       epochs=SYNTHETIC_EPOCHS, ft_epochs=SYNTHETIC_FT_EPOCHS,
                       lr=SYNTHETIC_LR, weight_decay=SYNTHETIC_WD,
                       T_grid=SYNTHETIC_T_GRID, alpha=BH_ALPHA, seed=SEED):
    case = get_synthetic_case(name)
    Xtr, ytr = reg.make_data(n_train, input_dim, case['make_y'], seed=seed)
    Xte, yte = reg.make_data(n_test, input_dim, case['make_y'], seed=seed + 1)
    from torch.utils.data import DataLoader, TensorDataset
    test_loader = DataLoader(TensorDataset(Xte, yte), batch_size=BATCH_SIZE, shuffle=False)

    model = reg.make_model(input_dim, list(hidden_sizes), seed, device)
    criterion = nn.MSELoss()
    finetune(model, Xtr, ytr, criterion, epochs, lr, weight_decay)
    _, r2_dense = reg.evaluate(model, test_loader, device)

    row = {'name': name, 'desc': case['desc'], 'r2_dense': r2_dense,
          'T_grid': list(T_grid), 'pruned_pct': [], 'simul': [], 'seq_freeze': [], 'seq_unfrozen': []}
    for T_z in T_grid:
        m_sim, sp_sim = deep_prune_simultaneous(model, Xtr, ytr, criterion, T_z, ft_epochs, lr, weight_decay)
        m_frz, _ = deep_prune_sequential(model, Xtr, ytr, criterion, T_z, ft_epochs, lr, weight_decay, freeze=True)
        m_unf, _ = deep_prune_sequential(model, Xtr, ytr, criterion, T_z, ft_epochs, lr, weight_decay, freeze=False)
        row['pruned_pct'].append(100.0 * float(np.mean(sp_sim)))
        row['simul'].append(reg.evaluate(m_sim, test_loader, device)[1])
        row['seq_freeze'].append(reg.evaluate(m_frz, test_loader, device)[1])
        row['seq_unfrozen'].append(reg.evaluate(m_unf, test_loader, device)[1])

    bh_layers = get_prunable_layers(model)
    row['bh_Tz'] = [round(t, 3) if np.isfinite(t) else None
                    for t in (layer_bh_threshold(lin.weight.data, alpha) for lin in bh_layers)]
    m_bh, sp_bh = deep_prune_simultaneous(model, Xtr, ytr, criterion, T_z=0.0, ft_epochs=ft_epochs,
                                          lr=lr, weight_decay=weight_decay, bh=True, alpha=alpha)
    row['bh_pruned_pct'] = 100.0 * float(np.mean(sp_bh))
    row['bh_r2'] = reg.evaluate(m_bh, test_loader, device)[1]
    return row


def run_synthetic_sweep(cases=SYNTHETIC_CASES):
    results = {}
    for name in cases:
        print(f'\n{name}: {get_synthetic_case(name)["desc"]}')
        row = run_synthetic_case(name)
        results[name] = row
        print(f'  dense R2={row["r2_dense"]:+.4f}  BH pruned={row["bh_pruned_pct"]:.1f}%  BH R2={row["bh_r2"]:+.4f}')
        for T_z, pct, s, f, u in zip(row['T_grid'], row['pruned_pct'], row['simul'], row['seq_freeze'], row['seq_unfrozen']):
            print(f'    T_z={T_z:5.2f}  pruned={pct:5.1f}%  simul={s:+.4f}  freeze={f:+.4f}  unfrozen={u:+.4f}')
    return results


def plot_synthetic_results(results, save=SAVE_RESULTS, plot_dir=RESULTS_DIR):
    if not results:
        return
    import matplotlib
    if save:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    names = list(results.keys())
    ncol = 4
    nrow = int(np.ceil(len(names) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, 3.5 * nrow), squeeze=False)
    style = {
        'simul':        ('simultaneous', 'tab:blue', 'o-'),
        'seq_freeze':   ('sequential (freeze)', 'tab:orange', 's--'),
        'seq_unfrozen': ('sequential (unfrozen)', 'tab:green', '^-.'),
    }
    for i, name in enumerate(names):
        ax = axes[i // ncol][i % ncol]
        row = results[name]
        Tz = row['T_grid']
        ax.axhline(row['r2_dense'], color='k', ls=':', lw=1, label='dense')
        for key, (lbl, c, st) in style.items():
            ax.plot(Tz, row[key], st, color=c, label=lbl, ms=4)
        ax.set_title(f'{name}\n{row["desc"]}', fontsize=8)
        ax.set_xlabel('per-layer threshold T_z')
        ax.set_ylabel('test R2')
        ax.set_ylim(-0.2, 1.05)
        ax.legend(fontsize=6)
    for j in range(len(names), nrow * ncol):
        axes[j // ncol][j % ncol].axis('off')
    fig.tight_layout()

    if save:
        os.makedirs(plot_dir, exist_ok=True)
        out_path = os.path.join(plot_dir, 'deep_pruning_synthetic_sweep.png')
        fig.savefig(out_path, dpi=150)
        print(f'\nSaved figure -> {out_path}')
    else:
        plt.show()


# ── Tabular experiment driver ──────────────────────────────────────────────

def run_tabular_dataset(did, hidden_sizes=HIDDEN_SIZES, epochs=EPOCHS, ft_epochs=FT_EPOCHS,
                        lr=LR, T_grid=T_GRID, alpha=BH_ALPHA, seed=SEED):
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    data = load_openml_binary(did)
    if data is None:
        return None
    X, y = data
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
    sc = StandardScaler().fit(Xtr)
    Xtr = sc.transform(Xtr).astype(np.float32)
    Xte = sc.transform(Xte).astype(np.float32)

    set_seed(seed)
    base = TabMLP(Xtr.shape[1], hidden_sizes, 2).to(device)
    criterion = nn.CrossEntropyLoss()
    Xtr_t = torch.tensor(Xtr, device=device)
    ytr_t = torch.tensor(ytr, device=device)
    finetune(base, Xtr_t, ytr_t, criterion, epochs, lr, weight_decay=0.0)

    dense_acc = accuracy(base, Xte, yte)
    majority_acc = float(np.max(np.bincount(ytr)) / len(ytr))
    layer_B = [int(l.weight.numel()) for l in get_prunable_layers(base)]

    row = {
        'did': did, 'd_in': Xtr.shape[1], 'dense': dense_acc, 'majority': majority_acc,
        'layer_B': layer_B, 'T_grid': list(T_grid),
        'simul': [], 'seq_freeze': [], 'seq_unfrozen': [],
        'pruned_pct': [], 'magnitude': [],
    }

    for T_z in T_grid:
        m_sim, sp_sim = deep_prune_simultaneous(base, Xtr_t, ytr_t, criterion, T_z, ft_epochs, lr)
        m_frz, _ = deep_prune_sequential(base, Xtr_t, ytr_t, criterion, T_z, ft_epochs, lr, freeze=True)
        m_unf, sp_unf_prune = deep_prune_sequential(base, Xtr_t, ytr_t, criterion, T_z, ft_epochs, lr, freeze=False)
        m_mag = magnitude_prune(base, Xtr_t, ytr_t, criterion, sp_sim, ft_epochs, lr)

        pruned_pct = 100.0 * float(np.mean(sp_sim))
        row['simul'].append(accuracy(m_sim, Xte, yte))
        row['seq_freeze'].append(accuracy(m_frz, Xte, yte))
        row['seq_unfrozen'].append(accuracy(m_unf, Xte, yte))
        row['magnitude'].append(accuracy(m_mag, Xte, yte))
        row['pruned_pct'].append(pruned_pct)
        print(f'  T_z={T_z:5.2f}  pruned={pruned_pct:5.1f}%  '
              f'simul={row["simul"][-1]:.3f}  freeze={row["seq_freeze"][-1]:.3f}  '
              f'unfrozen={row["seq_unfrozen"][-1]:.3f}  magnitude={row["magnitude"][-1]:.3f}')

    bh_layers = get_prunable_layers(base)
    bh_Tz = [layer_bh_threshold(lin.weight.data, alpha) for lin in bh_layers]
    m_bh, sp_bh = deep_prune_simultaneous(base, Xtr_t, ytr_t, criterion, T_z=0.0, ft_epochs=ft_epochs, lr=lr, bh=True, alpha=alpha)
    row['bh_Tz'] = [round(t, 3) if np.isfinite(t) else None for t in bh_Tz]
    row['bh_pruned_pct'] = 100.0 * float(np.mean(sp_bh))
    row['bh_acc'] = accuracy(m_bh, Xte, yte)

    return row


def run_tabular_sweep(openml_ids=OPENML_IDS, n_datasets=N_DATASETS):
    results = []
    for did in openml_ids:
        if len(results) >= n_datasets:
            break
        try:
            print(f'openml#{did} ...')
            row = run_tabular_dataset(did)
            if row is None:
                print(f'  skipped (shape/class constraints)')
                continue
            results.append(row)
            print(f'  dense={row["dense"]:.3f}  majority={row["majority"]:.3f}  '
                  f'BH pruned={row["bh_pruned_pct"]:.1f}%  BH acc={row["bh_acc"]:.3f}  '
                  f'loaded={len(results)}/{n_datasets}')
        except Exception as e:
            print(f'  skipped ({type(e).__name__}: {e})')
    return results


def summarize_tabular(results):
    if not results:
        print('No datasets loaded (offline?).')
        return
    header = (f'{"openml":>8}  {"dense":>6}  {"majority":>8}  {"max%pruned":>10}  '
              f'{"simul":>7}  {"freeze":>7}  {"unfroz":>7}  {"magnitude":>9}')
    print('\n' + header)
    print('-' * len(header))
    agg = {k: [] for k in ('dense', 'majority', 'simul', 'freeze', 'unfrozen', 'magnitude')}
    for row in results:
        j = int(np.argmax(row['pruned_pct']))
        s, f, u, mg = row['simul'][j], row['seq_freeze'][j], row['seq_unfrozen'][j], row['magnitude'][j]
        print(f'{row["did"]:>8}  {row["dense"]:>6.3f}  {row["majority"]:>8.3f}  '
              f'{row["pruned_pct"][j]:>9.1f}%  {s:>7.3f}  {f:>7.3f}  {u:>7.3f}  {mg:>9.3f}')
        agg['dense'].append(row['dense']); agg['majority'].append(row['majority'])
        agg['simul'].append(s); agg['freeze'].append(f)
        agg['unfrozen'].append(u); agg['magnitude'].append(mg)
    print('-' * len(header))
    print(f'{"MEAN":>8}  {np.mean(agg["dense"]):>6.3f}  {np.mean(agg["majority"]):>8.3f}  '
          f'{"":>10}  {np.mean(agg["simul"]):>7.3f}  {np.mean(agg["freeze"]):>7.3f}  '
          f'{np.mean(agg["unfrozen"]):>7.3f}  {np.mean(agg["magnitude"]):>9.3f}')


def plot_tabular_results(results, save=SAVE_RESULTS, plot_dir=RESULTS_DIR):
    if not results:
        return
    import matplotlib
    if save:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    ncol = 4
    nrow = int(np.ceil(len(results) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 4.0 * nrow), squeeze=False)
    style = {
        'simul':        ('simultaneous', 'tab:blue', 'o-'),
        'seq_freeze':   ('sequential (freeze)', 'tab:orange', 's--'),
        'seq_unfrozen': ('sequential (unfrozen)', 'tab:green', '^-.'),
        'magnitude':    ('magnitude (matched sparsity)', 'tab:red', 'x:'),
    }
    for i, row in enumerate(results):
        ax = axes[i // ncol][i % ncol]
        Tz = np.array(row['T_grid'])
        ax.axhline(row['dense'], color='k', ls='--', lw=1, label='dense')
        ax.axhline(row['majority'], color='0.5', ls=':', lw=1.2,
                   label=f'majority ({row["majority"]:.2f})')
        for key, (lbl, c, st) in style.items():
            ax.plot(Tz, row[key], st, color=c, label=lbl, ms=3)
        ax.set_title(f'openml#{row["did"]}', fontsize=9)
        ax.set_xlabel('threshold T_z')
        ax.set_ylabel('test accuracy')
        ax.legend(fontsize=6)
    for j in range(len(results), nrow * ncol):
        axes[j // ncol][j % ncol].axis('off')
    fig.tight_layout()

    if save:
        os.makedirs(plot_dir, exist_ok=True)
        out_path = os.path.join(plot_dir, 'deep_pruning_tabular_sweep.png')
        fig.savefig(out_path, dpi=150)
        print(f'\nSaved figure -> {out_path}')
    else:
        plt.show()


if __name__ == '__main__':
    if SAVE_RESULTS:
        os.makedirs(RESULTS_DIR, exist_ok=True)

    if RUN_SYNTHETIC:
        print('=' * 72)
        print(f'Synthetic sanity check: {SYNTHETIC_CASES}')
        print('=' * 72)
        synthetic_results = run_synthetic_sweep()
        plot_synthetic_results(synthetic_results)
        if SAVE_RESULTS:
            out_path = os.path.join(RESULTS_DIR, 'deep_pruning_synthetic_results.json')
            with open(out_path, 'w') as f:
                json.dump(synthetic_results, f, indent=2)
            print(f'Saved results -> {out_path}')

    if RUN_TABULAR:
        print('\n' + '=' * 72)
        print(f'Tabular sweep: up to {N_DATASETS} OpenML datasets '
              f'(hidden={HIDDEN_SIZES}, epochs={EPOCHS}, ft_epochs={FT_EPOCHS})')
        print('=' * 72)
        tabular_results = run_tabular_sweep()
        summarize_tabular(tabular_results)
        plot_tabular_results(tabular_results)
        if SAVE_RESULTS and tabular_results:
            out_path = os.path.join(RESULTS_DIR, 'deep_pruning_tabular_results.json')
            with open(out_path, 'w') as f:
                json.dump(tabular_results, f, indent=2)
            print(f'Saved results -> {out_path}')