"""Tabular-data pipeline for paper Figs 3A/3B, 4, 5 and Table 1 (driver: experiments/run_tabular_figs345.py)."""
import os
import math
import copy
import json
import time
import contextlib
import warnings
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.stats import beta as sp_beta, kstest
from sklearn.model_selection import train_test_split

import wvalue.core as wvalue_core
from wvalue.core import BetaSFLookupTable
import wvalue.regression as reg
import wvalue.analysis as null_a
from wvalue import plots
from wvalue.utils import set_seed
from wvalue.datasets import (DATASET_CASES, load_uci_dataset, list_uci_datasets, get_mlp_class,
                             _openml_fetch, _xy_from_openml, _std_split)

warnings.filterwarnings('ignore', category=RuntimeWarning)
NL = chr(10)

@dataclass
class TabularConfig:
    SEED: int = 42
    EPOCHS: int = 100
    FINETUNE_EPOCHS: int = 10
    BATCH_SIZE: int = 64
    LR: float = 3e-4
    WD_REGRESSION: float = 1e-3
    WD_CLASSIFICATION: float = 1e-2
    EVAL_BATCH_SIZE: int = 256
    BH_ALPHA: float = 0.05
    NULL_N_DRAWS: int = 5
    QQ_CI: int = 95
    LOOKUP_RESOLUTION: int = 500_000
    REGIMES: list = field(default_factory=lambda: ['fixed', 'tuned'])
    VAL_FRAC: float = 0.2
    TUNE_WD_GRID: list = field(default_factory=lambda: [1e-4, 1e-3, 1e-2])
    TUNE_MAX_EPOCHS: int = 300
    TUNE_PATIENCE: int = 25
    MIN_D: int = 20
    MIN_NTRAIN: int = 320
    MAX_NTRAIN: Optional[int] = 30_000
    INCLUDE_UCI: bool = True
    INCLUDE_REPO_REGRESSION: bool = True
    INCLUDE_LOW_D_REGRESSION: bool = True
    INCLUDE_OPENML_REGRESSION: bool = True
    SCREEN_SEEDS: list = field(default_factory=lambda: [0, 1, 2, 3, 4])
    TAIL_P: float = 1e-3
    SEL_MIN_BH_RATE: float = 0.6
    SEL_TOP_K: dict = field(default_factory=lambda: {'clf': 4, 'reg': 3})
    MANUAL_CASES: Optional[list] = None
    FULL_SEEDS: list = field(default_factory=lambda: [42])
    FIG5_T_MAX: int = 20
    FIG4_MIN_NTRAIN: int = 1500
    N_SAMPLE_STEPS: int = 10
    REPO_SAMPLE_START: int = 1_000
    REPO_SAMPLE_END: int = 10_000
    ANNOTATE_FIGS: bool = True

    def resume_signature(self):
        keys = ['SEED', 'EPOCHS', 'FINETUNE_EPOCHS', 'BATCH_SIZE', 'LR', 'WD_REGRESSION',
                'WD_CLASSIFICATION', 'BH_ALPHA', 'NULL_N_DRAWS', 'LOOKUP_RESOLUTION', 'VAL_FRAC',
                'TUNE_WD_GRID', 'TUNE_MAX_EPOCHS', 'TUNE_PATIENCE', 'MIN_D', 'MIN_NTRAIN',
                'SCREEN_SEEDS', 'TAIL_P', 'FIG5_T_MAX', 'N_SAMPLE_STEPS']
        d = asdict(self)
        return {k: d[k] for k in keys}

CFG = TabularConfig()
DEVICE = torch.device('cpu')
LOOKUP = None
UCI_DIR = None
REGISTRY = {}
DATA = {}
OPENML_REGRESSION = {
    'pol':          dict(desc='Pole telecommunications (OpenML pol)', d=48,
                         fetch=[dict(name='pol', version=1)]),
    'ailerons':     dict(desc='Ailerons (OpenML)', d=40,
                         fetch=[dict(name='Ailerons', version=1), dict(name='ailerons', version=1)]),
    'cpu_act':      dict(desc='Computer activity (OpenML cpu_act)', d=21,
                         fetch=[dict(name='cpu_act', version=1)]),
    'superconduct': dict(desc='Superconductivity critical temperature (OpenML)', d=81,
                         fetch=[dict(name='superconduct', version=1)]),
}

def pick_device(requested='auto'):
    """Pick cuda or cpu; 'auto' prefers cuda."""
    if requested == 'auto':
        requested = 'cuda' if torch.cuda.is_available() else 'cpu'
    if requested == 'cuda' and not torch.cuda.is_available():
        print('WARNING: --device cuda requested but no GPU found; falling back to CPU.')
        requested = 'cpu'
    dev = torch.device(requested)
    if dev.type != 'cuda':
        print('WARNING: running on CPU. The full pipeline is very slow on CPU; '
              'consider fewer SCREEN_SEEDS / EPOCHS or a GPU.')
    return dev

def configure(cfg=None, device='auto', uci_dir=None):
    """Set the global config, device, UCI folder and lookup table."""
    global CFG, DEVICE, LOOKUP, UCI_DIR
    if cfg is not None:
        CFG = cfg
    DEVICE = device if isinstance(device, torch.device) else pick_device(device)
    UCI_DIR = uci_dir
    wvalue_core.beta_sf_lookup = BetaSFLookupTable(resolution=CFG.LOOKUP_RESOLUTION)
    LOOKUP = wvalue_core.beta_sf_lookup
    DATA.clear()
    return DEVICE

def write_json(obj, path):
    with open(path, 'w') as fh:
        json.dump(obj, fh, indent=1)

def read_json(path):
    with open(path) as fh:
        return json.load(fh)

class RunPaths:
    """Output folders: figures, artifacts, tables."""

    def __init__(self, out_root):
        self.root = out_root
        self.fig = os.path.join(out_root, 'figures')
        self.art = os.path.join(out_root, 'artifacts')
        self.tab = os.path.join(out_root, 'tables')
        for d in (self.fig, self.art, self.tab):
            os.makedirs(d, exist_ok=True)

    def check_config(self, resume=True):
        """Error if the run folder was made with different settings."""
        path = os.path.join(self.root, 'config.json')
        sig = CFG.resume_signature()
        if resume and os.path.exists(path):
            old = read_json(path)
            diff = {k: (old.get(k), v) for k, v in sig.items() if old.get(k) != v}
            if diff:
                raise RuntimeError('Config differs from the one this run folder was created with: '
                                   + str(diff) + '. Use a different --run-name or pass --no-resume.')
        write_json(sig, path)

def auto_hidden(d):
    if d <= 20:
        return [64, 32]
    if d <= 100:
        return [128, 64]
    return [256, 128]

def build_registry():
    """Register UCI, repo and OpenML datasets."""
    REGISTRY.clear()
    if CFG.INCLUDE_UCI:
        if UCI_DIR and os.path.isdir(UCI_DIR):
            for name in list_uci_datasets(UCI_DIR):
                REGISTRY['uci:' + name] = dict(task='clf', source='UCI', desc='UCI ' + name, name=name)
        else:
            print('UCI directory not found:', UCI_DIR, '- UCI datasets skipped. '
                  'Run experiments/download_uci_data.py first.')
    if CFG.INCLUDE_REPO_REGRESSION:
        for name, c in DATASET_CASES.items():
            if 'load_data' in c:
                REGISTRY['repo:' + name] = dict(task='reg', source='repo', desc=c['desc'], name=name,
                                                expected_d=c['input_dim'])
    if CFG.INCLUDE_OPENML_REGRESSION:
        for name, spec in OPENML_REGRESSION.items():
            REGISTRY['openml:' + name] = dict(task='reg', source='OpenML', desc=spec['desc'], name=name,
                                              expected_d=spec['d'])
    return REGISTRY

def get_data(key):
    """Load a dataset (cached) with its tensors, architecture and chance level."""
    if key in DATA:
        return DATA[key]
    if not REGISTRY:
        build_registry()
    E = REGISTRY[key]
    if E['source'] == 'UCI':
        tl, vl, cfg = load_uci_dataset(E['name'], uci_dir=UCI_DIR, batch_size=CFG.BATCH_SIZE)
        Xtr, ytr = tl.dataset.tensors
        Xte, yte = vl.dataset.tensors
        Xtr, Xte, ytr, yte = Xtr.float(), Xte.float(), ytr.long(), yte.long()
        n_classes, hidden = int(cfg['num_classes']), list(cfg['hidden_sizes'])
    else:
        if E['source'] == 'OpenML':
            raw = _openml_fetch(OPENML_REGRESSION[E['name']]['fetch'])
            X, y = _xy_from_openml(raw)
            Xtr, ytr, Xte, yte = _std_split(X, y, seed=CFG.SEED)
        else:
            Xtr, ytr, Xte, yte = DATASET_CASES[E['name']]['load_data'](seed=CFG.SEED)
        Xtr, Xte, ytr, yte = Xtr.float(), Xte.float(), ytr.float(), yte.float()
        if Xtr.shape[1] != E['expected_d']:
            raise ValueError('expected d=' + str(E['expected_d']) + ' but loaded d=' + str(Xtr.shape[1])
                             + ' (wrong dataset fetched)')
        n_classes, hidden = None, auto_hidden(int(Xtr.shape[1]))
    d = int(Xtr.shape[1])
    if E['task'] == 'clf':
        counts = torch.bincount(yte, minlength=n_classes).float()
        chance = float(counts.max() / counts.sum())
    else:
        chance = 0.0
    D = dict(Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte, d=d, n_train=len(Xtr), n_test=len(Xte),
             n_classes=n_classes, hidden=hidden, B=hidden[0] * d, chance=chance,
             wd=(CFG.WD_CLASSIFICATION if E['task'] == 'clf' else CFG.WD_REGRESSION),
             feat_mean=float(Xtr.mean(0).abs().mean()), feat_std=float(Xtr.std(0).mean()))
    DATA[key] = D
    return D

def run_inventory(paths):
    """Load every dataset, save the inventory table, return eligible keys."""
    build_registry()
    rows = []
    for key, E in REGISTRY.items():
        row = dict(dataset=key, task=E['task'], source=E['source'])
        try:
            D = get_data(key)
            row.update(d=D['d'], n_train=D['n_train'], n_test=D['n_test'], n_classes=D['n_classes'],
                       hidden=str(D['hidden']), B=D['B'], chance=round(D['chance'], 3),
                       mean_abs_feat_mean=round(D['feat_mean'], 3), mean_feat_std=round(D['feat_std'], 3))
            reasons = []
            low_d_ok = E['source'] == 'repo' and CFG.INCLUDE_LOW_D_REGRESSION
            if D['d'] <= CFG.MIN_D and not low_d_ok:
                reasons.append('d<=' + str(CFG.MIN_D))
            if D['n_train'] < CFG.MIN_NTRAIN:
                reasons.append('n_train<' + str(CFG.MIN_NTRAIN))
            if CFG.MAX_NTRAIN is not None and D['n_train'] > CFG.MAX_NTRAIN:
                reasons.append('n_train>' + str(CFG.MAX_NTRAIN))
            row.update(low_d=D['d'] <= CFG.MIN_D, eligible=not reasons, reason=', '.join(reasons))
            if reasons:
                DATA.pop(key, None)
        except Exception as e:
            print('  [skip]', key, '-', type(e).__name__, str(e)[:120])
            row.update(eligible=False, reason='load failed: ' + type(e).__name__ + ': ' + str(e)[:150])
        rows.append(row)
    inv = pd.DataFrame(rows).sort_values(['eligible', 'task', 'd'], ascending=[False, True, False])
    inv.to_csv(os.path.join(paths.tab, 'dataset_inventory.csv'), index=False)
    eligible = inv[inv['eligible'] == True].sort_values('n_train')['dataset'].tolist()
    print(len(REGISTRY), 'datasets found,', len(eligible), 'eligible')
    with pd.option_context('display.max_rows', 400, 'display.width', 250):
        print(inv.to_string(index=False))
    return inv, eligible

def eligible_from_inventory(paths):
    """Eligible dataset keys from the saved inventory."""
    p = os.path.join(paths.tab, 'dataset_inventory.csv')
    if not os.path.exists(p):
        return run_inventory(paths)[1]
    build_registry()
    inv = pd.read_csv(p)
    return inv[inv['eligible'] == True].sort_values('n_train')['dataset'].tolist()

class GPULoader:
    """Mini-batch loader that keeps the training set on the device."""

    def __init__(self, X, y, batch_size, seed):
        self.X, self.y = X.to(DEVICE), y.to(DEVICE)
        self.batch_size = int(batch_size)
        self.gen = torch.Generator().manual_seed(int(seed))

    def __len__(self):
        return math.ceil(len(self.X) / self.batch_size)

    def __iter__(self):
        idx = torch.randperm(len(self.X), generator=self.gen).to(self.X.device)
        for i in range(0, len(idx), self.batch_size):
            j = idx[i:i + self.batch_size]
            yield self.X[j], self.y[j]

def eval_loader(X, y):
    return DataLoader(TensorDataset(X, y), batch_size=CFG.EVAL_BATCH_SIZE, shuffle=False)

def make_clf_model(d, hidden, n_classes, seed):
    """Classification MLP with Kaiming-normal weights and zero biases."""
    set_seed(seed)
    m = get_mlp_class()(d, hidden, n_classes).to(DEVICE)
    for layer in m.modules():
        if isinstance(layer, nn.Linear):
            nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
            nn.init.zeros_(layer.bias)
    return m

@torch.no_grad()
def evaluate_classification(model, loader, device):
    """Return (cross-entropy, accuracy) on a loader."""
    model.eval()
    tot_loss, correct, n = 0.0, 0, 0
    for Xb, yb in loader:
        out = model(Xb.to(device)).cpu()
        tot_loss += F.cross_entropy(out, yb, reduction='sum').item()
        correct += (out.argmax(1) == yb).sum().item()
        n += len(yb)
    return tot_loss / n, correct / n

def train_classification(model, train_loader, device, epochs, lr, weight_decay,
                         test_loader=None, print_every=25):
    """Adam training loop with cross-entropy."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.train()
    for epoch in range(1, epochs + 1):
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            criterion(model(Xb), yb).backward()
            optimizer.step()
        if test_loader is not None and print_every and (epoch % print_every == 0 or epoch == 1):
            loss, acc = evaluate_classification(model, test_loader, device)
            print(f'  Epoch {epoch:3d}/{epochs}  ce={loss:.4f}  acc={acc:.4f}')
            model.train()

def _finetune(model, loader, epochs, lr, weight_decay, criterion, freeze=None):
    """Fine-tune; with `freeze`, masked first-layer weights stay at zero."""
    W = reg._first_linear_weight(model)
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.train()
    for _ in range(epochs):
        for Xb, yb in loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            criterion(model(Xb), yb).backward()
            if freeze is not None and W.grad is not None:
                W.grad.masked_fill_(freeze.to(W.device), 0.0)
            opt.step()
            if freeze is not None:
                with torch.no_grad():
                    W.masked_fill_(freeze.to(W.device), 0.0)
    return model

def run_pruning_experiments_clf(model, snap_trained, train_loader, test_loader, device, T_z_bh,
                                N_WEIGHTS, finetune_epochs=10, lr=3e-4, weight_decay=0.01,
                                lookup_table=None):
    """Prune the first layer at the BH threshold and fine-tune (freeze, full, continued)."""
    crit = nn.CrossEntropyLoss()
    loss_trained, score_trained = evaluate_classification(model, test_loader, device)
    mask_2d = snap_trained['z'] >= T_z_bh
    mask_t = torch.tensor(mask_2d, dtype=torch.bool)
    n_kept, n_total = int(mask_2d.sum()), mask_2d.size
    print(f'Threshold  : T_z = {T_z_bh:.4f}')
    print(f'Kept       : {n_kept:,} / {n_total:,} ({100 * n_kept / n_total:.2f}%)')
    model_pruned = copy.deepcopy(model)
    W = reg._first_linear_weight(model_pruned)
    with torch.no_grad():
        W.data[~mask_t.to(W.device)] = 0.0
    m_mk = _finetune(copy.deepcopy(model_pruned), train_loader, finetune_epochs, lr, weight_decay, crit,
                     freeze=~mask_t)
    m_ft = _finetune(copy.deepcopy(model_pruned), train_loader, finetune_epochs, lr, weight_decay, crit)
    m_ct = _finetune(copy.deepcopy(model), train_loader, finetune_epochs, lr, weight_decay, crit)
    out = dict(mask_2d=mask_2d, n_kept=n_kept, n_total=n_total, model_pruned=model_pruned,
               loss_trained=loss_trained, score_trained=score_trained)
    for tag, m in (('masked_ft', m_mk), ('finetuned', m_ft), ('continued', m_ct)):
        out['snap_' + tag] = reg.capture(m, N_WEIGHTS, device, lookup_table)
        out['loss_' + tag], out['score_' + tag] = evaluate_classification(m, test_loader, device)
    return out

def threshold_sweep_clf(model, snap_trained, train_loader, test_loader, device, T_grid,
                        finetune_epochs=10, lr=3e-4, weight_decay=0.01):
    """Prune + fine-tune at each threshold in T_grid; return accuracies."""
    crit = nn.CrossEntropyLoss()
    s_m, s_f = [], []
    print(f'Sweeping {len(T_grid)} thresholds  [{T_grid[0]:.1f}, {T_grid[-1]:.1f}]')
    for T_z in T_grid:
        mask_t = torch.tensor(snap_trained['z'] >= T_z, dtype=torch.bool)
        base = copy.deepcopy(model)
        Wb = reg._first_linear_weight(base)
        with torch.no_grad():
            Wb.data[~mask_t.to(Wb.device)] = 0.0
        m_mk = _finetune(copy.deepcopy(base), train_loader, finetune_epochs, lr, weight_decay, crit,
                         freeze=~mask_t)
        s_m.append(evaluate_classification(m_mk, test_loader, device)[1])
        m_ft = _finetune(copy.deepcopy(base), train_loader, finetune_epochs, lr, weight_decay, crit)
        s_f.append(evaluate_classification(m_ft, test_loader, device)[1])
        kept = 100 * mask_t.float().mean().item()
        print(f'{T_z:8.3f}  {kept:6.2f}%  {s_m[-1]:10.4f}  {s_f[-1]:10.4f}')
    return np.array(s_m), np.array(s_f)

def _norm_reg_pruning(p):
    out = {k: p[k] for k in ('mask_2d', 'n_kept', 'n_total', 'model_pruned',
                             'snap_masked_ft', 'snap_finetuned', 'snap_continued')}
    for tag in ('trained', 'masked_ft', 'finetuned', 'continued'):
        out['loss_' + tag] = p['mse_' + tag]
        out['score_' + tag] = p['r2_' + tag]
    return out

TASK = {
    'reg': dict(score='R²', loss='MSE', criterion=nn.MSELoss,
                new_model=lambda d, hidden, n_classes, seed: reg.make_model(d, hidden, seed, DEVICE),
                train=reg.train_regression, evaluate=reg.evaluate,
                prune=lambda *a, **k: _norm_reg_pruning(reg.run_pruning_experiments(*a, **k)),
                sweep=reg.threshold_sweep),
    'clf': dict(score='Accuracy', loss='CE loss', criterion=nn.CrossEntropyLoss,
                new_model=make_clf_model, train=train_classification,
                evaluate=evaluate_classification, prune=run_pruning_experiments_clf,
                sweep=threshold_sweep_clf),
}

def _split_idx(n, y, frac, seed, stratify):
    idx = np.arange(n)
    try:
        return train_test_split(idx, test_size=frac, random_state=seed,
                                stratify=(y.numpy() if stratify else None))
    except ValueError:
        return train_test_split(idx, test_size=frac, random_state=seed)

def tune_hparams(task, X, y, d, hidden, n_classes, seed):
    """Pick weight decay and epoch count on a validation split."""
    T = TASK[task]
    tr, va = _split_idx(len(X), y, CFG.VAL_FRAC, seed, task == 'clf')
    tr, va = torch.as_tensor(tr), torch.as_tensor(va)
    tl, vl = GPULoader(X[tr], y[tr], CFG.BATCH_SIZE, seed), eval_loader(X[va], y[va])
    best = None
    for wd in CFG.TUNE_WD_GRID:
        model = T['new_model'](d, hidden, n_classes, seed)
        crit = T['criterion']()
        opt = optim.Adam(model.parameters(), lr=CFG.LR, weight_decay=wd)
        best_val, best_ep, bad = float('inf'), 1, 0
        for ep in range(1, CFG.TUNE_MAX_EPOCHS + 1):
            model.train()
            for Xb, yb in tl:
                opt.zero_grad()
                crit(model(Xb), yb).backward()
                opt.step()
            val_loss = T['evaluate'](model, vl, DEVICE)[0]
            if val_loss < best_val:
                best_val, best_ep, bad = val_loss, ep, 0
            else:
                bad += 1
                if bad >= CFG.TUNE_PATIENCE:
                    break
        if best is None or best_val < best['val_loss']:
            best = dict(wd=wd, epochs=best_ep, val_loss=float(best_val))
    print(f'  tuned: wd={best["wd"]:g}  epochs={best["epochs"]}  val_loss={best["val_loss"]:.4f}')
    return best

def diagnostics(snap_init, snap_trained, analysis, NW):
    """Init KS p-value, tail enrichment and lookup-table error."""
    b0 = (NW - 1) / 2.0
    ks_p = float(kstest(snap_init['x'].ravel(), 'beta', args=(0.5, b0)).pvalue)
    a_t, b_t = analysis['ecdf_fits']['trained']
    x_tr = snap_trained['x'].ravel()
    tail_enrich = float(np.mean(x_tr > sp_beta.isf(CFG.TAIL_P, a_t, b_t)) / CFG.TAIL_P)
    exact = -np.log(np.clip(sp_beta.sf(x_tr, 0.5, b0), 1e-300, 1.0))
    ok = exact < 30
    err = float(np.max(np.abs(exact[ok] - snap_trained['sig'].ravel()[ok]))) if ok.any() else float('nan')
    return dict(ks_init_p=ks_p, tail_enrich=tail_enrich, lookup_err=err)

def fit_and_prune(key, seed, regime, do_sweep=True, T_grid=None, verbose=True):
    """Train, run null analysis, prune at BH, fine-tune, optionally sweep thresholds."""
    t0 = time.time()
    E, D = REGISTRY[key], get_data(key)
    task, T = E['task'], TASK[E['task']]
    d, hidden, n_classes, n_train = D['d'], D['hidden'], D['n_classes'], D['n_train']
    NW = hidden[0] * d
    tune = None
    if regime == 'fixed':
        wd, epochs = D['wd'], CFG.EPOCHS
    else:
        tune = tune_hparams(task, D['Xtr'], D['ytr'], d, hidden, n_classes, seed)
        wd, epochs = tune['wd'], tune['epochs']
    print(f'--- {key} | {regime} | seed {seed} | d={d} B={NW} n_train={n_train} | {epochs} ep, wd={wd:g}')

    tl = GPULoader(D['Xtr'], D['ytr'], CFG.BATCH_SIZE, seed)
    vl = eval_loader(D['Xte'], D['yte'])
    model = T['new_model'](d, hidden, n_classes, seed)
    snap_init = reg.capture(model, NW, DEVICE, LOOKUP)
    T['train'](model, tl, DEVICE, epochs, CFG.LR, wd,
               test_loader=(vl if verbose else None), print_every=max(1, epochs // 4))
    snap_trained = reg.capture(model, NW, DEVICE, LOOKUP)

    analysis = null_a.run_null_analysis(snap_init, snap_trained, NW, beta_threshold_quantiles=[1.00],
                                        signal_cols=None, alpha_bh=CFG.BH_ALPHA)
    T_bh = float(analysis['T_z_bh'])
    fired = bool(np.isfinite(T_bh))

    p = T['prune'](model, snap_trained, tl, vl, DEVICE, T_z_bh=T_bh, N_WEIGHTS=NW,
                   finetune_epochs=CFG.FINETUNE_EPOCHS, lr=CFG.LR, weight_decay=wd, lookup_table=LOOKUP)
    loss_pruned, score_pruned = T['evaluate'](p.pop('model_pruned'), vl, DEVICE)

    res = dict(key=key, task=task, desc=E['desc'], regime=regime, seed=seed, d=d, hidden=hidden,
               N_WEIGHTS=NW, n_train=n_train, epochs=int(epochs), wd=float(wd), tune=tune,
               chance=D['chance'], snap_init=snap_init, snap_trained=snap_trained, analysis=analysis,
               T_z_bh=T_bh, bh_fired=fired, loss_pruned=float(loss_pruned),
               score_pruned=float(score_pruned), **p)
    res.update(diagnostics(snap_init, snap_trained, analysis, NW))

    if do_sweep:
        if T_grid is None:
            hi = CFG.FIG5_T_MAX
            if fired and T_bh >= CFG.FIG5_T_MAX:
                hi = int(np.ceil(T_bh)) + 2
            T_grid = np.arange(0.0, hi, 1)
        s_m, s_f = T['sweep'](model, snap_trained, tl, vl, DEVICE, T_grid=T_grid,
                              finetune_epochs=CFG.FINETUNE_EPOCHS, lr=CFG.LR, weight_decay=wd)
        res.update(T_grid=np.asarray(T_grid), sweep_masked=np.asarray(s_m), sweep_ft=np.asarray(s_f))

    res['seconds'] = round(time.time() - t0, 1)
    del model, tl
    LOOKUP.tables.clear()
    LOOKUP.pdf_tables.clear()
    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()
    return res

@contextlib.contextmanager
def _defer_show():
    orig = plt.show
    plt.show = lambda *a, **k: None
    try:
        yield
    finally:
        plt.show = orig

_SENTINEL = 1e9

def _slug(key):
    return key.replace(':', '_')

def _header(res):
    key, regime, ep, wd, seed = res['key'], res['regime'], res['epochs'], res['wd'], res['seed']
    d, NW, n = res['d'], res['N_WEIGHTS'], res['n_train']
    return (f'{key}  |  {regime}: {ep} ep (+{CFG.FINETUNE_EPOCHS} FT), wd={wd:g}, batch {CFG.BATCH_SIZE}, '
            f'lr {CFG.LR:g}  |  seed {seed}  |  d={d}, B={NW}, n_train={n}')

def _finish(fig, stem, note=None):
    """Save the figure as PNG and PDF."""
    if CFG.ANNOTATE_FIGS and note:
        fig.text(0.5, -0.04, note, ha='center', va='top', fontsize=10, color='0.3')
    for ext, kw in (('png', dict(dpi=150)), ('pdf', {})):
        fig.savefig(stem + '.' + ext, bbox_inches='tight', **kw)
    print('Saved:', stem + '.{png,pdf}')
    plt.close(fig)

def plot_fig3(res, paths):
    """Fig 3A and 3B for one run."""
    a, NW, fired = res['analysis'], res['N_WEIGHTS'], res['bh_fired']
    pfx = os.path.join(paths.fig, _slug(res['key']) + '_' + res['regime'] + '_s' + str(res['seed']))
    note = _header(res)
    if not fired:
        note += NL + 'BH (alpha = ' + str(CFG.BH_ALPHA) + ') found no significant weights: every first-layer weight is pruned.'

    T_plot = res['T_z_bh'] if fired else _SENTINEL
    with _defer_show():
        plots.fig1_weight_distributions(res['snap_init'], res['snap_trained'], T_plot, a['T_z_ecdf_all'],
                                        NW, res['epochs'], save=False, fits=a['fits'],
                                        beta_threshold_quantiles=[1.00])
        fig = plt.gcf()
    if not fired:
        for ax in fig.axes:
            removed = False
            for ln in list(ax.lines):
                xd = np.asarray(ln.get_xdata(), dtype=float)
                if xd.size and np.all(xd == _SENTINEL):
                    ln.remove()
                    removed = True
            if removed and ax.get_legend() is not None:
                ax.legend(fontsize=9, loc='upper right')
    _finish(fig, pfx + '_fig3A', note)

    null_sorted = null_a.build_null_sorted(res['hidden'][0], res['d'], n_draws=CFG.NULL_N_DRAWS)
    with _defer_show():
        plots.fig2_null_comparison(res['snap_init'], res['snap_trained'], a['fits'], a['ecdf_fits'],
                                   a['T_z_ecdf_all'], null_sorted, NW, res['epochs'], is_real_data=True,
                                   qq_ci=CFG.QQ_CI, beta_threshold_quantiles=[1.00], save=False)
        fig = plt.gcf()
    _finish(fig, pfx + '_fig3B', _header(res))

def plot_fig5(runs, stem, note):
    """Fig 5: score vs pruning threshold."""
    r0 = runs[0]
    score = TASK[r0['task']]['score']
    T_grid = r0['T_grid']
    S_m = np.array([r['sweep_masked'] for r in runs])
    S_f = np.array([r['sweep_ft'] for r in runs])
    m_m, m_f = S_m.mean(0), S_f.mean(0)
    s_trained = float(np.mean([r['score_trained'] for r in runs]))
    s_cont = float(np.mean([r['score_continued'] for r in runs]))
    epochs, ft = r0['epochs'], CFG.FINETUNE_EPOCHS
    N_WEIGHTS, T_optimal, fired = r0['N_WEIGHTS'], r0['T_z_bh'], r0['bh_fired']
    b_null = (N_WEIGHTS - 1) / 2.0

    fig5, ax5 = plt.subplots(figsize=(10, 6))
    ax5.plot(T_grid, m_m, color='steelblue', lw=2.0, marker='o', ms=4,
             label=f'Post-pruning + freeze-mask ({ft} ep)')
    ax5.plot(T_grid, m_f, color='darkorange', lw=2.0, marker='s', ms=4,
             label=f'Post-pruning + full fine-tune ({ft} ep)')
    if len(runs) > 1:
        ax5.fill_between(T_grid, m_m - S_m.std(0), m_m + S_m.std(0), color='steelblue', alpha=0.15)
        ax5.fill_between(T_grid, m_f - S_f.std(0), m_f + S_f.std(0), color='darkorange', alpha=0.15)
    ax5.axhline(s_trained, color='#444444', lw=1.2, ls='--', label=f'Trained ({epochs} ep)')
    ax5.axhline(s_cont, color='seagreen', lw=1.2, ls='--', label=f'Continued ({epochs + ft} ep, no prune)')
    if fired:
        i_opt = int(np.argmin(np.abs(T_grid - T_optimal)))
        ax5.axvline(T_optimal, color=plots.ECDF_COLOR, lw=2.0, ls='-', alpha=0.3, zorder=10,
                    label=f'BH pruning threshold  T_z={T_optimal:.3f}')
        ax5.plot(T_optimal, m_m[i_opt], marker='*', ms=14, color=plots.ECDF_COLOR, zorder=11)
        ax5.plot(T_optimal, m_f[i_opt], marker='*', ms=14, color=plots.ECDF_COLOR, zorder=11)
    else:
        ax5.text(0.02, 0.04, 'BH: no significant weights', transform=ax5.transAxes,
                 fontsize=11, color=plots.ECDF_COLOR)
    ax5.set_xlabel('Threshold', fontsize=14)
    ax5.set_ylabel(score, fontsize=14)
    ax5.set_title(f'{score} vs Pruning Threshold', fontsize=16, fontweight='bold')
    ax5.legend(fontsize=11, loc='lower right')
    ax5.tick_params(labelsize=12)
    ax5.grid(True, alpha=0.3, linestyle='--')

    ax5_top = ax5.twiny()
    ax5_top.set_xlim(ax5.get_xlim())
    for _side in ['bottom', 'right', 'left']:
        ax5_top.spines[_side].set_visible(False)
    ax5_top.xaxis.set_ticks_position('top')
    ax5_top.xaxis.set_label_position('top')
    _wval_pcts = np.array([50, 25, 10, 5, 1, 0.5, 0.1])
    _wval_z = np.array([sp_beta.isf(np.clip(p / 100, 1e-15, 1 - 1e-15), 0.5, b_null) * N_WEIGHTS
                        for p in _wval_pcts])
    _xlim = ax5.get_xlim()
    _valid_top = (_wval_z >= _xlim[0]) & (_wval_z <= _xlim[1])
    ax5_top.set_xticks(_wval_z[_valid_top])
    ax5_top.set_xticklabels([f'{p:.1f}' if p < 1 else f'{int(p)}' for p in _wval_pcts[_valid_top]],
                            fontsize=12)
    ax5_top.set_xlabel('w-value  (keep weights where w-value is <= this %)', fontsize=14)

    z_sorted = np.sort(r0['snap_trained']['z'].ravel())
    ax5_bot = ax5.twiny()
    for side in ['top', 'right', 'left']:
        ax5_bot.spines[side].set_visible(False)
    ax5_bot.xaxis.set_ticks_position('bottom')
    ax5_bot.xaxis.set_label_position('bottom')
    ax5_bot.spines['bottom'].set_position(('outward', 48))
    ax5_bot.set_xlim(ax5.get_xlim())
    _nice_pcts = np.array([75, 90, 95, 99, 99.5])
    _nice_z = np.percentile(z_sorted, _nice_pcts)
    _valid = (_nice_z >= _xlim[0]) & (_nice_z <= _xlim[1])
    ax5_bot.set_xticks(_nice_z[_valid])
    ax5_bot.set_xticklabels([f'{p:.1f}' if p > 99 else f'{int(p)}' for p in _nice_pcts[_valid]], fontsize=12)
    ax5_bot.set_xlabel('Weights pruned (%)', fontsize=14)
    fig5.subplots_adjust(left=0.09, right=0.97, top=0.87, bottom=0.20)
    if len(runs) > 1:
        note = note + NL + 'mean and std over seeds ' + str([r['seed'] for r in runs])
    _finish(fig5, stem, note)

def plot_fig4(key, regime, runs_rows, stem, note):
    """Fig 4: test score vs training samples."""
    E, D = REGISTRY[key], get_data(key)
    task, desc = E['task'], E['desc']
    score = TASK[task]['score']
    ns = np.array([r['n'] for r in runs_rows[0]])

    def arr(k):
        return np.array([[r[k] for r in rows] for rows in runs_rows], dtype=float)

    fired_frac = arr('bh_fired').mean(0)
    fig, ax = plt.subplots(figsize=(9, 5))
    shaded = False
    for n_, f_ in zip(ns, fired_frac):
        if f_ == 0:
            ax.axvspan(n_ / 1.04, n_ * 1.04, color='0.88', zorder=0)
            shaded = True
    ft = CFG.FINETUNE_EPOCHS
    series = [('score_base', 'steelblue', 'o', '-', 'Baseline MLP  (no pruning)'),
              ('score_freeze', 'forestgreen', 's', '-', f'Pruned + freeze FT  ({ft} ep)'),
              ('score_full', 'darkorange', '^', '-', f'Pruned + full FT  ({ft} ep)'),
              ('score_reset', 'crimson', 'x', '--', f'Control: layer-1 reset + full FT  ({ft} ep)')]
    for k, c, mk, ls, lbl in series:
        A = arr(k)
        mu = np.nanmean(A, 0)
        ax.plot(ns, mu, color=c, lw=2, marker=mk, ms=6, ls=ls, label=lbl)
        if A.shape[0] > 1:
            sd = np.nanstd(A, 0)
            ax.fill_between(ns, mu - sd, mu + sd, color=c, alpha=0.12)
    if task == 'clf':
        ax.axhline(D['chance'], color='#999999', lw=1.0, ls=':', label='Majority class')
    else:
        ax.axhline(0, color='#999999', lw=0.8, ls=':')
    handles, labels = ax.get_legend_handles_labels()
    if shaded:
        handles.append(Patch(color='0.88'))
        labels.append('BH found no significant weights')
    ax.set_xscale('log')
    ax.set_xlabel('Training samples  (n)', fontsize=13)
    ax.set_ylabel('Test ' + score, fontsize=13)
    ep_txt = f'{CFG.EPOCHS} training epochs' if regime == 'fixed' else 'validation-tuned epochs'
    ax.set_title(f'Sample size vs Test {score}  —  {desc}' + NL
                 + f'({ep_txt} + {ft} FT epochs,  BH α = {CFG.BH_ALPHA})', fontsize=12)
    ax.legend(handles, labels, fontsize=10, loc='lower right')
    ax.grid(True, alpha=0.3, linestyle='--', which='both')
    plt.tight_layout()
    _finish(fig, stem, note)

def run_screen(paths, eligible, resume=True):
    """Screen every eligible dataset; resumable, one CSV row per (dataset, seed)."""
    csv = os.path.join(paths.tab, 'screen_rows.csv')
    rows = pd.read_csv(csv).to_dict('records') if (resume and os.path.exists(csv)) else []
    done = {(r['dataset'], int(r['seed'])) for r in rows}
    failed = []
    try:
        for i, key in enumerate(eligible):
            for seed in CFG.SCREEN_SEEDS:
                if (key, seed) in done:
                    continue
                try:
                    r = fit_and_prune(key, seed, 'fixed', do_sweep=False, verbose=False)
                except Exception as e:
                    print('  FAILED', key, 'seed', seed, type(e).__name__, str(e)[:200])
                    failed.append((key, seed, type(e).__name__ + ': ' + str(e)[:200]))
                    continue
                kept_pct = 100.0 * r['n_kept'] / r['n_total']
                rows.append(dict(
                    dataset=key, task=r['task'], d=r['d'], n_train=r['n_train'], B=r['N_WEIGHTS'], seed=seed,
                    epochs=r['epochs'], wd=r['wd'], chance=r['chance'], bh_fired=r['bh_fired'],
                    T_z_bh=r['T_z_bh'], T_z_ecdf=float(r['analysis']['T_z_ecdf_all']), kept_pct=kept_pct,
                    score_trained=r['score_trained'], score_continued=r['score_continued'],
                    score_pruned=r['score_pruned'], score_freeze=r['score_masked_ft'],
                    score_full=r['score_finetuned'], ks_init_p=r['ks_init_p'],
                    tail_enrich=r['tail_enrich'], lookup_err=r['lookup_err'], seconds=r['seconds']))
                pd.DataFrame(rows).to_csv(csv, index=False)
                sname = TASK[r['task']]['score']
                print(f"[{i + 1}/{len(eligible)}] {key:<30s} seed={seed}  {sname}: "
                      f"dense={r['score_trained']:.4f} full-FT={r['score_finetuned']:.4f}  "
                      f"BH={r['bh_fired']}  kept={kept_pct:.2f}%  tail x{r['tail_enrich']:.1f}  "
                      f"({r['seconds']:.0f}s)")
    except KeyboardInterrupt:
        print('Interrupted. Progress is saved in', csv, '- re-run to resume.')
        raise
    finally:
        if failed:
            pd.DataFrame(failed, columns=['dataset', 'seed', 'error']).to_csv(
                os.path.join(paths.tab, 'screen_failures.csv'), index=False)
    print('screen rows:', len(rows))
    return pd.DataFrame(rows)

def summarize_screen(paths, screen_rows=None):
    """Summarize the screen, draw screen figures, pick datasets."""
    if screen_rows is None:
        screen_rows = pd.read_csv(os.path.join(paths.tab, 'screen_rows.csv'))
    S = pd.DataFrame(screen_rows)
    if S.empty:
        print('No screen rows; run the screen first.')
        return None, [], []
    S['bh_fired'] = S['bh_fired'].astype(bool)
    S['delta_freeze'] = np.where(S['bh_fired'], S['score_freeze'] - S['score_trained'], np.nan)
    S['delta_full'] = np.where(S['bh_fired'], S['score_full'] - S['score_trained'], np.nan)
    S['delta_continued'] = S['score_continued'] - S['score_trained']
    g = S.groupby('dataset')
    SUMMARY = pd.DataFrame(dict(
        task=g['task'].first(), d=g['d'].first(), n_train=g['n_train'].first(), B=g['B'].first(),
        seeds=g['seed'].count(), chance=g['chance'].first(),
        score_trained=g['score_trained'].mean(), score_trained_sd=g['score_trained'].std(),
        bh_rate=g['bh_fired'].mean(), kept_pct_median=g['kept_pct'].median(),
        delta_freeze=g['delta_freeze'].mean(), delta_full=g['delta_full'].mean(),
        delta_full_sd=g['delta_full'].std(), delta_continued=g['delta_continued'].mean(),
        tail_enrich=g['tail_enrich'].mean(), ks_init_p_min=g['ks_init_p'].min(),
        lookup_err_max=g['lookup_err'].max()))
    SUMMARY['learned'] = np.where(SUMMARY['task'] == 'reg', SUMMARY['score_trained'] > 0.05,
                                  SUMMARY['score_trained'] > SUMMARY['chance'] + 0.01)
    SUMMARY['low_d'] = SUMMARY['d'] <= CFG.MIN_D
    SUMMARY = SUMMARY.sort_values(['task', 'bh_rate', 'tail_enrich'], ascending=[True, False, False])
    SUMMARY.to_csv(os.path.join(paths.tab, 'screen_summary.csv'))
    with pd.option_context('display.max_rows', 400, 'display.width', 250):
        print(SUMMARY.round(4).to_string())

    collapsed = SUMMARY[SUMMARY['kept_pct_median'] < 1.0]
    for name, row in collapsed.iterrows():
        print(f'  NOTE: {name} keeps only {row["kept_pct_median"]:.2f}% of first-layer weights '
              f'(median over seeds); pruning results there reflect an extremely sparse first layer.')

    fig, ax = plt.subplots(figsize=(8, 5))
    for task, mk in (('clf', 'o'), ('reg', 's')):
        sub = SUMMARY[SUMMARY['task'] == task]
        ax.scatter(sub['d'], sub['bh_rate'], marker=mk, s=45, alpha=0.7,
                   label=('classification' if task == 'clf' else 'regression'))
    ax.set_xscale('log')
    ax.set_xlabel('input dimension d', fontsize=12)
    ax.set_ylabel('fraction of seeds where BH fired', fontsize=12)
    ax.set_title('Screen: does BH find significant first-layer weights?', fontsize=13)
    ax.grid(alpha=0.3, ls='--')
    ax.legend()
    _finish(fig, os.path.join(paths.fig, 'screen_bh_rate_vs_d'),
            f'{len(SUMMARY)} datasets, seeds {CFG.SCREEN_SEEDS}, {CFG.EPOCHS} epochs, batch {CFG.BATCH_SIZE}')

    order = SUMMARY.dropna(subset=['delta_full']).sort_values('delta_full')
    if len(order):
        fig, ax = plt.subplots(figsize=(8, max(4, 0.22 * len(order))))
        cols = ['darkorange' if t == 'clf' else 'steelblue' for t in order['task']]
        ax.barh(range(len(order)), order['delta_full'], xerr=order['delta_full_sd'].fillna(0),
                color=cols, alpha=0.8)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels(order.index, fontsize=7)
        ax.axvline(0, color='k', lw=0.8)
        ax.set_xlabel('pruned + full FT  minus  trained   (accuracy or R², seeds where BH fired)', fontsize=10)
        ax.set_title('Screen: effect of BH pruning', fontsize=13)
        ax.grid(alpha=0.3, ls='--', axis='x')
        plt.tight_layout()
        _finish(fig, os.path.join(paths.fig, 'screen_pruning_effect'),
                'orange = classification, blue = regression')

    cand = SUMMARY[(SUMMARY['bh_rate'] >= CFG.SEL_MIN_BH_RATE) & SUMMARY['learned']
                   & (SUMMARY['ks_init_p_min'] > 1e-3)]
    auto = []
    for task, k in CFG.SEL_TOP_K.items():
        auto += cand[cand['task'] == task].sort_values('tail_enrich', ascending=False).index[:k].tolist()
    selected = list(CFG.MANUAL_CASES) if CFG.MANUAL_CASES else auto
    print('Automatic pick (BH rate >= ' + str(CFG.SEL_MIN_BH_RATE)
          + ', model learned, init KS ok; ranked by tail enrichment):')
    for k in auto:
        print('   ', k)
    n_reg = sum(1 for k in auto if k.split(':')[0] != 'uci')
    if n_reg == 0 and not CFG.MANUAL_CASES:
        print('NOTE: the automatic rule selected no regression datasets. Pass --cases (e.g. '
              '--cases uci:spambase openml:pol) to choose datasets manually.')
    if CFG.MANUAL_CASES:
        print('Manual cases in use:', selected)
    if not selected:
        print('Nothing selected. Inspect tables/screen_summary.csv and use --cases.')
    write_json(dict(auto=auto, selected=selected), os.path.join(paths.tab, 'selection.json'))
    return SUMMARY, auto, selected

def selected_cases(paths):
    """Datasets for the full figures."""
    if CFG.MANUAL_CASES:
        return list(CFG.MANUAL_CASES)
    p = os.path.join(paths.tab, 'selection.json')
    if not os.path.exists(p):
        raise RuntimeError('No selection yet: run the screen first (or pass --cases).')
    return read_json(p)['selected']

def run_full_figures(paths, selected, resume=True):
    """Run fit_and_prune per dataset/regime/seed and draw Figs 3 and 5."""
    build_registry()
    FULL = {}
    for key in selected:
        if key not in REGISTRY:
            print('  [skip] unknown dataset', key)
            continue
        for regime in CFG.REGIMES:
            runs, T_grid = [], None
            try:
                for seed in CFG.FULL_SEEDS:
                    path = os.path.join(paths.art, 'full_' + _slug(key) + '_' + regime + '_s' + str(seed) + '.pt')
                    if resume and os.path.exists(path):
                        r = torch.load(path, weights_only=False)
                        print('[resume]', os.path.basename(path))
                    else:
                        r = fit_and_prune(key, seed, regime, do_sweep=True, T_grid=T_grid)
                        torch.save(r, path)
                    if T_grid is None:
                        T_grid = r['T_grid']
                    runs.append(r)
            except Exception as e:
                print('  FAILED', key, regime, type(e).__name__, str(e)[:200])
                continue
            FULL[(key, regime)] = runs
            plot_fig3(runs[0], paths)
            plot_fig5(runs, os.path.join(paths.fig, _slug(key) + '_' + regime + '_fig5'), _header(runs[0]))
            plt.close('all')
    return FULL

def make_table1(paths, FULL):
    """Table 1: scores per dataset and regime."""
    rows = []
    for (key, regime), runs in FULL.items():
        T = TASK[runs[0]['task']]
        ep, ft = runs[0]['epochs'], CFG.FINETUNE_EPOCHS
        entries = [(f'Trained {ep} epochs', 'trained'),
                   (f'Trained {ep + ft} epochs', 'continued'),
                   (f'Pruned after {ep} epochs', 'pruned'),
                   (f'Pruned & freeze fine-tuned {ft} ep.', 'masked_ft'),
                   (f'Pruned & fine-tuned {ft} ep.', 'finetuned')]
        for label, tag in entries:
            lv = np.array([r['loss_' + tag] for r in runs], dtype=float)
            sv = np.array([r['score_' + tag] for r in runs], dtype=float)
            rows.append(dict(dataset=key, regime=regime, model=label,
                             loss_name=T['loss'], loss=lv.mean(), loss_sd=(lv.std() if len(lv) > 1 else np.nan),
                             score_name=T['score'], score=sv.mean(),
                             score_sd=(sv.std() if len(sv) > 1 else np.nan),
                             wd=runs[0]['wd'], bh_fired_seeds=int(sum(r['bh_fired'] for r in runs)),
                             seeds=len(runs),
                             kept_pct=100.0 * float(np.mean([r['n_kept'] / r['n_total'] for r in runs]))))
    T1 = pd.DataFrame(rows)
    T1.to_csv(os.path.join(paths.tab, 'table1.csv'), index=False)
    with pd.option_context('display.max_rows', 400, 'display.width', 250):
        print(T1.round(4).to_string(index=False))
    return T1

def sample_sizes(task, n_train):
    """Training-set sizes for Fig 4."""
    if task == 'reg' and n_train >= CFG.REPO_SAMPLE_END:
        s = np.linspace(CFG.REPO_SAMPLE_START, CFG.REPO_SAMPLE_END, CFG.N_SAMPLE_STEPS)
    else:
        s = np.linspace(max(CFG.MIN_NTRAIN, int(0.1 * n_train)), n_train, CFG.N_SAMPLE_STEPS)
    return [int(v) for v in np.unique(np.round(s).astype(int))]

def _subset(task, X, y, n):
    if n >= len(X):
        return X, y
    if task == 'reg':
        return X[:n], y[:n]
    try:
        idx, _ = train_test_split(np.arange(len(X)), train_size=n, random_state=CFG.SEED, stratify=y.numpy())
    except ValueError:
        idx, _ = train_test_split(np.arange(len(X)), train_size=n, random_state=CFG.SEED)
    idx = torch.as_tensor(idx)
    return X[idx], y[idx]

def _prune_ft_score(task, model, keep, loader, vl, wd, freeze):
    T = TASK[task]
    m = copy.deepcopy(model)
    keep_t = torch.tensor(keep, dtype=torch.bool)
    W = reg._first_linear_weight(m)
    with torch.no_grad():
        W.data[~keep_t.to(W.device)] = 0.0
    _finetune(m, loader, CFG.FINETUNE_EPOCHS, CFG.LR, wd, T['criterion'](),
              freeze=(~keep_t if freeze else None))
    return float(T['evaluate'](m, vl, DEVICE)[1])

def sample_size_sweep_any(key, regime, seed):
    """Train, prune and fine-tune at each training-set size."""
    E, D = REGISTRY[key], get_data(key)
    task, T = E['task'], TASK[E['task']]
    d, hidden, n_classes = D['d'], D['hidden'], D['n_classes']
    NW = hidden[0] * d
    vl = eval_loader(D['Xte'], D['yte'])
    rows = []
    for n in sample_sizes(task, D['n_train']):
        t0 = time.time()
        Xs, ys = _subset(task, D['Xtr'], D['ytr'], n)
        if regime == 'fixed':
            wd, epochs = D['wd'], CFG.EPOCHS
        else:
            tune = tune_hparams(task, Xs, ys, d, hidden, n_classes, seed)
            wd, epochs = tune['wd'], tune['epochs']
        tl = GPULoader(Xs, ys, CFG.BATCH_SIZE, seed)
        model = T['new_model'](d, hidden, n_classes, seed)
        T['train'](model, tl, DEVICE, epochs, CFG.LR, wd)
        s_base = float(T['evaluate'](model, vl, DEVICE)[1])
        snap = reg.capture(model, NW, DEVICE, LOOKUP)
        T_bh = float(null_a.bh_threshold(snap, NW, CFG.BH_ALPHA))
        fired = bool(np.isfinite(T_bh))
        keep = snap['z'] >= T_bh
        s_frz = s_full = float('nan')
        if fired:
            s_frz = _prune_ft_score(task, model, keep, tl, vl, wd, freeze=True)
            s_full = _prune_ft_score(task, model, keep, tl, vl, wd, freeze=False)
        s_reset = _prune_ft_score(task, model, np.zeros_like(keep), tl, vl, wd, freeze=False)
        n_keep = int(keep.sum())
        rows.append(dict(n=n, epochs=int(epochs), wd=float(wd), score_base=s_base, score_freeze=s_frz,
                         score_full=s_full, score_reset=s_reset, bh_fired=fired, T_z_bh=T_bh,
                         kept=n_keep, kept_pct=100.0 * n_keep / keep.size, seconds=round(time.time() - t0, 1)))
        print(f'  n={n:>6d}  ep={epochs:>3d}  base={s_base:+.4f}  freeze={s_frz:+.4f}  full={s_full:+.4f}'
              f'  reset={s_reset:+.4f}  BH={fired}  kept={n_keep}')
        del model, tl
        LOOKUP.tables.clear()
        LOOKUP.pdf_tables.clear()
    return rows

def run_fig4(paths, selected, resume=True):
    """Run the sample-size sweep and draw Fig 4."""
    build_registry()
    cases = []
    for k in selected:
        if k not in REGISTRY:
            print('  [skip] unknown dataset', k)
            continue
        try:
            if get_data(k)['n_train'] >= CFG.FIG4_MIN_NTRAIN:
                cases.append(k)
            else:
                print(f'  [skip] {k}: n_train < FIG4_MIN_NTRAIN ({CFG.FIG4_MIN_NTRAIN})')
        except Exception as e:
            print('  [skip]', k, '-', type(e).__name__, str(e)[:120])
    print('Fig. 4 datasets:', cases)
    FIG4 = {}
    for key in cases:
        for regime in CFG.REGIMES:
            runs_rows = []
            for seed in CFG.FULL_SEEDS:
                path = os.path.join(paths.art, 'fig4_' + _slug(key) + '_' + regime + '_s' + str(seed) + '.json')
                if resume and os.path.exists(path):
                    rows = read_json(path)
                    print('[resume]', os.path.basename(path))
                else:
                    print('=== Fig. 4 |', key, '|', regime, '| seed', seed)
                    rows = sample_size_sweep_any(key, regime, seed)
                    write_json(rows, path)
                runs_rows.append(rows)
            FIG4[(key, regime)] = runs_rows
            note = f'{key}  |  {regime}  |  batch {CFG.BATCH_SIZE}, lr {CFG.LR:g}  |  seeds {CFG.FULL_SEEDS}'
            plot_fig4(key, regime, runs_rows, os.path.join(paths.fig, _slug(key) + '_' + regime + '_fig4'), note)
    print()
    print('Selection effect (full FT minus reset control), mean over n where BH fired:')
    for (key, regime), runs_rows in FIG4.items():
        diffs = [r['score_full'] - r['score_reset'] for rows in runs_rows for r in rows if r['bh_fired']]
        msg = f'{np.mean(diffs):+.4f} over {len(diffs)} points' if diffs else 'BH never fired'
        print(f'  {key:<30s} {regime:<6s} {msg}')
    return FIG4
