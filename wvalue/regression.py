"""Regression MLP utilities shared by notebooks 05 and 06."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_sizes):
        super().__init__()
        layers, prev = [], input_dim
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def make_model(input_dim, hidden_sizes, seed, device):
    from wvalue.utils import set_seed
    set_seed(seed)
    m = MLP(input_dim, hidden_sizes).to(device)
    for layer in m.modules():
        if isinstance(layer, nn.Linear):
            nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
            nn.init.zeros_(layer.bias)
    return m


def make_data(n, input_dim, make_y_fn, seed=0):
    gen = torch.Generator().manual_seed(seed)
    X   = torch.randn(n, input_dim, generator=gen)
    return X, make_y_fn(X)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, targets = [], []
    for Xb, yb in loader:
        preds.append(model(Xb.to(device)).cpu())
        targets.append(yb)
    preds, targets = torch.cat(preds), torch.cat(targets)
    mse = F.mse_loss(preds, targets).item()
    ss_res = ((targets - preds) ** 2).sum()
    ss_tot = ((targets - targets.mean()) ** 2).sum()
    r2     = (1 - ss_res / ss_tot).item()
    return mse, r2


def capture(model, n_weights, device, lookup_table=None):
    """Snapshot first-layer weights: raw W, Frobenius-normed x and z, and w-value sig."""
    import wvalue.core as _core
    first  = next(m for m in model.modules() if isinstance(m, nn.Linear))
    W_t    = first.weight.data.clone().cpu()
    W_np   = W_t.numpy()
    w_sq   = W_np ** 2
    x      = w_sq / w_sq.sum()
    z      = x * n_weights
    table  = lookup_table if lookup_table is not None else _core.beta_sf_lookup
    _, sig = _core.compute_w_value(W_t, use_lookup=True,
                                    beta_sf_lookup_table=table)
    return {
        'W':       W_np,
        'w_normed': W_np / np.sqrt(w_sq.sum()),
        'x':       x,
        'z':       z,
        'sig':     sig.detach().cpu().numpy(),
    }


def _first_linear_weight(model):
    """Return the weight parameter of the first Linear layer in model."""
    for m in model.modules():
        if isinstance(m, nn.Linear):
            return m.weight
    raise ValueError('No Linear layer found in model')


def train_regression(model, train_loader, device, epochs, lr, weight_decay,
                     test_loader=None, print_every=25):
    """Train a regression MLP with Adam. Prints MSE/R² periodically if test_loader given."""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.train()
    for epoch in range(1, epochs + 1):
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            criterion(model(Xb), yb).backward()
            optimizer.step()
        if test_loader is not None and print_every and (epoch % print_every == 0 or epoch == 1):
            mse, r2 = evaluate(model, test_loader, device)
            print(f'  Epoch {epoch:3d}/{epochs}  mse={mse:.4f}  r²={r2:+.4f}')
            model.train()


def run_pruning_experiments(
    model,
    snap_trained,
    train_loader,
    test_loader,
    device,
    T_z_bh,
    N_WEIGHTS,
    finetune_epochs=10,
    lr=3e-4,
    weight_decay=0.01,
    lookup_table=None,
):
    """Prune the first Linear layer by BH threshold and run three fine-tuning variants.

    Variants produced:
      freeze  — pruned init, gradient/weight mask frozen on zeroed positions
      full    — pruned init, all weights free to move
      continued — original model trained for finetune_epochs more epochs (no pruning)

    Returns a dict with model objects, snapshots, and MSE/R² for every variant plus
    the trained baseline (keys: mask_2d, n_kept, n_total, model_pruned,
    mse_trained, r2_trained, model_masked_ft, snap_masked_ft, mse_masked_ft,
    r2_masked_ft, model_finetuned, snap_finetuned, mse_finetuned, r2_finetuned,
    model_continued, snap_continued, mse_continued, r2_continued).
    """
    import copy
    criterion = nn.MSELoss()

    mse_trained, r2_trained = evaluate(model, test_loader, device)

    mask_2d = snap_trained['z'] >= T_z_bh
    mask_t  = torch.tensor(mask_2d, dtype=torch.bool)
    n_kept  = int(mask_2d.sum())
    n_total = mask_2d.size
    print(f'Threshold  : T_z = {T_z_bh:.4f}')
    print(f'Kept       : {n_kept:,} / {n_total:,} ({100*n_kept/n_total:.2f}%)')
    print(f'Zeroed out : {n_total-n_kept:,} / {n_total:,} ({100*(n_total-n_kept)/n_total:.2f}%)')

    # Shared pruned base
    model_pruned = copy.deepcopy(model)
    W_base = _first_linear_weight(model_pruned)
    with torch.no_grad():
        W_base.data[~mask_t.to(W_base.device)] = 0.0

    # ── Freeze-mask fine-tune ────────────────────────────────────────────────
    model_masked_ft = copy.deepcopy(model_pruned)
    W_mk   = _first_linear_weight(model_masked_ft)
    freeze = ~mask_t
    opt_mk = optim.Adam(model_masked_ft.parameters(), lr=lr, weight_decay=weight_decay)
    model_masked_ft.train()
    for _ in range(finetune_epochs):
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            opt_mk.zero_grad()
            criterion(model_masked_ft(Xb), yb).backward()
            if W_mk.grad is not None:
                W_mk.grad.masked_fill_(freeze.to(W_mk.device), 0.0)
            opt_mk.step()
            with torch.no_grad():
                W_mk.masked_fill_(freeze.to(W_mk.device), 0.0)
    snap_masked_ft              = capture(model_masked_ft, N_WEIGHTS, device, lookup_table)
    mse_masked_ft, r2_masked_ft = evaluate(model_masked_ft, test_loader, device)

    # ── Full fine-tune ───────────────────────────────────────────────────────
    model_finetuned = copy.deepcopy(model_pruned)
    opt_ft          = optim.Adam(model_finetuned.parameters(), lr=lr, weight_decay=weight_decay)
    model_finetuned.train()
    for _ in range(finetune_epochs):
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            opt_ft.zero_grad()
            criterion(model_finetuned(Xb), yb).backward()
            opt_ft.step()
    snap_finetuned               = capture(model_finetuned, N_WEIGHTS, device, lookup_table)
    mse_finetuned, r2_finetuned = evaluate(model_finetuned, test_loader, device)

    # ── Continued training (no pruning) ─────────────────────────────────────
    model_continued = copy.deepcopy(model)
    opt_ct          = optim.Adam(model_continued.parameters(), lr=lr, weight_decay=weight_decay)
    model_continued.train()
    for _ in range(finetune_epochs):
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            opt_ct.zero_grad()
            criterion(model_continued(Xb), yb).backward()
            opt_ct.step()
    snap_continued                 = capture(model_continued, N_WEIGHTS, device, lookup_table)
    mse_continued, r2_continued   = evaluate(model_continued, test_loader, device)

    return {
        'mask_2d':        mask_2d,
        'n_kept':         n_kept,
        'n_total':        n_total,
        'model_pruned':   model_pruned,
        'mse_trained':    mse_trained,
        'r2_trained':     r2_trained,
        'model_masked_ft':   model_masked_ft,
        'snap_masked_ft':    snap_masked_ft,
        'mse_masked_ft':     mse_masked_ft,
        'r2_masked_ft':      r2_masked_ft,
        'model_finetuned':   model_finetuned,
        'snap_finetuned':    snap_finetuned,
        'mse_finetuned':     mse_finetuned,
        'r2_finetuned':      r2_finetuned,
        'model_continued':   model_continued,
        'snap_continued':    snap_continued,
        'mse_continued':     mse_continued,
        'r2_continued':      r2_continued,
    }


def threshold_sweep(
    model,
    snap_trained,
    train_loader,
    test_loader,
    device,
    T_grid,
    finetune_epochs=10,
    lr=3e-4,
    weight_decay=0.01,
):
    """Sweep pruning thresholds and evaluate freeze-mask vs full fine-tune R².

    For each T_z in T_grid: prune first-layer weights below the threshold,
    then run freeze-mask fine-tune and full fine-tune from the same pruned init.

    Returns (r2_masked, r2_ft) as numpy arrays aligned with T_grid.
    """
    import copy
    import numpy as np
    criterion = nn.MSELoss()
    r2_masked_list, r2_ft_list = [], []

    print(f'Sweeping {len(T_grid)} thresholds  [{T_grid[0]:.1f}, {T_grid[-1]:.1f}]')
    print(f'{"T_z":>8}  {"kept%":>7}  {"R² freeze":>10}  {"R² ft":>10}')

    for T_z in T_grid:
        mask   = snap_trained['z'] >= T_z
        mask_t = torch.tensor(mask, dtype=torch.bool)

        m_base = copy.deepcopy(model)
        Wb = _first_linear_weight(m_base)
        with torch.no_grad():
            Wb.data[~mask_t.to(Wb.device)] = 0.0

        # Freeze-mask
        m_mk     = copy.deepcopy(m_base)
        W_tr     = _first_linear_weight(m_mk)
        freeze_m = ~mask_t
        opt_mk   = optim.Adam(m_mk.parameters(), lr=lr, weight_decay=weight_decay)
        m_mk.train()
        for _ in range(finetune_epochs):
            for Xb, yb in train_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                opt_mk.zero_grad()
                criterion(m_mk(Xb), yb).backward()
                if W_tr.grad is not None:
                    W_tr.grad.masked_fill_(freeze_m.to(W_tr.device), 0.0)
                opt_mk.step()
                with torch.no_grad():
                    W_tr.masked_fill_(freeze_m.to(W_tr.device), 0.0)
        _, r2_m = evaluate(m_mk, test_loader, device)

        # Full fine-tune
        m_ft   = copy.deepcopy(m_base)
        opt_ft = optim.Adam(m_ft.parameters(), lr=lr, weight_decay=weight_decay)
        m_ft.train()
        for _ in range(finetune_epochs):
            for Xb, yb in train_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                opt_ft.zero_grad()
                criterion(m_ft(Xb), yb).backward()
                opt_ft.step()
        _, r2_f = evaluate(m_ft, test_loader, device)

        r2_masked_list.append(r2_m)
        r2_ft_list.append(r2_f)
        pct = 100 * mask.sum() / mask.size
        print(f'{T_z:8.3f}  {pct:6.2f}%  {r2_m:+10.4f}  {r2_f:+10.4f}')

    return np.array(r2_masked_list), np.array(r2_ft_list)


def sample_size_sweep(
    case_cfg,
    sample_sizes,
    input_dim,
    hidden_sizes,
    epochs,
    finetune_epochs,
    lr,
    weight_decay,
    batch_size,
    bh_alpha,
    test_loader,
    device,
    N_WEIGHTS,
    lookup_table=None,
    x_train_full=None,
    y_train_full=None,
    seed=42,
):
    """Train + prune + fine-tune across a range of training set sizes.

    For real datasets pass x_train_full and y_train_full (tensors; rows are sliced by n).
    For synthetic datasets leave those as None and supply case_cfg['make_y'].

    Returns dict with keys: n, baseline, freeze, finetune, T_z_bh, kept_frac.
    """
    import copy
    import numpy as np
    from torch.utils.data import DataLoader, TensorDataset
    from wvalue.analysis import bh_threshold as _bh_threshold

    criterion = nn.MSELoss()
    results = {
        'n': [], 'baseline': [], 'freeze': [], 'finetune': [],
        'T_z_bh': [], 'kept_frac': [],
    }

    for n in sample_sizes:
        if x_train_full is not None:
            Xtr, ytr = x_train_full[:n], y_train_full[:n]
        else:
            Xtr, ytr = make_data(n, input_dim, case_cfg['make_y'], seed=seed)
        bs           = int(min(batch_size, n))
        train_loader = DataLoader(
            TensorDataset(Xtr, ytr), batch_size=bs, shuffle=True,
            generator=torch.Generator().manual_seed(seed),
        )

        # Baseline
        model = make_model(input_dim, hidden_sizes, seed, device)
        train_regression(model, train_loader, device, epochs, lr, weight_decay)
        _, r2_base = evaluate(model, test_loader, device)

        # BH threshold
        snap   = capture(model, N_WEIGHTS, device, lookup_table)
        T_z_bh = _bh_threshold(snap, N_WEIGHTS, bh_alpha)
        mask   = snap['z'] >= T_z_bh
        mask_t = torch.tensor(mask, dtype=torch.bool)
        n_kept = int(mask.sum())

        # Pruned base
        m_base = copy.deepcopy(model)
        Wf = _first_linear_weight(m_base)
        with torch.no_grad():
            Wf.data[~mask_t.to(Wf.device)] = 0.0

        # Freeze fine-tune
        m_f    = copy.deepcopy(m_base)
        Wf_p   = _first_linear_weight(m_f)
        freeze = ~mask_t
        opt_f  = optim.Adam(m_f.parameters(), lr=lr, weight_decay=weight_decay)
        m_f.train()
        for _ in range(finetune_epochs):
            for Xb, yb in train_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                opt_f.zero_grad()
                criterion(m_f(Xb), yb).backward()
                if Wf_p.grad is not None:
                    Wf_p.grad.masked_fill_(freeze.to(Wf_p.device), 0.0)
                opt_f.step()
                with torch.no_grad():
                    Wf_p.masked_fill_(freeze.to(Wf_p.device), 0.0)
        _, r2_freeze = evaluate(m_f, test_loader, device)

        # Full fine-tune
        m_ft   = copy.deepcopy(m_base)
        opt_ft = optim.Adam(m_ft.parameters(), lr=lr, weight_decay=weight_decay)
        m_ft.train()
        for _ in range(finetune_epochs):
            for Xb, yb in train_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                opt_ft.zero_grad()
                criterion(m_ft(Xb), yb).backward()
                opt_ft.step()
        _, r2_ft = evaluate(m_ft, test_loader, device)

        results['n'].append(n)
        results['baseline'].append(r2_base)
        results['freeze'].append(r2_freeze)
        results['finetune'].append(r2_ft)
        results['T_z_bh'].append(T_z_bh)
        results['kept_frac'].append(n_kept / mask.size)
        print(f'n={n:6d}  base={r2_base:+.3f}  freeze={r2_freeze:+.3f}  '
              f'ft={r2_ft:+.3f}  T_z_bh={T_z_bh:.2f}  kept={100*n_kept/mask.size:.1f}%')

    return results
