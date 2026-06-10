"""High-level experiment runner for broad multi-dataset evaluation (notebook 04)."""

import math
import time
import torch
import torch.nn as nn
import torch.optim as optim

from wvalue.datasets import load_dataset, get_mlp_class, load_uci_dataset
from wvalue.training import train_epoch, evaluate, train_with_l2_regularization
from wvalue.core import compute_and_cache_significance, train_with_wvalue_filtering
from wvalue.analysis import capture_snapshot
from wvalue.utils import set_seed


def run_broad_eval(
    datasets,
    device,
    seed=42,
    run_baseline=True,
    run_l2=False,
    l2_strength=1e-4,
    run_wval_filter=True,
    wval_prob_thresholds=None,
    wval_update_interval=1,
    epochs_override=None,
    lr_override=None,
    batch_size_override=None,
    snapshot_n=5,
    print_interval=10,
    lookup_table=None,
):
    """Train baseline, L2, and w-value-filtered models on a list of datasets with snapshots.

    Args:
        datasets: Resolved list of dataset names (e.g. ['MNIST', 'uci:iris']).
        device: torch.device.
        seed: Random seed for model initialisation.
        run_baseline: Train an unregularised baseline.
        run_l2: Train an L2-regularised variant.
        l2_strength: L2 lambda.
        run_wval_filter: Train w-value-filtered variants.
        wval_prob_thresholds: List of probability thresholds; significance = -ln(p).
            Defaults to [0.90, 0.50].
        wval_update_interval: Recompute cached significance every N epochs (Frobenius per layer).
        epochs_override: Override each dataset's default epoch count.
        lr_override: Override each dataset's default learning rate.
        batch_size_override: Override batch size for UCI loaders.
        snapshot_n: Number of snapshot intervals; produces N+1 snapshots (0% ... 100%).
        print_interval: Print epoch progress every N epochs.
        lookup_table: BetaSFLookupTable instance; falls back to module global if None.

    Returns:
        (all_dataset_results, failed_datasets)
        all_dataset_results: dict  dataset_name -> list of result dicts
        failed_datasets:     list of (dataset_name, error_string) pairs
    """
    import wvalue.core as _core

    if wval_prob_thresholds is None:
        wval_prob_thresholds = [0.90, 0.50]
    table = lookup_table if lookup_table is not None else _core.beta_sf_lookup

    wval_thresholds = [-math.log(p) for p in wval_prob_thresholds] if run_wval_filter else []
    wval_labels     = [f'w<{p:g}' for p in wval_prob_thresholds]   if run_wval_filter else []

    MLP = get_mlp_class()
    all_dataset_results = {}
    failed_datasets     = []

    for ds_idx, ds_name in enumerate(datasets):
        print(f'\n{"#"*70}')
        print(f'# Dataset {ds_idx+1}/{len(datasets)}: {ds_name}')
        print(f'{"#"*70}')

        try:
            if ds_name.startswith('uci:'):
                bs = batch_size_override or 64
                train_loader, test_loader, dataset_config = load_uci_dataset(
                    ds_name[4:], batch_size=bs)
            else:
                train_loader, test_loader, dataset_config = load_dataset(ds_name)
        except Exception as e:
            print(f'  SKIPPING: {e}')
            failed_datasets.append((ds_name, str(e)))
            continue

        input_size   = dataset_config['input_size']
        num_classes  = dataset_config['num_classes']
        hidden_sizes = dataset_config['hidden_sizes']
        num_epochs   = epochs_override or dataset_config.get('epochs', 50)
        lr           = lr_override     or dataset_config.get('learning_rate', 0.001)
        criterion    = nn.CrossEntropyLoss()

        epoch_to_labels = {}
        for k in range(1, snapshot_n + 1):
            pct   = round(k * 100 / snapshot_n)
            ep    = min(num_epochs, max(1, round(k * num_epochs / snapshot_n)))
            epoch_to_labels.setdefault(ep, []).append(f'{pct}%')

        schedule_str = '0% (init)' + ''.join(
            f' -> {"/".join(lbls)} @ep{ep}'
            for ep, lbls in sorted(epoch_to_labels.items())
        )
        print(f'  Snapshot schedule: {schedule_str}')
        print(f'  Architecture: {input_size} -> {hidden_sizes} -> {num_classes}')
        print(f'  Epochs: {num_epochs}, LR: {lr}')

        def _make_model():
            set_seed(seed)
            model = MLP(input_size, hidden_sizes, num_classes).to(device)
            for module in model.modules():
                if isinstance(module, nn.Linear):
                    nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                    nn.init.zeros_(module.bias)
            return model

        def _run_one(name, train_fn, threshold=None, probability=None):
            model     = _make_model()
            optimizer = optim.Adam(model.parameters(), lr=lr)
            snapshots = {'0%': capture_snapshot(model, table)}
            tl, ta, vl, va = [], [], [], []
            extra = {}
            t0    = time.time()
            for epoch in range(num_epochs):
                tl_e, ta_e = train_fn(model, optimizer, epoch, extra)
                vl_e, va_e = evaluate(model, test_loader, criterion, device)
                tl.append(tl_e); ta.append(ta_e); vl.append(vl_e); va.append(va_e)
                if (epoch + 1) in epoch_to_labels:
                    snap = capture_snapshot(model, table)
                    for lbl in epoch_to_labels[epoch + 1]:
                        snapshots[lbl] = snap
                if (epoch + 1) % print_interval == 0 or epoch == 0:
                    print(f'    Epoch {epoch+1}/{num_epochs}  '
                          f'train_acc={ta_e:.2f}%  test_acc={va_e:.2f}%')
            elapsed = time.time() - t0
            print(f'    Final test acc: {va[-1]:.2f}%')
            entry = {
                'name': name, 'train_losses': tl, 'train_accs': ta,
                'test_losses': vl, 'test_accs': va, 'final_acc': va[-1],
                'time_taken': elapsed, 'snapshots': snapshots, 'threshold': threshold,
            }
            if probability is not None:
                entry['probability'] = probability
            return entry

        results = []

        if run_baseline:
            print(f'\n  === Baseline ===')
            def _baseline(model, optimizer, epoch, extra):
                return train_epoch(model, train_loader, criterion, optimizer, device)
            results.append(_run_one('Baseline', _baseline))

        if run_l2:
            print(f'\n  === L2 (lambda={l2_strength:.0e}) ===')
            def _l2(model, optimizer, epoch, extra):
                return train_with_l2_regularization(
                    model, train_loader, criterion, optimizer, device,
                    l2_strength=l2_strength)
            results.append(_run_one(f'L2 {l2_strength:.0e}', _l2))

        if run_wval_filter:
            for prob, thresh, label in zip(wval_prob_thresholds, wval_thresholds, wval_labels):
                print(f'\n  === {label} (threshold={thresh:.4f}) ===')
                def _wval(model, optimizer, epoch, extra, _t=thresh):
                    if epoch % wval_update_interval == 0:
                        extra['sig'] = compute_and_cache_significance(
                            model, device,
                            beta_sf_lookup_table=table,
                        )
                    return train_with_wvalue_filtering(
                        model, train_loader, criterion, optimizer, device,
                        significance_threshold=_t,
                        cached_significance=extra['sig'],
                        filtering_type='weight',
                        cache_mask_every_batch=False,
                    )
                results.append(_run_one(label, _wval, threshold=thresh, probability=prob))

        all_dataset_results[ds_name] = results

    print(f'\n\n{"="*70}')
    n_ok   = len(all_dataset_results)
    n_fail = len(failed_datasets)
    print(f'All experiments complete. {n_ok} datasets succeeded, {n_fail} failed.')
    if failed_datasets:
        print('\nFailed datasets:')
        for name, err in failed_datasets:
            print(f'  {name}: {err}')

    return all_dataset_results, failed_datasets
