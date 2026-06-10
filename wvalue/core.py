"""
W-value (significance) computation for regularization.

Provides the significance regularizer term: sum of BetaSF(normalized_weight²)
over all linear layer weights, with optional lookup table for speed.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.special import betainc as scipy_betainc
from scipy.stats import beta


class _BetaSFAutograd(torch.autograd.Function):
    """
    Beta survival function SF(x) = 1 - I(x; a, b) with differentiable backward.
    Forward: uses lookup table if provided (fast), else scipy.special.betainc.
    Backward: uses lookup table for PDF if one was used in forward, else scipy.stats.beta.pdf.
    """

    @staticmethod
    def forward(ctx, x, alpha, beta_param, lookup_table, nearest):
        x_np = x.detach().cpu().double().numpy()
        x_np = np.clip(x_np, 1e-12, 1.0 - 1e-12)
        ctx.save_for_backward(x)
        ctx.alpha, ctx.beta_param = float(alpha), float(beta_param)
        ctx.lookup_table = lookup_table
        ctx.nearest = nearest
        if lookup_table is not None:
            sf_np = lookup_table.lookup(x_np, beta_param, nearest=nearest)
        else:
            sf_np = 1.0 - scipy_betainc(alpha, beta_param, x_np)
        return torch.as_tensor(sf_np, device=x.device, dtype=x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        a, b = ctx.alpha, ctx.beta_param
        x_np = x.detach().cpu().double().numpy()
        x_np = np.clip(x_np, 1e-12, 1.0 - 1e-12)
        lookup_table = getattr(ctx, 'lookup_table', None)
        nearest = getattr(ctx, 'nearest', False)
        if lookup_table is not None:
            pdf_np = lookup_table.lookup_pdf(x_np, b, nearest=nearest)
        else:
            pdf_np = beta.pdf(x_np, a, b)
        pdf = torch.as_tensor(pdf_np, device=x.device, dtype=x.dtype)
        return -grad_output * pdf, None, None, None, None


def _beta_sf(x, alpha, beta_param, lookup_table=None, nearest=False):
    """Element-wise Beta SF(x) = 1 - I(x; alpha, beta), differentiable w.r.t. x."""
    return _BetaSFAutograd.apply(x, alpha, beta_param, lookup_table, nearest)


def significance_regularizer_term(model, beta_sf_lookup_table=None, term_scale='mean'):
    """
    Compute the significance regularizer (Beta SF of normalized weight²) for all linear weights.
    For each linear layer: Frobenius-normalize the full weight matrix to unit norm, then sum BetaSF(x) for x = normed².
    Uses alpha=1/2, beta=(B-1)/2 with B = out_features * in_features per layer.

    Args:
        term_scale: 'sum' = raw sum over all entries (original behavior); 'mean' = divide by total number of
            weight entries (linear layers only), so the term is mean BetaSF per weight and lambda is comparable across models.
    """
    table = beta_sf_lookup_table if beta_sf_lookup_table is not None else beta_sf_lookup
    total = 0.0
    total_entries = 0
    for module in model.modules():
        if not isinstance(module, nn.Linear):
            continue
        w = module.weight
        n_entries = w.numel()
        B = n_entries
        w_normed = F.normalize(w.view(1, -1), p=2, dim=1).view_as(w)
        x = w_normed ** 2
        x = torch.clamp(x, min=1e-7, max=1.0 - 1e-7)
        alpha = 0.5
        beta_param = (B - 1) / 2.0
        sf = _beta_sf(x, alpha, beta_param, lookup_table=table)
        total = total + sf.sum()
        total_entries += n_entries
    if term_scale == 'mean' and total_entries > 0:
        total = total / total_entries
    return total


class BetaSFLookupTable:
    """Lookup table for Beta SF and PDF to speed up forward and backward passes."""

    def __init__(self, resolution=10000, min_x=1e-10, max_x=1 - 1e-10):
        self.resolution = resolution
        self.min_x = min_x
        self.max_x = max_x
        self.tables = {}
        self.pdf_tables = {}
        self.x_values = np.linspace(min_x, max_x, resolution)
        self._n_forward_lookups = 0
        self._n_backward_lookups = 0

    def get_table(self, beta_param):
        beta_key = round(beta_param, 6)
        if beta_key not in self.tables:
            alpha = 0.5
            self.tables[beta_key] = beta.sf(self.x_values, alpha, beta_param)
        return self.tables[beta_key]

    def get_pdf_table(self, beta_param):
        beta_key = round(beta_param, 6)
        if beta_key not in self.pdf_tables:
            alpha = 0.5
            self.pdf_tables[beta_key] = beta.pdf(self.x_values, alpha, beta_param)
        return self.pdf_tables[beta_key]

    def _interpolate(self, x_clamped, table):
        indices = (x_clamped - self.min_x) / (self.max_x - self.min_x) * (self.resolution - 1)
        indices = np.clip(indices, 0, self.resolution - 1)
        lower_idx = np.floor(indices).astype(int)
        upper_idx = np.minimum(lower_idx + 1, self.resolution - 1)
        frac = indices - lower_idx
        return (1 - frac) * table[lower_idx] + frac * table[upper_idx]

    def _nearest(self, x_clamped, table):
        indices = (x_clamped - self.min_x) / (self.max_x - self.min_x) * (self.resolution - 1)
        idx = np.clip(np.rint(indices), 0, self.resolution - 1).astype(int)
        return table[idx]

    def lookup(self, x_values, beta_param, nearest=False):
        self._n_forward_lookups += 1
        x_clamped = np.clip(x_values, self.min_x, self.max_x)
        table = self.get_table(beta_param)
        return self._nearest(x_clamped, table) if nearest else self._interpolate(x_clamped, table)

    def lookup_pdf(self, x_values, beta_param, nearest=False):
        self._n_backward_lookups += 1
        x_clamped = np.clip(x_values, self.min_x, self.max_x)
        table = self.get_pdf_table(beta_param)
        return self._nearest(x_clamped, table) if nearest else self._interpolate(x_clamped, table)

    def usage_counts(self):
        return (getattr(self, '_n_forward_lookups', 0), getattr(self, '_n_backward_lookups', 0))


# Global lookup table; set in notebook if desired (e.g. BetaSFLookupTable(resolution=50000))
beta_sf_lookup = None


def compute_w_value(weight_matrix, use_lookup=True, beta_sf_lookup_table=None):
    """
    Compute w-values and significance for a weight matrix.
    Frobenius-normalize the entire matrix to unit norm; for each normalized weight squared x,
    w_value = BetaSF(x), significance = -ln(w_value), with B = out_features * in_features.
    Returns (w_values, significance) tensors same shape as weight_matrix.
    """
    table = beta_sf_lookup_table if beta_sf_lookup_table is not None else beta_sf_lookup
    B = weight_matrix.numel()
    weight_normed = F.normalize(weight_matrix.view(1, -1), p=2, dim=1).view_as(weight_matrix)
    weight_squared = weight_normed ** 2
    weight_squared = torch.clamp(weight_squared, min=1e-10, max=1.0 - 1e-10)
    alpha = 0.5
    beta_param = (B - 1) / 2.0
    weight_squared_np = weight_squared.detach().cpu().numpy()
    if use_lookup and table is not None:
        w_values_np = table.lookup(weight_squared_np, beta_param)
    else:
        w_values_np = beta.sf(weight_squared_np, alpha, beta_param)
    w_values = torch.as_tensor(w_values_np, device=weight_matrix.device, dtype=weight_matrix.dtype)
    significance = -torch.log(w_values.clamp(min=torch.finfo(w_values.dtype).tiny))
    return w_values, significance


def compute_w_values_for_model(model, beta_sf_lookup_table=None):
    """Compute w-values and significance for all linear layers (Frobenius per layer).

    Returns dict layer_name -> (w_values, significance).
    """
    result = {}

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            w_values, significance = compute_w_value(
                module.weight.data, use_lookup=True,
                beta_sf_lookup_table=beta_sf_lookup_table,
            )
            result[name] = (w_values, significance)
    return result


def compute_and_cache_significance(model, device, beta_sf_lookup_table=None):
    """
    Compute significance (-ln(w_value)) for all linear layer weights (Frobenius-normalized per layer).

    Returns dict layer_name -> significance tensor (same shape as layer weight).
    Used as cached_significance for train_with_wvalue_filtering.
    """
    table = beta_sf_lookup_table if beta_sf_lookup_table is not None else beta_sf_lookup
    cached = {}
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        w = module.weight.data
        _, significance = compute_w_value(w, use_lookup=True, beta_sf_lookup_table=table)
        cached[name] = significance.to(device)
    return cached


def train_with_wvalue_filtering(model, train_loader, criterion, optimizer, device,
                                significance_threshold=2.0, cached_significance=None,
                                filtering_type='gradient',
                                cache_mask_every_batch=False):
    """
    One epoch of training with w-value filtering.
    significance_threshold is in -ln(w) space (higher = more selective).
    cached_significance: dict layer_name -> significance tensor (from compute_and_cache_significance).
    cache_mask_every_batch: if True and filtering_type=='weight', recompute significance (and thus mask)
        at every batch from current weights; otherwise use passed-in cached_significance. Ignored for gradient filtering.
    Significance uses Frobenius normalization per layer (same as compute_and_cache_significance).
    Returns (epoch_loss, epoch_acc).
    """
    from tqdm import tqdm
    if cached_significance is None:
        cached_significance = compute_and_cache_significance(model, device, beta_sf_lookup_table=beta_sf_lookup)
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in tqdm(train_loader, desc="W-value train", leave=False):
        # For weight filtering: optionally recompute mask from current weights every batch
        if filtering_type == 'weight' and cache_mask_every_batch:
            cached_significance = compute_and_cache_significance(
                model, device,
                beta_sf_lookup_table=beta_sf_lookup,
            )
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        # Weight mask: zero out insignificant weights before forward (pruning-style)
        if filtering_type == 'weight':
            with torch.no_grad():
                for name, module in model.named_modules():
                    if isinstance(module, nn.Linear) and name in cached_significance:
                        mask = (cached_significance[name] >= significance_threshold).to(module.weight.dtype)
                        module.weight.data.mul_(mask)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        # Mask gradients: only significant weights get updated (both modes need this so weight mask stays fixed)
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and name in cached_significance and module.weight.grad is not None:
                mask = (cached_significance[name] >= significance_threshold).to(module.weight.grad.dtype)
                module.weight.grad.data.mul_(mask)
        optimizer.step()
        # Re-apply weight mask after step so pruned weights stay zero (weight mode only)
        if filtering_type == 'weight':
            with torch.no_grad():
                for name, module in model.named_modules():
                    if isinstance(module, nn.Linear) and name in cached_significance:
                        mask = (cached_significance[name] >= significance_threshold).to(module.weight.dtype)
                        module.weight.data.mul_(mask)
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc
