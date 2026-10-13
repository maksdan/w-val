#!/usr/bin/env python3
"""
Deep pruning with w-values, applied to Llama.

Adapts experiments/run_deep_pruning.py's three deep-pruning regimes
(simultaneous / sequential-freeze / sequential-unfrozen) to a Llama decoder
model's MLP sublayers. The Llama-loading pattern (AutoModelForCausalLM in
bf16, WikiText-2 perplexity) is carried over from a prior null-analysis
notebook; nothing else from that notebook is reused, and its results are
NOT used as a baseline here -- that experiment is being scrapped.

Design decisions (confirmed with the user before writing this file):
  - Model: meta-llama/Llama-3.2-1B (small enough to iterate on with LoRA).
  - Prunable layers: MLP only -- gate_proj, up_proj, down_proj -- across
    every decoder layer. Attention projections are left alone.
  - Fine-tuning between prune steps: run_deep_pruning.py's `finetune()` does
    full-parameter training, which isn't viable at Llama scale. Here,
    "fine-tune between prune steps" means training a short LoRA adapter
    instead (see MaskedLoRALinear below). Since LoRA freezes 100% of the
    base weights everywhere, the freeze/unfrozen distinction is
    reinterpreted at the adapter level:
      * `run_simultaneous` and the freeze arm of `run_sequential` mask the
        LoRA delta itself at pruned positions on every forward pass, so
        those positions stay pinned at exactly zero -- the adapter-level
        analogue of the tabular version's `frozen_masks` (which zero the
        gradient at pruned positions every step).
      * The unfrozen arm never masks the delta, so pruned connections may
        regrow through the adapter -- the analogue of the tabular version
        calling `finetune()` with no frozen_masks at all.
  - Sequential granularity: one weight matrix per step (gate_proj, then
    up_proj, then down_proj, then the next layer's gate_proj, ...), matching
    run_deep_pruning.py's per-Linear-layer granularity exactly. For a model
    with N decoder layers this is 3*N steps rather than N.
  - Thresholding: both a shared T_z sweep (T_GRID) and a per-matrix BH
    threshold, exactly as in run_deep_pruning.py, but the Beta MLE fit is
    subsampled (FIT_SUBSAMPLE) since each Llama matrix has ~10-100x more
    weights than a tabular hidden layer.

FT_STEPS / T_GRID default to small values -- a full sequential run touches
every prunable matrix (3 per decoder layer), so runtime scales with
3 * N_LAYERS * FT_STEPS. Raise them once the pipeline is confirmed to work
end-to-end.

Run:
    python experiments/run_llama_deep_pruning.py
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# ── CONFIG ────────────────────────────────────────────────────────────────────
SEED          = 0

MODEL_NAME    = 'meta-llama/Llama-3.2-1B'
PRUNE_MODULES = ('gate_proj', 'up_proj', 'down_proj')   # MLP only, every decoder layer

T_GRID        = (1.0, 2.0, 4.0)      # shared per-weight z = (w^2 / sum w^2) * B threshold
BH_ALPHA      = 0.05
RUN_BH        = True
FIT_SUBSAMPLE = 50_000               # weights subsampled for the per-matrix Beta MLE fit

# LoRA "fine-tune between prune steps" config
LORA_RANK     = 8
LORA_ALPHA    = 16.0
FT_STEPS      = 5           # optimizer steps per prune step (per matrix for sequential, once for simultaneous)
FT_LR         = 1e-4
FT_BATCH_SIZE = 2           # batch_size * seq_len drives per-step activation memory --
FT_SEQ_LEN    = 256          # lowered from 4x512 after an MPS OOM that needed ~1 GiB more headroom

# Perplexity eval (WikiText-2, same pattern as the prior notebook)
EVAL_DATASET     = 'wikitext-2-raw-v1'
EVAL_N_TOKENS    = 4096
EVAL_CHUNK_TOKENS = 512   # eval forward passes are chunked to this size to bound peak memory

SAVE_RESULTS  = True
RESULTS_DIR   = 'results_llama_deep_pruning'
# ─────────────────────────────────────────────────────────────────────────────

import gc
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import beta as sp_beta
from tqdm import tqdm

from wvalue.analysis import select_threshold_mse, fit_beta
from wvalue.utils import set_seed

set_seed(SEED)

if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
print(f'Using device: {device}')


# ── Load Llama (pattern carried over from the prior null-analysis notebook) ──

def load_llama(model_name=MODEL_NAME):
    from huggingface_hub import login
    try:
        login()
    except Exception:
        pass  # already authenticated / token cached

    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f'Loading {model_name} ...')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map={'': device},   # pin the whole (small) model to one device --
    )                               # 'auto' was disk-offloading part of it on this machine
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)   # base weights are always frozen; only LoRA deltas train
    # Recompute (rather than retain) each layer's activations during backward,
    # so peak memory doesn't scale with the number of prunable matrices wrapped
    # at once -- without this, the `simultaneous` regime (which wraps every MLP
    # matrix in the model) needs to hold every layer's full-size delta tensor
    # alive simultaneously, which is what caused an MPS OOM in practice.
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.enable_input_require_grads()   # frozen embeddings otherwise break the checkpoint boundary
    print('Loaded.')
    return model, tokenizer


# ── WikiText-2 data: eval (perplexity) + fine-tune (LoRA) ───────────────────

def load_wikitext_ids(tokenizer, split, dataset=EVAL_DATASET):
    from datasets import load_dataset
    # The bare 'wikitext' repo used a loading script, which `datasets` no
    # longer supports; the maintained parquet mirror is under this namespace.
    ds = load_dataset('Salesforce/wikitext', dataset, split=split)
    text = '\n\n'.join(s for s in ds['text'] if s.strip())
    return tokenizer(text, return_tensors='pt').input_ids[0]


def make_ft_blocks(ids, seq_len=FT_SEQ_LEN):
    """Non-overlapping (n_blocks, seq_len) token blocks for causal-LM fine-tuning."""
    n_blocks = len(ids) // seq_len
    return ids[: n_blocks * seq_len].view(n_blocks, seq_len)


@torch.no_grad()
def eval_perplexity(model, eval_ids, chunk_tokens=EVAL_CHUNK_TOKENS):
    """Perplexity over eval_ids, computed in non-overlapping chunks so the
    logits tensor (1, chunk_tokens, vocab_size) never gets big enough to blow
    past the MPS/CUDA memory ceiling in one forward pass. Loss is averaged
    across chunks, weighted by the number of predicted tokens in each."""
    model.eval()
    device_ = next(model.parameters()).device
    n = eval_ids.shape[0]
    total_loss, total_tokens = 0.0, 0
    for start in range(0, n, chunk_tokens):
        chunk = eval_ids[start: start + chunk_tokens]
        if chunk.shape[0] < 2:
            continue
        ids = chunk.unsqueeze(0).to(device_)
        out = model(ids, labels=ids)
        n_pred = chunk.shape[0] - 1   # causal shift drops the last position's label
        total_loss += float(out.loss) * n_pred
        total_tokens += n_pred
    return float(np.exp(total_loss / total_tokens))


# ── Prunable-module discovery ─────────────────────────────────────────────────

def get_prunable_modules_flat(model, matrix_names=PRUNE_MODULES):
    """{dotted_name: nn.Linear} over every decoder layer's target matrices, in
    order (layer 0's gate/up/down_proj, then layer 1's, ...). This ordering
    is also what the sequential regimes step through, one matrix at a time."""
    flat = {}
    for idx, layer in enumerate(model.model.layers):
        for name in matrix_names:
            flat[f'model.layers.{idx}.mlp.{name}'] = getattr(layer.mlp, name)
    return flat


def get_module(model, dotted_name):
    obj = model
    for p in dotted_name.split('.'):
        obj = getattr(obj, p)
    return obj


def set_module(model, dotted_name, new_module):
    *parents, leaf = dotted_name.split('.')
    obj = model
    for p in parents:
        obj = getattr(obj, p)
    setattr(obj, leaf, new_module)


# ── w-value significance + thresholds (same formula as run_deep_pruning.py) ──

def layer_significance(W_t):
    """z = (w^2 / sum(w^2)) * num_weights, for an arbitrary weight matrix."""
    W = W_t.detach().float().cpu().numpy()
    w_sq = W ** 2
    s = w_sq.sum()
    x = w_sq / s if s > 0 else np.full_like(w_sq, 1.0 / w_sq.size)
    return x, x * W.size


def layer_bh_threshold(W_t, alpha=BH_ALPHA, subsample=FIT_SUBSAMPLE, rng_seed=0):
    """Per-matrix Benjamini-Hochberg threshold, in z-space. The Beta MLE fit is
    done on a random subsample of the matrix's weights -- Llama MLP matrices
    are 10-100x larger than the tabular experiment's, so an exact fit is too
    slow."""
    x, z = layer_significance(W_t)
    B = W_t.numel()
    x_flat, z_flat = x.ravel(), z.ravel()
    rng = np.random.default_rng(rng_seed)
    if B > subsample:
        idx = rng.choice(B, size=subsample, replace=False)
        x_sub, z_sub = x_flat[idx], z_flat[idx]
    else:
        x_sub, z_sub = x_flat, z_flat

    T_z_ecdf = select_threshold_mse(z_sub, n_weights=B, verbose=False)[0]
    a_fit, b_fit = fit_beta(x_sub, threshold=T_z_ecdf / B, n_weights=B, verbose=False)
    pvals = sp_beta.sf(x_sub, a_fit, b_fit)
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


def magnitude_keep_mask(W_t, sparsity):
    w = np.abs(W_t.detach().float().cpu().numpy())
    k = max(1, int(round((1 - sparsity) * w.size)))
    thresh = np.partition(w.ravel(), -k)[-k]
    return torch.tensor(w >= thresh, dtype=torch.bool)


def resolve_thresholds(modules, T_z, bh=False, alpha=BH_ALPHA):
    if bh:
        return {name: layer_bh_threshold(lin.weight.data, alpha) for name, lin in modules.items()}
    return {name: T_z for name in modules}


# ── Masked-LoRA linear: base frozen (and pruned), low-rank delta trainable ──

class MaskedLoRALinear(nn.Module):
    """Wraps an already-pruned nn.Linear: y = x @ (W0 + delta)^T + b, where
    W0 is the pruned, frozen base weight and delta = scaling * lora_B @ lora_A
    is a trainable low-rank update (standard LoRA init: lora_B = 0, so the
    initial delta is exactly 0, matching the just-pruned state).

    W0 is NOT cloned into a persistent buffer -- `base_weight` is only a
    reference to the original nn.Linear's weight tensor (never mutated in
    place), and the masked weight is recomputed fresh on every forward pass.
    Wrapping many matrices at once (as `simultaneous` does, across every MLP
    matrix in the model) would otherwise keep a full-size duplicate of every
    wrapped matrix alive at the same time as the untouched originals (which
    stay alive in `original_modules` for restoration), roughly doubling the
    model's weight-memory footprint -- this is what caused an MPS
    out-of-memory crash in practice, on the very first forward pass.

    mask_delta=True also re-masks the delta at pruned positions on every
    forward pass, so those positions can never move (the "freeze" regime).
    mask_delta=False never masks the delta, so pruned positions may regrow
    through the adapter (the "unfrozen" regime).
    """

    def __init__(self, base_linear, keep_mask, rank=LORA_RANK, alpha=LORA_ALPHA, mask_delta=True):
        super().__init__()
        out_f, in_f = base_linear.weight.shape
        dev = base_linear.weight.device
        self.base_weight = base_linear.weight.data   # reference only, no clone
        self.register_buffer('keep_mask', keep_mask.to(dev))
        if base_linear.bias is not None:
            self.register_buffer('bias', base_linear.bias.data.clone())
        else:
            self.bias = None
        self.lora_A = nn.Parameter(torch.zeros(rank, in_f, device=dev, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(out_f, rank, device=dev, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)
        self.scaling = alpha / rank
        self.mask_delta = mask_delta

    def forward(self, x):
        # lora_A/lora_B stay fp32 for training precision (they're tiny), but the
        # matmul's (out_features, in_features) RESULT is downcast to the base's
        # dtype (bf16) immediately -- that full-size result is what autograd has
        # to keep alive for backward, so this halves the dominant memory cost.
        dtype = self.base_weight.dtype
        mask = self.keep_mask.to(dtype)
        delta = ((self.lora_B @ self.lora_A) * self.scaling).to(dtype)
        if self.mask_delta:
            delta = delta * mask
        W_eff = self.base_weight * mask + delta
        return F.linear(x, W_eff, self.bias)

    def sparsity(self):
        return float((~self.keep_mask).float().mean())


# ── Fine-tune loop (shared by all regimes) ───────────────────────────────────

def run_ft_steps(model, opt, ft_blocks, steps, batch_size=FT_BATCH_SIZE, gen=None, pbar=None):
    if opt is None or steps <= 0:
        return
    gen = gen if gen is not None else torch.Generator().manual_seed(SEED)
    dev = next(model.parameters()).device
    n = ft_blocks.shape[0]
    model.train()
    for _ in range(steps):
        idx = torch.randint(0, n, (batch_size,), generator=gen)
        batch = ft_blocks[idx].to(dev)
        opt.zero_grad()
        out = model(batch, labels=batch)
        out.loss.backward()
        opt.step()
        if pbar is not None:
            pbar.set_postfix(loss=f'{out.loss.item():.3f}')
            pbar.update(1)
    model.eval()


def finetune_lora(model, params, ft_blocks, steps=FT_STEPS, lr=FT_LR, desc=None):
    if not params:
        return
    opt = torch.optim.Adam(params, lr=lr)
    with tqdm(total=steps, desc=desc, leave=False) as pbar:
        run_ft_steps(model, opt, ft_blocks, steps, pbar=pbar)


# ── Three deep-pruning regimes (analogues of run_deep_pruning.py) ──────────

def restore_originals(model, flat_modules, original_modules):
    for dotted in flat_modules:
        set_module(model, dotted, original_modules[dotted])
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()


def deep_prune_simultaneous(model, flat_modules, original_modules, thresholds_by_name, ft_blocks,
                             ft_steps=FT_STEPS, lr=FT_LR):
    restore_originals(model, flat_modules, original_modules)
    params, sparsities = [], []
    for dotted, lin in flat_modules.items():
        keep = keep_mask(lin.weight.data, thresholds_by_name[dotted])
        wrapped = MaskedLoRALinear(lin, keep, mask_delta=True)
        set_module(model, dotted, wrapped)
        params += [wrapped.lora_A, wrapped.lora_B]
        sparsities.append(wrapped.sparsity())
    finetune_lora(model, params, ft_blocks, ft_steps, lr, desc='simultaneous')
    return sparsities


def deep_prune_sequential(model, flat_modules, original_modules, thresholds_by_name, ft_blocks,
                           freeze, ft_steps=FT_STEPS, lr=FT_LR):
    """freeze=True pins pruned positions at zero for the rest of the run
    (sequential-freeze). freeze=False never masks the delta, so pruned
    connections may regrow as later matrices are tuned (sequential-unfrozen).
    Steps through one weight matrix at a time (gate_proj, up_proj, down_proj,
    next layer's gate_proj, ...), matching run_deep_pruning.py's
    per-Linear-layer granularity."""
    restore_originals(model, flat_modules, original_modules)
    trainable_params, opt = [], None
    sparsities = []
    desc = 'seq-freeze' if freeze else 'seq-unfrozen'
    for dotted, lin in tqdm(flat_modules.items(), desc=desc, total=len(flat_modules), leave=False):
        keep = keep_mask(lin.weight.data, thresholds_by_name[dotted])
        wrapped = MaskedLoRALinear(lin, keep, mask_delta=freeze)
        set_module(model, dotted, wrapped)
        sparsities.append(wrapped.sparsity())
        step_params = [wrapped.lora_A, wrapped.lora_B]
        trainable_params += step_params
        if opt is None:
            opt = torch.optim.Adam(trainable_params, lr=lr)
        else:
            opt.add_param_group({'params': step_params, 'lr': lr})
        run_ft_steps(model, opt, ft_blocks, ft_steps)
    return sparsities


def magnitude_prune(model, flat_modules, original_modules, sparsities_by_name, ft_blocks,
                     ft_steps=FT_STEPS, lr=FT_LR):
    """Matched-sparsity magnitude-pruning baseline (frozen, like simultaneous)."""
    restore_originals(model, flat_modules, original_modules)
    params = []
    for dotted, lin in flat_modules.items():
        keep = magnitude_keep_mask(lin.weight.data, sparsities_by_name[dotted])
        wrapped = MaskedLoRALinear(lin, keep, mask_delta=True)
        set_module(model, dotted, wrapped)
        params += [wrapped.lora_A, wrapped.lora_B]
    finetune_lora(model, params, ft_blocks, ft_steps, lr, desc='magnitude')


# ── Driver ────────────────────────────────────────────────────────────────────

def run_threshold_trial(model, flat_modules, original_modules, ft_blocks, eval_ids,
                         T_z=None, bh=False, alpha=BH_ALPHA):
    thresholds_by_name = resolve_thresholds(flat_modules, T_z, bh=bh, alpha=alpha)
    label = 'BH' if bh else f'T_z={T_z:.2f}'
    print(f'-- {label} --')

    print('  simultaneous...')
    sp_sim = deep_prune_simultaneous(model, flat_modules, original_modules, thresholds_by_name, ft_blocks)
    ppl_sim = eval_perplexity(model, eval_ids)

    print('  sequential (freeze)...')
    sp_freeze = deep_prune_sequential(model, flat_modules, original_modules, thresholds_by_name, ft_blocks, freeze=True)
    ppl_freeze = eval_perplexity(model, eval_ids)

    print('  sequential (unfrozen)...')
    sp_unfrozen = deep_prune_sequential(model, flat_modules, original_modules, thresholds_by_name, ft_blocks, freeze=False)
    ppl_unfrozen = eval_perplexity(model, eval_ids)

    print('  magnitude...')
    sparsities_by_name = dict(zip(flat_modules.keys(), sp_sim))
    magnitude_prune(model, flat_modules, original_modules, sparsities_by_name, ft_blocks)
    ppl_mag = eval_perplexity(model, eval_ids)

    restore_originals(model, flat_modules, original_modules)

    pruned_pct = 100.0 * float(np.mean(sp_sim))
    print(f'  {label:10s}  pruned={pruned_pct:5.1f}%  '
          f'simul={ppl_sim:8.3f}  freeze={ppl_freeze:8.3f}  '
          f'unfrozen={ppl_unfrozen:8.3f}  magnitude={ppl_mag:8.3f}')

    return {
        'T_z': T_z, 'bh': bh, 'pruned_pct': pruned_pct,
        'simul': ppl_sim, 'seq_freeze': ppl_freeze,
        'seq_unfrozen': ppl_unfrozen, 'magnitude': ppl_mag,
    }


RESULTS_PATH = os.path.join(RESULTS_DIR, 'llama_deep_pruning_results.json')


def trial_key(T_z, bh):
    return 'BH' if bh else f'T_z={T_z}'


def load_checkpoint(path=RESULTS_PATH):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def save_checkpoint(results, path=RESULTS_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)


def run_sweep(model, tokenizer):
    """Runs the threshold sweep, checkpointing to RESULTS_PATH after every
    threshold trial (not just at the end). Rerunning the script picks up
    from RESULTS_PATH and skips any threshold already recorded there, so an
    interrupted sweep (Ctrl-C, closed terminal, sleep, crash) can be resumed
    instead of starting over. A threshold trial (all 4 regimes at one T_z or
    BH) is the resume granularity -- interrupting mid-trial still loses that
    one trial's progress, since it's one continuous prune-finetune-eval run."""
    flat_modules = get_prunable_modules_flat(model)
    original_modules = dict(flat_modules)

    print('Loading WikiText-2 train (fine-tune) and test (eval) splits...')
    train_ids = load_wikitext_ids(tokenizer, split='train')
    ft_blocks = make_ft_blocks(train_ids)
    eval_ids = load_wikitext_ids(tokenizer, split='test')[:EVAL_N_TOKENS]
    print(f'FT blocks: {ft_blocks.shape}   Eval tokens: {eval_ids.shape[0]:,}')

    results = load_checkpoint() if SAVE_RESULTS else None
    if results is not None:
        done = {trial_key(r['T_z'], r['bh']) for r in results['trials']}
        print(f'Resuming from {RESULTS_PATH}  '
              f'({len(done)} threshold(s) already done: {sorted(done)})')
        print(f'Baseline (dense) perplexity: {results["baseline"]:.4f}\n')
    else:
        baseline_ppl = eval_perplexity(model, eval_ids)
        print(f'Baseline (dense) perplexity: {baseline_ppl:.4f}\n')
        results = {'baseline': baseline_ppl, 'trials': []}
        done = set()
        if SAVE_RESULTS:
            save_checkpoint(results)

    todo = [(T_z, False) for T_z in T_GRID]
    if RUN_BH:
        todo.append((None, True))

    print(f'Threshold sweep over {len(T_GRID)} T_z values'
          f'{" + BH" if RUN_BH else ""}  (ft_steps={FT_STEPS}, lora_rank={LORA_RANK})')
    for T_z, bh in todo:
        key = trial_key(T_z, bh)
        if key in done:
            print(f'  {key:10s}  already done, skipping')
            continue
        row = run_threshold_trial(model, flat_modules, original_modules, ft_blocks, eval_ids, T_z=T_z, bh=bh)
        results['trials'].append(row)
        done.add(key)
        if SAVE_RESULTS:
            save_checkpoint(results)
            print(f'  (checkpoint saved -> {RESULTS_PATH})')

    return results


def plot_sweep(results, save=SAVE_RESULTS, plot_dir=RESULTS_DIR):
    import matplotlib
    if save:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    grid_trials = [r for r in results['trials'] if not r['bh']]
    bh_trials = [r for r in results['trials'] if r['bh']]

    fig, ax = plt.subplots(figsize=(8, 6))
    Tz = [r['T_z'] for r in grid_trials]
    style = {
        'simul':        ('simultaneous', 'tab:blue', 'o-'),
        'seq_freeze':   ('sequential (freeze)', 'tab:orange', 's--'),
        'seq_unfrozen': ('sequential (unfrozen)', 'tab:green', '^-.'),
        'magnitude':    ('magnitude (matched sparsity)', 'tab:red', 'x:'),
    }
    ax.axhline(results['baseline'], color='k', ls=':', lw=1, label='dense baseline')
    for key, (lbl, c, st) in style.items():
        ax.plot(Tz, [r[key] for r in grid_trials], st, color=c, label=lbl, ms=5)
    if bh_trials:
        for key, (lbl, c, _) in style.items():
            ax.axhline(bh_trials[0][key], color=c, ls=':', lw=1, alpha=0.5)
    ax.set_xlabel('threshold T_z')
    ax.set_ylabel('perplexity (WikiText-2)')
    ax.set_yscale('log')
    ax.set_title(f'Deep pruning — {MODEL_NAME}\nMLP (gate/up/down_proj), all decoder layers')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, linestyle='--')
    fig.tight_layout()

    if save:
        os.makedirs(plot_dir, exist_ok=True)
        out_path = os.path.join(plot_dir, 'llama_deep_pruning_sweep.png')
        fig.savefig(out_path, dpi=150)
        print(f'\nSaved figure -> {out_path}')
    else:
        plt.show()


if __name__ == '__main__':
    if SAVE_RESULTS:
        os.makedirs(RESULTS_DIR, exist_ok=True)

    model, tokenizer = load_llama()
    results = run_sweep(model, tokenizer)   # checkpoints to RESULTS_PATH after each threshold trial
    plot_sweep(results)
