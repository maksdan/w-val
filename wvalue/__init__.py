"""
wvalue — W-value significance framework for neural network pruning and regularization.

Key idea: after Frobenius-normalizing a weight matrix, each weight's squared value follows
Beta(0.5, (B-1)/2) under the null hypothesis (random initialization). The "w-value" is
Beta.sf(x) — a p-value. This codebase uses w-values to regularize or prune networks.

Package structure:
  wvalue.core        — BetaSFLookupTable, compute_w_value, significance_regularizer_term,
                       train_with_wvalue_filtering, compute_and_cache_significance
  wvalue.training    — training loops: baseline, L1/L2/significance regularizers
  wvalue.utils       — set_seed (reproducibility)
  wvalue.datasets    — classification datasets (MNIST, UCI, etc.) + regression DATASET_CASES
  wvalue.analysis    — null distribution analysis (Beta fitting, BH pruning) +
                       weight snapshot capture (capture_snapshot, generate_null_significance)
  wvalue.regression  — MLP, make_model, make_data, evaluate, capture, train_regression,
                       run_pruning_experiments, threshold_sweep, sample_size_sweep
  wvalue.broad_eval  — run_broad_eval (multi-dataset classification experiment runner)
"""

from .core import (
    BetaSFLookupTable,
    compute_w_value,
    compute_w_values_for_model,
    significance_regularizer_term,
    train_with_wvalue_filtering,
    compute_and_cache_significance,
    beta_sf_lookup,
)
from .training import (
    train_epoch,
    evaluate,
    train_baseline_model,
    train_with_l1_regularization,
    train_with_l2_regularization,
    train_with_significance_regularization,
)
from .utils import set_seed
from .datasets import (
    load_dataset,
    get_mlp_class,
    load_uci_dataset,
    list_uci_datasets,
    make_modular_arithmetic,
    DATASET_CONFIGS,
    DATASET_CASES,
)
from .analysis import (
    capture_snapshot,
    generate_null_significance,
    qq_obs_and_null,
    run_null_analysis,
    bh_threshold,
    build_null_sorted,
    fit_beta,
    select_threshold_mse,
    qq_beta,
    qq_wvalue,
)
from .broad_eval import run_broad_eval
from .regression import run_pruning_experiments, threshold_sweep, sample_size_sweep

__all__ = [
    # core
    "BetaSFLookupTable", "compute_w_value", "compute_w_values_for_model",
    "significance_regularizer_term", "train_with_wvalue_filtering",
    "compute_and_cache_significance", "beta_sf_lookup",
    # training
    "train_epoch", "evaluate", "train_baseline_model",
    "train_with_l1_regularization", "train_with_l2_regularization",
    "train_with_significance_regularization",
    # utils
    "set_seed",
    # datasets — classification
    "load_dataset", "get_mlp_class", "load_uci_dataset",
    "list_uci_datasets", "make_modular_arithmetic", "DATASET_CONFIGS",
    # datasets — regression
    "DATASET_CASES",
    # analysis — null distribution
    "run_null_analysis", "bh_threshold", "build_null_sorted",
    "fit_beta", "select_threshold_mse", "qq_beta", "qq_wvalue",
    # analysis — snapshots
    "capture_snapshot", "generate_null_significance", "qq_obs_and_null",
    # high-level runners
    "run_broad_eval",
    "run_pruning_experiments", "threshold_sweep", "sample_size_sweep",
]
