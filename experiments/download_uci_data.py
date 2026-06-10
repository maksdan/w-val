#!/usr/bin/env python3
"""
Download UCI datasets for use with run_broad_eval.py.

Usage:
    python experiments/download_uci_data.py

This script checks which UCI classification datasets are already present in
../UCI_data/ (relative to this script, i.e. UCI_data/ at the repo root),
and prints instructions for obtaining any that are missing.

The download uses the KEEL repository format (_R.dat files). Since KEEL
does not provide a bulk download API, datasets must be obtained manually
from:  https://sci2s.ugr.es/keel/datasets.php

Each dataset should be placed as a folder under UCI_data/:
    UCI_data/
      iris/
        iris.txt          (metadata)
        iris_R.dat        (data with 'clase' column)
        conxuntos.dat     (train/test split indices)
      wine/
        wine.txt
        wine_R.dat
        conxuntos.dat
      ...

These datasets download automatically when first used (via sklearn/torchvision):
  - MNIST, FashionMNIST, CIFAR-10, SVHN  -> downloaded by run_broad_eval.py
  - Covertype, 20 Newsgroups             -> downloaded via sklearn.datasets
"""

# ── Path setup ────────────────────────────────────────────────────────────────
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

UCI_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'UCI_data'))

# ── Commonly used UCI datasets ────────────────────────────────────────────────
# These are datasets that work well for classification experiments.
# The full set of ~122 datasets supported by load_uci_dataset() can be used
# once you have the KEEL-format files.
RECOMMENDED_DATASETS = [
    'iris', 'wine', 'glass', 'ionosphere', 'sonar', 'heart-statlog',
    'balance', 'bupa', 'pima', 'vehicle', 'vowel', 'spambase',
    'australian', 'german', 'splice', 'letter', 'satimage',
    'wdbc', 'yeast', 'abalone-19', 'banana', 'magic', 'poker',
]


def check_uci_datasets(uci_dir=UCI_DATA_DIR):
    """Check which datasets are present and which are missing."""
    if not os.path.isdir(uci_dir):
        print(f'UCI_data directory not found: {uci_dir}')
        print('Create this directory and place dataset folders inside it.')
        return [], RECOMMENDED_DATASETS

    present = []
    missing = []
    for name in RECOMMENDED_DATASETS:
        dataset_dir = os.path.join(uci_dir, name)
        if os.path.isdir(dataset_dir):
            # Check for the main data file
            has_data = (
                os.path.exists(os.path.join(dataset_dir, f'{name}_R.dat')) or
                os.path.exists(os.path.join(dataset_dir, f'{name}_train_R.dat'))
            )
            if has_data:
                present.append(name)
            else:
                missing.append(name)
        else:
            missing.append(name)
    return present, missing


def list_all_available(uci_dir=UCI_DATA_DIR):
    """List all dataset folders that exist in UCI_data/."""
    if not os.path.isdir(uci_dir):
        return []
    return sorted([
        d for d in os.listdir(uci_dir)
        if os.path.isdir(os.path.join(uci_dir, d))
    ])


def download_sklearn_datasets():
    """Download datasets available via scikit-learn (no manual steps needed)."""
    print('\nDownloading sklearn/torchvision datasets...')
    print('(These are used by run_broad_eval.py automatically, but we download them now.)')

    try:
        from sklearn.datasets import fetch_covtype
        print('  Downloading Covertype via sklearn... ', end='', flush=True)
        fetch_covtype(data_home='./data', download_if_missing=True)
        print('done.')
    except Exception as e:
        print(f'failed: {e}')

    try:
        from sklearn.datasets import fetch_20newsgroups
        print('  Downloading 20 Newsgroups via sklearn... ', end='', flush=True)
        fetch_20newsgroups(subset='train', data_home='./data', download_if_missing=True)
        fetch_20newsgroups(subset='test',  data_home='./data', download_if_missing=True)
        print('done.')
    except Exception as e:
        print(f'failed: {e}')

    print('\nNote: MNIST, FashionMNIST, CIFAR-10, SVHN download automatically')
    print('      when run_broad_eval.py first runs them (via torchvision).')


if __name__ == '__main__':
    print('W-Value Dataset Setup')
    print('=' * 60)

    # Check UCI datasets
    print(f'\nUCI_data directory: {UCI_DATA_DIR}')
    present, missing = check_uci_datasets()
    all_available = list_all_available()

    if all_available:
        print(f'\nAll available UCI datasets ({len(all_available)} total):')
        for i, name in enumerate(all_available):
            print(f'  {name}')
    else:
        print('\nNo UCI datasets found in UCI_data/.')

    print(f'\nRecommended datasets present   : {len(present)} / {len(RECOMMENDED_DATASETS)}')
    if present:
        print('  Present:', ', '.join(present[:10]) + (f' ... +{len(present)-10}' if len(present) > 10 else ''))

    if missing:
        print(f'\nRecommended datasets missing   : {len(missing)}')
        print('  Missing:', ', '.join(missing[:10]) + (f' ... +{len(missing)-10}' if len(missing) > 10 else ''))
        print()
        print('To download missing UCI datasets:')
        print('  1. Go to: https://sci2s.ugr.es/keel/datasets.php')
        print('  2. Download datasets in KEEL format (.zip files)')
        print(f'  3. Extract each zip into: {UCI_DATA_DIR}/')
        print('     Each dataset should create a folder, e.g.:')
        print(f'       {UCI_DATA_DIR}/iris/iris_R.dat')
        print(f'       {UCI_DATA_DIR}/iris/conxuntos.dat')
        print()
        print('The UCI_data/ directory should be at the repo root (same level as wvalue/).')
    else:
        print('\nAll recommended datasets are present.')

    # Download sklearn datasets
    print()
    answer = input('Download Covertype and 20 Newsgroups via sklearn now? [y/N] ').strip().lower()
    if answer == 'y':
        download_sklearn_datasets()
    else:
        print('Skipped sklearn download. Run again and answer y, or let run_broad_eval.py download them.')

    print('\nSetup check complete.')
