"""
Dataset loaders for w-value experiments.

This module merges two dataset sources into one file:

Section 1 — Classification datasets (originally from data.py):
  - get_mlp_class()          — MLP factory for classification
  - DATASET_CONFIGS          — per-dataset epoch/arch/transform configs
  - load_dataset(name)       — load a built-in classification dataset
  - load_uci_dataset(name)   — load a UCI dataset from the UCI_data/ folder
  - list_uci_datasets()      — list available UCI dataset names
  - make_modular_arithmetic() — grokking dataset (a*b mod p)
  - UCI_DATA_DIR             — path to UCI data folder

Section 2 — Regression datasets (originally from datasets.py):
  - DATASET_CASES            — synthetic + real regression tasks for notebooks 05/06
  - Individual loaders: _load_california_housing, _load_diabetes, _load_wine_quality,
    _load_concrete, _load_abalone, _load_power_plant, _load_energy_efficiency, _load_auto_mpg
  - OpenML helpers: _openml_fetch, _std_split, _xy_from_openml
"""

# ══════════════════════════════════════════════════════════════════════════════
# Section 1 — Classification datasets
# ══════════════════════════════════════════════════════════════════════════════

import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from sklearn.datasets import fetch_covtype, fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
from torch.utils.data import TensorDataset
import os
import pandas as pd


def get_mlp_class():
    """Factory function to return MLP model class"""
    import torch.nn as nn

    class MLP(nn.Module):
        def __init__(self, input_size, hidden_sizes, num_classes,
                     activation_type='relu', wval_threshold=2.0):
            super(MLP, self).__init__()
            layers = []
            prev_size = input_size
            for hidden_size in hidden_sizes:
                layers.append(nn.Linear(prev_size, hidden_size))
                if activation_type == 'gelu':
                    layers.append(nn.GELU())
                elif activation_type == 'leaky_relu':
                    layers.append(nn.LeakyReLU())
                else:  # 'relu' default
                    layers.append(nn.ReLU())
                prev_size = hidden_size
            layers.append(nn.Linear(prev_size, num_classes))
            self.network = nn.Sequential(*layers)

        def forward(self, x):
            x = x.view(x.size(0), -1)
            return self.network(x)

    return MLP


# Dataset configurations
# Each config can include learning_rate and batch_size; defaults below are used if omitted.
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_BATCH_SIZE = 64

DATASET_CONFIGS = {
    'MNIST': {
        'epochs': 20,
        'input_size': 784,
        'num_classes': 10,
        'hidden_sizes': [128, 64],
        'learning_rate': 0.001,
        'batch_size': 64,
        'transform': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]),
        'model_class': get_mlp_class,
        'optimal_l1_lambda': 1e-06
    },
    'FashionMNIST': {
        'epochs': 20,
        'input_size': 784,
        'num_classes': 10,
        'hidden_sizes': [256, 128],
        'learning_rate': 0.001,
        'batch_size': 64,
        'transform': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,))
        ]),
        'model_class': get_mlp_class,
        'optimal_l1_lambda': 1e-05
    },
    'CIFAR10': {
        'epochs': 30,
        'input_size': 3072,
        'num_classes': 10,
        'hidden_sizes': [512, 256],
        'learning_rate': 0.001,
        'batch_size': 64,
        'transform': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]),
        'model_class': get_mlp_class,
        'optimal_l1_lambda': 1e-05
    },
    'CIFAR100': {
        'epochs': 30,
        'input_size': 3072,
        'num_classes': 100,
        'hidden_sizes': [512, 256],
        'learning_rate': 0.001,
        'batch_size': 64,
        'transform': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]),
        'model_class': get_mlp_class,
        'optimal_l1_lambda': 0.0001
    },
    'SVHN': {
        'epochs': 30,
        'input_size': 3072,
        'num_classes': 10,
        'hidden_sizes': [512, 256],
        'learning_rate': 0.001,
        'batch_size': 64,
        'transform': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]),
        'model_class': get_mlp_class,
        'optimal_l1_lambda': 1e-05
    },
    'Covertype': {
        'epochs': 30,
        'input_size': 54,
        'num_classes': 7,
        'hidden_sizes': [1024, 512, 256, 128],
        'learning_rate': 0.001,
        'batch_size': 256,
        'transform': None,  # Tabular data, no image transforms
        'model_class': get_mlp_class,
        'optimal_l1_lambda': 1e-05  # Will need to be tuned
    },
    'Newsgroups': {
        'epochs': 100,
        'input_size': 5000,  # TF-IDF features (max_features=5000)
        'num_classes': 20,  # 20 newsgroups
        'hidden_sizes': [1024, 512, 256],
        'learning_rate': 0.0001,
        'batch_size': 128,
        'transform': None,  # Text data, no image transforms
        'model_class': get_mlp_class,
        'optimal_l1_lambda': 1e-05
    }
}


def load_dataset(dataset_name):
    """
    Load a dataset by name.

    Args:
        dataset_name: Name of the dataset (must be in DATASET_CONFIGS)

    Returns:
        train_loader: DataLoader for training set
        test_loader: DataLoader for test set
        config: Dictionary with dataset configuration
    """
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASET_CONFIGS.keys())}")

    config = DATASET_CONFIGS[dataset_name]

    if dataset_name == 'MNIST':
        train_dataset = datasets.MNIST(
            root='./data', train=True, download=True, transform=config['transform'])
        test_dataset = datasets.MNIST(
            root='./data', train=False, download=True, transform=config['transform'])
    elif dataset_name == 'FashionMNIST':
        train_dataset = datasets.FashionMNIST(
            root='./data', train=True, download=True, transform=config['transform'])
        test_dataset = datasets.FashionMNIST(
            root='./data', train=False, download=True, transform=config['transform'])
    elif dataset_name == 'CIFAR10':
        train_dataset = datasets.CIFAR10(
            root='./data', train=True, download=True, transform=config['transform'])
        test_dataset = datasets.CIFAR10(
            root='./data', train=False, download=True, transform=config['transform'])
    elif dataset_name == 'CIFAR100':
        train_dataset = datasets.CIFAR100(
            root='./data', train=True, download=True, transform=config['transform'])
        test_dataset = datasets.CIFAR100(
            root='./data', train=False, download=True, transform=config['transform'])
    elif dataset_name == 'SVHN':
        train_dataset = datasets.SVHN(
            root='./data', split='train', download=True, transform=config['transform'])
        test_dataset = datasets.SVHN(
            root='./data', split='test', download=True, transform=config['transform'])
    elif dataset_name == 'Covertype':
        # Load UCI Covertype dataset
        print(f"Loading UCI Covertype dataset...")
        print("  (This may take a moment - downloading ~11MB dataset)")
        data = fetch_covtype(data_home='./data', download_if_missing=True)
        X, y = data.data, data.target

        # Convert labels from 1-7 to 0-6
        y = y - 1

        # Split into train/test (80/20 split)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Standardize features (important for neural networks)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.LongTensor(y_train)
        X_test_tensor = torch.FloatTensor(X_test)
        y_test_tensor = torch.LongTensor(y_test)

        # Create PyTorch datasets
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

        print(f"  Loaded {len(X_train)} training samples, {len(X_test)} test samples")
        print(f"  Features: {X_train.shape[1]}, Classes: {len(np.unique(y))}")
    elif dataset_name == 'Newsgroups':
        # Load 20 Newsgroups dataset
        print(f"Loading 20 Newsgroups dataset...")
        print("  (This may take a moment - downloading and processing text data)")

        # Fetch the dataset (removes headers, footers, quotes for cleaner text)
        newsgroups_train = fetch_20newsgroups(
            subset='train',
            remove=('headers', 'footers', 'quotes'),
            data_home='./data',
            download_if_missing=True
        )
        newsgroups_test = fetch_20newsgroups(
            subset='test',
            remove=('headers', 'footers', 'quotes'),
            data_home='./data',
            download_if_missing=True
        )

        # Convert text to TF-IDF features
        print("  Converting text to TF-IDF features...")
        vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        X_train = vectorizer.fit_transform(newsgroups_train.data).toarray()
        X_test = vectorizer.transform(newsgroups_test.data).toarray()

        # Get labels (already 0-19)
        y_train = newsgroups_train.target
        y_test = newsgroups_test.target

        # Normalize features (important for neural networks)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.LongTensor(y_train)
        X_test_tensor = torch.FloatTensor(X_test)
        y_test_tensor = torch.LongTensor(y_test)

        # Create PyTorch datasets
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

        print(f"  Loaded {len(X_train)} training samples, {len(X_test)} test samples")
        print(f"  Features: {X_train.shape[1]}, Classes: {len(np.unique(y_train))}")
    else:
        raise ValueError(f"Dataset loader not implemented for: {dataset_name}")

    batch_size = config.get('batch_size', DEFAULT_BATCH_SIZE)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, config


# ── UCI dataset loader ────────────────────────────────────────────────────────
#
# Generic loader for UCI datasets from the UCI_data folder.
#
# Each dataset folder contains:
#   - {name}.txt: metadata (n_entradas, n_clases, n_arquivos)
#   - {name}_R.dat: tab-separated data with header, last column 'clase'
#     OR {name}_train_R.dat + {name}_test_R.dat for pre-split datasets
#   - conxuntos.dat: train/test index split (2 lines) for single-file datasets
#
# Usage:
#     from wvalue.datasets import load_uci_dataset, list_uci_datasets
#
#     train_loader, test_loader, config = load_uci_dataset('iris')
#     all_names = list_uci_datasets()

UCI_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'UCI_data')


def list_uci_datasets(uci_dir=None):
    """Return sorted list of all available UCI dataset names."""
    uci_dir = uci_dir or UCI_DATA_DIR
    uci_dir = os.path.abspath(uci_dir)
    return sorted([
        d for d in os.listdir(uci_dir)
        if os.path.isdir(os.path.join(uci_dir, d))
    ])


def _parse_txt(txt_path):
    """Parse the {name}.txt metadata file."""
    info = {}
    with open(txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if '=' in line:
                key, val = line.split('=', 1)
                key = key.strip()
                val = val.strip()
                try:
                    val = int(val)
                except ValueError:
                    pass
                info[key] = val
    return info


def _load_r_dat(path):
    """Load a _R.dat file. Returns (X numpy array, y numpy array)."""
    df = pd.read_csv(path, sep='\t', index_col=0)
    y = df['clase'].values
    X = df.drop(columns=['clase']).values.astype(np.float32)
    return X, y


def _encode_labels(y_train, y_test):
    """Map arbitrary class labels to 0..K-1."""
    all_labels = np.unique(np.concatenate([y_train, y_test]))
    label_map = {label: i for i, label in enumerate(all_labels)}
    y_train = np.array([label_map[l] for l in y_train])
    y_test = np.array([label_map[l] for l in y_test])
    return y_train, y_test, len(all_labels)


def load_uci_dataset(name, uci_dir=None, batch_size=64, test_ratio=0.25):
    """
    Load a UCI dataset by folder name.

    Args:
        name: Dataset folder name (e.g. 'iris', 'wine', 'spambase')
        uci_dir: Path to UCI_data root. Defaults to ../UCI_data relative to this file.
        batch_size: Batch size for DataLoaders.
        test_ratio: Fallback test split ratio if conxuntos.dat is missing.

    Returns:
        train_loader, test_loader, config dict with keys:
            input_size, num_classes, n_train, n_test, name,
            epochs, hidden_sizes, learning_rate, batch_size, model_class
    """
    uci_dir = uci_dir or UCI_DATA_DIR
    uci_dir = os.path.abspath(uci_dir)
    dataset_dir = os.path.join(uci_dir, name)

    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(f"UCI dataset folder not found: {dataset_dir}")

    # Parse metadata
    txt_path = os.path.join(dataset_dir, f'{name}.txt')
    info = _parse_txt(txt_path) if os.path.exists(txt_path) else {}

    # Load data: check for train/test split files first
    train_path = os.path.join(dataset_dir, f'{name}_train_R.dat')
    test_path = os.path.join(dataset_dir, f'{name}_test_R.dat')
    single_path = os.path.join(dataset_dir, f'{name}_R.dat')

    if os.path.exists(train_path) and os.path.exists(test_path):
        X_train, y_train = _load_r_dat(train_path)
        X_test, y_test = _load_r_dat(test_path)
    elif os.path.exists(single_path):
        X, y = _load_r_dat(single_path)
        # Use conxuntos.dat for train/test split
        conxuntos_path = os.path.join(dataset_dir, 'conxuntos.dat')
        if os.path.exists(conxuntos_path):
            with open(conxuntos_path, 'r') as f:
                lines = f.readlines()
            train_idx = np.array([int(x) for x in lines[0].strip().split()])
            test_idx = np.array([int(x) for x in lines[1].strip().split()])
            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]
        else:
            # Fallback: random split
            n = len(X)
            n_test = max(1, int(n * test_ratio))
            rng = np.random.RandomState(42)
            perm = rng.permutation(n)
            X_train, y_train = X[perm[n_test:]], y[perm[n_test:]]
            X_test, y_test = X[perm[:n_test]], y[perm[:n_test]]
    else:
        raise FileNotFoundError(
            f"No data files found for '{name}'. Expected {single_path} or {train_path}/{test_path}"
        )

    # Encode labels to 0..K-1
    y_train, y_test, num_classes = _encode_labels(y_train, y_test)

    # Handle NaN values (replace with 0)
    X_train = np.nan_to_num(X_train, nan=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0)

    input_size = X_train.shape[1]

    # Build tensors and loaders
    train_ds = TensorDataset(
        torch.FloatTensor(X_train),
        torch.LongTensor(y_train),
    )
    test_ds = TensorDataset(
        torch.FloatTensor(X_test),
        torch.LongTensor(y_test),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # Auto-size hidden layers based on input size
    if input_size <= 20:
        hidden_sizes = [64, 32]
    elif input_size <= 100:
        hidden_sizes = [128, 64]
    else:
        hidden_sizes = [256, 128]

    config = {
        'input_size': input_size,
        'num_classes': num_classes,
        'n_train': len(X_train),
        'n_test': len(X_test),
        'name': name,
        'epochs': 50,
        'hidden_sizes': hidden_sizes,
        'learning_rate': 0.001,
        'batch_size': batch_size,
        'model_class': get_mlp_class,
    }

    return train_loader, test_loader, config


# ── Grokking dataset ──────────────────────────────────────────────────────────
#
# Modular arithmetic dataset (a * b mod p).
# Used to study grokking: train on a subset of pairs, test on all;
# model often memorizes then generalizes.


def make_modular_arithmetic(p=97, train_frac=0.5, seed=42, batch_size=64):
    """
    Create train/test loaders for a*b mod p. Input = one-hot(a) concat one-hot(b); output = a*b mod p.

    Args:
        p: modulus (number of classes and one-hot size per operand)
        train_frac: fraction of all p^2 pairs used for training (rest for test)
        seed: random seed for train/test split
        batch_size: DataLoader batch size

    Returns:
        train_loader, test_loader, config dict with input_size=2*p, num_classes=p
    """
    rng = np.random.default_rng(seed)
    pairs = np.array([(a, b) for a in range(p) for b in range(p)])
    rng.shuffle(pairs)
    n_train = max(1, int(len(pairs) * train_frac))
    train_pairs, test_pairs = pairs[:n_train], pairs[n_train:]

    def to_onehot(a, b):
        x = np.zeros(2 * p, dtype=np.float32)
        x[a] = 1.0
        x[p + b] = 1.0
        return x

    def labels_from_pairs(prs):
        return (prs[:, 0] * prs[:, 1]) % p

    X_train = np.array([to_onehot(a, b) for a, b in train_pairs])
    y_train = labels_from_pairs(train_pairs)
    X_test = np.array([to_onehot(a, b) for a, b in test_pairs])
    y_test = labels_from_pairs(test_pairs)

    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    test_ds = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    config = {'input_size': 2 * p, 'num_classes': p, 'p': p, 'n_train': n_train, 'n_test': len(test_pairs)}
    return train_loader, test_loader, config


# ══════════════════════════════════════════════════════════════════════════════
# Section 2 — Regression datasets
# ══════════════════════════════════════════════════════════════════════════════

import math


# ── OpenML / sklearn helpers ──────────────────────────────────────────────────

def _openml_fetch(attempts, as_frame=True, parser='auto'):
    from sklearn.datasets import fetch_openml
    errs = []
    for kw in attempts:
        try:
            return fetch_openml(**kw, as_frame=as_frame, parser=parser)
        except Exception as e:
            errs.append(f'{kw}: {e}')
    raise RuntimeError('Could not load dataset from OpenML. Attempts:\n' + '\n'.join(errs))


def _std_split(X, y, test_size=0.2, seed=42):
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    X = StandardScaler().fit_transform(X.astype('float32'))
    y = StandardScaler().fit_transform(y.astype('float32').reshape(-1, 1)).ravel()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=seed)
    return torch.tensor(Xtr), torch.tensor(ytr), torch.tensor(Xte), torch.tensor(yte)


def _xy_from_openml(d):
    df = d.data.copy()
    if d.target is None or (hasattr(d.target, 'empty') and d.target.empty):
        y_series = df.iloc[:, -1]
        df       = df.iloc[:, :-1]
    else:
        y_series = d.target.copy()
    y_series = pd.to_numeric(y_series, errors='coerce')
    df       = df.select_dtypes(include='number')
    mask     = df.notna().all(axis=1) & y_series.notna()
    df       = df.loc[mask]
    y_series = y_series.loc[mask]
    if len(df) == 0:
        raise ValueError(
            f'_xy_from_openml: 0 rows remain after cleaning. '
            f'Original shape: {d.data.shape}.'
        )
    return df.values.astype('float32'), y_series.values.astype('float32')


# ── Real-dataset loaders ──────────────────────────────────────────────────────

def _load_california_housing(seed=42):
    from sklearn.datasets import fetch_california_housing
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    d    = fetch_california_housing()
    X, y = d.data.astype('float32'), d.target.astype('float32')
    X    = StandardScaler().fit_transform(X)
    y    = StandardScaler().fit_transform(y.reshape(-1, 1)).ravel()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=seed)
    return torch.tensor(Xtr), torch.tensor(ytr), torch.tensor(Xte), torch.tensor(yte)


def _load_diabetes(seed=42):
    from sklearn.datasets import load_diabetes
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    d    = load_diabetes()
    X, y = d.data.astype('float32'), d.target.astype('float32')
    y    = StandardScaler().fit_transform(y.reshape(-1, 1)).ravel()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=seed)
    return torch.tensor(Xtr), torch.tensor(ytr), torch.tensor(Xte), torch.tensor(yte)


def _load_wine_quality(seed=42):
    from sklearn.datasets import load_wine
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    try:
        from sklearn.datasets import fetch_openml
        d    = fetch_openml('wine-quality-red', version=1, as_frame=True, parser='auto')
        X, y = _xy_from_openml(d)
        X, y = X.astype('float32'), y.astype('float32')
    except Exception:
        d    = load_wine()
        X, y = d.data.astype('float32'), d.target.astype('float32')
    X = StandardScaler().fit_transform(X)
    y = StandardScaler().fit_transform(y.reshape(-1, 1)).ravel()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=seed)
    return torch.tensor(Xtr), torch.tensor(ytr), torch.tensor(Xte), torch.tensor(yte)


def _load_concrete(seed=42):
    d = _openml_fetch([
        dict(name='concrete-compressive-strength', version=1),
        dict(name='concrete-compressive-strength', version=2),
        dict(data_id=4353),
    ])
    X, y = _xy_from_openml(d)
    return _std_split(X, y, seed=seed)


def _load_abalone(seed=42):
    d        = _openml_fetch([dict(name='abalone', version=1), dict(data_id=183)])
    df       = pd.get_dummies(d.data.copy(), columns=['Sex'], drop_first=False)
    _, y_abl = _xy_from_openml(d)
    return _std_split(df.values, y_abl, seed=seed)


def _load_power_plant(seed=42):
    d = _openml_fetch([
        dict(name='CCPP', version=1),
        dict(name='CCPP', version=2),
        dict(name='combined-cycle-power-plant', version=1),
        dict(name='combined_cycle_power_plant', version=1),
        dict(data_id=4553),
        dict(data_id=294),
    ])
    X, y = _xy_from_openml(d)
    return _std_split(X, y, seed=seed)


def _load_energy_efficiency(seed=42):
    d = _openml_fetch([
        dict(name='energy-efficiency', version=2),
        dict(name='energy-efficiency', version=1),
        dict(data_id=934),
    ])
    X, y = _xy_from_openml(d)
    y    = y[:, 0] if y.ndim > 1 else y
    return _std_split(X, y, seed=seed)


def _load_auto_mpg(seed=42):
    d   = _openml_fetch([
        dict(name='autoMpg', version=1),
        dict(name='auto-mpg', version=1),
        dict(data_id=196),
    ])
    df      = d.data.select_dtypes(include='number').dropna()
    _, y_all = _xy_from_openml(d)
    tgt     = y_all[df.index] if hasattr(y_all, '__getitem__') else d.target.loc[df.index].values
    return _std_split(df.values, tgt, seed=seed)


# ── Dataset cases ─────────────────────────────────────────────────────────────
# Each entry supplies:
#   desc        — human-readable task description
#   make_y      — callable(X: Tensor) -> Tensor   (synthetic cases only)
#   load_data   — callable(seed) -> (Xtr, ytr, Xte, yte)   (real cases only)
#   signal_cols — indices of causally relevant input features
#   input_dim   — overrides INPUT_DIM for real datasets
#   weight_decay — suggested weight decay for this task

DATASET_CASES = {
    'x1_x2': {
        'desc':        'y = x₁·x₂',
        'make_y':      lambda X: X[:, 0] * X[:, 1],
        'signal_cols': [0, 1],
    },
    'linear_sparse': {
        'desc':        'y = x₁ + x₂ + x₃',
        'make_y':      lambda X: X[:, 0] + X[:, 1] + X[:, 2],
        'signal_cols': [0, 1, 2],
    },
    'quadratic': {
        'desc':        'y = x₁² + x₂²',
        'make_y':      lambda X: X[:, 0]**2 + X[:, 1]**2,
        'signal_cols': [0, 1],
    },
    'two_products': {
        'desc':        'y = x₁x₂ + x₃x₄',
        'make_y':      lambda X: X[:, 0]*X[:, 1] + X[:, 2]*X[:, 3],
        'signal_cols': [0, 1, 2, 3],
    },
    'three_way': {
        'desc':        'y = x₁·x₂·x₃',
        'make_y':      lambda X: X[:, 0] * X[:, 1] * X[:, 2],
        'signal_cols': [0, 1, 2],
        'weight_decay': 1e-2,
    },
    'four_way': {
        'desc':        'y = x₁·x₂·x₃·x₄',
        'make_y':      lambda X: X[:, 0] * X[:, 1] * X[:, 2] * X[:, 3],
        'signal_cols': [0, 1, 2, 3],
        'weight_decay': 1e-2,
    },
    'sin_product': {
        'desc':        'y = sin(πx₁)·cos(πx₂)',
        'make_y':      lambda X: torch.sin(math.pi * X[:, 0]) * torch.cos(math.pi * X[:, 1]),
        'signal_cols': [0, 1],
        'weight_decay': 0.03,
    },
    'five_products': {
        'desc':        'y = x₁x₂ + x₃x₄ + x₅x₆ + x₇x₈ + x₉x₁₀',
        'make_y':      lambda X: torch.stack([X[:, 2*i] * X[:, 2*i+1] for i in range(5)]).sum(0),
        'signal_cols': list(range(10)),
    },
    # ── Real datasets ──────────────────────────────────────────────────────────
    'california_housing': {
        'desc':        'California Housing (sklearn)',
        'load_data':   _load_california_housing,
        'signal_cols': list(range(8)),
        'input_dim':   8,
        'weight_decay': 1e-3,
    },
    'diabetes': {
        'desc':        'Diabetes (sklearn)',
        'load_data':   _load_diabetes,
        'signal_cols': list(range(10)),
        'input_dim':   10,
        'weight_decay': 1e-3,
    },
    'wine_quality': {
        'desc':        'Wine Quality Red (UCI)',
        'load_data':   _load_wine_quality,
        'signal_cols': list(range(11)),
        'input_dim':   11,
        'weight_decay': 1e-2,
    },
    'concrete': {
        'desc':        'Concrete Compressive Strength (UCI)',
        'load_data':   _load_concrete,
        'signal_cols': list(range(8)),
        'input_dim':   8,
        'weight_decay': 1e-3,
    },
    'abalone': {
        'desc':        'Abalone Age (UCI)',
        'load_data':   _load_abalone,
        'signal_cols': list(range(10)),
        'input_dim':   10,
        'weight_decay': 1e-3,
    },
    'power_plant': {
        'desc':        'Combined Cycle Power Plant (UCI)',
        'load_data':   _load_power_plant,
        'signal_cols': list(range(4)),
        'input_dim':   4,
        'weight_decay': 1e-3,
    },
    'energy_efficiency': {
        'desc':        'Energy Efficiency — Heating Load (UCI)',
        'load_data':   _load_energy_efficiency,
        'signal_cols': list(range(8)),
        'input_dim':   8,
        'weight_decay': 1e-3,
    },
    'auto_mpg': {
        'desc':        'Auto MPG (UCI)',
        'load_data':   _load_auto_mpg,
        'signal_cols': list(range(6)),
        'input_dim':   6,
        'weight_decay': 1e-3,
    },
}
