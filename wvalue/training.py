"""
Baseline model training utilities.

This module provides functions for training baseline (unregularized) models.
Useful for establishing baseline performance before comparing regularization methods.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in tqdm(train_loader, desc="Training", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc


def evaluate(model, test_loader, criterion, device):
    """Evaluate model on test set, returns both loss and accuracy"""
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    avg_loss = total_loss / len(test_loader)
    return avg_loss, accuracy


def train_baseline_model(model, train_loader, test_loader, device, num_epochs,
                        verbose=True, print_interval=5, learning_rate=0.001):
    """
    Train a baseline (unregularized) model.

    Args:
        model: PyTorch model to train
        train_loader: DataLoader for training data
        test_loader: DataLoader for test data
        device: torch device
        num_epochs: Number of training epochs
        verbose: Whether to print progress
        print_interval: Print every N epochs
        learning_rate: Learning rate for Adam optimizer (default 0.001)

    Returns:
        train_losses: List of training losses per epoch
        train_accs: List of training accuracies per epoch
        test_losses: List of test losses per epoch
        test_accs: List of test accuracies per epoch
        final_test_acc: Final test accuracy
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []

    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

        if verbose and ((epoch + 1) % print_interval == 0 or epoch == 0):
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"  Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

    final_test_acc = test_accs[-1]
    if verbose:
        print(f"\nBaseline final test accuracy: {final_test_acc:.2f}%")

    return train_losses, train_accs, test_losses, test_accs, final_test_acc

def train_with_l1_regularization(model, train_loader, criterion, optimizer, device, l1_strength=0.001):
    """
    Train with L1 regularization.

    Args:
        model: PyTorch model
        train_loader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer
        device: torch device
        l1_strength: L1 regularization strength (lambda)

    Returns:
        epoch_loss: Average training loss for the epoch (classification loss only)
        epoch_acc: Training accuracy for the epoch
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in tqdm(train_loader, desc="Training with L1 reg", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Add L1 regularization
        l1_reg = 0.0
        for param in model.parameters():
            l1_reg += torch.sum(torch.abs(param))

        total_loss = loss + l1_strength * l1_reg
        total_loss.backward()
        optimizer.step()

        running_loss += loss.item()  # Track classification loss only
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc


def train_with_l2_regularization(model, train_loader, criterion, optimizer, device, l2_strength=0.001):
    """
    Train with L2 regularization (weight decay style in loss).

    Args:
        model: PyTorch model
        train_loader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer
        device: torch device
        l2_strength: L2 regularization strength (lambda)

    Returns:
        epoch_loss: Average training loss for the epoch (classification loss only)
        epoch_acc: Training accuracy for the epoch
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in tqdm(train_loader, desc="Training with L2 reg", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        l2_reg = 0.0
        for param in model.parameters():
            l2_reg += torch.sum(param ** 2)

        total_loss = loss + l2_strength * l2_reg
        total_loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc


def train_with_significance_regularization(model, train_loader, criterion, optimizer, device,
                                           sig_strength=0.001, significance_regularizer_term_fn=None,
                                           beta_sf_lookup_table=None, term_scale='mean'):
    """
    Train with significance regularizer: loss + lambda * sig_term (sig_term = sum or mean of BetaSF(normalized_weight_squared)).

    Significance is in probability space (Beta SF), not -ln space.
    Per-layer weights use Frobenius (matrix) normalization only.

    Args:
        model: PyTorch model
        train_loader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer
        device: torch device
        sig_strength: Regularization strength (lambda)
        significance_regularizer_term_fn: Callable(model, ...) -> scalar tensor.
            If None, uses wvalue_utils.significance_regularizer_term.
        beta_sf_lookup_table: Optional BetaSFLookupTable for faster forward pass. Pass explicitly so
            the regularizer uses the lookup table (e.g. wvalue_utils.beta_sf_lookup from the notebook).
        term_scale: 'mean' (default) or 'sum' for sig term; passed to significance_regularizer_term when using default fn.

    Returns:
        epoch_loss: Average training loss for the epoch (classification loss only)
        epoch_acc: Training accuracy for the epoch
    """
    if significance_regularizer_term_fn is None:
        from .core import significance_regularizer_term
        significance_regularizer_term_fn = significance_regularizer_term

    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in tqdm(train_loader, desc="Training with sig reg", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        try:
            sig_reg = significance_regularizer_term_fn(model, beta_sf_lookup_table=beta_sf_lookup_table, term_scale=term_scale)
        except TypeError:
            sig_reg = significance_regularizer_term_fn(model, beta_sf_lookup_table=beta_sf_lookup_table)
        total_loss = loss + sig_strength * sig_reg
        total_loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc
