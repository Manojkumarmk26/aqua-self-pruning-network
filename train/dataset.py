"""
Dataset Loaders and Synthetic Data Generators.
Per challenge restrictions: Scikit-learn is permitted ONLY to load standard datasets.
"""

from __future__ import annotations
import numpy as np
from typing import Tuple, Generator, Optional


def generate_spirals(
    n_points_per_class: int = 500,
    n_classes: int = 3,
    noise: float = 0.2,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic 2D spiral classification dataset.
    Non-linear manifold that tests capacity of multi-layer networks.
    """
    np.random.seed(seed)
    n = n_points_per_class * n_classes
    d = 2
    x = np.zeros((n, d), dtype=np.float32)
    y = np.zeros(n, dtype=np.int64)

    for j in range(n_classes):
        ix = range(n_points_per_class * j, n_points_per_class * (j + 1))
        r = np.linspace(0.0, 1.0, n_points_per_class)
        t = np.linspace(j * 4.0, (j + 1) * 4.0, n_points_per_class) + np.random.randn(n_points_per_class) * noise
        x[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[ix] = j

    # Shuffle
    indices = np.random.permutation(n)
    return x[indices], y[indices]


def load_classification_data(
    dataset_name: str = "digits",
    val_split: float = 0.2,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load dataset using permitted Scikit-Learn loader.
    Supports 'digits' (8x8 handwritten digits, 10 classes) and 'mnist' (28x28).
    """
    np.random.seed(seed)
    if dataset_name == "digits":
        from sklearn.datasets import load_digits
        data = load_digits()
        x = data.data.astype(np.float32) / 16.0  # Normalize pixel intensities to [0, 1]
        y = data.target.astype(np.int64)
    elif dataset_name == "spirals":
        x, y = generate_spirals(n_points_per_class=1000, n_classes=4, seed=seed)
    elif dataset_name == "mnist":
        from sklearn.datasets import fetch_openml
        try:
            print("Fetching MNIST dataset via scikit-learn...")
            mnist = fetch_openml('mnist_784', version=1, parser='auto', as_frame=False)
            x = mnist.data.astype(np.float32) / 255.0
            y = mnist.target.astype(np.int64)
        except Exception as e:
            print(f"Failed to fetch online MNIST ({e}), falling back to standard 10-class Digits dataset.")
            from sklearn.datasets import load_digits
            data = load_digits()
            x = data.data.astype(np.float32) / 16.0
            y = data.target.astype(np.int64)
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")

    n = len(x)
    n_val = int(n * val_split)
    indices = np.random.permutation(n)

    train_idx = indices[n_val:]
    val_idx = indices[:n_val]

    return x[train_idx], y[train_idx], x[val_idx], y[val_idx]


class DataLoader:
    """Mini-batch data loader."""
    def __init__(self, x: np.ndarray, y: np.ndarray, batch_size: int = 64, shuffle: bool = True):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n = len(x)

    def __len__(self) -> int:
        return int(np.ceil(self.n / self.batch_size))

    def __iter__(self) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        indices = np.arange(self.n)
        if self.shuffle:
            np.random.shuffle(indices)

        for i in range(0, self.n, self.batch_size):
            batch_idx = indices[i:i + self.batch_size]
            yield self.x[batch_idx], self.y[batch_idx]
