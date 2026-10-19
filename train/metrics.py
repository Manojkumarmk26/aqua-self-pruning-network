"""
Evaluation Metrics.
"""

from __future__ import annotations
from typing import Tuple
import numpy as np
from engine.tensor import Tensor
import engine.ops as ops
from nn.module import Module


def compute_accuracy(logits: np.ndarray, targets: np.ndarray) -> float:
    preds = np.argmax(logits, axis=-1)
    return float(np.mean(preds == targets))


def evaluate(model: Module, x: np.ndarray, y: np.ndarray, batch_size: int = 128) -> Tuple[float, float]:
    """Evaluate model over entire dataset without gradient tracking."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    n = len(x)

    for i in range(0, n, batch_size):
        bx = x[i:i + batch_size]
        by = y[i:i + batch_size]

        tx = Tensor(bx, requires_grad=False)
        logits = model(tx)
        loss = ops.softmax_cross_entropy(logits, by)

        total_loss += float(loss.item()) * len(bx)
        total_correct += int(np.sum(np.argmax(logits.data, axis=-1) == by))

    avg_loss = total_loss / n
    avg_acc = total_correct / n
    return avg_loss, avg_acc
