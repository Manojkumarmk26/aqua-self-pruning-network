"""
Mini-Batched Training Loop and Orchestration.
Integrates Model, Autodiff Backward, Optimizer Step, and Dynamic Self-Pruning.
"""

from __future__ import annotations
import time
import numpy as np
from typing import Optional, Dict, List, Any
from engine.tensor import Tensor
import engine.ops as ops
from nn.module import Module
from optim.optimizer import Optimizer
from prune.pruner import Pruner
from train.dataset import DataLoader
from train.metrics import evaluate, compute_accuracy


class Trainer:
    """
    Mini-batched Trainer for composable neural networks with dynamic pruning hooks.
    """
    def __init__(
        self,
        model: Module,
        optimizer: Optimizer,
        pruner: Optional[Pruner] = None,
        verbose: bool = True
    ):
        self.model = model
        self.optimizer = optimizer
        self.pruner = pruner
        self.verbose = verbose
        self.global_step = 0

    def train_step(self, bx: np.ndarray, by: np.ndarray) -> Tuple[float, float]:
        self.model.train()
        self.optimizer.zero_grad()

        tx = Tensor(bx, requires_grad=False)
        logits = self.model(tx)
        loss = ops.softmax_cross_entropy(logits, by)

        # Reverse-mode autodiff backward pass
        loss.backward()

        # Capture virtual sensitivity gradients before optimizer step
        if self.pruner is not None:
            self.pruner.capture_shadow_gradients()

        # Optimizer update (with honest mask invariant enforcement)
        self.optimizer.step()

        # Pruner step (evaluates importance criterion and schedule)
        self.global_step += 1
        if self.pruner is not None:
            self.pruner.step(self.global_step, optimizer=self.optimizer)

        step_loss = float(loss.item())
        step_acc = compute_accuracy(logits.data, by)
        return step_loss, step_acc

    def fit(
        self,
        train_loader: DataLoader,
        val_x: np.ndarray,
        val_y: np.ndarray,
        epochs: int = 20
    ) -> Dict[str, List[Any]]:
        history: Dict[str, List[Any]] = {
            "epoch": [],
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "sparsity": []
        }

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            epoch_loss = 0.0
            epoch_correct = 0
            n_samples = 0

            for bx, by in train_loader:
                loss_val, acc_val = self.train_step(bx, by)
                bs = len(bx)
                epoch_loss += loss_val * bs
                epoch_correct += int(round(acc_val * bs))
                n_samples += bs

            train_loss = epoch_loss / n_samples
            train_acc = epoch_correct / n_samples

            val_loss, val_acc = evaluate(self.model, val_x, val_y)
            sparsity = self.pruner.current_sparsity if self.pruner is not None else 0.0

            history["epoch"].append(epoch)
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            history["sparsity"].append(sparsity)

            elapsed = time.time() - t0
            if self.verbose:
                print(
                    f"Epoch {epoch:02d}/{epochs:02d} [{elapsed:.2f}s] - "
                    f"Loss: {train_loss:.4f} | Acc: {train_acc * 100:.2f}% | "
                    f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc * 100:.2f}% | "
                    f"Sparsity: {sparsity * 100:.1f}%"
                )

        return history
