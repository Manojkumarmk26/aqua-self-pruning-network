"""
Part 2 Reproduction Script - Train MLP with Adam from Scratch.
Demonstrates stable training, weight initialization justification, and NaN-free convergence.
"""

import os
import sys

# Ensure repository root is in python path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

from engine.tensor import Tensor
import nn
import optim
from train.dataset import load_classification_data, DataLoader
from train.trainer import Trainer
from train.metrics import evaluate


def build_mlp(in_features: int, hidden_dim: int, num_classes: int) -> nn.Sequential:
    """
    Construct MLP with Kaiming-initialized weights to preserve variance across ReLU layers.
    """
    return nn.Sequential(
        nn.Linear(in_features, hidden_dim, nonlinearity="relu", name="fc1"),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim, nonlinearity="relu", name="fc2"),
        nn.ReLU(),
        nn.Linear(hidden_dim, num_classes, nonlinearity="linear", name="fc3")
    )


def main():
    parser = argparse.ArgumentParser(description="Part 2: Train MLP with Adam from scratch")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=2e-3, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden dimension")
    parser.add_argument("--dataset", type=str, default="digits", choices=["digits", "spirals", "mnist"])
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_dir", type=str, default="artifacts")
    args = parser.parse_args()

    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"=== PART 2: Training Dense MLP on {args.dataset.upper()} with Custom Adam Optimizer ===")
    train_x, train_y, val_x, val_y = load_classification_data(args.dataset, seed=args.seed)
    in_features = train_x.shape[1]
    num_classes = len(np.unique(train_y))

    print(f"Dataset: {len(train_x)} train samples, {len(val_x)} val samples, {in_features} features, {num_classes} classes")

    # Instantiate model with Kaiming initialization
    model = build_mlp(in_features, args.hidden_dim, num_classes)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    train_loader = DataLoader(train_x, train_y, batch_size=args.batch_size, shuffle=True)
    trainer = Trainer(model, optimizer, verbose=True)

    history = trainer.fit(train_loader, val_x, val_y, epochs=args.epochs)

    final_val_loss, final_val_acc = evaluate(model, val_x, val_y)
    print(f"\nTraining Complete. Final Validation Accuracy: {final_val_acc * 100:.2f}% | Final Loss: {final_val_loss:.4f}")

    # Verify no NaNs occurred
    for p in model.parameters():
        assert not np.isnan(p.data).any(), "NaN detected in model parameters!"

    # Save results
    results_path = os.path.join(args.output_dir, "part2_results.json")
    with open(results_path, "w") as f:
        json.dump({
            "dataset": args.dataset,
            "final_val_accuracy": float(final_val_acc),
            "final_val_loss": float(final_val_loss),
            "history": history
        }, f, indent=2)
    print(f"Saved results to {results_path}")

    # Plot learning curve
    plot_path = os.path.join(args.output_dir, "part2_learning_curve.png")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.plot(history["epoch"], history["train_loss"], label="Train Loss", color="#1f77b4", lw=2)
    ax1.plot(history["epoch"], history["val_loss"], label="Val Loss", color="#ff7f0e", lw=2, linestyle="--")
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Loss", fontsize=11)
    ax1.set_title("Part 2: Cross-Entropy Loss Curve", fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(history["epoch"], [a * 100 for a in history["train_acc"]], label="Train Acc", color="#2ca02c", lw=2)
    ax2.plot(history["epoch"], [a * 100 for a in history["val_acc"]], label="Val Acc", color="#d62728", lw=2, linestyle="--")
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Accuracy (%)", fontsize=11)
    ax2.set_title("Part 2: Classification Accuracy Curve", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"Saved learning curve plot to {plot_path}")


if __name__ == "__main__":
    main()
