"""
Part 3 Reproduction Script - Self-Pruning Network under Hard Compute Budget.
Trains a network that progressively removes its own least-important connections
using derived first-order Taylor saliency and a cubic sparsity ramp to hit 90% sparsity.
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

import nn
import optim
from prune.masked_linear import MaskedLinear
from prune.criteria import TaylorSaliencyCriterion, MagnitudeCriterion, RegrowthSaliencyCriterion
from prune.scheduler import CubicScheduler
from prune.pruner import Pruner
from prune.sparse_engine import SparseLinear, count_flops, benchmark_layer_inference
from train.dataset import load_classification_data, DataLoader
from train.trainer import Trainer
from train.metrics import evaluate


def build_masked_mlp(in_features: int, hidden_dim: int, num_classes: int) -> nn.Sequential:
    """Construct MLP where weight layers are MaskedLinear."""
    return nn.Sequential(
        MaskedLinear(in_features, hidden_dim, nonlinearity="relu", name="fc1"),
        nn.ReLU(),
        MaskedLinear(hidden_dim, hidden_dim, nonlinearity="relu", name="fc2"),
        nn.ReLU(),
        MaskedLinear(hidden_dim, num_classes, nonlinearity="linear", name="fc3")
    )


def main():
    parser = argparse.ArgumentParser(description="Part 3: Self-Pruning Network")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--target_sparsity", type=float, default=0.90, help="Target connection sparsity")
    parser.add_argument("--lr", type=float, default=2e-3, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden dimension")
    parser.add_argument("--criterion", type=str, default="taylor", choices=["taylor", "magnitude", "regrowth"])
    parser.add_argument("--regrowth", action="store_true", help="Allow connection regrowth during training")
    parser.add_argument("--dataset", type=str, default="digits", choices=["digits", "spirals", "mnist"])
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_dir", type=str, default="artifacts")
    args = parser.parse_args()

    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n=======================================================")
    print(f"=== PART 3: Self-Pruning Network on {args.dataset.upper()} ===")
    print(f"Target Sparsity: {args.target_sparsity * 100:.1f}% | Criterion: {args.criterion} | Regrowth: {args.regrowth}")
    print(f"=======================================================\n")

    train_x, train_y, val_x, val_y = load_classification_data(args.dataset, seed=args.seed)
    in_features = train_x.shape[1]
    num_classes = len(np.unique(train_y))

    model = build_masked_mlp(in_features, args.hidden_dim, num_classes)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # Calculate steps for pruning ramp: ramp starts at step 50, ends at 75% of total steps
    train_loader = DataLoader(train_x, train_y, batch_size=args.batch_size, shuffle=True)
    steps_per_epoch = len(train_loader)
    total_steps = args.epochs * steps_per_epoch
    start_step = min(50, total_steps // 6)
    end_step = int(total_steps * 0.75)

    scheduler = CubicScheduler(
        initial_sparsity=0.0,
        target_sparsity=args.target_sparsity,
        start_step=start_step,
        end_step=end_step,
        update_freq=max(5, steps_per_epoch // 2)
    )

    if args.criterion == "taylor":
        criterion = TaylorSaliencyCriterion(ema_decay=0.9)
    elif args.criterion == "regrowth":
        criterion = RegrowthSaliencyCriterion(ema_decay=0.9)
        args.regrowth = True
    else:
        criterion = MagnitudeCriterion()

    pruner = Pruner(
        model=model,
        scheduler=scheduler,
        criterion=criterion,
        allow_regrowth=args.regrowth,
        regrowth_ratio=0.15
    )

    trainer = Trainer(model, optimizer, pruner=pruner, verbose=True)
    history = trainer.fit(train_loader, val_x, val_y, epochs=args.epochs)

    final_val_loss, final_val_acc = evaluate(model, val_x, val_y)
    achieved_sparsity = pruner.current_sparsity

    total_weights = sum(layer.weight.size for layer in pruner.layers)
    active_weights = sum(layer.num_active_params for layer in pruner.layers)

    print("\n-------------------------------------------------------")
    print(f"Final Validation Accuracy: {final_val_acc * 100:.2f}%")
    print(f"Target Sparsity:           {args.target_sparsity * 100:.1f}%")
    print(f"Achieved Sparsity:         {achieved_sparsity * 100:.2f}%")
    print(f"Active Weights:            {active_weights:,} / {total_weights:,}")
    print("-------------------------------------------------------\n")

    # True cost measurement on largest layer
    largest_layer = pruner.layers[1]
    cost_info = benchmark_layer_inference(largest_layer, batch_size=64, num_runs=200)
    print(f"FLOPs Reduction:           {cost_info['dense_flops']:,} dense -> {cost_info['sparse_flops']:,} active FLOPs")
    print(f"Theoretical Speedup:       {cost_info['theoretical_speedup']:.2f}x")
    print(f"Sparse CSR Latency:        {cost_info['sparse_csr_latency_ms']:.4f} ms")

    # Save results
    results_path = os.path.join(args.output_dir, "part3_results.json")
    with open(results_path, "w") as f:
        json.dump({
            "target_sparsity": args.target_sparsity,
            "achieved_sparsity": float(achieved_sparsity),
            "final_val_accuracy": float(final_val_acc),
            "final_val_loss": float(final_val_loss),
            "total_weights": int(total_weights),
            "active_weights": int(active_weights),
            "cost_measurement": cost_info,
            "history": history
        }, f, indent=2)
    print(f"Saved Part 3 results to {results_path}")

    # Plot dynamic pruning progression
    plot_path = os.path.join(args.output_dir, "part3_pruning_curve.png")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

    epochs = history["epoch"]
    ax1.plot(epochs, [s * 100 for s in history["sparsity"]], color="#8c564b", lw=2.5, label="Achieved Sparsity (%)")
    ax1.axhline(args.target_sparsity * 100, color="black", linestyle=":", lw=1.5, label=f"Target ({args.target_sparsity*100:.0f}%)")
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Sparsity (%)", fontsize=11)
    ax1.set_title("Part 3: Cubic Sparsity Ramp", fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="lower right")

    ax2.plot(epochs, [a * 100 for a in history["train_acc"]], label="Train Acc", color="#1f77b4", lw=2)
    ax2.plot(epochs, [a * 100 for a in history["val_acc"]], label="Val Acc", color="#2ca02c", lw=2)
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Accuracy (%)", fontsize=11)
    ax2.set_title("Part 3: Accuracy Recovery During Pruning", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"Saved Part 3 plot to {plot_path}")


if __name__ == "__main__":
    main()
