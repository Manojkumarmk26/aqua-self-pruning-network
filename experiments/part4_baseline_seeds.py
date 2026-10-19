
"""
Part 4 Baseline Comparison Across Random Seeds.
Evaluates Taylor Saliency versus Pure Magnitude Pruning across multiple random seeds
to prove statistical significance, variance bounds, and support the falsifiable claim.
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
from scipy import stats
from experiments.part4_pareto_sweep import run_pruning_experiment


def main():
    parser = argparse.ArgumentParser(description="Part 4: Multi-seed Baseline Comparison")
    parser.add_argument("--sparsity", type=float, default=0.90, help="Target connection sparsity")
    parser.add_argument("--epochs", type=int, default=15, help="Training epochs")
    parser.add_argument("--num_seeds", type=int, default=5, help="Number of random seeds")
    parser.add_argument("--dataset", type=str, default="digits")
    parser.add_argument("--output_dir", type=str, default="artifacts")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    seeds = [42 + i for i in range(args.num_seeds)]

    print("\n=======================================================")
    print(f"=== Multi-Seed Statistical Comparison at {args.sparsity*100:.0f}% Sparsity ===")
    print(f"Seeds: {seeds} | Dataset: {args.dataset} | Epochs: {args.epochs}")
    print("=======================================================\n")

    taylor_accs = []
    magnitude_accs = []
    seed_details = []

    for seed in seeds:
        print(f"Running Seed {seed}...")
        res_taylor = run_pruning_experiment(
            sparsity_target=args.sparsity,
            criterion_name="taylor",
            allow_regrowth=False,
            dataset=args.dataset,
            epochs=args.epochs,
            seed=seed
        )
        res_mag = run_pruning_experiment(
            sparsity_target=args.sparsity,
            criterion_name="magnitude",
            allow_regrowth=False,
            dataset=args.dataset,
            epochs=args.epochs,
            seed=seed
        )

        t_acc = res_taylor["accuracy"] * 100.0
        m_acc = res_mag["accuracy"] * 100.0
        diff = t_acc - m_acc

        taylor_accs.append(t_acc)
        magnitude_accs.append(m_acc)

        seed_details.append({
            "seed": seed,
            "taylor_accuracy": t_acc,
            "magnitude_accuracy": m_acc,
            "difference": diff,
            "achieved_sparsity_taylor": res_taylor["achieved_sparsity"],
            "achieved_sparsity_magnitude": res_mag["achieved_sparsity"]
        })

        print(f"  Seed {seed}: Taylor = {t_acc:.2f}% | Magnitude = {m_acc:.2f}% | Delta = {diff:+.2f}%")

    t_mean, t_std = float(np.mean(taylor_accs)), float(np.std(taylor_accs, ddof=1))
    m_mean, m_std = float(np.mean(magnitude_accs)), float(np.std(magnitude_accs, ddof=1))
    diff_mean = t_mean - m_mean

    # Paired t-test
    t_stat, p_val = stats.ttest_rel(taylor_accs, magnitude_accs)

    summary = {
        "target_sparsity": args.sparsity,
        "num_seeds": args.num_seeds,
        "seeds": seeds,
        "taylor_mean": t_mean,
        "taylor_std": t_std,
        "magnitude_mean": m_mean,
        "magnitude_std": m_std,
        "mean_difference": diff_mean,
        "t_statistic": float(t_stat),
        "p_value": float(p_val),
        "statistically_significant_p05": bool(p_val < 0.05),
        "seed_details": seed_details
    }

    out_file = os.path.join(args.output_dir, "seed_comparison.json")
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved statistical seed comparison to {out_file}")

    print("\n-------------------------------------------------------")
    print(f"Taylor Saliency Mean Accuracy: {t_mean:.2f}% ± {t_std:.2f}%")
    print(f"Magnitude Baseline Mean Acc:   {m_mean:.2f}% ± {m_std:.2f}%")
    print(f"Net Advantage (Taylor - Mag):  {diff_mean:+.2f}% (p-value: {p_val:.4e})")
    print(f"Statistically Significant:     {p_val < 0.05}")
    print("-------------------------------------------------------\n")


if __name__ == "__main__":
    main()
