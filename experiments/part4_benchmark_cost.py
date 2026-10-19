"""
Part 4 Real Cost Measurement - FLOPs, Latency & CSR vs Dense Scaling.
Demonstrates that honest pruning actually reduces work and measures wall-clock speedup.
"""

import os
import sys

# Ensure repository root is in python path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import json
import numpy as np
import matplotlib.pyplot as plt
from prune.masked_linear import MaskedLinear
from prune.sparse_engine import benchmark_layer_inference, count_flops, SparseLinear


def main():
    os.makedirs("artifacts", exist_ok=True)
    np.random.seed(42)

    in_dim, out_dim = 256, 256
    batch_size = 64
    layer = MaskedLinear(in_features=in_dim, out_features=out_dim)

    sparsities = [0.0, 0.50, 0.70, 0.80, 0.85, 0.90, 0.95, 0.98]
    benchmarks = []

    print("\n=== Real Cost & Latency Benchmark across Sparsity Levels ===")
    print(f"Layer: ({in_dim} x {out_dim}) | Total Weights: {in_dim*out_dim:,} | Batch Size: {batch_size}\n")

    for s in sparsities:
        # Create mask
        mask = (np.random.rand(in_dim, out_dim) >= s).astype(np.float32)
        layer.set_mask(mask)

        bench = benchmark_layer_inference(layer, batch_size=batch_size, num_runs=300)
        sparse_layer = SparseLinear(layer)

        # Estimate memory footprints in Kilobytes
        dense_mem_kb = (in_dim * out_dim * 4) / 1024.0
        # CSR stores: data (float32, 4B) + indices (int32, 4B) + indptr (int32, 4B * (rows+1))
        csr_mem_kb = (sparse_layer.csr_weight.data.nbytes + sparse_layer.csr_weight.indices.nbytes + sparse_layer.csr_weight.indptr.nbytes) / 1024.0

        bench["target_sparsity"] = s
        bench["dense_mem_kb"] = dense_mem_kb
        bench["csr_mem_kb"] = csr_mem_kb
        benchmarks.append(bench)

        print(
            f"Sparsity: {bench['sparsity']*100:4.1f}% | Active Params: {bench['active_params']:6,d} | "
            f"Active FLOPs: {bench['sparse_flops']:7,d} ({bench['theoretical_speedup']:4.1f}x) | "
            f"CSR Latency: {bench['sparse_csr_latency_ms']:.4f}ms vs Dense: {bench['dense_latency_ms']:.4f}ms | "
            f"Dense*0 Shortcut: {bench['dense_masked_latency_ms']:.4f}ms"
        )

    # Save to artifacts
    out_json = os.path.join("artifacts", "cost_benchmark.json")
    with open(out_json, "w") as f:
        json.dump(benchmarks, f, indent=2)
    print(f"\nSaved cost measurements to {out_json}")

    # Plot FLOPs and Latency comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    sp_pct = [b["sparsity"] * 100 for b in benchmarks]
    dense_flops = [b["dense_flops"] / 1e6 for b in benchmarks]
    sparse_flops = [b["sparse_flops"] / 1e6 for b in benchmarks]

    ax1.plot(sp_pct, dense_flops, label="Dense FLOPs (Constant)", color="black", linestyle="--", lw=2)
    ax1.plot(sp_pct, sparse_flops, label="Active Sparse FLOPs", color="#1f77b4", marker="o", lw=2.5)
    ax1.set_xlabel("Sparsity (%)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("MFLOPs per Batch", fontsize=11, fontweight="bold")
    ax1.set_title("True Work Reduction: Active FLOPs vs Sparsity", fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    csr_lat = [b["sparse_csr_latency_ms"] for b in benchmarks]
    dense_lat = [b["dense_latency_ms"] for b in benchmarks]
    dense_zero_lat = [b["dense_masked_latency_ms"] for b in benchmarks]

    ax2.plot(sp_pct, dense_lat, label="Dense GEMM Latency", color="black", linestyle="--", lw=2)
    ax2.plot(sp_pct, dense_zero_lat, label="Dense-Times-Zero Shortcut (No Speedup)", color="#d62728", linestyle=":", lw=2)
    ax2.plot(sp_pct, csr_lat, label="True Sparse CSR Latency", color="#2ca02c", marker="s", lw=2.5)
    ax2.set_xlabel("Sparsity (%)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Wall-Clock Latency (ms)", fontsize=11, fontweight="bold")
    ax2.set_title("Wall-Clock Inference: CSR vs Dense vs Dense*0", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plot_path = os.path.join("artifacts", "benchmark_flops.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"Saved cost benchmark plot to {plot_path}")


if __name__ == "__main__":
    main()
