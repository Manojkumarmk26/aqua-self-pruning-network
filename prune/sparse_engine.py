"""
True Sparse Inference Engine and Real Cost Measurement Primitives.
Implements Compressed Sparse Row (CSR) execution to avoid multiplying by zeros,
measuring true active FLOPs and wall-clock inference speedups.
"""

from __future__ import annotations
import time
import numpy as np
import scipy.sparse as sp
from typing import Dict, Any, Tuple
from prune.masked_linear import MaskedLinear


class SparseLinear:
    """
    Inference-only Compressed Sparse Row (CSR) Linear Layer.
    Extracts the non-zero parameters from a trained MaskedLinear layer and executes
    forward inference without performing operations or memory reads on zeros.
    """
    def __init__(self, masked_layer: MaskedLinear):
        self.in_features = masked_layer.in_features
        self.out_features = masked_layer.out_features

        # Extract active weights: W * mask
        effective_w = masked_layer.weight.data * masked_layer.mask
        
        # Convert to Scipy CSR format
        self.csr_weight = sp.csr_matrix(effective_w, dtype=np.float32)
        self.bias = masked_layer.bias.data.copy() if masked_layer.bias is not None else None

        self.nnz = self.csr_weight.nnz
        self.total_weights = self.in_features * self.out_features
        self.sparsity = 1.0 - (self.nnz / self.total_weights)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        True sparse matrix multiplication: y = x @ W_csr + b.
        Does not perform dense multiplies by zero.
        """
        # x is (batch_size, in_features)
        # csr_weight is (in_features, out_features)
        out = x @ self.csr_weight
        if self.bias is not None:
            out += self.bias
        return out


def count_flops(in_features: int, out_features: int, nnz: int, batch_size: int = 1) -> Dict[str, int]:
    """
    Compute theoretical floating point operations (FLOPs).
    Each multiply-accumulate (MAC) is counted as 2 FLOPs (1 multiply + 1 add).
    """
    dense_flops = 2 * in_features * out_features * batch_size
    sparse_flops = 2 * nnz * batch_size
    return {
        "dense_flops": dense_flops,
        "sparse_flops": sparse_flops,
        "theoretical_speedup": dense_flops / max(sparse_flops, 1)
    }


def benchmark_layer_inference(
    layer: MaskedLinear,
    batch_size: int = 64,
    num_runs: int = 200
) -> Dict[str, Any]:
    """
    Rigorous wall-clock benchmark comparing:
    1. Standard Dense Forward pass (full dense GEMM)
    2. Dense-times-zero shortcut (multiplying masked dense matrix)
    3. True Sparse CSR forward pass (bypassing zeros entirely)
    """
    # Sample input batch
    x = np.random.randn(batch_size, layer.in_features).astype(np.float32)

    dense_w = layer.weight.data.copy()
    masked_dense_w = dense_w * layer.mask
    sparse_layer = SparseLinear(layer)
    bias = layer.bias.data.copy() if layer.bias is not None else None

    # Warmup
    for _ in range(20):
        _ = x @ dense_w
        _ = x @ masked_dense_w
        _ = sparse_layer.forward(x)

    # 1. Benchmark Standard Dense
    t0 = time.perf_counter()
    for _ in range(num_runs):
        out_dense = x @ dense_w
        if bias is not None:
            out_dense += bias
    dense_time = (time.perf_counter() - t0) / num_runs

    # 2. Benchmark Dense-with-zeros shortcut
    t0 = time.perf_counter()
    for _ in range(num_runs):
        out_masked = x @ masked_dense_w
        if bias is not None:
            out_masked += bias
    dense_masked_time = (time.perf_counter() - t0) / num_runs

    # 3. Benchmark True Sparse CSR
    t0 = time.perf_counter()
    for _ in range(num_runs):
        out_sparse = sparse_layer.forward(x)
    sparse_time = (time.perf_counter() - t0) / num_runs

    flops_info = count_flops(layer.in_features, layer.out_features, sparse_layer.nnz, batch_size=batch_size)

    return {
        "sparsity": sparse_layer.sparsity,
        "total_params": sparse_layer.total_weights,
        "active_params": sparse_layer.nnz,
        "dense_flops": flops_info["dense_flops"],
        "sparse_flops": flops_info["sparse_flops"],
        "theoretical_speedup": flops_info["theoretical_speedup"],
        "dense_latency_ms": dense_time * 1000.0,
        "dense_masked_latency_ms": dense_masked_time * 1000.0,
        "sparse_csr_latency_ms": sparse_time * 1000.0,
        "measured_speedup_vs_dense": dense_time / max(sparse_time, 1e-9)
    }
