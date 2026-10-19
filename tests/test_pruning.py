"""
Pruning Schedule, Saliency, and Sparse Inference Unit Tests.
"""

import unittest
import numpy as np
from engine.tensor import Tensor
from prune.masked_linear import MaskedLinear
from prune.scheduler import CubicScheduler, OneShotScheduler
from prune.criteria import MagnitudeCriterion, TaylorSaliencyCriterion
from prune.sparse_engine import SparseLinear, count_flops


class TestPruning(unittest.TestCase):
    def test_cubic_scheduler_bounds(self):
        sched = CubicScheduler(initial_sparsity=0.0, target_sparsity=0.90, start_step=10, end_step=100, update_freq=5)
        self.assertEqual(sched.get_sparsity(0), 0.0)
        self.assertEqual(sched.get_sparsity(9), 0.0)
        self.assertEqual(sched.get_sparsity(100), 0.90)
        self.assertEqual(sched.get_sparsity(150), 0.90)

        # Monotonically increasing
        s_prev = 0.0
        for step in range(10, 101, 5):
            s_curr = sched.get_sparsity(step)
            self.assertGreaterEqual(s_curr, s_prev)
            s_prev = s_curr

    def test_sparse_linear_csr_equivalence(self):
        """SparseLinear CSR forward pass must match dense MaskedLinear output."""
        layer = MaskedLinear(in_features=8, out_features=6)
        x = np.random.randn(5, 8).astype(np.float32)

        # Apply 75% random sparsity
        mask = (np.random.rand(8, 6) > 0.75).astype(np.float32)
        layer.set_mask(mask)

        dense_out = layer(Tensor(x)).data
        sparse_layer = SparseLinear(layer)
        sparse_out = sparse_layer.forward(x)

        np.testing.assert_allclose(dense_out, sparse_out, atol=1e-5)

    def test_flops_counter(self):
        flops = count_flops(in_features=100, out_features=50, nnz=500, batch_size=2)
        # Dense: 2 * 100 * 50 * 2 = 20,000
        # Sparse: 2 * 500 * 2 = 2,000
        self.assertEqual(flops["dense_flops"], 20000)
        self.assertEqual(flops["sparse_flops"], 2000)
        self.assertAlmostEqual(flops["theoretical_speedup"], 10.0)


if __name__ == "__main__":
    unittest.main()
