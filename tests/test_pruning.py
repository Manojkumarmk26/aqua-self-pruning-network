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

    def test_exact_multi_layer_achieved_sparsity(self):
        """Verify exact per-layer and global achieved sparsity across multiple layers."""
        import nn
        from prune.pruner import Pruner
        from prune.scheduler import OneShotScheduler

        model = nn.Sequential(
            MaskedLinear(in_features=20, out_features=30, name="fc1"),
            nn.ReLU(),
            MaskedLinear(in_features=30, out_features=10, name="fc2")
        )
        # Total weights: 20*30 + 30*10 = 600 + 300 = 900
        target_s = 0.80
        sched = OneShotScheduler(target_sparsity=target_s, prune_step=1)
        pruner = Pruner(model, sched, MagnitudeCriterion())

        pruned = pruner.step(global_step=1)
        self.assertTrue(pruned)

        # Layer 1: 600 weights -> 80% pruned = 480 pruned, 120 active
        self.assertEqual(pruner.layers[0].num_active_params, 120)
        self.assertAlmostEqual(pruner.layers[0].sparsity, 0.80, places=4)

        # Layer 2: 300 weights -> 80% pruned = 240 pruned, 60 active
        self.assertEqual(pruner.layers[1].num_active_params, 60)
        self.assertAlmostEqual(pruner.layers[1].sparsity, 0.80, places=4)

        # Global sparsity: (120 + 60) / 900 = 180 / 900 = 0.20 active -> 0.80 sparsity
        self.assertAlmostEqual(pruner.current_sparsity, 0.80, places=4)

    def test_pruner_score_caching_ordering(self):
        """Verify importance score caching cleanly caches and consumes scores before optimizer step."""
        import nn
        from prune.pruner import Pruner
        from prune.scheduler import OneShotScheduler

        model = nn.Sequential(MaskedLinear(in_features=5, out_features=5, name="fc1"))
        sched = OneShotScheduler(target_sparsity=0.60, prune_step=1)
        pruner = Pruner(model, sched, MagnitudeCriterion())

        # Precompute scores
        pruner.compute_importance_scores()
        self.assertIn(id(pruner.layers[0]), pruner.cached_scores)

        # Step consumes cached scores
        pruner.step(global_step=1)
        self.assertNotIn(id(pruner.layers[0]), pruner.cached_scores)
        self.assertEqual(pruner.layers[0].num_active_params, 10)  # 25 * 0.4 = 10 active


if __name__ == "__main__":
    unittest.main()
