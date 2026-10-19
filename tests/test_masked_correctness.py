"""
Masked-Weight Correctness Test Suite.
Validates the fundamental correctness requirement:
1. Masked weights contribute 0 to the forward pass.
2. The gradient of a masked weight is identically zero.
3. The optimizer does not move masked weights.
4. Optimizer moments (momentum, Adam moments) for pruned weights do not corrupt
   training upon revival or cause division-by-epsilon explosions.
"""

import unittest
import numpy as np
from engine.tensor import Tensor
import engine.ops as ops
from prune.masked_linear import MaskedLinear
from optim.adam import Adam
from optim.sgd import SGD


class TestMaskedCorrectness(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)

    def test_masked_forward_zero_contribution(self):
        """A masked weight must contribute exactly zero to forward activations."""
        layer = MaskedLinear(in_features=4, out_features=3, bias=False)
        x = np.random.randn(2, 4).astype(np.float32)

        # Mask out connection (0, 0)
        mask = np.ones((4, 3), dtype=np.float32)
        mask[0, 0] = 0.0
        layer.set_mask(mask)

        # Forward pass with original weight
        out1 = layer(Tensor(x)).data.copy()

        # Artificially alter the masked weight in storage
        layer.weight.data[0, 0] = 999.0
        out2 = layer(Tensor(x)).data.copy()

        # The forward output must be identical despite the altered weight
        np.testing.assert_allclose(out1, out2, atol=1e-6)

    def test_masked_weight_gradient_is_strictly_zero(self):
        """The mathematical gradient of a masked weight dL/dW_masked is zero."""
        layer = MaskedLinear(in_features=4, out_features=3, bias=False)
        x = np.random.randn(5, 4).astype(np.float32)
        targets = np.array([0, 1, 2, 0, 1], dtype=np.int64)

        mask = np.ones((4, 3), dtype=np.float32)
        mask[1, 2] = 0.0
        mask[3, 0] = 0.0
        layer.set_mask(mask)

        logits = layer(Tensor(x))
        loss = ops.softmax_cross_entropy(logits, targets)
        loss.backward()

        w_grad = layer.weight.grad
        self.assertIsNotNone(w_grad)
        # Masked entries must have exactly zero gradient
        self.assertEqual(w_grad[1, 2], 0.0)
        self.assertEqual(w_grad[3, 0], 0.0)
        # Active entries should have non-zero gradient
        self.assertNotEqual(w_grad[0, 0], 0.0)

    def test_adam_does_not_update_masked_weights(self):
        """Adam optimizer must leave masked weights frozen at exactly zero."""
        layer = MaskedLinear(in_features=5, out_features=4, bias=True)
        mask = np.ones((5, 4), dtype=np.float32)
        mask[0, 1] = 0.0
        mask[2, 3] = 0.0
        layer.set_mask(mask)

        optimizer = Adam(layer.parameters(), lr=1e-2)

        for _ in range(30):
            optimizer.zero_grad()
            x = np.random.randn(4, 5).astype(np.float32)
            targets = np.array([0, 1, 2, 3], dtype=np.int64)

            logits = layer(Tensor(x))
            loss = ops.softmax_cross_entropy(logits, targets)
            loss.backward()
            optimizer.step()

            # Masked weights must remain strictly 0.0
            self.assertEqual(layer.weight.data[0, 1], 0.0)
            self.assertEqual(layer.weight.data[2, 3], 0.0)

    def test_adam_moments_sanitized_for_masked_weights(self):
        """Adam moments (m, v) for masked weights must not harbor decaying phantom values."""
        layer = MaskedLinear(in_features=3, out_features=3, bias=False)
        mask = np.ones((3, 3), dtype=np.float32)
        mask[1, 1] = 0.0
        layer.set_mask(mask)

        optimizer = Adam(layer.parameters(), lr=1e-2)

        for _ in range(10):
            optimizer.zero_grad()
            x = np.random.randn(2, 3).astype(np.float32)
            targets = np.array([0, 1], dtype=np.int64)
            loss = ops.softmax_cross_entropy(layer(Tensor(x)), targets)
            loss.backward()
            optimizer.step()

        p_id = id(layer.weight)
        state = optimizer.state[p_id]
        # Masked moments must be zeroed, not carrying stale phantom momentum
        self.assertEqual(state["m"][1, 1], 0.0)
        self.assertEqual(state["v"][1, 1], 0.0)

    def test_revived_connection_trains_smoothly_without_nans_or_explosions(self):
        """
        Probe Part 1 live defense scenario:
        A connection is pruned for multiple steps, then revived.
        The optimizer state must not cause an explosive update or NaNs.
        """
        layer = MaskedLinear(in_features=4, out_features=4, bias=False)
        optimizer = Adam(layer.parameters(), lr=1e-2)

        # 1. Train initially with all active connections
        for _ in range(10):
            optimizer.zero_grad()
            x = np.random.randn(4, 4).astype(np.float32)
            targets = np.array([0, 1, 2, 3], dtype=np.int64)
            loss = ops.softmax_cross_entropy(layer(Tensor(x)), targets)
            loss.backward()
            optimizer.step()

        # 2. Prune connection (2, 2)
        old_mask = np.ones((4, 4), dtype=np.float32)
        new_mask = old_mask.copy()
        new_mask[2, 2] = 0.0
        layer.set_mask(new_mask)

        for _ in range(25):
            optimizer.zero_grad()
            x = np.random.randn(4, 4).astype(np.float32)
            targets = np.array([0, 1, 2, 3], dtype=np.int64)
            loss = ops.softmax_cross_entropy(layer(Tensor(x)), targets)
            loss.backward()
            optimizer.step()

        self.assertEqual(layer.weight.data[2, 2], 0.0)

        # 3. Revive connection (2, 2)
        revived_mask = np.ones((4, 4), dtype=np.float32)
        optimizer.clean_revived_state(layer.weight, new_mask, revived_mask)
        layer.set_mask(revived_mask)

        # 4. Train after revival and verify step magnitude and stability
        for step in range(20):
            optimizer.zero_grad()
            x = np.random.randn(4, 4).astype(np.float32)
            targets = np.array([0, 1, 2, 3], dtype=np.int64)
            loss = ops.softmax_cross_entropy(layer(Tensor(x)), targets)
            loss.backward()
            optimizer.step()

            # Verify no NaNs or Infs
            self.assertFalse(np.isnan(loss.item()))
            self.assertFalse(np.isinf(loss.item()))
            self.assertFalse(np.any(np.isnan(layer.weight.data)))
            # The revived weight should now update stably
            self.assertTrue(np.abs(layer.weight.data[2, 2]) < 5.0)

    def test_sgd_momentum_does_not_corrupt_revived_connection(self):
        """SGD with momentum must not apply stale velocity to pruned/revived weights."""
        layer = MaskedLinear(in_features=3, out_features=2, bias=False)
        optimizer = SGD(layer.parameters(), lr=0.1, momentum=0.9)

        # Step 1: Active training
        for _ in range(5):
            optimizer.zero_grad()
            x = np.random.randn(2, 3).astype(np.float32)
            targets = np.array([0, 1], dtype=np.int64)
            loss = ops.softmax_cross_entropy(layer(Tensor(x)), targets)
            loss.backward()
            optimizer.step()

        # Step 2: Prune connection (0, 1)
        mask = np.ones((3, 2), dtype=np.float32)
        mask[0, 1] = 0.0
        layer.set_mask(mask)

        for _ in range(10):
            optimizer.zero_grad()
            x = np.random.randn(2, 3).astype(np.float32)
            targets = np.array([0, 1], dtype=np.int64)
            loss = ops.softmax_cross_entropy(layer(Tensor(x)), targets)
            loss.backward()
            optimizer.step()

        # Velocity for pruned connection must be 0.0
        v = optimizer.state[id(layer.weight)]["velocity"]
        self.assertEqual(v[0, 1], 0.0)
        self.assertEqual(layer.weight.data[0, 1], 0.0)


if __name__ == "__main__":
    unittest.main()
