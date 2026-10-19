"""
Autodiff Graph Topology and Gradient Accumulation Tests.
"""

import unittest
import numpy as np
from engine.tensor import Tensor


class TestAutodiff(unittest.TestCase):
    def test_diamond_graph_gradient_accumulation(self):
        # L = (2x + 3x)^2 = (5x)^2 = 25 x^2
        # dL/dx = 50 x
        x = Tensor(2.0, requires_grad=True)
        a = x * 2.0
        b = x * 3.0
        y = a + b
        loss = y * y
        loss.backward()

        expected_grad = 50.0 * 2.0
        self.assertAlmostEqual(x.grad.item(), expected_grad, places=5)

    def test_multi_branch_accumulation(self):
        # y = x * x + x
        # dy/dx = 2x + 1
        x = Tensor(3.0, requires_grad=True)
        y = x * x + x
        y.backward()
        self.assertAlmostEqual(x.grad.item(), 7.0, places=5)

    def test_tensor_detach(self):
        x = Tensor(4.0, requires_grad=True)
        y = x * 2.0
        y_detached = y.detach()
        self.assertFalse(y_detached.requires_grad)
        self.assertIsNone(y_detached.creator)

    def test_repeated_backward_accumulates_into_leaves(self):
        """Calling backward multiple times on the same graph accumulates linearly into leaf nodes."""
        x = Tensor(3.0, requires_grad=True)
        # Multi-layer graph: intermediate y and root loss
        y = x * 2.0
        loss = y * 3.0  # loss = 6x, dloss/dx = 6

        # Pass 1
        loss.backward()
        self.assertAlmostEqual(x.grad.item(), 6.0, places=5)
        self.assertAlmostEqual(y.grad.item(), 3.0, places=5)

        # Pass 2 without zero_grad: leaf node x should accumulate 6.0 + 6.0 = 12.0
        loss.backward()
        self.assertAlmostEqual(x.grad.item(), 12.0, places=5)
        # Intermediate node y was reset before backprop, so it represents single-pass grad
        self.assertAlmostEqual(y.grad.item(), 3.0, places=5)

        # Pass 3: 12.0 + 6.0 = 18.0
        loss.backward()
        self.assertAlmostEqual(x.grad.item(), 18.0, places=5)

    def test_zero_grad_resets_gradients(self):
        """Calling zero_grad resets leaf and intermediate gradients."""
        x = Tensor(2.5, requires_grad=True)
        y = x * 4.0
        y.backward()
        self.assertIsNotNone(x.grad)
        self.assertAlmostEqual(x.grad.item(), 4.0, places=5)

        x.zero_grad()
        self.assertIsNone(x.grad)

    def test_gradient_accumulation_across_microbatches(self):
        """Simulates micro-batch training: multiple forward/backward passes accumulate into shared leaf weights."""
        w = Tensor(np.array([[2.0, -1.0], [0.5, 3.0]], dtype=np.float32), requires_grad=True)
        
        # Batch 1
        x1 = Tensor(np.array([[1.0, 2.0]], dtype=np.float32))
        loss1 = (x1 @ w).sum()
        loss1.backward()
        # Grad from batch 1: x1.T @ ones = [[1, 1], [2, 2]]
        np.testing.assert_allclose(w.grad, np.array([[1.0, 1.0], [2.0, 2.0]], dtype=np.float32), atol=1e-5)

        # Batch 2 without zero_grad
        x2 = Tensor(np.array([[3.0, 1.0]], dtype=np.float32))
        loss2 = (x2 @ w).sum()
        loss2.backward()
        # Grad from batch 2: x2.T @ ones = [[3, 3], [1, 1]]
        # Total accumulated: [[4, 4], [3, 3]]
        np.testing.assert_allclose(w.grad, np.array([[4.0, 4.0], [3.0, 3.0]], dtype=np.float32), atol=1e-5)


if __name__ == "__main__":
    unittest.main()
