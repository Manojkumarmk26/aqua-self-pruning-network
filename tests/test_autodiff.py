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


if __name__ == "__main__":
    unittest.main()
