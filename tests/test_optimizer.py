"""
Optimizer Convergence and Dynamics Unit Tests.
"""

import unittest
import numpy as np
from nn.module import Parameter
from optim.adam import Adam
from optim.sgd import SGD


class TestOptimizers(unittest.TestCase):
    def test_adam_quadratic_convergence(self):
        # Minimize f(w) = 0.5 * (w - 3)^2
        w = Parameter(np.array([10.0], dtype=np.float32), requires_grad=True)
        opt = Adam([w], lr=0.1)

        for _ in range(200):
            opt.zero_grad()
            loss = 0.5 * (w - 3.0) * (w - 3.0)
            loss.backward()
            opt.step()

        self.assertAlmostEqual(w.data[0], 3.0, places=2)

    def test_sgd_momentum_convergence(self):
        w = Parameter(np.array([10.0], dtype=np.float32), requires_grad=True)
        opt = SGD([w], lr=0.05, momentum=0.9)

        for _ in range(150):
            opt.zero_grad()
            loss = 0.5 * (w - 2.0) * (w - 2.0)
            loss.backward()
            opt.step()

        self.assertAlmostEqual(w.data[0], 2.0, places=2)


if __name__ == "__main__":
    unittest.main()
