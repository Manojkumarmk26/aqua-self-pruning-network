"""
Automated Finite-Difference Gradient Checking Suite.
Validates analytical gradients of all engine operations against numerical approximations.
"""

import unittest
import numpy as np
from engine.tensor import Tensor
import engine.ops as ops


def numerical_gradient(f, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Compute two-sided central difference numerical gradient."""
    grad = np.zeros_like(x, dtype=np.float64)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        orig_val = x[idx]

        x[idx] = orig_val + eps
        pos_loss = float(f(x))

        x[idx] = orig_val - eps
        neg_loss = float(f(x))

        grad[idx] = (pos_loss - neg_loss) / (2.0 * eps)
        x[idx] = orig_val
        it.iternext()

    return grad


def check_gradient(f_forward, x: np.ndarray, eps: float = 1e-6, tol: float = 1e-5) -> float:
    """
    Compare analytical backward gradient against numerical gradient.
    Returns relative error: ||g_num - g_anal|| / (||g_num|| + ||g_anal|| + 1e-8).
    """
    x_64 = np.asarray(x, dtype=np.float64)
    t_x = Tensor(x_64.copy(), requires_grad=True)
    out = f_forward(t_x)
    out.backward()

    anal_grad = t_x.grad

    def scalar_f(x_arr):
        t = Tensor(x_arr, requires_grad=False)
        return f_forward(t).item()

    num_grad = numerical_gradient(scalar_f, x_64.copy(), eps=eps)

    diff = np.linalg.norm(num_grad - anal_grad)
    norm_sum = np.linalg.norm(num_grad) + np.linalg.norm(anal_grad) + 1e-8
    rel_error = float(diff / norm_sum)

    return rel_error


class TestGradientCheck(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)

    def test_add_grad(self):
        y = np.random.randn(4, 5).astype(np.float64)
        def f(x):
            return (x + Tensor(y)).sum()
        x = np.random.randn(4, 5).astype(np.float64)
        err = check_gradient(f, x)
        self.assertLess(err, 1e-5)

    def test_sub_grad(self):
        y = np.random.randn(4, 5).astype(np.float64)
        def f(x):
            return (x - Tensor(y)).sum()
        x = np.random.randn(4, 5).astype(np.float64)
        err = check_gradient(f, x)
        self.assertLess(err, 1e-5)

    def test_mul_grad(self):
        y = np.random.randn(4, 5).astype(np.float64)
        def f(x):
            return (x * Tensor(y)).sum()
        x = np.random.randn(4, 5).astype(np.float64)
        err = check_gradient(f, x)
        self.assertLess(err, 1e-5)

    def test_div_grad(self):
        y = np.random.uniform(1.0, 2.0, size=(4, 5)).astype(np.float64)
        def f(x):
            return (x / Tensor(y)).sum()
        x = np.random.uniform(1.0, 2.0, size=(4, 5)).astype(np.float64)
        err = check_gradient(f, x)
        self.assertLess(err, 1e-5)

    def test_neg_grad(self):
        def f(x):
            return (-x).sum()
        x = np.random.randn(3, 4).astype(np.float64)
        err = check_gradient(f, x)
        self.assertLess(err, 1e-5)

    def test_matmul_grad(self):
        w = np.random.randn(5, 3).astype(np.float64)
        def f(x):
            return (x @ Tensor(w)).sum()
        x = np.random.randn(4, 5).astype(np.float64)
        err = check_gradient(f, x)
        self.assertLess(err, 1e-5)

        # Test gradient w.r.t weights as well
        x_const = np.random.randn(4, 5).astype(np.float64)
        def f_w(weight):
            return (Tensor(x_const) @ weight).sum()
        err_w = check_gradient(f_w, w)
        self.assertLess(err_w, 1e-5)

    def test_sum_reduction(self):
        x = np.random.randn(3, 4, 5).astype(np.float64)
        err_full = check_gradient(lambda t: t.sum(), x)
        self.assertLess(err_full, 1e-5)

        err_axis = check_gradient(lambda t: t.sum(axis=1).sum(), x)
        self.assertLess(err_axis, 1e-5)

        err_keepdims = check_gradient(lambda t: t.sum(axis=(0, 2), keepdims=True).sum(), x)
        self.assertLess(err_keepdims, 1e-5)

    def test_mean_reduction(self):
        x = np.random.randn(3, 4, 5).astype(np.float64)
        err_full = check_gradient(lambda t: t.mean(), x)
        self.assertLess(err_full, 1e-5)

        err_axis = check_gradient(lambda t: t.mean(axis=1).sum(), x)
        self.assertLess(err_axis, 1e-5)

    def test_relu_grad(self):
        # Avoid point of non-differentiability at 0
        x = np.array([-2.5, -0.5, 0.5, 2.5, 4.0], dtype=np.float64)
        err = check_gradient(lambda t: t.relu().sum(), x)
        self.assertLess(err, 1e-5)

    def test_tanh_grad(self):
        x = np.random.randn(4, 4).astype(np.float64)
        err = check_gradient(lambda t: t.tanh().sum(), x)
        self.assertLess(err, 1e-5)

    def test_sigmoid_grad(self):
        x = np.random.randn(4, 4).astype(np.float64)
        err = check_gradient(lambda t: t.sigmoid().sum(), x)
        self.assertLess(err, 1e-5)

    def test_gelu_grad(self):
        x = np.random.randn(4, 4).astype(np.float64)
        err = check_gradient(lambda t: t.gelu().sum(), x)
        self.assertLess(err, 1e-5)

    def test_softmax_cross_entropy(self):
        targets = np.array([1, 0, 2, 1], dtype=np.int64)
        def f(logits):
            return ops.softmax_cross_entropy(logits, targets)
        logits = np.random.randn(4, 3).astype(np.float64)
        err = check_gradient(f, logits)
        self.assertLess(err, 1e-5)

    def test_broadcasting_grad(self):
        # (N, D) + (D,) broadcasting
        bias = np.random.randn(5).astype(np.float64)
        x = np.random.randn(4, 5).astype(np.float64)

        # Check gradient w.r.t bias (D,)
        def f_bias(b):
            return (Tensor(x) + b).sum()
        err_b = check_gradient(f_bias, bias)
        self.assertLess(err_b, 1e-5)

        # (N, D) * (1, D) broadcasting
        w = np.random.randn(1, 5).astype(np.float64)
        def f_w(weight):
            return (Tensor(x) * weight).sum()
        err_w = check_gradient(f_w, w)
        self.assertLess(err_w, 1e-5)

    def test_deep_composite_graph(self):
        # 2-layer MLP computation graph
        w1 = np.random.randn(6, 8).astype(np.float64)
        b1 = np.random.randn(8).astype(np.float64)
        w2 = np.random.randn(8, 4).astype(np.float64)
        b2 = np.random.randn(4).astype(np.float64)
        x = np.random.randn(5, 6).astype(np.float64)
        targets = np.array([0, 1, 2, 3, 1], dtype=np.int64)

        def f_w1(weight1):
            h = (Tensor(x) @ weight1 + Tensor(b1)).relu()
            logits = h @ Tensor(w2) + Tensor(b2)
            return ops.softmax_cross_entropy(logits, targets)

        err = check_gradient(f_w1, w1)
        self.assertLess(err, 1e-5)


if __name__ == "__main__":
    unittest.main()
