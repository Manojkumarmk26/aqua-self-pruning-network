"""
Differentiable Operations for Reverse-Mode Autodiff Engine.
All operations implement forward() and backward() with full broadcasting support.
"""

from __future__ import annotations
import numpy as np
from typing import Tuple, Optional, Union
from engine.tensor import Op, Tensor, unbroadcast


class Add(Op):
    def forward(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return a + b

    def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Gradient for a is grad_output; for b is grad_output
        return grad_output, grad_output


class Sub(Op):
    def forward(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return a - b

    def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return grad_output, -grad_output


class Mul(Op):
    def forward(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        self.a = a
        self.b = b
        return a * b

    def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return grad_output * self.b, grad_output * self.a


class Div(Op):
    def forward(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        self.a = a
        self.b = b
        return a / b

    def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        grad_a = grad_output / self.b
        grad_b = -grad_output * self.a / (self.b ** 2)
        return grad_a, grad_b


class Neg(Op):
    def forward(self, a: np.ndarray) -> np.ndarray:
        return -a

    def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray]:
        return (-grad_output,)


class MatMul(Op):
    def forward(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        self.a = a
        self.b = b
        return a @ b

    def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Y = A @ B -> dL/dA = grad @ B^T, dL/dB = A^T @ grad
        grad_a = grad_output @ np.swapaxes(self.b, -1, -2)
        grad_b = np.swapaxes(self.a, -1, -2) @ grad_output
        return grad_a, grad_b


class Sum(Op):
    def forward(self, a: np.ndarray, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> np.ndarray:
        self.input_shape = a.shape
        self.axis = axis
        self.keepdims = keepdims
        return np.sum(a, axis=axis, keepdims=keepdims)

    def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray]:
        if self.axis is None:
            return (np.broadcast_to(grad_output, self.input_shape),)
        
        grad = grad_output
        if not self.keepdims:
            # Reinsert reduced axes for broadcasting
            axes = (self.axis,) if isinstance(self.axis, int) else tuple(sorted(self.axis))
            for ax in axes:
                grad = np.expand_dims(grad, axis=ax)
                
        return (np.broadcast_to(grad, self.input_shape),)


class Mean(Op):
    def forward(self, a: np.ndarray, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> np.ndarray:
        self.input_shape = a.shape
        self.axis = axis
        self.keepdims = keepdims
        
        if axis is None:
            self.count = a.size
        elif isinstance(axis, int):
            self.count = a.shape[axis]
        else:
            self.count = np.prod([a.shape[i] for i in axis])
            
        return np.mean(a, axis=axis, keepdims=keepdims)

    def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray]:
        grad = grad_output / self.count
        if self.axis is None:
            return (np.broadcast_to(grad, self.input_shape),)
        
        if not self.keepdims:
            axes = (self.axis,) if isinstance(self.axis, int) else tuple(sorted(self.axis))
            for ax in axes:
                grad = np.expand_dims(grad, axis=ax)
                
        return (np.broadcast_to(grad, self.input_shape),)


class ReLU(Op):
    def forward(self, a: np.ndarray) -> np.ndarray:
        self.mask = (a > 0.0)
        return np.maximum(0.0, a)

    def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray]:
        return (grad_output * self.mask.astype(grad_output.dtype),)


class Tanh(Op):
    def forward(self, a: np.ndarray) -> np.ndarray:
        self.out = np.tanh(a)
        return self.out

    def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray]:
        return (grad_output * (1.0 - self.out ** 2),)


class Sigmoid(Op):
    def forward(self, a: np.ndarray) -> np.ndarray:
        # Numerically stable sigmoid
        pos_mask = (a >= 0)
        neg_mask = ~pos_mask
        out = np.empty_like(a)
        out[pos_mask] = 1.0 / (1.0 + np.exp(-a[pos_mask]))
        exp_neg = np.exp(a[neg_mask])
        out[neg_mask] = exp_neg / (1.0 + exp_neg)
        self.out = out
        return out

    def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray]:
        return (grad_output * self.out * (1.0 - self.out),)


class GELU(Op):
    """
    Gaussian Error Linear Unit (GELU) with standard accurate tanh approximation:
    GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    """
    def forward(self, a: np.ndarray) -> np.ndarray:
        self.a = a
        c = np.sqrt(2.0 / np.pi)
        self.u = c * (a + 0.044715 * (a ** 3))
        self.tanh_u = np.tanh(self.u)
        return 0.5 * a * (1.0 + self.tanh_u)

    def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray]:
        c = np.sqrt(2.0 / np.pi)
        du_dx = c * (1.0 + 3.0 * 0.044715 * (self.a ** 2))
        sech2_u = 1.0 - self.tanh_u ** 2
        dgelu_dx = 0.5 * (1.0 + self.tanh_u) + 0.5 * self.a * sech2_u * du_dx
        return (grad_output * dgelu_dx,)


class SoftmaxCrossEntropy(Op):
    """
    Combined Softmax and Cross-Entropy Loss for numerical stability.
    Uses log-sum-exp trick to avoid overflow/underflow.
    """
    def forward(self, logits: np.ndarray, targets: np.ndarray) -> np.ndarray:
        self.logits = logits
        self.targets = targets
        self.N = logits.shape[0]

        # Log-Sum-Exp trick
        max_logits = np.max(logits, axis=-1, keepdims=True)
        shifted = logits - max_logits
        exp_shifted = np.exp(shifted)
        sum_exp = np.sum(exp_shifted, axis=-1, keepdims=True)
        self.log_probs = shifted - np.log(sum_exp)
        self.probs = exp_shifted / sum_exp

        # Targets can be 1D class indices (N,) or 2D one-hot (N, C)
        if targets.ndim == 1:
            indices = np.arange(self.N)
            loss = -np.mean(self.log_probs[indices, targets.astype(int)])
        else:
            loss = -np.mean(np.sum(targets * self.log_probs, axis=-1))

        return np.asarray(loss, dtype=logits.dtype)

    def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        grad_scale = grad_output / self.N
        grad_logits = self.probs.copy()

        if self.targets.ndim == 1:
            indices = np.arange(self.N)
            grad_logits[indices, self.targets.astype(int)] -= 1.0
        else:
            grad_logits -= self.targets

        grad_logits *= grad_scale
        # Targets are non-differentiable labels
        return grad_logits, None


def softmax_cross_entropy(logits: Tensor, targets: Union[Tensor, np.ndarray]) -> Tensor:
    """Helper functional API for combined softmax cross-entropy loss."""
    if not isinstance(targets, Tensor):
        targets = Tensor(targets, requires_grad=False)
    return SoftmaxCrossEntropy.apply(logits, targets)


def relu(x: Tensor) -> Tensor:
    return ReLU.apply(x)


def tanh(x: Tensor) -> Tensor:
    return Tanh.apply(x)


def sigmoid(x: Tensor) -> Tensor:
    return Sigmoid.apply(x)


def gelu(x: Tensor) -> Tensor:
    return GELU.apply(x)
