"""
Autodiff Engine - Tensor and Computation Graph Primitives.
Built from scratch in pure Python and NumPy.
"""

from __future__ import annotations
import numpy as np
from typing import Optional, Tuple, List, Union, Set


def unbroadcast(grad: np.ndarray, target_shape: tuple) -> np.ndarray:
    """
    Sum out broadcasted dimensions of grad so it matches target_shape.
    Handles numpy broadcasting rules in reverse-mode automatic differentiation.
    """
    grad = np.asarray(grad)
    if grad.shape == target_shape:
        return grad
    if len(target_shape) == 0:
        return np.sum(grad)
    
    # 1. Sum leading dimensions that were added by broadcasting
    ndim_diff = grad.ndim - len(target_shape)
    for _ in range(ndim_diff):
        grad = grad.sum(axis=0)
    
    # 2. Sum dimensions where target has size 1 but grad has size > 1
    for axis, (g_dim, t_dim) in enumerate(zip(grad.shape, target_shape)):
        if t_dim == 1 and g_dim != 1:
            grad = grad.sum(axis=axis, keepdims=True)
            
    return grad.reshape(target_shape)


class Tensor:
    """
    Tensor wrapping a NumPy array.
    Tracks parents in the computation graph, caches forward activations,
    and accumulates gradients in the backward pass.
    """

    def __init__(
        self,
        data: Union[int, float, list, np.ndarray, np.generic],
        requires_grad: bool = False,
        creator: Optional[Op] = None,
        name: Optional[str] = None,
        dtype: Optional[Union[np.dtype, type]] = None,
    ):
        if isinstance(data, (np.ndarray, np.generic)):
            if dtype is not None and data.dtype != dtype:
                self.data = np.asarray(data, dtype=dtype)
            else:
                self.data = np.asarray(data)
        else:
            self.data = np.asarray(data, dtype=dtype if dtype is not None else np.float32)

        self.requires_grad = requires_grad
        self.creator = creator
        self.grad: Optional[np.ndarray] = None
        self.name = name

        # Optional parameter mask (1 = active, 0 = pruned/masked)
        self.mask: Optional[np.ndarray] = None

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def size(self) -> int:
        return self.data.size

    @property
    def dtype(self):
        return self.data.dtype

    def zero_grad(self) -> None:
        """Reset gradient to None."""
        self.grad = None

    def detach(self) -> Tensor:
        """Return a new Tensor detached from the current computation graph."""
        return Tensor(self.data.copy(), requires_grad=False)

    def numpy(self) -> np.ndarray:
        return self.data

    def item(self) -> Union[int, float]:
        return self.data.item()

    def __repr__(self) -> str:
        grad_info = f", grad_fn=<{self.creator.__class__.__name__}>" if self.creator else ""
        if self.requires_grad and self.creator is None:
            grad_info = ", requires_grad=True"
        return f"Tensor({self.data}{grad_info})"

    def backward(self, grad: Optional[Union[np.ndarray, Tensor]] = None) -> None:
        """
        Execute reverse-mode automatic differentiation starting from this node.
        Builds a reverse topological order of all ancestor nodes and backpropagates
        gradient accumulations through each creator Op.
        """
        if not self.requires_grad and self.creator is None:
            raise RuntimeError("Called backward() on a Tensor that does not require grad.")

        if grad is None:
            if self.size == 1:
                self.grad = np.ones_like(self.data, dtype=self.data.dtype)
            else:
                raise RuntimeError("grad can only be implicitly created for scalar outputs. Provide grad explicitly.")
        else:
            if isinstance(grad, Tensor):
                grad = grad.data
            self.grad = np.asarray(grad, dtype=self.data.dtype)

        # Build topological sort (DFS post-order traversal reversed)
        topo: List[Tensor] = []
        visited: Set[int] = set()

        def build_topo(v: Tensor):
            v_id = id(v)
            if v_id not in visited:
                visited.add(v_id)
                if v.creator is not None:
                    for parent in v.creator.inputs:
                        if isinstance(parent, Tensor):
                            build_topo(parent)
                topo.append(v)

        build_topo(self)

        # Clear intermediate node gradients from prior backward passes to prevent
        # stale accumulation into non-leaf computation nodes across repeated backward calls.
        for node in topo:
            if node is not self and node.creator is not None:
                node.grad = None

        # Backpropagate in reverse topological order
        for node in reversed(topo):
            if node.creator is None or node.grad is None:
                continue

            # Compute input gradients from node.grad
            parent_grads = node.creator.backward(node.grad)
            if not isinstance(parent_grads, tuple):
                parent_grads = (parent_grads,)

            for parent, p_grad in zip(node.creator.inputs, parent_grads):
                if isinstance(parent, Tensor) and parent.requires_grad:
                    if p_grad is None:
                        continue
                    p_grad_arr = unbroadcast(p_grad, parent.shape)
                    if parent.grad is None:
                        parent.grad = p_grad_arr.copy()
                    else:
                        parent.grad = parent.grad + p_grad_arr

    # Operator Overloads
    def __add__(self, other: Union[Tensor, float, int, np.ndarray]) -> Tensor:
        from engine.ops import Add
        return Add.apply(self, other)

    def __radd__(self, other: Union[Tensor, float, int, np.ndarray]) -> Tensor:
        from engine.ops import Add
        return Add.apply(other, self)

    def __sub__(self, other: Union[Tensor, float, int, np.ndarray]) -> Tensor:
        from engine.ops import Sub
        return Sub.apply(self, other)

    def __rsub__(self, other: Union[Tensor, float, int, np.ndarray]) -> Tensor:
        from engine.ops import Sub
        return Sub.apply(other, self)

    def __mul__(self, other: Union[Tensor, float, int, np.ndarray]) -> Tensor:
        from engine.ops import Mul
        return Mul.apply(self, other)

    def __rmul__(self, other: Union[Tensor, float, int, np.ndarray]) -> Tensor:
        from engine.ops import Mul
        return Mul.apply(other, self)

    def __truediv__(self, other: Union[Tensor, float, int, np.ndarray]) -> Tensor:
        from engine.ops import Div
        return Div.apply(self, other)

    def __rtruediv__(self, other: Union[Tensor, float, int, np.ndarray]) -> Tensor:
        from engine.ops import Div
        return Div.apply(other, self)

    def __neg__(self) -> Tensor:
        from engine.ops import Neg
        return Neg.apply(self)

    def __matmul__(self, other: Tensor) -> Tensor:
        from engine.ops import MatMul
        return MatMul.apply(self, other)

    def sum(self, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> Tensor:
        from engine.ops import Sum
        return Sum.apply(self, axis=axis, keepdims=keepdims)

    def mean(self, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> Tensor:
        from engine.ops import Mean
        return Mean.apply(self, axis=axis, keepdims=keepdims)

    def relu(self) -> Tensor:
        from engine.ops import ReLU
        return ReLU.apply(self)

    def tanh(self) -> Tensor:
        from engine.ops import Tanh
        return Tanh.apply(self)

    def sigmoid(self) -> Tensor:
        from engine.ops import Sigmoid
        return Sigmoid.apply(self)

    def gelu(self) -> Tensor:
        from engine.ops import GELU
        return GELU.apply(self)


class Op:
    """Base class for differentiable operations in computation graph."""
    inputs: Tuple[Union[Tensor, any], ...]

    @classmethod
    def apply(cls, *args, **kwargs) -> Tensor:
        op = cls()
        tensor_inputs = []
        requires_grad = False

        for arg in args:
            if isinstance(arg, Tensor):
                tensor_inputs.append(arg)
                if arg.requires_grad:
                    requires_grad = True
            elif isinstance(arg, (int, float, np.ndarray)):
                # Convert raw numeric scalars/arrays to non-differentiable tensors
                t = Tensor(arg, requires_grad=False)
                tensor_inputs.append(t)
            else:
                tensor_inputs.append(arg)

        op.inputs = tuple(tensor_inputs)
        out_data = op.forward(*(t.data if isinstance(t, Tensor) else t for t in tensor_inputs), **kwargs)
        
        creator = op if requires_grad else None
        return Tensor(out_data, requires_grad=requires_grad, creator=creator)

    def forward(self, *args, **kwargs) -> np.ndarray:
        raise NotImplementedError

    def backward(self, grad_output: np.ndarray) -> Tuple[Optional[np.ndarray], ...]:
        raise NotImplementedError
