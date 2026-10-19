"""
Masked Linear Layer with In-Graph Masking and Shadow Gradient Tracking.
"""

from __future__ import annotations
import numpy as np
from typing import Optional
from engine.tensor import Tensor
from nn.module import Module, Parameter
from nn.init import kaiming_normal, zeros


class MaskedLinear(Module):
    """
    Masked Fully Connected Layer for Dynamic Sparsity Learning.
    
    Weights are coupled with a binary mask M in the computation graph:
        W_eff = W * M
        y = x @ W_eff + b
        
    Properties:
    1. A masked connection (M_ij = 0) contributes exactly 0 to the forward pass.
    2. In the backward pass, dL/dW_ij = (dL/dW_eff_ij) * M_ij = 0.
    3. The shadow gradient (dL/dW_eff) is retained for sensitivity/regrowth analysis.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        nonlinearity: str = "relu",
        name: Optional[str] = None
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        w_data = kaiming_normal((in_features, out_features), fan_in=in_features, nonlinearity=nonlinearity)
        self.weight = Parameter(w_data, requires_grad=True, name=f"{name}.weight" if name else "weight")

        # Binary mask: 1 = active, 0 = pruned
        self.mask = np.ones((in_features, out_features), dtype=np.float32)
        self.weight.mask = self.mask

        if bias:
            b_data = zeros((out_features,))
            self.bias = Parameter(b_data, requires_grad=True, name=f"{name}.bias" if name else "bias")
        else:
            self.bias = None

        # Cache for effective weight tensor and shadow gradient
        self._w_eff_tensor: Optional[Tensor] = None
        self.shadow_grad: Optional[np.ndarray] = None

    def set_mask(self, new_mask: np.ndarray):
        """Update mask and enforce zero on pruned weights."""
        self.mask = new_mask.astype(np.float32)
        self.weight.mask = self.mask
        # Pruned weights are explicitly set to 0.0
        self.weight.data[self.mask == 0.0] = 0.0

    @property
    def sparsity(self) -> float:
        """Return current sparsity fraction of the weight matrix (0.0 to 1.0)."""
        return float(1.0 - (np.count_nonzero(self.mask) / self.mask.size))

    @property
    def num_active_params(self) -> int:
        """Return count of active non-zero weight parameters."""
        return int(np.count_nonzero(self.mask))

    @property
    def total_params(self) -> int:
        return self.weight.size

    def forward(self, x: Tensor) -> Tensor:
        # Honest graph embedding: W_eff = W * mask
        mask_tensor = Tensor(self.mask, requires_grad=False)
        self._w_eff_tensor = self.weight * mask_tensor

        out = x @ self._w_eff_tensor
        if self.bias is not None:
            out = out + self.bias
        return out

    def capture_shadow_grad(self):
        """
        Extract the shadow gradient dL/dW_eff from the computation graph.
        Represents the virtual sensitivity if pruned weights were active.
        """
        if self._w_eff_tensor is not None and self._w_eff_tensor.grad is not None:
            self.shadow_grad = self._w_eff_tensor.grad.copy()

    def __repr__(self) -> str:
        return (
            f"MaskedLinear(in_features={self.in_features}, out_features={self.out_features}, "
            f"sparsity={self.sparsity * 100:.1f}%, active={self.num_active_params}/{self.total_params})"
        )
