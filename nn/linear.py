"""
Linear (Fully Connected) Layer.
"""

from __future__ import annotations
import numpy as np
from typing import Optional
from engine.tensor import Tensor
from nn.module import Module, Parameter
from nn.init import kaiming_normal, zeros


class Linear(Module):
    """
    Standard fully connected linear layer: y = x @ W + b.
    Weight shape: (in_features, out_features)
    Bias shape: (out_features,)
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

        if bias:
            b_data = zeros((out_features,))
            self.bias = Parameter(b_data, requires_grad=True, name=f"{name}.bias" if name else "bias")
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        out = x @ self.weight
        if self.bias is not None:
            out = out + self.bias
        return out

    def __repr__(self) -> str:
        return f"Linear(in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None})"
