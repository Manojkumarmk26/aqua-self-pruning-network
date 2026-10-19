"""
Activation Module Wrappers.
"""

from engine.tensor import Tensor
from nn.module import Module
import engine.ops as ops


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return ops.relu(x)

    def __repr__(self) -> str:
        return "ReLU()"


class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor:
        return ops.tanh(x)

    def __repr__(self) -> str:
        return "Tanh()"


class Sigmoid(Module):
    def forward(self, x: Tensor) -> Tensor:
        return ops.sigmoid(x)

    def __repr__(self) -> str:
        return "Sigmoid()"


class GELU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return ops.gelu(x)

    def __repr__(self) -> str:
        return "GELU()"
