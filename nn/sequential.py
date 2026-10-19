"""
Sequential Container Module.
"""

from typing import List
from engine.tensor import Tensor
from nn.module import Module


class Sequential(Module):
    """
    Sequential container that feeds the output of each module into the next.
    """
    def __init__(self, *layers: Module):
        super().__init__()
        self.layers: List[Module] = list(layers)
        for i, layer in enumerate(self.layers):
            setattr(self, f"layer_{i}", layer)

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    def __getitem__(self, idx: int) -> Module:
        return self.layers[idx]

    def __len__(self) -> int:
        return len(self.layers)

    def __repr__(self) -> str:
        layer_strs = "\n  ".join(f"({i}): {repr(layer)}" for i, layer in enumerate(self.layers))
        return f"Sequential(\n  {layer_strs}\n)"
