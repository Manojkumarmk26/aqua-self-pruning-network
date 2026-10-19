"""
Optimizer Base Class.
"""

from typing import List, Dict, Any
from nn.module import Parameter


class Optimizer:
    """Base class for all gradient-based optimizers."""

    def __init__(self, params: List[Parameter], lr: float = 1e-3, weight_decay: float = 0.0):
        self.params = list(params)
        self.lr = lr
        self.weight_decay = weight_decay
        self.state: Dict[int, Dict[str, Any]] = {}

    def zero_grad(self):
        """Zero the gradients of all managed parameters."""
        for p in self.params:
            p.zero_grad()

    def step(self):
        """Perform a single optimization step."""
        raise NotImplementedError
