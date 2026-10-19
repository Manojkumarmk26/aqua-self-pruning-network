"""
Stochastic Gradient Descent (SGD) with Momentum and Honest Mask Awareness.
"""

import numpy as np
from typing import List
from nn.module import Parameter
from optim.optimizer import Optimizer


class SGD(Optimizer):
    """
    SGD with classical momentum and mask-aware state handling.
    """
    def __init__(
        self,
        params: List[Parameter],
        lr: float = 1e-2,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False
    ):
        super().__init__(params, lr=lr, weight_decay=weight_decay)
        self.momentum = momentum
        self.nesterov = nesterov

    def step(self):
        for p in self.params:
            if p.grad is None:
                continue

            grad = p.grad.copy()

            # Enforce honest parameter masking
            if p.mask is not None:
                grad *= p.mask

            if self.weight_decay != 0.0:
                grad += self.weight_decay * p.data

            p_id = id(p)
            if p_id not in self.state:
                self.state[p_id] = {
                    "velocity": np.zeros_like(p.data)
                }

            state = self.state[p_id]
            v = state["velocity"]

            if self.momentum != 0.0:
                v[:] = self.momentum * v + grad
                if self.nesterov:
                    step_val = grad + self.momentum * v
                else:
                    step_val = v
            else:
                step_val = grad

            p.data -= self.lr * step_val

            # Enforce mask invariance on data and velocity
            if p.mask is not None:
                p.data[p.mask == 0] = 0.0
                v[p.mask == 0] = 0.0
