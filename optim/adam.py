"""
Adam Optimizer with Honest Mask Handling and Momentum-Corruption Prevention.
"""

from __future__ import annotations
import numpy as np
from typing import List, Tuple
from nn.module import Parameter
from optim.optimizer import Optimizer


class Adam(Optimizer):
    """
    Adam optimizer with decoupled weight decay (AdamW style) and
    rigorous mask-aware state handling.

    Addresses the core correctness requirement:
    - Masked connections receive zero gradient.
    - Masked weights are strictly frozen at 0.0.
    - Optimizer moments (m, v) for masked connections are zeroed to prevent
      phantom momentum decay and catastrophic division-by-epsilon explosions upon revival.
    """
    def __init__(
        self,
        params: List[Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0
    ):
        super().__init__(params, lr=lr, weight_decay=weight_decay)
        self.beta1, self.beta2 = betas
        self.eps = eps

    def step(self):
        for p in self.params:
            if p.grad is None:
                continue

            grad = p.grad.copy()

            # Ensure gradient of masked parameters is strictly zero
            if p.mask is not None:
                grad *= p.mask

            p_id = id(p)
            if p_id not in self.state:
                self.state[p_id] = {
                    "step": 0,
                    "m": np.zeros_like(p.data),
                    "v": np.zeros_like(p.data)
                }

            state = self.state[p_id]
            state["step"] += 1
            t = state["step"]
            m = state["m"]
            v = state["v"]

            # Decoupled weight decay on active weights
            if self.weight_decay != 0.0:
                p.data -= self.lr * self.weight_decay * p.data

            # Update biased 1st and 2nd moment estimates
            m[:] = self.beta1 * m + (1.0 - self.beta1) * grad
            v[:] = self.beta2 * v + (1.0 - self.beta2) * (grad ** 2)

            # Bias corrections
            bias_correction1 = 1.0 - (self.beta1 ** t)
            bias_correction2 = 1.0 - (self.beta2 ** t)
            
            m_hat = m / bias_correction1
            v_hat = v / bias_correction2

            # Compute Adam update
            step_val = m_hat / (np.sqrt(v_hat) + self.eps)
            p.data -= self.lr * step_val

            # Enforce honest mask invariance on parameters and moments
            if p.mask is not None:
                pruned = (p.mask == 0)
                # Frozen at zero: masked weights do not move
                p.data[pruned] = 0.0
                # Zero out moments for pruned weights to prevent phantom decay & revival spikes
                m[pruned] = 0.0
                v[pruned] = 0.0

    def clean_revived_state(self, p: Parameter, old_mask: np.ndarray, new_mask: np.ndarray):
        """
        Explicitly cleans and warm-starts optimizer moments for newly revived connections.
        Prevents stale momentum from corrupting revived weights.
        """
        p_id = id(p)
        if p_id not in self.state:
            return

        state = self.state[p_id]
        revived = (old_mask == 0) & (new_mask == 1)
        if np.any(revived):
            # Reset revived moments cleanly
            state["m"][revived] = 0.0
            # Initialize revived second moment to active average to prevent large initial step
            active = (new_mask == 1) & (~revived)
            if np.any(active) and np.mean(state["v"][active]) > 0:
                state["v"][revived] = np.mean(state["v"][active])
            else:
                state["v"][revived] = 0.0
