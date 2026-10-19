"""
Importance Criteria for Pruning and Regrowth.
Includes Magnitude Baseline and Derived First-Order Taylor Saliency.
"""

from __future__ import annotations
import numpy as np
from typing import Dict
from prune.masked_linear import MaskedLinear


class ImportanceCriterion:
    """Base class for connection importance estimation."""
    def compute_scores(self, layer: MaskedLinear) -> np.ndarray:
        raise NotImplementedError


class MagnitudeCriterion(ImportanceCriterion):
    """
    Trivial Magnitude Baseline:
        S_ij = |W_ij|
    Assumes connection importance is purely proportional to absolute weight magnitude.
    """
    def compute_scores(self, layer: MaskedLinear) -> np.ndarray:
        return np.abs(layer.weight.data)


class TaylorSaliencyCriterion(ImportanceCriterion):
    """
    First-Order Taylor Expansion Saliency Criterion:
        Delta L_ij approx |W_ij * (dL / dW_ij)|

    Derivation:
    Removing connection (i, j) corresponds to perturbing parameter W by Delta W_ij = -W_ij.
    By Taylor series expansion of loss around W:
        L(W - W_ij e_ij) = L(W) - (dL / dW_ij) * W_ij + 1/2 H_ij,ij * (W_ij)^2 + O(||W_ij||^3)
    The first-order approximation of the magnitude of loss increase is:
        |Delta L_ij| approx |W_ij * g_ij|
    
    Includes an Exponential Moving Average (EMA) buffer to filter out 
    mini-batch stochastic gradient noise.
    """
    def __init__(self, ema_decay: float = 0.9):
        self.ema_decay = ema_decay
        self.ema_scores: Dict[int, np.ndarray] = {}

    def compute_scores(self, layer: MaskedLinear) -> np.ndarray:
        w = layer.weight.data
        # Use shadow gradient when available (contains unmasked virtual sensitivity dL/dW_eff)
        # For active weights, shadow_grad matches weight.grad identically.
        if layer.shadow_grad is not None:
            g = layer.shadow_grad
        elif layer.weight.grad is not None:
            g = layer.weight.grad
        else:
            g = np.zeros_like(w)

        instant_score = np.abs(w * g)
        layer_id = id(layer)

        if layer_id not in self.ema_scores:
            self.ema_scores[layer_id] = instant_score.copy()
        else:
            self.ema_scores[layer_id] = (
                self.ema_decay * self.ema_scores[layer_id] + (1.0 - self.ema_decay) * instant_score
            )

        return self.ema_scores[layer_id]


class RegrowthSaliencyCriterion(ImportanceCriterion):
    """
    Unified Importance Criterion supporting Regrowth (RigL / SET style):
    - For active connections (mask == 1):
        Score = |W_ij * g_ij| (Taylor saliency)
    - For pruned connections (mask == 0):
        Score = |shadow_grad_ij| (Directional sensitivity if connection is introduced)
    """
    def __init__(self, ema_decay: float = 0.9):
        self.ema_decay = ema_decay
        self.ema_scores: Dict[int, np.ndarray] = {}

    def compute_scores(self, layer: MaskedLinear) -> np.ndarray:
        w = layer.weight.data
        g = layer.weight.grad if layer.weight.grad is not None else np.zeros_like(w)
        shadow_g = layer.shadow_grad if layer.shadow_grad is not None else np.zeros_like(w)

        # Active score based on Taylor saliency
        active_score = np.abs(w * g)
        # Pruned score based on gradient sensitivity
        pruned_score = np.abs(shadow_g)

        combined = np.where(layer.mask == 1.0, active_score, pruned_score)

        layer_id = id(layer)
        if layer_id not in self.ema_scores:
            self.ema_scores[layer_id] = combined.copy()
        else:
            self.ema_scores[layer_id] = (
                self.ema_decay * self.ema_scores[layer_id] + (1.0 - self.ema_decay) * combined
            )

        return self.ema_scores[layer_id]
