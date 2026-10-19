"""
Pruner Engine Orchestrator.
Manages dynamic mask updates, scheduled sparsity ramps, and honest regrowth.
"""

from __future__ import annotations
import numpy as np
from typing import List, Optional
from nn.module import Module
from optim.optimizer import Optimizer
from prune.masked_linear import MaskedLinear
from prune.criteria import ImportanceCriterion, MagnitudeCriterion, TaylorSaliencyCriterion, RegrowthSaliencyCriterion
from prune.scheduler import PruningScheduler, CubicScheduler


class Pruner:
    """
    Self-Pruning Orchestrator that progressively prunes neural network connections
    during training according to an importance criterion and schedule.
    """
    def __init__(
        self,
        model: Module,
        scheduler: PruningScheduler,
        criterion: ImportanceCriterion,
        allow_regrowth: bool = False,
        regrowth_ratio: float = 0.15
    ):
        self.model = model
        self.scheduler = scheduler
        self.criterion = criterion
        self.allow_regrowth = allow_regrowth
        self.regrowth_ratio = regrowth_ratio
        self.cached_scores: dict[int, np.ndarray] = {}

        # Discover all MaskedLinear layers in model
        self.layers: List[MaskedLinear] = []
        for _, submodule in model.named_parameters():
            pass
        self._find_masked_layers(model)

    def _find_masked_layers(self, module: Module):
        if isinstance(module, MaskedLinear):
            self.layers.append(module)
        for sub in module._submodules.values():
            self._find_masked_layers(sub)

    @property
    def current_sparsity(self) -> float:
        """Calculate global sparsity across all masked layers."""
        total_elements = sum(layer.weight.size for layer in self.layers)
        active_elements = sum(layer.num_active_params for layer in self.layers)
        if total_elements == 0:
            return 0.0
        return 1.0 - (active_elements / total_elements)

    def capture_shadow_gradients(self):
        """Extract shadow gradients across all masked layers prior to zero_grad."""
        for layer in self.layers:
            layer.capture_shadow_grad()

    def compute_importance_scores(self):
        """
        Evaluate importance criteria on all layers prior to optimizer step.
        Ensures weights W_t and gradients g_t correspond to the exact same backward timestep.
        """
        for layer in self.layers:
            self.cached_scores[id(layer)] = self.criterion.compute_scores(layer)

    def step(self, global_step: int, optimizer: Optional[Optimizer] = None) -> bool:
        """
        Execute pruning step if scheduled.
        Returns True if masks were updated at this step.
        """
        if not self.scheduler.should_prune(global_step):
            return False

        target_sparsity = self.scheduler.get_sparsity(global_step)

        for layer in self.layers:
            old_mask = layer.mask.copy()
            layer_id = id(layer)
            if layer_id in self.cached_scores:
                scores = self.cached_scores.pop(layer_id)
            else:
                scores = self.criterion.compute_scores(layer)
            total = scores.size
            num_pruned_target = int(np.round(total * target_sparsity))
            num_active_target = total - num_pruned_target

            if not self.allow_regrowth:
                # Standard dynamic rank pruning
                flat_scores = scores.flatten()
                if num_active_target <= 0:
                    new_mask = np.zeros_like(layer.mask)
                elif num_active_target >= total:
                    new_mask = np.ones_like(layer.mask)
                else:
                    sorted_indices = np.argsort(flat_scores)
                    new_mask_flat = np.zeros(total, dtype=np.float32)
                    new_mask_flat[sorted_indices[num_pruned_target:]] = 1.0
                    new_mask = new_mask_flat.reshape(layer.mask.shape)
            else:
                # Dynamic Sparse Exploration with Regrowth (RigL / SET style)
                # First, ensure connection count matches target budget
                current_active_idx = np.where(old_mask.flatten() == 1.0)[0]
                current_inactive_idx = np.where(old_mask.flatten() == 0.0)[0]

                if len(current_active_idx) > num_active_target:
                    # Pruning phase: trim excess active connections based on Taylor saliency
                    active_scores = scores.flatten()[current_active_idx]
                    sorted_active = current_active_idx[np.argsort(active_scores)]
                    keep_idx = sorted_active[len(current_active_idx) - num_active_target:]
                    new_mask_flat = np.zeros(total, dtype=np.float32)
                    new_mask_flat[keep_idx] = 1.0
                    new_mask = new_mask_flat.reshape(layer.mask.shape)
                else:
                    # Rewiring phase at target sparsity budget:
                    # Drop lowest saliency active connections and regrow highest sensitivity inactive ones
                    num_to_regrow = int(np.round(num_active_target * self.regrowth_ratio))
                    if num_to_regrow > 0 and len(current_inactive_idx) >= num_to_regrow:
                        # 1. Select lowest saliency active connections to drop
                        active_scores = scores.flatten()[current_active_idx]
                        sorted_active = current_active_idx[np.argsort(active_scores)]
                        prune_candidates = sorted_active[:num_to_regrow]
                        surviving_active = sorted_active[num_to_regrow:]

                        # 2. Select highest gradient sensitivity inactive connections to revive
                        shadow_g = layer.shadow_grad if layer.shadow_grad is not None else np.zeros_like(layer.weight.data)
                        inactive_sens = np.abs(shadow_g).flatten()[current_inactive_idx]
                        sorted_inactive = current_inactive_idx[np.argsort(inactive_sens)[::-1]]
                        regrow_candidates = sorted_inactive[:num_to_regrow]

                        new_mask_flat = np.zeros(total, dtype=np.float32)
                        new_mask_flat[surviving_active] = 1.0
                        new_mask_flat[regrow_candidates] = 1.0
                        new_mask = new_mask_flat.reshape(layer.mask.shape)
                    else:
                        new_mask = old_mask.copy()

            # Update layer mask
            layer.set_mask(new_mask)

            # Inform optimizer to sanitize moments for revived connections
            if optimizer is not None and hasattr(optimizer, "clean_revived_state"):
                optimizer.clean_revived_state(layer.weight, old_mask, new_mask)

        return True
