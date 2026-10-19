"""
Pruning Schedules for Dynamic Sparsity Ramp.
"""

from typing import Optional


class PruningScheduler:
    """Base pruning schedule interface."""
    def get_sparsity(self, step: int) -> float:
        raise NotImplementedError

    def should_prune(self, step: int) -> bool:
        raise NotImplementedError


class CubicScheduler(PruningScheduler):
    """
    Polynomial (Cubic) Sparsity Schedule (Zhu & Gupta, 2017).
    
    Formula:
        s_t = s_f + (s_i - s_f) * (1 - (t - t_start) / (t_end - t_start))^3
        
    Properties:
    - Rapid initial pruning when network redundancy is high.
    - Decelerating pruning rate as target sparsity is reached, allowing the network
      to continuously heal remaining connections.
    - Fixed final sparsity phase between t_end and total steps for stable convergence.
    """
    def __init__(
        self,
        initial_sparsity: float = 0.0,
        target_sparsity: float = 0.90,
        start_step: int = 100,
        end_step: int = 1000,
        update_freq: int = 20
    ):
        self.initial_sparsity = initial_sparsity
        self.target_sparsity = target_sparsity
        self.start_step = start_step
        self.end_step = end_step
        self.update_freq = update_freq

    def should_prune(self, step: int) -> bool:
        if step < self.start_step or step > self.end_step:
            return False
        return (step - self.start_step) % self.update_freq == 0

    def get_sparsity(self, step: int) -> float:
        if step < self.start_step:
            return self.initial_sparsity
        if step >= self.end_step:
            return self.target_sparsity

        progress = (step - self.start_step) / float(self.end_step - self.start_step)
        ramp = (1.0 - progress) ** 3
        return float(self.target_sparsity + (self.initial_sparsity - self.target_sparsity) * ramp)


class OneShotScheduler(PruningScheduler):
    """
    One-Shot Pruning Schedule.
    Prunes instantly to target sparsity at a single step.
    Serves as an empirical baseline to demonstrate why gradual schedules are superior.
    """
    def __init__(self, target_sparsity: float = 0.90, prune_step: int = 500):
        self.target_sparsity = target_sparsity
        self.prune_step = prune_step
        self._pruned = False

    def should_prune(self, step: int) -> bool:
        if step >= self.prune_step and not self._pruned:
            self._pruned = True
            return True
        return False

    def get_sparsity(self, step: int) -> float:
        return self.target_sparsity if step >= self.prune_step else 0.0


class LinearScheduler(PruningScheduler):
    """Linear sparsity ramp schedule."""
    def __init__(
        self,
        initial_sparsity: float = 0.0,
        target_sparsity: float = 0.90,
        start_step: int = 100,
        end_step: int = 1000,
        update_freq: int = 20
    ):
        self.initial_sparsity = initial_sparsity
        self.target_sparsity = target_sparsity
        self.start_step = start_step
        self.end_step = end_step
        self.update_freq = update_freq

    def should_prune(self, step: int) -> bool:
        if step < self.start_step or step > self.end_step:
            return False
        return (step - self.start_step) % self.update_freq == 0

    def get_sparsity(self, step: int) -> float:
        if step < self.start_step:
            return self.initial_sparsity
        if step >= self.end_step:
            return self.target_sparsity
        progress = (step - self.start_step) / float(self.end_step - self.start_step)
        return float(self.initial_sparsity + progress * (self.target_sparsity - self.initial_sparsity))
