"""
Weight Initialization Strategies.
Provides mathematically derived initializations (He/Kaiming, Xavier/Glorot)
to preserve signal variance across deep layers and guarantee stable training without NaNs.
"""

import numpy as np


def kaiming_normal(shape: tuple, fan_in: int, nonlinearity: str = "relu") -> np.ndarray:
    """
    Kaiming (He) normal initialization.
    Designed for rectified activations (ReLU, GELU).
    Gain = sqrt(2) to account for ReLU zeroing half of activations on average:
    Var(W) = 2 / fan_in.
    """
    gain = np.sqrt(2.0) if nonlinearity in ("relu", "gelu") else 1.0
    std = gain / np.sqrt(fan_in)
    return np.random.randn(*shape).astype(np.float32) * std


def kaiming_uniform(shape: tuple, fan_in: int, nonlinearity: str = "relu") -> np.ndarray:
    """
    Kaiming (He) uniform initialization.
    Bound = sqrt(6 / fan_in).
    """
    gain = np.sqrt(2.0) if nonlinearity in ("relu", "gelu") else 1.0
    bound = gain * np.sqrt(3.0 / fan_in)
    return np.random.uniform(-bound, bound, size=shape).astype(np.float32)


def xavier_normal(shape: tuple, fan_in: int, fan_out: int) -> np.ndarray:
    """
    Xavier (Glorot) normal initialization.
    Designed for symmetric zero-centered activations (Tanh, Linear).
    Var(W) = 2 / (fan_in + fan_out).
    """
    std = np.sqrt(2.0 / (fan_in + fan_out))
    return np.random.randn(*shape).astype(np.float32) * std


def xavier_uniform(shape: tuple, fan_in: int, fan_out: int) -> np.ndarray:
    """
    Xavier (Glorot) uniform initialization.
    Bound = sqrt(6 / (fan_in + fan_out)).
    """
    bound = np.sqrt(6.0 / (fan_in + fan_out))
    return np.random.uniform(-bound, bound, size=shape).astype(np.float32)


def zeros(shape: tuple) -> np.ndarray:
    return np.zeros(shape, dtype=np.float32)


def ones(shape: tuple) -> np.ndarray:
    return np.ones(shape, dtype=np.float32)
