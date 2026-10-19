"""
Neural Network Base Modules and Parameters.
"""

from __future__ import annotations
import numpy as np
from typing import List, Dict, Generator, Any, Optional, Tuple, Union
from engine.tensor import Tensor


class Parameter(Tensor):
    """
    Trainable parameter tensor with requires_grad=True by default.
    Supports association with a binary pruning mask.
    """
    def __init__(
        self,
        data: Union[int, float, list, np.ndarray],
        requires_grad: bool = True,
        name: Optional[str] = None
    ):
        super().__init__(data, requires_grad=requires_grad, name=name)
        # Binary mask: 1 = active, 0 = pruned. Defaults to all active (ones).
        self.mask: Optional[np.ndarray] = None


class Module:
    """Base class for all neural network modules."""

    def __init__(self):
        self._parameters: Dict[str, Parameter] = {}
        self._submodules: Dict[str, Module] = {}
        self.training: bool = True

    def register_parameter(self, name: str, param: Optional[Parameter]):
        if param is not None and not isinstance(param, Parameter):
            raise TypeError(f"Expected Parameter, got {type(param)}")
        self._parameters[name] = param

    def __setattr__(self, name: str, value: Any):
        if isinstance(value, Parameter):
            if not hasattr(self, "_parameters"):
                super().__setattr__("_parameters", {})
            self._parameters[name] = value
        elif isinstance(value, Module):
            if not hasattr(self, "_submodules"):
                super().__setattr__("_submodules", {})
            self._submodules[name] = value
        super().__setattr__(name, value)

    def parameters(self) -> List[Parameter]:
        """Return a list of all parameters in this module and its submodules."""
        params: List[Parameter] = list(self._parameters.values())
        for submodule in self._submodules.values():
            params.extend(submodule.parameters())
        return params

    def named_parameters(self, prefix: str = "") -> Generator[Tuple[str, Parameter], None, None]:
        """Yield (name, parameter) tuples recursively."""
        for name, param in self._parameters.items():
            full_name = f"{prefix}.{name}" if prefix else name
            yield full_name, param
        for sub_name, submodule in self._submodules.items():
            sub_prefix = f"{prefix}.{sub_name}" if prefix else sub_name
            yield from submodule.named_parameters(prefix=sub_prefix)

    def zero_grad(self):
        """Zero the gradients of all parameters."""
        for param in self.parameters():
            param.zero_grad()

    def train(self, mode: bool = True):
        """Set module and submodules into training or evaluation mode."""
        self.training = mode
        for submodule in self._submodules.values():
            submodule.train(mode)

    def eval(self):
        """Set module into evaluation mode."""
        self.train(False)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
