from importlib.metadata import PackageNotFoundError, version

from .circuit import Circuit as Circuit
from .constants import constants as constants
from .optimize import optimize as optimize
from .sweep import sweep as sweep

"""
spicex: Differentiable SPICE circuit solver in JAX.

Philip Mocz (2026)
"""

try:
    __version__ = version("spicex")
except PackageNotFoundError:
    __version__ = "unknown"

__all__ = [
    "Circuit",
    "constants",
    "optimize",
    "sweep",
]
