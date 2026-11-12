from importlib.metadata import PackageNotFoundError, version

from .circuit import Circuit as Circuit
from .constants import constants as constants
from .events import first_crossing as first_crossing
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
    "first_crossing",
    "optimize",
    "sweep",
]
