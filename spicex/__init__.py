from importlib.metadata import version, PackageNotFoundError

from .circuit import Circuit as Circuit
from . import constants as constants


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
]
