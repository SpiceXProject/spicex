"""
Parameter sweep utility for spicex circuits
"""

import jax
from collections.abc import Callable


def sweep(fn: Callable, *param_arrays, jit: bool = True):
    """Sweep a circuit function over arrays of parameters (via jax.vmap).

    Each array in *param_arrays is mapped over its leading (batch) dimension
    element-wise. All arrays must share the same leading-axis size.

    Args:
        fn: fn(*scalars) -> output. Constructs and solves a Circuit; each
            scalar argument corresponds to one element from the matching entry
            in param_arrays.
        *param_arrays: One or more JAX arrays swept over their leading axis.
        jit: JIT-compile the vmapped function (default True).

    Returns:
        Stacked outputs of fn across the batch. Shape (batch, ...) for a
        scalar-returning fn, or a pytree of such arrays for tuple-returning fn.
    """
    batched = jax.vmap(fn)
    if jit:
        batched = jax.jit(batched)
    return batched(*param_arrays)
