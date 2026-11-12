"""
Event detection utilities for spicex transient results
"""

import jax.numpy as jnp


def first_crossing(t, y, level):
    """Time at which a sampled signal first crosses a level.

    The crossing time is found by linear interpolation between the two samples
    on either side of the first crossing. This makes it a smooth function of
    the sample values, so gradients flow through it. Taking the first sample
    past the crossing would give a step function with zero gradient.

    For a gradient to exist, `y` must keep evolving through the step where it
    crosses the level: do not clamp `y` to `level` (e.g. freeze an imploding
    radius at the step after it passes r_min, not at r_min itself).

    Args:
        t: shape (n,), sample times (increasing).
        y: shape (n,), signal samples.
        level: Level to detect.

    Returns:
        Scalar crossing time, or nan if `y` never crosses `level`.
    """
    t = jnp.asarray(t)
    y = jnp.asarray(y)
    above = y > level
    crossed = above[1:] != above[:-1]
    k = jnp.argmax(crossed)
    y0, y1 = y[k], y[k + 1]
    # Guard the division on the no-crossing branch (y0 may equal y1 there).
    dy = jnp.where(crossed[k], y1 - y0, 1.0)
    t_cross = t[k] + (level - y0) / dy * (t[k + 1] - t[k])
    return jnp.where(crossed[k], t_cross, jnp.nan)
