"""
Optimization utilities for spicex circuits
"""

from typing import NamedTuple

import chex
import jax
import optax


class _InfoState(NamedTuple):
    iter_num: chex.Numeric


def _print_info():
    def init_fn(params):
        del params
        return _InfoState(iter_num=0)

    def update_fn(updates, state, params, *, value, grad, **extra_args):
        del params, extra_args
        jax.debug.print(
            "Iteration: {i}, Loss: {v:.2e}, |grad|: {e:.2e}",
            i=state.iter_num,
            v=value,
            e=optax.tree_utils.tree_norm(grad),
        )
        return updates, _InfoState(iter_num=state.iter_num + 1)

    return optax.GradientTransformationExtraArgs(init_fn, update_fn)


def optimize(init_params, fun, opt=None, max_iter=100, tol=1e-8, verbose=True):
    """Run an optimizer loop until convergence or max_iter.

    Args:
        init_params: Initial parameters.
        fun: Scalar function to minimize.
        opt: An optax GradientTransformation. Defaults to L-BFGS.
        max_iter: Maximum number of iterations.
        tol: Stop when gradient norm falls below this threshold.
        verbose: Print loss and gradient norm each iteration.

    Returns:
        (final_params, final_state)
    """
    if opt is None:
        opt = optax.lbfgs()
    if verbose:
        opt = optax.chain(_print_info(), opt)

    value_and_grad_fun = optax.value_and_grad_from_state(fun)

    def step(carry):
        params, state = carry
        value, grad = value_and_grad_fun(params, state=state)
        updates, state = opt.update(
            grad, state, params, value=value, grad=grad, value_fn=fun
        )
        params = optax.apply_updates(params, updates)
        return params, state

    def continuing_criterion(carry):
        _, state = carry
        iter_num = optax.tree_utils.tree_get(state, "count")
        grad = optax.tree_utils.tree_get(state, "grad")
        err = optax.tree_utils.tree_norm(grad)
        return (iter_num == 0) | ((iter_num < max_iter) & (err >= tol))

    init_carry = (init_params, opt.init(init_params))
    final_params, final_state = jax.lax.while_loop(
        continuing_criterion, step, init_carry
    )
    return final_params, final_state
