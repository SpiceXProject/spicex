from typing import NamedTuple

import chex
import jax
import jax.numpy as jnp
import optax
import spicex

# switch on for double precision
jax.config.update("jax_enable_x64", True)

"""
Maximum Power Transfer

Philip Mocz (2026)

Usage:
  python maximum_power_transfer.py --plot
"""

V_S = 10.0  # source voltage (V)
R_S = 1e3  # source resistance (Ohm)


def power_in_load(log_R_L):
    """Power delivered to the load resistor R_L."""
    R_L = jnp.exp(log_R_L)
    circuit = spicex.Circuit(n_nodes=3)
    circuit.add_voltage_source(0, 1, V_S)
    circuit.add_resistor(1, 2, R_S)
    circuit.add_resistor(2, 0, R_L)
    v_nodes, _ = circuit.solve()
    return v_nodes[2] ** 2 / R_L


class InfoState(NamedTuple):
    iter_num: chex.Numeric


def print_info():
    def init_fn(params):
        del params
        return InfoState(iter_num=0)

    def update_fn(updates, state, params, *, value, grad, **extra_args):
        del params, extra_args

        jax.debug.print(
            "Iteration: {i}, Loss: {v:.2e}, |grad|: {e:.2e}",
            i=state.iter_num,
            v=value,
            e=optax.tree_utils.tree_norm(grad),
        )
        return updates, InfoState(iter_num=state.iter_num + 1)

    return optax.GradientTransformationExtraArgs(init_fn, update_fn)


def run_opt(init_params, fun, opt, max_iter, tol):
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


def main():
    def loss_fn(log_R_L):
        return -power_in_load(log_R_L)

    log_R_L = jnp.log(jnp.array(100.0))  # start at R_L = 100 Ohm

    opt = optax.chain(print_info(), optax.lbfgs())
    log_R_L_opt, _ = run_opt(log_R_L, loss_fn, opt, max_iter=100, tol=1e-8)

    R_L_opt = jnp.exp(log_R_L_opt)
    P_opt = power_in_load(log_R_L_opt)
    P_analytical = V_S**2 / (4.0 * R_S)

    print(f"Optimal R_L:   {float(R_L_opt):.2f} Ohm  (analytical: {R_S:.2f} Ohm)")
    print(
        f"Maximum power: {float(P_opt) * 1e3:.4f} mW  (analytical: {P_analytical * 1e3:.4f} mW)"
    )

    return R_L_opt, P_opt


if __name__ == "__main__":
    main()
