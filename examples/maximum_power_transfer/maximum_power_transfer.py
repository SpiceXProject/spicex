import jax
import jax.numpy as jnp
import spicex

# switch on for double precision
jax.config.update("jax_enable_x64", True)

"""
Maximum Power Transfer

Philip Mocz (2026)

Usage:
  python maximum_power_transfer.py
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


def main():
    @jax.jit
    def loss_fn(log_R_L):
        return -power_in_load(log_R_L)

    log_R_L = jnp.log(100.0)

    log_R_L_opt, _ = spicex.optimize(log_R_L, loss_fn, max_iter=100, tol=1e-8)

    R_L_opt = jnp.exp(log_R_L_opt)
    P_opt = power_in_load(log_R_L_opt)
    P_analytic = V_S**2 / (4.0 * R_S)

    print(f"Optimal R_L:   {float(R_L_opt):.2f} Ohm  (analytic: {R_S:.0f} Ohm)")
    print(
        f"Max power: {float(P_opt) * 1e3:.4f} mW  (analytic: {P_analytic * 1e3:.0f} mW)"
    )

    return R_L_opt, P_opt


if __name__ == "__main__":
    main()
