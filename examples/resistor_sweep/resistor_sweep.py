import jax
import jax.numpy as jnp
import spicex

# switch on for double precision
jax.config.update("jax_enable_x64", True)

"""
Resistor Sweep

Philip Mocz (2026)

Usage:
  python resistor_sweep.py
"""

V_S = 10.0  # source voltage (V)
R_S = 1e3  # source resistance (Ohm)
N = 100  # number of sweep points


def power_in_load(log_R_L):
    """Power delivered to the load resistor R_L (parameterised in log-space)."""
    R_L = jnp.exp(log_R_L)
    circuit = spicex.Circuit(n_nodes=3)
    circuit.add_voltage_source(0, 1, V_S)
    circuit.add_resistor(1, 2, R_S)
    circuit.add_resistor(2, 0, R_L)
    v_nodes, _ = circuit.solve()
    return v_nodes[2] ** 2 / R_L


def main():
    log_R_Ls = jnp.linspace(jnp.log(100.0), jnp.log(10_000.0), N)
    R_L_values = jnp.exp(log_R_Ls)

    powers = spicex.sweep(power_in_load, log_R_Ls)

    print(f"{'R_L (Ohm)':>12}  {'Power (mW)':>12}")
    for R_L, P in zip(R_L_values, powers):
        print(f"  {float(R_L):10.1f}  {float(P) * 1e3:12.4f}")

    idx_peak = int(jnp.argmax(powers))
    print(
        f"\nPeak power detected: {float(powers[idx_peak]) * 1e3:.4f} mW at R_L = {float(R_L_values[idx_peak]):.1f} Ohm"
    )
    print(f"Analytic max = {V_S**2 / (4 * R_S) * 1e3:.4f} mW at R_L = {R_S:.1f} Ohm")

    return R_L_values, powers


if __name__ == "__main__":
    main()
