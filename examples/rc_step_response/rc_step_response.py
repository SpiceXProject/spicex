import jax
import jax.numpy as jnp
import spicex

# switch on for double precision
jax.config.update("jax_enable_x64", True)

"""
RC step-response transient simulation.

A 1 V voltage source charges a 1 µF capacitor through a 1 kΩ resistor.
The time constant is τ = RC = 1 ms.
Analytic solution: V_C(t) = 1 − exp(−t / τ).

Philip Mocz (2026)

Usage:
  python rc_step_response.py
"""

R = 1e3  # resistance (Ω)
C = 1e-6  # capacitance (F)
V_S = 1.0  # source voltage (V)
t_end = 5e-3  # 5 ms = 5τ
dt = 0.01e-3  # 0.01 ms ==> 500 steps, 100 steps per τ


def main():
    circuit = spicex.Circuit(n_nodes=3)
    circuit.add_voltage_source(0, 1, V_S)  # 1 V source: GND → node 1
    circuit.add_resistor(1, 2, R)  # 1 kΩ: node 1 → node 2
    circuit.add_capacitor(2, 0, C)  # 1 µF: node 2 → GND

    t, v_nodes, i_vsrc, i_inductor, i_capacitor = circuit.solve_transient(
        t_end=t_end, dt=dt
    )

    tau = R * C
    v_analytic = V_S * (1.0 - jnp.exp(-t / tau))

    print(
        f"{'t (ms)':>8}  {'V_C sim (V)':>12}  {'V_C analytic (V)':>16}  {'err (%)':>8}"
    )
    for k in range(0, len(t), 50):
        sim = float(v_nodes[k, 2])
        ana = float(v_analytic[k])
        err = 100.0 * abs(sim - ana) / (abs(ana) + 1e-12)
        print(f"  {float(t[k]) * 1e3:6.2f}  {sim:12.6f}  {ana:16.6f}  {err:8.4f}")

    return t, v_nodes, i_vsrc, i_inductor, i_capacitor


if __name__ == "__main__":
    main()
