import argparse
import jax
import jax.numpy as jnp
import spicex

# switch on for double precision
jax.config.update("jax_enable_x64", True)

"""
RLC Series Circuit

A 1 V voltage source drives a series R-L-C circuit.

  R = 2 Ω,  L = 10 mH,  C = 100 µF
  ω₀ = 1/√(LC) = 1000 rad/s  (f₀ ≈ 159 Hz,  T₀ ≈ 6.28 ms)
  ζ  = R / (2√(L/C)) = 0.1   (underdamped — oscillatory step response)

Analytic capacitor voltage:
  V_C(t) = V_S [1 − e^(−αt)(cos ω_d t + (α/ω_d) sin ω_d t)]
where α = ζω₀,  ω_d = ω₀√(1−ζ²).

Philip Mocz (2026)

Usage:
  python rlc_series_circuit.py [--plot]
"""

R = 2.0  # resistance (Ω)
L = 10e-3  # inductance (H)
C = 100e-6  # capacitance (F)
V_S = 1.0  # source voltage (V)
t_end = 50e-3  # 50 ms ~= 5τ  (τ = 1/(ζω₀) = 10 ms)
dt = 0.01e-3  # 0.01 ms ==? 5000 steps, ~628 steps per oscillation period


def main():
    # Nodes: 0=GND, 1=V_source+, 2=L-R junction, 3=R-C junction (cap voltage)
    circuit = spicex.Circuit(n_nodes=4)
    circuit.add_voltage_source(0, 1, V_S)  # 1 V step: GND --> node 1
    circuit.add_inductor(1, 2, L)  # 10 mH: node 1 --> node 2
    circuit.add_resistor(2, 3, R)  # 2 Ω:   node 2 --> node 3
    circuit.add_capacitor(3, 0, C)  # 100 µF: node 3 --> GND

    t, v_nodes, i_vsrc, i_inductor, i_capacitor = circuit.solve_transient(
        t_end=t_end, dt=dt
    )

    # Analytic solution (underdamped series RLC)
    alpha = R / (2.0 * L)  # = ζω₀ = 100 rad/s
    omega0 = 1.0 / jnp.sqrt(L * C)  # = 1000 rad/s
    omega_d = jnp.sqrt(omega0**2 - alpha**2)  # ≈ 994.99 rad/s
    v_analytic = V_S * (
        1.0
        - jnp.exp(-alpha * t)
        * (jnp.cos(omega_d * t) + (alpha / omega_d) * jnp.sin(omega_d * t))
    )

    peak_idx = int(jnp.argmax(v_nodes[:, 3]))
    print(
        f"ω₀ = {float(omega0):.1f} rad/s,  ζ = {R / (2 * float(jnp.sqrt(L / C))):.2f}"
    )
    print(
        f"Peak V_C: {float(v_nodes[peak_idx, 3]):.4f} V at t = {float(t[peak_idx]) * 1e3:.3f} ms"
        f"  (analytic peak ≈ {float(jnp.max(v_analytic)):.4f} V)"
    )
    print()
    print(
        f"{'t (ms)':>8}  {'V_C sim (V)':>12}  {'V_C analytic (V)':>16}  {'err (%)':>8}"
    )
    for k in range(0, len(t), 500):
        sim = float(v_nodes[k, 3])
        ana = float(v_analytic[k])
        err = 100.0 * abs(sim - ana) / (abs(ana) + 1e-12)
        print(f"  {float(t[k]) * 1e3:6.2f}  {sim:12.6f}  {ana:16.6f}  {err:8.4f}")

    return t, v_nodes, i_vsrc, i_inductor, i_capacitor


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true", help="Plot V_C vs time")
    args = parser.parse_args()

    t, v_nodes, i_vsrc, i_inductor, i_capacitor = main()

    if args.plot:
        import matplotlib.pyplot as plt

        alpha = R / (2.0 * L)
        omega0 = 1.0 / jnp.sqrt(L * C)
        omega_d = jnp.sqrt(omega0**2 - alpha**2)
        v_analytic = V_S * (
            1.0
            - jnp.exp(-alpha * t)
            * (jnp.cos(omega_d * t) + (alpha / omega_d) * jnp.sin(omega_d * t))
        )

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.axhline(V_S, color="gray", linewidth=0.8, linestyle=":")
        ax.plot(t * 1e3, v_analytic, "--", color="black", label="analytic")
        ax.plot(t * 1e3, v_nodes[:, 3], color="red", label="spicex")
        ax.set_xlim(0, t_end * 1e3)
        ax.set_ylim(0, 1.8 * V_S)
        ax.set_xlabel("Time [ms]")
        ax.set_ylabel("V_C [V]")
        ax.set_title("Series RLC Step Response")
        ax.legend()
        plt.tight_layout()
        plt.savefig("rlc_series_circuit.png", dpi=300)
        plt.show()
