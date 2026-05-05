import argparse
import jax
import jax.numpy as jnp
import spicex

# switch on for double precision
jax.config.update("jax_enable_x64", True)

"""
PFN Type B

Philip Mocz (2026)

Usage:
  python pfn_type_b.py [--plot]
"""

# Component values
L1, L2, L3, L4, L5 = 6.6e-6, 5.4e-6, 5.6e-6, 6.5e-6, 9.3e-6  # inductances (H)
C1, C2, C3, C4, C5 = 250e-6, 250e-6, 300e-6, 350e-6, 800e-6  # capacitances (F)
R_esr = 5e-3  # 5 mΩ equivalent series resistance (ESR) for each capacitor branch
R_load = 100e-3  # 100 mΩ matched load

# Characteristic impedance and initial charge voltage
L_total = L1 + L2 + L3 + L4 + L5  # 33.4 µH
C_total = C1 + C2 + C3 + C4 + C5  # 1950 µF
Z0 = float(jnp.sqrt(L_total / C_total))  # ~ 0.131 Ω

# Calibrate V0 so the flat-top current ~ 800 A into R_load
I_target = 800.0
V0 = I_target * (Z0 + R_load)

# Pulse duration: T ~ 2*sqrt(L_total * C_total) ~ 0.51 ms
T_pulse = 2.0 * float(jnp.sqrt(L_total * C_total))

# Simulation parameters
t_end = 1.6e-3  # 1.6 ms
dt = 500e-9  # 500 ns


def main():
    circuit = spicex.Circuit(n_nodes=12)

    # Series inductors along top rail
    circuit.add_inductor(1, 2, L1)
    circuit.add_inductor(2, 3, L2)
    circuit.add_inductor(3, 4, L3)
    circuit.add_inductor(4, 5, L4)
    circuit.add_inductor(5, 6, L5)

    # Shunt branches: Ck + R_esr
    circuit.add_capacitor(2, 7, C1)
    circuit.add_resistor(7, 0, R_esr)
    circuit.add_capacitor(3, 8, C2)
    circuit.add_resistor(8, 0, R_esr)
    circuit.add_capacitor(4, 9, C3)
    circuit.add_resistor(9, 0, R_esr)
    circuit.add_capacitor(5, 10, C4)
    circuit.add_resistor(10, 0, R_esr)
    circuit.add_capacitor(6, 11, C5)
    circuit.add_resistor(11, 0, R_esr)

    # Load at output terminal
    circuit.add_resistor(1, 0, R_load)

    # Initial conditions:
    #   PFN nodes n2-n6 at V0 (capacitors fully charged).
    #   n1 at 0 V (output terminal was isolated before switch closed at t=0).
    #   Intermediate nodes n7-n11 at 0 V (no pre-discharge current through ESR).
    #   All inductor currents zero (no current before switch).
    v0 = jnp.array([0.0, 0.0, V0, V0, V0, V0, V0, 0.0, 0.0, 0.0, 0.0, 0.0])
    i_L0 = jnp.zeros(5)

    t, v_nodes, i_vsrc, i_inductor, i_capacitor = circuit.solve_transient(
        t_end=t_end, dt=dt, v0=v0, i_L0=i_L0
    )

    i_load = v_nodes[:, 1] / R_load  # load current (A)

    # Metrics
    i_peak = float(jnp.max(i_load))

    # Flat-top: central region 15%--85% of T_pulse
    t_flat_lo, t_flat_hi = 0.15 * T_pulse, 0.85 * T_pulse
    mask = (t >= t_flat_lo) & (t <= t_flat_hi)
    i_flat = i_load[mask]
    i_mean = float(jnp.mean(i_flat))
    flat_pct = float(100.0 * jnp.max(jnp.abs(i_flat - i_mean)) / i_mean)

    # Rise time: 10% --> 90% of peak
    idx10 = int(jnp.argmax(i_load >= 0.10 * i_peak))
    idx90 = int(jnp.argmax(i_load >= 0.90 * i_peak))
    rise_us = float((t[idx90] - t[idx10]) * 1e6)

    print(f"Z₀          = {Z0 * 1e3:.2f} mΩ  (PFN characteristic impedance)")
    print(f"V₀          = {V0:.1f} V  (initial charge voltage)")
    print(f"T_pulse     = {T_pulse * 1e3:.3f} ms  (approx. pulse duration)")
    print()
    print(f"Peak current : {i_peak:.1f} A")
    print(f"Rise time    : {rise_us:.1f} µs  (10%-->90% of peak)")
    print(
        f"Flat-top mean: {i_mean:.1f} A  (t = {t_flat_lo * 1e3:.2f}--{t_flat_hi * 1e3:.2f} ms)"
    )
    print(f"Flatness     : ±{flat_pct:.2f}%")

    return t, v_nodes, i_load


def plot(t, v_nodes, i_load):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))

    # ±2.5% flatness band around target
    ax.axhspan(
        I_target * 0.975, I_target * 1.025, alpha=0.12, color="gray", label="±2.5% band"
    )
    ax.axhline(I_target, color="gray", linewidth=0.8, linestyle="-")
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="-")

    ax.plot(t * 1e3, i_load, color="r", linewidth=1.5, label="spicex")

    ax.set_xlim(0, t_end * 1e3)
    ax.set_ylim(-400, 1000)
    ax.set_xlabel("time [ms]")
    ax.set_ylabel("load current [A]")
    ax.set_title("PFN Type B")
    ax.legend()
    plt.tight_layout()
    plt.savefig("pfn_type_b.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true", help="Plot load current vs time")
    args = parser.parse_args()

    t, v_nodes, i_load = main()

    if args.plot:
        plot(t, v_nodes, i_load)
