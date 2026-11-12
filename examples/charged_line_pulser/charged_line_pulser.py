import argparse

import jax
import jax.numpy as jnp

import spicex

# switch on for double precision
jax.config.update("jax_enable_x64", True)

"""
Charged transmission line pulser

A transmission line (impedance Z0, one-way delay tau) is charged to V0, with
its far end open. A triggered gas switch closes at t_close and discharges the
line into a resistive load R. The line then delivers a staircase of
rectangular pulses, each 2 tau long:

  V_k = V0 (1 + G)/2 G^k,   G = (R_eff - Z0) / (R_eff + Z0),   k = 0, 1, 2, ...

where R_eff = R + r_on includes the switch on-resistance (the load sees the
fraction R / R_eff of it). A matched load (R_eff = Z0) receives one clean
pulse of V0/2 that lasts 2 tau and takes all of the stored energy.

The script compares spicex with this bounce-diagram solution for matched,
under-matched and over-matched loads. It then uses gradient-based
optimisation to find the load that takes the most energy in the first
pulse, which should be the matched load.

Philip Mocz (2026)

Usage:
  python charged_line_pulser.py [--plot]
"""

Z0 = 5.0  # line impedance (Ω)
TAU = 50e-9  # one-way transit time (s)
V0 = 100e3  # charge voltage (V)
T_CLOSE = 10e-9  # switch trigger time (s)
T_RISE = 1e-9  # switch closing time scale (s)
R_ON = 5e-3  # switch on-resistance (Ω)
T_END = 400e-9  # simulation end time (s)
DT = 0.25e-9  # time step (s), tau = 200 steps

LOADS = [5.0, 15.0, 2.0]  # matched, over-matched, under-matched (Ω)


def simulate(r_load, z0=Z0):
    # Nodes: 0=GND, 1=line port 1 (switch side), 2=line port 2 (open end),
    #        3=load
    circuit = spicex.Circuit(n_nodes=4)
    circuit.add_transmission_line(1, 0, 2, 0, z0, TAU)
    circuit.add_switch(1, 3, T_CLOSE, r_on=R_ON, t_rise=T_RISE)
    circuit.add_resistor(3, 0, r_load)
    v0 = jnp.array([0.0, V0, V0, 0.0])  # line charged to V0 along its length
    t, v_nodes, *_ = circuit.solve_transient(T_END, DT, v0=v0)
    return t, v_nodes[:, 3]


def bounce_diagram(t, r_load, z0=Z0):
    """Ideal load voltage (instant switch closing)."""
    r_eff = r_load + R_ON
    gamma = (r_eff - z0) / (r_eff + z0)
    k = jnp.floor((t - T_CLOSE) / (2 * TAU))
    v_line = V0 * (1.0 + gamma) / 2.0 * gamma**k
    return jnp.where(t >= T_CLOSE, v_line * r_load / r_eff, 0.0)


def first_pulse_energy(log_r_load):
    """Energy into the load during the first pulse, [t_close, t_close + 2 tau]."""
    r_load = jnp.exp(log_r_load)
    t, v_load = simulate(r_load)
    window = (t > T_CLOSE - DT / 2) & (t < T_CLOSE + 2 * TAU - DT / 2)
    return jnp.sum(jnp.where(window, v_load**2 / r_load, 0.0)) * DT


def main():
    # 1. Staircase vs bounce diagram, sampled mid-plateau (away from the
    #    switch's rise time)
    print(f"Charged line: Z0 = {Z0} Ω, tau = {TAU * 1e9:.0f} ns, V0 = {V0 / 1e3} kV")
    print()
    print(f"{'R (Ω)':>6}  {'k':>2}  {'spicex (kV)':>12}  {'analytic (kV)':>14}")
    plateaus = {}
    for r_load in LOADS:
        t, v_load = simulate(r_load)
        v_ideal = bounce_diagram(t, r_load)
        rows = []
        for k in range(3):
            i = int(jnp.argmin(jnp.abs(t - (T_CLOSE + (2 * k + 1) * TAU))))
            rows.append((float(v_load[i]), float(v_ideal[i])))
            print(
                f"{r_load:6.1f}  {k:2d}  {rows[-1][0] / 1e3:12.4f}  "
                f"{rows[-1][1] / 1e3:14.4f}"
            )
        plateaus[r_load] = rows

    # 2. Load that takes the most energy in the first pulse
    stored = 0.5 * (TAU / Z0) * V0**2  # line capacitance tau/Z0
    log_r_opt, _ = spicex.optimize(
        jnp.log(2.0), lambda x: -first_pulse_energy(x) / stored, verbose=False
    )
    r_opt = float(jnp.exp(log_r_opt))
    e_opt = float(first_pulse_energy(log_r_opt))
    print()
    print(f"Stored energy: {stored:.1f} J")
    print(
        f"Optimal load for the first pulse: R = {r_opt:.4f} Ω "
        f"(Z0 - r_on = {Z0 - R_ON:.4f} Ω), takes {e_opt:.1f} J "
        f"({100 * e_opt / stored:.2f}% of stored)"
    )

    return {"plateaus": plateaus, "r_opt": r_opt, "e_opt": e_opt, "stored": stored}


def plot():
    import matplotlib.pyplot as plt

    _fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["red", "blue", "green"]
    for r_load, color in zip(LOADS, colors):
        t, v_load = simulate(r_load)
        ax.plot(t * 1e9, bounce_diagram(t, r_load) / 1e3, "--", color="black", lw=0.8)
        ax.plot(t * 1e9, v_load / 1e3, color=color, label=f"R = {r_load:g} Ω")
    ax.plot([], [], "--", color="black", lw=0.8, label="bounce diagram")
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_xlim(0, T_END * 1e9)
    ax.set_xlabel("Time [ns]")
    ax.set_ylabel("Load voltage [kV]")
    ax.set_title(f"Charged line pulser (Z0 = {Z0:g} Ω, τ = {TAU * 1e9:.0f} ns)")
    ax.legend()
    plt.tight_layout()
    plt.savefig("charged_line_pulser.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true", help="Plot load voltages")
    args = parser.parse_args()

    main()

    if args.plot:
        plot()
