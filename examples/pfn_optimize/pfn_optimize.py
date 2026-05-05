import argparse
import jax
import jax.numpy as jnp
import spicex

# switch on for double precision
jax.config.update("jax_enable_x64", True)

"""
PFN Inductor Optimization

Philip Mocz (2026)

Optimize the 5 inductor values of a PFN to maximize pulse flatness,
using automatic differentiation through the transient circuit simulation.

The inductor distribution is parameterized via a log-softmax so all
L_k > 0 and sum(L_k) = L_total (pulse duration is preserved).

c.f. `examples/pfn_type_b/pfn_type_b.py`

Usage:
  python pfn_optimize.py [--plot]
"""

# Fixed circuit parameters
L_total = 33.4e-6  # total inductance (H), shared equally at init
C_section = 390e-6  # equal section capacitance (F)  ~1950 µF total
C_total = 5 * C_section
R_esr = 5e-3  # 5 mΩ ESR per capacitor branch
R_load = 100e-3  # 100 mΩ matched load

Z0 = float(jnp.sqrt(L_total / C_total))  # ~0.131 Ω
I_target = 800.0
V0 = I_target * (Z0 + R_load)  # initial charge voltage
T_pulse = 2.0 * float(jnp.sqrt(L_total * C_total))  # ~0.51 ms

# Flat-top window
T_FLAT_LO = 0.15 * T_pulse
T_FLAT_HI = 0.85 * T_pulse

# Simulation parameters
t_end = 1.6e-3  # 1.6 ms
dt = 500e-9  # 500 ns


def simulate(log_L_weights):
    """Run transient simulation for given inductor log-weights.

    log_L_weights : shape (5,)
        Unnormalized log weights; Ls = L_total * softmax(log_L_weights)

    Returns
    -------
    t        : shape (n_steps,)
    i_load   : shape (n_steps,)  load current in A
    """
    Ls = L_total * jax.nn.softmax(log_L_weights)

    circuit = spicex.Circuit(n_nodes=12)

    # Series inductors along top rail
    circuit.add_inductor(1, 2, Ls[0])
    circuit.add_inductor(2, 3, Ls[1])
    circuit.add_inductor(3, 4, Ls[2])
    circuit.add_inductor(4, 5, Ls[3])
    circuit.add_inductor(5, 6, Ls[4])

    # Shunt branches: Ck + R_esr  (equal caps)
    circuit.add_capacitor(2, 7, C_section)
    circuit.add_resistor(7, 0, R_esr)
    circuit.add_capacitor(3, 8, C_section)
    circuit.add_resistor(8, 0, R_esr)
    circuit.add_capacitor(4, 9, C_section)
    circuit.add_resistor(9, 0, R_esr)
    circuit.add_capacitor(5, 10, C_section)
    circuit.add_resistor(10, 0, R_esr)
    circuit.add_capacitor(6, 11, C_section)
    circuit.add_resistor(11, 0, R_esr)

    # Load at output terminal
    circuit.add_resistor(1, 0, R_load)

    # Initial conditions: capacitor nodes at V0, all else at 0
    v0 = jnp.array([0.0, 0.0, V0, V0, V0, V0, V0, 0.0, 0.0, 0.0, 0.0, 0.0])
    i_L0 = jnp.zeros(5)

    t, v_nodes, *_ = circuit.solve_transient(t_end=t_end, dt=dt, v0=v0, i_L0=i_L0)
    return t, v_nodes[:, 1] / R_load


@jax.jit
def loss_fn(log_L_weights):
    """Normalized RMS deviation from I_target over the flat-top window."""
    t, i_load = simulate(log_L_weights)
    mask = (t >= T_FLAT_LO) & (t <= T_FLAT_HI)
    n_flat = jnp.sum(mask)
    return jnp.sum(mask * (i_load - I_target) ** 2) / (n_flat * I_target**2)


def flatness_pct(t, i_load):
    """Peak-to-peak flatness as ± % of mean in flat-top window."""
    mask = (t >= T_FLAT_LO) & (t <= T_FLAT_HI)
    i_flat = i_load[mask]
    i_mean = float(jnp.mean(i_flat))
    return float(100.0 * jnp.max(jnp.abs(i_flat - i_mean)) / i_mean)


def main():
    log_L_weights_init = jnp.zeros(5)  # equal inductors

    print("Optimizing inductor distribution for maximum flatness...")
    print()
    log_L_weights_opt, _ = spicex.optimize(
        log_L_weights_init, loss_fn, max_iter=200, tol=1e-10
    )

    t, i_before = simulate(log_L_weights_init)
    t, i_after = simulate(log_L_weights_opt)

    L_init = L_total * jax.nn.softmax(log_L_weights_init)
    L_opt = L_total * jax.nn.softmax(log_L_weights_opt)

    print()
    print(f"  {'':20s}  {'Initial':>12s}  {'Optimized':>12s}")
    print("  " + "-" * 50)
    for k in range(5):
        print(
            f"  L{k + 1}                      "
            f"  {float(L_init[k]) * 1e6:10.2f} µH"
            f"  {float(L_opt[k]) * 1e6:10.2f} µH"
        )
    print("  " + "-" * 50)
    print(
        f"  Peak current              "
        f"  {float(jnp.max(i_before)):10.1f} A "
        f"  {float(jnp.max(i_after)):10.1f} A"
    )
    print(
        f"  Flatness (±%)             "
        f"  {flatness_pct(t, i_before):10.2f} % "
        f"  {flatness_pct(t, i_after):10.2f} %"
    )

    return t, i_before, i_after


def plot(t, i_before, i_after):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    for ax, i_load, title in zip(
        axes, [i_before, i_after], ["Initial (equal L)", "Optimized L"]
    ):
        ax.axhspan(
            I_target * 0.975,
            I_target * 1.025,
            alpha=0.12,
            color="gray",
            label="±2.5% band",
        )
        ax.axhline(I_target, color="gray", linewidth=0.8, linestyle="-")
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="-")
        ax.axvspan(T_FLAT_LO * 1e3, T_FLAT_HI * 1e3, alpha=0.08, color="blue")
        ax.plot(t * 1e3, i_load, linewidth=1.5, label="spicex")
        ax.set_xlim(0, t_end * 1e3)
        ax.set_ylim(-400, 1000)
        ax.set_xlabel("time [ms]")
        ax.set_title(title)
        ax.legend()

    axes[0].set_ylabel("load current [A]")
    fig.suptitle("PFN Inductor Optimization")
    plt.tight_layout()
    plt.savefig("pfn_optimize.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true", help="Plot load current vs time")
    args = parser.parse_args()

    t, i_before, i_after = main()

    if args.plot:
        plot(t, i_before, i_after)
