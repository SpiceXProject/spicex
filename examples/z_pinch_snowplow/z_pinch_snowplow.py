import argparse

import jax
import jax.numpy as jnp

import spicex

# switch on for double precision
jax.config.update("jax_enable_x64", True)

"""
Z-pinch driven by a Marx generator (0D snowplow load)

A Marx generator (N stages of capacitance C_s charged to V_c, each with a spark
gap of inductance L_s and resistance R_s) erects into a single capacitor
C_s/N charged to N*V_c. It drives a z-pinch load through a lumped output line
of inductance L_line.

The load is a cylindrical plasma shell of height h that starts at radius r0
inside a return-current can of radius b. Its inductance grows as it implodes:

  L_p(r) = (mu0 h / 2 pi) ln(b / r)

The shell is a snowplow: it starts with a liner mass m_liner (per unit length)
and sweeps up the gas fill of density rho0 as it moves in:

  m(r) = m_liner + pi rho0 (r0^2 - r^2)               (per unit length)
  d(m v)/dt = -mu0 I^2 / (4 pi r)                     (J x B force)

The circuit sees the load as a time-varying inductor, V = d(L_p I)/dt, which
includes the motional voltage I dL_p/dt. The shell state is advanced with the
circuit current through the `update_fn` hook of solve_transient(). The run
stops tracking the shell once it passes r_min (stagnation): after that its
inductance is held fixed.

The script compares the fully coupled run with two common shortcuts:
  - a fixed-inductance load (shell frozen at r0)
  - one-way coupling: the shell moves, but its dL/dt is not fed back

It also checks convergence in dt and computes gradients of the implosion time
with respect to the design parameters.

Philip Mocz (2026)

Usage:
  python z_pinch_snowplow.py [--plot]
"""

mu0 = spicex.constants["vacuum_permeability"]

# Driver (Marx generator + output line)
N_STAGES = 10  # number of Marx stages
C_STAGE = 1.0e-6  # capacitance per stage (F)
V_CHARGE = 50e3  # charge voltage per stage (V)
L_SWITCH = 2e-9  # spark-gap inductance per stage (H)
R_SWITCH = 5e-3  # spark-gap resistance per stage (Ω)
L_LINE = 10e-9  # output line + feed inductance (H)

# Load (snowplow)
R0 = 10e-3  # initial shell radius (m)
B_RETURN = 15e-3  # return-current radius (m)
HEIGHT = 20e-3  # pinch height (m)
M_LINER = 1e-6  # initial liner mass per unit length (kg/m) = 10 µg/cm
RHO0 = 0.01  # gas fill density (kg/m^3), sweeps up 31 µg/cm by r = 0
R_MIN = 1e-3  # stagnation radius (m), convergence ratio 10

# Time stepping
T_END = 200e-9  # simulation end time (s)
DT = 20e-12  # time step (s)

# Node and inductor indices
LOAD_NODE = 4
I_LOAD = 2  # load current column: fixed inductors (L_marx, L_line) come first


def load_inductance(r, h=HEIGHT):
    return mu0 * h / (2.0 * jnp.pi) * jnp.log(B_RETURN / r)


def swept_mass(r, m_liner, rho0):
    return m_liner + jnp.pi * rho0 * (R0**2 - r**2)


def make_update_fn(m_liner, rho0, feedback=True):
    """Snowplow equation of motion, advanced with the previous-step current.

    State is (r, p) with p = m v the shell momentum per unit length.
    Semi-implicit Euler: update p with the force at r, then move r with the new
    velocity. The shell is frozen once it passes R_MIN. The step that crosses
    R_MIN is not clamped, so spicex.first_crossing() can interpolate the
    stagnation time smoothly (and differentiably).
    """

    def update_fn(state, t, dt, v_prev, i_L_prev):
        r, p = state
        current = i_L_prev[I_LOAD]
        moving = r > R_MIN
        force = -mu0 * current**2 / (4.0 * jnp.pi * r)
        p_new = jnp.where(moving, p + force * dt, p)
        r_new = jnp.where(moving, r + p_new / swept_mass(r, m_liner, rho0) * dt, r)
        r_new = jnp.maximum(r_new, 0.5 * R_MIN)  # keep ln(b/r) finite
        L_new = load_inductance(r_new) if feedback else load_inductance(R0)
        return (r_new, p_new), jnp.array([L_new])

    return update_fn


def build_circuit(n_stages, c_stage, l_switch, r_switch, l_line):
    """Erected Marx (C_s/N, N L_s, N R_s) -> output line -> z-pinch load."""
    # Nodes: 0=GND, 1=Marx capacitor, 2=after gap resistance,
    #        3=after Marx inductance, 4=load (top of variable inductor)
    circuit = spicex.Circuit(n_nodes=5)
    circuit.add_capacitor(1, 0, c_stage / n_stages)
    circuit.add_resistor(1, 2, n_stages * r_switch)
    circuit.add_inductor(2, 3, n_stages * l_switch)
    circuit.add_inductor(3, 4, l_line)
    circuit.add_variable_inductor(LOAD_NODE, 0, load_inductance(R0))
    return circuit


def simulate(
    v_charge=V_CHARGE,
    m_liner=M_LINER,
    rho0=RHO0,
    n_stages=N_STAGES,
    c_stage=C_STAGE,
    l_switch=L_SWITCH,
    r_switch=R_SWITCH,
    l_line=L_LINE,
    dt=DT,
    t_end=T_END,
    feedback=True,
):
    """Run the coupled driver + snowplow simulation (t = 0 sample prepended)."""
    circuit = build_circuit(n_stages, c_stage, l_switch, r_switch, l_line)
    v0 = jnp.zeros(circuit.n_nodes).at[1].set(n_stages * v_charge)
    state0 = (jnp.asarray(R0), jnp.asarray(0.0))
    t, v_nodes, _, i_ind, _, (r, p) = circuit.solve_transient(
        t_end,
        dt,
        v0=v0,
        state0=state0,
        update_fn=make_update_fn(m_liner, rho0, feedback),
    )
    zero = jnp.zeros(1)
    return {
        "t": jnp.concatenate([zero, t]),
        "I": jnp.concatenate([zero, i_ind[:, I_LOAD]]),
        "V_cap": jnp.concatenate([v0[1:2], v_nodes[:, 1]]),
        "r": jnp.concatenate([jnp.array([R0]), r]),
        "p": jnp.concatenate([zero, p]),
    }


def pinch_time(run):
    return spicex.first_crossing(run["t"], run["r"], R_MIN)


def diagnostics(run, m_liner=M_LINER, rho0=RHO0):
    """Implosion metrics and energy accounting up to stagnation."""
    t, current, r = run["t"], run["I"], run["r"]
    t_pinch = pinch_time(run)
    k = int(jnp.argmax(r <= R_MIN))  # first sample at or past stagnation
    sl = slice(0, k + 1)

    L_p = load_inductance(r)
    L_drive = N_STAGES * L_SWITCH + L_LINE
    C = C_STAGE / N_STAGES
    R = N_STAGES * R_SWITCH
    dt = t[1] - t[0]

    # Energy balance to stagnation: capacitor energy released must equal
    # magnetic energy + resistive loss + work done on the shell
    e_cap = 0.5 * C * (run["V_cap"][0] ** 2 - run["V_cap"][k] ** 2)
    e_mag = 0.5 * (L_drive + L_p[k]) * current[k] ** 2
    e_res = jnp.sum(R * current[1 : k + 1] ** 2) * dt
    # Work on the load, (1/2) I^2 dL (midpoint current squared)
    i2_mid = 0.5 * (current[1 : k + 1] ** 2 + current[:k] ** 2)
    e_load = jnp.sum(0.5 * i2_mid * jnp.diff(L_p[sl]))
    # Shell kinetic energy (the snowplow converts part of the work to heat)
    m = swept_mass(r[k], m_liner, rho0)
    e_kin = HEIGHT * run["p"][k] ** 2 / (2.0 * m)

    return {
        "t_pinch": float(t_pinch),
        "I_peak": float(jnp.max(current[sl])),
        "t_peak": float(t[jnp.argmax(current[sl])]),
        "I_pinch": float(current[k]),
        "v_implosion": float(run["p"][k] / m),
        "E_stored": float(0.5 * C * run["V_cap"][0] ** 2),
        "E_cap": float(e_cap),
        "E_mag": float(e_mag),
        "E_res": float(e_res),
        "E_load": float(e_load),
        "E_kin": float(e_kin),
        "energy_error": float((e_cap - e_mag - e_res - e_load) / e_cap),
    }


def main():
    # 1. Fully coupled run
    run = simulate()
    d = diagnostics(run)
    print("Coupled Marx + snowplow")
    print(f"  stored energy        {d['E_stored'] / 1e3:8.2f} kJ")
    print(f"  implosion time       {d['t_pinch'] * 1e9:8.2f} ns")
    print(
        f"  peak current         {d['I_peak'] / 1e3:8.1f} kA at {d['t_peak'] * 1e9:.1f} ns"
    )
    print(f"  current at pinch     {d['I_pinch'] / 1e3:8.1f} kA")
    print(f"  implosion velocity   {-d['v_implosion'] / 1e3:8.1f} km/s")
    print(f"  work on load         {d['E_load']:8.1f} J")
    print(f"  shell kinetic energy {d['E_kin']:8.1f} J")
    print(f"  energy balance error {d['energy_error']:8.1e}")

    # 2. Shortcuts that miss the timing
    fixed = simulate(feedback=False)
    i_fixed = fixed["I"]
    t_peak_fixed = float(fixed["t"][jnp.argmax(i_fixed)])
    t_pinch_oneway = float(pinch_time(fixed))  # shell moved, no dL/dt feedback
    print()
    print("Shortcuts")
    print(
        f"  fixed-L load: current peaks at {t_peak_fixed * 1e9:.1f} ns "
        f"({float(jnp.max(i_fixed)) / 1e3:.1f} kA)"
    )
    print(
        f"  one-way coupling (no dL/dt feedback): implosion at "
        f"{t_pinch_oneway * 1e9:.2f} ns "
        f"({(t_pinch_oneway - d['t_pinch']) * 1e9:+.2f} ns error)"
    )

    # 3. Convergence in dt
    d_half = diagnostics(simulate(dt=DT / 2))
    dt_pinch = abs(d_half["t_pinch"] - d["t_pinch"])
    dI_peak = abs(d_half["I_peak"] - d["I_peak"]) / d["I_peak"]
    print()
    print(
        f"Convergence (dt -> dt/2): implosion time changes {dt_pinch * 1e12:.1f} ps, "
        f"peak current {100 * dI_peak:.3f}%"
    )

    # 4. Gradients of the implosion time w.r.t. design parameters
    def t_pinch_of(v_charge, m_liner, l_line):
        return pinch_time(simulate(v_charge=v_charge, m_liner=m_liner, l_line=l_line))

    grads = jax.grad(t_pinch_of, argnums=(0, 1, 2))(V_CHARGE, M_LINER, L_LINE)
    grads = [float(g) for g in grads]
    print()
    print("Implosion-time sensitivities")
    print(f"  d t_pinch / d V_charge  {grads[0] * 1e9 * 1e3:+8.3f} ns/kV")
    print(f"  d t_pinch / d m_liner   {grads[1] * 1e9 * 1e-7:+8.3f} ns/(µg/cm)")
    print(f"  d t_pinch / d L_line    {grads[2] * 1e9 * 1e-9:+8.3f} ns/nH")

    return {
        "run": run,
        "metrics": d,
        "fixed": fixed,
        "t_peak_fixed": t_peak_fixed,
        "t_pinch_oneway": t_pinch_oneway,
        "convergence": {"dt_pinch": dt_pinch, "dI_peak": dI_peak},
        "grads": {"v_charge": grads[0], "m_liner": grads[1], "l_line": grads[2]},
    }


def plot(results):
    import matplotlib.pyplot as plt

    run, fixed, d = results["run"], results["fixed"], results["metrics"]
    t_ns = run["t"] * 1e9
    t_pinch_ns = d["t_pinch"] * 1e9

    _fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 6), sharex=True)
    ax1.plot(fixed["t"] * 1e9, fixed["I"] / 1e3, "--", color="gray", label="fixed-L")
    ax1.plot(t_ns, run["I"] / 1e3, color="red", label="coupled snowplow")
    ax1.axvline(t_pinch_ns, color="black", linewidth=0.8, linestyle=":")
    ax1.set_ylabel("Load current [kA]")
    ax1.legend(loc="lower left")
    ax1.set_title("Marx-driven z-pinch (0D snowplow)")

    ax2.plot(t_ns, run["r"] * 1e3, color="blue", label="shell radius")
    ax2.axvline(t_pinch_ns, color="black", linewidth=0.8, linestyle=":")
    ax2.set_ylabel("Radius [mm]", color="blue")
    ax2.set_xlabel("Time [ns]")
    ax2.set_ylim(0, 1.1 * R0 * 1e3)
    ax3 = ax2.twinx()
    ax3.plot(t_ns, load_inductance(run["r"]) * 1e9, color="green")
    ax3.set_ylabel("Load inductance [nH]", color="green")
    ax2.set_xlim(0, T_END * 1e9)

    plt.tight_layout()
    plt.savefig("z_pinch_snowplow.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true", help="Plot I, r and L_p")
    args = parser.parse_args()

    results = main()

    if args.plot:
        plot(results)
