import jax
import jax.numpy as jnp
import pytest

from spicex import Circuit, first_crossing

jax.config.update("jax_enable_x64", True)


def test_resistor_node_out_of_range():
    c = Circuit(2)
    c.add_resistor(0, 5, 1000.0)
    with pytest.raises(ValueError, match="out of range"):
        c.solve()


def test_voltage_source_node_out_of_range():
    c = Circuit(2)
    c.add_voltage_source(0, 9, 5.0)
    with pytest.raises(ValueError, match="out of range"):
        c.solve()


def test_current_source_node_out_of_range():
    c = Circuit(2)
    c.add_current_source(0, 9, 1e-3)
    with pytest.raises(ValueError, match="out of range"):
        c.solve()


def test_resistor_self_loop():
    c = Circuit(2)
    c.add_resistor(1, 1, 1000.0)
    with pytest.raises(ValueError, match="self-loop"):
        c.solve()


def test_voltage_source_self_loop():
    c = Circuit(2)
    c.add_voltage_source(1, 1, 5.0)
    with pytest.raises(ValueError, match="self-loop"):
        c.solve()


def test_current_source_self_loop():
    c = Circuit(2)
    c.add_current_source(1, 1, 1e-3)
    with pytest.raises(ValueError, match="self-loop"):
        c.solve()


def test_floating_node_no_elements():
    # Node 1 has no elements connecting it
    c = Circuit(2)
    with pytest.raises(ValueError, match="Floating"):
        c.solve()


def test_floating_node_isolated_from_ground():
    # Nodes 1 and 2 are connected to each other but not to ground
    c = Circuit(3)
    c.add_resistor(1, 2, 1000.0)
    with pytest.raises(ValueError, match="Floating"):
        c.solve()


def test_floating_node_partial_connection():
    # Node 1 connects to ground, node 2 is isolated
    c = Circuit(3)
    c.add_voltage_source(0, 1, 5.0)
    c.add_resistor(0, 1, 1000.0)
    with pytest.raises(ValueError, match="Floating"):
        c.solve()


def test_inductor_node_out_of_range():
    c = Circuit(2)
    c.add_inductor(0, 9, 1e-3)
    with pytest.raises(ValueError, match="out of range"):
        c.solve()


def test_inductor_self_loop():
    c = Circuit(2)
    c.add_inductor(1, 1, 1e-3)
    with pytest.raises(ValueError, match="self-loop"):
        c.solve()


def test_inductor_floating_node():
    # Inductor connects nodes 1 and 2, but neither connects to ground
    c = Circuit(3)
    c.add_inductor(1, 2, 1e-3)
    with pytest.raises(ValueError, match="Floating"):
        c.solve()


def test_capacitor_node_out_of_range():
    c = Circuit(2)
    c.add_capacitor(0, 9, 10e-6)
    with pytest.raises(ValueError, match="out of range"):
        c.solve()


def test_capacitor_self_loop():
    c = Circuit(2)
    c.add_capacitor(1, 1, 10e-6)
    with pytest.raises(ValueError, match="self-loop"):
        c.solve()


def test_capacitor_floating_node():
    # A capacitor is an open circuit in DC, so a node with only a capacitor
    # path to the rest of the circuit has no DC conduction path and is floating.
    c = Circuit(2)
    c.add_capacitor(0, 1, 10e-6)
    with pytest.raises(ValueError, match="Floating"):
        c.solve()


def test_capacitor_only_path_valid_in_transient():
    # vsrc(0,1), cap(1,2), cap(2,3), res(3,0): node 2 is floating in DC but valid in transient
    c = Circuit(4)
    c.add_voltage_source(0, 1, 1.0)
    c.add_capacitor(1, 2, 1e-6)
    c.add_capacitor(2, 3, 2e-6)
    c.add_resistor(3, 0, 1e3)
    with pytest.raises(ValueError, match="Floating"):
        c.solve()
    _t, v_nodes, *_ = c.solve_transient(t_end=1e-3, dt=1e-5)
    assert v_nodes.shape == (100, 4)


def test_valid_circuit_passes_validation():
    # A well-formed circuit should not raise
    c = Circuit(2)
    c.add_voltage_source(0, 1, 5.0)
    c.add_resistor(0, 1, 1000.0)
    v_nodes, *_ = c.solve()
    assert v_nodes[1] == pytest.approx(5.0)


def test_capacitor_only_path_valid_in_ac():
    # vsrc(0,1), cap(1,2), cap(2,3), res(3,0): node 2 is floating in DC but
    # valid in AC, since capacitors carry a finite admittance at omega != 0.
    c = Circuit(4)
    c.add_voltage_source(0, 1, 1.0)
    c.add_capacitor(1, 2, 1e-6)
    c.add_capacitor(2, 3, 2e-6)
    c.add_resistor(3, 0, 1e3)
    with pytest.raises(ValueError, match="Floating"):
        c.solve()
    v_nodes, i_vsrc = c.solve_ac(omega=1e3)
    assert v_nodes.shape == (4,)
    assert i_vsrc.shape == (1,)


def test_solve_ac_matches_analytic_rc_low_pass():
    # V-R-C low-pass: |H(jw)| = 1/sqrt(1+(wRC)^2), phase = -atan(wRC)
    R = 1e3
    C = 1e-6
    freq = jnp.array([10.0, 100.0, 1000.0])
    omega = 2.0 * jnp.pi * freq

    def probe(w):
        c = Circuit(3)
        c.add_voltage_source(0, 1, 1.0)
        c.add_resistor(1, 2, R)
        c.add_capacitor(2, 0, C)
        v_nodes, _ = c.solve_ac(w)
        return v_nodes[2]

    v_probe = jnp.array([probe(float(w)) for w in omega])
    mag = jnp.abs(v_probe)
    phase = jnp.angle(v_probe)

    analytic_mag = 1.0 / jnp.sqrt(1.0 + (omega * R * C) ** 2)
    analytic_phase = -jnp.arctan(omega * R * C)

    assert jnp.allclose(mag, analytic_mag, rtol=1e-6)
    assert jnp.allclose(phase, analytic_phase, rtol=1e-6)


def _rlc():
    circuit = Circuit(n_nodes=4)
    circuit.add_voltage_source(0, 1, 1.0)
    circuit.add_inductor(1, 2, 10e-3)
    circuit.add_resistor(2, 3, 2.0)
    circuit.add_capacitor(3, 0, 100e-6)
    return circuit


def test_transient_step_matches_solve_transient():
    dt, n_steps = 1.0e-5, 200

    _, v_nodes, _, _, _ = _rlc().solve_transient(t_end=n_steps * dt, dt=dt)

    circuit = _rlc()
    v = jnp.zeros(circuit.n_nodes)
    i_L = jnp.zeros(1)
    stepped = []
    for _ in range(n_steps):
        v, _, i_L, _ = circuit.transient_step(v, i_L, dt)
        stepped.append(v)

    assert jnp.allclose(jnp.stack(stepped), v_nodes, rtol=1e-12, atol=1e-12)


def test_transient_step_source_override_drives_the_circuit():
    dt, n_steps = 1.0e-5, 200

    _, v_nodes, _, _, _ = _rlc().solve_transient(t_end=n_steps * dt, dt=dt)

    def stepped(source_of):
        circuit = _rlc()
        v = jnp.zeros(circuit.n_nodes)
        i_L = jnp.zeros(1)
        out = []
        for k in range(n_steps):
            v, _, i_L, _ = circuit.transient_step(
                v, i_L, dt, vsrc_values=jnp.array([source_of(k)])
            )
            out.append(v)
        return jnp.stack(out)

    assert jnp.allclose(stepped(lambda k: 1.0), v_nodes, rtol=1e-12, atol=1e-12)
    assert not jnp.allclose(stepped(lambda k: 1.0 + k * dt), v_nodes, atol=1e-6)


def test_variable_inductor_node_out_of_range():
    c = Circuit(2)
    c.add_variable_inductor(0, 9, 1e-3)
    with pytest.raises(ValueError, match="out of range"):
        c.solve()


def test_variable_inductor_self_loop():
    c = Circuit(2)
    c.add_variable_inductor(1, 1, 1e-3)
    with pytest.raises(ValueError, match="self-loop"):
        c.solve()


def test_variable_inductor_floating_node():
    # Variable inductor connects nodes 1 and 2, but neither connects to ground
    c = Circuit(3)
    c.add_variable_inductor(1, 2, 1e-3)
    with pytest.raises(ValueError, match="Floating"):
        c.solve()


def test_variable_inductor_dc_is_short_and_reported_after_fixed():
    # 5 V -> L (fixed) -> L (variable) -> 1 kΩ: both inductors carry 5 mA
    c = Circuit(4)
    c.add_voltage_source(0, 1, 5.0)
    c.add_inductor(1, 2, 1e-3)
    c.add_variable_inductor(2, 3, 2e-3)
    c.add_resistor(3, 0, 1e3)
    v_nodes, _i_vsrc, i_inductor, _ = c.solve()
    assert jnp.allclose(v_nodes, jnp.array([0.0, 5.0, 5.0, 5.0]))
    assert jnp.allclose(i_inductor, jnp.array([5e-3, 5e-3]))


def test_variable_inductor_ac_uses_stored_value():
    def v_out(add):
        c = Circuit(3)
        c.add_voltage_source(0, 1, 1.0)
        add(c, 1, 2, 10e-3)
        c.add_resistor(2, 0, 10.0)
        return c.solve_ac(1e3)[0][2]

    fixed = v_out(Circuit.add_inductor)
    variable = v_out(Circuit.add_variable_inductor)
    assert jnp.allclose(fixed, variable, rtol=1e-12)


def test_constant_variable_inductor_matches_fixed_inductor():
    dt, n_steps = 1.0e-5, 200
    _, v_fixed, _, i_fixed, _ = _rlc().solve_transient(t_end=n_steps * dt, dt=dt)

    c = Circuit(n_nodes=4)
    c.add_voltage_source(0, 1, 1.0)
    c.add_variable_inductor(1, 2, 10e-3)
    c.add_resistor(2, 3, 2.0)
    c.add_capacitor(3, 0, 100e-6)
    _, v_var, _, i_var, _ = c.solve_transient(t_end=n_steps * dt, dt=dt)

    assert jnp.allclose(v_var, v_fixed, rtol=1e-12, atol=1e-12)
    assert jnp.allclose(i_var, i_fixed, rtol=1e-12, atol=1e-12)


def _ramp_update(L0, rate):
    # External "state" is just the step count; L(t) = L0 * (1 + rate * t)
    def update_fn(state, t, dt, v_prev, i_L_prev):
        return state + 1, jnp.array([L0 * (1.0 + rate * t)])

    return update_fn


def test_variable_inductor_conserves_flux_in_superconducting_loop():
    # Fixed L_f (1 -> 0) and variable L_v (0 -> 1) form a lossless loop, so the
    # total flux (L_f + L_v) I is conserved exactly, step by step.
    L_f, L_v0, I0 = 1e-6, 1e-6, 10.0
    dt, n_steps, rate = 1e-9, 500, 4e6  # L_v grows 3x over the run
    c = Circuit(2)
    c.add_inductor(1, 0, L_f)
    c.add_variable_inductor(0, 1, L_v0)
    t, _, _, i_L, _, steps = c.solve_transient(
        n_steps * dt,
        dt,
        i_L0=jnp.array([I0, I0]),
        state0=jnp.array(0),
        update_fn=_ramp_update(L_v0, rate),
    )
    assert int(steps[-1]) == n_steps
    assert jnp.allclose(i_L[:, 0], i_L[:, 1], rtol=1e-12)
    flux = (L_f + L_v0 * (1.0 + rate * t)) * i_L[:, 0]
    assert jnp.allclose(flux, (L_f + L_v0) * I0, rtol=1e-12)
    assert float(i_L[-1, 0]) == pytest.approx(I0 * 2.0 / 4.0, rel=1e-9)


def test_variable_inductor_rl_decay_matches_analytic():
    # Current I0 decaying through R with L(t) = L0 (1 + a t):
    # d(L I)/dt = -R I  =>  L I = L0 I0 (1 + a t)^(-R / (L0 a))
    L0, R, a, I0 = 1e-3, 1.0, 2e3, 1.0
    c = Circuit(2)
    c.add_variable_inductor(1, 0, L0)
    c.add_resistor(1, 0, R)

    def run(dt):
        t, _, _, i_L, _, _ = c.solve_transient(
            1e-3, dt, i_L0=jnp.array([I0]), state0=0, update_fn=_ramp_update(L0, a)
        )
        L = L0 * (1.0 + a * t)
        flux = L0 * I0 * (1.0 + a * t) ** (-R / (L0 * a))
        return jnp.max(jnp.abs(i_L[:, 0] - flux / L))

    err_coarse, err_fine = run(1e-6), run(5e-7)
    assert err_fine < 1e-3 * I0
    # Backward Euler is first order: halving dt halves the error
    assert float(err_coarse / err_fine) == pytest.approx(2.0, rel=0.05)


def test_coupled_transient_is_differentiable():
    L0, R, I0 = 1e-3, 1.0, 1.0

    def final_current(a):
        c = Circuit(2)
        c.add_variable_inductor(1, 0, L0)
        c.add_resistor(1, 0, R)
        _, _, _, i_L, _, _ = c.solve_transient(
            1e-3, 1e-6, i_L0=jnp.array([I0]), state0=0, update_fn=_ramp_update(L0, a)
        )
        return i_L[-1, 0]

    a, h = 2e3, 1.0
    grad = jax.grad(final_current)(a)
    fd = (final_current(a + h) - final_current(a - h)) / (2 * h)
    assert jnp.isfinite(grad)
    assert float(grad) == pytest.approx(float(fd), rel=1e-5)


def test_first_crossing_interpolates_and_is_differentiable():
    t = jnp.linspace(0.0, 1.0, 11)
    assert float(first_crossing(t, 1.0 - t, 0.25)) == pytest.approx(0.75)
    assert float(first_crossing(t, t**2, 0.5)) == pytest.approx(
        0.7 + (0.5 - 0.49) / (0.64 - 0.49) * 0.1
    )
    assert jnp.isnan(first_crossing(t, t, 2.0))
    # y = 1 - s t crosses 0.25 at t = 0.75 / s
    g = jax.grad(lambda s: first_crossing(t, 1.0 - s * t, 0.25))(1.0)
    assert float(g) == pytest.approx(-0.75)


def test_switch_node_out_of_range():
    c = Circuit(2)
    c.add_switch(0, 9, 1e-6, 1e-3)
    with pytest.raises(ValueError, match="out of range"):
        c.solve()


def test_switch_self_loop():
    c = Circuit(2)
    c.add_switch(1, 1, 1e-6, 1e-3)
    with pytest.raises(ValueError, match="self-loop"):
        c.solve()


def test_switch_floating_node():
    # Switch connects nodes 1 and 2, but neither connects to ground
    c = Circuit(3)
    c.add_switch(1, 2, 1e-6, 1e-3)
    with pytest.raises(ValueError, match="Floating"):
        c.solve()


def test_transmission_line_node_out_of_range():
    c = Circuit(3)
    c.add_transmission_line(1, 0, 9, 0, 50.0, 1e-9)
    with pytest.raises(ValueError, match="out of range"):
        c.solve()


def test_transmission_line_self_loop():
    c = Circuit(3)
    c.add_resistor(1, 0, 50.0)
    c.add_transmission_line(1, 0, 2, 2, 50.0, 1e-9)
    with pytest.raises(ValueError, match="port 2 has both terminals"):
        c.solve()


def test_transmission_line_floating_node():
    # Port 2 (nodes 2, 3) has no path to ground: the line does not join ports
    c = Circuit(4)
    c.add_voltage_source(0, 1, 1.0)
    c.add_transmission_line(1, 0, 2, 3, 50.0, 1e-9)
    c.add_resistor(2, 3, 50.0)
    with pytest.raises(ValueError, match="Floating"):
        c.solve_transient(1e-8, 1e-10)


def test_switch_dc_uses_state_at_t0():
    def v_out(t_close):
        c = Circuit(3)
        c.add_voltage_source(0, 1, 10.0)
        c.add_switch(1, 2, t_close, r_on=1.0, r_off=1e6)
        c.add_resistor(2, 0, 1.0)
        return float(c.solve()[0][2])

    assert v_out(-1.0) == pytest.approx(5.0)  # closed: 1 Ω + 1 Ω divider
    assert v_out(1e-6) == pytest.approx(10.0 / (1e6 + 1.0))  # open


def test_switch_discharges_capacitor_from_t_close():
    # C charged to V0 holds until the switch closes, then decays through R.
    # Backward Euler: V_k = V0 / (1 + dt/RC)^k after closing.
    C, R, V0, dt, t_close = 1e-6, 10.0, 1e3, 1e-7, 2e-6
    c = Circuit(3)
    c.add_capacitor(1, 0, C)
    c.add_switch(1, 2, t_close, r_on=1e-6, r_off=1e12)
    c.add_resistor(2, 0, R)
    t, v, *_ = c.solve_transient(1e-5, dt, v0=jnp.array([0.0, V0, 0.0]))
    before = t < t_close - dt / 2
    assert jnp.allclose(v[before, 1], V0, rtol=1e-6)
    k = jnp.round((t - t_close) / dt) + 1  # steps since closing
    expected = V0 / (1.0 + dt / ((R + 1e-6) * C)) ** k
    assert jnp.allclose(v[~before, 1], expected[~before], rtol=1e-6)


def test_switch_close_time_is_differentiable():
    def energy(t_close):
        c = Circuit(3)
        c.add_capacitor(1, 0, 1e-6)
        c.add_switch(1, 2, t_close, r_on=1.0, t_rise=2e-9)
        c.add_resistor(2, 0, 10.0)
        _, v, *_ = c.solve_transient(1e-7, 1e-10, v0=jnp.array([0.0, 1e3, 0.0]))
        return jnp.sum(v[:, 2] ** 2 / 10.0) * 1e-10

    for t_close in (30.0e-9, 30.05e-9):  # on and between samples
        grad = jax.grad(energy)(t_close)
        h = 1e-12
        fd = (energy(t_close + h) - energy(t_close - h)) / (2 * h)
        assert float(grad) == pytest.approx(float(fd), rel=1e-4)


def test_transient_step_requires_time_and_line_history():
    c = Circuit(3)
    c.add_voltage_source(0, 1, 1.0)
    c.add_switch(1, 2, 0.0, 1.0)
    c.add_resistor(2, 0, 1.0)
    with pytest.raises(ValueError, match="needs t"):
        c.transient_step(jnp.zeros(3), jnp.zeros(0), 1e-9)

    c = Circuit(3)
    c.add_voltage_source(0, 1, 1.0)
    c.add_transmission_line(1, 0, 2, 0, 50.0, 1e-9)
    c.add_resistor(2, 0, 50.0)
    with pytest.raises(ValueError, match="needs tline_hist"):
        c.transient_step(jnp.zeros(3), jnp.zeros(0), 1e-10)


def _charged_line(R_load, delay=50e-9, z0=5.0):
    # Line charged to V0 (open far end at node 2), switched into R at 10 ns
    c = Circuit(4)
    c.add_transmission_line(1, 0, 2, 0, z0, delay)
    c.add_switch(1, 3, 10e-9, r_on=1e-6)
    c.add_resistor(3, 0, R_load)
    return c


def test_charged_line_discharge_matches_bounce_diagram():
    # Load voltage steps every 2*delay: V_k = V0 (1 + G)/2 * G^k,
    # G = (R - Z0) / (R + Z0)
    z0, delay, V0, dt = 5.0, 50e-9, 1e3, 0.5e-9
    for R_load in (5.0, 15.0, 2.0):
        c = _charged_line(R_load)
        t, v, *_ = c.solve_transient(400e-9, dt, v0=jnp.array([0.0, V0, V0, 0.0]))
        gamma = (R_load - z0) / (R_load + z0)
        for k in range(3):
            i = int(jnp.argmin(jnp.abs(t - (10e-9 + (2 * k + 1) * delay))))
            expected = V0 * (1.0 + gamma) / 2.0 * gamma**k
            assert float(v[i, 3]) == pytest.approx(expected, rel=1e-6, abs=1e-6 * V0)


def test_transmission_line_fractional_delay():
    # Matched load: a V0/2 pulse lasting 2*delay, with delay = 100.5 steps
    V0, dt, delay = 1e3, 0.5e-9, 50.25e-9
    c = _charged_line(5.0, delay=delay)
    t, v, *_ = c.solve_transient(200e-9, dt, v0=jnp.array([0.0, V0, V0, 0.0]))
    on = v[:, 3] > V0 / 4
    width = float(jnp.sum(on)) * dt
    assert width == pytest.approx(2 * delay, abs=dt)
    mid = int(jnp.argmin(jnp.abs(t - 60e-9)))
    assert float(v[mid, 3]) == pytest.approx(V0 / 2, rel=1e-6)


def test_transmission_line_delay_must_cover_a_step_and_be_concrete():
    c = _charged_line(5.0, delay=0.4e-9)
    with pytest.raises(ValueError, match="at least one time step"):
        c.solve_transient(1e-8, 0.5e-9)

    def run(delay):
        return _charged_line(5.0, delay=delay).solve_transient(1e-8, 0.5e-9)[1]

    with pytest.raises(ValueError, match="concrete"):
        jax.jit(run)(5e-9)


def test_transmission_line_ac_input_impedance():
    z0, delay = 5.0, 50e-9

    def z_in(z_load, omega):
        c = Circuit(3)
        c.add_current_source(0, 1, 1.0)
        c.add_transmission_line(1, 0, 2, 0, z0, delay)
        c.add_resistor(2, 0, z_load)
        return c.solve_ac(omega)[0][1]

    # A matched line looks like Z0 at any frequency
    assert jnp.allclose(z_in(z0, 1.234e7), z0, rtol=1e-12)
    # A quarter-wave line transforms Z_L into Z0^2 / Z_L
    omega_q = jnp.pi / (2 * delay)
    assert jnp.allclose(z_in(20.0, omega_q), z0**2 / 20.0, rtol=1e-12)


def test_transmission_line_dc_is_a_through_connection():
    c = Circuit(3)
    c.add_voltage_source(0, 1, 10.0)
    c.add_transmission_line(1, 0, 2, 0, 5.0, 1e-9)
    c.add_resistor(2, 0, 4.0)
    v_nodes, i_vsrc, *_ = c.solve()
    assert jnp.allclose(v_nodes, jnp.array([0.0, 10.0, 10.0]))
    assert jnp.allclose(i_vsrc, jnp.array([2.5]))


def test_transient_step_with_line_history_matches_solve_transient():
    dt, n_steps, V0 = 0.5e-9, 300, 1e3
    c = _charged_line(15.0)
    v0 = jnp.array([0.0, V0, V0, 0.0])
    _, v_ref, *_ = c.solve_transient(n_steps * dt, dt, v0=v0)

    v, i_L = v0, jnp.zeros(0)
    hist = c.tline_history_init(v0, dt)
    stepped = []
    for k in range(1, n_steps + 1):
        v, _, i_L, _ = c.transient_step(v, i_L, dt, t=k * dt, tline_hist=hist)
        hist = c.tline_history_update(hist, v, dt)
        stepped.append(v)
    assert jnp.allclose(jnp.stack(stepped), v_ref, rtol=1e-12, atol=1e-9)


def test_transmission_line_impedance_is_differentiable():
    # Energy delivered in the first 2*delay is V0^2 R / (R + Z0)^2 * 2*delay
    V0, dt, delay, R_load = 1e3, 0.5e-9, 50e-9, 15.0

    def first_pulse_energy(z0):
        c = _charged_line(R_load, z0=z0)
        t, v, *_ = c.solve_transient(200e-9, dt, v0=jnp.array([0.0, V0, V0, 0.0]))
        # Samples from switch closing (t = 10 ns) to the end of the first pulse
        window = (t > 10e-9 - dt / 2) & (t < 10e-9 + 2 * delay - dt / 2)
        return jnp.sum(jnp.where(window, v[:, 3] ** 2 / R_load, 0.0)) * dt

    z0 = 5.0
    analytic = V0**2 * R_load / (R_load + z0) ** 2 * 2 * delay
    assert float(first_pulse_energy(z0)) == pytest.approx(analytic, rel=1e-6)
    d_analytic = -2 * V0**2 * R_load / (R_load + z0) ** 3 * 2 * delay
    assert float(jax.grad(first_pulse_energy)(z0)) == pytest.approx(
        d_analytic, rel=1e-6
    )
