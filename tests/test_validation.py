import jax.numpy as jnp
import pytest
from spicex import Circuit


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
    t, v_nodes, *_ = c.solve_transient(t_end=1e-3, dt=1e-5)
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
