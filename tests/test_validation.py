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


def test_valid_circuit_passes_validation():
    # A well-formed circuit should not raise
    c = Circuit(2)
    c.add_voltage_source(0, 1, 5.0)
    c.add_resistor(0, 1, 1000.0)
    v_nodes, *_ = c.solve()
    assert v_nodes[1] == pytest.approx(5.0)
