import jax
import spicex

# switch on for double precision
jax.config.update("jax_enable_x64", True)

"""
Capacitor with Resistor

Philip Mocz (2026)

Usage:
  python capacitor_with_resistor.py
"""


def main():
    circuit = spicex.Circuit(n_nodes=2)
    circuit.add_voltage_source(0, 1, 5.0)  # 5 V source: ground --> node 1
    circuit.add_resistor(1, 0, 1e3)  # 1 kΩ: node 1 --> ground
    circuit.add_capacitor(1, 0, 10e-6)  # 10 µF: node 1 --> ground (parallel)

    v_nodes, i_vsrc, i_inductor, i_capacitor = circuit.solve()

    print("Node voltages:", v_nodes)
    print("Current through voltage source:", i_vsrc)
    print("Current through capacitor:", i_capacitor)

    return v_nodes, i_vsrc, i_capacitor


if __name__ == "__main__":
    main()
