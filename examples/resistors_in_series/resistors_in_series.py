import jax
import spicex

# switch on for double precision
jax.config.update("jax_enable_x64", True)

"""
Two resistors in parallel with a voltage source

Philip Mocz (2026)

Usage:
  python resistors_in_series.py
"""


def main():
    circuit = spicex.Circuit(n_nodes=3)
    circuit.add_voltage_source(0, 1, 5.0)  # 5 V source: ground --> node 1
    circuit.add_resistor(1, 2, 1e3)  # 1 kΩ: node 1 --> node 2
    circuit.add_resistor(2, 0, 2e3)  # 2 kΩ: node 2 --> ground
    v_nodes, i_vsrc = circuit.solve()

    print("Node voltages:", v_nodes)
    print("Current through voltage source:", i_vsrc)

    return v_nodes, i_vsrc


if __name__ == "__main__":
    main()
