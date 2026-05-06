import jax
import spicex

# switch on for double precision
jax.config.update("jax_enable_x64", True)

"""
Inductor with Resistor

Philip Mocz (2026)

Usage:
  python inductor_with_resistor.py
"""


def main():
    circuit = spicex.Circuit(n_nodes=3)
    circuit.add_voltage_source(0, 1, 5.0)  # 5 V source: ground --> node 1
    circuit.add_inductor(1, 2, 10e-3)  # 10 mH: node 1 --> node 2
    circuit.add_resistor(2, 0, 1e3)  # 1 kΩ: node 2 --> ground

    v_nodes, i_vsrc, i_inductor, _ = circuit.solve()

    print("Node voltages:", v_nodes)
    print("Current through voltage source:", i_vsrc)
    print("Current through inductor:", i_inductor)

    return v_nodes, i_vsrc, i_inductor


if __name__ == "__main__":
    main()
