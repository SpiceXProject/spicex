import jax
import spicex

# switch on for double precision
jax.config.update("jax_enable_x64", True)

"""
Current source driving a resistor.

Philip Mocz (2026)

Usage:
  python current_source_with_resistor.py
"""


def main():
    circuit = spicex.Circuit(n_nodes=2)
    circuit.add_current_source(0, 1, 1e-3)  # 1 mA from ground to node 1
    circuit.add_resistor(1, 0, 1e3)  # 1 kΩ from node 1 to ground

    v_nodes, i_vsrc = circuit.solve()

    print(f"Node voltages: {v_nodes}")  # [0, 1.0] V
    print(f"Voltage source currents: {i_vsrc}")  # [] (no voltage sources)

    return v_nodes, i_vsrc


if __name__ == "__main__":
    main()
