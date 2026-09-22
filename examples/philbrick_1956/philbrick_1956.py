import argparse

import jax
import jax.numpy as jnp

import spicex

jax.config.update("jax_enable_x64", True)

"""
Philbrick 1956 RC Network

AC frequency response of the three-resistor, three-capacitor RC network
patented by Philbrick in 1956. A 1 V source drives RSRC into a bus shared
by three parallel R-C branches feeding a resistive ladder (R2-R3-R4) that
is probed at V and terminated by RLOAD.

The bus reaches the ladder only through capacitors, so the network has no
DC path: gain -> 0 as f -> 0. At high frequency the capacitors short the
ladder onto the bus and, since RLOAD >> RSRC, gain -> 1. In between, the
three-section CR ladder produces a resonance-like peak above 1.

Philip Mocz (2026)

Usage:
  python philbrick_1956.py [--plot]
"""

RSRC = 1e3  # source resistance (Ohm)
C1 = 10e-9  # F
C2 = 20e-9  # F
C3 = 50e-9  # F
R2 = 3.3e6  # Ohm
R3 = 1.0e6  # Ohm
R4 = 510e3  # Ohm
RLOAD = 100e6  # Ohm
V_S = 1.0  # source amplitude (V)

F_MIN = 1e-3  # Hz
F_MAX = 1e3  # Hz
N_FREQ = 601


def build_circuit():
    # Nodes: 0=GND, 1=VSRC+, 2=RSRC/C1/C2/C3 bus,
    #        3=probe V (C1-R2-RLOAD), 4=R2-C2-R3 junction, 5=R3-C3-R4 junction
    circuit = spicex.Circuit(n_nodes=6)
    circuit.add_voltage_source(0, 1, V_S)
    circuit.add_resistor(1, 2, RSRC)
    circuit.add_capacitor(2, 3, C1)
    circuit.add_capacitor(2, 4, C2)
    circuit.add_capacitor(2, 5, C3)
    circuit.add_resistor(3, 4, R2)
    circuit.add_resistor(4, 5, R3)
    circuit.add_resistor(5, 0, R4)
    circuit.add_resistor(3, 0, RLOAD)
    return circuit


def main():
    freq = jnp.logspace(jnp.log10(F_MIN), jnp.log10(F_MAX), N_FREQ)
    omega = 2.0 * jnp.pi * freq

    def probe_at(w):
        v_nodes, _ = build_circuit().solve_ac(w)
        return v_nodes[3]

    v_probe = spicex.sweep(probe_at, omega)
    gain = jnp.abs(v_probe)

    peak_idx = int(jnp.argmax(gain))
    print(
        f"Peak gain: {float(gain[peak_idx]):.3f} V at f = {float(freq[peak_idx]):.3f} Hz"
    )

    above_unity = freq[gain > 1.0]
    if above_unity.size:
        print(
            f"Gain > 1.0 for f in [{float(above_unity[0]):.2f}, "
            f"{float(above_unity[-1]):.2f}] Hz"
        )

    return freq, v_probe


def plot(freq, v_probe):
    import matplotlib.pyplot as plt

    gain = jnp.abs(v_probe)

    _fig, ax = plt.subplots(figsize=(6, 4))
    ax.axhline(1.0, color="gray", linewidth=0.8, linestyle=":")
    ax.plot(freq, gain, color="red", label="spicex")
    ax.set_xscale("log")
    ax.set_xlim(float(freq[0]), float(freq[-1]))
    ax.set_ylim(0.0, 1.5)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("V(probe) [V]")
    ax.set_title("Philbrick 1956 RC Network Frequency Response")
    ax.legend()
    plt.tight_layout()
    plt.savefig("philbrick_1956.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plot", action="store_true", help="Plot |V(probe)| vs frequency"
    )
    args = parser.parse_args()

    freq, v_probe = main()

    if args.plot:
        plot(freq, v_probe)
