"""
Modified Nodal Analysis (MNA) circuit solver in JAX.

Node 0 is always ground (reference node, V=0).

MNA system:
  [G  B] [v]   [i]
  [C  D] [j] = [e]

  G: conductance matrix (n_internal x n_internal)
  B: voltage source incidence (n_internal x n_vsrc)
  C = B^T
  D = 0
  v: internal node voltages
  j: currents through voltage sources (from - terminal to + terminal through source)
  i: injected currents at internal nodes (from current sources)
  e: voltage source values
"""

import jax.numpy as jnp


class Circuit:
    """
    JAX-based Modified Nodal Analysis (MNA) circuit solver.

    Node 0 is ground.
    """

    def __init__(self, n_nodes: int):
        self.n_nodes = n_nodes
        self._resistors: list[tuple] = []  # (node_a, node_b, resistance)
        self._vsources: list[tuple] = []  # (node_neg, node_pos, voltage)

    def add_resistor(self, node_a: int, node_b: int, resistance: float) -> None:
        """Add a resistor between node_a and node_b."""
        self._resistors.append((node_a, node_b, resistance))

    def add_voltage_source(self, node_neg: int, node_pos: int, voltage: float) -> None:
        """
        Add an independent voltage source.

        node_neg is the - terminal, node_pos is the + terminal.
        V(node_pos) - V(node_neg) = voltage.
        """
        self._vsources.append((node_neg, node_pos, voltage))

    def solve(self) -> tuple:
        """
        Solve the circuit with MNA.

        Returns:
            v_nodes: jnp array shape (n_nodes,), node voltages; v_nodes[0] = 0.
            i_vsrc:  jnp array shape (n_vsrc,), current through each voltage source (positive = current flowing from + terminal through the external circuit).
        """
        n = self.n_nodes - 1
        m = len(self._vsources)

        G = jnp.zeros((n, n))
        B = jnp.zeros((n, m))
        z = jnp.zeros(n + m)

        for node_a, node_b, r in self._resistors:
            g = 1.0 / r
            if node_a != 0:
                G = G.at[node_a - 1, node_a - 1].add(g)
            if node_b != 0:
                G = G.at[node_b - 1, node_b - 1].add(g)
            if node_a != 0 and node_b != 0:
                G = G.at[node_a - 1, node_b - 1].add(-g)
                G = G.at[node_b - 1, node_a - 1].add(-g)

        for k, (node_neg, node_pos, v) in enumerate(self._vsources):
            if node_pos != 0:
                B = B.at[node_pos - 1, k].add(1.0)
            if node_neg != 0:
                B = B.at[node_neg - 1, k].add(-1.0)
            z = z.at[n + k].set(v)

        # Assemble full MNA matrix A = [[G, B], [B^T, 0]]
        A = jnp.block(
            [
                [G, B],
                [B.T, jnp.zeros((m, m))],
            ]
        )

        x = jnp.linalg.solve(A, z)

        v_internal = x[:n]
        # Negate: raw MNA j_k is current injected into + terminal (negative when
        # source delivers power). Return SPICE convention: positive = current
        # flowing from + terminal through the external circuit.
        i_vsrc = -x[n:]

        v_nodes = jnp.concatenate([jnp.zeros(1), v_internal])
        return v_nodes, i_vsrc
