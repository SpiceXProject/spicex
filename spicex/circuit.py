"""
Modified Nodal Analysis (MNA) circuit solver in JAX.

Node 0 is always ground (reference node, V=0).

MNA system:
  [G  B] [v]   [i]
  [C  D] [j] = [e]

  G: conductance matrix (n_internal x n_internal)
  B: branch incidence (n_internal x (n_vsrc + n_inductor))
  C = B^T
  D = 0
  v: internal node voltages
  j: branch currents (voltage sources then inductors; from - to + through branch)
  i: injected currents at internal nodes (from current sources)
  e: branch voltage values (voltage source values; 0 for inductors in DC)

Inductors are modelled as ideal short circuits in DC (zero voltage drop).
Capacitors are modelled as open circuits in DC (zero current).
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
        self._csources: list[tuple] = []  # (node_neg, node_pos, current)
        self._inductors: list[tuple] = []  # (node_a, node_b, inductance)
        self._capacitors: list[tuple] = []  # (node_a, node_b, capacitance)

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

    def add_current_source(self, node_neg: int, node_pos: int, current: float) -> None:
        """
        Add an independent current source.

        node_neg is the - terminal, node_pos is the + terminal.
        Conventional current flows from node_neg to node_pos inside the source
        (i.e., current is injected into node_pos and extracted from node_neg).
        """
        self._csources.append((node_neg, node_pos, current))

    def add_capacitor(self, node_a: int, node_b: int, capacitance: float) -> None:
        """
        Add an ideal capacitor between node_a and node_b.

        In DC analysis the capacitor is an open circuit (zero current).
        The returned i_capacitor is always zero in DC.
        """
        self._capacitors.append((node_a, node_b, capacitance))

    def add_inductor(self, node_a: int, node_b: int, inductance: float) -> None:
        """
        Add an ideal inductor between node_a and node_b.

        node_a is the + terminal, node_b is the - terminal.
        In DC analysis the inductor is a short circuit (zero voltage drop).
        The returned i_inductor is positive when current flows from node_a
        through the inductor to node_b.
        """
        self._inductors.append((node_a, node_b, inductance))

    def _validate(self) -> None:
        all_elements = [
            ("Resistor", self._resistors),
            ("Voltage source", self._vsources),
            ("Current source", self._csources),
            ("Inductor", self._inductors),
            ("Capacitor", self._capacitors),
        ]

        for label, elements in all_elements:
            for node_a, node_b, _ in elements:
                for node in (node_a, node_b):
                    if not (0 <= node < self.n_nodes):
                        raise ValueError(
                            f"{label} references node {node}, which is out of range "
                            f"[0, {self.n_nodes - 1}]"
                        )
                if node_a == node_b:
                    raise ValueError(
                        f"{label} has both terminals at node {node_a} (self-loop)"
                    )

        # Capacitors are open circuits in DC and do not provide a conduction path,
        # so they are excluded from the adjacency graph for floating-node detection.
        adjacency: dict[int, set] = {i: set() for i in range(self.n_nodes)}
        dc_elements = (
            self._resistors + self._vsources + self._csources + self._inductors
        )
        for node_a, node_b, _ in dc_elements:
            adjacency[node_a].add(node_b)
            adjacency[node_b].add(node_a)

        # BFS from ground (node 0) to detect floating nodes
        visited = {0}
        queue = [0]
        while queue:
            node = queue.pop()
            for neighbor in adjacency[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        floating = sorted(set(range(self.n_nodes)) - visited)
        if floating:
            raise ValueError(f"Floating nodes detected (no path to ground): {floating}")

    def solve(self) -> tuple:
        """
        Solve the circuit with MNA.

        Returns:
            v_nodes:    jnp array shape (n_nodes,), node voltages; v_nodes[0] = 0.
            i_vsrc:     jnp array shape (n_vsrc,), current through each voltage
                        source (positive = current flowing from + terminal through
                        the external circuit).
            i_inductor:  jnp array shape (n_inductor,), current through each
                         inductor (positive = current flowing from node_a through
                         the inductor to node_b).
            i_capacitor: jnp array shape (n_capacitor,), current through each
                         capacitor; always zero in DC analysis.
        """
        self._validate()

        n = self.n_nodes - 1
        m_v = len(self._vsources)
        m_l = len(self._inductors)
        m = m_v + m_l

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

        # Inductors act as short circuits in DC (zero voltage drop).
        for k_l, (node_a, node_b, _) in enumerate(self._inductors):
            k = m_v + k_l
            if node_a != 0:
                B = B.at[node_a - 1, k].add(1.0)
            if node_b != 0:
                B = B.at[node_b - 1, k].add(-1.0)
            # z[n + k] stays 0: V(node_a) - V(node_b) = 0

        for node_neg, node_pos, i_val in self._csources:
            if node_pos != 0:
                z = z.at[node_pos - 1].add(i_val)
            if node_neg != 0:
                z = z.at[node_neg - 1].add(-i_val)

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
        i_vsrc = -x[n : n + m_v]
        # No negation: j_L is already the physical current from node_a to node_b.
        i_inductor = x[n + m_v :]

        # Capacitors are open circuits in DC: no stamp, current is always zero.
        i_capacitor = jnp.zeros(len(self._capacitors))

        v_nodes = jnp.concatenate([jnp.zeros(1), v_internal])
        return v_nodes, i_vsrc, i_inductor, i_capacitor
