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

import jax
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

    def _validate(self, include_capacitors_in_graph: bool = False) -> None:
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
        # In transient, capacitors have a companion conductance C/dt and must be
        # included so nodes reachable only via capacitors are not flagged as floating.
        adjacency: dict[int, set] = {i: set() for i in range(self.n_nodes)}
        conduction_elements = (
            self._resistors + self._vsources + self._csources + self._inductors
        )
        if include_capacitors_in_graph:
            conduction_elements = conduction_elements + self._capacitors
        for node_a, node_b, _ in conduction_elements:
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

    def solve_transient(
        self,
        t_end: float,
        dt: float,
        v0=None,
        i_L0=None,
    ) -> tuple:
        """
        Solve the circuit in the time domain using backward Euler integration.

        Capacitors and inductors are replaced with backward Euler companion models:
          - Capacitor: conductance G_eq = C/dt plus history current I_eq = G_eq*V_prev.
          - Inductor:  conductance G_eq = dt/L plus history current I_eq = i_L_prev.
        Inductors are NOT in the B matrix here; they contribute to G only.

        The time loop uses jax.lax.scan, keeping the solver JAX-differentiable.

        Args:
            t_end: Simulation end time (s).
            dt:    Time step (s).
            v0:    Initial node voltages shape (n_nodes,). Defaults to zeros.
                   v0[0] must be 0 (ground).
            i_L0:  Initial inductor currents shape (n_inductor,). Defaults to zeros.

        Returns:
            t:           shape (n_steps,), time points dt, 2*dt, …
            v_nodes:     shape (n_steps, n_nodes), node voltages; column 0 = 0.
            i_vsrc:      shape (n_steps, n_vsrc), voltage-source currents.
            i_inductor:  shape (n_steps, n_inductor), inductor currents
                         (positive = node_a to node_b).
            i_capacitor: shape (n_steps, n_capacitor), capacitor currents
                         (positive = node_a to node_b, i.e. C*dV/dt).
        """
        self._validate(include_capacitors_in_graph=True)

        n_steps = round(t_end / dt)
        n = self.n_nodes - 1
        m_v = len(self._vsources)
        m_l = len(self._inductors)
        m_c = len(self._capacitors)

        if v0 is None:
            v0 = jnp.zeros(self.n_nodes)
        if i_L0 is None:
            i_L0 = jnp.zeros(m_l)

        def step(carry, _):
            v_all_prev, i_L_prev = carry

            G = jnp.zeros((n, n))
            B = jnp.zeros((n, m_v))
            z = jnp.zeros(n + m_v)

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

            for node_neg, node_pos, i_val in self._csources:
                if node_pos != 0:
                    z = z.at[node_pos - 1].add(i_val)
                if node_neg != 0:
                    z = z.at[node_neg - 1].add(-i_val)

            # Capacitor companions: G_eq = C/dt, history current from node_b to node_a.
            # i_cap = G_eq*(V_a - V_b) - I_eq  (≈ C*dV/dt)
            I_eq_caps = jnp.zeros(m_c)
            for k, (node_a, node_b, C_val) in enumerate(self._capacitors):
                G_eq = C_val / dt
                I_eq = G_eq * (v_all_prev[node_a] - v_all_prev[node_b])
                if node_a != 0:
                    G = G.at[node_a - 1, node_a - 1].add(G_eq)
                if node_b != 0:
                    G = G.at[node_b - 1, node_b - 1].add(G_eq)
                if node_a != 0 and node_b != 0:
                    G = G.at[node_a - 1, node_b - 1].add(-G_eq)
                    G = G.at[node_b - 1, node_a - 1].add(-G_eq)
                if node_a != 0:
                    z = z.at[node_a - 1].add(I_eq)
                if node_b != 0:
                    z = z.at[node_b - 1].add(-I_eq)
                I_eq_caps = I_eq_caps.at[k].set(I_eq)

            # Inductor companions: G_eq = dt/L, history current from node_a to node_b.
            # i_L = G_eq*(V_a - V_b) + I_eq  (≈ integral of V/L)
            I_eq_inds = jnp.zeros(m_l)
            for k_l, (node_a, node_b, L_val) in enumerate(self._inductors):
                G_eq_L = dt / L_val
                I_eq_L = i_L_prev[k_l]
                if node_a != 0:
                    G = G.at[node_a - 1, node_a - 1].add(G_eq_L)
                if node_b != 0:
                    G = G.at[node_b - 1, node_b - 1].add(G_eq_L)
                if node_a != 0 and node_b != 0:
                    G = G.at[node_a - 1, node_b - 1].add(-G_eq_L)
                    G = G.at[node_b - 1, node_a - 1].add(-G_eq_L)
                if node_b != 0:
                    z = z.at[node_b - 1].add(I_eq_L)
                if node_a != 0:
                    z = z.at[node_a - 1].add(-I_eq_L)
                I_eq_inds = I_eq_inds.at[k_l].set(I_eq_L)

            A = jnp.block(
                [
                    [G, B],
                    [B.T, jnp.zeros((m_v, m_v))],
                ]
            )
            x = jnp.linalg.solve(A, z)

            v_internal = x[:n]
            i_vsrc = -x[n : n + m_v]
            v_all = jnp.concatenate([jnp.zeros(1), v_internal])

            i_L_new = jnp.zeros(m_l)
            for k_l, (node_a, node_b, L_val) in enumerate(self._inductors):
                i_L_new = i_L_new.at[k_l].set(
                    (dt / L_val) * (v_all[node_a] - v_all[node_b]) + I_eq_inds[k_l]
                )

            i_cap = jnp.zeros(m_c)
            for k, (node_a, node_b, C_val) in enumerate(self._capacitors):
                i_cap = i_cap.at[k].set(
                    (C_val / dt) * (v_all[node_a] - v_all[node_b]) - I_eq_caps[k]
                )

            return (v_all, i_L_new), (v_all, i_vsrc, i_L_new, i_cap)

        _, outputs = jax.lax.scan(step, (v0, i_L0), None, length=n_steps)
        t = jnp.arange(1, n_steps + 1) * dt
        v_nodes_t, i_vsrc_t, i_inductor_t, i_capacitor_t = outputs
        return t, v_nodes_t, i_vsrc_t, i_inductor_t, i_capacitor_t
