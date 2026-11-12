"""
Modified Nodal Analysis (MNA) circuit solver in JAX.

Node 0 is always ground (reference node, V=0).

MNA system:
  [G  B] [v]   [i]
  [C  D] [j] = [e]

  G: conductance matrix (n_internal x n_internal)
  B: branch incidence (n_internal x (n_vsrc + n_inductor + n_tline))
  C = B^T
  D = 0
  v: internal node voltages
  j: branch currents (voltage sources, then inductors, then transmission
     lines; from - to + through branch)
  i: injected currents at internal nodes (from current sources)
  e: branch voltage values (voltage source values; 0 for inductors and
     transmission lines in DC)

Inductors are modelled as ideal short circuits in DC (zero voltage drop).
Capacitors are modelled as open circuits in DC (zero current).
Transmission lines are modelled as ideal 1:1 two-ports in DC (equal port
voltages, opposite port currents).
"""

import math

import jax
import jax.numpy as jnp


def _stamp_conductance(G, node_a, node_b, g):
    """Add conductance g between node_a and node_b (node 0 is ground)."""
    if node_a != 0:
        G = G.at[node_a - 1, node_a - 1].add(g)
    if node_b != 0:
        G = G.at[node_b - 1, node_b - 1].add(g)
    if node_a != 0 and node_b != 0:
        G = G.at[node_a - 1, node_b - 1].add(-g)
        G = G.at[node_b - 1, node_a - 1].add(-g)
    return G


def _stamp_current(z, node_neg, node_pos, i_val):
    """Inject current i_val into node_pos and extract it from node_neg."""
    if node_pos != 0:
        z = z.at[node_pos - 1].add(i_val)
    if node_neg != 0:
        z = z.at[node_neg - 1].add(-i_val)
    return z


def _switch_conductance(t, t_close, r_on, r_off, t_rise):
    """Conductance of a triggered closing switch at time t (see add_switch)."""
    g_on, g_off = 1.0 / r_on, 1.0 / r_off
    if isinstance(t_rise, (int, float)) and t_rise == 0:
        return jnp.where(t >= t_close, g_on, g_off)
    # The max() keeps the exponent <= 0, so there is no overflow (and no nan
    # gradient) before the switch closes. The closing law starts with zero
    # slope, so results stay smooth in t_close as it moves between samples.
    x = jnp.maximum(t - t_close, 0.0) / t_rise
    return g_off + (g_on - g_off) * (1.0 - (1.0 + x) * jnp.exp(-x))


def _tline_delay_steps(delay, dt):
    """Split a line delay into whole steps n >= 1 plus a fraction f in [0, 1)."""
    try:
        steps = float(delay) / float(dt)
    except (TypeError, jax.errors.ConcretizationTypeError) as err:
        raise ValueError(
            "Transmission line delay and dt must be concrete numbers (not traced "
            "JAX values): they set the length of the delay buffer"
        ) from err
    n = math.floor(steps)
    f = steps - n
    if f > 1.0 - 1e-9:  # absorb round-off, e.g. 4.9999999999 -> 5
        n, f = n + 1, 0.0
    if n < 1:
        raise ValueError(
            f"Transmission line delay ({float(delay):g} s) must be at least one "
            f"time step (dt = {float(dt):g} s)"
        )
    return n, f


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
        # (node_a, node_b, initial inductance)
        self._var_inductors: list[tuple] = []
        # (node_a, node_b, t_close, r_on, r_off, t_rise)
        self._switches: list[tuple] = []
        # (port1_pos, port1_neg, port2_pos, port2_neg, impedance, delay)
        self._tlines: list[tuple] = []

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

    def add_variable_inductor(
        self, node_a: int, node_b: int, inductance: float
    ) -> None:
        """
        Add a time-varying inductor between node_a and node_b.

        node_a is the + terminal, node_b is the - terminal, as for
        add_inductor(). The terminal voltage is the rate of change of flux,

            V(node_a) - V(node_b) = d(L i)/dt = L di/dt + i dL/dt,

        so the motional term i dL/dt (e.g. from an imploding z-pinch load) is
        included. `inductance` is the value at t = 0. In transient analysis
        the value at each step can be set with the `var_ind_values` argument
        of transient_step(), or by the `update_fn` of solve_transient().
        solve() treats the element as a short circuit and solve_ac() uses
        `inductance`.

        Variable-inductor currents are returned after the fixed-inductor
        currents in every `i_inductor` / `i_L` array, in the order the
        elements were added.
        """
        self._var_inductors.append((node_a, node_b, inductance))

    def add_switch(
        self,
        node_a: int,
        node_b: int,
        t_close: float,
        r_on: float,
        r_off: float = 1e9,
        t_rise: float = 0.0,
    ) -> None:
        """
        Add a triggered closing switch between node_a and node_b.

        The switch is a time-dependent resistor. It has resistance r_off
        before t_close. After t_close, its conductance rises smoothly to
        1/r_on on the time scale t_rise:

            g(t) = 1/r_off + (1/r_on - 1/r_off) [1 - (1 + x) exp(-x)],
            x = (t - t_close) / t_rise

        The conductance starts rising with zero slope and reaches 26%, 59%,
        80% and 96% of its final value at x = 1, 2, 3 and 5 (10-90% in about
        3.4 t_rise).

        With t_rise = 0 (the default) the switch closes instantly at the first
        time step at or after t_close. Use t_rise > 0 (at least a few time
        steps) if you need gradients with respect to t_close, e.g. to
        optimise trigger timing or study jitter. The switch never reopens.
        solve() and solve_ac() use its state at t = 0.

        Keep r_on and r_off within about 1e-6 and 1e9 times the circuit's
        other impedances: more extreme ratios make the linear solve
        ill-conditioned and cost digits of accuracy.
        """
        self._switches.append((node_a, node_b, t_close, r_on, r_off, t_rise))

    def add_transmission_line(
        self,
        port1_pos: int,
        port1_neg: int,
        port2_pos: int,
        port2_neg: int,
        impedance: float,
        delay: float,
    ) -> None:
        """
        Add a lossless transmission line between two ports.

        Port 1 is (port1_pos, port1_neg) and port 2 is (port2_pos, port2_neg).
        `impedance` is the characteristic impedance Z0 (Ω). `delay` is the
        one-way transit time (s).

        In transient analysis the line is solved exactly by the method of
        characteristics (Bergeron model). Each port is a resistor Z0 in
        parallel with a current source set by the wave that left the other
        port one delay earlier. `delay` must be at least one time step and a
        concrete number (it sets a buffer length). If it is not a whole
        number of steps, the delayed waves are linearly interpolated. Z0 can
        be a traced JAX value. The line starts at rest, charged to its port
        voltages in `v0`, so a uniformly charged line needs equal voltages at
        both ports.

        solve() treats the line as an ideal 1:1 two-port (equal port
        voltages, opposite port currents). solve_ac() uses the exact phasor
        two-port, which is singular when omega * delay is a multiple of pi.

        The ports are not joined by a conduction path in the model, so each
        port must have its own path to ground (usually its - terminal is
        node 0).
        """
        self._tlines.append(
            (port1_pos, port1_neg, port2_pos, port2_neg, impedance, delay)
        )

    def _validate(self, include_capacitors_in_graph: bool = False) -> None:
        all_elements = [
            ("Resistor", self._resistors),
            ("Voltage source", self._vsources),
            ("Current source", self._csources),
            ("Inductor", self._inductors),
            ("Capacitor", self._capacitors),
            ("Variable inductor", self._var_inductors),
            ("Switch", self._switches),
        ]
        # Each transmission line port is checked like a two-terminal element
        for p1p, p1n, p2p, p2n, *_ in self._tlines:
            all_elements.append(("Transmission line port 1", [(p1p, p1n)]))
            all_elements.append(("Transmission line port 2", [(p2p, p2n)]))

        for label, elements in all_elements:
            for node_a, node_b, *_ in elements:
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
            self._resistors
            + self._vsources
            + self._csources
            + self._inductors
            + self._var_inductors
            + self._switches
        )
        for p1p, p1n, p2p, p2n, *_ in self._tlines:
            conduction_elements = conduction_elements + [(p1p, p1n), (p2p, p2n)]
        if include_capacitors_in_graph:
            conduction_elements = conduction_elements + self._capacitors
        for node_a, node_b, *_ in conduction_elements:
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

            i_inductor:  jnp array shape (n_inductor + n_var_inductor,),
                         current through each inductor, fixed inductors first
                         (positive = current flowing from node_a through the
                         inductor to node_b).

            i_capacitor: jnp array shape (n_capacitor,), current through each
                         capacitor; always zero in DC analysis.
        """
        self._validate()

        n = self.n_nodes - 1
        m_v = len(self._vsources)
        m_l = len(self._inductors) + len(self._var_inductors)
        m_t = len(self._tlines)
        m = m_v + m_l + m_t

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
        for k_l, (node_a, node_b, _) in enumerate(
            self._inductors + self._var_inductors
        ):
            k = m_v + k_l
            if node_a != 0:
                B = B.at[node_a - 1, k].add(1.0)
            if node_b != 0:
                B = B.at[node_b - 1, k].add(-1.0)
            # z[n + k] stays 0: V(node_a) - V(node_b) = 0

        # Switches in their t = 0 state
        for node_a, node_b, t_close, r_on, r_off, t_rise in self._switches:
            g = _switch_conductance(0.0, t_close, r_on, r_off, t_rise)
            G = _stamp_conductance(G, node_a, node_b, g)

        # Transmission lines are ideal 1:1 two-ports in DC. Branch current j
        # flows into port 1 (+) and out of port 2 (+); the constraint row is
        # V(port 1) - V(port 2) = 0.
        for k_t, (p1p, p1n, p2p, p2n, _, _) in enumerate(self._tlines):
            k = m_v + m_l + k_t
            for node, sign in ((p1p, 1.0), (p1n, -1.0), (p2p, -1.0), (p2n, 1.0)):
                if node != 0:
                    B = B.at[node - 1, k].add(sign)

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
        i_inductor = x[n + m_v : n + m_v + m_l]

        # Capacitors are open circuits in DC: no stamp, current is always zero.
        i_capacitor = jnp.zeros(len(self._capacitors))

        v_nodes = jnp.concatenate([jnp.zeros(1), v_internal])
        return v_nodes, i_vsrc, i_inductor, i_capacitor

    def solve_ac(self, omega: float) -> tuple:
        """
        Solve the circuit for sinusoidal steady state (phasor/AC analysis)
        at angular frequency omega (rad/s).

        Voltage and current source values are treated as complex phasor
        amplitudes (a real Python/JAX float is a phasor with zero phase).

        Unlike solve() and solve_transient(), capacitors and inductors are
        folded directly into the admittance matrix G as Y_C = j*omega*C and
        Y_L = 1/(j*omega*L). No branch-current unknowns are needed for them:
        in MNA, branch currents are only required to enforce a voltage
        constraint, which is the case for voltage sources but not for linear
        admittances in the frequency domain.

        Returns:

            v_nodes: complex jnp array shape (n_nodes,), phasor node
                     voltages; v_nodes[0] = 0.

            i_vsrc:  complex jnp array shape (n_vsrc,), phasor current
                     through each voltage source (SPICE convention:
                     positive = current flowing from + terminal through the
                     external circuit).
        """
        self._validate(include_capacitors_in_graph=True)

        n = self.n_nodes - 1
        m_v = len(self._vsources)

        G = jnp.zeros((n, n), dtype=complex)
        B = jnp.zeros((n, m_v), dtype=complex)
        z = jnp.zeros(n + m_v, dtype=complex)

        for node_a, node_b, r in self._resistors:
            y = 1.0 / r
            if node_a != 0:
                G = G.at[node_a - 1, node_a - 1].add(y)
            if node_b != 0:
                G = G.at[node_b - 1, node_b - 1].add(y)
            if node_a != 0 and node_b != 0:
                G = G.at[node_a - 1, node_b - 1].add(-y)
                G = G.at[node_b - 1, node_a - 1].add(-y)

        for node_a, node_b, c in self._capacitors:
            y = 1j * omega * c
            if node_a != 0:
                G = G.at[node_a - 1, node_a - 1].add(y)
            if node_b != 0:
                G = G.at[node_b - 1, node_b - 1].add(y)
            if node_a != 0 and node_b != 0:
                G = G.at[node_a - 1, node_b - 1].add(-y)
                G = G.at[node_b - 1, node_a - 1].add(-y)

        for node_a, node_b, ind in self._inductors + self._var_inductors:
            y = 1.0 / (1j * omega * ind)
            if node_a != 0:
                G = G.at[node_a - 1, node_a - 1].add(y)
            if node_b != 0:
                G = G.at[node_b - 1, node_b - 1].add(y)
            if node_a != 0 and node_b != 0:
                G = G.at[node_a - 1, node_b - 1].add(-y)
                G = G.at[node_b - 1, node_a - 1].add(-y)

        # Switches in their t = 0 state
        for node_a, node_b, t_close, r_on, r_off, t_rise in self._switches:
            y = _switch_conductance(0.0, t_close, r_on, r_off, t_rise)
            G = _stamp_conductance(G, node_a, node_b, y)

        # Lossless line, theta = omega * delay. With port currents flowing into
        # the + terminals: Y11 = Y22 = -j cot(theta) / Z0, Y12 = Y21 = j / (Z0
        # sin(theta)).
        for p1p, p1n, p2p, p2n, z0, delay in self._tlines:
            theta = omega * delay
            y_self = -1j / (z0 * jnp.tan(theta))
            y_mutual = 1j / (z0 * jnp.sin(theta))
            ports = ((p1p, p1n), (p2p, p2n))
            for k_row, (row_pos, row_neg) in enumerate(ports):
                for k_col, (col_pos, col_neg) in enumerate(ports):
                    y = y_self if k_row == k_col else y_mutual
                    for row, s_row in ((row_pos, 1.0), (row_neg, -1.0)):
                        for col, s_col in ((col_pos, 1.0), (col_neg, -1.0)):
                            if row != 0 and col != 0:
                                G = G.at[row - 1, col - 1].add(s_row * s_col * y)

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

        A = jnp.block(
            [
                [G, B],
                [B.T, jnp.zeros((m_v, m_v), dtype=complex)],
            ]
        )

        x = jnp.linalg.solve(A, z)

        v_internal = x[:n]
        i_vsrc = -x[n:]

        v_nodes = jnp.concatenate([jnp.zeros(1, dtype=complex), v_internal])
        return v_nodes, i_vsrc

    def tline_history_init(self, v0, dt: float) -> list:
        """
        Initial transmission-line history for transient_step().

        Each line starts at rest (zero current), charged to its port voltages
        in v0. Returns a list with one array per line, of shape
        (n_delay + 1, 2), where n_delay = floor(delay / dt). Row -1 is the
        most recent step. The columns hold the waves V/Z0 + I leaving port 1
        and port 2.
        """
        hist = []
        for p1p, p1n, p2p, p2n, z0, delay in self._tlines:
            n_d, _ = _tline_delay_steps(delay, dt)
            v1 = v0[p1p] - v0[p1n]
            v2 = v0[p2p] - v0[p2n]
            waves = jnp.stack([v1 / z0, v2 / z0])
            hist.append(jnp.broadcast_to(waves, (n_d + 1, 2)))
        return hist

    def _tline_history_currents(self, tline_hist, dt):
        """History current sources (I_h1, I_h2) of each line for this step."""
        currents = []
        for (_, _, _, _, _, delay), hist in zip(self._tlines, tline_hist):
            n_d, f = _tline_delay_steps(delay, dt)
            # Waves that left each port one delay ago: t - (n_d + f) dt
            delayed = (1.0 - f) * hist[-n_d] + f * hist[-(n_d + 1)]
            # Port 1 sees the wave from port 2, and vice versa
            currents.append((delayed[1], delayed[0]))
        return currents

    def tline_history_update(self, tline_hist, v_all, dt: float) -> list:
        """
        Append the step just solved by transient_step() to the line history.

        Args:
            tline_hist: history passed to the transient_step() call.
            v_all:      node voltages returned by that call.
            dt:         time step (s).

        Returns:
            The history to pass to the next transient_step() call.
        """
        new_hist = []
        currents = self._tline_history_currents(tline_hist, dt)
        for (p1p, p1n, p2p, p2n, z0, _), hist, (I_h1, I_h2) in zip(
            self._tlines, tline_hist, currents
        ):
            v1 = v_all[p1p] - v_all[p1n]
            v2 = v_all[p2p] - v_all[p2n]
            # i_k = v_k/Z0 - I_hk, so the outgoing wave v_k/Z0 + i_k is:
            waves = jnp.stack([2.0 * v1 / z0 - I_h1, 2.0 * v2 / z0 - I_h2])
            new_hist.append(jnp.concatenate([hist[1:], waves[None, :]]))
        return new_hist

    def transient_step(
        self,
        v_prev,
        i_L_prev,
        dt: float,
        vsrc_values=None,
        csrc_values=None,
        var_ind_values=None,
        var_ind_values_prev=None,
        t=None,
        tline_hist=None,
    ) -> tuple:
        """
        Advance the circuit by a single backward Euler timestep.

        This is the building block that solve_transient() iterates. It is
        exposed separately so an external solver can drive the circuit: pass
        vsrc_values (or csrc_values) to override the stored source values for
        this step, and step the circuit in lockstep with the other solver.

        Args:

            v_prev:      shape (n_nodes,), node voltages at the previous step.

            i_L_prev:    shape (n_inductor + n_var_inductor,), inductor
                         currents at the previous step (fixed inductors first).

            dt:          Time step (s).

            vsrc_values: optional shape (n_vsrc,), voltage source values to use
                         for this step in place of the stored ones.

            csrc_values: optional shape (n_csource,), current source values to
                         use for this step in place of the stored ones.

            var_ind_values: optional shape (n_var_inductor,), variable
                         inductances at the end of this step. Defaults to the
                         stored values.

            var_ind_values_prev: optional shape (n_var_inductor,), variable
                         inductances at the previous step. Defaults to the
                         stored values.

            t:           Time at the end of this step (s). Required if the
                         circuit has switches.

            tline_hist:  Transmission-line history. Required if the circuit
                         has transmission lines. Start from
                         tline_history_init(v0, dt) and, after each step,
                         advance it with tline_history_update().

        Variable inductors use a flux-conserving companion model. Backward
        Euler applied to V = d(L i)/dt gives

            i_new = (dt / L_new) V + (L_prev / L_new) i_prev,

        which reduces to the fixed-inductor model when L_new = L_prev.

        Returns:

            v_all:      shape (n_nodes,), node voltages; entry 0 = 0.

            i_vsrc:     shape (n_vsrc,), voltage-source currents.

            i_L_new:    shape (n_inductor + n_var_inductor,), inductor
                        currents (fixed inductors first).

            i_cap:      shape (n_capacitor,), capacitor currents.
        """
        n = self.n_nodes - 1
        m_v = len(self._vsources)
        m_l = len(self._inductors) + len(self._var_inductors)
        m_c = len(self._capacitors)

        if self._switches and t is None:
            raise ValueError("transient_step() needs t when the circuit has switches")
        if self._tlines and tline_hist is None:
            raise ValueError(
                "transient_step() needs tline_hist when the circuit has transmission "
                "lines (see tline_history_init and tline_history_update)"
            )

        # Inductor list (node_a, node_b, L at end of step, history-current
        # scale). Fixed inductors have scale 1; variable inductors carry
        # flux, so their history current is scaled by L_prev / L_new.
        inductors = [(a, b, L_val, None) for a, b, L_val in self._inductors]
        for k_v, (node_a, node_b, L_init) in enumerate(self._var_inductors):
            L_new = L_init if var_ind_values is None else var_ind_values[k_v]
            L_old = L_init if var_ind_values_prev is None else var_ind_values_prev[k_v]
            inductors.append((node_a, node_b, L_new, L_old / L_new))

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
            if vsrc_values is not None:
                v = vsrc_values[k]
            if node_pos != 0:
                B = B.at[node_pos - 1, k].add(1.0)
            if node_neg != 0:
                B = B.at[node_neg - 1, k].add(-1.0)
            z = z.at[n + k].set(v)

        for k_c, (node_neg, node_pos, i_val) in enumerate(self._csources):
            if csrc_values is not None:
                i_val = csrc_values[k_c]
            if node_pos != 0:
                z = z.at[node_pos - 1].add(i_val)
            if node_neg != 0:
                z = z.at[node_neg - 1].add(-i_val)

        for node_a, node_b, t_close, r_on, r_off, t_rise in self._switches:
            g = _switch_conductance(t, t_close, r_on, r_off, t_rise)
            G = _stamp_conductance(G, node_a, node_b, g)

        # Transmission lines (Bergeron): each port is a conductance 1/Z0 in
        # parallel with a history source I_h; current into the + terminal is
        # i_k = v_k/Z0 - I_hk.
        if self._tlines:
            currents = self._tline_history_currents(tline_hist, dt)
            for (p1p, p1n, p2p, p2n, z0, _), (I_h1, I_h2) in zip(
                self._tlines, currents
            ):
                G = _stamp_conductance(G, p1p, p1n, 1.0 / z0)
                G = _stamp_conductance(G, p2p, p2n, 1.0 / z0)
                z = _stamp_current(z, p1n, p1p, I_h1)
                z = _stamp_current(z, p2n, p2p, I_h2)

        # Capacitor companions: G_eq = C/dt, history current from node_b to node_a.
        # i_cap = G_eq*(V_a - V_b) - I_eq  (≈ C*dV/dt)
        I_eq_caps = jnp.zeros(m_c)
        for k, (node_a, node_b, C_val) in enumerate(self._capacitors):
            G_eq = C_val / dt
            I_eq = G_eq * (v_prev[node_a] - v_prev[node_b])
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
        for k_l, (node_a, node_b, L_val, scale) in enumerate(inductors):
            G_eq_L = dt / L_val
            I_eq_L = i_L_prev[k_l] if scale is None else scale * i_L_prev[k_l]
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
        for k_l, (node_a, node_b, L_val, _) in enumerate(inductors):
            i_L_new = i_L_new.at[k_l].set(
                (dt / L_val) * (v_all[node_a] - v_all[node_b]) + I_eq_inds[k_l]
            )

        i_cap = jnp.zeros(m_c)
        for k, (node_a, node_b, C_val) in enumerate(self._capacitors):
            i_cap = i_cap.at[k].set(
                (C_val / dt) * (v_all[node_a] - v_all[node_b]) - I_eq_caps[k]
            )

        return v_all, i_vsrc, i_L_new, i_cap

    def solve_transient(
        self,
        t_end: float,
        dt: float,
        v0=None,
        i_L0=None,
        state0=None,
        update_fn=None,
    ) -> tuple:
        """
        Solve the circuit in the time domain using backward Euler integration.

        Capacitors and inductors are replaced with backward Euler companion models:
          - Capacitor: conductance G_eq = C/dt plus history current I_eq = G_eq*V_prev.
          - Inductor:  conductance G_eq = dt/L plus history current I_eq = i_L_prev.
          - Variable inductor: G_eq = dt/L_new plus flux-conserving history
            current I_eq = (L_prev/L_new)*i_L_prev (see transient_step()).
          - Switch: a resistor whose value follows the switch's closing law
            at the end of each step (see add_switch()).
          - Transmission line: exact method of characteristics; each port is
            1/Z0 plus a history current from the other port one delay
            earlier (see add_transmission_line()).

        Inductors are NOT in the B matrix here; they contribute to G only.

        The time loop uses jax.lax.scan, keeping the solver JAX-differentiable.

        Coupling to an external model (e.g. the equation of motion of an
        imploding plasma load) is done with `state0` and `update_fn`. At the
        start of every step, update_fn advances the external state with the
        circuit solution from the previous step and returns the variable
        inductances for the new step:

            state, var_ind_values = update_fn(state, t, dt, v_prev, i_L_prev)

        where t is the time at the end of the step, v_prev has shape
        (n_nodes,) and i_L_prev has shape (n_inductor + n_var_inductor,).
        `state` may be any pytree of arrays. The circuit step itself is then
        solved implicitly. The external model therefore lags the circuit by
        one step (first-order accurate, like backward Euler). It must be
        written with jnp so the whole run stays differentiable.

        Args:

            t_end: Simulation end time (s).

            dt:    Time step (s).

            v0:    Initial node voltages shape (n_nodes,). Defaults to zeros.
                   v0[0] must be 0 (ground).

            i_L0:  Initial inductor currents shape
                   (n_inductor + n_var_inductor,). Defaults to zeros.

            state0:    Initial external state (pytree). Used only with
                       update_fn.

            update_fn: Optional coupling function, see above. Without it,
                       variable inductors keep their stored values.

        Returns:

            t:           shape (n_steps,), time points dt, 2*dt, …

            v_nodes:     shape (n_steps, n_nodes), node voltages; column 0 = 0.

            i_vsrc:      shape (n_steps, n_vsrc), voltage-source currents.

            i_inductor:  shape (n_steps, n_inductor + n_var_inductor), inductor
                         currents, fixed inductors first
                         (positive = node_a to node_b).

            i_capacitor: shape (n_steps, n_capacitor), capacitor currents
                         (positive = node_a to node_b, i.e. C*dV/dt).

            state:       only returned when update_fn is given: the external
                         state at each step, stacked along a leading axis of
                         length n_steps.
        """
        self._validate(include_capacitors_in_graph=True)

        n_steps = round(t_end / dt)
        m_l = len(self._inductors) + len(self._var_inductors)

        if v0 is None:
            v0 = jnp.zeros(self.n_nodes)
        if i_L0 is None:
            i_L0 = jnp.zeros(m_l)
        L_var0 = jnp.array([L_val for _, _, L_val in self._var_inductors])
        tline_hist0 = self.tline_history_init(v0, dt)

        def step(carry, t):
            v_all_prev, i_L_prev, L_var_prev, state, tline_hist = carry
            if update_fn is None:
                L_var = L_var_prev
            else:
                state, L_var = update_fn(state, t, dt, v_all_prev, i_L_prev)
                L_var = jnp.reshape(jnp.asarray(L_var), L_var_prev.shape)
            v_all, i_vsrc, i_L_new, i_cap = self.transient_step(
                v_all_prev,
                i_L_prev,
                dt,
                var_ind_values=L_var,
                var_ind_values_prev=L_var_prev,
                t=t,
                tline_hist=tline_hist,
            )
            tline_hist = self.tline_history_update(tline_hist, v_all, dt)
            outputs = (v_all, i_vsrc, i_L_new, i_cap)
            if update_fn is not None:
                outputs = outputs + (state,)
            return (v_all, i_L_new, L_var, state, tline_hist), outputs

        t = jnp.arange(1, n_steps + 1) * dt
        carry0 = (v0, i_L0, L_var0, state0, tline_hist0)
        _, outputs = jax.lax.scan(step, carry0, t)
        return (t,) + tuple(outputs)
