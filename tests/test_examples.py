from spicex.utils import run_example_main
import jax.numpy as jnp

rel_tol = 1e-8


def test_current_source_with_resistor():
    # 1 mA current source into node 1, 1 kΩ to ground: V(node_1) = 1 mA * 1 kΩ = 1 V
    v_nodes, i_vsrc = run_example_main(
        "examples/current_source_with_resistor/current_source_with_resistor.py"
    )
    assert jnp.allclose(v_nodes, jnp.array([0.0, 1.0]), rtol=rel_tol)
    assert i_vsrc.shape == (0,)


def test_maximum_power_transfer():
    # V_s=10V, R_s=1kΩ: optimal R_L = R_s = 1kΩ, P_max = V_s^2/(4*R_s) = 25 mW
    R_L_opt, P_opt = run_example_main(
        "examples/maximum_power_transfer/maximum_power_transfer.py"
    )
    assert jnp.isclose(R_L_opt, jnp.array(1e3), rtol=1e-3)
    assert jnp.isclose(P_opt, jnp.array(10.0**2 / (4.0 * 1e3)), rtol=1e-3)


def test_resistors_in_parallel():
    # 5 V source, 1 kΩ ‖ 2 kΩ = 2/3 kΩ: V = [0, 5] V, I = 5 / (2000/3) = 3/400 A
    v_nodes, i_vsrc = run_example_main(
        "examples/resistors_in_parallel/resistors_in_parallel.py"
    )
    assert jnp.allclose(v_nodes, jnp.array([0.0, 5.0]), rtol=rel_tol)
    assert jnp.allclose(i_vsrc, jnp.array([3.0 / 400.0]), rtol=rel_tol)


def test_resistors_in_series():
    # 5 V source, 1 kΩ + 2 kΩ in series: V = [0, 5, 10/3] V, I = 5/3000 A
    v_nodes, i_vsrc = run_example_main(
        "examples/resistors_in_series/resistors_in_series.py"
    )
    assert jnp.allclose(v_nodes, jnp.array([0.0, 5.0, 10.0 / 3.0]), rtol=rel_tol)
    assert jnp.allclose(i_vsrc, jnp.array([5.0 / 3000.0]), rtol=rel_tol)


def test_capacitor_with_resistor():
    # 5 V source, 1 kΩ and 10 µF in parallel: C open in DC, I_src = 5 mA, I_cap = 0
    v_nodes, i_vsrc, i_capacitor = run_example_main(
        "examples/capacitor_with_resistor/capacitor_with_resistor.py"
    )
    assert jnp.allclose(v_nodes, jnp.array([0.0, 5.0]), rtol=rel_tol)
    assert jnp.allclose(i_vsrc, jnp.array([5e-3]), rtol=rel_tol)
    assert jnp.allclose(i_capacitor, jnp.array([0.0]), rtol=rel_tol)


def test_inductor_with_resistor():
    # 5 V source, 10 mH inductor (short in DC), 1 kΩ: V(n2)=5 V, I=5 mA
    v_nodes, i_vsrc, i_inductor = run_example_main(
        "examples/inductor_with_resistor/inductor_with_resistor.py"
    )
    assert jnp.allclose(v_nodes, jnp.array([0.0, 5.0, 5.0]), rtol=rel_tol)
    assert jnp.allclose(i_vsrc, jnp.array([5e-3]), rtol=rel_tol)
    assert jnp.allclose(i_inductor, jnp.array([5e-3]), rtol=rel_tol)


def test_rc_step_response():
    # 1 V source, R=1kΩ, C=1µF: τ=1ms, V_C(t)=1−exp(−t/τ), simulate 5ms with dt=0.01ms
    t, v_nodes, i_vsrc, i_inductor, i_capacitor = run_example_main(
        "examples/rc_step_response/rc_step_response.py"
    )
    tau = 1e-3
    v_analytic = 1.0 * (1.0 - jnp.exp(-t / tau))
    assert jnp.allclose(v_nodes[:, 2], v_analytic, rtol=1e-2)
    assert float(v_nodes[-1, 2]) > 0.99


def test_rlc_series():
    # R=2Ω, L=10mH, C=100µF: ω₀=1000 rad/s, ζ=0.1 (underdamped), simulate 50ms
    t, v_nodes, i_vsrc, i_inductor, i_capacitor = run_example_main(
        "examples/rlc_series/rlc_series.py"
    )
    assert v_nodes.shape == (5000, 4)
    assert i_inductor.shape == (5000, 1)
    assert i_capacitor.shape == (5000, 1)
    # Underdamped: V_C must overshoot the 1 V source (analytic peak ≈ 1.73 V)
    assert float(jnp.max(v_nodes[:, 3])) > 1.5
    # Settled near V_S at t_end = 50 ms = 5τ (envelope decayed to e^-5 ≈ 0.7%)
    assert abs(float(v_nodes[-1, 3]) - 1.0) < 0.05


def test_resistor_sweep():
    R_L_values, powers = run_example_main("examples/resistor_sweep/resistor_sweep.py")
    V_S = 10.0
    R_S = 1e3

    def analytical_power(R_L):
        return V_S**2 * R_L / (R_S + R_L) ** 2

    expected = analytical_power(R_L_values)
    assert jnp.allclose(powers, expected, rtol=rel_tol)
    # Peak should be near R_L = R_S = 1 kΩ
    idx_peak = int(jnp.argmax(powers))
    assert 0.95 * R_S <= float(R_L_values[idx_peak]) <= 1.05 * R_S
