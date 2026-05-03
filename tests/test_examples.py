from spicex.utils import run_example_main
import jax.numpy as jnp

rel_tol = 1e-8


def test_resistors_in_series():
    # 5 V source, 1 kΩ + 2 kΩ in series: V = [0, 5, 10/3] V, I = 5/3000 A
    v_nodes, i_vsrc = run_example_main(
        "examples/resistors_in_series/resistors_in_series.py"
    )
    assert jnp.allclose(v_nodes, jnp.array([0.0, 5.0, 10.0 / 3.0]), rtol=rel_tol)
    assert jnp.allclose(i_vsrc, jnp.array([5.0 / 3000.0]), rtol=rel_tol)


def test_resistors_in_parallel():
    # 5 V source, 1 kΩ ‖ 2 kΩ = 2/3 kΩ: V = [0, 5] V, I = 5 / (2000/3) = 3/400 A
    v_nodes, i_vsrc = run_example_main(
        "examples/resistors_in_parallel/resistors_in_parallel.py"
    )
    assert jnp.allclose(v_nodes, jnp.array([0.0, 5.0]), rtol=rel_tol)
    assert jnp.allclose(i_vsrc, jnp.array([3.0 / 400.0]), rtol=rel_tol)


def test_current_source_with_resistor():
    # 1 mA current source into node 1, 1 kΩ to ground: V(node_1) = 1 mA * 1 kΩ = 1 V
    v_nodes, i_vsrc = run_example_main(
        "examples/current_source_with_resistor/current_source_with_resistor.py"
    )
    assert jnp.allclose(v_nodes, jnp.array([0.0, 1.0]), rtol=rel_tol)
    assert i_vsrc.shape == (0,)
