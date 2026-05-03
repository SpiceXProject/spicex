# SpiceX

SpiceX is a differentiable SPICE circuit solver in Python/JAX.
It supports automatic differentiation for gradient-based circuit optimization.

The library is in the `spicex/` folder.
Examples are in the `examples/` folder.
Unit and Integration tests are in the `tests/` folder.
Read all source files before making changes.

---

## Commands

```bash
# Run tests
pytest

# Run examples, e.g.
python examples/resistors_in_series/resistors_in_series.py
python examples/resistors_in_parallel/resistors_in_parallel.py
python examples/current_source_with_resistor/current_source_with_resistor.py
python examples/maximum_power_transfer/maximum_power_transfer.py
```

---

## Project Structure

```
spicex/          # Python library source code
tests/           # Unit and integration tests
examples/        # Example Python scripts
docs/            # Documentation
pyproject.toml   # Build config
```

---

## Conventions

**Node 0 is always ground (V=0).** It is never a free variable.


---

## JAX Rules

### Use `jnp`, not `np`, inside `solve()`

All arrays created inside `solve()` use `jnp` so JAX can differentiate through them.


---

## Testing Conventions

Unit and integration tests are in the `test/` folder.

Newly added examples should have an integration test added in the `test_examples.py` file, e.g.

```python
result = run_example_main("examples/<dir>/<script>.py")
v_nodes, i_vsrc = result
assert jnp.allclose(v_nodes[1], 5.0, rtol=1e-8)
```


---

## Example Conventions

Every example must follow this structure:

```python
import jax
import spicex

jax.config.update("jax_enable_x64", True)

"""
Brief description.

Author (Year)

Usage:
  python script_name.py
"""


def main():
    # build and solve circuit
    ...
    return results  # must return values for test assertions


if __name__ == "__main__":
    main()
```

Also include a `README.md` with an ASCII circuit diagram in each example folder


---


## Adding New Circuit Elements

To add a new linear element (e.g., controlled source, inductor):

1. Add a storage list in `Circuit.__init__`:
   ```python
   self._new_elements: list[tuple] = []  # (node_a, node_b, value)
   ```
2. Add an `add_*` method with a docstring explaining the terminal convention
3. Update `_validate()`: add the new list to `all_elements` so node validation and floating-node detection cover it
4. Add the MNA stamp in `solve()` — modify `G`, `B`, or `z` as required by the element's stamp
5. Add tests in `test_validation.py`: out-of-range nodes, self-loop, and floating node for the new element type
6. Add an example in `examples/<new_example>/` with a `README.md` and a corresponding test in `test_examples.py`

