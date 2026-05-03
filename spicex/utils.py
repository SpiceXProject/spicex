import sys
import os
import importlib.util
from pathlib import Path


def run_example_main(example_path, argv=None):
    """
    Utility to run an example script's main() as if from its directory, with optional argv patching.
    Args:
        example_path (str or Path): Path to the example script (e.g., 'examples/foo/foo.py')
        argv (list, optional): List of arguments to patch sys.argv with. If None, uses [script_name].
    """
    example_path = Path(example_path)
    example_dir = example_path.parent
    script_name = example_path.name
    old_cwd = os.getcwd()
    old_argv = sys.argv.copy()
    try:
        spec = importlib.util.spec_from_file_location(
            Path(script_name).stem, example_path
        )
        os.chdir(example_dir)
        sys.argv = [script_name] + (argv if argv is not None else [])
        module = importlib.util.module_from_spec(spec)
        sys.modules[Path(script_name).stem] = module
        spec.loader.exec_module(module)
        if hasattr(module, "main"):
            result = module.main()
        else:
            raise AttributeError(f"No main() in {example_path}")
    finally:
        os.chdir(old_cwd)
        sys.argv = old_argv
    return result
