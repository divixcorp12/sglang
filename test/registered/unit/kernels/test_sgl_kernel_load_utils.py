import builtins
import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch

LOAD_UTILS_PATH = (
    Path(__file__).parents[4]
    / "python/sglang/kernels/aot/python/sgl_kernel/load_utils.py"
)


def load_utils_module():
    spec = importlib.util.spec_from_file_location(
        "test_sgl_kernel_load_utils_module", LOAD_UTILS_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load sgl_kernel load utilities")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestArchitectureSpecificOpsLoading(unittest.TestCase):
    def test_missing_libraries_report_every_searched_fallback_pattern(self):
        module = load_utils_module()
        search_dirs = [Path("/wheel/sgl_kernel"), Path("/source/sgl_kernel")]
        original_import = builtins.__import__

        def import_without_common_ops(name, *args, **kwargs):
            if name == "common_ops":
                raise ImportError("common_ops unavailable")
            return original_import(name, *args, **kwargs)

        with (
            patch.object(module, "_get_compute_capability", return_value=100),
            patch.object(module, "_get_package_search_dirs", return_value=search_dirs),
            patch.object(module.glob, "glob", return_value=[]),
            patch("builtins.__import__", side_effect=import_without_common_ops),
            self.assertRaises(ImportError) as raised,
        ):
            module._load_architecture_specific_ops()

        message = str(raised.exception)
        self.assertIn("/wheel/sgl_kernel/common_ops.*", message)
        self.assertIn("/source/sgl_kernel/common_ops.*", message)
        self.assertIn("common_ops unavailable", message)


if __name__ == "__main__":
    unittest.main()
