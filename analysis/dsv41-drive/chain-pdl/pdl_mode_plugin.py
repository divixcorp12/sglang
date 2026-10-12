"""pytest plugin: run the lease-protocol suites against the PDL test build of the expert_stream device module.

    CHAIN_PDL_DEFINES=EXL3_RAM_MISS_TEST_PDL[,EXL3_RAM_MISS_TEST_PDL_EARLY] \\
      PYTHONPATH=<pdl-probe worktree>/python:<this dir> python -m pytest -p pdl_mode_plugin <suites>

With CHAIN_PDL_DEFINES empty or unset it changes nothing (the baseline). Otherwise every device module the suites
load is the hooked build: the production loader (`_device_module`) returns device_module_with_hooks(defines), and a
test's own hook build (device_module_with_hooks(extra)) gets the PDL defines added to its own.
"""
import os

DEFINES = [d for d in os.environ.get("CHAIN_PDL_DEFINES", "").split(",") if d]
_loads = {"production": 0, "hooked": 0}


def pytest_configure(config):
    if not DEFINES:
        return
    from sglang.kernels.ops.moe import expert_stream_transport as ops

    original = ops.device_module_with_hooks
    cache = {}

    def production(layout="exl3"):
        _loads["production"] += 1
        if layout not in cache:
            cache[layout] = original(DEFINES, layout)
        return cache[layout]

    def hooked(defines, layout="exl3"):
        _loads["hooked"] += 1
        return original(list(defines) + [d for d in DEFINES if d not in defines], layout)

    ops._device_module = production
    ops.device_module_with_hooks = hooked


def pytest_report_header(config):
    return f"chain-pdl: device module defines {DEFINES or 'none (production build)'}"


def pytest_terminal_summary(terminalreporter):
    terminalreporter.write_line(f"chain-pdl: defines {DEFINES or 'none'}; device-module loads through the plugin {_loads}")
