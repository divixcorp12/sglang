"""Nsight Systems options for run_arm.sh's NSYS_TRACE=1 path. Pure, no I/O.

`--cuda-graph-trace=graph` is the default: `node` makes cudaGraphLaunch cost ~0.77 us per traced node, so a
node-mode report fabricates a step-tail idle gap and inflates ms/token (CLAUDE.md). Node mode is for per-kernel
attribution only.

Graph mode is refused when the arm runs the RAM-miss copy engine: graph-mode CUPTI tracing deadlocks the copy wait
(LEASE_PROTOCOL.md 7.6; `analysis/dsv41-drive/copy-engine/smoke.sh` refuses it the same way).
"""

from __future__ import annotations

GRAPH_TRACE_MODES = ("graph", "node")
COPY_ENGINE_ENV = "SGLANG_DSV41_ENABLE_RAM_MISS_COPY_ENGINE"
# The spellings sglang's EnvBool reads as true (python/sglang/srt/environ.py).
_TRUE = ("true", "1", "yes", "y")
# nsys scratch and reports go here, never to divix01's root volume (~88% full): a full capture exhausts /tmp there
# and the server dies mid-run with a ~361 KB report (CLAUDE.md, DSV41_REFERENCE.md section 22).
NSYS_TMPDIR = "/mnt/nvme1/nsys-tmp"
REPORT_ROOT = "/mnt/nvme1/"
# The RTX 5090 is GB202; this set carries the PCIe read (RX) and write (TX) throughput rows.
GPU_METRICS_SET = "gb20x"
# divix01's driver keeps GPU counters admin-only (RmProfilingAdminOnly=1), so GPU metrics come from a second,
# metrics-only nsys session run as root through this NOPASSWD sudo wrapper (`exec nsys "$@"`).
NSYS_SUDO_WRAPPER = "/usr/local/sbin/nsys-profile"
_GPU_METRICS_UNSUPPORTED = "None of the installed GPUs are supported"
_SUDO_REFUSED = ("a password is required", "not allowed to execute", "command not found", "No such file")
GPU_METRICS_FIX = (
    f"GPU metrics need `sudo -n {NSYS_SUDO_WRAPPER}` (a NOPASSWD rule for a wrapper that execs nsys) or the driver "
    "option NVreg_RestrictProfilingToAdminUsers=0 (/etc/modprobe.d, dracut -f, reboot). "
    "Or set NSYS_GPU_METRICS=0 to trace without PCIe RX/TX"
)


def gpu_metrics_args(requested: str | None) -> list[str]:
    """`nsys start` options for the root metrics-only session; NSYS_GPU_METRICS is 1 (the default) or 0."""
    value = (requested or "1").strip()
    if value == "0":
        return []
    if value != "1":
        raise ValueError(f"NSYS_GPU_METRICS={requested!r}: must be 0 or 1")
    # sudo's env_reset drops CUDA_VISIBLE_DEVICES, so select every GPU; divix01 has one.
    return [
        "--sample=none",
        "--cpuctxsw=none",
        "--gpu-metrics-devices=all",
        f"--gpu-metrics-set={GPU_METRICS_SET}",
        "--force-overwrite=true",
    ]


def gpu_metrics_unavailable(devices_help: str) -> str | None:
    """Why GPU metrics cannot be collected, from `sudo -n <wrapper> profile --gpu-metrics-devices=help`, or None."""
    if _GPU_METRICS_UNSUPPORTED in devices_help or any(s in devices_help for s in _SUDO_REFUSED):
        return f"nsys cannot sample GPU metrics (PCIe RX/TX): {devices_help.strip()} -- {GPU_METRICS_FIX}"
    if "--gpu-metrics-devices values are" not in devices_help:
        return f"unrecognised nsys GPU metrics output: {devices_help.strip()!r} -- {GPU_METRICS_FIX}"
    return None


def copy_engine_on(env: dict[str, str]) -> bool:
    return env.get(COPY_ENGINE_ENV, "").strip().lower() in _TRUE


def graph_trace_mode(requested: str | None, env: dict[str, str]) -> str:
    """The --cuda-graph-trace value for this arm; raises ValueError on an unknown or refused mode.

    ``requested`` is NSYS_CUDA_GRAPH_TRACE (None or empty means the default, graph); ``env`` is the arm's full
    resolved server environment, so a copy engine turned on by default is refused as well as one turned on by an
    override.
    """
    mode = (requested or "graph").strip()
    if mode not in GRAPH_TRACE_MODES:
        raise ValueError(f"NSYS_CUDA_GRAPH_TRACE={requested!r}: must be one of {GRAPH_TRACE_MODES}")
    if mode == "graph" and copy_engine_on(env):
        raise ValueError(
            f"NSYS_CUDA_GRAPH_TRACE=graph with {COPY_ENGINE_ENV}={env[COPY_ENGINE_ENV]}: graph-mode tracing deadlocks "
            "the copy engine's copy wait (LEASE_PROTOCOL.md 7.6); use NSYS_CUDA_GRAPH_TRACE=node"
        )
    return mode


def check_report_dir(path: str) -> str:
    """Refuse a report directory off /mnt/nvme1 (a relative path, /tmp, or the root volume)."""
    if not path.startswith(REPORT_ROOT) or "/../" in path + "/":
        raise ValueError(f"NSYS_OUT_DIR={path!r}: nsys reports must stay under {REPORT_ROOT}")
    return path
