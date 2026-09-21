"""The launch of one benchmark arm: environment, Engine arguments, and the checks that refuse a wrong one.

Two arms share every environment variable and Engine argument except SGLANG_DSV41_ENABLE_RAM_MISS_LEASES,
which Exl3RamMissService reads once at start, so each arm is its own process.

The phase3a env.sh sets SGLANG_MOE_EXPERT_GRAPH_GATHER=0 and decodes eagerly. Lease mode lives in the in-graph
RAM-miss path, so that recipe as it stands makes both arms identical by construction. recipe_env applies the
phase3b env-full.sh override (GRAPH_GATHER=1) and check_recipe refuses anything else.

Nothing here imports sglang at module level: the import-path assertion must run before anything can pick up
the wrong tree.
"""

from __future__ import annotations

import contextlib
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import msgspec

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DSV41 = REPO_ROOT / "scripts" / "dsv41"

LEASE_ENV = "SGLANG_DSV41_ENABLE_RAM_MISS_LEASES"
GRAPH_GATHER_ENV = "SGLANG_MOE_EXPERT_GRAPH_GATHER"
TRACE_ENV = "SGLANG_DSV41_EXPERT_TRACE_PATH"
ARMS = ("lease_off", "lease_on")
# Differs per run, not per arm; kept out of the arm diff.
PER_RUN_ENV = frozenset({TRACE_ENV})

NVFP4_WORK = "/data/models/slang/nvfp4-work"
DEFAULT_GPU_RUN = f"{NVFP4_WORK}/cc-expert-prediction/analysis/dsv41-phase3b/gpu-run.sh"
DEFAULT_SESSIONS = "/mnt/nvme2/nvfp4-work/benchmarks/full/sessions.jsonl"
# run-step.sh puts this on PYTHONPATH after the tree under test (a pyarrow stand-in the venv lacks).
PYARROW_SHIM = f"{NVFP4_WORK}/cc-expert-prediction/analysis/dsv41-phase1/pyarrow-shim"


class ImportPathError(RuntimeError):
    pass


class RecipeError(ValueError):
    pass


class Paths(msgspec.Struct, frozen=True):
    # Paths as on divix01: phase3a env.sh, and $FULL from phase3b prof-graph.sh for the Engine's model_path.
    model: str = f"{NVFP4_WORK}/cc-expert-prediction/dsv41-full40"
    engram_table_dir: str = "/mnt/nvme2/DeepSeek-V4.1-Flash"
    expert_dir: str = "/mnt/nvme2/DeepSeek-V4.1-Flash-EXL3-3.0bpw"
    exl3_src: str = f"{NVFP4_WORK}/exllamav3"
    exl3_build_dir: str = f"{NVFP4_WORK}/cc-expert-prediction/exl3-build"
    sessions: str = DEFAULT_SESSIONS
    gpu_run: str = DEFAULT_GPU_RUN
    # Colon-separated; empty means no mirror, which is what the phase3a/3b recipe runs.
    mirror_dirs: str = ""
    cuda_home: str = "/usr/local/cuda-13.2"

    def required(self) -> dict[str, str]:
        out = {
            "model": self.model,
            "engram_table_dir": self.engram_table_dir,
            "expert_dir": self.expert_dir,
            "exl3_src": self.exl3_src,
            "exl3_build_dir": self.exl3_build_dir,
            "sessions": self.sessions,
            "gpu_run": self.gpu_run,
            "cuda_home": self.cuda_home,
            "interpreter": sys.executable,
        }
        for i, mirror in enumerate(d for d in self.mirror_dirs.split(":") if d):
            out[f"mirror_{i}"] = mirror
        return out


class Resources(msgspec.Struct, frozen=True):
    pinned_host_mb: int = 71680
    # 16384 left no GPU memory for the KV cache on the 32 GB card (phase3a smoke-attempt1-hot16384.log); env.sh uses 14336.
    hot_gpu_mb: int = 14336
    # The graph runs that worked (phase3b prof-graph.sh, step8-chain.sh) used 0.80.
    mem_fraction_static: float = 0.80
    chunked_prefill_size: int = 512
    context_length: int = 4096


def recipe_env(*, paths: Paths, res: Resources) -> dict[str, str]:
    env = {
        "CUDA_HOME": paths.cuda_home,
        "OMP_NUM_THREADS": "16",
        "MKL_NUM_THREADS": "16",
        "PYTHONDONTWRITEBYTECODE": "1",
        "SGLANG_EXL3_SRC": paths.exl3_src,
        "SGLANG_EXL3_BUILD_DIR": paths.exl3_build_dir,
        "SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK": "1",
        "SGLANG_DSV41_ENGRAM_TABLE_DIR": paths.engram_table_dir,
        "SGLANG_DSV41_TORCH_PREFILL_INDEXER": "1",
        "SGLANG_DSV41_ENGRAM_RAM_GIB": "5",
        "SGLANG_DSV41_EXPERT_STREAM": "1",
        "SGLANG_DSV41_EXPERT_DIR": paths.expert_dir,
        "SGLANG_MOE_EXPERT_ROW_SOURCE": "shards",
        "SGLANG_MOE_EXPERT_FILE_READER": "uring_direct",
        "SGLANG_MOE_PINNED_HOST_MB": str(res.pinned_host_mb),
        "SGLANG_MOE_HOT_GPU_MB": str(res.hot_gpu_mb),
        "SGLANG_MOE_HOT_DYNAMIC": "1",
        "SGLANG_MOE_HOT_UPDATE_PREFILL_TOKENS": "256",
        "SGLANG_MOE_HOT_UPDATE_DECODE_FORWARDS": "32",
        "SGLANG_MOE_HOT_MIN_RESIDENCE_FORWARDS": "8",
        "SGLANG_MOE_HOT_ASYNC_PROMOTIONS": "0",
        "SGLANG_MOE_HOT_LOG_INTERVAL": "64",
        "SGLANG_MOE_GPU_RESIDENCY_UPDATE": "0",
        "SGLANG_MOE_EXPERT_DOORBELL": "0",
        "SGLANG_MOE_PREFETCH_MAX_CANDIDATES": "0",
        GRAPH_GATHER_ENV: "1",
        "SGLANG_DSV41_RAM_MISS_TIMEOUT_MS": "2000",
        "SGLANG_DSV41_ENABLE_EXPERT_PREFETCH": "0",
    }
    if paths.mirror_dirs:
        env["SGLANG_MOE_EXPERT_MIRROR_DIRS"] = paths.mirror_dirs
    return env


def arm_env(arm: str, *, paths: Paths, res: Resources) -> dict[str, str]:
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS}")
    env = recipe_env(paths=paths, res=res)
    # Always explicit, so the run records the switch even for the off arm.
    env[LEASE_ENV] = "1" if arm == "lease_on" else "0"
    return env


def _import_trace_corpus():
    if str(SCRIPTS_DSV41) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_DSV41))
    import trace_corpus

    return trace_corpus


def import_provenance():
    _import_trace_corpus()
    import provenance

    return provenance


def engine_kwargs(*, paths: Paths, res: Resources) -> dict:
    """trace_corpus.engine_kwargs(graphs=True): the one graph shape the EXL3 gate accepts, identical in both arms."""
    args = SimpleNamespace(
        model=paths.model,
        mem_fraction_static=res.mem_fraction_static,
        chunked_prefill_size=res.chunked_prefill_size,
        graphs=True,
        dspark=None,
    )
    kwargs = _import_trace_corpus().engine_kwargs(args)
    kwargs["context_length"] = res.context_length
    return kwargs


def env_diff(
    a: dict, b: dict, *, a_name: str = "lease_off", b_name: str = "lease_on"
) -> dict:
    out = {}
    for name in sorted(set(a) | set(b)):
        if name not in PER_RUN_ENV and a.get(name) != b.get(name):
            out[name] = {a_name: a.get(name), b_name: b.get(name)}
    return out


def check_recipe(
    *, env: dict, kwargs: dict, concurrency: int, allow_eager_batches: bool
) -> None:
    """Refuse launches the EXL3 gate lets through but whose lease switch cannot change anything."""
    if env.get(GRAPH_GATHER_ENV) != "1":
        raise RecipeError(
            f"{GRAPH_GATHER_ENV}={env.get(GRAPH_GATHER_ENV)!r}: lease mode lives in the in-graph RAM-miss path, which "
            "graph gather feeds; with it off both arms decode eagerly and lease-on == lease-off by construction"
        )
    if kwargs.get("cuda_graph_backend_decode") != "breakable" or kwargs.get(
        "disable_cuda_graph"
    ):
        raise RecipeError(
            "decode must be a breakable CUDA graph; eager decode never runs the lease path"
        )
    if env.get(LEASE_ENV) not in ("0", "1"):
        raise RecipeError(f"{LEASE_ENV} must be 0 or 1, got {env.get(LEASE_ENV)!r}")
    if concurrency > 1 and not allow_eager_batches:
        raise RecipeError(
            f"concurrency {concurrency}: decode graphs are captured at batch size 1 only (EXL3 gate), so a batch of two "
            "or more runs eagerly and skips the in-graph lease path; pass --allow-eager-batches to measure it anyway"
        )


@contextlib.contextmanager
def patched_environ(env: dict):
    saved = dict(os.environ)
    os.environ.update(env)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(saved)


def assert_sglang_from_repo(repo_root: Path = REPO_ROOT) -> str:
    """Fail unless sglang imports from <repo_root>/python; the venv's default resolves to an unrelated tree."""
    import sglang

    got = Path(sglang.__file__).resolve()
    want = (Path(repo_root) / "python" / "sglang").resolve()
    if want not in got.parents:
        raise ImportPathError(
            f"sglang imported from {got}, not from {want}: this run would exercise the wrong tree. "
            f"Set PYTHONPATH={Path(repo_root) / 'python'} (PYTHONPATH is {os.environ.get('PYTHONPATH')!r})"
        )
    return str(got)


def check_paths(paths: Paths) -> dict[str, dict]:
    return {
        label: {"path": p, "exists": os.path.exists(p)}
        for label, p in paths.required().items()
    }


def _gate_namespace(*, kwargs: dict, model: str) -> SimpleNamespace:
    from sglang.srt.model_executor.cuda_graph_config import CudaGraphConfig

    def phase(name: str) -> dict:
        out = {"backend": kwargs.get(f"cuda_graph_backend_{name}")}
        out["max_bs"] = kwargs.get(f"cuda_graph_max_bs_{name}")
        out["bs"] = kwargs.get(f"cuda_graph_bs_{name}")
        return {k: v for k, v in out.items() if v is not None}

    graph = CudaGraphConfig.from_dict(
        {"decode": phase("decode"), "prefill": phase("prefill")}
    )
    return SimpleNamespace(
        model_path=model,
        quantization=None,
        ple_offload_embedding=False,
        cpu_offload_gb=0,
        offload_group_size=0,
        ple_offload_backend=None,
        moe_runner_backend="auto",
        tp_size=kwargs["tp_size"],
        ep_size=1,
        moe_a2a_backend="none",
        disable_overlap_schedule=False,
        enable_two_batch_overlap=False,
        enable_single_batch_overlap=False,
        max_running_requests=kwargs["max_running_requests"],
        expert_distribution_recorder_mode=kwargs["expert_distribution_recorder_mode"],
        enable_waterfill=False,
        enable_eplb=False,
        moe_offload_preset="off",
        speculative_algorithm=kwargs.get("speculative_algorithm"),
        cuda_graph_config=graph,
    )


def check_gate(*, env: dict, kwargs: dict, model: str) -> dict:
    """Run handle_offload_compatibility (which dispatches to the EXL3 requirements) on this launch, loading nothing.

    Raises the gate's own ValueError when it refuses.
    """
    from sglang.srt.arg_groups import memory_hook
    from sglang.srt.arg_groups.expert_stream_requirements import (
        expert_stream_requirements_for,
    )
    from sglang.srt.environ import envs

    cfg = _gate_namespace(kwargs=kwargs, model=model)
    with patched_environ(env):
        before = dict(os.environ)
        original_view = memory_hook.resolving_view
        # A plain namespace stands in for the resolving view, as in test_expert_stream_requirements_exl3.py.
        memory_hook.resolving_view = lambda a: a
        try:
            requirements = expert_stream_requirements_for(cfg, cfg)
            # Without a readable <model>/config.json naming exl3 the gate silently applies the NVFP4 rules instead.
            if requirements.label != "EXL3":
                raise RecipeError(
                    f"the gate resolved {requirements.label!r} requirements for {model}, not EXL3"
                )
            memory_hook.handle_offload_compatibility(cfg)
        finally:
            memory_hook.resolving_view = original_view
        after = dict(os.environ)
        sglang_env = [n for n in sorted(env) if n.startswith("SGLANG_")]
        known = [n for n in sglang_env if n in vars(type(envs))]
        return {
            "requirements": requirements.label,
            "accepted": True,
            "graph_gather_host_source": requirements.graph_gather_host_source,
            "cuda_graph_config": cfg.cuda_graph_config.to_dict(),
            "resolved_env": {n: getattr(envs, n).get() for n in known},
            "unknown_sglang_env": [n for n in sglang_env if n not in known],
            "env_edited_by_gate": {
                k: {"before": before.get(k), "after": after.get(k)}
                for k in sorted(set(before) | set(after))
                if before.get(k) != after.get(k)
            },
        }
