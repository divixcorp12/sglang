"""A fake EXL3 checkpoint plus per-layer pinned-slab stand-ins for the option C CPU tests."""

from __future__ import annotations

import contextlib
import dataclasses
import os
import pathlib
import shutil
import signal
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import torch

from sglang.srt.layers.moe.exl3_expert_format import EXL3_STREAMED_NAMES, Exl3ExpertFormat
from sglang.srt.layers.moe import exl3_row_image as ri
from sglang.srt.layers.moe.exl3_expert_layout import build_exl3_expert_layout
from sglang.srt.layers.moe.exl3_shard_row_source import Exl3ShardRowSource
from sglang.srt.layers.moe.expert_host_tier import allocate_host_slab
from sglang.test.dsv41_fake_exl3 import write_fake_exl3


@dataclass
class RamMissSetup:
    layout: object
    fmt: Exl3ExpertFormat
    specs: dict
    slabs: dict
    tables: object
    roots: tuple = ()  # the row-image roots, one per mirror weight

    def reference(self, layer: int, experts: list[int]) -> dict[str, torch.Tensor]:
        """Exl3ShardRowSource's split of ``experts`` of ``layer`` (the byte oracle)."""
        out = {
            name: torch.empty((len(experts),) + self.specs[name].row_shape, dtype=self.specs[name].dtype)
            for name in EXL3_STREAMED_NAMES
        }
        Exl3ShardRowSource.for_layer(self.layout, layer, self.fmt.segment_map(), direct=False).read(
            torch.tensor(experts), out
        )
        return out


# Fake-expert dimensions whose six streamed names' slab rows are all multiples of 512 bytes, as row images need
# (dsv41's are; write_fake_exl3's defaults give 256-byte w2_suh/w2_svh rows): trellis rows of 49152 (w13) and 24576
# (w2) bytes and scale rows of 512 and 1024, an image of 76800 bytes (18.75 pages, so the last part ends inside a
# page) and a row stride of 77824.
ROW_IMAGE_DIM = 256


def write_row_images(layout, segments, source_root: str, roots: Sequence[str], layer_ids: Optional[Iterable[int]] = None):
    """A reference build of row images on every root, from exl3_row_image's primitives only (the converter,
    scripts/dsv41/build_row_images.py, is what production uses; tests must not depend on it): each row's image by
    ``image_of_row`` of its checkpoint bytes, zero-padded to ``row_stride``, then the manifest. Returns the layout."""
    image_layout = ri.row_image_layout(segments)
    layer_ids = list(range(layout.num_layers)) if layer_ids is None else list(layer_ids)
    digests, files = {}, {}
    for layer in layer_ids:
        out = bytearray()
        digests[layer] = []
        for expert in range(layout.num_experts):
            record = layout.records[(layer, expert)]
            with open(record.path, "rb") as f:
                f.seek(record.file_offset)
                raw = f.read(record.nbytes)
            image = ri.image_of_row(image_layout, raw)
            digests[layer].append(ri.row_digest(image))
            out += image + bytes(image_layout.row_stride - image_layout.image_bytes)
        files[layer] = bytes(out)
    fingerprint = ri.source_fingerprint(layout, source_root)
    for root in roots:
        os.makedirs(ri.row_image_dir(root), exist_ok=True)
        for layer, data in files.items():
            with open(os.path.join(ri.row_image_dir(root), ri.layer_file_name(layer)), "wb") as f:
                f.write(data)
        ri.write_manifest(root, ri.manifest_json(image_layout, fingerprint, digests))
    return image_layout


def _takes_o_direct(path: str) -> bool:
    try:
        os.close(os.open(path, os.O_RDONLY | os.O_DIRECT))
        return True
    except OSError:
        return False


def require_o_direct(path: str) -> None:
    """The reader reads row images with O_DIRECT only (plan 2026-09-29-hotpath-zero-overhead): a test filesystem that
    refuses it (tmpfs before Linux 6.6, some overlays) must fail loudly here, not run a buffered read that production
    can no longer take."""
    if not _takes_o_direct(path):
        raise RuntimeError(f"{path}: the RAM-miss tests need a filesystem that takes O_DIRECT (tmpfs needs Linux 6.6+)")


def service_row_images(source) -> contextlib.ExitStack:
    """What an option C service needs to read row images, for a test that builds its tiers over the fake checkpoint
    at ``source`` (written by write_fake_exl3 with ``ROW_IMAGE_DIM`` multiples as ``hidden``/``inter``): a mirror root
    beside it holding a copy of its shards and a reference build of its row images, and the env the service reads
    them with. Returns the entered env overrides; close the stack when the test is done.

    The formats must be built inside the stack with ``source_root=source`` and no ``direct`` (so
    SGLANG_MOE_EXPERT_FILE_READER=uring_direct decides it, as in production). Row images and lease mode need no env:
    both are unconditional (plan 2026-09-29-hotpath-zero-overhead Task 7). Modelled on
    test_exl3_prefill_fills_service._build."""
    from sglang.srt.environ import envs

    source = pathlib.Path(source)
    root = source.parent / f"{source.name}_images_mirror"
    if not root.exists():  # built once per checkpoint: a harness may start the service more than once over it
        shutil.copytree(source, root)
        layout = build_exl3_expert_layout(str(source))
        segments = Exl3ExpertFormat(layout, 0, direct=False, source_root=str(source)).segment_map()
        write_row_images(layout, segments, str(source), [str(root)])
    require_o_direct(os.path.join(ri.row_image_dir(str(root)), ri.layer_file_name(0)))
    stack = contextlib.ExitStack()
    for env, value in (
        (envs.SGLANG_MOE_EXPERT_MIRROR_DIRS, str(root)),
        (envs.SGLANG_MOE_EXPERT_FILE_READER, "uring_direct"),
    ):
        stack.enter_context(env.override(value))
    return stack


def image_tables(layout, segments, slabs_by_layer, source_root, mirror_weights=None):
    """``exl3_ram_miss_tables`` over row images of the fake checkpoint at ``source_root``, built by
    ``write_row_images`` for the layers in ``slabs_by_layer``: one image root beside the checkpoint per mirror weight
    (``<source_root>_images<i>``; one root, weight 1, when ``mirror_weights`` is None), split as the weights say. The
    image files must take O_DIRECT (``require_o_direct``). Returns the tables and the roots. For harnesses that
    allocate their own slabs (the manual GPU suites); ``ram_miss_setup`` builds its tables with it too."""
    from sglang.srt.layers.moe.exl3_ram_miss import exl3_ram_miss_tables
    from sglang.srt.layers.moe.exl3_read_split import StaticSplitPolicy
    from sglang.srt.layers.moe.exl3_row_image import open_row_images

    weights = (1.0,) if mirror_weights is None else tuple(mirror_weights)
    source = str(source_root).rstrip("/")
    roots = tuple(f"{source}_images{i}" for i in range(len(weights)))
    layers = sorted(slabs_by_layer)
    write_row_images(layout, segments, source, roots, layers)
    images = open_row_images(roots, layout, segments, source, layers)
    tables = exl3_ram_miss_tables(
        layout, segments, slabs_by_layer, roots=roots, policy=StaticSplitPolicy(weights), row_images=images
    )
    require_o_direct(tables.paths[0])
    return tables, roots


@contextlib.contextmanager
def paused(host, timeout_s: float = 10.0):
    """The tier's owner for a block: the service thread paused (the class's own pause, not an instance attribute a
    test may have wrapped to count pauses). Not reentrant, like the pause."""
    type(host).pause(host, timeout_s)
    try:
        yield host
    finally:
        type(host).resume(host)


def same_bytes(a: torch.Tensor, b: torch.Tensor) -> bool:
    return torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


def ram_miss_setup(
    tmp_path,
    *,
    capacity: int = 3,
    layers: int = 2,
    experts: int = 6,
    mirror_weights=None,
    hidden=None,
    inter=None,
    row_images: bool = True,
) -> RamMissSetup:
    """``mirror_weights``: one weight per mirror root; image roots are made beside ``tmp_path`` and the
    tables split every row across them (``parts == len(mirror_weights)``). ``hidden``/``inter`` size the fake
    experts (write_fake_exl3's defaults when None): larger ones give rows many pages long.

    The tables read row images, the reader's only tables (plan 2026-09-29-hotpath-zero-overhead D4). The fake experts
    default to ``ROW_IMAGE_DIM`` (the images need 512-byte slab rows), each mirror root (one root, weight 1, when
    ``mirror_weights`` is None) holds images built by ``write_row_images`` and no shard copies, and ``reference`` is
    still the checkpoint read by Exl3ShardRowSource: the oracle every row the reader lands is compared against. The
    image files must take O_DIRECT (``require_o_direct``): production reads them with nothing else. ``row_images``
    stays only so existing ``row_images=True`` callers keep working; ``False`` (shard tables) is refused, since the
    packed reader that read them is gone."""
    if not row_images:
        raise ValueError(
            "ram_miss_setup(row_images=False): shard tables were read only by the deleted packed path; the reader "
            "reads row images only (plan 2026-09-29-hotpath-zero-overhead D4)"
        )
    hidden = ROW_IMAGE_DIM if hidden is None else hidden
    inter = ROW_IMAGE_DIM if inter is None else inter
    dims = {k: v for k, v in (("hidden", hidden), ("inter", inter)) if v is not None}
    write_fake_exl3(str(tmp_path), num_layers=layers, num_experts=experts, **dims)
    layout = build_exl3_expert_layout(str(tmp_path))
    fmt = Exl3ExpertFormat(layout, 0, direct=False)
    specs = {spec.name: spec for spec in fmt.tensor_specs(None)}
    slabs = {
        layer: {
            name: allocate_host_slab(capacity, specs[name].row_shape, specs[name].dtype, register=False)
            for name in EXL3_STREAMED_NAMES
        }
        for layer in range(layers)
    }
    tables, roots = image_tables(layout, fmt.segment_map(), slabs, tmp_path, mirror_weights)
    return RamMissSetup(layout, fmt, specs, slabs, tables, roots)


class DirectUpdaterStandIn:
    """The DIRECT updater surface the RAM-miss service reads at attach: ``insert_direct``, the hot bank
    ``slot_to_expert`` (``[layers, capacity + 1]``, the last column the dump slot), ``layer_ids``, ``caches`` and, for
    CPU experts, ``enable_miss_order``'s keys. A test sets a layer's hot set by writing its bank row."""

    insert_direct = True

    def __init__(self, layers: int, capacity: int, experts: int, device="cpu"):
        from types import SimpleNamespace

        self.device = torch.device(device)
        self.layer_ids = list(range(layers))
        self.slot_to_expert = torch.full((layers, capacity + 1), -1, dtype=torch.int64, device=self.device)
        self.caches = [SimpleNamespace(capacity=capacity) for _ in range(layers)]
        self.experts = experts
        self.miss_keys = None

    def enable_miss_order(self) -> None:
        self.miss_keys = torch.zeros((len(self.layer_ids), self.experts), dtype=torch.int64, device=self.device)


def attached_host(setup: "RamMissSetup", page: torch.Tensor, *, k: int = 1, slot_map=None, **host_kw):
    """An ExpertStreamHost over ``setup``'s tables with ``k`` staging slots reserved per row (the service reserves them
    at start), and ``slot_map`` (a fresh -1 map when None)."""
    from sglang.kernels.ops.moe.expert_stream_transport import ExpertStreamHost

    layers, experts = setup.tables.starts.shape[:2]
    if slot_map is None:
        slot_map = torch.full((layers, experts), -1, dtype=torch.int32)
    host = ExpertStreamHost(setup.tables, page=page, slot_map=slot_map, **host_kw)
    host.reserve_staging(k)
    return host


def fake_cpu_layer(hidden: int = 8):
    """A layer for the instr build's fake CPU expert kernel (ExpertStreamHost.test_kernel_address), which reads only
    its hidden size. Its capacity covers any test tier's row: the host refuses a layer smaller than the row."""
    from sglang.srt.layers.moe.cpu_experts.trait import CpuExpertLayerSpec

    return CpuExpertLayerSpec(capacity=1 << 20, hidden=hidden, intermediate=0, act_limit=0.0, slabs=(), params=b"")


def draft_cpu_host(mode: str, areas, kernel: int, *, cores, threads: int, keep_warm_us: int,
                   fatal_wait_s: float, tmp_path, variant: str = "instr", ns_per_expert: int = 0):
    """A DSpark draft channel server, unstarted, with DraftCpuHost's interface, in one of the two shapes the shared CPU
    team allows (plan 2026-10-06 Task 11):
      draft_only  a draft-only engine (DraftCpuHost), as a launch with the target's CPU experts off builds;
      shared      group 0's CPU expert engine of an ExpertStreamHost whose CPU experts are on (the fake kernel at
                  `kernel`), the draft attached as its second job source; .expert_host is that host and .sim its
                  ChainSim, so a test can post target jobs to the same team. stop() detaches the draft; the caller
                  stops .expert_host.
    """
    from sglang.kernels.ops.moe.dspark_draft_cpu import DraftCpuHost
    from sglang.kernels.ops.moe.expert_lease_block import wire_layout
    from sglang.kernels.ops.moe.expert_stream_transport import new_page
    from sglang.test.dsv41_chain_sim import ChainSim

    if mode == "draft_only":
        return DraftCpuHost(areas, cores=cores, threads=threads, keep_warm_us=keep_warm_us,
                            fatal_wait_s=fatal_wait_s, variant=variant)
    if mode != "shared":
        raise ValueError(mode)
    s = ram_miss_setup(tmp_path, capacity=7, mirror_weights=(1.0, 1.0), hidden=256, inter=512)
    page = new_page(pin=False, wire=wire_layout(8))
    host = attached_host(s, page, k=3)
    host.enable_copy_engine(-1)
    row = 1
    dst = {n: torch.zeros((6,) + tuple(t.shape[1:]), dtype=t.dtype) for n, t in s.slabs[row].items()}
    table = torch.tensor([[t.data_ptr(), dst[n].data_ptr(), t[0].numel() * t.element_size()] for n, t in s.slabs[row].items()],
                         dtype=torch.int64)
    host.set_copy_table(row, table, 6)
    host.arm_copy_engine()
    x_rows = torch.zeros((2, 16), dtype=torch.uint8)
    out_rows = torch.zeros((2, 2, 8), dtype=torch.float32)
    split = [0] * (host.wire.lanes + 1)
    split[1] = 1  # one eligible hit lane: the CPU's
    host.enable_cpu_experts(kernel, split, cores, x_rows, out_rows, threads=threads, keep_warm_us=keep_warm_us)
    # The target's layer holds fewer slots than a draft stage's (>= 1 << 20), which the fake kernel records per call: a test
    # tells a target forward from a draft one by it.
    host.set_cpu_layer(row, dataclasses.replace(fake_cpu_layer(8), capacity=1024))
    draft = host.draft_source(areas, fatal_wait_s=fatal_wait_s, group=0)
    draft.expert_host, draft.sim, draft.keep = host, ChainSim(host, page, s.slabs), (s, dst, x_rows, out_rows)
    return draft


# The host build and lane count ``run_host_script``'s child constructs; the parent warms exactly these.
HOST_SCRIPT_VARIANT = "instr"
HOST_SCRIPT_LANES = 8


def warm_host_modules(variant: Optional[str] = None, *, lanes: int = 8, nodes: int = 1) -> None:
    """Load, in this process, the JIT host modules a child interpreter will load, so the child finds them built.

    A cold module costs 50-100 s to compile, serialized across ``pytest -n`` workers by the JIT build lock; a child
    under a ``timeout`` would spend it on the compiler (nine ``run_host_script`` callers timed out at 60 s on
    2026-10-04). The parent has no timeout, so it waits on the lock. Load only: no host is built.

    ``variant``/``lanes``/``nodes`` name the module the child constructs its ``ExpertStreamHost`` with (None: the
    child's default build). The child has no conftest, so its ``host_layout()`` also loads the default build at one
    node, whatever the parent's autouse fixture says: that one is warmed too."""
    from sglang.kernels.ops.moe import expert_stream_transport as ops

    conftest_default, ops._DEFAULT_VARIANT = ops._DEFAULT_VARIANT, None
    try:
        child_default = ops.host_variant()
    finally:
        ops._DEFAULT_VARIANT = conftest_default
    ops._host_module("exl3", child_default, 8, 1)  # what host_layout() loads: its default lanes, one node
    ops._host_module("exl3", variant or child_default, lanes, nodes)
    # A child that serves a miss reads through this module, which the host modules do not load.
    from sglang.kernels.ops.io import uring_file_reader

    uring_file_reader._uring_file_reader_type()


# First line of a child that is meant to abort: every service failure is a deliberate ``std::abort``, and on divix01
# ``core_pattern`` pipes to systemd-coredump with ``ulimit -c unlimited``, so each such child spent longer than its
# 60 s timeout writing a ~440 MB core under ``pytest -n 8`` (2026-10-04). The limit is read at crash time.
NO_CORE_DUMP = "import resource; resource.setrlimit(resource.RLIMIT_CORE, (0, 0))\n"


def spawn_child(
    script: str, *args, timeout_s: float, variant: Optional[str] = None, lanes: int = 8, nodes: int = 1, env=None
) -> subprocess.CompletedProcess:
    """Run ``script`` in a fresh interpreter with ``args`` as its ``sys.argv[1:]``: the one way a test starts a child
    that builds expert-stream hosts. The JIT modules it loads are warmed here first (``warm_host_modules``, arguments
    as there), so the timeout measures the child; its core dumps are off (``NO_CORE_DUMP``), so a child that is meant
    to abort, or dies when its test is red, is not held up writing one. The pytest process's own limit is untouched.

    ``env`` must not change what picks the child's default build (``SGLANG_DSV41_EXPERT_TRACE_PATH``,
    ``SGLANG_TEST_DSV41_RAM_MISS_FAULT``): the warm-up reads this process's."""
    warm_host_modules(variant, lanes=lanes, nodes=nodes)
    return subprocess.run(
        [sys.executable, "-c", NO_CORE_DUMP + script, *map(str, args)],
        capture_output=True,
        text=True,
        timeout=timeout_s,
        env=env,
    )


_HOST_SCRIPT_HEAD = f"""
import pathlib, sys, time
import torch
from sglang.kernels.ops.moe.expert_lease_block import wire_layout
from sglang.kernels.ops.moe.expert_stream_transport import ExpertStreamHost, new_hot_page, new_page, page_word
from sglang.test.dsv41_chain_sim import ChainSim
from sglang.test.dsv41_ram_miss_fixtures import ram_miss_setup
s = ram_miss_setup(pathlib.Path(sys.argv[1]), capacity=int(sys.argv[2]))
page, hot_page = new_page(pin=False, wire=wire_layout({HOST_SCRIPT_LANES})), new_hot_page(6, pin=False)
host = ExpertStreamHost(s.tables, page=page, slot_map=torch.full((2, 6), -1, dtype=torch.int32), variant="{HOST_SCRIPT_VARIANT}"%s)
host.reserve_staging(int(sys.argv[3]))
sim = ChainSim(host, page, s.slabs)
"""


def run_host_script(
    tmp_path, body: str, *, capacity: int = 3, staging: int = 1, host_args: str = "", timeout_s: float = 60
) -> subprocess.CompletedProcess:
    """Run ``body`` in a fresh interpreter over an instrumented host (``s``, ``page``, ``host``, ``sim`` in scope),
    each of its two rows attached with ``staging`` staging slots; ``host_args`` is extra keyword source for its
    constructor (", hot_page=hot_page" hands it the ``hot_page`` in scope).

    Every service failure is fail-stop (``std::abort``), so a test of one must watch a child process die."""
    return spawn_child(
        _HOST_SCRIPT_HEAD % host_args + textwrap.dedent(body), tmp_path, capacity, staging,
        timeout_s=timeout_s, variant=HOST_SCRIPT_VARIANT, lanes=HOST_SCRIPT_LANES,
    )


def assert_aborted(result: subprocess.CompletedProcess, message: str) -> None:
    """The child died of SIGABRT, printing ``message`` on its FATAL line, and never reached its ``print``."""
    assert result.returncode == -signal.SIGABRT, (result.returncode, result.stderr[-2000:])
    assert "reached" not in result.stdout, result.stdout
    assert "FATAL" in result.stderr and message in result.stderr, result.stderr[-2000:]
