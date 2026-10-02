# DSV4.1 DSpark: hybrid draft experts, a static resident set on the GPU and the rest on the CPU (Implementation Plan)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep only each DSpark draft stage's most-routed experts in VRAM and compute the rest with the CPU expert
kernel, so most of the draft's 6.75 GiB goes to the target's hot cache. Measure whether eager DSpark gets faster.

**Architecture:**
- **Unchanged:** the target model and DSpark itself (eager, D1). Only the draft stages' `FusedMoE` changes.
- **Resident set:** a calibration file lists, per draft stage, the N experts to keep on the GPU. It is made offline
  from a routes probe (`SGLANG_DSPARK_DEBUG_DRAFT_ROUTES_PATH`, already on the branch) by
  `draft_resident_set.py`. `SGLANG_DSV41_DSPARK_DRAFT_RESIDENT_PATH` points the server at it.
- **Weights:** with `SGLANG_DSV41_ENABLE_DSPARK_CPU_EXPERTS=1`, a draft stage loads all experts into host RAM. After
  loading, its resident experts and any fused shared expert are copied to the GPU, and the full slabs register with
  a `CpuExpertPool`.
- **Compute, per stage call:** the routes come to the host (one small D2H). Routes to non-resident experts go to the
  CPU pool on a worker thread, via the kernel's multi-row op `compute_rows`, which releases the GIL
  (`bindings.cpp:161`). Meanwhile the caller runs the resident experts on the GPU with `exl3_moe_accumulate`, then
  adds the CPU result. A call with no CPU routes skips the CPU entirely.
- **Order:** the pieces, then a GPU parity test, then a served A/B on sessions the calibration did not see.

**Tech Stack:**
- SGLang fork (`master`), Python/PyTorch.
- The plain `sglang_exl3_ext` CPU kernel: upstream exllamav3's `exl3_moe_cpu_make_layer` and `exl3_moe_cpu_forward`.
  Under DSpark, `exl3_ext()` builds the plain flavor, because `SGLANG_DSV41_CPU_EXPERTS` is refused with
  speculation. At draft shapes it beats the optimized build (Evidence, below).
- pytest.
- divix01 (RTX 5090, 2-socket Xeon).

**Spec:** `DSV41_REFERENCE.md` §33 (DSpark D1 state), §30 (CPU experts) and §26.2 (the hot cache's value per GiB),
plus the evidence in `docs/superpowers/plans/2026-10-02-dsv41-dspark-cpu-draft.md` under "Results". This plan
supersedes that plan's Tasks 2–7. Its Task 1 (the benchmark) and its routes probe stand as the evidence.

## Evidence (why a hybrid, and why N = 32)

From the superseded plan's Results (divix01, 2026-10-02):
- **Kernel cost:** 0.52 ms per weight pass with the plain build, 4-bit, 12 threads, 6 rows. The kernel reads an
  expert once per two rows routed to it. The optimized `_resid_b128_cpu_v1` build is slower at multiple rows
  (30.9 vs 26.5 ms, independent; 20.8 vs 14.5 ms, shared).
- **Routes probe:** 8 sessions × 128 tokens, 352 draft steps, accept length 3.07.
  - Every draft call is 5 rows.
  - Per-stage mean union: 9.25, 7.59 and 5.61 experts.
- **All-CPU draft:** 15.7 ms per draft step (17.1 ms p90), against ~13–18 ms of hot-cache value. Break-even.
- **Hybrid, held out** (top-N chosen on steps 1–176, scored on 177–352):

| N per stage | VRAM freed | CPU ms per draft step (mean / p90) |
|---|---|---|
| 16 | 5.9 GiB | 3.58 / 5.70 |
| 32 | 5.1 GiB | 1.43 / 3.11 |
| 48 | 4.3 GiB | 0.67 / 1.55 |

N = 32 is the A/B's arm. It is the knee: 16 doubles the CPU time for 0.8 GiB more, and 48 gives back 0.8 GiB to save
0.8 ms. N is a calibration-file parameter, not code, so another value costs one file.

## Global Constraints

- **Branch and worktree:** branch `dsv41-dspark-cpu-draft`, laptop worktree `sglang-nvfp4-dspark-cpu`, push to `origin`
  only. On divix01, run in `/data/models/slang/nvfp4-work/wt-dspark-cpu` at the pushed commit
  (`.claude/rules/divix01-run-protocol.md`), with `PYTHONPATH=$PWD/python`.
- **Git:** no rebase, amend, force-push or `git stash`. Stage files by name. End commit messages with
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- **CPU jobs:** `taskset -c 0-63`, `OMP_NUM_THREADS=8`. The draft's kernel runs on cores `18-29`.
- **GPU jobs:** use `flock /data/models/slang/nvfp4-work/cc-gpu.lock`, after `rowimg-disk.lock` when a job streams
  from NVMe. Production must not be running; never start or stop it without the owner's OK.
- **Test status:** read pytest's own status through `PIPESTATUS[0]`, never a pipeline's.
- **Required reading:** `env-var-conventions` before touching `environ.py`. The names below follow it.
- **Kernel environment:** `EXL3_MOE_CPU_PIN=0` in every process that runs the CPU kernel. The trait's
  `check_environment` refuses otherwise.
- **No bulk reads** of `/mnt/nvme1` or `/mnt/nvme2`.
- **Units:** ms per stage call (one stage, one forward); ms per draft step (3 stage calls); tok/s per session.
- **Stage identity:** a draft stage is its `FusedMoE`'s `layer_id`. The routes probe recorded layer ids 0, 1 and 2.

## Review Focus

1. **A draft MoE call over prefill-sized input** (the startup dummy run, or an extend of hundreds of rows). Its CPU
   share scales with rows and would inflate startup and TTFT. The probe saw only 5-row calls at decode. Task 7
   compares startup time and TTFT between arms.
2. **A resident file that does not match the loaded draft.** Examples: a stage missing from the file, an id at or
   above `n_routed`, or a file from another checkpoint. The launch must refuse, never silently run everything on the
   CPU. Tasks 3 and 5 pin it.
3. **A fused shared expert** (id ≥ `n_routed`). It must always run on the GPU and never reach the kernel. Tasks 4
   and 5 pin it.
4. **The scheduler thread's CPU affinity.** Binding the kernel's worker must never pin the caller. Task 4 pins it.
5. **Routing drift away from the calibration.** The A/B runs on sessions 8–15, which the calibration (sessions 0–7)
   did not see. The runtime logs the share of stage calls that needed the CPU (Task 4), and Task 7 reports it.

---

### Task 1: `CpuExpertPool.compute_rows`: the pool for more than one row

**Files:**
- Modify: `python/sglang/srt/layers/moe/cpu_experts/pool.py`: add `compute_rows` after `compute`, and correct the
  `forward` protocol docstring to `[m, k]`.
- Modify: `python/sglang/srt/layers/moe/cpu_experts/exl3.py`: `Exl3CpuQuantTrait.forward` docstring only.
- Test: `test/registered/unit/kernels/test_cpu_expert_pool.py`

**Interfaces:**
- Produces: `CpuExpertPool.compute_rows(layer: int, host_slots: torch.Tensor, weights: torch.Tensor, x: torch.Tensor, out: torch.Tensor) -> None`.
  - `x` and `out` are `[m, H]`; `host_slots` is int64 `[m, k]` with -1 skipped; `weights` is `[m, k]`.
  - Dtypes are the trait's. It overwrites `out`.
  - `compute` (batch 1) is unchanged.

- [ ] **Step 1: Write the failing tests.** Append to `test/registered/unit/kernels/test_cpu_expert_pool.py`:

```python
class _RowsTrait:
    """A fake EXL3-shaped trait: out[r] = x[r] * sum of row r's weights over valid slots, recorded per call."""

    name = "fake-rows"
    slab_names = ("w13_trellis",)
    act_limit = 10.0
    x_dtype = torch.float16
    weights_dtype = torch.float16
    out_dtype = torch.float32

    def __init__(self):
        self.calls = []

    def check_environment(self):
        pass

    def register_layer(self, slabs, capacity):
        return capacity

    def forward(self, handle, x, slots, weights, out, threads):
        self.calls.append((handle, slots.clone(), threading.get_native_id()))
        valid = (slots >= 0).to(torch.float32)
        out.copy_(x.float() * (weights.float() * valid).sum(-1, keepdim=True))

    def free_layer(self, handle):
        pass


def _rows_pool(trait, capacity=4):
    cores = sorted(os.sched_getaffinity(0))[:2]
    if len(cores) < 2:
        pytest.skip("needs at least 2 cores in the affinity mask")
    slabs = {7: {"w13_trellis": torch.zeros(capacity, 2, dtype=torch.int16)}}
    return CpuExpertPool(trait, slabs, cores=cores, threads=2)


def _on_bound_thread(pool, fn):
    result = {}

    def run():
        pool.bind_current_thread()
        try:
            result["value"] = fn()
        except BaseException as error:  # re-raised on the caller
            result["error"] = error

    thread = threading.Thread(target=run)
    thread.start()
    thread.join()
    if "error" in result:
        raise result["error"]
    return result.get("value")


def test_compute_rows_runs_every_row_and_skips_minus_one():
    trait = _RowsTrait()
    pool = _rows_pool(trait)
    x = torch.arange(6 * 4, dtype=torch.float16).reshape(6, 4)
    slots = torch.tensor([[0, 1, -1]] * 6, dtype=torch.int64)
    weights = torch.full((6, 3), 0.5, dtype=torch.float16)
    out = torch.empty(6, 4, dtype=torch.float32)
    _on_bound_thread(pool, lambda: pool.compute_rows(7, slots, weights, x, out))
    assert torch.equal(out, x.float() * 1.0)  # two valid slots of 0.5 per row
    assert trait.calls[0][1].shape == (6, 3)


@pytest.mark.parametrize(
    "slots_shape,weights_shape,out_rows",
    [((6, 3), (6, 2), 6), ((5, 3), (5, 3), 6), ((6, 3), (6, 3), 5)],
)
def test_compute_rows_refuses_mismatched_shapes(slots_shape, weights_shape, out_rows):
    pool = _rows_pool(_RowsTrait())
    x = torch.zeros(6, 4, dtype=torch.float16)
    slots = torch.zeros(slots_shape, dtype=torch.int64)
    weights = torch.zeros(weights_shape, dtype=torch.float16)
    out = torch.empty(out_rows, 4, dtype=torch.float32)
    with pytest.raises(ValueError):
        _on_bound_thread(pool, lambda: pool.compute_rows(7, slots, weights, x, out))


def test_compute_rows_refuses_a_slot_outside_the_layer():
    pool = _rows_pool(_RowsTrait(), capacity=4)
    x = torch.zeros(2, 4, dtype=torch.float16)
    slots = torch.tensor([[0, 4], [1, 2]], dtype=torch.int64)
    weights = torch.zeros(2, 2, dtype=torch.float16)
    out = torch.empty(2, 4, dtype=torch.float32)
    with pytest.raises(ValueError, match="host slot 4"):
        _on_bound_thread(pool, lambda: pool.compute_rows(7, slots, weights, x, out))


def test_compute_rows_refuses_an_unbound_thread():
    pool = _rows_pool(_RowsTrait())
    x = torch.zeros(1, 4, dtype=torch.float16)
    with pytest.raises(RuntimeError, match="bind_current_thread"):
        pool.compute_rows(
            7,
            torch.zeros(1, 1, dtype=torch.int64),
            torch.zeros(1, 1, dtype=torch.float16),
            x,
            torch.empty(1, 4, dtype=torch.float32),
        )
```

  If the file does not already import `os`, `threading` and `CpuExpertPool`, add them at its top.

- [ ] **Step 2: Run them; they must fail** (on divix01: `os.sched_getaffinity` does not exist on macOS)

```bash
PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 taskset -c 0-63 /data/models/slang/.venv/bin/python -m pytest \
  test/registered/unit/kernels/test_cpu_expert_pool.py -q -p no:randomly -k compute_rows 2>&1 | tail -3; echo "EXIT=${PIPESTATUS[0]}"
```
Expected: `EXIT=1`, failing with `AttributeError: 'CpuExpertPool' object has no attribute 'compute_rows'`.

- [ ] **Step 3: Implement.** In `pool.py`, insert after `compute`:

```python
    def compute_rows(
        self,
        layer: int,
        host_slots: torch.Tensor,
        weights: torch.Tensor,
        x: torch.Tensor,
        out: torch.Tensor,
    ) -> None:
        """Overwrite ``out`` ``[m, H]`` with the routed sums of ``m`` rows.

        ``x`` is ``[m, H]``, ``host_slots`` int64 ``[m, k]`` (-1 skips) and ``weights`` ``[m, k]``. The DSpark
        draft's path (``cpu_experts/draft.py``): the kernel groups the rows' routes by expert, so it reads an expert
        once per two rows that route to it. Must run on a thread that called ``bind_current_thread``.
        """
        if threading.get_native_id() not in self._bound_threads:
            raise RuntimeError(
                "compute_rows() from a thread that has not called bind_current_thread()"
            )
        trait = self.trait
        rows = x.shape[0]
        if (
            x.dim() != 2
            or host_slots.dtype != torch.int64
            or host_slots.dim() != 2
            or host_slots.shape[0] != rows
            or weights.shape != host_slots.shape
            or out.shape != x.shape
        ):
            raise ValueError(
                "x and out must be [m, H] and host_slots int64 [m, k] with weights of the same shape, "
                f"got x {tuple(x.shape)}, host_slots {tuple(host_slots.shape)}, weights {tuple(weights.shape)}, "
                f"out {tuple(out.shape)}"
            )
        if (
            x.dtype != trait.x_dtype
            or weights.dtype != trait.weights_dtype
            or out.dtype != trait.out_dtype
        ):
            raise ValueError(
                f"{trait.name} takes x {trait.x_dtype}, weights {trait.weights_dtype}, out {trait.out_dtype}"
            )
        if layer not in self.capacity:
            raise ValueError(f"layer {layer} is not in the CPU expert pool")
        capacity = self.capacity[layer]
        top = int(host_slots.max()) if host_slots.numel() else -1
        if top >= capacity:
            raise ValueError(
                f"layer {layer}: host slot {top} is outside the tier's {capacity} rows"
            )
        if capacity == 0:
            out.zero_()
            return
        trait.forward(self._handles[layer], x, host_slots, weights, out, self.threads)
```

  Check the names against `compute` before relying on them: `self._bound_threads`, `self._handles`, `self.capacity`
  and `self.threads`. If `compute` uses different attribute names, use those, and record a ruling.

  Also change two docstrings:
  - `CpuExpertQuantTrait.forward`: "(int64 ``[1, k]``)" becomes "(int64 ``[m, k]``; ``compute`` passes one row)".
  - `Exl3CpuQuantTrait.forward` in `exl3.py`: "Overwrite ``out`` ``[m, H]`` with the routed sums of ``slots`` ``[m, k]`` over layer ``handle``."

- [ ] **Step 4: Run the whole file; it must pass**

```bash
PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 taskset -c 0-63 /data/models/slang/.venv/bin/python -m pytest \
  test/registered/unit/kernels/test_cpu_expert_pool.py -q -p no:randomly 2>&1 | tail -3; echo "EXIT=${PIPESTATUS[0]}"
```
Expected: `EXIT=0`.

- [ ] **Step 5: Commit**

```bash
git add python/sglang/srt/layers/moe/cpu_experts/pool.py python/sglang/srt/layers/moe/cpu_experts/exl3.py \
  test/registered/unit/kernels/test_cpu_expert_pool.py
git commit -m "cpu-experts: compute_rows, the pool's multi-row forward for the DSpark draft

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: The resident-set file: format, calibration tool and loader

**Files:**
- Create: `python/sglang/srt/layers/moe/cpu_experts/draft_resident.py`
- Create: `analysis/dsv41-drive/dspark/draft_resident_set.py`
- Test: `test/registered/unit/kernels/test_dspark_draft_resident.py`

**Interfaces:**
- Produces, in `draft_resident.py`:
  - `top_n_resident_set(route_lines: Iterable[str], n: int) -> dict[int, list[int]]`. It reads the routes probe's
    JSON lines (`{"layer": int, "ids": [[int, ...], ...]}`) and returns each layer's `n` most-routed ids, most
    routed first, ties to the lower id.
  - `write_resident_set(path: str, stages: Mapping[int, Sequence[int]], *, n: int, source: str) -> None`.
  - `load_resident_set(path: str) -> dict[int, frozenset[int]]`. It raises `ValueError` naming the path on a
    missing file, bad JSON, the wrong `version`, or a non-integer or negative or repeated id.
  - `resident_for(layer_id: int) -> frozenset[int]`. It reads `SGLANG_DSV41_DSPARK_DRAFT_RESIDENT_PATH` (Task 3),
    cached per path. An empty path gives `frozenset()` (every routed expert on the CPU). A set path with no entry for
    `layer_id` raises `ValueError`.
- File format (version 1):
  `{"version": 1, "n": 32, "source": "<routes path>", "stages": {"0": [ids], "1": [ids], "2": [ids]}}`.

- [ ] **Step 1: Write the failing tests** in `test/registered/unit/kernels/test_dspark_draft_resident.py`:

```python
"""The DSpark draft's resident-set file: calibration from a routes probe, and a loader that refuses bad files."""

import json

import pytest

from sglang.srt.layers.moe.cpu_experts import draft_resident
from sglang.srt.layers.moe.cpu_experts.draft_resident import (
    load_resident_set,
    resident_for,
    top_n_resident_set,
    write_resident_set,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _routes():
    return [
        json.dumps({"layer": 0, "ids": [[5, 1, 2], [5, 1, 3]]}),
        json.dumps({"layer": 1, "ids": [[7, 8, 9], [9, 8, 4]]}),
        json.dumps({"layer": 0, "ids": [[5, 2, 6], [2, 0, 1]]}),
    ]


def test_top_n_takes_the_most_routed_ids_per_layer_ties_to_the_lower_id():
    # layer 0 counts: 5:3, 1:3, 2:3, 3:1, 6:1, 0:1; layer 1: 8:2, 9:2, 7:1, 4:1
    assert top_n_resident_set(_routes(), 2) == {0: [1, 2], 1: [8, 9]}


def test_a_written_file_loads_back_as_frozensets(tmp_path):
    path = tmp_path / "resident.json"
    write_resident_set(str(path), {0: [1, 2], 1: [8, 9]}, n=2, source="routes.jsonl")
    assert json.loads(path.read_text())["version"] == 1
    assert load_resident_set(str(path)) == {0: frozenset({1, 2}), 1: frozenset({8, 9})}


@pytest.mark.parametrize(
    "content",
    [
        "not json",
        json.dumps({"version": 2, "n": 1, "source": "", "stages": {"0": [1]}}),
        json.dumps({"version": 1, "n": 1, "source": "", "stages": {"0": [-1]}}),
        json.dumps({"version": 1, "n": 2, "source": "", "stages": {"0": [1, 1]}}),
        json.dumps({"version": 1, "n": 1, "source": "", "stages": {"x": [1]}}),
    ],
)
def test_a_malformed_file_is_refused_naming_the_path(tmp_path, content):
    path = tmp_path / "bad.json"
    path.write_text(content)
    with pytest.raises(ValueError, match="bad.json"):
        load_resident_set(str(path))


def test_a_missing_file_is_refused(tmp_path):
    with pytest.raises(ValueError, match="nope.json"):
        load_resident_set(str(tmp_path / "nope.json"))


def test_resident_for_follows_the_env_and_refuses_an_unlisted_stage(tmp_path):
    from sglang.srt.environ import envs

    path = tmp_path / "resident.json"
    write_resident_set(str(path), {0: [3]}, n=1, source="")
    draft_resident._cache.clear()
    with envs.SGLANG_DSV41_DSPARK_DRAFT_RESIDENT_PATH.override(""):
        assert resident_for(0) == frozenset()
    with envs.SGLANG_DSV41_DSPARK_DRAFT_RESIDENT_PATH.override(str(path)):
        assert resident_for(0) == frozenset({3})
        with pytest.raises(ValueError, match="stage 1"):
            resident_for(1)
```

- [ ] **Step 2: Run them; they must fail**

```bash
PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 taskset -c 0-63 /data/models/slang/.venv/bin/python -m pytest \
  test/registered/unit/kernels/test_dspark_draft_resident.py -q -p no:randomly 2>&1 | tail -3; echo "EXIT=${PIPESTATUS[0]}"
```
Expected: `EXIT=2` (collection error: `No module named ...cpu_experts.draft_resident`). Task 3's env var does not
exist yet either; the module must not touch it at import time.

- [ ] **Step 3: Implement** `python/sglang/srt/layers/moe/cpu_experts/draft_resident.py`:

```python
"""Which DSpark draft experts stay on the GPU: a calibration file made from a routes probe.

The probe (``SGLANG_DSPARK_DEBUG_DRAFT_ROUTES_PATH``) logs each draft MoE call's topk ids per stage (``layer_id``);
``top_n_resident_set`` keeps each stage's N most-routed experts. Every other routed expert of the stage runs on the
CPU (``cpu_experts/draft.py``).
"""

import collections
import json
import threading
from typing import Iterable, Mapping, Sequence

from sglang.srt.environ import envs

VERSION = 1

_cache: dict[str, dict[int, frozenset[int]]] = {}
_lock = threading.Lock()


def top_n_resident_set(route_lines: Iterable[str], n: int) -> dict[int, list[int]]:
    counts: dict[int, collections.Counter] = collections.defaultdict(collections.Counter)
    for line in route_lines:
        record = json.loads(line)
        counts[int(record["layer"])].update(e for row in record["ids"] for e in row if e >= 0)
    return {
        layer: [e for e, _ in sorted(c.items(), key=lambda item: (-item[1], item[0]))[:n]]
        for layer, c in sorted(counts.items())
    }


def write_resident_set(path: str, stages: Mapping[int, Sequence[int]], *, n: int, source: str) -> None:
    body = {
        "version": VERSION,
        "n": n,
        "source": source,
        "stages": {str(layer): [int(e) for e in ids] for layer, ids in sorted(stages.items())},
    }
    with open(path, "w") as f:
        json.dump(body, f, indent=1)
        f.write("\n")


def load_resident_set(path: str) -> dict[int, frozenset[int]]:
    try:
        with open(path) as f:
            body = json.load(f)
    except (OSError, ValueError) as error:
        raise ValueError(f"DSpark draft resident set {path}: {error}") from error
    if not isinstance(body, dict) or body.get("version") != VERSION or not isinstance(body.get("stages"), dict):
        raise ValueError(f"DSpark draft resident set {path}: expected version {VERSION} with a 'stages' map")
    stages = {}
    for key, ids in body["stages"].items():
        if not key.isdigit() or not isinstance(ids, list):
            raise ValueError(f"DSpark draft resident set {path}: stage {key!r} is not a layer id with a list")
        if any(not isinstance(e, int) or e < 0 for e in ids) or len(set(ids)) != len(ids):
            raise ValueError(f"DSpark draft resident set {path}: stage {key} ids must be distinct ints >= 0")
        stages[int(key)] = frozenset(ids)
    return stages


def resident_for(layer_id: int) -> frozenset[int]:
    path = envs.SGLANG_DSV41_DSPARK_DRAFT_RESIDENT_PATH.get()
    if not path:
        return frozenset()
    with _lock:
        if path not in _cache:
            _cache[path] = load_resident_set(path)
        stages = _cache[path]
    if layer_id not in stages:
        raise ValueError(
            f"DSpark draft resident set {path} has no stage {layer_id} (it lists {sorted(stages)}); "
            "recalibrate it from this draft's routes"
        )
    return stages[layer_id]
```

  Then create `analysis/dsv41-drive/dspark/draft_resident_set.py`:

```python
"""Make a DSpark draft resident-set file from a routes probe: each stage's N most-routed experts stay on the GPU.

  python analysis/dsv41-drive/dspark/draft_resident_set.py ROUTES.jsonl N OUT.json
ROUTES.jsonl is SGLANG_DSPARK_DEBUG_DRAFT_ROUTES_PATH's output (ab_cpu_draft.py's "routes" arm). The server reads
OUT.json through SGLANG_DSV41_DSPARK_DRAFT_RESIDENT_PATH.
"""

import sys

from sglang.srt.layers.moe.cpu_experts.draft_resident import top_n_resident_set, write_resident_set


def main():
    routes, n, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    with open(routes) as f:
        stages = top_n_resident_set(f, n)
    write_resident_set(out, stages, n=n, source=routes)
    for layer, ids in stages.items():
        print(f"stage {layer}: {len(ids)} resident: {ids}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests.** `test_resident_for_...` needs Task 3's env var. Run the other four now:

```bash
PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 taskset -c 0-63 /data/models/slang/.venv/bin/python -m pytest \
  test/registered/unit/kernels/test_dspark_draft_resident.py -q -p no:randomly -k "not resident_for" 2>&1 | tail -3; echo "EXIT=${PIPESTATUS[0]}"
```
Expected: `EXIT=0`. `test_resident_for_follows_the_env_and_refuses_an_unlisted_stage` passes after Task 3. Task 3's
final run includes this file.

- [ ] **Step 5: Commit**

```bash
git add python/sglang/srt/layers/moe/cpu_experts/draft_resident.py analysis/dsv41-drive/dspark/draft_resident_set.py \
  test/registered/unit/kernels/test_dspark_draft_resident.py
git commit -m "cpu-experts(dspark): the draft resident-set file, its calibration tool and loader

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Environment variables and the launch gate

**Files:**
- Modify: `python/sglang/srt/environ.py`: add four entries directly after `SGLANG_DSV41_CPU_EXPERTS_MISSES`, the last
  entry of the CPU-experts block.
- Modify: `python/sglang/srt/arg_groups/expert_stream_requirements_exl3.py`: add a rule directly before
  `cpu_experts = envs.SGLANG_DSV41_CPU_EXPERTS.get()`, which comes after the `isinstance(graph, CudaGraphConfig)`
  early return.
- Test: `test/registered/unit/test_expert_stream_requirements_exl3.py`

**Interfaces:**
- Consumes: `load_resident_set` (Task 2); `parse_core_list` (`cpu_experts/policy.py`).
- Produces:
  - `envs.SGLANG_DSV41_ENABLE_DSPARK_CPU_EXPERTS` (`EnvBool(False)`);
  - `envs.SGLANG_DSV41_DSPARK_CPU_EXPERTS_CORES` (`EnvStr("")`);
  - `envs.SGLANG_DSV41_DSPARK_CPU_EXPERTS_THREADS` (`EnvInt(0)`);
  - `envs.SGLANG_DSV41_DSPARK_DRAFT_RESIDENT_PATH` (`EnvStr("")`).

- [ ] **Step 1: Write the failing tests.** Append to `test/registered/unit/test_expert_stream_requirements_exl3.py`:

```python
DSPARK_CPU_ENV = dict(
    SGLANG_DSV41_ENABLE_DSPARK_CPU_EXPERTS=True,
    SGLANG_DSV41_DSPARK_CPU_EXPERTS_CORES="18-29",
)


def test_dspark_cpu_experts_pass_with_dspark_and_cores(model_dir):
    _gate(_launch(model_dir, speculative_algorithm="DSPARK"), **DSPARK_CPU_ENV)


def test_dspark_cpu_experts_without_dspark_are_refused(model_dir):
    with pytest.raises(ValueError, match="--speculative-algorithm DSPARK"):
        _gate(_launch(model_dir), **DSPARK_CPU_ENV)


@pytest.mark.parametrize("cores", ["", "18"])
def test_dspark_cpu_experts_need_two_cores(model_dir, cores):
    with pytest.raises(ValueError, match="SGLANG_DSV41_DSPARK_CPU_EXPERTS_CORES"):
        _gate(
            _launch(model_dir, speculative_algorithm="DSPARK"),
            **{**DSPARK_CPU_ENV, "SGLANG_DSV41_DSPARK_CPU_EXPERTS_CORES": cores},
        )


def test_a_bad_resident_file_is_refused_at_launch(model_dir, tmp_path):
    bad = tmp_path / "resident.json"
    bad.write_text("{}")
    with pytest.raises(ValueError, match="resident.json"):
        _gate(
            _launch(model_dir, speculative_algorithm="DSPARK"),
            **{**DSPARK_CPU_ENV, "SGLANG_DSV41_DSPARK_DRAFT_RESIDENT_PATH": str(bad)},
        )
```

- [ ] **Step 2: Run them; they must fail**

```bash
PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 taskset -c 0-63 /data/models/slang/.venv/bin/python -m pytest \
  test/registered/unit/test_expert_stream_requirements_exl3.py -q -p no:randomly -k "dspark_cpu or resident_file" 2>&1 | tail -3; echo "EXIT=${PIPESTATUS[0]}"
```
Expected: `EXIT=1`, with `AttributeError` on `envs.SGLANG_DSV41_ENABLE_DSPARK_CPU_EXPERTS`.

- [ ] **Step 3: Add the variables** in `environ.py`, after `SGLANG_DSV41_CPU_EXPERTS_MISSES`:

```python
    # DSpark draft experts on the CPU (plan 2026-10-02-dsv41-dspark-hybrid-draft): each draft stage keeps the experts
    # its resident set lists on the GPU and computes the rest with the CPU expert kernel, eagerly, so their VRAM goes
    # to the target's hot cache. A fused shared expert stays on the GPU. Needs --speculative-algorithm DSPARK.
    SGLANG_DSV41_ENABLE_DSPARK_CPU_EXPERTS = EnvBool(False)
    # Cores of the draft's CPU expert pool, as a taskset list ("18-29"). At least two.
    SGLANG_DSV41_DSPARK_CPU_EXPERTS_CORES = EnvStr("")
    # Worker threads of the draft's CPU expert pool, at most one per core. 0 takes one per core.
    SGLANG_DSV41_DSPARK_CPU_EXPERTS_THREADS = EnvInt(0)
    # The draft experts kept on the GPU, per stage (analysis/dsv41-drive/dspark/draft_resident_set.py).
    # Empty keeps none: every routed draft expert runs on the CPU.
    SGLANG_DSV41_DSPARK_DRAFT_RESIDENT_PATH = EnvStr("")
```

- [ ] **Step 4: Add the gate rule** in `expert_stream_requirements_exl3.py`, directly before
  `cpu_experts = envs.SGLANG_DSV41_CPU_EXPERTS.get()`:

```python
    if envs.SGLANG_DSV41_ENABLE_DSPARK_CPU_EXPERTS.get():
        if getattr(cfg, "speculative_algorithm", None) != "DSPARK":
            raise ValueError(
                "SGLANG_DSV41_ENABLE_DSPARK_CPU_EXPERTS computes the DSpark draft's routed experts on the CPU; "
                "pass --speculative-algorithm DSPARK or unset it"
            )
        if len(parse_core_list(envs.SGLANG_DSV41_DSPARK_CPU_EXPERTS_CORES.get())) < 2:
            raise ValueError(
                "SGLANG_DSV41_ENABLE_DSPARK_CPU_EXPERTS needs SGLANG_DSV41_DSPARK_CPU_EXPERTS_CORES with at "
                "least two cores (one spinning worker per core)"
            )
        resident = envs.SGLANG_DSV41_DSPARK_DRAFT_RESIDENT_PATH.get()
        if resident:
            load_resident_set(resident)
```

  At the top of the module, add:

```python
from sglang.srt.layers.moe.cpu_experts.draft_resident import load_resident_set
from sglang.srt.layers.moe.cpu_experts.policy import parse_core_list
```

  If importing `layers.moe.cpu_experts` from `arg_groups` creates an import cycle (the test collection errors with
  `ImportError: cannot import name ... (most likely due to a circular import)`), move both imports inside the
  `if`, and record a ruling.

- [ ] **Step 5: Run the gate file and Task 2's file; both must pass**

```bash
PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 taskset -c 0-63 /data/models/slang/.venv/bin/python -m pytest \
  test/registered/unit/test_expert_stream_requirements_exl3.py test/registered/unit/kernels/test_dspark_draft_resident.py \
  -q -p no:randomly 2>&1 | tail -3; echo "EXIT=${PIPESTATUS[0]}"
```
Expected: `EXIT=0`.

- [ ] **Step 6: Commit**

```bash
git add python/sglang/srt/environ.py python/sglang/srt/arg_groups/expert_stream_requirements_exl3.py \
  test/registered/unit/test_expert_stream_requirements_exl3.py
git commit -m "env(dspark): draft CPU experts, cores, threads and resident set, and their launch rule

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: `cpu_experts/draft.py`: the draft's CPU runtime, worker thread and stats

**Files:**
- Create: `python/sglang/srt/layers/moe/cpu_experts/draft.py`
- Test: `test/registered/unit/kernels/test_dspark_draft_cpu_experts.py`

**Interfaces:**
- Consumes:
  - `CpuExpertPool.compute_rows` (Task 1);
  - `cpu_trait_for("exl3")` (`cpu_experts/service.py:38`, an `Exl3CpuQuantTrait` with `act_limit=None`);
  - `parse_core_list`;
  - Task 3's envs.
- Produces:
  - `DraftLayer(slabs: Mapping[str, torch.Tensor], on_cpu: torch.Tensor, act_limit: Optional[float])`, a frozen
    dataclass. `on_cpu` is bool `[E]`, True for experts the CPU computes.
  - `DraftCpuExperts(trait, layers: Mapping[int, DraftLayer], *, cores: Sequence[int], threads: int, log_every: int = 300)`, with:
    - `.cpu_slots(key, ids: Tensor[m,k] int64 cpu) -> Tensor[m,k] int64`: ids the CPU computes; -1 elsewhere.
    - `.submit(key, ids: Tensor[m,k] int64 cpu, x: Tensor[m,H], weights: Tensor[m,k]) -> Optional[Future[Tensor[m,H] fp32 cpu]]`:
      `None`, recorded as a skip, when no route is on the CPU. Otherwise it moves `x` and `weights` to host fp16 on
      the caller and runs the kernel on the worker.
    - `.stats: DraftCpuStats`;
    - `.close()`.
  - `DraftCpuStats(log_every: int)`, with `.record(slots, seconds)`, `.record_skip()`, `.summary() -> str`, and the
    counters `.calls`, `.skips`, `.unions`, `.passes`.
  - `DraftCpuExpertsRegistry()`, with:
    - `.register(slabs, on_cpu, act_limit) -> int`;
    - `.runtime() -> DraftCpuExperts`, built once from the Task 3 envs;
    - `.close()`.
  - Module-level `DRAFT_CPU_EXPERTS = DraftCpuExpertsRegistry()`.

- [ ] **Step 1: Write the failing tests** in `test/registered/unit/kernels/test_dspark_draft_cpu_experts.py`:

```python
"""The DSpark draft's CPU expert runtime: masking, skips, the worker thread, stats and the registry (fake kernel)."""

import os
import threading

import pytest
import torch

from sglang.srt.layers.moe.cpu_experts.draft import (
    DraftCpuExperts,
    DraftCpuExpertsRegistry,
    DraftCpuStats,
    DraftLayer,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

E, H = 6, 4  # 5 routed experts plus a fused shared one (id 5); experts 1 and 5 stay on the GPU


class _Trait:
    name = "fake-draft"
    slab_names = ("w13_trellis",)
    act_limit = None
    x_dtype = torch.float16
    weights_dtype = torch.float16
    out_dtype = torch.float32

    def __init__(self):
        self.seen = []

    def check_environment(self):
        pass

    def register_layer(self, slabs, capacity):
        return capacity

    def forward(self, handle, x, slots, weights, out, threads):
        self.seen.append((slots.clone(), threading.get_native_id()))
        valid = (slots >= 0).to(torch.float32)
        out.copy_(x.float() * (weights.float() * valid).sum(-1, keepdim=True))

    def free_layer(self, handle):
        pass


def _cores():
    cores = sorted(os.sched_getaffinity(0))[:2]
    if len(cores) < 2:
        pytest.skip("needs at least 2 cores in the affinity mask")
    return cores


def _on_cpu():
    on_cpu = torch.ones(E, dtype=torch.bool)
    on_cpu[[1, 5]] = False
    return on_cpu


def _runtime(trait):
    layer = DraftLayer({"w13_trellis": torch.zeros(E, 2, dtype=torch.int16)}, on_cpu=_on_cpu(), act_limit=10.0)
    return DraftCpuExperts(trait, {0: layer}, cores=_cores(), threads=2)


def test_resident_and_fused_shared_ids_never_reach_the_cpu():
    trait = _Trait()
    experts = _runtime(trait)
    try:
        ids = torch.tensor([[0, 1, 5], [2, 5, -1]], dtype=torch.int64)
        out = experts.submit(0, ids, torch.ones(2, H, dtype=torch.bfloat16), torch.full((2, 3), 0.5)).result()
    finally:
        experts.close()
    slots, _ = trait.seen[0]
    assert slots.tolist() == [[0, -1, -1], [2, -1, -1]]
    assert out.dtype == torch.float32 and out.tolist() == [[0.5] * H, [0.5] * H]


def test_a_call_with_only_gpu_routes_skips_the_cpu():
    trait = _Trait()
    experts = _runtime(trait)
    try:
        ids = torch.tensor([[1, 5, -1]], dtype=torch.int64)
        assert experts.submit(0, ids, torch.ones(1, H), torch.ones(1, 3)) is None
    finally:
        experts.close()
    assert trait.seen == [] and experts.stats.skips == 1 and experts.stats.calls == 1


def test_the_kernel_runs_on_the_worker_and_the_caller_keeps_its_affinity():
    trait = _Trait()
    before = os.sched_getaffinity(0)
    experts = _runtime(trait)
    try:
        experts.submit(0, torch.zeros(1, 3, dtype=torch.int64), torch.ones(1, H), torch.ones(1, 3)).result()
    finally:
        experts.close()
    assert trait.seen[0][1] != threading.get_native_id()
    assert os.sched_getaffinity(0) == before


def test_stats_count_union_passes_and_skips():
    stats = DraftCpuStats(log_every=1000)
    # 6 rows: expert 0 routed by all 6 (3 passes), expert 2 by 2 (1 pass), expert 3 by 1 (1 pass)
    slots = torch.tensor([[0, 2, -1]] * 2 + [[0, 3, -1]] + [[0, -1, -1]] * 3, dtype=torch.int64)
    stats.record(slots, 0.010)
    stats.record_skip()
    assert stats.unions == [3] and stats.passes == [5]
    assert stats.calls == 2 and stats.skips == 1
    summary = stats.summary()
    assert "2 stage calls" in summary and "1 without CPU work" in summary and "union 3.0" in summary


def test_the_registry_builds_once_from_the_envs_and_refuses_mixed_limits(monkeypatch):
    from sglang.srt.environ import envs
    from sglang.srt.layers.moe.cpu_experts import draft

    built = []
    monkeypatch.setattr(draft, "cpu_trait_for", lambda key: built.append(key) or _Trait())
    registry = DraftCpuExpertsRegistry()
    slabs = {"w13_trellis": torch.zeros(E, 2, dtype=torch.int16)}
    assert registry.register(slabs, _on_cpu(), 10.0) == 0
    assert registry.register(slabs, _on_cpu(), 10.0) == 1
    with pytest.raises(ValueError, match="activation limit"):
        registry.register(slabs, _on_cpu(), 7.0)
    cores = ",".join(str(c) for c in _cores())
    with envs.SGLANG_DSV41_DSPARK_CPU_EXPERTS_CORES.override(cores):
        try:
            runtime = registry.runtime()
            assert registry.runtime() is runtime and built == ["exl3"]
            assert runtime.pool.trait.act_limit == 10.0
            assert sorted(runtime.pool.capacity) == [0, 1]
        finally:
            registry.close()
```

- [ ] **Step 2: Run them; they must fail**

```bash
PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 taskset -c 0-63 /data/models/slang/.venv/bin/python -m pytest \
  test/registered/unit/kernels/test_dspark_draft_cpu_experts.py -q -p no:randomly 2>&1 | tail -3; echo "EXIT=${PIPESTATUS[0]}"
```
Expected: `EXIT=2` (collection error: `No module named ...cpu_experts.draft`).

- [ ] **Step 3: Implement** `python/sglang/srt/layers/moe/cpu_experts/draft.py`:

```python
"""The DSpark draft's non-resident routed experts on the CPU (eager).

With ``SGLANG_DSV41_ENABLE_DSPARK_CPU_EXPERTS`` each draft stage's ``FusedMoE`` loads its experts into host RAM and
keeps only its resident set (``draft_resident.py``) and any fused shared expert on the GPU. This module computes the
rest with the CPU expert kernel. The stages register here as they finish loading; the pool is built on first use.

One worker thread runs the kernel, bound to the pool's cores, so the caller (the scheduler) never is. The kernel's
op releases the GIL, so the caller runs the resident experts on the GPU meanwhile.
"""

import atexit
import logging
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.moe.cpu_experts.policy import parse_core_list
from sglang.srt.layers.moe.cpu_experts.pool import CpuExpertPool
from sglang.srt.layers.moe.cpu_experts.service import cpu_trait_for

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DraftLayer:
    """One draft stage: its expert tensors in the pinned tier's slab layout, which experts the CPU computes, and
    the SwiGLU clamp."""

    slabs: Mapping[str, torch.Tensor]
    on_cpu: torch.Tensor
    act_limit: Optional[float]


class DraftCpuStats:
    """Cost and routing of the draft's CPU share, logged every ``log_every`` stage calls.

    ``passes`` counts weight reads: the kernel reads an expert once per two rows routed to it. ``skips`` counts
    stage calls whose routes were all resident.
    """

    def __init__(self, log_every: int):
        self.log_every = log_every
        self.calls = 0
        self.skips = 0
        self.seconds: list[float] = []
        self.unions: list[int] = []
        self.passes: list[int] = []

    def record(self, slots: torch.Tensor, seconds: float) -> None:
        used = torch.bincount(slots[slots >= 0])
        used = used[used > 0]
        self.unions.append(int(used.numel()))
        self.passes.append(int(((used + 1) // 2).sum()))
        self.seconds.append(seconds)
        self._count()

    def record_skip(self) -> None:
        self.skips += 1
        self._count()

    def _count(self) -> None:
        self.calls += 1
        if self.calls % self.log_every == 0:
            logger.info(self.summary())
            self.seconds, self.unions, self.passes = [], [], []

    def summary(self) -> str:
        head = f"DSpark CPU experts: {self.calls} stage calls, {self.skips} without CPU work"
        if not self.seconds:
            return head
        ms = sorted(s * 1e3 for s in self.seconds)
        window = len(ms)
        return (
            f"{head}; last {window} CPU calls: {sum(ms) / window:.2f} ms mean, "
            f"{ms[int(0.9 * (window - 1))]:.2f} ms p90, union {sum(self.unions) / window:.1f}, "
            f"weight passes {sum(self.passes) / window:.1f}"
        )


class DraftCpuExperts:
    """The draft stages' CPU expert pool and the worker thread that runs it."""

    def __init__(
        self,
        trait,
        layers: Mapping[int, DraftLayer],
        *,
        cores: Sequence[int],
        threads: int,
        log_every: int = 300,
    ):
        self.pool = CpuExpertPool(
            trait, {key: layer.slabs for key, layer in layers.items()}, cores=cores, threads=threads
        )
        self.on_cpu = {key: layer.on_cpu for key, layer in layers.items()}
        self.stats = DraftCpuStats(log_every)
        self._worker = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="dspark-cpu-experts",
            initializer=self.pool.bind_current_thread,
        )

    def cpu_slots(self, key: int, ids: torch.Tensor) -> torch.Tensor:
        on_cpu = self.on_cpu[key]
        valid = (ids >= 0) & (ids < len(on_cpu))
        keep = valid & on_cpu[ids.clamp(0, len(on_cpu) - 1)]
        return torch.where(keep, ids, torch.full_like(ids, -1))

    def submit(
        self, key: int, ids: torch.Tensor, x: torch.Tensor, weights: torch.Tensor
    ) -> Optional[Future]:
        """Start stage ``key``'s CPU routes over ``x`` ``[m, H]``; None when every route is resident."""
        slots = self.cpu_slots(key, ids)
        if not bool((slots >= 0).any()):
            self.stats.record_skip()
            return None
        x16 = x.to(torch.float16).cpu()
        w16 = weights.to(torch.float16).cpu()
        return self._worker.submit(self._run, key, x16, slots, w16)

    def _run(self, key: int, x16: torch.Tensor, slots: torch.Tensor, w16: torch.Tensor) -> torch.Tensor:
        out = torch.empty(x16.shape, dtype=torch.float32)
        start = time.perf_counter()
        self.pool.compute_rows(key, slots, w16, x16, out)
        self.stats.record(slots, time.perf_counter() - start)
        return out

    def close(self) -> None:
        self._worker.shutdown(wait=True)
        self.pool.close()


class DraftCpuExpertsRegistry:
    """Draft stages registered at load, and the runtime built from them on first use."""

    def __init__(self):
        self._layers: dict[int, DraftLayer] = {}
        self._runtime: Optional[DraftCpuExperts] = None
        self._lock = threading.Lock()

    def register(
        self, slabs: Mapping[str, torch.Tensor], on_cpu: torch.Tensor, act_limit: Optional[float]
    ) -> int:
        """Add one stage; returns its key. Every stage must share one activation limit (one kernel trait)."""
        with self._lock:
            if self._runtime is not None:
                raise RuntimeError("a DSpark draft stage registered after the CPU experts started")
            limits = {layer.act_limit for layer in self._layers.values()}
            if limits and act_limit not in limits:
                raise ValueError(
                    f"DSpark draft stages disagree on the activation limit: {sorted(limits)} and {act_limit}"
                )
            key = len(self._layers)
            self._layers[key] = DraftLayer(slabs, on_cpu, act_limit)
            return key

    def runtime(self) -> DraftCpuExperts:
        with self._lock:
            if self._runtime is None:
                if not self._layers:
                    raise RuntimeError("no DSpark draft stage registered for CPU experts")
                cores = parse_core_list(envs.SGLANG_DSV41_DSPARK_CPU_EXPERTS_CORES.get())
                threads = envs.SGLANG_DSV41_DSPARK_CPU_EXPERTS_THREADS.get() or len(cores)
                trait = cpu_trait_for("exl3")
                trait.act_limit = next(iter(self._layers.values())).act_limit
                self._runtime = DraftCpuExperts(trait, self._layers, cores=cores, threads=threads)
                atexit.register(self.close)
                logger.info(
                    "DSpark CPU experts: %d draft stages on cores %s, %d threads; %s experts on the CPU",
                    len(self._layers),
                    cores,
                    threads,
                    [int(layer.on_cpu.sum()) for layer in self._layers.values()],
                )
            return self._runtime

    def close(self) -> None:
        with self._lock:
            runtime, self._runtime = self._runtime, None
        if runtime is not None:
            logger.info(runtime.stats.summary())
            runtime.close()


DRAFT_CPU_EXPERTS = DraftCpuExpertsRegistry()
```

  Check `CpuExpertPool`'s constructor against this call: `CpuExpertPool(trait, slabs_by_layer, cores=..., threads=...)`,
  `.capacity`, `.bind_current_thread` and `.close`. If its real signature differs, adapt the call and record a ruling.

- [ ] **Step 4: Run the tests; they must pass**

```bash
PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 taskset -c 0-63 /data/models/slang/.venv/bin/python -m pytest \
  test/registered/unit/kernels/test_dspark_draft_cpu_experts.py test/registered/unit/kernels/test_cpu_expert_pool.py \
  -q -p no:randomly 2>&1 | tail -3; echo "EXIT=${PIPESTATUS[0]}"
```
Expected: `EXIT=0`.

- [ ] **Step 5: Commit**

```bash
git add python/sglang/srt/layers/moe/cpu_experts/draft.py test/registered/unit/kernels/test_dspark_draft_cpu_experts.py
git commit -m "cpu-experts(dspark): the draft's CPU runtime (worker thread, resident masking, skips, stats)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Route the draft stages' `Exl3MoEMethod` through the hybrid

**Files:**
- Modify: `python/sglang/srt/models/deepseek_v4_exl3_weights.py`: add `DSPARK_DRAFT_EXPERT_MODULE_RE` and
  `is_dspark_draft_expert_module` after `is_streamed_expert_module`.
- Modify: `python/sglang/srt/layers/quantization/exl3.py`:
  - `Exl3Config.get_quant_method`;
  - `Exl3MoEMethod.__init__`, `create_weights`, `process_weights_after_loading` and `apply`;
  - new methods `_attach_cpu_draft` and `_apply_cpu_draft`.
- Test: `test/registered/unit/layers/quantization/test_exl3_stream_scope.py` (the module regex) and
  `test/registered/unit/layers/quantization/test_exl3_moe_method.py` (placement, registration, apply).

**Interfaces:**
- Consumes:
  - `resident_for` (Task 2);
  - `DRAFT_CPU_EXPERTS.register(...)`, `.runtime().submit(...)` (Task 4);
  - `exl3_moe_accumulate`, `Exl3Tensors` and `EXL3_STREAMED_NAMES`, all already imported in `exl3.py`;
  - `layer.layer_id`;
  - `layer.moe_runner_config.swiglu_limit` (set in `FusedMoE.__init__`, before the weights load).
- Produces:
  - `Exl3MoEMethod(config, *, streamed: bool, cpu_draft: bool = False)`.
  - On a CPU-draft layer:
    - `layer.exl3_cpu_draft_key: int`;
    - `layer.exl3_gpu_experts: frozenset[int]` (the resident set plus fused shared ids);
    - `layer.exl3_gpu_w13: dict[int, tuple[Exl3Tensors, Exl3Tensors]]`;
    - `layer.exl3_gpu_w2: dict[int, Exl3Tensors]`.

- [ ] **Step 1: Write the failing tests.**

Append to `test_exl3_stream_scope.py`:

```python
from sglang.srt.models.deepseek_v4_exl3_weights import is_dspark_draft_expert_module


@pytest.mark.parametrize(
    "prefix,expected",
    [
        ("stages.0.mlp.experts", True),
        ("model.stages.2.mlp.experts", True),
        ("model.layers.0.mlp.experts", False),
        ("stages.0.mlp.shared_experts", False),
    ],
)
def test_only_draft_stage_experts_are_draft_modules(prefix, expected):
    assert is_dspark_draft_expert_module(prefix) is expected
```

Append to `test_exl3_moe_method.py` (it already defines `CFG`, `E = 4`, `HIDDEN`, `INTER` and `_load_all`):

```python
from types import SimpleNamespace


def _cpu_draft_moe(monkeypatch, tmp_path, resident, fused_shared=0):
    from sglang.srt.environ import envs
    from sglang.srt.layers.moe.cpu_experts import draft, draft_resident

    registered = []
    registry = SimpleNamespace(register=lambda slabs, on_cpu, limit: registered.append((slabs, on_cpu, limit)) or 0)
    monkeypatch.setattr(draft, "DRAFT_CPU_EXPERTS", registry)
    path = tmp_path / "resident.json"
    draft_resident.write_resident_set(str(path), {0: resident}, n=len(resident), source="")
    draft_resident._cache.clear()
    layer = nn.Module()
    layer.layer_id = 0
    layer.num_experts = E
    layer.num_fused_shared_experts = fused_shared
    layer.moe_runner_config = SimpleNamespace(swiglu_limit=10.0)
    method = Exl3MoEMethod(Exl3Config.from_config(CFG), streamed=False, cpu_draft=True)
    method.create_weights(layer, E, HIDDEN, INTER, torch.bfloat16)
    _load_all(layer)
    with envs.SGLANG_DSV41_DSPARK_DRAFT_RESIDENT_PATH.override(str(path)):
        method.process_weights_after_loading(layer)
    return layer, method, registered


def test_a_cpu_draft_layer_loads_into_host_memory_and_keeps_its_resident_set_for_the_gpu(monkeypatch, tmp_path):
    layer, method, registered = _cpu_draft_moe(monkeypatch, tmp_path, resident=[2])
    assert layer.w13_trellis.device.type == "cpu" and layer.w2_svh.device.type == "cpu"
    (slabs, on_cpu, limit), = registered
    assert on_cpu.tolist() == [True, True, False, True] and limit == 10.0
    assert slabs["w13_trellis"].data_ptr() == layer.w13_trellis.data_ptr()
    assert layer.exl3_cpu_draft_key == 0 and layer.exl3_gpu_experts == frozenset({2})
    gate, up = layer.exl3_gpu_w13[2]
    assert int(gate.trellis[0, 0, 0]) == 10 * 2 + 1 and sorted(layer.exl3_gpu_w2) == [2]


def test_a_fused_shared_expert_is_always_on_the_gpu(monkeypatch, tmp_path):
    layer, method, registered = _cpu_draft_moe(monkeypatch, tmp_path, resident=[0], fused_shared=1)
    assert registered[0][1].tolist() == [False, True, True, False]
    assert layer.exl3_gpu_experts == frozenset({0, E - 1})


def test_a_resident_id_outside_the_routed_experts_is_refused(monkeypatch, tmp_path):
    with pytest.raises(ValueError, match="resident"):
        _cpu_draft_moe(monkeypatch, tmp_path, resident=[E - 1], fused_shared=1)


def test_apply_adds_the_gpu_and_cpu_shares(monkeypatch, tmp_path):
    from sglang.srt.layers.moe.cpu_experts import draft
    from sglang.srt.layers.quantization import exl3

    layer, method, _ = _cpu_draft_moe(monkeypatch, tmp_path, resident=[2])
    seen = {}

    class _Future:
        def result(self):
            return torch.full((2, HIDDEN), 1.0)

    def submit(key, ids, x, weights):
        seen["ids"] = ids.tolist()
        return _Future()

    monkeypatch.setattr(draft, "DRAFT_CPU_EXPERTS", SimpleNamespace(runtime=lambda: SimpleNamespace(submit=submit)))

    def accumulate(out, x, w, ids, w13, w2, limit, experts):
        seen["gpu"] = list(experts)
        out += 2.0

    monkeypatch.setattr(exl3, "exl3_moe_accumulate", accumulate)
    x = torch.zeros(2, HIDDEN, dtype=torch.bfloat16)
    ids = torch.tensor([[2, 0], [1, 3]])
    out = method._apply_cpu_draft(layer, x, torch.ones(2, 2), ids, 10.0)
    assert seen == {"ids": [[2, 0], [1, 3]], "gpu": [2]}
    assert out.dtype == torch.bfloat16 and torch.all(out == 3.0)


def test_without_the_flag_a_draft_layer_stays_on_the_default_device():
    method = Exl3MoEMethod(Exl3Config.from_config(CFG), streamed=False)
    assert method.cpu_draft is False
```

- [ ] **Step 2: Run them; they must fail**

```bash
PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 taskset -c 0-63 /data/models/slang/.venv/bin/python -m pytest \
  test/registered/unit/layers/quantization/test_exl3_stream_scope.py \
  test/registered/unit/layers/quantization/test_exl3_moe_method.py -q -p no:randomly 2>&1 | tail -3; echo "EXIT=${PIPESTATUS[0]}"
```
Expected: `EXIT=2` (`ImportError: cannot import name 'is_dspark_draft_expert_module'`).

- [ ] **Step 3: Add the module predicate** in `deepseek_v4_exl3_weights.py`, after `is_streamed_expert_module`:

```python
# The DSpark draft's stage MoEs: "stages.<S>.mlp.experts" (empty root prefix) or "model.stages.<S>.mlp.experts".
DSPARK_DRAFT_EXPERT_MODULE_RE = re.compile(r"^(?:model\.)?stages\.\d+\.mlp\.experts$")


def is_dspark_draft_expert_module(prefix: str) -> bool:
    """True for a DSpark draft stage's routed-expert FusedMoE."""
    return DSPARK_DRAFT_EXPERT_MODULE_RE.match(prefix) is not None
```

- [ ] **Step 4: Wire `Exl3MoEMethod`** in `exl3.py`.

  **4a.** In `Exl3Config.get_quant_method`, replace the `FusedMoE` branch's import and return
  (`return Exl3MoEMethod(self, streamed=is_streamed_expert_module(prefix))`):

```python
            from sglang.srt.models.deepseek_v4_exl3_weights import (
                is_dspark_draft_expert_module,
                is_streamed_expert_module,
            )

            return Exl3MoEMethod(
                self,
                streamed=is_streamed_expert_module(prefix),
                cpu_draft=envs.SGLANG_DSV41_ENABLE_DSPARK_CPU_EXPERTS.get()
                and is_dspark_draft_expert_module(prefix),
            )
```

  **4b.** Change `Exl3MoEMethod.__init__`:

```python
    def __init__(self, config: Exl3Config, *, streamed: bool, cpu_draft: bool = False):
        self.config = config
        self.streamed = streamed
        # A DSpark draft stage whose non-resident experts run on the CPU (cpu_experts/draft.py).
        self.cpu_draft = cpu_draft
        self.moe_runner_config = None
        self.cast_fusion = envs.SGLANG_DSV41_ENABLE_EXL3_CAST_FUSION.get()
        self.route_plan = envs.SGLANG_DSV41_ENABLE_PREFILL_ROUTE_PLAN.get()
```

  **4c.** In `create_weights`, make the parameter placeholders on the CPU for a CPU draft. `_materialize` allocates
  on `param.device`, so the experts never touch VRAM:

```python
        device = "cpu" if self.cpu_draft else None
        for prefix in ("w13", "w2"):
            for name in EXL3_PARAMS:
                param = nn.Parameter(
                    torch.empty(0, dtype=torch.int8, device=device), requires_grad=False
                )
```
  The rest of the loop body is unchanged.

  **4d.** At the end of `process_weights_after_loading`, after the `w2` shape-check loop:

```python
        if self.cpu_draft:
            self._attach_cpu_draft(layer)
```

  **4e.** Add the two methods to `Exl3MoEMethod`, after `process_weights_after_loading`:

```python
    def _attach_cpu_draft(self, layer: nn.Module) -> None:
        """Copy the resident set and fused shared experts to the GPU; register every slab with the CPU runtime.

        The parameters are the pinned tier's slab layout ([expert, part, ...]), so they register as they are.
        """
        from dataclasses import replace

        from sglang.srt.layers.moe.cpu_experts import draft
        from sglang.srt.layers.moe.cpu_experts.draft_resident import resident_for

        n_routed = layer.exl3_num_experts - getattr(layer, "num_fused_shared_experts", 0)
        resident = resident_for(layer.layer_id)
        outside = sorted(e for e in resident if e >= n_routed)
        if outside:
            raise ValueError(
                f"exl3: DSpark draft stage {layer.layer_id}'s resident set lists {outside}, "
                f"outside its {n_routed} routed experts"
            )
        gpu = frozenset(resident) | frozenset(range(n_routed, layer.exl3_num_experts))
        on_cpu = torch.ones(layer.exl3_num_experts, dtype=torch.bool)
        on_cpu[sorted(gpu)] = False
        slabs = {name: getattr(layer, name).data for name in EXL3_STREAMED_NAMES}
        layer.exl3_cpu_draft_key = draft.DRAFT_CPU_EXPERTS.register(
            slabs, on_cpu, layer.moe_runner_config.swiglu_limit
        )
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        def to_device(t: Exl3Tensors) -> Exl3Tensors:
            return replace(t, trellis=t.trellis.to(device), suh=t.suh.to(device), svh=t.svh.to(device))

        layer.exl3_gpu_experts = gpu
        layer.exl3_gpu_w13 = {e: tuple(to_device(t) for t in layer.exl3_w13[e]) for e in sorted(gpu)}
        layer.exl3_gpu_w2 = {e: to_device(layer.exl3_w2[e]) for e in sorted(gpu)}

    def _apply_cpu_draft(self, layer, x, topk_weights, topk_ids, swiglu_limit) -> torch.Tensor:
        """Non-resident routes on the CPU worker while the resident ones run on the GPU; eager."""
        from sglang.srt.layers.moe.cpu_experts import draft

        assert_not_capturing("Exl3MoEMethod._apply_cpu_draft")
        ids = topk_ids.to(torch.int64).cpu()
        pending = draft.DRAFT_CPU_EXPERTS.runtime().submit(layer.exl3_cpu_draft_key, ids, x, topk_weights)
        out = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
        present = sorted(set(ids[ids >= 0].tolist()) & layer.exl3_gpu_experts)
        if present:
            exl3_moe_accumulate(
                out,
                x,
                topk_weights,
                topk_ids,
                layer.exl3_gpu_w13,
                layer.exl3_gpu_w2,
                swiglu_limit,
                experts=present,
            )
        if pending is not None:
            out += pending.result().to(x.device)
        return out.to(x.dtype)
```

  **4f.** In `apply`, add a branch between `elif streamer is not None:` and the final `else:` (the `exl3_moe_loop`
  call):

```python
        elif self.cpu_draft:
            out = self._apply_cpu_draft(
                layer, dispatch_output.hidden_states, topk_weights, topk_ids, cfg.swiglu_limit
            )
```

  The routed scaling after the branches applies to this output as it does to `exl3_moe_loop`'s. Check that by
  reading the code after the `else:` branch. If the scaling lives inside the `else:`, apply the same scaling in the
  new branch and record a ruling.

- [ ] **Step 5: Run the tests; they must pass.** Run the touched files plus the EXL3 MoE files, so the constructor
  change cannot break another caller:

```bash
PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 taskset -c 0-63 /data/models/slang/.venv/bin/python -m pytest \
  test/registered/unit/layers/quantization/test_exl3_stream_scope.py \
  test/registered/unit/layers/quantization/test_exl3_moe_method.py \
  test/registered/unit/layers/quantization/test_exl3_moe_stream_mode.py \
  test/registered/unit/kernels/test_dspark_draft_cpu_experts.py \
  test/registered/unit/kernels/test_dspark_draft_resident.py \
  -q -p no:randomly 2>&1 | tail -3; echo "EXIT=${PIPESTATUS[0]}"
```
Expected: `EXIT=0`. If `test_exl3_moe_stream_mode.py` does not exist, drop it from the command and record that.

- [ ] **Step 6: Commit**

```bash
git add python/sglang/srt/models/deepseek_v4_exl3_weights.py python/sglang/srt/layers/quantization/exl3.py \
  test/registered/unit/layers/quantization/test_exl3_stream_scope.py \
  test/registered/unit/layers/quantization/test_exl3_moe_method.py
git commit -m "exl3(dspark): hybrid draft experts, resident set on the GPU and the rest on the CPU kernel

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: GPU parity: the hybrid path against `exl3_moe_loop`, at 3 and 4 bits

**Files:**
- Create: `test/manual/dsv41/test_dspark_hybrid_draft_gpu.py`

**Interfaces:**
- Consumes:
  - `Exl3MoEMethod(..., cpu_draft=True)` and its `_apply_cpu_draft` (Task 5);
  - `DraftCpuExpertsRegistry` (Task 4);
  - `write_resident_set` (Task 2);
  - `exl3_moe_loop` (`exl3_ops.py:191`).

The CPU kernel quantizes activations, so the result is close to the GPU's, not bit-identical. The same check at
3 bits (the target's validated bitrate) separates a 4-bit defect from that noise. The resident sets cover all three
paths: none resident (all CPU), some resident (both), and all resident (the CPU skipped).

- [ ] **Step 1: Write the test**

```python
"""The DSpark hybrid draft path matches exl3_moe_loop on the GPU, at the target's 3 and the draft's 4 bits.

Run on divix01 with the GPU lock, EXL3_MOE_CPU_PIN=0 and SGLANG_EXL3_SRC set (the CPU kernel's build):
  flock .../cc-gpu.lock taskset -c 18-29,32-63 python -m pytest test/manual/dsv41/test_dspark_hybrid_draft_gpu.py -s
"""

import os
from types import SimpleNamespace

import pytest
import torch
from torch import nn

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not os.environ.get("SGLANG_EXL3_SRC"),
    reason="needs CUDA and SGLANG_EXL3_SRC",
)

E_ROUTED, SHARED, HIDDEN, INTER, ROWS, TOPK = 8, 1, 512, 256, 5, 3
LIMIT = 10.0


def _layer(bits, resident, monkeypatch, registry, tmp_path):
    from sglang.srt.environ import envs
    from sglang.srt.layers.moe.cpu_experts import draft, draft_resident
    from sglang.srt.layers.quantization.exl3 import Exl3Config, Exl3MoEMethod

    monkeypatch.setattr(draft, "DRAFT_CPU_EXPERTS", registry)
    path = tmp_path / "resident.json"
    draft_resident.write_resident_set(str(path), {0: resident}, n=len(resident), source="")
    draft_resident._cache.clear()
    cfg = {"quant_method": "exl3", "version": "1.4.2", "bits": float(bits), "head_bits": 6, "codebook": "mul1"}
    e = E_ROUTED + SHARED
    layer = nn.Module()
    layer.layer_id = 0
    layer.num_experts = e
    layer.num_fused_shared_experts = SHARED
    layer.moe_runner_config = SimpleNamespace(swiglu_limit=LIMIT)
    method = Exl3MoEMethod(Exl3Config.from_config(cfg), streamed=False, cpu_draft=True)
    method.create_weights(layer, e, HIDDEN, INTER, torch.bfloat16)
    g = torch.Generator().manual_seed(bits)
    for expert in range(e):
        for shard, (in_f, out_f), prefix in (
            ("w1", (HIDDEN, INTER), "w13"),
            ("w3", (HIDDEN, INTER), "w13"),
            ("w2", (INTER, HIDDEN), "w2"),
        ):
            tensors = {
                "trellis": torch.randint(-32768, 32767, (in_f // 16, out_f // 16, 16 * bits), generator=g, dtype=torch.int16),
                "suh": (torch.randint(0, 2, (in_f,), generator=g) * 2 - 1).half(),
                "svh": (torch.randint(0, 2, (out_f,), generator=g) * 2 - 1).half(),
                "mul1": torch.tensor(1, dtype=torch.int32),
            }
            for name, tensor in tensors.items():
                param = getattr(layer, f"{prefix}_{name}")
                param.weight_loader(param, tensor, f"experts.{prefix}_{name}", shard_id=shard, expert_id=expert)
    with envs.SGLANG_DSV41_DSPARK_DRAFT_RESIDENT_PATH.override(str(path)):
        method.process_weights_after_loading(layer)
    return layer, method


def _routes(pattern, g):
    if pattern == "shared":
        routed = torch.randperm(E_ROUTED, generator=g)[:TOPK].repeat(ROWS, 1)
    else:
        routed = torch.stack([torch.randperm(E_ROUTED, generator=g)[:TOPK] for _ in range(ROWS)])
    shared = torch.full((ROWS, 1), E_ROUTED, dtype=torch.int64)
    return torch.cat([routed, shared], dim=1)


@pytest.mark.parametrize("resident", [[], [1, 5], list(range(E_ROUTED))], ids=["all-cpu", "hybrid", "all-gpu"])
@pytest.mark.parametrize("pattern", ["independent", "shared"])
@pytest.mark.parametrize("bits", [3, 4])
def test_hybrid_draft_matches_the_gpu_loop(bits, pattern, resident, monkeypatch, tmp_path):
    from dataclasses import replace

    from sglang.srt.environ import envs
    from sglang.srt.layers.moe.cpu_experts.draft import DraftCpuExpertsRegistry
    from sglang.srt.layers.quantization.exl3_ops import exl3_moe_loop

    monkeypatch.setenv("EXL3_MOE_CPU_PIN", "0")
    cores = sorted(os.sched_getaffinity(0) & set(range(18, 30))) or sorted(os.sched_getaffinity(0))[:4]
    registry = DraftCpuExpertsRegistry()
    layer, method = _layer(bits, resident, monkeypatch, registry, tmp_path)
    g = torch.Generator().manual_seed(100 + bits)
    x = (torch.randn(ROWS, HIDDEN, generator=g) * 0.5).to(torch.bfloat16).cuda()
    ids = _routes(pattern, g).cuda()
    weights = torch.rand(ROWS, TOPK + 1, generator=g).cuda()

    def gpu(t):
        return replace(t, trellis=t.trellis.cuda(), suh=t.suh.cuda(), svh=t.svh.cuda())

    w13 = [tuple(gpu(t) for t in pair) for pair in layer.exl3_w13]
    w2 = [gpu(t) for t in layer.exl3_w2]
    with envs.SGLANG_DSV41_DSPARK_CPU_EXPERTS_CORES.override(",".join(map(str, cores))):
        try:
            got = method._apply_cpu_draft(layer, x, weights, ids, LIMIT).float()
            stats = registry.runtime().stats
            skips = stats.skips
        finally:
            registry.close()
    ref = exl3_moe_loop(x, weights, ids, w13, w2, LIMIT).float()
    rel = float((got - ref).norm() / ref.norm())
    print(f"bits={bits} pattern={pattern} resident={len(resident)} rel_l2={rel:.4f} skips={skips}")
    assert torch.isfinite(got).all()
    assert rel < 0.05
    assert skips == (1 if len(resident) == E_ROUTED else 0)
```

- [ ] **Step 2: Commit, push, pull into the divix01 worktree, and run** (GPU lock; CPU cores 18-29 for the kernel)

```bash
git add test/manual/dsv41/test_dspark_hybrid_draft_gpu.py
git commit -m "test(dspark): hybrid draft experts against the GPU loop at 3 and 4 bits

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push origin dsv41-dspark-cpu-draft
ssh divix01 'cd /data/models/slang/nvfp4-work/wt-dspark-cpu && git fetch origin && git checkout --detach origin/dsv41-dspark-cpu-draft && \
  export PYTHONPATH=$PWD/python EXL3_MOE_CPU_PIN=0 && \
  export SGLANG_EXL3_SRC=$(PYTHONPATH=benchmarks/dsv41_baseline /data/models/slang/.venv/bin/python -c "import arm_env;print(arm_env.base_env()[\"SGLANG_EXL3_SRC\"])") && \
  export SGLANG_EXL3_BUILD_DIR=$(PYTHONPATH=benchmarks/dsv41_baseline /data/models/slang/.venv/bin/python -c "import arm_env;print(arm_env.base_env()[\"SGLANG_EXL3_BUILD_DIR\"])") && \
  flock /data/models/slang/nvfp4-work/cc-gpu.lock taskset -c 18-29,32-63 /data/models/slang/.venv/bin/python -m pytest \
  test/manual/dsv41/test_dspark_hybrid_draft_gpu.py -q -s -p no:randomly 2>&1 | grep -E "rel_l2|passed|failed|Error" ; echo "EXIT=${PIPESTATUS[0]}"'
```
Expected: `EXIT=0`, twelve `rel_l2=` lines, and 4-bit values of the same order as the 3-bit ones.

If a 4-bit case fails while 3-bit passes, the kernel mishandles the draft's bitrate. Debug that with
`superpowers:systematic-debugging`. Never loosen the 0.05.

- [ ] **Step 3: Record** the twelve `rel_l2` values under "Results" in this plan, then commit.

---

### Task 7: Calibrate, then the served A/B on held-out sessions, and the doc

**Files:**
- Modify: `analysis/dsv41-drive/dspark/ab_cpu_draft.py`: replace the `cpu` arm with `hybrid`, and add `AB_SKIP`.
- Modify: `DSV41_REFERENCE.md`: add §33.4.
- Output (divix01, not committed): `cc-expert-prediction/analysis/dsv41-dspark/cpu-draft-ab/`.

**Interfaces:**
- Consumes:
  - the routes probe output `cc-expert-prediction/analysis/dsv41-dspark/cpu-draft-routes/routes.jsonl` (sessions
    0–7);
  - `draft_resident_set.py` (Task 2);
  - the driver's `COMMON` overrides and `routes` arm, already on the branch;
  - `trace_corpus.py --skip`.

The `hybrid` arm's hot-cache budget is the resident arm's 7168 MiB plus the VRAM the non-resident experts free. That
is 288 experts (3 stages × (128 − 32)) × 17,739,276 B = 4,872 MiB, so 12,040 MiB.

- [ ] **Step 1: Change the driver.** In `ab_cpu_draft.py`, replace the `"cpu"` entry of `ARMS` with:

```python
    # Each stage's top-32 experts stay on the GPU (resident-top32.json, from the routes arm's sessions 0-7); the
    # other 288 free 4,872 MiB for the target's hot cache.
    "hybrid": {
        "SGLANG_MOE_HOT_GPU_MB": "12040",
        "SGLANG_DSV41_ENABLE_DSPARK_CPU_EXPERTS": "1",
        "SGLANG_DSV41_DSPARK_CPU_EXPERTS_CORES": "18-29",
        "SGLANG_DSV41_DSPARK_DRAFT_RESIDENT_PATH": RESIDENT,
        "EXL3_MOE_CPU_PIN": "0",
    },
```

  Add, next to `SESSIONS`:

```python
RESIDENT = "/data/models/slang/nvfp4-work/cc-expert-prediction/analysis/dsv41-dspark/cpu-draft-routes/resident-top32.json"
```

  In `run()`, add `"--skip", os.environ.get("AB_SKIP", "0"),` to `cmd` after the `"--n", str(n),` pair.

- [ ] **Step 2: Commit, push, and make the resident file** (CPU only)

```bash
git add analysis/dsv41-drive/dspark/ab_cpu_draft.py
git commit -m "bench(dspark): hybrid arm and held-out sessions in the A/B driver

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push origin dsv41-dspark-cpu-draft
ssh divix01 'cd /data/models/slang/nvfp4-work/wt-dspark-cpu && git fetch origin && git checkout --detach origin/dsv41-dspark-cpu-draft && \
  R=/data/models/slang/nvfp4-work/cc-expert-prediction/analysis/dsv41-dspark/cpu-draft-routes && \
  PYTHONPATH=$PWD/python taskset -c 0-63 /data/models/slang/.venv/bin/python analysis/dsv41-drive/dspark/draft_resident_set.py \
  $R/routes.jsonl 32 $R/resident-top32.json; echo EXIT=$?'
```
Expected: `EXIT=0` and three `stage N: 32 resident:` lines, for stages 0, 1 and 2.

- [ ] **Step 3: Smoke both arms** with one held-out session of 16 tokens:

```bash
ssh divix01 'cd /data/models/slang/nvfp4-work/wt-dspark-cpu && \
  OUT=/data/models/slang/nvfp4-work/cc-expert-prediction/analysis/dsv41-dspark/cpu-draft-smoke && \
  AB_SKIP=8 AB_SESSIONS=1 AB_NEW_TOKENS=16 flock /data/models/slang/nvfp4-work/rowimg-disk.lock \
  flock /data/models/slang/nvfp4-work/cc-gpu.lock taskset -c 0-17,30-63 \
  /data/models/slang/.venv/bin/python analysis/dsv41-drive/dspark/ab_cpu_draft.py $OUT resident hybrid; echo EXIT=$?; \
  grep -h "DSpark CPU experts\|Load weight end\|Traceback\|Error" $OUT/*.log | head -20'
```

  The driver's own affinity (`0-17,30-63`) leaves cores 18–29 to the kernel's worker, which binds itself there.

  Pass criteria:
  - `EXIT=0`.
  - The hybrid arm logs `DSpark CPU experts: 3 draft stages on cores [18, …, 29], 12 threads; [96, 96, 96] experts on the CPU`.
  - The hybrid arm's draft `Load weight end` shows about 4.9 GB less GPU memory than the resident arm's.
  - Both arms produce text.
  - Record each arm's wall time from launch to the first session. A much longer hybrid startup means a
    prefill-sized draft forward hit the CPU (Review Focus 1).

  If a gate refuses an override, change exactly the variable its message names. Record the change in this plan,
  then rerun. If the hybrid arm runs out of GPU memory at 12040, lower it by 1024 at a time and record the value that
  launches.

- [ ] **Step 4: Run the A/B** on held-out sessions 8–15 (`AB_SKIP=8`), 128 tokens, EOS honoured. Order the arms
  resident then hybrid, then repeat in reverse into a second directory:

```bash
ssh divix01 'cd /data/models/slang/nvfp4-work/wt-dspark-cpu && A=/data/models/slang/nvfp4-work/cc-expert-prediction/analysis/dsv41-dspark/cpu-draft-ab && \
  export AB_SKIP=8 AB_SESSIONS=8 AB_NEW_TOKENS=128 && \
  flock /data/models/slang/nvfp4-work/rowimg-disk.lock flock /data/models/slang/nvfp4-work/cc-gpu.lock taskset -c 0-17,30-63 \
  /data/models/slang/.venv/bin/python analysis/dsv41-drive/dspark/ab_cpu_draft.py $A/r1 resident hybrid; echo EXIT1=$?; \
  flock /data/models/slang/nvfp4-work/rowimg-disk.lock flock /data/models/slang/nvfp4-work/cc-gpu.lock taskset -c 0-17,30-63 \
  /data/models/slang/.venv/bin/python analysis/dsv41-drive/dspark/ab_cpu_draft.py $A/r2 hybrid resident; echo EXIT2=$?'
```
Expected: `EXIT1=0` and `EXIT2=0`.

- [ ] **Step 5: Analyze.** From each `<arm>.json` `per_session`, report:
  - the median of per-session `decode_tok_s` per arm and repeat;
  - the median paired ratio hybrid/resident across the 16 session pairs, and how many pairs the hybrid arm wins;
  - the median TTFT (`ttft_s`) per arm. More than 20% worse in the hybrid arm is a blocker (Review Focus 1);
  - the mean accept length (`completion_tokens / spec_verify_ct`) per arm. Only the draft's numerics differ, so it
    must agree within noise. Report a large drop rather than explain it away.
  - From the hybrid arm's last `DSpark CPU experts:` line: the share of stage calls without CPU work, and the mean
    and p90 ms per CPU call. Compare them with the Evidence table's held-out 1.43 ms per draft step (Review Focus 5).

  The result is a **win** if the median paired ratio is above 1 and the hybrid arm wins at least 12 of 16 pairs. It
  is a **loss** if the ratio is below 1 with at least 12 of 16 losses. Anything else is **inconclusive**: report the
  numbers and stop. Whatever the outcome, state that both arms are far below the production non-spec path
  (~13.5 tok/s, §30.1). This A/B measures the draft's VRAM trade, not whether DSpark ships.

- [ ] **Step 6: Record in the doc.** Add `### 33.4 The draft's experts: GPU resident set plus CPU (2026-10-…)` to
  `DSV41_REFERENCE.md`, after §33.3. It contains:
  - the benchmark table and the plain-vs-optimized kernel finding (superseded plan, Results);
  - the routes probe's per-stage union and the held-out hybrid table (Evidence, above);
  - Task 6's twelve `rel_l2` values;
  - the A/B numbers from Step 5, with the exact commands and output directories;
  - the verdict;
  - the master bug found on the way: `ExpertPinnedHostCacheManager.from_model` checked NUMA capacity for a model
    with no streamed experts, fixed in `d40d781771`.

  Commit:

```bash
git add DSV41_REFERENCE.md docs/superpowers/plans/2026-10-02-dsv41-dspark-hybrid-draft.md
git commit -m "docs(dsv41): section 33.4, DSpark hybrid draft experts

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

## Results

(Filled in by Tasks 6 and 7.)
