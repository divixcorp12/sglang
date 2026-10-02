# DSV4.1 DSpark: the draft's routed experts on the CPU (Implementation Plan)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Hold the DSpark draft's routed experts in host RAM and compute them with the CPU expert kernel, so the
draft's 6.75 GiB of VRAM returns to the target's hot cache. Measure whether that makes eager DSpark faster.

**Architecture:**
- **Unchanged:** the target model, and DSpark itself (eager, D1). Only the draft stages' `FusedMoE` changes.
- **Weights on the CPU:** when `SGLANG_DSV41_ENABLE_DSPARK_CPU_EXPERTS=1`, `Exl3MoEMethod` allocates a draft
  stage's expert parameters on the CPU.
- **Compute:** a new `cpu_experts/draft.py` registers the stages with a `CpuExpertPool` and computes each stage's
  routed experts on one dedicated worker thread. It goes through the kernel's multi-row torch op
  (`ext.exl3_moe_cpu_forward`, rows = the draft block's 6 tokens).
- **Fused shared expert:** if the draft folds one in (ids ≥ `n_routed`), it stays on the GPU and runs while the CPU
  job runs.
- **Order:** a microbenchmark goes first and gates the code. A served A/B against the resident draft goes last.

**Tech Stack:**
- SGLang fork (`master`), Python/PyTorch.
- The vendored exllamav3 CPU kernel (`layers/quantization/exl3_cpu/optimized/moe_mul1.cpp`, via `exl3_ext()`).
- pytest.
- divix01 (RTX 5090, 2-socket Xeon) for the benchmark, the GPU test and the A/B.

**Spec:** `DSV41_REFERENCE.md`:
- §33: DSpark's D1 state and the graph analysis;
- §30, §30.4: the CPU experts, their measured cost and NUMA cost;
- §28: the CPU kernel's throughput;
- §26.2 item 3: the hot-cache value of 1 GiB.

There is no separate spec; the design and its estimate are recorded here under "Why".

## Why (the estimate this plan tests)

- **The gain:** the draft is resident today: 3 stages × 128 EXL3 4-bit experts = 6.75 GiB, about 544 target hot
  slots. Each +1 GiB of hot cache saves 2–2.7 misses per token at ~1 ms each (§26.2). The draft's VRAM is therefore
  worth ~13–18 ms per target forward [estimate].
- **The cost:**
  - A draft forward runs 6 tokens (`dspark_draft.py:96`, `query_token_num = gamma + 1`) through 3 stages in series.
    Each token routes to top-3 of 128 experts, so at most 18 (token, expert) pairs per stage.
  - A draft expert is 17.7 MB, against the target's 13.3 MB (§10). The kernel is bandwidth-bound at ~0.5–0.65 ms per
    target expert (§28.1, §30.1), so ~0.85 ms per draft expert pass.
  - A pass serves at most 2 rows (`moe_mul1.cpp:84,94-96`: `CHUNK_M = 2` in the residual build).
  - Estimate: 10–30 ms per draft step. It is serial, before each verify.
- **The cores are free:** the draft and the verify are serial, and CPU experts are refused with speculation (§30.1),
  so the cores have no other CPU-expert work during DSpark.
- **The win, if there is one, is at most this:** this does not touch the verify forward's union cost (§33.3). It
  removes the draft's VRAM cost only.

## Global Constraints

- **Branch and worktree:** branch `dsv41-dspark-cpu-draft` from `master`, in a laptop worktree. Push to `origin` only.
  On divix01, run in a private worktree at the pushed commit (`.claude/rules/divix01-run-protocol.md`), with
  `PYTHONPATH=$PWD/python`. Print `sglang.__file__` once per worktree and check it.
- **Git:** no rebase, amend, force-push or `git stash`. Stage files by name. End commit messages with
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- **CPU jobs:** `taskset -c 0-63`, `OMP_NUM_THREADS=8`. The CPU-expert kernel itself runs on cores `18-29`
  (`SGLANG_DSV41_CPU_EXPERTS_CORES` of §30.1).
- **GPU jobs:** use `flock /data/models/slang/nvfp4-work/cc-gpu.lock taskset -c 32-63 …`. A job that also streams
  from NVMe takes `rowimg-disk.lock` first.
- **Production:** never start or stop production without the owner's OK. Tasks 6–7 need the GPU free.
- **Test status:** read pytest's own status through `PIPESTATUS[0]`, never a pipeline's.
- **Required reading:**
  - `env-var-conventions` before touching `environ.py`; done for this plan, and the names below follow it;
  - `large-class-style` if anything in `model_runner.py` is touched (nothing here should be).
- **Kernel environment:** set `EXL3_MOE_CPU_PIN=0` for every process that runs the CPU kernel. The trait's
  `check_environment` refuses otherwise.
- **No bulk reads** of `/mnt/nvme1` or `/mnt/nvme2`.
- **Units:** ms per draft *stage call* (one stage, one forward), ms per draft *step* (3 stage calls), tok/s per
  session.

## Review Focus

1. **A draft block whose 6 rows route to the same expert.** The union shrinks, and the kernel must still give the
   same per-row sums as the GPU path. Task 6's GPU test pins it (`pattern="shared"`).
2. **A fused shared expert** (`disable_shared_experts_fusion` off). Ids ≥ `n_routed` must never reach the CPU kernel
   (out of range there), and must still be added on the GPU. Tasks 4 and 6 pin it.
3. **A draft forward over a prefill chunk** (hundreds of rows, not 6), if the draft MoE runs on extend or in the
   startup dummy run. CPU cost scales with rows, so this would inflate TTFT and startup. Task 7 Steps 3 and 5 compare
   startup time and TTFT between arms. A -1 route (padding) is pinned by Task 2's test.
4. **The scheduler thread's CPU affinity.** Binding a kernel worker must never pin the caller, or the scheduler would
   run on the expert cores. Task 4 pins it.
5. **A process that turns the flag on without DSpark, or without cores.** It must be refused at launch, not fail at
   the first forward. Task 3 pins it.

---

### Task 1: Microbenchmark the kernel on 4-bit draft-shaped experts at 1–6 rows (the go/no-go gate)

**Files:**
- Create: `analysis/dsv41-drive/cpu-experts/draft_bench.py`
- Output (divix01, not committed): `analysis/dsv41-drive/cpu-experts/draft_bench_results.jsonl`

**Interfaces:**
- Consumes: `exl3_ext()` (`python/sglang/srt/layers/quantization/exl3_ext.py:129`). Its ops:
  - `ext.exl3_moe_cpu_make_layer(gate_t, gate_u, gate_v, up_t, up_u, up_v, down_t, down_u, down_v, [], [], [], activation, act_limit, swizzled) -> int`, called exactly as `test/manual/dsv41/test_cpu_expert_pool_exl3.py::_direct_layer`;
  - `ext.exl3_moe_cpu_forward(handle, x fp16 [m,H], selected int64 [m,k], weights fp16 [m,k], out fp32 [m,H], threads)` (`moe_mul1.cpp:3192-3230`).
- Produces: one JSON line per (bits, threads, rows, pattern) cell. The gate decision is recorded in this plan's
  "Results" section.

Random weights are valid for timing: the kernel's cost is the bytes it reads, not their values. Shapes are DSV4.1's
expert shapes, H = 5120 and I = 2304 (`analysis/dsv41-drive/cpu-experts/p0_real.py:35`). A draft expert at 4 bits is
17.7 MB, matching DSV41_REFERENCE's measured "one DSpark draft expert is 17,739,276 B". 128 experts × 17.7 MB = 2.26 GB, far past the LLC, so a random expert per call is a cold read.

- [ ] **Step 1: Write the script**

```python
"""DSpark draft-shaped experts through the CPU expert kernel: ms per stage call at 1-6 rows.

Plan: docs/superpowers/plans/2026-10-02-dsv41-dspark-cpu-draft.md, Task 1. Random EXL3 weights at DSV4.1's expert
shape (H 5120, I 2304) and BITS bits, 128 experts per layer as the draft's stages hold them. Each call routes ROWS rows
to top-3 experts:
  independent  every row draws its own 3 distinct experts (the largest union, up to 3*ROWS)
  shared       every row uses the same 3 experts (union 3)
Usage (divix01, repo root, PYTHONPATH=$PWD/python):
  EXL3_MOE_CPU_PIN=0 numactl --membind=1 taskset -c 18-29 python analysis/dsv41-drive/cpu-experts/draft_bench.py \
      BITS THREADS ROWS PATTERNS TAG
BITS, THREADS, ROWS and PATTERNS are comma lists. Results append to draft_bench_results.jsonl beside this file.
"""

import json
import os
import statistics
import sys
import time

import torch

os.environ.setdefault("EXL3_MOE_CPU_PIN", "0")

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "benchmarks", "dsv41_baseline"))
import arm_env  # noqa: E402

for key in ("SGLANG_EXL3_SRC", "SGLANG_EXL3_BUILD_DIR"):
    os.environ.setdefault(key, arm_env.base_env()[key])

H, I, E, TOPK = 5120, 2304, 128, 3
LIMIT = 10.0
WARMUP, CALLS = 5, 60
RESULTS = os.path.join(HERE, "draft_bench_results.jsonl")


def slabs(bits: int, gen: torch.Generator) -> dict[str, torch.Tensor]:
    def trellis(*shape):
        return torch.randint(-32768, 32767, shape, generator=gen, dtype=torch.int16)

    def signs(*shape):
        return (torch.randint(0, 2, shape, generator=gen) * 2 - 1).half()

    return {
        "w13_trellis": trellis(E, 2, H // 16, I // 16, 16 * bits),
        "w13_suh": signs(E, 2, H),
        "w13_svh": signs(E, 2, I),
        "w2_trellis": trellis(E, I // 16, H // 16, 16 * bits),
        "w2_suh": signs(E, I),
        "w2_svh": signs(E, H),
    }


def make_layer(ext, s) -> int:
    rows = range(E)
    return ext.exl3_moe_cpu_make_layer(
        [s["w13_trellis"][i, 0] for i in rows],
        [s["w13_suh"][i, 0] for i in rows],
        [s["w13_svh"][i, 0] for i in rows],
        [s["w13_trellis"][i, 1] for i in rows],
        [s["w13_suh"][i, 1] for i in rows],
        [s["w13_svh"][i, 1] for i in rows],
        [s["w2_trellis"][i] for i in rows],
        [s["w2_suh"][i] for i in rows],
        [s["w2_svh"][i] for i in rows],
        [],
        [],
        [],
        0,
        LIMIT,
        0,
    )


def routes(rows: int, pattern: str, gen: torch.Generator) -> torch.Tensor:
    if pattern == "shared":
        return torch.randperm(E, generator=gen)[:TOPK].repeat(rows, 1)
    if pattern == "independent":
        return torch.stack([torch.randperm(E, generator=gen)[:TOPK] for _ in range(rows)])
    raise SystemExit(f"unknown pattern {pattern}")


def weight_passes(ids: torch.Tensor) -> tuple[int, int]:
    """(union, sum over the union of ceil(t/2)): the kernel reads an expert once per 2 rows that route to it."""
    counts = torch.bincount(ids.reshape(-1), minlength=E)
    used = counts[counts > 0]
    return int(used.numel()), int(((used + 1) // 2).sum())


def cell(ext, handle, bits, threads, rows, pattern, gen):
    x = (torch.randn(rows, H, generator=gen) * 0.5).half()
    w = torch.full((rows, TOPK), 1.0 / TOPK).half()
    out = torch.empty(rows, H, dtype=torch.float32)
    ms, unions, passes = [], [], []
    for i in range(WARMUP + CALLS):
        ids = routes(rows, pattern, gen)
        start = time.perf_counter()
        ext.exl3_moe_cpu_forward(handle, x, ids, w, out, threads)
        elapsed = (time.perf_counter() - start) * 1e3
        if i >= WARMUP:
            u, p = weight_passes(ids)
            ms.append(elapsed)
            unions.append(u)
            passes.append(p)
    median = statistics.median(ms)
    mean_passes = statistics.mean(passes)
    return {
        "bits": bits,
        "threads": threads,
        "rows": rows,
        "pattern": pattern,
        "ms_median": round(median, 3),
        "ms_p90": round(sorted(ms)[int(0.9 * len(ms))], 3),
        "union_mean": round(statistics.mean(unions), 2),
        "passes_mean": round(mean_passes, 2),
        "ms_per_pass": round(median / mean_passes, 3),
        "draft_step_ms": round(3 * median, 2),
    }


def main():
    bits_list, threads_list, rows_list = (
        [int(v) for v in arg.split(",")] for arg in sys.argv[1:4]
    )
    patterns, tag = sys.argv[4].split(","), sys.argv[5]
    from sglang.srt.layers.quantization.exl3_ext import exl3_ext

    ext = exl3_ext()
    gen = torch.Generator().manual_seed(20261002)
    for bits in bits_list:
        handle = make_layer(ext, slabs(bits, gen))
        try:
            for threads in threads_list:
                for rows in rows_list:
                    for pattern in patterns:
                        rec = {
                            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                            "tag": tag,
                            "affinity": sorted(os.sched_getaffinity(0)),
                            **cell(ext, handle, bits, threads, rows, pattern, gen),
                        }
                        line = json.dumps(rec)
                        print(line, flush=True)
                        with open(RESULTS, "a") as f:
                            f.write(line + "\n")
        finally:
            ext.exl3_moe_cpu_free_layer(handle)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit, push, and set up the divix01 worktree**

```bash
git add analysis/dsv41-drive/cpu-experts/draft_bench.py
git commit -m "bench(cpu-experts): DSpark draft-shaped experts at 1-6 rows

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push origin dsv41-dspark-cpu-draft
ssh divix01 'git -C /data/models/slang/sglang fetch origin \
  && git -C /data/models/slang/sglang worktree add --detach /data/models/slang/nvfp4-work/wt-dspark-cpu origin/dsv41-dspark-cpu-draft \
  && git -C /data/models/slang/nvfp4-work/wt-dspark-cpu log -1 --oneline'
```

- [ ] **Step 3: Run it** (CPU only, but it loads the cores the GPU job's host threads use, so take the GPU lock)

```bash
ssh divix01 'cd /data/models/slang/nvfp4-work/wt-dspark-cpu && export PYTHONPATH=$PWD/python EXL3_MOE_CPU_PIN=0 && \
  flock /data/models/slang/nvfp4-work/cc-gpu.lock numactl --membind=1 taskset -c 18-29 \
  /data/models/slang/.venv/bin/python analysis/dsv41-drive/cpu-experts/draft_bench.py \
  3,4 8,12 1,2,3,6 independent,shared node1-r1; echo EXIT=$?'
```

Expected: `EXIT=0` and 32 JSON lines.
- **Sanity check:** the `bits=3, rows=1, independent` cells must sit near §28.1/§30.4's 0.49–0.58 ms per expert
  (`ms_per_pass`). If they are off by more than 2×, the bench is measuring something else. Stop and find out why
  before reading the 4-bit cells.
- **Failure mode:** if `make_layer` raises on `bits=4`, the kernel does not take the draft's bitrate. Stop and
  report; the plan does not continue.

- [ ] **Step 4: Decide.** Let `P` be `draft_step_ms` of the cell `bits=4, rows=6, pattern=independent` at the faster
  thread count. That is the worst-case union. `shared` is the best case and brackets it.
  - **Go** if `P ≤ 18` ms, the top of the ~13–18 ms the freed VRAM is worth per target forward.
  - **Ask the owner** if `18 < P ≤ 30`. The real union lies between `shared` and `independent`, and the verify
    forward's misses may be worth more than one decode's.
  - **Stop** if `P > 30` ms. Record the numbers in DSV41_REFERENCE §33 (as in Task 7 Step 6) and end the plan.

  Write the cells and the verdict into this plan under "Results" and commit:

```bash
git add docs/superpowers/plans/2026-10-02-dsv41-dspark-cpu-draft.md
git commit -m "plan(dspark-cpu-draft): Task 1 benchmark results and verdict

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: `CpuExpertPool.compute_rows`: the pool for more than one row

**Files:**
- Modify: `python/sglang/srt/layers/moe/cpu_experts/pool.py`: add `compute_rows` after `compute`, and correct the
  `forward` protocol docstring to `[m, k]`.
- Modify: `python/sglang/srt/layers/moe/cpu_experts/exl3.py`: `Exl3CpuQuantTrait.forward` docstring only (the op
  already takes m rows).
- Test: `test/registered/unit/kernels/test_cpu_expert_pool.py`

**Interfaces:**
- Produces: `CpuExpertPool.compute_rows(layer: int, host_slots: torch.Tensor, weights: torch.Tensor, x: torch.Tensor, out: torch.Tensor) -> None`, where:
  - `x` is `[m, H]` and `out` is `[m, H]`;
  - `host_slots` is int64 `[m, k]`, with -1 skipped;
  - `weights` is `[m, k]`;
  - dtypes are the trait's.
  It overwrites `out`. `compute` (batch 1) is unchanged: the native CPU-expert thread's contract is still one row.

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

- [ ] **Step 2: Run them; they must fail** (on divix01 after pushing, or on any Linux box: `os.sched_getaffinity` does
  not exist on macOS)

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

Also change these docstrings:
- `CpuExpertQuantTrait.forward`: "(int64 ``[1, k]``)" becomes "(int64 ``[m, k]``; ``compute`` passes one row)".
- `Exl3CpuQuantTrait.forward` in `exl3.py`: "Overwrite ``out`` ``[m, H]`` with the routed sums of ``slots`` ``[m, k]`` over layer ``handle``."

- [ ] **Step 4: Run the whole file; it must pass** (the new tests and every existing one)

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

### Task 3: Environment variables and the launch gate

**Files:**
- Modify: `python/sglang/srt/environ.py`: add three entries directly after `SGLANG_DSV41_CPU_EXPERTS_MISSES`, the
  last entry of the CPU-experts block (around line 1930).
- Modify: `python/sglang/srt/arg_groups/expert_stream_requirements_exl3.py`: add a rule directly before
  `cpu_experts = envs.SGLANG_DSV41_CPU_EXPERTS.get()` (around line 89).
- Test: `test/registered/unit/test_expert_stream_requirements_exl3.py`

**Interfaces:**
- Produces:
  - `envs.SGLANG_DSV41_ENABLE_DSPARK_CPU_EXPERTS` (`EnvBool(False)`);
  - `envs.SGLANG_DSV41_DSPARK_CPU_EXPERTS_CORES` (`EnvStr("")`);
  - `envs.SGLANG_DSV41_DSPARK_CPU_EXPERTS_THREADS` (`EnvInt(0)`).

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
```

- [ ] **Step 2: Run them; they must fail**

```bash
PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 taskset -c 0-63 /data/models/slang/.venv/bin/python -m pytest \
  test/registered/unit/test_expert_stream_requirements_exl3.py -q -p no:randomly -k dspark_cpu 2>&1 | tail -3; echo "EXIT=${PIPESTATUS[0]}"
```
Expected: `EXIT=1`, with `AttributeError` on `envs.SGLANG_DSV41_ENABLE_DSPARK_CPU_EXPERTS`.

- [ ] **Step 3: Add the variables** in `environ.py`, after `SGLANG_DSV41_CPU_EXPERTS_MISSES`:

```python
    # DSpark draft experts on the CPU (plan 2026-10-02-dsv41-dspark-cpu-draft): the draft stages' routed experts stay
    # in host RAM and the CPU expert kernel computes them, eagerly, so their VRAM goes to the target's hot cache. A
    # fused shared expert stays on the GPU. Needs --speculative-algorithm DSPARK. Off by default.
    SGLANG_DSV41_ENABLE_DSPARK_CPU_EXPERTS = EnvBool(False)
    # Cores of the draft's CPU expert pool, as a taskset list ("18-29"). At least two.
    SGLANG_DSV41_DSPARK_CPU_EXPERTS_CORES = EnvStr("")
    # Worker threads of the draft's CPU expert pool, at most one per core. 0 takes one per core.
    SGLANG_DSV41_DSPARK_CPU_EXPERTS_THREADS = EnvInt(0)
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
```

At the top of that module, add `from sglang.srt.layers.moe.cpu_experts.policy import parse_core_list`. If the module
already imports from `policy`, extend that import instead.

- [ ] **Step 5: Run the whole gate file; it must pass**

```bash
PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 taskset -c 0-63 /data/models/slang/.venv/bin/python -m pytest \
  test/registered/unit/test_expert_stream_requirements_exl3.py -q -p no:randomly 2>&1 | tail -3; echo "EXIT=${PIPESTATUS[0]}"
```
Expected: `EXIT=0`.

- [ ] **Step 6: Commit**

```bash
git add python/sglang/srt/environ.py python/sglang/srt/arg_groups/expert_stream_requirements_exl3.py \
  test/registered/unit/test_expert_stream_requirements_exl3.py
git commit -m "env(dspark): SGLANG_DSV41_ENABLE_DSPARK_CPU_EXPERTS, its cores and threads, and their launch rule

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: `cpu_experts/draft.py`: the draft's CPU runtime, worker thread and stats

**Files:**
- Create: `python/sglang/srt/layers/moe/cpu_experts/draft.py`
- Test: `test/registered/unit/kernels/test_dspark_draft_cpu_experts.py`

**Interfaces:**
- Consumes: `CpuExpertPool.compute_rows` (Task 2); `cpu_trait_for("exl3")` (`cpu_experts/service.py:38`, which
  returns an `Exl3CpuQuantTrait` with `act_limit=None`); `parse_core_list` (`cpu_experts/policy.py:19`); the Task 3
  envs.
- Produces:
  - `DraftLayer(slabs: Mapping[str, torch.Tensor], n_routed: int, act_limit: Optional[float])`, a frozen dataclass.
  - `DraftCpuExperts(trait, layers: Mapping[int, DraftLayer], *, cores: Sequence[int], threads: int, log_every: int = 300)`, with:
    - `.submit(key: int, x16: Tensor[m,H] fp16 cpu, ids: Tensor[m,k] int64 cpu, w16: Tensor[m,k] fp16 cpu) -> concurrent.futures.Future[Tensor[m,H] fp32 cpu]`;
    - `.stats: DraftCpuStats`;
    - `.close()`.
  - `DraftCpuStats(capacities: Mapping[int, int], log_every: int)`, with `.record(key, slots, seconds)`,
    `.summary() -> str` and `.coverage(n: int) -> float`.
  - `DraftCpuExpertsRegistry()`, with:
    - `.register(slabs, n_routed, act_limit) -> int` (the key);
    - `.runtime() -> DraftCpuExperts` (built once, from the Task 3 envs, on first call);
    - `.close()`.
  - Module-level `DRAFT_CPU_EXPERTS = DraftCpuExpertsRegistry()`.

- [ ] **Step 1: Write the failing tests** in `test/registered/unit/kernels/test_dspark_draft_cpu_experts.py`:

```python
"""The DSpark draft's CPU expert runtime: masking, the worker thread, stats and the registry (fake kernel)."""

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

E, H = 5, 4  # 4 routed experts plus a fused shared one (id 4)


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


def _runtime(trait):
    layer = DraftLayer({"w13_trellis": torch.zeros(E, 2, dtype=torch.int16)}, n_routed=4, act_limit=10.0)
    return DraftCpuExperts(trait, {0: layer}, cores=_cores(), threads=2)


def test_fused_shared_ids_never_reach_the_cpu():
    trait = _Trait()
    experts = _runtime(trait)
    try:
        ids = torch.tensor([[0, 1, 4], [2, 4, -1]], dtype=torch.int64)
        out = experts.submit(0, torch.ones(2, H, dtype=torch.float16), ids, torch.full((2, 3), 0.5, dtype=torch.float16)).result()
    finally:
        experts.close()
    slots, _ = trait.seen[0]
    assert slots.tolist() == [[0, 1, -1], [2, -1, -1]]
    assert out.tolist() == [[1.0] * H, [0.5] * H]


def test_the_kernel_runs_on_the_worker_and_the_caller_keeps_its_affinity():
    trait = _Trait()
    before = os.sched_getaffinity(0)
    experts = _runtime(trait)
    try:
        experts.submit(
            0,
            torch.ones(1, H, dtype=torch.float16),
            torch.zeros(1, 3, dtype=torch.int64),
            torch.ones(1, 3, dtype=torch.float16),
        ).result()
    finally:
        experts.close()
    assert trait.seen[0][1] != threading.get_native_id()
    assert os.sched_getaffinity(0) == before


def test_stats_count_union_passes_and_coverage():
    stats = DraftCpuStats({0: 8}, log_every=1000)
    # 6 rows: expert 0 routed by all 6 (3 passes), expert 1 by 2 (1 pass), expert 2 by 1 (1 pass)
    slots = torch.tensor([[0, 1, -1]] * 2 + [[0, 2, -1]] + [[0, -1, -1]] * 3, dtype=torch.int64)
    stats.record(0, slots, 0.010)
    assert stats.unions == [3] and stats.passes == [5]
    assert stats.coverage(1) == pytest.approx(6 / 9)
    assert "1 stage calls" in stats.summary() and "union 3.0" in stats.summary()


def test_the_registry_builds_once_from_the_envs_and_refuses_mixed_limits(monkeypatch):
    from sglang.srt.environ import envs
    from sglang.srt.layers.moe.cpu_experts import draft

    built = []
    monkeypatch.setattr(draft, "cpu_trait_for", lambda key: built.append(key) or _Trait())
    registry = DraftCpuExpertsRegistry()
    slabs = {"w13_trellis": torch.zeros(E, 2, dtype=torch.int16)}
    assert registry.register(slabs, 4, 10.0) == 0
    assert registry.register(slabs, 4, 10.0) == 1
    with pytest.raises(ValueError, match="activation limit"):
        registry.register(slabs, 4, 7.0)
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
"""The DSpark draft's routed experts on the CPU (eager).

With ``SGLANG_DSV41_ENABLE_DSPARK_CPU_EXPERTS`` each draft stage's ``FusedMoE`` keeps its routed experts in host
RAM instead of VRAM (6.75 GiB for V4.1's three stages, DSV41_REFERENCE.md section 33), and this module computes them
with the CPU expert kernel. The stages register here as they finish loading, and the pool is built on the first
forward.

One worker thread runs the kernel. It is bound to the pool's cores, so the caller (the scheduler) never is. Ids at
or above a stage's ``n_routed`` are a fused shared expert, which the caller runs on the GPU; they reach the kernel as
-1, which it skips.
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
    """One draft stage: its expert tensors in the pinned tier's slab layout, routed count and SwiGLU clamp."""

    slabs: Mapping[str, torch.Tensor]
    n_routed: int
    act_limit: Optional[float]


class DraftCpuStats:
    """Per-stage-call cost and routing of the draft's CPU experts, logged every ``log_every`` calls.

    ``passes`` counts weight reads: the kernel reads an expert once per two rows routed to it. ``coverage(n)`` is
    the share of routes on each stage's ``n`` most-routed experts, averaged over stages: the skew a small VRAM draft
    cache would exploit.
    """

    def __init__(self, capacities: Mapping[int, int], log_every: int):
        self.log_every = log_every
        self.route_counts = {key: torch.zeros(c, dtype=torch.int64) for key, c in capacities.items()}
        self.calls = 0
        self.seconds: list[float] = []
        self.unions: list[int] = []
        self.passes: list[int] = []

    def record(self, key: int, slots: torch.Tensor, seconds: float) -> None:
        counts = torch.bincount(slots[slots >= 0], minlength=len(self.route_counts[key]))
        self.route_counts[key] += counts
        used = counts[counts > 0]
        self.unions.append(int(used.numel()))
        self.passes.append(int(((used + 1) // 2).sum()))
        self.seconds.append(seconds)
        self.calls += 1
        if self.calls % self.log_every == 0:
            logger.info(self.summary())
            self.seconds, self.unions, self.passes = [], [], []

    def coverage(self, n: int) -> float:
        shares = []
        for counts in self.route_counts.values():
            total = int(counts.sum())
            if total:
                shares.append(int(counts.sort(descending=True).values[:n].sum()) / total)
        return sum(shares) / len(shares) if shares else 0.0

    def summary(self) -> str:
        if not self.seconds:
            return f"DSpark CPU experts: {self.calls} stage calls"
        ms = sorted(s * 1e3 for s in self.seconds)
        window = len(ms)
        return (
            f"DSpark CPU experts: {self.calls} stage calls; last {window}: "
            f"{sum(ms) / window:.2f} ms mean, {ms[int(0.9 * (window - 1))]:.2f} ms p90 per stage call, "
            f"union {sum(self.unions) / window:.1f}, weight passes {sum(self.passes) / window:.1f}; "
            "routes on the top 8/16/32 experts per stage: "
            + "/".join(f"{100 * self.coverage(n):.0f}%" for n in (8, 16, 32))
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
        self.n_routed = {key: layer.n_routed for key, layer in layers.items()}
        self.stats = DraftCpuStats(self.pool.capacity, log_every)
        self._worker = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="dspark-cpu-experts",
            initializer=self.pool.bind_current_thread,
        )

    def submit(self, key: int, x16: torch.Tensor, ids: torch.Tensor, w16: torch.Tensor) -> Future:
        """Start stage ``key``'s routed experts over ``x16`` ``[m, H]``; the future yields fp32 ``[m, H]``."""
        return self._worker.submit(self._run, key, x16, ids, w16)

    def _run(self, key: int, x16: torch.Tensor, ids: torch.Tensor, w16: torch.Tensor) -> torch.Tensor:
        slots = ids.masked_fill(ids >= self.n_routed[key], -1)
        out = torch.empty(x16.shape, dtype=torch.float32)
        start = time.perf_counter()
        self.pool.compute_rows(key, slots, w16, x16, out)
        self.stats.record(key, slots, time.perf_counter() - start)
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

    def register(self, slabs: Mapping[str, torch.Tensor], n_routed: int, act_limit: Optional[float]) -> int:
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
            self._layers[key] = DraftLayer(slabs, n_routed, act_limit)
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
                    "DSpark CPU experts: %d draft stages on cores %s, %d threads",
                    len(self._layers),
                    cores,
                    threads,
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
git commit -m "cpu-experts: the DSpark draft's CPU runtime (worker thread, fused-shared masking, stats)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Route the draft stages' `Exl3MoEMethod` to the CPU

**Files:**
- Modify: `python/sglang/srt/models/deepseek_v4_exl3_weights.py`: add `DSPARK_DRAFT_EXPERT_MODULE_RE` and
  `is_dspark_draft_expert_module` after `is_streamed_expert_module`.
- Modify: `python/sglang/srt/layers/quantization/exl3.py`:
  - `Exl3Config.get_quant_method` (around line 109);
  - `Exl3MoEMethod.__init__`, `create_weights`, `process_weights_after_loading` and `apply`;
  - new methods `_attach_cpu_draft` and `_apply_cpu_draft`.
- Test: `test/registered/unit/layers/quantization/test_exl3_stream_scope.py` (the module regex) and
  `test/registered/unit/layers/quantization/test_exl3_moe_method.py` (CPU allocation and registration).

**Interfaces:**
- Consumes: `DRAFT_CPU_EXPERTS.register(...)`, `.runtime().submit(...)` (Task 4); `exl3_moe_accumulate`
  (`exl3_ops.py:214`); `EXL3_STREAMED_NAMES` (`layers/moe/exl3_expert_format.py:43`);
  `layer.moe_runner_config.swiglu_limit` (set in `FusedMoE.__init__`, `fused_moe_triton/layer.py:447`, before the
  weights load).
- Produces:
  - `Exl3MoEMethod(config, *, streamed: bool, cpu_draft: bool = False)`;
  - on a CPU-draft layer: `layer.exl3_cpu_draft_key: int`, `layer.exl3_gpu_shared_w13: dict[int, tuple[Exl3Tensors, Exl3Tensors]]`, `layer.exl3_gpu_shared_w2: dict[int, Exl3Tensors]`.

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

Append to `test_exl3_moe_method.py`:

```python
from types import SimpleNamespace


def _cpu_draft_moe(monkeypatch, fused_shared=0):
    from sglang.srt.layers.moe.cpu_experts import draft

    registered = []
    registry = SimpleNamespace(register=lambda slabs, n_routed, limit: registered.append((slabs, n_routed, limit)) or 0)
    monkeypatch.setattr(draft, "DRAFT_CPU_EXPERTS", registry)
    layer = nn.Module()
    layer.num_experts = E
    layer.num_fused_shared_experts = fused_shared
    layer.moe_runner_config = SimpleNamespace(swiglu_limit=10.0)
    method = Exl3MoEMethod(Exl3Config.from_config(CFG), streamed=False, cpu_draft=True)
    method.create_weights(layer, E, HIDDEN, INTER, torch.bfloat16)
    return layer, method, registered


def test_a_cpu_draft_layer_loads_its_experts_into_host_memory(monkeypatch):
    with torch.device("meta"):  # stands in for the loader's CUDA default device
        layer, method, registered = _cpu_draft_moe(monkeypatch)
    _load_all(layer)
    method.process_weights_after_loading(layer)
    assert layer.w13_trellis.device.type == "cpu" and layer.w2_svh.device.type == "cpu"
    (slabs, n_routed, limit), = registered
    assert n_routed == E and limit == 10.0
    assert slabs["w13_trellis"].data_ptr() == layer.w13_trellis.data_ptr()
    assert layer.exl3_cpu_draft_key == 0 and layer.exl3_gpu_shared_w2 == {}


def test_a_fused_shared_expert_is_kept_for_the_gpu(monkeypatch):
    layer, method, registered = _cpu_draft_moe(monkeypatch, fused_shared=1)
    _load_all(layer)
    method.process_weights_after_loading(layer)
    assert registered[0][1] == E - 1
    assert sorted(layer.exl3_gpu_shared_w2) == [E - 1]
    gate, up = layer.exl3_gpu_shared_w13[E - 1]
    assert int(gate.trellis[0, 0, 0]) == 10 * (E - 1) + 1


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
# The DSpark draft's stage MoEs: "stages.<S>.mlp.experts" (empty root prefix) or "model.stages.<S>.mlp.experts"
# (see ROUTED_EXPERT_MODULE_RE above).
DSPARK_DRAFT_EXPERT_MODULE_RE = re.compile(r"^(?:model\.)?stages\.\d+\.mlp\.experts$")


def is_dspark_draft_expert_module(prefix: str) -> bool:
    """True for a DSpark draft stage's routed-expert FusedMoE."""
    return DSPARK_DRAFT_EXPERT_MODULE_RE.match(prefix) is not None
```

- [ ] **Step 4: Wire `Exl3MoEMethod`** in `exl3.py`.

  **4a.** In `Exl3Config.get_quant_method`, replace the `FusedMoE` branch:

```python
        if isinstance(layer, FusedMoE):
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

  **4b.** Change `Exl3MoEMethod.__init__` to take the flag:

```python
    def __init__(self, config: Exl3Config, *, streamed: bool, cpu_draft: bool = False):
        self.config = config
        self.streamed = streamed
        # A DSpark draft stage whose routed experts live in host RAM (cpu_experts/draft.py).
        self.cpu_draft = cpu_draft
        self.moe_runner_config = None
        self.cast_fusion = envs.SGLANG_DSV41_ENABLE_EXL3_CAST_FUSION.get()
        self.route_plan = envs.SGLANG_DSV41_ENABLE_PREFILL_ROUTE_PLAN.get()
```

  **4c.** In `create_weights`, the parameter loop makes the placeholder on the CPU for a CPU draft. `_materialize`
  allocates on `param.device`, so the experts never touch VRAM:

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
        """Register the routed experts with the draft's CPU runtime; keep fused shared experts for the GPU.

        The parameters are the pinned tier's slab layout ([expert, part, ...]), so they register as they are.
        """
        from dataclasses import replace

        from sglang.srt.layers.moe.cpu_experts import draft

        n_routed = layer.exl3_num_experts - getattr(layer, "num_fused_shared_experts", 0)
        slabs = {name: getattr(layer, name).data for name in EXL3_STREAMED_NAMES}
        layer.exl3_cpu_draft_key = draft.DRAFT_CPU_EXPERTS.register(
            slabs, n_routed, layer.moe_runner_config.swiglu_limit
        )
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        def to_device(t: Exl3Tensors) -> Exl3Tensors:
            return replace(t, trellis=t.trellis.to(device), suh=t.suh.to(device), svh=t.svh.to(device))

        shared = range(n_routed, layer.exl3_num_experts)
        layer.exl3_gpu_shared_w13 = {e: tuple(to_device(t) for t in layer.exl3_w13[e]) for e in shared}
        layer.exl3_gpu_shared_w2 = {e: to_device(layer.exl3_w2[e]) for e in shared}

    def _apply_cpu_draft(self, layer, x, topk_weights, topk_ids, swiglu_limit) -> torch.Tensor:
        """Routed experts on the CPU worker while any fused shared expert runs on the GPU; eager."""
        from sglang.srt.layers.moe.cpu_experts import draft

        assert_not_capturing("Exl3MoEMethod._apply_cpu_draft")
        pending = draft.DRAFT_CPU_EXPERTS.runtime().submit(
            layer.exl3_cpu_draft_key,
            x.to(torch.float16).cpu(),
            topk_ids.to(torch.int64).cpu(),
            topk_weights.to(torch.float16).cpu(),
        )
        out = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
        if layer.exl3_gpu_shared_w2:
            exl3_moe_accumulate(
                out,
                x,
                topk_weights,
                topk_ids,
                layer.exl3_gpu_shared_w13,
                layer.exl3_gpu_shared_w2,
                swiglu_limit,
                experts=sorted(layer.exl3_gpu_shared_w2),
            )
        out += pending.result().to(x.device)
        return out.to(x.dtype)
```

  If `exl3.py` does not already import them at module level, import `Exl3Tensors` and `exl3_moe_accumulate` from
  `exl3_ops` and `EXL3_STREAMED_NAMES` from `layers/moe/exl3_expert_format`. Run
  `grep -n "^from\|^import" python/sglang/srt/layers/quantization/exl3.py` to check. Without these the file does
  not import.

  **4f.** In `apply`, insert a branch between the `elif streamer is not None:` branch and the final `else:` (the
  `exl3_moe_loop` call):

```python
        elif self.cpu_draft:
            out = self._apply_cpu_draft(
                layer, dispatch_output.hidden_states, topk_weights, topk_ids, cfg.swiglu_limit
            )
```

  The routed scaling after the branches applies to this output unchanged, as it does to `exl3_moe_loop`'s.

- [ ] **Step 5: Run the tests; they must pass.** Run the touched files plus the EXL3 MoE files, so the constructor
  change cannot break another caller:

```bash
PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 taskset -c 0-63 /data/models/slang/.venv/bin/python -m pytest \
  test/registered/unit/layers/quantization/test_exl3_stream_scope.py \
  test/registered/unit/layers/quantization/test_exl3_moe_method.py \
  test/registered/unit/layers/quantization/test_exl3_moe_stream_mode.py \
  test/registered/unit/kernels/test_dspark_draft_cpu_experts.py \
  -q -p no:randomly 2>&1 | tail -3; echo "EXIT=${PIPESTATUS[0]}"
```
Expected: `EXIT=0`.

If `test_a_cpu_draft_layer_loads_its_experts_into_host_memory` fails because `nn.Module` under `torch.device("meta")`
is unsupported, drop the `with` line. The device assertion still holds, because the placeholder's `device="cpu"` is
explicit.

- [ ] **Step 6: Commit**

```bash
git add python/sglang/srt/models/deepseek_v4_exl3_weights.py python/sglang/srt/layers/quantization/exl3.py \
  test/registered/unit/layers/quantization/test_exl3_stream_scope.py \
  test/registered/unit/layers/quantization/test_exl3_moe_method.py
git commit -m "exl3(dspark): draft stage experts in host RAM, computed by the CPU expert kernel

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: GPU parity: the CPU draft path against `exl3_moe_loop`, at 3 and 4 bits

**Files:**
- Create: `test/manual/dsv41/test_dspark_cpu_draft_gpu.py`

**Interfaces:**
- Consumes: `Exl3MoEMethod(..., cpu_draft=True)` and its `_apply_cpu_draft` (Task 5); `DraftCpuExpertsRegistry`
  (Task 4); `exl3_moe_loop` (`exl3_ops.py:191`).

The CPU kernel quantizes activations to int8 in blocks of 128 (`SGLANG_EXL3_CPU_ACT_BLOCK`, §30.1), so the result is
close to the GPU's, not bit-identical. Running the same check at 3 bits (the target's bitrate, already validated in
P0/P2) separates a 4-bit defect from activation-quantization noise.

- [ ] **Step 1: Write the test**

```python
"""The DSpark draft's CPU expert path matches exl3_moe_loop on the GPU, at the target's 3 and the draft's 4 bits.

Run on divix01 with the GPU lock, EXL3_MOE_CPU_PIN=0 and SGLANG_EXL3_SRC set (the CPU kernel's build):
  flock .../cc-gpu.lock taskset -c 18-29,32-63 python -m pytest test/manual/dsv41/test_dspark_cpu_draft_gpu.py -s
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

E_ROUTED, SHARED, HIDDEN, INTER, ROWS, TOPK = 8, 1, 512, 256, 6, 3
LIMIT = 10.0


def _layer(bits, monkeypatch, registry):
    from sglang.srt.layers.moe.cpu_experts import draft
    from sglang.srt.layers.quantization.exl3 import Exl3Config, Exl3MoEMethod

    monkeypatch.setattr(draft, "DRAFT_CPU_EXPERTS", registry)
    cfg = {"quant_method": "exl3", "version": "1.4.2", "bits": float(bits), "head_bits": 6, "codebook": "mul1"}
    e = E_ROUTED + SHARED
    layer = nn.Module()
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
    method.process_weights_after_loading(layer)
    return layer, method


def _routes(pattern, g):
    if pattern == "shared":
        routed = torch.randperm(E_ROUTED, generator=g)[:TOPK].repeat(ROWS, 1)
    else:
        routed = torch.stack([torch.randperm(E_ROUTED, generator=g)[:TOPK] for _ in range(ROWS)])
    shared = torch.full((ROWS, 1), E_ROUTED, dtype=torch.int64)
    return torch.cat([routed, shared], dim=1)


@pytest.mark.parametrize("pattern", ["independent", "shared"])
@pytest.mark.parametrize("bits", [3, 4])
def test_cpu_draft_matches_the_gpu_loop(bits, pattern, monkeypatch):
    from dataclasses import replace

    from sglang.srt.environ import envs
    from sglang.srt.layers.moe.cpu_experts.draft import DraftCpuExpertsRegistry
    from sglang.srt.layers.quantization.exl3_ops import exl3_moe_loop

    monkeypatch.setenv("EXL3_MOE_CPU_PIN", "0")
    cores = sorted(os.sched_getaffinity(0) & set(range(18, 30))) or sorted(os.sched_getaffinity(0))[:4]
    registry = DraftCpuExpertsRegistry()
    layer, method = _layer(bits, monkeypatch, registry)
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
        finally:
            registry.close()
    ref = exl3_moe_loop(x, weights, ids, w13, w2, LIMIT).float()
    rel = float((got - ref).norm() / ref.norm())
    print(f"bits={bits} pattern={pattern} rel_l2={rel:.4f}")
    assert torch.isfinite(got).all()
    assert rel < 0.05
```

- [ ] **Step 2: Commit, push, pull into the divix01 worktree, and run** (GPU lock; CPU cores 18-29 for the kernel)

```bash
git add test/manual/dsv41/test_dspark_cpu_draft_gpu.py
git commit -m "test(dspark): CPU draft experts against the GPU loop at 3 and 4 bits

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push origin dsv41-dspark-cpu-draft
ssh divix01 'cd /data/models/slang/nvfp4-work/wt-dspark-cpu && git fetch origin && git checkout --detach origin/dsv41-dspark-cpu-draft && \
  export PYTHONPATH=$PWD/python EXL3_MOE_CPU_PIN=0 CUDA_MODULE_LOADING=EAGER && \
  export SGLANG_EXL3_SRC=$(PYTHONPATH=benchmarks/dsv41_baseline /data/models/slang/.venv/bin/python -c "import arm_env;print(arm_env.base_env()[\"SGLANG_EXL3_SRC\"])") && \
  flock /data/models/slang/nvfp4-work/cc-gpu.lock taskset -c 18-29,32-63 /data/models/slang/.venv/bin/python -m pytest \
  test/manual/dsv41/test_dspark_cpu_draft_gpu.py -q -s -p no:randomly 2>&1 | tail -8; echo "EXIT=${PIPESTATUS[0]}"'
```
Expected: `EXIT=0`, four `rel_l2=` lines, and 4-bit values of the same order as the 3-bit ones.

If a 4-bit case fails while 3-bit passes, the kernel mishandles the draft's bitrate. Debug that with
`superpowers:systematic-debugging`. Never loosen the 0.05.

- [ ] **Step 3: Record** the four `rel_l2` values under "Results" in this plan, then commit.

---

### Task 7: Served A/B: resident draft vs CPU draft, eager, and the doc

**Files:**
- Create: `analysis/dsv41-drive/dspark/ab_cpu_draft.py`
- Modify: `DSV41_REFERENCE.md`, adding §33.4.

**Interfaces:**
- Consumes: `scripts/dsv41/trace_corpus.py` (`--dspark`, `--stop-at-eos`; it writes per-session
  `decode_tok_s`, `completion_tokens` and `spec_verify_ct`); `benchmarks/dsv41_baseline/arm_env.py`
  (`arm_env(overrides)`, `MODEL_PATH`); the sessions corpus `/mnt/nvme2/nvfp4-work/benchmarks/full/sessions.jsonl`
  (the D1 α run's, `analysis/dsv41-dspark/run-alpha.sh`). Reading one JSONL file is not a bulk read.

Arms, eager (the gate forces it under DSpark), same sessions, same everything else:
- **resident:** `SGLANG_MOE_HOT_GPU_MB=7168` (D1's budget with the draft resident).
- **cpu:** `SGLANG_MOE_HOT_GPU_MB=14080` (7168 + the draft's 6912 MiB), plus
  `SGLANG_DSV41_ENABLE_DSPARK_CPU_EXPERTS=1`, `SGLANG_DSV41_DSPARK_CPU_EXPERTS_CORES=18-29` and
  `EXL3_MOE_CPU_PIN=0`.

- [ ] **Step 1: Write the driver**

```python
"""DSpark A/B (plan 2026-10-02-dsv41-dspark-cpu-draft, Task 7): the draft's experts resident in VRAM vs on the CPU.

Eager (the EXL3 gate refuses speculation under a decode graph), one Engine per arm through trace_corpus.py, the same
sessions. Run on divix01 from a worktree at the pushed branch, holding rowimg-disk.lock then cc-gpu.lock:
  python analysis/dsv41-drive/dspark/ab_cpu_draft.py OUTDIR [ARM ...]
"""

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "benchmarks", "dsv41_baseline"))
import arm_env  # noqa: E402

PYTHON = "/data/models/slang/.venv/bin/python"
DRAFT = "/data/models/slang/nvfp4-work/cc-expert-prediction/dsv41-dspark-draft"
SESSIONS = "/mnt/nvme2/nvfp4-work/benchmarks/full/sessions.jsonl"
# The 2026-09-24 DSpark launch's eager overrides, less the variables §32.7 retired.
COMMON = {
    "SGLANG_MOE_EXPERT_GRAPH_GATHER": "0",
    "SGLANG_MOE_GPU_RESIDENCY_UPDATE": "0",
    "SGLANG_MOE_HOT_INSERT_ON_MISS_STAGE": "0",
    "SGLANG_MOE_EXPERT_FUSED_PLAN": "0",
    "SGLANG_DSV41_ENGRAM_HOST_NODE_CACHE_URING": "0",
    # Device wait serves from the uring store, which the 2026-09-24 DSpark launch kept off.
    "SGLANG_DSV41_ENABLE_ENGRAM_DEVICE_WAIT": "0",
    "SGLANG_SM120_FLASHMLA_BACKEND": "triton",
    # Layer-major prefill needs --max-running-requests 1; trace_corpus launches with 4.
    "SGLANG_LAYER_MAJOR_PREFILL_MIN_TOKENS": "0",
    # Prefill fills are the RAM-miss service's reads, which only run with graph gather (option C).
    "SGLANG_DSV41_ENABLE_PREFILL_FILLS": "0",
}
ARMS = {
    "resident": {"SGLANG_MOE_HOT_GPU_MB": "7168"},
    "cpu": {
        "SGLANG_MOE_HOT_GPU_MB": "14080",
        "SGLANG_DSV41_ENABLE_DSPARK_CPU_EXPERTS": "1",
        "SGLANG_DSV41_DSPARK_CPU_EXPERTS_CORES": "18-29",
        "EXL3_MOE_CPU_PIN": "0",
    },
}


def run(arm: str, outdir: str, n: int, new_tokens: int) -> int:
    # arm_env holds only the recipe; without the inherited PATH the extension build cannot find cc1plus.
    env = os.environ | arm_env.arm_env({**COMMON, **ARMS[arm]})
    env["PYTHONPATH"] = os.path.join(REPO, "python")
    env.setdefault("OMP_NUM_THREADS", "16")
    cmd = [
        PYTHON, os.path.join(REPO, "scripts", "dsv41", "trace_corpus.py"),
        "--model", arm_env.MODEL_PATH,
        "--sessions", SESSIONS,
        "--n", str(n),
        "--prompt-tokens", "256",
        "--new-tokens", str(new_tokens),
        "--stop-at-eos",
        "--dspark", DRAFT,
        "--out", os.path.join(outdir, f"{arm}.json"),
    ]
    print(f"=== {arm}: {' '.join(cmd)}", flush=True)
    with open(os.path.join(outdir, f"{arm}.log"), "w") as log:
        return subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, cwd=REPO).returncode


def main():
    outdir = sys.argv[1]
    arms = sys.argv[2:] or list(ARMS)
    n = int(os.environ.get("AB_SESSIONS", "8"))
    new_tokens = int(os.environ.get("AB_NEW_TOKENS", "128"))
    os.makedirs(outdir, exist_ok=True)
    for arm in arms:
        rc = run(arm, outdir, n, new_tokens)
        print(f"{arm}: rc={rc}", flush=True)
        if rc:
            sys.exit(rc)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit and push**

```bash
git add analysis/dsv41-drive/dspark/ab_cpu_draft.py
git commit -m "bench(dspark): resident vs CPU draft experts A/B driver

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push origin dsv41-dspark-cpu-draft
```

- [ ] **Step 3: Ask the owner for a GPU window** (production must not hold `cc-gpu.lock`). Then smoke both arms with
  one session of 16 tokens:

```bash
ssh divix01 'cd /data/models/slang/nvfp4-work/wt-dspark-cpu && git fetch origin && git checkout --detach origin/dsv41-dspark-cpu-draft && \
  OUT=/data/models/slang/nvfp4-work/cc-expert-prediction/analysis/dsv41-dspark/cpu-draft-smoke && \
  AB_SESSIONS=1 AB_NEW_TOKENS=16 flock /data/models/slang/nvfp4-work/rowimg-disk.lock \
  flock /data/models/slang/nvfp4-work/cc-gpu.lock taskset -c 0-17,30-63 \
  /data/models/slang/.venv/bin/python analysis/dsv41-drive/dspark/ab_cpu_draft.py $OUT; echo EXIT=$?; \
  grep -h "DSpark CPU experts\|Load weight end\|Traceback\|Error" $OUT/*.log | head -20'
```

The driver's own affinity (`0-17,30-63`) leaves 18–29 to the kernel's worker, which binds itself there.

Pass criteria:
- `EXIT=0`.
- The cpu arm logs `DSpark CPU experts: 3 draft stages on cores [18, …, 29]`.
- The cpu arm's `Load weight end` line for the draft worker shows about 6.6 GB less memory than the resident arm's.
- Both arms produce text.
- Record each arm's wall time from launch to "ready" (from its log). A much longer cpu-arm startup means the dummy
  run's prefill-sized draft forward hit the CPU path (Review Focus 3).

If the gate refuses a `COMMON` override, change exactly the variable its message names, record the change in this
plan, and rerun. The overrides date from 2026-09-24, and §30–§32 changed the gate since. If the cpu arm runs out of
memory at 14080, lower it by 1024 at a time and record the value that launches.

- [ ] **Step 4: Run the A/B:** 8 sessions, 128 tokens, EOS honoured. Order the arms resident then cpu, then repeat
  in reverse order into a second directory, so drift and page-cache warm-up do not favour one arm.

```bash
ssh divix01 'cd /data/models/slang/nvfp4-work/wt-dspark-cpu && A=/data/models/slang/nvfp4-work/cc-expert-prediction/analysis/dsv41-dspark/cpu-draft-ab && \
  flock /data/models/slang/nvfp4-work/rowimg-disk.lock flock /data/models/slang/nvfp4-work/cc-gpu.lock taskset -c 0-17,30-63 \
  /data/models/slang/.venv/bin/python analysis/dsv41-drive/dspark/ab_cpu_draft.py $A/r1 resident cpu; echo EXIT1=$?; \
  flock /data/models/slang/nvfp4-work/rowimg-disk.lock flock /data/models/slang/nvfp4-work/cc-gpu.lock taskset -c 0-17,30-63 \
  /data/models/slang/.venv/bin/python analysis/dsv41-drive/dspark/ab_cpu_draft.py $A/r2 cpu resident; echo EXIT2=$?'
```

Expected: `EXIT1=0` and `EXIT2=0`.

- [ ] **Step 5: Analyze.** From each `<arm>.json` `per_session`, report:
  - the median of per-session `decode_tok_s` per arm and repeat;
  - the median paired ratio cpu/resident across the 16 session pairs, with the number of pairs the cpu arm wins;
  - the median TTFT (`ttft_s`) per arm. If the cpu arm's TTFT exceeds the resident arm's by more than 20%, the draft
    MoE runs over prefill rows on the CPU (Review Focus 3); report it as a blocker for the plan's follow-up;
  - the mean accept length `completion_tokens / spec_verify_ct` per arm. It must agree within noise, since only the
    draft's numerics differ. A large drop means the CPU draft hurts α; report it rather than explain it away.

  From the cpu arm's last `DSpark CPU experts:` log line, report the mean and p90 ms per stage call, the union, the
  weight passes, and the top-8/16/32 coverage.

  The result is a **win** if the median paired ratio is above 1 and the cpu arm wins at least 12 of 16 pairs. It is
  a **loss** if the ratio is below 1 with at least 12 of 16 losses. Anything else is **inconclusive**: report the
  numbers and stop. Whatever the outcome, record that both arms are far below the production non-spec path
  (~13.5 tok/s, §30.1). This A/B measures the draft's VRAM trade, not whether DSpark ships.

- [ ] **Step 6: Record in the doc.** Add `### 33.4 The draft's experts on the CPU (2026-10-…)` to
  `DSV41_REFERENCE.md`, after §33.3, containing:
  - Task 1's benchmark table (bits, rows, pattern, ms per stage call, passes, ms per pass);
  - Task 6's four `rel_l2` values;
  - the A/B numbers from Step 5, with the exact commands and output directories;
  - the verdict.
  - If coverage of the top 16 experts per stage is above 80%, add one line saying the hybrid in "Next" below is
    worth planning. Otherwise say it is not.

  Commit:

```bash
git add DSV41_REFERENCE.md docs/superpowers/plans/2026-10-02-dsv41-dspark-cpu-draft.md
git commit -m "docs(dsv41): DSpark draft experts on the CPU, benchmark, parity and A/B (section 33.4)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push origin dsv41-dspark-cpu-draft
```

- [ ] **Step 7: Clean up the divix01 worktree** after the owner has the results:
  `git -C /data/models/slang/sglang worktree remove /data/models/slang/nvfp4-work/wt-dspark-cpu`. Merge to `master`
  only when the owner asks.

---

## Next (not in this plan; decided by Task 7's coverage line)

- **Hybrid draft cache:** keep the few most-routed draft experts per stage in a small VRAM set and send only the rest
  to the CPU. It only pays if routing is skewed (Task 7's coverage).
- **Draft MoE in a graph:** not useful while DSpark is eager (§33.3).
- **NUMA re-homing of the draft's host weights:** §30.4 measured a remote penalty of only 2–8% per expert, so it is
  not worth the code until a profile says otherwise.

## Results

(Filled in by Tasks 1, 6 and 7.)

### Task 1: draft-shaped experts through the CPU kernel (2026-10-02, divix01, `node1-r1`)

Command: `EXL3_MOE_CPU_PIN=0 flock cc-gpu.lock numactl --membind=1 taskset -c 18-29 python
analysis/dsv41-drive/cpu-experts/draft_bench.py 3,4 8,12 1,2,3,6 independent,shared node1-r1` at `1f92bc4367`;
EXIT=0, 32 cells, raw lines in `wt-dspark-cpu/analysis/dsv41-drive/cpu-experts/draft_bench_results.jsonl`.

Sanity: `bits=3, rows=1, independent` is 0.505 ms per pass at 12 threads (0.677 at 8), inside §28.1/§30.4's
0.49-0.58 ms. `make_layer` takes 4-bit slabs.

`draft_step_ms` (3 stages x median stage call), union = mean distinct experts per stage call:

| bits | threads | rows | independent (union) | shared (union 3) |
|---|---|---|---|---|
| 4 | 12 | 1 | 4.94 (3) | 5.02 |
| 4 | 12 | 2 | 9.76 (6.0) | 6.31 |
| 4 | 12 | 3 | 13.86 (8.8) | 7.85 |
| 4 | 12 | 6 | **26.53** (17.1) | 14.46 |
| 4 | 8 | 6 | 35.03 (17.0) | 25.66 |
| 3 | 12 | 6 | 31.08 (17.1) | 15.87 |
| 3 | 8 | 6 | 39.43 (17.2) | 24.19 |

- 12 threads beats 8 in every 6-row cell. 4-bit is no slower than 3-bit per pass (0.52 vs 0.61 ms at 12 threads,
  6 rows), so the 17.7 MB size did not cost the ~0.85 ms per pass the estimate assumed.
- Shared routing confirms the `CHUNK_M = 2` grouping: 9 passes for 3 experts x 6 rows, at 0.54 ms per pass.
- **P = 26.53 ms** (bits 4, rows 6, independent, 12 threads). `18 < P <= 30`: **ask the owner.** The bracket is
  14.5 ms (shared) to 26.5 ms (independent) against the ~13-18 ms the freed 6.75 GiB is worth per target forward,
  so the sign of the win depends on the draft's real per-stage union, which this bench cannot see.
- **Kernel flavor.** The cells above ran the plain `sglang_exl3_ext` build (no `SGLANG_DSV41_CPU_EXPERTS`), i.e.
  upstream's CPU kernel. Rerun on the optimized `_resid_b128_cpu_v1` build (tag `node1-opt`, 12 threads, private
  build dir): faster at 1 row (3-bit 0.343 vs 0.505 ms per pass) but slower at 6 rows: bits 4 independent 30.92 vs
  26.53 ms, shared 20.75 vs 14.46 ms (~0.77 ms per pass when rows share an expert, vs 0.54). The draft should keep
  the plain build, which is what `exl3_ext()` loads under DSpark, since `SGLANG_DSV41_CPU_EXPERTS` is refused
  with speculation.
- **Owner's call (2026-10-02):** measure the draft's real per-stage union before Tasks 2-7. Probe:
  `SGLANG_DSPARK_DEBUG_DRAFT_ROUTES_PATH` and `ab_cpu_draft.py ... routes`, read by `draft_routes_report.py`.
