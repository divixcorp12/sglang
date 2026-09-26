# DSV41 Split Fill Gather Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop the GPU idling while a prefill chunk waits for its NVMe fills. Copy the chunk's rows that are already in
the pinned tier first, and the filled rows once they land. Today 1.93 s of a 9.28 s prefill is GPU idle spent in
those waits.

**Architecture:** Behind a new default-off flag, `SGLANG_DSV41_ENABLE_PREFILL_SPLIT_GATHER`,
`ExpertPinnedHostCache.gather_rows` changes when a chunk holds rows that the layer's prefill fill is still reading. It
copies the other rows first with a new indexed gather kernel, then waits (`_await_fills`) only for the fills, then
copies the filled rows. Every row gets the same bytes in the same staging position, so the MoE output is bitwise
unchanged. The only difference is that the first copy runs on the GPU while the host waits for the NVMe.

**Tech Stack:** Python 3.13, PyTorch (CUDA, RTX 5090 SM120), Triton, pytest; Nsight Systems for the before/after
trace.

**Spec:** `DSV41_REFERENCE.md` §27.11. Read "Where the remaining GPU idle goes, by site", then its "Next" item "Split
each chunk's gather around its fills", and §27.6 (prefill fills). Also read `MOE_PREFILL_OPT.md` for the repo rules
it lists.

## Global Constraints

- New env var: `SGLANG_DSV41_ENABLE_PREFILL_SPLIT_GATHER = EnvBool(False)` in `python/sglang/srt/environ.py`, and the
  field `enable_prefill_split_gather` in `Dsv41Config`. `test_one_field_per_knob` must pass.
- Flag off must be exactly today's path: the existing `_gather_host_rows_kernel` and its three call sites are
  untouched, and `copy_rows(..., rows=None)` runs today's code.
- Outputs must be byte-identical: staging bytes per row, the MoE output, and greedy output flag off vs on.
- A split copy must happen inside the same host use as today's copy, and must never read a slot that is still
  filling: a row the layer's fill claimed is copied only after `_await_fills` covers it.
- `msgspec.Struct`, not `@dataclass`; no defensive `getattr`/`hasattr`; comments per `.claude/rules/comment-style.md`.
- On divix01, per `.claude/rules/divix01-run-protocol.md`:
  - get code there by commit, push, and a private worktree;
  - set `PYTHONPATH=$PWD/python` and print `sglang.__file__`;
  - read `PIPESTATUS[0]`;
  - run CPU work under `taskset -c 0-63` and GPU work under `cc-gpu.lock` on cores 32-63;
  - lock order: `rowimg-disk.lock`, then `cc-gpu.lock`.
- The laptop Python has no `transformers`, so tests run on divix01. Commit each test before its implementation and
  push both together, then run RED at the test commit and GREEN at the tip. Scripts from the route plan work:
  `/mnt/nvme1/prefill-opt/rpt.sh COMMIT <pytest args>` (CPU) and `gpt.sh COMMIT <pytest args>` (GPU, takes the lock).
- Production (port 7867) is stopped and must stay stopped unless the user says otherwise. Never test in
  `dsv41-direct-prod` or `dsv41-direct-live`. Scratch goes on `/mnt/nvme1`.
- Traced arms use `NSYS_GPU_METRICS=0`. The root PCIe session's `nsys stop` failed on 2026-09-25, and the orphan held
  `cc-gpu.lock` until the user killed it.
- Arms: A (the current recipe, which has the route plan on) then B (+ the new flag), once each, no ABBA, port 30021.
- Git:
  - work on `origin/master` directly (the user's instruction; no separate branch or workspace);
  - never force-push;
  - stage by name, never `.omc/`;
  - fetch and check fast-forward before every push.
- Commit trailer:
  ```
  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01N985sZUTxTE9P7MP2rtJaF
  ```

## Why (measured, §27.11)

Traced arm at `035ce226d3`: route plan on, 9,280 ms prefill, 2,728 ms GPU idle.

| Host state while the GPU idles | GPU idle |
|---|---:|
| Fill wait before a layer's first chunk's gather (40 chunks, median 26.9 ms) | 1,296 ms |
| Fill wait before a later chunk's gather (13 chunks, median 51 ms) | 637 ms |
| Python and launches | 765 ms |
| Blocking CUDA calls | 30 ms |

`gather_rows` today calls `_await_fills(chunk_ids)` and only then `copy_rows(chunk, ...)`: nothing of the chunk moves
until its last NVMe row lands. A row gathers in ~1.08 ms (13.3 MB at ~12.3 GB/s) and reads from the two NVMe mirrors
in ~1.75 ms (~7.6 GB/s). A first chunk's median 27 ms wait is therefore about 15 fills, which leaves ~49 resident rows
taking ~53 ms to gather. That covers the wait in most layers.

Estimated: most of the 1.3 s, and part of the 0.64 s, for TTFT ~9.9 s to ~8.5 s. Task 7 measures this.

## File Structure

- Modify `python/sglang/srt/environ.py`, `python/sglang/srt/dsv41_config.py` and
  `test/registered/unit/test_dsv41_config.py`: the flag.
- Modify `python/sglang/srt/layers/moe/expert_stream.py`:
  - add `_gather_host_rows_to_kernel` next to `_gather_host_rows_kernel` (~line 695);
  - `ExpertPinnedHostCache.__init__` reads the flag;
  - `copy_rows` (~457) gains `rows`;
  - `gather_rows` (~496) splits;
  - two private helpers, `_filling_positions` and `_splits`.
- Test:
  - `test/registered/unit/layers/moe/test_expert_pinned_row_fills.py` (CPU, the `FakeFills` harness);
  - `test/registered/unit/layers/moe/test_expert_plugins_cuda.py` (CUDA, `TestPinnedTierCuda`).
- Create `analysis/dsv41-drive/split-gather/drive_arms.sh`.
- Docs: `DSV41_REFERENCE.md` §27.12 and §27.4 item 7.

## Review Focus

1. **A chunk whose rows are all filling.** The split buys nothing here, so it must take today's single wait and
   single copy. Pinned in Task 3 (`test_a_chunk_of_only_filling_rows_waits_then_copies_once`).
2. **A chunk miss the fill did not claim (overflow).** `ensure_rows` joins the fill (`finish_fills`), after which
   `_fill_order` is None and nothing is filling, so the chunk must take today's path. Pinned by the existing
   `test_a_chunk_miss_outside_the_prefetch_joins_it_before_admitting`, run with the flag on in Task 3.
3. **Outputs that would take the non-contiguous fallback copy.** A split is refused, and today's path runs. Pinned in
   Task 2 (`copy_rows` raises for `rows` on the fallback) and Task 3 (`_splits`).
4. **A fill that fails mid-chunk.** `_await_fills` must raise as today, after the resident rows' copy was queued. The
   caller then abandons the forward, so the rows already copied into staging do no harm. Pinned in Task 3
   (`test_a_failed_fill_still_raises_with_the_split`).
5. **Every chunk of a layer, not just the first.** Later chunks' filling rows sit at arbitrary positions; the indexed
   copy must put each at its own row. Pinned in Task 3 (the parity test's second chunk).

---

### Task 1: The flag

**Files:**
- Modify: `python/sglang/srt/environ.py` (after `SGLANG_DSV41_ENABLE_PREFILL_ROUTE_PLAN`, ~line 1929)
- Modify: `python/sglang/srt/dsv41_config.py`
- Modify: `test/registered/unit/test_dsv41_config.py` (`test_defaults_match_the_env_declarations`)

**Interfaces:**
- Produces: `envs.SGLANG_DSV41_ENABLE_PREFILL_SPLIT_GATHER` and `Dsv41Config.enable_prefill_split_gather: bool`.

- [ ] **Step 1: Add `enable_prefill_split_gather=False,` after `enable_prefill_route_plan=False,` in the defaults
  test; commit it**

- [ ] **Step 2: Run it on divix01 at that commit and see it fail**

Run: `bash /mnt/nvme1/prefill-opt/rpt.sh <sha> test/registered/unit/test_dsv41_config.py`
Expected: FAIL, `TypeError: Unexpected keyword argument 'enable_prefill_split_gather'`.

- [ ] **Step 3: Declare it**

`environ.py`, after `SGLANG_DSV41_ENABLE_PREFILL_ROUTE_PLAN = EnvBool(False)`:
```python
    # Split fill gather (plan 2026-09-26-dsv41-split-fill-gather): a prefill gather chunk holding rows the layer's
    # prefill fill is still reading copies its other rows first, then waits for the fill and copies the filled rows,
    # so the GPU gathers while the NVMe reads. Same bytes in the same staging rows. Off by default.
    SGLANG_DSV41_ENABLE_PREFILL_SPLIT_GATHER = EnvBool(False)
```
`dsv41_config.py`: add `enable_prefill_split_gather: bool` after `enable_prefill_route_plan: bool`. In `from_envs`, add
`enable_prefill_split_gather=envs.SGLANG_DSV41_ENABLE_PREFILL_SPLIT_GATHER.get(),`.

- [ ] **Step 4: Commit, push, and run at the tip**

Expected: `test_dsv41_config.py` passes, including `test_one_field_per_knob`.

---

### Task 2: `copy_rows(rows=...)` and the indexed gather kernel

**Files:**
- Modify: `python/sglang/srt/layers/moe/expert_stream.py` (the new kernel after `_gather_host_rows_kernel`;
  `copy_rows`)
- Test: `test/registered/unit/layers/moe/test_expert_pinned_row_fills.py` (CPU),
  `test/registered/unit/layers/moe/test_expert_plugins_cuda.py` (CUDA)

**Interfaces:**
- Produces:
  - `ExpertPinnedHostCache.copy_rows(source_ids: torch.Tensor, outputs: dict[str, torch.Tensor], rows: torch.Tensor | None = None) -> bool`.
    `rows` is int64 on `source_ids.device`, and names the output rows to copy, each from `source_ids` at the same
    position. Every other output row is left as it was.
  - `_gather_host_rows_to_kernel(src_ptr, index_ptr, rows_ptr, output_ptr, row_bytes, BLOCK)`.

- [ ] **Step 1: Write the failing CPU test** (append to `test_expert_pinned_row_fills.py`)

```python
def test_copy_rows_copies_only_the_named_rows_and_leaves_the_rest():
    streamer, cache, fills, reference = _setup()
    cache.ensure_rows(torch.tensor([3, 1, 5]))
    outputs = {n: torch.full_like(t, SENTINEL) for n, t in _outputs(reference, 3).items()}
    cache.copy_rows(torch.tensor([3, 1, 5]), outputs, rows=torch.tensor([2, 0]))
    for name in NAMES:
        assert torch.equal(outputs[name][0], reference[name][3])
        assert torch.equal(outputs[name][2], reference[name][5])
        assert (outputs[name][1] == SENTINEL).all()
```

- [ ] **Step 2: Write the failing CUDA test** (in `TestPinnedTierCuda`)

```python
    def test_copy_rows_to_named_rows_on_cuda_leaves_the_rest(self):
        from sglang.srt.layers.moe.expert_stream import ExpertPinnedHostCache

        experts = 8
        layer = torch.nn.Module()
        layer.host_rows = torch.nn.Parameter(
            torch.randint(0, 256, (experts, 3000), dtype=torch.uint8), requires_grad=False
        )
        streamer = ExpertStreamer(layer, ("host_rows",))
        cache = ExpertPinnedHostCache(streamer, 4)
        ids = torch.tensor([6, 2, 5], device="cuda")
        cache.ensure_rows(ids)
        out = torch.full((3, 3000), 7, dtype=torch.uint8, device="cuda")
        cache.copy_rows(ids, {"host_rows": out}, rows=torch.tensor([1, 2], device="cuda"))
        got = out.cpu()
        self.assertTrue(torch.equal(got[1], layer.host_rows.data[2]))
        self.assertTrue(torch.equal(got[2], layer.host_rows.data[5]))
        self.assertTrue((got[0] == 7).all())
        cache.close()
```
Before relying on it, check that `ensure_rows` on this streamer reads from `layer.host_rows`, as
`test_cached_gather_copies_misses_beyond_the_pinned_capacity` does. If this streamer needs a row source first, copy
that test's setup.

- [ ] **Step 3: Commit both tests; run them at that commit (CPU with `rpt.sh`, CUDA with `gpt.sh`)**

Expected: both FAIL with `TypeError: ... unexpected keyword argument 'rows'`.

- [ ] **Step 4: Implement**

Add the kernel after `_gather_host_rows_kernel`:
```python
@triton.jit
def _gather_host_rows_to_kernel(
    src_ptr,
    index_ptr,
    rows_ptr,
    output_ptr,
    row_bytes,
    BLOCK: tl.constexpr,
):
    source_row = tl.load(index_ptr + tl.program_id(0)).to(tl.int64)
    output_row = tl.load(rows_ptr + tl.program_id(0)).to(tl.int64)
    offsets = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < row_bytes
    values = tl.load(src_ptr + source_row * row_bytes + offsets, mask=mask, other=0)
    tl.store(output_ptr + output_row * row_bytes + offsets, values, mask=mask)
```
In `copy_rows`:
- Add `rows: torch.Tensor | None = None` to the signature.
- Add to the docstring: "``rows`` limits the copy to those output rows (int64, on ``source_ids``' device), each from
  ``source_ids`` at the same position; other rows are left as they are, and the non-contiguous fallback refuses it."
- After `slots = self.expert_to_slot[source_ids.long()]`, the three branches become:
  ```python
            if output.device.type == "cpu":
                if rows is None:
                    torch.index_select(source, 0, slots.cpu(), out=output)
                else:
                    output[rows.cpu()] = torch.index_select(source, 0, slots[rows].cpu())
            elif source.is_contiguous() and output.is_contiguous():
                row_bytes = source.numel() * source.element_size() // self.capacity
                if rows is None:
                    ...the existing _gather_host_rows_kernel launch, unchanged...
                else:
                    _gather_host_rows_to_kernel[(rows.numel(), triton.cdiv(row_bytes, 1024))](
                        source.view(torch.uint8),
                        slots[rows],
                        rows,
                        output.view(torch.uint8),
                        row_bytes,
                        BLOCK=1024,
                    )
            else:
                if rows is not None:
                    raise ValueError("a copy to named rows needs contiguous pinned rows and outputs")
                ...the existing fallback, unchanged...
  ```

- [ ] **Step 5: Commit, push, and run both tests plus `test_expert_pinned_row_fills.py` and
  `test_expert_plugins_cuda.py` at the tip**

Expected: all pass.

---

### Task 3: `gather_rows` splits a chunk around its fills

**Files:**
- Modify: `python/sglang/srt/layers/moe/expert_stream.py`:
  - `ExpertPinnedHostCache.__init__`: `self.split_fill_gather`, set after `self.row_fills`;
  - `gather_rows`: the `_await_fills` / `copy_rows` pair near its end;
  - the new `_filling_positions` and `_splits`.
- Test: `test/registered/unit/layers/moe/test_expert_pinned_row_fills.py`

**Interfaces:**
- Consumes: `copy_rows(..., rows=...)` (Task 2), `envs.SGLANG_DSV41_ENABLE_PREFILL_SPLIT_GATHER` (Task 1).
- Produces: `ExpertPinnedHostCache.split_fill_gather: bool`.

- [ ] **Step 1: Write the failing tests** (append; add `from sglang.srt.environ import envs` to the imports)

```python
def _split_setup():
    with envs.SGLANG_DSV41_ENABLE_PREFILL_SPLIT_GATHER.override(True):
        return _setup()


def _sentinel_outputs(reference, rows):
    return {n: torch.full_like(t, SENTINEL) for n, t in _outputs(reference, rows).items()}


def test_split_gather_copies_resident_rows_before_waiting_for_the_fills():
    """A chunk's resident rows must already be copied when the host starts waiting for its fills."""
    streamer, cache, fills, reference = _split_setup()
    cache.ensure_rows(torch.tensor([4]))
    seen = []
    wait = fills.fill_wait
    with cache.host_use():
        assert cache.prefetch_rows([1, 4, 6], protected=[1, 4, 6]) == 2  # claims 1 and 6; 4 is resident
        outputs = _sentinel_outputs(reference, 3)
        fills.fill_wait = lambda rows: (seen.append({n: o.clone() for n, o in outputs.items()}), wait(rows))
        cache.gather_rows(torch.tensor([1, 4, 6]), outputs)
        cache.finish_fills()
    for name in NAMES:
        assert torch.equal(seen[0][name][1], reference[name][4])  # the resident row, copied before the wait
        assert (seen[0][name][0] == SENTINEL).all() and (seen[0][name][2] == SENTINEL).all()
    _check(outputs, reference, [1, 4, 6])


def test_split_gather_places_every_chunks_rows_like_the_unsplit_gather():
    """Two chunks with filling rows at different positions: byte-identical to the flag-off gather."""
    got = None
    for split in (False, True):
        with envs.SGLANG_DSV41_ENABLE_PREFILL_SPLIT_GATHER.override(split):
            streamer, cache, fills, reference = _setup(capacity=6, experts=8)
        cache.ensure_rows(torch.tensor([0, 5]))
        with cache.host_use():
            cache.prefetch_rows([0, 2, 3, 5, 7], protected=[0, 2, 3, 5, 7])
            first, second = _sentinel_outputs(reference, 3), _sentinel_outputs(reference, 3)
            cache.gather_rows(torch.tensor([0, 2, 5]), first)
            cache.gather_rows(torch.tensor([7, 5, 3]), second)  # filling at 0 and 2, resident 5 between
            cache.finish_fills()
        _check(first, reference, [0, 2, 5])
        _check(second, reference, [7, 5, 3])
        if got is None:
            got = (first, second)
        else:
            for a, b in zip(got, (first, second)):
                for name in NAMES:
                    assert torch.equal(a[name], b[name])


def test_a_chunk_of_only_filling_rows_waits_then_copies_once():
    streamer, cache, fills, reference = _split_setup()
    copies = []
    copy = cache.copy_rows
    cache.copy_rows = lambda ids, outputs, rows=None: copies.append(rows) or copy(ids, outputs, rows=rows)
    with cache.host_use():
        cache.prefetch_rows([1, 6], protected=[1, 6])
        outputs = _sentinel_outputs(reference, 2)
        cache.gather_rows(torch.tensor([1, 6]), outputs)
        cache.finish_fills()
    assert copies == [None] and fills.events[1] == ("wait", 2)
    _check(outputs, reference, [1, 6])


def test_a_failed_fill_still_raises_with_the_split():
    streamer, cache, fills, reference = _split_setup()
    cache.ensure_rows(torch.tensor([4]))
    fills.fill_wait = lambda rows: (_ for _ in ()).throw(RuntimeError("a prefetch of pinned host rows failed"))
    with cache.host_use():
        cache.prefetch_rows([1, 4], protected=[1, 4])
        with pytest.raises(RuntimeError, match="prefetch"):
            cache.gather_rows(torch.tensor([1, 4]), _sentinel_outputs(reference, 2))
        fills.fill_wait = lambda rows: None
        cache.finish_fills()
```
Also parametrize the existing `test_a_chunk_miss_outside_the_prefetch_joins_it_before_admitting` over
`split in (False, True)`, building `_setup()` under
`envs.SGLANG_DSV41_ENABLE_PREFILL_SPLIT_GATHER.override(split)`.

Before relying on the failure test, check what `_await_fills` does when `fill_wait` raises: `fill_wait` returns
nothing and `_await_fills` does not check a result. If it cannot fail at that point today, rewrite the test to fail
through `finish_fills`, as `test_a_failed_prefetch_raises_when_joined` does. Record that in the plan's ledger.

- [ ] **Step 2: Commit the tests; run them at that commit**

Run: `bash /mnt/nvme1/prefill-opt/rpt.sh <sha> test/registered/unit/layers/moe/test_expert_pinned_row_fills.py`
Expected:
- `test_split_gather_copies_resident_rows_before_waiting_for_the_fills` FAILS: at the wait, the resident row is
  still SENTINEL.
- `test_a_chunk_of_only_filling_rows_...` may already pass (today's path copies once). That is fine: it guards the
  "no split when nothing is resident" branch after the change.
- The parity test passes in both states: it is the byte-identity guard.

- [ ] **Step 3: Implement**

`__init__`, after `self.row_fills = row_fills`:
```python
        self.split_fill_gather = envs.SGLANG_DSV41_ENABLE_PREFILL_SPLIT_GATHER.get()
```
In `gather_rows`, replace
```python
            self._await_fills(chunk_ids)
            fallback_used = self.copy_rows(chunk, chunk_outputs) or fallback_used
```
with
```python
            filling = self._filling_positions(chunk_ids)
            if filling and len(filling) < len(chunk_ids) and self._splits(chunk_outputs):
                # The resident rows' copy runs while the fills land: the same bytes to the same rows, in two
                # copies. The stream is idle here (the chunk's .item() above), so the row lists cost no wait.
                ready = [i for i in range(len(chunk_ids)) if i not in set(filling)]
                ready_rows = torch.tensor(ready, dtype=torch.int64, device=chunk.device)
                filling_rows = torch.tensor(filling, dtype=torch.int64, device=chunk.device)
                self.copy_rows(chunk, chunk_outputs, rows=ready_rows)
                self._await_fills([chunk_ids[i] for i in filling])
                self.copy_rows(chunk, chunk_outputs, rows=filling_rows)
            else:
                self._await_fills(chunk_ids)
                fallback_used = self.copy_rows(chunk, chunk_outputs) or fallback_used
```
The helpers, next to `_await_fills`:
```python
    def _filling_positions(self, chunk_ids: list[int]) -> list[int]:
        """Positions in ``chunk_ids`` of rows the running prefetch has not been waited for; [] unless splitting."""
        order = self._fill_order
        if not self.split_fill_gather or order is None:
            return []
        return [i for i, expert_id in enumerate(chunk_ids) if expert_id in order]

    def _splits(self, outputs: dict[str, torch.Tensor]) -> bool:
        """Whether copy_rows can copy named rows into ``outputs``: never through the non-contiguous fallback."""
        return all(
            outputs[name].device.type == "cpu"
            or (self.tensors[name].is_contiguous() and outputs[name].is_contiguous())
            for name in self.cached_names
        )
```

- [ ] **Step 4: Commit, push, run at the tip**

Run: `rpt.sh <sha> test/registered/unit/layers/moe/test_expert_pinned_row_fills.py test/registered/unit/layers/moe/test_exl3_prefill_fills_service.py test/registered/unit/layers/quantization/test_exl3_moe_stream_mode.py test/registered/unit/layers/quantization/test_exl3_route_plan.py`
Expected: all pass.

---

### Task 4: GPU tests, mutants, registered suite (divix01, production stopped)

**Files:** none new. This task runs what Tasks 2-3 wrote, on the GPU.

- [ ] **Step 1: GPU**

Run: `bash /mnt/nvme1/prefill-opt/gpt.sh <tip> test/registered/unit/layers/moe/test_expert_plugins_cuda.py test/manual/dsv41/test_exl3_stream_apply_gpu.py test/registered/unit/kernels/test_exl3_ram_miss_prefill_fills.py`
Expected: all pass, none skipped. Check with `--collect-only -q`.

- [ ] **Step 2: Mutants** (in `wt-route-plan`, via the `mutants.py` pattern in `/mnt/nvme1/prefill-opt/`; revert each)

| Mutant | Must be caught by |
|---|---|
| `_await_fills` moved before the first `copy_rows` (no overlap) | `test_split_gather_copies_resident_rows_before_...` |
| `ready`/`filling_rows` swapped | the parity test and the ordering test |
| `_gather_host_rows_to_kernel` stores to `tl.program_id(0)` instead of `output_row` | the CUDA `copy_rows` test |
| `_filling_positions` returns `[]` always | the ordering test |
| `_splits` returns True for non-contiguous outputs | the CPU fallback check: add a CPU test that passes a non-contiguous CUDA-like output if the mutant survives |

Re-run Steps 1 and 3's CPU command afterwards and record green.

- [ ] **Step 3: Registered suite, tip vs merge base, under the GPU lock**

Use `/mnt/nvme1/prefill-opt/suite.sh <tip> tip` and `suite.sh <base> base`, and record both counts. The known flake,
`test_no_worker_thread_exists_unless_asked_for_and_close_joins_them`, passes when rerun on its own.

---

### Task 5: A/B arms and a traced arm (divix01, production stopped)

**Files:**
- Create: `analysis/dsv41-drive/split-gather/drive_arms.sh`

- [ ] **Step 1: The driver**

```bash
#!/usr/bin/env bash
# Split fill gather A/B (plan 2026-09-26-dsv41-split-fill-gather): A = production recipe (route plan on), B = + the
# split, once each; then a node-mode traced B without the root PCIe session. Production must be stopped.
# Usage: drive_arms.sh SHA [ab|traced|all]
set -u
WT=/data/models/slang/nvfp4-work/wt-route-plan-arms
SHA=$1
MODE=${2:-all}
FLAG=SGLANG_DSV41_ENABLE_PREFILL_SPLIT_GATHER=1
GPU_LOCK=/data/models/slang/nvfp4-work/cc-gpu.lock
say() { echo "$(date +%T) $*"; }
wait_gpu() { while ! flock -n $GPU_LOCK true; do say "cc-gpu.lock held; waiting"; sleep 60; done; }
cd $WT
[ "$(git rev-parse HEAD)" = "$SHA" ] || { say "worktree not at $SHA"; exit 1; }
PYTHONPATH=$WT/python /data/models/slang/.venv/bin/python -c "import sglang; print(\"sglang from\", sglang.__file__)"
exec 8>/data/models/slang/nvfp4-work/rowimg-disk.lock
say "waiting for rowimg-disk.lock"; flock 8; say "rowimg-disk.lock held"
if [ "$MODE" = ab ] || [ "$MODE" = all ]; then
    wait_gpu; say "arm A"
    EXPECT_SHA=$SHA bash benchmarks/dsv41_baseline/run_arm.sh split-gather-A 30021; say "A rc=$?"
    wait_gpu; say "arm B"
    EXPECT_SHA=$SHA bash benchmarks/dsv41_baseline/run_arm.sh split-gather-B 30021 $FLAG; say "B rc=$?"
fi
if [ "$MODE" = traced ] || [ "$MODE" = all ]; then
    wait_gpu; say "arm B traced"
    NSYS_TMPDIR=/mnt/nvme1/nsys-tmp NSYS_TRACE=1 NSYS_CUDA_GRAPH_TRACE=node NSYS_GPU_METRICS=0 EXPECT_SHA=$SHA \
        bash benchmarks/dsv41_baseline/run_arm.sh split-gather-B-node 30021 $FLAG; say "B traced rc=$?"
fi
say "DRIVER DONE"
```

- [ ] **Step 2: Register and run**
  - Commit and push the driver.
  - In `wt-route-plan-arms`, `git checkout --detach <sha>` and register `git rev-parse HEAD:python` (the tree hash,
    not the path) in `benchmarks/dsv41_baseline/generations.json` via `generations.register`.
  - Launch with `setsid nohup bash analysis/dsv41-drive/split-gather/drive_arms.sh <full sha> all > /mnt/nvme1/prefill-opt/split-arms.log 2>&1 < /dev/null & disown`.
  - Afterwards, check that no orphaned `nsys` process holds `cc-gpu.lock` (`flock -n ... true`).

- [ ] **Step 3: Results**
  - TTFT per session and pooled ms/token for A and B, with output identity: `/mnt/nvme1/prefill-opt/arm_table.py`
    with the arm names changed.
  - Export the traced report to SQLite, then run `idle_by_host_state.py`, `compare_arms.py` (against
    `route-plan-fix-node-20260926-002944.sqlite`) and `chunk_host_time.py`.
  - Expected: the "fill wait, layer's first chunk" row falls well below 1,296 ms, with identical output.

---

### Task 6: Write-up

**Files:** `DSV41_REFERENCE.md` (new §27.12; §27.4 item 7)

- [ ] **Step 1:** Write §27.12. It covers:
  - what changed, with the flag and commits;
  - the tests with their commands and counts;
  - the mutant table;
  - the suite comparison;
  - the A/B table;
  - the idle-by-host-state table before and after;
  - what remains.
- [ ] **Step 2:** Update §27.4 item 7 with the new TTFT.
- [ ] **Step 3:** Commit and push. Ask the user before adding the flag to `arm_env.base_env()`, and before
  restarting production.

## Not in this plan

- **Reading a layer's fills before its routing is known.** That needs a predictor (Track B, §25.4).
- **Copying a filled row as soon as it lands**, instead of after the chunk's last fill. That needs a poll of
  `fill_landed` and more launches. Worth it only if Task 5's trace shows the later-chunk waits still large.
- **The 765 ms of Python and launch idle.** Attribute it separately, with the same host-state tooling.
