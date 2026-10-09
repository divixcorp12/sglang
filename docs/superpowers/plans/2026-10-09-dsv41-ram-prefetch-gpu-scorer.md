# DSV41 RAM Prefetch, Phase 2 (the gate scorer on the GPU) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Score layer T+1's router gate on the GPU right after layer T's post, inside the captured decode graph, and publish each record's up to 8 ranked candidates to a pinned page the host's speculative threads read, so a speculative NVMe read starts as soon as the host learns of the record instead of ~1.8 ms of CPU scoring later.

**Architecture:** Two kernels (`spec_score.cuh`, in the existing device module) run after `side.post` on the same stream: a multi-block score kernel writes `sqrt(softplus(w_e . x_t)) + b_e` for every live token into a device scratch, and a one-block select kernel ranks it exactly as `GateScorer::choose` does, past the experts VRAM-hot or RAM-mapped in layer T+1, and seqlock-publishes the candidates to slot `(seq - 1) % 16` of a pinned candidate page (`spec_candidates.h`). In GPU mode the host offers a job for every record of a row with a target; the speculative thread waits up to 200 us for the slot, re-checks the host map and the pool, and reads the first `per_layer` survivors of its own group, exactly as Phase 1 reads the CPU scorer's picks. `SGLANG_DSV41_RAM_PREFETCH_SCORER` selects `cpu` (default, unchanged) or `gpu`.

**Tech Stack:** CUDA (JIT device module `expert_stream_exl3*`, TVM FFI launchers), C++20 header-only host (`host/ram_tier.h`, ProdBuild/InstrBuild), Python 3 (torch, pytest), the RAM-miss service (`exl3_ram_miss.py`), the DSpark A/B driver (`analysis/dsv41-drive/dspark/both_cpu_ab.py`), Nsight Systems; every run on divix01 under the run protocol.

**Spec:** `docs/superpowers/specs/2026-10-09-dsv41-ram-prefetch-gpu-scorer-design.md` (binding). It builds on `docs/superpowers/specs/2026-10-08-dsv41-ram-prefetch-design.md` and `docs/superpowers/plans/2026-10-08-dsv41-ram-prefetch-phase1.md`; the CPU capture it argues from is `divix01:/data/models/slang/nvfp4-work/ram-prefetch/margin-20261009-002539`.

## Global Constraints

- Code is written on the laptop in `/Users/dnikolaidis/.codex/worktrees/ram-prefetch-margin` (branch `codex/dsv41-ram-prefetch-margin`), committed, pushed to `origin`, and run on divix01 only in the private worktree `/data/models/slang/nvfp4-work/wt-ram-prefetch-margin`, updated with `git fetch -q origin && git checkout -q --detach origin/codex/dsv41-ram-prefetch-margin`. Never rsync/scp a tree; never run in the production checkout (`.claude/rules/divix01-run-protocol.md`).
- Every test command is `env PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 taskset -c 0-63 /data/models/slang/.venv/bin/python -m pytest -q -p no:randomly <files>`, with pytest's own status read from `${PIPESTATUS[0]}`, never the `tail` status. Files of different test directories go in separate pytest invocations (pytest 9.1.1's conftest quirk). Record the exact selection next to any count quoted.
- GPU tests run as `flock /data/models/slang/nvfp4-work/cc-gpu.lock taskset -c 32-63 env PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 /data/models/slang/.venv/bin/python -m pytest -q -p no:randomly <files>`. A job needing both locks takes `rowimg-disk.lock` first, then `cc-gpu.lock`. Cores 64-71 stay free.
- Production on divix01 is stopped. Never restart it, never stop or kill a process you did not start; when a GPU step finds the GPU busy, stop and report.
- The first run after any C++/CUDA change rebuilds every JIT module it touches (50-100 s per host module, serialized). A `TimeoutExpired` on that run is the compiler: check the build-dir mtimes before calling it a hang.
- Mutants are applied in the divix01 worktree only, run, then reverted with `git checkout -- <file>`; never committed. After reverting, re-run the same selection and record that it is green.
- The option name is the spec's, verbatim: `SGLANG_DSV41_RAM_PREFETCH_SCORER`, an `EnvStr` in `python/sglang/srt/environ.py`'s DSV41 RAM-prefetch block, values `cpu` (default) and `gpu`, read with `.get()`, overridden in tests with `.override()` (`.claude/skills/env-var-conventions/SKILL.md`).
- The spec's numbers, verbatim: `kMaxCandidates` = 8, `kCandRecords` = 16 (= `Wire::kDemandRecords`), 128-byte slot stride, an 8-byte header (u32 seq, u16 count, u16 flags), 8-byte entries (u16 expert, u8 rank, a pad byte, fp32 margin), `kCandWait` = 200 us, `kDepth` = 12. Success: the two kernels <= 0.1 ms per layer; `spec_promoted / spec_used` < 15% (CPU capture: 42%); the A/B faster by the median and paired per session with the text inside the near-tie band.
- With the CPU scorer (the default) nothing changes on the request path: `offer_spec`'s CPU branch, the scorer and every Phase 1 test are as before; the hot-path golden test pins the option-off path.
- Nsight: graph-mode tracing is refused with the copy engine (`benchmarks/dsv41_baseline/nsys_capture.py`) and a graph-mode kernel table leaves out the graph body, so the kernels' per-launch cost comes from a `--cuda-graph-trace=node` trace of a one-session window, read only for those two kernels (never ms/token or step-tail idle from it). `NSYS_TMPDIR=/mnt/nvme1/nsys-tmp`, reports under `/mnt/nvme1/`, analysis on divix01 under `taskset -c 0-63`.
- Comments state constraints, not narration (`.claude/rules/comment-style.md`): one or two lines, ASCII, no TODO without an owner.
- Every commit ends with exactly these trailer lines; stage files by name; never amend or rebase:
  ```
  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01LxB2wQZv5EVNuBGUJJbkGk
  ```

## Review Focus

1. **The other group reads first.** In GPU mode group 0 can land its candidate for record S before group 1 reads S's slot. Group 1's list must still count that candidate toward the layer's `per_layer` (`pooled_before` excludes entries issued for the same seq), or group 1 reads the next candidate and the layer reads `per_layer + 1` rows. Pinned in Task 2 (`test_a_group_that_reads_first_does_not_shift_the_other_groups_list`).
2. **A slot that never comes.** A record whose select kernel never ran (a row posting without its input, an eager path) must cost its own job one 200 us wait and `spec_late`, never hold back the next job, which reads its own slot normally. Pinned in Task 2 (`test_a_slot_that_never_comes_costs_its_own_job_not_the_next`).
3. **Stale scratch rows.** The scratch is `[tokens_max, experts]` and keeps an earlier, larger record's rows past the current record's tokens; the select kernel must rank only the record's own live rows. Pinned in Task 3 (`test_a_record_with_fewer_tokens_ranks_only_its_own_rows_of_the_scratch`).
4. **NaN or infinite activations.** A NaN score (a NaN bias, or `inf * 0` in the dot product) must rank below every other expert in id order, never be chosen over a finite one, and never trap. Pinned in Task 3 (`test_nan_scores_rank_last_in_id_order` and `test_an_infinite_input_ranks_as_the_reference`).
5. **A gate wider than the select kernel.** A model with more experts than the select kernel's shared arrays (1024) must be refused when the scorer is enabled, at start, not overflow shared memory inside a replay. Pinned in Task 4 (`test_the_gpu_scorer_refuses_more_experts_than_the_select_kernel_holds`).

## File Structure

| File | Responsibility |
|---|---|
| `python/sglang/kernels/jit/csrc/moe/expert_stream/spec_candidates.h` (new) | `SpecCandidates`: the candidate page's layout, shared by the select kernel and the host |
| `python/sglang/kernels/jit/csrc/moe/expert_stream/spec_score.cuh` (new) | the score and select kernels, `SpecScoreParams`, `SpecSelectParams`, the checked launchers `SpecScoreKernel` |
| `python/sglang/kernels/jit/csrc/moe/exl3/exl3_ram_miss.cuh` | includes `spec_score.cuh` (one device module, already loaded before any capture) |
| `.../expert_stream/host/ram_prefetch.h` | `RamPrefetchConfig::candidates`; the wait constants |
| `.../expert_stream/host/tier_protocol.h` | `kSpecLate`, a core counter |
| `.../expert_stream/host/ram_tier.h` | GPU-mode `enable_ram_prefetch`, `offer_spec`, `serve_gpu_job`, `read_candidates`, `await_candidates` |
| `.../expert_stream/host/ffi_exports.h` | `enable_ram_prefetch` takes the candidate page |
| `python/sglang/kernels/ops/moe/expert_stream_transport.py` | `CAND_*`, `new_candidate_page`, `candidate_offset`, `read_candidates`, `spec_late`, the wrapper's `candidates=`, `run_spec_score`, `run_spec_select`, `ExpertStreamDevice.enable_spec_scorer` / `spec_score` |
| `python/sglang/srt/environ.py`, `python/sglang/srt/arg_groups/expert_stream_requirements_exl3.py` | the option and its launch rules |
| `python/sglang/srt/layers/moe/ram_prefetch.py` | `SCORERS`, `PrefetchTargets`, `prefetch_targets`, `SpecScoreRow`, `spec_score_rows`, `GpuScorer` |
| `python/sglang/srt/layers/moe/exl3_ram_miss.py` | the scorer in `_enable_ram_prefetch`, the per-row table at attach, the launch after `side.post`, quarantine |
| `python/sglang/test/dsv41_ram_prefetch_fixtures.py` | `gate_reference` and its cases (moved), `enable_gpu`, `write_candidate_slot` |
| `analysis/dsv41-drive/dspark/both_cpu_ab.py`, `spec_margin_capture.py` | the `dspark-both-prefetch-gpu` arms, `spec_late`, in-flight share and paired gain; `--scorer` |
| Tests | `test/registered/unit/kernels/test_exl3_ram_prefetch_gpu_host.py` (new), `test/manual/dsv41/test_exl3_ram_prefetch_gpu_scorer_cuda.py` (new), `test/registered/unit/scripts/test_spec_margin_capture.py` (new), additions to `test_expert_stream_requirements_exl3.py`, `test_exl3_ram_miss_device_args.py`, `test_exl3_ram_miss_attach_lanes.py`, `test_ram_prefetch_tables.py`, `test_both_cpu_ab.py`, `test_exl3_ram_prefetch_scorer.py` (imports only) |
| `DSV41_REFERENCE.md` | §33.15, the results |

## Task list

1. The option and its launch rules.
2. The candidate page and the host in GPU mode (`spec_late`, waiting, filters, budget, both groups).
3. The scoring kernels, with GPU parity, oversize, lap and publication-order tests.
4. Wiring: the per-row table at attach, the launch after `side.post`, the device scratch, quarantine, the captured-graph replay test.
5. The A/B arms and the capture's `--scorer`.
6. Regression and measurement on divix01 (Nsight, instrumented capture, the reversed A/B), results in `DSV41_REFERENCE.md` §33.15.

## Decisions this plan takes where the spec is silent

- The kernels live in the existing device module (`exl3_ram_miss.cuh` includes `spec_score.cuh`): no new JIT module exists to load, so nothing new can load during capture, and under `CUDA_MODULE_LOADING=EAGER` both instantiations load with the module at the first post.
- Score kernel: 256 threads (8 warps) per block, one expert per warp at a time, `ceil(experts / 8)` blocks (48 for 384 experts); lanes stride the hidden size, 8 tokens accumulated per pass over a gate row; scalar coalesced loads; fp32 `__fmaf_rn`, a xor-shuffle sum, then `__fsqrt_rn` and `__fadd_rn` so integer logits score bit for bit as the host does; NaN stored as -inf.
- Select kernel: one block of 256 threads. Warp w orders tokens w, w+8, ...: 12 rounds of a warp argmax by (score desc, id asc), each lane holding experts lane, lane+32, ... with a taken bitmask. Thread 0 then walks, merges and picks the top 8 by (best margin desc, id asc) over the compact list of picked experts, mirroring `GateScorer::choose` line for line. Shared arrays bound it to 1024 experts and 32 tokens; the launcher and `enable_spec_scorer` refuse more.
- Publication: thread 0 stores seq 0, a system release fence, the u32 `count | flags << 16`, each entry as two u32 (`expert | rank << 16`, the margin's bits), then the seq with a system release. Flags: bit 0 `kCandFlagOversize`, set with count 0 when the record's tokens are outside 1..`tokens_max` (also 0 tokens: a post without input).
- The record's seq is the post's `state[kPosted]`, read by the select kernel; both kernels launch without PDL, so stream order puts them after the post's completion.
- Scoring launches on every post of a row with a target, captured or eager, after `side.post` and before C1; the warm-up's eager forwards therefore exercise both kernels before capture.
- The "per-row device table" is a per-row Python entry (`SpecScoreRow`: the target's live gate weight and bias tensors, target row, hot-slot tensor and capacity) whose pointers become the launches' `__grid_constant__` parameters; the `ram_slot` row is the device map bank's row `target`. It is built at attach, as each row and its target attach (`_note_spec_row`), which `ModelRunner.__init__` runs in `maybe_init_expert_hot_cache` before graph capture; a post of a row with a target and no entry raises.
- `x` must be bf16 (DSV4's MoE input); the gate weight bf16 or fp32 (`router_fp32`); the bias fp32. Anything else is refused when the table is built.
- Host wait: `kCandWaitNs` = 200,000, spinning with `_mm_pause` for the first 20,000 ns (`kCandSpinNs`), then `sleep_for` 10,000 ns steps (`kCandSleepNs`), on `now_ns()`. The staleness check runs before the wait (a stale job never waits). A slot whose seq word is 0 (being rewritten) or older is "not yet"; newer, or changed across the copy, is lapped.
- Host consumption: a count above 8 is clamped to 8 and an entry naming an expert outside the gate is passed over (the kernel writes neither); neither fail-stops, since the candidates are a prediction. The host does not re-check the hot set (the GPU's view is the newer one).
- In GPU mode the host gets the target table (the gate column is ignored), no gates, and the candidate page; `hidden` and the gate count are not checked. CPU experts stay required on every group (the pool serves forced CPU misses).
- `kSpecLate` is a core counter placed after `kSpecDelayed`, so production A/Bs report it.
- The A/B summary gains `spec_late`, `spec_in_flight_at_use` (`spec_promoted / spec_used`) and a paired per-session gain against the arm's reference, keyed by `session_id`.
- A `dspark-both-prefetch-gpu-topk` arm is added for the spec's optional top-k-only A/B; it runs only if the owner asks.

---

### Task 1: The option and its launch rules

**Files:**
- Modify: `python/sglang/srt/environ.py` (directly after `SGLANG_DSV41_RAM_PREFETCH_TOP_K_ONLY = EnvBool(False)`, ~line 2013)
- Modify: `python/sglang/srt/layers/moe/ram_prefetch.py` (the bounds block at the top)
- Modify: `python/sglang/srt/arg_groups/expert_stream_requirements_exl3.py` (`_check_ram_prefetch`, ~lines 340-360)
- Test: `test/registered/unit/test_expert_stream_requirements_exl3.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `envs.SGLANG_DSV41_RAM_PREFETCH_SCORER` (`EnvStr("cpu")`); `ram_prefetch.SCORERS = ("cpu", "gpu")`; the launch gate refuses an unknown scorer, `gpu` without `SGLANG_DSV41_RAM_PREFETCH=1`, and `gpu` without `SGLANG_DSV41_ENABLE_RAM_MISS_COPY_ENGINE=1`.

- [ ] **Step 0: Prepare the divix01 worktree and check the interpreter**

```bash
git -C /Users/dnikolaidis/.codex/worktrees/ram-prefetch-margin push origin codex/dsv41-ram-prefetch-margin
ssh divix01 bash -s <<'REMOTE'
set -e
WT=/data/models/slang/nvfp4-work/wt-ram-prefetch-margin
if [ ! -d "$WT" ]; then
  git -C /data/models/slang/sglang fetch origin
  git -C /data/models/slang/sglang worktree add --detach "$WT" origin/codex/dsv41-ram-prefetch-margin
fi
cd "$WT" && git fetch -q origin && git checkout -q --detach origin/codex/dsv41-ram-prefetch-margin && git log -1 --oneline
PYTHONPATH=$PWD/python /data/models/slang/.venv/bin/python -c "import sglang; print(sglang.__file__)"
REMOTE
```
Expected: the branch head's one-line log, then `/data/models/slang/nvfp4-work/wt-ram-prefetch-margin/python/sglang/__init__.py`. Any other path means the runs would test other code: stop and fix `PYTHONPATH`.

- [ ] **Step 1: Write the failing tests**

Append to `test/registered/unit/test_expert_stream_requirements_exl3.py`:

```python
def test_the_ram_prefetch_scorer_defaults_to_the_cpu():
    """The GPU scorer is opt-in until its A/B is accepted (spec 2026-10-09-dsv41-ram-prefetch-gpu-scorer-design)."""
    assert envs.SGLANG_DSV41_RAM_PREFETCH_SCORER.get() == "cpu"


def test_the_gpu_scorer_runs_on_the_prefetch_and_the_captured_copy_engine_post(model_dir):
    """gpu needs the prefetch it feeds, and the copy-engine post whose capture the scoring kernels follow."""
    args = _launch(model_dir, cuda_graph_config=BREAKABLE_BS1)
    assert expert_stream_requirements_for(args, args).label == "EXL3"
    _gate(args, **CPU_EXPERTS_ENV, SGLANG_DSV41_RAM_PREFETCH=True, SGLANG_DSV41_RAM_PREFETCH_SCORER="gpu")
    with pytest.raises(ValueError, match="SGLANG_DSV41_RAM_PREFETCH_SCORER=gpu needs SGLANG_DSV41_RAM_PREFETCH=1"):
        _gate(args, **CPU_EXPERTS_ENV, SGLANG_DSV41_RAM_PREFETCH_SCORER="gpu")
    no_copy_engine = {**CPU_EXPERTS_ENV, "SGLANG_DSV41_ENABLE_RAM_MISS_COPY_ENGINE": False}
    with pytest.raises(ValueError, match="SGLANG_DSV41_RAM_PREFETCH_SCORER=gpu runs after the copy-engine post"):
        _gate(args, **no_copy_engine, SGLANG_DSV41_RAM_PREFETCH=True, SGLANG_DSV41_RAM_PREFETCH_SCORER="gpu")


@pytest.mark.parametrize("prefetch", [False, True])
def test_an_unknown_ram_prefetch_scorer_is_refused_whether_or_not_the_prefetch_is_on(model_dir, prefetch):
    """A typo must not fall back to the CPU scorer silently, nor pass while the prefetch is off."""
    args = _launch(model_dir, cuda_graph_config=BREAKABLE_BS1)
    with pytest.raises(ValueError, match="SGLANG_DSV41_RAM_PREFETCH_SCORER must be one of"):
        _gate(args, **CPU_EXPERTS_ENV, SGLANG_DSV41_RAM_PREFETCH=prefetch, SGLANG_DSV41_RAM_PREFETCH_SCORER="tpu")
```

- [ ] **Step 2: Commit the failing tests, push, and run them on divix01**

```bash
cd /Users/dnikolaidis/.codex/worktrees/ram-prefetch-margin
git add test/registered/unit/test_expert_stream_requirements_exl3.py
git commit -m "Test the RAM prefetch scorer option and its launch rules

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01LxB2wQZv5EVNuBGUJJbkGk"
git push origin codex/dsv41-ram-prefetch-margin
ssh divix01 bash -s <<'REMOTE'
set -u
cd /data/models/slang/nvfp4-work/wt-ram-prefetch-margin && git fetch -q origin && git checkout -q --detach origin/codex/dsv41-ram-prefetch-margin && git log -1 --oneline
env PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 taskset -c 0-63 /data/models/slang/.venv/bin/python -m pytest -q -p no:randomly test/registered/unit/test_expert_stream_requirements_exl3.py -k "scorer" 2>&1 | tail -12; echo "EXIT=${PIPESTATUS[0]}"
REMOTE
```
Expected: `EXIT=1`, failures with `AttributeError: ... SGLANG_DSV41_RAM_PREFETCH_SCORER`.

- [ ] **Step 3: Add the option and the rules**

In `python/sglang/srt/environ.py`, directly after `SGLANG_DSV41_RAM_PREFETCH_TOP_K_ONLY = EnvBool(False)`:

```python
    # Where the next layer's gate is scored: "cpu", each group's speculative thread on the record's staged input, or
    # "gpu", two kernels after the layer's post inside the decode graph (spec 2026-10-09-dsv41-ram-prefetch-gpu-scorer).
    SGLANG_DSV41_RAM_PREFETCH_SCORER = EnvStr("cpu")
```

In `python/sglang/srt/layers/moe/ram_prefetch.py`, directly after `MAX_SPEC_SHARE = 4`:

```python
# SGLANG_DSV41_RAM_PREFETCH_SCORER's values.
SCORERS = ("cpu", "gpu")
```

In `python/sglang/srt/arg_groups/expert_stream_requirements_exl3.py`, replace the body of `_check_ram_prefetch` (keep its docstring) from `if not envs.SGLANG_DSV41_RAM_PREFETCH.get():` through the end of the bounds loop with:

```python
    from sglang.srt.layers.moe import ram_prefetch

    scorer = envs.SGLANG_DSV41_RAM_PREFETCH_SCORER.get()
    if scorer not in ram_prefetch.SCORERS:
        raise ValueError(f"SGLANG_DSV41_RAM_PREFETCH_SCORER must be one of {ram_prefetch.SCORERS}, got {scorer!r}")
    if not envs.SGLANG_DSV41_RAM_PREFETCH.get():
        if scorer == "gpu":
            raise ValueError("SGLANG_DSV41_RAM_PREFETCH_SCORER=gpu needs SGLANG_DSV41_RAM_PREFETCH=1")
        return
    if not envs.SGLANG_DSV41_CPU_EXPERTS.get():
        # Only a record with a CPU lane stages the input the scorer reads, and only a forced CPU miss uses the pool.
        raise ValueError("SGLANG_DSV41_RAM_PREFETCH needs SGLANG_DSV41_CPU_EXPERTS=1")
    if scorer == "gpu" and not envs.SGLANG_DSV41_ENABLE_RAM_MISS_COPY_ENGINE.get():
        raise ValueError(
            "SGLANG_DSV41_RAM_PREFETCH_SCORER=gpu runs after the copy-engine post captured in the decode graph; "
            "set SGLANG_DSV41_ENABLE_RAM_MISS_COPY_ENGINE=1"
        )
    for name, high in (
        ("SGLANG_DSV41_RAM_PREFETCH_PER_TOKEN", ram_prefetch.MAX_PER_TOKEN),
        ("SGLANG_DSV41_RAM_PREFETCH_PER_LAYER", ram_prefetch.MAX_PER_LAYER),
        ("SGLANG_DSV41_RAM_PREFETCH_SPEC_SHARE", ram_prefetch.MAX_SPEC_SHARE),
    ):
        value = getattr(envs, name).get()
        if not 1 <= value <= high:
            raise ValueError(f"{name} must be in [1, {high}], got {value}")
```

- [ ] **Step 4: Commit, push, and run the file on divix01**

```bash
cd /Users/dnikolaidis/.codex/worktrees/ram-prefetch-margin
git add python/sglang/srt/environ.py python/sglang/srt/layers/moe/ram_prefetch.py python/sglang/srt/arg_groups/expert_stream_requirements_exl3.py
git commit -m "Add SGLANG_DSV41_RAM_PREFETCH_SCORER and refuse a GPU scorer the launch cannot run

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01LxB2wQZv5EVNuBGUJJbkGk"
git push origin codex/dsv41-ram-prefetch-margin
ssh divix01 bash -s <<'REMOTE'
set -u
cd /data/models/slang/nvfp4-work/wt-ram-prefetch-margin && git fetch -q origin && git checkout -q --detach origin/codex/dsv41-ram-prefetch-margin && git log -1 --oneline
env PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 taskset -c 0-63 /data/models/slang/.venv/bin/python -m pytest -q -p no:randomly test/registered/unit/test_expert_stream_requirements_exl3.py 2>&1 | tail -3; echo "EXIT=${PIPESTATUS[0]}"
REMOTE
```
Expected: `EXIT=0`; the whole file passes (the Phase 1 prefetch tests included).

---

### Task 2: The candidate page and the host in GPU mode

**Files:**
- Create: `python/sglang/kernels/jit/csrc/moe/expert_stream/spec_candidates.h`
- Modify: `python/sglang/kernels/jit/csrc/moe/expert_stream/host/tier_protocol.h` (`enum Counter`, `is_core_counter`)
- Modify: `python/sglang/kernels/jit/csrc/moe/expert_stream/host/ram_prefetch.h` (`RamPrefetchConfig`, new wait constants, `SpecJob` comment)
- Modify: `python/sglang/kernels/jit/csrc/moe/expert_stream/host/ram_tier.h` (include, `enable_ram_prefetch` ~1063, `offer_spec` ~2157, `serve_spec_job` ~2208, new `serve_gpu_job` / `read_candidates` / `await_candidates`)
- Modify: `python/sglang/kernels/jit/csrc/moe/expert_stream/host/ffi_exports.h` (`enable_ram_prefetch` ~742)
- Modify: `python/sglang/kernels/ops/moe/expert_stream_transport.py` (`COUNTERS`, `CORE_COUNTERS`, new `CAND_*` block, `ExpertStreamHost.enable_ram_prefetch`)
- Modify: `python/sglang/test/dsv41_ram_prefetch_fixtures.py` (`enable_gpu`, `write_candidate_slot`)
- Test: `test/registered/unit/kernels/test_exl3_ram_prefetch_gpu_host.py` (new), `test/registered/unit/kernels/test_exl3_ram_miss_device_args.py` (layout mirror)

**Interfaces:**
- Consumes: Task 1's option name (only in comments here).
- Produces:
  - C++ `sglang::expert_stream::wire::SpecCandidates` with `kMaxCandidates = 8`, `kCandRecords = 16`, `kCandSeq = 0`, `kCandCount = 4`, `kCandFlags = 6`, `kCandEntries = 8`, `kCandEntryBytes = 8`, `kCandExpert = 0`, `kCandRank = 2`, `kCandMargin = 4`, `kCandPayloadBytes = 72`, `kCandStride = 128`, `kCandPageBytes = 2048`, `kCandFlagOversize = 1`, `static constexpr int64_t slot_offset(uint32_t seq)` (host only).
  - C++ counter `kSpecLate` (core); `RamPrefetchConfig::candidates` (`const uint8_t*`, null = CPU scorer); `kCandWaitNs = 200'000`, `kCandSpinNs = 20'000`, `kCandSleepNs = 10'000`.
  - FFI `expert_stream_enable_ram_prefetch(handle, targets, gates, bias, cores, hidden, top_k, per_token, per_layer, top_k_only, candidates)`; `candidates` uint8 `[0]` (CPU scorer) or `[2048]` (GPU scorer, then `gates` is `[0, 0]`).
  - Python `CAND_MAX = 8`, `CAND_RECORDS = 16`, `CAND_STRIDE = 128`, `CAND_ENTRIES = 8`, `CAND_ENTRY_BYTES = 8`, `CAND_PAGE_BYTES = 2048`, `CAND_FLAG_OVERSIZE = 1`; `new_candidate_page(*, pin: bool) -> torch.Tensor`; `candidate_offset(seq: int) -> int`; `CandidateRead(status: str, count: int = 0, flags: int = 0, picks: tuple = ())` (a `NamedTuple`; `status` in `"ready"`, `"not_yet"`, `"lapped"`; `picks` of `(expert: int, rank: int, margin: float)`); `read_candidates(page, seq) -> CandidateRead`; `"spec_late"` in `COUNTERS` and `CORE_COUNTERS` after `"spec_delayed"`.
  - `ExpertStreamHost.enable_ram_prefetch(targets, gates, bias, *, top_k, per_token, per_layer, cores, top_k_only=False, candidates=None)`: with `candidates`, `gates` and `bias` must be `None`.
  - Fixtures: `enable_gpu(rig, *, targets=None, top_k=2, per_token=1, per_layer=1) -> torch.Tensor` (the page); `write_candidate_slot(page, seq, picks, *, flags=0, slot_seq=None) -> None`.

- [ ] **Step 1: Write the fixtures and the failing tests**

Append to `python/sglang/test/dsv41_ram_prefetch_fixtures.py` (add `import struct` to the imports, and `candidate_offset, new_candidate_page` to the `expert_stream_transport` import):

```python
def enable_gpu(rig: PrefetchRig, *, targets=None, top_k=2, per_token=1, per_layer=1) -> torch.Tensor:
    """The GPU scorer on the rig (spec 2026-10-09-dsv41-ram-prefetch-gpu-scorer-design): row 0 targets row 1 unless
    `targets` (int64 [rows, 2]) says otherwise. Returns the candidate page, which the test writes as the select kernel
    would (write_candidate_slot)."""
    rows = rig.x_rows.shape[0]
    if targets is None:
        targets = torch.tensor([[1, 0]] + [[-1, -1]] * (rows - 1), dtype=torch.int64)
    page = new_candidate_page(pin=False)
    rig.host.enable_ram_prefetch(
        targets,
        None,
        None,
        top_k=top_k,
        per_token=per_token,
        per_layer=per_layer,
        cores=[[] for _ in range(rig.host.nodes)],
        candidates=page,
    )
    return page


def write_candidate_slot(page: torch.Tensor, seq: int, picks, *, flags: int = 0, slot_seq=None) -> None:
    """Writes `seq`'s slot as the select kernel does: the count, flags and (expert, rank, margin) entries, then the seq
    word. `slot_seq` stores another word instead: a later record's (a lapping select) or 0 (a slot still open)."""
    off = candidate_offset(seq)
    payload = struct.pack("<HH", len(picks), flags) + b"".join(struct.pack("<HBxf", e, r, m) for e, r, m in picks)
    page[off + 4 : off + 4 + len(payload)] = torch.frombuffer(bytearray(payload), dtype=torch.uint8)
    word = (seq if slot_seq is None else slot_seq) & 0xFFFFFFFF
    page[off : off + 4] = torch.frombuffer(bytearray(struct.pack("<I", word)), dtype=torch.uint8)
```

Create `test/registered/unit/kernels/test_exl3_ram_prefetch_gpu_host.py`:

```python
"""The host's half of the GPU scorer (spec 2026-10-09-dsv41-ram-prefetch-gpu-scorer-design, "The host in GPU mode"),
driven on the test's thread with spec_pump: every record of a row with a target feeds each group's ring; the job waits
for the record's candidate slot, which the test writes as the select kernel would, then reads in the GPU's order the
first per_layer candidates still unmapped and not pooled before, its own group's only (CPU, ChainSim)."""

import json
import struct
import time

import pytest
import torch

from sglang.kernels.ops.moe.expert_stream_transport import CAND_FLAG_OVERSIZE
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.dsv41_ram_prefetch_fixtures import (
    LOGITS,
    enable,
    enable_gpu,
    load,
    prefetch_rig,
    trigger,
    write_candidate_slot,
)

register_cpu_ci(est_time=30, suite="base-a-test-cpu")


def _landed(host, row, group=None):
    return sorted(
        e["expert"] for e in host.spec_pool(row) if e["state"] == "landed" and (group is None or e["group"] == group)
    )


def _record(rig, row=0):
    """A record of `row` with no lane, served: with the GPU scorer it still feeds every group's ring."""
    req = rig.sim.post(row, [])
    assert rig.host.pump() == 1 and rig.sim.wait_handled(req)
    return req


def _counts(rig, *names):
    c = rig.host.counters()
    return tuple(c[n] for n in names)


def test_a_ready_slot_is_read_in_the_gpus_order_past_mapped_and_pooled_experts(tmp_path):
    """4 is mapped, 0 pooled for an earlier record, 2 hot in the host's view (the GPU filtered hot; the host does not):
    per_layer 2 reads 2 and 1. Mutants: drop the host's map check, or its pooled_before check -- red (each spends the
    budget on a candidate spec_read then drops)."""
    rig = prefetch_rig(tmp_path, capacity=9, share=3)
    try:
        page = enable_gpu(rig, per_layer=2)
        load(rig, 1, [4])
        rig.host.spec_place(1, 0)
        rig.host.set_hot(1, [2])
        req = _record(rig)
        write_candidate_slot(page, req.seq, [(2, 0, 1.5), (4, 1, 1.0), (0, 0, 0.5), (1, 2, 0.25), (3, 3, 0.0)])
        assert rig.host.spec_pump(0) and not rig.host.spec_pump(0)
        assert _landed(rig.host, 1) == [0, 1, 2]
        assert _counts(rig, "spec_issued", "spec_landed", "spec_late", "spec_dropped") == (2, 2, 0, 0)
    finally:
        rig.host.stop()


@pytest.mark.parametrize("slot_seq", [None, 0])
def test_a_slot_never_ready_counts_late_after_the_wait(tmp_path, slot_seq):
    """Never written, or left open (seq word 0): the job waits kCandWait (200 us) and reads nothing."""
    rig = prefetch_rig(tmp_path)
    try:
        page = enable_gpu(rig)
        req = _record(rig)
        if slot_seq is not None:
            write_candidate_slot(page, req.seq, [(2, 0, 1.0)], slot_seq=slot_seq)
        start = time.perf_counter()
        assert rig.host.spec_pump(0)
        assert time.perf_counter() - start >= 200e-6
        assert _counts(rig, "spec_late", "spec_issued", "spec_dropped") == (1, 0, 0)
    finally:
        rig.host.stop()


def test_a_lapped_slot_is_dropped(tmp_path):
    """The slot holds the record 16 seqs later: its candidates are another layer's. Mutant: treat any other seq as not
    yet -- red (spec_late instead)."""
    rig = prefetch_rig(tmp_path)
    try:
        page = enable_gpu(rig)
        req = _record(rig)
        write_candidate_slot(page, req.seq, [(2, 0, 1.0)], slot_seq=req.seq + 16)
        assert rig.host.spec_pump(0)
        assert _counts(rig, "spec_dropped", "spec_late", "spec_issued") == (1, 0, 0)
    finally:
        rig.host.stop()


def test_a_count_of_zero_is_dropped(tmp_path):
    """An oversize record (prefill) publishes count 0 with the oversize flag."""
    rig = prefetch_rig(tmp_path)
    try:
        page = enable_gpu(rig)
        req = _record(rig)
        write_candidate_slot(page, req.seq, [], flags=CAND_FLAG_OVERSIZE)
        assert rig.host.spec_pump(0)
        assert _counts(rig, "spec_dropped", "spec_late", "spec_issued") == (1, 0, 0)
    finally:
        rig.host.stop()


def test_every_record_of_a_row_with_a_target_feeds_the_ring_and_no_other_does(tmp_path):
    """Row 0 has a target and its record has no CPU lane: a job. Row 1 has none: no job. Mutant: keep the CPU
    scorer's staged-lane condition in GPU mode -- red."""
    rig = prefetch_rig(tmp_path)
    try:
        enable_gpu(rig)
        _record(rig, 1)
        assert not rig.host.spec_pump(0)
        _record(rig, 0)
        assert rig.host.spec_pump(0)
    finally:
        rig.host.stop()


def test_a_job_whose_target_was_served_first_is_dropped_without_waiting(tmp_path):
    rig = prefetch_rig(tmp_path)
    try:
        page = enable_gpu(rig)
        req = _record(rig)
        write_candidate_slot(page, req.seq, [(2, 0, 1.0)])
        _record(rig, 1)
        assert rig.host.spec_pump(0)
        assert _counts(rig, "spec_dropped", "spec_late", "spec_issued") == (1, 0, 0)
    finally:
        rig.host.stop()


def test_a_slot_that_never_comes_costs_its_own_job_not_the_next(tmp_path):
    """Review Focus 2. Rows 0 -> 1 and 2 -> 3: row 0's slot never comes, row 2's does. Mutant: return early from the
    ring on a late slot -- red (row 2's job is left)."""
    rig = prefetch_rig(tmp_path, rows=4)
    try:
        targets = torch.tensor([[1, 0], [-1, -1], [3, 1], [-1, -1]], dtype=torch.int64)
        page = enable_gpu(rig, targets=targets)
        _record(rig, 0)
        second = _record(rig, 2)
        write_candidate_slot(page, second.seq, [(2, 0, 1.0)])
        assert rig.host.spec_pump(0) and rig.host.spec_pump(0)
        assert _counts(rig, "spec_late", "spec_issued") == (1, 1) and _landed(rig.host, 3) == [2]
    finally:
        rig.host.stop()


def test_both_groups_read_the_same_list_and_each_its_own_experts(tmp_path):
    """Two groups, per_layer 2 over [2 (group 0), 3 (group 1), 5 (group 1)]: group 0 reads 2, group 1 reads 3."""
    rig = prefetch_rig(tmp_path, nodes=2, share=1)
    try:
        page = enable_gpu(rig, per_layer=2)
        req = _record(rig)
        write_candidate_slot(page, req.seq, [(2, 0, 1.0), (3, 0, 0.5), (5, 1, 0.25)])
        assert rig.host.spec_pump(0) and rig.host.spec_pump(1)
        assert _landed(rig.host, 1, 0) == [2] and _landed(rig.host, 1, 1) == [3]
        assert _counts(rig, "spec_issued") == (2,)
    finally:
        rig.host.stop()


def test_the_layer_budget_holds_over_both_groups(tmp_path):
    """per_layer 2 over [2, 4 (both group 0), 3 (group 1)]: group 0 reads both, group 1 nothing."""
    rig = prefetch_rig(tmp_path, nodes=2, share=2)
    try:
        page = enable_gpu(rig, per_layer=2)
        req = _record(rig)
        write_candidate_slot(page, req.seq, [(2, 0, 1.0), (4, 1, 0.5), (3, 0, 0.25)])
        assert rig.host.spec_pump(0) and rig.host.spec_pump(1)
        assert _landed(rig.host, 1, 0) == [2, 4] and _landed(rig.host, 1, 1) == []
        assert _counts(rig, "spec_issued") == (2,)
    finally:
        rig.host.stop()


def test_a_group_that_reads_first_does_not_shift_the_other_groups_list(tmp_path):
    """Review Focus 1. per_layer 1 over [2 (group 0), 3 (group 1)]: group 0 lands 2 first; group 1 must still count 2
    and read nothing. Mutant (ram_prefetch.h pooled_before): count an entry issued for the same seq -- red (group 1
    reads 3)."""
    rig = prefetch_rig(tmp_path, nodes=2, share=1)
    try:
        page = enable_gpu(rig, per_layer=1)
        req = _record(rig)
        write_candidate_slot(page, req.seq, [(2, 0, 1.0), (3, 0, 0.5)])
        assert rig.host.spec_pump(0)
        assert _landed(rig.host, 1, 0) == [2]
        assert rig.host.spec_pump(1)
        assert _landed(rig.host, 1, 1) == [] and _counts(rig, "spec_issued") == (1,)
    finally:
        rig.host.stop()


def test_the_cpu_scorer_never_counts_late(tmp_path):
    """The default scorer's path is Phase 1's: a CPU record scores and reads; nothing waits for a slot."""
    rig = prefetch_rig(tmp_path)
    try:
        enable(rig, LOGITS)
        trigger(rig)
        assert rig.host.spec_pump(0)
        assert _counts(rig, "spec_late", "spec_issued") == (0, 1)
    finally:
        rig.host.stop()


def test_spec_submit_carries_the_gpus_rank_and_margin_and_the_wait_is_metered(tmp_path, monkeypatch):
    """gen = rank << 32 | the margin's fp32 bits, from the slot; spec_scored / spec_score_ns count the host's wait."""
    (tmp_path / "rig").mkdir()
    rig = prefetch_rig(tmp_path / "rig")
    try:
        monkeypatch.setenv("SGLANG_DSV41_EXPERT_JOB_TRACE_PREFIX", str(tmp_path / "jobs"))
        page = enable_gpu(rig)
        req = _record(rig)
        write_candidate_slot(page, req.seq, [(2, 3, 0.75)])
        assert rig.host.spec_pump(0)
        c = rig.host.counters()
        assert c["spec_scored"] == 1 and c["spec_score_ns"] > 0
    finally:
        rig.host.stop()
    files = list(tmp_path.glob("jobs.*.exl3-spec0.*.jsonl"))
    assert len(files) == 1
    events = list(map(json.loads, files[0].read_text().splitlines()[1:-1]))
    submit = next(e for e in events if e["event"] == "spec_submit")
    assert (submit["a"], submit["seq"]) == (2, req.seq)
    assert submit["gen"] == (3 << 32) | struct.unpack("<I", struct.pack("<f", 0.75))[0]


def test_a_gpu_scorer_takes_no_host_gates_and_a_page_of_the_slots_size(tmp_path):
    rig = prefetch_rig(tmp_path)
    try:
        targets = torch.tensor([[1, 0], [-1, -1]], dtype=torch.int64)
        cores = [[]]
        with pytest.raises(ValueError, match="gates=None and bias=None"):
            rig.host.enable_ram_prefetch(
                targets, torch.zeros((1, 6, 8), dtype=torch.bfloat16), torch.zeros((1, 6)), top_k=2, per_token=1,
                per_layer=1, cores=cores, candidates=torch.zeros(2048, dtype=torch.uint8),
            )
        with pytest.raises(ValueError, match="2048 bytes"):
            rig.host.enable_ram_prefetch(
                targets, None, None, top_k=2, per_token=1, per_layer=1, cores=cores,
                candidates=torch.zeros(1024, dtype=torch.uint8),
            )
    finally:
        rig.host.stop()
```

Append to `test/registered/unit/kernels/test_exl3_ram_miss_device_args.py`:

```python
def test_the_candidate_page_layout_is_the_python_mirror():
    """spec_candidates.h's literal offsets against expert_stream_transport's CAND_* and read_candidates' struct
    formats (<IHH header, <HBxf entry)."""
    text = (CSRC / "expert_stream" / "spec_candidates.h").read_text()
    found = {k: int(v) for k, v in re.findall(r"static constexpr \w+ (k\w+) = (\d+);", text)}
    want = {
        "kMaxCandidates": ram_miss.CAND_MAX,
        "kCandRecords": ram_miss.CAND_RECORDS,
        "kCandSeq": 0,
        "kCandCount": 4,
        "kCandFlags": 6,
        "kCandEntries": ram_miss.CAND_ENTRIES,
        "kCandEntryBytes": ram_miss.CAND_ENTRY_BYTES,
        "kCandExpert": 0,
        "kCandRank": 2,
        "kCandMargin": 4,
        "kCandStride": ram_miss.CAND_STRIDE,
        "kCandFlagOversize": ram_miss.CAND_FLAG_OVERSIZE,
    }
    assert {k: found.get(k) for k in want} == want
    assert ram_miss.CAND_PAGE_BYTES == ram_miss.CAND_RECORDS * ram_miss.CAND_STRIDE == 2048
```

- [ ] **Step 2: Commit the failing tests, push, and run them on divix01**

```bash
cd /Users/dnikolaidis/.codex/worktrees/ram-prefetch-margin
git add python/sglang/test/dsv41_ram_prefetch_fixtures.py test/registered/unit/kernels/test_exl3_ram_prefetch_gpu_host.py test/registered/unit/kernels/test_exl3_ram_miss_device_args.py
git commit -m "Test the host's GPU-scorer mode against slots written as the select kernel would

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01LxB2wQZv5EVNuBGUJJbkGk"
git push origin codex/dsv41-ram-prefetch-margin
ssh divix01 bash -s <<'REMOTE'
set -u
cd /data/models/slang/nvfp4-work/wt-ram-prefetch-margin && git fetch -q origin && git checkout -q --detach origin/codex/dsv41-ram-prefetch-margin && git log -1 --oneline
env PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 taskset -c 0-63 /data/models/slang/.venv/bin/python -m pytest -q -p no:randomly test/registered/unit/kernels/test_exl3_ram_prefetch_gpu_host.py test/registered/unit/kernels/test_exl3_ram_miss_device_args.py -k "gpu or candidate" 2>&1 | tail -12; echo "EXIT=${PIPESTATUS[0]}"
REMOTE
```
Expected: `EXIT=2` (collection error: `ImportError: cannot import name 'candidate_offset' from 'sglang.kernels.ops.moe.expert_stream_transport'`).

- [ ] **Step 3: The shared layout header**

Create `python/sglang/kernels/jit/csrc/moe/expert_stream/spec_candidates.h`:

```cpp
// The GPU scorer's candidate page (spec docs/superpowers/specs/2026-10-09-dsv41-ram-prefetch-gpu-scorer-design.md):
// pinned host memory the select kernel writes (spec_score.cuh) and the speculative threads read (host/ram_tier.h).
//
//   slot (seq - 1) % kCandRecords, kCandStride bytes: u32 seq (the seqlock word, stored last), u16 count, u16 flags,
//   then kMaxCandidates entries of u16 expert, u8 rank, a pad byte, f32 margin
//
// python/sglang/kernels/ops/moe/expert_stream_transport.py mirrors it as CAND_*; test_exl3_ram_miss_device_args checks.
#pragma once

#include "lease_layout.h"
#include <cstdint>

namespace sglang::expert_stream::wire {

struct SpecCandidates {
  static constexpr int kMaxCandidates = 8;
  static constexpr uint32_t kCandRecords = 16;
  static constexpr int64_t kCandSeq = 0;      // u32: 0 while the slot is rewritten, the record's seq stored last
  static constexpr int64_t kCandCount = 4;    // u16
  static constexpr int64_t kCandFlags = 6;    // u16: kCandFlag*
  static constexpr int64_t kCandEntries = 8;  // kMaxCandidates entries, best margin first
  static constexpr int64_t kCandEntryBytes = 8;
  static constexpr int64_t kCandExpert = 0;  // u16
  static constexpr int64_t kCandRank = 2;    // u8: the pick's position in the token that gave its best margin
  static constexpr int64_t kCandMargin = 4;  // f32: that token's score for it less its top_k-th
  static constexpr int64_t kCandPayloadBytes = kCandEntries + kMaxCandidates * kCandEntryBytes;
  static constexpr int64_t kCandStride = 128;
  static constexpr int64_t kCandPageBytes = kCandRecords * kCandStride;
  static constexpr uint32_t kCandFlagOversize = 1;  // the record's tokens are outside 1..tokens_max: count 0

  // Host only: the device computes the same expression inline.
  static constexpr int64_t slot_offset(uint32_t seq) {
    return static_cast<int64_t>((seq - 1u) % kCandRecords) * kCandStride;
  }
};

static_assert(SpecCandidates::kCandRecords == Wire::kDemandRecords, "one slot per demand record of the ring");
static_assert(SpecCandidates::kCandPayloadBytes == 72 && SpecCandidates::kCandPayloadBytes <= SpecCandidates::kCandStride,
              "a slot's payload fits its stride");

}  // namespace sglang::expert_stream::wire
```

- [ ] **Step 4: The counter, the config and the wait constants**

In `host/tier_protocol.h`, in `enum Counter`, directly after `kSpecDelayed,`:

```cpp
  kSpecLate,  // GPU scorer: a record's candidate slot not ready within kCandWaitNs
```

and in `is_core_counter`, directly after `case kSpecDelayed:`:

```cpp
    case kSpecLate:
```

In `host/ram_prefetch.h`, in `struct RamPrefetchConfig`, directly after `std::vector<std::vector<int>> cores;`:

```cpp
  // The GPU scorer's candidate page (../spec_candidates.h), pinned memory the caller keeps alive; null: the CPU
  // scorer scores each record's staged input.
  const uint8_t* candidates = nullptr;
```

and directly after the struct:

```cpp
// The GPU scorer's wait for a record's candidate slot: spin, then sleep in steps (the speculative thread may share its
// core with the polling RAM thread), up to kCandWaitNs; a slot not ready by then counts spec_late.
constexpr int64_t kCandWaitNs = 200'000;
constexpr int64_t kCandSpinNs = 20'000;
constexpr int64_t kCandSleepNs = 10'000;
```

Replace `SpecJob`'s comment with:

```cpp
// One record handed from a group's service thread to its speculative thread: a record of `row`; with the CPU scorer,
// one that staged `tokens` live inputs.
```

- [ ] **Step 5: The host's GPU mode in `ram_tier.h`**

Add `#include "../spec_candidates.h"` directly after `#include "../row_layout.h"`.

Replace `RamTier::enable_ram_prefetch` (its comment and body, ~1059-1113) with:

```cpp
  // Enables the RAM prefetch over the reserved pool: per group a speculative thread (RamThread starts it with the
  // service threads) fed by the group's service. The CPU scorer gets every record that staged a CPU input and scores
  // it; the GPU scorer (config.candidates) gets every record of a row with a target and reads its candidate slot. Needs
  // CPU experts on every group. On the owner, before the service thread starts.
  void enable_ram_prefetch(RamPrefetchConfig config) {
    std::lock_guard<std::mutex> caller(caller_mutex_);
    require_owner("enable_ram_prefetch");
    const std::string prefix = error_prefix<Layout>() + "RAM prefetch: ";
    const bool gpu = config.candidates != nullptr;
    if (threaded_.load()) throw std::runtime_error(prefix + "enable it before the service thread starts");
    if (spec_ != nullptr) throw std::runtime_error(prefix + "it is already enabled");
    if (pool_ == nullptr) throw std::runtime_error(prefix + "reserve_spec_pool first");
    if (static_cast<int64_t>(config.target.size()) != layers_ || static_cast<int64_t>(config.gate.size()) != layers_)
      throw std::runtime_error(prefix + "the target table has one entry per streamed row");
    for (int64_t row = 0; row < layers_; ++row) {
      const int32_t target = config.target[row], gate = config.gate[row];
      if ((target < 0) != (gate < 0) || target >= layers_ || target == row || (!gpu && gate >= config.gate_count))
        throw std::runtime_error(
            prefix + "row " + std::to_string(row) + " names target " + std::to_string(target) + " with gate " +
            std::to_string(gate));
    }
    try {
      check_gate_choice(experts_, config.top_k, config.per_token, config.per_layer);
    } catch (const std::invalid_argument& error) {
      throw std::runtime_error(prefix + error.what());
    }
    if (gpu && config.gate_count != 0)
      throw std::runtime_error(prefix + "the GPU scorer reads the device's gates: pass none");
    if (!gpu && (config.gates == nullptr || config.bias == nullptr || config.gate_count < 1))
      throw std::runtime_error(prefix + "it needs at least one gate");
    if (static_cast<int>(config.cores.size()) != groups())
      throw std::runtime_error(prefix + "one core list per NUMA group");
    for (int g = 0; g < groups(); ++g)
      if (dist_.group(g).cpu == nullptr)
        throw std::runtime_error(
            prefix + "it needs CPU experts on group " + std::to_string(g) + ", whose records stage the scorer's input");
    const CpuExpertConfig& cpu = dist_.group(0).cpu->config();
    if (!gpu && config.hidden != cpu.hidden)
      throw std::runtime_error(
          prefix + "the gate's hidden size " + std::to_string(config.hidden) + " is not the CPU rows' " +
          std::to_string(cpu.hidden));
    auto spec = std::make_unique<SpecState>();
    spec->x_base = cpu.x_base;
    spec->x_stride = cpu.x_stride;
    spec->x_token_bytes = cpu.x_token_bytes;
    spec->tokens_max = cpu.tokens;
    for (int g = 0; g < groups(); ++g) {
      auto group = std::make_unique<SpecGroup>(std::string(Layout::kName) + "-spec" + std::to_string(g));
      if (!gpu) group->scorer.reserve(spec->tokens_max, config.hidden, experts_);
      group->skip.assign(static_cast<size_t>(experts_), 0);
      group->packed.reserve(1);
      group->cores = config.cores[g];
      group->row_seq = std::make_unique<std::atomic<uint32_t>[]>(static_cast<size_t>(layers_));
      spec->groups.push_back(std::move(group));
    }
    spec->config = std::move(config);
    spec_ = std::move(spec);
  }
```

Replace `offer_spec` (its comment and body) with:

```cpp
  // Service thread, after any whole record: hands it to the group's speculative thread when its row has a target and,
  // with the CPU scorer, it staged a CPU input (the GPU scorer scores every record). A full ring drops it: the service
  // never waits.
  void offer_spec(Group& group, const Request& request) {
    if (request.row < 0 || request.row >= layers_) return;
    SpecGroup& spec = *spec_->groups[group.index];
    if (spec_->config.target[request.row] < 0) return;
    int64_t tokens = 1;
    if (spec_->config.candidates == nullptr) {
      bool staged = false;
      for (const Lane& lane : request.lanes)
        staged |= lane.kind == Wire::kKindHitCpu || lane.kind == Wire::kKindMissCpu;
      if (!staged) return;
      if (spec_->tokens_max > 1) {
        uint32_t live;
        std::memcpy(&live, spec_->x_base + request.row * spec_->x_stride + spec_->tokens_max * spec_->x_token_bytes, 4);
        tokens = live;
        if (tokens < 1 || tokens > spec_->tokens_max) {
          count<kSpecDropped>(group);
          return;
        }
      }
    }
    if (!spec.ring.push(SpecJob{request.seq, request.row, tokens})) {
      count<kSpecDropped>(group);
      return;
    }
    spec.bell.ring();
  }
```

In `serve_spec_job`, directly after the block

```cpp
    if (spec_stale(spec, job.row, target, job.seq)) {
      spec_count<kSpecDropped>(spec);
      return;
    }
```

insert:

```cpp
    if (config.candidates != nullptr) {
      serve_gpu_job(g, job, target, threaded);
      return;
    }
```

Directly after `serve_spec_job`'s closing brace (before `spec_read`'s comment), insert:

```cpp
  // A candidate slot as read_candidates finds it: ready, not yet (an older record's, or open: seq word 0), or lapped (a
  // later record's, also when it changed across the copy).
  enum class CandRead { kReady, kNotYet, kLapped };
  struct CandSlot {
    int count = 0;
    int32_t expert[SpecCandidates::kMaxCandidates];
    int32_t rank[SpecCandidates::kMaxCandidates];
    float margin[SpecCandidates::kMaxCandidates];
  };

  // Copies `seq`'s slot as read_gpu_hot copies a hot record: acquire the seq, copy, acquire fence, re-check. A count
  // above kMaxCandidates is clamped (the kernel writes none).
  CandRead read_candidates(uint32_t seq, CandSlot* out) const {
    using Cand = SpecCandidates;
    const uint8_t* slot = spec_->config.candidates + Cand::slot_offset(seq);
    const uint32_t first = load_acquire(slot + Cand::kCandSeq);
    if (first != seq) return first != 0 && reached(first, skip_zero(seq + 1u)) ? CandRead::kLapped : CandRead::kNotYet;
    uint8_t bytes[Cand::kCandPayloadBytes];
    std::memcpy(bytes, slot, sizeof(bytes));
    std::atomic_thread_fence(std::memory_order_acquire);
    asm volatile("" ::: "memory");  // the copy's plain loads must stay before the seq re-check
    if (load_acquire(slot + Cand::kCandSeq) != seq) return CandRead::kLapped;
    uint16_t count;
    std::memcpy(&count, bytes + Cand::kCandCount, 2);
    out->count = std::min<int>(count, Cand::kMaxCandidates);
    for (int i = 0; i < out->count; ++i) {
      const uint8_t* entry = bytes + Cand::kCandEntries + i * Cand::kCandEntryBytes;
      uint16_t expert;
      std::memcpy(&expert, entry + Cand::kCandExpert, 2);
      out->expert[i] = expert;
      out->rank[i] = entry[Cand::kCandRank];
      std::memcpy(&out->margin[i], entry + Cand::kCandMargin, 4);
    }
    return CandRead::kReady;
  }

  // Polls `seq`'s slot until it is ready or lapped: _mm_pause for kCandSpinNs, then kCandSleepNs sleeps, up to
  // kCandWaitNs. kNotYet when it never became ready.
  CandRead await_candidates(uint32_t seq, CandSlot* out) const {
    const int64_t start = now_ns();
    for (;;) {
      const CandRead read = read_candidates(seq, out);
      if (read != CandRead::kNotYet) return read;
      const int64_t waited = now_ns() - start;
      if (waited >= kCandWaitNs) return CandRead::kNotYet;
      if (waited < kCandSpinNs)
        _mm_pause();
      else
        std::this_thread::sleep_for(std::chrono::nanoseconds(kCandSleepNs));
    }
  }

  // serve_spec_job with the GPU scorer: waits for the record's slot, then reads, in the GPU's order, the first
  // per_layer candidates still unmapped here and not pooled before, its own group's only. Both groups read the same slot
  // and pass the same filters, so the layer's budget holds over both. The GPU skipped the hot ones.
  void serve_gpu_job(int g, const SpecJob& job, int64_t target, bool threaded) {
    SpecGroup& spec = *spec_->groups[g];
    int64_t start = 0;
    if constexpr (Build::kMetrics) start = now_ns();
    CandSlot slot;
    const CandRead read = await_candidates(job.seq, &slot);
    if constexpr (Build::kMetrics) {
      stats_.add(kSpecScored);
      stats_.add(kSpecScoreNs, now_ns() - start);
    }
    if (read == CandRead::kNotYet) {
      spec_count<kSpecLate>(spec);
      return;
    }
    if (read == CandRead::kLapped || slot.count == 0) {
      spec_count<kSpecDropped>(spec);
      return;
    }
    int taken = 0;
    for (int i = 0; i < slot.count && taken < spec_->config.per_layer; ++i) {
      const int32_t expert = slot.expert[i];
      if (expert >= experts_ || __atomic_load_n(map_ + target * experts_ + expert, __ATOMIC_ACQUIRE) >= 0 ||
          pool_->pooled_before(target, expert, job.seq))
        continue;
      ++taken;
      if (Wire::home(expert) != g) continue;
      if (threaded && !spec_wait_unheld(spec)) return;  // a quiescer waits for one row, not the whole job
      uint64_t pick = 0;  // spec_submit's gen: rank << 32 | the margin's fp32 bits
      if constexpr (Build::kMetrics)
        pick = (static_cast<uint64_t>(slot.rank[i]) << 32) | std::bit_cast<uint32_t>(slot.margin[i]);
      spec_read(g, job.row, target, job.seq, expert, pick);
    }
  }
```

- [ ] **Step 6: The FFI export**

In `host/ffi_exports.h`, replace `enable_ram_prefetch` (its comment and body, ~738-801) with:

```cpp
  // Enables the RAM prefetch (RamTier::enable_ram_prefetch). `targets` int64 [rows, 2]: per source row its target row
  // and that row's gate index, or -1 -1. CPU scorer: `gates` uint8 [n, experts * hidden * 2] (bf16 [experts, hidden]
  // each) and `bias` float32 [n, experts], host memory the caller keeps alive, and `candidates` empty. GPU scorer:
  // `candidates` uint8 [SpecCandidates::kCandPageBytes], pinned memory the caller keeps alive, `gates` and `bias`
  // [0, 0], `hidden` unused. `cores` int64 [groups, width]: each group's speculative thread's cores, -1 padding (none:
  // the caller's affinity). Cores 64-71 are refused.
  static void enable_ram_prefetch(
      int64_t handle,
      TensorView targets,
      TensorView gates,
      TensorView bias,
      TensorView cores,
      int64_t hidden,
      int64_t top_k,
      int64_t per_token,
      int64_t per_layer,
      int64_t top_k_only,
      TensorView candidates) {
    using namespace host;
    using Cand = expert_stream::wire::SpecCandidates;
    auto cpu = SymbolicDevice{};
    auto host_mem = SymbolicDevice{};
    auto host_bias = SymbolicDevice{};
    auto host_page = SymbolicDevice{};
    auto rows = SymbolicSize{"rows"};
    auto n = SymbolicSize{"gates"};
    expert_stream::verify_named(
        "targets", TensorMatcher({rows, 2}).with_dtype<int64_t>().with_device<kDLCPU>(cpu), targets);
    expert_stream::verify_named(
        "gates", TensorMatcher({n, -1}).with_dtype<uint8_t>().with_device<kDLCPU, kDLCUDAHost>(host_mem), gates);
    expert_stream::verify_named(
        "bias", TensorMatcher({n, -1}).with_dtype<float>().with_device<kDLCPU, kDLCUDAHost>(host_bias), bias);
    expert_stream::verify_named(
        "cores", TensorMatcher({Wire::kNodes, -1}).with_dtype<int64_t>().with_device<kDLCPU>(cpu), cores);
    expert_stream::verify_named(
        "candidates",
        TensorMatcher({-1}).with_dtype<uint8_t>().with_device<kDLCPU, kDLCUDAHost>(host_page),
        candidates);
    const auto tier = find(handle);
    const bool gpu = candidates.size(0) != 0;
    if (gpu) {
      if (candidates.size(0) != Cand::kCandPageBytes || gates.size(0) != 0)
        throw std::runtime_error(
            error_prefix<Layout>() + "RAM prefetch: the GPU scorer takes a candidate page of " +
            std::to_string(Cand::kCandPageBytes) + " bytes and no gates");
    } else if (hidden < 1 || bias.size(1) != tier->experts() || gates.size(1) != tier->experts() * hidden * 2) {
      throw std::runtime_error(
          error_prefix<Layout>() +
          "RAM prefetch: the gates are not bf16 [n, experts, hidden] with an fp32 bias per expert");
    }
    expert_stream::RamPrefetchConfig config;
    const auto* t = static_cast<const int64_t*>(targets.data_ptr());
    for (int64_t row = 0; row < targets.size(0); ++row) {
      config.target.push_back(static_cast<int32_t>(t[2 * row]));
      config.gate.push_back(static_cast<int32_t>(t[2 * row + 1]));
    }
    config.gates = gpu ? nullptr : static_cast<const uint16_t*>(gates.data_ptr());
    config.bias = gpu ? nullptr : static_cast<const float*>(bias.data_ptr());
    config.gate_count = gates.size(0);
    config.hidden = hidden;
    config.top_k = static_cast<int>(top_k);
    config.per_token = static_cast<int>(per_token);
    config.per_layer = static_cast<int>(per_layer);
    config.top_k_only = top_k_only != 0;
    config.candidates = gpu ? static_cast<const uint8_t*>(candidates.data_ptr()) : nullptr;
    const auto* c = static_cast<const int64_t*>(cores.data_ptr());
    for (int g = 0; g < Wire::kNodes; ++g) {
      std::vector<int> own;
      for (int64_t j = 0; j < cores.size(1); ++j) {
        const int64_t core = c[g * cores.size(1) + j];
        if (core < 0) continue;
        if (core >= CPU_SETSIZE || (core >= 64 && core <= 71))
          throw std::runtime_error(
              error_prefix<Layout>() + "RAM prefetch: core " + std::to_string(core) +
              " is out of range or reserved (64-71 take NVMe completion interrupts)");
        own.push_back(static_cast<int>(core));
      }
      config.cores.push_back(std::move(own));
    }
    tier->enable_ram_prefetch(std::move(config));
  }
```

- [ ] **Step 7: The Python side**

In `python/sglang/kernels/ops/moe/expert_stream_transport.py`:

`struct` is already imported; add `NamedTuple` to the `from typing import ...` line.

In `COUNTERS` and in `CORE_COUNTERS`, directly after `"spec_delayed",` add `"spec_late",`.

Directly after `new_hot_page`, add:

```python
# The GPU scorer's candidate page (csrc/moe/expert_stream/spec_candidates.h): CAND_RECORDS slots of CAND_STRIDE bytes,
# slot (seq - 1) % CAND_RECORDS; u32 seq, u16 count, u16 flags, then CAND_MAX entries (u16 expert, u8 rank, pad, f32).
CAND_MAX = 8
CAND_RECORDS = 16
CAND_STRIDE = 128
CAND_ENTRIES = 8
CAND_ENTRY_BYTES = 8
CAND_PAGE_BYTES = CAND_RECORDS * CAND_STRIDE
CAND_FLAG_OVERSIZE = 1


def new_candidate_page(*, pin: bool) -> torch.Tensor:
    """A zeroed candidate page; pinned (the select kernel writes it through UVA) for a real device."""
    return torch.zeros(CAND_PAGE_BYTES, dtype=torch.uint8, pin_memory=pin)


def candidate_offset(seq: int) -> int:
    """The byte offset of record ``seq``'s slot."""
    return ((int(seq) - 1) % CAND_RECORDS) * CAND_STRIDE


class CandidateRead(NamedTuple):
    status: str  # "ready", "not_yet" (an older record's, or open) or "lapped" (a later record's): ram_tier.h CandRead
    count: int = 0
    flags: int = 0
    picks: tuple = ()  # (expert, rank, margin) per candidate, best margin first


def read_candidates(page: torch.Tensor, seq: int) -> CandidateRead:
    """Record ``seq``'s slot as the host reads it, without the torn-copy check: read it while no kernel writes."""
    off = candidate_offset(seq)
    raw = bytes(page[off : off + CAND_ENTRIES + CAND_MAX * CAND_ENTRY_BYTES].tolist())
    word, count, flags = struct.unpack_from("<IHH", raw)
    seq &= 0xFFFFFFFF
    if word != seq:
        following = (seq + 1) & 0xFFFFFFFF or 1
        lapped = word != 0 and ((word - following) & 0xFFFFFFFF) < 0x80000000
        return CandidateRead("lapped" if lapped else "not_yet")
    picks = tuple(
        struct.unpack_from("<HBxf", raw, CAND_ENTRIES + i * CAND_ENTRY_BYTES) for i in range(min(count, CAND_MAX))
    )
    return CandidateRead("ready", count, flags, picks)
```

Replace `ExpertStreamHost.enable_ram_prefetch` with:

```python
    def enable_ram_prefetch(
        self,
        targets: torch.Tensor,
        gates: Optional[torch.Tensor],
        bias: Optional[torch.Tensor],
        *,
        top_k: int,
        per_token: int,
        per_layer: int,
        cores: Sequence[Sequence[int]],
        top_k_only: bool = False,
        candidates: Optional[torch.Tensor] = None,
    ) -> None:
        """Enable the RAM prefetch over the pool ``reserve_spec_pool`` took, after ``enable_cpu_experts`` and before
        the thread. ``targets`` int64 ``[layers, 2]``: per source row its target row and gate index, or (-1, -1);
        ``cores`` is each NUMA group's speculative-thread core list (empty: the caller's affinity). CPU scorer:
        ``gates`` bf16 ``[n, experts, hidden]`` and ``bias`` fp32 ``[n, experts]``, host tensors this host keeps alive.
        GPU scorer: ``candidates`` (``new_candidate_page``), which the select kernel writes, and no gates."""
        from sglang.srt.layers.moe.cpu_experts.threading_config import check_not_reserved

        if candidates is not None:
            if gates is not None or bias is not None:
                raise ValueError("the GPU scorer reads the device's gates: pass gates=None and bias=None")
            if (
                candidates.dtype != torch.uint8
                or candidates.device.type != "cpu"
                or not candidates.is_contiguous()
                or candidates.numel() != CAND_PAGE_BYTES
            ):
                raise ValueError(
                    f"candidates must be a contiguous host uint8 tensor of {CAND_PAGE_BYTES} bytes (new_candidate_page)"
                )
            gate_bytes = torch.empty((0, 0), dtype=torch.uint8)
            bias = torch.empty((0, 0), dtype=torch.float32)
            hidden = 0
        else:
            if (
                gates is None
                or gates.dtype != torch.bfloat16
                or gates.dim() != 3
                or gates.device.type != "cpu"
                or not gates.is_contiguous()
            ):
                raise ValueError("gates must be a contiguous host bf16 [n, experts, hidden] tensor")
            if (
                bias is None
                or bias.dtype != torch.float32
                or tuple(bias.shape) != tuple(gates.shape[:2])
                or bias.device.type != "cpu"
            ):
                raise ValueError("bias must be a host fp32 [n, experts] tensor")
            gate_bytes = gates.view(gates.shape[0], -1).view(torch.uint8)
            bias = bias.contiguous()
            hidden = int(gates.shape[2])
        if len(cores) != self.nodes:
            raise ValueError(f"one core list per NUMA group ({self.nodes}), got {len(cores)}")
        table = torch.full((self.nodes, max(1, max(len(own) for own in cores))), -1, dtype=torch.int64)
        for g, own in enumerate(cores):
            for j, core in enumerate(own):
                check_not_reserved(int(core))
                table[g, j] = int(core)
        self._module.expert_stream_enable_ram_prefetch(
            self.handle,
            targets.to(torch.int64).contiguous(),
            gate_bytes,
            bias,
            table,
            hidden,
            int(top_k),
            int(per_token),
            int(per_layer),
            int(bool(top_k_only)),
            candidates if candidates is not None else torch.empty(0, dtype=torch.uint8),
        )
        self.ram_prefetch_tensors = (gates, bias, candidates)
```

- [ ] **Step 8: Commit, push, and run the selection on divix01**

```bash
cd /Users/dnikolaidis/.codex/worktrees/ram-prefetch-margin
git add python/sglang/kernels/jit/csrc/moe/expert_stream/spec_candidates.h python/sglang/kernels/jit/csrc/moe/expert_stream/host/tier_protocol.h python/sglang/kernels/jit/csrc/moe/expert_stream/host/ram_prefetch.h python/sglang/kernels/jit/csrc/moe/expert_stream/host/ram_tier.h python/sglang/kernels/jit/csrc/moe/expert_stream/host/ffi_exports.h python/sglang/kernels/ops/moe/expert_stream_transport.py
git commit -m "Read the GPU scorer's candidate slots on the host: wait, filter, budget, spec_late

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01LxB2wQZv5EVNuBGUJJbkGk"
git push origin codex/dsv41-ram-prefetch-margin
ssh divix01 bash -s <<'REMOTE'
set -u
cd /data/models/slang/nvfp4-work/wt-ram-prefetch-margin && git fetch -q origin && git checkout -q --detach origin/codex/dsv41-ram-prefetch-margin && git log -1 --oneline
env PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 taskset -c 0-63 /data/models/slang/.venv/bin/python -m pytest -q -p no:randomly test/registered/unit/kernels/test_exl3_ram_prefetch_gpu_host.py test/registered/unit/kernels/test_exl3_ram_miss_device_args.py test/registered/unit/kernels/test_exl3_ram_prefetch_step.py test/registered/unit/kernels/test_exl3_ram_prefetch_thread.py test/registered/unit/kernels/test_exl3_ram_prefetch_events.py test/registered/unit/kernels/test_exl3_ram_prefetch_pool.py test/registered/unit/kernels/test_exl3_ram_prefetch_swap.py test/registered/unit/kernels/test_exl3_ram_prefetch_scorer.py test/registered/unit/kernels/test_expert_stream_build_variants.py test/registered/unit/kernels/test_expert_stream_hotpath_golden.py 2>&1 | tail -4; echo "EXIT=${PIPESTATUS[0]}"
REMOTE
```
Expected: `EXIT=0` (the first run rebuilds the host modules; a `TimeoutExpired` there is the compiler, re-run once warm). Record the pass count with this selection.

- [ ] **Step 9: Mutants (divix01 worktree only, reverted after each)**

In `/data/models/slang/nvfp4-work/wt-ram-prefetch-margin`, apply each, run `test_exl3_ram_prefetch_gpu_host.py` with the Step 8 command (that file alone), confirm red, then `git checkout -- python/sglang/kernels/jit/csrc/moe/expert_stream/host/ram_tier.h python/sglang/kernels/jit/csrc/moe/expert_stream/host/ram_prefetch.h`:
1. `ram_tier.h` `serve_gpu_job`: delete `|| pool_->pooled_before(target, expert, job.seq)` -- red in `test_a_ready_slot_is_read_in_the_gpus_order_past_mapped_and_pooled_experts`.
2. `ram_prefetch.h` `pooled_before`: replace `!= seq) return true;` with `!= seq || true) return true;` -- red in `test_a_group_that_reads_first_does_not_shift_the_other_groups_list`.
3. `ram_tier.h` `read_candidates`: replace `? CandRead::kLapped : CandRead::kNotYet` with `? CandRead::kNotYet : CandRead::kNotYet` -- red in `test_a_lapped_slot_is_dropped`.
4. `ram_tier.h` `offer_spec`: replace `if (spec_->config.candidates == nullptr) {` with `if (true) {` -- red in `test_every_record_of_a_row_with_a_target_feeds_the_ring_and_no_other_does`.

After the last revert, re-run the Step 8 selection; expected `EXIT=0`. Record the four red names and the green re-run.

---

### Task 3: The scoring kernels

**Files:**
- Create: `python/sglang/kernels/jit/csrc/moe/expert_stream/spec_score.cuh`
- Modify: `python/sglang/kernels/jit/csrc/moe/exl3/exl3_ram_miss.cuh` (the includes)
- Modify: `python/sglang/kernels/ops/moe/expert_stream_transport.py` (`_SPEC_METHODS`, `_device_wrappers`, `SPEC_DEPTH`, `SPEC_SELECT_MAX_EXPERTS`, `run_spec_score`, `run_spec_select`)
- Modify: `python/sglang/test/dsv41_ram_prefetch_fixtures.py` (receives `gate_reference`, `exact_gate_case`, `lead_gate_case`)
- Modify: `test/registered/unit/kernels/test_exl3_ram_prefetch_scorer.py` (imports them instead of defining them)
- Test: `test/manual/dsv41/test_exl3_ram_prefetch_gpu_scorer_cuda.py` (new), `test/registered/unit/kernels/test_exl3_ram_miss_device_args.py`

**Interfaces:**
- Consumes: Task 2's `SpecCandidates` (C++), `CAND_*`, `new_candidate_page`, `candidate_offset`, `read_candidates`, `CandidateRead`.
- Produces:
  - Device kernels `exl3_ram_prefetch_score_kernel<W>` (`W` = `bf16_t` or `float`) and `exl3_ram_prefetch_select_kernel`; launchers `SpecScoreKernel::score(x, w, bias, scores)` and `SpecScoreKernel::select(scores, tokens, top_k, per_token, top_k_only, hot_slots, hot_capacity, ram_slot, target, state, candidates_address)`, exported as `expert_stream_spec_score` / `expert_stream_spec_select`; test hook `EXL3_RAM_MISS_TEST_SPEC_NO_SEQ`.
  - Python `SPEC_DEPTH = 12`, `SPEC_SELECT_MAX_EXPERTS = 1024`; `run_spec_score(x, w, bias, scores, *, module=None) -> None`; `run_spec_select(scores, tokens, *, top_k, per_token, top_k_only, hot_slots, hot_capacity, ram_slot, target, state, candidates, module=None) -> None`.
  - Fixtures `gate_reference(x, w, bias, skip, *, top_k, per_token, per_layer, top_k_only=False, meta=False)`, `exact_gate_case(seed, tokens=3, experts=16, hidden=64)`, `lead_gate_case(seed, tokens=3, experts=16, hidden=16)`, `GATE_DEPTH = 12`.

- [ ] **Step 1: Move the reference into the fixtures**

Append to `python/sglang/test/dsv41_ram_prefetch_fixtures.py` (moved verbatim from `test_exl3_ram_prefetch_scorer.py`'s `reference`, `_exact_case` and `_lead_case`, renamed):

```python
GATE_DEPTH = 12  # GateScorer::kDepth


def gate_reference(x, w, bias, skip, *, top_k, per_token, per_layer, top_k_only=False, meta=False):
    """A torch reference of the Phase 0 replay's ranking, which GateScorer::choose and the GPU select kernel both
    implement: sqrt(softplus(W x)) + b per live token, each token's top 12 walked in score order past the skipped
    experts for per_token picks, each pick's margin its score less the token's top_k-th, the union by margin (ties by
    id), the first per_layer."""
    logits = x.float() @ w.float().T
    scores = torch.where(logits > 20, logits, torch.log1p(torch.exp(logits))).sqrt() + bias
    scores = torch.where(scores.isnan(), torch.tensor(float("-inf")), scores)  # a NaN score ranks below every other
    best, ranks = {}, {}
    experts = w.shape[0]
    for m in range(x.shape[0]):
        s = scores[m]
        order = sorted(range(experts), key=lambda e: (-float(s[e]), e))[: min(GATE_DEPTH, experts)]
        walk = order[:top_k] if top_k_only else order
        kth = s[order[top_k - 1]]
        picked = 0
        for rank, e in enumerate(walk):
            if picked >= per_token:
                break
            if skip[e]:
                continue
            # an fp32 subtraction, as the host's; equal scores (also two infinities) give 0, -inf stays above unpicked
            margin = 0.0 if s[e] == kth else max(float(s[e] - kth), -3.4028234663852886e38)
            # an expert's rank is its position in the token that gave its best margin, the lowest on a tie
            if e not in best or margin > best[e] or (margin == best[e] and rank < ranks[e]):
                ranks[e] = rank
            best[e] = max(best.get(e, float("-inf")), margin)
            picked += 1
    chosen = [e for e, _ in sorted(best.items(), key=lambda kv: (-kv[1], kv[0]))][:per_layer]
    return [(e, ranks[e], best[e]) for e in chosen] if meta else chosen


def exact_gate_case(seed, tokens=3, experts=16, hidden=64):
    """Small non-negative integers with a constant 21 term: every logit is an integer above 20, exact in fp32 in any
    summation order, so softplus is the identity and host, GPU and torch produce the same fp32 scores bit for bit."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(0, 4, (tokens, hidden), generator=g).to(torch.float16)
    x[:, 0] = 1
    w = torch.randint(0, 4, (experts, hidden), generator=g).to(torch.bfloat16)
    w[:, 0] = 21
    bias = torch.randperm(experts, generator=g).float() * 1e-3
    return x, w, bias


def lead_gate_case(seed, tokens=3, experts=16, hidden=16):
    """exact_gate_case with a lead term of 128, so x may go negative while every logit stays an exact integer above
    20."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(-2, 4, (tokens, hidden), generator=g).to(torch.float16)
    x[:, 0] = 1
    w = torch.randint(0, 4, (experts, hidden), generator=g).to(torch.bfloat16)
    w[:, 0] = 128
    bias = torch.randperm(experts, generator=g).float() * 1e-3
    return x, w, bias
```

In `test/registered/unit/kernels/test_exl3_ram_prefetch_scorer.py`: delete the definitions of `reference`, `_exact_case` and `_lead_case` (the three functions and their docstrings; keep `DEPTH = 12`, which `_separated` reads), and add to the imports:

```python
from sglang.test.dsv41_ram_prefetch_fixtures import exact_gate_case as _exact_case
from sglang.test.dsv41_ram_prefetch_fixtures import gate_reference as reference
from sglang.test.dsv41_ram_prefetch_fixtures import lead_gate_case as _lead_case
```

Every test body stays as it is.

- [ ] **Step 2: Write the failing GPU tests**

Create `test/manual/dsv41/test_exl3_ram_prefetch_gpu_scorer_cuda.py`:

```python
"""The GPU scorer's kernels (spec 2026-10-09-dsv41-ram-prefetch-gpu-scorer-design, "The scoring kernels") on a real
GPU: their candidates against the host scorer's torch reference (gate_reference), the oversize and lapped slots, and the
seq word stored last. Task 4 adds a captured post with its scoring replayed through the production row backend.

Run on divix01 under cc-gpu.lock, with PYTHONPATH pointing at the tree under test.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

from sglang.kernels.ops.moe import expert_stream_transport as es  # noqa: E402
from sglang.test.dsv41_ram_prefetch_fixtures import exact_gate_case, gate_reference, lead_gate_case  # noqa: E402

KEEP = es.CAND_MAX  # a reference per_layer that lists every candidate the GPU writes
BF16 = torch.bfloat16


def _candidates(x, w, bias, *, top_k, per_token, top_k_only=False, hot=(), mapped=(), seq=1, tokens_max=None,
                page=None, scores=None, module=None):
    """Scores `x` (bf16 [tokens, hidden]) against (w, bias) for target row 1 of a two-row map, with `hot` VRAM-resident
    and `mapped` mapped in RAM there, and returns record `seq`'s slot."""
    experts, tokens = w.shape[0], x.shape[0]
    tokens_max = tokens_max or max(1, tokens)
    if scores is None:
        scores = torch.empty((tokens_max, experts), dtype=torch.float32, device="cuda")
    page = es.new_candidate_page(pin=True) if page is None else page
    state = torch.zeros(len(es.STATE_WORDS), dtype=torch.int32, device="cuda")
    state[es.STATE_WORDS["posted"]] = seq
    hot_slots = torch.full((max(1, len(hot)),), -1, dtype=torch.int64, device="cuda")
    if hot:
        hot_slots[: len(hot)] = torch.tensor(list(hot), dtype=torch.int64)
    ram_slot = torch.full((2, experts), -1, dtype=torch.int32, device="cuda")
    for slot, e in enumerate(mapped):
        ram_slot[1, e] = slot
    if 1 <= tokens <= tokens_max:
        es.run_spec_score(x.cuda(), w.cuda(), bias.cuda(), scores, module=module)
    es.run_spec_select(
        scores, tokens, top_k=top_k, per_token=per_token, top_k_only=top_k_only, hot_slots=hot_slots,
        hot_capacity=len(hot), ram_slot=ram_slot, target=1, state=state, candidates=page, module=module,
    )
    torch.cuda.synchronize()
    return es.read_candidates(page, seq)


def _skip(experts, hot=(), mapped=()):
    return [e in hot or e in mapped for e in range(experts)]


COMBOS = [(6, 1, 1), (6, 1, 3), (6, 2, 4), (6, 3, 8), (2, 1, 2), (1, 2, 5)]


@pytest.mark.parametrize("w_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("top_k_only", [False, True])
@pytest.mark.parametrize("top_k, per_token, per_layer", COMBOS)
@pytest.mark.parametrize("seed", range(8))
def test_the_gpu_ranks_exactly_as_the_host_reference(seed, top_k, per_token, per_layer, top_k_only, w_dtype):
    """Experts, ranks and margins bit for bit, hot and mapped experts skipped; the first per_layer are the CPU scorer's
    per_layer. Mutant: the margin to the (top_k - 1)-th score -- red."""
    x, w, bias = exact_gate_case(seed)
    order = torch.randperm(16, generator=torch.Generator().manual_seed(100 + seed)).tolist()
    hot, mapped = order[:2], order[2:4]
    got = _candidates(x.to(BF16), w.to(w_dtype), bias, top_k=top_k, per_token=per_token, top_k_only=top_k_only,
                      hot=hot, mapped=mapped)
    kw = dict(top_k=top_k, per_token=per_token, top_k_only=top_k_only, meta=True)
    skip = _skip(16, hot, mapped)
    assert got.status == "ready" and got.flags == 0
    assert list(got.picks) == gate_reference(x, w, bias, skip, per_layer=KEEP, **kw)
    assert list(got.picks[:per_layer]) == gate_reference(x, w, bias, skip, per_layer=per_layer, **kw)


@pytest.mark.parametrize("per_token", [1, 2, 12])
@pytest.mark.parametrize("seed", range(4))
def test_tied_scores_and_margins_go_to_the_lower_id(seed, per_token):
    """Experts e and e + 16 share a gate row and there is no bias. Mutants: either id tie-break dropped -- red."""
    x, w, _ = exact_gate_case(seed, tokens=6)
    w = torch.cat([w, w])
    bias = torch.zeros(32)
    got = _candidates(x.to(BF16), w, bias, top_k=6, per_token=per_token)
    assert list(got.picks) == gate_reference(x, w, bias, [False] * 32, top_k=6, per_token=per_token, per_layer=KEEP,
                                             meta=True)


@pytest.mark.parametrize("seed", range(4))
def test_negative_inputs_rank_as_the_reference(seed):
    x, w, bias = lead_gate_case(seed)
    assert (x < 0).any()
    got = _candidates(x.to(BF16), w, bias, top_k=6, per_token=2)
    assert list(got.picks) == gate_reference(x, w, bias, [False] * 16, top_k=6, per_token=2, per_layer=KEEP, meta=True)


def test_an_infinite_input_ranks_as_the_reference():
    """Review Focus 4. x[:, 2] = inf: an expert with w > 0 there scores +inf, one with w == 0 scores NaN (inf * 0) and
    ranks last."""
    x, w, bias = lead_gate_case(1)
    x[:, 2] = float("inf")
    w[:, 2] = torch.tensor([0, 1, 2, 3] * 4, dtype=torch.bfloat16)
    got = _candidates(x.to(BF16), w, bias, top_k=6, per_token=12)
    want = gate_reference(x, w, bias, [False] * 16, top_k=6, per_token=12, per_layer=KEEP, meta=True)
    assert list(got.picks) == want and all(e % 4 != 0 for e, _, _ in got.picks)


@pytest.mark.parametrize("seed", range(4))
def test_nan_scores_rank_last_in_id_order(seed):
    """Review Focus 4. A NaN bias on half the experts. Mutant: store the NaN score unmapped -- red."""
    x, w, bias = exact_gate_case(seed, hidden=16)
    bias[torch.randperm(16, generator=torch.Generator().manual_seed(200 + seed))[:8]] = float("nan")
    got = _candidates(x.to(BF16), w, bias, top_k=3, per_token=12)
    assert list(got.picks) == gate_reference(x, w, bias, [False] * 16, top_k=3, per_token=12, per_layer=KEEP,
                                             meta=True)


@pytest.mark.parametrize("w_dtype", [torch.bfloat16, torch.float32])
def test_on_random_data_the_top_candidate_agrees_with_torch_in_99_percent_of_records(w_dtype):
    """DSV4's shape (384 experts, hidden 7168), 1-6 tokens: the softplus branch and fp32 sums in another order may
    move a near-tie, so agreement, not identity."""
    g = torch.Generator().manual_seed(7)
    experts, hidden, trials = 384, 7168, 200
    w = (torch.randn((experts, hidden), generator=g) * 0.02).to(w_dtype)
    scores = torch.empty((6, experts), dtype=torch.float32, device="cuda")
    page, agree = es.new_candidate_page(pin=True), 0
    for trial in range(trials):
        x = torch.randn((1 + trial % 6, hidden), generator=g).to(BF16)
        bias = torch.randn(experts, generator=g) * 0.1
        got = _candidates(x, w, bias, top_k=6, per_token=1, seq=trial + 1, tokens_max=6, page=page, scores=scores)
        want = gate_reference(x, w, bias, [False] * experts, top_k=6, per_token=1, per_layer=1)
        agree += got.picks[0][0] == want[0]
    assert agree >= 0.99 * trials


@pytest.mark.parametrize("tokens", [0, 3])
def test_a_record_outside_one_to_tokens_max_tokens_publishes_count_zero(tokens):
    """Prefill (more tokens than the rows hold) or a post without input: count 0, the oversize flag, nothing scored."""
    x, w, bias = exact_gate_case(0, tokens=max(tokens, 1))
    got = _candidates(x[:tokens].to(BF16), w, bias, top_k=6, per_token=1, tokens_max=2)
    assert (got.status, got.count, got.flags, got.picks) == ("ready", 0, es.CAND_FLAG_OVERSIZE, ())


def test_a_record_with_fewer_tokens_ranks_only_its_own_rows_of_the_scratch():
    """Review Focus 3. A three-token record fills the scratch, then a one-token record rewrites row 0 only. Mutant:
    rank tokens_max rows -- red."""
    x3, w, bias = exact_gate_case(1, tokens=3)
    x1 = exact_gate_case(2, tokens=1)[0]
    kw = dict(top_k=6, per_token=2, per_layer=KEEP, meta=True)
    want = gate_reference(x1, w, bias, [False] * 16, **kw)
    stale = gate_reference(torch.cat([x1, x3[1:]]), w, bias, [False] * 16, **kw)
    assert want != stale, "pick seeds whose stale rows change the ranking"
    scores = torch.empty((3, 16), dtype=torch.float32, device="cuda")
    page = es.new_candidate_page(pin=True)
    _candidates(x3.to(BF16), w, bias, top_k=6, per_token=2, seq=1, tokens_max=3, page=page, scores=scores)
    got = _candidates(x1.to(BF16), w, bias, top_k=6, per_token=2, seq=2, tokens_max=3, page=page, scores=scores)
    assert list(got.picks) == want


def test_seventeen_records_lap_the_first_slot_and_the_seqlock_shows_it():
    x, w, bias = exact_gate_case(0, tokens=1)
    page = es.new_candidate_page(pin=True)
    scores = torch.empty((1, 16), dtype=torch.float32, device="cuda")
    for seq in range(1, 18):
        _candidates(x.to(BF16), w, bias, top_k=6, per_token=1, seq=seq, page=page, scores=scores)
    assert es.read_candidates(page, 1).status == "lapped"
    assert es.read_candidates(page, 17).status == "ready" and es.read_candidates(page, 2).status == "ready"
    assert es.read_candidates(page, 18).status == "not_yet"


def test_the_seq_word_is_stored_last():
    """With the closing store compiled out (EXL3_RAM_MISS_TEST_SPEC_NO_SEQ), a slot that already held this record's
    seq reads as open, its payload written: the kernel opens the slot (seq 0, a release fence) before any payload store,
    and only the final release publishes it. Mutant: drop the opening store -- red (the stale seq reads ready)."""
    module = es.device_module_with_hooks(["EXL3_RAM_MISS_TEST_SPEC_NO_SEQ"])
    x, w, bias = exact_gate_case(0, tokens=1)
    page = es.new_candidate_page(pin=True)
    off = es.candidate_offset(5)
    page[off : off + 4] = torch.tensor([5], dtype=torch.int32).view(torch.uint8)
    got = _candidates(x.to(BF16), w, bias, top_k=6, per_token=1, seq=5, page=page, module=module)
    assert got.status == "not_yet"
    assert int(page[off : off + 4].view(torch.int32)[0]) == 0 and int(page[off + 4 : off + 6].view(torch.int16)[0]) == 1
```

Append to `test/registered/unit/kernels/test_exl3_ram_miss_device_args.py`:

```python
def test_the_spec_kernels_bounds_are_the_python_mirror():
    spec = _constants(CSRC / "expert_stream" / "spec_score.cuh")
    assert (spec["kSelectMaxExperts"], spec["kSpecDepth"]) == (ram_miss.SPEC_SELECT_MAX_EXPERTS, ram_miss.SPEC_DEPTH)
    assert spec["kSelectMaxTokens"] == lease.CPU_TOKENS_MAX
```

- [ ] **Step 3: Commit the failing tests, push, and run them on divix01**

```bash
cd /Users/dnikolaidis/.codex/worktrees/ram-prefetch-margin
git add python/sglang/test/dsv41_ram_prefetch_fixtures.py test/registered/unit/kernels/test_exl3_ram_prefetch_scorer.py test/manual/dsv41/test_exl3_ram_prefetch_gpu_scorer_cuda.py test/registered/unit/kernels/test_exl3_ram_miss_device_args.py
git commit -m "Test the GPU scorer's kernels against the host scorer's reference

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01LxB2wQZv5EVNuBGUJJbkGk"
git push origin codex/dsv41-ram-prefetch-margin
ssh divix01 bash -s <<'REMOTE'
set -u
cd /data/models/slang/nvfp4-work/wt-ram-prefetch-margin && git fetch -q origin && git checkout -q --detach origin/codex/dsv41-ram-prefetch-margin && git log -1 --oneline
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
env PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 taskset -c 0-63 /data/models/slang/.venv/bin/python -m pytest -q -p no:randomly test/registered/unit/kernels/test_exl3_ram_prefetch_scorer.py 2>&1 | tail -3; echo "EXIT=${PIPESTATUS[0]}"
flock /data/models/slang/nvfp4-work/cc-gpu.lock taskset -c 32-63 env PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 /data/models/slang/.venv/bin/python -m pytest -q -p no:randomly test/manual/dsv41/test_exl3_ram_prefetch_gpu_scorer_cuda.py -x 2>&1 | tail -8; echo "EXIT=${PIPESTATUS[0]}"
REMOTE
```
Expected: no GPU process listed (production is stopped; if one is, stop and report); the scorer file `EXIT=0` (the move kept it green); the GPU file `EXIT=1` with `AttributeError: module ... has no attribute 'run_spec_score'`.

- [ ] **Step 4: The kernels**

Create `python/sglang/kernels/jit/csrc/moe/expert_stream/spec_score.cuh`:

```cpp
// The RAM prefetch's GPU scorer (spec docs/superpowers/specs/2026-10-09-dsv41-ram-prefetch-gpu-scorer-design.md):
// two kernels launched right after layer T's post, on its stream and inside its captured graph.
//
//   exl3_ram_prefetch_score_kernel   sqrt(softplus(w_e . x_t)) + b_e of layer T+1's gate for every live token, fp32,
//                                    into a [tokens_max, experts] scratch
//   exl3_ram_prefetch_select_kernel  GateScorer::choose's ranking (host/gate_scorer.h) past the experts VRAM-hot or
//                                    RAM-mapped in layer T+1, published to the record's candidate slot (spec_candidates.h)
//   SpecScoreKernel                  the checked host launchers for both
#pragma once

#include <sgl_kernel/tensor.h>

#include "lease_device.cuh"
#include "spec_candidates.h"
#include "tensor_checks.h"
#include <cuda_bf16.h>

namespace sglang {
namespace device::expert_stream {

constexpr int kScoreWarps = 8;           // a score block's warps; each scores one expert at a time
constexpr int kScoreTokenTile = 8;       // tokens accumulated per pass over a gate row
constexpr int kSelectThreads = 256;      // one warp per token while ordering, then thread 0 alone
constexpr int kSelectMaxExperts = 1024;  // the select kernel's shared per-expert arrays; 32 per lane while ordering
constexpr int kSelectMaxTokens = 32;     // CpuTokenTable::kMaxTokens
constexpr int kSpecDepth = 12;           // GateScorer::kDepth: each token's ranks walked

SGL_DEVICE float neg_inf() {
  return __uint_as_float(0xFF800000u);
}

SGL_DEVICE float gate_value(const bf16_t* w, int64_t i) {
  return __bfloat162float(w[i]);
}

SGL_DEVICE float gate_value(const float* w, int64_t i) {
  return w[i];
}

// GateScorer's order: the higher score first, the lower id on a tie.
SGL_DEVICE bool ranks_above(float sa, int32_t a, float sb, int32_t b) {
  return sa > sb || (sa == sb && a < b);
}

}  // namespace device::expert_stream

// Arguments of the score kernel, a __grid_constant__ a captured graph freezes.
struct SpecScoreParams {
  const bf16_t* x;    // [tokens, hidden]: layer T's MoE input
  const void* w;      // [experts, hidden]: layer T+1's router weight, bf16 or fp32 (router_fp32)
  const float* bias;  // [experts]: its score-correction bias
  float* scores;      // [tokens_max, experts]; rows past `tokens` keep an earlier record's scores
  int64_t tokens;
  int64_t hidden;
  int64_t experts;
};

// The score kernel: one warp per expert at a time, its lanes striding the hidden size for kScoreTokenTile tokens per
// pass, fp32 sums, then sqrt(softplus(z)) + b with torch's softplus threshold of 20. The sqrt and the add are
// correctly rounded whatever the module's math flags, so integer logits score as the host's do, bit for bit.
template <class W>
__global__ __launch_bounds__(device::expert_stream::kScoreWarps * 32) void exl3_ram_prefetch_score_kernel(
    const __grid_constant__ SpecScoreParams p) {
  using namespace device::expert_stream;
  const int lane = threadIdx.x % 32;
  const W* w = static_cast<const W*>(p.w);
  const int64_t warps = static_cast<int64_t>(gridDim.x) * kScoreWarps;
  for (int64_t e = static_cast<int64_t>(blockIdx.x) * kScoreWarps + threadIdx.x / 32; e < p.experts; e += warps) {
    const W* row = w + e * p.hidden;
    for (int64_t t0 = 0; t0 < p.tokens; t0 += kScoreTokenTile) {
      float acc[kScoreTokenTile] = {};
      for (int64_t h = lane; h < p.hidden; h += 32) {
        const float wv = gate_value(row, h);
#pragma unroll
        for (int j = 0; j < kScoreTokenTile; ++j)
          if (t0 + j < p.tokens) acc[j] = __fmaf_rn(wv, __bfloat162float(p.x[(t0 + j) * p.hidden + h]), acc[j]);
      }
#pragma unroll
      for (int j = 0; j < kScoreTokenTile; ++j) {
        float z = acc[j];
#pragma unroll
        for (int offset = 16; offset > 0; offset /= 2)
          z = __fadd_rn(z, __shfl_xor_sync(0xFFFFFFFFu, z, offset));
        if (lane == 0 && t0 + j < p.tokens) {
          const float softplus = z > 20.0f ? z : log1pf(expf(z));
          const float s = __fadd_rn(__fsqrt_rn(softplus), p.bias[e]);
          p.scores[(t0 + j) * p.experts + e] = isnan(s) ? neg_inf() : s;  // a NaN score ranks below every other
        }
      }
    }
  }
}

// Arguments of the select kernel, a __grid_constant__ a captured graph freezes.
struct SpecSelectParams {
  const float* scores;  // the score kernel's [tokens_max, experts]
  int64_t tokens;       // the record's live tokens; outside 1..tokens_max the slot gets count 0
  int64_t tokens_max;
  int64_t experts;
  int64_t top_k;
  int64_t per_token;
  int64_t top_k_only;
  const int64_t* hot_slots;  // layer T+1's VRAM slots' experts (-1 empty), hot_capacity of them
  int64_t hot_capacity;
  const int32_t* ram_slot;  // layer T+1's row of the device map: >= 0 mapped in RAM
  const int32_t* state;     // the post's state words: kPosted is the record's seq
  uint8_t* candidates;      // the pinned candidate page
};

// The select kernel, one block of kSelectThreads. Warp w orders tokens w, w + 8, ...: kSpecDepth rounds of a warp
// argmax, lane l holding experts l, l + 32, .... Thread 0 then walks each token's order past the skipped experts as
// GateScorer::choose does, merges the picks by best margin, and publishes up to kMaxCandidates to the record's slot:
// seq 0, a release fence, the payload, then the seq with a release (the hot page's order).
__global__ __launch_bounds__(device::expert_stream::kSelectThreads, 1) void exl3_ram_prefetch_select_kernel(
    const __grid_constant__ SpecSelectParams p) {
  using namespace device::expert_stream;
  using Cand = ::sglang::expert_stream::wire::SpecCandidates;
  __shared__ uint8_t skip[kSelectMaxExperts];
  __shared__ uint8_t picked[kSelectMaxExperts];  // 0 never picked, 1 picked, 2 published
  __shared__ uint8_t rank[kSelectMaxExperts];
  __shared__ float best[kSelectMaxExperts];
  __shared__ int16_t order[kSelectMaxTokens][kSpecDepth];
  __shared__ int16_t list[kSelectMaxTokens * kSpecDepth];
  const bool live = p.tokens >= 1 && p.tokens <= p.tokens_max;
  const int64_t depth = min(static_cast<int64_t>(kSpecDepth), p.experts);
  for (int64_t e = threadIdx.x; e < p.experts; e += blockDim.x) {
    skip[e] = p.ram_slot[e] >= 0 ? 1 : 0;
    picked[e] = 0;
  }
  __syncthreads();
  for (int64_t s = threadIdx.x; s < p.hot_capacity; s += blockDim.x) {
    const int64_t e = p.hot_slots[s];
    if (e >= 0 && e < p.experts) skip[e] = 1;
  }
  if (live) {
    const int lane = threadIdx.x % 32;
    for (int64_t t = threadIdx.x / 32; t < p.tokens; t += kSelectThreads / 32) {
      const float* s = p.scores + t * p.experts;
      uint32_t taken = 0;  // bit k: expert lane + 32k is already ordered
      for (int64_t i = 0; i < depth; ++i) {
        float top = neg_inf();
        int32_t id = 0x7FFFFFFF;
        for (int k = 0; k < kSelectMaxExperts / 32 && lane + 32 * k < p.experts; ++k) {
          const int32_t e = lane + 32 * k;
          if ((taken >> k & 1u) == 0 && ranks_above(s[e], e, top, id)) {
            top = s[e];
            id = e;
          }
        }
#pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
          const float other = __shfl_xor_sync(0xFFFFFFFFu, top, offset);
          const int32_t other_id = __shfl_xor_sync(0xFFFFFFFFu, id, offset);
          if (ranks_above(other, other_id, top, id)) {
            top = other;
            id = other_id;
          }
        }
        if (id % 32 == lane) taken |= 1u << (id / 32);
        if (lane == 0) order[t][i] = static_cast<int16_t>(id);
      }
    }
  }
  __syncthreads();
  if (threadIdx.x != 0) return;
  int n = 0;
  if (live) {
    const int64_t walk = p.top_k_only != 0 ? p.top_k : depth;
    for (int64_t t = 0; t < p.tokens; ++t) {
      const float* s = p.scores + t * p.experts;
      const float kth = s[order[t][p.top_k - 1]];
      int64_t taken = 0;
      for (int64_t i = 0; i < walk && taken < p.per_token; ++i) {
        const int32_t e = order[t][i];
        if (skip[e]) continue;
        // Equal scores (also two infinities) give 0; -inf less a finite score is clamped to the lowest finite float.
        const float margin = s[e] == kth ? 0.0f : fmaxf(__fsub_rn(s[e], kth), -3.40282347e38f);
        if (picked[e] == 0) {
          picked[e] = 1;
          best[e] = margin;
          rank[e] = static_cast<uint8_t>(i);
          list[n++] = static_cast<int16_t>(e);
        } else {
          if (margin > best[e] || (margin == best[e] && i < rank[e])) rank[e] = static_cast<uint8_t>(i);
          best[e] = fmaxf(best[e], margin);
        }
        ++taken;
      }
    }
  }
  const int count = min(n, Cand::kMaxCandidates);
  const uint32_t seq = static_cast<uint32_t>(p.state[kPosted]);
  uint8_t* const slot = p.candidates + static_cast<int64_t>((seq - 1u) % Cand::kCandRecords) * Cand::kCandStride;
  st_relaxed_sys<uint32_t>(slot + Cand::kCandSeq, 0u);
  cuda::atomic_thread_fence(cuda::memory_order_release, cuda::thread_scope_system);
  st_relaxed_sys<uint32_t>(
      slot + Cand::kCandCount, static_cast<uint32_t>(count) | (live ? 0u : Cand::kCandFlagOversize) << 16);
  for (int c = 0; c < count; ++c) {
    int at = -1;
    for (int j = 0; j < n; ++j) {
      const int32_t e = list[j];
      if (picked[e] == 1 && (at < 0 || ranks_above(best[e], e, best[list[at]], list[at]))) at = j;
    }
    const int32_t e = list[at];
    picked[e] = 2;
    uint8_t* const entry = slot + Cand::kCandEntries + c * Cand::kCandEntryBytes;
    st_relaxed_sys<uint32_t>(entry + Cand::kCandExpert, static_cast<uint32_t>(e) | static_cast<uint32_t>(rank[e]) << 16);
    st_relaxed_sys<uint32_t>(entry + Cand::kCandMargin, __float_as_uint(best[e]));
  }
#ifndef EXL3_RAM_MISS_TEST_SPEC_NO_SEQ
  st_release_sys(slot + Cand::kCandSeq, seq);  // orders the payload before the seq
#endif
}

/// \brief Checked host launchers for the GPU scorer's kernels: score, then select, on the stream of the tensors'
/// device and without PDL, so each starts after the post before it completed.
struct SpecScoreKernel {
  /// Launches the score kernel; see SpecScoreParams. `x` bf16 [tokens, hidden], `w` bf16 or fp32 [experts, hidden],
  /// `bias` fp32 [experts], `scores` fp32 [tokens_max, experts] with 1 <= tokens <= tokens_max.
  static void score(
      tvm::ffi::TensorView x, tvm::ffi::TensorView w, tvm::ffi::TensorView bias, tvm::ffi::TensorView scores) {
    using namespace host;
    using namespace device::expert_stream;
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    auto T_ = SymbolicSize{"tokens"};
    auto H_ = SymbolicSize{"hidden"};
    auto E_ = SymbolicSize{"experts"};
    auto M_ = SymbolicSize{"tokens_max"};
    expert_stream::verify_named("x", TensorMatcher({T_, H_}).with_dtype<bf16_t>().with_device<kDLCUDA>(device), x);
    expert_stream::verify_named(
        "w", TensorMatcher({E_, H_}).with_dtype<bf16_t, fp32_t>().with_device<kDLCUDA>(device), w);
    expert_stream::verify_named("bias", TensorMatcher({E_}).with_dtype<fp32_t>().with_device<kDLCUDA>(device), bias);
    expert_stream::verify_named(
        "scores", TensorMatcher({M_, E_}).with_dtype<fp32_t>().with_device<kDLCUDA>(device), scores);
    RuntimeCheck(
        x.is_contiguous() && w.is_contiguous() && bias.is_contiguous() && scores.is_contiguous(),
        "x, w, bias, scores: must be contiguous");
    RuntimeCheck(
        T_.unwrap() >= 1 && T_.unwrap() <= M_.unwrap(), "x: 1..tokens_max rows (a record outside them is not scored)");
    const auto params = SpecScoreParams{
        .x = static_cast<const bf16_t*>(x.data_ptr()),
        .w = w.data_ptr(),
        .bias = static_cast<const float*>(bias.data_ptr()),
        .scores = static_cast<float*>(scores.data_ptr()),
        .tokens = T_.unwrap(),
        .hidden = H_.unwrap(),
        .experts = E_.unwrap(),
    };
    const auto stream = LaunchKernel::resolve_device(scores.device());
    const auto blocks = static_cast<unsigned>((E_.unwrap() + kScoreWarps - 1) / kScoreWarps);
    const bool fp32 = w.dtype().code == kDLFloat;
    LaunchKernel(dim3(blocks), dim3(kScoreWarps * 32), stream)(
        fp32 ? exl3_ram_prefetch_score_kernel<float> : exl3_ram_prefetch_score_kernel<bf16_t>, params);
  }

  /// Launches the select kernel; see SpecSelectParams. `ram_slot` is the device map bank's [rows, experts] and
  /// `target` its row; `candidates_address` the pinned candidate page's address.
  static void select(
      tvm::ffi::TensorView scores,
      int64_t tokens,
      int64_t top_k,
      int64_t per_token,
      int64_t top_k_only,
      tvm::ffi::TensorView hot_slots,
      int64_t hot_capacity,
      tvm::ffi::TensorView ram_slot,
      int64_t target,
      tvm::ffi::TensorView state,
      int64_t candidates_address) {
    using namespace host;
    using namespace device::expert_stream;
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    auto E_ = SymbolicSize{"experts"};
    auto M_ = SymbolicSize{"tokens_max"};
    auto Rows_ = SymbolicSize{"rows"};
    expert_stream::verify_named(
        "scores", TensorMatcher({M_, E_}).with_dtype<fp32_t>().with_device<kDLCUDA>(device), scores);
    expert_stream::verify_named(
        "hot_slots", TensorMatcher({-1}).with_dtype<int64_t>().with_device<kDLCUDA>(device), hot_slots);
    expert_stream::verify_named(
        "ram_slot", TensorMatcher({Rows_, E_}).with_dtype<int32_t>().with_device<kDLCUDA>(device), ram_slot);
    expert_stream::verify_named(
        "state", TensorMatcher({kStateWords}).with_dtype<int32_t>().with_device<kDLCUDA>(device), state);
    const int64_t experts = E_.unwrap();
    const int64_t depth = std::min<int64_t>(kSpecDepth, experts);
    RuntimeCheck(experts <= kSelectMaxExperts, "experts: the select kernel holds at most ", kSelectMaxExperts);
    RuntimeCheck(M_.unwrap() <= kSelectMaxTokens, "scores: at most ", kSelectMaxTokens, " token rows");
    RuntimeCheck(tokens >= 0, "tokens: must not be negative");
    RuntimeCheck(top_k >= 1 && top_k <= depth, "top_k: must be in 1..", depth);
    RuntimeCheck(per_token >= 1 && per_token <= kSpecDepth, "per_token: must be in 1..", kSpecDepth);
    RuntimeCheck(hot_capacity >= 0 && hot_capacity <= hot_slots.size(0), "hot_capacity: at most hot_slots' size");
    RuntimeCheck(target >= 0 && target < Rows_.unwrap(), "target: outside the map bank");
    RuntimeCheck(
        candidates_address != 0 && candidates_address % 128 == 0, "candidates: a nonzero 128-byte aligned address");
    const auto params = SpecSelectParams{
        .scores = static_cast<const float*>(scores.data_ptr()),
        .tokens = tokens,
        .tokens_max = M_.unwrap(),
        .experts = experts,
        .top_k = top_k,
        .per_token = per_token,
        .top_k_only = top_k_only,
        .hot_slots = static_cast<const int64_t*>(hot_slots.data_ptr()),
        .hot_capacity = hot_capacity,
        .ram_slot = static_cast<const int32_t*>(ram_slot.data_ptr()) + target * experts,
        .state = static_cast<const int32_t*>(state.data_ptr()),
        .candidates = reinterpret_cast<uint8_t*>(candidates_address),
    };
    const auto stream = LaunchKernel::resolve_device(scores.device());
    LaunchKernel(dim3(1), dim3(kSelectThreads), stream)(exl3_ram_prefetch_select_kernel, params);
  }
};

}  // namespace sglang
```

In `python/sglang/kernels/jit/csrc/moe/exl3/exl3_ram_miss.cuh`, directly after `#include "../expert_stream/row_copy_kernels.cuh"`:

```cpp
#include "../expert_stream/spec_score.cuh"
```

- [ ] **Step 5: The Python launchers**

In `python/sglang/kernels/ops/moe/expert_stream_transport.py`, directly after `_ROW_COPY_METHODS`:

```python
_SPEC_METHODS = {
    "expert_stream_spec_score": "score",
    "expert_stream_spec_select": "select",
}
# The select kernel's bounds (spec_score.cuh kSpecDepth, kSelectMaxExperts).
SPEC_DEPTH = 12
SPEC_SELECT_MAX_EXPERTS = 1024
```

In `_device_wrappers`, extend the returned list:

```python
    return (
        [(name, f"LeaseProtocolKernel::{method}") for name, method in _LEASE_METHODS.items()]
        + [(name, f"RowCopyKernel<{device_layout}>::{method}") for name, method in _ROW_COPY_METHODS.items()]
        + [(name, f"SpecScoreKernel::{method}") for name, method in _SPEC_METHODS.items()]
    )
```

Directly after `device_module_with_hooks`, add:

```python
def run_spec_score(x, w, bias, scores, *, module=None) -> None:
    """The GPU scorer's score kernel (spec_score.cuh): sqrt(softplus(w @ x_t)) + bias for each row of ``x`` into the
    same row of ``scores``. ``x`` bf16 [tokens, hidden], ``w`` bf16 or fp32 [experts, hidden], ``bias`` fp32
    [experts], ``scores`` fp32 [tokens_max, experts], one CUDA device. ``module``: the device module (default: the
    8-lane, one-node build)."""
    (module or _device_module()).expert_stream_spec_score(x, w, bias, scores)


def run_spec_select(
    scores, tokens, *, top_k, per_token, top_k_only, hot_slots, hot_capacity, ram_slot, target, state, candidates,
    module=None,
) -> None:
    """The GPU scorer's select kernel: ranks ``scores``' first ``tokens`` rows past ``hot_slots[:hot_capacity]`` and
    the experts ``ram_slot[target]`` maps, and publishes up to CAND_MAX candidates to the slot of ``state``'s posted
    seq in the pinned ``candidates`` page; count 0 when ``tokens`` is outside 1..scores' rows."""
    (module or _device_module()).expert_stream_spec_select(
        scores,
        int(tokens),
        int(top_k),
        int(per_token),
        int(bool(top_k_only)),
        hot_slots,
        int(hot_capacity),
        ram_slot,
        int(target),
        state,
        int(candidates.data_ptr()),
    )
```

- [ ] **Step 6: Commit, push, and run on divix01**

```bash
cd /Users/dnikolaidis/.codex/worktrees/ram-prefetch-margin
git add python/sglang/kernels/jit/csrc/moe/expert_stream/spec_score.cuh python/sglang/kernels/jit/csrc/moe/exl3/exl3_ram_miss.cuh python/sglang/kernels/ops/moe/expert_stream_transport.py
git commit -m "Score and select the next layer's candidates on the GPU after the post

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01LxB2wQZv5EVNuBGUJJbkGk"
git push origin codex/dsv41-ram-prefetch-margin
ssh divix01 bash -s <<'REMOTE'
set -u
cd /data/models/slang/nvfp4-work/wt-ram-prefetch-margin && git fetch -q origin && git checkout -q --detach origin/codex/dsv41-ram-prefetch-margin && git log -1 --oneline
env PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 taskset -c 0-63 /data/models/slang/.venv/bin/python -m pytest -q -p no:randomly test/registered/unit/kernels/test_exl3_ram_miss_device_args.py 2>&1 | tail -3; echo "EXIT=${PIPESTATUS[0]}"
flock /data/models/slang/nvfp4-work/cc-gpu.lock taskset -c 32-63 env PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 /data/models/slang/.venv/bin/python -m pytest -q -p no:randomly test/manual/dsv41/test_exl3_ram_prefetch_gpu_scorer_cuda.py test/manual/dsv41/test_exl3_lease_kernels_cuda.py 2>&1 | tail -4; echo "EXIT=${PIPESTATUS[0]}"
REMOTE
```
Expected: both `EXIT=0` (the device module and the hook module rebuild on this first run). `test_exl3_lease_kernels_cuda.py` stays green: the chain's kernels are unchanged.

- [ ] **Step 7: Mutants (divix01 worktree only, reverted after each)**

Apply each to `python/sglang/kernels/jit/csrc/moe/expert_stream/spec_score.cuh`, run the GPU file with the Step 6 GPU command, confirm red, then `git checkout -- python/sglang/kernels/jit/csrc/moe/expert_stream/spec_score.cuh`:
1. `const float kth = s[order[t][p.top_k - 1]];` -> `s[order[t][p.top_k - 2 < 0 ? 0 : p.top_k - 2]]` -- red in `test_the_gpu_ranks_exactly_as_the_host_reference`.
2. In the warp argmax, `ranks_above(s[e], e, top, id)` -> `s[e] > top` -- red in `test_tied_scores_and_margins_go_to_the_lower_id`.
3. `for (int64_t t = 0; t < p.tokens; ++t)` (thread 0's walk) -> `t < p.tokens_max` and the warp loop's `t < p.tokens` -> `t < p.tokens_max` -- red in `test_a_record_with_fewer_tokens_ranks_only_its_own_rows_of_the_scratch`.
4. Delete `st_relaxed_sys<uint32_t>(slot + Cand::kCandSeq, 0u);` -- red in `test_the_seq_word_is_stored_last`.
5. `isnan(s) ? neg_inf() : s` -> `s` -- red in `test_nan_scores_rank_last_in_id_order`.

After the last revert, re-run the Step 6 GPU command; expected `EXIT=0`. Record the reds and the green re-run.

---

### Task 4: Wiring: the per-row table, the launch after the post, the replayed graph

**Files:**
- Modify: `python/sglang/srt/layers/moe/ram_prefetch.py` (`PrefetchTargets`, `prefetch_targets`, `prefetch_tables` on top of it, `SpecScoreRow`, `spec_score_rows`, `GpuScorer`)
- Modify: `python/sglang/kernels/ops/moe/expert_stream_transport.py` (`ExpertStreamDevice.__init__`, `enable_spec_scorer`, `spec_score`)
- Modify: `python/sglang/srt/layers/moe/exl3_ram_miss.py` (`Exl3RamMissRowBackend.__init__` / `post`; `Exl3RamMissService.__init__`, `ensure_started`, `_enable_ram_prefetch`, `attach`, new `_note_spec_row` / `_bind_spec_scores`, `_quarantine`)
- Test: `test/registered/unit/layers/moe/test_ram_prefetch_tables.py`, `test/registered/unit/kernels/test_exl3_ram_miss_attach_lanes.py`, `test/registered/unit/kernels/test_exl3_ram_miss_device_args.py`, `test/manual/dsv41/test_exl3_ram_prefetch_gpu_scorer_cuda.py`

**Interfaces:**
- Consumes: Task 1's `envs.SGLANG_DSV41_RAM_PREFETCH_SCORER` and `SCORERS`; Task 2's `new_candidate_page`, `CAND_PAGE_BYTES`, the host wrapper's `candidates=`; Task 3's `run_spec_score`, `run_spec_select`, `SPEC_DEPTH`, `SPEC_SELECT_MAX_EXPERTS`, `gate_reference`, `exact_gate_case`.
- Produces:
  - `ram_prefetch.PrefetchTargets(targets: Tensor, gates: tuple[RouterGate, ...], top_k: int)`; `prefetch_targets(layer_ids, gates, *, hidden) -> PrefetchTargets`; `SpecScoreRow(weight, bias, target: int, hot_slots, hot_capacity: int)`; `spec_score_rows(picked: PrefetchTargets, hot: Mapping[int, tuple[Tensor, int]]) -> dict[int, SpecScoreRow]`; `GpuScorer(picked: PrefetchTargets, candidates: Tensor, per_token: int, top_k_only: bool)`.
  - `ExpertStreamDevice.enable_spec_scorer(candidates, *, top_k, per_token, top_k_only) -> None`; `ExpertStreamDevice.spec_score(entry, x) -> None`; attributes `spec_candidates`, `spec_scores` (None until enabled).
  - `Exl3RamMissRowBackend.spec_expected: bool`, `.spec_score: Optional[SpecScoreRow]`; the post launches `side.spec_score(entry=..., x=...)` right after `side.post`.
  - `Exl3RamMissService._enable_ram_prefetch(host, layer_ids, numa, cpu_experts, *, pin=False) -> Optional[GpuScorer]`; `._spec_gpu`; `._note_spec_row(row, backend)`; `._bind_spec_scores()`.

- [ ] **Step 1: Write the failing tests**

Append to `test/registered/unit/layers/moe/test_ram_prefetch_tables.py` (add `GpuScorer, prefetch_targets, spec_score_rows` to the `ram_prefetch` import and `from sglang.kernels.ops.moe.expert_stream_transport import CAND_PAGE_BYTES, new_candidate_page`):

```python
def _numa():
    return SimpleNamespace(
        nodes=1, plans=[NodePlan(group=0, node=0, ram=17, cpu=(8, 9), sq=None, busy_poll=True, spec=(0, 1))]
    )


def test_the_gpu_scorer_enables_the_host_with_a_candidate_page_and_no_host_gate_copy():
    gate = _gate()
    register_router_gate(1, gate.weight, gate.bias, 6)
    calls = []
    host = SimpleNamespace(enable_ram_prefetch=lambda *a, **kw: calls.append((a, kw)))
    cpu = SimpleNamespace(services=[SimpleNamespace(hidden=8)])
    with envs.SGLANG_DSV41_RAM_PREFETCH_SCORER.override("gpu"):
        scorer = module.Exl3RamMissService._enable_ram_prefetch(host, [0, 1], _numa(), cpu)
    (targets, gates, bias), kw = calls[0]
    assert targets.tolist() == [[1, 0], [-1, -1]] and gates is None and bias is None
    page = kw.pop("candidates")
    assert page.dtype == torch.uint8 and page.numel() == CAND_PAGE_BYTES and not page.is_pinned()
    assert kw == dict(top_k=6, per_token=1, per_layer=1, cores=[[0, 1]], top_k_only=False)
    assert scorer.candidates is page and scorer.picked.gates[0].weight is gate.weight
    assert (scorer.per_token, scorer.top_k_only, scorer.picked.top_k) == (1, False, 6)


def test_the_cpu_scorer_returns_no_gpu_scorer_and_an_unknown_one_is_refused():
    gate = _gate()
    register_router_gate(1, gate.weight, gate.bias, 6)
    host = SimpleNamespace(enable_ram_prefetch=lambda *a, **kw: None)
    cpu = SimpleNamespace(services=[SimpleNamespace(hidden=8)])
    assert module.Exl3RamMissService._enable_ram_prefetch(host, [0, 1], _numa(), cpu) is None
    with envs.SGLANG_DSV41_RAM_PREFETCH_SCORER.override("tpu"):
        with pytest.raises(ValueError, match="SGLANG_DSV41_RAM_PREFETCH_SCORER"):
            module.Exl3RamMissService._enable_ram_prefetch(host, [0, 1], _numa(), cpu)


def test_the_gpu_scorers_table_has_a_row_once_its_target_attached():
    gate = _gate()
    register_router_gate(1, gate.weight, gate.bias, 6)
    picked = prefetch_targets([0, 1, 2], registered_gates(), hidden=8)
    hot = torch.full((3,), -1, dtype=torch.int64)
    assert spec_score_rows(picked, {}) == {}
    rows = spec_score_rows(picked, {1: (hot, 3)})
    assert list(rows) == [0]
    entry = rows[0]
    assert (entry.target, entry.hot_capacity) == (1, 3) and entry.hot_slots is hot
    assert torch.equal(entry.weight, gate.weight) and torch.equal(entry.bias, gate.bias)


@pytest.mark.parametrize(
    "weight_dtype, bias_dtype, why",
    [(torch.float16, torch.float32, "bf16 or fp32 gate"), (torch.bfloat16, torch.bfloat16, "fp32 bias")],
)
def test_a_gate_the_kernels_cannot_read_is_refused_when_the_table_is_built(weight_dtype, bias_dtype, why):
    gate = _gate()
    register_router_gate(1, gate.weight.to(weight_dtype), gate.bias.to(bias_dtype), 6)
    picked = prefetch_targets([0, 1], registered_gates(), hidden=8)
    with pytest.raises(ValueError, match=why):
        spec_score_rows(picked, {1: (torch.full((3,), -1, dtype=torch.int64), 3)})


def test_the_service_binds_each_row_as_its_target_attaches_and_flags_rows_with_a_target():
    gate = _gate()
    register_router_gate(1, gate.weight, gate.bias, 6)
    svc = module.Exl3RamMissService()
    picked = prefetch_targets([0, 1], registered_gates(), hidden=8)
    svc._spec_gpu = GpuScorer(picked, new_candidate_page(pin=False), per_token=1, top_k_only=False)
    hot0, hot1 = (torch.full((3,), -1, dtype=torch.int64) for _ in range(2))
    b0 = SimpleNamespace(spec_expected=False, spec_score=None, hot_slots=hot0, hot_capacity=3)
    b1 = SimpleNamespace(spec_expected=False, spec_score=None, hot_slots=hot1, hot_capacity=3)
    svc._note_spec_row(0, b0)
    assert b0.spec_expected and b0.spec_score is None  # row 1, its target, has not attached
    svc._note_spec_row(1, b1)
    assert b0.spec_score.target == 1 and b0.spec_score.hot_slots is hot1
    assert not b1.spec_expected and b1.spec_score is None


def test_the_candidate_page_and_the_score_scratch_are_quarantined(monkeypatch):
    """The select kernel writes the page through UVA: a shutdown that cannot establish the GPU stopped keeps it."""
    gate = _gate()
    register_router_gate(1, gate.weight, gate.bias, 6)
    svc = module.Exl3RamMissService()
    page = new_candidate_page(pin=False)
    svc._spec_gpu = GpuScorer(prefetch_targets([0, 1], registered_gates(), hidden=8), page, 1, False)
    owned = []
    monkeypatch.setattr(module, "quarantine_host_slabs", owned.extend)
    svc._quarantine("test")
    assert any(t is page for t in owned)
```

In `test/registered/unit/kernels/test_exl3_ram_miss_attach_lanes.py`, change `_Side.__getattr__`'s name tuple to `("post", "stream", "copy_wait", "spec_score")`, and append:

```python
def test_a_row_with_a_target_scores_right_after_its_post(monkeypatch):
    """The scoring kernels follow the post on its stream, before C1: the host learns the record and its candidates
    while this layer still runs (spec 2026-10-09, Approach A)."""
    streamer = SimpleNamespace(_plan_miss_keys=torch.zeros(EXPERTS, dtype=torch.int64))
    backend, side = _captured_backend(monkeypatch, lambda: streamer)
    entry = SimpleNamespace(target=1)
    backend.spec_expected, backend.spec_score = True, entry
    backend.post(0, _CapturedPlan())
    assert [name for name, _ in side.calls] == ["post", "spec_score", "copy", "stream", "copy_wait"]
    assert side.calls[1][1]["entry"] is entry and side.calls[1][1]["x"] is backend.cpu_input[0]


def test_a_row_whose_target_has_not_attached_is_refused_before_anything_posts(monkeypatch):
    """The table is built at attach, before capture; a post without its entry would freeze a graph that never scores."""
    streamer = SimpleNamespace(_plan_miss_keys=torch.zeros(EXPERTS, dtype=torch.int64))
    backend, side = _captured_backend(monkeypatch, lambda: streamer)
    backend.spec_expected = True
    with pytest.raises(RuntimeError, match="row 0 posted before its target row attached"):
        backend.post(0, _CapturedPlan())
    assert not side.calls


def test_a_row_without_a_target_posts_the_chain_alone(monkeypatch):
    streamer = SimpleNamespace(_plan_miss_keys=torch.zeros(EXPERTS, dtype=torch.int64))
    backend, side = _captured_backend(monkeypatch, lambda: streamer)
    backend.post(0, _CapturedPlan())
    assert [name for name, _ in side.calls] == ["post", "copy", "stream", "copy_wait"]
```

Append to `test/registered/unit/kernels/test_exl3_ram_miss_device_args.py` (add `from types import SimpleNamespace` to the imports):

```python
def test_the_gpu_scorer_takes_a_candidate_page_of_the_slots_size():
    with pytest.raises(ValueError, match="candidates must be a contiguous CPU uint8 tensor"):
        _device().enable_spec_scorer(torch.zeros(10, dtype=torch.uint8), top_k=2, per_token=1, top_k_only=False)


def test_the_gpu_scorer_refuses_more_experts_than_the_select_kernel_holds():
    """Review Focus 5: refused at start, not overflowing the select kernel's shared arrays inside a replay."""
    with pytest.raises(ValueError, match="1024"):
        _device(experts=1025).enable_spec_scorer(
            ram_miss.new_candidate_page(pin=False), top_k=2, per_token=1, top_k_only=False
        )


def test_the_gpu_scorer_refuses_a_choice_outside_the_kernels_bounds():
    for top_k, per_token in ((0, 1), (5, 1), (2, 0), (2, 13)):
        with pytest.raises(ValueError, match="top_k|per_token"):
            _device().enable_spec_scorer(
                ram_miss.new_candidate_page(pin=False), top_k=top_k, per_token=per_token, top_k_only=False
            )


def test_the_gpu_scorer_sizes_its_scratch_by_the_rows_tokens():
    dev = _device()
    dev.enable_spec_scorer(ram_miss.new_candidate_page(pin=False), top_k=2, per_token=1, top_k_only=False)
    assert tuple(dev.spec_scores.shape) == (dev.cpu_tokens_max, 4) and dev.spec_scores.dtype == torch.float32
    with pytest.raises(RuntimeError, match="already enabled"):
        dev.enable_spec_scorer(ram_miss.new_candidate_page(pin=False), top_k=2, per_token=1, top_k_only=False)


def test_scoring_before_the_gpu_scorer_is_enabled_is_refused():
    with pytest.raises(RuntimeError, match="enable_spec_scorer"):
        _device().spec_score(SimpleNamespace(target=0), None)
```

Append to `test/manual/dsv41/test_exl3_ram_prefetch_gpu_scorer_cuda.py` (add `import random` to the imports and these two imports after the existing ones):

```python
from lease_chain_rig import EXPERTS, TOP_K, Chain  # noqa: E402

from sglang.srt.layers.moe.ram_prefetch import SpecScoreRow  # noqa: E402


def _step(c, experts, row=0):
    c.plan(experts, row)
    c.gather(row)
    snapshot = c.snapshot(row)
    torch.cuda.synchronize()
    c.check(experts, snapshot, row)


def test_a_captured_post_and_its_scoring_replay_the_reference_candidates(tmp_path):
    """The production row backend's post with its scoring, captured once, replayed with new inputs, hot sets and
    plans twice round the 16-slot ring: each replay's slot holds the reference's candidates for that replay's x past
    its hot slots and row 1's device map. Mutant: score a copy of x taken at capture -- red."""
    c = Chain(tmp_path)
    try:
        x16, w16, bias = exact_gate_case(0, tokens=1, experts=EXPERTS)
        x = x16.to(BF16).cuda()
        hot = torch.full((4,), -1, dtype=torch.int64, device="cuda")
        page = es.new_candidate_page(pin=True)
        c.dev.enable_spec_scorer(page, top_k=TOP_K, per_token=2, top_k_only=False)
        backend = c.backends[0]
        backend.spec_expected = True
        backend.spec_score = SpecScoreRow(w16.cuda(), bias.cuda(), 1, hot, 4)
        backend.cpu_input = (x, None)  # the scoring reads x alone: this rig's rows have no CPU experts
        _step(c, [0, 1], 0)  # loads both kernels and scores eagerly
        _step(c, [2, 3], 1)
        _step(c, [2, 3], 1)  # applies row 1's delta: 2 and 3 mapped in the device map
        assert c.handled()
        mapped = {e for e, slot in enumerate(c.device_map(1)) if slot >= 0}
        assert {2, 3} <= mapped
        stream = torch.cuda.Stream()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
            c.gather(0)
        rng = random.Random(11)
        for i in range(2 * es.CAND_RECORDS + 1):
            xi = exact_gate_case(100 + i, tokens=1, experts=EXPERTS)[0]
            hot_i = rng.sample(range(EXPERTS), 4)
            with torch.cuda.stream(stream):
                x.copy_(xi.to(BF16))
                hot.copy_(torch.tensor(hot_i, dtype=torch.int64))
                c.plan(rng.sample(range(EXPERTS), rng.randint(1, TOP_K)), 0)
                graph.replay()
            stream.synchronize()
            seq = int(c.dev.stats()["posted"]) & 0xFFFFFFFF
            skip = [e in hot_i or e in mapped for e in range(EXPERTS)]
            want = gate_reference(xi, w16, bias, skip, top_k=TOP_K, per_token=2, per_layer=KEEP, meta=True)
            got = es.read_candidates(page, seq)
            assert got.status == "ready" and list(got.picks) == want, i
        assert c.handled()
    finally:
        c.close()
```

- [ ] **Step 2: Commit the failing tests, push, and run them on divix01**

```bash
cd /Users/dnikolaidis/.codex/worktrees/ram-prefetch-margin
git add test/registered/unit/layers/moe/test_ram_prefetch_tables.py test/registered/unit/kernels/test_exl3_ram_miss_attach_lanes.py test/registered/unit/kernels/test_exl3_ram_miss_device_args.py test/manual/dsv41/test_exl3_ram_prefetch_gpu_scorer_cuda.py
git commit -m "Test the GPU scorer's wiring: the per-row table, the launch after the post, the replay

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01LxB2wQZv5EVNuBGUJJbkGk"
git push origin codex/dsv41-ram-prefetch-margin
ssh divix01 bash -s <<'REMOTE'
set -u
cd /data/models/slang/nvfp4-work/wt-ram-prefetch-margin && git fetch -q origin && git checkout -q --detach origin/codex/dsv41-ram-prefetch-margin && git log -1 --oneline
PY="env PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 taskset -c 0-63 /data/models/slang/.venv/bin/python -m pytest -q -p no:randomly"
$PY test/registered/unit/layers/moe/test_ram_prefetch_tables.py 2>&1 | tail -6; echo "EXIT=${PIPESTATUS[0]}"
$PY test/registered/unit/kernels/test_exl3_ram_miss_attach_lanes.py test/registered/unit/kernels/test_exl3_ram_miss_device_args.py -k "target or gpu_scorer or scoring" 2>&1 | tail -6; echo "EXIT=${PIPESTATUS[0]}"
REMOTE
```
Expected: both `EXIT=1` or `EXIT=2`: `ImportError: cannot import name 'GpuScorer'` (tables), `AttributeError: 'ExpertStreamDevice' object has no attribute 'enable_spec_scorer'`, and the attach-lanes tests failing on the missing `spec_score` call and the missing refusal.

- [ ] **Step 3: The table in `ram_prefetch.py`**

Replace `prefetch_tables` (signature through `return PrefetchTables(...)`) with:

```python
@dataclass(frozen=True)
class PrefetchTargets:
    targets: torch.Tensor  # int64 [rows, 2]: per source row its target row and gate index, or (-1, -1)
    gates: tuple  # RouterGate per gate index: the target layer's registered gate, uncopied
    top_k: int


def prefetch_targets(layer_ids: Sequence[int], gates: Mapping[int, RouterGate], *, hidden: int) -> PrefetchTargets:
    """Row r targets row r + 1 when that row is the next layer (layer_ids[r] + 1) and its gate has a bias; the last row,
    a non-consecutive layer and a hash-routed one target nothing. Every target's gate has ``hidden`` columns, one
    expert count and one top_k."""
    picks = []
    for row in range(len(layer_ids) - 1):
        nxt = layer_ids[row + 1]
        gate = gates.get(nxt)
        if nxt == layer_ids[row] + 1 and gate is not None and gate.bias is not None:
            picks.append((row, gate, nxt))
    if not picks:
        raise ValueError("RAM prefetch: no streamed row has a next layer with a registered, biased router gate")
    experts = int(picks[0][1].weight.shape[0])
    top_k = picks[0][1].top_k
    for _, gate, layer_id in picks:
        if gate.weight.dim() != 2 or int(gate.weight.shape[1]) != hidden:
            raise ValueError(
                f"RAM prefetch: layer {layer_id}'s gate has hidden size {int(gate.weight.shape[-1])}, "
                f"the CPU rows {hidden}"
            )
        if int(gate.weight.shape[0]) != experts:
            raise ValueError(
                f"RAM prefetch: layer {layer_id}'s gate has {int(gate.weight.shape[0])} experts, not {experts}"
            )
        if gate.top_k != top_k:
            raise ValueError(f"RAM prefetch: layer {layer_id}'s top_k {gate.top_k} is not {top_k}")
    targets = torch.full((len(layer_ids), 2), -1, dtype=torch.int64)
    for index, (row, _, _) in enumerate(picks):
        targets[row, 0], targets[row, 1] = row + 1, index
    return PrefetchTargets(targets, tuple(gate for _, gate, _ in picks), top_k)


def prefetch_tables(
    layer_ids: Sequence[int], gates: Mapping[int, RouterGate], *, hidden: int, node: Optional[int] = None
) -> PrefetchTables:
    """prefetch_targets with each target's gate copied to host memory once, bound to NUMA ``node`` when given, for the
    CPU scorer. The host scorer reads bf16, so an fp32 router (``router_fp32``) is rounded to bf16 here: its ranking can
    differ slightly from the model's, which costs prefetch hit rate and never correctness."""
    picked = prefetch_targets(layer_ids, gates, hidden=hidden)
    n = len(picked.gates)
    experts = int(picked.gates[0].weight.shape[0])
    row_bytes = experts * hidden * 2
    if node is None:
        flat = torch.empty((n, row_bytes), dtype=torch.uint8)
    else:
        from sglang.srt.layers.moe.host_numa import allocate_bound

        flat = allocate_bound(n * row_bytes, [(node, 0, n)], row_bytes).view(n, row_bytes)
    weights = flat.view(torch.bfloat16).view(n, experts, hidden)
    bias = torch.empty((n, experts), dtype=torch.float32)
    for index, gate in enumerate(picked.gates):
        weights[index].copy_(gate.weight.detach().to(device="cpu", dtype=torch.bfloat16))
        bias[index].copy_(gate.bias.detach().to(device="cpu", dtype=torch.float32))
    return PrefetchTables(picked.targets, weights, bias, picked.top_k)


@dataclass(frozen=True)
class SpecScoreRow:
    """One source row's GPU scoring (SGLANG_DSV41_RAM_PREFETCH_SCORER=gpu): its target row's live router gate and VRAM
    hot slots, which the scoring kernels' captured parameters point at."""

    weight: torch.Tensor  # [experts, hidden], bf16 or fp32 (router_fp32): the router's own parameter
    bias: torch.Tensor  # [experts] fp32
    target: int
    hot_slots: torch.Tensor  # int64: the target layer's VRAM slots' experts, -1 empty
    hot_capacity: int


def spec_score_rows(picked: PrefetchTargets, hot: Mapping[int, tuple]) -> dict:
    """The GPU scorer's per-row table: each source row whose target row's hot slots are known (``hot``: row -> (hot
    slots, capacity), filled as rows attach). Refuses a gate the kernels cannot read."""
    rows = {}
    for row, (target, index) in enumerate(picked.targets.tolist()):
        if target < 0 or target not in hot:
            continue
        gate = picked.gates[index]
        if gate.weight.dtype not in (torch.bfloat16, torch.float32):
            raise ValueError(
                f"RAM prefetch: the GPU scorer reads a bf16 or fp32 gate; row {target}'s is {gate.weight.dtype}"
            )
        if gate.bias.dtype != torch.float32:
            raise ValueError(f"RAM prefetch: the GPU scorer reads an fp32 bias; row {target}'s is {gate.bias.dtype}")
        hot_slots, capacity = hot[target]
        rows[row] = SpecScoreRow(gate.weight.detach(), gate.bias.detach(), int(target), hot_slots, int(capacity))
    return rows


@dataclass(frozen=True)
class GpuScorer:
    """The GPU scorer as the service started it: the targets and their live gates, and the candidate page the select
    kernel writes and the host reads."""

    picked: PrefetchTargets
    candidates: torch.Tensor
    per_token: int
    top_k_only: bool
```

- [ ] **Step 4: The device side in `expert_stream_transport.py`**

In `ExpertStreamDevice.__init__`, directly after `self.piece_runs = piece_runs.to(device).contiguous()`:

```python
        # The GPU scorer (enable_spec_scorer): its (top_k, per_token, top_k_only), the candidate page and the score
        # scratch; None when off.
        self._spec = None
        self.spec_candidates = None
        self.spec_scores = None
```

Directly after `set_row_cpu`, add:

```python
    def enable_spec_scorer(self, candidates: torch.Tensor, *, top_k: int, per_token: int, top_k_only: bool) -> None:
        """The GPU scorer (SGLANG_DSV41_RAM_PREFETCH_SCORER=gpu): ``candidates`` is the host's candidate page
        (``new_candidate_page``); scores go to an fp32 [cpu_tokens_max, experts] scratch allocated here, so this runs
        after ``enable_cpu_experts`` and before any graph captures ``spec_score``."""
        if self._spec is not None:
            raise RuntimeError("the GPU scorer is already enabled on this device")
        if (
            candidates.dtype != torch.uint8
            or candidates.device.type != "cpu"
            or not candidates.is_contiguous()
            or candidates.numel() != CAND_PAGE_BYTES
        ):
            raise ValueError(
                f"candidates must be a contiguous CPU uint8 tensor of {CAND_PAGE_BYTES} bytes (new_candidate_page)"
            )
        if torch.device(self.state.device).type == "cuda" and not candidates.is_pinned():
            raise ValueError("candidates must be pinned: the select kernel writes it through UVA")
        if self.experts > SPEC_SELECT_MAX_EXPERTS:
            raise ValueError(
                f"the GPU scorer's select kernel holds at most {SPEC_SELECT_MAX_EXPERTS} experts, not {self.experts}"
            )
        if not 1 <= top_k <= min(SPEC_DEPTH, self.experts):
            raise ValueError(f"top_k must be in 1..{min(SPEC_DEPTH, self.experts)}, got {top_k}")
        if not 1 <= per_token <= SPEC_DEPTH:
            raise ValueError(f"per_token must be in 1..{SPEC_DEPTH}, got {per_token}")
        self._spec = (int(top_k), int(per_token), bool(top_k_only))
        self.spec_candidates = candidates
        self.spec_scores = torch.empty(
            (self.cpu_tokens_max, self.experts), dtype=torch.float32, device=self.state.device
        )

    def spec_score(self, entry, x: Optional[torch.Tensor]) -> None:
        """Score the record this device just posted: layer T's MoE input ``x`` (bf16 [tokens, hidden] or [hidden], or
        None) against ``entry`` (a ``SpecScoreRow``), then publish its candidates to the record's slot. A record of no
        input or more than cpu_tokens_max tokens publishes count 0. Right after ``post``, on its stream."""
        if self._spec is None:
            raise RuntimeError("the GPU scorer is not enabled on this device (enable_spec_scorer)")
        self._check_row(entry.target)
        top_k, per_token, top_k_only = self._spec
        tokens = 0 if x is None else (int(x.shape[0]) if x.dim() == 2 else 1)
        module = self._kernels()
        if 1 <= tokens <= self.cpu_tokens_max:
            run_spec_score(x.reshape(tokens, -1), entry.weight, entry.bias, self.spec_scores, module=module)
        run_spec_select(
            self.spec_scores,
            tokens,
            top_k=top_k,
            per_token=per_token,
            top_k_only=top_k_only,
            hot_slots=entry.hot_slots,
            hot_capacity=entry.hot_capacity,
            ram_slot=self.map_bank["ram_slot"],
            target=entry.target,
            state=self.state,
            candidates=self.spec_candidates,
            module=module,
        )
```

- [ ] **Step 5: The row backend and the service in `exl3_ram_miss.py`**

In `Exl3RamMissRowBackend.__init__`, directly after `self._delivered: Optional[torch.Tensor] = None`:

```python
        # The GPU scorer (Exl3RamMissService._bind_spec_scores): whether this row has a target, and its SpecScoreRow
        # once the target row attached.
        self.spec_expected = False
        self.spec_score = None
```

In `Exl3RamMissRowBackend.post`, directly before `side = self.device_side`:

```python
        if self.spec_expected and self.spec_score is None:
            raise RuntimeError(
                f"RAM prefetch: row {self.row} posted before its target row attached; the GPU scorer's table is built "
                "at attach"
            )
```

and directly after the `side.post(...)` call (before `side.copy_engine_captured |= captured`):

```python
        if self.spec_expected:
            # Layer T+1's gate on this layer's input, published before C1 so the host can read it while this layer runs.
            side.spec_score(entry=self.spec_score, x=self.cpu_input[0] if self.cpu_input is not None else None)
```

In `Exl3RamMissService.__init__`, directly after `self._spec_share = 0`:

```python
        # The GPU scorer (SGLANG_DSV41_RAM_PREFETCH_SCORER=gpu), set at start; its per-row table is built as rows attach.
        self._spec_gpu = None
        self._spec_hot: dict[int, tuple] = {}
        self._row_backends: dict[int, object] = {}
```

In `ensure_started`, replace

```python
            if spec_share:
                self._enable_ram_prefetch(host, list(tables.layer_ids), numa, cpu_experts)
```

with

```python
            spec_gpu = (
                self._enable_ram_prefetch(host, list(tables.layer_ids), numa, cpu_experts, pin=pin)
                if spec_share
                else None
            )
```

then directly after `self._spec_share = spec_share` add `self._spec_gpu = spec_gpu`, and in the start log replace `f"share {spec_share}" if spec_share else "off",` with `f"share {spec_share}, {'GPU' if spec_gpu else 'CPU'} scorer" if spec_share else "off",`.

Replace `_enable_ram_prefetch` with:

```python
    @staticmethod
    def _enable_ram_prefetch(host, layer_ids, numa, cpu_experts, *, pin: bool = False):
        """Start the RAM prefetch over the pool reserved at start, each group's speculative thread on its plan's spare
        cores. CPU scorer: the router gates the model registered at load, copied to host memory once; returns None.
        GPU scorer: the targets and a candidate page (pinned when ``pin``) the host reads; returns the GpuScorer whose
        per-row table attach builds."""
        from sglang.srt.layers.moe.ram_prefetch import (
            SCORERS,
            GpuScorer,
            prefetch_tables,
            prefetch_targets,
            registered_gates,
        )

        if cpu_experts is None:
            raise RuntimeError("exl3 RAM miss: SGLANG_DSV41_RAM_PREFETCH needs SGLANG_DSV41_CPU_EXPERTS")
        scorer = envs.SGLANG_DSV41_RAM_PREFETCH_SCORER.get()
        if scorer not in SCORERS:
            raise ValueError(f"SGLANG_DSV41_RAM_PREFETCH_SCORER must be one of {SCORERS}, got {scorer!r}")
        per_token = envs.SGLANG_DSV41_RAM_PREFETCH_PER_TOKEN.get()
        per_layer = envs.SGLANG_DSV41_RAM_PREFETCH_PER_LAYER.get()
        top_k_only = envs.SGLANG_DSV41_RAM_PREFETCH_TOP_K_ONLY.get()
        hidden = cpu_experts.services[0].hidden
        cores = [list(plan.spec) for plan in numa.plans]
        if scorer == "gpu":
            picked = prefetch_targets(layer_ids, registered_gates(), hidden=hidden)
            candidates = new_candidate_page(pin=pin)
            host.enable_ram_prefetch(
                picked.targets,
                None,
                None,
                top_k=picked.top_k,
                per_token=per_token,
                per_layer=per_layer,
                cores=cores,
                top_k_only=top_k_only,
                candidates=candidates,
            )
            targets, result = picked.targets, GpuScorer(picked, candidates, per_token, top_k_only)
        else:
            # Off node 0, which the tier keeps near full (DSV41_REFERENCE.md section 33.12).
            node = numa.plans[-1].node if numa.nodes > 1 else None
            tables = prefetch_tables(layer_ids, registered_gates(), hidden=hidden, node=node)
            host.enable_ram_prefetch(
                tables.targets,
                tables.gates,
                tables.bias,
                top_k=tables.top_k,
                per_token=per_token,
                per_layer=per_layer,
                cores=cores,
                top_k_only=top_k_only,
            )
            targets, result = tables.targets, None
        logger.info(
            "exl3 RAM miss prefetch: %d of %d rows target the next layer, %s scorer, %d per token, %d per layer%s, "
            "cores %s",
            int((targets[:, 0] >= 0).sum()),
            len(layer_ids),
            scorer.upper(),
            per_token,
            per_layer,
            ", predicted top-k only" if top_k_only else "",
            [plan.spec for plan in numa.plans],
        )
        return result
```

Add `new_candidate_page` to this module's `expert_stream_transport` import (the import that brings `new_page` and `new_hot_page`).

In `attach`, inside the `if self.device_side is None:` block, directly after the `if self.cpu_experts is not None:` statement that calls `enable_cpu_experts` and `attach_device` (at that `if`'s indentation, before the bulk-map snapshot):

```python
            if self._spec_gpu is not None:
                spec = self._spec_gpu
                self.device_side.enable_spec_scorer(
                    spec.candidates, top_k=spec.picked.top_k, per_token=spec.per_token, top_k_only=spec.top_k_only
                )
```

and directly after the `streamer.row_backend = Exl3RamMissRowBackend(...)` statement:

```python
        if self._spec_gpu is not None:
            self._note_spec_row(row, streamer.row_backend)
```

Directly after `attach`, add:

```python
    def _note_spec_row(self, row: int, backend) -> None:
        """attach's half of the GPU scorer's table: records the row's VRAM hot slots and rebinds every attached row."""
        self._spec_hot[row] = (backend.hot_slots, backend.hot_capacity)
        self._row_backends[row] = backend
        self._bind_spec_scores()

    def _bind_spec_scores(self) -> None:
        """Gives each attached row with a target its SpecScoreRow once the target row attached. ModelRunner attaches
        every row (maybe_init_expert_hot_cache) before it captures the decode graph, which freezes the entries."""
        from sglang.srt.layers.moe.ram_prefetch import spec_score_rows

        picked = self._spec_gpu.picked
        rows = spec_score_rows(picked, self._spec_hot)
        for row, backend in self._row_backends.items():
            backend.spec_expected = int(picked.targets[row, 0]) >= 0
            backend.spec_score = rows.get(row)
```

In `_quarantine`, directly after the `if self.host is not None:` block:

```python
        if self._spec_gpu is not None:
            owned.append(self._spec_gpu.candidates)  # the select kernel writes it through UVA
```

and inside the `if self.device_side is not None:` block, directly after the `owned += [...]` list:

```python
            if side.spec_scores is not None:
                owned.append(side.spec_scores)
```

- [ ] **Step 6: Verify the table is built before capture**

Run:
```bash
cd /Users/dnikolaidis/.codex/worktrees/ram-prefetch-margin
grep -n "self.load_model()\|self.maybe_init_expert_hot_cache()\|def init_cuda_graphs\|self.init_cuda_graphs(" python/sglang/srt/model_executor/model_runner.py
grep -n "manager._attach_formats()" python/sglang/srt/layers/moe/expert_hot_cache.py
grep -n "register_moe_gates(" python/sglang/srt/models/deepseek_v4.py
```
Expected: `self.load_model()` (~672, which runs `deepseek_v4`'s `register_moe_gates`) before `self.maybe_init_expert_hot_cache()` (~681), whose `ExpertHotCacheManager.from_model` ends with `manager._attach_formats()` (~1557), which calls `attach` for every streamed layer; `init_cuda_graphs` is called later in `ModelRunner` initialization. Record the line numbers in the Task 6 write-up. If the order differs, stop: the guard in `post` will refuse the capture, and the wiring needs revisiting before going on.

- [ ] **Step 7: Commit, push, and run on divix01**

```bash
cd /Users/dnikolaidis/.codex/worktrees/ram-prefetch-margin
git add python/sglang/srt/layers/moe/ram_prefetch.py python/sglang/kernels/ops/moe/expert_stream_transport.py python/sglang/srt/layers/moe/exl3_ram_miss.py
git commit -m "Wire the GPU scorer: the per-row table at attach and the launch after the post

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01LxB2wQZv5EVNuBGUJJbkGk"
git push origin codex/dsv41-ram-prefetch-margin
ssh divix01 bash -s <<'REMOTE'
set -u
cd /data/models/slang/nvfp4-work/wt-ram-prefetch-margin && git fetch -q origin && git checkout -q --detach origin/codex/dsv41-ram-prefetch-margin && git log -1 --oneline
PY="env PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 taskset -c 0-63 /data/models/slang/.venv/bin/python -m pytest -q -p no:randomly"
$PY test/registered/unit/layers/moe/test_ram_prefetch_tables.py test/registered/unit/layers/moe/test_exl3_ram_miss_service.py 2>&1 | tail -3; echo "EXIT=${PIPESTATUS[0]}"
$PY test/registered/unit/kernels/test_exl3_ram_miss_attach_lanes.py test/registered/unit/kernels/test_exl3_ram_miss_device_args.py 2>&1 | tail -3; echo "EXIT=${PIPESTATUS[0]}"
flock /data/models/slang/nvfp4-work/cc-gpu.lock taskset -c 32-63 env PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 /data/models/slang/.venv/bin/python -m pytest -q -p no:randomly test/manual/dsv41/test_exl3_ram_prefetch_gpu_scorer_cuda.py test/manual/dsv41/test_exl3_lease_kernels_cuda.py 2>&1 | tail -3; echo "EXIT=${PIPESTATUS[0]}"
REMOTE
```
Expected: three `EXIT=0`.

- [ ] **Step 8: Mutant (divix01 worktree only, reverted)**

In `python/sglang/kernels/ops/moe/expert_stream_transport.py` `ExpertStreamDevice.spec_score`, replace `run_spec_score(x.reshape(tokens, -1), ...` with `run_spec_score(x.reshape(tokens, -1).clone(), ...`, run the GPU command of Step 7 with `-k captured`, confirm `test_a_captured_post_and_its_scoring_replay_the_reference_candidates` is red, `git checkout -- python/sglang/kernels/ops/moe/expert_stream_transport.py`, re-run, record green.

---

### Task 5: The A/B arms and the capture's scorer

**Files:**
- Modify: `analysis/dsv41-drive/dspark/both_cpu_ab.py` (`ARMS`, `REFERENCE`, `RAM_KEYS`, new `_ms_by_session` / `_paired_gain_pct`, `summarize`)
- Modify: `analysis/dsv41-drive/dspark/spec_margin_capture.py` (`server_env`, `--scorer`, `counter_summary`, `main`)
- Test: `test/registered/unit/scripts/test_both_cpu_ab.py`, `test/registered/unit/scripts/test_spec_margin_capture.py` (new)

**Interfaces:**
- Consumes: Task 1's option name; Task 2's `spec_late` core counter.
- Produces: arms `dspark-both-prefetch-gpu` (= `dspark-both-prefetch` with `SGLANG_DSV41_RAM_PREFETCH_SCORER=gpu`) and `dspark-both-prefetch-gpu-topk`; `dspark-both-prefetch` pins `SGLANG_DSV41_RAM_PREFETCH_SCORER=cpu`; `REFERENCE` maps both GPU arms to `dspark-both`; summary fields `spec_in_flight_at_use`, `paired_gain_pct_vs_reference` (session_id -> %), `paired_gain_pct_median`; `spec_margin_capture.server_env(out_dir, top_k_only, scorer="cpu")`, `counter_summary(log_path) -> dict`, `--scorer {cpu,gpu}` writing `counters.json`.

- [ ] **Step 1: Write the failing tests**

In `test/registered/unit/scripts/test_both_cpu_ab.py`, in `test_the_prefetch_arm_is_dspark_both_with_the_prefetch_on_and_its_a_states_it_off`, add `"SGLANG_DSV41_RAM_PREFETCH_SCORER": "cpu",` to the `options` dict. Append:

```python
def test_the_gpu_arm_differs_from_the_prefetch_arm_in_the_scorer_alone(monkeypatch):
    ab = _ab()
    b, g = ab.ARMS["dspark-both-prefetch"][0], ab.ARMS["dspark-both-prefetch-gpu"][0]
    assert (b["SGLANG_DSV41_RAM_PREFETCH_SCORER"], g["SGLANG_DSV41_RAM_PREFETCH_SCORER"]) == ("cpu", "gpu")
    assert {k: v for k, v in g.items() if k != "SGLANG_DSV41_RAM_PREFETCH_SCORER"} == {
        k: v for k, v in b.items() if k != "SGLANG_DSV41_RAM_PREFETCH_SCORER"
    }
    assert ab.ARMS["dspark-both-prefetch-gpu"][1] is True and ab.REFERENCE["dspark-both-prefetch-gpu"] == "dspark-both"
    t = ab.ARMS["dspark-both-prefetch-gpu-topk"][0]
    assert {k: v for k, v in t.items() if k != "SGLANG_DSV41_RAM_PREFETCH_TOP_K_ONLY"} == {
        k: v for k, v in g.items() if k != "SGLANG_DSV41_RAM_PREFETCH_TOP_K_ONLY"
    }
    assert t["SGLANG_DSV41_RAM_PREFETCH_TOP_K_ONLY"] == "1"
    monkeypatch.setenv("SGLANG_DSV41_RAM_PREFETCH_SCORER", "cpu")
    assert ab._overrides("dspark-both-prefetch-gpu", "/out")["SGLANG_DSV41_RAM_PREFETCH_SCORER"] == "gpu"


def test_the_driver_runs_the_arms_in_the_order_given(monkeypatch, tmp_path):
    """The reversed A/B (B first) relies on it."""
    ab = _ab()
    ran = []
    monkeypatch.setattr(ab, "run_timed", lambda arm, out: ran.append(("timed", arm)) or 0)
    monkeypatch.setattr(ab, "run_probe", lambda arm, out: ran.append(("probe", arm)) or 0)
    monkeypatch.setattr(ab, "summarize", lambda out: {})
    monkeypatch.setattr(ab.sys, "argv", ["both_cpu_ab.py", str(tmp_path), "dspark-both-prefetch-gpu", "dspark-both"])
    ab.main()
    assert ran == [
        ("timed", "dspark-both-prefetch-gpu"),
        ("probe", "dspark-both-prefetch-gpu"),
        ("timed", "dspark-both"),
        ("probe", "dspark-both"),
    ]


def _session_rows(root, arm, ms):
    run = root / "servers" / arm / "run-1"
    run.mkdir(parents=True)
    (run / "results.jsonl").write_text(
        "".join(
            json.dumps({"session_id": f"s{i}", "decode_tokens_per_sec": 1000.0 / v, "completion_tokens": 10,
                        "spec_tokens_details": {}}) + "\n"
            for i, v in enumerate(ms)
        )
    )
    counters = {"rows_read": 100, "spec_used": 50, "spec_promoted": 5, "spec_late": 3}
    (run / "server.log").write_text("exl3 RAM miss thread counters " + json.dumps(counters) + "\n")


def test_the_summary_pairs_sessions_and_reports_reads_still_in_flight(tmp_path):
    ab = _ab()
    _session_rows(tmp_path, "dspark-both", [100.0, 200.0, 50.0])
    _session_rows(tmp_path, "dspark-both-prefetch-gpu", [90.0, 210.0, 40.0])
    g = ab.summarize(str(tmp_path))["dspark-both-prefetch-gpu"]
    assert g["paired_gain_pct_vs_reference"] == pytest.approx({"s0": 10.0, "s1": -5.0, "s2": 20.0})
    assert g["paired_gain_pct_median"] == pytest.approx(10.0)
    assert g["spec_in_flight_at_use"] == pytest.approx(0.1) and g["ram"]["spec_late"] == 3
```

Add `import pytest` to that file's imports if absent.

Create `test/registered/unit/scripts/test_spec_margin_capture.py`:

```python
"""The prefetch margin capture's server env and its counters summary (CPU)."""

import importlib.util
import json
import os

import pytest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))


def _capture():
    spec = importlib.util.spec_from_file_location(
        "spec_margin_capture", os.path.join(ROOT, "analysis", "dsv41-drive", "dspark", "spec_margin_capture.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_capture_scores_where_asked_and_otherwise_runs_the_prefetch_arm():
    cap = _capture()
    cpu, gpu = cap.server_env("/out", False), cap.server_env("/out", False, "gpu")
    assert (cpu["SGLANG_DSV41_RAM_PREFETCH_SCORER"], gpu["SGLANG_DSV41_RAM_PREFETCH_SCORER"]) == ("cpu", "gpu")
    assert {k: v for k, v in gpu.items() if k != "SGLANG_DSV41_RAM_PREFETCH_SCORER"} == {
        k: v for k, v in cpu.items() if k != "SGLANG_DSV41_RAM_PREFETCH_SCORER"
    }
    assert gpu["SGLANG_DSV41_EXPERT_JOB_TRACE_PREFIX"] == "/out/events"


def test_the_counter_summary_reads_the_last_counters_line(tmp_path):
    cap = _capture()
    log = tmp_path / "server.log"
    first = {"spec_scored": 1, "spec_score_ns": 1, "spec_used": 1, "spec_promoted": 1}
    last = {"spec_scored": 10, "spec_score_ns": 50_000, "spec_used": 40, "spec_promoted": 4, "spec_late": 2}
    log.write_text("".join("exl3 RAM miss thread counters " + json.dumps(c) + "\n" for c in (first, last)))
    s = cap.counter_summary(str(log))
    assert s["wait_or_score_us_per_record"] == pytest.approx(5.0)
    assert s["in_flight_at_use"] == pytest.approx(0.1) and s["spec_late"] == 2
    assert cap.counter_summary(str(tmp_path / "missing.log")) == {}
```

- [ ] **Step 2: Commit the failing tests, push, and run them on divix01**

```bash
cd /Users/dnikolaidis/.codex/worktrees/ram-prefetch-margin
git add test/registered/unit/scripts/test_both_cpu_ab.py test/registered/unit/scripts/test_spec_margin_capture.py
git commit -m "Test the GPU-scorer A/B arms, the paired summary and the capture's scorer

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01LxB2wQZv5EVNuBGUJJbkGk"
git push origin codex/dsv41-ram-prefetch-margin
ssh divix01 bash -s <<'REMOTE'
set -u
cd /data/models/slang/nvfp4-work/wt-ram-prefetch-margin && git fetch -q origin && git checkout -q --detach origin/codex/dsv41-ram-prefetch-margin && git log -1 --oneline
env PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 taskset -c 0-63 /data/models/slang/.venv/bin/python -m pytest -q -p no:randomly test/registered/unit/scripts/test_both_cpu_ab.py test/registered/unit/scripts/test_spec_margin_capture.py 2>&1 | tail -10; echo "EXIT=${PIPESTATUS[0]}"
REMOTE
```
Expected: `EXIT=1`: `KeyError: 'dspark-both-prefetch-gpu'`, `KeyError: 'SGLANG_DSV41_RAM_PREFETCH_SCORER'`, `TypeError: server_env() takes 2 positional arguments but 3 were given`, `AttributeError: ... 'counter_summary'`. The order test passes already (it pins existing behavior).

- [ ] **Step 3: The driver**

In `analysis/dsv41-drive/dspark/both_cpu_ab.py`, in `ARMS["dspark-both-prefetch"]`'s dict, directly after `"SGLANG_DSV41_RAM_PREFETCH_TOP_K_ONLY": "0",` add `"SGLANG_DSV41_RAM_PREFETCH_SCORER": "cpu",`. Directly after the `ARMS["dspark-both-prefetch-topk"]` assignment add:

```python
# The same arm scored on the GPU after each layer's post (spec 2026-10-09-dsv41-ram-prefetch-gpu-scorer-design).
ARMS["dspark-both-prefetch-gpu"] = (
    {**ARMS["dspark-both-prefetch"][0], "SGLANG_DSV41_RAM_PREFETCH_SCORER": "gpu"},
    True,
)
ARMS["dspark-both-prefetch-gpu-topk"] = (
    {**ARMS["dspark-both-prefetch-gpu"][0], "SGLANG_DSV41_RAM_PREFETCH_TOP_K_ONLY": "1"},
    True,
)
```

Replace `REFERENCE = {...}` with:

```python
REFERENCE = {
    "dspark-both-prefetch": "dspark-both",
    "dspark-both-prefetch-topk": "dspark-both",
    "dspark-both-prefetch-gpu": "dspark-both",
    "dspark-both-prefetch-gpu-topk": "dspark-both",
}
```

In `RAM_KEYS`, directly after `"spec_delayed",` add `"spec_late",`.

Directly after `_results`, add:

```python
def _ms_by_session(out: str, arm: str) -> dict:
    return {
        r["session_id"]: 1000.0 / r["decode_tokens_per_sec"]
        for r in _results(out, arm)
        if r.get("session_id") is not None and r.get("decode_tokens_per_sec")
    }


def _paired_gain_pct(out: str, reference: str, arm: str):
    """Per session (by session_id), the arm's ms/token gain over its reference's, in % of the reference's; None when
    the two share no session."""
    try:
        a, b = _ms_by_session(out, reference), _ms_by_session(out, arm)
    except FileNotFoundError:
        return None
    return {k: 100.0 * (a[k] - b[k]) / a[k] for k in sorted(set(a) & set(b))} or None
```

In `summarize`, inside `if counters:` directly after `entry["ram"]["scope"] = ...`:

```python
            if counters.get("spec_used"):
                # Used reads still in flight at their demand (the GPU scorer's success criterion: under 15%).
                entry["spec_in_flight_at_use"] = counters.get("spec_promoted", 0) / counters["spec_used"]
```

and directly before `summary[arm] = entry`:

```python
        if reference:
            paired = _paired_gain_pct(out, reference, arm)
            if paired:
                entry["paired_gain_pct_vs_reference"] = paired
                entry["paired_gain_pct_median"] = statistics.median(paired.values())
```

- [ ] **Step 4: The capture**

In `analysis/dsv41-drive/dspark/spec_margin_capture.py`: add `import json` to the imports; update the module docstring's usage line to `Usage: spec_margin_capture.py OUT_DIR [--prompts 8] [--max-tokens 128] [--top-k-only] [--scorer cpu|gpu]`; replace `server_env` with:

```python
def server_env(out_dir: str, top_k_only: bool, scorer: str = "cpu") -> dict:
    overrides, _ = both_cpu_ab.ARMS["dspark-both-prefetch"]
    return arm_env.arm_env({
        **overrides,
        "SGLANG_DSV41_RAM_PREFETCH_TOP_K_ONLY": "1" if top_k_only else "0",
        "SGLANG_DSV41_RAM_PREFETCH_SCORER": scorer,
        "SGLANG_DSV41_EXPERT_JOB_TRACE_PREFIX": os.path.join(out_dir, "events"),
        "SGLANG_DSV41_EXPERT_JOB_TRACE_CAPACITY": "524288",
        "SGLANG_MOE_HOT_METRICS_FILE": os.path.join(out_dir, "metrics.jsonl"),
    })


def counter_summary(log_path: str) -> dict:
    """From the server's last counters line: the speculative thread's time per record (the CPU scorer's scoring, the GPU
    scorer's wait for the slot) and the share of used reads still in flight at their demand."""
    try:
        with open(log_path, errors="replace") as f:
            lines = [line for line in f if both_cpu_ab.COUNTER_MARKER in line]
    except FileNotFoundError:
        return {}
    if not lines:
        return {}
    c = json.loads(lines[-1].split(both_cpu_ab.COUNTER_MARKER, 1)[1])
    keys = ("spec_scored", "spec_score_ns", "spec_issued", "spec_landed", "spec_used", "spec_promoted",
            "spec_dropped", "spec_late")
    out = {k: c.get(k) for k in keys}
    if c.get("spec_scored"):
        out["wait_or_score_us_per_record"] = c["spec_score_ns"] / c["spec_scored"] / 1000
    if c.get("spec_used"):
        out["in_flight_at_use"] = c.get("spec_promoted", 0) / c["spec_used"]
    return out
```

In `main`: add `p.add_argument("--scorer", choices=("cpu", "gpu"), default="cpu")` after `--top-k-only`; change `server_env(a.out_dir, a.top_k_only)` to `server_env(a.out_dir, a.top_k_only, a.scorer)`; replace the final `return subprocess.run([... spec_margin.py ...]).returncode` with:

```python
    rc = subprocess.run(
        [sys.executable, os.path.join(HERE, "spec_margin.py"), os.path.join(a.out_dir, "events.*.jsonl"),
         "--json", os.path.join(a.out_dir, "spec_margin.json")]
    ).returncode
    summary = counter_summary(os.path.join(a.out_dir, "server.log"))
    with open(os.path.join(a.out_dir, "counters.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print("counters", json.dumps(summary))
    return rc
```

- [ ] **Step 5: Commit, push, and run on divix01**

```bash
cd /Users/dnikolaidis/.codex/worktrees/ram-prefetch-margin
git add analysis/dsv41-drive/dspark/both_cpu_ab.py analysis/dsv41-drive/dspark/spec_margin_capture.py
git commit -m "Add the GPU-scorer A/B arms, a paired summary and the capture's --scorer

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01LxB2wQZv5EVNuBGUJJbkGk"
git push origin codex/dsv41-ram-prefetch-margin
ssh divix01 bash -s <<'REMOTE'
set -u
cd /data/models/slang/nvfp4-work/wt-ram-prefetch-margin && git fetch -q origin && git checkout -q --detach origin/codex/dsv41-ram-prefetch-margin && git log -1 --oneline
env PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 taskset -c 0-63 /data/models/slang/.venv/bin/python -m pytest -q -p no:randomly test/registered/unit/scripts/test_both_cpu_ab.py test/registered/unit/scripts/test_spec_margin_capture.py test/registered/unit/scripts/test_spec_margin.py 2>&1 | tail -3; echo "EXIT=${PIPESTATUS[0]}"
REMOTE
```
Expected: `EXIT=0`.

---

### Task 6: Regression and measurement on divix01

**Files:**
- Modify: `DSV41_REFERENCE.md` (new §33.15 directly after §33.14, before `## Sources`)
- Untracked on divix01 only: `benchmarks/dsv41_baseline/generations.json` in the divix01 worktree

**Interfaces:**
- Consumes: everything above; arms `dspark-both`, `dspark-both-prefetch-gpu`; `spec_margin_capture.py --scorer gpu`; `counter_summary`.
- Produces: the results directories under `/data/models/slang/nvfp4-work/ram-prefetch/` and `/mnt/nvme1/dsv41-nsys/`, and `DSV41_REFERENCE.md` §33.15.

- [ ] **Step 1: The regression selection, each directory in its own run**

```bash
ssh divix01 bash -s <<'REMOTE'
set -u
cd /data/models/slang/nvfp4-work/wt-ram-prefetch-margin && git fetch -q origin && git checkout -q --detach origin/codex/dsv41-ram-prefetch-margin && git log -1 --oneline
PY="env PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 taskset -c 0-63 /data/models/slang/.venv/bin/python -m pytest -q -p no:randomly"
$PY test/registered/unit/kernels/test_*exl3*.py test/registered/unit/kernels/test_*expert*.py 2>&1 | tail -3; echo "EXIT=${PIPESTATUS[0]}"
$PY test/registered/unit/layers/moe/test_ram_prefetch_tables.py test/registered/unit/layers/moe/test_exl3_ram_miss_service.py test/registered/unit/layers/moe/test_threading_config.py 2>&1 | tail -3; echo "EXIT=${PIPESTATUS[0]}"
$PY test/registered/unit/test_expert_stream_requirements_exl3.py 2>&1 | tail -3; echo "EXIT=${PIPESTATUS[0]}"
$PY test/registered/unit/scripts/test_both_cpu_ab.py test/registered/unit/scripts/test_spec_margin.py test/registered/unit/scripts/test_spec_margin_capture.py 2>&1 | tail -3; echo "EXIT=${PIPESTATUS[0]}"
flock /data/models/slang/nvfp4-work/cc-gpu.lock taskset -c 32-63 env PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 /data/models/slang/.venv/bin/python -m pytest -q -p no:randomly test/manual/dsv41/test_exl3_ram_prefetch_gpu_scorer_cuda.py test/manual/dsv41/test_exl3_lease_kernels_cuda.py test/manual/dsv41/test_exl3_lease_ordering_cuda.py test/manual/dsv41/test_exl3_copy_engine_cuda.py 2>&1 | tail -3; echo "EXIT=${PIPESTATUS[0]}"
REMOTE
```
Expected: five `EXIT=0`. A red outside this plan's files: run the same selection at the merge-base (`git merge-base origin/codex/dsv41-ram-prefetch-margin 6818b92db6`, in a second private worktree) and diff the counts before calling it a regression (§33.14 records one pre-existing red, the clock-guard test). Record each selection with its pass count.

- [ ] **Step 2: Check the GPU is free**

```bash
ssh divix01 nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
```
Expected: no line. Production is stopped and stays stopped: never restart it. If a process is listed, do not touch it; stop and report.

- [ ] **Step 3: Register the python tree's generation**

```bash
ssh divix01 bash -s <<'REMOTE'
set -u
WT=/data/models/slang/nvfp4-work/wt-ram-prefetch-margin
cd $WT && git log -1 --oneline
TREE=$(git rev-parse HEAD:python); SHA=$(git rev-parse --short=10 HEAD)
cd $WT/benchmarks/dsv41_baseline && /data/models/slang/.venv/bin/python -c "import generations, sys; generations.register(sys.argv[1], sys.argv[2]); print(generations.check_registered(sys.argv[1]))" "$TREE" "ram-prefetch-gpu-scorer-$SHA"
git -C $WT status --short
REMOTE
```
Expected: `ram-prefetch-gpu-scorer-<sha>`, and `git status` showing at most `?? benchmarks/dsv41_baseline/generations.json` (untracked; `run_arm.sh`'s preflight counts tracked changes only).

- [ ] **Step 4: The kernels' cost: a node-mode trace of one session**

```bash
ssh divix01 bash -s <<'REMOTE'
set -u
WT=/data/models/slang/nvfp4-work/wt-ram-prefetch-margin
cd $WT && git log -1 --oneline
STAMP=$(date +%Y%m%d-%H%M%S)
OUT=/data/models/slang/nvfp4-work/ram-prefetch/gpu-nsys-$STAMP; mkdir -p $OUT/tmp; echo "OUT=$OUT"
NSYS_OUT=/mnt/nvme1/dsv41-nsys/gpu-nsys-$STAMP; mkdir -p $NSYS_OUT /mnt/nvme1/nsys-tmp; echo "NSYS_OUT=$NSYS_OUT"
git rev-parse HEAD > $OUT/commit.txt
export TMPDIR=$OUT/tmp NSYS_TMPDIR=/mnt/nvme1/nsys-tmp SGLANG_JIT_CACHE_DIR=/data/models/slang/nvfp4-work/ram-prefetch/jit-cache SGLANG_EXL3_BUILD_DIR=/data/models/slang/nvfp4-work/ram-prefetch/exl3-build
flock /data/models/slang/nvfp4-work/rowimg-disk.lock taskset -c 0-63 /data/models/slang/.venv/bin/python - "$OUT" "$NSYS_OUT" <<'PY'
import os, shlex, subprocess, sys
sys.path.insert(0, "analysis/dsv41-drive/dspark")
import both_cpu_ab as ab
out, nsys_out = sys.argv[1], sys.argv[2]
arm = "dspark-both-prefetch-gpu"
env = os.environ | {
    "DSV41_RUN_ROOT": out,
    "DSV41_SESSION_INDICES": "0",
    "DSV41_EXTRA_SERVER_ARGS": shlex.join(ab.arm_env.DSPARK_ARGV),
    "DSV41_HEALTH_TIMEOUT_S": str(ab._health_timeout_s(True)),
    "DSV41_MEM_FRACTION_STATIC": ab.arm_env.DSPARK_MEM_FRACTION_STATIC,
    "NSYS_TRACE": "1",
    "NSYS_CUDA_GRAPH_TRACE": "node",
    "NSYS_GPU_METRICS": "0",
    "NSYS_OUT_DIR": nsys_out,
}
cmd = [os.path.join(ab.REPO, "benchmarks", "dsv41_baseline", "run_arm.sh"), arm, str(ab.PORT)]
cmd += [f"{k}={v}" for k, v in ab._overrides(arm, out).items()]
sys.exit(subprocess.run(cmd, env=env, cwd=ab.REPO).returncode)
PY
echo "RC=$?"
ls -la $NSYS_OUT
REMOTE
```
Expected: `RC=0`, one `.nsys-rep` of more than 10 MB under `$NSYS_OUT`. This run's ms/token is not a measurement (node mode inflates each graph launch): read only the two kernels from it.

Then:
```bash
ssh divix01 bash -s <<'REMOTE'
set -u
NSYS_OUT=$(ls -d /mnt/nvme1/dsv41-nsys/gpu-nsys-* | tail -1); REP=$(ls $NSYS_OUT/*.nsys-rep | grep -v -- -pcie | head -1); echo "REP=$REP"
NSYS=$(command -v nsys || ls /opt/nvidia/nsight-systems/*/bin/nsys | tail -1)
export NSYS_TMPDIR=/mnt/nvme1/nsys-tmp
taskset -c 0-63 "$NSYS" stats --report cuda_gpu_kern_sum --format csv --output - "$REP" 2>/dev/null | grep -E 'Time \(%\)|exl3_ram_prefetch_(score|select)_kernel|exl3_ram_miss_post_kernel' | tee $NSYS_OUT/spec-kernels.csv
REMOTE
```
Expected: the header, one row per score instantiation that ran (`bf16` for DSV4), one select row, and the post's row; the score and select `Instances` equal (each record of a row with a target), no more than the post's. The cost per layer is the score row's `Avg (ns)` plus the select row's `Avg (ns)`; the success criterion is <= 100,000 ns. If it is not met, record it; the spec's fallback (a side-stream fork) is a separate change.

- [ ] **Step 5: The instrumented capture with the GPU scorer**

```bash
ssh divix01 bash -s <<'REMOTE'
set -u
WT=/data/models/slang/nvfp4-work/wt-ram-prefetch-margin
cd $WT && git log -1 --oneline
OUT=/data/models/slang/nvfp4-work/ram-prefetch/margin-gpu-$(date +%Y%m%d-%H%M%S); mkdir -p $OUT/tmp; echo "OUT=$OUT"
git rev-parse HEAD > $OUT/commit.txt
export TMPDIR=$OUT/tmp SGLANG_JIT_CACHE_DIR=/data/models/slang/nvfp4-work/ram-prefetch/jit-cache SGLANG_EXL3_BUILD_DIR=/data/models/slang/nvfp4-work/ram-prefetch/exl3-build
taskset -c 0-63 /data/models/slang/.venv/bin/python analysis/dsv41-drive/dspark/spec_margin_capture.py $OUT --scorer gpu > $OUT/capture.log 2>&1; echo "RC=$?"
tail -40 $OUT/capture.log
/data/models/slang/.venv/bin/python - "$OUT" <<'PY'
import json, sys
sys.path.insert(0, "analysis/dsv41-drive/dspark")
import spec_margin_capture as cap
cpu = "/data/models/slang/nvfp4-work/ram-prefetch/margin-20261009-002539"
print("cpu", json.dumps(cap.counter_summary(cpu + "/server.log")))
print("gpu", json.dumps(cap.counter_summary(sys.argv[1] + "/server.log")))
PY
REMOTE
```
Expected: `RC=0`; the spec_margin tables (all, inside/outside top-k, by rank, by margin) and the timing lines (read, lead, slack) for the GPU run; a `cpu` line with `in_flight_at_use` about 0.42 (992 / 2373) and `wait_or_score_us_per_record` about 1820; a `gpu` line whose `in_flight_at_use` is the success criterion (< 0.15) and whose `wait_or_score_us_per_record` is the host's wait for the slot. The capture takes both locks itself (disk, then GPU). Its timings are the instrumented build's, not a throughput number.

- [ ] **Step 6: The A/B, the arm order reversed**

Confirm the order first:
```bash
grep -n "for arm in arms" /Users/dnikolaidis/.codex/worktrees/ram-prefetch-margin/analysis/dsv41-drive/dspark/both_cpu_ab.py
```
Expected: one line in `main`, iterating `sys.argv[2:]` in order (pinned by `test_the_driver_runs_the_arms_in_the_order_given`).

```bash
ssh divix01 bash -s <<'REMOTE'
set -u
WT=/data/models/slang/nvfp4-work/wt-ram-prefetch-margin
cd $WT && git log -1 --oneline
OUT=/data/models/slang/nvfp4-work/ram-prefetch/ab-gpu-$(date +%Y%m%d-%H%M%S); mkdir -p $OUT/tmp; echo "OUT=$OUT"
git rev-parse HEAD > $OUT/commit.txt
export TMPDIR=$OUT/tmp SGLANG_JIT_CACHE_DIR=/data/models/slang/nvfp4-work/ram-prefetch/jit-cache SGLANG_EXL3_BUILD_DIR=/data/models/slang/nvfp4-work/ram-prefetch/exl3-build
nohup flock /data/models/slang/nvfp4-work/rowimg-disk.lock taskset -c 0-63 /data/models/slang/.venv/bin/python analysis/dsv41-drive/dspark/both_cpu_ab.py $OUT dspark-both-prefetch-gpu dspark-both > $OUT/ab.log 2>&1 &
echo "pid=$!"
REMOTE
```
Expected: `OUT=...` and a pid. Poll every 10 minutes with `ssh divix01 "tail -4 <OUT>/ab.log"` until the summary JSON prints. Expected in order: `dspark-both-prefetch-gpu run_timed: rc=0`, `dspark-both-prefetch-gpu run_probe: rc=0`, `dspark-both run_timed: rc=0`, `dspark-both run_probe: rc=0`, then the summary. Any nonzero rc ends the A/B: rename `$OUT` to `$OUT-failed-<reason>`, keep it, and report.

The optional top-k-only variant runs only if the owner asks, the same way with `dspark-both-prefetch-gpu-topk dspark-both` in an `ab-gpu-topk-*` directory.

- [ ] **Step 7: Provenance and the verdict**

```bash
ssh divix01 bash -s <<'REMOTE'
set -u
OUT=$(ls -d /data/models/slang/nvfp4-work/ram-prefetch/ab-gpu-2* | tail -1); echo "OUT=$OUT"
find /data/models/slang/nvfp4-work/ram-prefetch/jit-cache -name '*.so' \( -path '*expert_stream_host*prod*' -o -path '*expert_stream_exl3*' \) -exec sha256sum {} + > $OUT/modules.sha256
find /data/models/slang/nvfp4-work/ram-prefetch/exl3-build -name '*.so' -exec sha256sum {} + > $OUT/exl3-ext.sha256
for arm in dspark-both-prefetch-gpu dspark-both; do run=$(ls -d $OUT/servers/$arm/run-* | tail -1); grep -o '"SGLANG_DSV41_RAM_PREFETCH[A-Z_]*": *"[^"]*"' $run/server-env-actual.json | tr '\n' ' '; echo; done
cd /data/models/slang/nvfp4-work/wt-ram-prefetch-margin/benchmarks/dsv41_baseline && /data/models/slang/.venv/bin/python paired.py $(ls -d $OUT/servers/dspark-both/run-* | tail -1) $(ls -d $OUT/servers/dspark-both-prefetch-gpu/run-* | tail -1) > $OUT/paired.txt 2>&1; echo "paired rc=$?"; tail -5 $OUT/paired.txt
/data/models/slang/.venv/bin/python - "$OUT" <<'PY'
import json, sys
s = json.load(open(sys.argv[1] + "/summary.json"))
a, b = s["dspark-both"], s["dspark-both-prefetch-gpu"]
gain = a["ms_per_token_median"] - b["ms_per_token_median"]
paired = b.get("paired_gain_pct_vs_reference") or {}
text = b.get("text_vs_reference") or {}
report = {
    "A_ms_per_token": a["ms_per_token_median"],
    "B_ms_per_token": b["ms_per_token_median"],
    "gain_ms": gain,
    "gain_pct": 100 * gain / a["ms_per_token_median"],
    "paired_gain_pct": paired,
    "paired_gain_pct_median": b.get("paired_gain_pct_median"),
    "sessions_faster": sum(v > 0 for v in paired.values()),
    "A_accept_length": a["accept_length"],
    "B_accept_length": b["accept_length"],
    "text_B_vs_A": text,
    "B_in_flight_at_use": b.get("spec_in_flight_at_use"),
    "B_ram": b.get("ram"),
    "rows_per_token_lifetime": [a.get("ram_rows_per_token_lifetime"), b.get("ram_rows_per_token_lifetime")],
}
print(json.dumps(report, indent=2))
accept = gain > 0 and (b.get("paired_gain_pct_median") or 0) > 0 and text.get("pass") is True
print("ACCEPT" if accept else "REJECT")
json.dump({**report, "verdict": "accept" if accept else "reject"}, open(sys.argv[1] + "/verdict.json", "w"), indent=2)
PY
REMOTE
```
Expected: at least one module hash per file; the A's env shows `SGLANG_DSV41_RAM_PREFETCH": "0"`, the B's `"1"` and `SGLANG_DSV41_RAM_PREFETCH_SCORER": "gpu"`; `paired.txt` holds the sign test or the refusal (tenancy, clock profile), recorded either way; the report and `ACCEPT` or `REJECT` by the spec's rule (faster by the median and by the paired median, text within the near-tie band). `B_in_flight_at_use` against 15% is the read-timing criterion.

- [ ] **Step 8: Hand the GPU back**

Tell the lead: the GPU is free, production is still stopped (not restarted by this work), and give the `OUT` paths of the trace, the capture and the A/B.

- [ ] **Step 9: Write §33.15 and commit**

Render the A/B table on divix01:
```bash
ssh divix01 bash -s <<'REMOTE'
OUT=$(ls -d /data/models/slang/nvfp4-work/ram-prefetch/ab-gpu-2* | tail -1)
/data/models/slang/.venv/bin/python - "$OUT" <<'PY'
import json, sys
v = json.load(open(sys.argv[1] + "/verdict.json"))
r = v["B_ram"]
print("| arm | ms/token (median) | accept length | RAM rows / token (lifetime) |")
print("|---|---:|---:|---:|")
print(f"| `dspark-both` (A) | {v['A_ms_per_token']:.2f} | {v['A_accept_length']:.2f} | {v['rows_per_token_lifetime'][0]:.1f} |")
print(f"| `dspark-both-prefetch-gpu` (B) | {v['B_ms_per_token']:.2f} | {v['B_accept_length']:.2f} | {v['rows_per_token_lifetime'][1]:.1f} |")
print()
print("Paired per-session gain (A -> B, % of A): " + ", ".join(f"{k} {g:+.1f}" for k, g in v["paired_gain_pct"].items())
      + f"; median {v['paired_gain_pct_median']:+.1f}, {v['sessions_faster']} of {len(v['paired_gain_pct'])} faster.")
print(f"Median gain {v['gain_ms']:.2f} ms/token ({v['gain_pct']:.1f}%); text B vs A: pass={v['text_B_vs_A'].get('pass')}; "
      f"verdict: {v['verdict']}.")
print(f"B's speculative rows (lifetime): issued {r['spec_issued']}, landed {r['spec_landed']}, used {r['spec_used']}, "
      f"promoted {r['spec_promoted']} ({v['B_in_flight_at_use']:.0%} of used), late {r['spec_late']}, dropped "
      f"{r['spec_dropped']}, failed {r['spec_failed']}, demand reads delayed {r['spec_delayed']}.")
PY
REMOTE
```

Add to `DSV41_REFERENCE.md`, directly after §33.14 (before `## Sources`), a section headed `### 33.15 NVMe-to-RAM prefetch scored on the GPU: cost, timing and the reversed A/B (YYYY-MM-DD)`, the date the A/B's (from its `ab-gpu-YYYYMMDD-HHMMSS` name), with these paragraphs in this order:
1. **Verdict**, one or two sentences: accepted or rejected by the spec's rule (median and paired median both faster, text within the band), with the median gain, the paired median and how many of 8 sessions were faster; and whether the kernels met <= 0.1 ms per layer and the reads met < 15% in flight.
2. **What runs**: one paragraph citing the spec and this plan: after each post of a row with a target, a score kernel and a select kernel inside the decode graph publish up to 8 candidates per record (GateScorer's ranking, past VRAM-hot and RAM-mapped experts) to a 16-slot pinned page; each group's speculative thread waits up to 200 us for the record's slot and reads the first `per_layer` that are still unmapped and not pooled, its own group's; the CPU scorer stays the default.
3. **Kernel cost**: the score and select rows of `spec-kernels.csv` (instances, average ns, their sum per layer), naming the node-mode one-session trace and why graph mode was not used (refused with the copy engine; a graph-mode kernel table omits the graph body).
4. **Read timing**: the `cpu` and `gpu` `counter_summary` lines side by side (wait or score per record, in flight at use, late) and the GPU capture's read, lead and slack quantiles and its by-rank table, against `margin-20261009-002539`.
5. **Result**: the rendered table and its paragraphs.
6. **Provenance**: the commit (`commit.txt`) and python tree with its generation label; the A/B, capture and trace paths; the module and EXL3 extension hashes; `paired.txt`'s outcome; the unit selections and pass counts from Step 1; the Step 6 order check; the attach-before-capture line numbers from Task 4 Step 6; the scope note (RAM counters are server lifetime).
7. **Pointers**: `expert_stream/spec_score.cuh`, `expert_stream/spec_candidates.h`, `ram_tier.h` (`serve_gpu_job`, `read_candidates`, `await_candidates`), `ram_prefetch.py` (`spec_score_rows`), `exl3_ram_miss.py` (`_bind_spec_scores`, the post); tests `test_exl3_ram_prefetch_gpu_host.py`, `test/manual/dsv41/test_exl3_ram_prefetch_gpu_scorer_cuda.py`; the arms `dspark-both-prefetch-gpu{,-topk}`.

Commit and push:
```bash
cd /Users/dnikolaidis/.codex/worktrees/ram-prefetch-margin
git add DSV41_REFERENCE.md
git commit -m "Record the GPU-scored RAM prefetch: kernel cost, read timing and the reversed A/B (DSV41_REFERENCE 33.15)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01LxB2wQZv5EVNuBGUJJbkGk"
git push origin codex/dsv41-ram-prefetch-margin
git status --short
```
Expected: the push succeeds; `git status --short` shows only files that were dirty before this plan started.
