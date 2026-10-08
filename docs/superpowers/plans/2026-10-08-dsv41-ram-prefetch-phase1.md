# NVMe-to-RAM Prefetch, Phase 1 (host-scored speculative reads) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Hide DSpark's forced NVMe misses by reading, one layer ahead, the next streamed layer's most likely missing expert into a private pool of RAM slots that the device never maps, and swapping it into the tier when a forced CPU miss asks for it.

**Architecture:** `RamTier` reserves `SPEC_SHARE` slots per row and NUMA group as a host-private pool (slot state `kSpec`), which no victim, admission or release path takes. Per group, a speculative thread is fed by its service thread over an SPSC ring with every record that staged a CPU input. It scores the next streamed layer's gate on that input (`sqrt(softplus(W x)) + b`, ranked as the Phase 0 replay ranked it, one layer-wide budget computed identically on both groups), and reads its group's picks through the group's existing `RowReader` under a turn lock that the demand read also takes. In `reserve_victims_locked` a forced CPU miss whose expert landed in the pool takes its victim as today, maps the pool slot instead, gives the victim slot to the pool, and submits its CPU job at once with no read.

**Tech Stack:** C++20 header-only host (`python/sglang/kernels/jit/csrc/moe/expert_stream/host/`, JIT-built ProdBuild/InstrBuild modules through TVM FFI), Python 3 (torch, pytest), the RAM-miss service (`exl3_ram_miss.py`), `ThreadingConfig`, the launch gate, the DSpark A/B driver (`analysis/dsv41-drive/dspark/both_cpu_ab.py`); all runs on divix01 under the run protocol.

**Spec:** `docs/superpowers/specs/2026-10-08-dsv41-ram-prefetch-design.md` (sections "Phase 1" as revised 2026-10-08, "Invariants", "Testing", "The A/B"). Phase 0 is done (`DSV41_REFERENCE.md` §33.13).

## Global Constraints

- Code is written on the laptop in `/Users/dnikolaidis/.codex/worktrees/ram-prefetch/sglang-nvfp4` (branch `codex/dsv41-ram-prefetch`), committed, pushed to `origin`, and run on divix01 only in the private worktree `/data/models/slang/nvfp4-work/wt-ram-prefetch`, updated with `git -C <wt> fetch origin && git -C <wt> checkout --detach origin/codex/dsv41-ram-prefetch`. Never rsync/scp a tree; never run in the production checkout (`.claude/rules/divix01-run-protocol.md`).
- Every test command is `PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 taskset -c 0-63 /data/models/slang/.venv/bin/python -m pytest <files> -q -p no:randomly`, with pytest's own status read from `${PIPESTATUS[0]}` (never the `tail` status). Run only the files a task names; record the exact selection next to any count quoted.
- Host tests run on the instrumented build in-process (the `kernels/` conftest makes `instr` the default); fail-stops run in a child through `spawn_child`/`run_host_script` (`python/sglang/test/dsv41_ram_miss_fixtures.py`), which warm the JIT modules first.
- The first run after any C++ change rebuilds every host JIT module (50-100 s each, serialized). A `TimeoutExpired` on that run is the compiler; check the build-dir mtimes before calling it a hang.
- Mutants are applied in the divix01 worktree only, run, then reverted with `git checkout -- <file>`; never committed. After reverting, re-run the same selection and record that it is green.
- GPU work holds `rowimg-disk.lock` then `cc-gpu.lock` (lock order), cores 64-71 stay free. Production must be stopped for the GPU runs: ask the owner first, never stop it without approval, and never restart it (hand it back to the owner).
- Env vars are `EnvField`s in `python/sglang/srt/environ.py`'s DSV41 block (`.claude/skills/env-var-conventions/SKILL.md`), read with `.get()`, overridden in tests with `.override()`. The four names are the spec's, verbatim: `SGLANG_DSV41_RAM_PREFETCH` (bool, default off), `SGLANG_DSV41_RAM_PREFETCH_PER_TOKEN` (1), `SGLANG_DSV41_RAM_PREFETCH_PER_LAYER` (1), `SGLANG_DSV41_RAM_PREFETCH_SPEC_SHARE` (2). Do not reuse any name in `_DEPRECATED_ENVS` (`SGLANG_DSV41_ENABLE_EXPERT_PREFETCH`, `SGLANG_DSV41_ENABLE_NATIVE_PREFETCH`).
- With the option off the request path is today's: no pool, no speculative thread, no turn lock, no extra clock read (`traced_clock_reads()` unchanged). Every new branch on the service path tests `pool_`/`spec_` for null first.
- The host's bounds, mirrored in Python: per token 1..12 (`GateScorer::kDepth`), per layer 1..8 (`GateScorer::kMaxPerLayer`), share 1..4 (`SpecPool::kMaxShare`); lookahead fixed at one layer; no margin floor.
- Comments state constraints, not narration (`.claude/rules/comment-style.md`): one or two lines, ASCII, no TODO without an owner. Exported C++ entities get Doxygen-style `///` or the file's existing `//` contract comments.
- Commit trailer, exactly: `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Stage files by name. Do not amend or rebase.

## Review Focus

1. A verify row whose staged token count is below the rows' capacity (padding tokens): the scorer must rank only the table's live count, or it reads rows for garbage routes. Pinned in Task 5 (`test_each_live_token_adds_its_own_pick`, the `live=1` case on a two-token row).
2. The speculative thread falling behind by more than its ring (64 jobs): the service must never wait for it; the excess is dropped and counted. Pinned in Task 5 (`test_a_full_ring_drops_jobs_and_never_blocks_the_service`).
3. A pool entry reclaimed for a new candidate (the oldest landed one, when no entry is empty): a later forced miss on its old expert must read the row, never take the entry's new bytes. Pinned in Task 5 (`test_a_reclaimed_entry_is_never_swapped_in_for_its_old_expert`).
4. A GPU miss (staging slot) on an expert the pool is still reading: the service must neither promote nor take the pool row; it reads the row into staging as today (after at most the one speculative row, at the reader's turn), and the pool row still lands. Pinned in Task 6 (`test_a_gpu_miss_on_a_row_still_reading_reads_into_staging_and_the_pool_row_lands`).
5. A record whose CPU lanes all belong to the other NUMA group: both groups must still score it (each group's ring gets the job), or the two groups' rankings diverge and the layer budget is broken. Pinned in Task 5 (`test_both_groups_rank_alike_and_each_reads_only_its_own_experts`, which triggers on expert 5, homed on group 1, and asserts both groups' `spec_pump` find a job).

## File Structure

| File | Responsibility |
|---|---|
| `python/sglang/kernels/jit/csrc/moe/expert_stream/host/ram_prefetch.h` (new) | `SpecPool` (entries, words, per-group mutex), `RamPrefetchConfig`, `SpecJob` |
| `python/sglang/kernels/jit/csrc/moe/expert_stream/host/gate_scorer.h` (new) | `GateScorer` (fp16/bf16 conversion, sqrt-softplus scoring, Phase 0 ranking), `check_gate_choice` |
| `.../host/tier_protocol.h` | `kSpec` slot state; the nine speculative counters |
| `.../host/ram_tier.h` | pool reservation and introspection, swap-on-use, compacted demand read, speculative state/threads, quiesce, test hooks |
| `.../host/ram_thread.h` | starting, quiescing and stopping the speculative threads; the watchdog over their busy episodes |
| `.../host/cpu_experts.h` | `config()` accessor (the staged-input geometry the scorer reads) |
| `.../host/ffi_exports.h`, `ffi_test_exports.h` | `reserve_spec_pool`, `spec_share`, `spec_pool`, `enable_ram_prefetch`; test-only `spec_place`, `spec_pump`, `inject_spec`; `score_gate` |
| `python/sglang/kernels/ops/moe/expert_stream_transport.py` | Python wrappers, `COUNTERS`/`CORE_COUNTERS`, `TEST_ONLY_EXPORTS`, `score_gate` |
| `python/sglang/srt/layers/moe/ram_prefetch.py` (new) | router-gate registry, `prefetch_tables` (per-row targets, host gate copy), bounds |
| `python/sglang/srt/environ.py` | the four options |
| `python/sglang/srt/arg_groups/expert_stream_requirements_exl3.py` | `_check_ram_prefetch` |
| `python/sglang/srt/layers/moe/cpu_experts/threading_config.py` | `NodePlan.spec` spare cores |
| `python/sglang/srt/layers/moe/exl3_ram_miss.py` | pool reservation, enabling, spill room |
| `python/sglang/srt/models/deepseek_v4.py` | registering the router gates after load |
| `python/sglang/test/dsv41_chain_sim.py` | `forced_from` on `post` |
| `python/sglang/test/dsv41_ram_prefetch_fixtures.py` (new) | the prefetch test rig |
| `analysis/dsv41-drive/dspark/both_cpu_ab.py`, `benchmarks/dsv41_baseline/arm_env.py` | the `dspark-both-prefetch` arm and its summary |
| Tests (new) | `test/registered/unit/kernels/test_exl3_ram_prefetch_{pool,swap,scorer,step,thread,events}.py`, `test/registered/unit/layers/moe/test_ram_prefetch_tables.py` |

## Task list

1. Options, launch gate, spare cores, spill room (Python only).
2. The `kSpec` pool: reservation, exclusion from every victim/admission/release path, introspection.
3. Swap-on-use in `reserve_victims_locked`, the compacted read, the immediate CPU job; the counters; `spec_place`.
4. The gate scorer and its torch reference.
5. The speculative step in pump mode: enabling, feeding, scoring, budget, staleness, failure, two groups.
6. The speculative threads: turn lock on the demand read, promotion, quiesce on pause/fill/shutdown, watchdog.
7. Python wiring: gate registry, target table, host copy, enabling from the service.
8. InstrBuild events and scorer metrics.
9. The A/B arm, the served smoke and the A/B on divix01, results in `DSV41_REFERENCE.md` §33.14 and the handoff.

The guidance's task 5 is split into Tasks 5 and 6: the pump-mode step (deterministic, single-threaded) can be rejected or approved independently of the threading (turn lock, promotion, quiesce, watchdog), and a reviewer of the threading needs the step already proven.

---

### Task 1: Options, launch gate, spare cores and spill room

**Files:**
- Modify: `python/sglang/srt/environ.py` (DSV41 block, after `SGLANG_DSV41_CPU_EXPERTS_KEEP_WARM_US`, ~line 1993)
- Create: `python/sglang/srt/layers/moe/ram_prefetch.py` (bounds only in this task)
- Modify: `python/sglang/srt/arg_groups/expert_stream_requirements_exl3.py` (`_check`, new `_check_ram_prefetch`)
- Modify: `python/sglang/srt/layers/moe/cpu_experts/threading_config.py` (`CoreSettings`, `NodePlan`, `ThreadingConfig.from_env`/`resolve`, new `_spec_cores`)
- Modify: `python/sglang/srt/layers/moe/exl3_ram_miss.py` (`spill_room_shortfall`, `_check_spill_room`, `Exl3RamMissService.__init__`)
- Test: `test/registered/unit/test_expert_stream_requirements_exl3.py`, `test/registered/unit/layers/moe/test_threading_config.py`, `test/registered/unit/kernels/test_exl3_ram_miss_attach_lanes.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `envs.SGLANG_DSV41_RAM_PREFETCH` (EnvBool False), `envs.SGLANG_DSV41_RAM_PREFETCH_PER_TOKEN` (EnvInt 1), `envs.SGLANG_DSV41_RAM_PREFETCH_PER_LAYER` (EnvInt 1), `envs.SGLANG_DSV41_RAM_PREFETCH_SPEC_SHARE` (EnvInt 2); `ram_prefetch.MAX_PER_TOKEN = 12`, `MAX_PER_LAYER = 8`, `MAX_SPEC_SHARE = 4`; `CoreSettings.ram_prefetch: bool = False`; `NodePlan.spec: tuple[int, ...] = ()`; `spill_room_shortfall(ranges, *, staging, lanes, hot, pool=0)`; `Exl3RamMissService._spec_share: int` (0 until Task 7 sets it at start).

- [ ] **Step 0: Prepare the divix01 worktree and check the interpreter**

Run:
```bash
git -C /Users/dnikolaidis/.codex/worktrees/ram-prefetch/sglang-nvfp4 push origin codex/dsv41-ram-prefetch
ssh divix01 bash -s <<'REMOTE'
set -e
WT=/data/models/slang/nvfp4-work/wt-ram-prefetch
if [ ! -d "$WT" ]; then
  git -C /data/models/slang/sglang fetch origin
  git -C /data/models/slang/sglang worktree add --detach "$WT" origin/codex/dsv41-ram-prefetch
fi
cd "$WT" && git fetch -q origin && git checkout -q --detach origin/codex/dsv41-ram-prefetch && git log -1 --oneline
PYTHONPATH=$PWD/python /data/models/slang/.venv/bin/python -c "import sglang; print(sglang.__file__)"
REMOTE
```
Expected: the branch head's one-line log, then `/data/models/slang/nvfp4-work/wt-ram-prefetch/python/sglang/__init__.py`. Any other path means the run would test unrelated code; stop and fix `PYTHONPATH`.

- [ ] **Step 1: Write the failing tests**

Append to `test/registered/unit/test_expert_stream_requirements_exl3.py`:

```python
def test_ram_prefetch_options_default_off_at_the_replays_budget():
    """Off until a served A/B is accepted; one candidate per token, one row per layer (DSV41_REFERENCE.md 33.13)."""
    assert envs.SGLANG_DSV41_RAM_PREFETCH.get() is False
    assert envs.SGLANG_DSV41_RAM_PREFETCH_PER_TOKEN.get() == 1
    assert envs.SGLANG_DSV41_RAM_PREFETCH_PER_LAYER.get() == 1
    assert envs.SGLANG_DSV41_RAM_PREFETCH_SPEC_SHARE.get() == 2


def test_ram_prefetch_needs_cpu_experts(model_dir):
    """Only a record with a CPU lane stages the input the scorer reads, and only a forced CPU miss uses the pool."""
    args = _launch(model_dir, cuda_graph_config=BREAKABLE_BS1)
    assert expert_stream_requirements_for(args, args).label == "EXL3"
    _gate(args, **CPU_EXPERTS_ENV, SGLANG_DSV41_RAM_PREFETCH=True)
    with pytest.raises(ValueError, match="SGLANG_DSV41_RAM_PREFETCH needs SGLANG_DSV41_CPU_EXPERTS=1"):
        _gate(args, SGLANG_DSV41_RAM_PREFETCH=True)


@pytest.mark.parametrize(
    "name, value",
    [
        ("SGLANG_DSV41_RAM_PREFETCH_PER_TOKEN", 0),
        ("SGLANG_DSV41_RAM_PREFETCH_PER_TOKEN", 13),
        ("SGLANG_DSV41_RAM_PREFETCH_PER_LAYER", 0),
        ("SGLANG_DSV41_RAM_PREFETCH_PER_LAYER", 9),
        ("SGLANG_DSV41_RAM_PREFETCH_SPEC_SHARE", 0),
        ("SGLANG_DSV41_RAM_PREFETCH_SPEC_SHARE", 5),
    ],
)
def test_ram_prefetch_options_outside_the_hosts_bounds_are_refused(model_dir, name, value):
    """Refused at launch, not at the service's start, where the host would refuse the same bound."""
    args = _launch(model_dir, cuda_graph_config=BREAKABLE_BS1)
    assert expert_stream_requirements_for(args, args).label == "EXL3"
    with pytest.raises(ValueError, match=f"{name} must be in"):
        _gate(args, **CPU_EXPERTS_ENV, SGLANG_DSV41_RAM_PREFETCH=True, **{name: value})
```

Append to `test/registered/unit/layers/moe/test_threading_config.py`:

```python
def test_ram_prefetch_takes_each_nodes_spare_affinity_cores_or_its_ram_core(divix01):
    """Spare: the server's affinity on the node less every assigned core and its SMT sibling (CPU experts 8-14 take
    44-50 with them, the copy core 15 takes 51). Node 1 has no affinity core, so its thread shares the RAM core."""
    config = resolve(divix01, cpu_experts=True, threads=16, ram_prefetch=True)
    assert config.plans[0].spec == (*range(0, 8), 16, *range(36, 44), 52)
    assert config.plans[1].spec == (35,)
    assert config.log_lines()[:2] == [
        "numa node0: ram=17 cpu=8-14 (7) sq=- spec=0-7,16,36-43,52",
        "numa node1: ram=35 cpu=18-33 (16) sq=- spec=35",
    ]


def test_without_ram_prefetch_no_plan_names_spec_cores(divix01):
    config = resolve(divix01, cpu_experts=True, threads=16)
    assert [plan.spec for plan in config.plans] == [(), ()]
    assert "spec=" not in " ".join(config.log_lines())
```

Append to `test/registered/unit/kernels/test_exl3_ram_miss_attach_lanes.py`:

```python
def test_the_speculative_pool_adds_its_share_to_the_spill_room():
    """The pool's kSpec slots are never victims, so a forced miss's room needs them on top (RAM prefetch)."""
    assert module.spill_room_shortfall([(0, 69), (69, 139)], staging=8, lanes=36, hot=24, pool=2) == [(0, 69, 70)]
    assert module.spill_room_shortfall([(0, 80), (80, 161)], staging=8, lanes=36, hot=24, pool=2) == []
```

- [ ] **Step 2: Commit the failing tests, push, and run them on divix01**

```bash
cd /Users/dnikolaidis/.codex/worktrees/ram-prefetch/sglang-nvfp4
git add test/registered/unit/test_expert_stream_requirements_exl3.py test/registered/unit/layers/moe/test_threading_config.py test/registered/unit/kernels/test_exl3_ram_miss_attach_lanes.py
git commit -m "Test the RAM prefetch options, launch gate, spare cores and spill room

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push origin codex/dsv41-ram-prefetch
ssh divix01 bash -s <<'REMOTE'
cd /data/models/slang/nvfp4-work/wt-ram-prefetch && git fetch -q origin && git checkout -q --detach origin/codex/dsv41-ram-prefetch && git log -1 --oneline
PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 taskset -c 0-63 /data/models/slang/.venv/bin/python -m pytest test/registered/unit/test_expert_stream_requirements_exl3.py test/registered/unit/layers/moe/test_threading_config.py test/registered/unit/kernels/test_exl3_ram_miss_attach_lanes.py -q -p no:randomly -k "ram_prefetch or spec_cores or speculative_pool" 2>&1 | tail -15; echo "EXIT=${PIPESTATUS[0]}"
REMOTE
```
Expected: `EXIT=1`; failures with `AttributeError: ... SGLANG_DSV41_RAM_PREFETCH`, `TypeError: CoreSettings.__init__() got an unexpected keyword argument 'ram_prefetch'`, and `TypeError: spill_room_shortfall() got an unexpected keyword argument 'pool'`.

- [ ] **Step 3: Add the options**

In `python/sglang/srt/environ.py`, directly after `SGLANG_DSV41_CPU_EXPERTS_KEEP_WARM_US = EnvInt(2000)`:

```python
    # NVMe-to-RAM prefetch (spec 2026-10-08-dsv41-ram-prefetch-design, Phase 1): per NUMA group a thread scores the
    # next streamed layer's gate on a record's staged CPU input and reads its pick into a pool of RAM slots the device
    # never maps; a forced CPU miss on a pooled expert swaps it in instead of reading. Needs SGLANG_DSV41_CPU_EXPERTS.
    SGLANG_DSV41_RAM_PREFETCH = EnvBool(False)
    # Candidates per live token, in its score order past the hot, mapped and pooled experts.
    SGLANG_DSV41_RAM_PREFETCH_PER_TOKEN = EnvInt(1)
    # Speculative rows per layer over both NUMA groups: the replay's best budget (DSV41_REFERENCE.md section 33.13).
    SGLANG_DSV41_RAM_PREFETCH_PER_LAYER = EnvInt(1)
    # Pool slots per row and NUMA group, taken out of the tier at start.
    SGLANG_DSV41_RAM_PREFETCH_SPEC_SHARE = EnvInt(2)
```

Create `python/sglang/srt/layers/moe/ram_prefetch.py`:

```python
"""The NVMe-to-RAM prefetch's Python side (spec docs/superpowers/specs/2026-10-08-dsv41-ram-prefetch-design.md,
Phase 1): the host's bounds, the model's router gates registered at load, and the per-row target table the host's
speculative threads score with."""

from __future__ import annotations

# The host's bounds: host/gate_scorer.h GateScorer::kDepth and kMaxPerLayer, host/ram_prefetch.h SpecPool::kMaxShare.
MAX_PER_TOKEN = 12
MAX_PER_LAYER = 8
MAX_SPEC_SHARE = 4
```

- [ ] **Step 4: Add the launch gate check**

In `python/sglang/srt/arg_groups/expert_stream_requirements_exl3.py`, in `_check`, directly after `_check_row_weighted_assignment()`:

```python
    _check_ram_prefetch()
```

and add, after `_check_slot_map`:

```python
def _check_ram_prefetch() -> None:
    """``SGLANG_DSV41_RAM_PREFETCH``'s prerequisite and its options' bounds, which the host refuses at enable too.

    Checked before the backend rules: a disabled decode graph returns early, and the option must not pass silently.
    """
    if not envs.SGLANG_DSV41_RAM_PREFETCH.get():
        return
    if not envs.SGLANG_DSV41_CPU_EXPERTS.get():
        # Only a record with a CPU lane stages the input the scorer reads, and only a forced CPU miss uses the pool.
        raise ValueError("SGLANG_DSV41_RAM_PREFETCH needs SGLANG_DSV41_CPU_EXPERTS=1")
    from sglang.srt.layers.moe import ram_prefetch

    for name, high in (
        ("SGLANG_DSV41_RAM_PREFETCH_PER_TOKEN", ram_prefetch.MAX_PER_TOKEN),
        ("SGLANG_DSV41_RAM_PREFETCH_PER_LAYER", ram_prefetch.MAX_PER_LAYER),
        ("SGLANG_DSV41_RAM_PREFETCH_SPEC_SHARE", ram_prefetch.MAX_SPEC_SHARE),
    ):
        value = getattr(envs, name).get()
        if not 1 <= value <= high:
            raise ValueError(f"{name} must be in [1, {high}], got {value}")
```

- [ ] **Step 5: Derive the speculative threads' cores**

In `python/sglang/srt/layers/moe/cpu_experts/threading_config.py`:

Add to `CoreSettings`, after `draft_threads`:
```python
    ram_prefetch: bool = False  # SGLANG_DSV41_RAM_PREFETCH: each group's speculative thread needs cores
```

Add to `NodePlan`, after `busy_poll: bool`:
```python
    spec: tuple[int, ...] = ()  # the RAM prefetch's speculative thread's cores; () without the prefetch
```

Replace `NodePlan.log_line` with:
```python
    def log_line(self) -> str:
        ram = "-" if self.ram is None else str(self.ram)
        sq = "-" if self.sq is None else str(self.sq)
        spec = f" spec={_format_cpus(self.spec)}" if self.spec else ""
        return f"numa node{self.node}: ram={ram} cpu={_format_cpus(self.cpu)} ({self.threads}) sq={sq}{spec}"
```

In `ThreadingConfig.from_env`, add to the `CoreSettings(...)` call:
```python
            ram_prefetch=envs.SGLANG_DSV41_RAM_PREFETCH.get(),
```

In `ThreadingConfig.resolve`, replace the last three statements

```python
        draft: tuple[int, ...] = ()
        if settings.draft and not settings.cpu_experts:
            draft = tuple(named_draft) or _derive_draft(gpu, topology, affinity, settings, taken)
        return ThreadingConfig(tuple(plans), (copy,), gpu, draft)
```
with
```python
        draft: tuple[int, ...] = ()
        if settings.draft and not settings.cpu_experts:
            draft = tuple(named_draft) or _derive_draft(gpu, topology, affinity, settings, taken)
        if settings.ram_prefetch:
            assigned = taken | {s for c in draft for s in topology.siblings[c]}
            plans = [dataclasses.replace(plan, spec=_spec_cores(plan, topology, affinity, assigned)) for plan in plans]
        return ThreadingConfig(tuple(plans), (copy,), gpu, draft)
```

Add `import dataclasses` to the module's imports, and add after `_derive_draft`:
```python
def _spec_cores(plan: NodePlan, topology: Topology, affinity: frozenset[int], assigned: set[int]) -> tuple[int, ...]:
    """A group's speculative-thread cores: the server's affinity on its node less every assigned core and its SMT
    sibling. With none spare it shares the RAM thread's core (spec 2026-10-08-dsv41-ram-prefetch-design)."""
    spare = tuple(c for c in topology.node_cpus[plan.node] if c in affinity and not (topology.siblings[c] & assigned))
    return spare or ((plan.ram,) if plan.ram is not None else ())
```

- [ ] **Step 6: Count the pool in the spill room**

In `python/sglang/srt/layers/moe/exl3_ram_miss.py`, replace `spill_room_shortfall` with:
```python
def spill_room_shortfall(
    ranges, *, staging: int, lanes: int, hot: int, pool: int = 0
) -> list[tuple[int, int, int]]:
    """The node ranges of a layer that cannot give every forced CPU miss a RAM victim, as (group, slots, needed).

    A forced miss is read into a victim of its node's range: neither a staging slot (`staging`), nor a speculative pool
    slot (`pool`, SGLANG_DSV41_RAM_PREFETCH_SPEC_SHARE), nor an expert the record routes (at most `lanes` less the
    forced misses themselves), nor a VRAM-hot expert (at most `hot`). So a range of at least
    staging + pool + lanes + hot slots always has one (plan 2026-10-06, Task 9)."""
    need = staging + pool + lanes + hot
    return [(g, hi - lo, need) for g, (lo, hi) in enumerate(ranges) if hi - lo < need]
```

In `Exl3RamMissService._check_spill_room`, add `pool=self._spec_share` to the `spill_room_shortfall(...)` call:
```python
        short = spill_room_shortfall(
            ranges,
            staging=self.staging_for(capacity),
            lanes=width,
            hot=int(streamer.hot_cache.capacity),
            pool=self._spec_share,
        )
```

In `Exl3RamMissService.__init__`, next to the other start-time fields (beside `self._node_ranges` or `self.gpu_group`'s initialisation; `grep -n "self.gpu_group = None\|self._gpu_node_placement" python/sglang/srt/layers/moe/exl3_ram_miss.py` finds them), add:
```python
        # The RAM prefetch's pool slots per row and group, set at start (0: no pool).
        self._spec_share = 0
```

- [ ] **Step 7: Commit, push, run the tests**

```bash
cd /Users/dnikolaidis/.codex/worktrees/ram-prefetch/sglang-nvfp4
git add python/sglang/srt/environ.py python/sglang/srt/layers/moe/ram_prefetch.py python/sglang/srt/arg_groups/expert_stream_requirements_exl3.py python/sglang/srt/layers/moe/cpu_experts/threading_config.py python/sglang/srt/layers/moe/exl3_ram_miss.py
git commit -m "Add the RAM prefetch options, launch gate, spare cores and spill room

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push origin codex/dsv41-ram-prefetch
ssh divix01 bash -s <<'REMOTE'
cd /data/models/slang/nvfp4-work/wt-ram-prefetch && git fetch -q origin && git checkout -q --detach origin/codex/dsv41-ram-prefetch && git log -1 --oneline
PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 taskset -c 0-63 /data/models/slang/.venv/bin/python -m pytest test/registered/unit/test_expert_stream_requirements_exl3.py test/registered/unit/layers/moe/test_threading_config.py test/registered/unit/kernels/test_exl3_ram_miss_attach_lanes.py -q -p no:randomly 2>&1 | tail -5; echo "EXIT=${PIPESTATUS[0]}"
REMOTE
```
Expected: `EXIT=0`, every test of the three files passed (the whole files, since `NodePlan.log_line`, `resolve` and the spill check changed for every caller).

- [ ] **Step 8: Mutant check (launch gate ordering)**

```bash
ssh divix01 bash -s <<'REMOTE'
cd /data/models/slang/nvfp4-work/wt-ram-prefetch
/data/models/slang/.venv/bin/python - <<'PY'
import pathlib
p = pathlib.Path("python/sglang/srt/arg_groups/expert_stream_requirements_exl3.py")
t = p.read_text()
old = "    _check_row_weighted_assignment()\n    _check_ram_prefetch()\n"
assert t.count(old) == 1
p.write_text(t.replace(old, "    _check_row_weighted_assignment()\n"))
PY
PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 taskset -c 0-63 /data/models/slang/.venv/bin/python -m pytest test/registered/unit/test_expert_stream_requirements_exl3.py -q -p no:randomly -k ram_prefetch 2>&1 | tail -3; echo "EXIT=${PIPESTATUS[0]}"
git checkout -- python/sglang/srt/arg_groups/expert_stream_requirements_exl3.py && git status --short
PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 taskset -c 0-63 /data/models/slang/.venv/bin/python -m pytest test/registered/unit/test_expert_stream_requirements_exl3.py -q -p no:randomly -k ram_prefetch 2>&1 | tail -3; echo "EXIT=${PIPESTATUS[0]}"
REMOTE
```
Expected: first `EXIT=1` (the refusals are not raised), empty `git status --short`, then `EXIT=0`.

---

### Task 2: The speculative pool (`kSpec`)

**Files:**
- Create: `python/sglang/kernels/jit/csrc/moe/expert_stream/host/ram_prefetch.h`
- Modify: `python/sglang/kernels/jit/csrc/moe/expert_stream/host/tier_protocol.h:23-25` (slot states)
- Modify: `python/sglang/kernels/jit/csrc/moe/expert_stream/host/ram_tier.h` (include; `release` :397; new `reserve_spec_pool`, `spec_pool`, `spec_share` after `reserve_staging` :941; member `pool_`)
- Modify: `python/sglang/kernels/jit/csrc/moe/expert_stream/host/ffi_exports.h` (exports after `reserve_staging` :712; macro list)
- Modify: `python/sglang/kernels/ops/moe/expert_stream_transport.py` (`SPEC_POOL_STATES`; `reserve_spec_pool`, `spec_pool`; `slot_info` docstring)
- Create: `python/sglang/test/dsv41_ram_prefetch_fixtures.py`
- Test: `test/registered/unit/kernels/test_exl3_ram_prefetch_pool.py`

**Interfaces:**
- Consumes: `RamTier::reserve_staging`, `GroupRow::{lo, hi, staging, staging_reserved}`, `Tier::state`.
- Produces: slot state `kSpec = 4`; `enum : uint32_t { kPoolEmpty, kPoolReading, kPoolLanded }`; `pool_word(state, expert)`, `pool_state(word)`, `pool_expert(word)`; `struct PoolEntry { int32_t slot; std::atomic<uint32_t> word, for_seq; std::atomic<uint64_t> landed; }`; `class SpecPool { static constexpr int kMaxShare = 4; SpecPool(int64_t rows, int groups, int share); int share() const; PoolEntry& entry(int64_t row, int g, int i); std::mutex& mutex(int g); int find(int64_t row, int g, int32_t expert) const; bool pooled_before(int64_t row, int32_t expert, uint32_t seq) const; int claimable_locked(int64_t row, int g) const; uint64_t next_landing(); }`; `RamTier::reserve_spec_pool(int64_t share)`, `RamTier::spec_pool(int64_t row, int64_t* out)`, `RamTier::spec_share() const`, member `std::unique_ptr<SpecPool> pool_`; FFI `expert_stream_reserve_spec_pool(handle, share)`, `expert_stream_spec_share(handle) -> int64`, `expert_stream_spec_pool(handle, row, out int64 [groups*share, 4])`; Python `ExpertStreamHost.reserve_spec_pool(share)`, `ExpertStreamHost.spec_pool(row) -> list[dict]` (`group`, `slot`, `state` in `SPEC_POOL_STATES = ("empty", "reading", "landed")`, `expert`); fixtures `prefetch_rig`, `PrefetchRig`, `load`, `forced`, `HIDDEN`, `ROWS`, `LANES`, `HALVES`, `x_token_bytes`.

- [ ] **Step 1: Write the rig and the failing tests**

Create `python/sglang/test/dsv41_ram_prefetch_fixtures.py`:

```python
"""Hosts for the NVMe-to-RAM prefetch tests (spec docs/superpowers/specs/2026-10-08-dsv41-ram-prefetch-design.md,
Phase 1): an instrumented ExpertStreamHost with a speculative pool, the copy engine's test backend and CPU experts on
the fake kernel, every eligible lane the CPU's, driven by ChainSim."""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch

from sglang.kernels.ops.moe.expert_lease_block import wire_layout
from sglang.kernels.ops.moe.expert_stream_transport import ExpertStreamHost, new_page
from sglang.test.dsv41_chain_sim import ChainSim, SimRequest
from sglang.test.dsv41_ram_miss_fixtures import RamMissSetup, fake_cpu_layer, ram_miss_setup

ROWS = 2
HIDDEN = 8  # the fake CPU kernel's hidden size, and the test gates'
LANES = 8
# Two groups over an 8-slot row: node 0 slots 0-3, node 1 slots 4-7.
HALVES = [[(0, 4)] * ROWS, [(4, 8)] * ROWS]


@dataclass
class PrefetchRig:
    setup: RamMissSetup
    page: torch.Tensor
    host: ExpertStreamHost
    sim: ChainSim
    x_rows: torch.Tensor
    out_rows: torch.Tensor
    tokens: int


def x_token_bytes() -> int:
    """Bytes between two tokens' staged inputs (enable_cpu_experts: fp16, padded to 16 bytes)."""
    return (2 * HIDDEN + 15) // 16 * 16


def prefetch_rig(tmp_path, *, nodes=1, capacity=None, staging=None, share=2, pool=True, tokens=1) -> PrefetchRig:
    """One node: 7 slots a row, 3 staging, `share` pooled, 6 experts. Two nodes: 8 slots split by HALVES, 1 staging and
    `share` pooled per group, 8 experts. CPU experts on every group (one fake-kernel worker each), every row
    registered, split[n] = n so every eligible lane is the CPU's; rows of `tokens` staged inputs."""
    experts = 6 if nodes == 1 else 8
    capacity = (7 if nodes == 1 else 8) if capacity is None else capacity
    staging = (3 if nodes == 1 else 1) if staging is None else staging
    s = ram_miss_setup(tmp_path, capacity=capacity, experts=experts)
    page = new_page(pin=False, wire=wire_layout(LANES, nodes))
    host = ExpertStreamHost(
        s.tables,
        page=page,
        slot_map=torch.full((ROWS, experts), -1, dtype=torch.int32),
        variant="instr",
        **({"node_ranges": HALVES} if nodes == 2 else {}),
    )
    try:
        host.reserve_staging(staging)
        if pool:
            host.reserve_spec_pool(share)
        host.enable_copy_engine(-1)
        host.arm_copy_engine()
        table = 16 + 4 * LANES * (1 + tokens) if tokens > 1 else 0
        x_rows = torch.zeros((ROWS, tokens * x_token_bytes() + table), dtype=torch.uint8)
        shape = (ROWS, 2 * nodes, HIDDEN) if tokens == 1 else (ROWS, 2 * nodes, tokens, HIDDEN)
        out_rows = torch.zeros(shape, dtype=torch.float32)
        kernel = host.test_kernel_address()
        cores = sorted(os.sched_getaffinity(0))
        for g in range(nodes):
            host.enable_cpu_experts(
                kernel, list(range(LANES + 1)), [cores[g % len(cores)]], x_rows, out_rows, threads=1, group=g
            )
        for row in range(ROWS):
            host.set_cpu_layer(row, fake_cpu_layer(HIDDEN))
    except BaseException:
        host.stop()
        raise
    return PrefetchRig(s, page, host, ChainSim(host, page, s.slabs), x_rows, out_rows, tokens)


def _served(rig: PrefetchRig, req: SimRequest) -> None:
    if rig.host.threaded:
        assert rig.sim.wait_handled(req, timeout_s=10.0)
    else:
        assert rig.host.pump() == 1
    assert rig.sim.wait_served(req, timeout_s=10.0) and rig.sim.copy_wait(req, timeout_s=10.0)


def load(rig: PrefetchRig, row: int, experts) -> SimRequest:
    """Makes `experts` resident in `row` through an uncaptured post (each a GPU miss into staging)."""
    req = rig.sim.post(row, list(experts))
    _served(rig, req)
    return req


def forced(rig: PrefetchRig, row: int, experts, *, serve: bool = True) -> SimRequest:
    """A captured post whose lanes are all forced (spill): a hit is a CPU lane at its RAM slot, a miss a CPU miss with
    slot -1 that the host reads into a RAM victim, or swaps in from the pool."""
    req = rig.sim.post(row, list(experts), captured=True, cpu_on=True, cpu_misses=True, forced_from=0)
    if serve:
        _served(rig, req)
    return req
```

(`forced_from` is added to `ChainSim.post` in Task 3; the pool tests below do not call `forced`.)

Create `test/registered/unit/kernels/test_exl3_ram_prefetch_pool.py`:

```python
"""The RAM prefetch's speculative pool (spec 2026-10-08-dsv41-ram-prefetch-design, Phase 1, "The pool"): per row and
group, SPEC_SHARE slots in state kSpec after the staging slots, which the device never maps and no victim, admission
or release path takes (CPU, ChainSim)."""

import pytest
import torch

from sglang.kernels.ops.moe.expert_lease_block import wire_layout
from sglang.kernels.ops.moe.expert_stream_transport import new_page
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.dsv41_ram_miss_fixtures import attached_host, ram_miss_setup
from sglang.test.dsv41_ram_prefetch_fixtures import HALVES, load, prefetch_rig

register_cpu_ci(est_time=40, suite="base-a-test-cpu")

FREE, READY, STAGING, SPEC = 0, 2, 3, 4


def _states(host, row):
    return [state for state, _, _ in host.slot_info(row)]


def _pool_slots(host, row, group=None):
    return [e["slot"] for e in host.spec_pool(row) if group is None or e["group"] == group]


def test_the_pool_takes_share_free_slots_after_staging_and_publishes_nothing(tmp_path):
    rig = prefetch_rig(tmp_path)
    try:
        for row in (0, 1):
            assert _states(rig.host, row) == [STAGING] * 3 + [SPEC] * 2 + [FREE] * 2
            assert rig.host.spec_pool(row) == [
                {"group": 0, "slot": 3, "state": "empty", "expert": -1},
                {"group": 0, "slot": 4, "state": "empty", "expert": -1},
            ]
            assert rig.host.mapping(row) == [-1] * 6
            assert rig.sim.delta(row)[0] == 1 and rig.sim.delta(row)[2] == []
        assert rig.host.take_bulk_delta().shape == (0, 3)
    finally:
        rig.host.stop()


def test_each_group_pools_in_its_own_range(tmp_path):
    rig = prefetch_rig(tmp_path, nodes=2, share=1)
    try:
        for row in (0, 1):
            assert _pool_slots(rig.host, row, 0) == [1] and _pool_slots(rig.host, row, 1) == [5]
            for group, slot in ((0, 1), (1, 5)):
                lo, hi = HALVES[group][row]
                assert lo <= slot < hi
    finally:
        rig.host.stop()


def test_no_demand_victim_and_no_eager_admission_takes_a_pool_slot(tmp_path):
    """Mutant: take_victim_locked taking kSpec as free -- red (the demand's miss lands in a pool slot)."""
    rig = prefetch_rig(tmp_path)
    try:
        load(rig, 1, [0, 1])  # the row's two mappable slots
        load(rig, 1, [2])  # a demand miss: lands in staging and evicts the LRU of 0 and 1, never a pool slot
        mapped = [slot for slot in rig.host.mapping(1) if slot >= 0]
        assert len(mapped) == 2 and not set(mapped) & {3, 4}
        assert _states(rig.host, 1)[3:5] == [SPEC, SPEC]
        assert [e["state"] for e in rig.host.spec_pool(1)] == ["empty", "empty"]
        assert rig.host.victim_census(1, wanted=[0, 1, 2]) == (0, 0)
        with pytest.raises(RuntimeError, match="protected or leased"):
            rig.host.assign(1, 3, protected=[0, 1, 2], protected_fallback=False)
        slots, _ = rig.host.fill_begin(1, [4], protected=[0, 1, 2])
        assert slots == [] and rig.host.fill_end()
        assert _states(rig.host, 1)[3:5] == [SPEC, SPEC]
    finally:
        rig.host.stop()


def test_release_refuses_a_pool_slot(tmp_path):
    """Mutant: drop release()'s kSpec refusal -- red."""
    rig = prefetch_rig(tmp_path)
    try:
        with pytest.raises(RuntimeError, match="speculative pool slot"):
            rig.host.release(1, 3)
        assert _states(rig.host, 1)[3] == SPEC
    finally:
        rig.host.stop()


def _bare(tmp_path, capacity=7, k=3):
    s = ram_miss_setup(tmp_path, capacity=capacity)
    page = new_page(pin=False, wire=wire_layout(8))
    return s, page, attached_host(s, page, k=k)


@pytest.mark.parametrize("share", [0, 5])
def test_a_share_outside_one_to_four_is_refused(tmp_path, share):
    s, page, host = _bare(tmp_path)
    try:
        with pytest.raises(RuntimeError, match=r"1\.\.4 slots per group"):
            host.reserve_spec_pool(share)
    finally:
        host.stop()


def test_the_pool_is_reserved_once_after_staging_before_any_slot_fills_and_leaves_two_slots(tmp_path):
    s = ram_miss_setup(tmp_path, capacity=7)
    page = new_page(pin=False, wire=wire_layout(8))
    from sglang.kernels.ops.moe.expert_stream_transport import ExpertStreamHost

    host = ExpertStreamHost(s.tables, page=page, slot_map=torch.full((2, 6), -1, dtype=torch.int32))
    try:
        with pytest.raises(RuntimeError, match="after reserve_staging"):
            host.reserve_spec_pool(2)
    finally:
        host.stop()
    (tmp_path / "small").mkdir()
    s, page, host = _bare(tmp_path / "small", capacity=6)
    try:
        with pytest.raises(RuntimeError, match="too few slots for a speculative pool of 2"):
            host.reserve_spec_pool(2)
    finally:
        host.stop()
    (tmp_path / "filled").mkdir()
    s, page, host = _bare(tmp_path / "filled")
    try:
        host.assign(0, 1)
        with pytest.raises(RuntimeError, match="before any slot is filled"):
            host.reserve_spec_pool(2)
    finally:
        host.stop()
    (tmp_path / "twice").mkdir()
    s, page, host = _bare(tmp_path / "twice")
    try:
        host.reserve_spec_pool(2)
        with pytest.raises(RuntimeError, match="reserve_spec_pool is once"):
            host.reserve_spec_pool(2)
    finally:
        host.stop()
```

- [ ] **Step 2: Commit the failing tests, push, and run them on divix01**

```bash
cd /Users/dnikolaidis/.codex/worktrees/ram-prefetch/sglang-nvfp4
git add python/sglang/test/dsv41_ram_prefetch_fixtures.py test/registered/unit/kernels/test_exl3_ram_prefetch_pool.py
git commit -m "Test the RAM prefetch's speculative pool

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push origin codex/dsv41-ram-prefetch
ssh divix01 bash -s <<'REMOTE'
cd /data/models/slang/nvfp4-work/wt-ram-prefetch && git fetch -q origin && git checkout -q --detach origin/codex/dsv41-ram-prefetch && git log -1 --oneline
PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 taskset -c 0-63 /data/models/slang/.venv/bin/python -m pytest test/registered/unit/kernels/test_exl3_ram_prefetch_pool.py -q -p no:randomly 2>&1 | tail -10; echo "EXIT=${PIPESTATUS[0]}"
REMOTE
```
Expected: `EXIT=1`, every test failing with `AttributeError: 'ExpertStreamHost' object has no attribute 'reserve_spec_pool'`.

- [ ] **Step 3: Add the slot state**

In `tier_protocol.h`, replace lines 23-25:
```cpp
// Slot states. kStaging is one of the row's K staging slots, never mapped; an NVMe miss is read into it
// (analysis/dsv41-drive/LEASE_PROTOCOL.md). kSpec is a slot of the speculative pool (ram_prefetch.h): never mapped,
// never a victim, never released.
enum : uint8_t { kFree = 0, kReady = 2, kStaging = 3, kSpec = 4 };
```

- [ ] **Step 4: Create the pool header**

Create `python/sglang/kernels/jit/csrc/moe/expert_stream/host/ram_prefetch.h`:

```cpp
// The NVMe-to-RAM prefetch's shared state (spec docs/superpowers/specs/2026-10-08-dsv41-ram-prefetch-design.md,
// Phase 1).
//
//   SpecPool   per streamed row and NUMA group, `share` RAM-tier slots in state kSpec that the device never maps; an
//              entry is empty, reading expert e, or landed with expert e
//
// Threads. An entry's word, for_seq and landed are atomics any thread may load. Its slot, and every transition of its
// word out of empty or landed, is under its group's mutex: the group's speculative thread claims entries, the group's
// service thread swaps them (RamTier::take_pooled_locked). Only the claimer stores a reading word's successor.
#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>

namespace sglang::expert_stream {

enum : uint32_t { kPoolEmpty = 0, kPoolReading = 1, kPoolLanded = 2 };

// An entry's word: the state above bit 16, the expert below (RamTier::reserve_spec_pool refuses 65536 experts or more).
inline uint32_t pool_word(uint32_t state, int32_t expert) {
  return state << 16 | (static_cast<uint32_t>(expert) & 0xFFFFu);
}
inline uint32_t pool_state(uint32_t word) {
  return word >> 16;
}
inline int32_t pool_expert(uint32_t word) {
  return static_cast<int32_t>(word & 0xFFFFu);
}

struct PoolEntry {
  int32_t slot = -1;  // under the group's mutex
  std::atomic<uint32_t> word{kPoolEmpty};
  std::atomic<uint32_t> for_seq{0};  // the target record the read was issued for
  std::atomic<uint64_t> landed{0};   // landing order: with no empty entry, the oldest landed one is reclaimed
};

class SpecPool {
 public:
  static constexpr int kMaxShare = 4;  // Python mirror: ram_prefetch.MAX_SPEC_SHARE

  SpecPool(int64_t rows, int groups, int share)
      : groups_(groups),
        share_(share),
        entries_(std::make_unique<PoolEntry[]>(static_cast<size_t>(rows * groups * share))),
        mutexes_(std::make_unique<std::mutex[]>(static_cast<size_t>(groups))) {}

  int share() const {
    return share_;
  }
  PoolEntry& entry(int64_t row, int g, int i) {
    return entries_[(row * groups_ + g) * share_ + i];
  }
  const PoolEntry& entry(int64_t row, int g, int i) const {
    return entries_[(row * groups_ + g) * share_ + i];
  }
  std::mutex& mutex(int g) {
    return mutexes_[g];
  }

  // Group g's entry of `row` holding `expert`, reading or landed, or -1. Atomic loads: any thread.
  int find(int64_t row, int g, int32_t expert) const {
    for (int i = 0; i < share_; ++i) {
      const uint32_t word = entry(row, g, i).word.load(std::memory_order_acquire);
      if (word != kPoolEmpty && pool_expert(word) == expert) return i;
    }
    return -1;
  }

  // True when some group's entry of `row` holds `expert` for a record other than `seq`. The scorer skips such an
  // expert; one issued for `seq` itself stays ranked, so both groups' lists agree whichever group read first.
  bool pooled_before(int64_t row, int32_t expert, uint32_t seq) const {
    for (int g = 0; g < groups_; ++g) {
      const int i = find(row, g, expert);
      if (i >= 0 && entry(row, g, i).for_seq.load(std::memory_order_relaxed) != seq) return true;
    }
    return false;
  }

  // Under mutex(g): an empty entry of `row`, else its oldest landed one, else -1.
  int claimable_locked(int64_t row, int g) const {
    int oldest = -1;
    for (int i = 0; i < share_; ++i) {
      const PoolEntry& e = entry(row, g, i);
      const uint32_t state = pool_state(e.word.load(std::memory_order_relaxed));
      if (state == kPoolEmpty) return i;
      if (state == kPoolLanded &&
          (oldest < 0 || e.landed.load(std::memory_order_relaxed) <
                             entry(row, g, oldest).landed.load(std::memory_order_relaxed)))
        oldest = i;
    }
    return oldest;
  }

  uint64_t next_landing() {
    return landings_.fetch_add(1, std::memory_order_relaxed) + 1;
  }

 private:
  int groups_;
  int share_;
  std::unique_ptr<PoolEntry[]> entries_;
  std::unique_ptr<std::mutex[]> mutexes_;
  std::atomic<uint64_t> landings_{0};
};

}  // namespace sglang::expert_stream
```

- [ ] **Step 5: Reserve, introspect and protect the pool in `RamTier`**

In `ram_tier.h`, add `#include "ram_prefetch.h"` after `#include "numa_distributor.h"`.

In `RamTier::release`, directly after the `kStaging` refusal block:
```cpp
    if (tiers_[row].state[slot] == kSpec) {
      throw std::runtime_error(
          error_prefix<Layout>() + "release of pinned slot " + std::to_string(slot) +
          ": it is a speculative pool slot");
    }
```

After `reserve_staging` (ends ~line 975), add:
```cpp
  // Reserves every row's speculative pool (ram_prefetch.h): in each group's range, the first `share` kFree slots after
  // its staging slots become kSpec. Once, on the owner, after reserve_staging and before any slot is filled or the
  // service thread starts; nothing is published, since the device never maps a pool slot. A range left with fewer
  // than 2 slots to serve from is refused.
  void reserve_spec_pool(int64_t share) {
    std::lock_guard<std::mutex> caller(caller_mutex_);
    require_owner("reserve_spec_pool");
    const std::string prefix = error_prefix<Layout>() + "speculative pool: ";
    if (threaded_.load()) throw std::runtime_error(prefix + "reserve it before the service thread starts");
    if (pool_ != nullptr) throw std::runtime_error(prefix + "reserve_spec_pool is once");
    if (share < 1 || share > SpecPool::kMaxShare)
      throw std::runtime_error(prefix + "a row has 1.." + std::to_string(SpecPool::kMaxShare) + " slots per group");
    if (experts_ > 0xFFFF) throw std::runtime_error(prefix + "an entry names experts below 65536");
    for (int64_t row = 0; row < layers_; ++row) {
      const Tier& tier = tiers_[row];
      for (int g = 0; g < dist_.size(); ++g) {
        const GroupRow& own = dist_.group(g).rows[row];
        if (!own.staging_reserved) throw std::runtime_error(prefix + "reserve_spec_pool is after reserve_staging");
        if (own.hi - own.lo - static_cast<int64_t>(own.staging.size()) - share < 2)
          throw std::runtime_error(
              prefix + "row " + std::to_string(row) + " has too few slots for a speculative pool of " +
              std::to_string(share) + (dist_.size() > 1 ? " in group " + std::to_string(g) : std::string()));
        for (int64_t slot = own.lo; slot < own.hi; ++slot)
          if (tier.state[slot] != kFree && tier.state[slot] != kStaging)
            throw std::runtime_error(prefix + "reserve_spec_pool is before any slot is filled");
      }
    }
    auto pool = std::make_unique<SpecPool>(layers_, dist_.size(), static_cast<int>(share));
    for (int64_t row = 0; row < layers_; ++row) {
      Tier& tier = tiers_[row];
      for (int g = 0; g < dist_.size(); ++g) {
        const GroupRow& own = dist_.group(g).rows[row];
        int taken = 0;
        for (int64_t slot = own.lo; slot < own.hi && taken < share; ++slot) {
          if (tier.state[slot] != kFree) continue;
          tier.state[slot] = kSpec;
          pool->entry(row, g, taken++).slot = static_cast<int32_t>(slot);
        }
      }
    }
    pool_ = std::move(pool);
  }

  // Pool slots per row and group; 0 without a pool.
  int64_t spec_share() const {
    return pool_ == nullptr ? 0 : pool_->share();
  }

  // Writes {group, slot, state, expert} per pool entry of `row`, group-major; expert -1 for an empty entry. Any
  // thread: each group's entries under its mutex.
  void spec_pool(int64_t row, int64_t* out) {
    row_capacity(row);
    if (pool_ == nullptr) throw std::runtime_error(error_prefix<Layout>() + "no speculative pool (reserve_spec_pool)");
    int64_t k = 0;
    for (int g = 0; g < dist_.size(); ++g) {
      std::lock_guard<std::mutex> lock(pool_->mutex(g));
      for (int i = 0; i < pool_->share(); ++i) {
        const PoolEntry& entry = pool_->entry(row, g, i);
        const uint32_t word = entry.word.load(std::memory_order_acquire);
        out[k++] = g;
        out[k++] = entry.slot;
        out[k++] = pool_state(word);
        out[k++] = word == kPoolEmpty ? -1 : pool_expert(word);
      }
    }
  }
```

In the private members, directly after `std::vector<Tier> tiers_;` add:
```cpp
  // The speculative pool, once reserved (reserve_spec_pool); null with the RAM prefetch off.
  std::unique_ptr<SpecPool> pool_;
```

- [ ] **Step 6: Export the pool**

In `ffi_exports.h`, after the `reserve_staging` export:
```cpp
  // Reserves every row's speculative pool, `share` slots per group after its staging slots (RamTier::reserve_spec_pool).
  // Once, after reserve_staging, before any slot fills or the thread starts.
  static void reserve_spec_pool(int64_t handle, int64_t share) {
    find(handle)->reserve_spec_pool(share);
  }

  static int64_t spec_share(int64_t handle) {
    return find(handle)->spec_share();
  }

  // Writes {group, slot, state, expert} per pool entry of `row` into out int64 [groups * share, 4]. Any thread.
  static void spec_pool(int64_t handle, int64_t row, TensorView out) {
    using namespace host;
    auto cpu = SymbolicDevice{};
    const auto tier = find(handle);
    expert_stream::verify_named(
        "out",
        TensorMatcher({tier->groups() * tier->spec_share(), 4}).with_dtype<int64_t>().with_device<kDLCPU>(cpu),
        out);
    tier->spec_pool(row, static_cast<int64_t*>(out.data_ptr()));
  }
```

In `EXPERT_STREAM_HOST_EXPORTS`, after the `reserve_staging` line:
```cpp
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_reserve_spec_pool, Exports::reserve_spec_pool);     \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_spec_share, Exports::spec_share);                   \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_spec_pool, Exports::spec_pool);                     \
```

- [ ] **Step 7: Wrap the exports in Python**

In `expert_stream_transport.py`, after the `CORE_COUNTERS` assert:
```python
# A speculative pool entry's states (host/ram_prefetch.h kPoolEmpty, kPoolReading, kPoolLanded).
SPEC_POOL_STATES = ("empty", "reading", "landed")
```

In `ExpertStreamHost`, after `reserve_staging`:
```python
    def reserve_spec_pool(self, share: int) -> None:
        """Reserve every row's speculative pool (SGLANG_DSV41_RAM_PREFETCH): ``share`` slots per row and NUMA group
        after its staging slots, never mapped, never a victim, never released. Once, after ``reserve_staging``, before
        any slot fills and before the thread starts."""
        self._module.expert_stream_reserve_spec_pool(self.handle, int(share))

    def spec_pool(self, row: int) -> list[dict]:
        """Every pool entry of ``row``, group-major: ``group``, ``slot``, ``state`` (one of ``SPEC_POOL_STATES``) and
        ``expert`` (-1 when empty). Any time."""
        self._check(row)
        share = int(self._module.expert_stream_spec_share(self.handle))
        out = torch.empty((self.nodes * share, 4), dtype=torch.int64)
        self._module.expert_stream_spec_pool(self.handle, row, out)
        return [
            {"group": g, "slot": slot, "state": SPEC_POOL_STATES[state], "expert": expert}
            for g, slot, state, expert in out.tolist()
        ]
```

Change the `slot_info` docstring's first line to:
```python
        """Return ``(state, expert, stamp)`` per slot; state 0 FREE, 2 READY, 3 STAGING, 4 SPEC (the prefetch pool).
```

- [ ] **Step 8: Commit, push, run the pool tests and the paths they touch**

```bash
cd /Users/dnikolaidis/.codex/worktrees/ram-prefetch/sglang-nvfp4
git add python/sglang/kernels/jit/csrc/moe/expert_stream/host/ram_prefetch.h python/sglang/kernels/jit/csrc/moe/expert_stream/host/tier_protocol.h python/sglang/kernels/jit/csrc/moe/expert_stream/host/ram_tier.h python/sglang/kernels/jit/csrc/moe/expert_stream/host/ffi_exports.h python/sglang/kernels/ops/moe/expert_stream_transport.py
git commit -m "Reserve the RAM prefetch's speculative pool out of the tier

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push origin codex/dsv41-ram-prefetch
ssh divix01 bash -s <<'REMOTE'
cd /data/models/slang/nvfp4-work/wt-ram-prefetch && git fetch -q origin && git checkout -q --detach origin/codex/dsv41-ram-prefetch && git log -1 --oneline
PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 taskset -c 0-63 /data/models/slang/.venv/bin/python -m pytest test/registered/unit/kernels/test_exl3_ram_prefetch_pool.py test/registered/unit/kernels/test_exl3_ram_miss_slot_map.py test/registered/unit/kernels/test_exl3_ram_miss_prefill_fills.py test/registered/unit/kernels/test_expert_stream_prod_build_symbols.py -q -p no:randomly 2>&1 | tail -5; echo "EXIT=${PIPESTATUS[0]}"
REMOTE
```
Expected: `EXIT=0`. The first run rebuilds the host modules (minutes).

- [ ] **Step 9: Mutant checks**

```bash
ssh divix01 bash -s <<'REMOTE'
cd /data/models/slang/nvfp4-work/wt-ram-prefetch
H=python/sglang/kernels/jit/csrc/moe/expert_stream/host/ram_tier.h
run() { PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 taskset -c 0-63 /data/models/slang/.venv/bin/python -m pytest test/registered/unit/kernels/test_exl3_ram_prefetch_pool.py -q -p no:randomly 2>&1 | tail -3; echo "EXIT=${PIPESTATUS[0]}"; }
mutate() { /data/models/slang/.venv/bin/python - "$H" "$1" "$2" <<'PY'
import pathlib, sys
p, old, new = pathlib.Path(sys.argv[1]), sys.argv[2].encode().decode("unicode_escape"), sys.argv[3].encode().decode("unicode_escape")
t = p.read_text(); assert t.count(old) == 1, t.count(old); p.write_text(t.replace(old, new))
PY
}
echo "== mutant: a victim taken from the pool"
mutate '    *old = -1;\n    for (int64_t slot = own.lo; slot < own.hi; ++slot) {\n      if (tier.state[slot] == kFree) return slot;' '    *old = -1;\n    for (int64_t slot = own.lo; slot < own.hi; ++slot) {\n      if (tier.state[slot] == kFree || tier.state[slot] == kSpec) return slot;'
run; git checkout -- "$H"
echo "== mutant: release accepts a pool slot"
mutate '    if (tiers_[row].state[slot] == kSpec) {' '    if (false) {'
run; git checkout -- "$H"
git status --short
echo "== restored"; run
REMOTE
```
Expected: `EXIT=1` after each mutant (red on `test_no_demand_victim_and_no_eager_admission_takes_a_pool_slot` and `test_release_refuses_a_pool_slot` respectively), empty `git status --short`, then `EXIT=0` restored.

---

### Task 3: Swap-on-use

**Files:**
- Modify: `python/sglang/kernels/jit/csrc/moe/expert_stream/host/tier_protocol.h:31-75` (counters)
- Modify: `python/sglang/kernels/jit/csrc/moe/expert_stream/host/ram_tier.h` (`RecordPlan` :1634; `submit_landed_cpu_misses` :1855; `reserve_victims_locked` :1798; `read_misses` :1882; `commit_inserted_locked` :1922; `serve_record` :1958; new `take_pooled_locked`, `miss_landed`, `read_rows`, `spec_place`)
- Modify: `python/sglang/kernels/jit/csrc/moe/expert_stream/host/ffi_test_exports.h` (`spec_place`)
- Modify: `python/sglang/kernels/ops/moe/expert_stream_transport.py` (`COUNTERS`, `CORE_COUNTERS`, `TEST_ONLY_EXPORTS`, `spec_place`)
- Modify: `python/sglang/test/dsv41_chain_sim.py` (`ChainSim.post`: `forced_from`)
- Test: `test/registered/unit/kernels/test_exl3_ram_prefetch_swap.py`

**Interfaces:**
- Consumes: `SpecPool`, `PoolEntry`, `pool_word`, `kPoolLanded`, `RamTier::pool_` (Task 2); `prefetch_rig`, `load`, `forced` (Task 2).
- Produces: counters `kSpecIssued, kSpecLanded, kSpecUsed, kSpecPromoted, kSpecDropped, kSpecFailed, kSpecDelayed` (core) and `kSpecScored, kSpecScoreNs` (metrics), Python names `spec_issued, spec_landed, spec_used, spec_promoted, spec_dropped, spec_failed, spec_delayed, spec_scored, spec_score_ns`; `RecordPlan::{pooled, read_experts, read_slots, read_lanes, ordinal}`; `int32_t RamTier::take_pooled_locked(Group&, const Request&, int32_t expert, int32_t victim)`; `void RamTier::read_rows(Group&, const Request&, const RecordPlan&, std::span<const int32_t>, std::span<const int64_t>, std::span<const int32_t>, CpuMissBatch*, StageRecord*)`; `int64_t RamTier::read_misses(Group&, const Request&, RecordPlan&, CpuMissBatch*, StageRecord*)`; `int64_t RamTier::spec_place(int64_t row, int64_t expert)` (InstrBuild); FFI test export `expert_stream_spec_place(handle, row, expert) -> slot`; Python `ExpertStreamHost.spec_place(row, expert) -> int`; `ChainSim.post(..., forced_from: Optional[int] = None)`.

- [ ] **Step 1: Let ChainSim post forced lanes**

In `python/sglang/test/dsv41_chain_sim.py`, add to `ChainSim.post`'s keyword parameters, after `chain: Optional[int] = None,`:
```python
        forced_from: Optional[int] = None,
```
and pass it in the `type_lanes(...)` call, after `cpu_misses=cpu_misses, cpu_ok=cpu_ok, ce_ok=ce_ok,`:
```python
                forced_from=forced_from,
```
Extend the docstring's last sentence: `` ``forced_from`` types the lanes from it on as spill does (type_lanes).``

- [ ] **Step 2: Write the failing tests**

Create `test/registered/unit/kernels/test_exl3_ram_prefetch_swap.py`:

```python
"""Swap-on-use (spec 2026-10-08-dsv41-ram-prefetch-design, Phase 1, "Swap on use"): a forced CPU miss whose expert
landed in the row's pool takes its victim as today, maps the pool slot, gives the victim to the pool, reads nothing and
goes to the CPU at once. spec_place lands a pool row as a speculative read would (CPU, ChainSim)."""

from sglang.srt.layers.moe.ram_slot_map import LaneKind
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.dsv41_ram_miss_fixtures import same_bytes
from sglang.test.dsv41_ram_prefetch_fixtures import forced, load, prefetch_rig

register_cpu_ci(est_time=40, suite="base-a-test-cpu")

SPEC, READY = 4, 2


def _entry(host, row, expert):
    return next((e for e in host.spec_pool(row) if e["expert"] == expert), None)


def _same_row(rig, row, slot, expert):
    got, ref = rig.sim.read_slot(row, slot), rig.setup.reference(row, [expert])
    return all(same_bytes(got[name], ref[name][0]) for name in got)


def test_a_pool_row_is_invisible_to_the_device_until_a_forced_miss_swaps_it_in(tmp_path):
    """No read, the pool slot mapped by the record's delta, the free victim pooled in its place, the CPU job at once,
    bytes equal to a fresh read. Mutant: spec_place publishing the mirror -- red on the first mapping check."""
    rig = prefetch_rig(tmp_path)
    try:
        slot = rig.host.spec_place(1, 4)
        assert _entry(rig.host, 1, 4) == {"group": 0, "slot": slot, "state": "landed", "expert": 4}
        assert rig.host.mapping(1)[4] == -1 and rig.sim.delta(1)[2] == []
        rows, layer_rows = rig.host.counters()["rows_read"], rig.host.layer_rows()
        req = forced(rig, 1, [4])
        assert req.kinds == [LaneKind.MISS_CPU] and req.slots == [-1]
        counters = rig.host.counters()
        assert counters["spec_used"] == 1
        assert counters["rows_read"] == rows and rig.host.layer_rows() == layer_rows, "a pooled miss was read"
        assert rig.host.mapping(1)[4] == slot and rig.sim.delta(1)[2] == [(4, slot)]
        assert rig.host.slot_info(1)[slot][0] == READY
        victim = 5  # the row's first free slot, now pooled
        assert rig.host.slot_info(1)[victim][0] == SPEC
        assert {"group": 0, "slot": victim, "state": "empty", "expert": -1} in rig.host.spec_pool(1)
        calls = rig.host.test_kernel_calls()
        assert calls[-1]["slots"] == [slot] and calls[-1]["accumulate"] is False
        assert _same_row(rig, 1, slot, 4)
    finally:
        rig.host.stop()


def test_the_swap_still_evicts_its_victim_in_the_delta(tmp_path):
    """Mutant: skip the victim's eviction entry in the forced loop -- red on the delta."""
    rig = prefetch_rig(tmp_path)
    try:
        load(rig, 1, [0, 1])
        slot = rig.host.spec_place(1, 4)
        victim = rig.host.mapping(1)[0]  # 0 is the LRU of the two
        forced(rig, 1, [4])
        assert rig.sim.delta(1)[2] == [(0, -1), (4, slot)]
        assert rig.host.mapping(1)[0] == -1 and rig.host.mapping(1)[4] == slot
        assert rig.host.slot_info(1)[victim][0] == SPEC
        assert {"group": 0, "slot": victim, "state": "empty", "expert": -1} in rig.host.spec_pool(1)
        assert rig.host.counters()["evictions"] >= 1
    finally:
        rig.host.stop()


def test_a_record_mixing_a_pooled_and_a_read_forced_miss_sends_the_pooled_one_first(tmp_path):
    rig = prefetch_rig(tmp_path)
    try:
        slot = rig.host.spec_place(1, 4)
        rows = rig.host.counters()["rows_read"]
        req = forced(rig, 1, [4, 2])
        assert req.kinds == [LaneKind.MISS_CPU, LaneKind.MISS_CPU]
        calls = rig.host.test_kernel_calls()[-2:]
        assert calls[0]["slots"] == [slot] and calls[0]["accumulate"] is False
        assert calls[1]["slots"] == [rig.host.mapping(1)[2]] and calls[1]["accumulate"] is True
        assert rig.host.counters()["rows_read"] == rows + 1 and rig.host.counters()["spec_used"] == 1
        assert _same_row(rig, 1, rig.host.mapping(1)[2], 2) and _same_row(rig, 1, slot, 4)
    finally:
        rig.host.stop()


def test_a_gpu_miss_on_a_pooled_expert_is_read_and_the_pool_row_stays(tmp_path):
    rig = prefetch_rig(tmp_path)
    try:
        slot = rig.host.spec_place(1, 4)
        req = load(rig, 1, [4])
        assert req.kinds == [LaneKind.MISS_GPU]
        assert rig.host.mapping(1)[4] not in (-1, slot)
        assert _entry(rig.host, 1, 4) == {"group": 0, "slot": slot, "state": "landed", "expert": 4}
        assert rig.host.counters()["spec_used"] == 0
    finally:
        rig.host.stop()


def test_a_two_group_swap_stays_in_the_experts_home_range(tmp_path):
    rig = prefetch_rig(tmp_path, nodes=2, share=1)
    try:
        slot = rig.host.spec_place(1, 3)  # expert 3 is group 1's
        assert 4 <= slot < 8
        forced(rig, 1, [3])
        assert rig.host.mapping(1)[3] == slot
        assert rig.host.group_counters(1)["spec_used"] == 1 and rig.host.group_counters(0)["spec_used"] == 0
    finally:
        rig.host.stop()
```

- [ ] **Step 3: Commit the failing tests, push, run them on divix01**

```bash
cd /Users/dnikolaidis/.codex/worktrees/ram-prefetch/sglang-nvfp4
git add python/sglang/test/dsv41_chain_sim.py test/registered/unit/kernels/test_exl3_ram_prefetch_swap.py
git commit -m "Test swap-on-use of a landed pool row by a forced CPU miss

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push origin codex/dsv41-ram-prefetch
ssh divix01 bash -s <<'REMOTE'
cd /data/models/slang/nvfp4-work/wt-ram-prefetch && git fetch -q origin && git checkout -q --detach origin/codex/dsv41-ram-prefetch && git log -1 --oneline
PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 taskset -c 0-63 /data/models/slang/.venv/bin/python -m pytest test/registered/unit/kernels/test_exl3_ram_prefetch_swap.py -q -p no:randomly 2>&1 | tail -8; echo "EXIT=${PIPESTATUS[0]}"
REMOTE
```
Expected: `EXIT=1`, failures with `AttributeError: 'ExpertStreamHost' object has no attribute 'spec_place'`.

- [ ] **Step 4: Add the counters**

In `tier_protocol.h`, replace
```cpp
  // CPU experts: CPU jobs (one per part of a record) and their lanes.
  kCpuJobs,
  kCpuLanes,
  kCounterCount,
};
```
with
```cpp
  // CPU experts: CPU jobs (one per part of a record) and their lanes.
  kCpuJobs,
  kCpuLanes,
  // RAM prefetch (ram_prefetch.h): speculative reads started, landed; pool rows a forced miss swapped in, and of those
  // the ones whose read was still in flight (promoted); candidates dropped (stale, mapped or pooled since scoring, or
  // a full ring); reads failed; demand reads that waited at the reader's turn for a speculative read.
  kSpecIssued,
  kSpecLanded,
  kSpecUsed,
  kSpecPromoted,
  kSpecDropped,
  kSpecFailed,
  kSpecDelayed,
  // ... the scorer's records scored and their scoring time in ns (metrics).
  kSpecScored,
  kSpecScoreNs,
  kCounterCount,
};
```
and in `is_core_counter`, after `case kRamInsertSkipped:` add:
```cpp
    case kSpecIssued:
    case kSpecLanded:
    case kSpecUsed:
    case kSpecPromoted:
    case kSpecDropped:
    case kSpecFailed:
    case kSpecDelayed:
```
Update the comment above `is_core_counter` to:
```cpp
// True for the counters the production build keeps: the shutdown line's served, rows and errors, the admission
// policy's outcomes, the RAM prefetch's outcomes (a counters-off A/B reports them), and the functional version.
```

In `expert_stream_transport.py`, append to `COUNTERS` after `"cpu_lanes",`:
```python
    # RAM prefetch.
    "spec_issued",
    "spec_landed",
    "spec_used",
    "spec_promoted",
    "spec_dropped",
    "spec_failed",
    "spec_delayed",
    "spec_scored",
    "spec_score_ns",
```
and to `CORE_COUNTERS` after `"ram_insert_skipped",`:
```python
    "spec_issued",
    "spec_landed",
    "spec_used",
    "spec_promoted",
    "spec_dropped",
    "spec_failed",
    "spec_delayed",
```
Then run `grep -rn '"cpu_lanes"' python test analysis benchmarks scripts` on the laptop; any other file that asserts the whole counter tuple gets the same nine names appended in the same order.

- [ ] **Step 5: Extend the record plan and the CPU-miss landing check**

In `ram_tier.h`, replace `struct RecordPlan { ... };` with:
```cpp
  struct RecordPlan {
    int64_t idx = 0;  // the record's index in the demand ring
    CopyJob job;
    FixedVec<int32_t, kWanted> wanted;
    FixedVec<int32_t, Wire::kLanes> missing;
    FixedVec<int64_t, Wire::kLanes> slots;
    FixedVec<int32_t, Wire::kLanes> miss_lane;  // the lane index of each read row
    Wire::LaneMask pooled = 0;  // bit i: miss i was swapped in from the speculative pool and is not read
    // With a pooled miss, the rows read_misses reads, in miss order, and each miss's read ordinal (-1: pooled).
    FixedVec<int32_t, Wire::kLanes> read_experts;
    FixedVec<int64_t, Wire::kLanes> read_slots;
    FixedVec<int32_t, Wire::kLanes> read_lanes;
    FixedVec<int32_t, Wire::kLanes> ordinal;
  };
```

Before `submit_landed_cpu_misses`, add:
```cpp
  // True once miss i's row is in RAM: swapped in from the pool, or packed by the read (indexed by its read ordinal when
  // some miss was pooled).
  static bool miss_landed(const Group& group, const RecordPlan& plan, size_t i) {
    if ((plan.pooled >> i & 1u) != 0) return true;
    const size_t at = plan.pooled != 0 ? static_cast<size_t>(plan.ordinal[i]) : i;
    return at < group.packed.size() && group.packed[at] != 0;
  }
```
and in `submit_landed_cpu_misses` replace
```cpp
      if (!(i < group.packed.size() && group.packed[i] != 0)) continue;
```
with
```cpp
      if (!miss_landed(group, plan, i)) continue;
```

- [ ] **Step 6: Swap in `reserve_victims_locked`**

Before `reserve_victims_locked`, add:
```cpp
  // A forced miss's pool row: when group g's pool of the row holds `expert` landed, the entry's slot is returned and
  // the entry takes `victim` (empty) in its place; else -1 and the miss is read. Under the group's pool mutex.
  int32_t take_pooled_locked(Group& group, const Request& request, int32_t expert, int32_t victim) {
    const int g = group.index;
    const int i = pool_->find(request.row, g, expert);
    if (i < 0) return -1;
    PoolEntry& entry = pool_->entry(request.row, g, i);
    std::lock_guard<std::mutex> lock(pool_->mutex(g));
    if (entry.word.load(std::memory_order_acquire) != pool_word(kPoolLanded, expert)) return -1;
    const int32_t slot = entry.slot;
    entry.slot = victim;
    entry.word.store(kPoolEmpty, std::memory_order_release);
    count<kSpecUsed>(group);
    return slot;
  }
```

In `reserve_victims_locked`, replace the forced-miss loop body after the eviction entry, i.e. replace
```cpp
      part.entries[part.count][0] = plan.missing[i];
      part.entries[part.count][1] = static_cast<int32_t>(victim);
      ++part.count;
      tier.state[victim] = kStaging;  // being read; commit_inserted_locked makes it READY
      plan.slots[i] = victim;
      placed[i] = true;
      inserted[i] = true;
    }
```
with
```cpp
      // A pooled row is mapped where it landed, and the victim takes its place in the pool.
      const int32_t pooled =
          pool_ != nullptr ? take_pooled_locked(group, request, plan.missing[i], static_cast<int32_t>(victim)) : -1;
      const int64_t slot = pooled >= 0 ? pooled : victim;
      part.entries[part.count][0] = plan.missing[i];
      part.entries[part.count][1] = static_cast<int32_t>(slot);
      ++part.count;
      tier.state[slot] = kStaging;  // being read, or landed in the pool; commit_inserted_locked makes it READY
      if (pooled >= 0) {
        tier.state[victim] = kSpec;
        plan.pooled |= Wire::LaneMask{1} << i;
      }
      plan.slots[i] = slot;
      placed[i] = true;
      inserted[i] = true;
    }
```

- [ ] **Step 7: Read only the unpooled misses, and send the pooled ones at once**

Replace `read_misses` with:
```cpp
  // Reads the misses into their staging slots, each CPU miss going to the CPU as its row lands. A miss swapped in from
  // the speculative pool is not read and its CPU job goes first. Fail-stops on a failed read. Returns the stage status.
  int64_t read_misses(
      Group& group, const Request& request, RecordPlan& plan, CpuMissBatch* misses, StageRecord* cur) {
    group.packed.clear();
    const bool pooled = plan.pooled != 0;
    if (pooled) {
      for (size_t i = 0; i < plan.missing.size(); ++i) {
        if ((plan.pooled >> i & 1u) != 0) {
          plan.ordinal.push_back(-1);
          continue;
        }
        plan.ordinal.push_back(static_cast<int32_t>(plan.read_experts.size()));
        plan.read_experts.push_back(plan.missing[i]);
        plan.read_slots.push_back(plan.slots[i]);
        plan.read_lanes.push_back(plan.miss_lane[i]);
      }
      submit_landed_cpu_misses(group, request, plan, misses);
    }
    const std::span<const int32_t> experts = pooled ? plan.read_experts.span() : plan.missing.span();
    const std::span<const int64_t> slots = pooled ? plan.read_slots.span() : plan.slots.span();
    const std::span<const int32_t> lanes = pooled ? plan.read_lanes.span() : plan.miss_lane.span();
    if (!experts.empty()) read_rows(group, request, plan, experts, slots, lanes, misses, cur);
    submit_landed_cpu_misses(group, request, plan, misses);
    if (misses->left != 0) fail_record(request, "a CPU miss's row never landed");
    return kStatusServed;
  }

  // read_misses' demand read of `experts` into `slots` (`lanes` their record lanes), in read order.
  void read_rows(
      Group& group,
      const Request& request,
      const RecordPlan& plan,
      std::span<const int32_t> experts,
      std::span<const int64_t> slots,
      std::span<const int32_t> lanes,
      CpuMissBatch* misses,
      StageRecord* cur) {
    const bool publishing = init_piece_words_locked(group, request, lanes, plan.idx);
    bool fail_reads = false;
    if constexpr (Build::kFaults) {  // the test faults (inject, inject_fault): InstrBuild only
      apply_pending_fault(group);
      const int64_t delay = faults_.delay_ns.load();
      if (delay > 0 && group.demands_read >= faults_.delay_after.load()) fault_delay(delay);
      fail_reads = faults_.fail_reads.load();
    }
    if (fail_reads) fail_record(request, "a test fault failed the read");
    const int result = group.reader.read(
        request.row,
        experts,
        slots,
        kBounceRows,
        [](size_t) { return false; },
        cur,
        &group.packed,
        SIZE_MAX,
        // read() runs this once per drain-loop turn and once per finished row.
        [&] { submit_landed_cpu_misses(group, request, plan, misses); },
        publishing ? &group.piece_publish : nullptr);
    // The tier's count is the sum over the groups' readers: each adds what its own reader refused since it last did.
    const int64_t refused = group.reader.publish_refused();
    stats_.add(kPiecePublishRefused, refused - group.publish_refused_seen);
    group.publish_refused_seen = refused;
    if (result != 1) {
      count<kReadErrors>(group);
      fail_record(request, "the read failed");
    }
    ++group.demands_read;
    _mm_sfence();  // the pieces' memcpy stores land before the mirror publishes them
  }
```

In `commit_inserted_locked`, replace
```cpp
    std::atomic_ref<int64_t>(tier.rows_demand)
        .fetch_add(static_cast<int64_t>(plan.missing.size()), std::memory_order_relaxed);
    count<kVersion>(group);
    count<kRowsRead>(group, static_cast<int64_t>(plan.missing.size()));
```
with
```cpp
    const int64_t read = static_cast<int64_t>(plan.missing.size()) - std::popcount(plan.pooled);
    std::atomic_ref<int64_t>(tier.rows_demand).fetch_add(read, std::memory_order_relaxed);
    count<kVersion>(group);
    count<kRowsRead>(group, read);
```

`serve_record` already passes its local `plan` to `read_misses`; no change is needed there beyond compiling against the new non-const signature.

- [ ] **Step 8: Add `spec_place` (test only)**

In `ram_tier.h`, after `inject_fault`, add:
```cpp
  // Test only (InstrBuild): lands `expert`'s row of `row` in an empty pool entry of its home group, as a speculative
  // read would, and returns the slot. On the owner; nothing is mapped or published.
  int64_t spec_place(int64_t row, int64_t expert)
    requires(Build::kFaults)
  {
    std::lock_guard<std::mutex> caller(caller_mutex_);
    require_owner("spec_place");
    row_capacity(row);
    const std::string prefix = error_prefix<Layout>() + "spec_place: ";
    if (pool_ == nullptr) throw std::runtime_error(prefix + "no speculative pool (reserve_spec_pool)");
    if (expert < 0 || expert >= experts_) throw std::runtime_error(prefix + "expert out of range");
    if (tiers_[row].expert_slot[expert] >= 0) throw std::runtime_error(prefix + "the tier maps the expert");
    const int g = Wire::home(expert);
    int i = -1;
    int64_t slot = -1;
    {
      std::lock_guard<std::mutex> lock(pool_->mutex(g));
      for (int k = 0; k < pool_->share() && i < 0; ++k)
        if (pool_->entry(row, g, k).word.load(std::memory_order_relaxed) == kPoolEmpty) i = k;
      if (i < 0) throw std::runtime_error(prefix + "no empty pool entry");
      slot = pool_->entry(row, g, i).slot;
      pool_->entry(row, g, i).word.store(pool_word(kPoolReading, static_cast<int32_t>(expert)), std::memory_order_release);
    }
    const int32_t id = static_cast<int32_t>(expert);
    std::vector<uint8_t> packed;
    const int result = dist_.group(g).reader.read(
        row, std::span<const int32_t>(&id, 1), std::span<const int64_t>(&slot, 1), 1, [](size_t) { return false; },
        nullptr, &packed);
    _mm_sfence();
    PoolEntry& entry = pool_->entry(row, g, i);
    if (result != 1) {
      entry.word.store(kPoolEmpty, std::memory_order_release);
      throw std::runtime_error(prefix + "the read failed");
    }
    entry.landed.store(pool_->next_landing(), std::memory_order_relaxed);
    entry.word.store(pool_word(kPoolLanded, id), std::memory_order_release);
    return slot;
  }
```

In `ffi_test_exports.h`, after `inject_fault`:
```cpp
  // Test only (RamTier::spec_place): lands `expert`'s row of `row` in a pool entry, as a speculative read does; returns
  // the slot. InstrBuild only.
  static int64_t spec_place(int64_t handle, int64_t row, int64_t expert) {
    if constexpr (!Build::kFaults) {
      test_only("spec_place");
    } else {
      return find(handle)->spec_place(row, expert);
    }
  }
```
and in `EXPERT_STREAM_HOST_TEST_EXPORTS_OF`, after the `inject_group_stall` line:
```cpp
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_spec_place, Exports::spec_place);                     \
```

In `expert_stream_transport.py`, add `"spec_place",` to `TEST_ONLY_EXPORTS` after `"inject_group_stall",`, and to `ExpertStreamHost` after `inject_group_stall`:
```python
    def spec_place(self, row: int, expert: int) -> int:
        """Test only: land ``expert``'s row of ``row`` in an empty pool entry of its home group, as a speculative read
        does, and return the slot. Paused or pumping only; nothing is mapped."""
        _refuse_test_only("spec_place", self.variant)
        self._check(row, expert)
        return int(self._module.expert_stream_spec_place(self.handle, row, expert))
```

- [ ] **Step 9: Commit, push, run the swap tests and every record-path file**

```bash
cd /Users/dnikolaidis/.codex/worktrees/ram-prefetch/sglang-nvfp4
git add python/sglang/kernels/jit/csrc/moe/expert_stream/host/tier_protocol.h python/sglang/kernels/jit/csrc/moe/expert_stream/host/ram_tier.h python/sglang/kernels/jit/csrc/moe/expert_stream/host/ffi_test_exports.h python/sglang/kernels/ops/moe/expert_stream_transport.py
git commit -m "Swap a landed pool row in for a forced CPU miss instead of reading it

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push origin codex/dsv41-ram-prefetch
ssh divix01 bash -s <<'REMOTE'
cd /data/models/slang/nvfp4-work/wt-ram-prefetch && git fetch -q origin && git checkout -q --detach origin/codex/dsv41-ram-prefetch && git log -1 --oneline
PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 taskset -c 0-63 /data/models/slang/.venv/bin/python -m pytest test/registered/unit/kernels/test_exl3_ram_prefetch_swap.py test/registered/unit/kernels/test_exl3_ram_prefetch_pool.py test/registered/unit/kernels/test_exl3_ram_miss_cpu_experts.py test/registered/unit/kernels/test_exl3_ram_miss_thread.py test/registered/unit/kernels/test_exl3_ram_miss_slot_map.py test/registered/unit/kernels/test_exl3_ram_miss_numa_groups.py test/registered/unit/kernels/test_exl3_ram_miss_numa_completion.py test/registered/unit/kernels/test_exl3_ram_miss_prefill_fills.py test/registered/unit/kernels/test_expert_stream_prod_build_symbols.py -q -p no:randomly 2>&1 | tail -5; echo "EXIT=${PIPESTATUS[0]}"
REMOTE
```
Expected: `EXIT=0`. If a pre-existing file fails, run the same selection at `origin/master` (`git checkout -q --detach origin/master`) before calling it a regression, and record both counts.

- [ ] **Step 10: Mutant checks (spec Testing list: the device map naming a kSpec slot before a swap; the swap skipping the victim's eviction)**

```bash
ssh divix01 bash -s <<'REMOTE'
cd /data/models/slang/nvfp4-work/wt-ram-prefetch
H=python/sglang/kernels/jit/csrc/moe/expert_stream/host/ram_tier.h
run() { PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 taskset -c 0-63 /data/models/slang/.venv/bin/python -m pytest test/registered/unit/kernels/test_exl3_ram_prefetch_swap.py -q -p no:randomly 2>&1 | tail -3; echo "EXIT=${PIPESTATUS[0]}"; }
mutate() { /data/models/slang/.venv/bin/python - "$H" "$1" "$2" <<'PY'
import pathlib, sys
p, old, new = pathlib.Path(sys.argv[1]), sys.argv[2].encode().decode("unicode_escape"), sys.argv[3].encode().decode("unicode_escape")
t = p.read_text(); assert t.count(old) == 1, t.count(old); p.write_text(t.replace(old, new))
PY
}
echo "== mutant: a pool row on the device's map before its swap"
mutate '    entry.word.store(pool_word(kPoolLanded, id), std::memory_order_release);\n    return slot;' '    entry.word.store(pool_word(kPoolLanded, id), std::memory_order_release);\n    publish_mirror(row, expert, static_cast<int32_t>(slot));\n    return slot;'
run; git checkout -- "$H"
echo "== mutant: the swap skips the victim's eviction"
mutate 'if (victim < 0) fail_record(request, "a forced CPU miss found no RAM victim on its node");\n      if (old >= 0) {' 'if (victim < 0) fail_record(request, "a forced CPU miss found no RAM victim on its node");\n      if (false && old >= 0) {'
run; git checkout -- "$H"
git status --short
echo "== restored"; run
REMOTE
```
Expected: `EXIT=1` after each mutant (red on `test_a_pool_row_is_invisible_to_the_device_until_a_forced_miss_swaps_it_in` and `test_the_swap_still_evicts_its_victim_in_the_delta`), empty status, `EXIT=0` restored.

---

### Task 4: The gate scorer

**Files:**
- Create: `python/sglang/kernels/jit/csrc/moe/expert_stream/host/gate_scorer.h`
- Modify: `python/sglang/kernels/jit/csrc/moe/expert_stream/host/ffi_test_exports.h` (`score_gate`; include)
- Modify: `python/sglang/kernels/ops/moe/expert_stream_transport.py` (module function `score_gate`)
- Test: `test/registered/unit/kernels/test_exl3_ram_prefetch_scorer.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `float half_to_float(uint16_t)`, `float bf16_to_float(uint16_t)`, `float softplus(float)`; `class GateScorer { static constexpr int kDepth = 12; static constexpr int kMaxPerLayer = 8; void reserve(int64_t tokens, int64_t hidden, int64_t experts); int choose(const uint8_t* x, int64_t tokens, int64_t x_token_bytes, const uint16_t* w, const float* bias, int64_t experts, int64_t hidden, int top_k, int per_token, int per_layer, const uint8_t* skip, int32_t* out); }`; `void check_gate_choice(int64_t experts, int64_t top_k, int64_t per_token, int64_t per_layer)` (throws `std::invalid_argument`); FFI `expert_stream_score_gate(x uint8 [M, 2H], w uint8 [E, 2H], bias f32 [E], skip uint8 [E], top_k, per_token, per_layer, out int64 [per_layer]) -> n`; Python `score_gate(x, w, bias, skip, *, top_k, per_token, per_layer, layout="exl3", variant=None) -> list[int]`.

- [ ] **Step 1: Write the failing tests**

Create `test/registered/unit/kernels/test_exl3_ram_prefetch_scorer.py`:

```python
"""The speculative thread's gate scorer (host/gate_scorer.h) against a torch reference of the Phase 0 replay's ranking
(analysis/dsv41-drive/prefetch-replay/verify_replay.py gate_choice and issue's budget): sqrt(softplus(W x)) + b per live
token, each token's top 12 walked in score order past the skipped experts for per_token picks, each pick's margin its
score less the token's top_k-th, the union by margin (ties by id), the first per_layer (CPU)."""

import pytest
import torch

from sglang.kernels.ops.moe import expert_stream_transport as es
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

DEPTH = 12


def reference(x, w, bias, skip, *, top_k, per_token, per_layer):
    logits = x.float() @ w.float().T
    scores = torch.where(logits > 20, logits, torch.log1p(torch.exp(logits))).sqrt() + bias
    best = {}
    experts = w.shape[0]
    for m in range(x.shape[0]):
        s = scores[m]
        order = sorted(range(experts), key=lambda e: (-float(s[e]), e))[: min(DEPTH, experts)]
        kth = s[order[top_k - 1]]
        picked = 0
        for e in order:
            if picked >= per_token:
                break
            if skip[e]:
                continue
            margin = float(s[e] - kth)  # an fp32 subtraction, as the host's
            best[e] = max(best.get(e, float("-inf")), margin)
            picked += 1
    return [e for e, _ in sorted(best.items(), key=lambda kv: (-kv[1], kv[0]))][:per_layer]


def _exact_case(seed, tokens=3, experts=16, hidden=64):
    """Small non-negative integers with a constant 21 term: every logit is an integer above 20, exact in fp32 in any
    summation order, so softplus is the identity and host and torch produce the same fp32 scores bit for bit."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(0, 4, (tokens, hidden), generator=g).to(torch.float16)
    x[:, 0] = 1
    w = torch.randint(0, 4, (experts, hidden), generator=g).to(torch.bfloat16)
    w[:, 0] = 21
    bias = torch.randperm(experts, generator=g).float() * 1e-3
    return x, w, bias


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize(
    "top_k, per_token, per_layer", [(6, 1, 1), (6, 1, 3), (6, 2, 4), (6, 3, 8), (2, 1, 2), (1, 2, 5)]
)
def test_the_host_ranks_exactly_as_the_replays_reference(seed, top_k, per_token, per_layer):
    """Mutants: the margin to the (top_k - 1)-th score -- red (the cross-token union order changes); per_layer + 1 rows
    -- red (one more pick)."""
    x, w, bias = _exact_case(seed)
    g = torch.Generator().manual_seed(100 + seed)
    skip = (torch.rand(w.shape[0], generator=g) < 0.25).tolist()
    kw = dict(top_k=top_k, per_token=per_token, per_layer=per_layer)
    assert es.score_gate(x, w, bias, skip, **kw) == reference(x, w, bias, skip, **kw)


def test_skipped_experts_are_passed_over_not_counted():
    x, w, bias = _exact_case(0, tokens=1)
    first = reference(x, w, bias, [False] * 16, top_k=6, per_token=1, per_layer=1)
    skip = [e in first for e in range(16)]
    assert es.score_gate(x, w, bias, skip, top_k=6, per_token=1, per_layer=1) not in ([], first)


def _separated(x, w, bias, top_k, per_token):
    """True when each token's top-12 scores, and the margins of every token's per_token picks, are at least 1e-4
    apart: the log1p/exp branch may differ from torch's by an ulp, which must not reorder anything."""
    logits = x.float() @ w.float().T
    scores = torch.where(logits > 20, logits, torch.log1p(torch.exp(logits))).sqrt() + bias
    top = scores.sort(dim=-1, descending=True).values[:, :DEPTH]
    margins = (top[:, :per_token] - top[:, top_k - 1 : top_k]).flatten().sort().values
    return bool((top[:, :-1] - top[:, 1:]).min() > 1e-4) and bool((margins[1:] - margins[:-1]).min() > 1e-4)


def test_the_softplus_branch_ranks_as_the_reference():
    for seed in range(200):
        g = torch.Generator().manual_seed(seed)
        x = torch.randint(-2, 3, (3, 16), generator=g).to(torch.float16)
        w = torch.randint(-2, 3, (16, 16), generator=g).to(torch.bfloat16)
        bias = torch.randperm(16, generator=g).float() * 1e-2
        if _separated(x, w, bias, 6, 2):
            break
    else:
        pytest.fail("no well-separated case in 200 seeds")
    kw = dict(top_k=6, per_token=2, per_layer=4)
    assert es.score_gate(x, w, bias, [False] * 16, **kw) == reference(x, w, bias, [False] * 16, **kw)


@pytest.mark.parametrize(
    "kw, why",
    [
        (dict(top_k=0, per_token=1, per_layer=1), "top_k"),
        (dict(top_k=13, per_token=1, per_layer=1), "top_k"),
        (dict(top_k=6, per_token=0, per_layer=1), "per_token"),
        (dict(top_k=6, per_token=13, per_layer=1), "per_token"),
        (dict(top_k=6, per_token=1, per_layer=0), "per_layer"),
        (dict(top_k=6, per_token=1, per_layer=9), "per_layer"),
    ],
)
def test_a_choice_outside_the_hosts_bounds_is_refused(kw, why):
    x, w, bias = _exact_case(0)
    with pytest.raises(Exception, match=why):
        es.score_gate(x, w, bias, [False] * 16, **kw)
```

- [ ] **Step 2: Commit the failing tests, push, run them on divix01**

```bash
cd /Users/dnikolaidis/.codex/worktrees/ram-prefetch/sglang-nvfp4
git add test/registered/unit/kernels/test_exl3_ram_prefetch_scorer.py
git commit -m "Test the RAM prefetch gate scorer against the replay's ranking

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push origin codex/dsv41-ram-prefetch
ssh divix01 bash -s <<'REMOTE'
cd /data/models/slang/nvfp4-work/wt-ram-prefetch && git fetch -q origin && git checkout -q --detach origin/codex/dsv41-ram-prefetch && git log -1 --oneline
PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 taskset -c 0-63 /data/models/slang/.venv/bin/python -m pytest test/registered/unit/kernels/test_exl3_ram_prefetch_scorer.py -q -p no:randomly 2>&1 | tail -5; echo "EXIT=${PIPESTATUS[0]}"
REMOTE
```
Expected: `EXIT=1`, `AttributeError: module 'sglang.kernels.ops.moe.expert_stream_transport' has no attribute 'score_gate'`.

- [ ] **Step 3: Write the scorer**

Create `python/sglang/kernels/jit/csrc/moe/expert_stream/host/gate_scorer.h`:

```cpp
// The RAM prefetch's gate scorer (spec docs/superpowers/specs/2026-10-08-dsv41-ram-prefetch-design.md, Phase 1): the
// next streamed layer's router on a record's staged input, ranked as the Phase 0 replay ranked it
// (analysis/dsv41-drive/prefetch-replay/verify_replay.py, gate_choice and issue's budget).
#pragma once

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace sglang::expert_stream {

// IEEE half to float, exact. In software: the host may be built without F16C (SGLANG_JIT_HOST_MARCH).
inline float half_to_float(uint16_t h) {
  const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
  uint32_t exponent = (h >> 10) & 0x1Fu;
  uint32_t mantissa = h & 0x3FFu;
  uint32_t bits;
  if (exponent == 0) {
    if (mantissa == 0) {
      bits = sign;
    } else {
      exponent = 127 - 15 + 1;
      while ((mantissa & 0x400u) == 0) {
        mantissa <<= 1;
        --exponent;
      }
      bits = sign | exponent << 23 | (mantissa & 0x3FFu) << 13;
    }
  } else if (exponent == 0x1F) {
    bits = sign | 0x7F800000u | mantissa << 13;
  } else {
    bits = sign | (exponent + 127 - 15) << 23 | mantissa << 13;
  }
  return std::bit_cast<float>(bits);
}

inline float bf16_to_float(uint16_t b) {
  return std::bit_cast<float>(static_cast<uint32_t>(b) << 16);
}

// torch's softplus (threshold 20), through log1p as the DSV4.1 router computes it (sqrtsoftplus_log1p).
inline float softplus(float z) {
  return z > 20.0f ? z : std::log1p(std::exp(z));
}

// Throws std::invalid_argument for a choice the scorer cannot make.
inline void check_gate_choice(int64_t experts, int64_t top_k, int64_t per_token, int64_t per_layer);

class GateScorer {
 public:
  static constexpr int kDepth = 12;       // each token's ranks walked: the replay's (verify_gate_rankings --depth)
  static constexpr int kMaxPerLayer = 8;  // Python mirror: ram_prefetch.MAX_PER_LAYER

  // Sizes the scratch once, so choose() allocates nothing.
  void reserve(int64_t tokens, int64_t hidden, int64_t experts) {
    x_.assign(static_cast<size_t>(tokens * hidden), 0.0f);
    score_.assign(static_cast<size_t>(tokens * experts), 0.0f);
    best_.assign(static_cast<size_t>(experts), 0.0f);
    order_.assign(static_cast<size_t>(experts), 0);
    picks_.reserve(static_cast<size_t>(experts));
  }

  // Scores `tokens` inputs (fp16, `x_token_bytes` apart from `x`) against the gate `w` (bf16 [experts, hidden]) and
  // `bias`, and writes up to per_layer experts to `out`, best margin first; returns how many. `skip[e]` nonzero passes
  // expert e over. Accumulates in fp32, in a fixed order.
  int choose(
      const uint8_t* x,
      int64_t tokens,
      int64_t x_token_bytes,
      const uint16_t* w,
      const float* bias,
      int64_t experts,
      int64_t hidden,
      int top_k,
      int per_token,
      int per_layer,
      const uint8_t* skip,
      int32_t* out) {
    for (int64_t t = 0; t < tokens; ++t) {
      const auto* row = reinterpret_cast<const uint16_t*>(x + t * x_token_bytes);
      for (int64_t h = 0; h < hidden; ++h)
        x_[t * hidden + h] = half_to_float(row[h]);
    }
    for (int64_t e = 0; e < experts; ++e) {
      const uint16_t* we = w + e * hidden;
      for (int64_t t = 0; t < tokens; ++t)
        score_[t * experts + e] = std::sqrt(softplus(dot(we, &x_[t * hidden], hidden))) + bias[e];
    }
    std::fill(best_.begin(), best_.begin() + experts, -std::numeric_limits<float>::infinity());
    const int64_t depth = std::min<int64_t>(kDepth, experts);
    for (int64_t t = 0; t < tokens; ++t) {
      const float* s = &score_[t * experts];
      std::iota(order_.begin(), order_.begin() + experts, 0);
      std::partial_sort(order_.begin(), order_.begin() + depth, order_.begin() + experts, [s](int32_t a, int32_t b) {
        return s[a] > s[b] || (s[a] == s[b] && a < b);
      });
      const float kth = s[order_[top_k - 1]];
      int picked = 0;
      for (int64_t i = 0; i < depth && picked < per_token; ++i) {
        const int32_t e = order_[i];
        if (skip[e]) continue;
        best_[e] = std::max(best_[e], s[e] - kth);
        ++picked;
      }
    }
    picks_.clear();
    for (int64_t e = 0; e < experts; ++e)
      if (best_[e] != -std::numeric_limits<float>::infinity()) picks_.push_back(static_cast<int32_t>(e));
    std::sort(picks_.begin(), picks_.end(), [this](int32_t a, int32_t b) {
      return best_[a] > best_[b] || (best_[a] == best_[b] && a < b);
    });
    const int n = static_cast<int>(std::min<size_t>(static_cast<size_t>(per_layer), picks_.size()));
    std::copy_n(picks_.begin(), n, out);
    return n;
  }

 private:
  // Sixteen partial sums in a fixed order: the compiler vectorizes the inner loop, and the result never depends on it.
  static float dot(const uint16_t* w, const float* x, int64_t n) {
    float acc[16] = {};
    int64_t h = 0;
    for (; h + 16 <= n; h += 16)
      for (int j = 0; j < 16; ++j)
        acc[j] += bf16_to_float(w[h + j]) * x[h + j];
    float sum = 0.0f;
    for (int j = 0; j < 16; ++j)
      sum += acc[j];
    for (; h < n; ++h)
      sum += bf16_to_float(w[h]) * x[h];
    return sum;
  }

  std::vector<float> x_;      // [tokens, hidden]
  std::vector<float> score_;  // [tokens, experts]
  std::vector<float> best_;   // [experts]: an expert's best margin over the tokens, -inf when not picked
  std::vector<int32_t> order_;
  std::vector<int32_t> picks_;
};

inline void check_gate_choice(int64_t experts, int64_t top_k, int64_t per_token, int64_t per_layer) {
  if (experts < 1 || experts > 0xFFFF) throw std::invalid_argument("the gate has 1..65535 experts");
  const int64_t depth = std::min<int64_t>(GateScorer::kDepth, experts);
  if (top_k < 1 || top_k > depth) throw std::invalid_argument("top_k must be in 1.." + std::to_string(depth));
  if (per_token < 1 || per_token > GateScorer::kDepth)
    throw std::invalid_argument("per_token must be in 1.." + std::to_string(GateScorer::kDepth));
  if (per_layer < 1 || per_layer > GateScorer::kMaxPerLayer)
    throw std::invalid_argument("per_layer must be in 1.." + std::to_string(GateScorer::kMaxPerLayer));
}

}  // namespace sglang::expert_stream
```

- [ ] **Step 4: Export the scorer**

In `ffi_test_exports.h`, add `#include "gate_scorer.h"` after `#include "ffi_exports.h"`, and after `pause_ns`:
```cpp
  // The RAM prefetch's gate scorer (gate_scorer.h) on given tensors: x uint8 [tokens, 2 * hidden] (fp16 rows), w uint8
  // [experts, 2 * hidden] (bf16), bias float32 [experts], skip uint8 [experts]; writes the chosen experts to out int64
  // [per_layer] and returns how many. Both builds: a pure function.
  static int64_t score_gate(
      TensorView x, TensorView w, TensorView bias, TensorView skip, int64_t top_k, int64_t per_token, int64_t per_layer,
      TensorView out) {
    using namespace host;
    auto cpu = SymbolicDevice{};
    auto T = SymbolicSize{"tokens"};
    auto E = SymbolicSize{"experts"};
    auto B = SymbolicSize{"row bytes"};
    expert_stream::verify_named("x", TensorMatcher({T, B}).with_dtype<uint8_t>().with_device<kDLCPU>(cpu), x);
    expert_stream::verify_named("w", TensorMatcher({E, B}).with_dtype<uint8_t>().with_device<kDLCPU>(cpu), w);
    expert_stream::verify_named("bias", TensorMatcher({E}).with_dtype<float>().with_device<kDLCPU>(cpu), bias);
    expert_stream::verify_named("skip", TensorMatcher({E}).with_dtype<uint8_t>().with_device<kDLCPU>(cpu), skip);
    expert_stream::verify_named("out", TensorMatcher({per_layer}).with_dtype<int64_t>().with_device<kDLCPU>(cpu), out);
    const int64_t experts = w.size(0);
    const int64_t hidden = x.size(1) / 2;
    if (x.size(1) % 2 != 0 || hidden < 1) throw std::runtime_error("score_gate: x rows are fp16");
    try {
      expert_stream::check_gate_choice(experts, top_k, per_token, per_layer);
    } catch (const std::invalid_argument& error) {
      throw std::runtime_error(std::string("score_gate: ") + error.what());
    }
    expert_stream::GateScorer scorer;
    scorer.reserve(x.size(0), hidden, experts);
    int32_t chosen[expert_stream::GateScorer::kMaxPerLayer];
    const int n = scorer.choose(
        static_cast<const uint8_t*>(x.data_ptr()), x.size(0), x.size(1), static_cast<const uint16_t*>(w.data_ptr()),
        static_cast<const float*>(bias.data_ptr()), experts, hidden, static_cast<int>(top_k),
        static_cast<int>(per_token), static_cast<int>(per_layer), static_cast<const uint8_t*>(skip.data_ptr()), chosen);
    auto* result = static_cast<int64_t*>(out.data_ptr());
    for (int i = 0; i < n; ++i)
      result[i] = chosen[i];
    return n;
  }
```
and in `EXPERT_STREAM_HOST_TEST_EXPORTS_OF`, after the `pause_ns` line add (moving the trailing backslash so the macro stays continuous):
```cpp
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_pause_ns, Exports::pause_ns);                         \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_score_gate, Exports::score_gate);
```

In `expert_stream_transport.py`, after `pause_ns`:
```python
def score_gate(
    x: torch.Tensor,
    w: torch.Tensor,
    bias: torch.Tensor,
    skip: Sequence[bool],
    *,
    top_k: int,
    per_token: int,
    per_layer: int,
    layout: str = "exl3",
    variant: Optional[str] = None,
) -> list[int]:
    """The RAM prefetch's gate scorer (``host/gate_scorer.h``): ``x`` fp16 ``[tokens, hidden]``, ``w`` bf16
    ``[experts, hidden]``, ``bias`` fp32 ``[experts]``, skipping ``skip``; returns the chosen experts, best first. Both
    builds."""
    if x.dtype != torch.float16 or w.dtype != torch.bfloat16 or bias.dtype != torch.float32:
        raise ValueError("score_gate takes an fp16 x, a bf16 w and an fp32 bias")
    out = torch.full((int(per_layer),), -1, dtype=torch.int64)
    n = int(
        _host_module(layout, variant).expert_stream_score_gate(
            x.contiguous().view(torch.uint8),
            w.contiguous().view(torch.uint8),
            bias.contiguous(),
            torch.tensor([bool(s) for s in skip], dtype=torch.uint8),
            int(top_k),
            int(per_token),
            int(per_layer),
            out,
        )
    )
    return out[:n].tolist()
```

- [ ] **Step 5: Commit, push, run**

```bash
cd /Users/dnikolaidis/.codex/worktrees/ram-prefetch/sglang-nvfp4
git add python/sglang/kernels/jit/csrc/moe/expert_stream/host/gate_scorer.h python/sglang/kernels/jit/csrc/moe/expert_stream/host/ffi_test_exports.h python/sglang/kernels/ops/moe/expert_stream_transport.py
git commit -m "Add the RAM prefetch gate scorer, ranked as the Phase 0 replay

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push origin codex/dsv41-ram-prefetch
ssh divix01 bash -s <<'REMOTE'
cd /data/models/slang/nvfp4-work/wt-ram-prefetch && git fetch -q origin && git checkout -q --detach origin/codex/dsv41-ram-prefetch && git log -1 --oneline
PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 taskset -c 0-63 /data/models/slang/.venv/bin/python -m pytest test/registered/unit/kernels/test_exl3_ram_prefetch_scorer.py test/registered/unit/kernels/test_expert_stream_prod_build_symbols.py -q -p no:randomly 2>&1 | tail -5; echo "EXIT=${PIPESTATUS[0]}"
REMOTE
```
Expected: `EXIT=0` (8 x 6 reference cases plus the rest).

- [ ] **Step 6: Mutant checks (a layer's budget exceeded; the margin's reference rank)**

```bash
ssh divix01 bash -s <<'REMOTE'
cd /data/models/slang/nvfp4-work/wt-ram-prefetch
H=python/sglang/kernels/jit/csrc/moe/expert_stream/host/gate_scorer.h
run() { PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 taskset -c 0-63 /data/models/slang/.venv/bin/python -m pytest test/registered/unit/kernels/test_exl3_ram_prefetch_scorer.py -q -p no:randomly 2>&1 | tail -3; echo "EXIT=${PIPESTATUS[0]}"; }
mutate() { /data/models/slang/.venv/bin/python - "$H" "$1" "$2" <<'PY'
import pathlib, sys
p, old, new = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3]
t = p.read_text(); assert t.count(old) == 1, t.count(old); p.write_text(t.replace(old, new))
PY
}
echo "== mutant: one row past the layer's budget"
mutate 'std::min<size_t>(static_cast<size_t>(per_layer), picks_.size())' 'std::min<size_t>(static_cast<size_t>(per_layer + 1), picks_.size())'
run; git checkout -- "$H"
echo "== mutant: the margin to the score above the top_k-th"
mutate 'const float kth = s[order_[top_k - 1]];' 'const float kth = s[order_[std::max(0, top_k - 2)]];'
run; git checkout -- "$H"
git status --short
echo "== restored"; run
REMOTE
```
Expected: a nonzero `EXIT` for each mutant (the budget mutant can also crash the run, since it writes one expert past `out`), empty status, `EXIT=0` restored.

---

### Task 5: The speculative step (pump mode)

**Files:**
- Modify: `python/sglang/kernels/jit/csrc/moe/expert_stream/host/ram_prefetch.h` (`RamPrefetchConfig`, `SpecJob`)
- Modify: `python/sglang/kernels/jit/csrc/moe/expert_stream/host/cpu_experts.h` (`BasicCpuExpertEngine::config()`)
- Modify: `python/sglang/kernels/jit/csrc/moe/expert_stream/host/ram_tier.h` (includes; `pump_demand` :280; `apply_gpu_hot` :566; `counters` :1142; `group_counters` :1154; `inject`-adjacent test hooks; new `enable_ram_prefetch`, `offer_spec`, `fill_spec_skip`, `serve_spec_job`, `spec_read`, `spec_count`, `spec_pump`, `inject_spec`; `TierFaults`; members `SpecGroup`, `SpecState`, `spec_`)
- Modify: `python/sglang/kernels/jit/csrc/moe/expert_stream/host/ffi_exports.h` (`enable_ram_prefetch`), `ffi_test_exports.h` (`spec_pump`, `inject_spec`)
- Modify: `python/sglang/kernels/ops/moe/expert_stream_transport.py` (`enable_ram_prefetch`, `spec_pump`, `inject_spec`, `TEST_ONLY_EXPORTS`)
- Modify: `python/sglang/test/dsv41_ram_prefetch_fixtures.py` (`LOGITS`, `gate`, `enable`, `write_x`, `trigger`)
- Test: `test/registered/unit/kernels/test_exl3_ram_prefetch_step.py`

**Interfaces:**
- Consumes: `SpecPool`, `pool_word`, `kPoolReading/kPoolLanded/kPoolEmpty`, `pool_->find/pooled_before/claimable_locked/next_landing` (Task 2); `GateScorer`, `check_gate_choice` (Task 4); counters `kSpec*` (Task 3); `prefetch_rig`, `load`, `forced` (Tasks 2-3).
- Produces: `struct RamPrefetchConfig { std::vector<int32_t> target, gate; const uint16_t* gates; const float* bias; int64_t gate_count, hidden; int top_k, per_token, per_layer; std::vector<std::vector<int>> cores; }`; `struct SpecJob { uint32_t seq; int64_t row; int64_t tokens; }`; `const CpuExpertConfig& BasicCpuExpertEngine::config() const`; `RamTier::enable_ram_prefetch(RamPrefetchConfig)`; private `RamTier::{SpecGroup, SpecState, spec_, offer_spec, fill_spec_skip, serve_spec_job, spec_read, spec_count}` (Task 6 drives `serve_spec_job` from threads and reads `SpecGroup::{turn, busy, idle, bell, ring, cores, thread}`); test hooks `bool RamTier::spec_pump(int g)`, `void RamTier::inject_spec(int64_t delay_ns, bool fail)`; FFI `expert_stream_enable_ram_prefetch(handle, targets int64 [rows,2], gates uint8 [n, E*H*2], bias f32 [n, E], cores int64 [groups, width], hidden, top_k, per_token, per_layer)`, test exports `expert_stream_spec_pump(handle, group) -> 0/1`, `expert_stream_inject_spec(handle, delay_ns, fail)`; Python `ExpertStreamHost.enable_ram_prefetch(targets, gates, bias, *, top_k, per_token, per_layer, cores)`, `spec_pump(group=0) -> bool`, `inject_spec(delay_s=0.0, fail=False)`; fixtures `LOGITS`, `gate(logits)`, `enable(rig, logits, *, top_k=2, per_token=1, per_layer=1, cores=None)`, `write_x(rig, row, tokens)`, `trigger(rig, *, tokens=1, resident=5)`.

- [ ] **Step 1: Extend the rig**

Append to `python/sglang/test/dsv41_ram_prefetch_fixtures.py`:

```python
# Token 0's logits over the one-node rig's six experts: expert 2 first, then 4, 0, 1, 3, 5.
LOGITS = [[30, 25, 40, 22, 35, 21]]


def gate(logits) -> tuple[torch.Tensor, torch.Tensor]:
    """A gate as bf16 [1, experts, HIDDEN] and its zero fp32 bias: token t's input is the unit vector e_t (write_x), so
    its logit for expert e is logits[t][e]. Logits above 20 keep softplus the identity: the score order is theirs."""
    experts = len(logits[0])
    w = torch.zeros((1, experts, HIDDEN), dtype=torch.bfloat16)
    for t, row in enumerate(logits):
        w[0, :, t] = torch.tensor(row, dtype=torch.bfloat16)
    return w, torch.zeros((1, experts), dtype=torch.float32)


def enable(rig: PrefetchRig, logits, *, top_k=2, per_token=1, per_layer=1, cores=None) -> None:
    """Row 0 targets row 1 with `logits`' gate; row 1, the last, targets nothing."""
    w, bias = gate(logits)
    targets = torch.tensor([[1, 0], [-1, -1]], dtype=torch.int64)
    rig.host.enable_ram_prefetch(
        targets,
        w,
        bias,
        top_k=top_k,
        per_token=per_token,
        per_layer=per_layer,
        cores=cores if cores is not None else [[] for _ in range(rig.host.nodes)],
    )


def write_x(rig: PrefetchRig, row: int, tokens: int) -> None:
    """Stages `row`'s input as the post kernel does: token t is the unit vector e_t (fp16); a multi-token row's table
    counts the live tokens."""
    stride = x_token_bytes()
    for t in range(tokens):
        x = torch.zeros(HIDDEN, dtype=torch.float16)
        x[t] = 1.0
        rig.x_rows[row, t * stride : t * stride + 2 * HIDDEN] = x.view(torch.uint8)
    if rig.tokens > 1:
        table = rig.tokens * stride
        rig.x_rows[row, table : table + 4] = torch.tensor([tokens], dtype=torch.int32).view(torch.uint8)


def trigger(rig: PrefetchRig, *, tokens: int = 1, resident: int = 5) -> SimRequest:
    """A captured post of row 0 with one CPU hit (`resident`, loaded first if absent), after staging row 0's input:
    the service hands it to every group's speculative thread."""
    if rig.host.mapping(0)[resident] < 0:
        load(rig, 0, [resident])
    write_x(rig, 0, tokens)
    req = rig.sim.post(0, [resident], captured=True, cpu_on=True)
    _served(rig, req)
    return req
```

- [ ] **Step 2: Write the failing tests**

Create `test/registered/unit/kernels/test_exl3_ram_prefetch_step.py`:

```python
"""The speculative step (spec 2026-10-08-dsv41-ram-prefetch-design, Phase 1, "The speculative thread"), driven on the
test's thread with spec_pump: a record that staged a CPU input feeds each group's ring; the step scores the target
row's gate, skips hot, mapped and earlier-pooled experts, keeps the layer's budget over both groups, reads its own
group's picks into the pool, and drops a candidate whose target record was served first (CPU, ChainSim)."""

import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.dsv41_ram_miss_fixtures import attached_host, ram_miss_setup, same_bytes
from sglang.test.dsv41_ram_prefetch_fixtures import (
    HALVES,
    LOGITS,
    enable,
    forced,
    gate,
    load,
    prefetch_rig,
    trigger,
    write_x,
)

register_cpu_ci(est_time=60, suite="base-a-test-cpu")


def _landed(host, row, group=None):
    return sorted(
        e["expert"] for e in host.spec_pool(row) if e["state"] == "landed" and (group is None or e["group"] == group)
    )


def test_a_cpu_record_reads_the_next_rows_top_pick_into_the_pool_and_the_target_swaps_it(tmp_path):
    """Mutant: the speculative read publishing the mirror -- red on mapping(1)[2] before the swap."""
    rig = prefetch_rig(tmp_path)
    try:
        enable(rig, LOGITS)
        trigger(rig)
        assert rig.host.spec_pump(0) and not rig.host.spec_pump(0)
        entry = next(e for e in rig.host.spec_pool(1) if e["state"] == "landed")
        assert entry["expert"] == 2
        assert rig.host.mapping(1)[2] == -1 and rig.sim.delta(1)[2] == []
        c = rig.host.counters()
        assert (c["spec_issued"], c["spec_landed"], c["spec_dropped"], c["spec_failed"]) == (1, 1, 0, 0)
        got, ref = rig.sim.read_slot(1, entry["slot"]), rig.setup.reference(1, [2])
        assert all(same_bytes(got[n], ref[n][0]) for n in got)
        rows = c["rows_read"]
        forced(rig, 1, [2])
        c = rig.host.counters()
        assert c["spec_used"] == 1 and c["rows_read"] == rows and rig.host.mapping(1)[2] == entry["slot"]
    finally:
        rig.host.stop()


def test_hot_mapped_and_earlier_pooled_experts_are_skipped(tmp_path):
    """2 is VRAM-hot, 4 mapped, 0 pooled for an earlier record: the pick is 1. Mutant: drop the hot skip -- red."""
    rig = prefetch_rig(tmp_path)
    try:
        enable(rig, LOGITS)
        load(rig, 1, [4])
        rig.host.set_hot(1, [2])
        rig.host.spec_place(1, 0)
        trigger(rig)
        assert rig.host.spec_pump(0)
        assert _landed(rig.host, 1) == [0, 1]
    finally:
        rig.host.stop()


@pytest.mark.parametrize("per_layer, picks", [(1, [2]), (2, [2, 4])])
def test_the_layer_reads_per_layer_rows_in_margin_order(tmp_path, per_layer, picks):
    """Two candidates per token, budget 1 or 2. Mutant (gate_scorer.h): per_layer + 1 -- red."""
    rig = prefetch_rig(tmp_path)
    try:
        enable(rig, LOGITS, per_token=2, per_layer=per_layer)
        trigger(rig)
        assert rig.host.spec_pump(0)
        assert _landed(rig.host, 1) == picks and rig.host.counters()["spec_issued"] == len(picks)
    finally:
        rig.host.stop()


@pytest.mark.parametrize("live, picks", [(2, [0, 2]), (1, [2])])
def test_each_live_token_adds_its_own_pick(tmp_path, live, picks):
    """Review Focus 1: a two-token row; token 0 prefers 2, token 1 prefers 0. With the table counting one live token,
    token 1's stale input is not scored."""
    rig = prefetch_rig(tmp_path, tokens=2)
    try:
        enable(rig, [LOGITS[0], [41, 21, 22, 23, 24, 25]], per_layer=2)
        trigger(rig, tokens=live)
        assert rig.host.spec_pump(0)
        assert _landed(rig.host, 1) == picks
    finally:
        rig.host.stop()


def test_a_candidate_whose_target_record_was_served_first_is_dropped(tmp_path):
    rig = prefetch_rig(tmp_path)
    try:
        enable(rig, LOGITS)
        trigger(rig)
        target = rig.sim.post(1, [])  # row 1's record, served before the step runs
        assert rig.host.pump() == 1 and rig.sim.wait_handled(target)
        assert rig.host.spec_pump(0)
        c = rig.host.counters()
        assert (c["spec_dropped"], c["spec_issued"]) == (1, 0) and _landed(rig.host, 1) == []
    finally:
        rig.host.stop()


def test_a_failed_speculative_read_empties_its_entry_and_the_demand_reads_the_row(tmp_path):
    rig = prefetch_rig(tmp_path)
    try:
        enable(rig, LOGITS)
        rig.host.inject_spec(fail=True)
        trigger(rig)
        assert rig.host.spec_pump(0)
        c = rig.host.counters()
        assert (c["spec_issued"], c["spec_failed"], c["spec_landed"], c["read_errors"]) == (1, 1, 0, 0)
        assert all(e["state"] == "empty" for e in rig.host.spec_pool(1))
        rows = c["rows_read"]
        forced(rig, 1, [2])
        c = rig.host.counters()
        assert c["rows_read"] == rows + 1 and c["spec_used"] == 0
    finally:
        rig.host.stop()


def test_a_reclaimed_entry_is_never_swapped_in_for_its_old_expert(tmp_path):
    """Review Focus 3: one entry, holding 0 for an earlier record; the step's pick 2 reclaims it. A forced miss on 0
    then reads 0's row."""
    rig = prefetch_rig(tmp_path, share=1)
    try:
        enable(rig, LOGITS)
        rig.host.spec_place(1, 0)
        trigger(rig)
        assert rig.host.spec_pump(0)
        assert _landed(rig.host, 1) == [2]
        rows = rig.host.counters()["rows_read"]
        forced(rig, 1, [0])
        c = rig.host.counters()
        assert c["rows_read"] == rows + 1 and c["spec_used"] == 0
        got, ref = rig.sim.read_slot(1, rig.host.mapping(1)[0]), rig.setup.reference(1, [0])
        assert all(same_bytes(got[n], ref[n][0]) for n in got)
    finally:
        rig.host.stop()


def test_a_full_ring_drops_jobs_and_never_blocks_the_service(tmp_path):
    """Review Focus 2: 70 CPU records with no step run; the ring holds 64."""
    rig = prefetch_rig(tmp_path)
    try:
        enable(rig, LOGITS)
        for _ in range(70):
            trigger(rig)
        assert rig.host.counters()["spec_dropped"] == 6
    finally:
        rig.host.stop()


@pytest.mark.parametrize("per_layer, by_group", [(2, {0: [2], 1: [3]}), (1, {0: [2], 1: []})])
def test_both_groups_rank_alike_and_each_reads_only_its_own_experts(tmp_path, per_layer, by_group):
    """Review Focus 5: the trigger's only CPU lane is expert 5, group 1's, yet both groups get the job. Expert 2 is
    group 0's, 3 group 1's. Mutants: read every pick on any group -- red; skip an expert pooled for this same record
    (pooled_before ignoring the seq) -- red at budget 1, since group 1 then reads 3."""
    rig = prefetch_rig(tmp_path, nodes=2, share=1)
    try:
        enable(rig, [[30, 25, 40, 39, 22, 21, 23, 24]], per_token=2, per_layer=per_layer)
        trigger(rig, resident=5)
        assert rig.host.spec_pump(0) and rig.host.spec_pump(1)
        for g in (0, 1):
            assert _landed(rig.host, 1, g) == by_group[g]
            assert rig.host.group_counters(g)["spec_issued"] == len(by_group[g])
            lo, hi = HALVES[g][1]
            assert all(lo <= e["slot"] < hi for e in rig.host.spec_pool(1) if e["group"] == g)
    finally:
        rig.host.stop()


def test_records_without_a_cpu_lane_or_without_a_target_feed_no_job(tmp_path):
    rig = prefetch_rig(tmp_path)
    try:
        enable(rig, LOGITS)
        load(rig, 0, [1])  # uncaptured: no CPU lane, nothing staged
        assert not rig.host.spec_pump(0)
        load(rig, 1, [3])
        write_x(rig, 1, 1)
        req = rig.sim.post(1, [3], captured=True, cpu_on=True)  # a CPU lane on row 1, which targets nothing
        assert rig.host.pump() == 1 and rig.sim.copy_wait(req)
        assert not rig.host.spec_pump(0)
    finally:
        rig.host.stop()


def test_enable_refuses_what_the_step_could_not_serve(tmp_path):
    for name in ("a", "b", "c"):
        (tmp_path / name).mkdir()
    rig = prefetch_rig(tmp_path / "a", pool=False)
    try:
        with pytest.raises(RuntimeError, match="reserve_spec_pool first"):
            enable(rig, LOGITS)
    finally:
        rig.host.stop()
    rig = prefetch_rig(tmp_path / "b")
    try:
        for kw, why in [
            (dict(per_layer=0), "per_layer must be in 1..8"),
            (dict(per_token=13), "per_token must be in 1..12"),
            (dict(top_k=7), "top_k must be in 1..6"),
        ]:
            with pytest.raises(RuntimeError, match=why):
                enable(rig, LOGITS, **kw)
        w, bias = gate(LOGITS)
        with pytest.raises(RuntimeError, match="one entry per streamed row"):
            rig.host.enable_ram_prefetch(
                torch.tensor([[1, 0], [-1, -1], [-1, -1]]), w, bias, top_k=2, per_token=1, per_layer=1, cores=[[]]
            )
        wide = torch.zeros((1, 6, 16), dtype=torch.bfloat16)
        with pytest.raises(RuntimeError, match="hidden size 16 is not the CPU rows' 8"):
            rig.host.enable_ram_prefetch(
                torch.tensor([[1, 0], [-1, -1]]), wide, bias, top_k=2, per_token=1, per_layer=1, cores=[[]]
            )
        with pytest.raises(ValueError, match="one core list per NUMA group"):
            enable(rig, LOGITS, cores=[[], []])
        enable(rig, LOGITS)
        with pytest.raises(RuntimeError, match="already enabled"):
            enable(rig, LOGITS)
    finally:
        rig.host.stop()
    s = ram_miss_setup(tmp_path / "c", capacity=7)
    from sglang.kernels.ops.moe.expert_lease_block import wire_layout
    from sglang.kernels.ops.moe.expert_stream_transport import new_page

    host = attached_host(s, new_page(pin=False, wire=wire_layout(8)), k=3)
    try:
        host.reserve_spec_pool(2)
        w, bias = gate(LOGITS)
        with pytest.raises(RuntimeError, match="CPU experts on group 0"):
            host.enable_ram_prefetch(
                torch.tensor([[1, 0], [-1, -1]]), w, bias, top_k=2, per_token=1, per_layer=1, cores=[[]]
            )
    finally:
        host.stop()
```


- [ ] **Step 3: Commit the failing tests, push, run them on divix01**

```bash
cd /Users/dnikolaidis/.codex/worktrees/ram-prefetch/sglang-nvfp4
git add python/sglang/test/dsv41_ram_prefetch_fixtures.py test/registered/unit/kernels/test_exl3_ram_prefetch_step.py
git commit -m "Test the RAM prefetch's speculative step in pump mode

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push origin codex/dsv41-ram-prefetch
ssh divix01 bash -s <<'REMOTE'
cd /data/models/slang/nvfp4-work/wt-ram-prefetch && git fetch -q origin && git checkout -q --detach origin/codex/dsv41-ram-prefetch && git log -1 --oneline
PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 taskset -c 0-63 /data/models/slang/.venv/bin/python -m pytest test/registered/unit/kernels/test_exl3_ram_prefetch_step.py -q -p no:randomly 2>&1 | tail -8; echo "EXIT=${PIPESTATUS[0]}"
REMOTE
```
Expected: `EXIT=1`, `AttributeError: 'ExpertStreamHost' object has no attribute 'enable_ram_prefetch'`.

- [ ] **Step 4: Add the configuration and job types, and the CPU engine's config accessor**

Append to `ram_prefetch.h`, before the closing namespace, and add `#include <vector>` to its includes:
```cpp
// The RAM prefetch's settings (RamTier::enable_ram_prefetch).
struct RamPrefetchConfig {
  std::vector<int32_t> target;  // per source row: the next streamed layer's row, or -1 (none, or no biased gate)
  std::vector<int32_t> gate;    // per source row: the target's index in gates and bias, or -1
  const uint16_t* gates = nullptr;  // bf16 [gate_count, experts, hidden], host memory the caller keeps alive
  const float* bias = nullptr;      // fp32 [gate_count, experts]
  int64_t gate_count = 0;
  int64_t hidden = 0;
  int top_k = 0;
  int per_token = 0;
  int per_layer = 0;
  std::vector<std::vector<int>> cores;  // per group: its speculative thread's cores; empty inherits the caller's
};

// One record handed from a group's service thread to its speculative thread: a record of `row` that staged `tokens`
// live inputs.
struct SpecJob {
  uint32_t seq = 0;
  int64_t row = 0;
  int64_t tokens = 1;
};
```

In `cpu_experts.h`, in `BasicCpuExpertEngine`'s public section after `tokens()`:
```cpp
  /// The tables and geometry it was enabled with: the RAM prefetch's scorer reads the staged inputs through them.
  const CpuExpertConfig& config() const {
    return config_;
  }
```

- [ ] **Step 5: Add the speculative state, enabling and feeding to `RamTier`**

In `ram_tier.h`, add `#include "gate_scorer.h"` after `#include "ram_prefetch.h"`.

At the start of the private section (directly after `private:`, ~line 1159), so every private helper's signature
below can name them:
```cpp
  // One group's speculative state (enable_ram_prefetch): its job ring from the group's service thread, the turn on the
  // group's reader, its busy episode for the watchdog and its own counter block (one writer each).
  struct SpecGroup {
    explicit SpecGroup(const std::string& name) : trace(name) {}
    SpscRing<SpecJob, 64> ring;  // arbitrary: one job per CPU record, the thread drains it every ~layer
    Doorbell bell;
    std::mutex turn;  // the group's reader: one demand read (read_rows) or one speculative read at a time
    std::atomic<uint64_t> busy{0};
    uint64_t episodes = 0;
    std::atomic<bool> idle{true};
    GateScorer scorer;
    std::vector<uint8_t> skip;
    std::vector<uint8_t> packed;
    std::vector<int> cores;
    std::thread thread;
    LineCounters<kCounterCount> core;
    [[no_unique_address]] JobTrace<Build::kMetrics> trace;
  };
  struct SpecState {
    RamPrefetchConfig config;
    const uint8_t* x_base = nullptr;  // the CPU rows' staged inputs (CpuExpertConfig)
    int64_t x_stride = 0;
    int64_t x_token_bytes = 0;
    int64_t tokens_max = 1;
    std::vector<std::unique_ptr<SpecGroup>> groups;
    std::atomic<bool> stop{false};
    std::atomic<bool> hold{false};
  };
```
and among the data members, directly after `pool_`:
```cpp
  std::unique_ptr<SpecState> spec_;  // null with the RAM prefetch off
```

Add to `TierFaults`, after `group_stall_ns`:
```cpp
    std::atomic<int64_t> spec_delay_ns{0};  // inject_spec: each speculative read sleeps first, holding the turn
    std::atomic<bool> spec_fail{false};
```

Public, after `spec_pool`:
```cpp
  // Enables the RAM prefetch over the reserved pool: per group a speculative thread (RamThread starts it with the
  // service threads) fed by the group's service with every record that staged a CPU input. Needs CPU experts on every
  // group, whose records stage that input. On the owner, before the service thread starts.
  void enable_ram_prefetch(RamPrefetchConfig config) {
    std::lock_guard<std::mutex> caller(caller_mutex_);
    require_owner("enable_ram_prefetch");
    const std::string prefix = error_prefix<Layout>() + "RAM prefetch: ";
    if (threaded_.load()) throw std::runtime_error(prefix + "enable it before the service thread starts");
    if (spec_ != nullptr) throw std::runtime_error(prefix + "it is already enabled");
    if (pool_ == nullptr) throw std::runtime_error(prefix + "reserve_spec_pool first");
    if (static_cast<int64_t>(config.target.size()) != layers_ || static_cast<int64_t>(config.gate.size()) != layers_)
      throw std::runtime_error(prefix + "the target table has one entry per streamed row");
    for (int64_t row = 0; row < layers_; ++row) {
      const int32_t target = config.target[row], gate = config.gate[row];
      if ((target < 0) != (gate < 0) || target >= layers_ || target == row || gate >= config.gate_count)
        throw std::runtime_error(
            prefix + "row " + std::to_string(row) + " names target " + std::to_string(target) + " with gate " +
            std::to_string(gate));
    }
    try {
      check_gate_choice(experts_, config.top_k, config.per_token, config.per_layer);
    } catch (const std::invalid_argument& error) {
      throw std::runtime_error(prefix + error.what());
    }
    if (config.gates == nullptr || config.bias == nullptr || config.gate_count < 1)
      throw std::runtime_error(prefix + "it needs at least one gate");
    if (static_cast<int>(config.cores.size()) != groups())
      throw std::runtime_error(prefix + "one core list per NUMA group");
    for (int g = 0; g < groups(); ++g)
      if (dist_.group(g).cpu == nullptr)
        throw std::runtime_error(
            prefix + "it needs CPU experts on group " + std::to_string(g) + ", whose records stage the scorer's input");
    const CpuExpertConfig& cpu = dist_.group(0).cpu->config();
    if (config.hidden != cpu.hidden)
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
      group->scorer.reserve(spec->tokens_max, config.hidden, experts_);
      group->skip.assign(static_cast<size_t>(experts_), 0);
      group->packed.reserve(1);
      group->cores = config.cores[g];
      spec->groups.push_back(std::move(group));
    }
    spec->config = std::move(config);
    spec_ = std::move(spec);
  }

  // Test only (InstrBuild): serves group g's next speculative job on the caller, as its speculative thread would;
  // false when the ring is empty. Refused while the service thread runs.
  bool spec_pump(int g)
    requires(Build::kFaults)
  {
    std::lock_guard<std::mutex> caller(caller_mutex_);
    if (spec_ == nullptr) throw std::runtime_error(error_prefix<Layout>() + "the RAM prefetch is not enabled");
    if (threaded_.load()) throw std::runtime_error(error_prefix<Layout>() + "spec_pump while the service thread runs");
    if (g < 0 || g >= groups()) throw std::runtime_error(error_prefix<Layout>() + "no NUMA group " + std::to_string(g));
    SpecJob job;
    if (!spec_->groups[g]->ring.pop(&job)) return false;
    serve_spec_job(g, job);
    return true;
  }

  // Test only (InstrBuild; ProdBuild throws): each speculative read sleeps `delay_ns` first, holding the reader's turn,
  // and with `fail` is reported failed without reading.
  void inject_spec(int64_t delay_ns, bool fail) {
    if constexpr (!Build::kFaults) {
      (void)delay_ns, (void)fail;
      test_only("inject_spec");
    } else {
      faults_.spec_delay_ns.store(delay_ns);
      faults_.spec_fail.store(fail);
    }
  }
```

In `pump_demand`, replace `if (!skip) handle_record(group, request);` with:
```cpp
      if (!skip) {
        handle_record(group, request);
        if (spec_ != nullptr) offer_spec(group, request);
      }
```

Replace `apply_gpu_hot` (with its comment) with:
```cpp
  // Replaces the row's hot set with the record's bitmap, for the group's own experts. Relaxed atomic stores: the RAM
  // prefetch's scorers read every expert's flag from any group's thread (fill_spec_skip).
  void apply_gpu_hot(Group& group, const Request& request) {
    Tier& tier = tiers_[request.row];
    for (int64_t expert = group.index; expert < experts_; expert += Wire::kNodes)
      std::atomic_ref<uint8_t>(tier.hot[expert])
          .store((request.hot_bitmap[expert / 8] >> (expert % 8)) & 1, std::memory_order_relaxed);
  }
```

Replace `counters` and `group_counters` with:
```cpp
  // Writes every counter with relaxed reads: a core counter is the sum of its writers' blocks (each word has one
  // writer; kSpinCpu is group 0's alone), a metric is stats_'s (always 0 in ProdBuild; Python reports only the core
  // counters of a production host).
  void counters(int64_t* out) const {
    for (int i = 0; i < kCounterCount; ++i) {
      int64_t core = 0;
      for (int g = 0; g < dist_.size(); ++g) {
        if (i == kSpinCpu && g > 0) break;
        core += dist_.group(g).core.get(i);
      }
      if (spec_ != nullptr)
        for (const auto& spec : spec_->groups)
          core += spec->core.get(i);
      out[i] = core + copy_core_.get(i) + stats_.get(i);
    }
  }

  // Group g's own core counters: its service thread's block and its speculative thread's.
  void group_counters(int g, int64_t* out) const {
    for (int i = 0; i < kCounterCount; ++i)
      out[i] = dist_.group(g).core.get(i) + (spec_ != nullptr ? spec_->groups[g]->core.get(i) : 0);
  }
```

Private, after `take_pooled_locked`:
```cpp
  template <Counter K>
  void spec_count(SpecGroup& spec, int64_t n = 1) {
    count_into<K>(spec.core, n);
  }

  // Service thread, after a record: hands it to the group's speculative thread when it staged a CPU input (some lane
  // is the CPU's) and its row has a target. A full ring drops it, counted: the service never waits for the scorer.
  void offer_spec(Group& group, const Request& request) {
    if (request.row < 0 || request.row >= layers_ || spec_->config.target[request.row] < 0) return;
    bool staged = false;
    for (const Lane& lane : request.lanes)
      staged |= lane.kind == Wire::kKindHitCpu || lane.kind == Wire::kKindMissCpu;
    if (!staged) return;
    int64_t tokens = 1;
    if (spec_->tokens_max > 1) {
      uint32_t live;
      std::memcpy(&live, spec_->x_base + request.row * spec_->x_stride + spec_->tokens_max * spec_->x_token_bytes, 4);
      tokens = live;
      if (tokens < 1 || tokens > spec_->tokens_max) {
        count<kSpecDropped>(group);
        return;
      }
    }
    SpecGroup& spec = *spec_->groups[group.index];
    if (!spec.ring.push(SpecJob{request.seq, request.row, tokens})) {
      count<kSpecDropped>(group);
      return;
    }
    spec.bell.ring();
  }

  // The scorer's skips for `target`: an expert VRAM-hot there, mapped there (the host mirror), or pooled for an earlier
  // record. Lock-free loads of words other groups' threads write; a race costs at most a boundary candidate.
  void fill_spec_skip(int64_t target, uint32_t target_seq, uint8_t* skip) {
    Tier& tier = tiers_[target];
    for (int64_t expert = 0; expert < experts_; ++expert) {
      const bool hot = std::atomic_ref<uint8_t>(tier.hot[expert]).load(std::memory_order_relaxed) != 0;
      const bool mapped = __atomic_load_n(map_ + target * experts_ + expert, __ATOMIC_ACQUIRE) >= 0;
      skip[expert] = hot || mapped || pool_->pooled_before(target, static_cast<int32_t>(expert), target_seq);
    }
  }

  // Speculative thread (or spec_pump): scores `job`'s target row and reads this group's picks into the pool. Both
  // groups compute the same list from the same input and state, so the layer's budget holds over both.
  void serve_spec_job(int g, const SpecJob& job) {
    SpecGroup& spec = *spec_->groups[g];
    const RamPrefetchConfig& config = spec_->config;
    const int64_t target = config.target[job.row];
    // One record per streamed layer call, and a target is the next consecutive layer: its record is the next seq.
    const uint32_t target_seq = skip_zero(job.seq + 1u);
    if (reached(dist_.group(g).handled.load(std::memory_order_acquire), target_seq)) {
      spec_count<kSpecDropped>(spec);
      return;
    }
    fill_spec_skip(target, target_seq, spec.skip.data());
    const int64_t gate = config.gate[job.row];
    int32_t chosen[GateScorer::kMaxPerLayer];
    const int n = spec.scorer.choose(
        spec_->x_base + job.row * spec_->x_stride, job.tokens, spec_->x_token_bytes,
        config.gates + gate * experts_ * config.hidden, config.bias + gate * experts_, experts_, config.hidden,
        config.top_k, config.per_token, config.per_layer, spec.skip.data(), chosen);
    for (int i = 0; i < n; ++i)
      if (Wire::home(chosen[i]) == g) spec_read(g, target, target_seq, chosen[i]);
  }

  // Reads `expert`'s row of `target` into a pool entry of group g under the group's reader turn, so a demand read
  // waits for at most this one row. Dropped when the target record was served first, or the expert was mapped or
  // pooled meanwhile.
  void spec_read(int g, int64_t target, uint32_t target_seq, int32_t expert) {
    SpecGroup& spec = *spec_->groups[g];
    Group& group = dist_.group(g);
    std::lock_guard<std::mutex> turn(spec.turn);
    if (reached(group.handled.load(std::memory_order_acquire), target_seq) ||
        __atomic_load_n(map_ + target * experts_ + expert, __ATOMIC_ACQUIRE) >= 0 ||
        pool_->find(target, g, expert) >= 0) {
      spec_count<kSpecDropped>(spec);
      return;
    }
    int i = -1;
    int64_t slot = -1;
    {
      std::lock_guard<std::mutex> lock(pool_->mutex(g));
      i = pool_->claimable_locked(target, g);
      if (i < 0) {
        spec_count<kSpecDropped>(spec);
        return;
      }
      PoolEntry& entry = pool_->entry(target, g, i);
      slot = entry.slot;
      entry.for_seq.store(target_seq, std::memory_order_relaxed);
      entry.word.store(pool_word(kPoolReading, expert), std::memory_order_release);
    }
    spec_count<kSpecIssued>(spec);
    spec.busy.store(++spec.episodes, std::memory_order_release);
    bool fail = false;
    if constexpr (Build::kFaults) {
      if (const int64_t ns = faults_.spec_delay_ns.load()) fault_delay(ns);
      fail = faults_.spec_fail.load();
    }
    int result = 0;
    if (!fail) {
      try {
        result = group.reader.read(
            target, std::span<const int32_t>(&expert, 1), std::span<const int64_t>(&slot, 1), 1,
            [](size_t) { return false; }, nullptr, &spec.packed);
      } catch (const std::exception& error) {
        std::fprintf(stderr, "ERROR %sspeculative read: %s\n", error_prefix<Layout>().c_str(), error.what());
      }
    }
    _mm_sfence();  // the row's bytes before the landed word
    spec.busy.store(0, std::memory_order_release);
    PoolEntry& entry = pool_->entry(target, g, i);
    if (result == 1) {
      entry.landed.store(pool_->next_landing(), std::memory_order_relaxed);
      entry.word.store(pool_word(kPoolLanded, expert), std::memory_order_release);
      spec_count<kSpecLanded>(spec);
    } else {
      entry.word.store(kPoolEmpty, std::memory_order_release);
      spec_count<kSpecFailed>(spec);
    }
  }
```

- [ ] **Step 6: Export enabling and the test hooks**

In `ffi_exports.h`, after `spec_pool`:
```cpp
  // Enables the RAM prefetch (RamTier::enable_ram_prefetch). `targets` int64 [rows, 2]: per source row its target row
  // and that row's gate index, or -1 -1. `gates` uint8 [n, experts * hidden * 2] (bf16 [experts, hidden] each) and
  // `bias` float32 [n, experts], host memory the caller keeps alive. `cores` int64 [groups, width]: each group's
  // speculative thread's cores, -1 padding (none: the caller's affinity). Cores 64-71 are refused.
  static void enable_ram_prefetch(
      int64_t handle,
      TensorView targets,
      TensorView gates,
      TensorView bias,
      TensorView cores,
      int64_t hidden,
      int64_t top_k,
      int64_t per_token,
      int64_t per_layer) {
    using namespace host;
    auto cpu = SymbolicDevice{};
    auto host_mem = SymbolicDevice{};
    auto host_bias = SymbolicDevice{};
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
    const auto tier = find(handle);
    if (hidden < 1 || bias.size(1) != tier->experts() || gates.size(1) != tier->experts() * hidden * 2)
      throw std::runtime_error(
          error_prefix<Layout>() + "RAM prefetch: the gates are not bf16 [n, experts, hidden] with an fp32 bias per expert");
    expert_stream::RamPrefetchConfig config;
    const auto* t = static_cast<const int64_t*>(targets.data_ptr());
    for (int64_t row = 0; row < targets.size(0); ++row) {
      config.target.push_back(static_cast<int32_t>(t[2 * row]));
      config.gate.push_back(static_cast<int32_t>(t[2 * row + 1]));
    }
    config.gates = static_cast<const uint16_t*>(gates.data_ptr());
    config.bias = static_cast<const float*>(bias.data_ptr());
    config.gate_count = gates.size(0);
    config.hidden = hidden;
    config.top_k = static_cast<int>(top_k);
    config.per_token = static_cast<int>(per_token);
    config.per_layer = static_cast<int>(per_layer);
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
and in `EXPERT_STREAM_HOST_EXPORTS` after the `spec_pool` line:
```cpp
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_enable_ram_prefetch, Exports::enable_ram_prefetch); \
```

In `ffi_test_exports.h`, after `spec_place`:
```cpp
  // Test only (RamTier::spec_pump): serves group `group`'s next speculative job on the caller; 1 if it served one.
  static int64_t spec_pump(int64_t handle, int64_t group) {
    if constexpr (!Build::kFaults) {
      test_only("spec_pump");
    } else {
      return find(handle)->spec_pump(static_cast<int>(group)) ? 1 : 0;
    }
  }

  // Test only (RamTier::inject_spec): each speculative read sleeps `delay_ns` first, and with `fail` fails. InstrBuild.
  static void inject_spec(int64_t handle, int64_t delay_ns, int64_t fail) {
    if constexpr (!Build::kFaults) {
      test_only("inject_spec");
    } else {
      find(handle)->inject_spec(delay_ns, fail != 0);
    }
  }
```
and in `EXPERT_STREAM_HOST_TEST_EXPORTS_OF`, after the `spec_place` line:
```cpp
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_spec_pump, Exports::spec_pump);                       \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_inject_spec, Exports::inject_spec);                   \
```

In `expert_stream_transport.py`, add `"spec_pump",` and `"inject_spec",` to `TEST_ONLY_EXPORTS` after `"spec_place",`; and in `ExpertStreamHost` after `spec_pool`:
```python
    def enable_ram_prefetch(
        self,
        targets: torch.Tensor,
        gates: torch.Tensor,
        bias: torch.Tensor,
        *,
        top_k: int,
        per_token: int,
        per_layer: int,
        cores: Sequence[Sequence[int]],
    ) -> None:
        """Enable the RAM prefetch over the pool ``reserve_spec_pool`` took, after ``enable_cpu_experts`` and before
        the thread. ``targets`` int64 ``[layers, 2]``: per source row its target row and gate index, or (-1, -1);
        ``gates`` bf16 ``[n, experts, hidden]`` and ``bias`` fp32 ``[n, experts]`` are host tensors this host keeps
        alive; ``cores`` is each NUMA group's speculative-thread core list (empty: the caller's affinity)."""
        from sglang.srt.layers.moe.cpu_experts.threading_config import check_not_reserved

        if gates.dtype != torch.bfloat16 or gates.dim() != 3 or gates.device.type != "cpu" or not gates.is_contiguous():
            raise ValueError("gates must be a contiguous host bf16 [n, experts, hidden] tensor")
        if bias.dtype != torch.float32 or tuple(bias.shape) != tuple(gates.shape[:2]) or bias.device.type != "cpu":
            raise ValueError("bias must be a host fp32 [n, experts] tensor")
        if len(cores) != self.nodes:
            raise ValueError(f"one core list per NUMA group ({self.nodes}), got {len(cores)}")
        table = torch.full((self.nodes, max(1, max(len(own) for own in cores))), -1, dtype=torch.int64)
        for g, own in enumerate(cores):
            for j, core in enumerate(own):
                check_not_reserved(int(core))
                table[g, j] = int(core)
        bias = bias.contiguous()
        self._module.expert_stream_enable_ram_prefetch(
            self.handle,
            targets.to(torch.int64).contiguous(),
            gates.view(gates.shape[0], -1).view(torch.uint8),
            bias,
            table,
            int(gates.shape[2]),
            int(top_k),
            int(per_token),
            int(per_layer),
        )
        self.ram_prefetch_tensors = (gates, bias)

    def spec_pump(self, group: int = 0) -> bool:
        """Test only: serve NUMA group ``group``'s next speculative job on the calling thread (no service thread)."""
        _refuse_test_only("spec_pump", self.variant)
        return bool(self._module.expert_stream_spec_pump(self.handle, int(group)))

    def inject_spec(self, delay_s: float = 0.0, fail: bool = False) -> None:
        """Test only: each speculative read sleeps ``delay_s`` first, holding the reader's turn; with ``fail`` it is
        reported failed without reading."""
        _refuse_test_only("inject_spec", self.variant)
        self._module.expert_stream_inject_spec(self.handle, int(delay_s * 1e9), int(fail))
```

- [ ] **Step 7: Commit, push, run the step tests and the files the record path touches**

```bash
cd /Users/dnikolaidis/.codex/worktrees/ram-prefetch/sglang-nvfp4
git add python/sglang/kernels/jit/csrc/moe/expert_stream/host/ram_prefetch.h python/sglang/kernels/jit/csrc/moe/expert_stream/host/cpu_experts.h python/sglang/kernels/jit/csrc/moe/expert_stream/host/ram_tier.h python/sglang/kernels/jit/csrc/moe/expert_stream/host/ffi_exports.h python/sglang/kernels/jit/csrc/moe/expert_stream/host/ffi_test_exports.h python/sglang/kernels/ops/moe/expert_stream_transport.py
git commit -m "Score the next layer's gate and read the pick into the pool, per group

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push origin codex/dsv41-ram-prefetch
ssh divix01 bash -s <<'REMOTE'
cd /data/models/slang/nvfp4-work/wt-ram-prefetch && git fetch -q origin && git checkout -q --detach origin/codex/dsv41-ram-prefetch && git log -1 --oneline
PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 taskset -c 0-63 /data/models/slang/.venv/bin/python -m pytest test/registered/unit/kernels/test_exl3_ram_prefetch_step.py test/registered/unit/kernels/test_exl3_ram_prefetch_swap.py test/registered/unit/kernels/test_exl3_ram_prefetch_pool.py test/registered/unit/kernels/test_exl3_ram_miss_cpu_experts.py test/registered/unit/kernels/test_exl3_ram_miss_numa_groups.py test/registered/unit/kernels/test_exl3_ram_miss_slot_map.py test/registered/unit/kernels/test_expert_stream_prod_build_symbols.py -q -p no:randomly 2>&1 | tail -5; echo "EXIT=${PIPESTATUS[0]}"
REMOTE
```
Expected: `EXIT=0`.

- [ ] **Step 8: Mutant checks (spec Testing list: the device map naming a kSpec slot; a candidate read on the wrong group; plus the same-record pool skip)**

```bash
ssh divix01 bash -s <<'REMOTE'
cd /data/models/slang/nvfp4-work/wt-ram-prefetch
T=python/sglang/kernels/jit/csrc/moe/expert_stream/host/ram_tier.h
P=python/sglang/kernels/jit/csrc/moe/expert_stream/host/ram_prefetch.h
run() { PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 taskset -c 0-63 /data/models/slang/.venv/bin/python -m pytest test/registered/unit/kernels/test_exl3_ram_prefetch_step.py -q -p no:randomly 2>&1 | tail -3; echo "EXIT=${PIPESTATUS[0]}"; }
mutate() { /data/models/slang/.venv/bin/python - "$1" "$2" "$3" <<'PY'
import pathlib, sys
p, old, new = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3]
t = p.read_text(); assert t.count(old) == 1, t.count(old); p.write_text(t.replace(old, new))
PY
}
echo "== mutant: a speculative row on the device's map before its swap"
mutate "$T" '      spec_count<kSpecLanded>(spec);' '      spec_count<kSpecLanded>(spec);
      publish_mirror(target, expert, static_cast<int32_t>(slot));'
run; git checkout -- "$T"
echo "== mutant: a candidate read on the wrong group"
mutate "$T" '      if (Wire::home(chosen[i]) == g) spec_read(g, target, target_seq, chosen[i]);' '      spec_read(g, target, target_seq, chosen[i]);'
run; git checkout -- "$T"
echo "== mutant: an expert pooled for this same record is skipped"
mutate "$P" '      if (i >= 0 && entry(row, g, i).for_seq.load(std::memory_order_relaxed) != seq) return true;' '      if (i >= 0) return true;'
run; git checkout -- "$P"
git status --short
echo "== restored"; run
REMOTE
```
Expected: `EXIT=1` after each mutant (red on `test_a_cpu_record_reads_...`, `test_both_groups_rank_alike_...[2-...]` and `[1-...]` respectively), empty status, `EXIT=0` restored.

---

### Task 6: The speculative threads

**Files:**
- Modify: `python/sglang/kernels/jit/csrc/moe/expert_stream/host/ram_tier.h` (`~RamTier` :151; `take_pooled_locked`; `read_rows`; new `start_spec`, `stop_spec`, `quiesce_spec`, `resume_spec`, `spec_busy_episode`, `pin_spec`, `run_spec`)
- Modify: `python/sglang/kernels/jit/csrc/moe/expert_stream/host/ram_thread.h` (`start` :62, `stop` :104, `pause` :123, `resume_locked` :168, `watch` :246, `stuck_group` :257, `abort_hung` :283)
- Test: `test/registered/unit/kernels/test_exl3_ram_prefetch_thread.py`

**Interfaces:**
- Consumes: `SpecState`, `SpecGroup`, `serve_spec_job`, `spec_read`, `spec_count`, `inject_spec`, `enable_ram_prefetch` (Task 5); `kSpecPromoted`, `kSpecDelayed` (Task 3); fixtures `prefetch_rig`, `enable`, `trigger`, `forced`, `load`, `LOGITS`.
- Produces: `RamTier::start_spec()` (throws on a failed pin after joining), `stop_spec()` (idempotent), `quiesce_spec()`, `resume_spec()`, `uint64_t spec_busy_episode(int g) const`; speculative threads named `exl3-spec<g>`; watchdog message `a speculative read stayed in service`; `kSpecDelayed` counted when a demand read waits at the turn; `kSpecPromoted` when a forced miss waits for its pool row's read.

- [ ] **Step 1: Write the failing tests**

Create `test/registered/unit/kernels/test_exl3_ram_prefetch_thread.py`:

```python
"""The speculative threads (spec 2026-10-08-dsv41-ram-prefetch-design, Phase 1, "Demand priority", "Threads and
cores", "Failure handling"): one per group, started and stopped with the service threads, taking turns with the
demand read on the group's reader, quiesced by pause (and so by a prefill fill), timed by the watchdog (CPU)."""

import faulthandler
import json
import os
import time

import pytest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.dsv41_ram_miss_fixtures import assert_aborted, spawn_child
from sglang.test.dsv41_ram_prefetch_fixtures import LOGITS, enable, forced, prefetch_rig, trigger

register_cpu_ci(est_time=90, suite="base-a-test-cpu")


@pytest.fixture(autouse=True)
def hang_guard():
    # Joins and handshakes run in C++: dump every stack and exit instead of hanging the suite.
    faulthandler.dump_traceback_later(120, exit=True)
    yield
    faulthandler.cancel_dump_traceback_later()


def _until(predicate, timeout_s=10.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


def _spec_tids():
    tids = []
    for tid in os.listdir("/proc/self/task"):
        with open(f"/proc/self/task/{tid}/comm") as f:
            if f.read().strip().startswith("exl3-spec"):
                tids.append(int(tid))
    return tids


def _started(tmp_path, *, delay_s=0.0, cores=None):
    rig = prefetch_rig(tmp_path)
    enable(rig, LOGITS, cores=cores)
    if delay_s:
        rig.host.inject_spec(delay_s=delay_s)
    rig.host.start_thread(fatal_wait_s=5.0)
    return rig


def test_the_speculative_thread_runs_on_its_cores_and_needs_no_pump(tmp_path):
    core = sorted(os.sched_getaffinity(0))[-1]
    rig = _started(tmp_path, cores=[[core]])
    try:
        trigger(rig)
        assert _until(lambda: rig.host.counters()["spec_landed"] == 1)
        tids = _spec_tids()
        assert len(tids) == 1 and os.sched_getaffinity(tids[0]) == {core}
    finally:
        rig.host.stop()
    assert _spec_tids() == []


def test_without_the_prefetch_no_speculative_thread_starts(tmp_path):
    rig = prefetch_rig(tmp_path)  # a pool, but never enabled
    try:
        rig.host.start_thread(fatal_wait_s=5.0)
        assert _spec_tids() == []
    finally:
        rig.host.stop()


def test_a_demand_read_waits_for_at_most_the_speculative_row_in_flight(tmp_path):
    """Mutant: the speculative read outside the turn lock -- red (the demand is served while the read still sleeps)."""
    rig = _started(tmp_path, delay_s=0.5)
    try:
        trigger(rig)
        assert _until(lambda: rig.host.counters()["spec_issued"] == 1)
        start = time.monotonic()
        req = rig.sim.post(1, [3])  # a GPU miss of row 1, not the pick
        assert rig.sim.wait_served(req, timeout_s=5.0)
        waited = time.monotonic() - start
        c = rig.host.counters()
        assert c["spec_landed"] == 1, "the demand read ran beside the speculative read"
        assert c["spec_delayed"] == 1 and waited < 2.0
    finally:
        rig.host.stop()


def test_a_forced_miss_on_a_row_still_reading_waits_for_it_and_swaps_it_in(tmp_path):
    rig = _started(tmp_path, delay_s=0.4)
    try:
        trigger(rig)
        assert _until(lambda: rig.host.counters()["spec_issued"] == 1)
        rows = rig.host.counters()["rows_read"]
        forced(rig, 1, [2])
        c = rig.host.counters()
        assert (c["spec_promoted"], c["spec_used"], c["rows_read"]) == (1, 1, rows)
        assert rig.host.mapping(1)[2] >= 0
    finally:
        rig.host.stop()


def test_a_gpu_miss_on_a_row_still_reading_reads_into_staging_and_the_pool_row_lands(tmp_path):
    """Review Focus 4."""
    rig = _started(tmp_path, delay_s=0.4)
    try:
        trigger(rig)
        assert _until(lambda: rig.host.counters()["spec_issued"] == 1)
        req = rig.sim.post(1, [2])
        assert rig.sim.wait_served(req, timeout_s=5.0) and rig.sim.wait_handled(req, timeout_s=5.0)
        assert _until(lambda: rig.host.counters()["spec_landed"] == 1)
        pooled = next(e for e in rig.host.spec_pool(1) if e["expert"] == 2)
        assert pooled["state"] == "landed" and rig.host.mapping(1)[2] not in (-1, pooled["slot"])
        c = rig.host.counters()
        assert (c["spec_promoted"], c["spec_used"]) == (0, 0)
    finally:
        rig.host.stop()


def test_pause_waits_for_the_speculative_read_and_a_prefill_fill_reads_alone(tmp_path):
    rig = _started(tmp_path, delay_s=0.4)
    try:
        trigger(rig)
        assert _until(lambda: rig.host.counters()["spec_issued"] == 1)
        rig.host.pause(10.0)
        try:
            assert rig.host.counters()["spec_landed"] == 1, "pause returned with a speculative read in flight"
            slots, _ = rig.host.fill_begin(1, [3])
            assert len(slots) == 1 and rig.host.fill_end()
            rig.sim.sync_bulk()
        finally:
            rig.host.resume()
        rig.host.inject_spec()
        trigger(rig)
        assert _until(lambda: rig.host.counters()["spec_landed"] == 2), "the thread did not resume"
    finally:
        rig.host.stop()


def test_stop_with_a_read_in_flight_joins_and_reports_the_pool(tmp_path, capfd):
    rig = _started(tmp_path, delay_s=0.4)
    trigger(rig)
    assert _until(lambda: rig.host.counters()["spec_issued"] == 1)
    rig.host.stop()
    lines = [l for l in capfd.readouterr().err.splitlines() if l.startswith("exl3 RAM miss thread counters ")]
    counters = json.loads(lines[-1].removeprefix("exl3 RAM miss thread counters "))
    assert (counters["spec_issued"], counters["spec_landed"]) == (1, 1)
    assert _spec_tids() == []


_HUNG = """
import pathlib, sys, time
from sglang.test.dsv41_ram_prefetch_fixtures import LOGITS, enable, prefetch_rig, trigger
rig = prefetch_rig(pathlib.Path(sys.argv[1]))
enable(rig, LOGITS)
rig.host.inject_spec(delay_s=30.0)
rig.host.start_thread(fatal_wait_s=0.3)
trigger(rig)
time.sleep(3.0)
print("reached")
"""


def test_the_watchdog_aborts_a_hung_speculative_read(tmp_path):
    result = spawn_child(_HUNG, tmp_path, timeout_s=90, variant="instr")
    assert_aborted(result, "a speculative read stayed in service")
```

- [ ] **Step 2: Commit the failing tests, push, run them on divix01**

```bash
cd /Users/dnikolaidis/.codex/worktrees/ram-prefetch/sglang-nvfp4
git add test/registered/unit/kernels/test_exl3_ram_prefetch_thread.py
git commit -m "Test the RAM prefetch's speculative threads

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push origin codex/dsv41-ram-prefetch
ssh divix01 bash -s <<'REMOTE'
cd /data/models/slang/nvfp4-work/wt-ram-prefetch && git fetch -q origin && git checkout -q --detach origin/codex/dsv41-ram-prefetch && git log -1 --oneline
PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 taskset -c 0-63 /data/models/slang/.venv/bin/python -m pytest test/registered/unit/kernels/test_exl3_ram_prefetch_thread.py -q -p no:randomly 2>&1 | tail -10; echo "EXIT=${PIPESTATUS[0]}"
REMOTE
```
Expected: `EXIT=1`: the thread, turn, promotion, pause and watchdog tests fail (no thread serves the ring; `_until` times out, the child does not abort); `test_without_the_prefetch_no_speculative_thread_starts` passes.

- [ ] **Step 3: Start, quiesce and stop the speculative threads in `RamTier`**

Add `#include <future>` to `ram_tier.h`'s includes.

In `~RamTier()`, make the first statement:
```cpp
    stop_spec();  // the speculative threads read through the groups' readers into the pool's slabs
```

Public, after `inject_spec`:
```cpp
  // ---- The speculative threads (RamThread drives them with the service threads) ----

  // Starts each group's speculative thread, pinned to its cores. Throws, after joining them, when one cannot be
  // pinned. A no-op with the RAM prefetch off.
  void start_spec() {
    if (spec_ == nullptr) return;
    spec_->stop.store(false);
    spec_->hold.store(false);
    std::vector<std::promise<int>> pinned(spec_->groups.size());
    for (size_t g = 0; g < spec_->groups.size(); ++g) {
      std::promise<int>* pin = &pinned[g];
      spec_->groups[g]->thread = std::thread([this, g, pin] {
        const int error = pin_spec(static_cast<int>(g));
        pin->set_value(error);
        if (error == 0) run_spec(static_cast<int>(g));
      });
    }
    int failed = -1;
    int error = 0;
    for (size_t g = 0; g < pinned.size(); ++g) {
      const int e = pinned[g].get_future().get();
      if (failed < 0 && e != 0) {
        failed = static_cast<int>(g);
        error = e;
      }
    }
    if (failed >= 0) {
      stop_spec();
      throw std::runtime_error(
          error_prefix<Layout>() + "could not pin the speculative thread of group " + std::to_string(failed) + ": " +
          std::strerror(error));
    }
  }

  // Stops and joins the speculative threads; each finishes the read it is in first. Idempotent.
  void stop_spec() {
    if (spec_ == nullptr) return;
    spec_->stop.store(true, std::memory_order_seq_cst);
    for (auto& spec : spec_->groups)
      spec->bell.ring();
    for (auto& spec : spec_->groups)
      if (spec->thread.joinable()) spec->thread.join();
  }

  // Holds the speculative threads between reads and returns once none reads: a pausing caller and a prefill fill use
  // the readers. Bounded by one row; the watchdog times a hung one.
  void quiesce_spec() {
    if (spec_ == nullptr) return;
    spec_->hold.store(true, std::memory_order_seq_cst);
    for (auto& spec : spec_->groups)
      spec->bell.ring();
    for (auto& spec : spec_->groups)
      while (spec->thread.joinable() && !spec->idle.load(std::memory_order_seq_cst))
        std::this_thread::sleep_for(std::chrono::microseconds(20));
  }

  void resume_spec() {
    if (spec_ == nullptr) return;
    spec_->hold.store(false, std::memory_order_seq_cst);
    for (auto& spec : spec_->groups)
      spec->bell.ring();
  }

  // The watchdog's marker for group g's speculative read: nonzero while one is in flight, a new value per read.
  uint64_t spec_busy_episode(int g) const {
    return spec_ == nullptr ? 0 : spec_->groups[g]->busy.load(std::memory_order_acquire);
  }
```

Private, after `spec_read`:
```cpp
  // Names group g's speculative thread and pins it to its cores; returns 0 or the errno.
  int pin_spec(int g) {
    SpecGroup& spec = *spec_->groups[g];
    const std::string name = std::string(Layout::kName) + "-spec" + std::to_string(g);
    pthread_setname_np(pthread_self(), name.substr(0, 15).c_str());
    if (spec.cores.empty()) return 0;
    cpu_set_t cpus;
    CPU_ZERO(&cpus);
    for (const int core : spec.cores)
      CPU_SET(core, &cpus);
    return pthread_setaffinity_np(pthread_self(), sizeof(cpus), &cpus);
  }

  // Group g's speculative thread: serves its ring, sleeping on its doorbell when the ring is empty, and holds between
  // jobs while quiesced. `idle` is stored before `hold` is loaded, both seq_cst, so a quiescer that stored `hold` and
  // then saw `idle` knows no read starts until resume_spec.
  void run_spec(int g) {
    SpecGroup& spec = *spec_->groups[g];
    while (!spec_->stop.load(std::memory_order_acquire)) {
      spec.idle.store(false, std::memory_order_seq_cst);
      if (spec_->hold.load(std::memory_order_seq_cst)) {
        spec.idle.store(true, std::memory_order_seq_cst);
        while (spec_->hold.load(std::memory_order_acquire) && !spec_->stop.load(std::memory_order_acquire))
          std::this_thread::sleep_for(std::chrono::microseconds(20));
        continue;
      }
      SpecJob job;
      if (spec.ring.pop(&job)) {
        serve_spec_job(g, job);
        continue;
      }
      spec.idle.store(true, std::memory_order_seq_cst);
      spec.bell.sleep_unless([&] {
        return !spec.ring.empty() || spec_->hold.load(std::memory_order_relaxed) ||
               spec_->stop.load(std::memory_order_relaxed);
      });
    }
    spec.idle.store(true, std::memory_order_seq_cst);
  }
```

- [ ] **Step 4: Take the turn in the demand read, and promote a forced miss on a reading row**

In `read_rows`, make the first statements:
```cpp
    // The group's reader takes turns with its speculative thread: a demand waits for at most the one speculative row.
    std::unique_lock<std::mutex> turn;
    if (spec_ != nullptr) {
      turn = std::unique_lock<std::mutex>(spec_->groups[group.index]->turn, std::try_to_lock);
      if (!turn.owns_lock()) {
        count<kSpecDelayed>(group);
        turn.lock();
      }
    }
```

In `take_pooled_locked`, directly after `PoolEntry& entry = pool_->entry(request.row, g, i);`:
```cpp
    if (entry.word.load(std::memory_order_acquire) == pool_word(kPoolReading, expert)) {
      // Promotion: the demand waits for the one speculative read in flight rather than read the row again; the
      // service's busy episode times the wait.
      count<kSpecPromoted>(group);
      while (entry.word.load(std::memory_order_acquire) == pool_word(kPoolReading, expert))
        _mm_pause();
    }
```

- [ ] **Step 5: Drive the threads from `RamThread`, and watch them**

In `ram_thread.h`:

In `start()`, replace `watchdog_ = std::thread([this] { watch(); });` with:
```cpp
    try {
      tier_->start_spec();
    } catch (...) {
      stop_.store(true);
      for (std::thread& thread : threads_)
        thread.join();
      tier_->set_threaded(false);
      throw;
    }
    watchdog_ = std::thread([this] { watch(); });
```

In `stop()`, make the first statement:
```cpp
    // First: a service thread promoting a pool row waits for its read, which must finish while the services run.
    tier_->stop_spec();
```

In `pause()`, directly after `std::lock_guard<std::mutex> caller(tier_->caller_mutex());`:
```cpp
    // The paused caller and a prefill fill use the readers the speculative reads take turns on.
    tier_->quiesce_spec();
```

In `resume_locked()`, directly after `tier_->set_parked(false);`:
```cpp
    tier_->resume_spec();
```

Replace `watch`, `stuck_group` and `abort_hung` with:
```cpp
  /// The watchdog: aborts the process, instead of hanging decode, when one demand or fill stays in service on a group,
  /// or one speculative read stays in flight, for fatal_wait_ns (a hung read, timed as one busy episode), or when the
  /// copy wait's gate stays closed on one value past the copy-wait timeout (the copy thread is stuck in a driver call,
  /// and the device's wait must still end). It reads the clock every 20 ms on its own thread, so a stuck read cannot
  /// silence it.
  void watch() {
    const size_t episodes = 2 * threads_.size();  // each group's service, then each group's speculative read
    WatchState seen{std::vector<uint64_t>(episodes, 0), std::vector<int64_t>(episodes, 0)};
    while (!watch_stop_.load()) {
      const int64_t now = now_ns();
      const int stuck = stuck_group(seen, now);
      if (stuck >= 0 || gate_held(seen, now)) abort_hung(stuck);
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
  }

  /// The first busy episode that has lasted past fatal_wait_ns, or -1: index g is group g's service, groups + g its
  /// speculative read.
  int stuck_group(WatchState& seen, int64_t now) const {
    const size_t groups = threads_.size();
    int stuck = -1;
    for (size_t i = 0; i < 2 * groups; ++i) {
      const int g = static_cast<int>(i % groups);
      const uint64_t busy = i < groups ? tier_->busy_episode(g) : tier_->spec_busy_episode(g);
      if (busy != seen.episode[i]) {
        seen.episode[i] = busy;
        seen.episode_since[i] = now;
      }
      if (stuck < 0 && seen.episode[i] != 0 && now - seen.episode_since[i] > fatal_wait_ns_) stuck = static_cast<int>(i);
    }
    return stuck;
  }
```
and
```cpp
  /// Reports the hang (`stuck` from stuck_group, or the copy gate when -1) and aborts without a core dump.
  [[noreturn]] void abort_hung(int stuck) const {
    const int groups = static_cast<int>(threads_.size());
    const bool request = stuck >= 0;
    const bool speculative = stuck >= groups;
    const std::string why = request ? "" : tier_->copy_stall();
    const std::string group = request && groups > 1 ? "group " + std::to_string(stuck % groups) + ": " : "";
    std::fprintf(
        stderr,
        "FATAL %s%s%s for %.1f s%s; aborting instead of hanging decode\n",
        error_prefix<typename Tier::Layout>().c_str(),
        group.c_str(),
        !request ? "a copy wait held the decode stream"
                 : (speculative ? "a speculative read stayed in service" : "a request stayed in service"),
        static_cast<double>(request ? fatal_wait_ns_ : tier_->copy_wait_timeout_ns()) / 1e9,
        why.c_str());
    std::fflush(stderr);
    prctl(PR_SET_DUMPABLE, 0);
    std::abort();
  }
```

- [ ] **Step 6: Commit, push, run the thread tests and the service-thread files**

```bash
cd /Users/dnikolaidis/.codex/worktrees/ram-prefetch/sglang-nvfp4
git add python/sglang/kernels/jit/csrc/moe/expert_stream/host/ram_tier.h python/sglang/kernels/jit/csrc/moe/expert_stream/host/ram_thread.h
git commit -m "Run one speculative thread per group, taking turns with the demand read

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push origin codex/dsv41-ram-prefetch
ssh divix01 bash -s <<'REMOTE'
cd /data/models/slang/nvfp4-work/wt-ram-prefetch && git fetch -q origin && git checkout -q --detach origin/codex/dsv41-ram-prefetch && git log -1 --oneline
PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 taskset -c 0-63 /data/models/slang/.venv/bin/python -m pytest test/registered/unit/kernels/test_exl3_ram_prefetch_thread.py test/registered/unit/kernels/test_exl3_ram_prefetch_step.py test/registered/unit/kernels/test_exl3_ram_prefetch_swap.py test/registered/unit/kernels/test_exl3_ram_miss_thread.py test/registered/unit/kernels/test_exl3_ram_miss_prefill_fills.py test/registered/unit/kernels/test_exl3_ram_miss_numa_completion.py test/registered/unit/kernels/test_exl3_ram_miss_stage_trace_causal.py test/registered/unit/kernels/test_expert_stream_prod_build_symbols.py -q -p no:randomly 2>&1 | tail -5; echo "EXIT=${PIPESTATUS[0]}"
REMOTE
```
Expected: `EXIT=0`. `test_exl3_ram_miss_stage_trace_causal.py` (if the file is named differently, `ls test/registered/unit/kernels | grep stage_trace` names it) shows the trace's clock-read count is unchanged with the prefetch off.

- [ ] **Step 7: Mutant check (spec Testing list: a speculative read outside the turn lock)**

```bash
ssh divix01 bash -s <<'REMOTE'
cd /data/models/slang/nvfp4-work/wt-ram-prefetch
T=python/sglang/kernels/jit/csrc/moe/expert_stream/host/ram_tier.h
run() { PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 taskset -c 0-63 /data/models/slang/.venv/bin/python -m pytest test/registered/unit/kernels/test_exl3_ram_prefetch_thread.py -q -p no:randomly -k "demand_read_waits" 2>&1 | tail -3; echo "EXIT=${PIPESTATUS[0]}"; }
/data/models/slang/.venv/bin/python - "$T" <<'PY'
import pathlib, sys
p = pathlib.Path(sys.argv[1]); old = "    std::lock_guard<std::mutex> turn(spec.turn);\n"
t = p.read_text(); assert t.count(old) == 1; p.write_text(t.replace(old, ""))
PY
run; git checkout -- "$T"; git status --short
echo "== restored"; run
REMOTE
```
Expected: `EXIT=1` (the demand is served with `spec_landed == 0`), empty status, `EXIT=0` restored.

---

### Task 7: Python wiring

**Files:**
- Modify: `python/sglang/srt/layers/moe/ram_prefetch.py` (registry, `register_moe_gates`, `PrefetchTables`, `prefetch_tables`)
- Modify: `python/sglang/srt/models/deepseek_v4.py` (`DeepseekV4ForCausalLM.post_load_weights` :5290)
- Modify: `python/sglang/srt/layers/moe/exl3_ram_miss.py` (`ensure_started` try block; new `_enable_ram_prefetch`; start log line)
- Test: `test/registered/unit/layers/moe/test_ram_prefetch_tables.py` (new), `test/registered/unit/layers/moe/test_exl3_ram_miss_service.py` (one test)

**Interfaces:**
- Consumes: `ExpertStreamHost.reserve_spec_pool`, `enable_ram_prefetch` (Tasks 2, 5); `NodePlan.spec`, `Exl3RamMissService._spec_share`, the options (Task 1); `host_numa.allocate_bound(nbytes, runs, row_bytes)`; `CpuExpertGroups.services[0].hidden`.
- Produces: `RouterGate(weight, bias, top_k)`; `register_router_gate(layer_id, weight, bias, top_k)`; `register_moe_gates(layers: Mapping[int, layer], top_k) -> int`; `registered_gates() -> dict[int, RouterGate]`; `clear_router_gates()`; `PrefetchTables(targets int64 [rows, 2], gates bf16 [n, experts, hidden], bias fp32 [n, experts], top_k)`; `prefetch_tables(layer_ids, gates, *, hidden, node=None) -> PrefetchTables`; `Exl3RamMissService._enable_ram_prefetch(host, layer_ids, numa, cpu_experts)` (static).

- [ ] **Step 1: Write the failing tests**

Create `test/registered/unit/layers/moe/test_ram_prefetch_tables.py`:

```python
"""The RAM prefetch's Python side (spec 2026-10-08-dsv41-ram-prefetch-design, Phase 1): the router gates registered at
load, the per-row target table and its host copy, and the service's call into the host (CPU)."""

from types import SimpleNamespace

import pytest
import torch

from sglang.srt.environ import envs
from sglang.srt.layers.moe import exl3_ram_miss as module
from sglang.srt.layers.moe.cpu_experts.threading_config import NodePlan
from sglang.srt.layers.moe.ram_prefetch import (
    RouterGate,
    clear_router_gates,
    prefetch_tables,
    register_moe_gates,
    register_router_gate,
    registered_gates,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


@pytest.fixture(autouse=True)
def no_gates():
    clear_router_gates()
    yield
    clear_router_gates()


def _gate(scale=1.0, experts=4, hidden=8, bias=True, top_k=6):
    w = (torch.arange(experts * hidden, dtype=torch.float32).reshape(experts, hidden) * scale).to(torch.bfloat16)
    return RouterGate(w, torch.arange(experts, dtype=torch.float32) * scale if bias else None, top_k)


def test_each_row_targets_the_next_consecutive_layer_with_a_biased_gate():
    """Row 0's next layer is hash-routed (no bias), row 2's is not consecutive, row 4 is last."""
    gates = {1: _gate(bias=False), 2: _gate(2.0), 4: _gate(4.0), 5: _gate(5.0)}
    t = prefetch_tables([0, 1, 2, 4, 5], gates, hidden=8)
    assert t.targets.tolist() == [[-1, -1], [2, 0], [-1, -1], [4, 1], [-1, -1]]
    assert t.gates.dtype == torch.bfloat16 and t.gates.shape == (2, 4, 8) and t.gates.device.type == "cpu"
    assert torch.equal(t.gates[0], gates[2].weight) and torch.equal(t.gates[1], gates[5].weight)
    assert torch.equal(t.bias[1], gates[5].bias) and t.bias.dtype == torch.float32 and t.top_k == 6


@pytest.mark.parametrize(
    "gates, why",
    [
        ({}, "no streamed row has a next layer"),
        ({1: _gate(hidden=16)}, "hidden size 16"),
        ({1: _gate(), 2: _gate(top_k=8)}, "top_k"),
        ({1: _gate(), 2: _gate(experts=5)}, "experts"),
    ],
)
def test_a_table_the_host_could_not_score_is_refused(gates, why):
    with pytest.raises(ValueError, match=why):
        prefetch_tables([0, 1, 2], gates, hidden=8)


def test_register_moe_gates_takes_every_moe_layer_and_drops_a_hash_layers_bias():
    gate = SimpleNamespace(weight=torch.zeros(4, 8), e_score_correction_bias=torch.ones(4))
    layers = {
        0: SimpleNamespace(mlp=SimpleNamespace(gate=gate, is_hash=True)),
        1: SimpleNamespace(mlp=SimpleNamespace(gate=gate, is_hash=False)),
        2: SimpleNamespace(mlp=SimpleNamespace()),  # a dense layer
    }
    assert register_moe_gates(layers, 6) == 2
    got = registered_gates()
    assert sorted(got) == [0, 1] and got[0].bias is None and got[1].bias is gate.e_score_correction_bias
    assert got[1].top_k == 6


def test_the_service_enables_the_host_with_the_registered_gates_options_and_spare_cores():
    gate = _gate()
    register_router_gate(1, gate.weight, gate.bias, 6)
    calls = []
    host = SimpleNamespace(enable_ram_prefetch=lambda *a, **kw: calls.append((a, kw)))
    numa = SimpleNamespace(
        nodes=1, plans=[NodePlan(group=0, node=0, ram=17, cpu=(8, 9), sq=None, busy_poll=True, spec=(0, 1))]
    )
    cpu = SimpleNamespace(services=[SimpleNamespace(hidden=8)])
    with envs.SGLANG_DSV41_RAM_PREFETCH_PER_LAYER.override(2):
        module.Exl3RamMissService._enable_ram_prefetch(host, [0, 1], numa, cpu)
    (targets, gates, bias), kw = calls[0]
    assert targets.tolist() == [[1, 0], [-1, -1]] and torch.equal(gates[0], gate.weight)
    assert kw == dict(top_k=6, per_token=1, per_layer=2, cores=[[0, 1]])
    with pytest.raises(RuntimeError, match="needs SGLANG_DSV41_CPU_EXPERTS"):
        module.Exl3RamMissService._enable_ram_prefetch(host, [0, 1], numa, None)
```

Append to `test/registered/unit/layers/moe/test_exl3_ram_miss_service.py`:

```python
def test_ram_prefetch_without_cpu_experts_is_refused_at_start(tiers):
    service, _, _ = tiers
    with envs.SGLANG_DSV41_RAM_PREFETCH.override(True):
        with pytest.raises(RuntimeError, match="SGLANG_DSV41_RAM_PREFETCH needs SGLANG_DSV41_CPU_EXPERTS"):
            service.ensure_started()
```

- [ ] **Step 2: Commit the failing tests, push, run them on divix01**

```bash
cd /Users/dnikolaidis/.codex/worktrees/ram-prefetch/sglang-nvfp4
git add test/registered/unit/layers/moe/test_ram_prefetch_tables.py test/registered/unit/layers/moe/test_exl3_ram_miss_service.py
git commit -m "Test the RAM prefetch's gate registry, target table and service wiring

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push origin codex/dsv41-ram-prefetch
ssh divix01 bash -s <<'REMOTE'
cd /data/models/slang/nvfp4-work/wt-ram-prefetch && git fetch -q origin && git checkout -q --detach origin/codex/dsv41-ram-prefetch && git log -1 --oneline
PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 taskset -c 0-63 /data/models/slang/.venv/bin/python -m pytest test/registered/unit/layers/moe/test_ram_prefetch_tables.py test/registered/unit/layers/moe/test_exl3_ram_miss_service.py -q -p no:randomly -k "prefetch or gate or table" 2>&1 | tail -6; echo "EXIT=${PIPESTATUS[0]}"
REMOTE
```
Expected: `EXIT=2` or `EXIT=1`: `ImportError: cannot import name 'RouterGate'` (collection error), and the service test fails without the refusal.

- [ ] **Step 3: Write the registry and the target table**

Append to `python/sglang/srt/layers/moe/ram_prefetch.py` (add the imports at the top: `from dataclasses import dataclass`, `from typing import Mapping, Optional, Sequence`, `import torch`):

```python
@dataclass(frozen=True)
class RouterGate:
    weight: torch.Tensor  # [experts, hidden], the router's own parameter (any device)
    bias: Optional[torch.Tensor]  # [experts] e_score_correction_bias; None on a hash-routed layer
    top_k: int


_GATES: dict[int, RouterGate] = {}


def register_router_gate(layer_id: int, weight: torch.Tensor, bias: Optional[torch.Tensor], top_k: int) -> None:
    _GATES[int(layer_id)] = RouterGate(weight, bias, int(top_k))


def register_moe_gates(layers: Mapping[int, object], top_k: int) -> int:
    """Registers every MoE layer's router: ``layers`` maps a layer id to its decoder layer, whose ``mlp`` holds
    ``gate`` (``weight``, ``e_score_correction_bias``) and ``is_hash``. A dense layer has no gate. Returns how many."""
    registered = 0
    for layer_id, layer in layers.items():
        mlp = getattr(layer, "mlp", None)
        gate = getattr(mlp, "gate", None)
        if gate is None or not hasattr(gate, "weight"):
            continue
        bias = None if getattr(mlp, "is_hash", False) else getattr(gate, "e_score_correction_bias", None)
        register_router_gate(layer_id, gate.weight, bias, top_k)
        registered += 1
    return registered


def registered_gates() -> dict[int, RouterGate]:
    return dict(_GATES)


def clear_router_gates() -> None:
    _GATES.clear()


@dataclass(frozen=True)
class PrefetchTables:
    targets: torch.Tensor  # int64 [rows, 2]: per source row its target row and gate index, or (-1, -1)
    gates: torch.Tensor  # bf16 [n, experts, hidden], host memory
    bias: torch.Tensor  # fp32 [n, experts]
    top_k: int


def prefetch_tables(
    layer_ids: Sequence[int], gates: Mapping[int, RouterGate], *, hidden: int, node: Optional[int] = None
) -> PrefetchTables:
    """Row r targets row r + 1 when that row is the next layer (layer_ids[r] + 1) and its gate has a bias; the last row,
    a non-consecutive layer and a hash-routed one target nothing. Each target's gate is copied to host memory once,
    bound to NUMA ``node`` when given."""
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
                f"RAM prefetch: layer {layer_id}'s gate has hidden size {int(gate.weight.shape[-1])}, the CPU rows {hidden}"
            )
        if int(gate.weight.shape[0]) != experts:
            raise ValueError(f"RAM prefetch: layer {layer_id}'s gate has {int(gate.weight.shape[0])} experts, not {experts}")
        if gate.top_k != top_k:
            raise ValueError(f"RAM prefetch: layer {layer_id}'s top_k {gate.top_k} is not {top_k}")
    n, row_bytes = len(picks), experts * hidden * 2
    if node is None:
        flat = torch.empty((n, row_bytes), dtype=torch.uint8)
    else:
        from sglang.srt.layers.moe.host_numa import allocate_bound

        flat = allocate_bound(n * row_bytes, [(node, 0, n)], row_bytes).view(n, row_bytes)
    weights = flat.view(torch.bfloat16).view(n, experts, hidden)
    bias = torch.empty((n, experts), dtype=torch.float32)
    targets = torch.full((len(layer_ids), 2), -1, dtype=torch.int64)
    for index, (row, gate, _) in enumerate(picks):
        weights[index].copy_(gate.weight.detach().to(device="cpu", dtype=torch.bfloat16))
        bias[index].copy_(gate.bias.detach().to(device="cpu", dtype=torch.float32))
        targets[row, 0], targets[row, 1] = row + 1, index
    return PrefetchTables(targets, weights, bias, top_k)
```

- [ ] **Step 4: Register the gates after load**

In `python/sglang/srt/models/deepseek_v4.py`, at the end of `post_load_weights` (after the `for layer_id in range(...)` loop, inside the non-nextn path):
```python
        if envs.SGLANG_DSV41_RAM_PREFETCH.get():
            from sglang.srt.layers.moe.ram_prefetch import register_moe_gates

            register_moe_gates(
                {i: self.model.layers[i] for i in range(self.model.start_layer, self.model.end_layer)},
                self.config.num_experts_per_tok,
            )
```
(`envs` is already imported at line 50.)

- [ ] **Step 5: Reserve and enable from the service**

In `python/sglang/srt/layers/moe/exl3_ram_miss.py`, in `ensure_started`, directly after `host.reserve_staging(self.staging_width())`:
```python
            self._spec_share = (
                envs.SGLANG_DSV41_RAM_PREFETCH_SPEC_SHARE.get() if envs.SGLANG_DSV41_RAM_PREFETCH.get() else 0
            )
            if self._spec_share:
                if not envs.SGLANG_DSV41_CPU_EXPERTS.get():
                    raise RuntimeError("exl3 RAM miss: SGLANG_DSV41_RAM_PREFETCH needs SGLANG_DSV41_CPU_EXPERTS")
                # Right after the staging slots, before the hot cache fills any slot.
                host.reserve_spec_pool(self._spec_share)
```
and directly after the `if envs.SGLANG_DSV41_CPU_EXPERTS.get(): cpu_experts = self._start_cpu_experts(...)` block, at the indentation of `cpu_experts = None`:
```python
            if self._spec_share:
                self._enable_ram_prefetch(host, list(tables.layer_ids), numa, cpu_experts)
```
Add after `_start_cpu_experts`:
```python
    @staticmethod
    def _enable_ram_prefetch(host, layer_ids, numa, cpu_experts) -> None:
        """Start the RAM prefetch (SGLANG_DSV41_RAM_PREFETCH) over the pool reserved at start: the router gates the
        model registered at load, copied to host memory once, and each group's speculative thread on its plan's spare
        cores."""
        from sglang.srt.layers.moe.ram_prefetch import prefetch_tables, registered_gates

        if cpu_experts is None:
            raise RuntimeError("exl3 RAM miss: SGLANG_DSV41_RAM_PREFETCH needs SGLANG_DSV41_CPU_EXPERTS")
        # Off node 0, which the tier keeps near full (DSV41_REFERENCE.md section 33.12).
        node = numa.plans[-1].node if numa.nodes > 1 else None
        tables = prefetch_tables(layer_ids, registered_gates(), hidden=cpu_experts.services[0].hidden, node=node)
        per_token = envs.SGLANG_DSV41_RAM_PREFETCH_PER_TOKEN.get()
        per_layer = envs.SGLANG_DSV41_RAM_PREFETCH_PER_LAYER.get()
        host.enable_ram_prefetch(
            tables.targets,
            tables.gates,
            tables.bias,
            top_k=tables.top_k,
            per_token=per_token,
            per_layer=per_layer,
            cores=[list(plan.spec) for plan in numa.plans],
        )
        logger.info(
            "exl3 RAM miss prefetch: %d of %d rows target the next layer, %d per token, %d per layer, cores %s",
            int((tables.targets[:, 0] >= 0).sum()),
            len(layer_ids),
            per_token,
            per_layer,
            [plan.spec for plan in numa.plans],
        )
```

- [ ] **Step 6: Commit, push, run the Python tests and the service files**

```bash
cd /Users/dnikolaidis/.codex/worktrees/ram-prefetch/sglang-nvfp4
git add python/sglang/srt/layers/moe/ram_prefetch.py python/sglang/srt/models/deepseek_v4.py python/sglang/srt/layers/moe/exl3_ram_miss.py
git commit -m "Wire the RAM prefetch into the service: gates, target table, pool, cores

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push origin codex/dsv41-ram-prefetch
ssh divix01 bash -s <<'REMOTE'
cd /data/models/slang/nvfp4-work/wt-ram-prefetch && git fetch -q origin && git checkout -q --detach origin/codex/dsv41-ram-prefetch && git log -1 --oneline
PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 taskset -c 0-63 /data/models/slang/.venv/bin/python -m pytest test/registered/unit/layers/moe/test_ram_prefetch_tables.py test/registered/unit/layers/moe/test_exl3_ram_miss_service.py test/registered/unit/layers/moe/test_exl3_ram_miss_service_numa.py test/registered/unit/kernels/test_exl3_ram_miss_attach_lanes.py -q -p no:randomly 2>&1 | tail -5; echo "EXIT=${PIPESTATUS[0]}"
REMOTE
```
Expected: `EXIT=0`.

---

### Task 8: InstrBuild events and scorer metrics

**Files:**
- Modify: `python/sglang/kernels/jit/csrc/moe/expert_stream/host/ram_tier.h` (`serve_spec_job`, `spec_read`, `take_pooled_locked`)
- Test: `test/registered/unit/kernels/test_exl3_ram_prefetch_events.py`

**Interfaces:**
- Consumes: `SpecGroup::trace` (`JobTrace<Build::kMetrics>`, Task 5), `stats_`, `kSpecScored`, `kSpecScoreNs` (Task 3).
- Produces: job-trace events, in `<SGLANG_DSV41_EXPERT_JOB_TRACE_PREFIX>.<pid>.exl3-spec<g>.<n>.jsonl`: `spec_submit` and `spec_land` with `row` = target row, `seq` = target record seq, `group`, `a` = expert, `b` = pool slot; `spec_use` with `row`, `gen`, `seq` of the demand record that swapped it in, `a` = expert, `b` = slot. Counters `spec_scored` (records scored) and `spec_score_ns` (scoring ns), InstrBuild only.

- [ ] **Step 1: Write the failing test**

Create `test/registered/unit/kernels/test_exl3_ram_prefetch_events.py`:

```python
"""The RAM prefetch's InstrBuild observability (spec 2026-10-08-dsv41-ram-prefetch-design, Phase 1, "Observability"):
spec_submit, spec_land and spec_use job-trace events carrying the record's seq and row, so the job-trace joins
attribute them, and the scorer's records scored and scoring time (CPU)."""

import json

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.dsv41_ram_prefetch_fixtures import LOGITS, enable, forced, prefetch_rig, trigger

register_cpu_ci(est_time=20, suite="base-a-test-cpu")


def test_a_speculative_row_leaves_submit_land_and_use_events_and_the_scorer_counts(tmp_path, monkeypatch):
    (tmp_path / "rig").mkdir()
    rig = prefetch_rig(tmp_path / "rig")
    try:
        # Read when enable_ram_prefetch builds each group's trace; the rig's other engines were built without it.
        monkeypatch.setenv("SGLANG_DSV41_EXPERT_JOB_TRACE_PREFIX", str(tmp_path / "jobs"))
        enable(rig, LOGITS)
        trig = trigger(rig)
        assert rig.host.spec_pump(0)
        slot = next(e["slot"] for e in rig.host.spec_pool(1) if e["expert"] == 2)
        use = forced(rig, 1, [2])
        c = rig.host.counters()
        assert c["spec_scored"] == 1 and c["spec_score_ns"] > 0
    finally:
        rig.host.stop()
    files = list(tmp_path.glob("jobs.*.exl3-spec0.*.jsonl"))
    assert len(files) == 1
    events = {e["event"]: e for e in map(json.loads, files[0].read_text().splitlines()[1:-1])}
    target = (trig.seq + 1) & 0xFFFFFFFF
    for kind in ("spec_submit", "spec_land"):
        e = events[kind]
        assert (e["row"], e["seq"], e["group"], e["a"], e["b"]) == (1, target, 0, 2, slot)
    e = events["spec_use"]
    assert (e["row"], e["seq"], e["gen"], e["a"], e["b"]) == (1, use.seq, use.gen, 2, slot)
    assert use.seq == target
```

- [ ] **Step 2: Commit the failing test, push, run it on divix01**

```bash
cd /Users/dnikolaidis/.codex/worktrees/ram-prefetch/sglang-nvfp4
git add test/registered/unit/kernels/test_exl3_ram_prefetch_events.py
git commit -m "Test the RAM prefetch's job-trace events and scorer metrics

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push origin codex/dsv41-ram-prefetch
ssh divix01 bash -s <<'REMOTE'
cd /data/models/slang/nvfp4-work/wt-ram-prefetch && git fetch -q origin && git checkout -q --detach origin/codex/dsv41-ram-prefetch && git log -1 --oneline
PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 taskset -c 0-63 /data/models/slang/.venv/bin/python -m pytest test/registered/unit/kernels/test_exl3_ram_prefetch_events.py -q -p no:randomly 2>&1 | tail -5; echo "EXIT=${PIPESTATUS[0]}"
REMOTE
```
Expected: `EXIT=1` on `assert c["spec_scored"] == 1` (0 before this task).

- [ ] **Step 3: Emit the events and time the scorer**

In `serve_spec_job`, replace
```cpp
    fill_spec_skip(target, target_seq, spec.skip.data());
```
with
```cpp
    int64_t start = 0;
    if constexpr (Build::kMetrics) start = now_ns();  // a metric: ProdBuild reads no clock here
    fill_spec_skip(target, target_seq, spec.skip.data());
```
and directly after the `spec.scorer.choose(...)` statement:
```cpp
    if constexpr (Build::kMetrics) {
      stats_.add(kSpecScored);
      stats_.add(kSpecScoreNs, now_ns() - start);
    }
```

In `spec_read`, replace `spec_count<kSpecIssued>(spec);` with:
```cpp
    spec_count<kSpecIssued>(spec);
    spec.trace.emit("spec_submit", target, 0, target_seq, g, expert, slot);
```
and replace `spec_count<kSpecLanded>(spec);` with:
```cpp
      spec_count<kSpecLanded>(spec);
      spec.trace.emit("spec_land", target, 0, target_seq, g, expert, slot);
```

In `take_pooled_locked`, replace `count<kSpecUsed>(group);` with:
```cpp
    count<kSpecUsed>(group);
    if (spec_ != nullptr) spec_->groups[g]->trace.emit("spec_use", request.row, request.gen, request.seq, g, expert, slot);
```

- [ ] **Step 4: Commit, push, run the events test, the step tests and the prod symbol check**

```bash
cd /Users/dnikolaidis/.codex/worktrees/ram-prefetch/sglang-nvfp4
git add python/sglang/kernels/jit/csrc/moe/expert_stream/host/ram_tier.h
git commit -m "Trace speculative submits, landings and uses; time the scorer (InstrBuild)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push origin codex/dsv41-ram-prefetch
ssh divix01 bash -s <<'REMOTE'
cd /data/models/slang/nvfp4-work/wt-ram-prefetch && git fetch -q origin && git checkout -q --detach origin/codex/dsv41-ram-prefetch && git log -1 --oneline
PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 taskset -c 0-63 /data/models/slang/.venv/bin/python -m pytest test/registered/unit/kernels/test_exl3_ram_prefetch_events.py test/registered/unit/kernels/test_exl3_ram_prefetch_step.py test/registered/unit/kernels/test_expert_job_trace.py test/registered/unit/kernels/test_expert_stream_prod_build_symbols.py -q -p no:randomly 2>&1 | tail -5; echo "EXIT=${PIPESTATUS[0]}"
REMOTE
```
Expected: `EXIT=0`.

---

### Task 9: The A/B arm, the served smoke and the A/B

**Files:**
- Modify: `analysis/dsv41-drive/dspark/both_cpu_ab.py` (`ARMS`, `PASSTHROUGH`, `_overrides`, `REFERENCE`, `_server_counters`, `summarize`)
- Test: `test/registered/unit/scripts/test_both_cpu_ab.py`
- Modify (results): `DSV41_REFERENCE.md` (new §33.14 after §33.13), `NVME_PINNED_PREFETCH_HANDOFF.md` (new closing section)

**Interfaces:**
- Consumes: everything above; `arm_env.dspark_env()`, `run_probe`, `run_timed`, `scripts/dsv41/dspark_text_band.py compare`.
- Produces: arm `dspark-both-prefetch` = `dspark_env()` with `SGLANG_DSV41_RAM_PREFETCH=1` (per token 1, per layer 1, share 2: the replay's best h=1, K=1); arm `dspark-both` states `SGLANG_DSV41_RAM_PREFETCH=0`; `summary.json` entries gain `ram` (server-lifetime `rows_read` and the seven core `spec_*` counters), `ram_rows_per_timed_token`, and for the prefetch arm `text_vs_reference` (against `dspark-both`).

- [ ] **Step 1: Write the failing tests**

Append to `test/registered/unit/scripts/test_both_cpu_ab.py`:

```python
import json


def test_the_prefetch_arm_is_dspark_both_with_the_prefetch_on_and_its_a_states_it_off():
    ab = _ab()
    a, b = ab.ARMS["dspark-both"][0], ab.ARMS["dspark-both-prefetch"][0]
    assert a["SGLANG_DSV41_RAM_PREFETCH"] == "0" and b["SGLANG_DSV41_RAM_PREFETCH"] == "1"
    assert {k: v for k, v in a.items() if k != "SGLANG_DSV41_RAM_PREFETCH"} == {
        k: v for k, v in b.items() if k != "SGLANG_DSV41_RAM_PREFETCH"
    }
    assert ab.ARMS["dspark-both-prefetch"][1] is True


def test_the_private_build_caches_reach_every_arms_server(monkeypatch):
    ab = _ab()
    monkeypatch.setenv("SGLANG_JIT_CACHE_DIR", "/private/jit")
    monkeypatch.setenv("SGLANG_EXL3_BUILD_DIR", "/private/exl3")
    for arm in ab.ARMS:
        overrides = ab._overrides(arm, "/out")
        assert overrides["SGLANG_JIT_CACHE_DIR"] == "/private/jit"
        assert overrides["SGLANG_EXL3_BUILD_DIR"] == "/private/exl3"


def test_summarize_reports_the_ram_counters_and_the_prefetch_arms_text_against_its_a(tmp_path):
    ab = _ab()
    for arm, rows_read, used in (("dspark-both", 3000, 0), ("dspark-both-prefetch", 2000, 700)):
        run = tmp_path / "servers" / arm / "run-1"
        run.mkdir(parents=True)
        (run / "results.jsonl").write_text(
            json.dumps({"decode_tokens_per_sec": 2.0, "completion_tokens": 100, "spec_tokens_details": {}}) + "\n"
        )
        counters = {"rows_read": rows_read, "spec_issued": 900, "spec_used": used}
        (run / "server.log").write_text("noise\nexl3 RAM miss thread counters " + json.dumps(counters) + "\n")
        probe = [{"session_id": "s0", "tokens": [{"token": 1, "top": [[1, -0.1], [2, -2.0]]}]}]
        (tmp_path / f"{arm}.probe.json").write_text(json.dumps(probe))
    summary = ab.summarize(str(tmp_path))
    b = summary["dspark-both-prefetch"]
    assert b["ram"]["spec_used"] == 700 and b["ram"]["rows_read"] == 2000
    assert b["ram_rows_per_timed_token"] == 20.0 and summary["dspark-both"]["ram_rows_per_timed_token"] == 30.0
    assert b["text_vs_reference"]["pass"] is True
```

- [ ] **Step 2: Commit the failing tests, push, run them on divix01**

```bash
cd /Users/dnikolaidis/.codex/worktrees/ram-prefetch/sglang-nvfp4
git add test/registered/unit/scripts/test_both_cpu_ab.py
git commit -m "Test the dspark-both-prefetch A/B arm and its summary

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push origin codex/dsv41-ram-prefetch
ssh divix01 bash -s <<'REMOTE'
cd /data/models/slang/nvfp4-work/wt-ram-prefetch && git fetch -q origin && git checkout -q --detach origin/codex/dsv41-ram-prefetch && git log -1 --oneline
PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 taskset -c 0-63 /data/models/slang/.venv/bin/python -m pytest test/registered/unit/scripts/test_both_cpu_ab.py -q -p no:randomly 2>&1 | tail -6; echo "EXIT=${PIPESTATUS[0]}"
REMOTE
```
Expected: `EXIT=1`: `KeyError: 'SGLANG_DSV41_RAM_PREFETCH'`, `KeyError: 'dspark-both-prefetch'`, `KeyError: 'SGLANG_JIT_CACHE_DIR'`, `KeyError: 'ram'`.

- [ ] **Step 3: Add the arm, the passthrough and the summary fields**

In `analysis/dsv41-drive/dspark/both_cpu_ab.py`, replace the `"dspark-both": (arm_env.dspark_env(), True),` entry with:
```python
    # States the RAM prefetch off, so a captured shell cannot carry it into the A of the prefetch pair.
    "dspark-both": ({**arm_env.dspark_env(), "SGLANG_DSV41_RAM_PREFETCH": "0"}, True),
```
and add after the row-weighted entries, inside `ARMS`:
```python
    # NVMe-to-RAM prefetch (spec 2026-10-08-dsv41-ram-prefetch-design, The A/B): the replay's best arm, h=1, one
    # candidate per token, one row per layer, a pool of 2 per row and group (the options' defaults).
    "dspark-both-prefetch": ({**arm_env.dspark_env(), "SGLANG_DSV41_RAM_PREFETCH": "1"}, True),
```

After `ARMS`, add:
```python
# Build caches an experiment keeps private (run protocol): passed to every arm's server when set in the driver's env.
PASSTHROUGH = ("SGLANG_JIT_CACHE_DIR", "SGLANG_EXL3_BUILD_DIR")
# An arm whose outputs are also compared with its A's, not only with prod's.
REFERENCE = {"dspark-both-prefetch": "dspark-both"}
COUNTER_MARKER = "exl3 RAM miss thread counters "
RAM_KEYS = (
    "rows_read",
    "spec_issued",
    "spec_landed",
    "spec_used",
    "spec_promoted",
    "spec_dropped",
    "spec_failed",
    "spec_delayed",
)
```

Replace `_overrides` with:
```python
def _overrides(arm: str, out: str) -> dict:
    overrides, _ = ARMS[arm]
    passed = {k: os.environ[k] for k in PASSTHROUGH if os.environ.get(k)}
    return {**overrides, **passed, "SGLANG_MOE_HOT_METRICS_FILE": os.path.join(out, f"{arm}.metrics.jsonl")}
```

After `_results`, add:
```python
def _server_counters(out: str, arm: str):
    """The last counters line of the arm's timed server: the service's lifetime (warm-up and the timed set)."""
    runs = sorted(os.listdir(os.path.join(out, "servers", arm)))
    path = os.path.join(out, "servers", arm, runs[-1], "server.log")
    if not os.path.exists(path):
        return None
    with open(path, errors="replace") as f:
        lines = [line for line in f if COUNTER_MARKER in line]
    return json.loads(lines[-1].split(COUNTER_MARKER, 1)[1]) if lines else None
```

In `summarize`, directly before `summary[arm] = entry`:
```python
        counters = _server_counters(out, arm)
        if counters:
            entry["ram"] = {k: counters.get(k) for k in RAM_KEYS}
            entry["ram"]["scope"] = "server lifetime: warm-up and the timed set"
            tokens = sum(r["completion_tokens"] for r in rows)
            entry["ram_rows_per_timed_token"] = counters["rows_read"] / tokens if tokens else None
        reference = REFERENCE.get(arm)
        ref_probes = {a: os.path.join(out, f"{a}.probe.json") for a in (reference, arm)} if reference else {}
        if ref_probes and all(os.path.exists(p) for p in ref_probes.values()):
            with open(ref_probes[reference]) as f:
                base = json.load(f)
            with open(ref_probes[arm]) as f:
                other = json.load(f)
            entry["text_vs_reference"] = dspark_text_band.compare(base, other)
```

- [ ] **Step 4: Commit, push, run the driver tests**

```bash
cd /Users/dnikolaidis/.codex/worktrees/ram-prefetch/sglang-nvfp4
git add analysis/dsv41-drive/dspark/both_cpu_ab.py
git commit -m "Add the dspark-both-prefetch A/B arm and its RAM counters summary

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push origin codex/dsv41-ram-prefetch
ssh divix01 bash -s <<'REMOTE'
cd /data/models/slang/nvfp4-work/wt-ram-prefetch && git fetch -q origin && git checkout -q --detach origin/codex/dsv41-ram-prefetch && git log -1 --oneline
PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 taskset -c 0-63 /data/models/slang/.venv/bin/python -m pytest test/registered/unit/scripts/test_both_cpu_ab.py benchmarks/dsv41_baseline/test_dspark_recipe.py -q -p no:randomly 2>&1 | tail -5; echo "EXIT=${PIPESTATUS[0]}"
REMOTE
```
Expected: `EXIT=0`.

- [ ] **Step 5: Run the whole prefetch selection once on the final commit**

```bash
ssh divix01 bash -s <<'REMOTE'
cd /data/models/slang/nvfp4-work/wt-ram-prefetch && git fetch -q origin && git checkout -q --detach origin/codex/dsv41-ram-prefetch && git log -1 --oneline
PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 taskset -c 0-63 /data/models/slang/.venv/bin/python -m pytest test/registered/unit/kernels/test_exl3_ram_prefetch_pool.py test/registered/unit/kernels/test_exl3_ram_prefetch_swap.py test/registered/unit/kernels/test_exl3_ram_prefetch_scorer.py test/registered/unit/kernels/test_exl3_ram_prefetch_step.py test/registered/unit/kernels/test_exl3_ram_prefetch_thread.py test/registered/unit/kernels/test_exl3_ram_prefetch_events.py test/registered/unit/layers/moe/test_ram_prefetch_tables.py test/registered/unit/test_expert_stream_requirements_exl3.py test/registered/unit/layers/moe/test_threading_config.py test/registered/unit/kernels/test_exl3_ram_miss_cpu_experts.py test/registered/unit/kernels/test_exl3_ram_miss_thread.py test/registered/unit/kernels/test_exl3_ram_miss_slot_map.py test/registered/unit/kernels/test_exl3_ram_miss_numa_groups.py test/registered/unit/kernels/test_exl3_ram_miss_prefill_fills.py test/registered/unit/kernels/test_expert_stream_prod_build_symbols.py test/registered/unit/scripts/test_both_cpu_ab.py -q -p no:randomly 2>&1 | tail -3; echo "EXIT=${PIPESTATUS[0]}"
REMOTE
```
Expected: `EXIT=0`. Record the pass count with this exact selection in the §33.14 write-up.

- [ ] **Step 6: Ask the owner to stop production**

Use AskUserQuestion: "The RAM prefetch smoke and A/B need the GPU on divix01 (several hours: the smoke's two DSpark probe servers, then two arms of a timed server and a probe server each). May I stop production now? I will not restart it; I will tell you when the GPU is free." Options: "Yes, stop it now", "Not now". Proceed only on yes. Then check the GPU is free:
```bash
ssh divix01 nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
```
Expected: no line (nothing on the GPU). If a process remains, stop and ask again; never kill a process you did not start.

- [ ] **Step 7: The served smoke (one DSpark probe per side, option on and off)**

```bash
ssh divix01 bash -s <<'REMOTE'
set -u
WT=/data/models/slang/nvfp4-work/wt-ram-prefetch
cd $WT && git fetch -q origin && git checkout -q --detach origin/codex/dsv41-ram-prefetch && git log -1 --oneline
OUT=/data/models/slang/nvfp4-work/ram-prefetch/smoke-$(date +%Y%m%d-%H%M%S); mkdir -p $OUT/tmp; echo "OUT=$OUT"
git rev-parse HEAD > $OUT/commit.txt
export TMPDIR=$OUT/tmp SGLANG_JIT_CACHE_DIR=/data/models/slang/nvfp4-work/ram-prefetch/jit-cache SGLANG_EXL3_BUILD_DIR=/data/models/slang/nvfp4-work/ram-prefetch/exl3-build
flock /data/models/slang/nvfp4-work/rowimg-disk.lock taskset -c 0-63 /data/models/slang/.venv/bin/python - "$OUT" <<'PY'
import sys
sys.path.insert(0, "analysis/dsv41-drive/dspark")
import both_cpu_ab as ab
for arm in ("dspark-both-prefetch", "dspark-both"):
    print(arm, "rc", ab.run_probe(arm, sys.argv[1]), flush=True)
PY
grep -h "exl3 RAM miss prefetch:" $OUT/dspark-both-prefetch.probe-server.log | tail -1
grep -h "exl3 RAM miss thread counters" $OUT/dspark-both-prefetch.probe-server.log | tail -1
/data/models/slang/.venv/bin/python scripts/dsv41/dspark_text_band.py $OUT/dspark-both.probe.json $OUT/dspark-both-prefetch.probe.json
REMOTE
```
Expected: `dspark-both-prefetch rc 0` and `dspark-both rc 0`; a `exl3 RAM miss prefetch: K of 40 rows target the next layer, 1 per token, 1 per layer, cores [...]` line with K > 0; a counters line with `spec_issued > 0`, `spec_landed > 0`, `spec_used > 0`, `spec_failed == 0`; the text band's JSON with `"pass": true`. Any other outcome stops the task here: report the logs under `$OUT` and do not start the A/B.

- [ ] **Step 8: The A/B (8 sessions, 104 GiB tier, counters off, private caches)**

```bash
ssh divix01 bash -s <<'REMOTE'
set -u
WT=/data/models/slang/nvfp4-work/wt-ram-prefetch
cd $WT && git fetch -q origin && git checkout -q --detach origin/codex/dsv41-ram-prefetch && git log -1 --oneline
OUT=/data/models/slang/nvfp4-work/ram-prefetch/ab-$(date +%Y%m%d-%H%M%S); mkdir -p $OUT/tmp; echo "OUT=$OUT"
git rev-parse HEAD > $OUT/commit.txt
export TMPDIR=$OUT/tmp SGLANG_JIT_CACHE_DIR=/data/models/slang/nvfp4-work/ram-prefetch/jit-cache SGLANG_EXL3_BUILD_DIR=/data/models/slang/nvfp4-work/ram-prefetch/exl3-build
nohup flock /data/models/slang/nvfp4-work/rowimg-disk.lock taskset -c 0-63 /data/models/slang/.venv/bin/python analysis/dsv41-drive/dspark/both_cpu_ab.py $OUT dspark-both dspark-both-prefetch > $OUT/ab.log 2>&1 &
echo "pid=$!"
REMOTE
```
Expected: `OUT=...` and a pid. Poll every 10 minutes until the driver prints its summary JSON:
```bash
ssh divix01 "tail -4 <OUT>/ab.log"
```
Expected at the end: `dspark-both run_timed: rc=0`, `dspark-both run_probe: rc=0`, `dspark-both-prefetch run_timed: rc=0`, `dspark-both-prefetch run_probe: rc=0`, then the summary JSON. Any nonzero rc ends the A/B: keep `$OUT` as a failed attempt and report it.

- [ ] **Step 9: Record provenance and evaluate the acceptance rule**

```bash
ssh divix01 bash -s <<'REMOTE'
OUT=$(ls -d /data/models/slang/nvfp4-work/ram-prefetch/ab-* | tail -1); echo "OUT=$OUT"
find /data/models/slang/nvfp4-work/ram-prefetch/jit-cache -name '*.so' -path '*expert_stream_host*prod*' -exec sha256sum {} + > $OUT/host-modules.sha256
find /data/models/slang/nvfp4-work/ram-prefetch/exl3-build -name '*.so' -exec sha256sum {} + > $OUT/exl3-ext.sha256
for arm in dspark-both dspark-both-prefetch; do run=$(ls -d $OUT/servers/$arm/run-* | tail -1); grep -o '"SGLANG_DSV41_RAM_PREFETCH": *"[01]"' $run/server-env-actual.json; grep -o '"SGLANG_JIT_CACHE_DIR": *"[^"]*"' $run/server-env-actual.json; done
cat $OUT/host-modules.sha256 $OUT/exl3-ext.sha256 | wc -l
/data/models/slang/.venv/bin/python - "$OUT" <<'PY'
import json, sys
s = json.load(open(sys.argv[1] + "/summary.json"))
a, b = s["dspark-both"], s["dspark-both-prefetch"]
gain = a["ms_per_token_median"] - b["ms_per_token_median"]
report = {
    "A_ms_per_token": a["ms_per_token_median"],
    "B_ms_per_token": b["ms_per_token_median"],
    "gain_ms": gain,
    "gain_pct": 100 * gain / a["ms_per_token_median"],
    "A_accept_length": a["accept_length"],
    "B_accept_length": b["accept_length"],
    "text_B_vs_A": b.get("text_vs_reference"),
    "rows_per_timed_token": [a.get("ram_rows_per_timed_token"), b.get("ram_rows_per_timed_token")],
    "B_ram": b.get("ram"),
}
print(json.dumps(report, indent=2))
accept = (
    gain > 0.03 * a["ms_per_token_median"]
    and (b.get("text_vs_reference") or {}).get("pass") is True
    and b["ram_rows_per_timed_token"] < a["ram_rows_per_timed_token"]
)
print("ACCEPT" if accept else "REJECT")
json.dump({**report, "verdict": "accept" if accept else "reject"}, open(sys.argv[1] + "/verdict.json", "w"), indent=2)
PY
REMOTE
```
Expected: `"SGLANG_DSV41_RAM_PREFETCH": "0"` for the A and `"1"` for the B, the private cache path for both (a missing line means the passthrough did not reach the server: note it in the write-up), at least one host module hash, the report JSON, and `ACCEPT` or `REJECT` by the spec's rule (median gain beyond the suite's ~3% spread, outputs within the near-tie band, RAM misses per token down). Harmful evictions are zero by construction (a wrong prediction costs a read and a pool slot, never an eviction); "demand delayed" is `spec_delayed` plus `spec_promoted` in `B_ram`.

- [ ] **Step 10: Hand the GPU back**

Tell the owner: production is still stopped, the GPU is free, and give the `OUT` paths of the smoke and the A/B. Do not restart production.

- [ ] **Step 11: Write the results into `DSV41_REFERENCE.md` §33.14 and the handoff**

Render the table from the verdict on divix01 and paste it:
```bash
ssh divix01 bash -s <<'REMOTE'
OUT=$(ls -d /data/models/slang/nvfp4-work/ram-prefetch/ab-* | tail -1)
/data/models/slang/.venv/bin/python - "$OUT" <<'PY'
import json, sys
v = json.load(open(sys.argv[1] + "/verdict.json"))
r = v["B_ram"]
print("| arm | ms/token (median) | accept length | RAM rows / timed token |")
print("|---|---:|---:|---:|")
print(f"| `dspark-both` (A) | {v['A_ms_per_token']:.2f} | {v['A_accept_length']:.2f} | {v['rows_per_timed_token'][0]:.1f} |")
print(f"| `dspark-both-prefetch` (B) | {v['B_ms_per_token']:.2f} | {v['B_accept_length']:.2f} | {v['rows_per_timed_token'][1]:.1f} |")
print()
print(f"Gain {v['gain_ms']:.2f} ms/token ({v['gain_pct']:.1f}%); text B vs A: pass={v['text_B_vs_A']['pass']}, "
      f"max gap {v['text_B_vs_A']['max_gap']}; verdict: {v['verdict']}.")
print(f"B's speculative rows (server lifetime): issued {r['spec_issued']}, landed {r['spec_landed']}, used {r['spec_used']}, "
      f"promoted {r['spec_promoted']}, dropped {r['spec_dropped']}, failed {r['spec_failed']}, demand reads delayed "
      f"{r['spec_delayed']}; precision at use {r['spec_used'] / max(1, r['spec_landed']):.2f}.")
PY
REMOTE
```

Add to `DSV41_REFERENCE.md`, directly after §33.13 (before `## Sources`), a section headed
`### 33.14 NVMe-to-RAM prefetch under DSpark: the live A/B (YYYY-MM-DD)`, the date being the A/B's, from its `ab-YYYYMMDD-HHMMSS` directory name, with these paragraphs in this order:
1. **Verdict**, one sentence: accepted or rejected by the spec's rule, and the gain in ms/token against the 4-6 ms/token Phase 0 expected.
2. **What runs**: one paragraph citing the spec and this plan; per group a speculative thread scores the next layer's gate (`sqrt(softplus(W x)) + b`) on a CPU record's staged input, one row per layer over both groups, read under the reader's turn into a private `kSpec` pool (2 slots per row and group) the device never maps; a forced CPU miss on a landed row swaps it in with no read.
3. **Result**: the rendered table and the two rendered sentences.
4. **Provenance**: the commit (`commit.txt`), the A/B and smoke `OUT` paths, the host module and EXL3 extension hashes, the unit selection and pass count from Step 5, and the scope note (RAM counters are server lifetime).
5. **Pointers**: code (`host/ram_prefetch.h`, `host/gate_scorer.h`, `ram_tier.h` swap and speculative state, `ram_thread.h`), tests (`test_exl3_ram_prefetch_*.py`), the driver arm.

Append to `NVME_PINNED_PREFETCH_HANDOFF.md` a section `## 13. 2026-10: the live prefetch under DSpark` of three sentences: the reopening (spec, §33.13), the A/B verdict with the gain and `spec_used/spec_landed`, and the pointer to §33.14. The default stays off unless the owner accepts it for production.

Commit and push:
```bash
cd /Users/dnikolaidis/.codex/worktrees/ram-prefetch/sglang-nvfp4
git add DSV41_REFERENCE.md NVME_PINNED_PREFETCH_HANDOFF.md
git commit -m "Record the RAM prefetch A/B under DSpark (DSV41_REFERENCE 33.14)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push origin codex/dsv41-ram-prefetch
```
Expected: the push succeeds; `git status --short` shows only the files that were dirty before this plan started.
