# DSV4.1: DSpark with the target's and the draft's CPU experts at once — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Serve DSV4.1 EXL3 with DSpark speculative decoding while both the target's CPU experts
(`SGLANG_DSV41_CPU_EXPERTS=1`) and the draft's CPU experts (`SGLANG_DSV41_ENABLE_DSPARK_CPU_EXPERTS=1`) are on, and
make that the production recipe once a server A/B against today's production clears the owner's bar.

**Architecture:** The target's CPU experts learn to serve a DSpark verify (6 tokens per layer) inside the captured
decode graph. The post kernel stages all M token rows and writes a per-lane **token table** (each token's routing
weight per lane). The CPU expert thread runs one M-row forward per job from that table, and the route tables seed each
token's output from its own partial. A new DIRECT knob, `SGLANG_MOE_EXPERT_GRAPH_GATHER_VICTIM_LANES` (V), gives VRAM
victims only to the first V miss lanes of a 32-lane verify record. The post makes every lane past them a **forced CPU
lane** (spill) instead of overflowing. When a forced lane cannot be the CPU's, the post serves the victims' prefix and
flags the forward, which takes the existing eager re-verify, kept GPU-only. The draft keeps its own lease channel and
thread on dedicated node-0 cores (12-15). The target's node-0 team shrinks to 6-11, and node 1 keeps 18-27.

**Tech Stack:** CUDA (JIT `lease_kernels.cuh`, `lease_device.cuh`, `exl3_route_tables.cuh`, `direct_gather.cuh`),
C++20 host (`host/cpu_experts.h`, `host/ram_tier.h`, `host/ffi_exports.h`), Python (sglang EXL3 expert stream, DIRECT
residency, server-args gate), pytest (registered CPU/GPU units, manual `test/manual/dsv41` CUDA suites), bash
(`run_arm.sh`, `launch_prod.sh`).

**Spec:** The owner's request (2026-10-06, relayed by team-lead): "we absolutely need to support both cpu_experts and
dspark cpu_experts at the same time". The evidence this plan argues from is `DSV41_REFERENCE.md` §33.3–§33.9 on
`origin/master`; §33.9 reaches this branch in Task 1. Base: `origin/dsv41-cpu-plan-m2` (the multi-token DSV4.1 CPU
plan, `CHUNK_M` token chunks, `MAX_M = 4`), which the verify's M-row forwards use.

---

## Owner decisions

Each needs the owner's call. The plan proceeds on the recommendation and records the choice where it lands.

1. **When production flips to DSpark.** Evidence today: graphed DSpark ran 2.85 tok/s (§33.9) against production's
   ~13 tok/s plain decode with CPU experts (§30.1, §33.5). §33.5's v2 projection, multi-token CPU experts in the verify,
   ranges from +25% to −13% against 13.17 tok/s. This plan builds the mode and a one-line switch (`PROD_DSPARK` in
   `arm_env.py`).
   **Recommendation:** flip only if the A/B (Task 13) shows `dspark-both` at or below production's median ms/token
   and passes the text bar, or on the owner's explicit go regardless. Task 14 is gated on this.
2. **Cores for the two CPU-expert clients.** Node 0 has no free physical core under today's recipe: 0-5 server, 6-15
   target, 16 copy, 17 RAM. Options:
   - **(a) Recommended.** The draft takes named cores 12-15 and the target's node-0 team shrinks to 6-11, leaving
     node 1's 18-27 as is. No code change: `ThreadingConfig` already leaves named draft cores out of every derived
     role. No spin interaction: each team spins only on its own cores. The draft weights are first-touch on node 0,
     local to its cores.
   - **(b) Time-share 6-15.** Both teams' keep-warm and idle spin (100 ms PAUSE after 2 ms of register work) would
     fight for the same logical cores. That needs spin coordination code and `ThreadingConfig` changes that refuse
     today's rules.
   - **(c) The draft on node 1's 32-34.** Only 3 cores, and the draft weights would be remote (node 0).
3. **Victim lanes V.** A 6-token verify routes up to 36 distinct experts per layer (§33.5: mean 21, p99 32).
   **Recommendation:** a 32-lane record (`MISS_LANES=32`) with `VICTIM_LANES=8`.
   - V = 8 is D2-3's miss width. It gives the same 8 VRAM victims and 8 staging slots per node as the measured
     graphed arms, so the DIRECT floor is 16 slots per layer (≤ ~23.7 at the hot cache below).
   - A larger V moves more lanes to the link and costs 2 pinned rows per layer per extra lane (~12.4 MiB each, §33.3).
4. **Hot cache under DSpark.** **Recommendation:** `SGLANG_MOE_HOT_GPU_MB=12040`, the hybrid draft's value (§33.4,
   §33.5), which leaves 4040 MiB of production's 16080 for the draft's resident experts, dense weights and KV. Task 13
   gates it: the server's KV pool must be no smaller than the production arm's in the same A/B.
5. **The eager re-verify stays GPU-only.** **Recommendation:** keep it so. Spill removes the cause, not the symptom.
   - At W = 8 every verify re-ran eagerly because every verify overflowed (§33.8, §33.9: 348/348).
   - With spill, a verify overflows only when a layer has more than 32 distinct misses, the copy engine is not yet
     armed, or a forced NVMe miss finds no staging slot.
   - Putting CPU experts in `_apply_streamed` would need a host-submitted job path into an engine that today only the
     tier thread feeds, while the service is paused for eager pinned-tier use (`before_host_use`).
   - Task 13 measures the re-verify rate. If it stays high, that is the next plan.

## Global Constraints

- Branch `dsv41-dspark-both-cpu`, worktree `/Users/dnikolaidis/Desktop/divix/sglang-nvfp4-dspark-both-cpu`, cut from
  `origin/dsv41-cpu-plan-m2`. Push only this branch (`git push origin dsv41-dspark-both-cpu`).
- No push to `master`. No amend, rebase, stash or force. A fix-up is a new commit.
- Do not edit `python/sglang/srt/model_executor/model_runner.py`: it is frozen (`.claude/rules/modify-component-must-read.md`).
- Never use the `fable` model for any subagent.
- Stage files by name (`git add <path>...`). Never `git add -A` or `.`.
- Every commit ends with:
  ```
  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01Y3EXKJdMpP7nwX3qcBi8Vq
  ```
- Running code (`.claude/rules/divix01-run-protocol.md`):
  - Commit and push, then on divix01 run in a private worktree at the pushed commit:
    ```bash
    ssh divix01 'git -C /data/models/slang/sglang fetch origin && \
      (git -C /data/models/slang/sglang worktree add --detach /data/models/slang/nvfp4-work/wt-both-cpu origin/dsv41-dspark-both-cpu \
       || git -C /data/models/slang/nvfp4-work/wt-both-cpu checkout --detach origin/dsv41-dspark-both-cpu) && \
      git -C /data/models/slang/nvfp4-work/wt-both-cpu log -1 --oneline'
    ```
  - Every command sets `PYTHONPATH=$PWD/python` in that worktree. Print `sglang.__file__` once per session and check
    it is under `wt-both-cpu`.
  - CPU jobs run under `taskset -c 0-63` with `OMP_NUM_THREADS=8`.
  - Read pytest's status as `; echo "EXIT=${PIPESTATUS[0]}"` whenever output is piped.
  - Record the exact command next to every result.
- GPU steps:
  - GPU work runs under `flock /data/models/slang/nvfp4-work/cc-gpu.lock`, on cores 32-63 unless the step names cores.
  - A step that needs both locks takes `rowimg-disk.lock` first.
  - Production holds `cc-gpu.lock` for its whole lifetime. The executor waits on the lock and **never starts or stops
    production**; GPU windows are the owner's.
- Environment variables go through `sglang.srt.environ.envs` (`env-var-conventions` skill). The one new variable is
  `SGLANG_MOE_EXPERT_GRAPH_GATHER_VICTIM_LANES = EnvInt(0)`.
- Speculative names follow the `speculative-naming` skill (`accept_length`, `spec_verify_ct`, `verify`).
- Non-speculative decode must stay byte-for-byte what it is:
  - Every new path is gated on `cpu_tokens_max > 1`, on `spill`, or on `victim_lanes < miss_rows`. A BS1 launch has
    none of them.
  - The BS1 suites named in each task must stay green.
- The verify is `gamma + 1 = 6` tokens at `speculative_dspark_block_size=5` (`speculative_hook.py:680-704`,
  `dspark_config.py:131-132`), not 5. Tests and comments say 6.

## Review Focus

1. **A captured verify before the copy engine arms** (the first `COPY_ENGINE_ARM_DECODES` forwards after capture, and
   every eager graph-path verify). No forced lane can be a CPU lane: `host_lanes` is false. Expected: the post serves
   the victims' prefix, writes the count, flags the forward, and the eager re-verify gives the right text. No trap.
   Owner: Task 3 (`armed=False` parametrization), Task 9 (unarmed replay).
2. **A forced NVMe miss when its home node's staging list is exhausted.** Expected: overflow and re-verify, never the
   unforced path's `__trap`. Owner: Task 3 (`test_a_forced_miss_without_staging_overflows_not_traps`), Task 9.
3. **A verify with fewer tokens than the rows hold** (a 3-token post into 4-token rows, a short eager verify). Expected:
   the host reads the count from the table header. It neither assumes `tokens_max` nor reads stale rows. Owner: Task 4,
   Task 5 (3 tokens in 4-token rows).
4. **A token that routes none of a job's lanes, and stale output rows from an earlier record.** Expected:
   - that token's row gets slot −1 everywhere and its partial is exactly 0;
   - the engine zeroes a non-accumulating per-token job's rows itself instead of trusting the kernel to;
   - and the real kernel skips only slot −1, never a zero weight (`forward_plan.hpp:699-702`).
   Owner: Task 5 (out rows pre-filled with 7.0, tokens that skip lanes).
5. **More than 32 distinct misses in one layer with spill on** (§33.5: max 36). Expected: the clamp flags it and serves
   the victims' prefix, exactly as without CPU experts. The post never sees a count above the record's lanes. Owner:
   Task 7 (`test_spill_clamps_only_a_count_past_the_lanes`).

---

## File structure

| File | Change | Responsibility |
|---|---|---|
| `python/sglang/srt/layers/moe/ram_slot_map.py` | modify | Python reference of lane typing: forced lanes, `LaneOverflow` |
| `python/sglang/kernels/jit/csrc/moe/expert_stream/lease_device.cuh` | modify | device `type_lanes`: forced lanes, returns false on overflow |
| `python/sglang/kernels/jit/csrc/moe/expert_stream/lease_kernels.cuh` | modify | post: spill fallback, M-token staging, token table |
| `python/sglang/kernels/jit/csrc/moe/expert_stream/cpu_token_table.h` | create | the token table's layout constants, shared by device and host |
| `python/sglang/kernels/ops/moe/expert_lease_block.py` | modify | `cpu_row_bytes`, the Python mirror of the row layout |
| `python/sglang/kernels/ops/moe/expert_stream_transport.py` | modify | `ExpertStreamDevice.post` (`spill`, M tokens), both `enable_cpu_experts` (4-dim rows) |
| `python/sglang/kernels/jit/csrc/moe/expert_stream/host/cpu_experts.h` | modify | `CpuJob.per_token/lanes`, `CpuExpertConfig.tokens`, `run_job` expansion |
| `python/sglang/kernels/jit/csrc/moe/expert_stream/host/ram_tier.h` | modify | record jobs carry their lanes and `per_token` |
| `python/sglang/kernels/jit/csrc/moe/expert_stream/host/ffi_exports.h` | modify | `enable_cpu_experts(tokens)` |
| `python/sglang/kernels/jit/csrc/moe/exl3/exl3_route_tables.cuh` | modify | lift the one-token CPU refusal |
| `python/sglang/srt/layers/quantization/exl3/fused_moe.py` | modify | lift `m != 1` with `cpu` |
| `python/sglang/srt/environ.py` | modify | `SGLANG_MOE_EXPERT_GRAPH_GATHER_VICTIM_LANES` |
| `python/sglang/srt/layers/moe/expert_residency_gpu.py` | modify | victim cap, idle destinations, clamp under spill, floor 2V |
| `python/sglang/kernels/jit/csrc/moe/expert_residency/direct_gather.cuh` | modify | idle destination params |
| `python/sglang/kernels/ops/moe/expert_residency_direct_gather.py` | modify | pass them |
| `python/sglang/srt/layers/moe/expert_hot_cache.py` | modify | allocator floor 2V |
| `python/sglang/srt/layers/moe/exl3_expert_format.py` | modify | plan the staging width V |
| `python/sglang/srt/layers/moe/exl3_ram_miss.py` | modify | staging width, CPU rows by tokens, attach, spill words |
| `python/sglang/srt/layers/moe/cpu_experts/service.py` | modify | rows sized by tokens |
| `python/sglang/srt/arg_groups/expert_stream_requirements_exl3.py` | modify | the gate |
| `benchmarks/dsv41_baseline/arm_env.py`, `launch_prod.sh` | modify | DSpark mode, `PROD_DSPARK` |
| `scripts/expert_prediction/benchmarks/run_capture_sessions.py` | modify | record `sglext.spec_tokens_details` |
| `scripts/dsv41/dspark_text_band.py` | create | the §33.2/§33.9 near-tie text bar |
| `analysis/dsv41-drive/dspark/both_cpu_ab.py` | create | the three-arm server A/B |
| `analysis/dsv41-drive/LEASE_PROTOCOL.md`, `DSV41_REFERENCE.md` | modify | protocol and results |

---

### Task 1: Bring master's DSpark draft graph into the branch

The base lacks `origin/master`'s last 24 commits: §33.9's draft graph, the draft watchdog fixes, and §33.9 itself. A
trial merge conflicts in one hunk only: both sides added fields to `DraftCpuThread`'s member block.

**Files:**
- Modify (merge): `python/sglang/kernels/jit/csrc/moe/expert_stream/host/draft_cpu_thread.h`

**Interfaces:**
- Consumes: `origin/master` at or after `3c115b9fb5`.
- Produces: the branch with `DraftResidentMoe`, the draft graph, and both sides' `DraftCpuThread` counters.

- [ ] **Step 1: Merge**

```bash
cd /Users/dnikolaidis/Desktop/divix/sglang-nvfp4-dspark-both-cpu
git fetch origin
git merge --no-ff --no-commit origin/master
git status --short | grep '^UU'
```
Expected: exactly `UU python/sglang/kernels/jit/csrc/moe/expert_stream/host/draft_cpu_thread.h`.

- [ ] **Step 2: Resolve the one hunk as the union of both sides**

Replace the conflict block (from `<<<<<<< HEAD` to `>>>>>>> origin/master`) with:

```cpp
  std::atomic<bool> stop_{false}, watchdog_stop_{false};  // the watchdog stops after the run thread has joined
  std::atomic<uint32_t> completed_{0}, head_at_stop_{0};
  std::atomic<int64_t> jobs_{0}, rows_{0}, forward_ns_{0}, holds_{0}, collided_jobs_{0}, shared_routes_{0},
      collided_forward_ns_{0};
```

Then `grep -n '<<<<<<<\|>>>>>>>' python/sglang/kernels/jit/csrc/moe/expert_stream/host/draft_cpu_thread.h`. Expected:
no output.

- [ ] **Step 3: Commit the merge and push**

```bash
git add python/sglang/kernels/jit/csrc/moe/expert_stream/host/draft_cpu_thread.h
git commit --no-edit \
  --trailer "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" \
  --trailer "Claude-Session: https://claude.ai/code/session_01Y3EXKJdMpP7nwX3qcBi8Vq"
git push origin dsv41-dspark-both-cpu
```

- [ ] **Step 4: Run the tests the merge touched, on divix01**

```bash
cd /data/models/slang/nvfp4-work/wt-both-cpu
PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 taskset -c 0-63 /data/models/slang/.venv/bin/python -m pytest -q -p no:randomly \
  test/registered/unit/kernels/test_dspark_draft_cpu_thread.py \
  test/registered/unit/kernels/test_expert_stream_build_variants.py 2>&1 | tail -3; echo "EXIT=${PIPESTATUS[0]}"
```
Expected: `EXIT=0`. Record the pass counts with the command.

---

### Task 2: The reference types forced lanes

`ram_slot_map.type_lanes` is the host reference the device typing transcribes, and the CUDA parity suite compares
against it. It learns **forced lanes**: the lanes from `forced_from` on found no VRAM victim and must be CPU lanes.

**Files:**
- Modify: `python/sglang/srt/layers/moe/ram_slot_map.py` (`type_lanes`, new `LaneOverflow`)
- Test: `test/registered/unit/kernels/test_ram_slot_map.py`

**Interfaces:**
- Produces: `type_lanes(..., forced_from: Optional[int] = None)`. It returns `(kinds, slots)` as before, and raises
  `LaneOverflow(ValueError)` when a forced lane cannot be the CPU's. The split counts only lanes `< forced_from`.

- [ ] **Step 1: Write the failing tests** (append to `test_ram_slot_map.py`; add `LaneOverflow` to the import)

```python
from sglang.srt.layers.moe.ram_slot_map import LaneKind, LaneOverflow, MapReplica, type_lanes


def test_forced_lanes_are_cpu_lanes_outside_the_split():
    """Lanes 0-1 found VRAM victims, lanes 2-3 did not (spill). The split sees only lane 0, the one eligible unforced
    lane (split[1] = 1), so it is the CPU's; lane 1, an unforced miss, stays on the GPU; the forced hit and the forced
    miss are CPU lanes whatever the split, the miss in the next staging slot."""
    ram = [-1] * 16
    ram[1], ram[2] = 4, 5
    staging = [9, 10] + [-1] * 6
    kinds, slots = _type([1, 0, 2, 7], ram, staging, cpu_on=True, forced_from=2)
    assert kinds == [LaneKind.HIT_CPU, LaneKind.MISS_GPU, LaneKind.HIT_CPU, LaneKind.MISS_CPU]
    assert slots == [4, 9, 5, 10]


def test_forced_from_the_count_forces_nothing():
    ram = [-1] * 16
    ram[1] = 4
    staging = [9, 10] + [-1] * 6
    assert _type([1, 0], ram, staging, cpu_on=True, forced_from=2) == _type([1, 0], ram, staging, cpu_on=True)


@pytest.mark.parametrize(
    "changes",
    [{"copy_armed": False}, {"captured": False}, {"cpu_on": False}, {"cpu_ok": False}],
    ids=["unarmed", "eager", "cpu-off", "no-cpu-layer"],
)
def test_a_forced_lane_that_cannot_be_the_cpus_overflows(changes):
    ram = [-1] * 16
    ram[1], ram[2] = 4, 5
    with pytest.raises(LaneOverflow):
        _type([1, 2], ram, forced_from=1, **{"cpu_on": True, **changes})


def test_a_forced_miss_without_staging_overflows_and_an_unforced_one_still_raises_plainly():
    with pytest.raises(LaneOverflow):
        _type([0, 7], [-1] * 16, [9] + [-1] * 7, cpu_on=True, forced_from=1)
    with pytest.raises(ValueError) as refused:
        _type([0, 7], [-1] * 16, NO_STAGING, cpu_on=True, forced_from=1)
    assert not isinstance(refused.value, LaneOverflow)
```

- [ ] **Step 2: Run them and see them fail**

On divix01 in the worktree (pushed first):
`PYTHONPATH=$PWD/python taskset -c 0-63 /data/models/slang/.venv/bin/python -m pytest -q test/registered/unit/kernels/test_ram_slot_map.py 2>&1 | tail -3; echo "EXIT=${PIPESTATUS[0]}"`.
Expected: an `ImportError` (`LaneOverflow`), `EXIT=2`.

- [ ] **Step 3: Implement**

In `ram_slot_map.py`, add above `type_lanes`:

```python
class LaneOverflow(ValueError):
    """A forced lane (one DIRECT found no VRAM victim for, the post's spill) cannot be a CPU lane. The post then serves
    the unforced prefix and flags the forward (exl3_ram_miss_post_kernel)."""
```

Add `forced_from: Optional[int] = None` as the last keyword of `type_lanes`. Append to its docstring:

```
    Lanes from ``forced_from`` on (spill, SGLANG_MOE_EXPERT_GRAPH_GATHER_VICTIM_LANES) found no VRAM victim: each is a
    CPU lane whatever the split, a hit or a miss with a staging slot, and the split counts only the lanes before it.
    Raises LaneOverflow when one cannot be (no host lanes, no CPU layer, no staging slot on its node).
```

Replace the body from `home = wire_layout(lanes, nodes).home` through the end of the `for j in reversed(...)` loop:

```python
    forced_from = len(experts) if forced_from is None else forced_from
    home = wire_layout(lanes, nodes).home
    slots, hit, taken = [], [], [0] * nodes
    for j, e in enumerate(experts):
        s = ram_slot[e]
        if s >= 0:
            slots.append(s)
            hit.append(True)
            continue
        node = home(e)
        m = taken[node]
        if m >= lanes or staging[node * lanes + m] < 0:
            if j >= forced_from:
                raise LaneOverflow(f"forced lane {j} has no staging slot on node {node}")
            raise ValueError(f"a miss lane has no staging slot on node {node}")
        slots.append(staging[node * lanes + m])
        hit.append(False)
        taken[node] += 1
    host_lanes = captured and copy_armed
    can_cpu = host_lanes and cpu_on and cpu_ok
    if forced_from < len(experts) and not can_cpu:
        raise LaneOverflow(f"lanes {forced_from}.. found no victim and cannot be CPU lanes")
    eligible = [j < forced_from and can_cpu and (h or cpu_misses) for j, h in enumerate(hit)]
    take = [0] * nodes
    for node in range(nodes):
        n = sum(1 for e, ok in zip(experts, eligible) if ok and home(e) == node)
        take[node] = split[node * (lanes + 1) + n] if n else 0
    cpu = [j >= forced_from for j in range(len(experts))]
    for j in reversed(range(len(experts))):
        node = home(experts[j])
        if take[node] > 0 and eligible[j]:
            cpu[j] = True
            take[node] -= 1
```
Everything after (`copy_ok = ...`, the kinds loop) is unchanged.

- [ ] **Step 4: Run and pass**

Same command. Expected: `EXIT=0`, every earlier test in the file still passing.

- [ ] **Step 5: Commit**

```bash
git add python/sglang/srt/layers/moe/ram_slot_map.py test/registered/unit/kernels/test_ram_slot_map.py
git commit -m "feat(exl3-ram-miss): the lane-typing reference takes forced lanes (spill) and raises LaneOverflow

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Y3EXKJdMpP7nwX3qcBi8Vq"
```

---

### Task 3: The post types forced lanes on the device and falls back to the victims' prefix

**Files:**
- Modify: `python/sglang/kernels/jit/csrc/moe/expert_stream/lease_device.cuh` (`LanePlan`, `type_lanes`)
- Modify: `python/sglang/kernels/jit/csrc/moe/expert_stream/lease_kernels.cuh` (`PostParams`, post kernel, launcher)
- Modify: `python/sglang/kernels/ops/moe/expert_stream_transport.py` (`ExpertStreamDevice.__init__`, `.post`)
- Modify: `test/manual/dsv41/test_exl3_lease_kernels_cuda.py` (the two raw `expert_stream_post` calls)
- Test: `test/manual/dsv41/test_exl3_slot_map_kernels_cuda.py`

**Interfaces:**
- Consumes: Task 2's `type_lanes(..., forced_from=)` and `LaneOverflow`.
- Produces:
  - `ExpertStreamDevice.post(..., spill=None)`, where `spill = (overflow_flag int32[1], gather_overflow int64[1])`
    on the device;
  - the post's FFI tail `..., cpu_weights, spill, overflow_flag, gather_overflow, use_pdl`;
  - with `spill`, a lane whose `dst_slots` entry is −1 is forced. On a forced lane that cannot be the CPU's, the post
    writes `count[0] = live`, sets `*overflow_flag = 1`, and adds 1 to `*gather_overflow`.

- [ ] **Step 1: Write the failing tests** (append to `test_exl3_slot_map_kernels_cuda.py`)

```python
from sglang.srt.layers.moe.ram_slot_map import LaneOverflow  # noqa: E402

ALL_CPU = list(range(W.lanes + 1))  # split[n] = n: every eligible unforced lane is the CPU's


def _post_spill(c, experts, forced_from, flag, overflows, row=0):
    """A captured post whose lanes from forced_from on have no VRAM victim (dst -1), as DIRECT's spill leaves them."""
    c.plan(experts, row)
    backend, plan = c.backends[row], c.plans[row]
    plan.slots[forced_from : len(experts)] = -1
    backend._stage_planned(plan)
    cpu_input = (torch.zeros(1, 64, device="cuda"), torch.ones(TOP_K, device="cuda"))
    c.dev.post(row, backend.planned, plan.count, backend.routes, plan.slots, captured=True, cpu_input=cpu_input,
               spill=(flag, overflows))
    torch.cuda.synchronize()
    count = int(plan.count[0])
    return count, c.kinds(count), c.dev.lane_slot[:count].tolist()


@pytest.mark.parametrize("armed", [True, False], ids=["armed", "unarmed"])
def test_post_spills_forced_lanes_like_the_reference(tmp_path, armed):
    """200 random maps, plans and spill points. Armed, every forced lane becomes a CPU lane whatever the split and the
    count stands. Unarmed (Review Focus 1), any forced lane overflows: the post serves the live prefix, writes its
    count, sets the flag and counts the overflow once. Mutation: type a forced lane by the split -- red."""
    c = Chain(tmp_path, start=False, copy_engine=True, hit_copy="ce", cpu_misses=False)
    try:
        hidden = 64
        c.dev.cpu_x_rows = torch.zeros((2, 2 * hidden), dtype=torch.uint8).pin_memory()
        c.dev.set_row_cpu(0)
        _set_host_words(c, armed=armed)
        flag = torch.zeros(1, dtype=torch.int32, device="cuda")
        overflows = torch.zeros(1, dtype=torch.int64, device="cuda")
        rng = random.Random(11)
        staging = rng.sample(range(8, 14), 6)
        _write_delta(c, 0, 1, staging)
        spilled = overflowed = 0
        for _ in range(200):
            ram = [-1] * EXPERTS
            for e, slot in zip(rng.sample(range(EXPERTS), rng.randint(0, W.lanes)), rng.sample(range(W.lanes), W.lanes)):
                ram[e] = slot
            c.dev.map_bulk_apply(torch.tensor([[0, e, s] for e, s in enumerate(ram)], dtype=torch.int32))
            experts = rng.sample(range(EXPERTS), rng.randint(1, TOP_K))
            forced_from = rng.randint(0, len(experts))
            flag.zero_()
            before = int(overflows.item())
            got = _post_spill(c, experts, forced_from, flag, overflows)
            ref = dict(captured=True, copy_armed=armed, hit_copy="ce", cpu_on=True, cpu_misses=False, lanes=W.lanes)
            try:
                kinds, slots = type_lanes(experts, ram, staging, SPLIT, forced_from=forced_from, **ref)
                want = (len(experts), [int(k) for k in kinds], slots, 0)
            except LaneOverflow:
                kinds, slots = type_lanes(experts[:forced_from], ram, staging, SPLIT, **ref)
                want = (forced_from, [int(k) for k in kinds], slots, 1)
            assert (*got, int(flag.item())) == want, (experts, ram, forced_from)
            assert int(overflows.item()) - before == want[3]
            spilled += want[3] == 0 and forced_from < len(experts)
            overflowed += want[3]
            chain = _chain_of_last_record(c)
            if chain:
                _write_delta(c, 0, chain, staging)
        assert spilled > 0 if armed else overflowed > 0
    finally:
        c.close()


def test_a_forced_miss_without_staging_overflows_not_traps(tmp_path):
    """Review Focus 2: one staging slot; an unforced miss takes it, a forced miss finds none. The post serves the
    prefix (2 lanes: a hit and the unforced miss) and flags the forward instead of trapping."""
    c = Chain(tmp_path, start=False, copy_engine=True, hit_copy="ce")
    try:
        c.dev.cpu_x_rows = torch.zeros((2, 128), dtype=torch.uint8).pin_memory()
        c.dev.set_row_cpu(0)
        _set_host_words(c, armed=True, split=ALL_CPU)
        _write_delta(c, 0, 1, [9])
        c.dev.map_bulk_apply(torch.tensor([[0, 1, 4]], dtype=torch.int32))
        flag = torch.zeros(1, dtype=torch.int32, device="cuda")
        overflows = torch.zeros(1, dtype=torch.int64, device="cuda")
        count, kinds, slots = _post_spill(c, [1, 0, 7], 2, flag, overflows)
        assert (count, int(flag.item()), int(overflows.item())) == (2, 1, 1)
        assert slots == [4, 9]
    finally:
        c.close()
```

- [ ] **Step 2: Run and see them fail**

On divix01, GPU lock, cores 32-63:
```bash
PYTHONPATH=$PWD/python flock /data/models/slang/nvfp4-work/cc-gpu.lock taskset -c 32-63 \
  /data/models/slang/.venv/bin/python -m pytest -q test/manual/dsv41/test_exl3_slot_map_kernels_cuda.py \
  -k "spill or forced" 2>&1 | tail -3; echo "EXIT=${PIPESTATUS[0]}"
```
Expected: `TypeError: post() got an unexpected keyword argument 'spill'`.

- [ ] **Step 3: Implement the device typing** (`lease_device.cuh`)

Replace `struct LanePlan`:

```cpp
// The plan as the post reads it, in device memory: `count` planned experts and their VRAM destination slots. Lanes
// from forced_from on found no VRAM victim (spill: DIRECT's narrowed verify with CPU experts) and must be CPU lanes;
// forced_from == count forces none.
struct LanePlan {
  const int64_t* planned;
  const int32_t* dst;
  int64_t count;
  int64_t forced_from;
};
```

Replace the comment above `type_lanes` and its signature, and change its first loop and its typing loop as below.
The `expert`/`dst`/`ram`/`staging` loads and the `take` computation stay as they are.

```cpp
// Types each lane of the plan: its kind and source slot. Transcribes ram_slot_map.type_lanes, the host reference.
//
// A hit takes its RAM slot, and a miss the next slot of its home node's staging list. Node n's CPU takes the last
// split[n][k] of its k eligible unforced lanes in plan order; a forced lane (j >= plan.forced_from) is a CPU lane
// whatever the split. Traps where the reference raises ValueError: a plan wider than Wire::kLanes, an expert out of
// range or repeated, a hit slot past the row's capacity, an unforced miss with no staging slot on its node, a split
// entry above its n. Returns false where the reference raises LaneOverflow: a forced lane that cannot be the CPU's (no
// host lanes, no CPU layer, no staging slot); `out` is then partial. Reads no host memory; the caller loads the split
// table into the policy.
SGL_DEVICE bool type_lanes(const LanePlan& plan, const RowMap& map, const LanePolicy& policy, TypedLanes& out) {
```

The first per-lane loop becomes:

```cpp
  bool hit[Wire::kLanes];
  bool eligible[Wire::kLanes];
  int m[Wire::kNodes] = {};
  int n[Wire::kNodes] = {};
  const bool can_cpu = policy.host_lanes && policy.cpu_on && policy.cpu_ok;
#pragma unroll
  for (int j = 0; j < Wire::kLanes; ++j) {
    if (j >= plan.count) break;
    for (int i = 0; i < j; ++i)
      if (expert[i] == expert[j]) __trap();
    const int node = Wire::home(expert[j]);
    const bool forced = j >= plan.forced_from;
    out.node[j] = node;
    hit[j] = ram[j] >= 0;
    if (hit[j]) {
      if (static_cast<uint32_t>(ram[j]) >= map.row_capacity) __trap();
      out.slot[j] = ram[j];
    } else {
      if (m[node] >= Wire::kLanes || staging[node * Wire::kLanes + m[node]] < 0) {
        if (forced) return false;
        __trap();
      }
      out.slot[j] = staging[node * Wire::kLanes + m[node]++];
    }
    if (forced && !can_cpu) return false;
    eligible[j] = !forced && can_cpu && (hit[j] || policy.cpu_misses);
    n[node] += eligible[j] ? 1 : 0;
  }
```

The typing loop becomes:

```cpp
  for (int64_t j = plan.count - 1; j >= 0; --j) {
    const int node = out.node[j];
    const bool forced = j >= plan.forced_from;
    const bool cpu = forced || (take[node] > 0 && eligible[j]);
    if (cpu && !forced) --take[node];
    uint8_t kind;
    if (cpu) {
      kind = hit[j] ? Wire::kKindHitCpu : Wire::kKindMissCpu;
    } else if (hit[j]) {
      kind = copy_ok && dst[j] >= 0 && dst[j] < policy.dst_rows ? Wire::kKindHitCopy : Wire::kKindHitSm;
    } else {
      kind = Wire::kKindMissGpu;
    }
    out.kind[j] = kind;
  }
  return true;
}
```

- [ ] **Step 4: Implement the post** (`lease_kernels.cuh`)

In `PostParams`, change `const int32_t* count;` to `int32_t* count;  // the plan's miss count; a spill overflow lowers it`
and append after `cpu_weights_count`:

```cpp
  // Spill (GpuResidencyUpdater.victim_lanes below its miss lanes, CPU experts on): a lane whose dst_slots entry is -1
  // found no VRAM victim and must be a CPU lane. When one cannot be, the post serves the live prefix, writes it to
  // count and flags the forward in DIRECT's words: overflow_flag (int32, sticky) and gather_overflow (this layer's
  // int64 counter). Both unused when spill is 0.
  int64_t spill;
  int32_t* overflow_flag;
  int64_t* gather_overflow;
```

In `exl3_ram_miss_post_kernel`, change `const int64_t count = ...` to `int64_t count = ...` (thread 0 may lower it; the
other threads read only `any_cpu`). Replace

```cpp
      type_lanes(LanePlan{.planned = p.planned, .dst = p.dst_slots, .count = count}, map, policy, typed);
```
with
```cpp
      // Spill: DIRECT gives the lanes past its victims destination -1 (GpuResidencyUpdater.gather_destinations), and
      // the live lanes are a prefix.
      int64_t live = count;
      if (p.spill != 0)
        for (int64_t j = 0; j < count; ++j)
          if (p.dst_slots[j] < 0) {
            live = j;
            break;
          }
      if (!type_lanes(
              LanePlan{.planned = p.planned, .dst = p.dst_slots, .count = count, .forced_from = live},
              map,
              policy,
              typed)) {
        // A forced lane cannot be the CPU's: serve the live prefix and flag the forward, as clamp_gather_misses does
        // without CPU experts. S, CW, CC and the DIRECT commit read the count written here.
        count = live;
        p.count[0] = static_cast<int32_t>(live);
        *p.overflow_flag = 1;
        *p.gather_overflow += 1;
        type_lanes(
            LanePlan{.planned = p.planned, .dst = p.dst_slots, .count = live, .forced_from = live}, map, policy, typed);
      }
```

In the launcher `post(...)`, add parameters after `tvm::ffi::TensorView cpu_weights,`:

```cpp
      int64_t spill,
      tvm::ffi::TensorView overflow_flag,
      tvm::ffi::TensorView gather_overflow,
```
Before `const auto stream = ...`, add:
```cpp
    if (spill != 0) {
      expert_stream::verify_named(
          "overflow_flag", TensorMatcher({1}).with_dtype<int32_t>().with_device<kDLCUDA>(device), overflow_flag);
      expert_stream::verify_named(
          "gather_overflow", TensorMatcher({1}).with_dtype<int64_t>().with_device<kDLCUDA>(device), gather_overflow);
    }
```
In the `PostParams{...}` initializer, change `.count = static_cast<const int32_t*>(count.data_ptr()),` to
`.count = static_cast<int32_t*>(count.data_ptr()),` and append:
```cpp
        .spill = spill,
        .overflow_flag = spill != 0 ? static_cast<int32_t*>(overflow_flag.data_ptr()) : nullptr,
        .gather_overflow = spill != 0 ? static_cast<int64_t*>(gather_overflow.data_ptr()) : nullptr,
```

- [ ] **Step 5: Python** (`expert_stream_transport.py`)

In `ExpertStreamDevice.__init__`, after `self._no_cpu = ...`:
```python
        # The spill words of a post without spill: never read (the post's spill is 0).
        self._no_spill = (
            torch.zeros(1, dtype=torch.int32, device=device),
            torch.zeros(1, dtype=torch.int64, device=device),
        )
```
`post` gains a last keyword `spill=None`. Append to its docstring:
```
        ``spill`` is DIRECT's ``(overflow_flag int32 [1], gather_overflow int64 [1])`` for this row, given when lanes
        may have no VRAM victim (SGLANG_MOE_EXPERT_GRAPH_GATHER_VICTIM_LANES): a lane whose ``dst_slots`` entry is -1
        becomes a CPU lane, or, when one cannot, the post serves the live prefix, lowers ``count`` and flags both words.
```
Before `bank = self.map_bank`:
```python
        spill_on, (overflow_flag, gather_overflow) = 0, self._no_spill
        if spill is not None:
            overflow_flag, gather_overflow = spill
            for name, t, dtype in (
                ("overflow_flag", overflow_flag, torch.int32),
                ("gather_overflow", gather_overflow, torch.int64),
            ):
                if t.dtype != dtype or t.numel() != 1 or t.device != self.state.device:
                    raise ValueError(f"{name} must be one {dtype} word on {self.state.device}")
            spill_on = 1
```
In the `expert_stream_post(...)` call, insert after `cpu_weights,`:
```python
            spill_on,
            overflow_flag,
            gather_overflow,
```

- [ ] **Step 6: Update the raw calls in `test_exl3_lease_kernels_cuda.py`**

Both direct `expert_stream_post(` calls end in the CPU triple, then `use_pdl`:
- In the first, replace the tail `no_i32, 0, no_i32, 0,` with
  `no_i32, 0, no_i32, 0, torch.zeros(1, dtype=torch.int32, device="cuda"), torch.zeros(1, dtype=torch.int64, device="cuda"), 0,`.
- In the second, replace `*cpu_args, 0,` with
  `*cpu_args, 0, torch.zeros(1, dtype=torch.int32, **cuda), torch.zeros(1, dtype=torch.int64, **cuda), 0,`.

Then confirm nothing else calls the module directly:
`grep -rn "expert_stream_post(" python test analysis scripts benchmarks | grep -v "\.cuh:\|\.h:"`. Expected: only
`expert_stream_transport.py` and the two lines above.

- [ ] **Step 7: Run and pass, plus the BS1 lease suites**

```bash
PYTHONPATH=$PWD/python flock /data/models/slang/nvfp4-work/cc-gpu.lock taskset -c 32-63 \
  /data/models/slang/.venv/bin/python -m pytest -q -p no:randomly \
  test/manual/dsv41/test_exl3_slot_map_kernels_cuda.py test/manual/dsv41/test_exl3_lease_kernels_cuda.py \
  test/manual/dsv41/test_exl3_cpu_lane_order_cuda.py test/manual/dsv41/test_exl3_lease_ordering_cuda.py \
  2>&1 | tail -3; echo "EXIT=${PIPESTATUS[0]}"
```
Expected: `EXIT=0`. The first run rebuilds the JIT modules (50-100 s each, `divix01-run-protocol.md`); a
`TimeoutExpired` on a cold cache is the compiler, not a hang.

- [ ] **Step 8: Commit**

```bash
git add python/sglang/kernels/jit/csrc/moe/expert_stream/lease_device.cuh \
  python/sglang/kernels/jit/csrc/moe/expert_stream/lease_kernels.cuh \
  python/sglang/kernels/ops/moe/expert_stream_transport.py \
  test/manual/dsv41/test_exl3_slot_map_kernels_cuda.py test/manual/dsv41/test_exl3_lease_kernels_cuda.py
git commit -m "feat(exl3-ram-miss): the post makes victimless lanes CPU lanes (spill), else serves the prefix and flags

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Y3EXKJdMpP7nwX3qcBi8Vq"
```

---

### Task 4: The post stages every token and writes the token table

**Files:**
- Create: `python/sglang/kernels/jit/csrc/moe/expert_stream/cpu_token_table.h`
- Modify: `python/sglang/kernels/jit/csrc/moe/expert_stream/lease_kernels.cuh`
- Modify: `python/sglang/kernels/ops/moe/expert_lease_block.py` (`cpu_row_bytes`, constants)
- Modify: `python/sglang/kernels/ops/moe/expert_stream_transport.py` (`ExpertStreamDevice.__init__`, `.enable_cpu_experts`, `.post`, `.cpu_out_part_stride`)
- Modify: `test/manual/dsv41/test_exl3_lease_kernels_cuda.py` (raw calls again)
- Test: `test/manual/dsv41/test_exl3_slot_map_kernels_cuda.py`

**Interfaces:**
- Consumes: Task 3's FFI tail.
- Produces:
  - FFI tail `..., cpu_weights, cpu_tokens_max, cpu_x_token_bytes, spill, overflow_flag, gather_overflow, use_pdl`.
  - `expert_lease_block.cpu_row_bytes(hidden, tokens, lanes) -> int`, `CPU_TOKEN_TABLE_HEADER = 16`,
    `CPU_TOKENS_MAX = 32`.
  - `ExpertStreamDevice.cpu_tokens_max`, `.cpu_x_token_bytes`. `enable_cpu_experts` takes `out_rows` of shape
    `[layers, 2 * nodes, tokens, hidden]` for a multi-token row.
  - Row layout: `tokens_max` inputs of `ceil16(2 * hidden)` bytes each. With `tokens_max > 1` the table follows them:
    - `+0`: u32 tokens;
    - `+16 + 4 j`: u32 lane j's token mask;
    - `+16 + 4 lanes + 4 (t lanes + j)`: fp32 bits of token t's weight for lane j.

- [ ] **Step 1: Write the failing test** (append to `test_exl3_slot_map_kernels_cuda.py`)

```python
from sglang.kernels.ops.moe.expert_lease_block import CPU_TOKEN_TABLE_HEADER, cpu_row_bytes  # noqa: E402


def test_a_verify_post_stages_every_token_and_the_token_table(idle):
    """Three tokens in rows that hold four (Review Focus 3); three RAM-hit lanes, all the CPU's. Each token's input is
    staged at its own offset as fp16, the header holds 3, each lane's mask has a bit per token that routes its expert,
    and each such token's own weight sits in the table. The record's lane weight is still the sum (the one-token
    path's). Mutations: stage token 0 only; t = r / route_count instead of r / top_k; mask from the summed weight."""
    c = idle
    hidden, tokens_max, lanes = 64, 4, W.lanes
    xb = 2 * hidden
    x_rows = torch.zeros((2, cpu_row_bytes(hidden, tokens_max, lanes)), dtype=torch.uint8).pin_memory()
    out_rows = torch.zeros((2, 2, tokens_max, hidden), dtype=torch.float32).pin_memory()
    c.dev.enable_cpu_experts(x_rows, out_rows)
    assert (c.dev.cpu_tokens_max, c.dev.cpu_x_token_bytes) == (tokens_max, xb)
    c.dev.set_row_cpu(0)
    _set_host_words(c, armed=True, split=ALL_CPU)
    _write_delta(c, 0, 1, [9, 10, 11, 12, 13, 14])
    experts = [3, 5, 7]
    c.dev.map_bulk_apply(torch.tensor([[0, e, s] for s, e in enumerate(experts)], dtype=torch.int32))
    c.plan(experts)
    backend, plan = c.backends[0], c.plans[0]
    backend._stage_planned(plan)
    routes = [[3, 5, 8, 9, 10, 11], [7, 3, 12, 13, 14, 15], [5, 7, 8, 12, 9, 13]]
    weights = torch.tensor([[0.01 * (10 * t + i + 1) for i in range(TOP_K)] for t in range(3)], device="cuda")
    x = torch.randn(3, hidden, device="cuda")
    route_ids = torch.tensor(sum(routes, []), dtype=torch.int64, device="cuda")
    c.dev.post(0, backend.planned, plan.count, route_ids, plan.slots, captured=True, cpu_input=(x, weights))
    torch.cuda.synchronize()
    assert c.kinds(3) == [LaneKind.HIT_CPU] * 3
    row = x_rows[0]
    for t in range(3):
        assert torch.equal(row[t * xb : (t + 1) * xb].view(torch.float16), x[t].half().cpu()), t
    table = tokens_max * xb
    assert int(row[table : table + 4].view(torch.int32)[0]) == 3
    planned = backend.planned[:3].tolist()
    for j, expert in enumerate(planned):
        at = table + CPU_TOKEN_TABLE_HEADER + 4 * j
        mask = int(row[at : at + 4].view(torch.int32)[0])
        assert mask == sum(1 << t for t in range(3) if expert in routes[t]), (j, expert)
        for t in range(3):
            if expert in routes[t]:
                at = table + CPU_TOKEN_TABLE_HEADER + 4 * lanes + 4 * (t * lanes + j)
                assert float(row[at : at + 4].view(torch.float32)[0]) == pytest.approx(
                    float(weights[t, routes[t].index(expert)])
                ), (j, t)
```

- [ ] **Step 2: Run and see it fail**

Same command as Task 3 Step 2 with `-k "every_token"`. Expected: `ImportError: cannot import name 'CPU_TOKEN_TABLE_HEADER'`.

- [ ] **Step 3: Create `cpu_token_table.h`**

```cpp
// The token table of a CPU experts input row that holds several tokens (plan
// 2026-10-06-dsv41-dspark-both-cpu-experts, a DSpark verify).
//
// A row holds tokens_max staged inputs, x_token_bytes apart (fp16, each padded to 16 bytes). With tokens_max > 1 the
// table follows them; the post kernel writes it for every lane of the record:
//   +0                                       u32  the record's tokens, 1..tokens_max
//   +kHeaderBytes + 4 j                      u32  lane j's token mask: bit t when token t routes lane j's expert
//   +kHeaderBytes + 4 lanes + 4 (t lanes + j)  u32  the fp32 bits of token t's routing weight for lane j's expert
// The CPU expert thread reads it to run one forward of `tokens` rows, each with its own weights
// (CpuExpertEngine::run_job). A one-token row has no table: the record's lane weight serves. The Python mirror is
// expert_lease_block.cpu_row_bytes.
#pragma once

#include <cstdint>

namespace sglang::expert_stream {

struct CpuTokenTable {
  static constexpr int64_t kMaxTokens = 32;    // one u32 mask bit per token
  static constexpr int64_t kHeaderBytes = 16;  // the token count, padded
};

}  // namespace sglang::expert_stream
```

- [ ] **Step 4: The post** (`lease_kernels.cuh`)

Add `#include "cpu_token_table.h"` after `#include "lease_device.cuh"`. In `PostParams`, after `cpu_weights_count`:
```cpp
  // A verify's tokens (cpu_x's rows; 1 for one token), the host rows' token capacity (1: one-token rows with no token
  // table) and the bytes between two tokens' staged inputs (cpu_token_table.h).
  int64_t cpu_tokens;
  int64_t cpu_tokens_max;
  int64_t cpu_x_token_bytes;
```
Replace `stage_cpu_input`:
```cpp
// Stages the layer's input rows as fp16 into the host row, token t at t * cpu_x_token_bytes, 8 elements per 16-byte
// store, using the whole block. The launcher checks hidden % 8 == 0 and the alignment. Each thread fences its own
// stores at system scope before the barrier, so thread 0's later release of the record and demand_head orders all.
SGL_DEVICE void stage_cpu_input(const PostParams& p) {
  const int64_t vectors = p.cpu_hidden / 8;
  for (int64_t v = threadIdx.x; v < p.cpu_tokens * vectors; v += blockDim.x) {
    const int64_t t = v / vectors, c = v % vectors;
    __align__(16) __half h[8];
#pragma unroll
    for (int k = 0; k < 8; ++k)
      h[k] = __float2half_rn(cpu_input_value(p.cpu_x_src, p.cpu_x_dtype, t * p.cpu_hidden + 8 * c + k));
    __stwt(
        reinterpret_cast<uint4*>(p.cpu_x_dst + t * p.cpu_x_token_bytes + 16 * c),
        *reinterpret_cast<const uint4*>(h));
  }
  __threadfence_system();
  __syncthreads();
}
```
In the kernel, after `if (threadIdx.x != 0) return;`, add:
```cpp
  // A multi-token row's token table (cpu_token_table.h): written for every lane before the record's release.
  uint8_t* const table = p.cpu_x_dst != nullptr && p.cpu_tokens_max > 1
                             ? p.cpu_x_dst + p.cpu_tokens_max * p.cpu_x_token_bytes
                             : nullptr;
  const int64_t top_k = p.cpu_tokens > 0 ? p.cpu_weights_count / p.cpu_tokens : 0;
  if (table != nullptr) st_relaxed_sys<uint32_t>(table, static_cast<uint32_t>(p.cpu_tokens));
```
Replace the lane-weight block (`weight[j] = 0.0f; if (p.cpu_x_dst != nullptr) { for ... } }`) with:
```cpp
    // The lane expert's routing weight summed over the routes that name it (one route at batch size 1), and for a
    // multi-token row each token's own weight and the mask of tokens that route it. 0 when CPU experts are off.
    weight[j] = 0.0f;
    if (p.cpu_x_dst != nullptr) {
      uint32_t routed = 0;
      for (int64_t r = 0; r < p.route_count && r < p.cpu_weights_count; ++r) {
        if (p.routes[r] != p.planned[j]) continue;
        const float w = cpu_input_value(p.cpu_weights, p.cpu_weights_dtype, r);
        weight[j] += w;
        if (table != nullptr) {
          const int64_t t = r / top_k;
          routed |= 1u << t;
          st_relaxed_sys<uint32_t>(
              table + expert_stream::CpuTokenTable::kHeaderBytes + 4 * Wire::kLanes + 4 * (t * Wire::kLanes + j),
              __float_as_uint(w));
        }
      }
      if (table != nullptr) st_relaxed_sys<uint32_t>(table + expert_stream::CpuTokenTable::kHeaderBytes + 4 * j, routed);
    }
```
In the launcher, add parameters after `cpu_weights,` (before Task 3's `spill`): `int64_t cpu_tokens_max, int64_t cpu_x_token_bytes,`.
Replace the check `RuntimeCheck(cpu_x.dim() == 2 && cpu_x.size(0) == 1, ...)` with:
```cpp
      RuntimeCheck(
          cpu_tokens_max >= 1 && cpu_tokens_max <= expert_stream::CpuTokenTable::kMaxTokens,
          "CPU experts: the rows hold 1-32 tokens");
      RuntimeCheck(
          cpu_x.dim() == 2 && cpu_x.size(0) >= 1 && cpu_x.size(0) <= cpu_tokens_max,
          "CPU experts: cpu_x is [tokens, hidden] with at most the rows' tokens");
      RuntimeCheck(cpu_weights.numel() % cpu_x.size(0) == 0, "CPU experts: cpu_weights holds top_k weights a token");
```
After `RuntimeCheck(cpu_hidden > 0 && cpu_hidden % 8 == 0, ...)` add:
```cpp
      RuntimeCheck(
          cpu_x_token_bytes >= 2 * cpu_hidden && cpu_x_token_bytes % 16 == 0,
          "CPU experts: cpu_x_token_bytes holds one fp16 row, in 16-byte steps");
```
Append to `PostParams{...}`:
```cpp
        .cpu_tokens = cpu_input ? cpu_x.size(0) : 1,
        .cpu_tokens_max = cpu_tokens_max,
        .cpu_x_token_bytes = cpu_x_token_bytes,
```

- [ ] **Step 5: Python**

`expert_lease_block.py`, at module level after the existing constants:
```python
# The CPU experts' input rows (cpu_token_table.h): CpuTokenTable::kHeaderBytes and kMaxTokens.
CPU_TOKEN_TABLE_HEADER = 16
CPU_TOKENS_MAX = 32


def cpu_row_bytes(hidden: int, tokens: int, lanes: int) -> int:
    """Bytes of one CPU experts input row: ``tokens`` fp16 inputs, each padded to 16 bytes, then, for more than one
    token, the token table: a 16-byte header, a u32 mask per lane and an fp32 weight per token and lane."""
    x = -(-2 * hidden // 16) * 16
    return tokens * x + (CPU_TOKEN_TABLE_HEADER + 4 * lanes + 4 * tokens * lanes if tokens > 1 else 0)
```
`expert_stream_transport.py`:
- Import `CPU_TOKENS_MAX, cpu_row_bytes` from `expert_lease_block`.
- In `ExpertStreamDevice.__init__`, after `self.cpu_out_rows = None`:
  ```python
          # The rows' token capacity and the bytes between two tokens' staged inputs (enable_cpu_experts); 1 and 0
          # until then.
          self.cpu_tokens_max, self.cpu_x_token_bytes = 1, 0
  ```
- In `enable_cpu_experts`, replace the final `self.cpu_x_rows, self.cpu_out_rows = x_rows, out_rows` with:
  ```python
          tokens = int(out_rows.shape[2]) if out_rows.dim() == 4 else 1
          hidden = int(out_rows.shape[-1])
          if not 1 <= tokens <= CPU_TOKENS_MAX:
              raise ValueError(f"CPU expert rows hold 1-{CPU_TOKENS_MAX} tokens, not {tokens}")
          if x_rows.shape[1] < cpu_row_bytes(hidden, tokens, self.wire.lanes):
              raise ValueError(
                  f"x_rows holds {x_rows.shape[1]} bytes a row; {tokens} tokens of {hidden} need "
                  f"{cpu_row_bytes(hidden, tokens, self.wire.lanes)}"
              )
          self.cpu_x_rows, self.cpu_out_rows = x_rows, out_rows
          self.cpu_tokens_max, self.cpu_x_token_bytes = tokens, -(-2 * hidden // 16) * 16
  ```
  Append to its docstring: ``A multi-token row's ``out_rows`` is ``[layers, 2 * nodes, tokens, hidden]``; ``x_rows``
  then holds ``expert_lease_block.cpu_row_bytes`` a row (the token table follows the inputs).``
- `cpu_out_part_stride`: change `if self.cpu_out_rows.dim() == 3` to `if self.cpu_out_rows.dim() >= 3`. Its docstring
  gains: "(``tokens * hidden`` for a multi-token row)".
- In `post`, replace the CPU input block's shape handling:
  ```python
              cpu_x, cpu_weights = cpu_input
              tokens = cpu_x.shape[0] if cpu_x.dim() == 2 else 1
              if tokens > self.cpu_tokens_max:
                  raise ValueError(f"a {tokens}-token input does not fit rows of {self.cpu_tokens_max} tokens")
              if cpu_x.shape[-1] * 2 > self.cpu_x_rows.shape[1]:
                  raise ValueError(
                      f"a {cpu_x.shape[-1]}-wide input does not fit the {self.cpu_x_rows.shape[1]}-byte row"
                  )
              cpu_x, cpu_weights = cpu_x.reshape(tokens, -1), cpu_weights.reshape(-1)
              cpu_x_dst = int(self.cpu_x_rows[row].data_ptr())
  ```
  Before the module call, add
  `token_bytes = self.cpu_x_token_bytes or -(-2 * int(cpu_x.shape[-1]) // 16) * 16`. In the call, insert after
  `cpu_weights,` (before `spill_on,`): `self.cpu_tokens_max, token_bytes,`. The `cpu_x.reshape(1, -1)` this replaces
  was the latent overflow the research found (an `[M, H]` input became one `[1, M*H]` row).

- [ ] **Step 6: Raw calls again** (`test_exl3_lease_kernels_cuda.py`)

Insert `1, 16,` right after the CPU triple in both calls, before Task 3's three spill arguments. The first call's tail
becomes `no_i32, 0, no_i32, 1, 16, 0, torch.zeros(1, ...int32...), torch.zeros(1, ...int64...), 0,`. The second's
becomes `*cpu_args, 1, 32, 0, ...` (its CPU input is `[1, 8]` fp32, so 16 bytes of fp16 suffice; 32 is fine too).

- [ ] **Step 7: Run and pass** — same suites as Task 3 Step 7. Expected: `EXIT=0`.

- [ ] **Step 8: Commit**

```bash
git add python/sglang/kernels/jit/csrc/moe/expert_stream/cpu_token_table.h \
  python/sglang/kernels/jit/csrc/moe/expert_stream/lease_kernels.cuh \
  python/sglang/kernels/ops/moe/expert_lease_block.py python/sglang/kernels/ops/moe/expert_stream_transport.py \
  test/manual/dsv41/test_exl3_slot_map_kernels_cuda.py test/manual/dsv41/test_exl3_lease_kernels_cuda.py
git commit -m "feat(exl3-cpu-experts): a verify's post stages every token and each lane's per-token weights

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Y3EXKJdMpP7nwX3qcBi8Vq"
```

---

### Task 5: The CPU expert thread runs one M-row forward per job

**Files:**
- Modify: `python/sglang/kernels/jit/csrc/moe/expert_stream/host/cpu_experts.h`
- Modify: `python/sglang/kernels/jit/csrc/moe/expert_stream/host/ram_tier.h` (`submit_host_lanes`, `submit_landed_cpu_misses`)
- Modify: `python/sglang/kernels/jit/csrc/moe/expert_stream/host/ffi_exports.h` (`enable_cpu_experts`)
- Modify: `python/sglang/kernels/ops/moe/expert_stream_transport.py` (`ExpertStreamHost.enable_cpu_experts`)
- Test: `test/manual/dsv41/test_exl3_cpu_lane_order_cuda.py`

**Interfaces:**
- Consumes: Task 4's row layout and `cpu_row_bytes`.
- Produces:
  - `CpuJob.per_token` (bool) and `CpuJob.lanes[kLanes]`;
  - `CpuExpertConfig.tokens`, `.x_token_bytes`;
  - FFI `expert_stream_enable_cpu_experts(handle, group, kernel, split, cores, x_rows, out_rows, hidden, parts, tokens, threads, spin_ns, keep_warm_ns)`;
  - `ExpertStreamHost.enable_cpu_experts` accepts 4-dim `out_rows`.
  - A per-token job on a multi-token row runs `rows = tokens` from the table. The engine zeroes its rows first when the
    job does not accumulate, then always accumulates.

- [ ] **Step 1: Write the failing test** (append to `test_exl3_cpu_lane_order_cuda.py`)

```python
@pytest.mark.parametrize("tokens", [1, 3])
def test_a_verify_records_cpu_lanes_run_one_forward_of_its_tokens(tmp_path, tokens):
    """Three RAM hits, every lane the CPU's (split[3] = 3), in rows that hold 4 tokens. The record's CPU job runs one
    forward of `tokens` rows: row t names a lane's RAM slot only where token t routes its expert (-1 elsewhere, Review
    Focus 4) with t's own weight, and its output row is t's partial alone, over stale bytes (7.0). One token in 4-token
    rows reads the table too. Mutations: every row the record's summed weight; the slot kept where t does not route;
    no zeroing of a non-accumulating job's rows."""
    from sglang.kernels.ops.moe.expert_cache_transfer import copy_expert_row_segments_gpu

    row, experts, tokens_max = 0, [3, 5, 7], 4
    lanes = lease.wire_layout(8).lanes
    c = Chain(tmp_path, copy_engine=True, start=False)
    try:
        x_rows = torch.zeros((LAYERS, lease.cpu_row_bytes(HIDDEN, tokens_max, lanes)), dtype=torch.uint8).pin_memory()
        out_rows = torch.full((LAYERS, 2, tokens_max, HIDDEN), 7.0, dtype=torch.float32).pin_memory()
        cores = sorted(os.sched_getaffinity(0))[:2]
        c.host.enable_cpu_experts(c.host.test_kernel_address(), list(range(lanes + 1)), cores, x_rows, out_rows,
                                  threads=2)
        c.host.start_thread(fatal_wait_s=60.0)
        c.dev.enable_cpu_experts(x_rows, out_rows)
        c.dev.set_row_cpu(row)
        backend, plan, dev = c.backends[row], c.plans[row], c.dev
        c.plan(experts, row)
        c.gather(row)
        torch.cuda.synchronize()
        assert c.handled() and set(experts) <= c.resident(row)
        with paused(c.host):
            ram_slot = {e: s for s, (state, e, _) in enumerate(c.host.slot_info(row)) if e >= 0}
        c.host.set_cpu_layer(row, fake_cpu_layer(HIDDEN))
        c.host.arm_copy_engine()

        c.plan(experts, row)
        backend._stage_planned(plan)
        planned = backend.planned[:3].tolist()
        routes = [[3, 5, 8, 9, 10, 11], [7, 3, 12, 13, 14, 15], [5, 8, 12, 9, 13, 10]][:tokens]
        weights = torch.tensor([[0.01 * (10 * t + i + 1) for i in range(TOP_K)] for t in range(tokens)], device="cuda")
        route_ids = torch.tensor(sum(routes, []), dtype=torch.int64, device="cuda")
        x = torch.randn(tokens, HIDDEN, device="cuda").half()
        dev.post(row, backend.planned, plan.count, route_ids, plan.slots, captured=True, cpu_input=(x, weights))
        copy_expert_row_segments_gpu(backend.segments[0], dev.host_rows_1, dev.dst_slots_1, dev.go_1)
        dev.stream(row, backend.planned, plan.count, plan.slots, backend.segments[0], backend.stream_maps[0])
        dev.copy_wait(plan.count, plan.slots, backend.copy_sm_table)
        assert _until(lambda: len(c.host.test_kernel_calls()) == tokens)
        torch.cuda.synchronize()

        assert c.kinds(3) == [LaneKind.HIT_CPU] * 3
        assert dev.cpu_lanes.tolist() == [0b111, PART_HITS]
        for t, call in enumerate(c.host.test_kernel_calls()):
            want_slots = [ram_slot[e] if e in routes[t] else -1 for e in planned]
            want_weights = [float(weights[t, routes[t].index(e)]) if e in routes[t] else 0.0 for e in planned]
            assert call["slots"] == want_slots, (t, call)
            assert call["weights"] == pytest.approx(want_weights), (t, call)
            assert call["accumulate"], "a per-token job accumulates into the rows the engine zeroed"
            partial = sum(w * (s + 1) for s, w in zip(want_slots, want_weights))
            assert torch.allclose(out_rows[row, 0, t], torch.full((HIDDEN,), partial), atol=1e-4), t
    finally:
        c.close()
```
Token 2 (`[5, 8, 12, 9, 13, 10]`) routes expert 5 only: lanes 3 and 7 get −1 there.

- [ ] **Step 2: Run and see it fail**

`-k "one_forward_of_its_tokens"` under the GPU lock, as in Task 3. Expected: a `ValueError` from
`ExpertStreamHost.enable_cpu_experts` (`out_rows must be ... tensor`).

- [ ] **Step 3: `cpu_experts.h`**

Add `#include "../cpu_token_table.h"` and `#include <array>`. Replace `struct CpuJob`:

```cpp
/// One forward over up to Wire::kLanes lanes of one row.
///
/// A record produces at most one job for its CPU hits (part 0) and one per batch of CPU misses that landed together
/// (part 1). Every miss job after the first adds into part 1 in landing order, so that part's fp32 sum order varies
/// from run to run. A record's job is per_token: on a multi-token row (CpuExpertConfig::tokens > 1) the row's token
/// table gives each token its own slots and weights, `lanes` naming each job lane's column there.
struct CpuJob {
  int64_t row = 0;
  int32_t part = 0;
  bool accumulate = false;  // add into the part rather than overwrite it
  bool per_token = false;   // a record's job: a multi-token row's table gives each token its weights
  uint32_t seq = 0;         // from claim()
  int32_t k = 0;
  int32_t slots[wire::Wire::kLanes] = {};
  float weights[wire::Wire::kLanes] = {};
  int32_t lanes[wire::Wire::kLanes] = {};  // each job lane's record lane
};
```
In `CpuExpertConfig`, after `hidden`:
```cpp
  int64_t tokens = 1;         // tokens a row holds; above 1 a token table follows the inputs (cpu_token_table.h)
  int64_t x_token_bytes = 0;  // bytes between two tokens' staged inputs
```
In `validate()`, after the hidden check:
```cpp
    if (c.tokens < 1 || c.tokens > CpuTokenTable::kMaxTokens || (c.tokens > 1 && c.x_token_bytes < 2 * c.hidden))
      throw std::runtime_error(prefix_ + "the CPU expert rows hold 1-32 tokens of the hidden size");
```
Replace `run_job`'s head, from `cpu_experts::ForwardCall call;` through `call.cores = config_.cores;`:
```cpp
    cpu_experts::ForwardCall call;
    call.rows = 1;
    call.k = job.k;
    call.threads = config_.threads;
    call.x = config_.x_base + job.row * config_.x_stride;
    call.slots = job.slots;
    call.weights = job.weights;
    call.out =
        reinterpret_cast<float*>(config_.out_base + job.row * config_.out_stride + job.part * config_.out_part_stride);
    call.accumulate = job.accumulate;
    call.cores = config_.cores;
    if (job.per_token && config_.tokens > 1) {
      call.rows = static_cast<int32_t>(expand_tokens(job, call.x));
      call.slots = token_slots_.data();
      call.weights = token_weights_.data();
      // A token that routes none of the lanes must read 0, whatever an earlier record left: zero the rows here rather
      // than trust the kernel's overwrite, then accumulate.
      if (!job.accumulate) std::memset(call.out, 0, static_cast<size_t>(call.rows) * config_.hidden * sizeof(float));
      call.accumulate = true;
    }
```
Add these private members after `run_job`:
```cpp
  /// A record's job on a multi-token row: each token's slots and weights from the row's token table
  /// (cpu_token_table.h), -1 where the token does not route the lane's expert (the kernel skips only slot -1, never a
  /// zero weight). Returns the record's tokens; fails stop on a count the row cannot hold.
  uint32_t expand_tokens(const CpuJob& job, const uint8_t* x) {
    constexpr int64_t kLanes = wire::Wire::kLanes;
    const uint8_t* table = x + config_.tokens * config_.x_token_bytes;
    const uint32_t tokens = load_u32(table);
    if (tokens < 1 || tokens > static_cast<uint32_t>(config_.tokens)) {
      fail_stop(prefix_ + "row " + std::to_string(job.row) + "'s token table holds " + std::to_string(tokens) +
                " tokens");
      return 1;
    }
    for (uint32_t t = 0; t < tokens; ++t)
      for (int32_t i = 0; i < job.k; ++i) {
        const int32_t lane = job.lanes[i];
        const bool routed = (load_u32(table + CpuTokenTable::kHeaderBytes + 4 * lane) >> t & 1u) != 0;
        const uint32_t bits = load_u32(table + CpuTokenTable::kHeaderBytes + 4 * kLanes + 4 * (t * kLanes + lane));
        token_slots_[t * job.k + i] = routed ? job.slots[i] : -1;
        token_weights_[t * job.k + i] = routed ? std::bit_cast<float>(bits) : 0.0f;
      }
    return tokens;
  }

  static uint32_t load_u32(const uint8_t* p) {
    uint32_t v;
    std::memcpy(&v, p, sizeof v);
    return v;
  }
```
And data members next to `compute_ns_`:
```cpp
  // A per-token job's expanded slots and weights, [tokens][k]; this thread only.
  std::array<int32_t, CpuTokenTable::kMaxTokens * wire::Wire::kLanes> token_slots_{};
  std::array<float, CpuTokenTable::kMaxTokens * wire::Wire::kLanes> token_weights_{};
```

- [ ] **Step 4: `ram_tier.h`**

In `submit_host_lanes`, after `cpu_job.seq = first;` add `cpu_job.per_token = true;`. Inside its lane loop, before
`++cpu_job.k;`, add `cpu_job.lanes[cpu_job.k] = lane.lane;`.
In `submit_landed_cpu_misses`, after `cpu_job.accumulate = ...;` add `cpu_job.per_token = true;`. Before
`++cpu_job.k;`, add `cpu_job.lanes[cpu_job.k] = static_cast<int32_t>(plan.miss_lane[i]);`.
`split_calibration.h`'s jobs stay `per_token = false`: they are single-token by construction.

- [ ] **Step 5: `ffi_exports.h` `enable_cpu_experts`**

Add `int64_t tokens,` after `int64_t parts,`. Replace the block from `if (out_rows.size(1) < ...` through
`config.out_part_stride = ...;` with:
```cpp
    if (tokens < 1 || tokens > expert_stream::CpuTokenTable::kMaxTokens)
      throw std::runtime_error(error_prefix<Layout>() + "CPU expert rows hold 1-32 tokens");
    if (tokens > 1 && parts != 2)
      throw std::runtime_error(error_prefix<Layout>() + "a multi-token CPU expert row has two parts per group");
    config.tokens = tokens;
    config.x_token_bytes = (2 * hidden + 15) / 16 * 16;
    const int64_t table = tokens > 1 ? expert_stream::CpuTokenTable::kHeaderBytes +
                                           4 * expert_stream::Wire::kLanes * (1 + tokens)
                                     : 0;
    if (x_rows.size(1) < tokens * config.x_token_bytes + table)
      throw std::runtime_error(error_prefix<Layout>() + "x_rows is narrower than its tokens and token table");
    if (out_rows.size(1) < expert_stream::Wire::kNodes * parts * tokens * hidden)
      throw std::runtime_error(error_prefix<Layout>() + "out_rows is narrower than every group's parts");
    config.out_base = static_cast<uint8_t*>(out_rows.data_ptr()) + group * parts * tokens * hidden * sizeof(float);
    config.out_stride = out_rows.size(1) * static_cast<int64_t>(sizeof(float));
    config.out_part_stride = parts == 2 ? tokens * hidden * static_cast<int64_t>(sizeof(float)) : 0;
```
Include `cpu_token_table.h` there if it is not reached through `cpu_experts.h`.

- [ ] **Step 6: `ExpertStreamHost.enable_cpu_experts`** (`expert_stream_transport.py`)

Replace the `out_rows` validation and the `parts`/`hidden` lines:
```python
        if (
            out_rows.dtype != torch.float32
            or out_rows.device.type != "cpu"
            or not out_rows.is_contiguous()
            or not (
                (out_rows.dim() == 2 and self.nodes == 1)
                or (out_rows.dim() in (3, 4) and out_rows.shape[1] == 2 * self.nodes)
            )
        ):
            raise ValueError(
                "out_rows must be a contiguous host float32 [rows, hidden] at one node, [rows, 2 * nodes, hidden], or "
                "[rows, 2 * nodes, tokens, hidden] for multi-token rows"
            )
        parts = 1 if out_rows.dim() == 2 else 2
        tokens = int(out_rows.shape[2]) if out_rows.dim() == 4 else 1
        hidden = int(out_rows.shape[-1])
```
Pass `int(tokens),` after `parts,` in the module call. Append to the docstring: ``With ``[rows, 2 * nodes, tokens,
hidden]`` rows a record's CPU job runs one forward of its tokens from the row's token table.``

- [ ] **Step 7: Run and pass**

GPU lock, cores 32-63:
`test/manual/dsv41/test_exl3_cpu_lane_order_cuda.py test/manual/dsv41/test_exl3_copy_engine_cuda.py test/manual/dsv41/test_cpu_split_calibration_cuda.py`.
CPU, cores 0-63:
`test/registered/unit/kernels/test_exl3_ram_miss_cpu_experts.py test/registered/unit/kernels/test_cpu_expert_service.py test/registered/unit/kernels/test_cpu_expert_keep_warm.py test/registered/unit/kernels/test_exl3_cpu_split_calibration.py`.
Expected: `EXIT=0` for both. If a registered test calls `expert_stream_enable_cpu_experts` with the old arity, it
fails with a TypeError; add `1` for `tokens` after `parts` there, and list each such file in the commit.

- [ ] **Step 8: Commit**

```bash
git add python/sglang/kernels/jit/csrc/moe/expert_stream/host/cpu_experts.h \
  python/sglang/kernels/jit/csrc/moe/expert_stream/host/ram_tier.h \
  python/sglang/kernels/jit/csrc/moe/expert_stream/host/ffi_exports.h \
  python/sglang/kernels/ops/moe/expert_stream_transport.py test/manual/dsv41/test_exl3_cpu_lane_order_cuda.py
git commit -m "feat(exl3-cpu-experts): a verify's CPU job runs one forward of its tokens from the row's token table

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Y3EXKJdMpP7nwX3qcBi8Vq"
```

---

### Task 6: The route tables seed every token from its own partial

**Files:**
- Modify: `python/sglang/kernels/jit/csrc/moe/exl3/exl3_route_tables.cuh`
- Modify: `python/sglang/srt/layers/quantization/exl3/fused_moe.py` (`Exl3FusedMoE.run`)
- Test: `test/manual/dsv41/test_exl3_moe_split_parity_cuda.py`

**Interfaces:**
- Consumes: Task 5's part layout. Part p's token t is at `cpu_out + p * part_stride + t * hidden`, with
  `part_stride = tokens * hidden`.
- Produces: `Exl3FusedMoE.run(x [M, H], ..., cpu=(cpu_lanes, dst_slots, cpu_out, part_stride))` for any
  `1 <= M <= tokens`.

- [ ] **Step 1: Write the failing test** (append to `test_exl3_moe_split_parity_cuda.py`)

```python
VERIFY_TOKENS = 4


def _fused_tokens(slot_rows, device, tokens):
    from sglang.srt.environ import envs
    from sglang.srt.layers.quantization.exl3.fused_moe import Exl3FusedMoE

    with envs.SGLANG_DSV41_ENABLE_LAYER_FUSION.override(True):
        return Exl3FusedMoE(
            slot_rows, slot_rows["w13_trellis"].shape[0], hidden=slot_rows["w13_suh"].shape[-1],
            inter=slot_rows["w2_suh"].shape[-1], top_k=TOP_K, device=device, tokens=tokens,
        )


def test_a_verify_cpu_partial_seeds_every_token(slot_rows):
    """M = 4 tokens of a DSpark verify. The CPU lanes are distinct slots of the routes' union (lane i's dst_slots
    entry is its slot), so every route on one leaves the fused MoE in every token, and token t's [hidden] partial seeds
    token t. With a zero partial the run is, bit for bit, the run with those routes' weights zeroed; with the GPU's own
    per-token partial of those routes it is the full run up to fp32 reassociation. Mutation: seed only i < hidden (the
    one-token kernel) -- tokens 1-3 lose their partial."""
    device = slot_rows["w13_trellis"].device
    fused = _fused_tokens(slot_rows, device, VERIFY_TOKENS)
    hidden = slot_rows["w13_suh"].shape[-1]
    partial = torch.zeros((VERIFY_TOKENS, hidden), dtype=torch.float32).pin_memory()
    zero = torch.zeros_like(partial).pin_memory()
    keep = torch.ones(1, device=device)
    gen = torch.Generator().manual_seed(932)
    for trial in range(TRIALS * 4):
        remap = torch.cat([torch.randperm(fused.slots, generator=gen)[:TOP_K] for _ in range(VERIFY_TOKENS)]).to(device)
        weights = torch.softmax(torch.randn(VERIFY_TOKENS, TOP_K, generator=gen), 1).reshape(-1).to(device)
        x = (torch.randn((VERIFY_TOKENS, hidden), generator=gen) * 0.5).to(device)
        union = sorted(set(remap.tolist()))
        picked = [s for s in union if int(torch.randint(0, 2, (1,), generator=gen))] or union[:1]
        lanes = torch.tensor(picked, dtype=torch.int32, device=device)
        on_cpu = torch.isin(remap, lanes.long())
        words = torch.tensor([(1 << len(picked)) - 1, PART_HITS], dtype=torch.int32, device=device)
        want = fused.run(x, weights, remap, keep, ACT_LIMIT).clone()
        without = fused.run(x, torch.where(on_cpu, torch.zeros_like(weights), weights), remap, keep, ACT_LIMIT).clone()
        partial.copy_(fused.run(x, torch.where(on_cpu, weights, torch.zeros_like(weights)), remap, keep, ACT_LIMIT))
        got_zero = fused.run(x, weights, remap, keep, ACT_LIMIT, cpu=(words, lanes, zero.data_ptr(), 0)).clone()
        assert torch.equal(_bits(got_zero), _bits(without)), f"picked={picked}"
        got = fused.run(x, weights, remap, keep, ACT_LIMIT, cpu=(words, lanes, partial.data_ptr(), 0)).clone()
        scale = float(want.abs().max())
        assert torch.allclose(got, want, rtol=1e-5, atol=1e-5 * scale), f"picked={picked}"
```

- [ ] **Step 2: Run and see it fail**

On divix01, GPU lock, `SGLANG_EXL3_SRC=/data/models/slang/nvfp4-work/exllamav3`:
`-k a_verify_cpu_partial_seeds_every_token`. Expected: `RuntimeError: CPU experts run one token, not 4`.

- [ ] **Step 3: Implement**

`fused_moe.py` `run`: delete
```python
        if cpu is not None and m != 1:
            raise RuntimeError(f"CPU experts run one token, not {m}")
```
and replace the docstring's "CPU experts only, one token:" with "CPU experts only, any M (each token's partial at
``t * hidden`` within each part):".

`exl3_route_tables.cuh`:
- Delete `RuntimeCheck(!cpu_on || M_.unwrap() == 1, "CPU experts run one token (x has one row)");`.
- In the kernel's comment, replace "(plan 2026-09-29-dsv41-cpu-experts; cpu_lanes non-null; one token only)" with
  "(plan 2026-09-29-dsv41-cpu-experts; cpu_lanes non-null; any M tokens, plan 2026-10-06-dsv41-dspark-both-cpu-experts)".
- Replace "part p at cpu_out + p * part_stride (host memory, [hidden] fp32)" with "part p at cpu_out + p *
  part_stride (host memory, [tokens][hidden] fp32)".
- Replace the loop comment `// CPU experts run one token, so a seeded part is read only where i < hidden.` with
  `// Each token's partial is its own [hidden] row of each part, so element i of the output reads element i of each.`
  The loop itself already reads `cpu_out + part * part_stride + i` for `i < tokens * hidden`, which is exactly the
  `[tokens][hidden]` part layout.

- [ ] **Step 4: Run and pass**

The whole file `test/manual/dsv41/test_exl3_moe_split_parity_cuda.py` and
`test/manual/dsv41/test_exl3_fused_moe_multitoken_gpu.py`, GPU lock. Expected: `EXIT=0`.

- [ ] **Step 5: Commit**

```bash
git add python/sglang/kernels/jit/csrc/moe/exl3/exl3_route_tables.cuh \
  python/sglang/srt/layers/quantization/exl3/fused_moe.py test/manual/dsv41/test_exl3_moe_split_parity_cuda.py
git commit -m "feat(exl3-cpu-experts): the fused MoE's route tables seed each verify token from its own CPU partial

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Y3EXKJdMpP7nwX3qcBi8Vq"
```

---

### Task 7: DIRECT gives victims to the first V lanes only

**Files:**
- Modify: `python/sglang/srt/environ.py` (next to `SGLANG_MOE_EXPERT_GRAPH_GATHER_MISS_LANES`, line ~524)
- Modify: `python/sglang/srt/layers/moe/expert_residency_gpu.py` (`_init_insert_direct`, `_rank_victims`, `gather_destinations`, `fused_gather_destinations`, `clamp_gather_misses`)
- Modify: `python/sglang/kernels/jit/csrc/moe/expert_residency/direct_gather.cuh` (destinations kernel and launcher)
- Modify: `python/sglang/kernels/ops/moe/expert_residency_direct_gather.py` (`direct_gather_destinations`)
- Modify: `python/sglang/srt/layers/moe/expert_hot_cache.py` (allocator floors)
- Modify: `python/sglang/srt/layers/moe/exl3_expert_format.py` (`plan_graph_gather`)
- Modify: `python/sglang/srt/layers/moe/exl3_ram_miss.py` (`plan_staging_width`, `staging_width`)
- Test: `test/registered/unit/layers/moe/test_expert_residency_gpu.py`, `test/registered/unit/kernels/test_exl3_ram_miss_attach_lanes.py`

**Interfaces:**
- Produces:
  - `envs.SGLANG_MOE_EXPERT_GRAPH_GATHER_VICTIM_LANES` (`EnvInt(0)`);
  - `GpuResidencyUpdater.victim_lanes: int`, equal to `miss_rows` when the env is unset;
  - a lane that is not live gets destination `max_capacity` and destination slot −1 when
    `victim_lanes < miss_rows` (the post's spill marker), and (0, 0) otherwise;
  - `direct_gather_destinations(..., idle_destination=0, idle_slot=0)`;
  - `Exl3RamMissService.plan_staging_width(rows)`.

- [ ] **Step 1: Write the failing tests**

Append to `TestInsertOnMiss` in `test_expert_residency_gpu.py` (the class holding `_narrow`):

```python
    # ----- spill: victims for the first V lanes only (SGLANG_MOE_EXPERT_GRAPH_GATHER_VICTIM_LANES) -----

    def _spill(self, fused):
        from sglang.srt.environ import envs

        with envs.SGLANG_DSV41_CPU_EXPERTS.override(True), envs.SGLANG_MOE_EXPERT_GRAPH_GATHER_VICTIM_LANES.override(1):
            return self._narrow(_model(), 2, fused)

    def test_victim_lanes_need_cpu_experts_and_a_value_below_the_lanes(self):
        from sglang.srt.environ import envs

        with envs.SGLANG_MOE_EXPERT_GRAPH_GATHER_VICTIM_LANES.override(1), self.assertRaisesRegex(
            ValueError, "VICTIM_LANES=1 needs SGLANG_DSV41_CPU_EXPERTS=1"
        ):
            self._narrow(_model(), 2, False)
        with envs.SGLANG_DSV41_CPU_EXPERTS.override(True), envs.SGLANG_MOE_EXPERT_GRAPH_GATHER_VICTIM_LANES.override(
            2
        ), self.assertRaisesRegex(ValueError, "below the 2 miss lanes"):
            self._narrow(_model(), 2, False)

    def test_victim_lanes_cap_the_shortlist_and_the_floor(self):
        """Only shortlist column 0 is ever a victim once a boundary ranks; the floor is twice the victim lanes."""
        for fused in (False, True):
            self.model = _model()
            manager = self._spill(fused)
            updater = manager.gpu_residency
            self.assertEqual((updater.miss_rows, updater.victim_lanes), (2, 1))
            graph, static, outputs = self.capture(manager, tokens=2)
            self.replay_verify(manager, graph, static, outputs, [[[0, 1], [0, 1]]] * LAYERS, check_outputs=False)
            self.replay_verify(manager, graph, static, outputs, [[[0, 1], [0, 1]]] * LAYERS, check_outputs=False)
            self.assertFalse(bool(updater.victim_valid[:, 1:].any()))
            self.assertTrue(bool(updater.victim_valid[:, 0].all()))

    def test_victimless_lanes_take_the_idle_destination_in_both_paths(self):
        """Two misses, one victim: lane 0 is live at the victim, lane 1 is not and gets (slot_dump, -1), the marker the
        post spills to the CPU; its routes remap past every slot column. The torch chain and the fused kernel agree."""
        from sglang.kernels.ops.moe.expert_residency_direct_gather import direct_gather_destinations

        manager = self._spill(False)
        updater, streamer = manager.gpu_residency, manager.streamers[0]
        dump = updater.max_capacity
        base = streamer.row_planner.scratch_base
        updater.victims[0].copy_(torch.tensor([3, 5], device="cuda"))
        updater.victim_valid[0].copy_(torch.tensor([True, False], device="cuda"))
        streamer._graph_miss_count.fill_(2)
        remap = torch.tensor([base, base + 1, base + 1, base], dtype=torch.int64, device="cuda")
        route_slots = torch.full((4,), -1, dtype=torch.int64, device="cuda")
        got = updater.gather_destinations(0, remap, route_slots, base)
        _, _, destinations, live = updater._pending_commit
        self.assertEqual(destinations.tolist(), [3, dump])
        self.assertEqual(live.tolist(), [True, False])
        self.assertEqual(streamer._graph_destination_slots[:2].tolist(), [3, -1])
        self.assertEqual(got.tolist(), [3, dump, dump, 3])
        ids = torch.tensor([7, 9, 9, 7], dtype=torch.int64, device="cuda")
        expert_to_slot = torch.full((EXPERTS,), -1, dtype=torch.int64, device="cuda")
        slots_out = torch.zeros(2, dtype=torch.int32, device="cuda")
        dest_out = torch.zeros(2, dtype=torch.int64, device="cuda")
        live_out = torch.zeros(2, dtype=torch.bool, device="cuda")
        remap_out = torch.zeros(4, dtype=torch.int64, device="cuda")
        direct_gather_destinations(
            ids, expert_to_slot, updater.victims[0], updater.victim_valid[0], streamer._graph_miss_count, remap, base,
            slots_out, dest_out, live_out, remap_out, idle_destination=dump, idle_slot=-1,
        )
        self.assertEqual((slots_out.tolist(), dest_out.tolist(), live_out.tolist()), ([3, -1], [3, dump], [True, False]))
        self.assertEqual(remap_out.tolist(), got.tolist())

    def test_spill_clamps_only_a_count_past_the_lanes(self):
        """Review Focus 5. With spill the post decides the victimless lanes, so the clamp leaves a count within the
        lanes alone (no flag) and flags only one past them, serving the live prefix as without CPU experts."""
        manager = self._spill(False)
        updater, streamer = manager.gpu_residency, manager.streamers[0]
        base = streamer.row_planner.scratch_base
        updater.victims[0].copy_(torch.tensor([3, 5], device="cuda"))
        updater.victim_valid[0].copy_(torch.tensor([True, False], device="cuda"))
        remap = torch.tensor([base, base + 1, base + 1, base], dtype=torch.int64, device="cuda")
        for count, flagged, kept in ((2, 0, 2), (3, 1, 1)):
            updater.overflow_flag.zero_()
            streamer._graph_miss_count.fill_(count)
            updater.gather_destinations(0, remap, torch.full((4,), -1, dtype=torch.int64, device="cuda"), base)
            updater.clamp_gather_misses()
            self.assertEqual(int(updater.overflow_flag.item()), flagged, count)
            self.assertEqual(int(streamer._graph_miss_count.item()), kept, count)
```
`EXPERTS` (12) is the module constant. Victims 3 and 5 are slot ids under the 8-slot layers `_narrow` builds.

Append to `test_exl3_ram_miss_attach_lanes.py`:
```python
def test_victim_lanes_stage_their_width_not_the_lanes(tiers):
    """Spill: the post types up to 32 lanes, but only the V victim lanes and the CPU's forced misses stage, and a forced
    miss with no slot overflows. So a row reserves V staging slots (Exl3ExpertFormat.plan_graph_gather plans it)."""
    service, streamers = tiers
    with envs.SGLANG_MOE_EXPERT_GRAPH_GATHER_VICTIM_LANES.override(2):
        streamers[0].format.plan_graph_gather(streamers[0], 32)
    assert service.resolved_lanes() == 32
    assert service.staging_width() == 2
    assert service.staging_for(CAPACITY) == 2
```

- [ ] **Step 2: Run and see them fail**

On divix01:
- GPU lock, `test/registered/unit/layers/moe/test_expert_residency_gpu.py -k "victim or spill"`;
- CPU, `test/registered/unit/kernels/test_exl3_ram_miss_attach_lanes.py -k victim_lanes`.
Expected: `AttributeError: ... SGLANG_MOE_EXPERT_GRAPH_GATHER_VICTIM_LANES`.

- [ ] **Step 3: The env** (`environ.py`, right after `SGLANG_MOE_EXPERT_GRAPH_GATHER_MISS_LANES = EnvInt(0)`)

```python
    # Spill (SGLANG_DSV41_CPU_EXPERTS with a narrowed verify gather): only the first V of a gather's miss lanes in plan
    # order take a VRAM victim and a staging slot; the post makes every later lane a CPU lane, or flags the forward
    # when one cannot be. 0: every miss lane may take a victim. 1 <= V < MISS_LANES.
    SGLANG_MOE_EXPERT_GRAPH_GATHER_VICTIM_LANES = EnvInt(0)
```

- [ ] **Step 4: The updater** (`expert_residency_gpu.py`)

In `_init_insert_direct`, right after `width = self.miss_rows`:
```python
        victims = envs.SGLANG_MOE_EXPERT_GRAPH_GATHER_VICTIM_LANES.get()
        if victims and not envs.SGLANG_DSV41_CPU_EXPERTS.get():
            raise ValueError(
                f"SGLANG_MOE_EXPERT_GRAPH_GATHER_VICTIM_LANES={victims} needs SGLANG_DSV41_CPU_EXPERTS=1: only the "
                "CPU serves the lanes that take no victim"
            )
        if victims and not 1 <= victims < width:
            raise ValueError(
                f"SGLANG_MOE_EXPERT_GRAPH_GATHER_VICTIM_LANES={victims} must be below the {width} miss lanes"
            )
        # Lanes that may take a victim; the rest are the CPU's (the post's spill).
        self.victim_lanes = victims or width
```
Change the floor check to `if cache.capacity < 2 * self.victim_lanes:`. In its message, "twice its graph-gather
miss lanes; layer {layer_id} has {cache.capacity} slots for {width} lanes" becomes "twice its graph-gather victim lanes;
layer {layer_id} has {cache.capacity} slots for {self.victim_lanes} lanes".
Append to the docstring: "With spill (``victim_lanes`` below ``miss_rows``) only the first ``victim_lanes`` shortlist
columns are victims, so the floor is ``2 * victim_lanes``."

In `_rank_victims`, after `self.victim_valid.copy_(ranked.values[:, :width] < never)`:
```python
        if self.victim_lanes < width:
            # Spill: only the first victim_lanes columns are victims; the post sends the other lanes to the CPU.
            self.victim_valid[:, self.victim_lanes :] = False
```
Add a method after `_rank_victims`:
```python
    def _idle_destination(self) -> tuple[int, int]:
        """(destination, destination slot) of a lane that is not live. (0, 0) normally, a slot the clamp keeps any copy
        out of. With spill (slot_dump, -1): -1 tells the post the lane found no victim, and a route remapped to
        slot_dump is past every slot column, so the fused MoE skips it and the commit writes only the dump column."""
        return (self.max_capacity, -1) if self.victim_lanes < self.miss_rows else (0, 0)
```
In `gather_destinations`, replace
```python
        destinations = torch.where(live, usable, torch.zeros_like(usable))
        # The plan's slot buffer has a row per route; the lanes are the first miss_rows.
        streamer._graph_destination_slots[: self.miss_rows].copy_(destinations.to(torch.int32))
```
with
```python
        idle, idle_slot = self._idle_destination()
        destinations = torch.where(live, usable, torch.full_like(usable, idle))
        # The plan's slot buffer has a row per route; the lanes are the first miss_rows.
        streamer._graph_destination_slots[: self.miss_rows].copy_(
            torch.where(live, destinations, torch.full_like(destinations, idle_slot)).to(torch.int32)
        )
```
In `fused_gather_destinations`, pass
`idle_destination=self._idle_destination()[0], idle_slot=self._idle_destination()[1]` to `direct_gather_destinations`.

Replace `clamp_gather_misses`'s body after `served = live.sum(dtype=torch.int32)`:
```python
        # Spill: every lane up to the record's width is posted, and the post makes the victimless ones CPU lanes or
        # flags the forward (exl3_ram_miss_post_kernel); only a count past the lanes is the clamp's.
        over = count > (self.miss_rows if self.victim_lanes < self.miss_rows else served)
        self.gather_overflow[row].add_(over.sum())
        self.overflow_flag.bitwise_or_(over.to(torch.int32))
        count.copy_(torch.where(over, served, count))
```
Without spill this is the old `minimum(count, served)`: `over` is `count > served`. Append to the docstring: "With
spill (``victim_lanes`` below ``miss_rows``) it flags only a count past the miss lanes."

- [ ] **Step 5: The kernel** (`direct_gather.cuh`, `expert_residency_direct_gather.py`)

Kernel: add parameters `int64_t idle_destination, int32_t idle_slot` after `remap_out`. Replace
```cpp
    const int64_t destination = live ? usable[lane] : 0;
    destinations[lane] = destination;
    destinations_out[lane] = destination;
    destination_slots_out[lane] = static_cast<int32_t>(destination);
```
with
```cpp
    const int64_t destination = live ? usable[lane] : idle_destination;
    destinations[lane] = destination;
    destinations_out[lane] = destination;
    destination_slots_out[lane] = live ? static_cast<int32_t>(destination) : idle_slot;
```
In the kernel comment, "any other lane's is 0" becomes "any other lane's is idle_destination (0, or slot_dump with
spill), its destination slot idle_slot (0, or -1 with spill)". In the launcher, add the same two parameters after
`remap_out` and pass them last.
Python: `direct_gather_destinations(..., remap_out, idle_destination: int = 0, idle_slot: int = 0)`. Pass
`int(idle_destination), int(idle_slot)` after `remap_out` in `.run(...)`.

- [ ] **Step 6: Allocator floor and staging width**

`expert_hot_cache.py`, before `floors = {`:
```python
        # Spill: only the victim lanes need VRAM slots (GpuResidencyUpdater._init_insert_direct's floor).
        victim_lanes = envs.SGLANG_MOE_EXPERT_GRAPH_GATHER_VICTIM_LANES.get()
```
and change the floor expression to `2 * (min(miss_lanes[layer_id], victim_lanes) if victim_lanes else miss_lanes[layer_id]) if direct else 0`.
Change "(twice its graph-gather miss lanes)" in the refusal text to "(twice its graph-gather victim lanes)".

`exl3_ram_miss.py`, in `Exl3RamMissService` after `plan_gather_width`:
```python
    def plan_staging_width(self, rows: int) -> None:
        """Plan the staging slots a row reserves when fewer lanes than the record's take a victim (spill): the victim
        lanes. Only valid before the service starts; the widest plan wins."""
        if self.host is not None:
            raise RuntimeError("exl3 RAM miss: the staging width was planned after the service started")
        planned = max(1, int(rows))
        self._staging_planned = planned if self._staging_planned is None else max(self._staging_planned, planned)
```
Initialize `self._staging_planned = None` in `__init__`, next to `self._gather_planned`. `staging_width` returns
`self._staging_planned or self._gather_planned or self.resolved_lanes()`, and its docstring gains "the planned victim
lanes under spill,".

`exl3_expert_format.py` `plan_graph_gather`, inside the `if isinstance(...)` branch after `plan_gather_width(rows)`:
```python
            victims = envs.SGLANG_MOE_EXPERT_GRAPH_GATHER_VICTIM_LANES.get()
            if victims and victims < rows:
                Exl3RamMissService.get().plan_staging_width(victims)
```
Import `envs` there if the module does not already.

- [ ] **Step 7: Run and pass**

GPU lock: the whole `test/registered/unit/layers/moe/test_expert_residency_gpu.py`,
`test/manual/dsv41/test_exl3_verify_miss_lanes_gpu.py` and `test/manual/dsv41/test_exl3_cpu_lane_order_cuda.py`.
CPU: `test/registered/unit/kernels/test_exl3_ram_miss_attach_lanes.py`. Expected: `EXIT=0`.

- [ ] **Step 8: Commit**

```bash
git add python/sglang/srt/environ.py python/sglang/srt/layers/moe/expert_residency_gpu.py \
  python/sglang/kernels/jit/csrc/moe/expert_residency/direct_gather.cuh \
  python/sglang/kernels/ops/moe/expert_residency_direct_gather.py python/sglang/srt/layers/moe/expert_hot_cache.py \
  python/sglang/srt/layers/moe/exl3_expert_format.py python/sglang/srt/layers/moe/exl3_ram_miss.py \
  test/registered/unit/layers/moe/test_expert_residency_gpu.py test/registered/unit/kernels/test_exl3_ram_miss_attach_lanes.py
git commit -m "feat(moe-residency): SGLANG_MOE_EXPERT_GRAPH_GATHER_VICTIM_LANES gives victims to the first V lanes; the rest are marked for the CPU

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Y3EXKJdMpP7nwX3qcBi8Vq"
```

---

### Task 8: The service sizes CPU rows for the verify and wires the spill words

**Files:**
- Modify: `python/sglang/srt/layers/moe/cpu_experts/service.py` (`CpuExpertService.__init__`, `CpuExpertGroups.__init__`)
- Modify: `python/sglang/srt/layers/moe/exl3_ram_miss.py` (`_start_cpu_experts`, `attach`, `Exl3RamMissRowBackend.__init__`/`.post`)
- Modify: `python/sglang/test/dsv41_ram_miss_fixtures.py` (`DirectUpdaterStandIn`)
- Modify: `analysis/dsv41-drive/LEASE_PROTOCOL.md` (CPU expert section)
- Test: `test/registered/unit/kernels/test_exl3_ram_miss_attach_lanes.py`

**Interfaces:**
- Consumes: Tasks 3-7.
- Produces:
  - `CpuExpertService(..., tokens: int = 1)` and `CpuExpertGroups(..., tokens: int = 1)`;
  - `Exl3RamMissRowBackend.spill`, `None` or `(overflow_flag, gather_overflow[row : row + 1])`;
  - `attach` accepts CPU experts at a narrowed (verify) gather.

- [ ] **Step 1: Write the failing tests** (in `test_exl3_ram_miss_attach_lanes.py`)

Delete `test_cpu_experts_refuse_a_miss_width_below_the_routes`. Add:
```python
def test_cpu_experts_attach_a_verify_gather_and_wire_its_spill_words(tiers, monkeypatch):
    """Six tokens of top-6 at 32 miss lanes, 8 of them victim lanes: the layer attaches with CPU experts on, and its
    backend posts with DIRECT's overflow flag and its own row of the overflow counter (a view: the post's increment is
    the updater's)."""
    service, streamers = tiers
    service.plan_gather_width(32)
    service.ensure_started()
    built = []
    monkeypatch.setattr(
        module, "Exl3RamMissRowBackend", lambda *args, **kwargs: built.append(kwargs) or SimpleNamespace()
    )
    service.cpu_experts = SimpleNamespace(attach_device=lambda device_side: None)
    updater = DirectUpdaterStandIn(LAYERS, CAPACITY, EXPERTS)
    updater.miss_rows, updater.victim_lanes = 32, 8
    updater.overflow_flag = torch.zeros(1, dtype=torch.int32)
    updater.gather_overflow = torch.zeros(LAYERS, dtype=torch.int64)
    manager = SimpleNamespace(register_fail_stop_check=lambda check: None, gpu_residency=updater)
    try:
        streamer = streamers[0]
        streamer._graph_pinned_tier = True
        streamer.hot_cache = SimpleNamespace(device="cpu", capacity=CAPACITY)
        streamer.graph_gather_rows, streamer.graph_miss_lanes = 36, 32
        streamer.residency_row = 1
        streamer.row_backend = SimpleNamespace(segments={0: None}, host_row_map=torch.full((EXPERTS,), -1, dtype=torch.int64))
        service.attach(manager, streamer)
    finally:
        service.cpu_experts = None
    flag, counter = streamer.row_backend.spill
    assert flag is updater.overflow_flag
    counter.add_(1)
    assert updater.gather_overflow.tolist() == [0, 1]


def test_a_captured_cpu_expert_gather_posts_its_spill_words(monkeypatch):
    streamer = SimpleNamespace(_plan_miss_keys=torch.zeros(EXPERTS, dtype=torch.int64))
    backend, side = _captured_backend(monkeypatch, lambda: streamer)
    assert backend.spill is None
    backend.spill = (torch.zeros(1, dtype=torch.int32), torch.zeros(1, dtype=torch.int64))
    backend.post(0, _CapturedPlan())
    assert side.calls[0][1]["spill"] is backend.spill


def test_cpu_rows_hold_the_verify_tokens():
    """A 6-token verify's service rows: x rows of cpu_row_bytes(hidden, 6, lanes), out rows [rows, 2 * nodes, 6,
    hidden]; one token keeps today's shapes."""
    from sglang.srt.layers.moe.cpu_experts.service import CpuExpertService

    for tokens, out_shape in ((1, (2, 2, 64)), (6, (2, 2, 6, 64))):
        enabled = {}
        host = SimpleNamespace(
            wire=lease.wire_layout(32), nodes=1, enable_cpu_experts=lambda *a, **k: enabled.update(args=a),
            cpu_stats=lambda group: {},
        )
        trait = SimpleNamespace(check_environment=lambda: None, kernel_address=lambda: 1, name="t")
        service = CpuExpertService(
            host, trait, {0: {}, 1: {}}, hidden=64, cores=[0, 1], threads=2, split=[0] * 33, pin=False, tokens=tokens,
        )
        assert tuple(service.out_rows.shape) == out_shape
        assert service.x_rows.shape[1] == lease.cpu_row_bytes(64, tokens, 32)
```
`CpuExpertService.__init__` calls `check_engine_cores(cores, threads)`. If cores 0-1 are refused on the test host
(reserved or absent), name two cores from `os.sched_getaffinity(0)`.

- [ ] **Step 2: Run and see them fail**

CPU: `test/registered/unit/kernels/test_exl3_ram_miss_attach_lanes.py -k "spill or verify_tokens or verify_gather"`.
Expected: failures on `spill` (no attribute) and `tokens` (unexpected keyword).

- [ ] **Step 3: Implement**

`service.py`:
- `CpuExpertService.__init__` gains `tokens: int = 1` after `shared`.
- Replace the row allocation in the `else:` branch:
  ```python
              # Row layout is in the class doc; pinned so the device reaches them by UVA. A verify's rows hold its tokens
              # and the token table (cpu_token_table.h).
              self.x_rows = torch.zeros((rows, cpu_row_bytes(self.hidden, tokens, self.lanes)), dtype=torch.uint8)
              shape = (2 * host.nodes, self.hidden) if tokens == 1 else (2 * host.nodes, tokens, self.hidden)
              self.out_rows = torch.zeros((rows, *shape), dtype=torch.float32)
  ```
  Import `cpu_row_bytes` from `sglang.kernels.ops.moe.expert_lease_block`.
- Class doc: after "Each output row has two parts, ..." add "A verify's rows (``tokens`` > 1) hold each token's input
  and output, and the post's token table after the inputs."
- `CpuExpertGroups.__init__` gains `tokens: int = 1` and passes `tokens=tokens` to each `CpuExpertService`.

`exl3_ram_miss.py`:
- `_start_cpu_experts`: before `return CpuExpertGroups(`:
  ```python
          # A DSpark verify gathers tokens x top_k routes a layer; the CPU rows hold that many tokens.
          tokens = max((s.graph_gather_rows // s.layer.top_k for s in streamers.values() if s.graph_gather_rows), default=1)
  ```
  Pass `tokens=tokens`. Drop "(a verify)" refusals from its docstring if any.
- `attach`: delete the block
  ```python
        if self.cpu_experts is not None and width < streamer.graph_gather_rows:
            raise ValueError(
                f"exl3 RAM miss: CPU experts serve one token; ..."
            )
  ```
- The staging-warning line `want = max(1, streamer.graph_miss_width)` becomes:
  ```python
        victims = getattr(manager.gpu_residency, "victim_lanes", None)
        want = max(1, min(streamer.graph_miss_width, victims) if victims else streamer.graph_miss_width)
  ```
  Its comment gains "; under spill, the victim lanes".
- Right after `streamer.row_backend = Exl3RamMissRowBackend(...)`:
  ```python
        updater = manager.gpu_residency
        if self.cpu_experts is not None and getattr(updater, "victim_lanes", width) < width:
            # Spill: the post flags this layer itself when a victimless lane cannot be the CPU's.
            residency_row = streamer.residency_row
            streamer.row_backend.spill = (
                updater.overflow_flag,
                updater.gather_overflow[residency_row : residency_row + 1],
            )
  ```
  `width` here is the `streamer.graph_miss_width` read earlier in `attach`.
- `Exl3RamMissRowBackend.__init__`: after `self.cpu_input = None` add
  `self.spill = None  # DIRECT's (overflow flag, this layer's counter) under spill; set by Exl3RamMissService.attach`.
- `Exl3RamMissRowBackend.post`: add `spill=self.spill,` to `side.post(...)`.

`dsv41_ram_miss_fixtures.py` `DirectUpdaterStandIn`: nothing to add. `attach` reads `victim_lanes` with `getattr`.

`LEASE_PROTOCOL.md`, at the end of the CPU expert thread section, add:
```markdown
**A verify's CPU lanes (plan 2026-10-06-dsv41-dspark-both-cpu-experts).** A row's pinned input holds the verify's
tokens (`cpu_tokens_max`, 6 at `speculative_dspark_block_size=5`), then a token table the post writes for every lane:
the token count, a mask of the tokens that route the lane's expert, and each such token's weight
(`cpu_token_table.h`). A record's CPU job runs one forward of the record's tokens from it, and each token's partial is
its own `[hidden]` row of the part, which the route tables add to that token. The record's summed lane weight still
serves one-token rows.

**Spill.** With `SGLANG_MOE_EXPERT_GRAPH_GATHER_VICTIM_LANES` below the miss lanes, DIRECT marks the lanes past its
victims with destination slot -1. The post makes each one a CPU lane, outside the split. When one cannot be (copy
engine unarmed, no CPU layer, a miss with no staging slot), the post serves the live prefix, writes that count, and
sets DIRECT's overflow flag and the layer's counter, and the DSpark worker re-runs the verify eagerly. The draft (the
second client) is untouched: separate channel, areas, thread and cores, on the same stream strictly before the verify.
```

- [ ] **Step 4: Run and pass**

CPU: `test/registered/unit/kernels/test_exl3_ram_miss_attach_lanes.py test/registered/unit/kernels/test_exl3_ram_miss_cpu_experts.py test/registered/unit/kernels/test_cpu_expert_service.py`.
Expected: `EXIT=0`.

- [ ] **Step 5: Commit**

```bash
git add python/sglang/srt/layers/moe/cpu_experts/service.py python/sglang/srt/layers/moe/exl3_ram_miss.py \
  analysis/dsv41-drive/LEASE_PROTOCOL.md test/registered/unit/kernels/test_exl3_ram_miss_attach_lanes.py
git commit -m "feat(exl3-cpu-experts): CPU rows hold a verify's tokens; a verify gather attaches with CPU experts and posts the spill words

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Y3EXKJdMpP7nwX3qcBi8Vq"
```

---

### Task 9: End to end on the GPU: a captured 6-token verify with CPU experts and spill

The real EXL3 lease chain, the real optimized CPU kernel and a captured verify, the D2-2 rig
(`test_exl3_verify_miss_lanes_gpu.py`) with CPU experts on.

**Files:**
- Create: `test/manual/dsv41/test_exl3_verify_cpu_spill_gpu.py`

**Interfaces:**
- Consumes: everything above. `Exl3MoEMethod._apply_graph`, `_apply_streamed`, `ExpertHotCacheManager.from_model(...,
  graph_gather_miss_lanes=)`, `manager.take_verify_overflow()`, `manager.suspend_graph_gather()`,
  `exl3_ram_miss.COPY_ENGINE_ARM_DECODES`.

- [ ] **Step 1: Write the test**

```python
"""A captured 6-token verify through the real EXL3 lease chain with CPU experts and spill (GPU, real CPU kernel).

Plan 2026-10-06-dsv41-dspark-both-cpu-experts Task 9. Eight miss lanes, two of them victim lanes. A verify whose
union of misses are RAM hits serves two on the GPU and the rest on the CPU, with no overflow, and every token's output
is within the CPU kernel's bar of the fp32 reference. A replay before the copy engine arms overflows instead (Review
Focus 1), and the eager re-run is exact. A union with more NVMe misses than the node's staging overflows (Review Focus
2) and re-runs exactly.

Run on divix01 from the pushed worktree, holding rowimg-disk.lock then cc-gpu.lock, on the recipe's server cores (the
CPU expert team derives node 0's 6-15):
  CUDA_MODULE_LOADING=EAGER SGLANG_DSV41_CPU_EXPERTS=1 EXL3_MOE_CPU_PIN=0 SGLANG_EXL3_SRC=... SGLANG_EXL3_CPU_CXX=...
  taskset -c 0-5,36-41 python -m pytest -q test/manual/dsv41/test_exl3_verify_cpu_spill_gpu.py
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(__file__))
from test_exl3_ram_miss_graph_gpu import HIDDEN, INTER, _reference, _rel, _source_rows  # noqa: E402

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and os.environ.get("SGLANG_EXL3_SRC") and os.environ.get("SGLANG_DSV41_CPU_EXPERTS") == "1"),
    reason="needs a GPU, SGLANG_EXL3_SRC and SGLANG_DSV41_CPU_EXPERTS=1 (the optimized CPU build)",
)

ACT_LIMIT = 10.0
TOP_K, TOKENS, LANES, VICTIMS, EXPERTS = 6, 6, 8, 2, 48
CPU_BOUND = 2e-2  # the CPU kernel's own bar (test_exl3_moe_split_parity_cuda.py)


def test_a_verify_spills_its_victimless_lanes_to_the_cpu(tmp_path):
    from sglang.srt.environ import envs
    from sglang.srt.layers.moe import exl3_ram_miss as service_module
    from sglang.srt.layers.moe.exl3_expert_format import Exl3ExpertFormat
    from sglang.srt.layers.moe.exl3_expert_layout import build_exl3_expert_layout
    from sglang.srt.layers.moe.expert_hot_cache import ExpertHotCacheManager
    from sglang.srt.layers.moe.expert_stream import ExpertPinnedHostCache, ExpertStreamer
    from sglang.srt.layers.moe.ram_slot_map import LaneKind
    from sglang.srt.layers.quantization.exl3 import Exl3MoEMethod
    from sglang.test.dsv41_fake_exl3 import write_fake_exl3
    from sglang.test.dsv41_ram_miss_fixtures import service_row_images

    write_fake_exl3(str(tmp_path), num_layers=1, num_experts=EXPERTS, hidden=HIDDEN, inter=INTER, finite=True)
    source = _source_rows(tmp_path, num_experts=EXPERTS)
    source_cuda = {name: rows.cuda() for name, rows in source.items()}
    layout = build_exl3_expert_layout(str(tmp_path))
    service_module.Exl3RamMissService._instance = None
    service = service_module.Exl3RamMissService.get()
    try:
        with (
            service_row_images(tmp_path),
            envs.SGLANG_MOE_EXPERT_ROW_SOURCE.override("shards"),
            envs.SGLANG_MOE_EXPERT_GRAPH_GATHER.override(True),
            envs.SGLANG_MOE_EXPERT_PREFETCH_PULL_MODE.override("off"),
            envs.SGLANG_MOE_EXPERT_FUSED_PLAN.override(True),
            envs.SGLANG_DSV41_ENABLE_LAYER_FUSION.override(True),
            envs.SGLANG_DSV41_ENABLE_RAM_MISS_COPY_ENGINE.override(True),
            envs.SGLANG_DSV41_ENABLE_CPU_EXPERTS_CALIBRATION.override(False),
            envs.SGLANG_MOE_EXPERT_GRAPH_GATHER_VICTIM_LANES.override(VICTIMS),
        ):
            model, layer = torch.nn.Module(), torch.nn.Module()
            layer.layer_id, layer.top_k = 0, TOP_K
            fmt = Exl3ExpertFormat(layout, 0, source_root=str(tmp_path))
            fmt.max_gather_rows = LANES
            streamer = ExpertStreamer(layer, fmt.names, layer_id=0, format=fmt)
            layer._nvfp4_expert_streamer = streamer
            model.add_module("expert_layer", layer)
            ExpertPinnedHostCache(streamer, 4 * LANES, **fmt.pinned_tier_options(layer))
            manager = ExpertHotCacheManager.from_model(
                model, budget_bytes=2 * LANES * streamer.bytes_per_expert, seed_path=None, dynamic=True,
                update_prefill_tokens=16, min_residence_forwards=0, benefit_ratio=0.0,
                graph_gather_batch_size=TOKENS, graph_gather_miss_lanes=LANES, update_decode_forwards=1,
                gpu_residency_update=True, insert_on_miss=2,
            )
            updater = manager.gpu_residency
            assert (updater.miss_rows, updater.victim_lanes) == (LANES, VICTIMS)
            assert service.staging_width() == VICTIMS

            generator = torch.Generator(device="cpu").manual_seed(41)
            x = (torch.randn((TOKENS, HIDDEN), generator=generator) * 0.5).to("cuda", torch.bfloat16)
            weights = torch.softmax(torch.randn((TOKENS, TOP_K), generator=generator), 1).to("cuda")
            ids = torch.tensor([list(range(TOP_K))] * TOKENS, device="cuda", dtype=torch.int32)
            Exl3MoEMethod._apply_graph(layer, streamer, x, weights, ids, ACT_LIMIT)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out = Exl3MoEMethod._apply_graph(layer, streamer, x, weights, ids, ACT_LIMIT)
            manager.discard_graph_capture_routes()
            assert service.cpu_experts is not None and service.cpu_experts.x_rows.shape[0] == 1

            def outsiders():
                mapping = updater.mapping[0, :EXPERTS].cpu().tolist()
                return [expert for expert, slot in enumerate(mapping) if slot < 0]

            def in_ram(experts):
                """Load experts into the pinned tier through the eager path (the service maps them on the device)."""
                with manager.suspend_graph_gather():
                    for start in range(0, len(experts), TOP_K):
                        chunk = experts[start : start + TOP_K]
                        chunk = chunk + [e for e in range(EXPERTS) if e not in chunk][: TOP_K - len(chunk)]
                        Exl3MoEMethod._apply_streamed(
                            layer, streamer, x[:1], weights[:1], torch.tensor([chunk], device="cuda", dtype=torch.int32),
                            ACT_LIMIT,
                        )
                torch.cuda.synchronize()

            def replay(routes):
                updater.overflow_flag.zero_()
                ids.copy_(torch.tensor(routes, device="cuda", dtype=torch.int32))
                graph.replay()
                torch.cuda.synchronize()
                service.fail_stop_check()
                assert updater.insertion_truncated[0].item() == 0
                return int(updater.overflow_flag.item())

            def rerun_is_exact(routes):
                assert manager.take_verify_overflow()
                with manager.suspend_graph_gather():
                    eager = Exl3MoEMethod._apply_streamed(layer, streamer, x, weights, ids, ACT_LIMIT)
                torch.cuda.synchronize()
                for t, route in enumerate(routes):
                    ref = _reference(x[t : t + 1], weights[t], torch.tensor(route), source_cuda)
                    assert _rel(eager[t : t + 1], ref) <= CPU_BOUND, (t, route)

            # Eight RAM-resident outsiders, routed by every token in turn: a union of 8 misses, all RAM hits.
            ram_set = outsiders()[:LANES]
            in_ram(ram_set)
            routes = [[ram_set[(t + k) % LANES] for k in range(TOP_K)] for t in range(TOKENS)]

            # Review Focus 1: before the copy engine arms, no forced lane can be the CPU's.
            assert replay(routes) == 1
            rerun_is_exact(routes)

            service._copy_decodes = service_module.COPY_ENGINE_ARM_DECODES
            service._arm_copy_engine()
            assert service._copy_armed
            inserted = updater.gather_insertions[0].item()
            assert replay(routes) == 0, "a union of RAM hits spills, it does not overflow"
            count = int(streamer._graph_miss_count.item())
            kinds = service.device_side.lane_kind[:count].tolist()
            cpu = sum(k in (int(LaneKind.HIT_CPU), int(LaneKind.MISS_CPU)) for k in kinds)
            live = updater.gather_insertions[0].item() - inserted
            assert count == LANES and live <= VICTIMS and cpu >= LANES - VICTIMS, (count, live, kinds)
            mapping = updater.mapping[0, :EXPERTS].cpu()
            for t, route in enumerate(routes):
                ref = _reference(x[t : t + 1], weights[t], torch.tensor(route), source_cuda)
                assert _rel(out[t : t + 1], ref) <= CPU_BOUND, (t, route)

            # Review Focus 2: more NVMe misses than the node's VICTIMS staging slots. Forced misses past the staging
            # overflow (never trap); each flagged verify re-runs exactly, and the post counted each flag once.
            before = updater.gather_overflow[0].item()
            flagged = 0
            for extra in range(2, 2 + LANES - VICTIMS):
                cold = [e for e in outsiders() if e not in ram_set][: VICTIMS + extra]
                routes = [[cold[(t + k) % len(cold)] if k < 2 else ram_set[(t + k) % LANES] for k in range(TOP_K)]
                          for t in range(TOKENS)]
                if replay(routes):
                    flagged += 1
                    rerun_is_exact(routes)
                    break
            assert flagged == 1, "no union of cold misses exhausted the staging"
            assert updater.gather_overflow[0].item() - before == flagged
    finally:
        service.shutdown()
```
The last block widens the cold set until a forced miss finds no staging slot. Which plan lanes DIRECT makes live (and
so which misses take staging first) is not known in advance. The bar is fixed: never a trap (`fail_stop_check`), an
exact re-run when flagged (`rerun_is_exact`), and one counter increment per flag. Record the routes that tripped it.

- [ ] **Step 2: Run on divix01**

```bash
cd /data/models/slang/nvfp4-work/wt-both-cpu
exec 8>/data/models/slang/nvfp4-work/rowimg-disk.lock; flock 8
exec 9>/data/models/slang/nvfp4-work/cc-gpu.lock; flock 9
PYTHONPATH=$PWD/python CUDA_MODULE_LOADING=EAGER SGLANG_DSV41_CPU_EXPERTS=1 EXL3_MOE_CPU_PIN=0 \
  SGLANG_EXL3_SRC=/data/models/slang/nvfp4-work/exllamav3 \
  SGLANG_EXL3_BUILD_DIR=/data/models/slang/nvfp4-work/cc-expert-prediction/exl3-build \
  SGLANG_EXL3_CPU_CXX=/opt/rh/gcc-toolset-15/root/usr/bin/g++ CXX=/opt/rh/gcc-toolset-15/root/usr/bin/g++ \
  CUDA_HOME=/usr/local/cuda-13.2 OMP_NUM_THREADS=8 taskset -c 0-5,36-41 \
  /data/models/slang/.venv/bin/python -m pytest -q -s test/manual/dsv41/test_exl3_verify_cpu_spill_gpu.py \
  2>&1 | tail -20; echo "EXIT=${PIPESTATUS[0]}"
```
Expected: `EXIT=0`. If it fails, debug with `superpowers:systematic-debugging` before changing the bar. A
`fail_stop` or `__trap` is a defect in Tasks 3-8. It is never a reason to drop an assertion.

- [ ] **Step 3: Re-run the D2-2 rig and the BS1 graph suites** (same locks, cores 32-63):
`test/manual/dsv41/test_exl3_verify_miss_lanes_gpu.py test/manual/dsv41/test_exl3_ram_miss_graph_gpu.py test/manual/dsv41/test_exl3_graph_apply_gpu.py`.
Expected: `EXIT=0`.

- [ ] **Step 4: Commit**

```bash
git add test/manual/dsv41/test_exl3_verify_cpu_spill_gpu.py
git commit -m "test(exl3-cpu-experts): a captured 6-token verify spills its victimless lanes to the CPU, end to end

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Y3EXKJdMpP7nwX3qcBi8Vq"
```

---

### Task 10: The gate admits the target's CPU experts under a graphed DSpark verify

The refusal at `expert_stream_requirements_exl3.py:171-181` dates from `3937a8833a` (2026-09-30), before a verify
could run in the breakable graph. The narrowest change:
- CPU experts still require the breakable decode graph.
- Under speculation they ride `_check_graphed_verify`, which already admits only DSpark, static verify, DIRECT, and 1-32
  miss lanes, with a remedy that does not point at eager decode.
- A non-speculative launch takes exactly today's path.

**Files:**
- Modify: `python/sglang/srt/arg_groups/expert_stream_requirements_exl3.py`
- Test: `test/registered/unit/test_expert_stream_requirements_exl3.py`

**Interfaces:**
- Produces: `_check_graphed_verify(cfg, remedy: str = _EAGER_VERIFY_REMEDY)`, `_CPU_EXPERTS_VERIFY_REMEDY`.

- [ ] **Step 1: Update and add tests**

In the parametrize of `test_a_graphed_dspark_verify_needs_its_configuration`, replace the last row with
`({}, {**GRAPHED_VERIFY, "SGLANG_DSV41_CPU_EXPERTS": True}, "SGLANG_DSV41_CPU_EXPERTS needs"),` and change the
function's tail to:
```python
    if match != "SGLANG_DSV41_CPU_EXPERTS needs":
        assert "--cuda-graph-backend-decode disabled" in str(raised.value)
```
Replace `test_cpu_experts_need_the_breakable_decode_graph_without_speculation` with:
```python
@pytest.mark.parametrize(
    "changes, env",
    [
        ({}, _EAGER_DECODE),  # both phases disabled
        ({"cuda_graph_config": FULL_BS1}, {}),
        ({"speculative_algorithm": "DSPARK"}, _EAGER_DECODE),
    ],
    ids=["disabled", "full", "spec-disabled"],
)
def test_cpu_experts_need_the_breakable_decode_graph(model_dir, changes, env):
    """Refused before the speculative and backend rules; the refusal never suggests disabled decode graphs."""
    with pytest.raises(ValueError, match="pass --cuda-graph-backend-decode breakable") as refused:
        _gate(_launch(model_dir, **changes), **{**CPU_EXPERTS_ENV, **env})
    assert "disabled" not in str(refused.value)


DSPARK_BREAKABLE = dict(speculative_algorithm="DSPARK", cuda_graph_config=BREAKABLE_BS1)


def test_cpu_experts_with_a_graphed_dspark_verify_pass(model_dir):
    _gate(_launch(model_dir, **DSPARK_BREAKABLE), **CPU_EXPERTS_ENV, **GRAPHED_VERIFY)
    _gate(
        _launch(model_dir, **DSPARK_BREAKABLE), **CPU_EXPERTS_ENV, **GRAPHED_VERIFY,
        SGLANG_MOE_EXPERT_GRAPH_GATHER_MISS_LANES=32, SGLANG_MOE_EXPERT_GRAPH_GATHER_VICTIM_LANES=8,
    )


@pytest.mark.parametrize(
    "launch, env, match",
    [
        ({"speculative_algorithm": "EAGLE"}, GRAPHED_VERIFY, "graphs the verify of DSpark only"),
        ({}, {**GRAPHED_VERIFY, "SGLANG_MOE_EXPERT_GRAPH_GATHER_MISS_LANES": 0}, "MISS_LANES=1-32"),
        ({}, {**GRAPHED_VERIFY, "SGLANG_RAGGED_VERIFY_MODE": "compact"}, "SGLANG_RAGGED_VERIFY_MODE=static"),
    ],
)
def test_cpu_experts_under_speculation_need_the_graphed_dspark_verify(model_dir, launch, env, match):
    """The graphed verify's own rules, with a remedy for a CPU-experts launch: the verify must be graphed, so the
    eager-decode remedy is never offered."""
    with pytest.raises(ValueError, match=match) as refused:
        _gate(_launch(model_dir, **(DSPARK_BREAKABLE | launch)), **CPU_EXPERTS_ENV, **env)
    assert "--cuda-graph-backend-decode disabled" not in str(refused.value)
    assert "SGLANG_DSV41_CPU_EXPERTS" in str(refused.value)


@pytest.mark.parametrize(
    "env, match",
    [
        ({"SGLANG_DSV41_CPU_EXPERTS": False, "SGLANG_MOE_EXPERT_GRAPH_GATHER_VICTIM_LANES": 4}, "needs SGLANG_DSV41_CPU_EXPERTS=1"),
        ({"SGLANG_MOE_EXPERT_GRAPH_GATHER_VICTIM_LANES": 8}, "below SGLANG_MOE_EXPERT_GRAPH_GATHER_MISS_LANES"),
    ],
)
def test_victim_lanes_need_cpu_experts_and_fewer_than_the_miss_lanes(model_dir, env, match):
    with pytest.raises(ValueError, match=match):
        _gate(_launch(model_dir, **DSPARK_BREAKABLE), **{**CPU_EXPERTS_ENV, **GRAPHED_VERIFY, **env})
```
`GRAPHED_VERIFY` has `MISS_LANES` 8, so V = 8 is not below it.

- [ ] **Step 2: Run and see them fail** — `test/registered/unit/test_expert_stream_requirements_exl3.py`, CPU.
Expected: failures in the new tests, still refused "without speculative decoding".

- [ ] **Step 3: Implement**

```python
_EAGER_VERIFY_REMEDY = "or pass --cuda-graph-backend-decode disabled to run the DSpark verify eagerly"
# SGLANG_DSV41_CPU_EXPERTS computes in the captured graph's copy wait, so its verify cannot go eager.
_CPU_EXPERTS_VERIFY_REMEDY = "SGLANG_DSV41_CPU_EXPERTS serves the DSpark verify only in the decode graph"


def _check_graphed_verify(cfg, remedy: str = _EAGER_VERIFY_REMEDY) -> None:
```
Inside, replace each `{_EAGER_VERIFY_REMEDY}` with `{remedy}`. Then:
```python
def _check_victim_lanes() -> None:
    """SGLANG_MOE_EXPERT_GRAPH_GATHER_VICTIM_LANES: spill needs the CPU to take the victimless lanes."""
    victims = envs.SGLANG_MOE_EXPERT_GRAPH_GATHER_VICTIM_LANES.get()
    if not victims:
        return
    if not envs.SGLANG_DSV41_CPU_EXPERTS.get():
        raise ValueError(f"SGLANG_MOE_EXPERT_GRAPH_GATHER_VICTIM_LANES={victims} needs SGLANG_DSV41_CPU_EXPERTS=1")
    lanes = envs.SGLANG_MOE_EXPERT_GRAPH_GATHER_MISS_LANES.get()
    if not 1 <= victims < lanes:
        raise ValueError(
            f"SGLANG_MOE_EXPERT_GRAPH_GATHER_VICTIM_LANES={victims} must be below "
            f"SGLANG_MOE_EXPERT_GRAPH_GATHER_MISS_LANES={lanes}"
        )
```
In `_check`, replace the CPU-experts refusal and the speculative block:
```python
    cpu_experts = envs.SGLANG_DSV41_CPU_EXPERTS.get()
    speculative = getattr(cfg, "speculative_algorithm", None) is not None
    if cpu_experts and graph.decode.backend != Backend.BREAKABLE:
        # Checked before the speculative and backend rules: each would point a CPU-experts launch at the other's
        # decode backend.
        raise ValueError(
            "SGLANG_DSV41_CPU_EXPERTS computes experts inside the captured decode graph's copy wait; "
            "pass --cuda-graph-backend-decode breakable"
        )
    if speculative and graph.decode.backend != Backend.DISABLED:
        _check_graphed_verify(cfg, _CPU_EXPERTS_VERIFY_REMEDY if cpu_experts else _EAGER_VERIFY_REMEDY)
    _check_victim_lanes()
```
Module docstring: replace the speculative bullet with:
```
* Speculative decoding as DSpark: its verify eager (decode graphs disabled), or in the breakable decode graph on DIRECT
  residency at 1-32 miss lanes with a static verify (§33.8). The target's CPU experts serve that graphed verify; with
  SGLANG_MOE_EXPERT_GRAPH_GATHER_VICTIM_LANES below the miss lanes, the lanes past the victims are theirs (spill).
```
In `_check_dspark_cpu_experts`, the comment "SGLANG_DSV41_CPU_EXPERTS, which also selects it, is refused under
speculation, so these two defines are the way in" becomes "SGLANG_DSV41_CPU_EXPERTS also selects it, but a draft-only
launch has it off, so these two defines are the way in. The recipe sets them in either case."

- [ ] **Step 4: Run and pass** — the whole file. Expected: `EXIT=0`.

- [ ] **Step 5: Commit**

```bash
git add python/sglang/srt/arg_groups/expert_stream_requirements_exl3.py test/registered/unit/test_expert_stream_requirements_exl3.py
git commit -m "feat(exl3-gate): the target's CPU experts serve a graphed DSpark verify; VICTIM_LANES needs them

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Y3EXKJdMpP7nwX3qcBi8Vq"
```

---

### Task 11: The recipe's DSpark mode and the production switch

Which of the DSpark arms' overrides are still required with a graphed verify, and why (research, 2026-10-06):

| Override in `ab_cpu_draft.COMMON` | Verdict for this mode | Evidence |
|---|---|---|
| `SGLANG_DSV41_CPU_EXPERTS=0` | **Dropped**: the point of this plan | Tasks 3-10 |
| `SGLANG_LAYER_MAJOR_PREFILL_MIN_TOKENS=0` | **Required** | `layer_major/gate.py:38-50` refuses any speculative algorithm |
| `SGLANG_DSV41_ENABLE_PREFILL_FILLS=0` | **Dropped** | refused only without graph gather (`exl3_expert_format.py:226-231`); `graphed_verify.py` already keeps 1 |
| `SGLANG_MOE_EXPERT_GRAPH_GATHER/GPU_RESIDENCY_UPDATE/INSERT_ON_MISS_STAGE/FUSED_PLAN=0` | **Dropped** | the graphed verify needs them on (`_check_graphed_verify`) |
| `SGLANG_DSV41_ENGRAM_HOST_NODE_CACHE_URING=0`, `..._ENGRAM_DEVICE_WAIT=0` | **Dropped**, with a fallback | A verify never takes the native captured lookup (`engram.py:124-139`: one token, decode mode); it runs the eager break, and the uring store serves N tokens (`engram_file_table.py:147-163`). Device wait engages only in a one-token decode capture, which a DSpark target never captures. No DSpark run has had them on, so Task 13's smoke checks them, with the fallback named there. |
| `SGLANG_SM120_FLASHMLA_BACKEND=triton` | **Kept** | No refusal: flashinfer handles up to 64 query rows (`flash_mla_sm120.py:212, 317-341`). But every DSpark text result (§33.2, §33.8, §33.9) is triton, and §33.2 names the sm120 attention at 6 rows as a drift suspect. Dropping it is its own A/B. |
| `SGLANG_EXL3_CPU_ACT_RESIDUAL=1`, `SGLANG_EXL3_CPU_ACT_BLOCK=128` | **Kept** | `_check_dspark_cpu_experts` requires them; with CPU experts on they equal the build's own values (`ext.py:76-89`) |

**Files:**
- Modify: `benchmarks/dsv41_baseline/arm_env.py`, `benchmarks/dsv41_baseline/launch_prod.sh`
- Create: `benchmarks/dsv41_baseline/test_dspark_recipe.py`
- Test: `test/registered/unit/layers/moe/test_threading_config.py`

**Interfaces:**
- Produces:
  - `arm_env.DSPARK_DRAFT`, `DSPARK_RESIDENT`, `DSPARK_DRAFT_CORES = "12-15"`, `DSPARK_ARGV`;
  - `dspark_env() -> dict[str, str]`, the overrides on top of `base_env()`;
  - `prod_env() -> dict[str, str]`;
  - `PROD_DSPARK = False`;
  - `ServerArgs.dspark: bool = False`.
  - `ServerArgs.prod()` sets `dspark=PROD_DSPARK`, and `launch_prod.sh` uses `prod_env()`.

- [ ] **Step 1: Write the failing tests**

`benchmarks/dsv41_baseline/test_dspark_recipe.py`:
```python
"""The DSpark mode of the recipe: both CPU-expert clients on, the graphed verify's configuration, and argv the server
parses as DSpark (plan 2026-10-06-dsv41-dspark-both-cpu-experts Task 11)."""

import argparse

import arm_env


def test_dspark_env_turns_both_cpu_expert_clients_on_with_spill():
    env = arm_env.arm_env(arm_env.dspark_env())
    assert env["SGLANG_DSV41_CPU_EXPERTS"] == "1"
    assert env["SGLANG_DSV41_ENABLE_DSPARK_CPU_EXPERTS"] == "1"
    assert env["SGLANG_DSV41_DSPARK_CPU_EXPERTS_CORES"] == "12-15"
    assert (env["SGLANG_MOE_EXPERT_GRAPH_GATHER_MISS_LANES"], env["SGLANG_MOE_EXPERT_GRAPH_GATHER_VICTIM_LANES"]) == ("32", "8")
    assert env["SGLANG_RAGGED_VERIFY_MODE"] == "static"
    assert env["SGLANG_LAYER_MAJOR_PREFILL_MIN_TOKENS"] == "0"
    assert env["SGLANG_MOE_HOT_GPU_MB"] == "12040"
    assert env["SGLANG_DSV41_ENABLE_PREFILL_FILLS"] == "1"  # the recipe's, kept


def test_prod_is_unchanged_until_the_switch():
    assert arm_env.PROD_DSPARK is False
    assert arm_env.prod_env() == arm_env.base_env()
    assert "--speculative-algorithm" not in arm_env.ServerArgs.prod().argv()


def test_the_dspark_argv_parses_as_dspark_at_block_size_5():
    from sglang.srt.server_args import ServerArgs

    argv = arm_env.ServerArgs(port=1, dspark=True).argv()
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    ns = parser.parse_args(argv[3:])
    assert ns.speculative_algorithm == "DSPARK"
    assert ns.speculative_draft_model_path == arm_env.DSPARK_DRAFT
    assert int(ns.speculative_dspark_block_size) == 5
    assert ns.max_running_requests == 1
```
Append to `test_threading_config.py`:
```python
def test_the_dspark_recipe_layout_keeps_both_cpu_expert_clients_apart(divix01):
    """arm_env's DSpark mode: server 0-5,36-41, RAM 17 by name, target THREADS=10, the draft named on 12-15. The draft's
    cores leave every derived role, so node 0's target team is 6-11, node 1's 18-27, copy 16, node 1's RAM thread 35;
    no core or SMT sibling is in two roles. Mutation: leave named draft cores out of `taken` -- the target takes 6-15."""
    config = resolve(divix01, affinity=RECIPE_SERVER, cpu_experts=True, threads=10, spin_core=17, draft=True,
                     draft_cores="12-15")
    assert config.copy_cpus == (16,)
    assert [(p.ram, p.cpu) for p in config.plans] == [(17, tuple(range(6, 12))), (35, tuple(range(18, 28)))]
    assert config.draft_cpus == (12, 13, 14, 15)
    roles = [*config.copy_cpus, *config.draft_cpus] + [c for p in config.plans for c in (p.ram, *p.cpu)]
    physical = [c % 36 for c in roles]
    assert len(set(physical)) == len(physical)
```

- [ ] **Step 2: Run and see them fail**

On divix01:
- `cd benchmarks/dsv41_baseline && PYTHONPATH=$WT/python:. taskset -c 0-63 /data/models/slang/.venv/bin/python -m pytest -q test_dspark_recipe.py`;
- `test/registered/unit/layers/moe/test_threading_config.py -k dspark_recipe`.
Expected: `AttributeError: module 'arm_env' has no attribute 'dspark_env'`. The threading test may already pass:
`ThreadingConfig` needs no change for option (a). It then stands as the guard for that layout.

- [ ] **Step 3: Implement `arm_env.py`**

After `PROD_HOST`:
```python
# DSpark with both CPU-expert clients (plan 2026-10-06-dsv41-dspark-both-cpu-experts). Production serves it only once
# PROD_DSPARK is True (Owner decision 1, the A/B of that plan's Task 13).
PROD_DSPARK = False
DSPARK_DRAFT = f"{CC}/dsv41-dspark-draft"
# Each stage's top-32 draft experts stay on the GPU, the other 96 on the CPU (§33.4).
DSPARK_RESIDENT = f"{CC}/analysis/dsv41-dspark/cpu-draft-routes/resident-top32.json"
# The draft's CPU experts by name: node 0's 12-15, so the target's node-0 team derives 6-11 (Owner decision 2a).
DSPARK_DRAFT_CORES = "12-15"
# gamma = 5 draft tokens, a 6-token verify (speculative_hook.py).
DSPARK_ARGV = (
    "--speculative-algorithm", "DSPARK",
    "--speculative-draft-model-path", DSPARK_DRAFT,
    "--speculative-dspark-block-size", "5",
)


def dspark_env() -> dict[str, str]:
    """The DSpark mode's overrides on base_env: the graphed verify with the target's CPU experts and spill, and the
    draft's CPU experts. Every value's reason is in the plan's Task 11 table."""
    return {
        # The verify in the breakable decode graph at a 32-lane record, 8 of them VRAM victims; the rest are CPU lanes.
        "SGLANG_MOE_EXPERT_GRAPH_GATHER_MISS_LANES": "32",
        "SGLANG_MOE_EXPERT_GRAPH_GATHER_VICTIM_LANES": "8",
        "SGLANG_RAGGED_VERIFY_MODE": "static",
        # Layer-major prefill refuses speculative decoding (layer_major/gate.py).
        "SGLANG_LAYER_MAJOR_PREFILL_MIN_TOKENS": "0",
        # The draft's CPU experts on their own cores, reading the optimized build (its gate names both defines).
        "SGLANG_DSV41_ENABLE_DSPARK_CPU_EXPERTS": "1",
        "SGLANG_DSV41_DSPARK_CPU_EXPERTS_CORES": DSPARK_DRAFT_CORES,
        "SGLANG_DSV41_DSPARK_DRAFT_RESIDENT_PATH": DSPARK_RESIDENT,
        "SGLANG_EXL3_CPU_ACT_RESIDUAL": "1",
        "SGLANG_EXL3_CPU_ACT_BLOCK": "128",
        # 4040 MiB of the hot cache go to the draft's resident experts, dense weights and KV (Owner decision 4).
        "SGLANG_MOE_HOT_GPU_MB": "12040",
        # Every DSpark text result so far ran triton attention (§33.2, §33.8, §33.9).
        "SGLANG_SM120_FLASHMLA_BACKEND": "triton",
    }


def prod_env() -> dict[str, str]:
    """Production's env: the base recipe, with the DSpark mode once PROD_DSPARK is set."""
    return arm_env(dspark_env()) if PROD_DSPARK else base_env()
```
`ServerArgs`: add the field `dspark: bool = False` with the comment "the DSpark mode's argv (DSPARK_ARGV); its env is
dspark_env()". `prod` returns `cls(port=PROD_PORT, host=PROD_HOST, dspark=PROD_DSPARK)`. At the end of `argv`, before
`return argv`:
```python
        if self.dspark:
            argv += list(DSPARK_ARGV)
```

- [ ] **Step 4: `launch_prod.sh`**

Replace both uses of `arm_env.base_env()`, the DRY_RUN printer and the exec'd launcher, with `arm_env.prod_env()`.
In the header comment, "the arm_env base recipe, unchanged" becomes "arm_env.prod_env(): the base recipe, plus the
DSpark mode when arm_env.PROD_DSPARK is set". The DRY_RUN branch also prints the line
`print("dspark:", arm_env.PROD_DSPARK)`.

- [ ] **Step 5: Run and pass**, plus `DRY_RUN=1 benchmarks/dsv41_baseline/launch_prod.sh` from the worktree on divix01.
It takes no lock and starts nothing. Expected: `dspark: False`, and the env and argv byte-identical to production's
current DRY_RUN output. Diff them and record it.

- [ ] **Step 6: Commit**

```bash
git add benchmarks/dsv41_baseline/arm_env.py benchmarks/dsv41_baseline/launch_prod.sh \
  benchmarks/dsv41_baseline/test_dspark_recipe.py test/registered/unit/layers/moe/test_threading_config.py
git commit -m "feat(dsv41-baseline): the recipe's DSpark mode with both CPU-expert clients; production behind PROD_DSPARK (off)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Y3EXKJdMpP7nwX3qcBi8Vq"
```

---

### Task 12: A/B tooling: accept length from the driver, the text band, the three-arm driver

**Files:**
- Modify: `scripts/expert_prediction/benchmarks/run_capture_sessions.py`
- Create: `scripts/dsv41/dspark_text_band.py`
- Create: `analysis/dsv41-drive/dspark/both_cpu_ab.py`
- Test: `test/registered/unit/scripts/test_dspark_text_band.py`, `test/registered/unit/scripts/test_run_capture_sessions_spec.py`

**Interfaces:**
- Produces:
  - `run_capture_sessions.chunk_spec_details(chunk) -> dict | None`. Each results record gains `spec_tokens_details`,
    `None` without speculation. The request sends `return_spec_tokens_details: true`.
  - `dspark_text_band.compare(base, other, band=1.4) -> dict`.
  - `both_cpu_ab.py OUT [ARM...]`.

- [ ] **Step 1: Write the failing tests**

`test/registered/unit/scripts/test_run_capture_sessions_spec.py`:
```python
"""The session driver records a speculative server's per-request details (accept length's inputs) (CPU)."""

import importlib.util
import os

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))


def _driver():
    path = os.path.join(ROOT, "scripts", "expert_prediction", "benchmarks", "run_capture_sessions.py")
    spec = importlib.util.spec_from_file_location("run_capture_sessions", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_sglext_chunk_carries_the_spec_details():
    driver = _driver()
    chunk = {"choices": [], "sglext": {"spec_tokens_details": {"spec_verify_ct": 40, "spec_accept_length": 3.2}}}
    assert driver.chunk_spec_details(chunk) == {"spec_verify_ct": 40, "spec_accept_length": 3.2}
    assert driver.chunk_spec_details({"choices": [{"delta": {}}]}) is None
    assert driver.chunk_spec_details({"choices": [], "sglext": {"cached_tokens_details": {}}}) is None
```
`test/registered/unit/scripts/test_dspark_text_band.py`:
```python
"""The DSpark text bar: a run's text may leave the base run's only at a near-tie of the base (§33.2, §33.9) (CPU)."""

import importlib.util
import os

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))


def _band():
    spec = importlib.util.spec_from_file_location("dspark_text_band", os.path.join(ROOT, "scripts", "dsv41", "dspark_text_band.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tokens(*rows):
    return [{"token": chosen, "top": top} for chosen, top in rows]


def test_identical_text_passes_with_no_flips():
    band = _band()
    run = [{"session_id": "s", "tokens": _tokens(("a", [["a", -0.1], ["b", -2.0]]))}]
    assert band.compare(run, run)["pass"] and band.compare(run, run)["flips"] == []


def test_a_flip_within_the_band_passes_and_one_past_it_or_off_the_top_k_fails():
    band = _band()
    base = [{"session_id": "s", "tokens": _tokens(("a", [["a", -0.1], ["b", -1.2], ["c", -3.0]]))}]
    near = [{"session_id": "s", "tokens": _tokens(("b", [["b", -0.2], ["a", -0.3]]))}]
    far = [{"session_id": "s", "tokens": _tokens(("c", [["c", -0.2], ["a", -0.3]]))}]
    off = [{"session_id": "s", "tokens": _tokens(("z", [["z", -0.2], ["a", -0.3]]))}]
    assert band.compare(base, near)["pass"]  # 1.1 behind the base argmax
    assert not band.compare(base, far)["pass"]  # 2.9 behind
    assert not band.compare(base, off)["pass"]  # not in the base's top-k: unbounded
    assert band.compare(base, near)["flips"][0]["gap"] == 1.1


def test_different_prompts_are_refused():
    band = _band()
    try:
        band.compare([{"session_id": "a", "tokens": []}], [{"session_id": "b", "tokens": []}])
    except ValueError:
        return
    raise AssertionError("ran different prompts")
```

- [ ] **Step 2: Run and see them fail** — CPU. Expected: `AttributeError: ... chunk_spec_details` and `FileNotFoundError`.

- [ ] **Step 3: Implement**

`run_capture_sessions.py`:
- Add above `_stream_chat`:
  ```python
  def chunk_spec_details(chunk: dict):
      """A speculative server's per-request details (spec_verify_ct, spec_accept_length, ...) from its sglext chunk, or
      None (return_spec_tokens_details; a non-speculative server sends none)."""
      return (chunk.get("sglext") or {}).get("spec_tokens_details")
  ```
- In `_stream_chat`'s request body, add `"return_spec_tokens_details": True,`. Initialize `spec_details = None`. In
  the chunk loop, right after `chunk = json.loads(payload)`:
  ```python
              spec_details = chunk_spec_details(chunk) or spec_details
  ```
  Return `"spec_tokens_details": spec_details` in the result dict.
- In the success `record`, add `"spec_tokens_details": result["spec_tokens_details"],`.

`scripts/dsv41/dspark_text_band.py`:
```python
"""The DSpark text bar (DSV41_REFERENCE.md §33.2, §33.9): exact text is the wrong test for a speculative run, because
its verify forward settles near-ties differently from one-token decode. Of two logprob_probe.py outputs of the same
prompts (scripts/expert_prediction/prefetch/logprob_probe.py, --top-logprobs 5), the other run may leave the base run
only where its token is within `band` nats of the base's argmax at the first divergence. 1.4 is §33.2's
verify-vs-decode band; §33.9's largest draft-to-draft divergence was 1.125.

    python scripts/dsv41/dspark_text_band.py BASE.json OTHER.json [--band 1.4]
"""

import argparse
import json

BAND_NATS = 1.4


def compare(base, other, band: float = BAND_NATS) -> dict:
    if [a["session_id"] for a in base] != [b["session_id"] for b in other]:
        raise ValueError("the two probes ran different prompts")
    flips, compared = [], 0
    for a, b in zip(base, other):
        for index, (x, y) in enumerate(zip(a["tokens"], b["tokens"])):
            compared += 1
            if x["token"] == y["token"]:
                continue
            top = dict(x["top"])
            gap = round(x["top"][0][1] - top[y["token"]], 6) if y["token"] in top else float("inf")
            flips.append({"session_id": a["session_id"], "index": index, "gap": gap, "within": gap <= band})
            break
    return {
        "prompts": len(base),
        "compared_tokens": compared,
        "flips": flips,
        "max_gap": max((f["gap"] for f in flips), default=0.0),
        "pass": all(f["within"] for f in flips),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("base")
    parser.add_argument("other")
    parser.add_argument("--band", type=float, default=BAND_NATS)
    args = parser.parse_args()
    with open(args.base) as f:
        base = json.load(f)
    with open(args.other) as f:
        other = json.load(f)
    report = compare(base, other, args.band)
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["pass"] else 1)


if __name__ == "__main__":
    main()
```

`analysis/dsv41-drive/dspark/both_cpu_ab.py`:
```python
"""Server A/B (plan 2026-10-06-dsv41-dspark-both-cpu-experts Task 13): today's production recipe against DSpark with
both CPU-expert clients, and DSpark with the draft's only (the target's CPU experts off, §33.9's configuration in a
server) to isolate them.

Per arm: run_arm.sh's timed set over the 8 corpus sessions (ms/token, and accept length from the driver's
spec_tokens_details), then a short-lived server on the same env for logprob_probe.py (8 prompts, 128 tokens, top-5).
The text bar is dspark_text_band.py against prod. Run on divix01 from the pushed worktree holding rowimg-disk.lock
only: run_arm.sh takes cc-gpu.lock itself, and the probe phase takes it here (lock order: disk, then GPU).

    flock /data/models/slang/nvfp4-work/rowimg-disk.lock python analysis/dsv41-drive/dspark/both_cpu_ab.py OUT [ARM ...]
"""

import fcntl
import json
import os
import shlex
import signal
import statistics
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "benchmarks", "dsv41_baseline"))
import arm_env  # noqa: E402

PORT = 30017
PROBE_PORT = 30018
SESSIONS = "/mnt/nvme2/nvfp4-work/benchmarks/full/sessions.jsonl"
DRAFT_ONLY = {
    "SGLANG_DSV41_CPU_EXPERTS": "0",
    "SGLANG_MOE_EXPERT_GRAPH_GATHER_MISS_LANES": "8",
    "SGLANG_MOE_EXPERT_GRAPH_GATHER_VICTIM_LANES": "0",
}
# arm: (overrides on base_env, DSpark argv)
ARMS = {
    "prod": ({}, False),
    "dspark-draft-only": ({**arm_env.dspark_env(), **DRAFT_ONLY}, True),
    "dspark-both": (arm_env.dspark_env(), True),
}


def _overrides(arm: str, out: str) -> dict:
    overrides, _ = ARMS[arm]
    return {**overrides, "SGLANG_MOE_HOT_METRICS_FILE": os.path.join(out, f"{arm}.metrics.jsonl")}


def run_timed(arm: str, out: str) -> int:
    _, dspark = ARMS[arm]
    env = os.environ | {
        "DSV41_RUN_ROOT": out,
        "DSV41_SESSION_INDICES": ",".join(str(i) for i in range(8)),
        "DSV41_EXTRA_SERVER_ARGS": shlex.join(arm_env.DSPARK_ARGV) if dspark else "",
    }
    cmd = [os.path.join(REPO, "benchmarks", "dsv41_baseline", "run_arm.sh"), arm, str(PORT)]
    cmd += [f"{k}={v}" for k, v in _overrides(arm, out).items()]
    print("===", " ".join(cmd), flush=True)
    return subprocess.run(cmd, env=env, cwd=REPO).returncode


def _healthy(port: int, deadline: float) -> bool:
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=10) as r:
                if r.status == 200:
                    return True
        except OSError:
            pass
        time.sleep(5)
    return False


def run_probe(arm: str, out: str) -> int:
    """The arm's server on PROBE_PORT under cc-gpu.lock, for logprob_probe.py only."""
    _, dspark = ARMS[arm]
    env = os.environ | arm_env.arm_env(_overrides(arm, out)) | {"PYTHONPATH": os.path.join(REPO, "python")}
    argv = arm_env.ServerArgs(port=PROBE_PORT, dspark=dspark).argv()
    with open(arm_env.GPU_LOCK, "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with open(os.path.join(out, f"{arm}.probe-server.log"), "w") as log:
            server = subprocess.Popen(["taskset", "-c", arm_env.SERVER_CORES, *argv], env=env, stdout=log,
                                      stderr=subprocess.STDOUT, cwd=REPO, start_new_session=True)
        try:
            if not _healthy(PROBE_PORT, time.monotonic() + arm_env.HEALTH_TIMEOUT_S):
                return 1
            probe = os.path.join(REPO, "scripts", "expert_prediction", "prefetch", "logprob_probe.py")
            return subprocess.run(
                ["taskset", "-c", arm_env.DRIVER_CORES, arm_env.PYTHON, probe, "--port", str(PROBE_PORT),
                 "--sessions", SESSIONS, "--prompts", "8", "--max-tokens", "128", "--top-logprobs", "5",
                 "--out", os.path.join(out, f"{arm}.probe.json")], cwd=REPO,
            ).returncode
        finally:
            os.killpg(server.pid, signal.SIGTERM)
            try:
                server.wait(timeout=120)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGKILL)
                server.wait()


def _results(out: str, arm: str) -> list[dict]:
    runs = sorted(os.listdir(os.path.join(out, "servers", arm)))
    with open(os.path.join(out, "servers", arm, runs[-1], "results.jsonl")) as f:
        return [json.loads(line) for line in f if line.strip()]


def summarize(out: str) -> dict:
    sys.path.insert(0, os.path.join(REPO, "scripts", "dsv41"))
    import dspark_text_band

    summary = {}
    for arm in ARMS:
        try:
            rows = _results(out, arm)
        except FileNotFoundError:
            continue
        ms = [1000.0 / r["decode_tokens_per_sec"] for r in rows if r.get("decode_tokens_per_sec")]
        spec = [r.get("spec_tokens_details") or {} for r in rows]
        verifies = sum(s.get("spec_verify_ct", 0) for s in spec)
        entry = {
            "sessions": len(rows),
            "ms_per_token_median": statistics.median(ms) if ms else None,
            "ms_per_token": ms,
            "accept_length": (sum(r["completion_tokens"] for r in rows) / verifies) if verifies else None,
        }
        metrics = os.path.join(out, f"{arm}.metrics.jsonl")
        if os.path.exists(metrics):
            with open(metrics) as f:
                last = json.loads([line for line in f if line.strip()][-1])
            graphed = last.get("counters", {}).get("graphed_verify")
            if graphed and graphed.get("graphed_verify_ct"):
                entry["reverify_rate"] = graphed["verify_overflow_ct"] / graphed["graphed_verify_ct"]
        probes = {a: os.path.join(out, f"{a}.probe.json") for a in ("prod", arm)}
        if arm != "prod" and all(os.path.exists(p) for p in probes.values()):
            with open(probes["prod"]) as f:
                base = json.load(f)
            with open(probes[arm]) as f:
                other = json.load(f)
            entry["text_vs_prod"] = dspark_text_band.compare(base, other)
        summary[arm] = entry
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def main():
    out = os.path.abspath(sys.argv[1])
    arms = sys.argv[2:] or list(ARMS)
    os.makedirs(out, exist_ok=True)
    for arm in arms:
        for step in (run_timed, run_probe):
            rc = step(arm, out)
            print(f"{arm} {step.__name__}: rc={rc}", flush=True)
            if rc:
                sys.exit(rc)
    print(json.dumps(summarize(out), indent=2), flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run and pass** — both new test files, CPU. Then
`/data/models/slang/.venv/bin/python -c "import ast,sys; ast.parse(open('analysis/dsv41-drive/dspark/both_cpu_ab.py').read())"`.
Expected: `EXIT=0`, no output.

- [ ] **Step 5: Commit**

```bash
git add scripts/expert_prediction/benchmarks/run_capture_sessions.py scripts/dsv41/dspark_text_band.py \
  analysis/dsv41-drive/dspark/both_cpu_ab.py test/registered/unit/scripts/test_dspark_text_band.py \
  test/registered/unit/scripts/test_run_capture_sessions_spec.py
git commit -m "feat(dspark-ab): the driver records accept-length inputs; the near-tie text bar; the three-arm server A/B

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Y3EXKJdMpP7nwX3qcBi8Vq"
```

---

### Task 13: Smoke, then the server A/B on divix01, then §33.10

Needs a GPU window: production down, owned by the owner. The executor waits on the locks.

**Files:**
- Modify: `DSV41_REFERENCE.md` (new §33.10, after §33.9)

- [ ] **Step 1: Smoke `dspark-both` (2 sessions)**

```bash
cd /data/models/slang/nvfp4-work/wt-both-cpu
G=/data/models/slang/nvfp4-work/cc-expert-prediction/analysis/dsv41-dspark/both-cpu
mkdir -p $G/smoke
flock /data/models/slang/nvfp4-work/rowimg-disk.lock env DSV41_RUN_ROOT=$G/smoke \
  DSV41_EXTRA_SERVER_ARGS="$(/data/models/slang/.venv/bin/python -c 'import sys; sys.path.insert(0,"benchmarks/dsv41_baseline"); import arm_env, shlex; print(shlex.join(arm_env.DSPARK_ARGV))')" \
  benchmarks/dsv41_baseline/run_arm.sh dspark-both 30017 \
  $(/data/models/slang/.venv/bin/python -c 'import sys; sys.path.insert(0,"benchmarks/dsv41_baseline"); import arm_env; print(" ".join(f"{k}={v}" for k,v in arm_env.dspark_env().items()))')
```
Every one of these must hold in `$G/smoke/servers/dspark-both/run-*/server.log`. Record each line, or record its
absence as a failure:
1. `DSpark: EXL3 draft graphs on`.
2. `numa dspark draft: node0 cpus=12-15 (4)`, and CPU experts groups on cores `[6, ..., 11]` and `[18, ..., 27]`.
3. The decode graph captured as `TARGET_VERIFY`, and `exl3 RAM miss copy engine armed after`.
4. No `fail-stop`, `__trap`, `RemoteDisconnected`, `CUDA error`, or `out of memory`.
5. The KV pool line (`max_total_num_tokens`). Record it for Step 2's bar.
6. The CPU calibration line (`CPU experts group N: ... split`, or the `calibration skipped/failed` warning), with how
   long startup took from `Load weight end` to `/health` 200.

Engram fallback: if the log shows an Engram failure (`engram` in a traceback), re-run with
`SGLANG_DSV41_ENGRAM_HOST_NODE_CACHE_URING=0 SGLANG_DSV41_ENABLE_ENGRAM_DEVICE_WAIT=0` appended. Then commit that pair
into `dspark_env()` with the traceback quoted in its comment, as a fix-up commit.

- [ ] **Step 2: The A/B**

```bash
flock /data/models/slang/nvfp4-work/rowimg-disk.lock /data/models/slang/.venv/bin/python \
  analysis/dsv41-drive/dspark/both_cpu_ab.py $G/ab 2>&1 | tee $G/ab/driver.log; echo "EXIT=${PIPESTATUS[0]}"
```
Bars, each recorded from `$G/ab/summary.json` and the logs:
1. All three arms finish with `rc=0`, with no fail-stop or trap.
2. `dspark-both` and `dspark-draft-only` both have `text_vs_prod.pass` true: every first divergence from production
   is within 1.4 nats of production's argmax (§33.2, §33.9). Report the flips and `max_gap`.
3. `dspark-both.reverify_rate` is strictly below `dspark-draft-only.reverify_rate` (which §33.8/§33.9 put at 1.00).
   That is the spill working.
4. `dspark-both`'s server log shows nonzero target CPU-expert jobs (the periodic `CPU experts group` stats).
5. `dspark-both`'s KV pool is at least the `prod` arm's.
6. Report for all three: `ms_per_token_median` with the per-session list, `accept_length`, and the reverify rate.
   This is the input to Owner decision 1.

A failed bar is reported as failed, with its numbers. Do not re-run an arm to get a different number without saying so.

- [ ] **Step 3: Write §33.10** in `DSV41_REFERENCE.md`, after §33.9's "What this plan does not do" list:
  - the commit and the plan;
  - what changed (Tasks 2-11, one line each);
  - Steps 1-2's commands;
  - a table of the three arms × (ms/token median, accept length, reverify rate, text pass and max gap, KV pool);
  - each bar's verdict;
  - the gaps: the draft CPU forward ms is still unmeasured (§33.9), and the CPU split is calibrated per one-token job;
  - the recommendation for Owner decision 1.

Use only measured numbers. Where a number is missing, say so.

- [ ] **Step 4: Commit and push**

```bash
git add DSV41_REFERENCE.md
git commit -m "docs(dsv41): section 33.10 -- DSpark with both CPU-expert clients, the server A/B against production

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Y3EXKJdMpP7nwX3qcBi8Vq"
git push origin dsv41-dspark-both-cpu
```

---

### Task 14: Flip production to DSpark (owner-gated)

Run only on the owner's go (Owner decision 1), given in their own words after reading §33.10.

**Files:**
- Modify: `benchmarks/dsv41_baseline/arm_env.py` (`PROD_DSPARK = True`), `benchmarks/dsv41_baseline/test_dspark_recipe.py`

- [ ] **Step 1: Change the test first**

Replace `test_prod_is_unchanged_until_the_switch` with:
```python
def test_prod_serves_the_dspark_mode():
    assert arm_env.PROD_DSPARK is True
    assert arm_env.prod_env() == arm_env.arm_env(arm_env.dspark_env())
    argv = arm_env.ServerArgs.prod().argv()
    assert argv[argv.index("--speculative-algorithm") + 1] == "DSPARK"
```
Run it and see it fail (`PROD_DSPARK is False`).

- [ ] **Step 2: Set `PROD_DSPARK = True`.** Add a comment above it naming §33.10 and the owner's go (date, words).
Run the test and see it pass. Then run `DRY_RUN=1 benchmarks/dsv41_baseline/launch_prod.sh` and check `dspark: True`,
the DSpark argv, and `dspark_env()`'s values in the printed env.

- [ ] **Step 3: Commit and push the branch.** Merging to `master` and restarting production are the owner's steps.
This plan does neither.

```bash
git add benchmarks/dsv41_baseline/arm_env.py benchmarks/dsv41_baseline/test_dspark_recipe.py
git commit -m "feat(dsv41-baseline): production serves DSpark with both CPU-expert clients (owner go, §33.10)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Y3EXKJdMpP7nwX3qcBi8Vq"
git push origin dsv41-dspark-both-cpu
```

---

## Self-review notes (writer's pass, 2026-10-06)

- **Spec coverage.**
  - Both clients at once: Tasks 3-11.
  - The verify, graphed (Tasks 3-9) and eager re-verify: unchanged and GPU-only (Owner decision 5), its rate measured
    in Task 13.
  - The overflow fallback: Tasks 3 and 7.
  - Prefill: unchanged. The target's CPU experts type lanes only on a captured post (`lease_kernels.cuh:151`), prefill
    is eager, and the DSpark draft does not run at prefill (`dspark_worker_v2.py:653-745`).
  - Cores: Owner decision 2 and Task 11's layout test.
  - Lease and pinned-tier interplay: separate channels and areas (`LEASE_PROTOCOL.md:71`); strictly sequential on one
    stream; draft weights pageable, not tier rows (`exl3.py:431-441`). Task 8 documents it.
  - Gate: Task 10. Recipe and launch: Task 11. Validation: Tasks 2-9, 12, 13.
- **Type consistency.**
  - The FFI tail order is `cpu_x, cpu_x_dst, cpu_weights, cpu_tokens_max, cpu_x_token_bytes, spill, overflow_flag,
    gather_overflow, use_pdl` in Tasks 3, 4 and 6's raw calls.
  - `victim_lanes` is used in Tasks 7, 8 and 10.
  - `cpu_row_bytes(hidden, tokens, lanes)` is used in Tasks 4, 5 and 8.
  - `spill = (overflow_flag, gather_overflow[row:row+1])` is used in Tasks 3 and 8.
- **Known open item, not a placeholder.** Task 9's last block cannot know in advance which plan lanes DIRECT makes
  live, so it widens the cold set until staging runs out. Its bar is fixed; the routes that trip it are recorded.
