# CPU-Expert Insert-on-Miss: Replay Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Decide, by replay, whether CPU-computed experts should also be copied into the GPU hot cache off the critical
path. If they should, this plan's results become the spec for the device change.

**Architecture:** `tier_sim.DirectInsertReplay.graph_forward` gains a `cpu_insert` hook that models the device design
below. A chosen CPU lane frees its own victim slot at this forward and leaves it FILLING, which is neither mapped nor
shortlisted. The expert is mapped at the next forward's commit of the same layer. `cpu_expert_sim.py` adds policies
that insert the head-most (highest-scored) P CPU lanes per layer, and a `--cpu-scale` sweep that replays every policy
at CPU speed-ups 1, 2, 4 and 8. Each policy is costed with the existing `split_costs` bounds. A pre-registered gate
on the router capture decides whether the device plan gets written. No served code changes in this plan.

**Tech Stack:** Python 3, numpy, pytest. Runs on divix01, CPU only.

**Spec:** the following sections of `DSV41_REFERENCE.md`:
- §30.2: CPU lanes take the lowest-scored RAM hits, and a CPU lane is never inserted.
- §30.3: insertion policies replayed. Deferred inserts lost at today's CPU speed once the link cost was counted.
- §31: the slot-map protocol, its staging slots, and the "Why it is safe" invariants in
  `analysis/dsv41-drive/LEASE_PROTOCOL.md`.

The evidence that motivates this plan is under "Evidence" below.

## Evidence (why now)

The CPU-speed sweep was run on 2026-10-02 with `cpu_expert_sim` in slot-map mode, a 16,080 MB hot cache, CPU
experts on, and flat CPU cost scaled by 1/k. Outputs are in
`divix01:cc-expert-prediction/analysis/dsv41-dspark/verify-union/cpu-sweep/`.

Two arms, in tok/s:
- **insert:** CPU-computed experts are still inserted, with the copy not costed. This is §30.3's upper bound.
- **no-insert:** today's behaviour.

| CPU speed-up k | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| insert | 14.6 | 17.5 | 20.0 | 21.7 | 22.6 |
| no-insert | 13.8 | 15.6 | 15.3 | 18.5 | 20.8 |

At k ≥ 4 the CPU takes nearly every RAM hit, and the hot hit rate falls from 0.65 to 0.20. The gap between the arms
is the prize, up to about 23% at k = 4. §30.3 measured the opposite at k = 1: deferred inserts lost once the link was
costed, because the link had no room. This plan measures the middle ground. It inserts a bounded number of CPU lanes
per layer, through the device-shaped mechanism, costed by the three existing bounds.

The insert budget is per layer (P lanes), not taken from the layer's link slack. When the CPU is fast it takes every
lane, and the layer lasts less than one row's copy (1 ms). The idle link a background copy can use is mostly outside
the layer: GPU compute and NVMe waits. That is exactly what `split_costs`' `optimistic` bound credits and its
`amortised` bound does not.

## Global Constraints

- **Running code:**
  - Code runs on divix01 only from a pushed commit, in a private worktree `/data/models/slang/nvfp4-work/wt-cpu-insert`.
  - Use `PYTHONPATH=$PWD/python` and `/data/models/slang/.venv/bin/python`. `cpu_expert_sim` imports sglang's
    `split_table`, so a missing `PYTHONPATH` silently runs another tree's copy.
- **CPU jobs:** run under `taskset -c 0-63` with `OMP_NUM_THREADS` capped. No GPU, and no server is started or
  stopped.
- **Pytest status:** read it from `${PIPESTATUS[0]}`, never from a pipe's last command.
- **Git:**
  - Push only `dsv41-cpu-insert`.
  - No amend, rebase or stash.
  - Commit trailer: `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` and
    `Claude-Session: https://claude.ai/code/session_01Y3EXKJdMpP7nwX3qcBi8Vq`.
- **Defaults stay put:** every existing policy, and `graph_forward` without `cpu_insert`, must replay bit-identically.
  The existing tests in `test/registered/unit/kernels/test_cpu_expert_sim.py` stay green, unchanged.
- **Comments:** follow `.claude/rules/comment-style.md`. One or two lines, and only facts the code cannot show.
- **Trace:** `divix01:/data/models/slang/nvfp4-work/direct-two-phase-tests/hot-cache-policy/router-capture/stages.jsonl`
  (6,153 decode tokens, 40 layers, top-6, 384 experts).

## Review Focus

1. **A FILLING slot that never resolves.** The fill's expert may be inserted by a normal lane before the fill lands.
   The slot must then go FREE and the fill count as dropped, never leaving two slots holding one expert. Task 1 pins
   it.
2. **A FILLING slot reused as a victim**, by the next forward's shortlist or by a deferred insert's spare entries. It
   must be excluded from both. Task 1 pins it.
3. **An expert both landing and filling in one commit.** It is routed again as a CPU lane in the forward its fill
   lands, so it must not register a second fill. Task 1 pins it.
4. **The insert count exceeding the layer's CPU lanes,** or inserting when the split gives the CPU nothing. The hook
   only ever sees the CPU lanes. Task 2 pins it.
5. **Unchanged defaults.** `INSERT_POLICIES` grows, so the default `insert_policies` report gains rows. The existing
   rows must not change. Three things pin it:
   - Task 2 checks that no pre-existing policy issues a CPU insert.
   - The unchanged pre-existing tests stay green.
   - Task 4's sanity check must reproduce §30.3's 75.94 ms and 0.672 hit rate.

---

### Task 1: The FILLING slot in `DirectInsertReplay`

**Files:**
- Modify: `scripts/dsv41/tier_sim.py` (`DirectInsertReplay.__init__`, `_rank`, `graph_forward`)
- Test: `test/registered/unit/kernels/test_cpu_expert_sim.py`

**Interfaces:**
- Produces:
  - `tier_sim.FILLING = -2`, the slot value of an insert whose bytes are in flight.
  - `DirectInsertReplay.graph_forward(..., cpu_insert: Optional[Callable[[int, list[int]], list[int]]] = None)`.
    The hook is called with `(layer, cpu_lanes_in_plan_order)` and returns the CPU lanes to insert.
  - Counters: `cpu_insert_last: dict[int, int]` (fills issued this forward, per layer), `cpu_insert_issued: int`,
    `cpu_inserted: int` (landed) and `cpu_insert_dropped: int`.

- [ ] **Step 1: Write the failing tests.** Append to `test/registered/unit/kernels/test_cpu_expert_sim.py`:

```python
def _insert_every_cpu_lane(layer, lanes):
    return lanes


def test_a_cpu_insert_frees_its_victim_now_and_maps_the_expert_a_forward_later():
    # Shortlist [expert 2, 1, 0]: lane 0 (expert 5, CPU) takes expert 2's slot as a fill, lane 1 (6) expert 1's.
    sim = tier_sim.DirectInsertReplay({0: [0, 1, 2]}, {0: 3}, 8, miss_rows=3)
    sim.graph_forward({0: [5, 6]}, cpu_lanes=lambda layer, misses: {misses[0]}, cpu_insert=_insert_every_cpu_lane)
    assert sim.resident(0) == {0, 6}
    assert sim.slots[0][2] == tier_sim.FILLING
    assert sim.cpu_insert_last == {0: 1} and sim.cpu_insert_issued == 1 and sim.cpu_inserted == 0
    assert 2 not in sim._rank(0)  # a FILLING slot is never shortlisted
    sim.graph_forward({0: [0]})
    assert sim.resident(0) == {0, 5, 6} and sim.cpu_inserted == 1
    assert tier_sim.FILLING not in sim.slots[0]


def test_a_fill_whose_expert_was_inserted_elsewhere_frees_its_slot():
    sim = tier_sim.DirectInsertReplay({0: [0, 1, 2]}, {0: 3}, 8, miss_rows=3)
    sim.graph_forward({0: [5, 6]}, cpu_lanes=lambda layer, misses: {misses[0]}, cpu_insert=_insert_every_cpu_lane)
    sim.graph_forward({0: [5]})  # 5 misses (its slot is FILLING) and a normal lane inserts it elsewhere
    assert sim.slots[0].count(5) == 1
    assert sim.slots[0][2] == -1 and sim.cpu_insert_dropped == 1 and sim.cpu_inserted == 0


def test_a_landing_expert_routed_again_on_the_cpu_does_not_fill_a_second_slot():
    sim = tier_sim.DirectInsertReplay({0: [0, 1, 2]}, {0: 3}, 8, miss_rows=3)
    on_cpu = lambda layer, misses: set(misses)  # noqa: E731
    sim.graph_forward({0: [5]}, cpu_lanes=on_cpu, cpu_insert=_insert_every_cpu_lane)
    sim.graph_forward({0: [5]}, cpu_lanes=on_cpu, cpu_insert=_insert_every_cpu_lane)
    assert sim.slots[0].count(5) == 1 and tier_sim.FILLING not in sim.slots[0]
    assert sim.cpu_inserted == 1 and sim.cpu_insert_issued == 1


def test_deferred_inserts_never_take_a_filling_slot():
    # Same forward: the fill holds expert 2's old slot, so the deferred 7 must take another spare entry.
    sim = tier_sim.DirectInsertReplay({0: [0, 1, 2]}, {0: 3}, 8, miss_rows=3)
    sim.graph_forward(
        {0: [5]}, cpu_lanes=lambda layer, misses: set(misses), cpu_insert=_insert_every_cpu_lane, deferred={0: [7]}
    )
    assert sim.slots[0][2] == tier_sim.FILLING and 7 in sim.resident(0)
```

- [ ] **Step 2: Commit, push, set up the divix01 worktree, and run the tests to see them fail.**

```bash
cd /Users/dnikolaidis/Desktop/divix/sglang-nvfp4-cpu-insert
git add test/registered/unit/kernels/test_cpu_expert_sim.py
git commit -m "test(cpu-expert-sim): a CPU lane's insert fills its victim and lands a forward later (failing)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Y3EXKJdMpP7nwX3qcBi8Vq"
git push -u origin dsv41-cpu-insert
ssh divix01 'git -C /data/models/slang/sglang fetch origin \
  && git -C /data/models/slang/sglang worktree add --detach /data/models/slang/nvfp4-work/wt-cpu-insert origin/dsv41-cpu-insert \
  && cd /data/models/slang/nvfp4-work/wt-cpu-insert && git log -1 --oneline \
  && PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 taskset -c 0-63 /data/models/slang/.venv/bin/python -m pytest \
     test/registered/unit/kernels/test_cpu_expert_sim.py -q -p no:randomly 2>&1 | tail -6; echo "EXIT=${PIPESTATUS[0]}"'
```

Expected: `EXIT=1`. The four new tests fail with `TypeError: ... unexpected keyword argument 'cpu_insert'` or
`AttributeError: module 'tier_sim' has no attribute 'FILLING'`. Every pre-existing test passes.

- [ ] **Step 3: Implement.** In `scripts/dsv41/tier_sim.py`, add a module constant just above `class DirectInsertReplay:`:

```python
# A slot whose insert's bytes are in flight: neither mapped nor a victim until the next commit of its layer.
FILLING = -2
```

In `DirectInsertReplay.__init__`, after `self.deferred_last: dict[int, int] = {}`:

```python
        self.filling: list[dict[int, int]] = [{} for _ in range(layers)]  # per row: slot -> expert in flight
        self.cpu_insert_last: dict[int, int] = {}
        self.cpu_insert_issued = 0
        self.cpu_inserted = 0
        self.cpu_insert_dropped = 0
```

In `_rank`, a free slot is `-1` only:

```python
        free = [slot for slot, expert in enumerate(slots) if expert == -1]
```

In `graph_forward`:
- Add the keyword `cpu_insert: Optional[Callable[[int, list[int]], list[int]]] = None` after `lane_order`.
- Extend the docstring's last sentence with: ``cpu_insert(layer, cpu_lanes)`` picks CPU lanes whose rows are also
  copied off the critical path. The victim is freed and left FILLING now, and the expert is mapped at this layer's
  next graph commit, unless it was inserted elsewhere first.
- Reset `self.cpu_insert_last = {}` next to `self.deferred_last = {}`.
- Replace the per-layer body from `usable = ...` through the `for expert, slot in live:` loop with:

```python
            filling = self.filling[row]
            usable = [slot for slot in self.shortlist[row] if slot not in hits and slot not in filling]
            on_cpu = cpu_lanes(layer, missing) if cpu_lanes is not None and missing else set()
            # Lane j's destination is usable[j] whether or not it is live (direct_gather_destinations_kernel).
            pairs = list(zip(missing, usable))
            live = [(expert, slot) for expert, slot in pairs if expert not in on_cpu]
            fills: list[tuple[int, int]] = []
            if cpu_insert is not None and on_cpu:
                cpu_pairs = [(expert, slot) for expert, slot in pairs if expert in on_cpu]
                wanted = set(cpu_insert(layer, [expert for expert, _ in cpu_pairs]))
                fills = [(expert, slot) for expert, slot in cpu_pairs if expert in wanted]
            if deferred is not None and deferred.get(layer):
                taken = {slot for _, slot in live} | {slot for _, slot in fills}
                inserted = {expert for expert, _ in live}
                spare = [
                    slot
                    for slot in usable
                    if slot not in taken and not (deferred_unrouted and slots[slot] >= 0 and self.routed[row, slots[slot]])
                ]
                late = [e for e in dict.fromkeys(deferred[layer]) if where[e] < 0 and e not in inserted]
                live += list(zip(late, spare))
                self.deferred_last[layer] = min(len(late), len(spare))
                self.deferred_inserted += self.deferred_last[layer]
                self.deferred_dropped += max(len(late) - len(spare), 0)
            for expert, slot in live:
                old = slots[slot]
                if old >= 0:
                    where[old] = -1
                slots[slot] = expert
                where[expert] = slot
            # The previous forward's fills land at this commit; one whose expert a lane inserted meanwhile frees its slot.
            for slot, expert in list(filling.items()):
                del filling[slot]
                if where[expert] >= 0:
                    slots[slot] = -1
                    self.cpu_insert_dropped += 1
                else:
                    slots[slot] = expert
                    where[expert] = slot
                    self.cpu_inserted += 1
            issued = 0
            for expert, slot in fills:
                if where[expert] >= 0:
                    continue
                old = slots[slot]
                if old >= 0:
                    where[old] = -1
                slots[slot] = FILLING
                filling[slot] = expert
                issued += 1
            self.cpu_insert_last[layer] = issued
            self.cpu_insert_issued += issued
```

The lines after the loop (`self.truncated += ...` and `misses[layer] = len(missing)`) stay as they are.

- [ ] **Step 4: Run the tests to see them pass**

```bash
git add scripts/dsv41/tier_sim.py
git commit -m "feat(tier-sim): a CPU lane's insert frees its victim as FILLING and lands at the next commit

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Y3EXKJdMpP7nwX3qcBi8Vq"
git push origin dsv41-cpu-insert
ssh divix01 'cd /data/models/slang/nvfp4-work/wt-cpu-insert && git fetch -q origin && git checkout -q --detach origin/dsv41-cpu-insert \
  && PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 taskset -c 0-63 /data/models/slang/.venv/bin/python -m pytest \
     test/registered/unit/kernels/test_cpu_expert_sim.py test/manual/dsv41/test_tier_sim.py -q -p no:randomly 2>&1 | tail -3; echo "EXIT=${PIPESTATUS[0]}"'
```

Expected: `EXIT=0`. All tests pass, the pre-existing ones unchanged, including `test_tier_sim.py`.

---

### Task 2: Insert policies in `cpu_expert_sim.py`

**Files:**
- Modify: `scripts/dsv41/cpu_expert_sim.py` (`INSERT_POLICIES`, `replay_nm`)
- Test: `test/registered/unit/kernels/test_cpu_expert_sim.py`

**Interfaces:**
- Consumes: `graph_forward(..., cpu_insert=...)`, `cpu_insert_last`, `cpu_insert_issued`, `cpu_inserted` and
  `cpu_insert_dropped` (Task 1).
- Produces:
  - Policies `cpu_insert_p1`, `cpu_insert_p2` and `cpu_insert_all`. Each is the merged policy (`tail`/`desc`) plus
    `insert_per_layer` P.
  - `replay_nm` returns `b` including the issued CPU inserts, and its `residency` gains
    `cpu_insert_issued_per_token`, `cpu_insert_rows_per_token` (landed) and `cpu_insert_dropped_per_token`.

- [ ] **Step 1: Write the failing tests.** Append:

```python
SPLIT = [0, 1, 1, 2, 3, 3, 4]


def test_cpu_insert_p1_lands_the_cpu_ram_hit_a_forward_later_and_costs_one_background_row():
    loaded = _ram_hit_loaded()
    merged = replay_nm(loaded, ram_rows=4, num_experts=8, policy="cpu_by_score_desc_tail", split=SPLIT)
    assert merged["n"].tolist() == [[0], [0], [1], [1], [1]]  # the CPU lane never enters VRAM
    out = replay_nm(loaded, ram_rows=4, num_experts=8, policy="cpu_insert_p1", split=SPLIT)
    # Forward 3: expert 2 on the CPU fills the only slot (3 evicted). Forward 4: still FILLING, so a RAM hit again;
    # it lands at forward 4's commit and hits at forward 5.
    assert out["n"].tolist() == [[0], [0], [1], [1], [0]]
    assert out["b"].tolist() == [[0], [0], [1], [0], [0]]
    res = out["residency"]
    assert res["cpu_insert_issued_per_token"] == pytest.approx(1 / 5)
    assert res["cpu_insert_rows_per_token"] == pytest.approx(1 / 5)
    assert res["cpu_insert_dropped_per_token"] == 0.0
    assert res["hot_hit_rate"] == pytest.approx(1 / 5)


def test_no_cpu_insert_without_a_cpu_lane():
    out = replay_nm(_ram_hit_loaded(), ram_rows=4, num_experts=8, policy="cpu_insert_all", split=[0] * 7)
    assert out["residency"]["cpu_insert_issued_per_token"] == 0.0
    assert out["b"].sum() == 0


def test_cpu_insert_policies_are_the_merged_order_with_a_per_layer_cap():
    from cpu_expert_sim import policy_config

    for name, p in (("cpu_insert_p1", 1), ("cpu_insert_p2", 2), ("cpu_insert_all", MAX_ROUTES)):
        config = policy_config(name)
        assert config["choice"] == "tail" and config["sort"] == "desc" and config["insert_per_layer"] == p


def test_the_existing_policies_never_insert_a_cpu_lane():
    from cpu_expert_sim import INSERT_POLICIES

    loaded = _ram_hit_loaded()
    for name in (n for n in INSERT_POLICIES if not n.startswith("cpu_insert_")):
        res = replay_nm(loaded, ram_rows=4, num_experts=8, policy=name, split=SPLIT)["residency"]
        assert res["cpu_insert_issued_per_token"] == 0.0 and res["cpu_insert_rows_per_token"] == 0.0, name
```

- [ ] **Step 2: Commit, push and run to see them fail**

```bash
git add test/registered/unit/kernels/test_cpu_expert_sim.py
git commit -m "test(cpu-expert-sim): head-most CPU lanes per layer are inserted off the critical path (failing)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Y3EXKJdMpP7nwX3qcBi8Vq"
git push origin dsv41-cpu-insert
ssh divix01 'cd /data/models/slang/nvfp4-work/wt-cpu-insert && git fetch -q origin && git checkout -q --detach origin/dsv41-cpu-insert \
  && PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 taskset -c 0-63 /data/models/slang/.venv/bin/python -m pytest \
     test/registered/unit/kernels/test_cpu_expert_sim.py -q -p no:randomly 2>&1 | tail -6; echo "EXIT=${PIPESTATUS[0]}"'
```

Expected: `EXIT=1`. The new tests fail with `ValueError: unknown insertion policy 'cpu_insert_p1'` or
`KeyError: 'cpu_insert_issued_per_token'`.

- [ ] **Step 3: Implement.** In `scripts/dsv41/cpu_expert_sim.py`, add to `INSERT_POLICIES` after
  `"cpu_by_score_desc_tail"`:

```python
    # The merged policy, and the head-most (highest-scored) P CPU lanes of each layer also copied into their own
    # victim slot off the critical path (tier_sim FILLING): mapped at the layer's next commit.
    "cpu_insert_p1": {"choice": "tail", "sort": "desc", "insert_per_layer": 1},
    "cpu_insert_p2": {"choice": "tail", "sort": "desc", "insert_per_layer": 2},
    "cpu_insert_all": {"choice": "tail", "sort": "desc", "insert_per_layer": MAX_ROUTES},
```

In `replay_nm`, just before `simulated = sim.graph_forward(`:

```python
            per_layer = config.get("insert_per_layer", 0)
            insert_hook = (lambda layer, lanes: lanes[:per_layer]) if decode and per_layer else None
```

Pass `cpu_insert=insert_hook,` to `sim.graph_forward(...)` after `lane_order=...`. In the decode bookkeeping, replace
the `b_tok.append(...)` line with:

```python
                b_tok.append([
                    sim.deferred_last.get(layer, 0) + promoted.get(layer, 0) + sim.cpu_insert_last.get(layer, 0)
                    for layer in forward["routes"]
                ])
```

In the returned `residency` dict, after `"promoted_rows_per_token"`:

```python
            "cpu_insert_issued_per_token": sim.cpu_insert_issued / max(tokens, 1),
            "cpu_insert_rows_per_token": sim.cpu_inserted / max(tokens, 1),
            "cpu_insert_dropped_per_token": sim.cpu_insert_dropped / max(tokens, 1),
```

Extend the module docstring's "Residency" bullet with one sentence: ``cpu_insert_p<P>`` / ``_all`` add, to the
merged policy, a copy of the head-most P CPU lanes per layer into their own victim slot (tier_sim FILLING), costed
as background rows ``b``.

- [ ] **Step 4: Run to see them pass**

```bash
git add scripts/dsv41/cpu_expert_sim.py
git commit -m "feat(cpu-expert-sim): cpu_insert_p1/p2/all policies, the CPU lanes' inserts costed as background rows

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Y3EXKJdMpP7nwX3qcBi8Vq"
git push origin dsv41-cpu-insert
ssh divix01 'cd /data/models/slang/nvfp4-work/wt-cpu-insert && git fetch -q origin && git checkout -q --detach origin/dsv41-cpu-insert \
  && PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 taskset -c 0-63 /data/models/slang/.venv/bin/python -m pytest \
     test/registered/unit/kernels/test_cpu_expert_sim.py test/manual/dsv41/test_tier_sim.py -q -p no:randomly 2>&1 | tail -3; echo "EXIT=${PIPESTATUS[0]}"'
```

Expected: `EXIT=0`.

---

### Task 3: The `--cpu-scale` sweep

**Files:**
- Modify: `scripts/dsv41/cpu_expert_sim.py` (new `CPU_INSERT_POLICIES`, `cpu_insert_results`, a `summary` block, a
  `--cpu-scale` flag, wiring in `main`)
- Test: `test/registered/unit/kernels/test_cpu_expert_sim.py`

**Interfaces:**
- Consumes: the Task 2 policies and residency keys; `split_table` and `split_costs`.
- Produces: `cpu_insert_results(loaded, args) -> list[dict]`. There is one row per (scale, policy) with keys
  `cpu_scale`, `policy`, `split`, `hot_hit_rate`, `cpu_lanes_per_token`, `cpu_insert_issued_per_token`,
  `cpu_insert_rows_per_token`, `nvme_reads_per_token`, and `ms_per_token_{uncosted,worst,amortised,optimistic}`.
  The report key is `cpu_insert`.

- [ ] **Step 1: Write the failing test.** Add `cpu_insert_results` to the `from cpu_expert_sim import (...)` list,
  then append:

```python
def test_cpu_insert_sweep_scales_the_split_and_costs_the_inserts():
    args = argparse.Namespace(
        ram_rows=4, num_experts=8, no_initial_from_log=False, numa_mb="0:1", cpu_node=0, measured_c_cpu=0.63,
        split_c_cpu=0.52, c_link=1.0, handoff=0.02, nvme_ms=[1.5], gpu_ms=14.0, cpu_scale=[1.0, 4.0],
    )
    rows = cpu_insert_results(_ram_hit_loaded(), args)
    assert [(r["cpu_scale"], r["policy"]) for r in rows] == [
        (s, p) for s in (1.0, 4.0)
        for p in ("insert_all", "cpu_by_score_desc_tail", "cpu_insert_p1", "cpu_insert_p2", "cpu_insert_all")
    ]
    by = {(r["cpu_scale"], r["policy"]): r for r in rows}
    assert all(a >= b for a, b in zip(by[4.0, "insert_all"]["split"], by[1.0, "insert_all"]["split"]))
    for r in rows:
        assert r["ms_per_token_uncosted"] <= r["ms_per_token_amortised"] <= r["ms_per_token_worst"]
        assert r["ms_per_token_optimistic"] <= r["ms_per_token_amortised"]
    p1 = by[1.0, "cpu_insert_p1"]
    assert p1["cpu_insert_issued_per_token"] == pytest.approx(1 / 5)
    assert p1["hot_hit_rate"] > by[1.0, "cpu_by_score_desc_tail"]["hot_hit_rate"]
    # A faster CPU makes the same CPU-lane work cheaper.
    assert by[4.0, "cpu_by_score_desc_tail"]["ms_per_token_uncosted"] < by[1.0, "cpu_by_score_desc_tail"]["ms_per_token_uncosted"]
```

- [ ] **Step 2: Commit, push and run to see it fail**

```bash
git add test/registered/unit/kernels/test_cpu_expert_sim.py
git commit -m "test(cpu-expert-sim): a CPU speed-up sweep over the insert policies (failing)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Y3EXKJdMpP7nwX3qcBi8Vq"
git push origin dsv41-cpu-insert
ssh divix01 'cd /data/models/slang/nvfp4-work/wt-cpu-insert && git fetch -q origin && git checkout -q --detach origin/dsv41-cpu-insert \
  && PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 taskset -c 0-63 /data/models/slang/.venv/bin/python -m pytest \
     test/registered/unit/kernels/test_cpu_expert_sim.py -q -p no:randomly 2>&1 | tail -4; echo "EXIT=${PIPESTATUS[0]}"'
```

Expected: `EXIT=2`, a collection error: `ImportError: cannot import name 'cpu_insert_results'`.

- [ ] **Step 3: Implement.** In `scripts/dsv41/cpu_expert_sim.py`, after `staging_results`:

```python
CPU_INSERT_POLICIES = ("insert_all", STAGING_MERGED, "cpu_insert_p1", "cpu_insert_p2", "cpu_insert_all")


def cpu_insert_results(loaded: dict, args) -> list[dict]:
    """Per CPU speed-up s in ``--cpu-scale``: the split and flat c_cpu at 1/s, each insert policy's own replay, and
    ms/token with its background rows costed by split_costs. ``insert_all`` inserts CPU lanes uncosted (upper bound)."""
    rows = []
    for scale in args.cpu_scale:
        split = split_table(MAX_ROUTES, args.split_c_cpu / scale, args.c_link, args.handoff).tolist()
        model = CostModel(
            [args.measured_c_cpu / scale] * len(CALIBRATED_KS), c_link=args.c_link, handoff=args.handoff,
            nvme_ms=args.nvme_ms[0], gpu_ms=args.gpu_ms,
        )
        for policy in CPU_INSERT_POLICIES:
            nm = replay_nm(
                loaded, args.ram_rows, args.num_experts, not args.no_initial_from_log, policy, split,
                numa_mb=args.numa_mb, cpu_node=args.cpu_node,
            )
            res, tokens = nm["residency"], nm["validation"]["decode_tokens"]
            cost = split_costs(nm["n"], nm["m"], model, split, nm["b"])
            rows.append({
                "cpu_scale": scale,
                "policy": policy,
                "split": split,
                "hot_hit_rate": res["hot_hit_rate"],
                "cpu_lanes_per_token": res["cpu_lanes_per_token"],
                "cpu_insert_issued_per_token": res["cpu_insert_issued_per_token"],
                "cpu_insert_rows_per_token": res["cpu_insert_rows_per_token"],
                "nvme_reads_per_token": float(nm["m"].sum()) / max(tokens, 1),
                **{f"ms_per_token_{name}": value for name, value in cost.items()},
            })
    return rows
```

In `summary`, before the final `lines.append(f"\ngate ...")`:

```python
    if report.get("cpu_insert"):
        params = report["params"]
        lines.append(
            f"\nCPU speed-up sweep (flat c_cpu {params['measured_c_cpu']}/s, split from {params['split_c_cpu']}/s, "
            f"nvme {params['nvme_ms'][0]} ms/miss; ms/token uncosted | worst | amortised | optimistic, tok/s at optimistic)"
        )
        lines.append(
            f"{'s':>4s} {'policy':24s} {'hot hit':>7s} {'cpu/tok':>7s} {'ins/tok':>7s} {'NVMe/tok':>8s} "
            f"{'unc':>7s} {'worst':>7s} {'amort':>7s} {'opt':>7s} {'tok/s':>6s}"
        )
        for r in report["cpu_insert"]:
            lines.append(
                f"{r['cpu_scale']:4g} {r['policy']:24s} {r['hot_hit_rate']:7.3f} {r['cpu_lanes_per_token']:7.2f} "
                f"{r['cpu_insert_issued_per_token']:7.2f} {r['nvme_reads_per_token']:8.2f} "
                f"{r['ms_per_token_uncosted']:7.2f} {r['ms_per_token_worst']:7.2f} {r['ms_per_token_amortised']:7.2f} "
                f"{r['ms_per_token_optimistic']:7.2f} {1000.0 / r['ms_per_token_optimistic']:6.2f}"
            )
```

In `main`, add the flag after `--staging-reserve`:

```python
    p.add_argument(
        "--cpu-scale", type=lambda v: [float(x) for x in v.split(",")], default=None,
        help="comma-separated CPU speed-ups to sweep the insert policies at (cpu_insert_results), e.g. 1,2,4,8",
    )
```

Add `"split_c_cpu": args.split_c_cpu,` to `report["params"]`. Before `if args.out:`, add:

```python
    if args.cpu_scale:
        report["cpu_insert"] = cpu_insert_results(loaded, args)
```

- [ ] **Step 4: Run to see it pass**

```bash
git add scripts/dsv41/cpu_expert_sim.py
git commit -m "feat(cpu-expert-sim): --cpu-scale sweeps the insert policies at faster CPU experts

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Y3EXKJdMpP7nwX3qcBi8Vq"
git push origin dsv41-cpu-insert
ssh divix01 'cd /data/models/slang/nvfp4-work/wt-cpu-insert && git fetch -q origin && git checkout -q --detach origin/dsv41-cpu-insert \
  && PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 taskset -c 0-63 /data/models/slang/.venv/bin/python -m pytest \
     test/registered/unit/kernels/test_cpu_expert_sim.py test/manual/dsv41/test_tier_sim.py -q -p no:randomly 2>&1 | tail -3; echo "EXIT=${PIPESTATUS[0]}"'
```

Expected: `EXIT=0`.

---

### Task 4: Run the gate on the router capture, decide, and record

**Files:**
- Modify: `docs/superpowers/plans/2026-10-03-dsv41-cpu-expert-insert.md` (Results)
- Modify: `DSV41_REFERENCE.md` (one paragraph appended to §30.3)
- Output (divix01, not committed): `cc-expert-prediction/analysis/dsv41-cpu-insert/`

- [ ] **Step 1: Run the sweep** (CPU only, single process)

```bash
ssh divix01 'cd /data/models/slang/nvfp4-work/wt-cpu-insert && git log -1 --oneline \
  && OUT=/data/models/slang/nvfp4-work/cc-expert-prediction/analysis/dsv41-cpu-insert && mkdir -p $OUT \
  && PYTHONPATH=$PWD/python OMP_NUM_THREADS=4 taskset -c 0-63 /data/models/slang/.venv/bin/python scripts/dsv41/cpu_expert_sim.py \
     /data/models/slang/nvfp4-work/direct-two-phase-tests/hot-cache-policy/router-capture/stages.jsonl \
     --no-insert-policies --nvme-ms 1.5 --cpu-scale 1,2,4,8 --out $OUT/cpu-insert.json > $OUT/cpu-insert.txt 2>&1; \
     echo "EXIT=$?"; sed -n "/CPU speed-up sweep/,/^gate/p" $OUT/cpu-insert.txt'
```

Expected: `EXIT=0` and a 20-row table (4 scales × 5 policies).

**Sanity check, first:** at `s = 1`, `cpu_by_score_desc_tail`'s uncosted ms/token must be 75.94 ± 0.2 (§30.3's merged
row at the same defaults), and `insert_all`'s hot hit 0.672 ± 0.005. If either is off, the replay changed, so stop
and find out why before reading anything else.

- [ ] **Step 2: Decide.** Let `M(s)` be `cpu_by_score_desc_tail`'s uncosted ms/token, today's behaviour. For each
  `cpu_insert_p*` policy, let `O(s)` and `A(s)` be its optimistic and amortised ms/token. The verdict is pre-registered:
  - **Go** (write the device plan below) if some policy at `s = 2` or `s = 4` has `O(s) ≤ 0.95 · M(s)` and
    `A(s) ≤ 1.01 · M(s)`. The insert pays even when the copies get only the layers' own slack.
  - **Measure first** if `O(s) ≤ 0.95 · M(s)` but `A(s) > 1.01 · M(s)`. The win depends on the link being idle during
    GPU compute and NVMe waits. Capture one production decode trace with PCIe metrics (CLAUDE.md, Nsight section) and
    read the link's idle share per step before writing the device plan.
  - **Stop** otherwise. Record the numbers; CPU-computed experts stay uninserted.
  - Separately, at `s = 1`, record whether the chosen policy's `A(1) ≤ 1.005 · M(1)`. If not, the device change must
    take P from the startup calibration, with P = 0 at today's speed, rather than a fixed P.

- [ ] **Step 3: Record.**
  - Write the table, the sanity values, the verdict and the chosen P under "Results" below.
  - Append one paragraph to the end of `DSV41_REFERENCE.md` §30.3 (before `## 31.`). It gives the policy names, the
    sweep command and its output directory, the sanity values, the verdict, and a pointer to this plan.

```bash
git add docs/superpowers/plans/2026-10-03-dsv41-cpu-expert-insert.md DSV41_REFERENCE.md
git commit -m "docs(dsv41): CPU-expert insert-on-miss replay gate, results and verdict

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Y3EXKJdMpP7nwX3qcBi8Vq"
git push origin dsv41-cpu-insert
```

---

## Next (not in this plan): the device change, if the gate says Go

This section is the design the replay models. It becomes its own plan, with exact code, only after Task 4's verdict.
The references are from a code audit on 2026-10-03; their line numbers are at `9361644c82`, which equals master for
these files.

- **Typing (device).**
  - Add a host-written `kInsert[9]` table next to `kSplit` in the lease block. The space check is
    `lease_layout.h:93`; `kSplit` is at `:71`.
  - In `type_lanes` (`lease_device.cuh:352-405`), give the head-most `insert[n]` CPU lanes a new kind,
    `kKindHitCpuInsert`. The CPU lanes are walked from the tail, so the head-most CPU lanes are the highest-scored.
  - Update `ram_slot_map.type_lanes` (the Python reference), the wire mirrors, and CW's `is_cpu_kind`
    (`row_copy_kernels.cuh:269-345`).
- **DMA (host).**
  - In `RamTier::submit_host_lanes` (`ram_tier.h:1476-1511`), submit a second, insert-only `CopyJob` after the
    record's job. It writes into the lane's existing `dst_slot`, which is already the VRAM victim slot (`:1454`).
  - The copy thread must not call `copy_completed` for that job, so it never gates CopyDone. It stores
    `InsertLanded[row] = G` with release ordering.
  - Issue it on a second, low-priority stream, chunked so that a critical copy waits at most one chunk behind it. The
    existing stream is FIFO (`copy_engine.h:578-596`), and a 1 ms insert ahead of the next layer's hit copies would
    delay them.
- **Commit (device).**
  - CC also writes the insert lanes.
  - `direct_commit_gather_kernel` (`direct_gather.cuh:95-150`) unmaps the victim's old expert and sets a new `_FILLING`
    state. Today only `_FREE = 0` and `_READY = 3` exist (`expert_residency_gpu.py:37-38`). It records
    `pending[row] = (slot, expert, G)`.
  - A later commit of the row maps the expert once `InsertLanded[row] ≥ G`, or frees the slot if the expert was
    inserted elsewhere. That is the behaviour Task 1 replays.
- **Guards.** The insert DMA outlives "one record in flight; CopyDone covers every reader" (§31). That is the riskiest
  point, because a missed guard corrupts output silently instead of failing stop.
  - A FILLING slot is never shortlisted: `_rank_victims` admits only FREE and READY (`expert_residency_gpu.py:839-842`).
  - A FILLING slot is a hazard in both gather-destination kernels, for a shortlist ranked before the eviction.
  - The source RAM slot stays `tier.filling` until the DMA lands. `take_victim_locked` already skips filling slots
    (`ram_tier.h:1354`).
  - `before_host_use` drains insert jobs (`CopyEngine::wait_idle`).
  - Teardown waits for, or quarantines, an in-flight insert.
- **Env.** `SGLANG_DSV41_CPU_EXPERTS_INSERT_PER_LAYER` (`EnvInt`, default 0) goes in the CPU-experts block of
  `environ.py` (`:1909-1935`). The gate requires `SGLANG_DSV41_CPU_EXPERTS=1`, modelled on `_check_slot_map`
  (`expert_stream_requirements_exl3.py:188-206`).
- **Tests to update:**
  - `test_dsv41_layer_fusion_gpu.py:226` (CPU lanes unmapped);
  - `test_exl3_cpu_lane_order_cuda.py`, `test_exl3_lease_kernels_cuda.py`, `test_exl3_slot_map_kernels_cuda.py`,
    `test_exl3_copy_engine_cuda.py`, `test_exl3_ram_miss_graph_gpu.py`;
  - `test/registered/unit/kernels/test_exl3_ram_miss_cpu_experts.py`, `test_ram_slot_map.py`,
    `test_exl3_ram_miss_device_args.py`.
- **Proof:** a served A/B on the production recipe, with the insert off and on at the chosen P, run with the faster
  CPU expert kernel the gate assumed. The gate's tok/s gain at that kernel's measured speed-up is the prediction to
  check.

## Results

(Filled in by Task 4.)

### Task 4: the gate (2026-10-03, divix01, `cad2f94ae6`)

Command: Task 4 Step 1. Output: `divix01:cc-expert-prediction/analysis/dsv41-cpu-insert/cpu-insert.{json,txt}`. EXIT=0.

**Sanity.** At s = 1 the merged policy gives 75.94 ms/token and insert_all's hot hit is 0.672. Both reproduce §30.3.

| s | policy | hot hit | CPU inserts/token | uncosted | amortised | optimistic (tok/s) |
|---|---|---|---|---|---|---|
| 1 | insert_all (bound) | 0.672 | – | 73.03 | 73.03 | 73.03 (13.69) |
| 1 | merged (today) | 0.643 | 0 | 75.94 | 75.94 | 75.94 (13.17) |
| 1 | cpu_insert_p1 | 0.647 | 33.1 | 75.51 | 99.13 | 76.15 (13.13) |
| 2 | insert_all (bound) | 0.672 | – | 59.11 | 59.11 | 59.11 (16.92) |
| 2 | merged (today) | 0.564 | 0 | 65.53 | 65.53 | 65.53 (15.26) |
| 2 | cpu_insert_p1 | 0.633 | 30.7 | 61.35 | 77.69 | 61.37 (16.30) |
| 2 | cpu_insert_p2 | 0.636 | 51.8 | 61.19 | 98.71 | 69.89 (14.31) |
| 4 | insert_all (bound) | 0.672 | – | 51.03 | 51.03 | 51.03 (19.60) |
| 4 | merged (today) | 0.187 | 0 | 66.33 | 66.33 | 66.33 (15.08) |
| 4 | cpu_insert_p1 | 0.625 | 29.8 | 52.41 | 72.13 | 52.67 (18.99) |
| 4 | cpu_insert_p2 | 0.632 | 50.6 | 52.21 | 92.86 | 63.61 (15.72) |
| 8 | merged (today) | 0.187 | 0 | 54.69 | 54.69 | 54.69 (18.28) |
| 8 | cpu_insert_p1 | 0.625 | 29.8 | 47.62 | 72.11 | 48.88 (20.46) |

`cpu_insert_all` is worse than `p2` at every s, and the full table is in the output file.

**What the numbers say:**
- **The merged policy collapses as the CPU gets faster.** At s = 4 its hot hit is 0.187, and it is slower than at s = 2 (66.33 against 65.53 ms).
- **`cpu_insert_p1` keeps hot hit near 0.63.** At s = 4 its optimistic cost is within 1.6 ms of the uncosted insert-all bound.
- **The copies need idle link outside the layers.** P1 issues about 30 copies a token, so about 30 ms of link. The layers' own slack holds only a fraction of that: amortised is 72 ms against optimistic 53 ms. GPU compute plus NVMe waits per token are about 31 ms, so P1 just fits the optimistic idle budget. P2 and all overflow it.

**Verdict: measure first.**
- At s = 2, P1 gives O = 61.37 against M = 65.53, a 0.936 ratio, under the 0.95 bar. But A = 77.69, well over 1.01 · M.
- At s = 4, O / M = 0.794, and A = 72.13 is again over 1.01 · M.
- The win therefore depends on the link really being idle for about 30 row-copies a token during GPU compute and NVMe waits. A production decode trace with PCIe metrics must show that before the device plan is written.
- At s = 1, P1 costs more even optimistically (76.15 against 75.94). So the device change must take P from the CPU speed, with P = 0 at today's kernel, not a fixed P.

**Chosen P:** 1, for s ≥ 2.
