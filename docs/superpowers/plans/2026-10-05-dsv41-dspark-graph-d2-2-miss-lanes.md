# DSpark graphed verify D2-2: a miss width below the route count, and an overflow flag

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A DIRECT graph gather of a verify serves at most W distinct misses per layer, with W ≤ 32 and below the
route count. The RAM-miss lease chain is built for W lanes. A gather whose misses do not all find a lane serves the
ones that do, keeps residency exact, and raises a device overflow flag that says the forward is not a verify result.

**Architecture:** D2-2 of four v2 plans. D2-1 (`2026-10-05-dsv41-dspark-graph-d2-1-multitoken-moe.md`) made the
planner and the in-graph MoE multi-token. D2-3 runs the graphed verify end to end and re-verifies on the flag; D2-4
adds multi-token CPU experts.

§33.3 item 2 is out of date. The wire is no longer 8 lanes: `LeaseLayout<NumLanes, NumNodes>` takes 1-32 lanes,
one JIT build per lane count (`0b82fc37c5`, `5813a147d5`, `7a77475a20`). What still assumes one token is the unit
the width counts. `graph_gather_rows` counts routes (36 for 6 tokens at top-6), and three places size from it:
- the lane build: `plan_gather_width(rows)` raises above 32;
- DIRECT's victim shortlist: `miss_rows` and `capacity ≥ 2 × miss_rows`;
- the attach check `graph_gather_rows > lanes`.

This plan adds a second width, `graph_miss_lanes` (W), set by `SGLANG_MOE_EXPERT_GRAPH_GATHER_MISS_LANES`.
- Routes keep their own width (≤ 64): planner, remap and protect list.
- Misses take W: lane build, staging, shortlist, commit.
- A capture-safe step after the DIRECT destinations sets the shared miss count (which the copy, record and commit
  read) to the lanes that found a victim, and flags the gather when that is fewer than its misses.
  - The live lanes are always a prefix (usable shortlist entries first, then `lane < count`), so `live.sum()` is the
    number served.
  - Residency stays exact: every copied row is committed, and nothing else.
  - Unserved misses' routes read a served lane's slot: a valid slot, the wrong expert. The flag marks the output.

**Tech Stack:** CUDA JIT kernels (`sglang.kernels.jit`, tvm-ffi), torch, pytest; GPU runs on divix01.

**Spec:** `DSV41_REFERENCE.md` §33.3 items 1 and 2 (width against VRAM; the record), §33.5 (union curve: 6:3 has
mean 21 and p99 32 distinct experts per layer, and only W ≤ 11 fits beside the hybrid draft), §33.6 (D2-1, what it
left to this plan).

## Global Constraints

- BS1 stays bit for bit what it is today. With `SGLANG_MOE_EXPERT_GRAPH_GATHER_MISS_LANES` unset (0), every layer's
  miss width equals its routes, the new clamp step is never called, and every buffer has today's shape.
- A miss width below the routes needs DIRECT residency (`SGLANG_MOE_HOT_INSERT_ON_MISS_STAGE=2`). Only DIRECT's gather
  knows which lanes found a slot; SCRATCH needs a scratch row per route.
- A miss width below the routes refuses CPU experts. CPU experts are one token (§33.3 item 5); D2-4 lifts this.
- W ≤ 32: the wire (`LeaseLayout`) and the DIRECT gather and commit kernels (one warp, a lane per shortlist entry).
  Routes ≤ 64: the dedup planner and route tables (D2-1).
- Everything on the graph path is capture-safe: no host read of a device value, and no allocation that depends on
  data. Host-side shape and attribute reads are allowed.
- The overflow flag is sticky and device-side. Nothing in this plan reads it on the host during serving; D2-3 owns
  that read and the re-verify.
- Env vars follow `.claude/skills/env-var-conventions/SKILL.md`: an `EnvInt` in `environ.py` beside
  `SGLANG_MOE_EXPERT_GRAPH_GATHER_SCRATCH_ROWS`, read through `envs.X.get()`. Tests use `.override()`.
- Do not edit `model_runner.py` (a frozen core file). `from_model` reads the new env var itself when its keyword is
  `None`, as it already does for `insert_on_miss`.
- divix01 protocol (`.claude/rules/divix01-run-protocol.md`):
  - Push the branch, then `git fetch` and `git checkout --detach origin/dsv41-dspark-graph` in
    `/data/models/slang/nvfp4-work/wt-dsv41-dspark-graph`.
  - Run with `PYTHONPATH=$PWD/python`, and print `sglang.__file__` first.
  - Read `${PIPESTATUS[0]}`, never a pipe's status.
  - CPU jobs run under `taskset -c 0-63`.
  - GPU work runs under `flock /data/models/slang/nvfp4-work/cc-gpu.lock taskset -c 32-63`. A job that also reads
    checkpoint rows takes `rowimg-disk.lock` first.
  - EXL3 GPU tests need `SGLANG_EXL3_SRC=/data/models/slang/nvfp4-work/exllamav3`.
- Every commit ends with:
  ```
  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01Y3EXKJdMpP7nwX3qcBi8Vq
  ```

The laptop has no torch. Every test runs on divix01. The two commands used below, written once (`<FILES>` and `<K>`
vary):

```bash
# CPU
git push -q origin dsv41-dspark-graph && ssh divix01 'cd /data/models/slang/nvfp4-work/wt-dsv41-dspark-graph \
  && git fetch -q origin && git checkout -q --detach origin/dsv41-dspark-graph \
  && export PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 \
  && taskset -c 0-63 /data/models/slang/.venv/bin/python -c "import sglang; print(sglang.__file__)" \
  && taskset -c 0-63 /data/models/slang/.venv/bin/python -m pytest <FILES> -q -p no:randomly -k "<K>" > /tmp/d22.log 2>&1; echo EXIT=${PIPESTATUS[0]}; tail -15 /tmp/d22.log'

# GPU
git push -q origin dsv41-dspark-graph && ssh divix01 'cd /data/models/slang/nvfp4-work/wt-dsv41-dspark-graph \
  && git fetch -q origin && git checkout -q --detach origin/dsv41-dspark-graph \
  && export PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 SGLANG_EXL3_SRC=/data/models/slang/nvfp4-work/exllamav3 \
  && /data/models/slang/.venv/bin/python -c "import sglang; print(sglang.__file__)" \
  && flock /data/models/slang/nvfp4-work/cc-gpu.lock taskset -c 32-63 /data/models/slang/.venv/bin/python -m pytest <FILES> -q -p no:randomly -k "<K>" > /tmp/d22.log 2>&1; echo EXIT=${PIPESTATUS[0]}; tail -15 /tmp/d22.log'
```

`<K>` = `not nothing_matches_this` selects every test in `<FILES>`.

## Review Focus

1. **Hits that empty the shortlist.** A verify whose hits disqualify shortlist entries, so fewer than its misses
   survive, even with misses ≤ W. Expected: served = surviving entries, flag set, residency exact, no copy into slot
   0. Task 3 pins it with a crafted shortlist.
2. **The commit and the copy after a clamp.** Expected: the post kernel, S, the copy wait and the commit all read the
   clamped count; `insertion_truncated` stays 0; every mapped expert's slot holds its own bytes. Task 3 (NVFP4,
   generic and layer-fusion kernels) and Task 4 (EXL3 lease chain) both check slot bytes after an overflow.
3. **A narrow gather that does not overflow is today's gather.** Expected: bit for bit the residency state of the
   same routes at W = routes. Task 3's twin test compares the two until the first overflow.
4. **Routes 33-64 through the fused DIRECT gather.** Every route's remap must be translated, not just the first 32
   (one warp). Task 1's parity cases at 36 and 64 routes.
5. **Unset is today.** No env var means W = routes everywhere, the clamp is never called, and the metrics snapshot
   has no new key. Task 2 checks the defaults; Task 3 checks that BS1 never calls the clamp.

---

### Task 1: The fused DIRECT gather kernel takes up to 64 routes

**Files:**
- Modify: `python/sglang/kernels/jit/csrc/moe/expert_residency/direct_gather.cuh` (`direct_gather_destinations_kernel`,
  `direct_gather_destinations_gpu`)
- Test: `test/manual/dsv41/test_dsv41_layer_fusion_gpu.py` (`_scenario`, `test_gather_and_commit_match_the_torch_chain`)
- Test: `test/manual/dsv41/test_layer_fusion_launcher_checks_gpu.py` (`GATHER_REFUSALS`)

**Interfaces:**
- Produces: `direct_gather_destinations(...)` accepts `topk_ids` and `remap` of 1-64 routes. The shortlist stays
  1-32. The launcher has separate messages: "the shortlist must hold 1-32 entries" and "the routes must hold 1-64
  entries".

The kernel is one warp. It already loops the hazard check over every route (`for (int i = 0; i < top_k; ++i)`). Only
the remap translation is per thread: `if (lane < top_k)`. A verify of 36 routes leaves routes 32-35 with the
planner's scratch-based remap. The launcher refuses that today; once it is admitted, the kernel must stride.

- [ ] **Step 1: Write the failing tests**

In `test_dsv41_layer_fusion_gpu.py`, `_scenario`, the padding of the planner's source rows must not go negative when
the routes outnumber the shortlist. Replace:

```python
    source_rows = torch.cat([flat[miss], flat[~miss], unrouted[: width - routes]])
```

with:

```python
    source_rows = torch.cat([flat[miss], flat[~miss], unrouted[: max(0, width - routes)]])
```

Add three cases to the parametrize list of `test_gather_and_commit_match_the_torch_chain`, after `(32, 32, 400, 64)`:

```python
        (8, 36, 256, 40),  # a 6-token verify at top-6 over an 8-lane shortlist
        (16, 64, 400, 64),
        (32, 64, 400, 64),
```

In `test_layer_fusion_launcher_checks_gpu.py`, `GATHER_REFUSALS`, replace the `"width_past_32"` and `"routes_past_32"`
entries with:

```python
    "width_past_32": (
        lambda: _gather_args(width=33),
        "the shortlist must hold 1-32 entries",
    ),
    "routes_past_64": (
        lambda: _gather_args(routes=65),
        "the routes must hold 1-64 entries",
    ),
```

- [ ] **Step 2: Run them and watch them fail**

Commit `test(direct-gather): the fused DIRECT gather over 33-64 routes (failing)`. Run the GPU command with `<FILES>` =
`test/manual/dsv41/test_dsv41_layer_fusion_gpu.py test/manual/dsv41/test_layer_fusion_launcher_checks_gpu.py` and
`<K>` = `gather_and_commit_match or gather_launcher_refuses`.
Expected: FAIL.
- The new parity cases raise `RuntimeError` with "the shortlist and the routes must hold 1-32 entries".
- Both refusal cases fail with "Regex pattern did not match".
- The old parity cases pass.

- [ ] **Step 3: Implement**

In `direct_gather.cuh`, after `constexpr int kDirectGatherWarp = 32;` add:

```cpp
// Routes a gather may carry: a verify's tokens x top_k, the dedup planner's bound (expert_route_plan.cuh).
constexpr int kDirectGatherMaxRoutes = 64;
```

Update the kernel's comment block. After "A remap entry at or past scratch_base is a miss lane's rank and becomes
that lane's destination." add: "Routes may outnumber the warp (a verify carries up to 64); every lane translates
routes lane, lane + 32, ...".

Replace the remap block at the end of `direct_gather_destinations_kernel`:

```cpp
  if (static_cast<int>(lane) < top_k) {
    const int64_t remap = static_cast<int64_t>(remap_in[lane]);
    int64_t rank = remap - scratch_base;
    rank = rank < 0 ? 0 : (rank > width - 1 ? width - 1 : rank);
    remap_out[lane] = static_cast<RemapOutT>(remap >= scratch_base ? destinations[rank] : remap);
  }
```

with:

```cpp
  for (int i = static_cast<int>(lane); i < top_k; i += kDirectGatherWarp) {
    const int64_t remap = static_cast<int64_t>(remap_in[i]);
    int64_t rank = remap - scratch_base;
    rank = rank < 0 ? 0 : (rank > width - 1 ? width - 1 : rank);
    remap_out[i] = static_cast<RemapOutT>(remap >= scratch_base ? destinations[rank] : remap);
  }
```

In `direct_gather_destinations_gpu`, replace the single check:

```cpp
  RuntimeCheck(
      0 < W_.unwrap() && W_.unwrap() <= kDirectGatherWarp && 0 < K_.unwrap() && K_.unwrap() <= kDirectGatherWarp,
      "the shortlist and the routes must hold 1-32 entries");
```

with:

```cpp
  RuntimeCheck(0 < W_.unwrap() && W_.unwrap() <= kDirectGatherWarp, "the shortlist must hold 1-32 entries");
  RuntimeCheck(0 < K_.unwrap() && K_.unwrap() <= kDirectGatherMaxRoutes, "the routes must hold 1-64 entries");
```

- [ ] **Step 4: Run the tests and watch them pass**

Run the GPU command with Step 2's `<FILES>` and `<K>` = `not nothing_matches_this`.
Expected: all pass, the pre-existing gather, commit, route-table and refusal cases included.

- [ ] **Step 5: Commit**

```bash
git add python/sglang/kernels/jit/csrc/moe/expert_residency/direct_gather.cuh
git commit -m "feat(direct-gather): the fused DIRECT gather translates up to 64 routes over its 32-entry shortlist" -m "<trailers>"
```

---

### Task 2: A miss width W, separate from the routes

**Files:**
- Modify: `python/sglang/srt/environ.py` (beside `SGLANG_MOE_EXPERT_GRAPH_GATHER_SCRATCH_ROWS`)
- Modify: `python/sglang/srt/layers/moe/expert_stream.py` (`__init__` gather state, `enable_graph_gather`, a new
  `graph_miss_width` property)
- Modify: `python/sglang/srt/layers/moe/expert_hot_cache.py` (`from_model`)
- Modify: `python/sglang/srt/layers/moe/expert_residency_gpu.py` (`_init_insert_on_miss`, `_init_insert_direct`,
  `gather_destinations`, `fused_gather_destinations`)
- Modify: `python/sglang/srt/layers/moe/exl3_ram_miss.py` (`Exl3RamMissService.attach`)
- Modify: `python/sglang/srt/layers/quantization/exl3/fused_moe.py` (`exl3_fused_moe_for`)
- Test: `test/registered/unit/layers/moe/test_expert_residency_gpu.py` (CUDA; `TestInsertOnMissDirect`)
- Test: `test/registered/unit/kernels/test_exl3_ram_miss_attach_lanes.py` (CPU)
- Test: `test/registered/unit/layers/quantization/test_exl3_fused_moe.py` (CPU)

**Interfaces:**
- Consumes: Task 1's 64-route gather kernel.
- Produces:
  - `envs.SGLANG_MOE_EXPERT_GRAPH_GATHER_MISS_LANES`, an `EnvInt(0)`.
  - `ExpertHotCacheManager.from_model(..., graph_gather_miss_lanes: int | None = None)`. `None` reads the env var.
    0 means one lane per route. A positive value is capped at each layer's routes.
  - `ExpertStreamer.enable_graph_gather(max_rows, scratch_destinations=True, miss_lanes=0)`, and attributes
    `graph_miss_lanes` (0 when not narrowed) and `graph_miss_width` (property: `graph_miss_lanes or
    graph_gather_rows`).
  - `GpuResidencyUpdater.miss_rows` is the miss width W. Its shortlist (`victims`, `victim_valid`) and lane buffers
    are `[layers, W]`. The capacity floor is `2 × W`.
  - `Exl3RamMissService.attach` compares the miss width, not the routes, against its lanes.

- [ ] **Step 1: Write the failing tests**

In `test_expert_residency_gpu.py`, class `TestInsertOnMissDirect`, update the regex of
`test_a_layer_too_small_to_guarantee_a_victim_is_refused` from `"twice its graph-gather rows"` to
`"twice its graph-gather miss lanes"`, and add:

```python
    # ----- a miss width below the routes (a verify) -----

    def test_a_narrow_miss_width_sizes_the_shortlist_not_the_routes(self):
        """Three verify tokens route 6 ids a layer; the gather serves 2 distinct misses. The routes keep their width,
        the shortlist and the lanes take the miss width, and DIRECT still holds no scratch."""
        manager = _manager(_model(), gpu=True, graph_gather_batch_size=3, graph_gather_miss_lanes=2, **DIRECT)
        updater = manager.gpu_residency
        for streamer in manager.streamers.values():
            self.assertEqual(streamer.graph_gather_rows, 3 * TOP_K)
            self.assertEqual(streamer.graph_miss_lanes, 2)
            self.assertEqual(streamer.graph_miss_width, 2)
            self.assertEqual(streamer.hot_cache.scratch_rows, 0)
        self.assertEqual(updater.miss_rows, 2)
        self.assertEqual(tuple(updater.victims.shape), (LAYERS, 2))

    def test_the_miss_width_defaults_to_one_lane_per_route_and_reads_the_env(self):
        from sglang.srt.environ import envs

        # Two tokens route 4 ids: one lane per route needs a floor of 8 slots a layer.
        budget = dict(budget_bytes=56 * LAYERS * 8, graph_gather_batch_size=2)
        wide = _manager(_model(), gpu=True, **budget, **DIRECT)
        capped = _manager(_model(), gpu=True, graph_gather_miss_lanes=99, **budget, **DIRECT)
        with envs.SGLANG_MOE_EXPERT_GRAPH_GATHER_MISS_LANES.override(3):
            narrow = _manager(_model(), gpu=True, **budget, **DIRECT)
        for manager, lanes, width in ((wide, 0, 4), (capped, 0, 4), (narrow, 3, 3)):
            for streamer in manager.streamers.values():
                self.assertEqual((streamer.graph_miss_lanes, streamer.graph_miss_width), (lanes, width))
            self.assertEqual(manager.gpu_residency.miss_rows, width)

    def test_a_narrow_miss_width_needs_direct(self):
        with self.assertRaisesRegex(ValueError, "SGLANG_MOE_HOT_INSERT_ON_MISS_STAGE=2"):
            _manager(_model(), gpu=True, graph_gather_batch_size=3, graph_gather_miss_lanes=2, **IOM)

    def test_a_negative_miss_width_is_refused(self):
        with self.assertRaisesRegex(ValueError, "miss lanes"):
            _manager(_model(), gpu=True, graph_gather_batch_size=3, graph_gather_miss_lanes=-1, **DIRECT)
```

In `test_exl3_ram_miss_attach_lanes.py`, after `test_a_gather_wider_than_the_built_lanes_is_refused_before_anything_is_built`:

```python
def test_a_verify_gather_attaches_by_its_miss_width_not_its_routes(tiers):
    """Six tokens of top-6 route 36 ids a layer, past the wire's 32; the gather serves 8 misses, so it builds 8 lanes."""
    service, streamers = tiers
    service.plan_gather_width(8)
    streamers[0].graph_miss_lanes = 8
    _attach(service, streamers[0], 36)
    assert service.lanes == 8 and service.routed_rows_per_step == 36


def test_cpu_experts_refuse_a_miss_width_below_the_routes(tiers):
    """CPU experts are one token: their lanes and partials cover one token's routes (D2-4 lifts this)."""
    service, streamers = tiers
    service.plan_gather_width(8)
    service.ensure_started()
    service.cpu_experts = SimpleNamespace(attach_device=lambda device_side: None)
    try:
        streamers[0].graph_miss_lanes = 8
        with pytest.raises(ValueError, match="CPU experts serve one token"):
            _attach(service, streamers[0], 36)
    finally:
        service.cpu_experts = None
```

In `test_exl3_fused_moe.py`, after `test_direct_fused_moe_covers_resident_slots_with_zero_scratch`:

```python
def test_direct_fused_moe_for_a_verify_needs_one_token_of_resident_slots_not_every_route(monkeypatch):
    """Six tokens route 36 ids into 16 resident slots: tokens share slots, and the gather flags what it cannot serve,
    so DIRECT needs a token's top_k slots, not a slot per route."""
    from types import SimpleNamespace

    from sglang.srt.layers.quantization.exl3 import fused_moe as module

    calls = []
    monkeypatch.setattr(module, "Exl3FusedMoE", lambda tensors, slots, **kw: calls.append((slots, kw)) or object())
    layer = torch.nn.Module()
    layer.top_k = 6
    streamer = _stub_streamer(SimpleNamespace(name="exl3_ram_miss"), graph_gather_rows=36, scratch_rows=0)
    streamer.hot_cache.capacity = 16
    streamer.hot_cache.device_residency = SimpleNamespace(insert_on_miss=True, insert_direct=True)
    streamer.hot_cache.device = torch.device("cpu")
    streamer.hot_cache.tensors = {"w13_suh": torch.zeros((16, 2, 8)), "w2_suh": torch.zeros((16, 1, 8))}
    module.exl3_fused_moe_for(layer, streamer)
    assert calls == [(16, calls[0][1])] and calls[0][1]["tokens"] == 6
```

- [ ] **Step 2: Run them and watch them fail**

Commit `test(graph-gather): a miss width below the routes (failing)`.

Run the CPU command with `<FILES>` =
`test/registered/unit/kernels/test_exl3_ram_miss_attach_lanes.py test/registered/unit/layers/quantization/test_exl3_fused_moe.py`
and `<K>` = `miss_width or verify`.
Expected: FAIL.
- The attach test raises "gathers up to 36 rows ... at most 8 lanes".
- The CPU-experts test fails on the same refusal, not "CPU experts serve one token".
- The fused-MoE test raises "exl3 DIRECT needs at least top_k resident slots per token, one per route (16 < 36)".

Run the GPU command with `<FILES>` = `test/registered/unit/layers/moe/test_expert_residency_gpu.py` and `<K>` =
`miss_width or too_small`.
Expected: FAIL.
- The new tests fail with `TypeError: ... unexpected keyword argument 'graph_gather_miss_lanes'`.
- The `"twice its graph-gather miss lanes"` regex does not match.

- [ ] **Step 3: The env var and the streamer**

In `environ.py`, after `SGLANG_MOE_EXPERT_GRAPH_GATHER_SCRATCH_ROWS = EnvInt(0)`:

```python
    # Speculative decoding with DIRECT residency (SGLANG_MOE_HOT_INSERT_ON_MISS_STAGE=2) only: each layer's graph gather
    # serves at most this many distinct misses (the RAM-miss lanes and the victim shortlist), below one per verify
    # route. A gather with more serves the lanes that find a victim and sets the residency's overflow flag; its output
    # is then not a verify result. 0 keeps one lane per route.
    SGLANG_MOE_EXPERT_GRAPH_GATHER_MISS_LANES = EnvInt(0)
```

In `expert_stream.py`, `ExpertStreamer.__init__`, after `self.graph_gather_rows = 0` add:

```python
        # Distinct misses a graph gather serves when below its routes (a verify); 0: one per route.
        self.graph_miss_lanes = 0
```

Change `enable_graph_gather`'s signature to
`def enable_graph_gather(self, max_rows: int, scratch_destinations: bool = True, miss_lanes: int = 0) -> None:`.
Add to its docstring:
"``miss_lanes`` > 0 serves at most that many distinct misses, below ``max_rows`` routes (a verify). It needs
``scratch_destinations=False``: only DIRECT's victim slots know which misses found a row (see
``GpuResidencyUpdater.clamp_gather_misses``)."
Then add, as the method body's first statements:

```python
        if not 0 <= miss_lanes <= max_rows:
            raise ValueError(f"graph gather miss lanes must be 0-{max_rows}, got {miss_lanes}")
        if 0 < miss_lanes < max_rows and scratch_destinations:
            raise ValueError(
                "a graph gather that serves fewer misses than its routes needs DIRECT residency's victim slots"
            )
```

After `self.graph_gather_rows = max_rows` at the end of the method, add
`self.graph_miss_lanes = miss_lanes if miss_lanes < max_rows else 0`.

Add the property right after `serves_graph_gather`:

```python
    @property
    def graph_miss_width(self) -> int:
        """Distinct misses one graph gather serves: ``graph_miss_lanes`` when narrowed, else one per route."""
        return self.graph_miss_lanes or self.graph_gather_rows
```

- [ ] **Step 4: The manager**

In `expert_hot_cache.py`, `from_model`:
- Add `graph_gather_miss_lanes: int | None = None,` after `fused_insert: bool | None = None,`.
- Add to the docstring: "``graph_gather_miss_lanes`` caps each layer's distinct misses per graph gather below its
  routes (a verify); ``None`` reads ``SGLANG_MOE_EXPERT_GRAPH_GATHER_MISS_LANES``, and 0 keeps one lane per route.
  It needs stage DIRECT."
- Right after `insert_on_miss = int(insert_on_miss)`, add:

```python
        if graph_gather_miss_lanes is None:
            graph_gather_miss_lanes = envs.SGLANG_MOE_EXPERT_GRAPH_GATHER_MISS_LANES.get()
        graph_gather_miss_lanes = index(graph_gather_miss_lanes)
        if graph_gather_miss_lanes < 0:
            raise ValueError(f"graph gather miss lanes cannot be negative, got {graph_gather_miss_lanes}")
```

- After the `scratch_rows = {...}` dict (which follows `direct = insert_on_miss == InsertOnMissStage.DIRECT`), add:

```python
        # A verify's gather may serve fewer distinct misses than it has routes; the misses take the lanes, the
        # routes keep their width.
        miss_lanes = {
            layer_id: min(rows, graph_gather_miss_lanes) if graph_gather_miss_lanes else rows
            for layer_id, rows in gather_rows.items()
        }
        if not direct and any(miss_lanes[layer_id] < rows for layer_id, rows in gather_rows.items()):
            raise ValueError(
                f"SGLANG_MOE_EXPERT_GRAPH_GATHER_MISS_LANES={graph_gather_miss_lanes} below a layer's graph-gather "
                "routes needs SGLANG_MOE_HOT_INSERT_ON_MISS_STAGE=2 (DIRECT): only its gathers flag the misses they "
                "cannot serve"
            )
```

- The DIRECT floor the allocator gives every layer first counts lanes, not routes. In
  `floors = {layer_id: 2 * rows if direct else 0 for layer_id, rows in gather_rows.items()}`, use
  `2 * miss_lanes[layer_id]`. In the refusal below it that ends `(twice its graph-gather rows)"`, change the end to
  `(twice its graph-gather miss lanes)"`. Unchanged at one token. At a 36-route verify the old floor of 72 slots
  would refuse any layer of fewer experts.
- In the loop that calls the format's `plan_graph_gather`, pass `miss_lanes[layer_id]` instead of `rows`:
  `plan(streamers[layer_id], miss_lanes[layer_id])`. Keep the `if rows and plan is not None` guard.
- In the `enable_graph_gather` loop, pass the width:
  ```python
                streamers[layer_id].enable_graph_gather(
                    rows, scratch_destinations=not direct, miss_lanes=miss_lanes[layer_id]
                )
  ```

- [ ] **Step 5: Residency, the RAM-miss attach, and the fused MoE**

In `expert_residency_gpu.py`, `_init_insert_on_miss`, replace:

```python
        rows = {streamer.graph_gather_rows for streamer in self.streamers}
        if len(rows) != 1:
            raise ValueError(
                "insert-on-miss needs one graph-gather row count on every layer"
            )
```

with:

```python
        # The misses one gather serves, not its routes: a verify's routes may outnumber them.
        rows = {streamer.graph_miss_width for streamer in self.streamers}
        if len(rows) != 1:
            raise ValueError(
                "insert-on-miss needs one graph-gather miss width on every layer"
            )
```

In `_init_insert_direct`:
- In the capacity check, change the message's first two f-string lines to:
  ```python
                    "SGLANG_MOE_HOT_INSERT_ON_MISS_STAGE=2 needs every layer to hold at least "
                    f"twice its graph-gather miss lanes; layer {layer_id} has {cache.capacity} "
                    f"slots for {width} lanes. Raise SGLANG_MOE_HOT_GPU_MB or use stage 1."
  ```
- Append to the docstring's "Capacity guarantee" paragraph: "That holds when the miss width equals the routes (one
  token). A verify's narrower width loses it: ``H + M`` may exceed ``miss_rows``. Its gathers serve the misses that
  find a victim and flag the rest (:meth:`clamp_gather_misses`), so the floor stays ``2 * miss_rows``."

In `gather_destinations`, change `streamer._graph_destination_slots.copy_(destinations.to(torch.int32))` to:

```python
        # The plan's slot buffer has a row per route; the lanes are the first miss_rows.
        streamer._graph_destination_slots[: self.miss_rows].copy_(destinations.to(torch.int32))
```

In `fused_gather_destinations`, pass `streamer._graph_destination_slots[: self.miss_rows],` in place of
`streamer._graph_destination_slots,`.

In `exl3_ram_miss.py`, `Exl3RamMissService.attach`, replace:

```python
        if streamer.graph_gather_rows > self.lanes:
            # The post kernel requests min(count, lanes) lanes and traps on a wider
            # plan.
            raise ValueError(
                f"exl3 RAM miss: layer {streamer.layer_id} gathers up to {streamer.graph_gather_rows} rows "
                f"per call but the service requests at most {self.lanes} lanes"
            )
```

with:

```python
        width = streamer.graph_miss_width
        if width > self.lanes:
            # The post kernel requests min(count, lanes) lanes and traps on a wider
            # plan. A verify's misses take lanes, not its routes.
            raise ValueError(
                f"exl3 RAM miss: layer {streamer.layer_id} gathers up to {streamer.graph_gather_rows} rows "
                f"({width} miss lanes) per call but the service requests at most {self.lanes} lanes"
            )
        if self.cpu_experts is not None and width < streamer.graph_gather_rows:
            raise ValueError(
                f"exl3 RAM miss: CPU experts serve one token; layer {streamer.layer_id}'s gather serves {width} "
                f"misses of {streamer.graph_gather_rows} routes (a verify)"
            )
```

In `fused_moe.py`, `exl3_fused_moe_for`, replace the DIRECT check:

```python
        if direct and cache.capacity < rows:
            raise ValueError(
                f"exl3 DIRECT needs at least top_k resident slots per token, one per route ({cache.capacity} < {rows})"
            )
```

with:

```python
        # A token's routes are distinct slots; tokens share slots, and a verify's gather flags the misses it
        # cannot place (GpuResidencyUpdater.clamp_gather_misses).
        if direct and cache.capacity < top_k:
            raise ValueError(
                f"exl3 DIRECT needs at least top_k resident slots per token, one per route ({cache.capacity} < {top_k})"
            )
```

- [ ] **Step 6: Run the tests and watch them pass**

Run the CPU command with Step 2's CPU `<FILES>` plus `test/registered/unit/layers/moe/test_exl3_ram_miss_service.py`,
and `<K>` = `not nothing_matches_this`.
Expected: all pass. The pre-existing attach refusals still match: a layer with no miss lanes has width = rows.

Run the GPU command with `<FILES>` =
`test/registered/unit/layers/moe/test_expert_residency_gpu.py test/registered/unit/layers/moe/test_expert_graph_gather.py`
and `<K>` = `not nothing_matches_this`.
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add python/sglang/srt/environ.py python/sglang/srt/layers/moe/expert_stream.py \
  python/sglang/srt/layers/moe/expert_hot_cache.py python/sglang/srt/layers/moe/expert_residency_gpu.py \
  python/sglang/srt/layers/moe/exl3_ram_miss.py python/sglang/srt/layers/quantization/exl3/fused_moe.py
git commit -m "feat(graph-gather): a verify's gather serves a miss width W below its routes (SGLANG_MOE_EXPERT_GRAPH_GATHER_MISS_LANES, DIRECT only)" -m "<trailers>"
```

---

### Task 3: Serve the lanes that found a victim, flag the rest

**Files:**
- Modify: `python/sglang/srt/layers/moe/expert_residency_gpu.py` (`_init_insert_direct`, new `clamp_gather_misses`,
  `snapshot`)
- Modify: `python/sglang/srt/layers/moe/expert_stream.py` (`_gather_graph`)
- Test: `test/registered/unit/layers/moe/test_expert_residency_gpu.py` (CUDA; `TestInsertOnMissDirect`)

**Interfaces:**
- Consumes: Task 2's `graph_miss_width` and `miss_rows`.
- Produces:
  - `GpuResidencyUpdater.clamp_gather_misses() -> None`. It is capture-safe, called after `gather_destinations` or
    `fused_gather_destinations` and before the backend's post.
  - `GpuResidencyUpdater.gather_overflow`: int64 `[layers]`, gathers whose misses outnumbered their served lanes.
  - `GpuResidencyUpdater.overflow_flag`: int32 `[1]`, sticky. Nothing in serving clears it; D2-3 reads and clears it
    around each verify.
  - `snapshot()` adds `"gather_overflow"` only when the miss width is below the routes.

- [ ] **Step 1: Write the failing tests**

Add to `test_expert_residency_gpu.py`, module level, after `_random_routes`:

```python
def _verify_routes(generator, tokens):
    """Each layer's routes for a ``tokens``-token verify: top-k distinct per token, tokens may share experts."""
    return [[generator.sample(range(EXPERTS), TOP_K) for _ in range(tokens)] for _ in range(LAYERS)]
```

Add to class `TestInsertOnMissDirect`:

```python
    # ----- a narrow gather: serve what found a victim, flag the rest -----

    def replay_verify(self, manager, graph, static, outputs, routes, check_outputs):
        """One captured verify; checks every token's gathered rows when ``check_outputs``."""
        static.copy_(torch.tensor(routes, dtype=torch.int32, device="cuda"))
        graph.replay()
        torch.cuda.synchronize()
        if check_outputs:
            for layer in range(LAYERS):
                source_layer = self.model.get_submodule(str(layer))
                experts = torch.tensor(routes[layer]).reshape(-1)
                for name in NVFP4_STREAM_TENSORS:
                    source = getattr(source_layer, name)
                    expected = source[experts.to(source.device)].reshape(-1).view(torch.uint8).cpu()
                    actual = outputs[layer][name].view(torch.uint8).reshape(-1).cpu()
                    self.assertTrue(torch.equal(actual, expected), f"layer {layer} {name}")
        manager.on_expert_distribution(_decode_batch(), {"global_physical_count": _counts(routes)})
        torch.cuda.synchronize()

    def _narrow(self, model, lanes, fused):
        from sglang.srt.environ import envs

        with envs.SGLANG_MOE_EXPERT_FUSED_PLAN.override(fused), envs.SGLANG_DSV41_ENABLE_LAYER_FUSION.override(fused):
            # Zero seed scores fill every layer evenly (8 slots each) whatever its floor, so the narrow and the wide
            # twin hold the same capacities.
            manager = _manager(
                model, gpu=True, seed_scale=(0, 0, 0), graph_gather_batch_size=2, graph_gather_miss_lanes=lanes,
                budget_bytes=56 * LAYERS * 8, **DIRECT,
            )
        self.assertEqual(manager.gpu_residency.layer_fusion, fused)
        return manager

    def test_a_narrow_gather_is_the_wide_gather_until_it_overflows(self):
        """Two tokens route 4 ids a layer. Until the narrow gather first flags, its residency is the wide one's bit for
        bit: the served lanes are the same prefix of the same usable shortlist entries."""
        for fused in (False, True):
            self.model = _model()
            wide, narrow = self._narrow(self.model, 0, fused), self._narrow(_model(), 2, fused)
            self.assertEqual(wide.gpu_residency.miss_rows, 4)
            self.assertEqual(narrow.gpu_residency.miss_rows, 2)
            for layer_id in wide.caches:
                self.assertEqual(wide.caches[layer_id].capacity, 8)
                self.assertEqual(narrow.caches[layer_id].capacity, 8)
            twins = [(manager, *self.capture(manager, tokens=2)) for manager in (wide, narrow)]
            generator = random.Random(23)
            compared, overflowed = 0, False
            for step in range(60):
                routes = _verify_routes(generator, 2)
                for manager, graph, static, outputs in twins:
                    flagged = int(narrow.gpu_residency.overflow_flag.item()) > 0
                    self.replay_verify(manager, graph, static, outputs, routes, check_outputs=not flagged)
                if int(narrow.gpu_residency.overflow_flag.item()) > 0:
                    overflowed = True
                    break
                context = f"fused={fused} step {step}"
                assert_states_equal(self, device_state(narrow), device_state(wide), context)
                assert_slot_rows(self, narrow, self.model, context)
                self.assertEqual(narrow.gpu_residency.snapshot()["gather_overflow"], [0] * LAYERS)
                compared += 1
            self.assertGreaterEqual(compared, 3, f"fused={fused}: too few steps before the first overflow")
            self.assertTrue(overflowed, f"fused={fused}: no step overflowed two lanes")
            self.assertEqual(wide.gpu_residency.overflow_flag.item(), 0)
            self.assertNotIn("gather_overflow", wide.gpu_residency.snapshot())

    def test_an_overflowing_gather_serves_its_lanes_and_flags_its_layer(self):
        """Layer 0's two tokens route four non-resident experts into two lanes: two are copied and committed, the
        layer is flagged, and every mapped expert's slot still holds its own bytes."""
        for fused in (False, True):
            self.model = _model()
            manager = self._narrow(self.model, 2, fused)
            updater = manager.gpu_residency
            graph, static, outputs = self.capture(manager, tokens=2)
            mapping = manager.caches[0].expert_to_slot.tolist()
            outsiders = [expert for expert, slot in enumerate(mapping) if slot < 0]
            self.assertGreaterEqual(len(outsiders), 4)
            hits = [
                [[e for e, s in enumerate(manager.caches[layer].expert_to_slot.tolist()) if s >= 0][:TOP_K]] * 2
                for layer in range(1, LAYERS)
            ]
            routes = [[outsiders[0:2], outsiders[2:4]]] + hits
            inserted = updater.gather_insertions.clone()
            self.replay_verify(manager, graph, static, outputs, routes, check_outputs=False)
            context = f"fused={fused}"
            self.assertEqual(int(updater.overflow_flag.item()), 1, context)
            self.assertEqual(updater.gather_overflow.tolist(), [1, 0, 0], context)
            self.assertEqual((updater.gather_insertions - inserted).tolist(), [2, 0, 0], context)
            self.assertEqual(int(manager.streamers[0]._graph_miss_count.item()), 2, context)
            self.assertEqual(updater.insertion_truncated.tolist(), [0] * LAYERS, context)
            assert_slot_rows(self, manager, self.model, context)
            # The next all-hit verify neither flags nor reads a wrong row.
            updater.overflow_flag.zero_()
            self.replay_verify(manager, graph, static, outputs, [[[0, 1], [0, 1]]] * LAYERS, check_outputs=True)
            self.assertEqual(int(updater.overflow_flag.item()), 0, context)

    def test_hits_that_empty_the_shortlist_flag_the_gather(self):
        """Two misses fit two lanes, but the other token routes both shortlisted residents: no entry survives, nothing
        is copied, and the gather is flagged instead of copying into slot 0."""
        for fused in (False, True):
            self.model = _model()
            manager = self._narrow(self.model, 2, fused)
            updater = manager.gpu_residency
            graph, static, outputs = self.capture(manager, tokens=2)
            victims = updater.victims[0].tolist()
            self.assertTrue(bool(updater.victim_valid[0].all()))
            shortlisted = [int(updater.slot_to_expert[0, slot]) for slot in victims]
            self.assertTrue(all(expert >= 0 for expert in shortlisted), "the shortlist names free slots")
            mapping = manager.caches[0].expert_to_slot.tolist()
            outsiders = [expert for expert, slot in enumerate(mapping) if slot < 0]
            routes = [[shortlisted, outsiders[:2]]] + [[[0, 1], [0, 1]]] * (LAYERS - 1)
            before = _cache_bytes(manager.caches[0])
            self.replay_verify(manager, graph, static, outputs, routes, check_outputs=False)
            context = f"fused={fused}"
            self.assertEqual(int(updater.overflow_flag.item()), 1, context)
            self.assertEqual(int(manager.streamers[0]._graph_miss_count.item()), 0, context)
            self.assertEqual(updater.insertion_truncated.tolist(), [0] * LAYERS, context)
            after = _cache_bytes(manager.caches[0])
            for name in before:
                self.assertTrue(torch.equal(after[name], before[name]), f"{context}: {name} was written")

    def test_a_one_token_gather_never_clamps(self):
        from sglang.srt.layers.moe.expert_residency_gpu import GpuResidencyUpdater

        manager = _manager(self.model, gpu=True, **DIRECT)
        with unittest.mock.patch.object(GpuResidencyUpdater, "clamp_gather_misses", side_effect=AssertionError):
            graph, static, outputs = self.capture(manager)
            self.step(manager, graph, static, outputs, _random_routes(random.Random(5)), "one token")
        self.assertNotIn("gather_overflow", manager.gpu_residency.snapshot())
```

In the shortlist test, layers 1-2 route experts 0 and 1 on both tokens: at most two misses, which fit two lanes.
The test reads only layer 0's state and the flag, which is global and set by layer 0 either way.

- [ ] **Step 2: Run them and watch them fail**

Commit `test(direct-residency): a narrow verify gather serves what found a victim and flags the rest (failing)`. Run
the GPU command with `<FILES>` = `test/registered/unit/layers/moe/test_expert_residency_gpu.py` and `<K>` =
`narrow_gather or overflowing or empty_the_shortlist or never_clamps`.
Expected: FAIL.
- The twin, overflow and shortlist tests fail with `AttributeError: ... 'overflow_flag'`. The twin test may first
  fail on its `"gather_overflow"` snapshot key.
- Without the clamp, a gather with more misses than lanes traps in the copy, or its counted dead lane copies into slot
  0. Read the actual first failure and record it.
- `test_a_one_token_gather_never_clamps` fails with `AttributeError: ... does not have the attribute
  'clamp_gather_misses'`.

- [ ] **Step 3: Implement**

In `_init_insert_direct`, after `self._pending_commit = None`, add:

```python
        # A verify's gather may hold more misses than lanes: per layer, the gathers that did, and one sticky word
        # that says this forward's output is not a verify result (clamp_gather_misses).
        self.gather_overflow = torch.zeros(layers, dtype=torch.long, device=device)
        self.overflow_flag = torch.zeros(1, dtype=torch.int32, device=device)
        self.narrow_gather = any(
            streamer.graph_miss_width < streamer.graph_gather_rows for streamer in self.streamers
        )
```

Add the method after `fused_gather_destinations`:

```python
    def clamp_gather_misses(self) -> None:
        """Serve the miss lanes that found a victim, and flag the gather when its misses outnumber them.

        Only for a gather whose miss width is below its routes (a verify). There, the misses can exceed the shortlist,
        and the hits can disqualify more entries than the shortlist has to spare. The lanes
        :meth:`gather_destinations` made live are a prefix (usable entries first, then ``lane < count``), so their sum
        is the number served. The shared miss count, which the copy, the lease record and the commit all read, drops
        to it: every row copied is committed, and nothing is copied into slot 0. The unserved misses' routes read a
        served lane's slot (a valid slot, the wrong expert), so the forward's output is not a verify result, and
        ``overflow_flag`` says so. Device only, capture-safe.
        """
        row, streamer, _, live = self._pending_commit
        count = streamer._graph_miss_count
        served = live.sum(dtype=torch.int32)
        over = count > served
        self.gather_overflow[row].add_(over.sum())
        self.overflow_flag.bitwise_or_(over.to(torch.int32))
        count.copy_(torch.minimum(count, served))
```

In `snapshot()`, before `return snapshot`, add:

```python
        if self.insert_direct and self.narrow_gather:
            snapshot["gather_overflow"] = self.gather_overflow.cpu().tolist()
```

In `expert_stream.py`, `_gather_graph`, right after the `elif direct is not None:` branch that calls
`direct.gather_destinations(...)` and before `if prefetch_puller is not None:`, add:

```python
        if direct is not None and self.graph_miss_width < self.graph_gather_rows:
            # A verify: serve the misses that found a victim; the rest flag the forward.
            direct.clamp_gather_misses()
```

- [ ] **Step 4: Run the tests and watch them pass**

Run the GPU command with `<FILES>` =
`test/registered/unit/layers/moe/test_expert_residency_gpu.py test/registered/unit/layers/moe/test_expert_graph_gather.py`
and `<K>` = `not nothing_matches_this`.
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add python/sglang/srt/layers/moe/expert_residency_gpu.py python/sglang/srt/layers/moe/expert_stream.py
git commit -m "feat(direct-residency): a verify gather serves the misses that found a victim and flags the forward when it cannot serve them all" -m "<trailers>"
```

---

### Task 4: The EXL3 lease chain at W lanes for a 6-token verify, and the record

**Files:**
- Create: `test/manual/dsv41/test_exl3_verify_miss_lanes_gpu.py`
- Modify: `DSV41_REFERENCE.md` (new `### 33.7` after §33.6, before `## Sources`)

**Interfaces:**
- Consumes: Tasks 1-3 end to end: `from_model(graph_gather_batch_size=6, graph_gather_miss_lanes=8,
  insert_on_miss=2)`, `Exl3MoEMethod._apply_graph` with `x [6, H]` and `ids [6, 6]`, and `overflow_flag` and
  `gather_overflow` on `manager.gpu_residency`.
- Reuses `_source_rows`, `_reference` and `_rel` from `test/manual/dsv41/test_exl3_ram_miss_graph_gpu.py` (imported
  via `sys.path`). The setup mirrors that file's `test_direct_insert_replay_hit_evict_refetch_and_prefill_handoff`.

- [ ] **Step 1: Write the test**

```python
"""D2-2 gate: a captured 6-token verify through the EXL3 lease chain at 8 miss lanes (GPU).

Six tokens of top-6 route 36 ids a layer, past the wire's 32 lanes. The service builds 8 lanes (the miss width). A
verify whose union of misses fits the lanes is served exactly: every routed expert is mapped, its slot holds its
checkpoint bytes, and every token's output meets the probe's bar. A verify with more misses than lanes serves 8,
flags the forward, and leaves residency exact: every mapped expert's slot still holds its own bytes, and the chain
neither traps nor fail-stops.
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(__file__))
from test_exl3_ram_miss_graph_gpu import HIDDEN, INTER, REL_BOUND, _reference, _rel, _source_rows  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

ACT_LIMIT = 10.0
TOP_K, TOKENS, LANES, EXPERTS = 6, 6, 8, 48


@pytest.mark.parametrize("fused", [False, True], ids=["generic", "layer_fusion"])
def test_a_verify_gather_serves_its_lanes_and_flags_what_it_cannot(tmp_path, fused):
    from sglang.srt.environ import envs
    from sglang.srt.layers.moe import exl3_ram_miss as service_module
    from sglang.srt.layers.moe.exl3_expert_format import Exl3ExpertFormat
    from sglang.srt.layers.moe.exl3_expert_layout import build_exl3_expert_layout
    from sglang.srt.layers.moe.expert_hot_cache import ExpertHotCacheManager
    from sglang.srt.layers.moe.expert_stream import ExpertPinnedHostCache, ExpertStreamer
    from sglang.srt.layers.quantization.exl3 import Exl3MoEMethod
    from sglang.test.dsv41_fake_exl3 import write_fake_exl3
    from sglang.test.dsv41_ram_miss_fixtures import service_row_images

    write_fake_exl3(str(tmp_path), num_layers=1, num_experts=EXPERTS, hidden=HIDDEN, inter=INTER, finite=True)
    source = _source_rows(tmp_path, num_experts=EXPERTS)
    layout = build_exl3_expert_layout(str(tmp_path))
    service_module.Exl3RamMissService._instance = None
    service = service_module.Exl3RamMissService.get()
    service.plan_gather_width(LANES)
    try:
        with (
            service_row_images(tmp_path),
            envs.SGLANG_MOE_EXPERT_ROW_SOURCE.override("shards"),
            envs.SGLANG_MOE_EXPERT_GRAPH_GATHER.override(True),
            envs.SGLANG_MOE_EXPERT_PREFETCH_PULL_MODE.override("off"),
            envs.SGLANG_MOE_EXPERT_FUSED_PLAN.override(fused),
            envs.SGLANG_DSV41_ENABLE_LAYER_FUSION.override(fused),
        ):
            model = torch.nn.Module()
            layer = torch.nn.Module()
            layer.layer_id, layer.top_k = 0, TOP_K
            fmt = Exl3ExpertFormat(layout, 0, source_root=str(tmp_path))
            fmt.max_gather_rows = TOKENS * TOP_K
            streamer = ExpertStreamer(layer, fmt.names, layer_id=0, format=fmt)
            layer._nvfp4_expert_streamer = streamer
            model.add_module("expert_layer", layer)
            ExpertPinnedHostCache(streamer, 3 * LANES, **fmt.pinned_tier_options(layer))
            manager = ExpertHotCacheManager.from_model(
                model, budget_bytes=2 * LANES * streamer.bytes_per_expert,
                seed_path=None, dynamic=True, update_prefill_tokens=16,
                min_residence_forwards=0, benefit_ratio=0.0,
                graph_gather_batch_size=TOKENS, graph_gather_miss_lanes=LANES, update_decode_forwards=1,
                gpu_residency_update=True, insert_on_miss=2,
            )
            updater = manager.gpu_residency
            assert streamer.graph_gather_rows == TOKENS * TOP_K and streamer.graph_miss_width == LANES
            assert service.lanes == LANES and updater.miss_rows == LANES
            assert manager.caches[0].capacity == 2 * LANES and manager.caches[0].scratch_rows == 0
            assert updater.layer_fusion is fused

            generator = torch.Generator(device="cpu").manual_seed(37)
            x = (torch.randn((TOKENS, HIDDEN), generator=generator) * 0.5).to("cuda", torch.bfloat16)
            weights = torch.full((TOKENS, TOP_K), 1.0 / TOP_K, device="cuda")
            ids = torch.tensor([list(range(TOP_K))] * TOKENS, device="cuda", dtype=torch.int32)
            Exl3MoEMethod._apply_graph(layer, streamer, x, weights, ids, ACT_LIMIT)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out = Exl3MoEMethod._apply_graph(layer, streamer, x, weights, ids, ACT_LIMIT)
            manager.discard_graph_capture_routes()

            def residency_is_exact():
                mapping = updater.mapping[0, :EXPERTS].cpu()
                for expert in range(EXPERTS):
                    slot = int(mapping[expert])
                    if slot < 0:
                        continue
                    for name, rows in manager.caches[0].tensors.items():
                        assert torch.equal(rows[slot].cpu(), source[name][expert]), (expert, slot, name)

            def replay(routes, overflow):
                updater.overflow_flag.zero_()
                ids.copy_(torch.tensor(routes, device="cuda", dtype=torch.int32))
                graph.replay()
                torch.cuda.synchronize()
                service.fail_stop_check()
                assert streamer.row_backend.keep.item() == 1.0
                assert int(updater.overflow_flag.item()) == int(overflow), routes
                assert updater.insertion_truncated[0].item() == 0
                residency_is_exact()
                if overflow:
                    return
                mapping = updater.mapping[0, :EXPERTS].cpu()
                for t, route in enumerate(routes):
                    slots = torch.tensor([int(mapping[e]) for e in route])
                    assert bool((slots >= 0).all()), route
                    ref = _reference(x[t : t + 1], weights[t], slots, manager.caches[0].tensors)
                    assert _rel(out[t : t + 1], ref) <= REL_BOUND, (t, route)

            def outsiders():
                """Experts with no slot right now, so a route to them is a miss."""
                mapping = updater.mapping[0, :EXPERTS].cpu().tolist()
                return [expert for expert, slot in enumerate(mapping) if slot < 0]

            shared = outsiders()[:TOP_K]
            replay([shared] * TOKENS, overflow=False)  # 6 misses shared by every token, copied once
            rows_read = service.host.counters()["rows_read"]
            replay([shared] * TOKENS, overflow=False)  # all hits now
            assert service.host.counters()["rows_read"] == rows_read
            eight = outsiders()[:LANES]
            replay([[eight[(t + k) % 8] for k in range(TOP_K)] for t in range(TOKENS)], overflow=False)  # 8 fill the lanes
            overflowed = updater.gather_overflow[0].item()
            twelve = outsiders()[:12]
            replay([[twelve[(2 * t + k) % 12] for k in range(TOP_K)] for t in range(TOKENS)], overflow=True)  # 12 > 8
            assert updater.gather_overflow[0].item() == overflowed + 1
            replay([outsiders()[:TOP_K]] * TOKENS, overflow=False)  # served exactly again afterwards
    finally:
        service.shutdown()
```

- [ ] **Step 2: Run it**

Commit `test(exl3-ram-miss): a 6-token verify at 8 miss lanes through the lease chain` and push. Then on divix01 (it
writes a fake checkpoint under `tmp_path` and reads no production rows, so the GPU lock alone suffices):

```bash
ssh divix01 'cd /data/models/slang/nvfp4-work/wt-dsv41-dspark-graph && git fetch -q origin && git checkout -q --detach origin/dsv41-dspark-graph \
  && export PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 SGLANG_EXL3_SRC=/data/models/slang/nvfp4-work/exllamav3 \
  && /data/models/slang/.venv/bin/python -c "import sglang; print(sglang.__file__)" \
  && flock /data/models/slang/nvfp4-work/cc-gpu.lock taskset -c 32-63 /data/models/slang/.venv/bin/python -m pytest \
     test/manual/dsv41/test_exl3_verify_miss_lanes_gpu.py test/manual/dsv41/test_exl3_ram_miss_graph_gpu.py \
     test/manual/dsv41/test_exl3_lease_kernels_cuda.py test/manual/dsv41/test_dsv41_layer_fusion_gpu.py \
     -q -p no:randomly > /tmp/d22-gate.log 2>&1; echo EXIT=${PIPESTATUS[0]}; tail -8 /tmp/d22-gate.log'
```

Expected: `EXIT=0`. Both new cases pass, and the BS1 lease-chain, lease-kernel and layer-fusion files stay green.

If a bar fails, use superpowers:systematic-debugging. Do not loosen a bar.
- A trap or fail-stop points at a count read before the clamp: the post, S or CW was handed the planner's count.
- A slot holding the wrong bytes points at a commit that disagrees with the copy.
- A wrong output on a non-overflow step points at the remap translation of routes past 32 (Task 1) or at a
  destination buffer that is not the first `miss_rows` entries (Task 2).

- [ ] **Step 3: Record it**

Add `### 33.7 D2-2: a miss width below the routes, and the overflow flag (2026-10-05)` to `DSV41_REFERENCE.md`, after
§33.6 and before `## Sources`. It covers:
- that §33.3 item 2's 8-lane record is out of date: `LeaseLayout` takes 1-32 lanes (`0b82fc37c5`, `5813a147d5`,
  `7a77475a20`), and the real one-token assumption was that the width counted routes;
- what changed, with the commits: the 64-route DIRECT gather kernel, `SGLANG_MOE_EXPERT_GRAPH_GATHER_MISS_LANES` and
  `graph_miss_width`, and `clamp_gather_misses` with `gather_overflow` and `overflow_flag`;
- why the clamp keeps residency exact (live lanes are a prefix; one count for copy, record and commit) and what an
  overflowed forward computes (valid slots, wrong experts, flagged);
- why `capacity ≥ 2W` stays as the floor although its guarantee no longer holds for a verify (the flag covers the
  shortfall). Whether a smaller floor is safe is §33.5's measurement 3, still open;
- the gate (`test_exl3_verify_miss_lanes_gpu.py`, both arms) and its result;
- what D2-2 does not do:
  - D2-3: read and clear the flag, re-verify an overflowed verify, the gate lift, and the protect list, which still
    truncates silently at the lane count (recency stamps only, §33.3 item 6);
  - the choice of W, which waits for D2-3's overflow rate on real routes against §33.5's projection (W = 8 overflows
    half the layers in the offline model);
  - D2-4: CPU experts at W < routes.

```bash
git add test/manual/dsv41/test_exl3_verify_miss_lanes_gpu.py DSV41_REFERENCE.md
git commit -m "docs(dsv41): section 33.7 -- a verify's miss width below its routes, gated through the lease chain" -m "<trailers>"
git push -q origin dsv41-dspark-graph
```
