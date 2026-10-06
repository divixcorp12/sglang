# DSpark graphed verify D2-1: multi-token route plan and in-graph EXL3 MoE

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The in-graph EXL3 MoE path serves M ≤ 10 tokens of top-k routes per layer: a fused route plan that dedups
experts across tokens, and an `Exl3FusedMoE` that takes `[M, H]` with a real token index per route.

**Architecture:** D2-1 of four v2 plans (D2-2 wire/staging at W, D2-3 end-to-end graphed verify with CPU off, D2-4
multi-token CPU experts). This plan is self-contained and does not touch the RAM-miss wire, DIRECT residency or
DSpark. It has three parts:
- The BS1 planner kernel stays as it is. A second kernel, one block of 64 threads, reproduces `plan_graph_routes`
  (which already dedups) for multi-token calls.
- exllamav3's `exl3_moe` is multi-token by design. `token_sorted` holds each route's token sorted by slot, a slot
  holds one row per token, and experts are scheduled by ticket, so `num_active` only sizes the launch. The work is in
  our route tables (the torch chain and the layer-fusion kernel) and the buffers.
- A real-row GPU gate closes the plan, with timings per M recorded in `DSV41_REFERENCE.md`.

**Tech Stack:** CUDA JIT kernels (`sglang.kernels.jit`, tvm-ffi), torch, pytest; GPU runs on divix01.

**Spec:** `DSV41_REFERENCE.md` §33.3 items 3 and 4 (the blockers), and §33.5 (why v2 is being built: verify width 6,
stride 3, up to 36 routes per layer).

## Global Constraints

- BS1 stays bit for bit what it is today.
  - The one-token planner path keeps running `plan_unique_routes_kernel` unchanged.
  - At M = 1, the route tables (torch chain and kernel) produce exactly today's outputs.
  - At M = 1, `Exl3FusedMoE.run` passes `NUM_ACTIVE` (6), as today.
- Everything on the graph path is capture-safe: no host read of a device value, and no allocation inside a run.
  Shape reads (`x.shape[0]`) are host-side and allowed.
- Never pass `-use_fast_math` to `exl3_route_tables.cuh`. It implies `-ftz`, and bit parity with the torch chain
  breaks.
- Route limits:
  - The multi-token planner and the route tables take at most 64 routes, which is 10 tokens at top-6.
  - The one-token planner keeps its 32.
  - `Exl3FusedMoE` takes at most `ROW_TILE` = 16 tokens. A slot holds one row per token, and the fused kernel skips
    a slot with more rows than its temp tile.
- CPU experts stay one token in this plan. `Exl3FusedMoE.run` and the route-tables launcher refuse `cpu` with M > 1;
  D2-4 lifts that.
- divix01 protocol (`.claude/rules/divix01-run-protocol.md`):
  - Push the branch, then `git fetch` and `git checkout --detach origin/dsv41-dspark-graph` in
    `/data/models/slang/nvfp4-work/wt-dsv41-dspark-graph`.
  - Run with `PYTHONPATH=$PWD/python`, and print `sglang.__file__` first.
  - Read `${PIPESTATUS[0]}`, never a pipe's status.
  - GPU work runs under `flock /data/models/slang/nvfp4-work/cc-gpu.lock taskset -c 32-63`. A job that also reads
    checkpoint rows takes `rowimg-disk.lock` first.
- Every commit ends with:
  ```
  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01Y3EXKJdMpP7nwX3qcBi8Vq
  ```

The GPU test command used below, written once (`<FILES>` and `<K>` vary):

```bash
git push -q origin dsv41-dspark-graph && ssh divix01 'cd /data/models/slang/nvfp4-work/wt-dsv41-dspark-graph \
  && git fetch -q origin && git checkout -q --detach origin/dsv41-dspark-graph \
  && export PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 \
  && flock /data/models/slang/nvfp4-work/cc-gpu.lock taskset -c 32-63 /data/models/slang/.venv/bin/python -c "import sglang; print(sglang.__file__)" \
  && flock /data/models/slang/nvfp4-work/cc-gpu.lock taskset -c 32-63 /data/models/slang/.venv/bin/python -m pytest <FILES> -q -p no:randomly -k "<K>" > /tmp/d21.log 2>&1; echo EXIT=${PIPESTATUS[0]}; tail -15 /tmp/d21.log'
```

The laptop has no torch, so CPU tests run on divix01 too. The template (`<FILES>` and `<K>` vary):

```bash
git push -q origin dsv41-dspark-graph && ssh divix01 'cd /data/models/slang/nvfp4-work/wt-dsv41-dspark-graph \
  && git fetch -q origin && git checkout -q --detach origin/dsv41-dspark-graph \
  && export PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 \
  && taskset -c 0-63 /data/models/slang/.venv/bin/python -c "import sglang; print(sglang.__file__)" \
  && taskset -c 0-63 /data/models/slang/.venv/bin/python -m pytest <FILES> -q -p no:randomly -k "<K>" > /tmp/d21cpu.log 2>&1; echo EXIT=${PIPESTATUS[0]}; tail -15 /tmp/d21cpu.log'
```

A RED run pushes a failing-test commit first. That is intended: the rule is "commit before running
unverified code", and nothing is amended.

## Review Focus

1. **One expert, several tokens, first route prefetch-covered.** Every route of that expert remaps to the prefetch
   slot, and none takes a scratch row (Task 1's random dedup test draws prefetch offers).
2. **One `Exl3FusedMoE` serving M = 6 then M = 1 in the same process, with graphs captured at both widths.** A wider
   run must leave nothing behind for a narrower one: rows past M, counts or det columns (Task 4 runs M = 6, then
   M = 1, and compares against a fresh object).
3. **`keep = 0` at M > 1.** A dropped layer computes no expert for any token, and every token's output is zero
   (Task 3).
4. **The router's int32 remap under layer fusion at M > 1.** The int64 copy and the ranks match the torch chain
   (Task 2's parity test runs both dtypes at M > 1).
5. **Boundaries.**
   - 64 routes plan; 65 are refused.
   - A one-token call of 33-64 routes is still refused by the predicate (Task 1).
   - Routes that are not a multiple of the tokens are refused by the route-tables launcher (Task 2).

---

### Task 1: The multi-token fused route plan

**Files:**
- Modify: `python/sglang/kernels/jit/csrc/moe/expert_route_plan.cuh` (append a kernel and its launcher)
- Modify: `python/sglang/kernels/ops/moe/expert_route_plan.py`
- Modify: `python/sglang/srt/layers/moe/expert_route_plan.py:123-156,184-264`
- Modify: `python/sglang/srt/layers/moe/expert_stream.py:1454-1456,1496-1512`
- Test: `test/registered/unit/kernels/test_expert_route_plan_fused.py`
- Test: `test/registered/unit/layers/moe/test_expert_graph_gather.py`

**Interfaces:**
- Produces:
  - `plan_unique_routes_cuda(..., miss_keys=None, dedup: bool = False)`.
  - `MAX_DEDUP_ROUTES = 64` in `sglang.kernels.ops.moe.expert_route_plan`.
  - `FUSED_MAX_DEDUP_ROUTES = 64` in `sglang.srt.layers.moe.expert_route_plan`.
  - `plan_graph_routes_fused(..., miss_keys=None, dedup: bool = False)`.
  - `supports_fused_graph_routes` admits `topk_ids.shape[0] != 1` up to 64 routes.
  - `ExpertStreamer._gather_graph` passes `dedup=topk_ids.shape[0] != 1`.

- [ ] **Step 1: Write the failing tests**

In `test_expert_route_plan_fused.py`:
- Add `dedup: bool = False` to `run_fused_case`, passed through as `plan_unique_routes_cuda(..., miss_keys, dedup=dedup)`.
- Add `dedup: bool = False` to `assert_matches_reference`, and pass it to `run_fused_case`.
- In `assert_matches_reference`, replace `_expected_slots(ids, oracle)` with `_expected_slots_any(ids, oracle)`.
- Replace `test_gate_refuses_multi_token_calls_even_when_shape_would_otherwise_qualify` with the gate test below.
- Add the rest:

```python
def _expected_slots_any(ids: torch.Tensor, oracle: SimpleNamespace) -> torch.Tensor:
    """Every route of one expert shares its destination, so a compacted position's slot is the remap of any route
    of the expert it holds (the first one here); for unique ids this is `_expected_slots`."""
    first = (ids.unsqueeze(1) == oracle.source_rows.unsqueeze(0)).float().argmax(dim=0)
    return oracle.remap[first]


def test_dedup_plan_worked_case():
    """Two tokens of three routes: 2 is a hit (slot 7); 5 is missed by both tokens and takes one scratch row."""
    ids = torch.tensor([5, 2, 9, 2, 7, 5], device="cuda", dtype=torch.int64)
    expert_to_slot = _expert_to_slot([])
    expert_to_slot[2] = 7
    result = run_fused_case(ids, expert_to_slot, scratch_base=10, dedup=True)
    assert result.source_rows.tolist() == [5, 9, 7, 2, 2, 5]
    assert result.slots.tolist() == [10, 11, 12, 7, 7, 10]
    assert result.remap.tolist() == [10, 7, 11, 7, 12, 10]
    assert int(result.count.item()) == 3
    assert result.graph_counters.tolist() == [6, 4]
    assert result.graph_unique_counters.tolist() == [1, 3]
    oracle = plan_graph_routes(ids, expert_to_slot, ids.numel(), 10)
    assert torch.equal(result.source_rows, oracle.source_rows) and torch.equal(result.remap, oracle.remap)


def _multi_token_ids(rng, tokens, top_k):
    return torch.tensor(
        [e for _ in range(tokens) for e in rng.sample(range(EXPERTS), top_k)], device="cuda", dtype=torch.int64
    )


@pytest.mark.parametrize("seed", range(200))
def test_random_multi_token_ids_match_the_reference(seed):
    rng = random.Random(seed)
    tokens = rng.randint(2, 10)
    top_k = rng.randint(1, 64 // tokens)
    ids = _multi_token_ids(rng, tokens, top_k)
    resident = rng.sample(range(EXPERTS), rng.randint(0, EXPERTS // 2))
    assert_matches_reference(ids, resident, scratch_base=len(resident), dedup=True)


@pytest.mark.parametrize("seed", range(50))
def test_multi_token_prefetch_coverage_matches_the_reference(seed):
    """Every route of a covered expert goes to the prefetch slot, duplicates included, and takes no scratch row."""
    rng = random.Random(seed)
    ids = _multi_token_ids(rng, rng.randint(2, 6), 6)
    resident = rng.sample(range(EXPERTS), rng.randint(0, EXPERTS // 2))
    expert_to_slot = _expert_to_slot(resident)
    predicted = int(ids[rng.randrange(ids.numel())])
    posted = rng.choice([0, 1])
    oracle = plan_graph_routes(
        ids, expert_to_slot, ids.numel(), len(resident),
        prefetch_expert=torch.tensor([predicted], device="cuda"), prefetch_slot=99,
        prefetch_count=torch.tensor([posted], dtype=torch.int32, device="cuda"),
    )
    from sglang.kernels.ops.moe.expert_route_plan import plan_unique_routes_cuda

    n = ids.numel()
    out = SimpleNamespace(
        source_rows=torch.full((n,), -1, dtype=torch.int64, device="cuda"),
        slots=torch.full((n,), -1, dtype=torch.int32, device="cuda"),
        count=torch.full((1,), -1, dtype=torch.int32, device="cuda"),
        remap=torch.full((n,), -1, dtype=torch.int64, device="cuda"),
    )
    plan_unique_routes_cuda(
        ids, expert_to_slot, len(resident), out.source_rows, out.slots, out.count, out.remap, None, None, None,
        torch.tensor([predicted], device="cuda"), torch.tensor([posted], dtype=torch.int32, device="cuda"), 99,
        None, None, dedup=True,
    )
    assert torch.equal(out.remap, oracle.remap)
    assert torch.equal(out.source_rows, oracle.source_rows)
    assert int(out.count.item()) == int(oracle.miss_plan_rows)


@pytest.mark.parametrize("seed", range(50))
def test_multi_token_miss_keys_sort_distinct_residual_experts(seed):
    """Residual rows hold each missed expert once, by key descending, ties by first appearance; every route of a
    missed expert remaps to its expert's row; everything past the residual rows is the unsorted plan's."""
    rng = random.Random(seed)
    ids = _multi_token_ids(rng, rng.randint(2, 8), 6)
    resident = rng.sample(range(EXPERTS), rng.randint(0, EXPERTS // 2))
    expert_to_slot = _expert_to_slot(resident)
    base = len(resident)
    high = 4 if seed % 2 else 1 << 62
    keys = torch.tensor([rng.randrange(-high, high) for _ in range(EXPERTS)], dtype=torch.int64, device="cuda")
    unsorted = run_fused_case(ids, expert_to_slot, base, dedup=True)
    result = run_fused_case(ids, expert_to_slot, base, miss_keys=keys, dedup=True)
    first = {}
    for i, e in enumerate(ids.tolist()):
        if int(expert_to_slot[e]) < 0:
            first.setdefault(e, i)
    order = sorted(first, key=lambda e: (-int(keys[e]), first[e]))
    count = len(order)
    assert int(result.count.item()) == count
    assert result.source_rows[:count].tolist() == order
    assert result.slots[:count].tolist() == [base + p for p in range(count)]
    assert torch.equal(result.source_rows[count:], unsorted.source_rows[count:])
    for i, e in enumerate(ids.tolist()):
        want = base + order.index(e) if e in first else int(unsorted.remap[i])
        assert int(result.remap[i]) == want
    for name in ("graph_counters", "graph_unique_counters", "route_counts"):
        assert torch.equal(getattr(result, name), getattr(unsorted, name))


def test_dedup_plans_64_routes_and_refuses_65():
    rng = random.Random(7)
    assert_matches_reference(_multi_token_ids(rng, 8, 8), [1, 2, 3], scratch_base=3, dedup=True)
    with pytest.raises(ValueError, match="1-64"):
        run_fused_case(_multi_token_ids(rng, 5, 13), _expert_to_slot([]), scratch_base=0, dedup=True)


def test_dedup_plan_reads_no_device_value_on_the_host_and_replays():
    from sglang.kernels.ops.moe.expert_route_plan import plan_unique_routes_cuda

    rng = random.Random(3)
    ids = _multi_token_ids(rng, 6, 6)
    expert_to_slot = _expert_to_slot([1, 4, 9])
    n = ids.numel()
    bufs = [torch.full((n,), -1, dtype=torch.int64, device="cuda"), torch.full((n,), -1, dtype=torch.int32, device="cuda"),
            torch.zeros(1, dtype=torch.int32, device="cuda"), torch.full((n,), -1, dtype=torch.int64, device="cuda")]
    pe, pc = torch.zeros(1, dtype=torch.int64, device="cuda"), torch.zeros(1, dtype=torch.int32, device="cuda")

    def plan():
        plan_unique_routes_cuda(ids, expert_to_slot, 3, *bufs, None, None, None, pe, pc, 0, None, None, dedup=True)

    with _NoHostReads():
        plan()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        plan()
    ids.copy_(_multi_token_ids(rng, 6, 6))
    expert_to_slot.copy_(_expert_to_slot([2, 5, 30]))
    graph.replay()
    torch.cuda.synchronize()
    oracle = plan_graph_routes(ids, expert_to_slot, n, 3)
    assert torch.equal(bufs[3], oracle.remap) and torch.equal(bufs[0], oracle.source_rows)


def test_gate_admits_multi_token_calls_up_to_64_routes():
    """One token keeps the warp kernel's 32-route limit; several tokens take the dedup kernel, up to 64 routes."""
    from sglang.srt.layers.moe.expert_route_plan import supports_fused_graph_routes

    expert_to_slot = _expert_to_slot([1, 6])
    single = torch.zeros((1, 32), device="cuda", dtype=torch.int64)
    assert supports_fused_graph_routes(single, expert_to_slot, scratch_rows=64)
    assert not supports_fused_graph_routes(torch.zeros((1, 33), device="cuda", dtype=torch.int64), expert_to_slot, 64)
    multi = torch.tensor([[1, 2, 2, 6], [3, 4, 4, 5]], device="cuda", dtype=torch.int64)
    assert supports_fused_graph_routes(multi, expert_to_slot, scratch_rows=8)
    assert supports_fused_graph_routes(torch.zeros((8, 8), device="cuda", dtype=torch.int64), expert_to_slot, 64)
    assert not supports_fused_graph_routes(torch.zeros((9, 8), device="cuda", dtype=torch.int64), expert_to_slot, 72)
    assert not supports_fused_graph_routes(multi, expert_to_slot, scratch_rows=7)
```

In `test_expert_graph_gather.py`, inside `TestExpertGraphGather`, add the streamer-level wiring test:

```python
    def test_fused_plan_serves_verify_shaped_routes_through_the_dedup_kernel(self):
        from sglang.srt.environ import envs
        from sglang.srt.layers.moe.expert_hot_cache import ExpertHotCache

        layer = _layer()
        streamer = ExpertStreamer(layer, NVFP4_STREAM_TENSORS)
        routes = 4 * TOP_K
        with envs.SGLANG_MOE_EXPERT_FUSED_PLAN.override("true"):
            cache = ExpertHotCache(streamer, 3, scratch_rows=routes)
            cache.reassign([1, 4, 6])
            streamer.enable_graph_gather(routes)
        ids = torch.tensor(
            [[0, 1, 2, 3], [2, 3, 4, 5], [5, 0, 6, 7], [7, 2, 1, 3]], dtype=torch.int32, device="cuda"
        )

        compact, tensors = streamer.gather(ids)

        self._assert_rows(layer, ids, compact, tensors)
        # The fused planner wrote its remap: the dedup kernel served the call, not the generic plan.
        self.assertTrue(torch.equal(streamer._graph_fused_remaps[torch.int32][:routes], compact.reshape(-1)))
        self.assertEqual(streamer.graph_counters.tolist(), [routes, 12])
        self.assertEqual(streamer.graph_unique_counters.tolist(), [3, 5])
        self.assertEqual(compact[compact >= cache.capacity].unique().numel(), 5)
```

- [ ] **Step 2: Run them and watch them fail**

Commit `test(expert-route-plan): multi-token fused plan with cross-token dedup (failing)`, then run the GPU command
with
`<FILES>` = `test/registered/unit/kernels/test_expert_route_plan_fused.py test/registered/unit/layers/moe/test_expert_graph_gather.py`
and `<K>` = `dedup or multi_token or gate_admits or verify_shaped`.
Expected: FAIL. `plan_unique_routes_cuda() got an unexpected keyword argument 'dedup'`. The gate test fails on
`supports_fused_graph_routes(multi, ...)`. The streamer test fails on the `_graph_fused_remaps` equality.

- [ ] **Step 3: Add the kernel**

Append inside `namespace sglang` in `expert_route_plan.cuh`, after `plan_unique_routes_gpu`:

```cuda
constexpr int kExpertRoutePlanDedupMaxRoutes = 64;

// Several tokens' routes (up to 64): `plan_graph_routes` in one block, one thread per route. An expert's first
// route is its earliest; only a residual expert's first route takes a scratch row, and every route of that expert
// remaps to it. Compaction puts those rows first, by first appearance (or by `miss_keys`, highest first, ties by
// first appearance), then every other route in route order. Counters are `plan_graph_routes`': routes and
// nonresident routes with multiplicity, hits and misses by distinct expert.
template <typename IdT, typename RemapT>
__global__ __launch_bounds__(kExpertRoutePlanDedupMaxRoutes, 1) void plan_dedup_routes_kernel(
    const IdT* __restrict__ topk_ids,
    const int64_t* __restrict__ expert_to_slot,
    int routes,
    int32_t scratch_base,
    int64_t* __restrict__ source_rows_out,
    int32_t* __restrict__ slots_out,
    int32_t* __restrict__ count_out,
    RemapT* __restrict__ remap_out,
    int64_t* __restrict__ graph_counters,
    int64_t* __restrict__ graph_unique_counters,
    float* __restrict__ route_counts,
    const int64_t* __restrict__ prefetch_expert,
    const int32_t* __restrict__ prefetch_count,
    int32_t prefetch_slot,
    int64_t* __restrict__ outcome_counters,
    const int64_t* __restrict__ miss_keys) {
  enum : uint8_t { kInactive = 0, kHit = 1, kPrefetched = 2, kResidual = 3 };
  __shared__ int64_t s_expert[kExpertRoutePlanDedupMaxRoutes];
  __shared__ int64_t s_key[kExpertRoutePlanDedupMaxRoutes];
  __shared__ uint8_t s_kind[kExpertRoutePlanDedupMaxRoutes];
  __shared__ uint8_t s_first[kExpertRoutePlanDedupMaxRoutes];
  __shared__ int32_t s_rank[kExpertRoutePlanDedupMaxRoutes];
  const int i = static_cast<int>(threadIdx.x);
  const bool active = i < routes;
  const int64_t expert = active ? static_cast<int64_t>(topk_ids[i]) : -1;
  const int64_t slot = active ? expert_to_slot[expert] : -1;
  const bool hot_hit = active && slot >= 0;
  const bool prefetched = active && !hot_hit && prefetch_count[0] == 1 && expert == prefetch_expert[0];
  const uint8_t kind = !active ? kInactive : hot_hit ? kHit : prefetched ? kPrefetched : kResidual;
  s_expert[i] = expert;
  s_key[i] = (miss_keys != nullptr && active) ? miss_keys[expert] : 0;
  s_kind[i] = kind;
  __syncthreads();
  int first = i;
  for (int j = 0; active && j < i; ++j) {
    if (s_expert[j] == expert) {
      first = j;
      break;
    }
  }
  s_first[i] = active && first == i;
  __syncthreads();
  const bool first_residual = s_first[i] && kind == kResidual;
  int miss_rows = 0;
  int before = 0;
  for (int j = 0; j < routes; ++j) {
    const bool fr = s_first[j] && s_kind[j] == kResidual;
    miss_rows += fr;
    before += fr && j < i;
  }
  if (first_residual) {
    int rank = before;
    if (miss_keys != nullptr) {
      rank = 0;
      for (int j = 0; j < routes; ++j)
        if (s_first[j] && s_kind[j] == kResidual)
          rank += s_key[j] > s_key[i] || (s_key[j] == s_key[i] && j < i);
    }
    s_rank[i] = rank;
  }
  __syncthreads();
  if (active) {
    const int rank = kind == kResidual ? s_rank[first] : 0;
    const int32_t destination =
        hot_hit ? static_cast<int32_t>(slot) : (prefetched ? prefetch_slot : scratch_base + rank);
    const int dest_pos = first_residual ? rank : miss_rows + (i - before);
    source_rows_out[dest_pos] = expert;
    slots_out[dest_pos] = destination;
    remap_out[i] = static_cast<RemapT>(destination);
    if (route_counts != nullptr) atomicAdd(route_counts + expert, 1.0f);
  }
  if (i == 0) {
    unsigned demand = 0, covered = 0, residual_routes = 0, unique_hits = 0, unique_misses = 0;
    for (int j = 0; j < routes; ++j) {
      demand += s_kind[j] >= kPrefetched;
      covered += s_kind[j] == kPrefetched;
      residual_routes += s_kind[j] == kResidual;
      unique_hits += s_first[j] && s_kind[j] == kHit;
      unique_misses += s_first[j] && s_kind[j] >= kPrefetched;
    }
    count_out[0] = miss_rows;
    if (graph_counters != nullptr) {
      auto* counters = reinterpret_cast<unsigned long long*>(graph_counters);
      atomicAdd(counters, static_cast<unsigned long long>(routes));
      atomicAdd(counters + 1, static_cast<unsigned long long>(demand));
    }
    if (graph_unique_counters != nullptr) {
      auto* unique_counters = reinterpret_cast<unsigned long long*>(graph_unique_counters);
      atomicAdd(unique_counters, static_cast<unsigned long long>(unique_hits));
      atomicAdd(unique_counters + 1, static_cast<unsigned long long>(unique_misses));
    }
    if (outcome_counters != nullptr) {
      const unsigned posted_rows = prefetch_count[0] == 1 ? 1u : 0u;
      const unsigned wasted_rows = posted_rows && covered == 0 ? 1u : 0u;
      auto* outcomes = reinterpret_cast<unsigned long long*>(outcome_counters);
      atomicAdd(outcomes, static_cast<unsigned long long>(covered));
      atomicAdd(outcomes + 1, static_cast<unsigned long long>(residual_routes));
      atomicAdd(outcomes + 2, static_cast<unsigned long long>(wasted_rows));
      atomicAdd(outcomes + 3, static_cast<unsigned long long>(posted_rows));
    }
  }
}

// `plan_unique_routes_gpu`'s arguments, for `plan_dedup_routes_kernel`: one block of 64 threads.
template <typename IdT, typename RemapT>
void plan_dedup_routes_gpu(
    tvm::ffi::TensorView topk_ids,
    tvm::ffi::TensorView expert_to_slot,
    int64_t scratch_base,
    tvm::ffi::TensorView source_rows_out,
    tvm::ffi::TensorView slots_out,
    tvm::ffi::TensorView count_out,
    tvm::ffi::TensorView remap_out,
    tvm::ffi::Optional<tvm::ffi::TensorView> graph_counters,
    tvm::ffi::Optional<tvm::ffi::TensorView> graph_unique_counters,
    tvm::ffi::Optional<tvm::ffi::TensorView> route_counts,
    tvm::ffi::TensorView prefetch_expert,
    tvm::ffi::TensorView prefetch_count,
    int64_t prefetch_slot,
    tvm::ffi::Optional<tvm::ffi::TensorView> outcome_counters,
    tvm::ffi::Optional<tvm::ffi::TensorView> miss_keys) {
  host::RuntimeCheck(
      0 < topk_ids.numel() && topk_ids.numel() <= kExpertRoutePlanDedupMaxRoutes, "topk_ids must hold 1-64 routes");
  const auto stream = host::LaunchKernel::resolve_device(topk_ids.device());
  auto ptr = [](auto& opt) { return opt.has_value() ? opt.value().data_ptr() : nullptr; };
  host::LaunchKernel(1, kExpertRoutePlanDedupMaxRoutes, stream)(
      plan_dedup_routes_kernel<IdT, RemapT>,
      static_cast<const IdT*>(topk_ids.data_ptr()),
      static_cast<const int64_t*>(expert_to_slot.data_ptr()),
      static_cast<int>(topk_ids.numel()),
      static_cast<int32_t>(scratch_base),
      static_cast<int64_t*>(source_rows_out.data_ptr()),
      static_cast<int32_t*>(slots_out.data_ptr()),
      static_cast<int32_t*>(count_out.data_ptr()),
      static_cast<RemapT*>(remap_out.data_ptr()),
      static_cast<int64_t*>(ptr(graph_counters)),
      static_cast<int64_t*>(ptr(graph_unique_counters)),
      static_cast<float*>(ptr(route_counts)),
      static_cast<const int64_t*>(prefetch_expert.data_ptr()),
      static_cast<const int32_t*>(prefetch_count.data_ptr()),
      static_cast<int32_t>(prefetch_slot),
      static_cast<int64_t*>(ptr(outcome_counters)),
      static_cast<const int64_t*>(ptr(miss_keys)));
}
```

If `host::RuntimeCheck` is not visible from this file's includes, add `#include "expert_stream/tensor_checks.h"`
the way `exl3/exl3_route_tables.cuh` reaches it, or use the include that file gets `RuntimeCheck` from.

- [ ] **Step 4: Wire the Python side**

`python/sglang/kernels/ops/moe/expert_route_plan.py`:
- Under `MAX_ROUTES = 32`, add `MAX_DEDUP_ROUTES = 64`.
- Make the module's wrappers
  `cuda_wrappers=[("plan_unique_routes_gpu", f"plan_unique_routes_gpu<{args}>"), ("plan_dedup_routes_gpu", f"plan_dedup_routes_gpu<{args}>")]`.
- Give `_validate_route_plan_inputs` a trailing `max_routes: int` parameter. Its route check becomes:
  ```python
  if not 0 < topk_ids.numel() <= max_routes:
      raise ValueError(f"topk_ids must hold 1-{max_routes} routes.")
  ```
- `plan_unique_routes_cuda` gains a trailing `dedup: bool = False`. It passes
  `MAX_DEDUP_ROUTES if dedup else MAX_ROUTES` to the validator and calls
  `(module.plan_dedup_routes_gpu if dedup else module.plan_unique_routes_gpu)(...)` with the same arguments.
- Add one docstring sentence: "``dedup`` plans several tokens' routes (up to 64) with the kernel that shares a
  scratch row between an expert's routes."

`python/sglang/srt/layers/moe/expert_route_plan.py`:
- Add `FUSED_MAX_DEDUP_ROUTES = 64` under `FUSED_MAX_ROUTES`.
- Replace `supports_fused_graph_routes`' body and docstring:

```python
def supports_fused_graph_routes(
    topk_ids: torch.Tensor, expert_to_slot: torch.Tensor, scratch_rows: int
) -> bool:
    """Whether `plan_graph_routes_fused` may serve this gather.

    One token row (``topk_ids.shape[0] == 1``) takes the one-warp kernel, up to 32 routes: a token's top-k ids are
    distinct (`torch.topk`, and the logical-to-physical remap maps distinct experts to distinct replicas), which
    that kernel relies on. Several token rows take the dedup kernel (``dedup=True``), up to 64 routes, which
    shares one scratch row between an expert's routes as `plan_graph_routes` does.
    """
    if not (
        topk_ids.is_cuda
        and topk_ids.dtype in (torch.int32, torch.int64)
        and topk_ids.ndim >= 1
        and expert_to_slot.dtype == torch.int64
        and expert_to_slot.device == topk_ids.device
    ):
        return False
    limit = FUSED_MAX_ROUTES if topk_ids.shape[0] == 1 else FUSED_MAX_DEDUP_ROUTES
    return 0 < topk_ids.numel() <= min(limit, scratch_rows)
```

- `plan_graph_routes_fused` gains a trailing `dedup: bool = False`, documented as "``dedup``: the routes are
  several tokens' (`supports_fused_graph_routes`), planned by the dedup kernel." It passes `dedup=dedup` to
  `plan_unique_routes_cuda`.
- Its first docstring line becomes: "`plan_graph_routes` in one fused kernel launch: one token's unique ids, or
  several tokens' with ``dedup``."

`python/sglang/srt/layers/moe/expert_stream.py`, in `_gather_graph`'s `plan_graph_routes_fused(...)` call, add
`dedup=topk_ids.shape[0] != 1,` after `miss_keys=self._plan_miss_keys,`. In the error raised for
`_plan_miss_keys` without the fused plan, replace "a BS1 gather" with "at most 32 routes of one token or 64 of
several".

- [ ] **Step 5: Run the tests and watch them pass**

Run the GPU command with Step 2's `<FILES>` and `<K>`.
Expected: all selected pass, about 310 tests, with a first-run JIT build of the planner module.

- [ ] **Step 6: Run both files whole and commit**

Run the GPU command with Step 2's `<FILES>` and `<K>` = `not nothing_matches_this`.
Expected: everything passes. That includes every pre-existing BS1 test, which proves the one-warp path is unchanged.

```bash
git add python/sglang/kernels/jit/csrc/moe/expert_route_plan.cuh python/sglang/kernels/ops/moe/expert_route_plan.py \
  python/sglang/srt/layers/moe/expert_route_plan.py python/sglang/srt/layers/moe/expert_stream.py
git commit -m "feat(expert-route-plan): a dedup kernel plans several tokens' routes (up to 64) in one launch" -m "<trailers>"
```

---

### Task 2: Route tables for M tokens (torch chain and the layer-fusion kernel)

**Files:**
- Modify: `python/sglang/srt/layers/quantization/exl3/fused_moe.py:75-93` (`route_tables`)
- Modify: `python/sglang/kernels/jit/csrc/moe/exl3/exl3_route_tables.cuh`
- Modify: `python/sglang/kernels/ops/moe/exl3_route_tables.py`
- Test: `test/registered/unit/layers/quantization/test_exl3_fused_moe.py`
- Test: `test/manual/dsv41/test_dsv41_layer_fusion_gpu.py`
- Test: `test/manual/dsv41/test_layer_fusion_launcher_checks_gpu.py`

**Interfaces:**
- Produces:
  - `route_tables(remap, expert_count, ones, weights, keep, token_sorted=None, top_k=1) -> (inv_order, weight_sorted, det)`.
    It sorts stably by slot. When `token_sorted` (int64 `[routes]`) is given, it writes `order // top_k` into it.
  - `exl3_moe_route_tables(remap, weights, keep, x, remap64_out, x16_out, out_zero, expert_count, inv_order, weight_sorted, det, cpu_lanes=None, dst_slots=None, cpu_out=0, cpu_part_stride=0, token_sorted_out=None)`.
    `x`, `x16_out` and `out_zero` are `[M, H]`. `remap` holds `M * top_k` ≤ 64 routes. `token_sorted_out` (int64
    `[routes]`) receives `rank -> route // top_k`.
  - FFI order of `exl3_moe_route_tables_gpu`: `remap, weights, keep, x, remap64_out, x16_out, out_zero, expert_count, inv_order, weight_sorted, det, token_sorted_out, cpu_lanes, dst_slots, cpu_out, cpu_part_stride`.

- [ ] **Step 1: Write the failing tests**

Add to `test_exl3_fused_moe.py`:

```python
def test_route_tables_rank_cross_token_duplicates_by_route_and_name_their_tokens():
    # Two tokens of top-3; slot 4 is routed by both.
    remap = torch.tensor([4, 1, 3, 2, 4, 0])
    count = torch.zeros(6, dtype=torch.long)
    token_sorted = torch.full((6,), -1, dtype=torch.long)
    weights = torch.tensor([0.5, 0.25, 0.125, 1.0, 2.0, 4.0])
    inv, ws, det = route_tables(
        remap, count, torch.ones(6, dtype=torch.long), weights, torch.tensor([1.0]), token_sorted=token_sorted, top_k=3
    )
    assert count.tolist() == [1, 1, 1, 1, 2, 0]
    assert inv.tolist() == [4, 1, 3, 2, 5, 0]  # slot order, route order within slot 4
    assert token_sorted.tolist() == [1, 0, 1, 0, 0, 1]
    assert ws.tolist() == [4.0, 0.25, 1.0, 0.125, 0.5, 2.0]
    assert det[0].tolist() == [0, 1, 2, 3, 4, 6] and det[2].tolist() == [1, 1, 1, 1, 1, 0]
```

In `test_dsv41_layer_fusion_gpu.py`, add a multi-token parity test next to `test_route_tables_match_the_torch_chain`:

```python
@pytest.mark.parametrize("tokens,top_k,slots,hidden", [(1, 6, 12, 256), (2, 6, 12, 256), (6, 6, 40, 4096), (8, 8, 70, 1024), (3, 5, 16, 1025)])
@pytest.mark.parametrize("remap_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("x_dtype", [torch.bfloat16, torch.float32])
def test_route_tables_match_the_torch_chain_for_several_tokens(tokens, top_k, slots, hidden, remap_dtype, x_dtype):
    """Each token routes top_k distinct slots; tokens share slots. Stable ranks make kernel and chain agree bit for
    bit, token_sorted included."""
    from sglang.kernels.ops.moe.exl3_route_tables import exl3_moe_route_tables
    from sglang.srt.layers.quantization.exl3.fused_moe import route_tables

    gen = torch.Generator().manual_seed(tokens * 131 + top_k * 7 + slots)
    dev = "cuda"
    routes = tokens * top_k
    for trial in range(10):
        remap = torch.cat([torch.randperm(slots, generator=gen)[:top_k] for _ in range(tokens)]).to(dev, remap_dtype)
        weights = (torch.rand(routes, generator=gen) * 3).to(dev, torch.bfloat16)
        keep = torch.tensor([[1.0, 0.0, 0.75][trial % 3]], device=dev)
        x = (torch.randn(tokens, hidden, generator=gen) * 40).to(dev, x_dtype)

        count_ref = torch.full((slots + 1,), 99, dtype=torch.int64, device=dev)
        ts_ref = torch.full((routes,), -3, dtype=torch.int64, device=dev)
        inv_ref, ws_ref, det_ref = route_tables(
            remap.long(), count_ref, torch.ones(routes, dtype=torch.int64, device=dev), weights, keep,
            token_sorted=ts_ref, top_k=top_k,
        )
        x16_ref = torch.empty(tokens, hidden, dtype=torch.float16, device=dev).copy_(x)

        remap64 = torch.full((routes,), -3, dtype=torch.int64, device=dev)
        x16 = torch.full((tokens, hidden), 7.0, dtype=torch.float16, device=dev)
        out = torch.full((tokens, hidden), 5.0, dtype=torch.float32, device=dev)
        count = torch.full((slots + 1,), 99, dtype=torch.int64, device=dev)
        inv = torch.full((routes,), -3, dtype=torch.int64, device=dev)
        ws = torch.full((routes,), 9.0, dtype=torch.float16, device=dev)
        det = torch.full((3, slots + 1), -3, dtype=torch.int64, device=dev)
        ts = torch.full((routes,), -3, dtype=torch.int64, device=dev)
        exl3_moe_route_tables(remap, weights, keep, x, remap64, x16, out, count, inv, ws, det, token_sorted_out=ts)

        assert torch.equal(remap64, remap.long())
        assert torch.equal(x16.view(torch.int16), x16_ref.view(torch.int16)), f"trial {trial}: x16"
        assert torch.equal(out, torch.zeros_like(out))
        assert torch.equal(count, count_ref), f"trial {trial}: expert_count"
        assert torch.equal(inv, inv_ref), f"trial {trial}: inv_order"
        assert torch.equal(ws.view(torch.int16), ws_ref.view(torch.int16)), f"trial {trial}: weight_sorted"
        assert torch.equal(det, det_ref), f"trial {trial}: det"
        assert torch.equal(ts, ts_ref), f"trial {trial}: token_sorted"
```

In `test_layer_fusion_launcher_checks_gpu.py`:
- In `_route_args`, insert `"token_sorted_out": torch.empty(0, dtype=torch.int64, device=CUDA),` after `"det"`.
  This is the new FFI position.
- Give `_route_args` a `tokens: int = 1` parameter, and size `x`, `x16_out` and `out_zero` as `(tokens, hidden)`.
- In `ROUTE_REFUSALS`, replace the `"routes_past_32"` entry and add four more:

```python
    "routes_past_64": (lambda: _route_args(routes=65), "remap must hold 1-64 routes"),
    "routes_not_a_multiple_of_tokens": (lambda: _route_args(routes=7, tokens=2), "multiple of the tokens"),
    "cpu_experts_with_two_tokens": (
        lambda: {**_route_args(routes=12, tokens=2), "cpu_lanes": torch.zeros(2, dtype=torch.int32, device=CUDA), "cpu_out": 16},
        "CPU experts run one token",
    ),
    "token_sorted_out_wrong_size": (
        lambda: {**_route_args(), "token_sorted_out": torch.zeros(5, dtype=torch.int64, device=CUDA)},
        "token_sorted_out: ",
    ),
    "x16_out_wrong_tokens": (
        lambda: {**_route_args(routes=12, tokens=2), "x16_out": torch.zeros(1, 64, dtype=torch.float16, device=CUDA)},
        "^x16_out: ",
    ),
```

- [ ] **Step 2: Run them and watch them fail**

Commit `test(exl3-route-tables): M-token route tables (failing)`. Then run the CPU command with `<FILES>` =
`test/registered/unit/layers/quantization/test_exl3_fused_moe.py` and `<K>` = `cross_token`.
Expected: FAIL with `TypeError: route_tables() got an unexpected keyword argument 'token_sorted'`.

Then run the GPU command with
`<FILES>` = `test/manual/dsv41/test_dsv41_layer_fusion_gpu.py test/manual/dsv41/test_layer_fusion_launcher_checks_gpu.py`
and `<K>` = `several_tokens or route_tables_launcher`.
Expected: FAIL. The parity test fails on the unexpected keyword `token_sorted_out`, or on the launcher's `x` shape
check. The refusal cases fail on the FFI argument count.

- [ ] **Step 3: The torch chain**

Replace `route_tables` in `fused_moe.py`:

```python
def route_tables(remap, expert_count, ones, weights, keep, token_sorted=None, top_k=1):
    """Fill ``expert_count`` from ``remap``; return (inv_order, weight_sorted fp16, det tables).

    Routes are ranked by slot, ties by route index (a stable sort: several tokens may route one slot); with
    ``token_sorted`` given, each rank's token, ``route // top_k``, is written into it.
    ``keep`` (fp32 [1]) scales every route weight and, when 0, empties ``expert_count``,
    so a dropped layer runs no expert.
    ``det`` is exllamav3's device-built deterministic table stack
    ``[expert_start, expert_start, count > 0]``.
    """
    expert_count.zero_().index_add_(0, remap, ones)
    order = torch.argsort(remap, stable=True)
    inv_order = torch.empty_like(order).scatter_(
        0, order, torch.arange(order.numel(), device=order.device)
    )
    if token_sorted is not None:
        torch.floor_divide(order, top_k, out=token_sorted)
    weight_sorted = (weights[order].float() * keep).to(torch.float16)
    # A dropped layer runs no expert: nothing reads rows that may be half written.
    expert_count.mul_((keep > 0).to(torch.int64))
    expert_start = torch.cumsum(expert_count, 0) - expert_count
    det = torch.stack([expert_start, expert_start, (expert_count > 0).long()])
    return inv_order, weight_sorted, det
```

- [ ] **Step 4: The kernel**

In `exl3_route_tables.cuh`:
- Kernel signature:
  - Rename the `top_k` parameter to `routes`.
  - Add `int64_t tokens` after `hidden`.
  - Add `int64_t* __restrict__ token_sorted_out` after `det`.
  - Update every use of `top_k` in the body to `routes`.
- The staging loop becomes
  `for (int64_t i = tid; i < tokens * hidden; i += stride) {`. The body stays the same: the seed reads
  `cpu_out + part * part_stride + i`, and CPU experts run one token, so there `i < hidden`.
- In the `if (tid < routes)` block, after `inv_order[tid] = rank;`, add:
  ```cuda
  if (token_sorted_out != nullptr) token_sorted_out[rank] = tid / (routes / tokens);
  ```
- Update the comment block. The fused MoE's input is "the M tokens' `[M, hidden]` rows". Ranks are stable, so
  "a BS1 remap does not [share a slot]" is replaced by "torch's stable argsort ranks the same way, so the two agree
  for any remap, several tokens sharing a slot included".

Launcher `exl3_moe_route_tables_gpu`:
- Add `tvm::ffi::TensorView token_sorted_out` after `det`.
- `constexpr int64_t kMaxRoutes = 64;`. Add `auto M_ = SymbolicSize{"tokens"};`.
- The matchers for `x`, `x16_out` and `out_zero` become `TensorMatcher({M_, H_})`.
- Add, after the `det` check:
  ```cpp
  expert_stream::verify_named(
      "token_sorted_out", TensorMatcher({-1}).with_dtype<int64_t>().with_device<kDLCUDA>(device), token_sorted_out);
  RuntimeCheck(0 < K_.unwrap() && K_.unwrap() <= kMaxRoutes, "remap must hold 1-64 routes");
  RuntimeCheck(K_.unwrap() % M_.unwrap() == 0, "remap's routes must be a multiple of the tokens (x's rows)");
  RuntimeCheck(
      token_sorted_out.size(0) == 0 || token_sorted_out.size(0) == K_.unwrap(),
      "token_sorted_out: one entry per route, or empty");
  ```
  The old `1-32` check is deleted.
- After `cpu_on` is computed:
  `RuntimeCheck(!cpu_on || M_.unwrap() == 1, "CPU experts run one token (x has one row)");`.
- `hidden` becomes `x.size(1)` and `tokens` becomes `x.size(0)`. The work size is
  `max(tokens * hidden, columns)`.
- Pass `tokens` and
  `token_sorted_out.size(0) ? static_cast<int64_t*>(token_sorted_out.data_ptr()) : nullptr`
  in the kernel's new positions.
- Update the `\brief` comment: "`x`, `x16_out` and `out_zero` are the M tokens' `[M, hidden]` rows".

In `exl3_route_tables.py`:
- `exl3_moe_route_tables` gains a trailing `token_sorted_out: Optional[torch.Tensor] = None`. It passes
  `token_sorted_out if token_sorted_out is not None else _empty_i64(det.device)` right after `det`.
- Add `_empty_i64`, the int64 twin of `_empty_i32`, with its own `_EMPTY_I64` cache.
- Docstring: `x` and the staging buffers are `[M, H]`; `token_sorted_out` receives each rank's token.

- [ ] **Step 5: Run the tests and watch them pass**

Run the CPU command with `<FILES>` = `test/registered/unit/layers/quantization/test_exl3_fused_moe.py` and `<K>` =
`not nothing_matches_this`.
Expected: all pass.

GPU: run the command with Step 2's `<FILES>` and `<K>` = `not nothing_matches_this`.
Expected: all pass, including the pre-existing one-token parity and refusal cases.

- [ ] **Step 6: Commit**

```bash
git add python/sglang/srt/layers/quantization/exl3/fused_moe.py python/sglang/kernels/jit/csrc/moe/exl3/exl3_route_tables.cuh \
  python/sglang/kernels/ops/moe/exl3_route_tables.py
git commit -m "feat(exl3-route-tables): route tables for M tokens: stable ranks, each rank's token, up to 64 routes" -m "<trailers>"
```

---

### Task 3: `Exl3FusedMoE` over M tokens

**Files:**
- Modify: `python/sglang/srt/layers/quantization/exl3/fused_moe.py` (`Exl3FusedMoE`, `exl3_fused_moe_for`, the
  module docstring)
- Test: `test/registered/unit/layers/quantization/test_exl3_fused_moe.py`

**Interfaces:**
- Consumes: Task 2's `route_tables(..., token_sorted=, top_k=)` and
  `exl3_moe_route_tables(..., token_sorted_out=)`.
- Produces:
  - `Exl3FusedMoE(tensors, slots, hidden, inter, top_k, device, tokens=1)`.
  - `.run(x [M, H], topk_weights [M*top_k], remap [M*top_k], keep, act_limit, cpu=None) -> out [M, H] fp32 view`,
    for `1 <= M <= tokens`.
  - `exl3_fused_moe_for` builds it with `tokens = graph_gather_rows // layer.top_k`.

- [ ] **Step 1: Write the failing tests**

Add to `test_exl3_fused_moe.py`:

```python
class _FakeExt:
    """exl3_ext() stand-in: records what the fused MoE hands exllamav3."""

    def __init__(self):
        self.moe, self.gather = [], []

    def exl3_moe_max_concurrency(self, device):
        return 2

    def exl3_moe(self, *args):
        self.moe.append(args)

    def exl3_moe_gather(self, *args):
        self.gather.append(args)


def _fused(monkeypatch, tokens, hidden=8, slots=10):
    from sglang.srt.layers.quantization.exl3 import fused_moe as module

    ext = _FakeExt()
    monkeypatch.setattr(module, "exl3_ext", lambda: ext)
    module._SHARED_TEMPS.clear()
    tensors = {n: torch.zeros((slots, 2 if n.startswith("w13") else 1, hidden), dtype=torch.int16) for n in NAMES}
    tensors["w13_trellis"] = torch.zeros((slots, 2, 16 * 3), dtype=torch.int16)
    tensors["w2_trellis"] = torch.zeros((slots, 1, 16 * 3), dtype=torch.int16)
    fused = module.Exl3FusedMoE(tensors, slots, hidden=hidden, inter=hidden, top_k=6, device="cpu", tokens=tokens)
    return fused, ext


def test_fused_moe_hands_exllamav3_m_tokens_with_their_token_index(monkeypatch):
    fused, ext = _fused(monkeypatch, tokens=6)
    remap = torch.tensor([0, 1, 2, 3, 4, 5, 3, 4, 5, 6, 7, 8])  # token 1 shares slots 3-5 with token 0
    out = fused.run(torch.ones((2, 8)), torch.ones(12), remap, torch.ones(1), 10.0)
    args = ext.moe[-1]
    assert args[0].shape == (2, 8) and args[1].shape == (2, 8) and out.shape == (2, 8)
    assert args[2][:9].tolist() == [1, 1, 1, 2, 2, 2, 1, 1, 1]
    assert args[3].tolist() == [0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1]  # token of each rank
    assert args[4].shape == (12,) and args[30].shape == (12, 8)
    assert args[29] == -1  # num_active: launch-sized for any count of active slots
    assert ext.gather[-1][2].shape == (12,)


def test_fused_moe_one_token_is_todays_launch(monkeypatch):
    fused, ext = _fused(monkeypatch, tokens=6)
    fused.run(torch.ones((1, 8)), torch.ones(6), torch.tensor([5, 1, 3, 0, 2, 4]), torch.ones(1), 10.0)
    args = ext.moe[-1]
    assert args[0].shape == (1, 8) and args[3].tolist() == [0] * 6 and args[29] == 6 and args[30].shape == (6, 8)


def test_fused_moe_drops_every_token_when_keep_is_zero(monkeypatch):
    fused, ext = _fused(monkeypatch, tokens=2)
    out = fused.run(torch.ones((2, 8)), torch.ones(12), torch.arange(12) % 9, torch.zeros(1), 10.0)
    assert ext.moe[-1][2].sum().item() == 0 and torch.equal(out, torch.zeros((2, 8)))


@pytest.mark.parametrize("m", [0, 3])
def test_fused_moe_refuses_more_tokens_than_its_buffers(monkeypatch, m):
    fused, _ = _fused(monkeypatch, tokens=2)
    with pytest.raises(ValueError, match="1-2 tokens"):
        fused.run(torch.ones((m, 8)), torch.ones(6 * m), torch.zeros(6 * m, dtype=torch.long), torch.ones(1), 10.0)


def test_fused_moe_refuses_cpu_experts_for_several_tokens(monkeypatch):
    fused, _ = _fused(monkeypatch, tokens=2)
    with pytest.raises(RuntimeError, match="one token"):
        fused.run(torch.ones((2, 8)), torch.ones(12), torch.zeros(12, dtype=torch.long), torch.ones(1), 10.0,
                  cpu=(None, None, 16, 0))


def test_fused_moe_for_sizes_tokens_from_the_gather_rows(monkeypatch):
    from types import SimpleNamespace

    from sglang.srt.layers.quantization.exl3 import fused_moe as module

    calls = []
    monkeypatch.setattr(module, "Exl3FusedMoE", lambda tensors, slots, **kw: calls.append(kw) or object())
    layer = torch.nn.Module()
    layer.top_k = 6
    streamer = _stub_streamer(SimpleNamespace(name="other"), graph_gather_rows=36, scratch_rows=36)
    streamer.hot_cache.device = torch.device("cpu")
    streamer.hot_cache.tensors = {"w13_suh": torch.zeros((39, 2, 8)), "w2_suh": torch.zeros((39, 1, 8))}
    module.exl3_fused_moe_for(layer, streamer)
    assert calls[-1]["tokens"] == 6 and calls[-1]["top_k"] == 6
    del layer._exl3_fused_moe
    streamer.graph_gather_rows = 6 * 17
    streamer.hot_cache.scratch_rows = 6 * 17
    with pytest.raises(ValueError, match="row tile"):
        module.exl3_fused_moe_for(layer, streamer)
```

Expected values in the first test:
- `remap` sorted stably gives ranks for slots 0,1,2,3,3,4,4,5,5,6,7,8, from routes 0,1,2,3,6,4,7,5,8,9,10,11.
- Those routes' tokens (`route // 6`) are `0,0,0,0,1,0,1,0,1,1,1,1`.
- `expert_count[:9]` is `[1,1,1,2,2,2,1,1,1]`.

- [ ] **Step 2: Run them and watch them fail**

Commit `test(exl3-fused-moe): M-token in-graph MoE (failing)`. Then run the CPU command with `<FILES>` =
`test/registered/unit/layers/quantization/test_exl3_fused_moe.py` and `<K>` = `not nothing_matches_this`.
Expected: FAIL with `TypeError: Exl3FusedMoE.__init__() got an unexpected keyword argument 'tokens'`. The sizing
test fails on `KeyError: 'tokens'`.

- [ ] **Step 3: Implement**

In `Exl3FusedMoE.__init__`:
- Add a `tokens: int = 1` parameter.
- Store `self.top_k = top_k` and `self.tokens = tokens`, and set `routes = tokens * top_k`.
- `ones`, `token_sorted`, `remap64`, `inv_order` and `weight_sorted` become `[routes]`.
- `scratch` becomes `[routes, hidden]`, and `out` and `x16` become `[tokens, hidden]`.
- The temp buffers stay `ROW_TILE` rows: a slot holds at most one row per token.

`_fused_route_tables(self, x, topk_weights, remap, keep, m, cpu=None)` slices to the call's M:

```python
        routes = m * self.top_k
        exl3_moe_route_tables(
            remap.contiguous(),
            topk_weights.contiguous(),
            keep,
            x.contiguous(),
            self.remap64[:routes],
            self.x16[:m],
            self.out[:m],
            self.expert_count,
            self.inv_order[:routes],
            self.weight_sorted[:routes],
            self.det,
            cpu_lanes=cpu_lanes,
            dst_slots=dst_slots,
            cpu_out=cpu_out,
            cpu_part_stride=cpu_part_stride,
            token_sorted_out=self.token_sorted[:routes],
        )
        return self.remap64[:routes], self.inv_order[:routes], self.weight_sorted[:routes], self.det
```

Replace `run`:

```python
    def run(self, x, topk_weights, remap, keep, act_limit: float, cpu=None) -> torch.Tensor:
        """x [M, H], 1 <= M <= tokens, any float dtype; topk_weights [M * top_k]; remap [M * top_k] slots, int64
        (int32 too with layer fusion); keep fp32 [1]. Returns the fp32 [M, H] output, a view of this object's buffer.

        ``cpu`` = (cpu_lanes, dst_slots, cpu_out address, part stride), CPU experts only, one token: the routes the
        CPU computed are left out and the partial sums CC flagged seed the output (exl3_route_tables.cuh). Layer
        fusion only."""
        m = x.shape[0]  # a host-side shape read: capture-safe
        if not 1 <= m <= self.tokens:
            raise ValueError(f"exl3 in-graph MoE runs 1-{self.tokens} tokens, not {m}")
        if cpu is not None and m != 1:
            raise RuntimeError(f"CPU experts run one token, not {m}")
        if cpu is not None and not self.layer_fusion:
            raise RuntimeError("CPU experts need SGLANG_DSV41_ENABLE_LAYER_FUSION: only its route tables leave CPU routes out")
        routes = m * self.top_k
        x16, out, token_sorted, scratch = self.x16[:m], self.out[:m], self.token_sorted[:routes], self.scratch[:routes]
        if self.layer_fusion:
            remap, inv_order, weight_sorted, det = self._fused_route_tables(x, topk_weights, remap, keep, m, cpu)
        else:
            x16.copy_(x)
            inv_order, weight_sorted, det = route_tables(
                remap, self.expert_count, self.ones[:routes], topk_weights, keep, token_sorted=token_sorted,
                top_k=self.top_k,
            )
            out.zero_()
        t = self.tables
        self.ext.exl3_moe(
            x16, out, self.expert_count, token_sorted, weight_sorted,
            self.temp_state_g, self.temp_state_u, self.temp_intermediate_g, self.temp_intermediate_u,
            ACT_SILU, self.bits["gate"], self.bits["up"], self.bits["down"],
            t["gate_trellis"], t["gate_suh"], t["gate_svh"],
            t["up_trellis"], t["up_suh"], t["up_svh"],
            t["down_trellis"], t["down_suh"], t["down_svh"],
            False, True, False, True, False, True,
            float(act_limit),
            # One token routes NUM_ACTIVE distinct slots; several share slots unpredictably, and exllamav3 takes
            # any count by ticket, so -1 sizes the launch for the most it can hold.
            NUM_ACTIVE if m == 1 else -1,
            scratch, det[0], 1, ROW_TILE, 16,
        )
        self.ext.exl3_moe_gather(
            out, scratch, remap, inv_order,
            det[1, : self.slots], det[0, : self.slots], det[2, : self.slots], weight_sorted,
        )
        return out
```

Keep the original one-argument-per-line formatting of the `exl3_moe` call. The compressed form above is for
reading only; only the four changed arguments differ: `x16`, `out`, `token_sorted` and the `num_active`
expression, plus `scratch`.

Replace `exl3_fused_moe_for`'s row check:

```python
        rows = streamer.graph_gather_rows
        top_k = layer.top_k
        # The route buffers hold up to `tokens` tokens' top_k routes. A slot holds at most one route per token, and
        # the fused kernel skips a slot with more rows than its ROW_TILE temp tile.
        if rows % top_k:
            raise ValueError(
                f"exl3 in-graph MoE needs graph_gather_rows ({rows}) to be a multiple of top_k ({top_k})"
            )
        tokens = rows // top_k
        if tokens > ROW_TILE:
            raise ValueError(f"exl3 in-graph MoE: {tokens} tokens exceed the fused kernel's {ROW_TILE}-row tile")
```

Pass `top_k=top_k, tokens=tokens` to `Exl3FusedMoE`, replacing `top_k=rows`. The DIRECT and scratch checks keep
comparing against `rows` (one slot per route). The DIRECT message keeps its "DIRECT needs at least top_k" prefix,
followed by "resident slots per route".

Update the module docstring:
- "for in-graph decode at BS1" becomes "for in-graph decode and verify, 1-16 tokens".
- "one hot-cache slot per route ... distinct at BS1" becomes "one hot-cache slot per route (tokens may share a
  slot; a token's own routes are distinct)".
- Update `ROW_TILE`'s comment to "fused-kernel rows per slot tile: a slot holds one row per token".

- [ ] **Step 4: Run the tests and watch them pass**

Run the CPU command with the same `<FILES>` and `<K>`.
Expected: all pass, the pre-existing ones included. The `graph_gather_rows=4` case still matches "top_k".

- [ ] **Step 5: Commit**

```bash
git add python/sglang/srt/layers/quantization/exl3/fused_moe.py test/registered/unit/layers/quantization/test_exl3_fused_moe.py
git commit -m "feat(exl3-fused-moe): the in-graph MoE runs 1-16 tokens over shared slots, each route with its token" -m "<trailers>"
```

---

### Task 4: The real-row gate, and the numbers

**Files:**
- Create: `test/manual/dsv41/test_exl3_fused_moe_multitoken_gpu.py`
- Modify: `DSV41_REFERENCE.md` (new `### 33.6` after §33.5, before `## Sources`)

**Interfaces:**
- Consumes: Task 3's `Exl3FusedMoE(..., tokens=)`. It reuses `_load_slots`, `_views` and `_reference` from
  `test/manual/dsv41/test_exl3_moe_probe_gpu.py`, imported via `sys.path` from `test/manual/dsv41`.

- [ ] **Step 1: Write the test**

```python
"""D2-1 gate: the in-graph EXL3 MoE over M tokens that share slots, on real layer rows (GPU).

16 real experts of one layer fill 16 slots. Every M in (1, 2, 4, 6) draws 8 route sets of top-6 slots per token,
so tokens share slots. Bars, per token:
  * rel(fused) <= 1.2e-2 and rel(fused) <= 2 * rel(exl3_moe_loop) + 1e-3, against the probe's fp32 reference;
  * the layer-fusion and torch-chain route tables give the same output bitwise, and a second eager run repeats it;
  * a graph captured at M replays rewritten routes and inputs bitwise equal to an eager run;
  * one object serving M = 6 then M = 1 gives what a fresh object gives at M = 1.
Reports eager and replay microseconds per M to DSV41_MULTITOKEN_OUT (JSON) when set.
Env: DSV41_EXL3_DIR, DSV41_PROBE_LAYER (as the probe).
"""

import json
import os
import sys
import time

import pytest
import torch

sys.path.insert(0, os.path.dirname(__file__))
from test_exl3_moe_probe_gpu import ACT_LIMIT, LAYER, REL_BOUND, _load_slots, _reference, _rel, _views  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

EXPERTS = list(range(0, 384, 24))  # 16 experts -> 16 slots
TOP_K = 6
TOKENS = 6
WIDTHS = (1, 2, 4, 6)
ROUTE_SETS = 8
OUT = os.environ.get("DSV41_MULTITOKEN_OUT")


@pytest.fixture(scope="module")
def rows():
    import test_exl3_moe_probe_gpu as probe

    device = torch.device("cuda", torch.cuda.current_device())
    saved, probe.EXPERTS = probe.EXPERTS, EXPERTS  # _load_slots reads the module global
    try:
        tensors = _load_slots(device)
    finally:
        probe.EXPERTS = saved  # the probe test may run later in the same session
    return tensors, [_views(tensors, s) for s in range(len(EXPERTS))]


def _fused(tensors, layer_fusion):
    from sglang.srt.environ import envs
    from sglang.srt.layers.quantization.exl3.fused_moe import Exl3FusedMoE

    with envs.SGLANG_DSV41_ENABLE_LAYER_FUSION.override(layer_fusion):
        return Exl3FusedMoE(
            tensors, len(EXPERTS), hidden=tensors["w13_suh"].shape[-1], inter=tensors["w2_suh"].shape[-1],
            top_k=TOP_K, device=tensors["w13_suh"].device, tokens=TOKENS,
        )


def _inputs(gen, m, hidden, device):
    remap = torch.cat([torch.randperm(len(EXPERTS), generator=gen)[:TOP_K] for _ in range(m)]).to(device)
    weights = torch.softmax(torch.randn(m, TOP_K, generator=gen), -1).reshape(-1).to(device)
    x16 = (torch.randn((m, hidden), generator=gen) * 0.5).to(device, torch.float16)
    return x16, weights, remap


def _run(fused, x16, weights, remap):
    keep = torch.ones(1, device=x16.device)
    return fused.run(x16, weights, remap if fused.layer_fusion else remap.long(), keep, ACT_LIMIT).clone()


def test_multitoken_fused_moe(rows):
    from sglang.srt.layers.quantization.exl3.ops import exl3_moe_loop

    tensors, views = rows
    hidden = tensors["w13_suh"].shape[-1]
    device = tensors["w13_suh"].device
    w13, w2 = [v[0] for v in views], [v[1] for v in views]
    chain, fusion = _fused(tensors, False), _fused(tensors, True)
    gen = torch.Generator().manual_seed(4321)
    report = {"layer": LAYER, "experts": EXPERTS, "widths": {}}
    failures = []
    for m in WIDTHS:
        entry = {"route_sets": []}
        for _ in range(ROUTE_SETS):
            x16, weights, remap = _inputs(gen, m, hidden, device)
            a, b, again = _run(chain, x16, weights, remap), _run(fusion, x16, weights, remap), _run(fusion, x16, weights, remap)
            loop = exl3_moe_loop(x16, weights.view(m, TOP_K), remap.view(m, TOP_K), w13, w2, ACT_LIMIT).float()
            if not torch.equal(a, b):
                failures.append(f"M={m}: layer fusion != torch chain")
            if not torch.equal(b, again):
                failures.append(f"M={m}: eager rerun differs")
            for t in range(m):
                ref = _reference(x16[t : t + 1], weights.view(m, TOP_K)[t], remap.view(m, TOP_K)[t], views)
                rf, rl = _rel(a[t : t + 1], ref), _rel(loop[t : t + 1], ref)
                entry["route_sets"].append({"token": t, "rel_fused": rf, "rel_loop": rl})
                if not (rf <= REL_BOUND and rf <= 2 * rl + 1e-3):
                    failures.append(f"M={m} token {t}: rel {rf:.4g} (loop {rl:.4g})")
        # Capture at M over static inputs, replay after rewriting them in place.
        x_s, w_s, r_s = _inputs(gen, m, hidden, device)
        keep = torch.ones(1, device=device)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            fusion.run(x_s, w_s, r_s, keep, ACT_LIMIT)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out_s = fusion.run(x_s, w_s, r_s, keep, ACT_LIMIT)
        for _ in range(4):
            x_n, w_n, r_n = _inputs(gen, m, hidden, device)
            x_s.copy_(x_n), w_s.copy_(w_n), r_s.copy_(r_n)
            graph.replay()
            replayed = out_s.clone()
            if not torch.equal(replayed, _run(fusion, x_s, w_s, r_s)):
                failures.append(f"M={m}: replay != eager")
        torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(100):
            fusion.run(x_s, w_s, r_s, keep, ACT_LIMIT)
        torch.cuda.synchronize()
        entry["eager_us"] = (time.perf_counter() - started) * 1e4
        started = time.perf_counter()
        for _ in range(100):
            graph.replay()
        torch.cuda.synchronize()
        entry["replay_us"] = (time.perf_counter() - started) * 1e4
        entry["max_rel_fused"] = max(r["rel_fused"] for r in entry["route_sets"])
        report["widths"][str(m)] = entry
    # A wider run leaves nothing behind for a narrower one.
    x16, weights, remap = _inputs(gen, 1, hidden, device)
    _run(fusion, *_inputs(gen, TOKENS, hidden, device))
    if not torch.equal(_run(fusion, x16, weights, remap), _run(_fused(tensors, True), x16, weights, remap)):
        failures.append("M=1 after M=6 differs from a fresh object")
    report["failures"] = failures
    if OUT:
        with open(OUT, "w") as f:
            json.dump(report, f, indent=2)
    print(json.dumps({m: (e["max_rel_fused"], round(e["eager_us"], 1), round(e["replay_us"], 1))
                      for m, e in report["widths"].items()}))
    assert not failures, failures
```

- [ ] **Step 2: Run it**

Commit `test(exl3-fused-moe): real-row gate for M tokens` and push. Then, on divix01, take the disk lock first
(the test reads 16 rows from the checkpoint), then the GPU lock:

```bash
ssh divix01 'cd /data/models/slang/nvfp4-work/wt-dsv41-dspark-graph && git fetch -q origin && git checkout -q --detach origin/dsv41-dspark-graph \
  && export PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 G=/data/models/slang/nvfp4-work/cc-expert-prediction/analysis/dsv41-dspark/graph-verify \
  && /data/models/slang/.venv/bin/python -c "import sglang; print(sglang.__file__)" \
  && DSV41_MULTITOKEN_OUT=$G/d2-1-multitoken.json flock /data/models/slang/nvfp4-work/rowimg-disk.lock flock /data/models/slang/nvfp4-work/cc-gpu.lock taskset -c 32-63 \
     /data/models/slang/.venv/bin/python -m pytest test/manual/dsv41/test_exl3_fused_moe_multitoken_gpu.py test/manual/dsv41/test_exl3_moe_probe_gpu.py -q -s -p no:randomly > $G/d2-1-multitoken.log 2>&1; echo EXIT=${PIPESTATUS[0]}; tail -6 $G/d2-1-multitoken.log'
```

Expected: `EXIT=0`, 2 passed. The printed line maps each M to (max rel, eager µs, replay µs). The probe passing
again shows BS1 is intact.

If a bar fails, use superpowers:systematic-debugging. Do not loosen a bar.
- A torch-chain/fusion mismatch is a Task 2 bug.
- A parity failure only at M > 1 points at `token_sorted` or the slices.
- A replay mismatch points at a buffer that is not static.

- [ ] **Step 3: Record it**

Add `### 33.6 D2-1: the route plan and the in-graph MoE over M tokens (2026-10-05)` to `DSV41_REFERENCE.md`, after
§33.5 and before `## Sources`. It covers:
- what changed (the dedup planner kernel, M-token route tables, `Exl3FusedMoE(tokens=)`), with the three commits;
- that exllamav3's `exl3_moe` needed no change, and why: `token_sorted`, tickets, `num_active` only sizes the launch.
  This closes §33.3 item 4's "Unverified" line;
- the gate's table from `d2-1-multitoken.json`, with columns M, max rel_fused, eager µs and replay µs;
- the replay-vs-M=1 ratio as the first measured input to §33.5's "verify GPU ms" (MoE kernels only, one layer);
- `num_active = -1` at M > 1 as untuned;
- what D2-1 does not do: the record/wire at W, DIRECT at 2W, and CPU experts (D2-2 to D2-4).

```bash
git add test/manual/dsv41/test_exl3_fused_moe_multitoken_gpu.py DSV41_REFERENCE.md
git commit -m "docs(dsv41): section 33.6 -- the in-graph EXL3 MoE over M tokens, gated on real rows" -m "<trailers>"
git push -q origin dsv41-dspark-graph
```
