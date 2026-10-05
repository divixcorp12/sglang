# DSpark graphed verify D2-3: the target verify in the decode graph, end to end, with CPU experts off

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** DSpark's target verify runs in the breakable decode graph on DSV4.1 EXL3 at W miss lanes. A verify whose
expert gather overflowed is re-run eagerly before anything reads it. One measurement run on divix01 then gives the
real verify GPU time and overflow rate that §33.5 says decide v2.

**Architecture:** This is D2-3 of four v2 plans. D2-1 made the planner and the in-graph MoE multi-token. D2-2 added
the miss width W and the sticky device `overflow_flag`. D2-4 adds multi-token CPU experts.

D2-3 has six parts:
- **The manager** reads and clears the flag after each graphed verify, and can suspend its graph gather for one
  forward.
- **The DSpark verify** re-runs a flagged verify with the decode graph and the narrowed gather both off. The re-run
  takes the EXL3 eager MoE (`_apply_streamed`), the path DSpark verify ran on before D2.
- **Startup.** The verify epilogue (an in-graph accept that writes draft KV) is not built while the gather is
  narrowed. The draft's graphs are not captured for an EXL3 draft.
- **The copy-engine barrier** counts graphed verifies.
- **The gate** admits a DSpark verify under the breakable graph with that configuration.
- **A driver** runs eager, graphed, and graphed with every verify re-run, and summarizes the three.

**Tech Stack:** Python, torch, pytest, CUDA graphs (breakable backend). GPU runs on divix01.

**Spec:**
- `DSV41_REFERENCE.md` §33.3 items 6-8: the small items, the DSpark side and the gate.
- §33.5: measurement 1, the verify GPU ms, and the overflow rate that sets W.
- §33.7: what D2-2 left to this plan.
  - Read and clear the flag, and re-verify.
  - A flagged forward reads slot 0 and can output NaN. Everything it wrote must be discarded.
  - The protect list (see the finding below).
  - Choosing W.

**Finding, from the code read for this plan: the protect list needs no change.**
- The post kernel's protect list keeps the first `Wire::kLanes` distinct routes (`lease_kernels.cuh:164-169`).
- With no overflow, every route is either a VRAM hit or a lane. Hits are skipped by `tier.hot`, and lanes enter
  `wanted` in `collect_wanted_locked` (`host/ram_tier.h:1634-1647`) on their own.
- The routes the truncation drops are therefore exactly an overflowed verify's clamped misses, and that verify is
  re-run. §33.8 records this instead of widening the wire.

**Finding: why the re-run must suspend the narrowed gather.**
- `ExpertStreamer.gather` and `serves_graph_gather` choose the graph gather by route count alone
  (`expert_stream.py:1286-1288`, `:2090`), not by whether a graph is running.
- An eager re-run of the same verify would therefore go through `_gather_graph`, overflow again, and set the flag
  again.

## Global Constraints

- **BS1 is unchanged.**
  - Without speculation every new hook is inert. With speculation under an eager decode (`--cuda-graph-backend-decode
    disabled`), DSpark behaves exactly as today.
  - The re-verify runs only when the manager has a narrowed gather and the forward replayed a graph.
- **CPU experts stay off for a target verify.** `SGLANG_DSV41_CPU_EXPERTS=1` with speculation stays refused by the
  gate (§33.3 item 5). The draft's CPU experts (`SGLANG_DSV41_ENABLE_DSPARK_CPU_EXPERTS`) are allowed, as today.
- **A graphed verify's output is used only if `overflow_flag` read 0 after it.**
  - A flagged verify is re-run with the decode graph and the narrowed gather both off.
  - Nothing reads its logits, its epilogue buffers or its draft KV. While the gather is narrowed the epilogue is not
    built.
- **One host read per graphed verify.** That is the flag's `.item()`, made after the target forward and before the
  grammar mask and the accept. The accept already synchronizes with the host. Nothing else adds a sync to the graph
  path.
- **Do not edit `model_runner.py`** (a frozen core file). The re-verify hooks into
  `speculative/dspark_components/dspark_verify.py`. The force-eager switch lives on `DecodeCudaGraphRunner`
  (`model_executor/runner/decode_cuda_graph_runner.py`, not frozen).
- **Speculative identifiers follow `.claude/skills/speculative-naming/SKILL.md`.** Counters end in `_ct`:
  `graphed_verify_ct`, `verify_overflow_ct`.
- **Env vars follow `.claude/skills/env-var-conventions/SKILL.md`.** The one new variable is the test-only
  `SGLANG_TEST_DSPARK_FORCE_REVERIFY = EnvBool(False)`, placed beside the other `SGLANG_DSPARK_*` entries in
  `environ.py` and read through `envs.X.get()`.
- **divix01 protocol** (`.claude/rules/divix01-run-protocol.md`):
  - Push the branch, then `git fetch` and `git checkout --detach origin/dsv41-dspark-graph` in
    `/data/models/slang/nvfp4-work/wt-dsv41-dspark-graph`.
  - Run with `PYTHONPATH=$PWD/python`, and print `sglang.__file__` first.
  - Read `${PIPESTATUS[0]}`, never a pipe's status.
  - CPU jobs run under `taskset -c 0-63`.
  - GPU work runs under `flock /data/models/slang/nvfp4-work/cc-gpu.lock taskset -c 32-63`.
  - A job that also reads checkpoint rows (the Task 6 arms) takes `rowimg-disk.lock` first.
  - EXL3 GPU tests need `SGLANG_EXL3_SRC=/data/models/slang/nvfp4-work/exllamav3`.
- **Every commit ends with:**
  ```
  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01Y3EXKJdMpP7nwX3qcBi8Vq
  ```

The laptop has no torch, so every test runs on divix01. The two commands used below are written out once; `<FILES>`
and `<K>` vary:

```bash
# CPU
git push -q origin dsv41-dspark-graph && ssh divix01 'cd /data/models/slang/nvfp4-work/wt-dsv41-dspark-graph \
  && git fetch -q origin && git checkout -q --detach origin/dsv41-dspark-graph \
  && export PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 \
  && taskset -c 0-63 /data/models/slang/.venv/bin/python -c "import sglang; print(sglang.__file__)" \
  && taskset -c 0-63 /data/models/slang/.venv/bin/python -m pytest <FILES> -q -p no:randomly -k "<K>" > /tmp/d23.log 2>&1; echo EXIT=${PIPESTATUS[0]}; tail -15 /tmp/d23.log'

# GPU
git push -q origin dsv41-dspark-graph && ssh divix01 'cd /data/models/slang/nvfp4-work/wt-dsv41-dspark-graph \
  && git fetch -q origin && git checkout -q --detach origin/dsv41-dspark-graph \
  && export PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 SGLANG_EXL3_SRC=/data/models/slang/nvfp4-work/exllamav3 \
  && /data/models/slang/.venv/bin/python -c "import sglang; print(sglang.__file__)" \
  && flock /data/models/slang/nvfp4-work/cc-gpu.lock taskset -c 32-63 /data/models/slang/.venv/bin/python -m pytest <FILES> -q -p no:randomly -k "<K>" > /tmp/d23.log 2>&1; echo EXIT=${PIPESTATUS[0]}; tail -15 /tmp/d23.log'
```

`<K>` = `not nothing_matches_this` selects every test in `<FILES>`.

## Review Focus

1. **Re-running a verify with the same `ForwardBatch`.** The first forward may mutate the batch: positions, `spec_info`,
   attention metadata, `seq_lens`. Expected: the eager re-run computes what a fully eager verify computes. Task 6's
   `reverify` arm (`SGLANG_TEST_DSPARK_FORCE_REVERIFY=1`, every graphed verify re-run) must produce byte-equal text to
   the `eager` arm in every session. Both arms run the same eager MoE and the same eager attention.
2. **The re-run must not overflow or flag again.** Expected: under `suspend_graph_gather()` no layer takes
   `_gather_graph`, the flag stays 0, and every token's output meets the probe's bar. Task 2 pins this on the EXL3
   lease chain after a real overflow.
3. **Unset is today.** Expected:
   - with no narrowed gather, the reader does no host sync;
   - `suspend_graph_gather()` restores the streamers on an exception;
   - `eager_only()` restores the runner on an exception;
   - a DSpark launch with decode `disabled` passes the gate unchanged.

   Tasks 1, 3 and 5 pin these.
4. **An eager forward inside a run that captured copy-engine graphs.** Expected: the re-run is eager, so the
   copy-engine barrier drains the device first once the engine is armed, and does not count the re-run toward
   arming. A graphed verify is counted. Task 4 pins both.
5. **The epilogue and the draft graphs at startup.**
   - Expected: while the target's gather is narrowed, no verify epilogue is built, so the flagged forward writes no
     draft KV.
   - An EXL3 draft captures no draft graph. `exl3_moe_loop` raises under capture (`assert_not_capturing`).
   - Task 3 pins both helpers. Task 6's graphed arm proves startup on the real model.

---

### Task 1: The manager reads the flag, suspends its graph gather, and reports both

**Files:**
- Modify: `python/sglang/srt/layers/moe/expert_stream.py` (`ExpertStreamer.__init__`, `serves_graph_gather`,
  `gather`)
- Modify: `python/sglang/srt/layers/moe/expert_hot_cache.py` (`ExpertHotCacheManager`: new `narrow_graph_gather`,
  `take_verify_overflow`, `suspend_graph_gather` and `graph_gather_suspended`; `_trace_metadata`;
  `_trace_counters_from_host`; the `cls()` setup in `from_model`)
- Modify: `python/sglang/srt/layers/moe/expert_residency_gpu.py` (`insertion_counters`, `snapshot`)
- Test: `test/registered/unit/layers/moe/test_expert_residency_gpu.py` (CUDA; `TestInsertOnMissDirect`)

**Interfaces:**
- Consumes: D2-2's `GpuResidencyUpdater.overflow_flag`, `gather_overflow`, `narrow_gather`; the test helpers
  `_narrow`, `capture(manager, tokens=)`, `replay_verify`, `_verify_routes`.
- Produces:
  - `ExpertStreamer.graph_gather_suspended: bool`, default `False`. While True, `serves_graph_gather` returns False and
    `gather` never takes `_gather_graph`.
  - `ExpertHotCacheManager.narrow_graph_gather -> bool` (property). True when the in-graph updater runs DIRECT with
    some layer's miss width below its routes.
  - `ExpertHotCacheManager.take_verify_overflow() -> bool`. Call it only when `narrow_graph_gather`, after a graphed
    verify. It reads `overflow_flag` (one `.item()`), zeroes it when set, and counts `graphed_verify_ct` and
    `verify_overflow_ct`.
  - `ExpertHotCacheManager.suspend_graph_gather()`, a context manager that sets every streamer's
    `graph_gather_suspended`, plus the property `graph_gather_suspended -> bool`.
  - The metrics trace record (`SGLANG_MOE_HOT_METRICS_FILE`):
    - `counters.residency_gpu.gather_overflow`, a per-layer list, when narrowed;
    - `counters.graphed_verify = {"graphed_verify_ct": int, "verify_overflow_ct": int}`, when narrowed.

- [ ] **Step 1: Write the failing tests**

Add to class `TestInsertOnMissDirect` in `test_expert_residency_gpu.py`, after
`test_a_one_token_gather_never_clamps`:

```python
    # ----- the host side of a graphed verify -----

    def _overflow_routes(self, manager):
        """Layer 0's two tokens route four non-resident experts into two lanes; the other layers route hits."""
        mapping = manager.caches[0].expert_to_slot.tolist()
        outsiders = [expert for expert, slot in enumerate(mapping) if slot < 0]
        hits = [
            [[e for e, s in enumerate(manager.caches[layer].expert_to_slot.tolist()) if s >= 0][:TOP_K]] * 2
            for layer in range(1, LAYERS)
        ]
        return [[outsiders[0:2], outsiders[2:4]]] + hits

    def test_the_verify_overflow_is_read_once_and_cleared(self):
        manager = self._narrow(self.model, 2, False)
        self.assertTrue(manager.narrow_graph_gather)
        graph, static, outputs = self.capture(manager, tokens=2)
        self.replay_verify(manager, graph, static, outputs, self._overflow_routes(manager), check_outputs=False)
        self.assertTrue(manager.take_verify_overflow())
        self.assertEqual(int(manager.gpu_residency.overflow_flag.item()), 0)
        self.assertFalse(manager.take_verify_overflow())
        self.assertEqual((manager.graphed_verify_ct, manager.verify_overflow_ct), (2, 1))

    def test_a_manager_without_a_narrow_gather_reads_nothing(self):
        wide = self._narrow(self.model, 0, False)
        self.assertFalse(wide.narrow_graph_gather)
        one_token = _manager(_model(), gpu=True, **DIRECT)
        self.assertFalse(one_token.narrow_graph_gather)
        self.assertFalse(_manager(_model(), gpu=True, **IOM).narrow_graph_gather)

    def test_suspending_the_graph_gather_turns_it_off_for_every_streamer_and_restores_it(self):
        manager = self._narrow(self.model, 2, False)
        ids = torch.zeros((2, TOP_K), dtype=torch.int32, device="cuda")
        topk = SimpleNamespace(topk_ids=ids)
        streamers = list(manager.streamers.values())
        self.assertTrue(all(s.serves_graph_gather(topk) for s in streamers))
        self.assertFalse(manager.graph_gather_suspended)
        with self.assertRaises(KeyError):
            with manager.suspend_graph_gather():
                self.assertTrue(manager.graph_gather_suspended)
                self.assertFalse(any(s.serves_graph_gather(topk) for s in streamers))
                with unittest.mock.patch.object(ExpertStreamer, "_gather_graph", side_effect=_GraphPathTaken):
                    try:
                        streamers[0].gather(ids)
                    except _GraphPathTaken:
                        self.fail("a suspended gather took the graph path")
                    except Exception:
                        pass  # the eager path's own requirements are not under test here
                raise KeyError("restore on error")
        self.assertFalse(manager.graph_gather_suspended)
        self.assertTrue(all(s.serves_graph_gather(topk) for s in streamers))

    def test_the_metrics_trace_reports_the_overflow_and_the_graphed_verifies(self):
        import json

        with tempfile.NamedTemporaryFile() as trace:
            from sglang.srt.environ import envs

            with envs.SGLANG_MOE_EXPERT_FUSED_PLAN.override(False), envs.SGLANG_DSV41_ENABLE_LAYER_FUSION.override(False):
                manager = _manager(
                    self.model, gpu=True, seed_scale=(0, 0, 0), graph_gather_batch_size=2, graph_gather_miss_lanes=2,
                    budget_bytes=56 * LAYERS * 8, metrics_path=trace.name, log_interval=1, **DIRECT,
                )
            graph, static, outputs = self.capture(manager, tokens=2)
            self.replay_verify(manager, graph, static, outputs, self._overflow_routes(manager), check_outputs=False)
            self.assertTrue(manager.take_verify_overflow())
            self.replay_verify(manager, graph, static, outputs, [[[0, 1], [0, 1]]] * LAYERS, check_outputs=False)
            self.assertFalse(manager.take_verify_overflow())
            self.replay_verify(manager, graph, static, outputs, [[[0, 1], [0, 1]]] * LAYERS, check_outputs=False)
            manager.close_telemetry()
            trace.seek(0)
            last = json.loads(trace.read().splitlines()[-1])
        self.assertEqual(last["counters"]["residency_gpu"]["gather_overflow"], [1, 0, 0])
        self.assertEqual(last["counters"]["graphed_verify"], {"graphed_verify_ct": 2, "verify_overflow_ct": 1})
        self.assertIs(manager._trace_sources()["gpu_residency:gather_overflow"], manager.gpu_residency.gather_overflow)
```

Add at module level, before the class:

```python
class _GraphPathTaken(Exception):
    """Raised by a patched _gather_graph: a gather that should be eager took the graph path."""
```

The suspension test checks that the eager `gather` does not reach `_gather_graph`. Its eager result is not the
point here, so any other exception the eager path raises is accepted. Task 2 checks the eager re-run's output on the
EXL3 chain.

- [ ] **Step 2: Run them and watch them fail**

Commit `test(direct-residency): the host reads a verify's overflow, suspends the graph gather, and reports both (failing)`.
Run the GPU command with `<FILES>` = `test/registered/unit/layers/moe/test_expert_residency_gpu.py` and `<K>` =
`read_once or without_a_narrow or suspending or reports_the_overflow`.
Expected: FAIL. Each test fails with `AttributeError`: `narrow_graph_gather`, `take_verify_overflow` or
`suspend_graph_gather`.

- [ ] **Step 3: The streamer**

In `expert_stream.py`, `ExpertStreamer.__init__`, after `self.graph_miss_lanes = 0` add:

```python
        # Set by ExpertHotCacheManager.suspend_graph_gather: an eager re-run of a verify whose graph gather
        # overflowed must not take the same narrowed gather again.
        self.graph_gather_suspended = False
```

In `serves_graph_gather`, add `not self.graph_gather_suspended` as the return expression's first term:

```python
        return (
            not self.graph_gather_suspended
            and self.graph_gather_rows > 0
            and isinstance(topk_ids, torch.Tensor)
            and 0 < topk_ids.numel() <= self.graph_gather_rows
        )
```

In `gather`, change `if 0 < topk_ids.numel() <= self.graph_gather_rows:` to
`if not self.graph_gather_suspended and 0 < topk_ids.numel() <= self.graph_gather_rows:`.

- [ ] **Step 4: The updater's counters**

In `expert_residency_gpu.py`, `insertion_counters`, replace the final `return {...}` with:

```python
        counters = {
            "insertions": insertions,
            "insertion_evictions": evictions,
            "insertion_truncated": self.insertion_truncated,
        }
        if self.insert_direct and self.narrow_gather:
            counters["gather_overflow"] = self.gather_overflow
        return counters
```

In `snapshot()`, delete the two lines D2-2 added:

```python
        if self.insert_direct and self.narrow_gather:
            snapshot["gather_overflow"] = self.gather_overflow.cpu().tolist()
```

The loop over `insertion_counters()` now publishes it. `_trace_sources` already publishes every
`insertion_counters()` entry as `gpu_residency:<name>`.

- [ ] **Step 5: The manager**

In `expert_hot_cache.py`, in `from_model` right after `manager = cls()`, add:

```python
        manager.graphed_verify_ct = 0
        manager.verify_overflow_ct = 0
```

Add these methods to `ExpertHotCacheManager`, after `snapshot_counters`:

```python
    @property
    def narrow_graph_gather(self) -> bool:
        """Whether a graph gather may serve fewer misses than its routes (a verify at W miss lanes)."""
        updater = getattr(self, "gpu_residency", None)
        return updater is not None and bool(getattr(updater, "narrow_gather", False))

    def take_verify_overflow(self) -> bool:
        """After a graphed verify: whether a layer's gather could not serve its misses. Clears the flag.

        One host read of the sticky device flag (GpuResidencyUpdater.clamp_gather_misses). A True result means the
        verify's output is not a verify result and must be re-run with the graph gather suspended.
        """
        flag = self.gpu_residency.overflow_flag
        self.graphed_verify_ct += 1
        if not int(flag.item()):
            return False
        flag.zero_()
        self.verify_overflow_ct += 1
        return True

    @property
    def graph_gather_suspended(self) -> bool:
        return any(streamer.graph_gather_suspended for streamer in self.streamers.values())

    @contextmanager
    def suspend_graph_gather(self):
        """Every layer gathers eagerly inside the block (an overflowed verify's re-run)."""
        for streamer in self.streamers.values():
            streamer.graph_gather_suspended = True
        try:
            yield
        finally:
            for streamer in self.streamers.values():
                streamer.graph_gather_suspended = False
```

In `_trace_metadata`, inside `if updater is not None:`, after
`metadata["gpu_residency_insert_on_miss"] = updater.insert_on_miss`, add:

```python
            if self.narrow_graph_gather:
                metadata["graphed_verify"] = {
                    "graphed_verify_ct": self.graphed_verify_ct,
                    "verify_overflow_ct": self.verify_overflow_ct,
                }
```

In `_trace_counters_from_host`, after `if "residency_async" in metadata: result["residency_async"] =
metadata["residency_async"]`, add:

```python
        if "graphed_verify" in metadata:
            result["graphed_verify"] = metadata["graphed_verify"]
```

In the same function's `for name in (...)` tuple that builds `device`, append the overflow when it was published:

```python
            ) + (
                _INSERTION_TRACE_NAMES
                if metadata["gpu_residency_insert_on_miss"]
                else ()
            ) + (
                ("gpu_residency:gather_overflow",)
                if "gpu_residency:gather_overflow" in buffers
                else ()
            ):
```

- [ ] **Step 6: Run the tests and watch them pass**

Run the GPU command with `<FILES>` =
`test/registered/unit/layers/moe/test_expert_residency_gpu.py test/registered/unit/layers/moe/test_expert_graph_gather.py test/registered/unit/layers/moe/test_expert_hot_cache.py`
and `<K>` = `not nothing_matches_this`.
Expected: all pass. That includes D2-2's snapshot tests: a narrow snapshot still has `gather_overflow`, and a wide one
does not.

- [ ] **Step 7: Commit**

```bash
git add python/sglang/srt/layers/moe/expert_stream.py python/sglang/srt/layers/moe/expert_hot_cache.py \
  python/sglang/srt/layers/moe/expert_residency_gpu.py
git commit -m "feat(expert-hot-cache): read and clear a graphed verify's overflow, suspend the graph gather, and trace both" -m "<trailers>"
```

---

### Task 2: An overflowed verify's eager re-run is exact on the EXL3 lease chain

**Files:**
- Modify: `test/manual/dsv41/test_exl3_verify_miss_lanes_gpu.py`

**Interfaces:**
- Consumes: Task 1's `take_verify_overflow()` and `suspend_graph_gather()`, and
  `Exl3MoEMethod._apply_streamed(layer, streamer, x, topk_weights, topk_ids, swiglu_limit)`.

No production code changes in this task. It proves that the re-run path Task 3 wires in serves a verify the narrowed
gather could not. If it fails, the fix belongs in the eager path, and the executor debugs it under
superpowers:systematic-debugging before Task 3.

- [ ] **Step 1: Write the test**

In `test_a_verify_gather_serves_its_lanes_and_flags_what_it_cannot`, replace the overflow lines:

```python
            twelve = outsiders()[:12]
            replay([[twelve[(2 * t + k) % 12] for k in range(TOP_K)] for t in range(TOKENS)], overflow=True)  # 12 > 8
            assert updater.gather_overflow[0].item() == overflowed + 1
```

with:

```python
            twelve = outsiders()[:12]
            routes = [[twelve[(2 * t + k) % 12] for k in range(TOP_K)] for t in range(TOKENS)]
            replay(routes, overflow=True, clear=False)  # 12 > 8
            assert updater.gather_overflow[0].item() == overflowed + 1
            # The DSpark re-run: read and clear the flag, then the same verify with the graph gather suspended.
            assert manager.take_verify_overflow()
            with manager.suspend_graph_gather():
                assert not streamer.serves_graph_gather(SimpleNamespace(topk_ids=ids))
                eager = Exl3MoEMethod._apply_streamed(layer, streamer, x, weights, ids, ACT_LIMIT)
            torch.cuda.synchronize()
            service.fail_stop_check()
            assert int(updater.overflow_flag.item()) == 0, "the eager re-run flagged"
            assert updater.insertion_truncated[0].item() == 0
            residency_is_exact()
            for t, route in enumerate(routes):
                ref = _reference(x[t : t + 1], weights[t], torch.tensor(route), source_cuda)
                assert _rel(eager[t : t + 1], ref) <= REL_BOUND, (t, route)
```

Change `replay`'s signature and its first line so the overflow step can leave the flag for `take_verify_overflow`:

```python
            def replay(routes, overflow, clear=True):
                if clear:
                    updater.overflow_flag.zero_()
```

`replay`'s flag assertion only reads the flag, so `take_verify_overflow` still sees it after an overflow step.

Add `from types import SimpleNamespace` to the imports. Right after `source = _source_rows(tmp_path, num_experts=EXPERTS)`,
add `source_cuda = {name: rows.cuda() for name, rows in source.items()}`, and pass `source_cuda`, not `source`, to the
re-run's `_reference`.

`_reference(x, weights, slots, tensors)` indexes `tensors[name][slot]` on `x`'s device. Here it is handed the
checkpoint rows (`_source_rows` reads them to the CPU) and expert ids: `source[name][expert]` is that expert's row, the
reference for an expert that has no slot.

- [ ] **Step 2: Run it**

Commit `test(exl3-ram-miss): an overflowed verify re-runs eagerly with the graph gather suspended, exactly`. Run the GPU
command with `<FILES>` = `test/manual/dsv41/test_exl3_verify_miss_lanes_gpu.py` and `<K>` = `not nothing_matches_this`.
Expected: both arms (generic, layer fusion) pass.

---

### Task 3: The DSpark verify re-runs a flagged forward eagerly; startup drops the epilogue and the EXL3 draft's graphs

**Files:**
- Modify: `python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py` (`DecodeCudaGraphRunner`: class
  attribute `_eager_only`, `eager_only()`, the first lines of `can_run_graph`)
- Create: `python/sglang/srt/speculative/dspark_components/dspark_graphed_verify.py`
- Modify: `python/sglang/srt/speculative/dspark_components/dspark_verify.py` (`_forward_prepared_verify`)
- Modify: `python/sglang/srt/speculative/dspark_components/dspark_worker_v2.py` (the epilogue condition in
  `__init__`, and `init_cuda_graphs`)
- Modify: `python/sglang/srt/environ.py` (`SGLANG_TEST_DSPARK_FORCE_REVERIFY`)
- Test: `test/registered/unit/speculative/test_dspark_graphed_verify.py` (CPU)

**Interfaces:**
- Consumes: Task 1's `narrow_graph_gather`, `take_verify_overflow()` and `suspend_graph_gather()`.
- Produces:
  - `DecodeCudaGraphRunner.eager_only()`, a context manager inside which `can_run_graph` returns False.
  - `forward_verify_with_reverify(model_runner, forward) -> out`. `forward` is a zero-argument callable returning an
    object with `can_run_cuda_graph`.
  - `draft_runs_exl3(draft_model) -> bool` and `target_gather_is_narrow(model_runner) -> bool`.
  - `envs.SGLANG_TEST_DSPARK_FORCE_REVERIFY`.

- [ ] **Step 1: Write the failing tests**

Create `test/registered/unit/speculative/test_dspark_graphed_verify.py`:

```python
"""DSpark's graphed verify: a flagged forward is re-run eagerly, and startup skips what cannot be captured (CPU)."""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from sglang.srt.environ import envs
from sglang.srt.model_executor.runner.decode_cuda_graph_runner import DecodeCudaGraphRunner
from sglang.srt.speculative.dspark_components.dspark_graphed_verify import (
    draft_runs_exl3,
    forward_verify_with_reverify,
    target_gather_is_narrow,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _Manager:
    def __init__(self, narrow, overflows):
        self.narrow_graph_gather = narrow
        self._overflows = list(overflows)
        self.suspended = False
        self.reads = 0

    def take_verify_overflow(self):
        self.reads += 1
        return self._overflows.pop(0)

    @contextmanager
    def suspend_graph_gather(self):
        self.suspended = True
        try:
            yield
        finally:
            self.suspended = False


def _runner(manager):
    graph_runner = object.__new__(DecodeCudaGraphRunner)
    return SimpleNamespace(expert_hot_cache_manager=manager, decode_cuda_graph_runner=graph_runner)


def _forward(runner, log):
    def forward():
        graphed = runner.decode_cuda_graph_runner.can_run_graph(SimpleNamespace(replace_embeds=None)) is not False
        log.append((graphed, runner.expert_hot_cache_manager.suspended))
        return SimpleNamespace(can_run_cuda_graph=graphed, tag=len(log))

    return forward


def test_a_flagged_graphed_verify_is_re_run_eagerly_with_the_gather_suspended(monkeypatch):
    manager = _Manager(narrow=True, overflows=[True])
    runner = _runner(manager)
    monkeypatch.setattr(DecodeCudaGraphRunner, "can_run_graph", lambda self, batch: not self._eager_only)
    log = []
    out = forward_verify_with_reverify(runner, _forward(runner, log))
    assert log == [(True, False), (False, True)]
    assert out.tag == 2 and not out.can_run_cuda_graph
    assert not manager.suspended and not runner.decode_cuda_graph_runner._eager_only


def test_an_unflagged_graphed_verify_is_kept(monkeypatch):
    manager = _Manager(narrow=True, overflows=[False])
    runner = _runner(manager)
    monkeypatch.setattr(DecodeCudaGraphRunner, "can_run_graph", lambda self, batch: not self._eager_only)
    log = []
    out = forward_verify_with_reverify(runner, _forward(runner, log))
    assert log == [(True, False)] and out.tag == 1 and manager.reads == 1


@pytest.mark.parametrize("narrow, graphed", [(False, True), (True, False)])
def test_no_narrow_gather_or_no_graph_reads_nothing(narrow, graphed):
    manager = _Manager(narrow=narrow, overflows=[])
    runner = SimpleNamespace(expert_hot_cache_manager=manager, decode_cuda_graph_runner=None)
    out = forward_verify_with_reverify(runner, lambda: SimpleNamespace(can_run_cuda_graph=graphed))
    assert manager.reads == 0 and out.can_run_cuda_graph is graphed


def test_no_hot_cache_reads_nothing():
    runner = SimpleNamespace(expert_hot_cache_manager=None, decode_cuda_graph_runner=None)
    out = forward_verify_with_reverify(runner, lambda: SimpleNamespace(can_run_cuda_graph=True))
    assert out.can_run_cuda_graph


def test_the_test_switch_re_runs_every_graphed_verify(monkeypatch):
    manager = _Manager(narrow=True, overflows=[False])
    runner = _runner(manager)
    monkeypatch.setattr(DecodeCudaGraphRunner, "can_run_graph", lambda self, batch: not self._eager_only)
    log = []
    with envs.SGLANG_TEST_DSPARK_FORCE_REVERIFY.override(True):
        forward_verify_with_reverify(runner, _forward(runner, log))
    assert log == [(True, False), (False, True)] and manager.reads == 1


def test_eager_only_refuses_the_graph_and_restores_on_error():
    runner = object.__new__(DecodeCudaGraphRunner)
    assert runner._eager_only is False
    with pytest.raises(KeyError):
        with runner.eager_only():
            assert runner.can_run_graph(SimpleNamespace(replace_embeds=None)) is False
            raise KeyError("restore")
    assert runner._eager_only is False


def test_draft_runs_exl3():
    exl3 = SimpleNamespace(quant_config=SimpleNamespace(get_name=lambda: "exl3"))
    fp8 = SimpleNamespace(quant_config=SimpleNamespace(get_name=lambda: "fp8"))
    assert draft_runs_exl3(exl3) and not draft_runs_exl3(fp8)
    assert not draft_runs_exl3(SimpleNamespace(quant_config=None)) and not draft_runs_exl3(SimpleNamespace())


def test_target_gather_is_narrow():
    assert target_gather_is_narrow(SimpleNamespace(expert_hot_cache_manager=SimpleNamespace(narrow_graph_gather=True)))
    assert not target_gather_is_narrow(SimpleNamespace(expert_hot_cache_manager=SimpleNamespace(narrow_graph_gather=False)))
    assert not target_gather_is_narrow(SimpleNamespace(expert_hot_cache_manager=None))
    assert not target_gather_is_narrow(SimpleNamespace())
```

- [ ] **Step 2: Run them and watch them fail**

Commit `test(dspark): a flagged graphed verify is re-run eagerly; startup skips the epilogue and the EXL3 draft graphs (failing)`.
Run the CPU command with `<FILES>` = `test/registered/unit/speculative/test_dspark_graphed_verify.py` and `<K>` =
`not nothing_matches_this`.
Expected: FAIL, a collection error with
`ModuleNotFoundError: ... dspark_components.dspark_graphed_verify`.

- [ ] **Step 3: The force-eager switch**

In `decode_cuda_graph_runner.py`, add `from contextlib import contextmanager` to the imports if it is absent. Add a
class attribute in `DecodeCudaGraphRunner`'s class body, right after the docstring:

```python
    # Set by eager_only(): a DSpark verify whose graphed expert gather overflowed is re-run eagerly.
    _eager_only = False
```

Add the method right before `can_run_graph`:

```python
    @contextmanager
    def eager_only(self):
        """Every forward inside the block runs eagerly (can_run_graph returns False)."""
        self._eager_only = True
        try:
            yield
        finally:
            self._eager_only = False
```

Make these the first two lines of `can_run_graph`'s body:

```python
        if self._eager_only:
            return False
```

- [ ] **Step 4: The env var and the module**

In `environ.py`, next to the other `SGLANG_DSPARK_*` entries (after `SGLANG_DSPARK_FOLDED_PROPOSAL`), add:

```python
    # Test only: re-run every graphed DSpark verify eagerly, as an overflowed one is. Its output must equal an eager
    # verify's (D2-3's end-to-end check of the re-run path).
    SGLANG_TEST_DSPARK_FORCE_REVERIFY = EnvBool(False)
```

Create `python/sglang/srt/speculative/dspark_components/dspark_graphed_verify.py`:

```python
"""DSpark's target verify in the breakable decode graph with EXL3 expert caching (DSV41_REFERENCE.md §33.8).

A verify's graph gather serves at most W distinct misses per layer. A verify with more raises a sticky device flag,
and its output (logits, epilogue buffers, draft KV) is not a verify result. The verify is then re-run with the decode
graph and the narrowed gather both off: the EXL3 eager MoE, the path DSpark verify ran on before D2.
"""

from sglang.srt.environ import envs


def forward_verify_with_reverify(model_runner, forward):
    """Run a target verify; re-run it eagerly when its graphed expert gather overflowed.

    ``forward`` runs the verify and returns an object with ``can_run_cuda_graph``. The flag is read only after a
    graphed verify on a narrowed gather: one host read, before anything reads the logits.
    """
    out = forward()
    manager = getattr(model_runner, "expert_hot_cache_manager", None)
    if manager is None or not manager.narrow_graph_gather or not out.can_run_cuda_graph:
        return out
    overflowed = manager.take_verify_overflow()
    if not overflowed and not envs.SGLANG_TEST_DSPARK_FORCE_REVERIFY.get():
        return out
    with manager.suspend_graph_gather(), model_runner.decode_cuda_graph_runner.eager_only():
        return forward()


def draft_runs_exl3(draft_model) -> bool:
    """Whether the draft's MoE is EXL3: exl3_moe_loop reads expert counts on the host and refuses capture."""
    config = getattr(draft_model, "quant_config", None)
    return config is not None and config.get_name() == "exl3"


def target_gather_is_narrow(model_runner) -> bool:
    """Whether the target's verify gather serves fewer misses than its routes, and may flag its forward."""
    manager = getattr(model_runner, "expert_hot_cache_manager", None)
    return manager is not None and bool(manager.narrow_graph_gather)
```

- [ ] **Step 5: Wire it in**

In `dspark_verify.py`, add
`from sglang.srt.speculative.dspark_components.dspark_graphed_verify import forward_verify_with_reverify` to the
imports. In `_forward_prepared_verify`, replace:

```python
        target_out = self.target_worker.forward_batch_generation(
            batch=None,
            forward_batch=verify_forward_batch,
            is_verify=True,
            skip_attn_backend_init=True if not _is_npu else None,
        )
```

with:

```python
        target_out = forward_verify_with_reverify(
            self.target_worker.model_runner,
            lambda: self.target_worker.forward_batch_generation(
                batch=None,
                forward_batch=verify_forward_batch,
                is_verify=True,
                skip_attn_backend_init=True if not _is_npu else None,
            ),
        )
```

In `dspark_worker_v2.py`, add
`from sglang.srt.speculative.dspark_components.dspark_graphed_verify import draft_runs_exl3, target_gather_is_narrow`
to the imports.
- In `__init__`, in the `static_epilogue_supported = (...)` expression, append
  `and not target_gather_is_narrow(self.target_worker.model_runner)` as its last term. Put this comment above it:
  `# A narrowed verify gather may flag its forward, whose in-graph epilogue would commit draft KV from wrong hidden states.`
- In `init_cuda_graphs`, replace `capture_decode_cuda_graph = self._decode_graph_allowed` with:

```python
        # An EXL3 draft's MoE (exl3_moe_loop) reads expert counts on the host and refuses capture; the draft then runs
        # eagerly while the target keeps its decode graph.
        capture_decode_cuda_graph = self._decode_graph_allowed and not draft_runs_exl3(self.draft_model)
```

- [ ] **Step 6: Run the tests and watch them pass**

Run the CPU command with `<FILES>` =
`test/registered/unit/speculative/test_dspark_graphed_verify.py test/registered/unit/speculative/test_dspark_residency_commit.py test/registered/spec/dspark`
and `<K>` = `not nothing_matches_this`.
Expected: all pass. Record the `test/registered/spec/dspark` count, and compare it against the same selection at this
task's BASE commit if any test there fails.

- [ ] **Step 7: Commit**

```bash
git add python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py python/sglang/srt/environ.py \
  python/sglang/srt/speculative/dspark_components/dspark_graphed_verify.py \
  python/sglang/srt/speculative/dspark_components/dspark_verify.py \
  python/sglang/srt/speculative/dspark_components/dspark_worker_v2.py \
  test/registered/unit/speculative/test_dspark_graphed_verify.py
git commit -m "feat(dspark): re-run a graphed verify whose expert gather overflowed, eagerly; no epilogue or EXL3 draft graphs while narrowed" -m "<trailers>"
```

---

### Task 4: The copy-engine barrier counts a graphed verify and drains before its eager re-run

**Files:**
- Modify: `python/sglang/srt/layers/moe/exl3_ram_miss.py` (`Exl3RamMissService._copy_engine_barrier`)
- Test: `test/registered/unit/layers/moe/test_exl3_ram_miss_service.py`

**Interfaces:**
- Consumes: Task 1's `ExpertHotCacheManager.graph_gather_suspended`. The service holds its manager as
  `self._manager`, set in `attach`.

Today the barrier tests `is_decode()`, which is False for `TARGET_VERIFY`. A DSpark server's target forwards are all
verifies, so it never arms the copy engine. If it were armed, every graphed verify would synchronize first
(§33.3 item 6). A verify is graphed unless the manager's graph gather is suspended: that is Task 3's re-run, which is
eager.

- [ ] **Step 1: Write the failing test**

In `test_exl3_ram_miss_service.py`, after `test_the_copy_engine_arms_only_after_enough_decode_forwards_not_batches`, add:

```python
def _verify_batch():
    return SimpleNamespace(forward_mode=SimpleNamespace(is_decode=lambda: False, is_target_verify=lambda: True))


def test_a_graphed_verify_counts_toward_arming_and_its_eager_re_run_drains(monkeypatch):
    """A DSpark server's target forwards are all verifies: a graphed one counts like a decode, and the eager re-run
    of an overflowed one (the manager's graph gather suspended) drains the device once armed."""
    service, armed, syncs = _copy_engine_service(monkeypatch)
    service.device_side.copy_engine_captured = True
    service._manager = SimpleNamespace(graph_gather_suspended=False)
    for _ in range(module.COPY_ENGINE_ARM_DECODES):
        service._copy_engine_barrier(0, _verify_batch())
    service._arm_copy_engine()
    assert armed == [True] and not syncs
    service._copy_engine_barrier(0, _verify_batch())
    assert not syncs, "a graphed verify drained the device"
    service._manager = SimpleNamespace(graph_gather_suspended=True)
    service._copy_engine_barrier(0, _verify_batch())
    assert syncs == [True], "an eager re-run did not drain the device once armed"
```

Every fake mode handed to `_copy_engine_barrier` must now answer `is_target_verify`. Run
`grep -n "_copy_engine_barrier(\|forward_mode=SimpleNamespace" test/registered/unit/layers/moe/test_exl3_ram_miss_service.py`
and give each fake that reaches the barrier `is_target_verify=lambda: False`. Update `_batch` in the same way:

```python
def _batch(decode: bool):
    return SimpleNamespace(forward_mode=SimpleNamespace(is_decode=lambda: decode, is_target_verify=lambda: False))
```

- [ ] **Step 2: Run it and watch it fail**

Commit `test(exl3-ram-miss): a graphed verify counts toward arming the copy engine; its eager re-run drains (failing)`.
Run the CPU command with `<FILES>` = `test/registered/unit/layers/moe/test_exl3_ram_miss_service.py` and `<K>` =
`graphed_verify_counts`.
Expected: FAIL with `assert [] == [True]`: verifies never arm the engine.

- [ ] **Step 3: Implement**

In `_copy_engine_barrier`, replace:

```python
        if not forward_batch.forward_mode.is_decode():
```

with:

```python
        mode = forward_batch.forward_mode
        # A DSpark verify replays the decode graph unless its eager re-run suspended the graph gather.
        graphed = mode.is_decode() or (
            mode.is_target_verify()
            and not (self._manager is not None and self._manager.graph_gather_suspended)
        )
        if not graphed:
```

Add to the docstring: "A DSpark verify counts as a captured forward unless the manager's graph gather is suspended
(the eager re-run of an overflowed verify)."

- [ ] **Step 4: Run the tests and watch them pass**

Run the CPU command with `<FILES>` = `test/registered/unit/layers/moe/test_exl3_ram_miss_service.py` and `<K>` =
`not nothing_matches_this`.
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add python/sglang/srt/layers/moe/exl3_ram_miss.py
git commit -m "feat(exl3-ram-miss): the copy-engine barrier counts graphed verifies and drains before an eager re-run" -m "<trailers>"
```

---

### Task 5: The gate admits a DSpark verify in the breakable decode graph

**Files:**
- Modify: `python/sglang/srt/arg_groups/expert_stream_requirements_exl3.py` (`_check`, a new `_check_graphed_verify`)
- Test: `test/registered/unit/test_expert_stream_requirements_exl3.py`

**Interfaces:**
- Consumes: the env vars `SGLANG_MOE_EXPERT_GRAPH_GATHER`, `SGLANG_MOE_GPU_RESIDENCY_UPDATE`,
  `SGLANG_MOE_HOT_INSERT_ON_MISS_STAGE`, `SGLANG_MOE_EXPERT_GRAPH_GATHER_MISS_LANES` and `SGLANG_RAGGED_VERIFY_MODE`.

- [ ] **Step 1: Write the failing tests**

In `test_expert_stream_requirements_exl3.py`, after `test_dspark_speculation_passes_with_decode_disabled`, add:

```python
# DSpark's verify in the breakable decode graph (DSV41_REFERENCE.md §33.8): DIRECT residency at W miss lanes.
GRAPHED_VERIFY = {"SGLANG_MOE_EXPERT_GRAPH_GATHER": True, **DIRECT, "SGLANG_MOE_EXPERT_GRAPH_GATHER_MISS_LANES": 8}


def test_dspark_verify_in_the_breakable_decode_graph_passes(model_dir):
    _gate(_launch(model_dir, speculative_algorithm="DSPARK", cuda_graph_config=BREAKABLE_BS1), **GRAPHED_VERIFY)


@pytest.mark.parametrize(
    "launch_changes, env_changes, match",
    [
        ({"speculative_algorithm": "EAGLE"}, GRAPHED_VERIFY, "graphs the verify of DSpark only"),
        ({}, {**GRAPHED_VERIFY, "SGLANG_MOE_EXPERT_GRAPH_GATHER_MISS_LANES": 0}, "MISS_LANES=1-32"),
        ({}, {**GRAPHED_VERIFY, "SGLANG_MOE_EXPERT_GRAPH_GATHER_MISS_LANES": 33}, "MISS_LANES=1-32"),
        ({}, {**GRAPHED_VERIFY, "SGLANG_MOE_HOT_INSERT_ON_MISS_STAGE": 1}, "INSERT_ON_MISS_STAGE=2"),
        ({}, {**GRAPHED_VERIFY, "SGLANG_MOE_EXPERT_GRAPH_GATHER": False}, "SGLANG_MOE_EXPERT_GRAPH_GATHER=1"),
        ({}, {**GRAPHED_VERIFY, "SGLANG_RAGGED_VERIFY_MODE": "compact"}, "SGLANG_RAGGED_VERIFY_MODE=static"),
        ({}, {**GRAPHED_VERIFY, "SGLANG_DSV41_CPU_EXPERTS": True}, "without speculative decoding"),
    ],
)
def test_a_graphed_dspark_verify_needs_its_configuration(model_dir, launch_changes, env_changes, match):
    launch = dict(speculative_algorithm="DSPARK", cuda_graph_config=BREAKABLE_BS1) | launch_changes
    with pytest.raises(ValueError, match=match) as raised:
        _gate(_launch(model_dir, **launch), **env_changes)
    if match != "without speculative decoding":
        assert "--cuda-graph-backend-decode disabled" in str(raised.value)
```

In `test_dspark_speculation_passes_with_decode_disabled`, replace the comment with:

```python
    # An eager DSpark verify needs none of the graphed verify's configuration (§33.8).
```

- [ ] **Step 2: Run them and watch them fail**

Commit `test(exl3-gate): a DSpark verify in the breakable decode graph, and what it needs (failing)`. Run the CPU
command with `<FILES>` = `test/registered/unit/test_expert_stream_requirements_exl3.py` and `<K>` =
`graphed_dspark or breakable_decode_graph_passes`.
Expected: FAIL.
- The passing test raises "EXL3 expert caching runs DSpark verify eagerly only".
- Every refusal case except CPU experts fails with "Regex pattern did not match".
- The CPU-experts case passes: that refusal runs first and is unchanged.

- [ ] **Step 3: Implement**

In `expert_stream_requirements_exl3.py`, add above `_check`:

```python
_EAGER_VERIFY_REMEDY = "or pass --cuda-graph-backend-decode disabled to run the DSpark verify eagerly"


def _check_graphed_verify(cfg) -> None:
    """A speculative verify in the breakable decode graph (DSV41_REFERENCE.md §33.7-§33.8).

    Only DSpark's static verify, on DIRECT residency at W miss lanes: a verify routes more than the wire's 32 lanes,
    and only DIRECT's gather flags the misses it cannot serve, which the DSpark worker re-runs eagerly.
    """
    algorithm = cfg.speculative_algorithm
    if str(algorithm).upper() != "DSPARK":
        raise ValueError(
            f"EXL3 expert caching graphs the verify of DSpark only, not {algorithm}; {_EAGER_VERIFY_REMEDY}"
        )
    lanes = envs.SGLANG_MOE_EXPERT_GRAPH_GATHER_MISS_LANES.get()
    if not (
        envs.SGLANG_MOE_EXPERT_GRAPH_GATHER.get()
        and envs.SGLANG_MOE_GPU_RESIDENCY_UPDATE.get()
        and envs.SGLANG_MOE_HOT_INSERT_ON_MISS_STAGE.get() == 2
        and 1 <= lanes <= 32
    ):
        raise ValueError(
            "EXL3 expert caching runs a DSpark verify in the decode graph only with SGLANG_MOE_EXPERT_GRAPH_GATHER=1, "
            "SGLANG_MOE_GPU_RESIDENCY_UPDATE=1, SGLANG_MOE_HOT_INSERT_ON_MISS_STAGE=2 and "
            f"SGLANG_MOE_EXPERT_GRAPH_GATHER_MISS_LANES=1-32 (got {lanes}): a verify routes more experts than the 32 "
            f"lanes; {_EAGER_VERIFY_REMEDY}"
        )
    if envs.SGLANG_RAGGED_VERIFY_MODE.get() != "static":
        raise ValueError(
            "EXL3 expert caching runs a DSpark verify in the decode graph with SGLANG_RAGGED_VERIFY_MODE=static only "
            f"(compact mode reads the host); {_EAGER_VERIFY_REMEDY}"
        )
```

In `_check`, replace the speculation block:

```python
    if (
        getattr(cfg, "speculative_algorithm", None) is not None
        and graph.decode.backend != Backend.DISABLED
    ):
        # The graph-gather scratch and RAM-miss posting are sized for one token per
        # step; a DSpark verify runs up to block_size + 1 tokens.
        raise ValueError(
            "EXL3 expert caching runs DSpark verify eagerly only; pass "
            "--cuda-graph-backend-decode disabled (or --disable-cuda-graph)"
        )
```

with:

```python
    if (
        getattr(cfg, "speculative_algorithm", None) is not None
        and graph.decode.backend != Backend.DISABLED
    ):
        _check_graphed_verify(cfg)
```

The existing tests `test_unsupported_launches_are_refused[...DSpark...]` and
`test_dspark_with_a_decode_graph_names_the_remedy` still match: without `GRAPHED_VERIFY` the message names "DSpark"
and "--cuda-graph-backend-decode disabled".

- [ ] **Step 4: Run the tests and watch them pass**

Run the CPU command with `<FILES>` = `test/registered/unit/test_expert_stream_requirements_exl3.py` and `<K>` =
`not nothing_matches_this`.
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add python/sglang/srt/arg_groups/expert_stream_requirements_exl3.py test/registered/unit/test_expert_stream_requirements_exl3.py
git commit -m "feat(exl3-gate): admit a DSpark static verify in the breakable decode graph on DIRECT residency at W miss lanes" -m "<trailers>"
```

---

### Task 6: The measurement — eager, graphed, and graphed with every verify re-run

**Files:**
- Modify: `scripts/dsv41/trace_corpus.py` (`engine_kwargs`, a new `dspark_info`, `main`, the `--dspark` help)
- Modify: `test/manual/dsv41/test_trace_corpus.py`
- Create: `analysis/dsv41-drive/dspark/graphed_verify.py`
- Test: `test/registered/unit/scripts/test_graphed_verify_summary.py` (CPU)
- Modify: `DSV41_REFERENCE.md` (new `### 33.8` after §33.7, before `## Sources`)

**Interfaces:**
- Consumes:
  - Tasks 1-5;
  - `ab_cpu_draft.py`'s `COMMON`, `ARMS`, `DRAFT`, `SESSIONS`, `PYTHON`, `REPO`;
  - the metrics record of Task 1;
  - `dspark_info_record = {"records": [{"target_verify_gpu_ms": float | None, ...}], ...}`, from
    `engine.get_server_info()["internal_states"][0]`, enabled by `SGLANG_DSPARK_DEBUG_DUMP=target_verify_gpu_time`.
- Produces: `graphed_verify.summarize(outdir) -> dict`, written to `<outdir>/summary.json`.

- [ ] **Step 1: Write the failing tests**

In `test/manual/dsv41/test_trace_corpus.py`, replace
`test_dspark_overrides_graphs_since_the_gate_refuses_speculation_under_a_graph` with:

```python
def test_dspark_with_graphs_runs_the_verify_in_the_breakable_decode_graph():
    args = SimpleNamespace(
        model="/m", mem_fraction_static=0.85, chunked_prefill_size=512,
        new_tokens=128, graphs=True, dspark="/draft/dir",
    )
    kwargs = trace_corpus.engine_kwargs(args)
    assert "disable_cuda_graph" not in kwargs
    assert kwargs["cuda_graph_backend_decode"] == "breakable" and kwargs["cuda_graph_max_bs_decode"] == 1
    assert kwargs["speculative_algorithm"] == "DSPARK"


def test_dspark_info_reads_the_first_internal_state():
    engine = SimpleNamespace(get_server_info=lambda: {"internal_states": [{"dspark_info_record": {"records": [1]}}]})
    assert trace_corpus.dspark_info(engine) == {"records": [1]}
    assert trace_corpus.dspark_info(SimpleNamespace(get_server_info=lambda: {"internal_states": [{}]})) is None
```

Create `test/registered/unit/scripts/test_graphed_verify_summary.py`:

```python
"""The D2-3 driver's summary of an eager, a graphed and a re-verify-all arm (CPU)."""

import json
import os
import sys

import pytest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "analysis", "dsv41-drive", "dspark")
)
import graphed_verify  # noqa: E402


def _arm(outdir, arm, texts, verify_ms, overflow=None, graphed=None, truncated=None):
    sessions = [
        {"decode_tok_s": 10.0 + i, "completion_tokens": 30, "spec_verify_ct": 10, "output_text": text}
        for i, text in enumerate(texts)
    ]
    report = {
        "per_session": sessions,
        "mean_decode_tok_s": 10.5,
        "dspark_info_record": {"records": [{"target_verify_gpu_ms": ms} for ms in verify_ms]},
    }
    with open(os.path.join(outdir, f"{arm}.json"), "w") as f:
        json.dump(report, f)
    if overflow is not None:
        record = {"counters": {"residency_gpu": {"gather_overflow": overflow, "insertion_truncated": truncated},
                               "graphed_verify": graphed}}
        with open(os.path.join(outdir, f"{arm}.metrics.jsonl"), "w") as f:
            f.write(json.dumps({"counters": {}}) + "\n" + json.dumps(record) + "\n")


def test_the_summary_reads_every_arm(tmp_path):
    out = str(tmp_path)
    _arm(out, "eager", ["a", "b"], [20.0, 30.0, None])
    _arm(out, "graphed", ["a", "c"], [10.0, 14.0], overflow=[2, 0, 5], graphed={"graphed_verify_ct": 10, "verify_overflow_ct": 6},
         truncated=[0, 0, 0])
    _arm(out, "reverify", ["a", "b"], [40.0], overflow=[0, 0, 0], graphed={"graphed_verify_ct": 10, "verify_overflow_ct": 0},
         truncated=[0, 0, 0])
    summary = graphed_verify.summarize(out)
    assert summary["eager"]["accept_length"] == 3.0 and summary["eager"]["verify_ms"]["mean"] == 25.0
    graphed = summary["graphed"]
    assert graphed["reverify_rate"] == pytest.approx(0.6)
    assert graphed["layer_overflow_rate"] == {"mean": pytest.approx(0.7 / 3), "max": pytest.approx(0.5)}
    assert graphed["insertion_truncated"] == 0 and graphed["text_matches_eager"] == 1
    assert summary["reverify"]["text_matches_eager"] == 2
    with open(os.path.join(out, "summary.json")) as f:
        assert json.load(f) == summary
```

- [ ] **Step 2: Run them and watch them fail**

Commit `test(dsv41-dspark): the graphed-verify driver and its summary (failing)`. Run the CPU command with `<FILES>` =
`test/manual/dsv41/test_trace_corpus.py test/registered/unit/scripts/test_graphed_verify_summary.py` and `<K>` =
`not nothing_matches_this`.
Expected: FAIL.
- `test_dspark_with_graphs...` fails on `"disable_cuda_graph" not in kwargs`.
- `test_dspark_info...` fails with `AttributeError: ... 'dspark_info'`.
- The summary test fails at collection with `ModuleNotFoundError: No module named 'graphed_verify'`.
- `test_dspark_sets_speculative_kwargs_and_forces_eager` still passes: without `--graphs`, a DSpark run is eager.

- [ ] **Step 3: `trace_corpus.py`**

In `engine_kwargs`, replace:

```python
    if getattr(args, "graphs", False) and not dspark_draft:
        # The capture, RAM-miss thread and hot cache startup lines are info logs.
        kwargs.update(GRAPH_KWARGS, log_level="info")
    else:
        # The EXL3 expert-caching gate refuses speculation under a decode CUDA
        # graph, so a --dspark run is always eager regardless of --graphs.
        kwargs["disable_cuda_graph"] = True
```

with:

```python
    if getattr(args, "graphs", False):
        # The capture, RAM-miss thread and hot cache startup lines are info logs. With
        # --dspark the verify runs in the decode graph, which the EXL3 gate admits on
        # DIRECT residency at W miss lanes (DSV41_REFERENCE.md §33.8).
        kwargs.update(GRAPH_KWARGS, log_level="info")
    else:
        kwargs["disable_cuda_graph"] = True
```

Change the `--dspark` help to `"run DSpark speculative decoding with this draft checkpoint dir (eager unless --graphs)"`.

Add after `mean_decode_tok_s`:

```python
def dspark_info(engine):
    """The DSpark worker's info records (SGLANG_DSPARK_DEBUG_DUMP), or None when the dump is off."""
    states = engine.get_server_info().get("internal_states") or [{}]
    return states[0].get("dspark_info_record")
```

In `main`, replace `engine.shutdown()` with:

```python
    info = dspark_info(engine) if args.dspark else None
    engine.shutdown()
```

In the `report = {...}` dict, add `"dspark_info_record": info`.

- [ ] **Step 4: The driver**

Create `analysis/dsv41-drive/dspark/graphed_verify.py`:

```python
"""DSpark graphed verify, D2-3 (plan 2026-10-05-dsv41-dspark-graph-d2-3-graphed-verify Task 6).

Three arms, one Engine each through trace_corpus.py, the same sessions, the hybrid draft (§33.4), CPU experts off:
  eager     the verify eager (decode disabled), as §33.4's hybrid arm;
  graphed   the verify in the breakable decode graph at W miss lanes; an overflowed verify is re-run eagerly;
  reverify  graphed, with every verify re-run eagerly (SGLANG_TEST_DSPARK_FORCE_REVERIFY): its text must equal eager's.
Run on divix01 from a worktree at the pushed branch, holding rowimg-disk.lock then cc-gpu.lock:
  python analysis/dsv41-drive/dspark/graphed_verify.py OUTDIR [ARM ...]
"""

import json
import os
import statistics
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ab_cpu_draft import ARMS, COMMON, DRAFT, PYTHON, REPO, SESSIONS  # noqa: E402

sys.path.insert(0, os.path.join(REPO, "benchmarks", "dsv41_baseline"))
import arm_env  # noqa: E402

MISS_LANES = os.environ.get("D23_MISS_LANES", "8")
# COMMON turns the recipe's in-graph expert path off for an eager verify; a graphed verify keeps the recipe's.
EAGER_ONLY = (
    "SGLANG_MOE_EXPERT_GRAPH_GATHER",
    "SGLANG_MOE_GPU_RESIDENCY_UPDATE",
    "SGLANG_MOE_HOT_INSERT_ON_MISS_STAGE",
    "SGLANG_MOE_EXPERT_FUSED_PLAN",
    "SGLANG_DSV41_ENABLE_PREFILL_FILLS",
)
GRAPHED = {
    **{k: v for k, v in COMMON.items() if k not in EAGER_ONLY},
    "SGLANG_MOE_EXPERT_GRAPH_GATHER": "1",
    "SGLANG_MOE_GPU_RESIDENCY_UPDATE": "1",
    "SGLANG_MOE_HOT_INSERT_ON_MISS_STAGE": "2",
    "SGLANG_MOE_EXPERT_GRAPH_GATHER_MISS_LANES": MISS_LANES,
    "SGLANG_RAGGED_VERIFY_MODE": "static",
}
D23_ARMS = {
    "eager": ({**COMMON, **ARMS["hybrid"]}, False),
    "graphed": ({**GRAPHED, **ARMS["hybrid"]}, True),
    "reverify": ({**GRAPHED, **ARMS["hybrid"], "SGLANG_TEST_DSPARK_FORCE_REVERIFY": "1"}, True),
}


def run(arm: str, outdir: str, n: int, new_tokens: int) -> int:
    overrides, graphs = D23_ARMS[arm]
    overrides = {
        **overrides,
        "SGLANG_DSPARK_DEBUG_DUMP": "target_verify_gpu_time",
        "SGLANG_MOE_HOT_METRICS_FILE": os.path.join(outdir, f"{arm}.metrics.jsonl"),
    }
    env = os.environ | arm_env.arm_env(overrides)
    env["PYTHONPATH"] = os.path.join(REPO, "python")
    env.setdefault("OMP_NUM_THREADS", "16")
    cmd = [
        PYTHON, os.path.join(REPO, "scripts", "dsv41", "trace_corpus.py"),
        "--model", arm_env.MODEL_PATH,
        "--sessions", SESSIONS,
        "--n", str(n),
        "--skip", os.environ.get("AB_SKIP", "8"),
        "--prompt-tokens", "256",
        "--new-tokens", str(new_tokens),
        "--stop-at-eos",
        "--log-level", "info",
        "--dspark", DRAFT,
        "--out", os.path.join(outdir, f"{arm}.json"),
    ] + (["--graphs"] if graphs else [])
    print(f"=== {arm}: {' '.join(cmd)}", flush=True)
    with open(os.path.join(outdir, f"{arm}.log"), "w") as log:
        return subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, cwd=REPO).returncode


def _last_metrics(path: str):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        lines = [line for line in f.read().splitlines() if line.strip()]
    return json.loads(lines[-1]) if lines else None


def summarize(outdir: str) -> dict:
    """Per arm: tok/s, accept length, verify GPU ms; for graphed arms the overflow, re-verify rate and text parity."""
    reports = {}
    for arm in D23_ARMS:
        path = os.path.join(outdir, f"{arm}.json")
        if os.path.exists(path):
            with open(path) as f:
                reports[arm] = json.load(f)
    eager_texts = [s.get("output_text") for s in reports["eager"]["per_session"]] if "eager" in reports else None
    summary = {}
    for arm, report in reports.items():
        sessions = report["per_session"]
        records = (report.get("dspark_info_record") or {}).get("records", [])
        verify_ms = sorted(r["target_verify_gpu_ms"] for r in records if r.get("target_verify_gpu_ms") is not None)
        entry = {
            "mean_decode_tok_s": report["mean_decode_tok_s"],
            "accept_length": sum(s["completion_tokens"] for s in sessions) / sum(s["spec_verify_ct"] for s in sessions),
            "verify_ms": {
                "n": len(verify_ms),
                "mean": statistics.fmean(verify_ms) if verify_ms else None,
                "p50": verify_ms[len(verify_ms) // 2] if verify_ms else None,
                "p95": verify_ms[min(len(verify_ms) - 1, int(0.95 * len(verify_ms)))] if verify_ms else None,
            },
        }
        if eager_texts is not None and arm != "eager":
            entry["text_matches_eager"] = sum(
                s.get("output_text") == text for s, text in zip(sessions, eager_texts)
            )
        metrics = _last_metrics(os.path.join(outdir, f"{arm}.metrics.jsonl"))
        counters = (metrics or {}).get("counters", {})
        graphed = counters.get("graphed_verify")
        if graphed:
            verifies = graphed["graphed_verify_ct"]
            overflow = counters["residency_gpu"]["gather_overflow"]
            rates = [layer / verifies for layer in overflow]
            entry["graphed_verify_ct"] = verifies
            entry["reverify_rate"] = graphed["verify_overflow_ct"] / verifies
            entry["layer_overflow_rate"] = {"mean": sum(rates) / len(rates), "max": max(rates)}
            entry["insertion_truncated"] = sum(counters["residency_gpu"]["insertion_truncated"])
        summary[arm] = entry
    with open(os.path.join(outdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def main():
    outdir = sys.argv[1]
    arms = sys.argv[2:] or list(D23_ARMS)
    n = int(os.environ.get("AB_SESSIONS", "8"))
    new_tokens = int(os.environ.get("AB_NEW_TOKENS", "128"))
    os.makedirs(outdir, exist_ok=True)
    for arm in arms:
        rc = run(arm, outdir, n, new_tokens)
        print(f"{arm}: rc={rc}", flush=True)
        if rc:
            sys.exit(rc)
    print(json.dumps(summarize(outdir), indent=2), flush=True)


if __name__ == "__main__":
    main()
```

`AB_SKIP` defaults to 8: sessions 0-7 chose the hybrid draft's resident set (`resident-top32.json`), as in §33.4.

- [ ] **Step 5: Run the tests and watch them pass**

Run the CPU command with `<FILES>` =
`test/manual/dsv41/test_trace_corpus.py test/registered/unit/scripts/test_graphed_verify_summary.py` and `<K>` =
`not nothing_matches_this`.
Expected: all pass.

Commit:

```bash
git add scripts/dsv41/trace_corpus.py test/manual/dsv41/test_trace_corpus.py \
  analysis/dsv41-drive/dspark/graphed_verify.py test/registered/unit/scripts/test_graphed_verify_summary.py
git commit -m "feat(dsv41-dspark): the graphed-verify driver -- eager, graphed and re-verify-all arms, and their summary" -m "<trailers>"
git push -q origin dsv41-dspark-graph
```

- [ ] **Step 6: Run the arms on divix01**

Each arm reads checkpoint rows, so take the disk lock, then the GPU lock. The hybrid draft's CPU experts own cores
6-17, so the driver is pinned outside them, as in §33.4's port run.

```bash
ssh divix01 'cd /data/models/slang/nvfp4-work/wt-dsv41-dspark-graph && git fetch -q origin && git checkout -q --detach origin/dsv41-dspark-graph \
  && git log -1 --oneline && export PYTHONPATH=$PWD/python \
  && /data/models/slang/.venv/bin/python -c "import sglang; print(sglang.__file__)" \
  && G=/data/models/slang/nvfp4-work/cc-expert-prediction/analysis/dsv41-dspark/graph-verify/d2-3 && mkdir -p $G \
  && flock /data/models/slang/nvfp4-work/rowimg-disk.lock flock /data/models/slang/nvfp4-work/cc-gpu.lock \
     taskset -c 0-5,18-63 /data/models/slang/.venv/bin/python analysis/dsv41-drive/dspark/graphed_verify.py $G eager graphed reverify \
     > $G/driver.log 2>&1; echo EXIT=${PIPESTATUS[0]}; tail -40 $G/driver.log'
```

Run it with `run_in_background` and wait for its exit. Expected: `EXIT=0` with all three arms at `rc=0`. Then read
`$G/summary.json`.

**Bars.** If one fails, use superpowers:systematic-debugging. Do not loosen a bar.
- `reverify.text_matches_eager == 8`. Every verify of that arm ran the eager MoE and eager attention, as the eager
  arm's did. A mismatch means the graphed forward left state the re-run read (Review Focus 1).
- `graphed.insertion_truncated == 0` and `reverify.insertion_truncated == 0`.
- Every arm finishes its 8 sessions. The graphed arms' logs show the target's decode graph captured, and no
  `RuntimeError`, `trap` or `fail-stop`.

**Recorded, not bars:**
- each arm's tok/s and accept length;
- verify GPU ms (mean, p50, p95), eager against graphed;
- `graphed.reverify_rate`;
- `layer_overflow_rate`: the mean, and the max over layers;
- `graphed.text_matches_eager`.

The graphed arm's MoE runs the fused in-graph kernel while the eager arm runs `_apply_streamed`. Its text may part
from eager's at a near-tie argmax, so that count is reported, not gated.

- [ ] **Step 7: Record it**

Add `### 33.8 D2-3: the DSpark verify in the decode graph, end to end (2026-10-05)` to `DSV41_REFERENCE.md`, after
§33.7 and before `## Sources`. It covers:
- **What changed**, with commits:
  - the manager's flag read and clear, graph-gather suspension, and the trace fields;
  - the DSpark re-verify;
  - no epilogue while the gather is narrowed, and no EXL3 draft graphs;
  - the copy-engine barrier;
  - the gate's new rule (DSpark, static, DIRECT, `MISS_LANES` 1-32), which replaces "verify eagerly only".
- **Why a re-run, and why eager:**
  - the narrowed gather would overflow again;
  - the eager MoE is the path DSpark verify ran on before D2;
  - KV and compressed-cache writes are overwrites at the same `out_cache_loc`;
  - the residency and recorder counters count an overflowed verify twice, a known bias.
- **The protect-list finding** from this plan's header (no wire change).
- **The run:**
  - the command;
  - the three arms' table: tok/s, accept length, verify ms mean/p50/p95, re-verify rate, layer overflow mean/max,
    text matches;
  - the raw data path (`cc-expert-prediction/analysis/dsv41-dspark/graph-verify/d2-3/`).
- **What it decides, against §33.5:**
  - Measurement 1 is the graphed arm's verify ms. It includes the in-graph miss wait, so it is an upper bound on GPU
    compute.
  - The measured re-verify rate is set against §33.5's projected 0.50 at W = 8.
  - Whether v2 (D2-4, multi-token CPU experts) is worth building, by §33.5's rule: measurement 1 ≤ 28 ms, and an
    overflow path that is cheap at an admissible W.
  - If not, graphed DSpark stays shelved and §33.4's eager hybrid draft remains the DSpark path.
- **What D2-3 does not do:**
  - multi-token CPU experts (D2-4);
  - the epilogue under a narrowed gather;
  - draft graphs for an EXL3 draft;
  - the num_active launch sizing (§33.6).

```bash
git add DSV41_REFERENCE.md
git commit -m "docs(dsv41): section 33.8 -- the DSpark verify in the decode graph, end to end, measured" -m "<trailers>"
git push -q origin dsv41-dspark-graph
```
