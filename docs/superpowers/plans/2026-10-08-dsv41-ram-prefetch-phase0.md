# NVMe-to-RAM Prefetch, Phase 0 (replay go/no-go) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Decide, from a DSpark route capture replayed through a per-group gate-chain model, whether a next-layer gate predictor hides enough forced NVMe misses to build the live prefetch (the spec's Phase 1).

**Architecture:** Widen the stage trace's router capture from one token to the verify's `[M, H]` so a `dspark-both` arm logs every verify forward's routes, misses, hot sets and per-token router inputs; rank each token's next-layer experts offline with the checkpoint's gates; replay the capture through a simulator whose layer is the measured DSpark chain per NUMA group (DMA, hit job, serial forced-miss landings, miss job, gate) over a two-group pinned tier, with the existing priority NVMe queue; compare `gate` h=1 against `oracle` h=1.

**Tech Stack:** Python 3 (torch, numpy, msgspec), the stage trace (`exl3_stream_trace.py`), `scripts/dsv41/tier_sim.py`, `analysis/dsv41-drive/prefetch-replay/`, `benchmarks/dsv41_baseline/arm_env.py`; runs on divix01 under the run protocol.

**Spec:** `docs/superpowers/specs/2026-10-08-dsv41-ram-prefetch-design.md` (section "Phase 0: replay go/no-go")

## Global Constraints

- Code is written on the laptop, committed, pushed, and run on divix01 in a private worktree (`.claude/rules/divix01-run-protocol.md`): `PYTHONPATH=$PWD/python`, `taskset -c 0-63`, `OMP_NUM_THREADS` capped, read `PIPESTATUS[0]`, never the production checkout, no rsync/scp of trees.
- divix01 root disk is nearly full: every output goes under `/data/models/slang/nvfp4-work/ram-prefetch/`; set `TMPDIR` there.
- GPU work takes `rowimg-disk.lock` then `cc-gpu.lock` (`arm_env.GPU_LOCK`), on cores 32-63; cores 64-71 stay free. The capture arm runs only while production is down (it needs the GPU); ask the owner before stopping production, never stop it yourself.
- The capture arm is a diagnostic, not a throughput number; never quote its ms/token as a result.
- Env vars follow `.claude/skills/env-var-conventions/SKILL.md`; no new env var is added in this phase.
- Comments explain why, never what (`.claude/rules/comment-style.md`); no TODOs.
- Commit trailers: `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>` and `Claude-Session: https://claude.ai/code/session_01UuvidggXj3iNqZnXmZSTqQ`. Stage files by name.
- Go rule (owner's): build Phase 1 if `gate` h=1 at its best budget saves at least half of what `oracle` h=1 saves, at the 2.2 ms/row calibration.

## Review Focus

1. A verify forward whose token count is below the graph's `M_max` (the last draft tokens are padding): the replay must score only the forward's live tokens, never padding rows, or the predictor issues reads for garbage routes. Pinned in Task 4 (`test_live_tokens_only_are_ranked`).
2. A schema-1 router capture (one token, no `.ids.bin`) opened by the new loader: it must still load with `tokens == 1`, since every earlier replay reads one. Pinned in Task 2 (`test_schema_1_capture_loads_as_one_token`).
3. A forward whose two groups both miss: the gate is the slower group, and a speculative row for group 1 must not shorten group 0's chain. Pinned in Task 5 (`test_gate_is_the_slower_group`).
4. A demand for a row still filling speculatively: the lane waits for that landing (promotion) and is not read twice; counted as `late`, not as a RAM hit. Pinned in Task 5 (`test_a_demand_on_a_filling_row_waits_and_is_late`).
5. The capture arm killed before shutdown: the route log's `final` read never runs and the last entries are lost; the driver must shut the server down normally and the loader must refuse a run with `dropped_before`. Pinned in Task 6 (`test_the_driver_refuses_a_capture_with_dropped_forwards`).

---

### Task 1: Router capture of `[M, H]` verify inputs (schema 2)

**Files:**
- Modify: `python/sglang/srt/layers/moe/exl3_stream_trace.py` (`ROUTER_CAPTURE_SCHEMA`, `RouterCapture.write`, `GraphRouteLog.record_router`)
- Modify: `python/sglang/srt/layers/quantization/exl3/exl3.py:708` (the `record_router` call)
- Test: `test/registered/unit/layers/moe/test_exl3_stream_trace.py`

**Interfaces:**
- Consumes: `GraphRouteLog.record(row, routes, count)` (unchanged), `on_pre_forward` meta `tokens` (unchanged).
- Produces: `GraphRouteLog.record_router(row, x: Tensor[M, H], topk_ids: Tensor[M, topk], topk_weights: Tensor[M, topk])`; side files `<prefix>.x.bin` bf16 `[records, layers, M_max, hidden]`, `<prefix>.ids.bin` int32 `[records, layers, M_max, topk]`, `<prefix>.w.bin` fp32 `[records, layers, M_max, topk]`, `<prefix>.seq.bin` int64 `[records, 3]` = (seq, pass_id, tokens); header `{"schema": 2, "tokens": M_max, "hidden", "topk", "layer_ids", ...}`.

- [ ] **Step 1: Read the current tests that pin the one-token behaviour**

Run: `grep -n "_router_input\|_load_router\|refuses_a_second_token" test/registered/unit/layers/moe/test_exl3_stream_trace.py`
Expected: the helpers at lines ~358-378 and the test `test_router_capture_refuses_a_second_token_and_a_ring_sized_after_reads` at ~438.

- [ ] **Step 2: Rewrite the helpers for M tokens and add the failing tests**

Replace `_router_input` and `_load_router` and add two tests at the end of the file:

```python
def _router_input(step, row, hidden=16, topk=6, tokens=1):
    """A step-, layer- and token-dependent router input (exact in bf16), top-k ids and weights."""
    base = torch.arange(hidden, dtype=torch.float32) + 100.0 * step + 1000.0 * row
    x = torch.stack([base + 10000.0 * t for t in range(tokens)]).to(torch.bfloat16)
    ids = torch.stack([(torch.arange(topk) + step + row + t) % 64 for t in range(tokens)]).long()
    w = torch.stack([torch.linspace(0.1, 0.6, topk) + 0.01 * t for t in range(tokens)]).float()
    return x, ids, w


def _load_router(prefix):
    import numpy as np

    with open(prefix + ".json") as f:
        header = json.load(f)
    layers, tokens = len(header["layer_ids"]), header["tokens"]
    hidden, topk = header["hidden"], header["topk"]
    x = np.fromfile(prefix + ".x.bin", dtype=np.uint16).reshape(-1, layers, tokens, hidden)
    ids = np.fromfile(prefix + ".ids.bin", dtype=np.int32).reshape(-1, layers, tokens, topk)
    w = np.fromfile(prefix + ".w.bin", dtype=np.float32).reshape(-1, layers, tokens, topk)
    keys = np.fromfile(prefix + ".seq.bin", dtype=np.int64).reshape(-1, 3)
    return header, x, ids, w, keys


def test_router_capture_holds_every_token_of_a_verify_forward(tmp_path, monkeypatch):
    """A verify forward records [M, H] inputs, [M, topk] ids and weights per layer, and the forward's live token
    count beside its seq, so a replay can score each token's next layer."""
    from sglang.srt.layers.moe import exl3_stream_trace as module

    path, prefix = tmp_path / "trace.jsonl", str(tmp_path / "router")
    trace = module.Exl3StreamTrace(str(path))
    layers, tokens = 3, 4
    log = module.GraphRouteLog(layers=layers, width=8, device="cpu", depth=8, margin=2)
    log.enable_router(prefix)
    for row, layer in enumerate((4, 5, 9)):
        log.bind(row, layer, 28)
    trace.graph_seq_source, trace.forward_meta_source = log.read_seq, log.current_meta

    def forward(step):
        for row, routes in enumerate(_forward_routes(step, layers)):
            log.record(row, torch.tensor(routes, dtype=torch.int64), torch.tensor([1], dtype=torch.int32))
            log.record_router(row, *_router_input(step, row, tokens=tokens))

    monkeypatch.setattr(module, "capturing_graphs", lambda: True)
    forward(99)  # warmup: sizes the rings at the graph's widest verify, never written
    monkeypatch.setattr(module, "capturing_graphs", lambda: False)
    assert log.router_x.shape == (8, layers, tokens, 16) and log.router_ids.shape == (8, layers, tokens, 6)
    live = [tokens, tokens - 1, tokens, tokens - 1, tokens]
    for step in range(5):
        log.on_pre_forward(10 + step, _batch("target_verify", ["req-a"], live[step]))
        forward(step)
        log.poll(trace)
    log.poll(trace, final=True)
    trace.close()
    header, x, ids, w, keys = _load_router(prefix)
    assert header["schema"] == 2 and header["tokens"] == tokens
    assert x.shape == (5, layers, tokens, 16) and ids.shape == w.shape == (5, layers, tokens, 6)
    assert keys[:, 2].tolist() == live
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    steps = [line for line in lines if line["kind"] == "graph_routes"]
    assert [line["router"] for line in steps] == list(range(5))
    assert [line["phase"] for line in steps] == ["target_verify"] * 5 and [line["forward_tokens"] for line in steps] == live
    for step in range(5):
        for row in range(layers):
            want_x, want_ids, want_w = _router_input(step, row, tokens=tokens)
            assert torch.equal(torch.from_numpy(x[step, row].astype("int16")).view(torch.bfloat16), want_x)
            assert torch.equal(torch.from_numpy(ids[step, row]).long(), want_ids)
            assert torch.allclose(torch.from_numpy(w[step, row]), want_w)


def test_router_capture_refuses_a_different_token_count_and_a_ring_sized_after_reads(tmp_path):
    from sglang.srt.layers.moe import exl3_stream_trace as module

    log = module.GraphRouteLog(layers=2, width=8, device="cpu", depth=8, margin=2)
    log.enable_router(str(tmp_path / "router"))
    log.record(0, torch.tensor([1], dtype=torch.int64), torch.tensor([0], dtype=torch.int32))
    log.record_router(0, *_router_input(0, 0, tokens=3))
    with pytest.raises(ValueError, match="holds 3 tokens"):
        log.record_router(0, *_router_input(0, 0, tokens=2))
    late = module.GraphRouteLog(layers=2, width=8, device="cpu", depth=8, margin=2)
    late.enable_router(str(tmp_path / "late"))
    late._host = []  # a read already sized its pinned copies without the router rings
    with pytest.raises(RuntimeError, match="warmup forward"):
        late.record_router(0, *_router_input(0, 0, tokens=1))
```

The file's `_batch(mode, rids, tokens)` fake (line ~244) treats only `extend` as an extend mode; the real `ForwardMode.is_extend()` is also true for `TARGET_VERIFY`, and the verify's token count arrives as `extend_num_tokens`. Change it to:

```python
def _batch(mode, rids, tokens):
    extend = mode in ("extend", "target_verify")
    return SimpleNamespace(
        forward_mode=SimpleNamespace(name=mode.upper(), is_extend=lambda: extend),
        rids=rids, batch_size=1, extend_num_tokens=tokens if extend else None,
    )
```

Then update the existing tests that call `record_router(row, *_router_input(step, row))` and `_load_router` (lines ~395-421, ~455-480): the helper now returns three tensors and `_load_router` five values, with `tokens == 1`, so `log.router_x.shape == (8, layers, 1, 16)` there. Delete `test_router_capture_refuses_a_second_token_and_a_ring_sized_after_reads` (replaced above). (`_forward_routes`, `_batch` and the `module` import are the file's existing helpers.)

- [ ] **Step 3: Run the router tests to verify they fail**

Run: `ssh divix01 'cd /data/models/slang/nvfp4-work/wt-ram-prefetch && PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 TMPDIR=/data/models/slang/nvfp4-work/ram-prefetch/tmp taskset -c 0-63 /data/models/slang/.venv/bin/python -m pytest -q -p no:randomly test/registered/unit/layers/moe/test_exl3_stream_trace.py -k router 2>&1 | tail -5; echo EXIT=${PIPESTATUS[0]}'`
Expected: FAIL — `TypeError: record_router() takes 4 positional arguments but 5 were given` for the new and updated tests; EXIT=1.

(Commit the tests first so the worktree exists: `git add test/registered/unit/layers/moe/test_exl3_stream_trace.py && git commit -m "Test the router capture of every verify token"`, push, then `git -C /data/models/slang/sglang fetch origin && git -C /data/models/slang/sglang worktree add --detach /data/models/slang/nvfp4-work/wt-ram-prefetch origin/codex/dsv41-ram-prefetch`; later tasks `git -C /data/models/slang/nvfp4-work/wt-ram-prefetch fetch origin && git -C ... checkout --detach origin/codex/dsv41-ram-prefetch`.)

- [ ] **Step 4: Implement schema 2**

In `exl3_stream_trace.py`:

```python
ROUTER_CAPTURE_SCHEMA = 2
```

`RouterCapture.write` becomes:

```python
    def write(
        self,
        log: GraphRouteLog,
        seq: int,
        pass_id: int,
        tokens: int,
        x: torch.Tensor,
        ids: torch.Tensor,
        weights: torch.Tensor,
    ) -> int:
        """Append one forward's record for ``log``; returns its record number.

        ``x`` is ``[layers, M_max, hidden]``, ``ids`` and ``weights`` ``[layers, M_max, topk]``; ``tokens`` is the
        forward's live count, the rows past it being the graph's padding. The first call writes the ``.json``
        header and opens the data files.
        """
        if self._files is None:
            header = {
                "schema": ROUTER_CAPTURE_SCHEMA,
                "run": log.run,
                "layer_ids": list(log.layer_ids),
                "tokens": int(x.shape[-2]),
                "hidden": int(x.shape[-1]),
                "topk": int(weights.shape[-1]),
                "x_dtype": "bfloat16",
                "ids_dtype": "int32",
                "w_dtype": "float32",
                "depth": log.depth,
            }
            with open(self.prefix + ".json", "w") as f:
                json.dump(header, f)
            self._files = [
                open(self.prefix + suffix, "ab")
                for suffix in (".x.bin", ".ids.bin", ".w.bin", ".seq.bin")
            ]
        x_file, ids_file, w_file, seq_file = self._files
        x_file.write(x.contiguous().view(torch.int16).numpy().tobytes())
        ids_file.write(ids.contiguous().to(torch.int32).numpy().tobytes())
        w_file.write(weights.contiguous().numpy().tobytes())
        seq_file.write(
            torch.tensor([seq, pass_id, tokens], dtype=torch.int64).numpy().tobytes()
        )
        self.records += 1
        return self.records - 1
```

Update the class docstring's file list to the four files and shapes above.

`GraphRouteLog.record_router` becomes:

```python
    def record_router(
        self, row: int, x: torch.Tensor, topk_ids: torch.Tensor, topk_weights: torch.Tensor
    ) -> None:
        """Log one layer's router input, top-k ids and weights for every token of the forward into the slot.

        Captured in the graph. Must follow the layer's ``record``: row 0's takes the
        slot. The rings take the first call's token count: a graph is captured at its
        widest verify, so every replay writes that many rows and the live count travels
        in the forward's meta.
        """
        tokens, hidden = int(x.shape[0]) if x.dim() > 1 else 1, int(x.shape[-1])
        topk = int(topk_weights.shape[-1])
        if self.router_x is None:
            if _stream_capturing() or self._host is not None:
                raise RuntimeError(
                    "the router rings must be allocated by a warmup forward, before capture and reads"
                )
            device = self.routes.device
            shape = (self.depth, self.layers, tokens)
            self.router_x = torch.zeros((*shape, hidden), dtype=torch.bfloat16, device=device)
            self.router_ids = torch.zeros((*shape, topk), dtype=torch.int32, device=device)
            self.router_w = torch.zeros((*shape, topk), dtype=torch.float32, device=device)
        want = tuple(self.router_x.shape[-2:]), tuple(self.router_w.shape[-2:])
        if (tokens, hidden) != want[0] or (tokens, topk) != want[1]:
            raise ValueError(
                f"router capture holds {want[0][0]} tokens of [{want[0][1]}] and [{want[1][1]}], got "
                f"{tokens} tokens of {tuple(x.shape)} and {tuple(topk_weights.shape)}"
            )
        self.router_x[:, row].index_copy_(
            0, self.slot, x.reshape(1, tokens, hidden).to(torch.bfloat16)
        )
        self.router_ids[:, row].index_copy_(
            0, self.slot, topk_ids.reshape(1, tokens, topk).to(torch.int32)
        )
        self.router_w[:, row].index_copy_(
            0, self.slot, topk_weights.reshape(1, tokens, topk).float()
        )
```

Add `self.router_ids: Optional[torch.Tensor] = None` beside `router_x`/`router_w` in `__init__`. In `_ring`, replace the router line with `ring += [self.router_x, self.router_ids, self.router_w] if self.router_x is not None else []`. In `_emit`, replace the unpacking and the write:

```python
        router_x, router_ids, router_w = ring[-3:] if self.router_x is not None else (None, None, None)
        ...
            if router_x is not None:
                record = self.router.write(
                    self, s, pass_id, int((meta or {}).get("tokens", 0)) or router_x.shape[-2],
                    router_x[slot], router_ids[slot], router_w[slot],
                )
```

(`meta` is popped before this point in the loop; keep the existing order and read `tokens` from it.)

In `exl3.py:708`: `backend.route_log.record_router(backend.row, x, topk_ids, topk_weights)`.

- [ ] **Step 5: Run the router tests to verify they pass, then the whole file**

Run: the Step 3 command, then without `-k router`.
Expected: PASS for every test; EXIT=0 both times.

- [ ] **Step 6: Commit**

```bash
git add python/sglang/srt/layers/moe/exl3_stream_trace.py python/sglang/srt/layers/quantization/exl3/exl3.py test/registered/unit/layers/moe/test_exl3_stream_trace.py
git commit -m "Capture every verify token's router input, ids and weights (router capture schema 2)"
```

---

### Task 2: Load schema-2 captures in `router_score`

**Files:**
- Modify: `analysis/dsv41-drive/router-capture/router_score.py` (`RouterCapture`, `load_capture`)
- Test: `test/manual/dsv41/test_router_score.py` (CPU-only tests; the file already imports the module)

**Interfaces:**
- Produces: `RouterCapture(header, x, ids, w, tokens, record_of_seq)` where `x` is a memmap `uint16 [records, layers, M_max, hidden]`, `ids` `int32 [records, layers, M_max, topk]` (all -1 for schema 1), `w` `float32 [records, layers, M_max, topk]`, `tokens` `int64 [records]` (all 1 for schema 1).

- [ ] **Step 1: Write the failing tests**

Append to `test/manual/dsv41/test_router_score.py`:

```python
def _write_capture(prefix, *, schema, records=3, layers=2, tokens=2, hidden=4, topk=3):
    import numpy as np

    header = {"schema": schema, "layer_ids": list(range(layers)), "hidden": hidden, "topk": topk}
    if schema == 2:
        header["tokens"] = tokens
    else:
        tokens = 1
    with open(prefix + ".json", "w") as f:
        json.dump(header, f)
    np.arange(records * layers * tokens * hidden, dtype=np.uint16).tofile(prefix + ".x.bin")
    np.arange(records * layers * tokens * topk, dtype=np.float32).tofile(prefix + ".w.bin")
    if schema == 2:
        np.arange(records * layers * tokens * topk, dtype=np.int32).tofile(prefix + ".ids.bin")
        keys = np.stack([np.arange(records), np.arange(records) + 100, np.array([2, 1, 2])], axis=1)
    else:
        keys = np.stack([np.arange(records), np.arange(records) + 100], axis=1)
    keys.astype(np.int64).tofile(prefix + ".seq.bin")
    return tokens


def test_schema_2_capture_loads_every_token_with_its_live_count(tmp_path):
    prefix = str(tmp_path / "router")
    _write_capture(prefix, schema=2)
    cap = router_score.load_capture(prefix)
    assert cap.x.shape == (3, 2, 2, 4) and cap.ids.shape == cap.w.shape == (3, 2, 2, 3)
    assert cap.tokens.tolist() == [2, 1, 2]
    assert cap.record_of_seq == {0: 0, 1: 1, 2: 2}
    assert cap.ids[1, 0, 1].tolist() == [9, 10, 11]


def test_schema_1_capture_loads_as_one_token(tmp_path):
    prefix = str(tmp_path / "router")
    _write_capture(prefix, schema=1)
    cap = router_score.load_capture(prefix)
    assert cap.x.shape == (3, 2, 1, 4) and cap.w.shape == (3, 2, 1, 3)
    assert cap.tokens.tolist() == [1, 1, 1] and (cap.ids == -1).all()
```

- [ ] **Step 2: Run them to verify they fail**

Run: `ssh divix01 'cd /data/models/slang/nvfp4-work/wt-ram-prefetch && PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES= TMPDIR=/data/models/slang/nvfp4-work/ram-prefetch/tmp taskset -c 0-63 /data/models/slang/.venv/bin/python -m pytest -q -p no:randomly test/manual/dsv41/test_router_score.py -k "schema" 2>&1 | tail -5; echo EXIT=${PIPESTATUS[0]}'`
Expected: FAIL — `AttributeError: 'RouterCapture' object has no attribute 'tokens'` (schema 2 also fails the reshape); EXIT=1.

- [ ] **Step 3: Implement the loader**

In `router_score.py`, the struct and loader become:

```python
class RouterCapture(msgspec.Struct):
    """A router capture's header, its ``x`` memmap ``[records, layers, tokens, hidden]`` (bf16 words), ``ids``
    ``[records, layers, tokens, topk]`` (-1 for a schema-1 capture), ``w`` ``[records, layers, tokens, topk]``,
    each record's live token count and the record of each graph seq."""

    header: dict
    x: np.ndarray
    ids: np.ndarray
    w: np.ndarray
    tokens: np.ndarray
    record_of_seq: dict[int, int]


def load_capture(prefix: str) -> RouterCapture:
    with open(prefix + ".json") as f:
        header = json.load(f)
    layers, hidden, topk = len(header["layer_ids"]), header["hidden"], header["topk"]
    tokens = int(header.get("tokens", 1))
    x = np.memmap(prefix + ".x.bin", dtype=np.uint16, mode="r").reshape(-1, layers, tokens, hidden)
    w = np.fromfile(prefix + ".w.bin", dtype=np.float32).reshape(-1, layers, tokens, topk)
    if header.get("schema", 1) >= 2:
        ids = np.fromfile(prefix + ".ids.bin", dtype=np.int32).reshape(-1, layers, tokens, topk)
        keys = np.fromfile(prefix + ".seq.bin", dtype=np.int64).reshape(-1, 3)
        live = keys[:, 2]
    else:
        ids = np.full(w.shape, -1, dtype=np.int32)
        keys = np.fromfile(prefix + ".seq.bin", dtype=np.int64).reshape(-1, 2)
        live = np.ones(len(keys), dtype=np.int64)
    if not (len(x) == len(w) == len(keys) == len(ids)):
        raise ValueError(f"{prefix}: side files disagree on the record count ({len(x)}, {len(w)}, {len(keys)})")
    return RouterCapture(header, x, ids, w, live, {int(seq): i for i, seq in enumerate(keys[:, 0])})
```

Then fix the one-token readers in the same file: every `capture.x[records, src_layer]` becomes `capture.x[records, src_layer, 0]` (grep `capture.x[` and `capture.w[`; `gate_rankings.py` line ~96 too: `capture.x[records, src_layer, 0]`). The h=0 self-check and `rank_horizon` keep working on token 0, which is the only token of a schema-1 capture.

- [ ] **Step 4: Run the tests to verify they pass, plus the file's other CPU tests**

Run: the Step 2 command without `-k`.
Expected: PASS (GPU tests in the file skip without a device); EXIT=0.

- [ ] **Step 5: Commit**

```bash
git add analysis/dsv41-drive/router-capture/router_score.py analysis/dsv41-drive/prefetch-replay/gate_rankings.py test/manual/dsv41/test_router_score.py
git commit -m "Load schema-2 router captures with every token and its live count"
```

---

### Task 3: Treat the DSpark verify as the replay's decode step

**Files:**
- Modify: `scripts/dsv41/tier_sim.py` (`load_forwards`)
- Test: `test/registered/unit/layers/moe/test_exl3_stream_trace.py`

**Interfaces:**
- Produces: each graph forward dict gains `"router": int | None` (its router capture record) and `"verify": bool` (`phase == "target_verify"`); `phase` is unchanged.

- [ ] **Step 1: Write the failing test**

Append to `test/registered/unit/layers/moe/test_exl3_stream_trace.py`:

```python
def test_load_forwards_marks_verify_forwards_and_keeps_their_router_record(tmp_path):
    """A DSpark verify forward is a graph forward in phase target_verify with M tokens; the replay treats it as the
    decode step, and its router capture record travels with it."""
    import sys
    sys.path.insert(0, "scripts/dsv41")
    from tier_sim import load_forwards

    path = tmp_path / "trace.jsonl"
    lines = [
        {"kind": "graph_routes_header", "schema": 3, "run": "r", "layer_ids": [0, 1], "hot_capacity": [4, 4]},
        {"kind": "graph_routes", "schema": 3, "seq": 0, "forward_pass_id": 7, "phase": "target_verify",
         "rids": ["a"], "forward_tokens": 6, "routes": [[1, 2], [3]], "misses": [1, 0], "router": 0},
        {"kind": "graph_routes", "schema": 3, "seq": 1, "forward_pass_id": 8, "phase": "decode",
         "rids": ["a"], "forward_tokens": 1, "routes": [[1], [3]], "misses": [0, 0]},
    ]
    path.write_text("".join(json.dumps(line) + "\n" for line in lines))
    forwards = load_forwards(str(path))["forwards"]
    assert [f["verify"] for f in forwards] == [True, False]
    assert [f["router"] for f in forwards] == [0, None]
    assert forwards[0]["tokens"] == 6
```

- [ ] **Step 2: Run it to verify it fails**

Run: `ssh divix01 'cd /data/models/slang/nvfp4-work/wt-ram-prefetch && PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 TMPDIR=/data/models/slang/nvfp4-work/ram-prefetch/tmp taskset -c 0-63 /data/models/slang/.venv/bin/python -m pytest -q -p no:randomly test/registered/unit/layers/moe/test_exl3_stream_trace.py -k verify_forwards 2>&1 | tail -5; echo EXIT=${PIPESTATUS[0]}'`
Expected: FAIL — `KeyError: 'verify'`; EXIT=1.

- [ ] **Step 3: Implement**

In `tier_sim.load_forwards`, the graph event dict gains two keys after `"hot"`:

```python
            "router": line.get("router"),
            "verify": line.get("phase") == "target_verify",
```

and the docstring's list of graph keys names them: "``router`` (its router capture record, or None) and ``verify`` (a DSpark target verify: the step of a speculative decode)".

- [ ] **Step 4: Run the test to verify it passes, then the trace test file and the service test that imports tier_sim**

Run: the Step 2 command without `-k`, then `... -m pytest -q -p no:randomly test/registered/unit/layers/moe/test_exl3_ram_miss_service.py 2>&1 | tail -3; echo EXIT=${PIPESTATUS[0]}`.
Expected: PASS; EXIT=0 both.

- [ ] **Step 5: Commit**

```bash
git add scripts/dsv41/tier_sim.py test/registered/unit/layers/moe/test_exl3_stream_trace.py
git commit -m "Mark DSpark verify forwards and their router records in load_forwards"
```

---

### Task 4: Per-token next-layer gate rankings for a verify capture

**Files:**
- Create: `analysis/dsv41-drive/prefetch-replay/verify_gate_rankings.py`
- Test: `analysis/dsv41-drive/prefetch-replay/test_verify_gate_rankings.py`

**Interfaces:**
- Consumes: `router_score.load_capture` (Task 2), `gate_rankings.load_gates(model_dir, layer_ids) -> (W [layers, 384, hidden] fp32, bias [layers, 384])`, `tier_sim.load_forwards` (Task 3).
- Produces: `rank_verify(loaded, capture, W, bias, *, horizons, depth) -> dict` with `order int16 [steps, layers, M_max, H+1, depth]`, `score float32 [same]` (the biased score), `valid bool [steps, layers, H+1]`, `tokens int64 [steps]`, `seq int64 [steps]`, `h0_route_match float`; the CLI writes it as `.npz`. Step `s` is the s-th forward with `verify == True`; `valid[s, T, h]` is false when `T - h < 0` (no cross-step source: the next verify's tokens are unknown).

- [ ] **Step 1: Write the failing tests**

```python
"""verify_gate_rankings.py: per-token next-layer rankings of a verify capture. CPU only."""

import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / ".." / "router-capture"))

import router_score  # noqa: E402
from verify_gate_rankings import rank_verify  # noqa: E402

LAYERS, EXPERTS, HIDDEN, TOPK, M = 3, 8, 4, 2, 3


def _gates(seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(LAYERS, EXPERTS, HIDDEN, generator=g), torch.randn(LAYERS, EXPERTS, generator=g)


def _biased(W, bias, layer, x):
    return torch.nn.functional.softplus(x @ W[layer].T).sqrt() + bias[layer]


def _capture(tmp_path, W, bias, tokens_per_step):
    """A capture whose ids are each token's true top-k of its own layer, so h=0 reproduces them."""
    prefix = str(tmp_path / "router")
    g = torch.Generator().manual_seed(1)
    steps = len(tokens_per_step)
    x = torch.randn(steps, LAYERS, M, HIDDEN, generator=g).to(torch.bfloat16)
    ids = torch.full((steps, LAYERS, M, TOPK), -1, dtype=torch.int32)
    w = torch.zeros(steps, LAYERS, M, TOPK)
    for s in range(steps):
        for layer in range(LAYERS):
            top = torch.topk(_biased(W, bias, layer, x[s, layer].float()), TOPK, dim=-1)
            ids[s, layer], w[s, layer] = top.indices.int(), top.values
    with open(prefix + ".json", "w") as f:
        json.dump({"schema": 2, "layer_ids": list(range(LAYERS)), "tokens": M, "hidden": HIDDEN, "topk": TOPK}, f)
    x.view(torch.int16).numpy().astype(np.uint16).tofile(prefix + ".x.bin")
    ids.numpy().tofile(prefix + ".ids.bin")
    w.numpy().tofile(prefix + ".w.bin")
    np.stack([np.arange(steps), np.arange(steps) + 10, np.asarray(tokens_per_step)], axis=1).astype(np.int64).tofile(prefix + ".seq.bin")
    forwards = [{"kind": "graph", "seq": s, "phase": "target_verify", "verify": True, "tokens": int(t),
                 "router": s, "rids": ["a"], "routes": {layer: sorted(set(ids[s, layer, :t].flatten().tolist())) for layer in range(LAYERS)},
                 "hot": None, "misses": {}} for s, t in enumerate(tokens_per_step)]
    return {"layer_ids": list(range(LAYERS)), "forwards": forwards}, router_score.load_capture(prefix), x


def test_h0_reproduces_each_live_tokens_routes_and_h1_scores_the_next_layer_on_this_layers_input(tmp_path):
    W, bias = _gates()
    loaded, cap, x = _capture(tmp_path, W, bias, [3, 2])
    out = rank_verify(loaded, cap, W, bias, horizons=1, depth=TOPK)
    assert out["order"].shape == (2, LAYERS, M, 2, TOPK) and out["tokens"].tolist() == [3, 2]
    assert out["h0_route_match"] == 1.0
    want = torch.topk(_biased(W, bias, 1, x[0, 0, 1].float()), TOPK).indices.tolist()
    assert out["order"][0, 1, 1, 1].tolist() == want  # step 0, target layer 1, token 1, h=1: gate_1 on layer 0's x
    assert out["valid"][0, 0, 1] == False and out["valid"][0, 1, 1] == True  # noqa: E712


def test_live_tokens_only_are_ranked(tmp_path):
    """Padding rows past the forward's live count must not rank: their routes are garbage."""
    W, bias = _gates()
    loaded, cap, _ = _capture(tmp_path, W, bias, [2])
    out = rank_verify(loaded, cap, W, bias, horizons=1, depth=TOPK)
    assert (out["order"][0, :, 2] == -1).all() and (out["order"][0, :, :2] >= 0).all()
```

- [ ] **Step 2: Run them to verify they fail**

Run: `ssh divix01 'cd /data/models/slang/nvfp4-work/wt-ram-prefetch && PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES= TMPDIR=/data/models/slang/nvfp4-work/ram-prefetch/tmp taskset -c 0-63 /data/models/slang/.venv/bin/python -m pytest -q -p no:randomly analysis/dsv41-drive/prefetch-replay/test_verify_gate_rankings.py 2>&1 | tail -5; echo EXIT=${PIPESTATUS[0]}'`
Expected: FAIL — `ModuleNotFoundError: No module named 'verify_gate_rankings'`; EXIT=1.

- [ ] **Step 3: Implement**

```python
"""Per-token next-layer gate rankings of a DSpark verify capture, for verify_replay.py.

For every verify forward (step), target layer T, live token m and horizon h = 0..H, the top ``depth`` experts of
gate_T(x[step, T-h, m]) by biased score sqrt(softplus(W x)) + b, the score the model routes on. h = 0 is the
self-check against the captured top-k ids; h >= 1 is the predictor: layer T's gate on the same token's layer T-h
input. There is no cross-step source (the next verify's tokens come from a draft that has not run), so targets
T < h are invalid. Padding tokens (m >= tokens[step]) rank -1.

Output (npz): ``order`` int16 [steps, layers, M, H+1, depth], ``score`` float32 [same], ``valid`` bool
[steps, layers, H+1], ``tokens`` int64 [steps], ``seq`` int64 [steps], ``h0_route_match``.

Usage: verify_gate_rankings.py TRACE_JSONL ROUTER_PREFIX MODEL_DIR --out rank.npz [--horizons 2] [--depth 12]
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "router-capture"))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "scripts", "dsv41"))
import router_score  # noqa: E402
import tier_sim  # noqa: E402
from gate_rankings import load_gates  # noqa: E402


def verify_steps(loaded: dict) -> list[dict]:
    return [f for f in loaded["forwards"] if f.get("verify")]


def rank_verify(loaded: dict, capture, W: torch.Tensor, bias: torch.Tensor, *, horizons: int, depth: int) -> dict:
    steps = verify_steps(loaded)
    records = np.asarray([capture.record_of_seq[f["seq"]] for f in steps], dtype=np.int64)
    tokens = capture.tokens[records]
    n, layers, M = len(steps), len(loaded["layer_ids"]), int(capture.x.shape[2])
    H = horizons
    order = np.full((n, layers, M, H + 1, depth), -1, dtype=np.int16)
    score = np.zeros((n, layers, M, H + 1, depth), dtype=np.float32)
    valid = np.zeros((n, layers, H + 1), dtype=bool)
    live = np.arange(M)[None, :] < tokens[:, None]  # [steps, M]
    for src in range(layers):
        x = torch.from_numpy(router_score.bf16_to_f32(np.asarray(capture.x[records, src]))).reshape(n * M, -1)
        for h in range(H + 1):
            target = src + h
            if target >= layers:
                continue
            biased = torch.nn.functional.softplus(x @ W[target].T).sqrt() + bias[target]
            top = torch.topk(biased, depth, dim=-1)
            o = top.indices.numpy().astype(np.int16).reshape(n, M, depth)
            sc = top.values.numpy().astype(np.float32).reshape(n, M, depth)
            o[~live] = -1
            sc[~live] = 0.0
            order[:, target, :, h] = o
            score[:, target, :, h] = sc
            valid[:, target, h] = True
    ids = capture.ids[records]  # [steps, layers, M, topk]
    topk = ids.shape[-1]
    matches = [
        set(order[s, t, m, 0, :topk].tolist()) == set(ids[s, t, m].tolist())
        for s in range(n) for t in range(layers) for m in range(int(tokens[s]))
    ]
    return {
        "order": order, "score": score, "valid": valid, "tokens": tokens.astype(np.int64),
        "seq": np.asarray([f["seq"] for f in steps], dtype=np.int64),
        "h0_route_match": float(np.mean(matches)) if matches else float("nan"),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("trace")
    p.add_argument("router_prefix")
    p.add_argument("model")
    p.add_argument("--out", required=True)
    p.add_argument("--horizons", type=int, default=2)
    p.add_argument("--depth", type=int, default=12)
    a = p.parse_args()
    capture = router_score.load_capture(a.router_prefix)
    loaded = tier_sim.load_forwards(a.trace)
    if loaded["layer_ids"] != capture.header["layer_ids"]:
        raise ValueError("route log and router capture disagree on the layers")
    W, bias = load_gates(a.model, capture.header["layer_ids"])
    out = rank_verify(loaded, capture, W, bias, horizons=a.horizons, depth=a.depth)
    print(f"verify steps {len(out['seq'])}; h=0 self-check: route sets reproduced {out['h0_route_match']:.6f}")
    np.savez_compressed(a.out, **out)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: the Step 2 command.
Expected: PASS, 2 passed; EXIT=0.

- [ ] **Step 5: Commit**

```bash
git add analysis/dsv41-drive/prefetch-replay/verify_gate_rankings.py analysis/dsv41-drive/prefetch-replay/test_verify_gate_rankings.py
git commit -m "Rank each verify token's next-layer experts with the checkpoint's gates"
```

---

### Task 5: The verify-chain replay

**Files:**
- Create: `analysis/dsv41-drive/prefetch-replay/verify_replay.py`
- Test: `analysis/dsv41-drive/prefetch-replay/test_verify_replay.py`

**Interfaces:**
- Consumes: `pinned_prefetch_replay.Nvme` (unchanged: `submit(job)`, `promote(job)`, `advance(t)`, `finish(jobs)`, `drain()`, `busy_ms`, `delayed`, `delay_ms`), `tier_sim.load_forwards` (Task 3), `tier_sim.ram_rows_per_layer`, the Task 4 npz.
- Produces: CLI `verify_replay.py TRACE --predictor {none,oracle,gate,noisy} [--ranks NPZ] [--h 1] [--k 1] [--depth 6] [--p 0.5] [--budget 4] [--spec-share 4] [--admit cold|mru] [--queue prio|fifo] [--nvme-row-ms 2.2] [--pieces 8] [--dma-ms 1.2] [--cpu-fixed-ms 0.3] [--cpu-lane-ms 0.6] [--miss-job-ms 1.0] [--gpu-ms 1.0] [--step-ms 20] [--ram-rows 8385] [--node-share 0.5385] [--split "0 1 2 2 3 3 4 5 5"] [--accept 4.3] [--seed 0] [--out JSON]`; `run_one(args, loaded, ranks) -> dict` with the keys in `Replay.report`.

The model, per verify forward (`verify == True`; every other graph forward is skipped, eager forwards replay prefill admissions untimed as before):

- Per layer in order, at time `now` (the layer's post): for each routed expert (deduped), `vram_miss = expert not in hot`. Group `g = expert % 2`. In RAM and ready: a hit lane of group g. In RAM but `ready > now` (filling): a promoted job of group g, `late += 1`. Not in RAM: a forced miss of group g, admitted as a demand read.
- Per group: `cpu_hits = SPLIT[min(hits, len(SPLIT) - 1)]`, `dma = hits - cpu_hits`; `hit_end = max(dma_ms if dma else 0, cpu_fixed_ms + cpu_lane_ms * cpu_hits if cpu_hits else 0)`. The group's miss jobs run serially on its team after the hit job: for each landed row in landing order, `job_end = max(prev_end, hit_end, done) + miss_job_ms`. `gate[g] = max(hit_end, last job_end)` (relative to `now`); a group with no lanes has `gate 0`.
- `layer_end = now + max(gate) + gpu_ms`; `exposed += max(gate)`; speculative candidates for layer `T + h` are issued at `now` (the post), before the reads of this layer's misses are queued, so they queue behind nothing; `now = layer_end`.
- After the last layer, `now += step_ms` (the draft graph and the host between verifies).
- Tier per layer: two `Tier`s, capacities `round(rows_per_layer * node_share)` and the rest, the same victim rule as `pinned_prefetch_replay.Tier`; a group's reads and admissions touch only its tier.
- Predictors: `oracle` (the target layer's true forced misses of group g at issue time, up to `k` per live token on average: `k * tokens`), `gate` (from the npz: for the target layer, every live token's top-`depth` at horizon `h`, candidates ordered by their best score across tokens, those not hot, not in RAM, not filling, first `k * tokens`), `noisy` (oracle with probability `p` per row, else a random wrong expert of the same group). Budget: `--budget` speculative rows per layer across both groups; `--spec-share`, `--admit cold|mru` as before.
- Report: per verify step `step_ms`, `gate_ms` (sum of per-layer exposed), `ram_misses`, `spec_*` as in `pinned_prefetch_replay.report`, plus `ms_per_token = step_ms / accept`, `saved_ms_per_token` against a `--baseline JSON` when given, per-layer exposed, `late`, `harmful`, `demand_rows_delayed`, `demand_delay_mean_ms`, `nvme_busy_frac`, and `gate_p50/p90/p99` over all (layer, step) for calibration.

- [ ] **Step 1: Write the failing tests**

```python
"""verify_replay.py: the per-group gate chain, the two-group tier and the predictors. CPU only."""

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import verify_replay as vr  # noqa: E402

LAYERS = [0, 1, 2]


def _args(**over):
    a = vr.parse(["trace.jsonl"])
    for k, v in over.items():
        setattr(a, k, v)
    return a


def _loaded(forwards):
    return {"layer_ids": LAYERS, "forwards": forwards}


def _verify(seq, routes, hot=None, tokens=2):
    return {"kind": "graph", "seq": seq, "phase": "target_verify", "verify": True, "tokens": tokens, "rids": ["a"],
            "router": seq, "routes": {l: routes.get(l, []) for l in LAYERS}, "hot": {l: (hot or {}).get(l, []) for l in LAYERS},
            "misses": {}}


def _run(forwards, **over):
    a = _args(**over)
    return vr.run_one(a, _loaded(forwards), None)


ROWS = 40  # 14/13/13 rows over the three layers, 8/7/7 of them for group 0: room for every scenario below


def test_a_layer_with_one_forced_miss_waits_for_its_landing_and_its_job():
    """One expert of group 0, not in RAM: gate = row landing (2.2) + miss job (1.0); the layer ends gpu_ms later,
    and the two layers with no lanes cost gpu_ms each."""
    out = _run([_verify(0, {0: [2]})], ram_rows=ROWS, nvme_row_ms=2.2, miss_job_ms=1.0, gpu_ms=1.0, step_ms=0.0)
    assert out["gate_ms_per_step"] == 3.2 and out["step_ms_per_step"] == 3.2 + 1.0 + 2 * 1.0
    assert out["ram_misses_per_step"] == 1


def test_hits_cost_the_split_cpu_job_or_the_dma_whichever_is_later():
    """Four hits of group 0 all in RAM: SPLIT gives 3 CPU lanes (0.3 + 3 * 0.6 = 2.1) against one DMA lane (1.2)."""
    first = _verify(0, {0: [0, 2, 4, 6]})
    second = _verify(1, {0: [0, 2, 4, 6]})
    out = _run([first, second], ram_rows=ROWS, split="0 1 2 2 3 3 4 5 5", dma_ms=1.2, cpu_fixed_ms=0.3, cpu_lane_ms=0.6, gpu_ms=0.0, step_ms=0.0)
    # The first verify reads the four rows (misses); the second hits all four.
    assert out["per_step_gate_ms"][1] == 2.1


def test_gate_is_the_slower_group():
    """Three forced misses issued in route order share one queue: group 0's rows (experts 0, 2) land at 2.2 and 4.4,
    group 1's (expert 1) at 6.6. Group 0's serial jobs end at 5.4, group 1's at 7.6: the layer waits for the slower
    group, and the groups' gates are reported apart."""
    out = _run([_verify(0, {0: [0, 2, 1]})], ram_rows=ROWS, nvme_row_ms=2.2, miss_job_ms=1.0, gpu_ms=0.0, step_ms=0.0)
    assert out["gate_ms_per_step"] == 7.6
    assert out["per_group_gate_ms"] == [5.4, 7.6]


def test_an_oracle_prefetch_turns_the_next_layers_miss_into_a_hit():
    """Layer 0 routes expert 0; layer 1 will route expert 2. With oracle h=1 the read for 2 is issued at layer 0's
    post, queues behind layer 0's demand row (lands at 4.4), and layer 0's chain (3.2) plus its GPU work (3.0)
    outlasts it, so layer 1's gate is one CPU hit lane, not a landing."""
    routes = {0: [0], 1: [2]}
    base = _run([_verify(0, routes)], ram_rows=ROWS, gpu_ms=3.0, step_ms=0.0)
    arm = _run([_verify(0, routes)], ram_rows=ROWS, gpu_ms=3.0, step_ms=0.0, predictor="oracle", h=1, k=1)
    assert base["per_layer_exposed_ms"][1] == 3.2
    assert arm["per_layer_exposed_ms"][1] == 0.3 + 0.6
    assert arm["spec_rows_per_step"] == 1 and arm["precision_target"] == 1.0
    assert arm["step_ms_per_step"] < base["step_ms_per_step"]


def test_a_demand_on_a_filling_row_waits_and_is_late():
    """A speculative read still in flight when its row is demanded is promoted, not read again, and counts late."""
    routes = {0: [0], 1: [2]}
    arm = _run([_verify(0, routes)], ram_rows=ROWS, gpu_ms=0.0, step_ms=0.0, predictor="oracle", h=1, k=1,
               nvme_row_ms=50.0, miss_job_ms=1.0)
    assert arm["late_per_step"] == 1 and arm["ram_misses_per_step"] == 1  # layer 0's own miss only
    assert arm["spec_rows_per_step"] == 1


def test_demand_reads_outrank_speculative_ones_in_prio_mode():
    """With a wrong speculative row queued ahead, a demand row waits at most one piece under prio, a whole row under fifo."""
    routes = {0: [0], 1: [4]}  # the oracle is wrong on purpose: noisy with p=0 issues a random other expert
    prio = _run([_verify(0, routes)], ram_rows=ROWS, gpu_ms=0.0, step_ms=0.0, predictor="noisy", p=0.0, h=1, k=1, seed=3, queue="prio")
    fifo = _run([_verify(0, routes)], ram_rows=ROWS, gpu_ms=0.0, step_ms=0.0, predictor="noisy", p=0.0, h=1, k=1, seed=3, queue="fifo")
    assert prio["demand_delay_mean_ms"] <= 2.2 / 8 + 1e-9
    assert fifo["demand_delay_mean_ms"] >= prio["demand_delay_mean_ms"]


def test_the_tier_splits_rows_between_the_two_groups_by_node_share():
    a = _args(ram_rows=10, node_share=0.6)
    tiers = vr.make_tiers(a, LAYERS)
    assert [[t.capacity for t in pair] for pair in tiers] == [[2, 2], [2, 1], [2, 1]]  # rows per layer 4, 3, 3
```

- [ ] **Step 2: Run them to verify they fail**

Run: `ssh divix01 'cd /data/models/slang/nvfp4-work/wt-ram-prefetch && PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES= TMPDIR=/data/models/slang/nvfp4-work/ram-prefetch/tmp taskset -c 0-63 /data/models/slang/.venv/bin/python -m pytest -q -p no:randomly analysis/dsv41-drive/prefetch-replay/test_verify_replay.py 2>&1 | tail -5; echo EXIT=${PIPESTATUS[0]}'`
Expected: FAIL — `ModuleNotFoundError: No module named 'verify_replay'`; EXIT=1.

- [ ] **Step 3: Implement the simulator**

```python
"""Replay a DSpark verify capture through the per-group lease-gate chain, with NVMe-to-RAM prefetch.

The step is the verify forward (phase target_verify, M tokens). Per layer, each NUMA group g (expert % 2) serves
its lanes: hits split between the DMA (``--dma-ms``) and the CPU hit job (``--cpu-fixed-ms`` + ``--cpu-lane-ms``
per lane, lanes per the calibration ``--split``); forced misses are demand reads on the shared NVMe queue
(pinned_prefetch_replay.Nvme, ``--nvme-row-ms`` per row, priority queue) whose rows, as they land, each take a
serial CPU job (``--miss-job-ms``) after the hit job; the group's gate opens at the last finisher, the layer ends at
the slower group plus ``--gpu-ms``, and the next layer posts then. A verify adds ``--step-ms`` (the draft graph and
the host between verifies). So the lead a speculative read gets is whatever the chain gives, and hiding one layer's
misses shortens the next layer's lead.

Tier: per layer two Tiers, ``round(rows * --node-share)`` slots for group 0 and the rest for group 1, the RamTier
victim rule (lowest stamp, not hot, not wanted, not filling). Prefill forwards replay their admissions untimed.

Predictors, issued at a layer's post for layer T + h of the same verify (there is no cross-step source):
``oracle`` (the target's true forced misses), ``gate`` (verify_gate_rankings.py: every live token's top-depth at
horizon h, ordered by best score, not hot / in RAM / filling), ``noisy`` (oracle rows right with probability p).
``--budget`` speculative rows per layer, ``--k`` per live token, ``--spec-share`` unused rows a layer's group may hold,
``--admit cold|mru``.

Report: per-step and per-token (``--accept`` tokens per verify) times, the gate per layer and group, RAM misses,
speculative precision, late, harmful evictions, demand delay, NVMe busy, and gate percentiles for calibration.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "scripts", "dsv41"))
from pinned_prefetch_replay import CHUNK, NUM_EXPERTS, Nvme, Tier  # noqa: E402
from tier_sim import load_forwards, ram_rows_per_layer  # noqa: E402

GROUPS = 2


def group_of(expert: int) -> int:
    return expert % GROUPS


def make_tiers(a, layer_ids) -> list[list[Tier]]:
    rows = ram_rows_per_layer(a.ram_rows, len(layer_ids), NUM_EXPERTS)
    out = []
    for n in rows:
        g0 = int(round(n * a.node_share))
        out.append([Tier(g0), Tier(n - g0)])
    return out


class Replay:
    def __init__(self, a, layer_ids, ranks) -> None:
        self.a = a
        self.layer_ids = list(layer_ids)
        self.tiers = make_tiers(a, layer_ids)
        self.nvme = Nvme(a.nvme_row_ms, a.pieces, a.queue, 0.0)
        self.split = [int(s) for s in a.split.split()]
        self.ranks = ranks
        self.rng = random.Random(a.seed)
        self.tick = 0
        self.evicted_by_spec = [[dict() for _ in range(GROUPS)] for _ in layer_ids]
        self.st = dict(demand_misses=0, spec_issued=0, spec_used_target=0, spec_used_any=0, spec_late=0, harmful=0,
                       budget_capped=0, ram_hits=0, prefill_misses=0, steps=0, step_ms=0.0, gate_ms=0.0)
        self.per_layer_exposed = np.zeros(len(layer_ids))
        self.per_layer_misses = np.zeros(len(layer_ids))
        self.per_group_gate = np.zeros(GROUPS)
        self.gates: list[float] = []
        self.per_step_gate: list[float] = []
        self.per_step_ms: list[float] = []

    def _next(self) -> int:
        self.tick += 1
        return self.tick

    # ---- tier

    def admit(self, li: int, expert: int, hot: set, protect: set, now: float, kind: str, record=None):
        tier = self.tiers[li][group_of(expert)]
        slot = gone = -1
        owned = {i for i, r in enumerate(tier.spec) if r is not None and not r["used"]}
        if kind == "spec" and self.a.spec_share and len(owned) >= self.a.spec_share:
            try:
                slot, gone = tier.take(hot, protect, now, only=owned)
            except RuntimeError:
                slot = -1
        if slot < 0:
            slot, gone = tier.take(hot, protect, now)
        if gone >= 0 and kind == "spec":
            self.evicted_by_spec[li][group_of(expert)][gone] = True
        tier.spec[slot] = record
        tier.slot_expert[slot] = expert
        tier.where[expert] = slot
        tier.stamp[slot] = 0 if (kind == "spec" and self.a.admit == "cold") else self._next()
        self.evicted_by_spec[li][group_of(expert)].pop(expert, None)
        job = {"t": now, "kind": kind, "layer": li, "expert": expert, "group": group_of(expert)}
        tier.ready[slot] = float("inf")
        tier.job[slot] = job

        def on_done(j, tier=tier, slot=slot, expert=expert):
            if tier.slot_expert[slot] == expert and tier.job[slot] is j:
                tier.ready[slot] = j["done"]

        job["on_done"] = on_done
        self.nvme.submit(job)
        return job

    def where(self, li: int, expert: int):
        tier = self.tiers[li][group_of(expert)]
        slot = tier.where.get(expert)
        return tier, slot

    # ---- prediction

    def candidates(self, s: int, fwd, target_li: int) -> list[int]:
        a = self.a
        if a.predictor == "gate":
            if not self.ranks["valid"][s, target_li, a.h]:
                return []
            order = self.ranks["order"][s, target_li, :, a.h, : a.depth]  # [M, depth]
            score = self.ranks["score"][s, target_li, :, a.h, : a.depth]
            best: dict[int, float] = {}
            for m in range(int(self.ranks["tokens"][s])):
                for e, sc in zip(order[m].tolist(), score[m].tolist()):
                    if e >= 0 and sc > best.get(e, -1e30):
                        best[e] = sc
            return [e for e, _ in sorted(best.items(), key=lambda kv: (-kv[1], kv[0]))]
        if a.predictor in ("oracle", "noisy"):
            return list(dict.fromkeys(fwd["routes"][self.layer_ids[target_li]]))
        return []

    def issue(self, s: int, fwd, li: int, now: float, budget: list) -> None:
        a = self.a
        target_li = li + a.h
        if target_li >= len(self.layer_ids) or self.layer_ids[target_li] == 0:
            return
        target_layer = self.layer_ids[target_li]
        hot = set(fwd["hot"][target_layer]) if fwd.get("hot") else set()
        want = a.k * max(int(fwd.get("tokens", 1)), 1)
        chosen = []
        for e in self.candidates(s, fwd, target_li):
            if len(chosen) >= want:
                break
            tier, slot = self.where(target_li, e)
            if e in hot or slot is not None or e in chosen:
                continue
            chosen.append(e)
        if a.predictor == "noisy":
            out = []
            for e in chosen:
                if self.rng.random() < a.p:
                    out.append(e)
                    continue
                routed = set(fwd["routes"][target_layer])
                while True:
                    w = self.rng.randrange(NUM_EXPERTS)
                    tier, slot = self.where(target_li, w)
                    if w not in routed and w not in hot and slot is None and w not in out and group_of(w) == group_of(e):
                        break
                out.append(w)
            chosen = out
        target_routes = set(fwd["routes"][target_layer])
        for e in chosen:
            if budget[0] <= 0:
                self.st["budget_capped"] += 1
                return
            budget[0] -= 1
            rec = {"target": (s, target_li), "used": False, "right": e in target_routes and e not in hot}
            self.admit(target_li, e, hot, set(chosen), now, "spec", rec)
            self.st["spec_issued"] += 1

    # ---- the chain

    def layer(self, s: int, fwd, li: int, now: float) -> tuple[float, list[float]]:
        """Serve one layer at ``now``; returns (gate of the slower group, per-group gates), in ms after ``now``."""
        layer = self.layer_ids[li]
        hot = set(fwd["hot"][layer]) if fwd.get("hot") else set()
        routes = list(dict.fromkeys(fwd["routes"][layer]))
        wanted = set(routes)
        hits = [0] * GROUPS
        landing: list[list[dict]] = [[] for _ in range(GROUPS)]
        for expert in routes:
            if expert in hot:
                continue
            g = group_of(expert)
            tier, slot = self.where(li, expert)
            if slot is not None:
                rec = tier.spec[slot]
                if rec is not None and not rec["used"]:
                    rec["used"] = True
                    self.st["spec_used_any"] += 1
                    if rec["target"] == (s, li):
                        self.st["spec_used_target"] += 1
                    tier.spec[slot] = None
                tier.stamp[slot] = self._next()
                if tier.ready[slot] > now:
                    job = tier.job[slot]
                    self.nvme.promote(job)
                    landing[g].append(job)
                    self.st["spec_late"] += 1
                else:
                    hits[g] += 1
                    self.st["ram_hits"] += 1
                continue
            self.st["demand_misses"] += 1
            self.per_layer_misses[li] += 1
            if expert in self.evicted_by_spec[li][g]:
                self.st["harmful"] += 1
            landing[g].append(self.admit(li, expert, hot, wanted, now, "demand"))
        self.nvme.finish([j for group in landing for j in group])
        a = self.a
        gates = []
        for g in range(GROUPS):
            cpu = self.split[min(hits[g], len(self.split) - 1)]
            dma = hits[g] - cpu
            hit_end = max(a.dma_ms if dma else 0.0, a.cpu_fixed_ms + a.cpu_lane_ms * cpu if cpu else 0.0)
            end = hit_end
            for job in sorted(landing[g], key=lambda j: j["done"]):
                end = max(end, job["done"] - now) + a.miss_job_ms
            gates.append(end if (hits[g] or landing[g]) else 0.0)
        return max(gates), gates

    def run(self, loaded) -> dict:
        a = self.a
        now = 0.0
        s = -1
        for fwd in loaded["forwards"]:
            if fwd["kind"] != "graph":
                self.nvme.drain()
                now = max(now, self.nvme.free) + 1000.0
                for layer, (experts, _) in fwd["counts"].items():
                    self.prefill(self.layer_ids.index(layer), list(experts), now)
                continue
            if not fwd.get("verify"):
                continue
            s += 1
            step_start, step_gate = now, 0.0
            budget = [a.budget]
            for li in range(len(self.layer_ids)):
                self.nvme.advance(now)
                if a.predictor != "none":
                    self.issue(s, fwd, li, now, budget)
                gate, gates = self.layer(s, fwd, li, now)
                self.per_layer_exposed[li] += gate
                self.per_group_gate += gates
                self.gates.append(gate)
                step_gate += gate
                now += gate + a.gpu_ms
            now += a.step_ms
            self.st["steps"] += 1
            self.st["step_ms"] += now - step_start
            self.st["gate_ms"] += step_gate
            self.per_step_gate.append(step_gate)
            self.per_step_ms.append(now - step_start)
        self.nvme.drain()
        return self.report()

    def prefill(self, li: int, experts: list[int], now: float) -> None:
        for start in range(0, len(experts), CHUNK):
            chunk = experts[start : start + CHUNK]
            protect = set(chunk)
            for expert in chunk:
                tier, slot = self.where(li, expert)
                if slot is not None:
                    tier.stamp[slot] = self._next()
                    continue
                self.st["prefill_misses"] += 1
                slot, _ = tier.take(set(), protect, float("inf"))
                tier.slot_expert[slot] = expert
                tier.where[expert] = slot
                tier.stamp[slot] = self._next()
                tier.ready[slot] = 0.0
                tier.job[slot] = None
                tier.spec[slot] = None

    def report(self) -> dict:
        st, n = self.st, max(self.st["steps"], 1)
        issued = max(st["spec_issued"], 1)
        gates = np.asarray(self.gates) if self.gates else np.zeros(1)
        return {
            "args": {k: (sorted(v) if isinstance(v, set) else v) for k, v in vars(self.a).items()},
            "steps": st["steps"],
            "step_ms_per_step": st["step_ms"] / n,
            "gate_ms_per_step": st["gate_ms"] / n,
            "ms_per_token": st["step_ms"] / n / self.a.accept,
            "ram_misses_per_step": st["demand_misses"] / n,
            "ram_hits_per_step": st["ram_hits"] / n,
            "spec_rows_per_step": st["spec_issued"] / n,
            "precision_target": st["spec_used_target"] / issued,
            "precision_any_use": st["spec_used_any"] / issued,
            "late_per_step": st["spec_late"] / n,
            "wasted_rows_per_step": (st["spec_issued"] - st["spec_used_any"]) / n,
            "harmful_evictions_per_step": st["harmful"] / n,
            "budget_capped_per_step": st["budget_capped"] / n,
            "demand_rows_delayed_per_step": self.nvme.delayed / n,
            "demand_delay_mean_ms": self.nvme.delay_ms / max(self.nvme.delayed, 1),
            "nvme_busy_frac": self.nvme.busy_ms / max(st["step_ms"], 1e-9),
            "gate_p50_ms": float(np.percentile(gates, 50)),
            "gate_p90_ms": float(np.percentile(gates, 90)),
            "gate_p99_ms": float(np.percentile(gates, 99)),
            "per_layer_exposed_ms": (self.per_layer_exposed / n).round(3).tolist(),
            "per_layer_ram_misses": (self.per_layer_misses / n).round(3).tolist(),
            "per_group_gate_ms": (self.per_group_gate / n).round(3).tolist(),
            "per_step_gate_ms": [round(v, 3) for v in self.per_step_gate],
            "per_step_ms": [round(v, 3) for v in self.per_step_ms],
        }


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("trace")
    p.add_argument("--ranks", help="verify_gate_rankings.py output (npz)")
    p.add_argument("--predictor", default="none", choices=["none", "oracle", "gate", "noisy"])
    p.add_argument("--h", type=int, default=1)
    p.add_argument("--k", type=int, default=1, help="speculative rows per live token")
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--p", type=float, default=0.5)
    p.add_argument("--budget", type=int, default=4, help="speculative rows per layer")
    p.add_argument("--queue", default="prio", choices=["prio", "fifo"])
    p.add_argument("--admit", default="cold", choices=["mru", "cold"])
    p.add_argument("--spec-share", type=int, default=4)
    p.add_argument("--nvme-row-ms", type=float, default=2.2)
    p.add_argument("--pieces", type=int, default=8)
    p.add_argument("--dma-ms", type=float, default=1.2)
    p.add_argument("--cpu-fixed-ms", type=float, default=0.3)
    p.add_argument("--cpu-lane-ms", type=float, default=0.6)
    p.add_argument("--miss-job-ms", type=float, default=1.0)
    p.add_argument("--gpu-ms", type=float, default=1.0)
    p.add_argument("--step-ms", type=float, default=20.0)
    p.add_argument("--ram-rows", type=int, default=8385, help="106496 MiB / 13,315,584 B per row")
    p.add_argument("--node-share", type=float, default=57344 / 106496)
    p.add_argument("--split", default="0 1 2 2 3 3 4 5 5", help="CPU hit lanes by hit count (the calibration)")
    p.add_argument("--accept", type=float, default=4.3, help="accepted tokens per verify, for ms per token")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--name")
    p.add_argument("--out")
    return p.parse_args(argv)


def run_one(a, loaded=None, ranks=None) -> dict:
    loaded = loaded or load_forwards(a.trace)
    if ranks is None and a.predictor == "gate":
        ranks = dict(np.load(a.ranks))
    return Replay(a, loaded["layer_ids"], ranks).run(loaded)


def main() -> None:
    a = parse()
    out = run_one(a)
    text = json.dumps(out, indent=1)
    if a.out:
        with open(a.out, "w") as f:
            f.write(text)
    print(text)


if __name__ == "__main__":
    main()
```

Notes for the implementer: `Tier.take(hot, protect, now, only)` from `pinned_prefetch_replay` refuses a filling slot (`ready > now`). Demand rows issued at the same `now` are served in submit order, i.e. route order, which is what `test_gate_is_the_slower_group`'s numbers assume. Floating-point sums of 2.2, 1.0 and 0.6 may need `pytest.approx` in the equality assertions; use it rather than rounding in the model.

- [ ] **Step 4: Run the tests to verify they pass; fix the model, not the tests, when a number disagrees with the docstring's rules**

Run: the Step 2 command.
Expected: PASS, 7 passed; EXIT=0.

- [ ] **Step 5: Commit**

```bash
git add analysis/dsv41-drive/prefetch-replay/verify_replay.py analysis/dsv41-drive/prefetch-replay/test_verify_replay.py
git commit -m "Replay a DSpark verify capture through the per-group gate chain with RAM prefetch"
```

---

### Task 6: The capture driver and the arm definitions

**Files:**
- Create: `analysis/dsv41-drive/prefetch-replay/capture_verify.py`
- Create: `analysis/dsv41-drive/prefetch-replay/verify.arms`
- Create: `analysis/dsv41-drive/prefetch-replay/verify_table.py`
- Test: `analysis/dsv41-drive/prefetch-replay/test_capture_verify.py`

**Interfaces:**
- Consumes: `benchmarks/dsv41_baseline/arm_env.py` (`dspark_env()`, `arm_env(overrides)`, `ServerArgs(port, dspark).argv()`, `SERVER_CORES`, `DRIVER_CORES`, `PYTHON`, `GPU_LOCK`, `DSPARK_HEALTH_TIMEOUT_S`), `scripts/expert_prediction/prefetch/logprob_probe.py` (`--port --sessions --prompts --max-tokens --top-logprobs --out`), `benchmarks/dsv41_baseline/session_subset.py` (`CORPUS_PATH`, `load_raw_sessions(path, n=, skip=)`), the stage trace (`SGLANG_DSV41_EXPERT_TRACE_PATH`, `SGLANG_DSV41_ROUTER_CAPTURE_PATH`).
- Produces: `capture_verify.py OUT_DIR --sessions-skip {0|9} [--prompts 8] [--max-tokens 128]` writing `OUT_DIR/{sessions.jsonl, trace.jsonl, router.{json,x.bin,ids.bin,w.bin,seq.bin}, server.log, probe.json, capture.json}`; `check_capture(out_dir) -> dict` (verify forwards, dropped, records) that raises on `dropped > 0` or a record-count mismatch; `verify_table.py RESULTS.jsonl` printing one markdown row per arm.

- [ ] **Step 1: Write the failing test**

```python
"""capture_verify.py: the sessions file, the server env and the capture check. CPU only."""

import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import capture_verify as cv  # noqa: E402


def test_the_sessions_file_skips_the_suite_and_the_warmup_row(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text("".join(json.dumps({"session_id": f"s{i}"}) + "\n" for i in range(20)))
    held_out = cv.write_sessions(str(corpus), str(tmp_path / "sessions.jsonl"), skip=9, n=8)
    assert [json.loads(l)["session_id"] for l in open(held_out)] == [f"s{i}" for i in range(9, 17)]
    suite = cv.write_sessions(str(corpus), str(tmp_path / "suite.jsonl"), skip=0, n=8)
    assert [json.loads(l)["session_id"] for l in open(suite)] == [f"s{i}" for i in range(8)]


def test_the_server_env_names_the_trace_and_router_capture_under_the_out_dir(tmp_path):
    env = cv.server_env(str(tmp_path))
    assert env["SGLANG_DSV41_EXPERT_TRACE_PATH"] == str(tmp_path / "trace.jsonl")
    assert env["SGLANG_DSV41_ROUTER_CAPTURE_PATH"] == str(tmp_path / "router")
    assert env["SGLANG_DSV41_CPU_EXPERTS"] == "1" and env["SGLANG_DSV41_ENABLE_DSPARK_CPU_EXPERTS"] == "1"


def test_the_driver_refuses_a_capture_with_dropped_forwards(tmp_path):
    trace = tmp_path / "trace.jsonl"
    lines = [
        {"kind": "graph_routes_header", "schema": 3, "run": "r", "layer_ids": [0], "hot_capacity": [1]},
        {"kind": "graph_routes", "schema": 3, "seq": 0, "forward_pass_id": 1, "phase": "target_verify", "rids": ["a"],
         "forward_tokens": 6, "routes": [[1]], "misses": [0], "router": 0},
        {"kind": "graph_routes", "schema": 3, "seq": 5, "forward_pass_id": 2, "phase": "target_verify", "rids": ["a"],
         "forward_tokens": 6, "routes": [[1]], "misses": [0], "router": 1, "dropped_before": 4},
    ]
    trace.write_text("".join(json.dumps(l) + "\n" for l in lines))
    (tmp_path / "router.json").write_text(json.dumps({"schema": 2, "layer_ids": [0], "tokens": 6, "hidden": 4, "topk": 2}))
    with pytest.raises(RuntimeError, match="lost 4 graph forwards"):
        cv.check_capture(str(tmp_path))
```

- [ ] **Step 2: Run it to verify it fails**

Run: `ssh divix01 'cd /data/models/slang/nvfp4-work/wt-ram-prefetch && PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES= TMPDIR=/data/models/slang/nvfp4-work/ram-prefetch/tmp taskset -c 0-63 /data/models/slang/.venv/bin/python -m pytest -q -p no:randomly analysis/dsv41-drive/prefetch-replay/test_capture_verify.py 2>&1 | tail -5; echo EXIT=${PIPESTATUS[0]}'`
Expected: FAIL — `ModuleNotFoundError: No module named 'capture_verify'`; EXIT=1.

- [ ] **Step 3: Implement the driver**

```python
"""Capture a DSpark verify route log with router inputs, for verify_replay.py.

Starts the dspark-both server (the recipe's DSpark env and argv, both CPU-expert clients, the 104 GiB tier) with the
stage trace and router capture pointed at OUT_DIR, serves ``--prompts`` corpus sessions from ``--sessions-skip``
through logprob_probe.py, and shuts the server down normally so the route log's final read lands. Then checks the
capture: no forward lost, every verify forward with a router record. A diagnostic run: its timings are not a
throughput number.

Locks: rowimg-disk.lock, then cc-gpu.lock (the protocol's order); the server runs under taskset on SERVER_CORES.

Usage: capture_verify.py OUT_DIR [--sessions-skip 0|9] [--prompts 8] [--max-tokens 128]
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "benchmarks", "dsv41_baseline"))
sys.path.insert(0, os.path.join(REPO, "scripts", "dsv41"))
import arm_env  # noqa: E402
import session_subset  # noqa: E402

PORT = 30031
DISK_LOCK = "/data/models/slang/nvfp4-work/rowimg-disk.lock"
PROBE = os.path.join(REPO, "scripts", "expert_prediction", "prefetch", "logprob_probe.py")


def write_sessions(corpus: str, out: str, *, skip: int, n: int) -> str:
    """The first ``n`` corpus rows from ``skip``: 0 is the suite's eight, 9 skips the suite and its warm-up row."""
    rows = session_subset.load_raw_sessions(corpus, n=n, skip=skip)
    with open(out, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return out


def server_env(out_dir: str) -> dict:
    return arm_env.arm_env({
        **arm_env.dspark_env(),
        "SGLANG_DSV41_EXPERT_TRACE_PATH": os.path.join(out_dir, "trace.jsonl"),
        "SGLANG_DSV41_ROUTER_CAPTURE_PATH": os.path.join(out_dir, "router"),
    })


def check_capture(out_dir: str) -> dict:
    from tier_sim import load_forwards

    loaded = load_forwards(os.path.join(out_dir, "trace.jsonl"), allow_dropped=True)
    if loaded["dropped"]:
        raise RuntimeError(f"the route reader lost {loaded['dropped']} graph forwards; the capture cannot be replayed")
    verifies = [f for f in loaded["forwards"] if f.get("verify")]
    without = [f["seq"] for f in verifies if f.get("router") is None]
    if without:
        raise RuntimeError(f"{len(without)} verify forwards have no router record (first seq {without[0]})")
    with open(os.path.join(out_dir, "router.json")) as f:
        header = json.load(f)
    return {"verify_forwards": len(verifies), "forwards": len(loaded["forwards"]), "tokens": header["tokens"],
            "layers": len(header["layer_ids"])}


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


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("out_dir")
    p.add_argument("--sessions-skip", type=int, default=0)
    p.add_argument("--prompts", type=int, default=8)
    p.add_argument("--max-tokens", type=int, default=128)
    a = p.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    sessions = write_sessions(session_subset.CORPUS_PATH, os.path.join(a.out_dir, "sessions.jsonl"), skip=a.sessions_skip, n=a.prompts)
    env = os.environ | server_env(a.out_dir) | {"PYTHONPATH": os.path.join(REPO, "python")}
    argv = arm_env.ServerArgs(port=PORT, dspark=True).argv()
    with open(DISK_LOCK, "w") as disk, open(arm_env.GPU_LOCK, "w") as gpu:
        fcntl.flock(disk, fcntl.LOCK_EX)
        fcntl.flock(gpu, fcntl.LOCK_EX)
        with open(os.path.join(a.out_dir, "server.log"), "w") as log:
            server = subprocess.Popen(["taskset", "-c", arm_env.SERVER_CORES, *argv], env=env, stdout=log,
                                      stderr=subprocess.STDOUT, cwd=REPO, start_new_session=True)
        rc = 1
        try:
            if not _healthy(PORT, time.monotonic() + arm_env.DSPARK_HEALTH_TIMEOUT_S):
                return 1
            rc = subprocess.run(
                ["taskset", "-c", arm_env.DRIVER_CORES, arm_env.PYTHON, PROBE, "--port", str(PORT),
                 "--sessions", sessions, "--prompts", str(a.prompts), "--max-tokens", str(a.max_tokens),
                 "--top-logprobs", "1", "--out", os.path.join(a.out_dir, "probe.json")], cwd=REPO,
            ).returncode
        finally:
            # A normal shutdown runs the route log's final read; a SIGKILL would lose the last ring entries.
            os.killpg(server.pid, signal.SIGTERM)
            try:
                server.wait(timeout=300)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGKILL)
                server.wait()
    if rc != 0:
        return rc
    summary = check_capture(a.out_dir)
    summary.update({"sessions_skip": a.sessions_skip, "prompts": a.prompts, "max_tokens": a.max_tokens,
                    "commit": subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()})
    with open(os.path.join(a.out_dir, "capture.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

Check before committing that `arm_env.ServerArgs`, `arm_env.arm_env`, `arm_env.dspark_env`, `SERVER_CORES`, `DRIVER_CORES`, `PYTHON`, `GPU_LOCK`, `DSPARK_HEALTH_TIMEOUT_S` exist with these names (`grep -n "^def \|^class \|^[A-Z_]* = " benchmarks/dsv41_baseline/arm_env.py`); `both_cpu_ab.py::run_probe` uses exactly these, so copy its spelling if any differs. If `logprob_probe.py` needs `--top-logprobs` ≥ 2, pass 2.

`verify.arms` (one arm per line: name then flags; `run_arms.py` already takes a `.arms` file but calls `pinned_prefetch_replay`; write `verify_arms.sh` instead, a loop that runs `verify_replay.py` once per line into `$OUT/<name>.json` and appends to `$OUT/results.jsonl`):

```
none
none_r13 --nvme-row-ms 1.3
oracle_h1 --predictor oracle --h 1 --k 1
oracle_h1_k2 --predictor oracle --h 1 --k 2 --budget 8
oracle_h2 --predictor oracle --h 2 --k 1
oracle_h1_r13 --predictor oracle --h 1 --k 1 --nvme-row-ms 1.3
gate_h1_k1 --predictor gate --h 1 --k 1
gate_h1_k1_b2 --predictor gate --h 1 --k 1 --budget 2
gate_h1_k2 --predictor gate --h 1 --k 2 --budget 8
gate_h1_k1_d3 --predictor gate --h 1 --k 1 --depth 3
gate_h1_k1_mru --predictor gate --h 1 --k 1 --admit mru
gate_h1_k1_s0 --predictor gate --h 1 --k 1 --spec-share 0
gate_h1_k1_fifo --predictor gate --h 1 --k 1 --queue fifo
gate_h2_k1 --predictor gate --h 2 --k 1
gate_h1_k1_r13 --predictor gate --h 1 --k 1 --nvme-row-ms 1.3
noisy_h1_p25 --predictor noisy --h 1 --k 1 --p 0.25
noisy_h1_p40 --predictor noisy --h 1 --k 1 --p 0.4
noisy_h1_p60 --predictor noisy --h 1 --k 1 --p 0.6
```

`verify_arms.sh`:

```bash
#!/usr/bin/env bash
# verify_arms.sh TRACE RANKS ARMS OUT_DIR [extra verify_replay flags, e.g. the calibration]
set -euo pipefail
trace=$1; ranks=$2; arms=$3; out=$4; shift 4
here=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$out"; : > "$out/results.jsonl"
while read -r name flags; do
  [ -z "$name" ] && continue
  # shellcheck disable=SC2086
  "${PYTHON:-python3}" "$here/verify_replay.py" "$trace" --ranks "$ranks" --name "$name" --out "$out/$name.json" $flags "$@" > /dev/null
  python3 -c "import json,sys; d=json.load(open('$out/$name.json')); d['name']='$name'; print(json.dumps(d))" >> "$out/results.jsonl"
  echo "$name done"
done < "$arms"
```

`verify_table.py`:

```python
"""One markdown row per arm of verify_arms.sh's results.jsonl, saved ms against the ``none`` arm."""

import json
import sys

KEYS = ["precision_target", "precision_any_use", "spec_rows_per_step", "ram_misses_per_step", "late_per_step",
        "harmful_evictions_per_step", "demand_rows_delayed_per_step", "demand_delay_mean_ms", "gate_ms_per_step",
        "step_ms_per_step", "ms_per_token", "nvme_busy_frac"]


def main() -> None:
    rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
    base = {r["args"]["nvme_row_ms"]: r for r in rows if r["name"].startswith("none")}
    print("| arm | " + " | ".join(KEYS) + " | saved ms/token |")
    print("|---|" + "---:|" * (len(KEYS) + 1))
    for r in rows:
        b = base.get(r["args"]["nvme_row_ms"])
        saved = b["ms_per_token"] - r["ms_per_token"] if b else float("nan")
        cells = [f"{r[k]:.3f}" if isinstance(r[k], float) else str(r[k]) for k in KEYS]
        print(f"| {r['name']} | " + " | ".join(cells) + f" | {saved:+.2f} |")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the test to verify it passes**

Run: the Step 2 command.
Expected: PASS, 3 passed; EXIT=0.

- [ ] **Step 5: Commit**

```bash
chmod +x analysis/dsv41-drive/prefetch-replay/verify_arms.sh
git add analysis/dsv41-drive/prefetch-replay/capture_verify.py analysis/dsv41-drive/prefetch-replay/verify.arms analysis/dsv41-drive/prefetch-replay/verify_arms.sh analysis/dsv41-drive/prefetch-replay/verify_table.py analysis/dsv41-drive/prefetch-replay/test_capture_verify.py
git commit -m "Capture a DSpark verify route log and run the verify replay arms"
```

---

### Task 7: Capture on divix01 (suite and held-out), validate, calibrate

**Files:**
- divix01 outputs: `/data/models/slang/nvfp4-work/ram-prefetch/capture-suite/`, `.../capture-heldout/`, `.../calib/`
- No repo files change in this task except `findings.md` notes carried into Task 8.

**Interfaces:**
- Consumes: Tasks 1-6 on branch `codex/dsv41-ram-prefetch`, pushed; the divix01 worktree at that commit.
- Produces: `trace.jsonl` + `router.*` per capture, `rank.npz` per capture, the chosen calibration flags (recorded in `calib/chosen.txt`).

- [ ] **Step 1: Confirm production is down and the GPU free; otherwise stop here and ask the owner**

Run: `ssh divix01 'nvidia-smi --query-gpu=memory.used --format=csv,noheader; pgrep -af "sglang.launch_server|dsv41-direct-prod" | grep -v pgrep | head -3; ls /data/models/slang/nvfp4-work/cc-gpu.lock'`
Expected: `0 MiB` (or a few hundred) and no server process. If production is running, do not proceed: report to the owner and wait for a window. Never stop it yourself.

- [ ] **Step 2: Bring the divix01 worktree to the branch head and run the CPU tests once**

Run:
```bash
ssh divix01 'git -C /data/models/slang/sglang fetch origin && git -C /data/models/slang/nvfp4-work/wt-ram-prefetch checkout --detach origin/codex/dsv41-ram-prefetch && cd /data/models/slang/nvfp4-work/wt-ram-prefetch && git log -1 --oneline && mkdir -p /data/models/slang/nvfp4-work/ram-prefetch/tmp && PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES= TMPDIR=/data/models/slang/nvfp4-work/ram-prefetch/tmp taskset -c 0-63 /data/models/slang/.venv/bin/python -m pytest -q -p no:randomly test/registered/unit/layers/moe/test_exl3_stream_trace.py test/manual/dsv41/test_router_score.py analysis/dsv41-drive/prefetch-replay/test_verify_gate_rankings.py analysis/dsv41-drive/prefetch-replay/test_verify_replay.py analysis/dsv41-drive/prefetch-replay/test_capture_verify.py 2>&1 | tail -3; echo EXIT=${PIPESTATUS[0]}'
```
Expected: the branch head's SHA; all passed; EXIT=0.

- [ ] **Step 3: Capture the suite (sessions 0-7)**

Run (background; ~25 min: DSpark loads the target and draft in ~15 min, then 8 sessions):
```bash
ssh divix01 'cd /data/models/slang/nvfp4-work/wt-ram-prefetch && mkdir -p /data/models/slang/nvfp4-work/ram-prefetch/capture-suite && PYTHONPATH=$PWD/python TMPDIR=/data/models/slang/nvfp4-work/ram-prefetch/tmp nohup taskset -c 0-63 /data/models/slang/.venv/bin/python analysis/dsv41-drive/prefetch-replay/capture_verify.py /data/models/slang/nvfp4-work/ram-prefetch/capture-suite --sessions-skip 0 --prompts 8 --max-tokens 128 > /data/models/slang/nvfp4-work/ram-prefetch/capture-suite/driver.log 2>&1 &'
```
Then watch `driver.log` and `server.log` (`tail -f`, line-buffered) until `capture.json` exists.
Expected: `capture.json` with `verify_forwards` in the low hundreds (8 sessions × ~100 tokens / ~4.3 accepted ≈ 190-250), `tokens` 6, `layers` 40, no exception. If the server refuses the trace or router capture at startup (read `server.log` for `exl3 RAM miss:` lines), fix the cause in a new commit (likely the route ring width or `record_router` being reached by the draft path) and rerun; do not patch the worktree by hand.

- [ ] **Step 4: Capture the held-out sessions (corpus rows 9-16)**

Run: the Step 3 command with `capture-heldout` and `--sessions-skip 9`.
Expected: as Step 3.

- [ ] **Step 5: Rank both captures**

Run:
```bash
ssh divix01 'cd /data/models/slang/nvfp4-work/wt-ram-prefetch && for c in suite heldout; do d=/data/models/slang/nvfp4-work/ram-prefetch/capture-$c; PYTHONPATH=$PWD/python OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES= taskset -c 0-31 /data/models/slang/.venv/bin/python analysis/dsv41-drive/prefetch-replay/verify_gate_rankings.py $d/trace.jsonl $d/router /data/models/slang/nvfp4-work/cc-expert-prediction/dsv41-full40 --out $d/rank.npz --horizons 2 --depth 12 2>&1 | tail -2; done'
```
Expected: `h=0 self-check: route sets reproduced` ≥ 0.999 for both (the gate on the captured input reproduces the captured ids; §18.4 got 20,480/20,480). Below 0.99 means the capture's `x` is not what the gate saw (wrong tensor at the call site) — stop and fix Task 1's call site.

- [ ] **Step 6: Validate the tier model against the capture's own counters**

The server's metrics line (`server.log`, the `exl3 RAM miss` stats at shutdown, or `SGLANG_MOE_HOT_METRICS_FILE` if set) reports served RAM misses and `rows_read` for the run. Run the replay with no predictor and compare:
```bash
ssh divix01 'cd /data/models/slang/nvfp4-work/wt-ram-prefetch && d=/data/models/slang/nvfp4-work/ram-prefetch/capture-suite; PYTHONPATH=$PWD/python CUDA_VISIBLE_DEVICES= taskset -c 0-7 /data/models/slang/.venv/bin/python analysis/dsv41-drive/prefetch-replay/verify_replay.py $d/trace.jsonl --out $d/none.json | python3 -c "import json,sys; d=json.load(sys.stdin); print({k: d[k] for k in (\"steps\",\"ram_misses_per_step\",\"gate_ms_per_step\",\"step_ms_per_step\",\"gate_p50_ms\",\"gate_p90_ms\",\"gate_p99_ms\",\"nvme_busy_frac\")})"; grep -h "rows_read\|ram_miss\|served" $d/server.log | tail -5'
```
Expected: replay RAM misses per step within 15% of the measured served misses per verify (measured total / `verify_forwards`). If further off, the per-node share or the hot sets are wrong: check `hot` is non-null in the trace lines (the GPU residency updater must be on, as the recipe has it) and `node_share` against `SGLANG_MOE_PINNED_HOST_NUMA_MB` in `server.log`. Record the numbers in `calib/validation.txt`.

- [ ] **Step 7: Calibrate the chain constants against the capture's own forward times and the 2026-10-07 gate distribution**

The route log's lines carry `t` (monotonic seconds per forward); the mean inter-verify time within a session is the measured step. Sweep with no predictor:
```bash
ssh divix01 'cd /data/models/slang/nvfp4-work/wt-ram-prefetch && d=/data/models/slang/nvfp4-work/ram-prefetch/capture-suite; mkdir -p /data/models/slang/nvfp4-work/ram-prefetch/calib; python3 - <<EOF
import json
ts=[(json.loads(l)) for l in open("$d/trace.jsonl") if "graph_routes\"" in l]
v=[x for x in ts if x.get("phase")=="target_verify"]
gaps=[b["t"]-a["t"] for a,b in zip(v,v[1:]) if a["rids"]==b["rids"] and b["t"]-a["t"]<2]
gaps.sort(); n=len(gaps); print("measured step ms p50/p90", round(1000*gaps[n//2],1), round(1000*gaps[int(.9*n)],1), "n", n)
EOF
for row in 2.0 2.2 2.5; do for job in 0.8 1.0 1.2; do for gpu in 0.8 1.0 1.5; do PYTHONPATH=$PWD/python CUDA_VISIBLE_DEVICES= taskset -c 0-7 /data/models/slang/.venv/bin/python analysis/dsv41-drive/prefetch-replay/verify_replay.py $d/trace.jsonl --nvme-row-ms $row --miss-job-ms $job --gpu-ms $gpu --out /data/models/slang/nvfp4-work/ram-prefetch/calib/r${row}_j${job}_g${gpu}.json | python3 -c "import json,sys; d=json.load(sys.stdin); print(\"$row $job $gpu\", round(d[\"step_ms_per_step\"],1), round(d[\"gate_p50_ms\"],2), round(d[\"gate_p90_ms\"],2), round(d[\"gate_p99_ms\"],2), round(d[\"nvme_busy_frac\"],2))"; done; done; done | tee /data/models/slang/nvfp4-work/ram-prefetch/calib/sweep.txt'
```
Expected: one triple whose step ms is within ~5% of the measured p50 step and whose gate p50/p90/p99 are within ~20% of 4.9 / 9.5 / 19.5 ms. Prefer `--nvme-row-ms 2.2` (the measured landing) and move `--miss-job-ms` and `--gpu-ms` first; `--step-ms` absorbs the remaining per-step offset (draft graph + host, ~20 ms). Write the chosen flags to `calib/chosen.txt` as one line, e.g. `--nvme-row-ms 2.2 --miss-job-ms 1.0 --gpu-ms 1.0 --step-ms 20`.

- [ ] **Step 8: Record the capture facts in the findings draft**

Append to `/data/models/slang/nvfp4-work/ram-prefetch/findings.md` (create it): the commit, both captures' `capture.json`, the self-check values, the validation numbers, the sweep table and the chosen calibration. No commit (divix01 artifacts only).

---

### Task 8: Run the arms, apply the go rule, write the findings

**Files:**
- divix01 outputs: `/data/models/slang/nvfp4-work/ram-prefetch/arms-{suite,heldout}/`, `findings.md`
- Modify: `DSV41_REFERENCE.md` (new `### 33.13`), `NVME_PINNED_PREFETCH_HANDOFF.md` (a status line at the top)

**Interfaces:**
- Consumes: Task 7's captures, `rank.npz`, `calib/chosen.txt`; Task 6's `verify_arms.sh`, `verify.arms`, `verify_table.py`.
- Produces: the decision, recorded in three places.

- [ ] **Step 1: Run every arm on both captures**

Run:
```bash
ssh divix01 'cd /data/models/slang/nvfp4-work/wt-ram-prefetch && calib=$(cat /data/models/slang/nvfp4-work/ram-prefetch/calib/chosen.txt); for c in suite heldout; do d=/data/models/slang/nvfp4-work/ram-prefetch/capture-$c; PYTHON=/data/models/slang/.venv/bin/python PYTHONPATH=$PWD/python CUDA_VISIBLE_DEVICES= taskset -c 0-15 analysis/dsv41-drive/prefetch-replay/verify_arms.sh $d/trace.jsonl $d/rank.npz analysis/dsv41-drive/prefetch-replay/verify.arms /data/models/slang/nvfp4-work/ram-prefetch/arms-$c $calib 2>&1 | tail -2; /data/models/slang/.venv/bin/python analysis/dsv41-drive/prefetch-replay/verify_table.py /data/models/slang/nvfp4-work/ram-prefetch/arms-$c/results.jsonl | tee /data/models/slang/nvfp4-work/ram-prefetch/arms-$c/table.md; done'
```
Expected: 18 rows per table; `none_r13` and `*_r13` saved against `none_r13`, the rest against `none`; `gate_h1_k1_fifo` worse than `gate_h1_k1` (demand delay); `noisy_h1_p60` between `gate` and `oracle`.

- [ ] **Step 2: Apply the go rule**

For each capture: `ratio = saved(gate_h1 best of {k1, k1_b2, k2, k1_d3, k1_mru, k1_s0}) / saved(oracle_h1)`. Go if `ratio ≥ 0.5` on the suite and the held-out capture does not contradict it (its ratio ≥ 0.4). Also report, for the record, `oracle_h1` saved ms/token in absolute terms, `gate_h1_k1_r13`'s saving (the fourth-mirror world), and the drive busy fraction of the chosen gate arm.

- [ ] **Step 3: Write `findings.md`**

Sections: Capture (both), Validation, Calibration, Results (both tables), The go rule (ratio, numbers, decision), What the model leaves out (the two groups share one queue; no NVMe latency tail; the hit job's lane count follows the calibration split, not the live one; node 0's draft slabs are outside the tier model; a wrong prefetch's eviction cost is bounded by spec-share, as the live design will do), Next step (Phase 1 plan, or close with the reason). Keep every number next to its file path.

- [ ] **Step 4: Record in the repo**

In `DSV41_REFERENCE.md` after §33.12 add `### 33.13 NVMe-to-RAM prefetch under DSpark: the replay go/no-go (the run date)` with: the three overturned premises, the capture and calibration, the two tables (or their key rows), the decision, and the pointers (`analysis/dsv41-drive/prefetch-replay/verify_*.py`, divix01 paths, the spec). At the top of `NVME_PINNED_PREFETCH_HANDOFF.md` add one line under the status: `Revisited <run date> under DSpark with CPU-served misses: DSV41_REFERENCE.md §33.13.` Commit:

```bash
git add DSV41_REFERENCE.md NVME_PINNED_PREFETCH_HANDOFF.md
git commit -m "Record the DSpark RAM-prefetch replay go/no-go"
```

- [ ] **Step 5: Hindsight**

`hindsight_ingest_document` with the findings (title `DSpark RAM prefetch replay go/no-go (<run date>)`), and `hindsight_capture_initiative` with `relates_to_page_id` on the initiative page if one exists (the capture failed with ECONNREFUSED on 2026-10-08; retry, and if it still fails, say so in the final report).

- [ ] **Step 6: Push and report**

`git push origin codex/dsv41-ram-prefetch`; report the decision, the ratios, the absolute savings, and the branch state. The owner decides whether Phase 1 is planned.

---

## Self-review notes

- Spec coverage: capture (Tasks 1, 3, 6, 7), simulator chain/tier/calibration/predictors/metrics (Task 5, 7), go rule (Task 8), findings to divix01/Hindsight/§33 (Task 8). The spec's `--hit-ms` is realised as `--cpu-fixed-ms` + `--cpu-lane-ms` with the calibration split, so the hit job's cost follows its lane count as the measured 2.5 ms (4.4 lanes) does.
- Interfaces: `record_router(row, x, topk_ids, topk_weights)` (Task 1) is what `exl3.py` calls; `RouterCapture.tokens/ids` (Task 2) is what `rank_verify` (Task 4) reads; `verify`/`router` keys (Task 3) are what `rank_verify`, `verify_replay` and `check_capture` read; the npz keys `order/score/valid/tokens/seq` (Task 4) are what `verify_replay.candidates` reads.
- Review Focus tests: 1 → Task 4 `test_live_tokens_only_are_ranked`; 2 → Task 2 `test_schema_1_capture_loads_as_one_token`; 3 → Task 5 `test_gate_is_the_slower_group`; 4 → Task 5 `test_a_demand_on_a_filling_row_waits_and_is_late`; 5 → Task 6 `test_the_driver_refuses_a_capture_with_dropped_forwards`.
