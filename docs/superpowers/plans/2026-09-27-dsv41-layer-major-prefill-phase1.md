# Layer-Major Prefill, Phases 0-1: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task by task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prefill a long prompt one layer at a time: every chunk runs through layer L before layer L+1. Output must
match today's chunked prefill at 8k-32k tokens. The pass is built from a model- and quant-agnostic strategy layer
plus a DeepSeek-V4.1 model adapter.

**Architecture:**
- **Layer 1, the strategy.** A new package, `sglang/srt/layer_major/`. It owns the admission gate, a pinned host
  StateStore on NUMA node 1, the layer-outer/chunk-inner driver, a watchdog heartbeat, and the worker entry point.
  It sees the model only through the `LayerMajorModelAdapter` protocol and experts only through the `ExpertResidency`
  protocol (`NullResidency` in this phase).
- **Layer 2, the model adapter.** `models/deepseek_v4_layer_major.py` builds per-chunk ForwardBatches and metadata,
  maps the window-KV ring, runs one layer on one chunk, and runs the late-layer tail with tail-only logits.
- **The generic hybrid-SWA memory code** gains a ring admission budget and ring allocation, release and finalize.
- **Out of scope:** the EXL3 quant adapter and expert residency (phase 2, its own plan). Experts come through
  today's prefill path.

**Tech stack:** Python 3.13 and PyTorch on divix01 (RTX 5090, sm_120), sglang fork, `unittest`/pytest, msgspec.

**Spec:** `docs/superpowers/specs/2026-09-26-dsv41-layer-major-prefill-design.md` (revision 2 plus §4.0 layering).
This plan's corrections to it are in Task 1.

## Global Constraints

Every task's requirements include these.

**Code style:**
- New data containers are `msgspec.Struct`, not `@dataclass` (`.claude/rules/no-dataclasses.md`).
- No defensive `getattr`/`hasattr` on fields that always exist (`.claude/rules/no-getattr-defensive.md`).
- Comments follow `.claude/rules/comment-style.md`: one or two lines, facts a reader cannot see, ASCII only.
- `ForwardBatch.init_new` must not mutate the ScheduleBatch. Per-forward overrides go through kw-only params
  (`.claude/rules/forward-batch-init-new-purity.md`).
- ScheduleBatch fields are rebound, never mutated in place (`.claude/rules/schedule-batch-out-of-place-mutation.md`).
- `python/sglang/srt/model_executor/model_runner.py` is frozen: do not edit it.
- Env vars follow `.claude/skills/env-var-conventions/SKILL.md`. Read it before Task 1.

**Layering (spec §4.0):**
- `srt/layer_major/` must not import anything from `models/deepseek_v4*`, `layers/quantization/exl3*` or the dsv4
  backend.
- The DSV4 adapter must not import EXL3 modules.

**Running on divix01** (`.claude/rules/divix01-run-protocol.md`):
- Code is committed, pushed to `origin`, and run in a private worktree:
  `/data/models/slang/nvfp4-work/wt-layer-major` at the pushed commit.
- Set `PYTHONPATH=$PWD/python`, and print `sglang.__file__` before trusting a result.
- CPU unit runs: `OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES= taskset -c 18-35,54-63`, with
  `--basetemp=/mnt/nvme1/pytest-tmp/<name>`.
- GPU runs: `flock /data/models/slang/nvfp4-work/cc-gpu.lock taskset -c 32-63 ...`. A server arm takes
  `rowimg-disk.lock` first, then `cc-gpu.lock`.
- Scratch goes under `/mnt/nvme1/layer-major/`, never `/` or `/tmp`.
- Read pytest's own status: `...; echo "EXIT=${PIPESTATUS[0]}"`.
- Registered-suite baseline: `test/registered/unit/kernels`. The wider `test/registered` tree fails collection on
  divix01 (pyarrow). New unit tests use `unittest.TestCase`, because `test_utils` imports fail on divix01.

**Git:**
- Stage files by name, never `.omc/`.
- Commit trailer:

  ```
  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01N985sZUTxTE9P7MP2rtJaF
  ```
- Production (`dsv41-direct-prod`) is never used for tests. Nothing here restarts production.

**Constants** (from the spec and the traced run, `DSV41_REFERENCE.md` §27.17):

| Name | Value |
|---|---|
| Chunk size | `CHUNKED_PREFILL_SIZE` = 4096 |
| Page size | 256 |
| Window | 128 |
| Ring size | chunk + page = 4352 window slots (17 pages) |
| Window pool at chunk 4096 | 10,752 slots |
| Layers run layer-major | `start_layer` .. `late_layer_start - 1` = 0..20 |
| Late layers | 21..39, on the final chunk's tail only |
| Carried state per token | `hidden` [4, 5120] bf16 + `prev_pre` [4] fp32 (Task 1 explains why `normalized` is dropped) |

## Review Focus

These are the input classes most likely to hurt a user that no single task's main tests cover. Each has its pinning
test in the named task.

1. **A cached prefix (radix hit) before a long suffix.** Only the suffix runs layer-major. Chunk 0 reads its
   predecessor window from the prefix's own window slots, not the ring. Pinned in Task 8
   (`test_first_chunk_with_prefix_is_not_remapped`) and Task 12 (the `prefix` equivalence case).
2. **A request finished or aborted right after the pass.** The request's window slots must return to the pool
   exactly: no leak, no double free. Pinned in Task 6 (`test_release_after_finalize_returns_every_slot`).
3. **An exception in the middle of the pass.** Residency `end()` still runs, the ring is released, the request fails
   and the server keeps serving. Pinned in Task 3 (`test_exception_mid_pass_still_ends_residency`) and Task 11
   (`test_seam_propagates_pass_failure_after_release`).
4. **A pass longer than the 300 s watchdog.** It must not be killed. Pinned in Task 5
   (`test_watchdog_counter_advances_with_heartbeat`).
5. **A short request right after a long one.** It must take the normal chunked path with the window pool fully
   available. Pinned in Task 6 (`test_ring_admission_restores_available_size`) and Task 12 (the `after` case).

---

### Task 0: Phase 0: measure the chunked path at 128k (no product code)

**Why:** spec §11 phase 0. Four things are unknown past 32k: per-chunk time, peak VRAM, which indexer-logits path
sm_120 takes, and node 1's free memory under load. If the dense indexer's unchunked logits run out of memory at
128k, phase 0b (row-chunking the dense indexer) must be planned before Task 2.

**Files:**
- Create: `analysis/dsv41-drive/layer-major/phase0_probe.py`
- Modify: `analysis/dsv41-drive/prefill-chunk/chunk_smoke.sh` (add a per-node memory sampler)

**Interfaces:** none. This is a measurement.

- [ ] **Step 1: Write the indexer-path probe**

```python
"""Report which DSV4 prefill indexer-logits path this machine takes (phase 0 of the layer-major plan).

Usage: PYTHONPATH=<wt>/python python phase0_probe.py
"""

import importlib

from sglang.srt.environ import envs

print("SGLANG_OPT_USE_TOPK_V2 =", envs.SGLANG_OPT_USE_TOPK_V2.get())
try:
    deep_gemm = importlib.import_module("deep_gemm")
    print("deep_gemm.fp8_fp4_mqa_logits:", hasattr(deep_gemm, "fp8_fp4_mqa_logits"))
except ImportError as exc:
    print("deep_gemm not importable:", exc)
```

- [ ] **Step 2: Add a per-node memory sampler to `chunk_smoke.sh`**

Insert directly after the line that starts the `nvidia-smi` sampler (`SMI=$!`):

```bash
# Node free memory every 5 s: phase 0 of the layer-major plan sizes the host state store on node 1.
( while true; do echo "$(date +%s) $(numactl --hardware | awk '/free:/ {printf "%s=%s ", $2, $4}')"; sleep 5; done ) \
  > $OUT/numa.log 2>&1 &
NUMA=$!
```

In `stop_server()`, next to `kill $SMI 2>/dev/null`, add:

```bash
  kill $NUMA 2>/dev/null
```

- [ ] **Step 3: Commit and push**

```bash
git add analysis/dsv41-drive/layer-major/phase0_probe.py analysis/dsv41-drive/prefill-chunk/chunk_smoke.sh
git commit -m "analysis(dsv41): layer-major phase 0 probe and per-node memory sampler" -m "<trailer>"
git push origin cc/layer-major-prefill
```

- [ ] **Step 4: Run the probe and one 131,072-token prompt on today's recipe (divix01)**

```bash
git -C /data/models/slang/sglang fetch -q origin
git -C /data/models/slang/sglang worktree add --detach /data/models/slang/nvfp4-work/wt-layer-major origin/cc/layer-major-prefill
cd /data/models/slang/nvfp4-work/wt-layer-major
PYTHONPATH=$PWD/python /data/models/slang/.venv/bin/python analysis/dsv41-drive/layer-major/phase0_probe.py
cd /mnt/nvme1/prefill-chunk
LONG_MAX_NEW=8 setsid nohup bash /data/models/slang/nvfp4-work/wt-layer-major/analysis/dsv41-drive/prefill-chunk/chunk_smoke.sh \
  phase0-128k /data/models/slang/nvfp4-work/wt-layer-major 4096 16100 131072 > phase0-128k.nohup 2>&1 < /dev/null &
```

Expected: `driver.log` ends with `long rc=0` and `DONE`, after about 35 minutes if chunk time stays at ~17 s.

- [ ] **Step 5: Read the results**

```bash
cd /mnt/nvme1/prefill-chunk
/data/models/slang/.venv/bin/python /data/models/slang/nvfp4-work/wt-layer-major/analysis/dsv41-drive/prefill-chunk/chunk_times.py phase0-128k | head -4
grep -c "memory allocation failed with OOM" phase0-128k/server.log
sort -k2 phase0-128k/numa.log | head -3
```

Record in the ledger: TTFT; median, first and last chunk seconds; peak VRAM; OOM retries; minimum `node1=` free.

- [ ] **Step 6: Decision gate**

| Result | Action |
|---|---|
| `long rc=0` and last chunk within 1.5x the first | Continue to Task 1 |
| Out of memory in `_dense_fp4_mqa_logits` / `_publish_or_consume_candidates` | Stop. Write the phase 0b plan (row-chunk the dense indexer logits within `_TORCH_INDEXER_SCORE_BUDGET_BYTES`, as the torch path does) and ask the user |
| Minimum node 1 free under ~24 GB | Ledger a ruling: StateStore capacity = what fits with `host_numa.NODE_HEADROOM_BYTES` to spare; `LayerMajorGate` caps admitted lengths to it |

---

### Task 1: Spec corrections and environment variables

**Why:** two facts found while planning change the spec, and the env names must be model-agnostic (spec §4.0).

- **Correction 1, carried state.** Under decoder bounded replay, the chunked path always has a tail
  (`deepseek_v4.py:4474-4480`), and `next_combined` requires `tail is None` (`:4549`). So today's chunked prefill
  never carries the fused `normalized` input. The layer-major pass therefore carries only `hidden` and `prev_pre`,
  calls each layer with `next_combined=None`, and should be bitwise identical to the chunked path.
- **Correction 2, env names.** `SGLANG_DSV41_LAYER_MAJOR_PREFILL_MIN_TOKENS` becomes
  `SGLANG_LAYER_MAJOR_PREFILL_MIN_TOKENS`, and `SGLANG_DSV41_LAYER_MAJOR_VERIFY` becomes
  `SGLANG_LAYER_MAJOR_PREFILL_VERIFY`. The latter is added in phase 2, where it is used.

**Files:**
- Modify: `docs/superpowers/specs/2026-09-26-dsv41-layer-major-prefill-design.md` (§2, §6.2, §8, §10.2, §7.3)
- Modify: `python/sglang/srt/environ.py` (a new section after the DSV4 block ends at `# Kernels and indexer`)
- Test: `test/registered/unit/layer_major/test_layer_major_env.py`

**Interfaces:**
- Produces:
  - `envs.SGLANG_LAYER_MAJOR_PREFILL_MIN_TOKENS` (EnvInt, default 0 = off);
  - `envs.SGLANG_LAYER_MAJOR_STATE_NUMA_NODE` (EnvInt, default 1).

- [ ] **Step 1: Write the failing test**

```python
import os
import unittest
from unittest import mock

from sglang.srt.environ import envs


class TestLayerMajorEnv(unittest.TestCase):
    def test_defaults_leave_the_path_off(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            for name in (
                "SGLANG_LAYER_MAJOR_PREFILL_MIN_TOKENS",
                "SGLANG_LAYER_MAJOR_STATE_NUMA_NODE",
            ):
                os.environ.pop(name, None)
            self.assertEqual(envs.SGLANG_LAYER_MAJOR_PREFILL_MIN_TOKENS.get(), 0)
            self.assertEqual(envs.SGLANG_LAYER_MAJOR_STATE_NUMA_NODE.get(), 1)

    def test_threshold_reads_from_the_environment(self):
        with mock.patch.dict(os.environ, {"SGLANG_LAYER_MAJOR_PREFILL_MIN_TOKENS": "32768"}):
            self.assertEqual(envs.SGLANG_LAYER_MAJOR_PREFILL_MIN_TOKENS.get(), 32768)


if __name__ == "__main__":
    unittest.main()
```

Also create an empty `test/registered/unit/layer_major/__init__.py` if sibling unit directories have one (check
`ls test/registered/unit/mem_cache/__init__.py`).

- [ ] **Step 2: Run the test and watch it fail**

Run (divix01, CPU): `PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES= taskset -c 18-35,54-63 /data/models/slang/.venv/bin/python -m pytest test/registered/unit/layer_major/test_layer_major_env.py -q -p no:randomly --basetemp=/mnt/nvme1/pytest-tmp/lm1; echo "EXIT=${PIPESTATUS[0]}"`

Expected: FAIL, an `AttributeError` on `SGLANG_LAYER_MAJOR_PREFILL_MIN_TOKENS`.

- [ ] **Step 3: Add the env vars**

In `python/sglang/srt/environ.py`, directly before `    # Kernels and indexer` (line ~1944), add:

```python
    # Layer-major prefill (plan 2026-09-27-dsv41-layer-major-prefill-phase1): a request whose uncached prompt suffix is
    # at least this many tokens runs every chunk through a layer before the next layer, so each layer's experts
    # stream once per prompt. 0 turns it off. Model- and quant-agnostic; the model must provide an adapter.
    SGLANG_LAYER_MAJOR_PREFILL_MIN_TOKENS = EnvInt(0)
    # NUMA node for the layer-major host state store; node 0 is kept near full by the expert tier (DSV41_REFERENCE
    # §25.4, 27.17).
    SGLANG_LAYER_MAJOR_STATE_NUMA_NODE = EnvInt(1)
```

- [ ] **Step 4: Run the test and watch it pass**

Same command. Expected: `2 passed`, `EXIT=0`.

- [ ] **Step 5: Correct the spec**

- §2: rename both env vars.
- §6.2 table: delete the `normalized` row. The total becomes ~49.2 KB per token, plus the top-k bundle.
- §6.3: the StateStore is ~12.9 GB + ~3 GB of metadata at 262,144 tokens.
- §8: the device staging is two ~168 MB buffers, ~0.39 GB of new staging in all.
- Add this paragraph to §6.2:

  > The chunked path under decoder bounded replay always has a late-layer tail (`deepseek_v4.py:4474-4480`), and the
  > fused carry-over requires `tail is None` (`:4549`), so it never carries `normalized`. The layer-major pass calls
  > every layer with `next_combined=None`, `next_norm=None` and `precomputed_attn=None`, the chunked path's own
  > arguments, and carries only `hidden` and `prev_pre`.

- §10.2: replace the tolerance bullets with:

  > Expected: bitwise identity of layer 20's output per chunk and of the final logits. Any mismatch is a bug to
  > investigate before it is waived. The greedy 64-token outputs must be identical at every length.

  Delete the fusion-off reference bullet.

- [ ] **Step 6: Commit**

```bash
git add python/sglang/srt/environ.py test/registered/unit/layer_major/test_layer_major_env.py \
  docs/superpowers/specs/2026-09-26-dsv41-layer-major-prefill-design.md
git commit -m "feat(layer-major): env vars; spec: carried state is hidden+prev_pre, generic env names" -m "<trailer>"
```

---

### Task 2: Layer 1: tensor-tree mover and StateStore

**Files:**
- Create: `python/sglang/srt/layer_major/__init__.py` (a one-line docstring, nothing else)
- Create: `python/sglang/srt/layer_major/tensor_tree.py`
- Create: `python/sglang/srt/layer_major/state_store.py`
- Test: `test/registered/unit/layer_major/test_state_store.py`

**Interfaces:**
- Produces:
  - `map_tensors(obj, fn) -> Any`: a copy of `obj` with every tensor replaced by `fn(tensor)`. It recurses into
    dataclass instances (all fields, including `init=False` ones that are set), msgspec Structs, dicts, lists and
    tuples. A tensor referenced twice maps to one result.
  - `FieldSpec(msgspec.Struct, frozen=True)` with `name: str`, `per_token_shape: tuple[int, ...]`, `dtype: str`
    (a torch dtype name such as `"bfloat16"`).
  - `StateStore(fields: list[FieldSpec], capacity_tokens: int, *, numa_node: int | None, pin: bool)` with:
    - `.capacity_tokens`;
    - `.write(field: str, start: int, rows: torch.Tensor) -> None` (host copy of `rows` at token offset `start`);
    - `.read_into(field: str, start: int, out: torch.Tensor, stream: torch.cuda.Stream | None) -> None`
      (host to `out`);
    - `.write_from(field: str, start: int, src: torch.Tensor, stream: torch.cuda.Stream | None) -> None`
      (device to host);
    - `.park(key: int, obj) -> None` and `.unpark(key: int, device: torch.device) -> Any` (per-chunk opaque objects,
      tensors held in host memory);
    - `.clear_parked() -> None`.
  - `state_store_bytes(fields, capacity_tokens) -> int`.

- [ ] **Step 1: Write the failing tests**

```python
import dataclasses
import unittest

import msgspec
import torch

from sglang.srt.layer_major.state_store import FieldSpec, StateStore, state_store_bytes
from sglang.srt.layer_major.tensor_tree import map_tensors


@dataclasses.dataclass
class _Meta:
    a: torch.Tensor
    b: list
    late: torch.Tensor = dataclasses.field(init=False, default=None)


class _S(msgspec.Struct):
    x: torch.Tensor
    n: int


class TestTensorTree(unittest.TestCase):
    def test_maps_every_tensor_and_keeps_shared_tensors_shared(self):
        t = torch.arange(4)
        meta = _Meta(a=t, b=[t, {"k": torch.ones(2)}, (torch.zeros(1),)])
        meta.late = torch.full((3,), 7)
        out = map_tensors(meta, lambda x: x + 1)
        self.assertTrue(torch.equal(out.a, t + 1))
        self.assertIs(out.a, out.b[0])
        self.assertTrue(torch.equal(out.b[1]["k"], torch.full((2,), 2.0)))
        self.assertTrue(torch.equal(out.late, torch.full((3,), 8)))
        self.assertTrue(torch.equal(meta.a, t))  # the input is left alone

    def test_maps_msgspec_structs(self):
        out = map_tensors(_S(x=torch.ones(1), n=3), lambda x: x * 5)
        self.assertEqual(out.n, 3)
        self.assertEqual(out.x.item(), 5.0)


class TestStateStore(unittest.TestCase):
    def _store(self, capacity=16):
        fields = [FieldSpec(name="hidden", per_token_shape=(2, 3), dtype="bfloat16"),
                  FieldSpec(name="prev_pre", per_token_shape=(2,), dtype="float32")]
        return StateStore(fields, capacity, numa_node=None, pin=False)

    def test_bytes_formula(self):
        fields = [FieldSpec(name="h", per_token_shape=(4, 5120), dtype="bfloat16"),
                  FieldSpec(name="p", per_token_shape=(4,), dtype="float32")]
        self.assertEqual(state_store_bytes(fields, 10), 10 * (4 * 5120 * 2 + 4 * 4))

    def test_round_trip_at_offsets(self):
        store = self._store()
        rows = torch.randn(5, 2, 3).to(torch.bfloat16)
        store.write("hidden", 4, rows)
        out = torch.empty(5, 2, 3, dtype=torch.bfloat16)
        store.read_into("hidden", 4, out, stream=None)
        self.assertTrue(torch.equal(out, rows))
        store.write_from("hidden", 4, rows * 2, stream=None)
        store.read_into("hidden", 4, out, stream=None)
        self.assertTrue(torch.equal(out, rows * 2))

    def test_refuses_rows_past_capacity(self):
        store = self._store(capacity=4)
        with self.assertRaises(ValueError):
            store.write("prev_pre", 2, torch.zeros(3, 2))

    def test_park_and_unpark_hold_objects_on_host(self):
        store = self._store()
        store.park(0, {"t": torch.arange(3)})
        got = store.unpark(0, torch.device("cpu"))
        self.assertTrue(torch.equal(got["t"], torch.arange(3)))
        store.clear_parked()
        with self.assertRaises(KeyError):
            store.unpark(0, torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run and watch it fail**

Run: the Task 1 command with `test_state_store.py` and `--basetemp=/mnt/nvme1/pytest-tmp/lm2`.
Expected: FAIL with `ModuleNotFoundError: No module named 'sglang.srt.layer_major'`.

- [ ] **Step 3: Implement `tensor_tree.py`**

```python
"""Copy a nested structure with every tensor replaced, to park per-chunk state off the device and bring it back."""

from __future__ import annotations

import copy
import dataclasses
from typing import Any, Callable

import msgspec
import torch


def map_tensors(obj: Any, fn: Callable[[torch.Tensor], torch.Tensor]) -> Any:
    return _map(obj, fn, {})


def _map(obj: Any, fn: Callable[[torch.Tensor], torch.Tensor], memo: dict[int, Any]) -> Any:
    if id(obj) in memo:
        return memo[id(obj)]
    if isinstance(obj, torch.Tensor):
        out = fn(obj)
    elif dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        out = copy.copy(obj)
        memo[id(obj)] = out
        for f in dataclasses.fields(obj):
            # An init=False field without a default is absent until someone sets it.
            if f.name in obj.__dict__:
                setattr(out, f.name, _map(obj.__dict__[f.name], fn, memo))
        return out
    elif isinstance(obj, msgspec.Struct):
        out = copy.copy(obj)
        memo[id(obj)] = out
        for name in obj.__struct_fields__:
            setattr(out, name, _map(getattr(obj, name), fn, memo))
        return out
    elif isinstance(obj, dict):
        out = {k: _map(v, fn, memo) for k, v in obj.items()}
    elif isinstance(obj, list):
        out = [_map(v, fn, memo) for v in obj]
    elif isinstance(obj, tuple):
        items = [_map(v, fn, memo) for v in obj]
        out = type(obj)(*items) if hasattr(type(obj), "_fields") else tuple(items)
    else:
        return obj
    memo[id(obj)] = out
    return out
```

(`getattr(obj, name)` over `__struct_fields__` is a field walk, not defensive access. `hasattr(type(obj), "_fields")`
detects a namedtuple type, which has no other test.)

- [ ] **Step 4: Implement `state_store.py`**

```python
"""Pinned host storage for per-token state carried between layers of a layer-major prefill, plus parked per-chunk
objects. Holds named fields only: what a field means belongs to the model adapter."""

from __future__ import annotations

import math
from typing import Any

import msgspec
import torch

from sglang.srt.layer_major.tensor_tree import map_tensors


class FieldSpec(msgspec.Struct, frozen=True):
    name: str
    per_token_shape: tuple[int, ...]
    dtype: str


def _dtype(spec: FieldSpec) -> torch.dtype:
    return getattr(torch, spec.dtype)


def state_store_bytes(fields: list[FieldSpec], capacity_tokens: int) -> int:
    return capacity_tokens * sum(math.prod(f.per_token_shape) * _dtype(f).itemsize for f in fields)


class StateStore:
    def __init__(self, fields: list[FieldSpec], capacity_tokens: int, *, numa_node: int | None, pin: bool):
        self.capacity_tokens = capacity_tokens
        self._fields = {f.name: f for f in fields}
        self._host = {f.name: _allocate(f, capacity_tokens, numa_node=numa_node, pin=pin) for f in fields}
        self._pin = pin
        self._parked: dict[int, Any] = {}

    def _rows(self, field: str, start: int, count: int) -> torch.Tensor:
        if start < 0 or start + count > self.capacity_tokens:
            raise ValueError(
                f"rows [{start}, {start + count}) of {field!r} exceed the store's {self.capacity_tokens} tokens"
            )
        return self._host[field][start : start + count]

    def write(self, field: str, start: int, rows: torch.Tensor) -> None:
        self._rows(field, start, rows.shape[0]).copy_(rows)

    def read_into(self, field: str, start: int, out: torch.Tensor, stream: torch.cuda.Stream | None) -> None:
        src = self._rows(field, start, out.shape[0])
        with _on(stream):
            out.copy_(src, non_blocking=self._pin)

    def write_from(self, field: str, start: int, src: torch.Tensor, stream: torch.cuda.Stream | None) -> None:
        dst = self._rows(field, start, src.shape[0])
        with _on(stream):
            dst.copy_(src, non_blocking=self._pin)

    def park(self, key: int, obj: Any) -> None:
        self._parked[key] = map_tensors(obj, self._to_host)

    def unpark(self, key: int, device: torch.device) -> Any:
        return map_tensors(self._parked[key], lambda t: t.to(device, non_blocking=self._pin))

    def clear_parked(self) -> None:
        self._parked.clear()

    def _to_host(self, t: torch.Tensor) -> torch.Tensor:
        if t.device.type == "cpu":
            return t.clone()
        host = torch.empty(t.shape, dtype=t.dtype, device="cpu", pin_memory=self._pin)
        host.copy_(t, non_blocking=self._pin)
        return host


def _allocate(spec: FieldSpec, capacity_tokens: int, *, numa_node: int | None, pin: bool) -> torch.Tensor:
    shape = (capacity_tokens,) + tuple(spec.per_token_shape)
    if numa_node is None and not pin:
        return torch.empty(shape, dtype=_dtype(spec))
    from sglang.srt.layers.moe.expert_host_tier import allocate_host_slab

    nbytes = math.prod(shape) * _dtype(spec).itemsize
    placement = [(numa_node, nbytes)] if numa_node is not None else []
    return allocate_host_slab(capacity_tokens, spec.per_token_shape, _dtype(spec), register=pin, placement=placement)


class _on:
    def __init__(self, stream: torch.cuda.Stream | None):
        self._ctx = torch.cuda.stream(stream) if stream is not None else None

    def __enter__(self):
        if self._ctx is not None:
            self._ctx.__enter__()

    def __exit__(self, *exc):
        if self._ctx is not None:
            self._ctx.__exit__(*exc)
```

- [ ] **Step 5: Run and watch it pass**

Same command. Expected: `6 passed`, `EXIT=0`.

- [ ] **Step 6: Commit**

```bash
git add python/sglang/srt/layer_major/__init__.py python/sglang/srt/layer_major/tensor_tree.py \
  python/sglang/srt/layer_major/state_store.py test/registered/unit/layer_major/test_state_store.py
git commit -m "feat(layer-major): tensor-tree mover and pinned NUMA-placed state store" -m "<trailer>"
```

---

### Task 3: Layer 1: protocols, NullResidency, heartbeat, driver

**Files:**
- Create: `python/sglang/srt/layer_major/protocols.py`
- Create: `python/sglang/srt/layer_major/heartbeat.py`
- Create: `python/sglang/srt/layer_major/driver.py`
- Test: `test/registered/unit/layer_major/test_driver.py`

**Interfaces:**
- Consumes: `StateStore`, `FieldSpec` (Task 2).
- Produces:
  - **`LayerMajorModelAdapter` (Protocol):**
    - `field_specs() -> list[FieldSpec]`
    - `begin_pass(forward_batch, schedule_batch, store: StateStore) -> Any` (returns an opaque pass handle)
    - `layer_ids(handle) -> range`
    - `num_chunks(handle) -> int`
    - `run_layer(handle, layer_id: int, chunk: int, store: StateStore) -> None`
    - `finish_pass(handle, store: StateStore) -> Any` (returns the model's logits output)
    - `release_pass(handle, store: StateStore, *, failed: bool) -> None`
  - **`ExpertResidency` (Protocol):** `begin(layer_ids: range)`, `make_resident(layer_id: int)`, `restore()`,
    `end()`.
  - **`NullResidency`:** every method is a no-op.
  - **Heartbeat:** `PassHeartbeat` with `.tick()`; module functions `pass_progress() -> int` and
    `current_heartbeat() -> PassHeartbeat`.
  - **Driver:** `run_pass(adapter, residency, forward_batch, schedule_batch, store) -> Any`.
- Call order, which the tests pin:
  1. `begin_pass`, then `residency.begin`.
  2. For each layer, `residency.make_resident(layer)`, then `run_layer(layer, chunk)` for each chunk in order, with a
     heartbeat tick after each.
  3. `residency.restore()`, then `finish_pass`.
  4. Finally, whether or not the pass raised: `residency.end()` and `release_pass(failed=...)`.
  5. On an exception, `residency.restore()` runs before `end()`, and the exception propagates.

- [ ] **Step 1: Write the failing tests**

```python
import unittest

from sglang.srt.layer_major.driver import run_pass
from sglang.srt.layer_major.heartbeat import pass_progress


class _Adapter:
    def __init__(self, layers=range(0, 3), chunks=2, fail_at=None):
        self.calls, self._layers, self._chunks, self._fail_at = [], layers, chunks, fail_at

    def field_specs(self):
        return []

    def begin_pass(self, forward_batch, schedule_batch, store):
        self.calls.append("begin_pass")
        return "handle"

    def layer_ids(self, handle):
        return self._layers

    def num_chunks(self, handle):
        return self._chunks

    def run_layer(self, handle, layer_id, chunk, store):
        if (layer_id, chunk) == self._fail_at:
            raise RuntimeError("boom")
        self.calls.append(("layer", layer_id, chunk))

    def finish_pass(self, handle, store):
        self.calls.append("finish_pass")
        return "logits"

    def release_pass(self, handle, store, *, failed):
        self.calls.append(("release", failed))


class _Residency:
    def __init__(self, calls):
        self.calls = calls

    def begin(self, layer_ids):
        self.calls.append(("res.begin", tuple(layer_ids)))

    def make_resident(self, layer_id):
        self.calls.append(("res.make", layer_id))

    def restore(self):
        self.calls.append("res.restore")

    def end(self):
        self.calls.append("res.end")


class TestDriver(unittest.TestCase):
    def test_layer_outer_chunk_inner_order(self):
        a = _Adapter()
        out = run_pass(a, _Residency(a.calls), "fb", "sb", store=None)
        self.assertEqual(out, "logits")
        self.assertEqual(
            a.calls,
            ["begin_pass", ("res.begin", (0, 1, 2)),
             ("res.make", 0), ("layer", 0, 0), ("layer", 0, 1),
             ("res.make", 1), ("layer", 1, 0), ("layer", 1, 1),
             ("res.make", 2), ("layer", 2, 0), ("layer", 2, 1),
             "res.restore", "finish_pass", "res.end", ("release", False)],
        )

    def test_heartbeat_ticks_once_per_layer_chunk(self):
        before = pass_progress()
        a = _Adapter()
        run_pass(a, _Residency(a.calls), "fb", "sb", store=None)
        self.assertEqual(pass_progress() - before, 3 * 2)

    def test_exception_mid_pass_still_ends_residency(self):
        a = _Adapter(fail_at=(1, 1))
        with self.assertRaisesRegex(RuntimeError, "boom"):
            run_pass(a, _Residency(a.calls), "fb", "sb", store=None)
        self.assertEqual(a.calls[-3:], ["res.restore", "res.end", ("release", True)])
        self.assertNotIn("finish_pass", a.calls)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run and watch it fail**

Run: the Task 1 command on `test_driver.py`. Expected: FAIL with `ModuleNotFoundError ... layer_major.driver`.

- [ ] **Step 3: Implement `protocols.py`**

```python
"""The two seams of the layer-major strategy: the model adapter and expert residency. The strategy knows nothing
else about the model or its quantization."""

from __future__ import annotations

from typing import Any, Protocol

from sglang.srt.layer_major.state_store import FieldSpec, StateStore


class LayerMajorModelAdapter(Protocol):
    def field_specs(self) -> list[FieldSpec]: ...

    def begin_pass(self, forward_batch: Any, schedule_batch: Any, store: StateStore) -> Any: ...

    def layer_ids(self, handle: Any) -> range: ...

    def num_chunks(self, handle: Any) -> int: ...

    def run_layer(self, handle: Any, layer_id: int, chunk: int, store: StateStore) -> None: ...

    def finish_pass(self, handle: Any, store: StateStore) -> Any: ...

    def release_pass(self, handle: Any, store: StateStore, *, failed: bool) -> None: ...


class ExpertResidency(Protocol):
    def begin(self, layer_ids: range) -> None: ...

    def make_resident(self, layer_id: int) -> None: ...

    def restore(self) -> None: ...

    def end(self) -> None: ...


class NullResidency:
    """Experts come through the model's normal path; nothing is borrowed, so nothing is restored."""

    def begin(self, layer_ids: range) -> None:
        pass

    def make_resident(self, layer_id: int) -> None:
        pass

    def restore(self) -> None:
        pass

    def end(self) -> None:
        pass
```

- [ ] **Step 4: Implement `heartbeat.py`**

```python
"""Progress of the running layer-major pass, read by the scheduler watchdog: one pass is a single forward that can
outlast the watchdog timeout, and forward_ct only moves between forwards."""

from __future__ import annotations

import itertools


class PassHeartbeat:
    def __init__(self):
        self._counter = itertools.count(1)
        self._value = 0

    def tick(self) -> None:
        # itertools.count is atomic under the GIL; the watchdog thread only reads _value.
        self._value = next(self._counter)

    @property
    def value(self) -> int:
        return self._value


_HEARTBEAT = PassHeartbeat()


def current_heartbeat() -> PassHeartbeat:
    return _HEARTBEAT


def pass_progress() -> int:
    return _HEARTBEAT.value
```

- [ ] **Step 5: Implement `driver.py`**

```python
"""Layer-major prefill: every chunk through a layer before the next layer."""

from __future__ import annotations

from typing import Any

from sglang.srt.layer_major.heartbeat import current_heartbeat
from sglang.srt.layer_major.protocols import ExpertResidency, LayerMajorModelAdapter
from sglang.srt.layer_major.state_store import StateStore


def run_pass(
    adapter: LayerMajorModelAdapter,
    residency: ExpertResidency,
    forward_batch: Any,
    schedule_batch: Any,
    store: StateStore,
) -> Any:
    heartbeat = current_heartbeat()
    handle = adapter.begin_pass(forward_batch, schedule_batch, store)
    failed = True
    restored = False
    try:
        layers = adapter.layer_ids(handle)
        residency.begin(layers)
        for layer_id in layers:
            residency.make_resident(layer_id)
            for chunk in range(adapter.num_chunks(handle)):
                adapter.run_layer(handle, layer_id, chunk, store)
                heartbeat.tick()
        residency.restore()
        restored = True
        output = adapter.finish_pass(handle, store)
        failed = False
        return output
    finally:
        # Borrowed expert bytes must be back before anything else reads them, even on failure.
        if not restored:
            residency.restore()
        residency.end()
        adapter.release_pass(handle, store, failed=failed)
```

- [ ] **Step 6: Run and watch it pass**

Same command. Expected: `3 passed`, `EXIT=0`.

- [ ] **Step 7: Commit**

```bash
git add python/sglang/srt/layer_major/protocols.py python/sglang/srt/layer_major/heartbeat.py \
  python/sglang/srt/layer_major/driver.py test/registered/unit/layer_major/test_driver.py
git commit -m "feat(layer-major): adapter and residency protocols, heartbeat, layer-outer driver" -m "<trailer>"
```

---

### Task 4: Layer 1 gate, plus the DSV4.1 launch refusals

**Files:**
- Create: `python/sglang/srt/layer_major/gate.py`
- Modify: `python/sglang/srt/arg_groups/deepseek_v4_hook.py` (`validate_deepseek_v41_features`, after the
  `if cfg.enable_decoder_swa_bounded_replay:` block at ~366-383)
- Test: `test/registered/unit/layer_major/test_gate.py`

**Interfaces:**
- Produces:
  - `LayerMajorGate(msgspec.Struct, frozen=True)` with `min_tokens: int` and `max_tokens: int`, and
    `.admits(*, extend_len: int, wants_prompt_logprobs: bool, wants_hidden_states: bool) -> bool`.
  - `gate_from_env(*, max_tokens: int) -> LayerMajorGate | None` (None when the threshold is 0).
  - `launch_refusal(*, max_running_requests, speculative_algorithm, enable_dp_attention, attn_cp_size,
    enable_two_batch_overlap) -> str | None`: the generic refusals.

- [ ] **Step 1: Write the failing tests**

```python
import os
import unittest
from unittest import mock

from sglang.srt.layer_major.gate import LayerMajorGate, gate_from_env, launch_refusal


class TestGate(unittest.TestCase):
    def test_threshold_and_capacity(self):
        g = LayerMajorGate(min_tokens=32768, max_tokens=262144)
        self.assertFalse(g.admits(extend_len=32767, wants_prompt_logprobs=False, wants_hidden_states=False))
        self.assertTrue(g.admits(extend_len=32768, wants_prompt_logprobs=False, wants_hidden_states=False))
        self.assertFalse(g.admits(extend_len=262145, wants_prompt_logprobs=False, wants_hidden_states=False))

    def test_refuses_prompt_logprobs_and_hidden_states(self):
        g = LayerMajorGate(min_tokens=10, max_tokens=100)
        self.assertFalse(g.admits(extend_len=50, wants_prompt_logprobs=True, wants_hidden_states=False))
        self.assertFalse(g.admits(extend_len=50, wants_prompt_logprobs=False, wants_hidden_states=True))

    def test_gate_from_env_off_by_default(self):
        with mock.patch.dict(os.environ, {"SGLANG_LAYER_MAJOR_PREFILL_MIN_TOKENS": "0"}):
            self.assertIsNone(gate_from_env(max_tokens=1000))
        with mock.patch.dict(os.environ, {"SGLANG_LAYER_MAJOR_PREFILL_MIN_TOKENS": "64"}):
            self.assertEqual(gate_from_env(max_tokens=1000), LayerMajorGate(min_tokens=64, max_tokens=1000))

    def test_launch_refusals(self):
        ok = dict(max_running_requests=1, speculative_algorithm=None, enable_dp_attention=False, attn_cp_size=1,
                  enable_two_batch_overlap=False)
        self.assertIsNone(launch_refusal(**ok))
        self.assertIn("max-running-requests", launch_refusal(**{**ok, "max_running_requests": 2}))
        self.assertIn("speculative", launch_refusal(**{**ok, "speculative_algorithm": "EAGLE"}))
        self.assertIn("DP attention", launch_refusal(**{**ok, "enable_dp_attention": True}))
        self.assertIn("context parallelism", launch_refusal(**{**ok, "attn_cp_size": 2}))
        self.assertIn("two-batch overlap", launch_refusal(**{**ok, "enable_two_batch_overlap": True}))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run and watch it fail**

Expected: `ModuleNotFoundError ... layer_major.gate`.

- [ ] **Step 3: Implement `gate.py`**

```python
"""Which requests the layer-major prefill takes, and which launches cannot use it at all."""

from __future__ import annotations

import msgspec

from sglang.srt.environ import envs


class LayerMajorGate(msgspec.Struct, frozen=True):
    min_tokens: int
    max_tokens: int

    def admits(self, *, extend_len: int, wants_prompt_logprobs: bool, wants_hidden_states: bool) -> bool:
        # Prompt logprobs and hidden states need every prompt row past the last layer; the pass computes the tail only.
        if wants_prompt_logprobs or wants_hidden_states:
            return False
        return self.min_tokens <= extend_len <= self.max_tokens


def gate_from_env(*, max_tokens: int) -> LayerMajorGate | None:
    min_tokens = envs.SGLANG_LAYER_MAJOR_PREFILL_MIN_TOKENS.get()
    if min_tokens <= 0:
        return None
    return LayerMajorGate(min_tokens=min_tokens, max_tokens=max_tokens)


def launch_refusal(
    *,
    max_running_requests: int | None,
    speculative_algorithm: str | None,
    enable_dp_attention: bool,
    attn_cp_size: int,
    enable_two_batch_overlap: bool,
) -> str | None:
    # One blocking pass per request; nothing may run between its chunks.
    if max_running_requests != 1:
        return "SGLANG_LAYER_MAJOR_PREFILL_MIN_TOKENS requires --max-running-requests 1"
    for feature, enabled in (
        ("speculative decoding", speculative_algorithm is not None),
        ("DP attention", enable_dp_attention),
        ("context parallelism", attn_cp_size > 1),
        ("two-batch overlap", enable_two_batch_overlap),
    ):
        if enabled:
            return f"SGLANG_LAYER_MAJOR_PREFILL_MIN_TOKENS cannot be combined with {feature} yet"
    return None
```

- [ ] **Step 4: Wire the launch refusals into the DSV4.1 hook**

In `validate_deepseek_v41_features`, after the decoder-replay block, add (the field names are from
`arg_groups/fields`; check each with `grep -n "enable_two_batch_overlap\|attn_cp_size\|speculative_algorithm"
python/sglang/srt/arg_groups/fields/*.py`):

```python
    if envs.SGLANG_LAYER_MAJOR_PREFILL_MIN_TOKENS.get() > 0:
        from sglang.srt.layer_major.gate import launch_refusal

        reason = launch_refusal(
            max_running_requests=cfg.max_running_requests,
            speculative_algorithm=cfg.speculative_algorithm,
            enable_dp_attention=cfg.enable_dp_attention,
            attn_cp_size=cfg.attn_cp_size,
            enable_two_batch_overlap=cfg.enable_two_batch_overlap,
        )
        # The DSV4.1 adapter runs the late layers on the tail and maps a window ring on the paged SWA allocator.
        if reason is None and not cfg.enable_decoder_swa_bounded_replay:
            reason = "layer-major prefill on DeepSeek-V4.1 requires --enable-decoder-swa-bounded-replay"
        if reason is None and cfg.enable_encoder_swa_bounded_replay:
            reason = "layer-major prefill cannot be combined with --enable-encoder-swa-bounded-replay"
        if reason is not None:
            raise ValueError(reason)
```

Add `from sglang.srt.environ import envs` at the top of `deepseek_v4_hook.py` if it is not already imported.

- [ ] **Step 5: Run and watch it pass**

Expected: `4 passed`, `EXIT=0`. Also run the existing hook tests:
`grep -rln validate_deepseek_v41_features test/registered/unit | xargs` through the same pytest command. Expected:
the same pass count as before this task (record both runs in the ledger).

- [ ] **Step 6: Commit**

```bash
git add python/sglang/srt/layer_major/gate.py python/sglang/srt/arg_groups/deepseek_v4_hook.py \
  test/registered/unit/layer_major/test_gate.py
git commit -m "feat(layer-major): admission gate and launch refusals" -m "<trailer>"
```

---

### Task 5: The watchdog counts pass progress

**Files:**
- Modify: `python/sglang/srt/managers/scheduler_components/invariant_checker.py:507`
  (`get_counter=lambda: scheduler.forward_ct`)
- Test: `test/registered/unit/layer_major/test_watchdog_heartbeat.py`

**Interfaces:**
- Consumes: `pass_progress()` and `current_heartbeat()` (Task 3).
- Produces: `create_scheduler_watchdog(...)` counter = `scheduler.forward_ct + pass_progress()`.

- [ ] **Step 1: Write the failing test**

```python
import unittest
from types import SimpleNamespace
from unittest import mock

from sglang.srt.layer_major.heartbeat import current_heartbeat
from sglang.srt.managers.scheduler_components import invariant_checker


class TestWatchdogHeartbeat(unittest.TestCase):
    def test_watchdog_counter_advances_with_heartbeat(self):
        captured = {}

        def fake_watchdog(**kwargs):
            captured.update(kwargs)
            return SimpleNamespace()

        scheduler = SimpleNamespace(forward_ct=7, is_initializing=False, cur_batch_for_debug=object())
        with mock.patch.object(invariant_checker, "WatchdogRaw", side_effect=fake_watchdog):
            invariant_checker.create_scheduler_watchdog(scheduler, watchdog_timeout=300)
        before = captured["get_counter"]()
        current_heartbeat().tick()
        self.assertEqual(captured["get_counter"](), before + 1)
        scheduler.forward_ct += 1
        self.assertEqual(captured["get_counter"](), before + 2)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run and watch it fail**

Expected: FAIL, `AssertionError: 7 != 8`, since today's counter ignores the heartbeat.

- [ ] **Step 3: Implement**

In `invariant_checker.py`, replace `get_counter=lambda: scheduler.forward_ct,` with:

```python
        # A layer-major prefill is one forward that can outlast the timeout; it ticks per layer and chunk.
        get_counter=lambda: scheduler.forward_ct + pass_progress(),
```

Add at the top: `from sglang.srt.layer_major.heartbeat import pass_progress`.

- [ ] **Step 4: Run and watch it pass**

Expected: `1 passed`, `EXIT=0`.

- [ ] **Step 5: Commit**

```bash
git add python/sglang/srt/managers/scheduler_components/invariant_checker.py \
  test/registered/unit/layer_major/test_watchdog_heartbeat.py
git commit -m "feat(layer-major): scheduler watchdog counts pass progress" -m "<trailer>"
```

---

### Task 6: Window-KV ring: admission budget, allocation, finalize, release (generic hybrid SWA)

**Why:** spec §5.1 and §6.4. A whole-suffix extend must reserve full KV for every token but window KV for a ring of
`chunk + page` slots only. After the pass, only the final window's ring slots stay mapped. This code is generic to
any hybrid-SWA model: the DSV4 adapter only chooses the ring size.

**Files:**
- Modify: `python/sglang/srt/mem_cache/prefill_budget.py` (add `SWAPrefillBudget.check_prefill_ring`)
- Modify: `python/sglang/srt/mem_cache/allocator/swa.py` (add `SWATokenToKVPoolAllocator.ring_slots`,
  `map_ring_positions` and `finalize_ring`)
- Modify: `python/sglang/srt/mem_cache/allocation.py` (`alloc_paged_token_slots_extend` takes `swa_ring_tokens`)
- Test: `test/registered/unit/layer_major/test_window_ring.py`

**Interfaces:**
- Produces:
  - `SWAPrefillBudget.check_prefill_ring(*, total_tokens: int, max_new_tokens: int, ring_tokens: int) -> bool`:
    True iff the whole extend's full KV fits `remaining_total` and `ring_tokens + page_size` fits `remaining_swa`
    (strict `<`, as `check_prefill` uses for the paged allocator).
  - `alloc_paged_token_slots_extend(..., swa_ring_tokens: int | None = None)`: when set, calls
    `allocator.alloc_extend_swa_tail(..., swa_tail_len=swa_ring_tokens)` (bs=1).
  - `SWATokenToKVPoolAllocator.ring_slots(full_locs_tail: torch.Tensor) -> torch.Tensor`: the ring's SWA slots, read
    from the mapping of the last `ring` extend positions right after allocation.
  - `SWATokenToKVPoolAllocator.map_ring_positions(full_locs: torch.Tensor, positions: torch.Tensor,
    ring: torch.Tensor) -> None`: maps each position's full loc to `ring[(pos // page) % pages * page + pos % page]`.
  - `SWATokenToKVPoolAllocator.finalize_ring(extend_full_locs: torch.Tensor, extend_start: int, keep_from: int,
    ring: torch.Tensor) -> None`:
    - clears the mapping of every extend position below `keep_from`;
    - releases every ring slot not mapped by a kept position;
    - leaves the kept positions mapped.

- [ ] **Step 1: Write the failing tests**

Build a real allocator as `test_swa_eviction_boundary.py:_build_swa_tree` does, with page_size 4 and a small pool.

```python
import unittest

import torch

from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
from sglang.srt.mem_cache.prefill_budget import SWAPrefillBudget
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool

PAGE = 4
CHUNK = 16
RING = CHUNK + PAGE  # 5 pages


def _allocator(size=256, size_swa=64):
    kv = SWAKVPool(size=size, size_swa=size_swa, page_size=PAGE, dtype=torch.bfloat16, head_num=1, head_dim=8,
                   swa_attention_layer_ids=[1], full_attention_layer_ids=[0], device="cpu")
    return SWATokenToKVPoolAllocator(size=size, size_swa=size_swa, page_size=PAGE, dtype=torch.bfloat16,
                                     device="cpu", kvcache=kv, need_sort=False)


def _alloc_whole(a, n):
    t = torch.tensor
    full = a.alloc_extend_swa_tail(t([0]), t([0]), t([n]), t([n]), t([-1]), n, swa_tail_len=RING)
    assert full is not None
    return full


class TestWindowRing(unittest.TestCase):
    def test_ring_admission_restores_available_size(self):
        a = _allocator()
        swa_before = a.swa_available_size()
        full = _alloc_whole(a, 64)
        ring = a.ring_slots(full[-RING:])
        self.assertEqual(ring.numel(), RING)
        self.assertEqual(swa_before - a.swa_available_size(), RING)
        a.finalize_ring(full, extend_start=0, keep_from=64 - 8, ring=ring)
        a.free(full)
        self.assertEqual(a.swa_available_size(), swa_before)

    def test_every_chunk_sees_its_predecessor_page_in_distinct_slots(self):
        a = _allocator()
        n = 64
        full = _alloc_whole(a, n)
        ring = a.ring_slots(full[-RING:])
        for start in range(0, n, CHUNK):
            pos = torch.arange(start, min(n, start + CHUNK))
            a.map_ring_positions(full[pos], pos, ring)
            window = torch.arange(max(0, start - PAGE), min(n, start + CHUNK))
            slots = a.full_to_swa_index_mapping[full[window]]
            self.assertEqual(slots.unique().numel(), window.numel())

    def test_release_after_finalize_returns_every_slot(self):
        a = _allocator()
        swa_before, full_before = a.swa_available_size(), a.full_available_size()
        n = 50  # final chunk of 2 tokens
        full = _alloc_whole(a, n)
        ring = a.ring_slots(full[-RING:])
        for start in range(0, n, CHUNK):
            pos = torch.arange(start, min(n, start + CHUNK))
            a.map_ring_positions(full[pos], pos, ring)
        keep_from = (n - 8) // PAGE * PAGE
        a.finalize_ring(full, extend_start=0, keep_from=keep_from, ring=ring)
        kept = a.full_to_swa_index_mapping[full[keep_from:]]
        self.assertTrue(bool((kept > 0).all()))
        self.assertTrue(bool((a.full_to_swa_index_mapping[full[:keep_from]] == 0).all()))
        a.free(full)
        self.assertEqual((a.swa_available_size(), a.full_available_size()), (swa_before, full_before))


class TestRingBudget(unittest.TestCase):
    def _budget(self, remaining_total, remaining_swa):
        b = object.__new__(SWAPrefillBudget)
        b.page_size = PAGE
        b.req_ring = False
        type(b).remaining_total = property(lambda self: remaining_total)
        type(b).remaining_swa = property(lambda self: remaining_swa)
        return b

    def test_ring_budget(self):
        self.assertTrue(self._budget(1000, RING + PAGE + 1).check_prefill_ring(total_tokens=500, max_new_tokens=8,
                                                                               ring_tokens=RING))
        self.assertFalse(self._budget(1000, RING + PAGE).check_prefill_ring(total_tokens=500, max_new_tokens=8,
                                                                            ring_tokens=RING))
        self.assertFalse(self._budget(400, 1000).check_prefill_ring(total_tokens=500, max_new_tokens=8,
                                                                    ring_tokens=RING))


if __name__ == "__main__":
    unittest.main()
```

`_budget` patches the class properties. If other tests in the file share `SWAPrefillBudget`, set the properties on a
throwaway subclass instead:

```python
cls = type("B", (SWAPrefillBudget,), {"remaining_total": property(...), "remaining_swa": property(...)})
b = object.__new__(cls)
```

Prefer the subclass form from the start.

- [ ] **Step 2: Run and watch it fail**

Expected: FAIL, an `AttributeError` on `ring_slots` or `check_prefill_ring`.

If `SWAKVPool` or the allocator constructor rejects these arguments on this branch, copy the exact construction from
`test_swa_eviction_boundary.py:_build_swa_tree` and ledger it as a ruling.

- [ ] **Step 3: Implement `check_prefill_ring`**

In `SWAPrefillBudget`, after `check_prefill`:

```python
    def check_prefill_ring(self, *, total_tokens: int, max_new_tokens: int, ring_tokens: int) -> bool:
        """Admission of a layer-major extend: full KV for every token, window KV for one ring of slots only."""
        if total_tokens >= self.remaining_total:
            return False
        # One page of decode headroom past the ring, as estimate_swa_kv_tokens charges a normal prefill.
        return ring_tokens + self.page_size < self.remaining_swa
```

- [ ] **Step 4: Implement the allocator methods**

In `SWATokenToKVPoolAllocator`, after `clear_full_to_swa_mapping`:

```python
    def ring_slots(self, full_locs_tail: torch.Tensor) -> torch.Tensor:
        """The window slots alloc_extend_swa_tail gave the last ring positions of an extend, in position order."""
        return self.full_to_swa_index_mapping[full_locs_tail.to(torch.int64)].clone()

    def map_ring_positions(self, full_locs: torch.Tensor, positions: torch.Tensor, ring: torch.Tensor) -> None:
        # Ring page (pos // page) mod pages: a chunk of whole pages plus its predecessor page never share a slot.
        pages = ring.numel() // self.page_size
        idx = (positions // self.page_size) % pages * self.page_size + positions % self.page_size
        self.set_full_to_swa_mapping(full_locs, ring[idx.to(ring.device)])

    def finalize_ring(
        self, extend_full_locs: torch.Tensor, extend_start: int, keep_from: int, ring: torch.Tensor
    ) -> None:
        """Keep the window from keep_from (an absolute position) to the end of the extend; drop the rest of the ring."""
        split = keep_from - extend_start
        self.clear_full_to_swa_mapping(extend_full_locs[:split])
        kept = self.full_to_swa_index_mapping[extend_full_locs[split:].to(torch.int64)]
        unused = ring[~torch.isin(ring, kept)]
        self._release_swa(unused)
```

If `_release_swa` expects page-level indices or a padding filter (read it at `swa.py` before implementing), adapt
the call and ledger it. The release test pins the result: every slot comes back.

- [ ] **Step 5: Thread `swa_ring_tokens` through `alloc_paged_token_slots_extend`**

In `mem_cache/allocation.py`:
- add a keyword parameter `swa_ring_tokens: Optional[int] = None` to `alloc_paged_token_slots_extend`;
- before the `out = allocator.alloc_extend(` call (~211), add:

```python
    if swa_ring_tokens is not None:
        # Layer-major prefill: full KV for the whole extend, window KV for one ring only.
        out = allocator.alloc_extend_swa_tail(
            prefix_lens, prefix_lens_cpu, seq_lens, seq_lens_cpu, last_loc, extend_num_tokens,
            swa_tail_len=swa_ring_tokens,
        )
    else:
        out = allocator.alloc_extend(
            prefix_lens,
            prefix_lens_cpu,
            seq_lens,
            seq_lens_cpu,
            last_loc,
            extend_num_tokens,
            **extra_alloc_kwargs,
        )
```

Then pass `swa_ring_tokens=batch.layer_major_ring_tokens` from `alloc_for_extend` into the call at ~403. The field is
added in Task 7; until then, pass `None` explicitly.

- [ ] **Step 6: Run and watch it pass**

Expected: `4 passed`, `EXIT=0`. Then run the existing budget and allocator tests with the same command:
`test/registered/unit/mem_cache/test_prefill_memory_budget.py test/registered/unit/mem_cache/test_swa_eviction_boundary.py`.
Expected: the same result as at the task's BASE commit (record both runs).

- [ ] **Step 7: Commit**

```bash
git add python/sglang/srt/mem_cache/prefill_budget.py python/sglang/srt/mem_cache/allocator/swa.py \
  python/sglang/srt/mem_cache/allocation.py test/registered/unit/layer_major/test_window_ring.py
git commit -m "feat(layer-major): window-KV ring admission, allocation, mapping and finalize" -m "<trailer>"
```

---

### Task 7: Scheduler: admit a whole-suffix layer-major extend

**Files:**
- Modify: `python/sglang/srt/managers/schedule_policy.py` (`PrefillAdder.__init__`, `_select_prefill_admission`)
- Modify: `python/sglang/srt/managers/schedule_batch.py` (the `Req` field `layer_major`; the ScheduleBatch field
  `layer_major_ring_tokens`, set in `ScheduleBatch.init_new`)
- Modify: `python/sglang/srt/mem_cache/allocation.py` (pass `batch.layer_major_ring_tokens`)
- Modify: `python/sglang/srt/managers/scheduler.py` (build the gate once in `__init__`, pass it to `PrefillAdder`)
- Test: `test/registered/unit/layer_major/test_layer_major_admission.py`

**Interfaces:**
- Consumes: `LayerMajorGate` and `gate_from_env` (Task 4); `check_prefill_ring` (Task 6).
- Produces:
  - A module-level pure function in `schedule_policy.py`:
    `layer_major_admission(*, gate, budget, req_wants_prompt_logprobs: bool, req_wants_hidden: bool, prefix_len: int,
    extend_len: int, total_tokens: int, max_new_tokens: int, ring_tokens: int) -> _PrefillAdmission | None`.
    It returns `_PrefillAdmission(prefix_len, extend_len, max_new_tokens, False)` when the gate admits and the ring
    budget fits, else None.
  - `PrefillAdder(..., layer_major_gate: LayerMajorGate | None = None, layer_major_ring_tokens: int = 0)`.
  - `Req.layer_major: bool = False`.
  - `ScheduleBatch.layer_major_ring_tokens: int | None`: the ring size when the batch's single request is
    layer-major, else None.

- [ ] **Step 1: Write the failing test** (a pure-function test, so no scheduler is needed)

```python
import unittest
from types import SimpleNamespace

from sglang.srt.layer_major.gate import LayerMajorGate
from sglang.srt.managers.schedule_policy import layer_major_admission


def _budget(fits):
    return SimpleNamespace(check_prefill_ring=lambda **kw: fits)


class TestLayerMajorAdmission(unittest.TestCase):
    ARGS = dict(req_wants_prompt_logprobs=False, req_wants_hidden=False, prefix_len=512, extend_len=40000,
                total_tokens=41000, max_new_tokens=64, ring_tokens=4352)

    def test_admits_whole_suffix_when_gate_and_ring_fit(self):
        adm = layer_major_admission(gate=LayerMajorGate(min_tokens=32768, max_tokens=262144), budget=_budget(True),
                                    **self.ARGS)
        self.assertEqual((adm.prefix_len, adm.extend_len, adm.max_new_tokens, adm.is_chunked),
                         (512, 40000, 64, False))

    def test_falls_back_when_ring_does_not_fit_or_gate_refuses(self):
        gate = LayerMajorGate(min_tokens=32768, max_tokens=262144)
        self.assertIsNone(layer_major_admission(gate=gate, budget=_budget(False), **self.ARGS))
        self.assertIsNone(layer_major_admission(gate=gate, budget=_budget(True), **{**self.ARGS, "extend_len": 1000}))
        self.assertIsNone(layer_major_admission(gate=None, budget=_budget(True), **self.ARGS))


if __name__ == "__main__":
    unittest.main()
```

Check the `_PrefillAdmission` field names with `grep -n "class _PrefillAdmission" -A8
python/sglang/srt/managers/schedule_policy.py`. If they differ from `prefix_len, extend_len, max_new_tokens,
is_chunked`, fix the test to the real names and ledger it.

- [ ] **Step 2: Run and watch it fail**

Expected: `ImportError: cannot import name 'layer_major_admission'`.

- [ ] **Step 3: Implement the pure function and the adder wiring**

In `schedule_policy.py`, next to `_PrefillAdmission`:

```python
def layer_major_admission(
    *,
    gate,
    budget,
    req_wants_prompt_logprobs: bool,
    req_wants_hidden: bool,
    prefix_len: int,
    extend_len: int,
    total_tokens: int,
    max_new_tokens: int,
    ring_tokens: int,
) -> Optional["_PrefillAdmission"]:
    """The whole uncached suffix as one extend, or None to fall back to chunked prefill."""
    if gate is None or not gate.admits(
        extend_len=extend_len, wants_prompt_logprobs=req_wants_prompt_logprobs, wants_hidden_states=req_wants_hidden
    ):
        return None
    if not budget.check_prefill_ring(
        total_tokens=total_tokens, max_new_tokens=max_new_tokens, ring_tokens=ring_tokens
    ):
        return None
    return _PrefillAdmission(prefix_len, extend_len, max_new_tokens, False)
```

In `PrefillAdder.__init__`, add the keyword parameters `layer_major_gate=None` and `layer_major_ring_tokens: int = 0`,
and store them as `self.layer_major_gate` and `self.layer_major_ring_tokens`.

At the top of `_select_prefill_admission`, after `input_tokens = ...`:

```python
        max_new_tokens = min(req.sampling_params.max_new_tokens, CLIP_MAX_NEW_TOKENS)
        layer_major = layer_major_admission(
            gate=self.layer_major_gate,
            budget=self.memory_budget,
            req_wants_prompt_logprobs=req.return_logprob and req.logprob_start_len < len(req.origin_input_ids) - 1,
            req_wants_hidden=req.return_hidden_states,
            prefix_len=prefix_len,
            extend_len=extend_len,
            total_tokens=total_tokens,
            max_new_tokens=max_new_tokens,
            ring_tokens=self.layer_major_ring_tokens,
        )
        if layer_major is not None and host_hit_length == 0 and swa_host_hit_length == 0:
            req.layer_major = True
            return layer_major
```

Reuse the existing `max_new_tokens` computation lower in the function instead of recomputing it.

Before relying on them, verify the Req attribute names with `grep -n "return_hidden_states\|logprob_start_len\|
origin_input_ids" python/sglang/srt/managers/schedule_batch.py`. A HiCache host hit falls back to chunked
(`host_hit_length == 0`) because host-loaded prefixes are admitted through their own path. Ledger that as a ruling.

- [ ] **Step 4: Req and ScheduleBatch fields; allocation; scheduler construction**

- **`Req.__init__`:** `self.layer_major: bool = False`.
- **`ScheduleBatch`:** declare `layer_major_ring_tokens: Optional[int] = None`. In `ScheduleBatch.init_new` (the
  classmethod that builds the batch from reqs), compute
  `layer_major_ring_tokens = ring if any(r.layer_major for r in reqs) else None` and pass it to the constructor.
  `ring` is `chunked_prefill_size + page_size`, read from the server args object `init_new` already has; if it has
  none, add a keyword argument and pass it from the scheduler.
- **`alloc_for_extend`:** pass `swa_ring_tokens=batch.layer_major_ring_tokens` into `alloc_paged_token_slots_extend`.
- **`Scheduler.__init__`:** after the server args are resolved,
  `self.layer_major_gate = gate_from_env(max_tokens=self.model_config.context_len)`. Where `PrefillAdder(...)` is
  constructed (`grep -n "PrefillAdder(" python/sglang/srt/managers/scheduler.py`), pass
  `layer_major_gate=self.layer_major_gate` and
  `layer_major_ring_tokens=self.server_args.chunked_prefill_size + self.page_size`. Read
  `.claude/skills/large-class-style/SKILL.md` first: the Scheduler `__init__` is frozen-style. Put the gate
  construction in the existing `init_*` step that builds other scheduling policy objects.
- **Clear the flag once the request's prefill forward returns.** In the batch-result path that runs after an extend
  (`grep -n "def process_batch_result_prefill" python/sglang/srt/managers/scheduler*/*.py`), set
  `req.layer_major = False` for each request. A retracted request that comes back is then re-gated from scratch.

- [ ] **Step 5: Run and watch it pass; then the scheduler-adjacent suites**

Expected: `2 passed`. Then run `test/registered/unit/mem_cache/test_prefill_memory_budget.py` and
`test/registered/unit/kernels` as in the Global Constraints. Expected: the same counts as BASE.

- [ ] **Step 6: Commit**

```bash
git add python/sglang/srt/managers/schedule_policy.py python/sglang/srt/managers/schedule_batch.py \
  python/sglang/srt/mem_cache/allocation.py python/sglang/srt/managers/scheduler.py \
  test/registered/unit/layer_major/test_layer_major_admission.py
git commit -m "feat(layer-major): scheduler admits a whole-suffix extend with a window ring" -m "<trailer>"
```

---

### Task 8: DSV4 adapter, pure part: chunk plan, ring positions, keep window, Engram history

**Files:**
- Create: `python/sglang/srt/models/deepseek_v4_layer_major.py` (the pure helpers now; the adapter class in Task 10)
- Test: `test/registered/unit/layer_major/test_dsv4_chunk_plan.py`

**Interfaces:**
- Produces:
  - `ChunkSpan(msgspec.Struct, frozen=True)` with `index: int`, `start: int` and `end: int` (absolute positions).
  - `chunk_spans(*, prefix_len: int, seq_len: int, chunk: int) -> list[ChunkSpan]`: chunks of the uncached suffix,
    each `chunk` tokens except a shorter final one.
  - `keep_window_start(*, seq_len: int, window: int, page: int) -> int`: the page floor of `seq_len - window`.
  - `engram_history(ids: list[int], start: int, n: int) -> list[int]`: the `n` ids before `start`, left-padded
    with 0 (the encoder_swa_replay convention).
  - `DSV4_WINDOW = 128`.

- [ ] **Step 1: Write the failing tests**

```python
import unittest

from sglang.srt.models.deepseek_v4_layer_major import ChunkSpan, chunk_spans, engram_history, keep_window_start


class TestDsv4ChunkPlan(unittest.TestCase):
    def test_spans_cover_the_suffix_with_a_short_final_chunk(self):
        spans = chunk_spans(prefix_len=0, seq_len=33000, chunk=4096)
        self.assertEqual(len(spans), 9)
        self.assertEqual(spans[0], ChunkSpan(index=0, start=0, end=4096))
        self.assertEqual(spans[-1], ChunkSpan(index=8, start=32768, end=33000))

    def test_first_chunk_with_prefix_is_not_remapped(self):
        # The ring maps suffix positions only; chunk 0's predecessor window lives in the prefix's own slots.
        spans = chunk_spans(prefix_len=512, seq_len=512 + 8192, chunk=4096)
        self.assertEqual([(s.start, s.end) for s in spans], [(512, 4608), (4608, 8704)])

    def test_keep_window_floors_to_a_page(self):
        self.assertEqual(keep_window_start(seq_len=33000, window=128, page=256), 32768)
        self.assertEqual(keep_window_start(seq_len=32868, window=128, page=256), 32512)

    def test_engram_history_pads_left(self):
        self.assertEqual(engram_history([5, 6, 7, 8], start=2, n=3), [0, 5, 6])
        self.assertEqual(engram_history([5, 6, 7, 8], start=4, n=3), [6, 7, 8])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run and watch it fail**

Expected: `ModuleNotFoundError ... deepseek_v4_layer_major`.

- [ ] **Step 3: Implement the helpers** (the start of `deepseek_v4_layer_major.py`)

```python
"""DeepSeek-V4.1 adapter for the layer-major prefill strategy (sglang.srt.layer_major): chunk plan, window-KV ring,
per-chunk metadata, one layer on one chunk, and the late-layer tail. Knows nothing about the expert quant format."""

from __future__ import annotations

import msgspec

DSV4_WINDOW = 128


class ChunkSpan(msgspec.Struct, frozen=True):
    index: int
    start: int
    end: int


def chunk_spans(*, prefix_len: int, seq_len: int, chunk: int) -> list[ChunkSpan]:
    return [
        ChunkSpan(index=i, start=start, end=min(seq_len, start + chunk))
        for i, start in enumerate(range(prefix_len, seq_len, chunk))
    ]


def keep_window_start(*, seq_len: int, window: int, page: int) -> int:
    return max(0, seq_len - window) // page * page


def engram_history(ids: list[int], start: int, n: int) -> list[int]:
    window = list(ids[max(0, start - n) : start])
    return [0] * (n - len(window)) + window
```

- [ ] **Step 4: Run and watch it pass**

Expected: `4 passed`, `EXIT=0`.

- [ ] **Step 5: Commit**

```bash
git add python/sglang/srt/models/deepseek_v4_layer_major.py test/registered/unit/layer_major/test_dsv4_chunk_plan.py
git commit -m "feat(layer-major): DSV4 chunk plan, keep window and Engram history helpers" -m "<trailer>"
```

---

### Task 9: dsv4 backend and model hooks

**Why:** the adapter needs:
- installing a kept metadata object;
- skipping candidate masks on non-final chunks;
- running the late tail from a given layer with given `hidden`, `prev_pre` and `hash_ids`;
- tail-only logits.

**Files:**
- Modify: `python/sglang/srt/layers/attention/deepseek_v4_backend.py`:
  - `install_forward_metadata`, after `init_forward_metadata` (~2448);
  - the `DSV4Metadata` field `layer_major_skip_candidates` (~1044);
  - an early return in `_publish_or_consume_candidates` (~3342).
- Modify: `python/sglang/srt/models/deepseek_v4.py`:
  - kw-only params `layer_ids`, `prev_pre_in` and `hash_ids_in` on `_forward_layers_hc_pre_from_prev` (~4427);
  - `DeepseekV4ForCausalLM.forward_late_tail` (after `forward`, ~5197).
- Test: `test/registered/unit/layer_major/test_dsv4_backend_install.py`

**Interfaces:**
- Produces:
  - **`DeepseekV4AttnBackend.install_forward_metadata(metadata, *, tail_metadata=None) -> None`:** sets
    `forward_metadata = metadata` and `tail_forward_metadata = tail_metadata`, sets `encoder_replay = False`, and
    activates the request window when one exists (as `init_forward_metadata` does).
  - **`DSV4Metadata.layer_major_skip_candidates: bool = False`:** when True, the candidate source publishes an empty
    `CandidateMasks` instead of building masks.
  - **`_forward_layers_hc_pre_from_prev(..., *, layer_ids: range | None = None, prev_pre_in=None,
    hash_ids_in=None)`:**
    - when `layer_ids` is given, the loop runs over it, not over `range(self.start_layer, self.end_layer)`;
    - `prev_pre` starts at `prev_pre_in`;
    - when `hash_ids_in` is given, hashing and `post_engram_device_lookups` are skipped.
  - **`DeepseekV4ForCausalLM.forward_late_tail(*, forward_batch, hidden_states, prev_pre, hash_ids) ->
    LogitsProcessorOutput`:**
    - runs layers `late_layer_start..end_layer` through `_forward_layers_hc_pre_from_prev`, which enters the tail
      itself;
    - then `hc_combine` + norm on the tail rows, with no scatter;
    - then logits with `LogitsMetadata.extend_seq_lens` rewritten to the tail's (the dspark precedent,
      `deepseek_v4.py:5172-5183`).

- [ ] **Step 1: Write the failing test for `install_forward_metadata`** (CPU, a backend object with no `__init__`)

```python
import unittest
from types import SimpleNamespace

from sglang.srt.layers.attention.deepseek_v4_backend import DeepseekV4AttnBackend


class TestInstallForwardMetadata(unittest.TestCase):
    def _backend(self, window=None):
        b = object.__new__(DeepseekV4AttnBackend)
        b.encoder_replay = True
        b.forward_metadata = None
        b.tail_forward_metadata = "stale"
        b.token_to_kv_pool = SimpleNamespace(request_window=window)
        return b

    def test_installs_and_clears_per_forward_state(self):
        b = self._backend()
        meta = SimpleNamespace(core_attn_metadata=SimpleNamespace(request_window_layout=None))
        b.install_forward_metadata(meta)
        self.assertIs(b.forward_metadata, meta)
        self.assertIsNone(b.tail_forward_metadata)
        self.assertFalse(b.encoder_replay)

    def test_activates_request_window_when_present(self):
        activated = []
        window = SimpleNamespace(activate=activated.append)
        b = self._backend(window=window)
        meta = SimpleNamespace(core_attn_metadata=SimpleNamespace(request_window_layout="L"))
        b.install_forward_metadata(meta, tail_metadata="T")
        self.assertEqual((activated, b.tail_forward_metadata), (["L"], "T"))


if __name__ == "__main__":
    unittest.main()
```

Check the backend class name with `grep -n "^class .*Backend" python/sglang/srt/layers/attention/deepseek_v4_backend.py`
and use the real name.

- [ ] **Step 2: Run and watch it fail**

Expected: `AttributeError: ... has no attribute 'install_forward_metadata'`.

- [ ] **Step 3: Implement the backend pieces**

After `init_forward_metadata`:

```python
    def install_forward_metadata(self, metadata, *, tail_metadata=None) -> None:
        """Install metadata built earlier by init_forward_metadata, for a layer-major prefill's chunk."""
        self.encoder_replay = False
        self.forward_metadata = metadata
        self.tail_forward_metadata = tail_metadata
        if self.token_to_kv_pool.request_window is not None:
            self.token_to_kv_pool.request_window.activate(metadata.core_attn_metadata.request_window_layout)
```

In `DSV4Metadata`, after `late_layer_tail`:

```python
    # Layer-major prefill: only the final chunk's tail reads candidate masks, so earlier chunks skip building them.
    layer_major_skip_candidates: bool = False
```

In `_publish_or_consume_candidates`, as its first statements:

```python
        if indexer.is_candidate_source and self.forward_metadata.layer_major_skip_candidates:
            self.forward_metadata.candidate_metadata = CandidateMasks(request_masks=[])
            return
```

- [ ] **Step 4: Run and watch it pass**

Expected: `2 passed`, `EXIT=0`.

- [ ] **Step 5: Implement the model hooks**

In `_forward_layers_hc_pre_from_prev`, add the keyword-only parameters `*, layer_ids: Optional[range] = None,
prev_pre_in: Optional[torch.Tensor] = None, hash_ids_in: Optional[torch.Tensor] = None`. Then:
- replace `hash_ids = None` / `if self.engram_hasher is not None:` with `hash_ids = hash_ids_in` /
  `if self.engram_hasher is not None and hash_ids_in is None:`;
- replace `prev_pre = None` with `prev_pre = prev_pre_in`;
- replace `for i in range(self.start_layer, self.end_layer):` with
  `for i in (layer_ids if layer_ids is not None else range(self.start_layer, self.end_layer)):`.

Add `forward_late_tail` to `DeepseekV4ForCausalLM`:

```python
    def forward_late_tail(self, *, forward_batch, hidden_states, prev_pre, hash_ids):
        """Layers late_layer_start.. on the final chunk's tail, for a layer-major prefill; logits of the tail only.

        The backend holds the final chunk's metadata and tail metadata (install_forward_metadata).
        """
        from sglang.kernels.ops.layernorm.mhc import hc_combine

        model = self.model
        input_ids = forward_batch.input_ids
        hidden_states, last_pre, tail = model._forward_layers_hc_pre_from_prev(
            forward_batch.positions,
            hidden_states,
            forward_batch,
            input_ids,
            input_ids,
            False,
            [],
            layer_ids=range(model.late_layer_start, model.end_layer),
            prev_pre_in=prev_pre,
            hash_ids_in=hash_ids,
        )
        pre_hc_head = hidden_states.flatten(1)
        hidden_states = model.norm(hc_combine(pre_hc_head.float(), last_pre, model.hc_mult, hidden_states.dtype))
        # Tail rows only: scattering back to the full extend would allocate [T, 5120] and [T, 20480] tensors.
        logits_metadata = LogitsMetadata.from_forward_batch(forward_batch)
        logits_metadata.extend_seq_lens = tail.extend_seq_lens
        logits_metadata.extend_seq_lens_cpu = tail.extend_seq_lens_cpu
        logits_metadata.extend_logprob_start_lens_cpu = tail.extend_seq_lens_cpu
        output = self.logits_processor(
            tail.rows(input_ids), hidden_states, self.lm_head, logits_metadata, None,
            hidden_states_before_norm=pre_hc_head,
        )
        output.hidden_states_token_indices = tail.token_indices
        return output
```

- [ ] **Step 6: Run the registered kernels suite** (a regression guard for the edited loop)

Run: `test/registered/unit/kernels` as in the Global Constraints. Expected: the same pass/skip counts as BASE.

- [ ] **Step 7: Commit**

```bash
git add python/sglang/srt/layers/attention/deepseek_v4_backend.py python/sglang/srt/models/deepseek_v4.py \
  test/registered/unit/layer_major/test_dsv4_backend_install.py
git commit -m "feat(layer-major): dsv4 metadata install, candidate skip, late-tail entry with tail-only logits" -m "<trailer>"
```

---

### Task 10: The DSV4 adapter class

**Files:**
- Modify: `python/sglang/srt/models/deepseek_v4_layer_major.py` (add the adapter class)
- Modify: `python/sglang/srt/models/deepseek_v4.py` (`DeepseekV4ForCausalLM.make_layer_major_adapter`)

**Interfaces:**
- Consumes:
  - Task 2: `FieldSpec`, `StateStore`.
  - Task 6: `map_ring_positions`, `finalize_ring`, `ring_slots`.
  - Task 8: `chunk_spans`, `keep_window_start`, `engram_history`, `DSV4_WINDOW`.
  - Task 9: `install_forward_metadata`, `layer_major_skip_candidates`, `forward_late_tail`.
- Produces:
  - `DeepseekV4LayerMajorAdapter(model_runner)`, implementing `LayerMajorModelAdapter`.
  - `DeepseekV4ForCausalLM.make_layer_major_adapter(model_runner) -> DeepseekV4LayerMajorAdapter`.

There is no unit test in this task. The adapter needs the real model and GPU, and Task 12's equivalence probe is its
test. The code must still pass `python -m py_compile`.

- [ ] **Step 1: Implement the adapter** (append to `deepseek_v4_layer_major.py`)

```python
from contextlib import nullcontext
from copy import copy
from typing import Any

import torch

from sglang.srt.layer_major.state_store import FieldSpec, StateStore


class _Pass(msgspec.Struct):
    schedule_batch: Any
    spans: list
    forward_batches: list
    hash_ids: list
    extend_full_locs: torch.Tensor
    ring: torch.Tensor
    final_tail_metadata: Any
    copy_stream: Any


class DeepseekV4LayerMajorAdapter:
    def __init__(self, model_runner):
        self.runner = model_runner
        self.causal_lm = model_runner.model
        self.model = model_runner.model.model
        self.backend = model_runner.attn_backend
        self.allocator = model_runner.token_to_kv_pool_allocator
        self.page = model_runner.page_size
        self.chunk = model_runner.server_args.chunked_prefill_size

    def field_specs(self) -> list[FieldSpec]:
        return [
            FieldSpec(name="hidden", per_token_shape=(self.model.hc_mult, self.model.hidden_size), dtype="bfloat16"),
            FieldSpec(name="prev_pre", per_token_shape=(self.model.hc_mult,), dtype="float32"),
        ]

    # --- pass setup -------------------------------------------------------------------------------------------------

    def begin_pass(self, forward_batch, schedule_batch, store: StateStore) -> _Pass:
        assert len(schedule_batch.reqs) == 1, "layer-major prefill runs one request"
        req = schedule_batch.reqs[0]
        slot = int(schedule_batch.req_pool_indices_cpu[0])
        prefix_len = int(schedule_batch.prefix_lens[0])
        seq_len = int(schedule_batch.seq_lens_cpu[0])
        full = self.runner.req_to_token_pool.req_to_token[slot, prefix_len:seq_len].to(torch.int64)
        ring_len = self.chunk + self.page
        ring = self.allocator.ring_slots(full[-ring_len:])
        spans = chunk_spans(prefix_len=prefix_len, seq_len=seq_len, chunk=self.chunk)
        handle = _Pass(schedule_batch=schedule_batch, spans=spans, forward_batches=[], hash_ids=[],
                       extend_full_locs=full, ring=ring, final_tail_metadata=None,
                       copy_stream=torch.cuda.Stream())
        for span in spans:
            fb = self._chunk_forward_batch(schedule_batch, req, slot, span, last=span is spans[-1])
            positions = torch.arange(span.start, span.end, device=full.device)
            self.allocator.map_ring_positions(full[span.start - prefix_len : span.end - prefix_len], positions, ring)
            # Builds forward_metadata (and, bounded replay being on, tail metadata) from the mapping just set.
            self.backend.init_forward_metadata(fb)
            meta = self.backend.forward_metadata
            meta.layer_major_skip_candidates = span is not spans[-1]
            if span is spans[-1]:
                handle.final_tail_metadata = self.backend.tail_forward_metadata
            store.park(span.index, meta)
            hash_ids = None
            if self.model.engram_hasher is not None:
                hash_ids = self.model.engram_hasher(fb.input_ids, fb)
            handle.hash_ids.append(hash_ids)
            embedded = self.model.embed_tokens(fb.input_ids).unsqueeze(1).repeat(1, self.model.hc_mult, 1)
            store.write("hidden", span.start - prefix_len, embedded.to("cpu"))
            handle.forward_batches.append(fb)
        return handle

    def _chunk_forward_batch(self, batch, req, slot, span, *, last):
        from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardBatch

        device = self.runner.device
        sub = copy(batch)
        sub.reqs = [req]
        sub.input_ids = torch.tensor(list(req.full_untruncated_fill_ids[span.start : span.end]), dtype=torch.int64,
                                     device=device)
        sub.prefill_input_ids_cpu = None
        sub.prefix_lens = [span.start]
        sub.extend_lens = [span.end - span.start]
        sub.extend_num_tokens = span.end - span.start
        sub.seq_lens = torch.tensor([span.end], dtype=torch.int64, device=device)
        sub.seq_lens_cpu = torch.tensor([span.end], dtype=torch.int64)
        sub.seq_lens_sum = span.end
        sub.orig_seq_lens = sub.seq_lens
        sub.out_cache_loc = self.runner.req_to_token_pool.req_to_token[slot, span.start : span.end].long()
        sub.extend_logprob_start_lens = [span.end - span.start]
        if not last:
            sub.return_logprob = False
            sub.sampling_info = None
            sub.is_prefill_only = True
        sub.engram_history = None
        if self.model.engram_hasher is not None:
            n = self.model.engram_hasher.max_ngram_size - 1
            sub.engram_history = torch.tensor([engram_history(req.full_untruncated_fill_ids, span.start, n)],
                                              dtype=torch.int32, device=device)
        return ForwardBatch.init_new(sub, self.runner, capture_hidden_mode=CaptureHiddenMode.NULL,
                                     return_hidden_states_before_norm=False)

    # --- the strategy's loop ----------------------------------------------------------------------------------------

    def layer_ids(self, handle: _Pass) -> range:
        return range(self.model.start_layer, self.model.late_layer_start)

    def num_chunks(self, handle: _Pass) -> int:
        return len(handle.spans)

    def run_layer(self, handle: _Pass, layer_id: int, chunk: int, store: StateStore) -> None:
        from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder

        span = handle.spans[chunk]
        fb = handle.forward_batches[chunk]
        offset = span.start - int(handle.schedule_batch.prefix_lens[0])
        rows = span.end - span.start
        device = self.runner.device
        hidden = torch.empty((rows, self.model.hc_mult, self.model.hidden_size), dtype=torch.bfloat16, device=device)
        store.read_into("hidden", offset, hidden, stream=None)
        prev_pre = None
        if layer_id > self.model.start_layer:
            prev_pre = torch.empty((rows, self.model.hc_mult), dtype=torch.float32, device=device)
            store.read_into("prev_pre", offset, prev_pre, stream=None)
        meta = store.unpark(span.index, torch.device(device))
        self.backend.install_forward_metadata(meta)
        layer = self.model.layers[layer_id]
        if layer.engram is not None:
            hidden = layer.engram(hidden, handle.hash_ids[chunk][:, layer.engram.layer_hash_index], fb,
                                  cp_all_tokens=False)
        with get_global_expert_distribution_recorder().with_current_layer(layer_id):
            # The chunked path's own arguments under bounded replay: its tail makes next_combined None (4549).
            hidden, prev_pre = layer.forward_hc_pre_from_prev(
                positions=fb.positions, hidden_states=hidden, input_ids=fb.input_ids, forward_batch=fb,
                input_ids_global=fb.input_ids, prev_pre=prev_pre, precomputed_attn=None, next_norm=None,
                next_input=[], combined_attn=None, normalized_attn=None, next_combined=None,
            )
        store.write_from("hidden", offset, hidden, stream=None)
        store.write_from("prev_pre", offset, prev_pre, stream=None)
        # Top-k written in place by index-source layers travels with the chunk's metadata.
        store.park(span.index, self.backend.forward_metadata)

    def finish_pass(self, handle: _Pass, store: StateStore) -> Any:
        span = handle.spans[-1]
        fb = handle.forward_batches[-1]
        offset = span.start - int(handle.schedule_batch.prefix_lens[0])
        rows = span.end - span.start
        device = self.runner.device
        hidden = torch.empty((rows, self.model.hc_mult, self.model.hidden_size), dtype=torch.bfloat16, device=device)
        prev_pre = torch.empty((rows, self.model.hc_mult), dtype=torch.float32, device=device)
        store.read_into("hidden", offset, hidden, stream=None)
        store.read_into("prev_pre", offset, prev_pre, stream=None)
        self.backend.install_forward_metadata(store.unpark(span.index, torch.device(device)),
                                              tail_metadata=handle.final_tail_metadata)
        output = self.causal_lm.forward_late_tail(forward_batch=fb, hidden_states=hidden, prev_pre=prev_pre,
                                                  hash_ids=handle.hash_ids[-1])
        self._finalize_ring(handle)
        return output

    def _finalize_ring(self, handle: _Pass) -> None:
        seq_len = handle.spans[-1].end
        prefix_len = int(handle.schedule_batch.prefix_lens[0])
        keep_from = max(prefix_len, keep_window_start(seq_len=seq_len, window=DSV4_WINDOW, page=self.page))
        self.allocator.finalize_ring(handle.extend_full_locs, extend_start=prefix_len, keep_from=keep_from,
                                     ring=handle.ring)
        req = handle.schedule_batch.reqs[0]
        req.kv.swa_evicted_seqlen = max(req.kv.swa_evicted_seqlen, keep_from)

    def release_pass(self, handle: _Pass, store: StateStore, *, failed: bool) -> None:
        store.clear_parked()
        if failed:
            # Leave no stale ring mapping for the request's release to follow.
            self._finalize_ring(handle)
```

**Notes for the implementer.** Record a ledger ruling for each deviation; the Task 12 probe is the judge.
- **Positions:** `fb.positions` comes from `ForwardBatch.init_new` for the sub-batch. Check it holds absolute
  positions `span.start..span.end-1`.
- **`req.kv.swa_evicted_seqlen`:** confirm the name with
  `grep -n "swa_evicted_seqlen" python/sglang/srt/managers/schedule_batch.py`.
- **Engram vision branch:** if the model has `vision_n_layers > 0`, copy the `torch.where` from
  `deepseek_v4.py:4513-4521` after the Engram call.
- **Engram lookups:** `post_engram_device_lookups` is not called. Prefill lookups are eager (`engram.py:971-1011`).
  If Task 12 shows an Engram-layer mismatch, post them per chunk right before its first Engram layer.
- **Copy streams:** state copies run on the current stream in phase 1. The unused `copy_stream` field is kept for
  phase 2's overlap, so there are no side streams to join yet.

- [ ] **Step 2: Add the model hook**

In `DeepseekV4ForCausalLM`:

```python
    def make_layer_major_adapter(self, model_runner):
        from sglang.srt.models.deepseek_v4_layer_major import DeepseekV4LayerMajorAdapter

        return DeepseekV4LayerMajorAdapter(model_runner)
```

- [ ] **Step 3: Compile check**

Run: `python -m py_compile python/sglang/srt/models/deepseek_v4_layer_major.py python/sglang/srt/models/deepseek_v4.py`.
Expected: no output, exit 0.

- [ ] **Step 4: Commit**

```bash
git add python/sglang/srt/models/deepseek_v4_layer_major.py python/sglang/srt/models/deepseek_v4.py
git commit -m "feat(layer-major): DeepSeek-V4.1 adapter" -m "<trailer>"
```

---

### Task 11: Worker seam and strategy entry point

**Files:**
- Create: `python/sglang/srt/layer_major/worker_entry.py`
- Modify: `python/sglang/srt/managers/tp_worker.py` (`forward_batch_generation`, the last-rank branch at ~659)
- Test: `test/registered/unit/layer_major/test_worker_entry.py`

**Interfaces:**
- Consumes: `run_pass` and `NullResidency` (Task 3); `StateStore` (Task 2); `make_layer_major_adapter` (Task 10).
- Produces:
  - `LayerMajorRuntime(adapter, store, residency)`, built lazily once per worker by
    `layer_major_runtime(model_runner) -> LayerMajorRuntime | None`. It returns None when the env threshold is 0.
    It raises when the model has no `make_layer_major_adapter`.
  - `run_layer_major_prefill(runtime, model_runner, schedule_batch, forward_batch) -> ModelRunnerOutput`. It runs
    the pass inside `forward_context(ForwardContext(attn_backend=...))` and the expert-distribution recorder's
    `with_forward_pass`, and returns `ModelRunnerOutput(logits_output=..., can_run_graph=False)`.

- [ ] **Step 1: Write the failing tests**

```python
import unittest
from types import SimpleNamespace
from unittest import mock

from sglang.srt.layer_major import worker_entry


class TestWorkerEntry(unittest.TestCase):
    def test_runtime_off_when_threshold_is_zero(self):
        with mock.patch.object(worker_entry.envs.SGLANG_LAYER_MAJOR_PREFILL_MIN_TOKENS, "get", return_value=0):
            self.assertIsNone(worker_entry.layer_major_runtime(SimpleNamespace(model=object())))

    def test_runtime_refuses_a_model_without_an_adapter(self):
        with mock.patch.object(worker_entry.envs.SGLANG_LAYER_MAJOR_PREFILL_MIN_TOKENS, "get", return_value=64):
            with self.assertRaisesRegex(ValueError, "no layer-major adapter"):
                worker_entry.layer_major_runtime(SimpleNamespace(model=object()))

    def test_seam_propagates_pass_failure_after_release(self):
        runtime = SimpleNamespace(adapter="a", store="s", residency="r")
        runner = SimpleNamespace(attn_backend="backend", forward_pass_id=0)
        with mock.patch.object(worker_entry, "run_pass", side_effect=RuntimeError("pass failed")):
            with self.assertRaisesRegex(RuntimeError, "pass failed"):
                worker_entry.run_layer_major_prefill(runtime, runner, "sb", SimpleNamespace())


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run and watch it fail**

Expected: `ModuleNotFoundError ... layer_major.worker_entry`.

- [ ] **Step 3: Implement `worker_entry.py`**

```python
"""Where a TP worker hands a layer-major batch to the strategy, in place of model_runner.forward."""

from __future__ import annotations

from typing import Any

import msgspec

from sglang.srt.environ import envs
from sglang.srt.layer_major.driver import run_pass
from sglang.srt.layer_major.protocols import NullResidency
from sglang.srt.layer_major.state_store import StateStore


class LayerMajorRuntime(msgspec.Struct):
    adapter: Any
    store: StateStore
    residency: Any


def layer_major_runtime(model_runner) -> LayerMajorRuntime | None:
    if envs.SGLANG_LAYER_MAJOR_PREFILL_MIN_TOKENS.get() <= 0:
        return None
    make = getattr(type(model_runner.model), "make_layer_major_adapter", None)
    if make is None:
        raise ValueError(f"{type(model_runner.model).__name__} has no layer-major adapter")
    adapter = make(model_runner.model, model_runner)
    store = StateStore(
        adapter.field_specs(),
        model_runner.model_config.context_len,
        numa_node=envs.SGLANG_LAYER_MAJOR_STATE_NUMA_NODE.get(),
        pin=True,
    )
    return LayerMajorRuntime(adapter=adapter, store=store, residency=NullResidency())


def run_layer_major_prefill(runtime: LayerMajorRuntime, model_runner, schedule_batch, forward_batch):
    from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
    from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
    from sglang.srt.model_executor.model_runner import ModelRunnerOutput

    with forward_context(ForwardContext(attn_backend=model_runner.attn_backend)):
        with get_global_expert_distribution_recorder().with_forward_pass(model_runner.forward_pass_id, forward_batch):
            logits_output = run_pass(runtime.adapter, runtime.residency, forward_batch, schedule_batch, runtime.store)
    return ModelRunnerOutput(logits_output=logits_output, can_run_graph=False)
```

(`getattr(type(...), "make_layer_major_adapter", None)` is a capability check across model classes, not defensive
field access.) Check the import paths of `forward_context`, `ForwardContext` and `ModelRunnerOutput` with
`grep -rn "^def forward_context\|^class ForwardContext\|^class ModelRunnerOutput" python/sglang/srt` and fix them.

- [ ] **Step 4: Wire the seam in `tp_worker.py`**

In `TpModelWorker.__init__`, after `self.model_runner` exists, add
`self._layer_major = layer_major_runtime(self.model_runner)` (import from `sglang.srt.layer_major.worker_entry`).

In `forward_batch_generation`, replace

```python
            out = self.model_runner.forward(
                forward_batch,
                pp_proxy_tensors=pp_proxy_tensors,
            )
```

in the last-rank branch with:

```python
            if batch is not None and batch.layer_major_ring_tokens is not None:
                out = run_layer_major_prefill(self._layer_major, self.model_runner, batch, forward_batch)
            else:
                out = self.model_runner.forward(
                    forward_batch,
                    pp_proxy_tensors=pp_proxy_tensors,
                )
```

- [ ] **Step 5: Run and watch it pass**

Expected: `3 passed`, `EXIT=0`.

- [ ] **Step 6: Commit**

```bash
git add python/sglang/srt/layer_major/worker_entry.py python/sglang/srt/managers/tp_worker.py \
  test/registered/unit/layer_major/test_worker_entry.py
git commit -m "feat(layer-major): worker seam and strategy entry point" -m "<trailer>"
```

---

### Task 12: GPU equivalence against chunked prefill (divix01)

**Files:**
- Create: `analysis/dsv41-drive/layer-major/equiv.py` (the client: prompts, cases, comparison)
- Create: `analysis/dsv41-drive/layer-major/drive_equiv.sh` (two servers, chunked then layer-major)
- Modify: `DSV41_REFERENCE.md` (add §27.18 with phase 0 and phase 1 results)

**Interfaces:** none new. This task is the adapter's test.

**Cases:** all greedy, 64 new tokens, `ignore_eos`.
- Plain prompts of 8,192, 16,384, 32,768, 33,000 (final chunk 232) and 32,868 (final chunk 100) tokens.
- A `prefix` case: a 1,024-token prompt, then the same 1,024 plus 32,768 more (a radix hit, then a layer-major
  suffix).
- An `after` case: a 256-token prompt right after the 33,000 case.

**Pass criteria:**
- Byte-identical text and output token ids between the two servers for every case.
- On the layer-major server, `server.log` shows `layer-major` passes for the long cases only. Add a
  `logger.info("layer-major prefill: %d tokens in %d chunks", ...)` in `run_layer_major_prefill`.
- Zero `memory allocation failed with OOM` lines and no watchdog timeout.

- [ ] **Step 1: Write `equiv.py`**

```python
"""Layer-major vs chunked prefill: greedy 64-token outputs must be identical (plan 2026-09-27, Task 12).

Usage: equiv.py run --port P --model M --text FILE --out OUT.jsonl
       equiv.py compare A.jsonl B.jsonl
"""

import argparse
import json
import sys
import urllib.request

from transformers import AutoTokenizer

LENGTHS = [8192, 16384, 32768, 33000, 32868]


def _generate(port, ids):
    body = {"input_ids": ids, "sampling_params": {"max_new_tokens": 64, "temperature": 0, "ignore_eos": True}}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/generate", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    d = json.load(urllib.request.urlopen(req, timeout=7200))
    return {"text": d["text"], "ids": d.get("output_ids"), "meta": d.get("meta_info", {})}


def run(a):
    tok = AutoTokenizer.from_pretrained(a.model)
    ids = tok(open(a.text).read(), add_special_tokens=False)["input_ids"]
    while len(ids) < 70000:
        ids = ids + ids
    cases = [(f"len{n}", ids[:n]) for n in LENGTHS]
    cases.append(("prefix-warm", ids[5000:6024]))
    cases.append(("prefix", ids[5000:6024] + ids[:32768]))
    cases.append(("after", ids[100:356]))
    with open(a.out, "w") as f:
        for name, prompt in cases:
            r = _generate(a.port, prompt)
            f.write(json.dumps({"case": name, **r}) + "\n")
            f.flush()
            print(name, len(prompt), repr(r["text"][:60]), flush=True)


def compare(path_a, path_b):
    a = {json.loads(l)["case"]: json.loads(l) for l in open(path_a)}
    b = {json.loads(l)["case"]: json.loads(l) for l in open(path_b)}
    bad = 0
    for case in a:
        same = a[case]["text"] == b[case]["text"] and a[case]["ids"] == b[case]["ids"]
        bad += not same
        print(f"{case:12s} {'IDENTICAL' if same else 'DIFFERENT'}")
        if not same:
            first = next((i for i, (x, y) in enumerate(zip(a[case]["text"], b[case]["text"])) if x != y), None)
            print(f"   first differing character: {first}")
    return 1 if bad else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--port", type=int, required=True)
    r.add_argument("--model", required=True)
    r.add_argument("--text", required=True)
    r.add_argument("--out", required=True)
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    args = ap.parse_args()
    sys.exit(run(args) or 0 if args.cmd == "run" else compare(args.a, args.b))
```

- [ ] **Step 2: Write `drive_equiv.sh`**

Copy `analysis/dsv41-drive/prefill-chunk/chunk_smoke.sh`'s structure: locks, the environment and argv from
`arm_env`, the health loop, and `stop_server`. Replace its warm-up and long-prompt calls with
`equiv.py run --out $OUT/<arm>.jsonl`. It takes `<arm> <worktree> <min_tokens>` and exports
`SGLANG_LAYER_MAJOR_PREFILL_MIN_TOKENS=<min_tokens>` through the `arm_env` override dict. Arms:
- `chunked`, min_tokens 0;
- `layer-major`, min_tokens 32768 (the 8k and 16k cases then run chunked);
- `layer-major-8k`, min_tokens 8192, so every length case runs layer-major.

Output goes to `/mnt/nvme1/layer-major/equiv/`.

- [ ] **Step 3: Commit, push, and run the three arms on divix01, in order**

```bash
cd /mnt/nvme1/layer-major && setsid nohup bash -c '
  W=/data/models/slang/nvfp4-work/wt-layer-major
  for arm in "chunked 0" "layer-major 32768" "layer-major-8k 8192"; do
    set -- $arm; bash $W/analysis/dsv41-drive/layer-major/drive_equiv.sh $1 $W $2
  done; echo EQUIV DONE' > equiv.log 2>&1 < /dev/null &
```

Before starting, update the worktree to the pushed head with `git -C <wt> checkout --detach origin/cc/layer-major-prefill`.

- [ ] **Step 4: Compare**

```bash
cd /mnt/nvme1/layer-major/equiv
python equiv.py compare chunked.jsonl layer-major.jsonl; echo "EXIT=$?"
python equiv.py compare chunked.jsonl layer-major-8k.jsonl; echo "EXIT=$?"
grep -c "memory allocation failed with OOM" */server.log
grep -h "layer-major prefill:" layer-major*/server.log
```

Expected: every case IDENTICAL, `EXIT=0` twice, zero OOM retries, and layer-major lines for the expected cases only.

- [ ] **Step 5: If any case differs**

Use superpowers:systematic-debugging, with a per-layer dump probe modelled on
`analysis/dsv41-drive/hicache/swa_window_probe.py`: dump `hidden` after each layer for chunk k in both modes and find
the first differing layer. Known suspects, in order:
1. Engram lookups (Task 10 notes).
2. The candidate skip on the wrong chunk.
3. Ring mapping on the first chunk after a prefix.
4. `fb.positions`.
5. Top-k not carried: `store.park` after `run_layer`.

Do not mark this task complete while any case differs.

- [ ] **Step 6: Write §27.18 in `DSV41_REFERENCE.md`**

Record:
- phase 0's numbers (TTFT, chunk times at 128k, peak VRAM, indexer path, node 1 free);
- the equivalence table (case, chunked vs layer-major, IDENTICAL or first difference);
- layer-major TTFT against chunked at 32k and 33k (phase 1 is not expected to be faster: experts still stream per
  chunk);
- the commands and paths.

- [ ] **Step 7: Commit**

```bash
git add analysis/dsv41-drive/layer-major/equiv.py analysis/dsv41-drive/layer-major/drive_equiv.sh DSV41_REFERENCE.md \
  python/sglang/srt/layer_major/worker_entry.py
git commit -m "test(layer-major): chunked vs layer-major equivalence on DSV4.1, phase 0-1 results (27.18)" -m "<trailer>"
```

---

## After the last task

- Run `test/registered/unit/kernels` and `test/registered/unit/layer_major` as in the Global Constraints, and record
  both commands with their counts.
- The phase 2 plan (EXL3 quant adapter, expert residency, hot-cache borrow and restore) is written next. It builds on
  §27.18's numbers.
