# MoE Expert Prediction Capture Mode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Record full, unsampled per-token routing data from a live NVFP4 server, so that LLaPor (next layer) and APEX (same layer) can both be trained and replayed from one dataset.

**Architecture:** Capture reuses the shadow framework's taps.
- **Decode forwards** (CUDA graph replays) already fill fixed tap buffers. The runtime copies those rows into pinned host frames.
- **Prefill forwards** are eager and larger than those buffers. The taps "spill" them straight into a pinned frame during the forward.
- **The writer thread** waits on each frame's CUDA event, then:
  - drops prefill rows whose exact token prefix was already captured (production re-prefills whole conversations),
  - appends rows plus identity and hot-cache residency to safetensors shards on `/mnt/nvme2`.
- **Capture never samples.** Anything that would lose rows (byte cap, oversized forward, write error) stops capture permanently and records why.

**Tech Stack:** Python 3.13, torch 2.13 (pinned memory, `torch.cuda.Event`), `safetensors.torch`, `msgspec`, SGLang `envs`.

**Spec:** `docs/superpowers/specs/2026-09-14-moe-expert-prediction-framework-design.md` (parent framework), plus the Design section below. Research inputs:
- `crypto/trading-framework/mechanism_hunter/docs/superpowers/plans/2026-09-14-llapor-gpu-only.md` (identity: request/session, sequence, forward, branch, position)
- `...-apex-gpu-only.md` (same identity; ideal-predictor replay before training)

## Global Constraints

- Work on branch `master` in `/home/dimitri/data/divix/sglang-nvfp4`. No worktrees on the laptop; nothing in `/home/dimitri/data/divix/crypto`.
- **Commits:**
  - Stage by name; commit with `git commit -m ... -- <paths>`.
  - Never stage `.omc/`, `.superpowers/`, or `docs/superpowers/experiments/nvfp4-expert-offload-experiment-log.md`.
  - Trailer: `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>` and `Claude-Session: https://claude.ai/code/session_01NurjMVe2nS3PGBqBr8M8MZ`.
- **`python/sglang/srt/model_executor/model_runner.py` is frozen.** Only orchestration edits (gate, delegate, pass arguments).
- **Code conventions:**
  - `msgspec.Struct`, never `@dataclass`.
  - No defensive `getattr`/`hasattr` in new code.
  - Env vars only through `envs` descriptors in `python/sglang/srt/environ.py`.
  - Comments only for non-obvious facts, one or two lines, ASCII.
- **Off by default.** With `SGLANG_MOE_EXPERT_PREDICTOR` and `SGLANG_MOE_EXPERT_PREDICTOR_CAPTURE_DIR` both empty, no runtime, hooks, buffers, frames, or threads exist.
- **No host synchronization on the serving thread.** Device-to-host copies are `non_blocking` into pinned memory, and only the writer thread waits on CUDA events. The one intended stall is back-pressure: when every frame is still held by the writer, the next forward waits for a free frame.
- **Never thin.** Every PREFILL and DECODE forward is either recorded in full or capture stops permanently with a reason in `capture-stopped.json`.
- **Scope:** single GPU and no speculative decoding (`tokens_per_request == 1`). Both are rejected at startup when capture is on.
- **Testing on divix01, not the laptop:**
  - Python: `/data/models/slang/.venv/bin/python`.
  - `PYTHONPATH=/data/models/slang/nvfp4-work/flashinfer-0.6.18-cu130-overlay:$PWD/python:/data/models/slang/nvfp4-work/uring-test-deps`.
  - CPU tests: add `CUDA_VISIBLE_DEVICES=""`.
- **Experiment worktree:** `/data/models/slang/nvfp4-work/cc-expert-prediction/worktree`. Never modify the serving worktree `/data/models/slang/nvfp4-work/main-port-probe-7bc4eb` or `run-nvfp4-e16c-public.sh`.
- **MVP:** tests are committed together with their implementation (no red commits). Reviews stay light.

---

## Design

### Why capture everything

- **The replay needs the exact forward sequence.** Hot-cache residency and transfer costs depend on every earlier token's routes. The APEX plan requires that replay before any training.
- **Both research plans require unbroken identity:** request, sequence, forward, branch, position.
- **Selection unit:** whole sessions, chosen by what traffic is sent to the capture server. There is no per-token or per-forward sampling.

### Row schema (one row per captured token; all layers share the row)

| Tensor key | dtype | Shape | Meaning |
|---|---|---|---|
| `row.forward_index` | int64 | [R] | Index of the forward in this capture run |
| `row.request_index` | int32 | [R] | Index into the shard metadata `request_ids` JSON list |
| `row.position` | int64 | [R] | Token position in its request |
| `row.token_id` | int64 | [R] | Input token at that position |
| `row.prefix_hash` | int64 | [R] | 64-bit chained hash of tokens `0..position`; 0 means the earlier tokens were not observed |
| `layer.{L}.pre_mixer` | model dtype | [R, hidden] | APEX input (tensor the attention/GDN mixer consumes) |
| `layer.{L}.router_input` | model dtype | [R, hidden] | LLaPor input; router logits are recomputed offline from this and the gate weight |
| `layer.{L}.topk_ids` | int16 | [R, top_k] | Routed experts, rank order |
| `layer.{L}.topk_weights` | float32 | [R, top_k] | Routing weights |
| `forward.index` | int64 | [F] | Every captured forward, consecutive |
| `forward.kind` | uint8 | [F] | 0 prefill, 1 decode |
| `forward.rows` | int64 | [F] | Rows the forward ran, before dedupe |
| `forward.expert_to_slot` | int16 | [F, layers, experts] | Hot-cache residency when the forward ran; -1 not resident |

- **Files:** one `capture.json` header, `shard-NNNNNN.safetensors` files, `manifest.jsonl` with one line per shard, and `capture-stopped.json` only if capture stopped.
- **Size:** about 494 KB per row for the Qwen3.8 NVFP4 model (48 layers x (2 x 2560 x 2 B + 60 B)).

### Prefix dedupe

- **Why:** production runs `--disable-radix-cache`, so turn N re-prefills turns 1..N-1.
- **How:** the writer chains `blake2b` over each request's tokens. A prefill row whose chain hash was already seen (in any earlier prefill or decode row) is dropped. Decode rows are always kept.
- **Linking sessions:** hashes are stored, so offline tools can link turns: a turn-2 row with the same `prefix_hash` as a turn-1 row has the same history.
- **With radix cache on:** a request that starts at position > 0 has unknown earlier tokens. Its rows get hash 0 and are never deduped. The radix cache already skips the re-prefill, so little is duplicated.

### Data flow per forward

1. **During the forward (TopK hook and pre-mixer adapter):**
   - rows <= `max_rows`: `FeatureStore.write` copies into the fixed buffers, as in shadow mode (graph-safe).
   - rows > `max_rows` (eager prefill only): `FeatureStore.write` calls `store.spill`. `RouteCapture._spill` takes a frame from the pool (blocking if all are busy) and enqueues `non_blocking` copies into pinned memory.
2. **`ExpertPredictionRuntime.on_forward_end`:**
   - `RouteCapture.on_forward_end` classifies the forward. Only PREFILL and DECODE are recorded.
   - It checks rows against per-request rows and capacity. A mismatch stops capture.
   - If nothing spilled, it copies the store views into a frame. It then copies positions, token ids and `expert_to_slot`, records a CUDA event, and submits `(ForwardRecord, frame)` to the writer.
   - Shadow scoring then runs unchanged.
3. **Writer thread:**
   - waits on the event;
   - hashes and dedupes;
   - appends kept rows (copies, so the frame can be released) and releases the frame;
   - flushes a shard at `SHARD_ROWS`, after 5 s idle, and on close.
   - A flush that would exceed the byte cap discards that shard and stops capture.

### Configuration (environ.py, next to the other `SGLANG_MOE_EXPERT_PREDICTOR_*`)

| Variable | Default | Meaning |
|---|---|---|
| `SGLANG_MOE_EXPERT_PREDICTOR_CAPTURE_DIR` | `""` | Non-empty enables capture into this new or empty directory |
| `SGLANG_MOE_EXPERT_PREDICTOR_CAPTURE_MAX_GB` | 500 | Shard bytes allowed before capture stops |
| `SGLANG_MOE_EXPERT_PREDICTOR_CAPTURE_SHARD_ROWS` | 4096 | Kept rows per shard |
| `SGLANG_MOE_EXPERT_PREDICTOR_CAPTURE_FRAMES` | 2 | Pinned frames; each holds one forward of up to `max(max_rows, chunked_prefill_size)` rows (about 2 GB at 4096 rows) |

### Known approximations

- **Kill loss:** a killed server loses at most the unflushed rows (under 5 s of idle, or up to one shard). Stop load, wait 10 s, then stop the server.
- **Residency race:** `expert_to_slot` is snapshotted on the current stream before the hot cache observer runs. A hot-cache update that runs on another stream could race the snapshot, the same approximation shadow scoring makes.
- **Speculative decoding (VERIFY/DRAFT) and multi-GPU are out of scope** and rejected at startup.

---

## File Structure

| File | Responsibility |
|---|---|
| `python/sglang/srt/layers/moe/expert_prediction/capture_schema.py` (new) | Tensor names, `CaptureKind`, `ForwardRecord`, disk dtypes, per-request row counts |
| `python/sglang/srt/layers/moe/expert_prediction/prefix_hash.py` (new) | `PrefixHasher`, `SeenPrefixes` |
| `python/sglang/srt/layers/moe/expert_prediction/capture_frames.py` (new) | `CaptureFrame`, `FramePool`, `copy_rows` |
| `python/sglang/srt/layers/moe/expert_prediction/capture_writer.py` (new) | `ShardWriter` thread, `PendingForward`, file names |
| `python/sglang/srt/layers/moe/expert_prediction/capture_reader.py` (new) | `load_shard`, `read_manifest`, `check_capture`, CLI |
| `python/sglang/srt/layers/moe/expert_prediction/capture.py` (new) | `CaptureSettings`, `RouteCapture` (spill sink + per-forward staging) |
| `python/sglang/srt/layers/moe/expert_prediction/feature_store.py` (modify) | `spill` sink for batches above `max_rows` |
| `python/sglang/srt/layers/moe/expert_prediction/runtime.py` (modify) | Build and call `RouteCapture`; `from_env` capture settings |
| `python/sglang/srt/environ.py` (modify) | Four capture env vars |
| `python/sglang/srt/model_executor/model_runner.py` (modify, orchestration) | Gate on capture dir; pass `max_prefill_rows` |
| `scripts/expert_prediction/run-shadow-server.sh` (modify) | Optional 4th arg `capture` |
| `scripts/expert_prediction/capture-two-turn-smoke.py` (new) | Two-turn chat driver for the smoke |
| `scripts/expert_prediction/check-capture-gate-topk.py` (new) | Offline check that gate weights reproduce captured top-k |
| `test/registered/unit/layers/moe/test_expert_prediction_capture_schema.py` (new) | CPU |
| `test/registered/unit/layers/moe/test_expert_prediction_capture_writer.py` (new) | CPU |
| `test/registered/unit/layers/moe/test_expert_prediction_capture.py` (new) | CPU, runtime integration |
| `test/registered/unit/layers/moe/test_expert_prediction_capture_graph.py` (new) | CUDA graph replay + eager prefill, sync-free |

---

### Task 1: Row schema and prefix hashing

**Files:**
- Create: `python/sglang/srt/layers/moe/expert_prediction/capture_schema.py`
- Create: `python/sglang/srt/layers/moe/expert_prediction/prefix_hash.py`
- Test: `test/registered/unit/layers/moe/test_expert_prediction_capture_schema.py`

**Interfaces:**
- Consumes: `RouteFeature` from `contracts.py`.
- Produces:
  - `CAPTURE_FEATURES`, `CaptureKind`, `ForwardRecord(forward_index, kind, rids, rows_per_request)`.
  - `feature_key(layer_id, feature) -> str`, `disk_dtype(feature, hidden_dtype)`.
  - `rows_per_request(*, is_extend, batch_size, extend_seq_lens) -> tuple[int, ...]`.
  - Constants `SCHEMA_VERSION`, `ROW_*`, `FORWARD_*`.
  - `PrefixHasher(*, max_requests).hash_rows(*, rid, positions, token_ids) -> list[int]`.
  - `SeenPrefixes().keep_mask(*, hashes, is_prefill) -> list[bool]`.

- [x] **Step 1: Write `capture_schema.py`**

```python
"""Row identity and tensor names for captured expert-prediction training data."""

from __future__ import annotations

from enum import IntEnum
from typing import Sequence

import msgspec
import torch

from sglang.srt.layers.moe.expert_prediction.contracts import RouteFeature

SCHEMA_VERSION = 1

CAPTURE_FEATURES = (
    RouteFeature.PRE_MIXER,
    RouteFeature.ROUTER_INPUT,
    RouteFeature.TOPK_IDS,
    RouteFeature.TOPK_WEIGHTS,
)

ROW_FORWARD = "row.forward_index"
ROW_REQUEST = "row.request_index"
ROW_POSITION = "row.position"
ROW_TOKEN = "row.token_id"
# 0 means the request's earlier tokens were not observed.
ROW_PREFIX_HASH = "row.prefix_hash"
FORWARD_INDEX = "forward.index"
FORWARD_KIND = "forward.kind"
FORWARD_ROWS = "forward.rows"
# int16 [forwards, layers, experts]; -1 marks a non-resident expert.
FORWARD_RESIDENCY = "forward.expert_to_slot"


class CaptureKind(IntEnum):
    PREFILL = 0
    DECODE = 1


class ForwardRecord(msgspec.Struct, frozen=True):
    forward_index: int
    kind: CaptureKind
    rids: tuple[str, ...]
    rows_per_request: tuple[int, ...]


def feature_key(layer_id: int, feature: RouteFeature) -> str:
    return f"layer.{layer_id}.{feature.value}"


def disk_dtype(feature: RouteFeature, hidden_dtype: torch.dtype) -> torch.dtype:
    if feature is RouteFeature.TOPK_IDS:
        return torch.int16
    if feature is RouteFeature.TOPK_WEIGHTS:
        return torch.float32
    return hidden_dtype


def rows_per_request(
    *, is_extend: bool, batch_size: int, extend_seq_lens: Sequence[int] | None
) -> tuple[int, ...]:
    if is_extend:
        return tuple(int(rows) for rows in extend_seq_lens)
    return (1,) * batch_size
```

- [x] **Step 2: Write `prefix_hash.py`**

```python
"""Chain token hashes per request so identical prefixes share keys across requests."""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from typing import Sequence

UNKNOWN_PREFIX = 0
_BROKEN_CHAIN = -1


def _step(previous: int, token_id: int) -> int:
    digest = hashlib.blake2b(
        previous.to_bytes(8, "little", signed=True)
        + token_id.to_bytes(8, "little", signed=True),
        digest_size=8,
    ).digest()
    return int.from_bytes(digest, "little", signed=True) or 1


class PrefixHasher:
    """``hash_rows`` returns the chain hash of tokens ``0..position`` for each row."""

    def __init__(self, *, max_requests: int) -> None:
        self._max_requests = max_requests
        # rid -> (next expected position, chain hash so far)
        self._state: OrderedDict[str, tuple[int, int]] = OrderedDict()

    def hash_rows(
        self, *, rid: str, positions: Sequence[int], token_ids: Sequence[int]
    ) -> list[int]:
        next_position, value = self._state.pop(rid, (0, UNKNOWN_PREFIX))
        hashes = []
        for position, token_id in zip(positions, token_ids):
            if next_position == _BROKEN_CHAIN or position != next_position:
                next_position = _BROKEN_CHAIN
                hashes.append(UNKNOWN_PREFIX)
                continue
            value = _step(value, token_id)
            hashes.append(value)
            next_position = position + 1
        self._state[rid] = (next_position, value)
        while len(self._state) > self._max_requests:
            self._state.popitem(last=False)
        return hashes


class SeenPrefixes:
    """Drop prefill rows whose exact token prefix was already captured."""

    def __init__(self) -> None:
        self._seen: set[int] = set()

    def keep_mask(self, *, hashes: Sequence[int], is_prefill: bool) -> list[bool]:
        keep = []
        for value in hashes:
            known = value != UNKNOWN_PREFIX
            keep.append(not (is_prefill and known and value in self._seen))
            if known:
                self._seen.add(value)
        return keep
```

- [x] **Step 3: Write the tests**

```python
import unittest

from sglang.srt.layers.moe.expert_prediction.capture_schema import rows_per_request
from sglang.srt.layers.moe.expert_prediction.prefix_hash import PrefixHasher, SeenPrefixes


class TestPrefixHasher(unittest.TestCase):
    def test_same_prefix_same_hash_across_requests(self):
        hasher = PrefixHasher(max_requests=8)
        first = hasher.hash_rows(rid="a", positions=[0, 1, 2], token_ids=[5, 6, 7])
        second = hasher.hash_rows(rid="b", positions=[0, 1, 2], token_ids=[5, 6, 9])
        self.assertEqual(first[:2], second[:2])
        self.assertNotEqual(first[2], second[2])
        self.assertNotIn(0, first + second)

    def test_chunks_continue_the_chain(self):
        whole = PrefixHasher(max_requests=8).hash_rows(
            rid="a", positions=[0, 1, 2, 3], token_ids=[1, 2, 3, 4]
        )
        chunked = PrefixHasher(max_requests=8)
        parts = chunked.hash_rows(rid="a", positions=[0, 1], token_ids=[1, 2])
        parts += chunked.hash_rows(rid="a", positions=[2, 3], token_ids=[3, 4])
        self.assertEqual(whole, parts)

    def test_gap_marks_the_rest_of_the_request_unknown(self):
        hasher = PrefixHasher(max_requests=8)
        hasher.hash_rows(rid="a", positions=[0, 1], token_ids=[1, 2])
        self.assertEqual(hasher.hash_rows(rid="a", positions=[5, 6], token_ids=[3, 4]), [0, 0])
        self.assertEqual(hasher.hash_rows(rid="a", positions=[7], token_ids=[5]), [0])

    def test_request_starting_after_zero_is_unknown(self):
        hasher = PrefixHasher(max_requests=8)
        self.assertEqual(hasher.hash_rows(rid="a", positions=[64, 65], token_ids=[1, 2]), [0, 0])

    def test_evicts_oldest_request(self):
        hasher = PrefixHasher(max_requests=1)
        hasher.hash_rows(rid="a", positions=[0], token_ids=[1])
        hasher.hash_rows(rid="b", positions=[0], token_ids=[1])
        self.assertEqual(hasher.hash_rows(rid="a", positions=[1], token_ids=[2]), [0])


class TestSeenPrefixes(unittest.TestCase):
    def test_prefill_duplicates_dropped_decode_and_unknown_kept(self):
        seen = SeenPrefixes()
        self.assertEqual(seen.keep_mask(hashes=[11, 12], is_prefill=True), [True, True])
        self.assertEqual(
            seen.keep_mask(hashes=[11, 12, 13], is_prefill=True), [False, False, True]
        )
        self.assertEqual(seen.keep_mask(hashes=[13], is_prefill=False), [True])
        self.assertEqual(seen.keep_mask(hashes=[0, 0], is_prefill=True), [True, True])


class TestRowsPerRequest(unittest.TestCase):
    def test_extend_uses_extend_lengths(self):
        self.assertEqual(
            rows_per_request(is_extend=True, batch_size=2, extend_seq_lens=[3, 5]), (3, 5)
        )

    def test_decode_is_one_row_per_request(self):
        self.assertEqual(
            rows_per_request(is_extend=False, batch_size=3, extend_seq_lens=None), (1, 1, 1)
        )


if __name__ == "__main__":
    unittest.main()
```

- [x] **Step 4: Commit, push, sync the experiment worktree, run the tests on divix01**

```bash
git add python/sglang/srt/layers/moe/expert_prediction/capture_schema.py \
  python/sglang/srt/layers/moe/expert_prediction/prefix_hash.py \
  test/registered/unit/layers/moe/test_expert_prediction_capture_schema.py
git commit -m "feat(moe): add expert capture row schema and prefix hashing" -- \
  python/sglang/srt/layers/moe/expert_prediction/capture_schema.py \
  python/sglang/srt/layers/moe/expert_prediction/prefix_hash.py \
  test/registered/unit/layers/moe/test_expert_prediction_capture_schema.py
git push origin master
ssh -n divix01 'cd /data/models/slang/nvfp4-work/cc-expert-prediction/worktree && git fetch -q origin master && git checkout -q --detach FETCH_HEAD && git log -1 --oneline'
ssh -n divix01 'cd /data/models/slang/nvfp4-work/cc-expert-prediction/worktree && \
  CUDA_VISIBLE_DEVICES="" PYTHONPATH=/data/models/slang/nvfp4-work/flashinfer-0.6.18-cu130-overlay:$PWD/python:/data/models/slang/nvfp4-work/uring-test-deps \
  timeout 900 /data/models/slang/.venv/bin/python -m pytest -q -p no:cacheprovider -rfEs \
  test/registered/unit/layers/moe/test_expert_prediction_capture_schema.py; echo EXIT=$?'
```

Expected: `8 passed`, `EXIT=0`. (Commit messages end with the two trailer lines from Global Constraints.)

---

### Task 2: Pinned frames, store spill, shard writer, reader

**Files:**
- Create: `python/sglang/srt/layers/moe/expert_prediction/capture_frames.py`
- Create: `python/sglang/srt/layers/moe/expert_prediction/capture_writer.py`
- Create: `python/sglang/srt/layers/moe/expert_prediction/capture_reader.py`
- Modify: `python/sglang/srt/layers/moe/expert_prediction/feature_store.py` (`__init__`, `write`)
- Test: `test/registered/unit/layers/moe/test_expert_prediction_capture_writer.py`

**Interfaces:**
- Consumes (from Task 1): `CAPTURE_FEATURES`, `CaptureKind`, `ForwardRecord`, `feature_key`, `disk_dtype`, `SCHEMA_VERSION`, `ROW_*`, `FORWARD_*`, `PrefixHasher`, `SeenPrefixes`. Also `MoeLayerSpec`, `feature_width`, `feature_dtype` from `contracts.py`.
- Produces:
  - **Frames:** `copy_rows(destination, source)`; `CaptureFrame(*, specs, capacity, hidden_dtype, pin_memory)` with `.features[(layer_id, feature)]`, `.positions`, `.token_ids`, `.residency`, `.event`, `.capacity`, `.nbytes`, `.stage(*, layer_id, feature, rows)`; `FramePool(frames)` with `.acquire(timeout=None)`, `.release(frame)`, `.capacity`, `.nbytes`.
  - **Store:** `FeatureStore.spill: Callable[[int, RouteFeature, Tensor], None] | None`.
  - **Writer:** `PendingForward(record, frame)`; `ShardWriter(*, directory, specs, hidden_dtype, pool, shard_rows, max_bytes, idle_flush_s=5.0)` with `.submit(pending)`, `.stop(reason)`, `.close()`, `.stopped` (a `threading.Event`), `.written_bytes`; constants `MANIFEST_NAME`, `HEADER_NAME`, `STOPPED_NAME`.
  - **Reader:** `CaptureShard(name, request_ids, tensors)`, `CaptureReport`, `read_manifest(directory)`, `load_shard(directory, name)`, `check_capture(directory) -> CaptureReport`.

- [x] **Step 1: Write `capture_frames.py`**

```python
"""Pinned host frames that receive one forward's captured rows without blocking the GPU."""

from __future__ import annotations

import queue
from typing import Sequence

import torch

from sglang.srt.layers.moe.expert_prediction.capture_schema import CAPTURE_FEATURES
from sglang.srt.layers.moe.expert_prediction.contracts import (
    MoeLayerSpec,
    RouteFeature,
    feature_dtype,
    feature_width,
)


def copy_rows(destination: torch.Tensor, source: torch.Tensor) -> None:
    # Cast and compact on the source device so the host copy stays non-blocking.
    destination[: source.shape[0]].copy_(
        source.to(destination.dtype).contiguous(), non_blocking=True
    )


class CaptureFrame:
    def __init__(
        self,
        *,
        specs: Sequence[MoeLayerSpec],
        capacity: int,
        hidden_dtype: torch.dtype,
        pin_memory: bool,
    ) -> None:
        self.capacity = capacity
        self.features = {
            (spec.layer_id, feature): torch.empty(
                (capacity, feature_width(feature, spec)),
                dtype=feature_dtype(feature, hidden_dtype),
                pin_memory=pin_memory,
            )
            for spec in specs
            for feature in CAPTURE_FEATURES
        }
        self.positions = torch.empty(capacity, dtype=torch.int64, pin_memory=pin_memory)
        self.token_ids = torch.empty(capacity, dtype=torch.int64, pin_memory=pin_memory)
        self.residency = torch.full(
            (len(specs), max(spec.num_experts for spec in specs)),
            -1,
            dtype=torch.int64,
            pin_memory=pin_memory,
        )
        self.event: torch.cuda.Event | None = None

    @property
    def nbytes(self) -> int:
        tensors = [*self.features.values(), self.positions, self.token_ids, self.residency]
        return sum(tensor.numel() * tensor.element_size() for tensor in tensors)

    def stage(self, *, layer_id: int, feature: RouteFeature, rows: torch.Tensor) -> None:
        copy_rows(self.features[(layer_id, feature)], rows)


class FramePool:
    """Fixed frames; ``acquire`` blocks while the writer still holds every frame."""

    def __init__(self, frames: Sequence[CaptureFrame]) -> None:
        if not frames:
            raise ValueError("SGLANG_MOE_EXPERT_PREDICTOR_CAPTURE_FRAMES must be positive")
        self._frames = tuple(frames)
        self._free: queue.SimpleQueue[CaptureFrame] = queue.SimpleQueue()
        for frame in frames:
            self._free.put(frame)

    @property
    def capacity(self) -> int:
        return self._frames[0].capacity

    @property
    def nbytes(self) -> int:
        return sum(frame.nbytes for frame in self._frames)

    def acquire(self, timeout: float | None = None) -> CaptureFrame:
        return self._free.get(timeout=timeout)

    def release(self, frame: CaptureFrame) -> None:
        frame.event = None
        self._free.put(frame)
```

- [x] **Step 2: Add the spill sink to `FeatureStore`**

In `__init__`, after the `self._buffers = ...` assignment, add:

```python
        # Receives batches above max_rows (eager prefill) when capture is on.
        self.spill: Callable[[int, RouteFeature, torch.Tensor], None] | None = None
```

Change `from typing import Iterable` to `from typing import Callable, Iterable`. Replace `write` with:

```python
    def write(self, layer_id: int, feature: RouteFeature, source: torch.Tensor) -> None:
        """Copy ``source`` rows in; batches above ``max_rows`` go to ``spill`` or are skipped."""
        buffer = self._buffers.get((layer_id, feature))
        rows = source.shape[0]
        if buffer is None or rows == 0:
            return
        oversized = rows > self.max_rows
        if oversized and self.spill is None:
            return
        width = buffer.shape[1]
        flat = source.reshape(rows, -1)
        prefix_ok = feature in _PREFIX_FEATURES and flat.shape[1] > width
        if flat.shape[1] != width and not prefix_ok:
            raise ValueError(
                f"layer {layer_id} {feature.value} has width {flat.shape[1]}, "
                f"expected {width}"
            )
        with torch.no_grad():
            if oversized:
                self.spill(layer_id, feature, flat[:, :width])
            else:
                buffer[:rows].copy_(flat[:, :width])
```

- [x] **Step 3: Write `capture_writer.py`**

```python
"""Background thread that turns captured frames into safetensors shards on disk."""

from __future__ import annotations

import json
import logging
import os
import queue
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable, Sequence

import msgspec
import torch
from safetensors.torch import save_file

from sglang.srt.layers.moe.expert_prediction.capture_frames import CaptureFrame, FramePool
from sglang.srt.layers.moe.expert_prediction.capture_schema import (
    CAPTURE_FEATURES,
    FORWARD_INDEX,
    FORWARD_KIND,
    FORWARD_RESIDENCY,
    FORWARD_ROWS,
    ROW_FORWARD,
    ROW_POSITION,
    ROW_PREFIX_HASH,
    ROW_REQUEST,
    ROW_TOKEN,
    SCHEMA_VERSION,
    CaptureKind,
    ForwardRecord,
    disk_dtype,
    feature_key,
)
from sglang.srt.layers.moe.expert_prediction.contracts import MoeLayerSpec
from sglang.srt.layers.moe.expert_prediction.prefix_hash import PrefixHasher, SeenPrefixes

logger = logging.getLogger(__name__)

MANIFEST_NAME = "manifest.jsonl"
HEADER_NAME = "capture.json"
STOPPED_NAME = "capture-stopped.json"
# Arbitrary; bounds what a killed server loses once traffic pauses.
IDLE_FLUSH_S = 5.0
# Arbitrary; requests whose hash chains are remembered at once.
MAX_HASHED_REQUESTS = 4096
# Arbitrary; how often the writer checks whether a frame's copies finished.
_EVENT_POLL_S = 0.001


class PendingForward(msgspec.Struct, frozen=True):
    record: ForwardRecord
    frame: CaptureFrame


class ShardWriter:
    """Dedupe re-prefilled prefixes and append rows to shards; stops, never thins."""

    def __init__(
        self,
        *,
        directory: Path,
        specs: Sequence[MoeLayerSpec],
        hidden_dtype: torch.dtype,
        pool: FramePool,
        shard_rows: int,
        max_bytes: int,
        idle_flush_s: float = IDLE_FLUSH_S,
    ) -> None:
        if shard_rows < 1:
            raise ValueError("SGLANG_MOE_EXPERT_PREDICTOR_CAPTURE_SHARD_ROWS must be positive")
        if max_bytes < 1:
            raise ValueError("SGLANG_MOE_EXPERT_PREDICTOR_CAPTURE_MAX_GB must be positive")
        if directory.exists() and any(directory.iterdir()):
            raise ValueError(f"capture directory {directory} is not empty")
        directory.mkdir(parents=True, exist_ok=True)
        self._directory = directory
        self._specs = tuple(specs)
        self._hidden_dtype = hidden_dtype
        self._pool = pool
        self._shard_rows = shard_rows
        self._max_bytes = max_bytes
        self._idle_flush_s = idle_flush_s
        self._hasher = PrefixHasher(max_requests=MAX_HASHED_REQUESTS)
        self._seen = SeenPrefixes()
        self._shard_index = 0
        self.written_bytes = 0
        self.stopped = threading.Event()
        self._reset_shard()
        self._write_header()
        self._queue: queue.SimpleQueue[PendingForward | None] = queue.SimpleQueue()
        self._thread = threading.Thread(
            target=self._run, name="moe-expert-capture-writer", daemon=True
        )
        self._thread.start()

    def submit(self, pending: PendingForward) -> None:
        self._queue.put(pending)

    def stop(self, reason: str) -> None:
        if self.stopped.is_set():
            return
        self.stopped.set()
        logger.warning(
            "MoE expert capture stopped: %s (written_bytes=%d)", reason, self.written_bytes
        )
        try:
            (self._directory / STOPPED_NAME).write_text(
                json.dumps(
                    {
                        "reason": reason,
                        "written_bytes": self.written_bytes,
                        "timestamp_ns": time.time_ns(),
                    }
                )
            )
        except OSError:
            logger.exception("MoE expert capture could not record its stop reason")

    def close(self) -> None:
        if self._thread.is_alive():
            self._queue.put(None)
            self._thread.join()

    def _run(self) -> None:
        while True:
            try:
                item = self._queue.get(timeout=self._idle_flush_s)
            except queue.Empty:
                self._guarded(self._flush)
                continue
            if item is None:
                self._guarded(self._flush)
                return
            try:
                if not self.stopped.is_set():
                    self._guarded(lambda: self._ingest(item))
            finally:
                self._pool.release(item.frame)

    def _guarded(self, action: Callable[[], None]) -> None:
        try:
            action()
        except Exception as error:
            # The writer must keep releasing frames or the serving thread deadlocks.
            logger.exception("MoE expert capture writer failed")
            self.stop(f"writer error: {error!r}")

    def _reset_shard(self) -> None:
        self._columns: dict[str, list[torch.Tensor]] = defaultdict(list)
        self._request_ids: dict[str, int] = {}
        self._forward_count = 0
        self._row_count = 0

    def _write_header(self) -> None:
        header = {
            "schema_version": SCHEMA_VERSION,
            "hidden_dtype": str(self._hidden_dtype).removeprefix("torch."),
            "features": [feature.value for feature in CAPTURE_FEATURES],
            "layers": [msgspec.to_builtins(spec) for spec in self._specs],
            "created_ns": time.time_ns(),
        }
        (self._directory / HEADER_NAME).write_text(json.dumps(header, indent=2))

    def _ingest(self, pending: PendingForward) -> None:
        record, frame = pending.record, pending.frame
        # Polling instead of Event.synchronize keeps sync debug mode quiet in this thread.
        while frame.event is not None and not frame.event.query():
            time.sleep(_EVENT_POLL_S)
        rows = sum(record.rows_per_request)
        positions = frame.positions[:rows].tolist()
        token_ids = frame.token_ids[:rows].tolist()
        hashes: list[int] = []
        request_index: list[int] = []
        start = 0
        for rid, count in zip(record.rids, record.rows_per_request):
            end = start + count
            hashes += self._hasher.hash_rows(
                rid=rid, positions=positions[start:end], token_ids=token_ids[start:end]
            )
            request_index += [self._request_ids.setdefault(rid, len(self._request_ids))] * count
            start = end
        keep = self._seen.keep_mask(
            hashes=hashes, is_prefill=record.kind is CaptureKind.PREFILL
        )
        kept = torch.tensor([row for row, flag in enumerate(keep) if flag], dtype=torch.long)
        columns = self._columns
        columns[ROW_FORWARD].append(
            torch.full((kept.numel(),), record.forward_index, dtype=torch.int64)
        )
        columns[ROW_REQUEST].append(torch.tensor(request_index, dtype=torch.int32)[kept])
        columns[ROW_POSITION].append(frame.positions[:rows][kept])
        columns[ROW_TOKEN].append(frame.token_ids[:rows][kept])
        columns[ROW_PREFIX_HASH].append(torch.tensor(hashes, dtype=torch.int64)[kept])
        for (layer_id, feature), tensor in frame.features.items():
            columns[feature_key(layer_id, feature)].append(
                tensor[:rows][kept].to(disk_dtype(feature, self._hidden_dtype))
            )
        columns[FORWARD_INDEX].append(torch.tensor([record.forward_index], dtype=torch.int64))
        columns[FORWARD_KIND].append(torch.tensor([int(record.kind)], dtype=torch.uint8))
        columns[FORWARD_ROWS].append(torch.tensor([rows], dtype=torch.int64))
        columns[FORWARD_RESIDENCY].append(frame.residency.to(torch.int16).unsqueeze(0))
        self._forward_count += 1
        self._row_count += kept.numel()
        if self._row_count >= self._shard_rows:
            self._flush()

    def _flush(self) -> None:
        if self._forward_count == 0:
            return
        if self.stopped.is_set():
            self._reset_shard()
            return
        tensors = {key: torch.cat(parts).contiguous() for key, parts in self._columns.items()}
        nbytes = sum(tensor.numel() * tensor.element_size() for tensor in tensors.values())
        if self.written_bytes + nbytes > self._max_bytes:
            self._reset_shard()
            self.stop(f"byte cap {self._max_bytes} reached")
            return
        name = f"shard-{self._shard_index:06d}.safetensors"
        path = self._directory / name
        partial = self._directory / f"{name}.partial"
        request_ids = sorted(self._request_ids, key=self._request_ids.__getitem__)
        save_file(
            tensors,
            str(partial),
            metadata={
                "schema_version": str(SCHEMA_VERSION),
                "request_ids": json.dumps(request_ids),
            },
        )
        os.replace(partial, path)
        size = path.stat().st_size
        self.written_bytes += size
        forward_index = tensors[FORWARD_INDEX]
        with open(self._directory / MANIFEST_NAME, "a") as manifest:
            manifest.write(
                json.dumps(
                    {
                        "shard": name,
                        "rows": self._row_count,
                        "forwards": self._forward_count,
                        "first_forward": int(forward_index[0]),
                        "last_forward": int(forward_index[-1]),
                        "bytes": size,
                    }
                )
                + "\n"
            )
        self._shard_index += 1
        self._reset_shard()
```

- [x] **Step 4: Write `capture_reader.py`**

```python
"""Load expert capture shards and check that no forward or row went missing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import msgspec
import torch
from safetensors import safe_open

from sglang.srt.layers.moe.expert_prediction.capture_schema import (
    FORWARD_INDEX,
    FORWARD_KIND,
    FORWARD_ROWS,
    ROW_FORWARD,
    ROW_POSITION,
    ROW_REQUEST,
    CaptureKind,
)
from sglang.srt.layers.moe.expert_prediction.capture_writer import (
    MANIFEST_NAME,
    STOPPED_NAME,
)

_MAX_REPORTED_VIOLATIONS = 50


class CaptureShard(msgspec.Struct, frozen=True):
    name: str
    request_ids: tuple[str, ...]
    tensors: dict[str, torch.Tensor]


class CaptureReport(msgspec.Struct, frozen=True):
    shards: int
    rows: int
    prefill_rows: int
    decode_rows: int
    forwards: int
    ran_rows: int
    requests: int
    bytes: int
    bytes_per_row: float
    stopped_reason: str | None
    violation_count: int
    violations: list[str]


def read_manifest(directory: Path) -> list[dict]:
    path = directory / MANIFEST_NAME
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def load_shard(directory: Path, name: str) -> CaptureShard:
    with safe_open(str(directory / name), framework="pt") as handle:
        metadata = handle.metadata()
        tensors = {key: handle.get_tensor(key) for key in handle.keys()}
    return CaptureShard(
        name=name, request_ids=tuple(json.loads(metadata["request_ids"])), tensors=tensors
    )


def check_capture(directory: Path) -> CaptureReport:
    entries = read_manifest(directory)
    violations: list[str] = []
    last_position: dict[str, int] = {}
    prefill_rows = decode_rows = ran_rows = 0
    next_forward: int | None = None
    for entry in entries:
        shard = load_shard(directory, entry["shard"])
        tensors = shard.tensors
        forward_index = tensors[FORWARD_INDEX]
        if next_forward is not None and int(forward_index[0]) != next_forward:
            violations.append(f"{shard.name}: forwards jump from {next_forward - 1}")
        if forward_index.numel() > 1 and bool((forward_index.diff() != 1).any()):
            violations.append(f"{shard.name}: forward indices are not consecutive")
        next_forward = int(forward_index[-1]) + 1
        ran_rows += int(tensors[FORWARD_ROWS].sum())
        kinds = dict(zip(forward_index.tolist(), tensors[FORWARD_KIND].tolist()))
        for forward, request, position in zip(
            tensors[ROW_FORWARD].tolist(),
            tensors[ROW_REQUEST].tolist(),
            tensors[ROW_POSITION].tolist(),
        ):
            rid = shard.request_ids[request]
            if position <= last_position.get(rid, -1):
                violations.append(f"{shard.name}: request {rid} repeats position {position}")
            last_position[rid] = position
            if kinds[forward] == CaptureKind.PREFILL:
                prefill_rows += 1
            else:
                decode_rows += 1
    rows = sum(entry["rows"] for entry in entries)
    total_bytes = sum(entry["bytes"] for entry in entries)
    stopped = directory / STOPPED_NAME
    return CaptureReport(
        shards=len(entries),
        rows=rows,
        prefill_rows=prefill_rows,
        decode_rows=decode_rows,
        forwards=sum(entry["forwards"] for entry in entries),
        ran_rows=ran_rows,
        requests=len(last_position),
        bytes=total_bytes,
        bytes_per_row=total_bytes / rows if rows else 0.0,
        stopped_reason=json.loads(stopped.read_text())["reason"] if stopped.exists() else None,
        violation_count=len(violations),
        violations=violations[:_MAX_REPORTED_VIOLATIONS],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    print(msgspec.json.encode(check_capture(args.directory)).decode())


if __name__ == "__main__":
    main()
```

- [x] **Step 5: Write the tests**

```python
import tempfile
import time
import unittest
from pathlib import Path

import torch

from sglang.srt.layers.moe.expert_prediction.capture_frames import CaptureFrame, FramePool
from sglang.srt.layers.moe.expert_prediction.capture_reader import (
    check_capture,
    load_shard,
    read_manifest,
)
from sglang.srt.layers.moe.expert_prediction.capture_schema import (
    CaptureKind,
    ForwardRecord,
)
from sglang.srt.layers.moe.expert_prediction.capture_writer import (
    STOPPED_NAME,
    PendingForward,
    ShardWriter,
)
from sglang.srt.layers.moe.expert_prediction.contracts import MoeLayerSpec, RouteFeature
from sglang.srt.layers.moe.expert_prediction.feature_store import FeatureStore

SPECS = [
    MoeLayerSpec(layer_id=0, num_experts=8, top_k=2, hidden_size=4),
    MoeLayerSpec(layer_id=1, num_experts=8, top_k=2, hidden_size=4),
]


def _pool():
    return FramePool(
        [
            CaptureFrame(specs=SPECS, capacity=8, hidden_dtype=torch.float32, pin_memory=False)
            for _ in range(2)
        ]
    )


def _writer(directory, pool, **overrides):
    settings = dict(shard_rows=4, max_bytes=1 << 30, idle_flush_s=30.0)
    settings.update(overrides)
    return ShardWriter(
        directory=directory, specs=SPECS, hidden_dtype=torch.float32, pool=pool, **settings
    )


def _submit(writer, pool, *, index, kind, rid, positions, token_ids):
    frame = pool.acquire(timeout=5)
    rows = len(positions)
    frame.positions[:rows] = torch.tensor(positions)
    frame.token_ids[:rows] = torch.tensor(token_ids)
    for (layer_id, feature), tensor in frame.features.items():
        if feature is RouteFeature.TOPK_IDS:
            tensor[:rows] = torch.tensor(positions).unsqueeze(1) % 8
        else:
            tensor[:rows] = torch.tensor(positions, dtype=tensor.dtype).unsqueeze(1)
    writer.submit(
        PendingForward(
            record=ForwardRecord(
                forward_index=index, kind=kind, rids=(rid,), rows_per_request=(rows,)
            ),
            frame=frame,
        )
    )


class TestShardWriter(unittest.TestCase):
    def test_round_trip_through_reader(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "capture"
            pool = _pool()
            writer = _writer(directory, pool)
            _submit(writer, pool, index=0, kind=CaptureKind.PREFILL, rid="a",
                    positions=[0, 1, 2], token_ids=[1, 2, 3])
            _submit(writer, pool, index=1, kind=CaptureKind.DECODE, rid="a",
                    positions=[3], token_ids=[4])
            writer.close()
            report = check_capture(directory)
            self.assertEqual((report.rows, report.prefill_rows, report.decode_rows), (4, 3, 1))
            self.assertEqual(report.violations, [])
            shard = load_shard(directory, read_manifest(directory)[0]["shard"])
            self.assertEqual(shard.request_ids, ("a",))
            self.assertEqual(
                shard.tensors["layer.0.router_input"][:, 0].tolist(), [0.0, 1.0, 2.0, 3.0]
            )
            self.assertEqual(shard.tensors["layer.1.topk_ids"].dtype, torch.int16)
            self.assertEqual(tuple(shard.tensors["forward.expert_to_slot"].shape), (2, 2, 8))

    def test_reprefilled_prefix_is_dropped(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "capture"
            pool = _pool()
            writer = _writer(directory, pool, shard_rows=100)
            _submit(writer, pool, index=0, kind=CaptureKind.PREFILL, rid="a",
                    positions=[0, 1, 2], token_ids=[1, 2, 3])
            _submit(writer, pool, index=1, kind=CaptureKind.DECODE, rid="a",
                    positions=[3], token_ids=[4])
            _submit(writer, pool, index=2, kind=CaptureKind.PREFILL, rid="b",
                    positions=[0, 1, 2, 3, 4], token_ids=[1, 2, 3, 4, 5])
            writer.close()
            report = check_capture(directory)
            self.assertEqual((report.rows, report.ran_rows, report.forwards), (5, 9, 3))
            shard = load_shard(directory, read_manifest(directory)[0]["shard"])
            self.assertEqual(shard.tensors["row.position"].tolist(), [0, 1, 2, 3, 4])
            self.assertEqual(shard.tensors["row.request_index"].tolist(), [0, 0, 0, 0, 1])

    def test_shard_with_every_row_deduplicated_still_records_its_forward(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "capture"
            pool = _pool()
            writer = _writer(directory, pool, shard_rows=1)
            _submit(writer, pool, index=0, kind=CaptureKind.PREFILL, rid="a",
                    positions=[0, 1], token_ids=[1, 2])
            _submit(writer, pool, index=1, kind=CaptureKind.PREFILL, rid="b",
                    positions=[0, 1], token_ids=[1, 2])
            writer.close()
            report = check_capture(directory)
            self.assertEqual(
                (report.shards, report.rows, report.forwards, report.ran_rows), (2, 2, 2, 4)
            )
            self.assertEqual(report.violations, [])

    def test_byte_cap_stops_instead_of_thinning(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "capture"
            pool = _pool()
            writer = _writer(directory, pool, shard_rows=1, max_bytes=1)
            for index in range(3):
                _submit(writer, pool, index=index, kind=CaptureKind.DECODE, rid="a",
                        positions=[index], token_ids=[index])
            writer.close()
            self.assertTrue(writer.stopped.is_set())
            self.assertEqual(read_manifest(directory), [])
            self.assertIn("byte cap", check_capture(directory).stopped_reason)
            self.assertTrue((directory / STOPPED_NAME).exists())
            pool.acquire(timeout=1)
            pool.acquire(timeout=1)

    def test_idle_flush_writes_a_partial_shard(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "capture"
            pool = _pool()
            writer = _writer(directory, pool, shard_rows=100, idle_flush_s=0.05)
            _submit(writer, pool, index=0, kind=CaptureKind.DECODE, rid="a",
                    positions=[0], token_ids=[7])
            deadline = time.monotonic() + 5
            while not read_manifest(directory) and time.monotonic() < deadline:
                time.sleep(0.05)
            self.assertEqual(read_manifest(directory)[0]["rows"], 1)
            writer.close()

    def test_refuses_non_empty_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "leftover").write_text("x")
            with self.assertRaisesRegex(ValueError, "not empty"):
                _writer(Path(tmp), _pool())


class TestFeatureStoreSpill(unittest.TestCase):
    def test_oversized_batches_go_to_spill_with_routed_columns_only(self):
        store = FeatureStore(
            specs=SPECS, features=[RouteFeature.TOPK_IDS], max_rows=2,
            device=torch.device("cpu"), hidden_dtype=torch.float32,
        )
        calls = []
        store.spill = lambda layer_id, feature, rows: calls.append((layer_id, feature, rows))
        store.write(0, RouteFeature.TOPK_IDS, torch.arange(9).reshape(3, 3))
        store.write(0, RouteFeature.TOPK_IDS, torch.arange(6).reshape(2, 3))
        self.assertEqual(len(calls), 1)
        self.assertEqual(tuple(calls[0][2].shape), (3, 2))
        self.assertEqual(store.view(0, RouteFeature.TOPK_IDS, 2).tolist(), [[0, 1], [3, 4]])


if __name__ == "__main__":
    unittest.main()
```

Test `test_shard_with_every_row_deduplicated_still_records_its_forward` is the zero-row-tensor safetensors case: its second shard holds one forward and no rows. If `save_file` rejects zero-sized tensors, the test fails with a safetensors error. In that case, skip writing empty row tensors, store `row.*`/`layer.*` keys only when the shard has at least one kept row, and make `check_capture` treat missing row keys as zero rows.

- [x] **Step 6: Commit, push, sync the worktree, run the new tests plus the existing feature store tests on divix01**

```bash
git add python/sglang/srt/layers/moe/expert_prediction/capture_frames.py \
  python/sglang/srt/layers/moe/expert_prediction/capture_writer.py \
  python/sglang/srt/layers/moe/expert_prediction/capture_reader.py \
  python/sglang/srt/layers/moe/expert_prediction/feature_store.py \
  test/registered/unit/layers/moe/test_expert_prediction_capture_writer.py
git commit -m "feat(moe): add expert capture frames, shard writer, and reader" -- \
  python/sglang/srt/layers/moe/expert_prediction/capture_frames.py \
  python/sglang/srt/layers/moe/expert_prediction/capture_writer.py \
  python/sglang/srt/layers/moe/expert_prediction/capture_reader.py \
  python/sglang/srt/layers/moe/expert_prediction/feature_store.py \
  test/registered/unit/layers/moe/test_expert_prediction_capture_writer.py
git push origin master
ssh -n divix01 'cd /data/models/slang/nvfp4-work/cc-expert-prediction/worktree && git fetch -q origin master && git checkout -q --detach FETCH_HEAD && git log -1 --oneline'
ssh -n divix01 'cd /data/models/slang/nvfp4-work/cc-expert-prediction/worktree && \
  CUDA_VISIBLE_DEVICES="" PYTHONPATH=/data/models/slang/nvfp4-work/flashinfer-0.6.18-cu130-overlay:$PWD/python:/data/models/slang/nvfp4-work/uring-test-deps \
  timeout 900 /data/models/slang/.venv/bin/python -m pytest -q -p no:cacheprovider -rfEs \
  test/registered/unit/layers/moe/test_expert_prediction_capture_writer.py \
  test/registered/unit/layers/moe/test_expert_prediction_taps.py \
  test/registered/unit/layers/moe/test_expert_prediction_runtime.py; echo EXIT=$?'
```

Expected: the 7 new tests pass, the existing taps/runtime tests still pass, `EXIT=0`.

---

### Task 3: RouteCapture, runtime wiring, env vars, ModelRunner gate, CUDA test

**Files:**
- Create: `python/sglang/srt/layers/moe/expert_prediction/capture.py`
- Modify: `python/sglang/srt/layers/moe/expert_prediction/runtime.py`
- Modify: `python/sglang/srt/environ.py`
- Modify: `python/sglang/srt/model_executor/model_runner.py` (`maybe_init_expert_prediction` only)
- Modify: `scripts/expert_prediction/run-shadow-server.sh`
- Test: `test/registered/unit/layers/moe/test_expert_prediction_capture.py`
- Test: `test/registered/unit/layers/moe/test_expert_prediction_capture_graph.py`

**Interfaces:**
- Consumes (Tasks 1-2):
  - `CAPTURE_FEATURES`, `CaptureKind`, `ForwardRecord`, `rows_per_request`.
  - `CaptureFrame`, `FramePool`, `copy_rows`.
  - `PendingForward`, `ShardWriter`.
  - `FeatureStore.spill`.
  - `check_capture`, `load_shard`, `read_manifest` (tests).
  - From the framework: `classify_forward`, `ForwardKind`.
- Produces:
  - `CaptureSettings(directory, capacity, frames, shard_rows, max_bytes)`.
  - `RouteCapture.build(*, specs, store, device, hidden_dtype, settings, hot_caches)`, `.on_forward_end(forward_batch, *, taps_supported)`, `.close()`, `.stopped`, `.forwards`.
  - `ExpertPredictionRuntime.build(..., capture: CaptureSettings | None = None)` and `.capture`.
  - `ExpertPredictionRuntime.from_env(..., max_prefill_rows: int = 0)`.

- [x] **Step 1: Write `capture.py`**

```python
"""Stage each prefill and decode forward's taps, identity, and residency for the writer."""

from __future__ import annotations

import atexit
import logging
from pathlib import Path
from typing import Any, Mapping, Sequence

import msgspec
import torch

from sglang.srt.layers.moe.expert_prediction.capture_frames import (
    CaptureFrame,
    FramePool,
    copy_rows,
)
from sglang.srt.layers.moe.expert_prediction.capture_schema import (
    CAPTURE_FEATURES,
    CaptureKind,
    ForwardRecord,
    rows_per_request,
)
from sglang.srt.layers.moe.expert_prediction.capture_writer import PendingForward, ShardWriter
from sglang.srt.layers.moe.expert_prediction.contracts import MoeLayerSpec, RouteFeature
from sglang.srt.layers.moe.expert_prediction.feature_store import FeatureStore
from sglang.srt.layers.moe.expert_residency_clock import ForwardKind, classify_forward

logger = logging.getLogger(__name__)

_CAPTURE_KINDS = {
    ForwardKind.PREFILL: CaptureKind.PREFILL,
    ForwardKind.DECODE: CaptureKind.DECODE,
}


class CaptureSettings(msgspec.Struct, frozen=True):
    directory: Path
    capacity: int
    frames: int
    shard_rows: int
    max_bytes: int


class RouteCapture:
    """Record every prefill and decode row; a forward it cannot record stops capture."""

    def __init__(
        self,
        *,
        specs: Sequence[MoeLayerSpec],
        store: FeatureStore,
        pool: FramePool,
        writer: ShardWriter,
        hot_caches: Mapping[int, Any],
        device: torch.device,
    ) -> None:
        self._specs = tuple(specs)
        self._store = store
        self._pool = pool
        self._writer = writer
        self._hot_caches = dict(hot_caches)
        self._use_events = device.type == "cuda"
        self._frame: CaptureFrame | None = None
        self.forwards = 0
        store.spill = self._spill

    @classmethod
    def build(
        cls,
        *,
        specs: Sequence[MoeLayerSpec],
        store: FeatureStore,
        device: torch.device,
        hidden_dtype: torch.dtype,
        settings: CaptureSettings,
        hot_caches: Mapping[int, Any],
    ) -> "RouteCapture":
        if any(spec.num_experts > torch.iinfo(torch.int16).max for spec in specs):
            raise ValueError("expert capture stores expert ids as int16")
        pool = FramePool(
            [
                CaptureFrame(
                    specs=specs,
                    capacity=settings.capacity,
                    hidden_dtype=hidden_dtype,
                    pin_memory=device.type == "cuda",
                )
                for _ in range(settings.frames)
            ]
        )
        writer = ShardWriter(
            directory=settings.directory,
            specs=specs,
            hidden_dtype=hidden_dtype,
            pool=pool,
            shard_rows=settings.shard_rows,
            max_bytes=settings.max_bytes,
        )
        capture = cls(
            specs=specs,
            store=store,
            pool=pool,
            writer=writer,
            hot_caches=hot_caches,
            device=device,
        )
        atexit.register(capture.close)
        logger.info(
            "MoE expert capture: directory=%s capacity_rows=%d frames=%d "
            "pinned_bytes=%d max_bytes=%d",
            settings.directory,
            settings.capacity,
            settings.frames,
            pool.nbytes,
            settings.max_bytes,
        )
        return capture

    @property
    def stopped(self) -> bool:
        return self._writer.stopped.is_set()

    def close(self) -> None:
        self._writer.close()

    def on_forward_end(self, forward_batch: Any, *, taps_supported: bool) -> None:
        frame, self._frame = self._frame, None
        kind = _CAPTURE_KINDS.get(classify_forward(forward_batch)[0])
        if kind is None or self.stopped:
            self._release(frame)
            return
        rows = forward_batch.input_ids.shape[0]
        per_request = rows_per_request(
            is_extend=kind is CaptureKind.PREFILL,
            batch_size=forward_batch.batch_size,
            extend_seq_lens=forward_batch.extend_seq_lens_cpu,
        )
        reason = _unrecordable_reason(
            rows=rows,
            per_request=per_request,
            positions=forward_batch.positions,
            taps_supported=taps_supported,
            spilled=frame is not None,
            max_rows=self._store.max_rows,
            capacity=self._pool.capacity,
        )
        if reason is not None:
            self._release(frame)
            self._writer.stop(reason)
            return
        if frame is None:
            frame = self._stage_store_rows(rows)
        self._stage_identity(frame=frame, forward_batch=forward_batch, rows=rows)
        if self._use_events:
            frame.event = torch.cuda.Event()
            frame.event.record()
        self._writer.submit(
            PendingForward(
                record=ForwardRecord(
                    forward_index=self.forwards,
                    kind=kind,
                    rids=tuple(forward_batch.rids),
                    rows_per_request=per_request,
                ),
                frame=frame,
            )
        )
        self.forwards += 1

    def _spill(self, layer_id: int, feature: RouteFeature, rows: torch.Tensor) -> None:
        if self.stopped:
            return
        if rows.shape[0] > self._pool.capacity:
            self._writer.stop(
                f"forward with {rows.shape[0]} rows exceeds capture capacity "
                f"{self._pool.capacity}"
            )
            return
        if self._frame is None:
            self._frame = self._pool.acquire()
        self._frame.stage(layer_id=layer_id, feature=feature, rows=rows)

    def _stage_store_rows(self, rows: int) -> CaptureFrame:
        frame = self._pool.acquire()
        for spec in self._specs:
            for feature in CAPTURE_FEATURES:
                frame.stage(
                    layer_id=spec.layer_id,
                    feature=feature,
                    rows=self._store.view(spec.layer_id, feature, rows),
                )
        return frame

    def _stage_identity(self, *, frame: CaptureFrame, forward_batch: Any, rows: int) -> None:
        copy_rows(frame.positions, forward_batch.positions[:rows])
        copy_rows(frame.token_ids, forward_batch.input_ids[:rows])
        width = frame.residency.shape[1]
        for index, spec in enumerate(self._specs):
            frame.residency[index].fill_(-1)
            cache = self._hot_caches.get(spec.layer_id)
            if cache is not None:
                copy_rows(frame.residency[index], cache.expert_to_slot[:width])

    def _release(self, frame: CaptureFrame | None) -> None:
        if frame is not None:
            self._pool.release(frame)


def _unrecordable_reason(
    *,
    rows: int,
    per_request: tuple[int, ...],
    positions: torch.Tensor,
    taps_supported: bool,
    spilled: bool,
    max_rows: int,
    capacity: int,
) -> str | None:
    if not taps_supported:
        return "some MoE layers lack standard top-k outputs"
    if rows == 0 or sum(per_request) != rows:
        return f"forward rows {rows} do not match per-request rows {per_request}"
    if rows > capacity:
        return f"forward with {rows} rows exceeds capture capacity {capacity}"
    if rows > max_rows and not spilled:
        return f"forward with {rows} rows reached no tap"
    if positions.dim() != 1:
        return "expert capture needs 1-D positions"
    return None
```

- [x] **Step 2: Wire capture into `runtime.py`**

- **Imports:**

```python
from sglang.srt.layers.moe.expert_prediction.capture import CaptureSettings, RouteCapture
from sglang.srt.layers.moe.expert_prediction.capture_schema import CAPTURE_FEATURES
```

- **`__init__`:** add a keyword parameter `capture: RouteCapture | None = None` after `score_interval`, and store it with `self.capture = capture`.
- **`from_env`:** add a parameter `max_prefill_rows: int = 0` after `tokens_per_request`. Right after the `max_rows < 1` check, add:

```python
        capture_dir = envs.SGLANG_MOE_EXPERT_PREDICTOR_CAPTURE_DIR.get()
        capture = None
        if capture_dir:
            if tokens_per_request != 1:
                raise ValueError(
                    "SGLANG_MOE_EXPERT_PREDICTOR_CAPTURE_DIR does not support speculative decoding"
                )
            if max_prefill_rows < 1:
                raise ValueError(
                    "SGLANG_MOE_EXPERT_PREDICTOR_CAPTURE_DIR needs --chunked-prefill-size"
                )
            capture = CaptureSettings(
                directory=Path(capture_dir),
                capacity=max(max_rows, max_prefill_rows),
                frames=envs.SGLANG_MOE_EXPERT_PREDICTOR_CAPTURE_FRAMES.get(),
                shard_rows=envs.SGLANG_MOE_EXPERT_PREDICTOR_CAPTURE_SHARD_ROWS.get(),
                max_bytes=envs.SGLANG_MOE_EXPERT_PREDICTOR_CAPTURE_MAX_GB.get() * 2**30,
            )
```

  Then pass `capture=capture` into the `cls.build(...)` call.

- **`build`:**
  - Add a parameter `capture: CaptureSettings | None = None` after `experts_type`.
  - Replace the `features = ...` statement with:

```python
        features = {RouteFeature.TOPK_IDS}.union(
            *(predictor.required_features for predictor in predictors)
        )
        if capture is not None:
            features.update(CAPTURE_FEATURES)
```

  - After `pre_mixer_removers = ...`, build the capture:

```python
        route_capture = (
            None
            if capture is None
            else RouteCapture.build(
                specs=specs,
                store=store,
                device=device,
                hidden_dtype=hidden_dtype,
                settings=capture,
                hot_caches=hot_caches,
            )
        )
```

  - Pass `capture=route_capture` into `cls(...)`.

- **`on_forward_end`:** make the first statement

```python
        if self.capture is not None:
            self.capture.on_forward_end(
                forward_batch, taps_supported=not self._taps.unsupported_layers
            )
```

- **`close`:** add `if self.capture is not None: self.capture.close()` as its first statement.
- **If `build_predictors(())` or `ShadowMetrics(predictor_names=(), ...)` raises**, make it accept an empty predictor list (zero predictors, empty counters). Capture-only runs build no predictors.

- [x] **Step 3: Env vars in `python/sglang/srt/environ.py`**

After `SGLANG_MOE_EXPERT_PREDICTOR_METRICS_FILE = EnvStr("")`:

```python
    SGLANG_MOE_EXPERT_PREDICTOR_CAPTURE_DIR = EnvStr("")
    SGLANG_MOE_EXPERT_PREDICTOR_CAPTURE_MAX_GB = EnvInt(500)
    SGLANG_MOE_EXPERT_PREDICTOR_CAPTURE_SHARD_ROWS = EnvInt(4096)
    SGLANG_MOE_EXPERT_PREDICTOR_CAPTURE_FRAMES = EnvInt(2)
```

- [x] **Step 4: ModelRunner orchestration edit**

In `maybe_init_expert_prediction` (`python/sglang/srt/model_executor/model_runner.py:784`), replace the gate:

```python
        if self.is_draft_worker or not (
            envs.SGLANG_MOE_EXPERT_PREDICTOR.get()
            or envs.SGLANG_MOE_EXPERT_PREDICTOR_CAPTURE_DIR.get()
        ):
            return
```

Then add `max_prefill_rows=get_schedule().chunked_prefill_size or 0,` right after the `tokens_per_request=...` argument. `get_schedule` is already imported at line 184. Update the docstring to `"""Attach MoE expert prediction shadow scoring and capture before CUDA graph capture."""`.

- [x] **Step 5: Server script capture argument**

In `scripts/expert_prediction/run-shadow-server.sh`:
The 4th positional argument is already `radix` (commit `92e9217851`), and `HOT_GPU_MB` is already an env knob (`c19b41a4d0`). Capture follows the env-knob pattern:
- Add a comment line under the usage line: `# Env: HOT_GPU_MB (default 14336), CAPTURE=1 to record expert prediction training data.`
- After `hot_gpu_mb=${HOT_GPU_MB:-14336}`, add:

```bash
capture_dir=""
if [ "${CAPTURE:-}" = 1 ]; then
    capture_dir=/mnt/nvme2/nvfp4-work/expert-prediction-capture/$name/$(date +%Y%m%d-%H%M%S)
fi
```

- Add `capture_dir=${capture_dir:-none}` to the `echo "cc-expert-prediction server ..."` line.
- Add `SGLANG_MOE_EXPERT_PREDICTOR_CAPTURE_DIR="$capture_dir" \` after the `SGLANG_MOE_EXPERT_PREDICTOR_METRICS_FILE=...` line.

- [x] **Step 6: Write the CPU integration tests `test_expert_prediction_capture.py`**

```python
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from sglang.srt.environ import envs
from sglang.srt.layers.moe.expert_prediction.capture import CaptureSettings
from sglang.srt.layers.moe.expert_prediction.capture_reader import (
    check_capture,
    load_shard,
    read_manifest,
)
from sglang.srt.layers.moe.expert_prediction.runtime import ExpertPredictionRuntime
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.model_executor.forward_batch_info import ForwardMode


class FakeTopK(nn.Module):
    def __init__(self):
        super().__init__()
        self.topk_config = SimpleNamespace(top_k=2, num_fused_shared_experts=0)

    def forward(self, hidden_states, router_logits):
        weights, ids = torch.topk(router_logits.softmax(dim=-1), 2, dim=-1)
        return StandardTopKOutput(topk_weights=weights, topk_ids=ids, router_logits=router_logits)


class FakeMoE(nn.Module):
    def __init__(self, layer_id):
        super().__init__()
        self.layer_id = layer_id
        self.num_experts = 8
        self.hidden_size = 6
        self.num_fused_shared_experts = 0

    def forward(self, hidden_states, topk_output):
        return hidden_states


class FakeBlock(nn.Module):
    def __init__(self, layer_id):
        super().__init__()
        self.gate = nn.Linear(6, 8, bias=False)
        self.topk = FakeTopK()
        self.experts = FakeMoE(layer_id)

    def forward(self, hidden_states):
        return self.experts(hidden_states, self.topk(hidden_states, self.gate(hidden_states)))


class FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList(FakeBlock(i) for i in range(3))

    def forward(self, hidden_states):
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


def _runtime(directory, *, capacity=8, capture=True):
    model = FakeModel()
    runtime = ExpertPredictionRuntime.build(
        model=model,
        predictor_names=(),
        device=torch.device("cpu"),
        hidden_dtype=torch.float32,
        max_rows=1,
        max_candidates=4,
        hot_caches={},
        log_interval=100,
        metrics_path=None,
        score_interval=1,
        topk_type=FakeTopK,
        experts_type=FakeMoE,
        capture=(
            CaptureSettings(
                directory=directory, capacity=capacity, frames=2,
                shard_rows=1000, max_bytes=1 << 30,
            )
            if capture
            else None
        ),
    )
    return model, runtime


def _forward(model, runtime, *, mode, rid, positions, token_ids):
    rows = len(positions)
    hidden = torch.randn(rows, 6)
    with torch.no_grad():
        model(hidden)
    runtime.on_forward_end(
        SimpleNamespace(
            forward_mode=mode,
            input_ids=torch.tensor(token_ids),
            positions=torch.tensor(positions),
            batch_size=1,
            spec_info=None,
            extend_num_tokens=rows,
            rids=[rid],
            extend_seq_lens_cpu=[rows] if mode is ForwardMode.EXTEND else None,
        )
    )
    return hidden


class TestRouteCapture(unittest.TestCase):
    def test_records_spilled_prefill_and_buffered_decode_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "capture"
            model, runtime = _runtime(directory)
            prefill = _forward(model, runtime, mode=ForwardMode.EXTEND, rid="r",
                               positions=[0, 1, 2, 3, 4], token_ids=[1, 2, 3, 4, 5])
            decode = _forward(model, runtime, mode=ForwardMode.DECODE, rid="r",
                              positions=[5], token_ids=[6])
            runtime.close()
            report = check_capture(directory)
            self.assertEqual((report.rows, report.prefill_rows, report.decode_rows), (6, 5, 1))
            self.assertEqual((report.violations, report.stopped_reason), ([], None))
            tensors = load_shard(directory, read_manifest(directory)[0]["shard"]).tensors
            expected = torch.cat([prefill, decode])
            torch.testing.assert_close(tensors["layer.0.router_input"], expected)
            torch.testing.assert_close(tensors["layer.2.pre_mixer"], expected)
            block = model.layers[1]
            expected_ids = block.topk(expected, block.gate(expected)).topk_ids
            self.assertEqual(tensors["layer.1.topk_ids"].tolist(), expected_ids.tolist())
            self.assertEqual(tensors["row.token_id"].tolist(), [1, 2, 3, 4, 5, 6])

    def test_second_turn_prefill_drops_the_seen_prefix(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "capture"
            model, runtime = _runtime(directory)
            _forward(model, runtime, mode=ForwardMode.EXTEND, rid="turn1",
                     positions=[0, 1, 2, 3], token_ids=[1, 2, 3, 4])
            _forward(model, runtime, mode=ForwardMode.EXTEND, rid="turn2",
                     positions=[0, 1, 2, 3, 4, 5], token_ids=[1, 2, 3, 4, 9, 9])
            runtime.close()
            report = check_capture(directory)
            self.assertEqual((report.rows, report.ran_rows, report.requests), (6, 10, 2))

    def test_oversized_forward_stops_capture_without_hanging(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "capture"
            model, runtime = _runtime(directory, capacity=4)
            _forward(model, runtime, mode=ForwardMode.EXTEND, rid="r",
                     positions=[0, 1, 2, 3, 4], token_ids=[1, 2, 3, 4, 5])
            for step in range(3):
                _forward(model, runtime, mode=ForwardMode.DECODE, rid="r",
                         positions=[5 + step], token_ids=[6])
            runtime.close()
            report = check_capture(directory)
            self.assertIn("exceeds capture capacity", report.stopped_reason)
            self.assertEqual(report.rows, 0)

    def test_capture_off_leaves_store_without_spill(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, runtime = _runtime(Path(tmp) / "capture", capture=False)
            self.assertIsNone(runtime.capture)
            self.assertIsNone(runtime.store.spill)

    def test_from_env_rejects_unsupported_capture_launches(self):
        common = dict(
            model=FakeModel(), gpu_id=0, hidden_dtype=torch.float32, decode_max_bs=1,
            tp_size=1, moe_ep_size=1, attn_dp_size=None, pp_size=1,
            expert_hot_cache_manager=None,
        )
        with tempfile.TemporaryDirectory() as tmp, \
                envs.SGLANG_MOE_EXPERT_PREDICTOR_CAPTURE_DIR.override(str(Path(tmp) / "c")):
            with self.assertRaisesRegex(ValueError, "speculative"):
                ExpertPredictionRuntime.from_env(
                    tokens_per_request=2, max_prefill_rows=4096, **common
                )
            with self.assertRaisesRegex(ValueError, "chunked-prefill-size"):
                ExpertPredictionRuntime.from_env(
                    tokens_per_request=1, max_prefill_rows=-1, **common
                )


if __name__ == "__main__":
    unittest.main()
```

- [x] **Step 7: Write the CUDA test `test_expert_prediction_capture_graph.py`**

```python
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from sglang.srt.layers.moe.expert_prediction.capture import CaptureSettings
from sglang.srt.layers.moe.expert_prediction.capture_reader import (
    check_capture,
    load_shard,
    read_manifest,
)
from sglang.srt.layers.moe.expert_prediction.runtime import ExpertPredictionRuntime
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=15, stage="base-a", runner_config="1-gpu-small")

DECODE_ROWS = 2
PREFILL_ROWS = 8
HIDDEN = 32
EXPERTS = 16
TOP_K = 4


class FakeTopK(nn.Module):
    def __init__(self):
        super().__init__()
        self.topk_config = SimpleNamespace(top_k=TOP_K, num_fused_shared_experts=0)

    def forward(self, hidden_states, router_logits):
        weights, ids = torch.topk(router_logits.float().softmax(dim=-1), TOP_K, dim=-1)
        return StandardTopKOutput(
            topk_weights=weights, topk_ids=ids.to(torch.int32), router_logits=router_logits
        )


class FakeMoE(nn.Module):
    def __init__(self, layer_id):
        super().__init__()
        self.layer_id = layer_id
        self.num_experts = EXPERTS
        self.hidden_size = HIDDEN
        self.num_fused_shared_experts = 0

    def forward(self, hidden_states, topk_output):
        return hidden_states * 1.0


class FakeBlock(nn.Module):
    def __init__(self, layer_id):
        super().__init__()
        self.gate = nn.Linear(HIDDEN, EXPERTS, bias=False)
        self.topk = FakeTopK()
        self.experts = FakeMoE(layer_id)

    def forward(self, hidden_states):
        return self.experts(hidden_states, self.topk(hidden_states, self.gate(hidden_states)))


class FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList(FakeBlock(i) for i in range(3))

    def forward(self, hidden_states):
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


def _batch(mode, positions, token_ids, rids, extend_lens):
    rows = positions.shape[0]
    return SimpleNamespace(
        forward_mode=mode,
        input_ids=token_ids,
        positions=positions,
        batch_size=len(rids),
        spec_info=None,
        extend_num_tokens=rows,
        rids=rids,
        extend_seq_lens_cpu=extend_lens,
    )


class TestCaptureGraph(unittest.TestCase):
    def test_graph_decode_and_eager_prefill_are_captured_without_sync(self):
        device = torch.device("cuda")
        model = FakeModel().to(device=device, dtype=torch.bfloat16)
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "capture"
            runtime = ExpertPredictionRuntime.build(
                model=model, predictor_names=(), device=device, hidden_dtype=torch.bfloat16,
                max_rows=DECODE_ROWS, max_candidates=4, hot_caches={}, log_interval=100,
                metrics_path=None, score_interval=1, topk_type=FakeTopK, experts_type=FakeMoE,
                capture=CaptureSettings(
                    directory=directory, capacity=PREFILL_ROWS, frames=2,
                    shard_rows=1000, max_bytes=1 << 30,
                ),
            )
            static_input = torch.zeros(DECODE_ROWS, HIDDEN, device=device, dtype=torch.bfloat16)
            with torch.no_grad():
                side = torch.cuda.Stream()
                with torch.cuda.stream(side):
                    for _ in range(3):
                        model(static_input)
                torch.cuda.current_stream().wait_stream(side)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    model(static_input)

                prefill_input = torch.randn(PREFILL_ROWS, HIDDEN, device=device, dtype=torch.bfloat16)
                decode_input = torch.randn(DECODE_ROWS, HIDDEN, device=device, dtype=torch.bfloat16)
                prefill_positions = torch.arange(4, device=device).repeat(2)
                decode_positions = torch.tensor([4, 4], device=device)
                # Distinct tokens per request so request b's prefill is not deduplicated.
                prefill_tokens = torch.arange(PREFILL_ROWS, device=device) + 100
                decode_tokens = torch.tensor([200, 201], device=device)
                torch.cuda.synchronize()
                torch.cuda.set_sync_debug_mode("error")
                try:
                    model(prefill_input)
                    runtime.on_forward_end(
                        _batch(ForwardMode.EXTEND, prefill_positions, prefill_tokens,
                               ["a", "b"], [4, 4])
                    )
                    static_input.copy_(decode_input)
                    graph.replay()
                    runtime.on_forward_end(
                        _batch(ForwardMode.DECODE, decode_positions, decode_tokens,
                               ["a", "b"], None)
                    )
                finally:
                    torch.cuda.set_sync_debug_mode("default")
                expected_ids = model.layers[0].topk(
                    decode_input, model.layers[0].gate(decode_input)
                ).topk_ids.cpu()
            runtime.close()
            report = check_capture(directory)
            self.assertEqual((report.violations, report.stopped_reason), ([], None))
            self.assertEqual((report.prefill_rows, report.decode_rows), (PREFILL_ROWS, DECODE_ROWS))
            tensors = load_shard(directory, read_manifest(directory)[0]["shard"]).tensors
            torch.testing.assert_close(
                tensors["layer.0.router_input"], torch.cat([prefill_input, decode_input]).cpu()
            )
            self.assertEqual(
                tensors["layer.0.topk_ids"][PREFILL_ROWS:].tolist(), expected_ids.tolist()
            )


if __name__ == "__main__":
    unittest.main()
```

- [x] **Step 8: Commit, push, sync the worktree, run the tests on divix01**

The GPU is free (production can stay down). If a server holds the GPU, stop it first.

```bash
git add python/sglang/srt/layers/moe/expert_prediction/capture.py \
  python/sglang/srt/layers/moe/expert_prediction/runtime.py \
  python/sglang/srt/environ.py \
  python/sglang/srt/model_executor/model_runner.py \
  scripts/expert_prediction/run-shadow-server.sh \
  test/registered/unit/layers/moe/test_expert_prediction_capture.py \
  test/registered/unit/layers/moe/test_expert_prediction_capture_graph.py
git commit -m "feat(moe): capture full-sequence expert routing data for predictor training" -- \
  python/sglang/srt/layers/moe/expert_prediction/capture.py \
  python/sglang/srt/layers/moe/expert_prediction/runtime.py \
  python/sglang/srt/environ.py \
  python/sglang/srt/model_executor/model_runner.py \
  scripts/expert_prediction/run-shadow-server.sh \
  test/registered/unit/layers/moe/test_expert_prediction_capture.py \
  test/registered/unit/layers/moe/test_expert_prediction_capture_graph.py
git push origin master
ssh -n divix01 'cd /data/models/slang/nvfp4-work/cc-expert-prediction/worktree && git fetch -q origin master && git checkout -q --detach FETCH_HEAD && git log -1 --oneline'
ssh -n divix01 'cd /data/models/slang/nvfp4-work/cc-expert-prediction/worktree && \
  PYTHONPATH=/data/models/slang/nvfp4-work/flashinfer-0.6.18-cu130-overlay:$PWD/python:/data/models/slang/nvfp4-work/uring-test-deps \
  timeout 900 /data/models/slang/.venv/bin/python -m pytest -q -p no:cacheprovider -rfEs \
  test/registered/unit/layers/moe/test_expert_prediction_capture_schema.py \
  test/registered/unit/layers/moe/test_expert_prediction_capture_writer.py \
  test/registered/unit/layers/moe/test_expert_prediction_capture.py \
  test/registered/unit/layers/moe/test_expert_prediction_capture_graph.py \
  test/registered/unit/layers/moe/test_expert_prediction_taps.py \
  test/registered/unit/layers/moe/test_expert_prediction_adapters.py \
  test/registered/unit/layers/moe/test_expert_prediction_predictors.py \
  test/registered/unit/layers/moe/test_expert_prediction_metrics.py \
  test/registered/unit/layers/moe/test_expert_prediction_runtime.py \
  test/registered/unit/layers/moe/test_expert_prediction_graph.py; echo EXIT=$?'
```

Expected: all pass (the 21 new capture tests plus the existing framework tests), `EXIT=0`.

---

### Task 4: Live capture smoke on divix01

**Files:**
- Create: `scripts/expert_prediction/capture-two-turn-smoke.py`
- Create: `scripts/expert_prediction/check-capture-gate-topk.py`
- Create: `docs/superpowers/experiments/2026-09-14-expert-prediction-capture-smoke.md`

**Interfaces:**
- Consumes: `HOT_GPU_MB=12288 CAPTURE=1 run-shadow-server.sh <name> <port> off radix` (Task 3; radix on and a 12 GB hot cache match the settings `docs/superpowers/experiments/2026-09-14-radix-cache-ab.md` recommends for production); `python -m sglang.srt.layers.moe.expert_prediction.capture_reader <dir>` (Task 2); shard tensor keys (Task 1).
- Produces: measured bytes per row, dedupe effect, decode tok/s with capture on, and gate top-k agreement. These are recorded in the experiment doc.

- [x] **Step 1: Write `capture-two-turn-smoke.py`**

```python
"""Send a two-turn chat (turn 2 repeats turn 1 as history) and print token usage as JSON."""

import argparse
import json
import urllib.request


def _chat(port, messages, max_tokens):
    body = json.dumps(
        {"model": "default", "messages": messages, "max_tokens": max_tokens, "temperature": 0}
    ).encode()
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=1800) as response:
        return json.loads(response.read())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--max-tokens", type=int, default=256)
    args = parser.parse_args()
    messages = [
        {
            "role": "user",
            "content": "Write a Python function that computes the rolling Sharpe ratio "
            "of a pandas Series of daily returns over a 63-day window, annualized.",
        }
    ]
    first = _chat(args.port, messages, args.max_tokens)
    messages.append({"role": "assistant", "content": first["choices"][0]["message"]["content"] or ""})
    messages.append(
        {"role": "user", "content": "Now make it robust to missing days and explain the bias."}
    )
    second = _chat(args.port, messages, args.max_tokens)
    print(json.dumps({"turn1": first["usage"], "turn2": second["usage"]}))


if __name__ == "__main__":
    main()
```

- [x] **Step 2: Write `check-capture-gate-topk.py`**

```python
"""Check that gate weights times captured router input reproduce the captured top-k ids."""

import argparse
import json
import re
from pathlib import Path

import torch
from safetensors import safe_open

from sglang.srt.layers.moe.expert_prediction.capture_reader import load_shard, read_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture_dir", type=Path)
    parser.add_argument("model_dir", type=Path)
    args = parser.parse_args()
    header = json.loads((args.capture_dir / "capture.json").read_text())
    shard = load_shard(args.capture_dir, read_manifest(args.capture_dir)[0]["shard"])
    weight_map = json.loads((args.model_dir / "model.safetensors.index.json").read_text())["weight_map"]
    results = {}
    for layer in header["layers"]:
        layer_id, top_k = layer["layer_id"], layer["top_k"]
        pattern = re.compile(rf"(^|\.)layers\.{layer_id}\.mlp\.gate\.weight$")
        keys = [key for key in weight_map if pattern.search(key)]
        if len(keys) != 1:
            results[layer_id] = f"gate key not unique: {keys}"
            continue
        with safe_open(str(args.model_dir / weight_map[keys[0]]), framework="pt") as handle:
            weight = handle.get_tensor(keys[0])
        if not weight.is_floating_point():
            results[layer_id] = f"gate {keys[0]} is {weight.dtype}"
            continue
        router_input = shard.tensors[f"layer.{layer_id}.router_input"].float()
        captured = shard.tensors[f"layer.{layer_id}.topk_ids"].long()
        predicted = torch.topk(router_input @ weight.float().T, top_k, dim=-1).indices
        agreement = (predicted.unsqueeze(-1) == captured.unsqueeze(-2)).any(-1).float().mean()
        results[layer_id] = round(float(agreement), 5)
    print(json.dumps({"gate_key_example": keys, "agreement": results}))


if __name__ == "__main__":
    main()
```

- [x] **Step 3: Commit, push, sync the worktree**

Commit both scripts (`-- <paths>`), push, and sync the experiment worktree with the same three commands as Task 3 Step 8.

- [x] **Step 4: Launch the capture server**

The shadow script refuses to start while the GPU is in use. Production may stay down.

```bash
ssh -n divix01 'HOT_GPU_MB=12288 CAPTURE=1 nohup /data/models/slang/nvfp4-work/cc-expert-prediction/worktree/scripts/expert_prediction/run-shadow-server.sh capture-smoke 31010 off radix > /dev/null 2>&1 &'
ssh -n divix01 'timeout 1800 bash -c "until curl -sf http://127.0.0.1:31010/health > /dev/null; do sleep 10; done"; echo HEALTH_EXIT=$?'
ssh -n divix01 'grep -E "MoE expert capture|shadow mode" /data/models/slang/nvfp4-work/cc-expert-prediction/servers/capture-smoke/latest/server.log'
```

Expected: `HEALTH_EXIT=0`, and a `MoE expert capture: directory=/mnt/nvme2/nvfp4-work/expert-prediction-capture/capture-smoke/... capacity_rows=4096 frames=2` log line.

- [x] **Step 5: Drive two turns, wait for the idle flush, check the capture**

```bash
ssh -n divix01 'cd /data/models/slang/nvfp4-work/cc-expert-prediction/worktree && /data/models/slang/.venv/bin/python scripts/expert_prediction/capture-two-turn-smoke.py --port 31010'
ssh -n divix01 'sleep 15; d=$(ls -d /mnt/nvme2/nvfp4-work/expert-prediction-capture/capture-smoke/*/ | tail -1); echo $d; cd /data/models/slang/nvfp4-work/cc-expert-prediction/worktree && PYTHONPATH=$PWD/python /data/models/slang/.venv/bin/python -m sglang.srt.layers.moe.expert_prediction.capture_reader "$d"; du -sh "$d"'
```

Expected:
- `violations: []` and `stopped_reason: null`.
- `decode_rows == turn1.completion_tokens + turn2.completion_tokens - 2` (a request's first token comes from its prefill forward).
- `prefill_rows` equals the sum of `#new-token` over the server log's `Prefill batch` lines. With radix on, turn 2 prefills only the tokens after its cached prefix, so this is clearly less than `turn1.prompt_tokens + turn2.prompt_tokens`. Turn 2's rows start at a position above 0 and carry `prefix_hash` 0; that is expected.
- `bytes_per_row` near 494,000.

If `decode_rows` or `prefill_rows` differ, record the numbers and explain them from the forward kinds before continuing.

- [x] **Step 6: Gate top-k agreement and decode speed**

```bash
ssh -n divix01 'd=$(ls -d /mnt/nvme2/nvfp4-work/expert-prediction-capture/capture-smoke/*/ | tail -1); cd /data/models/slang/nvfp4-work/cc-expert-prediction/worktree && PYTHONPATH=$PWD/python /data/models/slang/.venv/bin/python scripts/expert_prediction/check-capture-gate-topk.py "$d" /mnt/nvme2/huggingface_hub/hub/models--nvidia--Qwen3.8-Flash-Next-NVFP4/snapshots/fc694b54fb0174e0913e6adf86691ef85a4ead47'
ssh -n divix01 'grep -E "Decode batch" /data/models/slang/nvfp4-work/cc-expert-prediction/servers/capture-smoke/latest/server.log | tail -20'
```

Expected:
- **Top-k agreement:** at least 0.99 on every layer. Lower agreement, or a non-float or missing gate key, means router logits cannot be dropped from capture. Record it as an open item.
- **Decode tok/s:** compare with capture off, 13.76 tok/s from `docs/superpowers/experiments/2026-09-14-expert-prediction-shadow-smoke.md`.

- [x] **Step 7: Stop the server, write the experiment doc, commit**

```bash
ssh -n divix01 'pkill -f "[s]glang serve.*--port 31010"; sleep 20; nvidia-smi --query-compute-apps=pid --format=csv,noheader'
```

Write `docs/superpowers/experiments/2026-09-14-expert-prediction-capture-smoke.md` with:
- **Setup:** commit, flags (the shadow script with `off capture`), capture directory.
- **Usage JSON** from Step 5.
- **Reader report** from Step 5, including the dedupe arithmetic.
- **Measured values:** bytes per row, gate agreement per layer (min, mean), decode tok/s with capture on vs off.
- **Disk projection:** tokens per 100 GB and hours of decode per 100 GB.
- **Open items:**
  - radix cache decision for the capture server (must match production);
  - speculative decoding support;
  - per-session tagging from the traffic driver.

Commit it with `-- docs/superpowers/experiments/2026-09-14-expert-prediction-capture-smoke.md`.

---

## Open items after this plan

- **Radix cache:** decided in `docs/superpowers/experiments/2026-09-14-radix-cache-ab.md` (enable, `extra_buffer`, 8 mamba slots, 12 GB hot cache). Capture runs use the same flags. Production's launcher still needs the change.
- **Speculative decoding (MTP):** VERIFY rows would need a branch tag and an acceptance join. Acceptance can be derived offline: a verify row is accepted iff the next forward for that request starts at a later position.
- **Session and domain tags:** the traffic driver should log `rid -> session, domain, split`, so shards can be split by session without parsing prompts.
