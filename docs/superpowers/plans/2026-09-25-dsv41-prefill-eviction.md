# DSV4.1: keep prefill from evicting decode's RAM set (§27.4 item 2)

**Flag:** `SGLANG_DSV41_ENABLE_PREFILL_SHARE`, default off. `Dsv41Config.enable_prefill_share`.

## Problem

The eager prefill path admits every miss into the pinned tier (`gather_rows` → `ensure_rows` →
`NativePinnedSlotTable.assign` → `RamTier::assign`), evicting by LRU. The decode RAM-miss service uses the
same slots. A 260-token prompt misses ~146 non-VRAM experts per layer, of ~172 evictable rows, so after a
prefill most of decode's rows are gone (§27.2: S 64 ms/step at decode steps 0–4, 36 ms at 70+).

## Evidence first: a replay of the RAM tier

`analysis/dsv41-drive/prefill-evict/ram_replay.py` replays a graph route capture through `RamTier`'s victim
rule, with the logged VRAM hot sets, and changes only how prefill admits. The base arm reproduces the run
(11.33 RAM misses/token against 11.26 measured on varied24). Decode RAM misses per token, by steps after
the prefill, and prefill RAM misses (rows read) for the whole run:

| Trace | Arm | 0–4 | 5–14 | 70+ | all | prefill rows |
|---|---|---:|---:|---:|---:|---:|
| varied24 (24–44-token prompts) | base | 19.53 | 19.13 | 9.34 | 11.33 | 18,122 |
| | (a) no admit | 49.75 | 26.89 | 9.42 | 12.61 | 19,449 |
| | (b) cold (stamp 0) | 21.85 | 21.32 | 9.30 | 11.58 | 18,946 |
| | (b) share 64 | 19.53 | 19.13 | 9.34 | 11.33 | 18,122 |
| | bound: prefill evicts nothing | 16.85 | 17.09 | 9.33 | 11.07 | 17,807 |
| ce-soak s4 (up to many 512-token chunks) | base | 24.05 | 20.83 | 9.79 | 15.21 | 188,177 |
| | (a) no admit | 29.83 | 20.30 | 9.33 | 15.05 | 255,700 |
| | (b) cold | 21.77 | 18.83 | 9.29 | 14.09 | 262,991 |
| | (b) share 64 | 21.74 | 18.68 | 9.66 | 14.38 | 248,164 |
| | (b) share 128 | 23.78 | 20.65 | 9.79 | 15.15 | 187,227 |
| ce-soak d4 | base | 24.80 | 20.96 | 11.00 | 16.06 | 68,236 |
| | (a) no admit | 29.95 | 18.87 | 10.23 | 14.98 | 102,549 |
| | (b) cold | 19.92 | 16.53 | 10.51 | 13.79 | 94,578 |
| | (b) share 64 | 21.08 | 17.67 | 10.91 | 14.86 | 87,310 |

(The "evicts nothing" bound is only meaningful for short prompts: with long ones the tier grows by whole chunks.)

What this says:

- **Most of the early-decode excess is a session's cold start, not eviction.** Even with no decode row ever lost
  to a prefill, varied24's steps 0–4 miss 16.85 rows/token against 9.3 at steps 70+.
- **(a) staging without admission is the worst option.** A prompt's experts are the ones its first decode
  tokens route to, so not admitting them multiplies steps 0–4's misses (2.5× on short prompts). It would also
  need a new pinned staging area (~850 MB for one 64-row chunk) outside the C++ service's slab tables, which the
  item-1 prefill-fill path reads into. Rejected.
- **(b) helps only long prompts, and costs prefill re-reads.** Consecutive 512-token chunks of one prompt reuse
  most of a layer's experts; any bound on prefill's rows loses that reuse. Cold admission saves the most decode
  rows but adds ~40% prefill reads and is worse than base on short prompts. A share of 64 rows is never worse
  than base for decode, and costs fewer extra prefill reads (+28–32% on long prompts, none on short ones).

**Choice: (b) with a bounded prefill share of one gather chunk (64 rows per layer, `EXL3_MAX_GATHER_ROWS`).**
Default off: on the long-prompt soaks it trades ~4–5k decode row reads for ~19–60k prefill row reads, so it is
not a net win there, and on short prompts it does nothing. The arms below measure the 260-token case.

## Design

`RamTier` gains a per-slot prefill-ownership flag and a share `K` (0 = off, the only state with the flag off).

- `take_admit_slot_locked(row, protect, fallback, evicted)`: when `K > 0` and the layer holds ≥ K prefill-owned
  rows, the victim is the LRU owned row that is READY, unleased, not hot and not protected; with none, or with
  fewer than K owned, it is `take_slot_locked`. With `K > 0` the chosen slot becomes prefill-owned. With `K = 0`
  it is exactly `take_slot_locked`. `RamTier::assign` calls it.
- Ownership ends when the slot is freed or evicted, and when decode uses the row: a served demand's hit stamp,
  an unarmed touch, a prefetch lease, and a Python `touch` made with `K = 0`.
- `set_prefill_share(k)` (FFI `exl3_ram_miss_set_prefill_share`). `Exl3RamMissService` registers a pre-forward
  observer, only with the flag on, that sets `K = EXL3_MAX_GATHER_ROWS` for a non-decode forward and 0 for a
  decode forward.
- Nothing in `gather_rows`/`ensure_rows` changes.

**Leases.** The victim rule keeps every exclusion of `take_slot_locked` (READY only, unleased, not hot, not
protected), so a slot a GPU reader may copy from is never chosen, as today. Eager prefill runs with the stream
synced and the thread paused (`before_host_use`), as today.

**Memory.** No new allocation beyond one byte per pinned slot.

**Item 1 (`prefill-fills`, `SGLANG_DSV41_ENABLE_PREFILL_FILLS`).** Its `fill_begin` claims slots with
`take_slot_locked`; whichever lands second switches that call to `take_admit_slot_locked`. Both flags then work
alone and together.

## Tasks (TDD)

1. Tests first (`test/registered/unit/kernels/test_exl3_ram_miss_prefill_share.py`, CPU, the C++ tier by hand):
   decode-hot rows survive a prefill's admissions with a share; without it the same admissions evict them; a
   decode touch ends ownership; share 0 is the old victim order; a protected/leased owned row is never taken.
2. C++ + FFI + `Exl3RamMissHost.set_prefill_share`.
3. Env var, `Dsv41Config` field, the service's observer, and a service test (a prefill forward sets the share,
   a decode forward clears it, flag off registers nothing).
4. divix01: CPU suite `test/registered/unit/kernels` + the touched `layers/moe` tests; the manual RAM-miss GPU
   suite (`gpu_suite.sh`).
5. Arms on port 30021, node-mode traced (`NSYS_TRACE=1 NSYS_CUDA_GRAPH_TRACE=node`), A off then B on, once each:
   S and step wall for decode steps 0–4, 5–14, 70+ (`s_decay.py`), ms/token, TTFT, byte-identical outputs.
6. Rebase on `origin/master`, fast-forward `master`, `DSV41_REFERENCE.md` §27 result, mark §27.4 item 2.
