# ld.global.nc visibility experiment: pre-registration and result

Status of this file: **PRE-REGISTRATION ONLY** until a "Result" section is appended below the
line marked `END OF PRE-REGISTRATION`. The text above that line is committed *before* the
program is run and is not edited afterwards; anything learned during the run goes below it.

Context: `LEASE_PROTOCOL.md` section 6.6 (the argument and its limit) and OPEN 6. The copy
path reads pinned host memory with `ld.global.nc.v2.b64` (`copy_expert_host_unit16` in
`expert_cache_transfer.cuh`) and stores with `st.global.cg`. Under the lease protocol a slot's
bytes are immutable while leased, so the intra-kernel contract of `.nc` is met by construction.
What is not established is the **cross-kernel** case: the host rewrites a slot's bytes between
two kernels that read it, and the second kernel must see the new bytes.

## 1. The question, and the step-4 decision it feeds

Q: On this machine (RTX 5090, PCIe to a Xeon Gold 6154 host), when the host rewrites a
`cudaHostRegister`-ed row, does a kernel launched **afterwards** always read the new bytes
through `ld.global.nc`, in the shapes the service and the graph produce?

Decision it feeds (LEASE_PROTOCOL section 20, step 4): whether the lease-mode copy keeps
`ld.global.nc` or uses the non-`nc` variant of `copy_expert_host_unit16`.

## 2. Design (fixed now)

**Program.** `analysis/dsv41-drive/nc_visibility.cu`, built with nvcc 13.4 for `sm_120`,
run through `gpu-run.sh` (cc-gpu.lock, cores 32-63).

**Memory.** Host region: plain page-aligned memory registered with `cudaHostRegister(ptr, size,
0)`, as `allocate_host_slab` + `_cuda_host_register` do in production. Device destination:
`cudaMalloc`. Copy kernel geometry: 8 blocks x 256 threads, unit = 16 bytes, grid-stride, the
production geometry (`kExpertTransferGridSize`, `kExpertTransferBlockSize`).

**Visibility region.** 4 rows x 128 KiB = **512 KiB**. It is deliberately far *below* L2
(about 128 MB on this card): the worst case for a stale line is one that can stay resident, so
the visibility cells are sized to let it. (The project rule "size the working set past L2" is
about *bandwidth* measurements; section 2.6 applies it there.)

**Pattern.** Iteration `i` writes tag `t = i + 1` into every 8-byte word `w` of the region as
`(t << 32) | w`. A device-side check kernel compares every word of the copied bytes with the
expected value and classifies each mismatch: **stale** if its tag is `t - 1` (the previous
iteration's bytes), else **other** (a mixed, torn or foreign value). The tag word lives in
device memory, is bumped by a third tiny kernel, and never crosses PCIe.

**Loads under test (the variants).**

| Variant | Instruction | Role |
|---|---|---|
| `nc` | `ld.global.nc.v2.b64` | **production**, exactly as `copy_expert_host_unit16` |
| `cv` | `ld.global.cv.v2.b64` | the candidate non-`nc` load ("do not cache") |
| `sys` | `ld.relaxed.sys.global.u64` x2 | characterization only |
| `plain` | `ld.global.v2.b64` | characterization only |

**Cells (primary: `nc` and `cv`).** Each of `{nc, cv}` x `{boundary, graph, thrash,
concurrent}` x host store `{regular, nt}` = **16 primary cells**, each **N = 200,000
iterations** (= 200,000 x 65,536 words = 1.3e10 words checked per cell, before any
reduction):

- `boundary`: `fill(i)`; launch `copy, check, bump` on a stream; synchronize. The bytes at the
  same addresses were read by the previous iteration's copy, so its lines may be cached.
- `graph`: the same three kernels captured once in a CUDA graph and replayed (production runs
  graph replays).
- `thrash`: `boundary`, with a kernel streaming 256 MiB of device memory between iterations to
  evict L2. A control for whether residency matters; expected to be the safest cell.
- `concurrent`: launch a device-only busy kernel first, then `fill(i)` **while it runs**, then
  queue `copy, check, bump` behind it. This is the service-publishes-while-the-GPU-is-busy
  shape. Nothing overlaps a reader of the *same* rows.
- Host store `regular`: ordinary 64-bit stores then `_mm_sfence()`. `nt`: non-temporal 128-bit
  stores (`_mm_stream_si128`) then `_mm_sfence()` (the service already needs the sfence because
  its packing stores are not assumed ordered by a release store).

**Characterization cells (not gating):** `sys` and `plain` in `boundary`/`regular` only,
**N = 50,000** each.

**Controls (sensitivity).**

- **C-self (gate).** One iteration where the check kernel is told to expect `t + 1` although
  the host wrote `t`: it must report 100% of words as `stale`, and the words-checked counter
  must equal the region's word count. If either fails, the harness cannot see staleness.
- **C1 (characterization).** A bounded single-thread spin on a host-written flag (device-side
  timeout 20 ms, the host sets the flag ~2 ms after the kernel signals it started), 100
  repeats per variant. Reports the fraction that saw the flag and the latency. This is an
  *intra-kernel* host-visibility test, which PTX does not promise for `.nc`: it is here to show
  whether `nc` and `cv` differ *at all* on this part, so that "no difference seen across
  kernels" is not confused with "the harness cannot tell them apart".
- **C2 (characterization).** In one kernel, `x = ld.nc(a); st.cg(a, x+1); fence; y = ld.nc(a)`
  on device memory, 100,000 trials, for `nc` and `cv`: how often `y == x` (the second read was
  served from the per-SM cache). Also intra-kernel and outside PTX's guarantee; a sensitivity
  demonstration, not a supported use.

**Bandwidth cell (section 2.6).** A host region of **1 GiB** (8x L2), copied to a 1 GiB device
buffer by `nc` and `cv` at geometries 8x256 and 128x256, 10 repetitions each, timed with CUDA
events, plus `cudaMemcpyAsync` from the same region as the reference. The GPU is on a PCIe
link whose maximum generation the driver reports as **3**, so the ceiling is Gen3 x16,
15.75 GB/s theoretical (about 12-13 GB/s achievable). **Any reported figure above 15.75 GB/s
is a broken measurement, not a result.** The link generation is sampled (nvidia-smi, 1 Hz)
throughout, because it is 1 while idle.

**Conditions recorded with the result.** load1 before/after, the busiest foreign processes and
their cores (`ps`), GPU clocks, power and PCIe link state during the run (1 Hz), the git
commit of the program, the nvcc and driver versions, region sizes, geometry, iteration counts,
and the wall time of each cell. The box is known to be CPU-contended (load ~5, foreign
services): the host `fill` is CPU work, so launch overhead and iteration time will show it;
correctness cells are unaffected in kind, and this is reported, not corrected for.

## 3. Definitions of what is counted

- **Stale word**: tag `t - 1`. **Other word**: any other mismatch. **Fresh**: exact match.
- A cell **shows staleness** if `stale + other > 0` over its whole run.
- "Words checked" is read from the device counter, not computed from the loop count.

## 4. Decision rule (fixed before the run)

Let `S(v, cell)` be `stale + other` for variant `v` in a primary cell.

1. **Harness gate.** C-self must pass, every primary cell must complete its full N, and the
   words-checked counter must equal `N x 65,536` in each. If any of these fails, the result is
   **INCONCLUSIVE** (rule 5).
2. **Keep `ld.global.nc`** in lease-mode copies **iff** `S(nc, cell) == 0` in **all eight `nc`
   primary cells** **and** the gate passed. The finding is then
   worded "not observed in 1.3e10 words per cell across 8 cells on this part, driver and
   kernel", never "safe" or "coherent".
3. **Switch to `cv`** (the non-`nc` load, as a compile-time variant of `copy_expert_host_unit16`
   used by lease mode) **iff** `S(nc, cell) > 0` in **any** of the eight `nc` cells **and**
   `S(cv, cell) == 0` in all eight `cv` cells.
4. **Escalate** (neither variant is adequate; a visibility problem beyond the load
   instruction, needing an explicit invalidate, a different memory type or a different
   design) iff `S(cv, cell) > 0` in **any** `cv` cell. The lease protocol's cross-kernel
   argument is then not established for this part.
5. **INCONCLUSIVE**: lease mode ships with the **non-`nc` (`cv`) variant** as the compile-time
   default until the experiment is rerun and satisfies rule 2. "Not established" is not treated
   as "safe".
6. **Cost is not a tiebreaker for correctness.** The bandwidth cell informs only how much a
   switch to `cv` costs (rule 3 or 5). If rule 2 holds, `nc` is kept regardless of any
   throughput difference.
7. C1 and C2 and the `sys`/`plain` cells are reported and interpreted, but **do not enter
   rules 1-5**.

## 5. What this experiment does not establish (drafted before the run)

- It does not prove visibility. It reports a count of stale words in a finite number of
  iterations; zero is "not observed".
- It covers one GPU, one host (Skylake-SP Xeon over PCIe Gen3 x16 per the driver), one
  driver/toolkit, and the memory type used in production (registered host memory). It says
  nothing about other parts, `cudaHostAlloc` memory, GDS or DMA copies.
- It exercises *kernel-boundary* visibility. The *intra-kernel* case (host writes while a
  kernel that reads the same lines is running) is outside PTX's `.nc` contract and is only
  probed by C1/C2 as sensitivity checks; it must not be relied on.
- Its 512 KiB visibility region keeps lines L2-resident, which is the intended worst case for
  stale lines; a different access pattern or an eviction by unrelated traffic is covered only
  by the `thrash` cell.
- The host stores are two fixed shapes (regular and non-temporal + `sfence`); the service's
  real `memcpy` may compile to a different instruction mix.
- The CPU is contended; that changes timing, not what is checked, but it also means the
  interleavings reached are those this box produced.
- It says nothing about the model (`lease_model.py`) or the lease protocol's other steps.

END OF PRE-REGISTRATION
