# Topology, registration limits and the io_uring feature audit — 2026-09-20

Plan: `docs/superpowers/plans/2026-09-20-storage-cpu-pipeline-v2.md`, Task 3, the
part that needs no GPU. Worktree HEAD `099eadba33` (`dsv41`).

**Every claim carries one tag.**

| tag | meaning |
|---|---|
| **[M]** | I measured it on divix01 on 2026-09-20, with the probes in the appendices |
| **[S]** | I read the machine's state from sysfs, procfs or a config file (a fact about the box, not a behaviour I exercised) |
| **[C]** | I read it in our code; the citation is file:line at the HEAD above |
| **[D]** | it comes from documentation or from another recorded document, cited |
| **[U]** | undetermined; the reason is given |

Measurements are CPU-only: `CUDA_VISIBLE_DEVICES=`, `taskset`/`numactl` pins on
cores 0-63 (cores 64-71 never used), no CUDA allocation. Drive reads went to
`/mnt/nvme0` only, with O_DIRECT. `/proc/diskstats` showed the drive idle (0 sectors in
2-3 s) before the first two batches of runs; the last two batches (THP matrix, `MADV_HUGEPAGE`)
followed within minutes and their idleness was **not** sampled. Nothing was read from
`/mnt/nvme1` or `/mnt/nvme2`. The diskstats read delta of the first batch equalled the
bytes the probe issued exactly (41,875 MB, 12 processes × 4 passes × 128 × 6.5 MiB),
so the O_DIRECT reads reached the drive.

## 0. Revision 2 (2026-09-21): what it supersedes and what it adds

Revision 1 (sections 1-6 and Appendices A-D below, commit `50cf23b619`) was written at
`099eadba33`. Revision 2 re-audits the reader code at `0752c0c0e6` (HEAD moved on to `e7a9cc3e7c` while I wrote; the audited blobs are named in §0.1 and did not change) and adds the topology facts
Revision 1 lacked (§7), a reader audit with current citations (§8), and the two measurements Task 3
still owes, **designed and not run** (§9). Where §0.1 contradicts sections 1-6, §0.1 wins.

Revision 2 was collected on divix01 between about 05:56 and 06:10 UTC on 2026-09-21 (`date -u`,
`date` printed CDT = UTC-5 at the end). It used **only** sysfs/procfs/`fincore`/`filefrag` reads,
two 32 MiB O_DIRECT `dd` reads, and one CPU-only pipe probe (Appendix E) pinned with
`taskset -c 2`. No GPU work (`nvidia-smi` queries only), no bulk drive read, nothing under
`/mnt/nvme1` or `/mnt/nvme2` beyond metadata. The machine was quiet: `nvidia-smi` showed 63 MiB used, P8,
0 % utilisation; `ps` showed no sglang or production process; load average 8.4 from unrelated
services (nimbus, reth, questdb). New tags: **[M2]** measured 2026-09-21 with the stated
command, **[S2]** read from sysfs/procfs/a tool on 2026-09-21, **[C2]** read in code at
`0752c0c0e6` (cited against `git show HEAD:<path>`, not the working tree).

### 0.1 Superseded or corrected

| Revision 1 said | Now | source |
|---|---|---|
| Worktree HEAD `099eadba33` | `0752c0c0e6`; 25 commits later (`git log --oneline 099eadba33..HEAD`). The native service file was rewritten in that span (two-bank pipeline `ddcb0d55ff`, stage trace, fault hooks): **every `exl3_ram_miss_host.cpp:NNN` citation in sections 2-4 is stale** and is remapped below | `git diff --shortstat 099eadba33 HEAD -- <native file>`: 774 insertions, 272 deletions [C2] |
| Native bounce is `kBounceRows x slot_bytes` = 8 slots, about 106 MB (§1.6, §4.3) | **`kBanks x kBounceRows` = 2 x 8 = 16 slots, about 213 MB.** "Task 4 asks for two banks" is already done (`ddcb0d55ff`), not future. The ring depth is `kQueueDepth x parts` = 16 x parts and is independent of the banks. 213 MB is 16 x the slot size Revision 1 took from `MIRROR_ROWS.md:42` (about 13.3 MB); the slot size was not read back from a live table | `exl3_ram_miss_host.cpp:49-52, :464, :634` [C2] |
| `cudaHostRegister` at `expert_host_tier.py:174-183` | `expert_host_tier.py:249` (`_cuda_host_register(slab, ...)`), `expert_host_arena.py:124`, implemented in `pool_host/common.py:132-167` | `grep -n _cuda_host_register` [C2] |
| The native service "creates its ring in `RowReader::open`, drives it from the service thread, `flags=0`" | Still true, at new lines (map below). The working tree of `exl3_ram_miss_host.cpp`, `ops/moe/exl3_ram_miss.py` and `exl3_stream_trace.py` carries **uncommitted edits by another session** (`git status`: `M`, +68/-12 in the native file); I audited the committed blob, not those edits | `git rev-parse HEAD:python/sglang/kernels/jit/csrc/moe/exl3_ram_miss_host.cpp` = `284a4d7e33d8d08471e33b2dc96670f513ac0267` [C2] |
| `uring_file_reader.cpp` citations | **Unchanged and still correct**: the file's last commit is `9ae02e21e9`, older than Revision 1's HEAD; blob `5e2f3744cd2af0c73ac7d1f7e93ff2d25609cf4d` | `git log -1 -- <path>`; `git rev-parse HEAD:<path>` [C2] |
| §5 "no serving process exists now" | Still true at 05:56 UTC (no `sglang` in `ps`, GPU 63 MiB); the live pinned-tier placement therefore remains undetermined | `ps -eo ... --sort=-pcpu`; `nvidia-smi` [S2] |
| Plan Task 3 "existing READ_FIXED and SINGLE_ISSUER/DEFER_TASKRUN support" | Adds one measured constraint Revision 1 missed: **a `DEFER_TASKRUN` ring shows no completion after `io_uring_submit()`, `submit_and_wait(0)` or `peek_cqe`, only after `io_uring_get_events()`, `submit_and_wait(1)` or with `IORING_SETUP_TASKRUN_FLAG`** (§8.3) | Appendix E [M2] |

Citation remap for `exl3_ram_miss_host.cpp` (Revision 1 line -> `0752c0c0e6` line):

| what | Rev 1 | now |
|---|---|---|
| `kBounceRows`, `kBanks`, `kQueueDepth`, `kPage` | `:46-47` | `:49`, `:50`, `:52`, `:53` |
| construction validation `dest + length <= slot_bytes` | `:237` | `:253` |
| bounce `posix_memalign` | `:339` | `:464` |
| `io_uring_queue_init(..., 0)` in `RowReader::open` | `:343` | `:482` |
| `open()` opens files `O_RDONLY\|O_CLOEXEC\|O_DIRECT` (when `direct_`) | (not cited) | `:435` |
| `io_uring_prep_read` (the only op) | `:425` | `:807` |
| `read()` "every return leaves the ring empty (I1)" | `:351` | `:495-504` |
| `reap()`; `submit(ready ? 0 : 1)` | `:600`, `:613` | `:822`, `:841` |
| `submit()` and its `io_uring_submit` / `_and_wait` | `:600` | `:1042`, `:1057` |
| `drain()`: `wait_cqe`, then `queue_exit` + `queue_init` | `:613`, `:621-622` | `:1065`, `:1070`, `:1078-1079` |
| destructor `io_uring_queue_exit` before `free(bounce_)` | `:293-295` | `:411-413` |
| `RamTier::open` -> `RowReader::open` | `:940-941` | `:1415-1416` |
| `exl3_ram_miss_open` (Python thread) | `:1456` | `:1935`, `:1957` |
| `exl3_ram_miss_pump` (caller-thread pump) | `:1467-1472` | `:1969` |
| `reader_.read(...)` in the service | `:1782` | `:1806` |
| service `std::thread`; `RamThread::run()` | `:1708`, run `:1782` | `:2210`, `:2259` |
| idle `_mm_pause` | `:1788` | `:2290` |
| `exl3_ram_miss_start_thread` | (not cited) | `:2362` |
| test-only `read_rows*` entry points | `:673-756` | `:1132`, `:1159`, `:1197` |

## 1. Static topology and budget

### 1.1 Machine, kernel, versions

| item | value | tag |
|---|---|---|
| host | divix01, 2× Xeon Gold 6154 (Skylake-SP, PCIe 3.0 root ports), 72 logical CPUs, 188 GiB | [S] |
| kernel | `6.12.0-211.51.1.el10_2.x86_64` (Rocky Linux 10.2). The laptop is a different kernel (`7.0.0-31`); nothing here applies to it | [S] |
| liburing | 2.12 (`pkg-config --modversion`, `/usr/lib64/liburing.so.2.12`); the JIT'd reader includes `/usr/include/liburing.h` from this package | [S] |
| NVIDIA driver | 610.57.04 | [S] |
| PyTorch / CUDA | 2.13.0+cu130 / 13.0 (`/data/models/slang/.venv`) | [S] |
| `kernel.io_uring_disabled` | 0 (io_uring enabled for all users) | [S] |
| `nvme` module `poll_queues` | 0 (see §2.4: IOPOLL cannot be exercised) | [S] |
| THP | `enabled=always`, `defrag=madvise`, `khugepaged/defrag=1` | [S] |
| `vm.zone_reclaim_mode` | 0 | [S] |

### 1.2 NUMA and CPU layout

| | CPUs | RAM | free at the time |
|---|---|---|---|
| node 0 | 0-17, 36-53 | 93.6 GiB | 56.7 GiB free, 20.6 GiB file cache |
| node 1 | 18-35, **54-71** | 94.4 GiB | 0.2-0.9 GiB free, **69.6 GiB file cache** |

`numactl -H` distance 10/21 [S]. Cores 64-71 (production's, core 71 the doorbell
spin core) are on **node 1**; the allowed 0-63 range spans both nodes [S].

**The GPU is on node 0. All four NVMe drives are on node 1** (`numa_node` of the PCI
functions, GPU `0000:37:00.0` root `pci0000:36`, drives under `pci0000:85`) [S].
`nvidia-smi topo -m`: GPU0 CPU affinity 0-17,36-53, NUMA 0 [S]. So a byte that goes
drive → bounce → pinned slab → GPU crosses the socket interconnect at least once,
whichever node the bounce and the slabs are allocated on.

### 1.3 Devices and negotiated links

| device | role | PCI addr | negotiated | capable | theoretical one-way | tag |
|---|---|---|---|---|---|---|
| RTX 5090 | GPU | 37:00.0 | **2.5 GT/s x16 at idle** (Gen 1; nvidia-smi `gpucurrent`=1) | GPU Gen5 x16; host root port Gen3 (`hostmax`=3, parent 36:00.0 max 8 GT/s x16) | Gen3 x16: 15.75 GB/s; idle Gen1 x16: 4 GB/s | [S] |
| Samsung 990 EVO Plus 2 TB | `/mnt/nvme0` (mirror root A) | 86:00.0 | 8 GT/s **x4** | 32 GT/s x4 | Gen3 x4: 3.94 GB/s | [S] |
| Crucial P310 4 TB | `/mnt/nvme1` (xfs; not used) | 87:00.0 | 8 GT/s x4 | 16 GT/s x4 | 3.94 GB/s | [S] |
| Samsung 990 EVO Plus 2 TB | `/mnt/nvme2` (**source** checkpoint) | 88:00.0 | 8 GT/s **x2** | 32 GT/s **x4**, root port 85:02.0 also x2 of x4 | Gen3 x2: **1.97 GB/s** | [S] |
| SPCC M.2 (`nvme3n1`) | `/mnt/nvme4` (mirror root B, ext4) | 89:00.0 | 8 GT/s x4 | 16 GT/s x4 | 3.94 GB/s | [S] |

- The four drives sit behind root ports `85:00.0`-`85:03.0` of the same Skylake-E root complex [S].
  Every drive negotiates Gen3, the root port's ceiling, so Gen4/5 drive capability is unused [S].
- **`/mnt/nvme2` (the source drive) trained at x2.** Its port and its device both report
  x2, both are capable of x4, and the AER correctable counters of the device and the port
  are all 0 [S]. The cause (a slot/riser lane split, or a downtrain) is **[U]**: seeing
  it needs `lspci -vv` LnkSta and dmesg, which are root-only here.
- The GPU link was Gen1 at every idle read; an earlier `nvidia-smi` read in this session
  showed generation 3 (`pcie.link.gen.current`) while the GPU was in use. A link state
  read at idle is not the loaded state. What the loaded, simultaneous link achieves is **[U]**
  (needs the GPU).
- The Gen3 x16 "theoretical" is 8 GT/s × 128b/130b × 16 lanes, a spec value, not a measurement.
- **Measured single-drive ceiling [M]:** `/mnt/nvme0`, O_DIRECT, 6.5 MiB extents, queue
  depth 16, one thread: **3.56 GB/s = 90 % of the 3.94 GB/s Gen3 x4 ceiling.** This passes
  the plan's bandwidth sanity check (below the link, far below anything cache-like).
  `SCHEDULING.md` recorded 3.3-3.5 GB/s per drive with both mirrors reading at once.

### 1.4 Filesystems and alignment

| mount | fs / block | partition start | notes | tag |
|---|---|---|---|---|
| `/mnt/nvme0` | xfs, 4096 B block, `sunit=0 swidth=0` | sector 2048 (1 MiB) | `relatime` | [S] |
| `/mnt/nvme2` | xfs, 4096 B, `sunit=0 swidth=0` | sector 2048 (1 MiB) | `noatime` | [S] |
| `/mnt/nvme4` | ext4, 4096 B (`stat -f`); stride/stripe **[U]** (`tune2fs` needs root) | sector 2048 (1 MiB) | `relatime` | [S] |

`statx(STATX_DIOALIGN)` on a shard of each root: `dio_mem_align=4`, `dio_offset_align=512`
on all three [M]. All four drives report 512 B logical and physical sectors, no I/O
scheduler (`none`), `nr_requests` 1023 [S]; the drives expose 16 (the Samsungs) to 32
hardware queues [S]. The reader still issues 4 KiB-aligned reads (`kPage`,
`exl3_ram_miss_host.cpp:339`, `:426`), which is stricter than the kernel requires;
file start offsets are not 4 KiB-aligned, so a superset read wastes 500-4,596 B
(`DSV41_REFERENCE.md:198`) [D].

### 1.5 First touch and NUMA behaviour

**Measured with plain anonymous memory (not the CUDA-pinned tier, which needs the GPU):**

- **First touch follows the touching CPU, but spills when that node is full [M].**
  `numa_touch`, 1 GiB touched from a core, pages located with `move_pages`:
  from a node-0 core 100 % on node 0; from a node-1 core **62.9 % node 1, 37.1 % node 0**,
  both times, because node 1 had only ~0.9 GiB free (its RAM is full of file cache) and
  `zone_reclaim_mode=0` lets the allocator fall back to the other node before it reclaims.
- **The drive → bounce DMA does not care which node the bounce is on, at one drive [M].**
  `nvme0` (node 1) into a bounce bound to node 0 or node 1, thread on either node: wall
  time 244.9-245.5 ms (p50 per process) for 128 × 6.5 MiB in **every one of 32 combinations**, 3.55-3.56 GB/s.
  Two drives at once into a remote bounce were not measured.
- **The row scatter is sensitive to node placement [M].** Single-thread `memcpy` of 13.3 MB
  rows (the mirror row size, `MIRROR_ROWS.md:42`), 40 rows per pass, medians of 6 passes, two rounds:

  | thread on | src (bounce) | dst (slab) | ms/row | vs all-local |
  |---|---|---|---|---|
  | node 0 | 0 | 0 | 2.64-2.65 | 1.00 |
  | node 0 | 1 | 0 | 3.14-3.27 | 1.19-1.24 |
  | node 0 | 0 | 1 | 4.08-4.12 | 1.55 |
  | node 0 | 1 | 1 | 4.64-4.69 | 1.75-1.77 |
  | node 1 | 1 | 1 | 2.59-2.83 | 1.00 (the same, mirrored) |
  | node 1 | 0 | 1 | 3.07-3.12 | 1.18 |
  | node 1 | 1 | 0 | 4.02-4.15 | 1.55 |
  | node 1 | 0 | 0 | 4.65-4.76 | 1.79 |

  A remote **write** costs more than a remote read. **The absolute numbers do not match
  the recorded scatter** (1.58-1.60 ms/row, `MIRROR_ROWS.md:36`, 8.4 GB/s here against
  5 GB/s from a plain `memcpy`); the production split is not one `memcpy`, and the box
  was not idle. Use the ratios only.
- **The 75 GB pinned tier cannot sit on one node.** It is 75,153,156,096 B resident
  (`SGLANG_MOE_PINNED_HOST_MB=71680`, 5,644 rows, `DSV41_REFERENCE.md:1404`) [D] against
  93.6/94.4 GiB nodes that are already largely file cache [S]. Which node each slab landed
  on is **[U]**: no serving process exists on divix01 right now (no sglang process in
  `ps`; node `Unevictable` 30 MB in total [S]), and the registration is
  `cudaHostRegister` on `torch.empty` memory (`expert_host_tier.py:174-183`,
  `pool_host/common.py:167`) [C], which needs CUDA. That path takes the plain-anonymous
  first-touch behaviour above only if `cudaHostRegister` populates on the calling
  thread, which I have not verified.
- **No code sets a NUMA policy.** No `mbind`, `numa` or `set_mempolicy` call exists under
  `srt/layers/moe`, `srt/mem_cache/pool_host` or the two native sources [C] (searched).
  `exl3_ram_miss_start_thread(cpu_core=-1)` inherits the caller's affinity
  (`ops/moe/exl3_ram_miss.py:418-421`) [C], so under `taskset -c 0-63` the service thread
  can run on either node and so can its bounce. Nothing records which.
- **THP backing of the bounce varies with the node [M].** The native bounce is a plain
  `posix_memalign` with no `madvise` (`exl3_ram_miss_host.cpp:339`) [C]. With THP=always a
  104 MiB bounce bound to node 0 came out **fully THP-backed (102 MiB `AnonHugePages`)**;
  bound to node 1 it came out **0 KiB THP**, and with `MADV_HUGEPAGE` and 2 MiB alignment
  only **52 of 104 MiB**. System-wide since boot, 17.3 M THP faults fell back to 4 KiB
  pages against 10.7 M that succeeded (`/proc/vmstat`) [S]. This matters because the
  page-pinning cost of an O_DIRECT read depends on it, 3-6× (§3).

### 1.6 Pinned-memory budget actually in use

| item | value | tag |
|---|---|---|
| pinned tier (recorded run) | 71,680 MB requested → 5,644 rows, 75,153,156,096 B resident, six slabs | [D] `DSV41_REFERENCE.md:1404` |
| hot GPU cache | 14,336 MB → 1,128 slots, 15.0 GB | [D] same |
| native bounce | **superseded by §0.1: now 16 slots ≈ 213 MB.** Rev 1: `kBounceRows × slot_bytes` = 8 × ~13.3 MB ≈ **106 MB** (slot_bytes from `MIRROR_ROWS.md:42`, not read back from a live table) | [C] `exl3_ram_miss_host.cpp:46,339`; [D] |
| Python EXL3 bounce | `BOUNCE_ROWS=8` rows of `slot_bytes`, plain `torch.empty` (not CUDA-pinned) | [C] `exl3_shard_row_source.py:29,51-60,99` |
| registration granularity | one `cudaHostRegister` per slab (chunk limit `SGLANG_HICACHE_HOST_REGISTER_CHUNK_GB`, default 256) | [C] `environ.py:996`, `pool_host/common.py:139-160` |
| live process value | **[U]**: there is no sglang process on divix01 at this moment; the budget above is the last recorded run's |

### 1.7 Registration limits

| limit | value | tag |
|---|---|---|
| `RLIMIT_MEMLOCK` | **unlimited** (soft and hard) in a login shell, and for an unrelated running user process (`/proc/<pid>/limits`); set by `/etc/security/limits.d/99-memlock.conf` (`* - memlock unlimited`) and systemd `DefaultLimitMEMLOCK=infinity` | [S] |
| `RLIMIT_MEMLOCK` is enforced for io_uring buffer registration | yes. With the limit lowered to 64 MiB in-process: 3 successive 32 MiB registrations succeed (each unregistered before the next), a 128 MiB registration fails `-ENOMEM`, and two 48 MiB in one call fail `-ENOMEM` | [M] (unprivileged user) |
| `vm.max_map_count` | 1,048,576 (an unrelated process holds 3,778 maps). Not a constraint: the tier registers ~6 ranges and the bounce is one mapping; I did not test whether registration splits VMAs | [S] |
| io_uring registered-buffer table | **16,384 entries** (`register_buffers_sparse` 16,384 succeeds, 16,385 and 65,536 fail `-EINVAL`) | [M] |
| io_uring registered buffer, one iovec | **1 GiB** (a 1 GiB buffer registers in 399 ms; 1 GiB + 4 KiB fails `-EFAULT`) | [M] |
| general reader's own caps | 1,024 registration slots, 1 GiB chunks | [C] `uring_file_reader.cpp:26-27,96,162` |

`cudaHostRegister` accounting against `RLIMIT_MEMLOCK` is **[U]** (CUDA); it is moot on
this box while the limit is unlimited.

## 2. io_uring feature audit

> **Line numbers for `exl3_ram_miss_host.cpp` in sections 1-4 are Revision 1's and are stale; see the remap in §0.1. §8 re-audits at `0752c0c0e6`.**

Two rings exist. Keep them apart.

- **General reader**: `UringFileReaderObj`, `csrc/io/uring_file_reader.cpp`, wrapped by
  `ops/io/uring_file_reader.py`, one process-wide shared instance
  (`get_shared_uring_file_reader`, queue depth `SGLANG_URING_FILE_READER_QUEUE_DEPTH`=128,
  `environ.py:516`, `uring_file_reader.py:168-173`). The eager EXL3 path
  (`Exl3RowReader`, `Exl3ShardRowSource`) reads through it [C] `exl3_row_reader.py:41,108`.
- **Native RAM-miss service**: `RowReader` inside `RamTier`,
  `csrc/moe/exl3_ram_miss_host.cpp`, driven by the `RamThread` service thread. Ring depth
  `kQueueDepth * parts` = 16 × 2 = 32 (`:47,592`).

### 2.1 Summary

| feature | general reader | native service | kernel 6.12 + liburing 2.12 |
|---|---|---|---|
| `IORING_OP_READ_FIXED` (registered buffers) | **implemented, used only when a caller registers** (`:379`); EXL3 does not register, so its reads are `IORING_OP_READ` | **not used**: only `io_uring_prep_read` (`:425`) | supported [M] |
| `IORING_SETUP_SINGLE_ISSUER` | **requested** with fallback (`:45-58`) | **not requested**: `io_uring_queue_init(..., 0)` (`:343`, `:622`) | supported [M] |
| `IORING_SETUP_DEFER_TASKRUN` | **requested** with fallback (`:45-58`) | **not requested** | supported [M] |

### 2.2 `READ_FIXED`

- General reader: it registers a sparse table of 1,024 slots at construction
  (`:96`), `register_buffer` chunks a range into ≤1 GiB entries (`:142-177`), each extent
  calls `find_buffer_` (`:231`) and uses `io_uring_prep_read_fixed` when the destination lies
  wholly inside one registration, `io_uring_prep_read` otherwise (`:379-381`) [C]. A
  registered range must stay alive until unregistered or the reader closes (the ring holds page
  references), which `ops/io/uring_file_reader.py:50-116` enforces with a `weakref.finalize` [C].
- Who registers: only `ExpertFileRowReader.register_destinations`
  (`expert_file_reader.py:185-192`), called from `expert_stream.py:212` [C].
  **`Exl3ShardRowSource.register_destinations` returns 0** (`exl3_shard_row_source.py:128-131`)
  and its bounce is an unregistered `torch.empty` (`:51-60`) [C]. So the EXL3 eager path
  issues no `READ_FIXED`. (The comment at `:129-130` says no destination is handed to
  io_uring; the destination *is* the bounce, so the comment is misleading.)
- Native service: the reads target `bounce_ + slot*slot_bytes + dest + done`
  (`:426`), and construction validates `e.dest + e.length <= slot_bytes` (`:237`) [C]. Every
  extent therefore lies inside `[bounce_, bounce_ + kBounceRows*slot_bytes)`.
- **Works on this kernel [M]:** `READ_FIXED` from `/mnt/nvme0` O_DIRECT into sub-ranges of one
  registered 104 MiB iovec, 128 reads × 6.5 MiB, no short or failed read.
- **What a native registered bounce would require [C]:**
  1. `io_uring_register_buffers` once after the ring exists (`:343`) on one iovec covering
     the whole bounce, and `io_uring_prep_read_fixed(..., buf_index=0)` at `:425`.
  2. **Re-register after `drain()` resets the ring** (`:621-622`): `io_uring_queue_exit` + `queue_init`
     discards the registration. Without that, the reset ring falls back silently to plain reads.
  3. Keep the teardown order the destructor already has, ring exit before `free(bounce_)`
     (`:293-295`).
  4. A counter of fixed vs plain SQEs, because a registration failure changes no result.
  5. An explicit decision on registration failure (`-ENOMEM` under a finite `RLIMIT_MEMLOCK`, measured
     above): fall back with a logged reason, do not fail the service.

### 2.3 `SINGLE_ISSUER` and `DEFER_TASKRUN`

**Facts in the general reader.** `init_ring` requests both flags and retries with 0 flags
only on `-EINVAL` (`:45-58`); the reader belongs to its constructing thread, and every
public call runs `check_owner_` (`:586-593`, `enter_` `:598`) [C]. Whether the shared
reader actually runs in this mode on divix01 was not observed inside a process; the probe shows
this kernel and liburing accept both flags for an unprivileged user, and the header
defines them, so the `#if` at `:47` is compiled in [M]/[C]. Nothing logs which mode the
reader ended up in. `get_shared_uring_file_reader` creates the reader on the first caller's
thread (`uring_file_reader.py:168-173`) [C], so that thread becomes the owner of every later call.

**Facts in the native service.** Neither flag is used (`:343`). More important, the ring
is **created on one thread and driven by another**:

- created in `RamTier::open` → `RowReader::open`, reached from `exl3_ram_miss_open`
  (`:940-941`, `:1456`) on the Python caller's thread [C];
- driven from the `RamThread` service thread (`std::thread` at `:1708`, `run()` calling
  `pump_demand`/`pump_advice` at `:1782`), which enters `RowReader::read` [C];
- recreated on whichever thread runs `drain()` (`:621-622`) [C], normally the service thread.

**So the plan's "one owning CPU thread drives each ring" is true of the native ring only by
convention.** Nothing enforces it, and the creator is not the driver.

**What the kernel does with these flags, measured [M]** (`uring_probe flags`):

| probe | result |
|---|---|
| ring init: `0`, `SINGLE_ISSUER`, `SINGLE_ISSUER\|DEFER_TASKRUN` | all `rc=0` |
| `DEFER_TASKRUN` without `SINGLE_ISSUER` | `-EINVAL` (so DEFER requires SINGLE) |
| `SINGLE_ISSUER` ring created on thread A, `submit` from thread B | **`-EEXIST`** |
| `flags=0` ring created on A, submit from B (**the native service today**) | works, `rc=1` |
| `SINGLE_ISSUER\|DEFER_TASKRUN\|R_DISABLED` created on A, `io_uring_enable_rings` + submit + wait on B | **works**; afterwards A's submit fails `-EEXIST`, i.e. the *enabling* thread becomes the issuer |
| read of an empty pipe submitted; a helper thread writes at 50 ms; the submitter spins on `io_uring_cq_ready` (memory only, no `io_uring_enter`) | `flags=0`: CQE visible at **51.6 ms**. `SINGLE_ISSUER\|DEFER_TASKRUN`: **no CQE in 500 ms**; it appeared only after `io_uring_wait_cqe` (a kernel entry) |

**Constraints this imposes on the native service, from the measurements above:**

1. **`SINGLE_ISSUER` as the code stands would break every read.** The ring belongs to the
   Python thread, the service thread submits, so the first submit returns `-EEXIST` and the
   read fails (`:449-457` treats only `-EINTR/-EAGAIN/-EBUSY` as soft). Adopting it needs either
   creating the ring on the service thread (which requires moving `RowReader::open`'s ring
   creation into `RamThread::run` and reporting failure back, since `open` currently fails
   synchronously at `:1456`) **or** `IORING_SETUP_R_DISABLED`, created where it is now and
   enabled by the service thread on its first call (measured to work). The `drain()` reset
   (`:621-622`) must then be repeated in the same mode on the same thread.
2. **The non-threaded test path breaks unless it uses the same thread.** `exl3_ram_miss_pump`
   (`:1467-1472`) and the test-only `read_rows_once` functions (`:673-756`) drive the ring from the
   caller's thread; a tier pumped on one test thread and later threaded would change issuer.
3. **`DEFER_TASKRUN` needs the owner to make regular kernel entries.** Measured: spinning on CQ
   memory never sees a completion. Today this is satisfied, because `RowReader::read` always
   enters through `io_uring_submit_and_wait` (`:600`) or `io_uring_wait_cqe` (`:613`) while
   reads are outstanding, and returns with the ring empty (`:351`). The service's idle spin
   (`_mm_pause` at `:1788`) polls the mailbox, not the ring, and nothing is in flight then.
   **It stops being satisfied for the plan's asynchronous service** (Task 4 and after,
   "a new asynchronous service interface may return with owned work outstanding"): a
   `progress()` that only inspects the CQ, or a thread that spins on the mailbox while reads
   are in flight, would hang with DEFER_TASKRUN. `progress()` must call
   `io_uring_submit_and_wait(…, 0)`/`io_uring_get_events` every iteration.
4. **The dedicated thread is compatible**, provided the ring is created or enabled on it.
   Note the doorbell spin thread and the service thread are different threads from the one
   that built the ring today.

**Benefit, measured [M]:** none visible. Same drive-bound loop as §3, thread CPU per pass
(0.87 GB), p50 of each process's 3 passes, three processes per arm, first batch (bounce on
node 1, which the later matrix showed to be 4 KiB-backed; the first batch did not print it): plain `READ` 27.6, 28.8, 36.9 ms against `SINGLE_ISSUER|DEFER_TASKRUN`
+ `READ` 27.2, 38.8, 39.0 ms; `READ_FIXED` 12.2, 13.3, 16.4 ms against SI/DTR + `READ_FIXED`
11.4, 15.6, 17.7 ms. Wall time 245.1-246.7 ms (p50 per process) in all. The ranges overlap. The probe issues
only 128 completions per pass on one drive, so it cannot show a per-completion saving; it
does show that the flags are not the lever for large, drive-bound reads.

**Recommendation:** do not adopt `SINGLE_ISSUER`/`DEFER_TASKRUN` in the native service
in this phase. The benefit is not visible, and the cost is an ownership refactor plus a
standing hazard for the asynchronous `progress()` design. If the asynchronous service later
wants them, use `R_DISABLED` + enable on the service thread and make `progress()`
enter the kernel every iteration.

### 2.4 `IOPOLL` and `SQPOLL` (optional experiments in the plan)

- `IORING_SETUP_IOPOLL`: the ring initialises [M], but the NVMe driver has `poll_queues=0`
  and every `queue/io_poll` is 0 [S], so polled reads to these drives would be refused.
  Enabling poll queues is a privileged module-parameter change: **[U]**, not attempted.
- `IORING_SETUP_SQPOLL`: **not tested.** It dedicates a spinning kernel thread on a shared box
  where cores 64-71 are production's; I did not spend one to test a gain the plan labels
  optional. Result: undetermined.

## 3. Registered bounce: cost, fallback, memory, benefit

Probe `uring_probe read`: `/mnt/nvme0/dsv41_flash/model-00001-of-00041.safetensors` (6.5 GB),
O_DIRECT, 128 random 4 KiB-aligned reads of 6.5 MiB per pass (about half of a two-root row's extent),
queue depth 16, the reader's own `submit_and_wait(1)` loop, same offsets in every arm, one warm-up pass
discarded, 3 passes per process, 2-3 rounds (3 in the first batch, 2 in the later ones). **Thread CPU** is `getrusage(RUSAGE_THREAD)` user+system
of the submitting thread (includes page pinning done in `io_uring_enter`), per pass of 0.872 GB.

### 3.1 Measured

Thread CPU per pass, p50 of 3 passes, round 1 / round 2 (ms). Wall time was **244.9-246.7 ms
in every process** (3.54-3.56 GB/s), so no arm was faster end to end.

| bounce backing | plain `READ` | `READ_FIXED` | fixed saves |
|---|---|---|---|
| node 0, default THP (THP=always; 102 MiB huge) | 9.2 / 10.9 | 13.2 / 12.7 | **none** (fixed slower) |
| node 0, `MADV_HUGEPAGE`, 2 MiB aligned (104 MiB huge) | 9.1 / 9.5 | 7.4 / 7.4 | ~2 ms |
| node 0, `NOHUGEPAGE` | 55.5 / 52.8 | 25.5 / 24.4 | ~28-30 ms |
| node 1, default (0 KiB huge) | 29.4 / 33.0 | 16.1 / 16.4 | ~13-17 ms |
| node 1, `NOHUGEPAGE` | 36.2 / 31.1 | 17.7 / 17.5 | ~14-19 ms |
| node 1, `MADV_HUGEPAGE` (52 of 104 MiB huge) | 35.6 / 37.3 | 22.3 / 23.6 | ~13-14 ms |

The pass-to-pass spread inside a process is up to ±20 % (min-max in the raw output).
I cannot explain why fixed is slower than plain in row 1 but faster in row 2, both fully
THP-backed; the alignment differs (row 1's region is 4 KiB aligned, so its head and tail are 4 KiB pages).

**Converted to a production batch** (8 rows ≈ 106 MB = 0.122 of a pass; scale, not a new measurement):

| bounce backing | plain, ms/batch | fixed, ms/batch | saves |
|---|---|---|---|
| 4 KiB pages | 3.6-6.8 | 2.0-3.1 | **1.6-3.7 ms** |
| THP | 1.1-1.3 | 0.9-1.6 | −0.5 to +0.2 ms |

For scale: an 8-row within-row 1:1 batch reads in 16.2 ms (`SCHEDULING.md`, run 1) and its
scatter is ~8 × 1.6 = 12.6 ms (`MIRROR_ROWS.md:36`, `SCHEDULING.md:112`) [D].

**Initialisation and registration cost [M]** (`uring_probe cost`, median of 7; the bounce is
untouched until first use in production, so the "not faulted" rows are the relevant ones):

| size | THP-ok, not faulted | THP-ok, faulted | 4 KiB pages, not faulted | 4 KiB, faulted | unregister |
|---|---|---|---|---|---|
| 8 MiB | 0.74 ms | 0.04 ms | 1.84 ms | 0.12 ms | 0.03 ms |
| 104 MiB (≈ the bounce) | **20.0 ms** | 1.7 ms | **39.7 ms** | 3.3 ms | 0.4 ms |
| 1 GiB | 390 ms | 199 ms | 414 ms | 27.7 ms | 4-8 ms |

So a one-off 20-40 ms at start-up (registration faults the pages in), or 2-3 ms if the bounce
was pre-touched, and the same again after every ring reset (`:621-622`).

### 3.2 Expected fallback rate

**0 % by construction.** Every extent's destination is validated inside one bounce slot at
construction (`exl3_ram_miss_host.cpp:237`) and computed from `bounce_` at `:426`, so one iovec
covering `[bounce_, bounce_ + 8*slot_bytes)` contains all of them; short-read resubmits stay
inside their extent (`:426` adds `done[j]`). Fallback is all-or-nothing: 100 % if the
registration fails (a finite `RLIMIT_MEMLOCK` below ~106 MB, measured `-ENOMEM`), or 100 % after a
ring reset that does not re-register. It is not a fractional rate. The plan's "fallback rate" is
therefore a counter to assert is zero, not a quantity to measure.

### 3.3 Memory pressure

~106 MB of extra pinned, unswappable memory (0.14 % of the 75.2 GB tier), one table entry of
16,384, one iovec of ≤1 GiB, against an unlimited `RLIMIT_MEMLOCK` here [S]/[M]. The bounce is
not CUDA-registered, so the two pins do not overlap. Not a constraint on this box.

### 3.4 Is the end-to-end benefit measurable?

**Probably not, and where it could be it is conditional. Recommendation: do not build the
registered-bounce experiment now. Revisit it after Task 4, on one condition.**

- **The drive-bound loop cannot show it.** Wall time is identical (244.9-246.7 ms, all
  44 processes); the drives, not the CPU, set the pace. Registration saves CPU time
  only, and only on the submitting thread.
- **The saving depends on the bounce being 4 KiB-backed.** On THP it is zero or negative;
  on 4 KiB pages it is 1.6-3.7 ms per 8-row batch (5-13 % of a ~29 ms read+scatter batch
  run serially). The bounce's backing is not controlled by anything today (§1.5).
- **The only place it could matter is the owner thread becoming CPU-bound under Task 4.** That
  thread does the scatter (~12.6 ms per 8-row batch) plus the submit-side pinning (1-7 ms) while
  reads take ~16 ms. With 4 KiB pages the sum can exceed the read time and Task 4's overlap would
  be limited by the CPU; with registration it is ~15-16 ms, about balanced. That is a computed
  scenario from the numbers above, not a measurement of the pipeline; Task 4 will show whether the owner
  is idle-waiting or saturated.
- **Cheaper levers first.** Pinning the service thread to a node-0 core so its bounce is first-touched
  on node 0 (fully THP-backed in the probe: 9 ms/pass against 29-37 ms on 4 KiB pages, no registration,
  no ring-reset hazard) removes most of the same CPU cost. This is a candidate, with supporting
  measurements listed in §4, not a tested end-to-end result.
- **Run the experiment after Task 4 only if** (a) the Task 4 timeline shows the owner thread
  saturated (no idle wait between batches), **and** (b) the recorded bounce placement shows 4 KiB
  backing after node placement is settled. Record the fixed/plain SQE counts and re-registration after `drain()`.
  If either is false, close it as not worth it.

## 4. Placement and budget for Task 4 (the Gate)

This is a recommendation to test, not a result.

1. **Service thread and bounce on node 0.** Evidence: bounce on node 0 is fully THP-backed and cheap
   to pin [M]; the drive → bounce DMA rate does not depend on the bounce node at one drive [M]; a
   scatter whose bounce and slab are local runs 1.0× against 1.2-1.8× with remote ends [M]; the GPU is on
   node 0 [S]. Against it: node 0 must then hold the 75 GB tier too or its slabs are remote
   to the scatter thread (up to 1.55× on the write side [M]), node 0 has 93.6 GiB in total [S], and drives
   are on node 1 (two-drive remote DMA is **[U]**).
2. **Pass `cpu_core` explicitly** to `start_thread` instead of `-1` (`ops/moe/exl3_ram_miss.py:418`), and
   **record `AnonHugePages`/`numa_maps` of the bounce and the slabs** in the Task 1 baseline, since
   today the node is whatever the scheduler picked.
3. **Budget:** bounce 8 × ~13.3 MB ≈ 106 MB (Task 4 asks for two banks: ≈ 213 MB); pinned tier 75.2 GB
   unchanged; no registered buffer.
4. **Storage alone [M]:** 3.56 GB/s per mirror drive; **current SM transfer alone and both
   together are [U]** (the plan's second Task 3 bullet needs the GPU). Both mirrors together are at
   most ~7.1 GB/s of a Gen3 x16 link (15.75 GB/s spec), with the drive → GPU path crossing sockets; whether
   that adds independently is exactly what the GPU run must show.

## 5. Negative and undetermined results

Negative:

- `SINGLE_ISSUER`/`DEFER_TASKRUN` showed no CPU or wall benefit on large drive-bound reads (§2.3).
- `READ_FIXED` showed no CPU benefit, and no wall benefit, on a THP-backed bounce (§3.1).
- Wall time (244.9-246.7 ms) did not move with any of: `READ` vs `READ_FIXED`, the two ring modes, buffer
  NUMA node, THP backing. The drive is the bottleneck in every arm.
- The EXL3 eager path does not use `READ_FIXED`, contrary to the impression the plan's Task 3
  wording gives ("existing READ_FIXED … support in the general reader"): the general reader implements
  it, the EXL3 reader never engages it (`exl3_shard_row_source.py:128-131`).
- One anomalous first run of `memlock` returned `-EOPNOTSUPP` for all five registrations with an unchecked
  `queue_init`; four later runs with the check added were deterministic (`-ENOMEM` at 128 MiB). I could not
  reproduce the anomaly and do not know its cause. It does not change any conclusion.

Undetermined:

- Why `/mnt/nvme2` is x2 (root-only tools).
- GPU link state under load, SM transfer alone, storage and transfer together (GPU).
- Placement of the live pinned tier and bounce, and `RLIMIT_MEMLOCK` of the production process (no
  serving process exists now).
- `cudaHostRegister` first-touch node and MEMLOCK accounting (CUDA).
- `IOPOLL` (needs `nvme` poll queues, privileged), `SQPOLL` (not tried), ext4 stripe of `/mnt/nvme4` (root).
- Two mirrors reading at once into a remote-node bounce.
- Whether the shared general reader actually runs in `SINGLE_ISSUER|DEFER_TASKRUN` mode inside the server
  (inferred from the probe and the code, not logged).

Revision 2 additions to the negative and undetermined lists (2026-09-21):

- Negative for adoption: plain `SINGLE_ISSUER|DEFER_TASKRUN` would leave the native loop's `submit(0)` unable to reap
  completions while a row packs (§8.3); it needs `TASKRUN_FLAG` or an explicit `io_uring_get_events()`. The native service
  has no `READ_FIXED` at all and the EXL3 eager path does not engage the general reader's (§8.1). Neither is
  needed for anything measured so far.
- Negative for the pipeline's premises: "one owning thread per ring" and "creator-thread ownership" do not describe the
  native service today (§8.2).
- Not measured, by instruction: storage alone on nvme4, SM transfer alone, both together, the registered-bounce
  experiment (all designed in §9). `IOPOLL` cannot be exercised and `SQPOLL` is untried (§7.4, §9.D).
- Three things Revision 1 could not have seen and Revision 2 found: mirror B is 7 % in page cache and 47x more
  fragmented than mirror A (§7.2, §7.4); NVMe completion interrupts reach reserved cores 64-71 from submitters
  on some cores in 0-63 (§7.3); the label `nvme4` is device `nvme3n1` (§7.1).

## 6. Reproduce

Probes ran from `/dev/shm` on divix01 with `CUDA_VISIBLE_DEVICES=` and the cores stated. Sources are in the
appendices; the `read` runs used the file above. Sample commands:

```
gcc -O2 -pthread uring_probe.c -o uring_probe -luring
taskset -c 2 ./uring_probe flags
taskset -c 2 ./uring_probe limits ; taskset -c 2 ./uring_probe memlock ; taskset -c 2 ./uring_probe cost
numactl --physcpubind=20 --membind=1 ./uring_probe read <file> plain 128 3     # arms: plain fixed si sifixed
NOHUGE=1 numactl ... ./uring_probe read ...      HUGE=1 numactl ... ./uring_probe read ...
gcc -O2 numa_touch.c -o numa_touch -lnuma ; ./numa_touch 20 1024
gcc -O2 memcpy_numa.c -o memcpy_numa -lnuma ; ./memcpy_numa <core> <src node> <dst node>
```

Before a drive run: `awk '$3=="nvme0n1"{print $6}' /proc/diskstats`, wait 3 s, and require the
delta to be 0.

## 7. Topology facts added in Revision 2

All rows are 2026-09-21 reads on divix01; the command is in the "source" column. Nothing here
touched the GPU or read data from `/mnt/nvme1` or `/mnt/nvme2`.

### 7.1 Mount label, block device, PCI function: the `nvme4` trap

**There is no `nvme4` device.** `/mnt/nvme4` is a label. Its filesystem is on `/dev/nvme3n1p1`.
`ls /dev/nvme* /sys/block` lists `nvme0..nvme3` and `nvme0n1..nvme3n1`, and `/proc/diskstats`
has no `nvme4*` row [S2]. A check written as `awk '$3=="nvme4n1"' /proc/diskstats` matches nothing,
prints nothing, and reads as "0 sectors, drive idle" (or, in a delta, as "this run read 0 bytes").

| mount label | `/proc/mounts` source | `/proc/diskstats` name | PCI function | model (`lsblk`) | filesystem |
|---|---|---|---|---|---|
| `/mnt/nvme0` | `/dev/nvme0n1p1` | `nvme0n1` | 0000:86:00.0 | Samsung 990 EVO Plus 2 TB (Revision 1) | xfs |
| `/mnt/nvme1` | `/dev/nvme1n1p1` | `nvme1n1` | 0000:87:00.0 | `CT4000P310SSD8` | xfs |
| `/mnt/nvme2` | `/dev/nvme2n1p1` | `nvme2n1` | 0000:88:00.0 | Samsung 990 EVO Plus 2 TB (Revision 1) | xfs |
| **`/mnt/nvme4`** | **`/dev/nvme3n1p1`** | **`nvme3n1`** | 0000:89:00.0 | `SPCC M.2 PCIe SSD` | **ext4** |

Sources: `grep nvme /proc/mounts`; `ls -l /sys/block/`; `lsblk -o NAME,MODEL,...`. Resolve a
mount to its device with `st_dev` (`os.stat(path).st_dev` -> major:minor -> the `/proc/diskstats`
row) or `findmnt -no SOURCE <mount>`, never by the label string. **The existing tooling already does
this:** `bench_row_scheduling.py:771-796` and `native_mirror_report.py:10,27` resolve by `st_dev`, and the
`*.sh` arm scripts hard-code `[nvme4]=nvme3n1` (`task1-baseline-arms.sh:28`, `eager-cache-arms.sh:36`,
`run-native-mirror-arm.sh:33`) [C2]. A new script or an ad-hoc `awk` is where the mistake would enter.

### 7.2 What is in the page cache: the nvme4 mirror is not cold

`fincore -b -n -o RES,SIZE <root>/*.safetensors`, 41 files per root [S2]:

| tree | resident | files with any resident page | total size |
|---|---|---|---|
| `/mnt/nvme0/dsv41_flash` (mirror A, xfs) | 106,430,464 B (101.5 MiB) | 3 of 41 | 219,195,394,582 B |
| `/mnt/nvme4/dsv41_flash` (mirror B, ext4) | **15,335,759,872 B (14.3 GiB, 7.0 %)** | **23 of 41** | same |
| `/mnt/nvme2/DeepSeek-V4.1-Flash-EXL3-3.0bpw` (source) | 7,722,295,296 B (7.2 GiB) | 15 of 41 | same |

`du -sb` reports 219,273,773,785 B for all three trees (78,379,203 B more than the summed shard size: non-shard files and directory entries, not itemised) [S2]. **O_DIRECT reads are not affected by this residency**, but any
buffered access to mirror B (a `cat`, a mmap, a Python `safetensors` open that maps) hits RAM for those
files and the drive for the rest. A benchmark that reports "cold-file" numbers for mirror B has to
record this table before and after, which `task1-baseline-arms.sh` does per directory (`expert_resident_by_dir_*`).

Node memory at the same time [S2] (`/sys/devices/system/node/node{0,1}/meminfo`, `free -b`):

| | MemTotal | MemFree | FilePages | Unevictable | Mlocked |
|---|---|---|---|---|---|
| node 0 | 93.6 GiB | **64.0 GiB** | 15.7 GiB | 7.3 MiB | 0 |
| node 1 | 94.4 GiB | **1.29 GiB** | **75.1 GiB** | 22 MiB | 0 |
| system | 188.0 GiB | (`MemAvailable` 159.9 GiB) | | 30,000 kB | 0 |

Node 1 holds the page cache and almost no free memory; node 0 has 64 GiB free. **The 70.0 GiB pinned tier
(75,153,156,096 B) is 74.8 % of node 0's total, 37.2 % of the system's, and 87.9 % of what node 0 can
supply without evicting anything** (`MemFree` + `FilePages` = 79.7 GiB). So a node-0-bound tier fits only
by reclaiming node-0 cache; a default first-touch tier spills to node 1 as Revision 1 measured
(62.9 % / 37.1 % for 1 GiB touched from a node-1 core). Where the live tier landed is **[U]** (§7.9).

### 7.3 NVMe queues, interrupts, and the reserved cores

**Queue-to-CPU maps.** `/sys/block/nvmeXn1/mq/*/cpu_list` (which CPUs submit into a hardware queue) and
`/proc/irq/<n>/effective_affinity_list` (the CPU that takes that queue's interrupt), joined through
`/proc/interrupts` names `nvme<N>q<hctx+1>` [S2]. The script is Appendix F. For every submitting core
in `0-63`, where does its completion interrupt run?

| drive | hardware queues | submitter cores in 0-63 whose completion IRQ lands on a reserved core (64-71) | node-0 submitters (0-17, 36-53): IRQ node |
|---|---|---|---|
| nvme0 (mirror A) | 16 | **34, 35, 56, 59, 62** | all node 0 |
| nvme2 (source) | 16 | 27-35, 63 | all node 0 |
| nvme3 (mirror B, "nvme4") | 31 | 28-35, 63 | all node 0 |

Examples from the same maps: nvme0 hctx6 serves CPUs 34, 35, 70, 71 and its interrupt is on **core 71**
(production's doorbell spin core); nvme3 hctx11 serves 35, 71 and its interrupt is on core 71; nvme3
hctx7 (CPUs 31, 67) is on core 67. `rq_affinity` is 1 on nvme0n1 and nvme3n1 (`cat
/sys/block/*/queue/rq_affinity`), which by the kernel's documented semantics moves the completion softirq, not the hard interrupt (documentation, not tested here).

Cumulative interrupts serviced on cores 64-71 by NVMe vectors since boot (`/proc/interrupts`, 9 days
uptime): 64: 7.6 M, 65: 1.7 M, 66: 25.6 M, 67: 20.9 M, 68: 24.1 M, 69: 21.3 M, 70: 24.7 M, **71: 20.3 M**
(of which nvme3q12 alone 20.1 M) [S2]. **This does not show who submitted them:** core 71 is in its
own hardware queue's CPU list, so work running on core 71 itself produces these; the count is a reason
to look, not evidence of leakage from cores 0-63.

Caveats, all binding: `irqbalance` is **active** (`systemctl is-active irqbalance`, pid 3369), so the
effective affinity is a snapshot that can move; `default_smp_affinity` is `ff,ffffffff,ffffffff` (all
72 CPUs) [S2]; the mapping of a submitter to its queue is the kernel's (blk-mq default), not set
here. Appendix F was run twice about 25 minutes apart, with `irqbalance` active, and printed identical lists both times. **`taskset -c 0-63` keeps our threads off cores 64-71 but does not keep NVMe interrupts off them.**
A service thread on any of cores 27-35 would still take its completions on a core in 64-71 for two of the
three drives. Pinning it to a node-0 core avoids this (every node-0 submitter's IRQ is on node 0 for all
three drives). This is the second independent argument, after THP backing, for Revision 1's §4 rec. 1-2.

### 7.4 Block layer limits and file layout

| item | nvme0n1 | nvme1n1 | nvme2n1 | nvme3n1 | source |
|---|---|---|---|---|---|
| `queue/max_sectors_kb` | **512** | 256 | **512** | **256** | sysfs [S2] |
| hardware queues (`mq/`) | 16 | 32 | 16 | 31 | sysfs [S2] |
| `nr_requests` / scheduler / `io_poll` / `read_ahead_kb` | 1023 / none / 0 / 4096 | same | same | same | sysfs [S2] |
| NVMe `poll_queues` (module) | 0 | | | | `/sys/module/nvme/parameters/poll_queues` [S2] |

So a 6.5 MiB extent (Revision 1's read size; half a two-root mirror row) is split by the block layer
into about **13 requests on the Samsungs (512 KiB) and about 26 on mirror B (256 KiB)**. This is
arithmetic from `max_sectors_kb`, not an observation of the request stream. Per-drive I/O count is
therefore not comparable across the two mirrors.

**File layout differs sharply between the mirrors.** `filefrag` (FIEMAP, metadata only) over all 41
shards of each tree [S2]:

| tree | fs | total extents | per-file min / max | extent length (median / p10 / p90) |
|---|---|---|---|---|
| `/mnt/nvme0/dsv41_flash` | xfs | **66** | 1 / 3 | 4,096 / 885 / 4,981 MiB |
| `/mnt/nvme4/dsv41_flash` | ext4 | **3,112** | 16 / 255 | **16 / 8 / 184 MiB** |
| `/mnt/nvme2/DeepSeek-V4.1-Flash-EXL3-3.0bpw` | xfs | 175 | 2 / 6 | (not computed) |

`model-00020` on nvme4 starts with extents of 8, 56, 8, 16 and 200 MiB (`filefrag -v`). A 6.5 MiB read
against a 16 MiB median extent crosses a physical discontinuity often, and the ext4 file is
physically scattered while the xfs one is one or two runs. **Whether this costs anything is [U]:** the
prior sessions measured 3.3-3.5 GB/s per drive with both mirrors reading (`SCHEDULING.md`), and
Revision 1 measured only nvme0 alone (3.56 GB/s). Nobody has recorded nvme4 alone. §9.A does.

### 7.5 Filesystem parameters and file age

| item | nvme0 | nvme2 | nvme4 | source |
|---|---|---|---|---|
| type; block | xfs, 4096 | xfs, 4096 | ext4, 4096 | `/proc/mounts`; `stat -f` [S2] |
| mount options | `relatime`, `attr2,inode64,logbufs=8,logbsize=32k` | `noatime`, same xfs opts | `relatime` (defaults) | `/proc/mounts` [S2] |
| xfs geometry | agcount 72, `sunit=0 swidth=0`, reflink=1 | agcount 72, `sunit=0 swidth=0` | ext4 stripe **[U]** (`tune2fs` needs root) | `xfs_info` [S2] |
| partition start | sector 2048 (1 MiB) | 2048 | 2048 | Revision 1 §1.4 [S] |
| free / used | 757 G free, 60 % | 219 G free, 89 % | 1.4 T free, 21 % | `df -h` [S2] |
| DIO alignment | mem 4, offset 512 | same | same | Revision 1 `statx` [M] |

`/mnt/nvme1` is xfs with `sunit=8,swidth=8` (a stripe hint) and is not used. **File age:** the 41
`model-*.safetensors` in each mirror were written 2026-09-17 16:03-16:28 local time (CDT, UTC-5)
(`ls -l --time-style=long-iso`), so 3.5 days old at the audit; both are complete copies of the
source (equal `du -sb`). The mirrors are **existing, not fresh, files**; mirror B had 3,112 extents at
audit. The plan's "distinguish fresh-file tests from existing-file tests" applies: any test that
rewrites a mirror changes its layout.

**O_DIRECT is honoured on both filesystems, re-verified [M2]:** `dd if=<cold shard> bs=1M count=32
skip=2000 iflag=direct` on `model-00001` of nvme0 (xfs) and of nvme4 (ext4), `fincore` residency before
and after: `0 -> 0` on both (32 MiB read: 743 MB/s and 1.6 GB/s, single-shot, not a throughput measure).
The product code opens `O_RDONLY|O_CLOEXEC|O_DIRECT` when `direct_` (`exl3_ram_miss_host.cpp:435`) [C2], and
the general reader comment at `expert_file_reader.py:47-50` (HEAD) says its direct mode opens `O_DIRECT` unconditionally and never reads through the
page cache.

### 7.6 PCIe, IOMMU, GPU state

| item | value | source |
|---|---|---|
| Links | as Revision 1 §1.3: nvme0 8 GT/s x4, nvme1 x4, **nvme2 x2 (capable x4; root port `85:02.0` x2 of x4)**, nvme3 x4; every port's max is 8 GT/s | `/sys/bus/pci/devices/0000:*/current_link_{speed,width}`, `max_link_*` [S2]; identical to Revision 1 |
| GPU link at audit | 2.5 GT/s x16 (`37:00.0` and its port `36:00.0`, whose max is 8 GT/s x16); `nvidia-smi` gen current 1, gpumax 5, hostmax 3; P8; persistence on | sysfs; `nvidia-smi --query-gpu=pcie.link.gen.*` [S2] |
| GPU BAR1 | 256 MiB (`nvidia-smi -q`), so Resizable BAR is not in effect | [S2] |
| IOMMU | `intel_iommu=on iommu=pt`; the IOMMU group of nvme0 (`86:00.0`) and of the GPU (`37:00.0`) are both type `identity` (no translation on their DMA) | `/proc/cmdline`; `/sys/kernel/iommu_groups/<g>/type` [S2] |
| CPU caches | 36 x 1 MiB L2 (36 MiB total), 2 x 24.75 MiB L3 (49.5 MiB total); no `cpufreq` directory is exposed (`ls /sys/devices/system/cpu/cpu0/cpufreq` fails), so the governor is not observable | `lscpu`; sysfs [S2] |
| GPU L2 | **not queried**: `nvidia-smi` has no field for it and a CUDA call would use the GPU. `CLAUDE.md` says ~128 MB and the plan requires verifying it; §9.B sizes the working set past any plausible value | [U] |

The GPU link was at gen 1 at every idle read. Whether it reaches Gen3 under the SM path, and how
long the ramp takes, is a GPU measurement (§9.B), and the idle value must not be reported as the
link's capability.

### 7.7 Pinned-memory budget and registration limits, restated

The 70 GiB tier is the recorded run's (Revision 1 §1.6: 71,680 MB requested -> 5,644 rows,
75,153,156,096 B, six slabs, one `cudaHostRegister` per slab). New in Revision 2:

- **There is no `mlock` accounting to run into here:** `RLIMIT_MEMLOCK` is unlimited (Revision 1 §1.7). The
  budget constraint is physical placement (§7.2), not a limit.
- **`Unevictable` was 30,000 kB and `Mlocked` 0 kB** with no serving process (`/proc/meminfo`) [S2]. A live
  `cudaHostRegister`'d tier would appear there; recording it during the first production or arm run
  after this is the cheap way to obtain the live budget.
- **The bounce grew.** 213 MB (16 slots), not 106 MB (§0.1). Registering it with io_uring would be one
  iovec (limit 1 GiB, Revision 1 §1.7). Registration cost re-measured [M2] with Revision 1's `cost` probe
  under `numactl --physcpubind=2 --membind=0`, one run, each cell the median of 7 as before: 104 MiB not-faulted 21.2 ms (THP-ok) / 42.6 ms (4 KiB pages),
  matching Revision 1's 20.0 / 39.7; **256 MiB not-faulted 59.9 ms (THP-ok) / 106.6 ms (4 KiB pages); faulted 11.7 / 9.4 ms; unregister about 1 ms**.
  A 203 MiB (16 x 13.3 MB) bounce sits between the 104 and 256 MiB rows, so expect roughly 45-60 ms (THP) or
  85-107 ms (4 KiB) once at start-up and again after each ring reset; that is an interpolation, not a run.

### 7.8 Placement and budget for Task 4 (gate): what changes from Revision 1 §4

Revision 1's recommendation stands. The added evidence and the one added requirement:

1. **Service thread on a node-0 core.** For: bounce THP-backed on node 0 (Revision 1), local row scatter (1.0x against
   1.2-1.8x, Revision 1), GPU on node 0, and **node-0 submitters take their NVMe completion interrupts on node-0 cores
   for all three drives, never on 64-71 (§7.3)**. Against: node 0 has 64.0 GiB free against a 70.0 GiB
   tier, so the tier's slabs will not all be node-0 unless node-0 cache is reclaimed (`numactl --membind` /
   `MPOL_BIND` reclaims; default first-touch spills, Revision 1 §1.5). **Two-drive DMA into a remote or
   local bounce is still [U]** and is now part of §9.C.
2. **Pass `cpu_core` explicitly.** The production call is `host.start_thread(fatal_wait_s=...)`
   (`srt/layers/moe/exl3_ram_miss.py:383`), so `cpu_core=-1` inherits the launcher's affinity. The wrapper
   and the native `start_thread` now refuse 64-71 and the native one warns when an inherited mask contains
   them (`ops/moe/exl3_ram_miss.py:448`, `exl3_ram_miss_host.cpp:2362+`) [C2], which stops the mistake but
   does not choose a node.
3. **Budget:** bounce 213 MB (two banks) plus the unchanged 75.2 GB tier; no registered buffer in Task 4.
   The gate's "resource budget" is: 70.0 GiB pinned host tier, 0.2 GiB bounce, one service thread on a
   node-0 core, no core in 64-71, both mirrors' rows read with O_DIRECT (no page-cache growth).

### 7.9 Undetermined after Revision 2

- Where the live pinned tier landed (§7.2): no serving process exists; `numa_maps` of a 75 GB process walks its whole
  address space and can stall it, so I did not read one from a running arm and I ask before doing so.
- GPU link under load, GPU L2 size, `cudaHostRegister` first-touch and page-locking behaviour (need a CUDA context).
- Two mirrors at once into a remote bounce; nvme4 alone (§9.A).
- Why `88:00.0` is x2, ext4 stripe of `/mnt/nvme4`, `dmesg` for AER (root-only).
- Stability of the IRQ map under `irqbalance` (one snapshot).

## 8. Reader audit at `0752c0c0e6`: `READ_FIXED`, `SINGLE_ISSUER`, `DEFER_TASKRUN`

Two rings, as Revision 1 §2 says. Citations are `git show HEAD:<path>` line numbers (blobs in §0.1).
`G:` is `csrc/io/uring_file_reader.cpp`; `N:` is `csrc/moe/exl3_ram_miss_host.cpp`.

### 8.1 Support matrix

| feature | general reader (G) | native service (N) | works on this kernel/liburing |
|---|---|---|---|
| `READ_FIXED` (registered buffers) | **Implemented.** 1,024-slot sparse table registered at construction (`G:26,96`), registrations chunked to 1 GiB (`G:27,142-177`), `find_buffer_` (`G:231,624`), `io_uring_prep_read_fixed` only when the whole destination lies inside one registration, else `io_uring_prep_read` (`G:379-381`). **Engaged only when a caller registers**: `ExpertFileRowReader.register_destinations` (`expert_file_reader.py:188`, from `expert_stream.py:212`). The EXL3 row source's `register_destinations` returns 0 (`exl3_shard_row_source.py:128-131`), so EXL3 eager reads are `IORING_OP_READ` | **Not supported.** The only opcode is `io_uring_prep_read` (`N:807`); no `io_uring_register_*` anywhere in the file (`grep`: 0 hits) | yes (Revision 1 [M]) |
| `SINGLE_ISSUER` | **Requested**, fallback to flags 0 only on `-EINVAL` (`G:45-58`) | **Not requested**: `io_uring_queue_init(depth, &ring_, 0)` (`N:482`, re-created `N:1079`) | yes (Revision 1 [M]) |
| `DEFER_TASKRUN` | **Requested** with `SINGLE_ISSUER` (`G:49`) | Not requested | yes; requires `SINGLE_ISSUER` (Revision 1 [M]) |
| `TASKRUN_FLAG`, `COOP_TASKRUN`, `SQPOLL`, `IOPOLL` | absent (`grep -c` = 0 in both files) | absent | `IOPOLL` ring initialises; NVMe `poll_queues=0` means polled reads cannot work (§7.4) |

The plan's wording, "existing `READ_FIXED` ... support in the general reader **versus** native service", has a
literal answer: the general reader has it and does not use it for EXL3; the native service has none. This
is a finding, not a gap: nothing measured so far says the native service needs it (Revision 1 §3, §8.4 below).

### 8.2 Creator-thread ownership

**General reader: enforced.** `owner_(current_thread_id())` is captured in the constructor (`G:89`); every public
call runs `check_owner_()` (`G:586-593`) and `enter_()` (`G:597`), and a call from another thread throws
`"io_uring file reader belongs to thread ..."`. Its waits are `io_uring_submit_and_wait(&ring_, 1)`
(`G:389`) and `io_uring_wait_cqe` (`G:519`), both blocking entries with `GETEVENTS`, so `DEFER_TASKRUN`
never leaves a completion stranded. The shared instance belongs to whichever thread first calls
`get_shared_uring_file_reader` (`ops/io/uring_file_reader.py:168-173`, Revision 1 §2.3). **Whether the
running server's shared reader is in `SINGLE_ISSUER|DEFER_TASKRUN` or in the fallback mode is not logged
and not observed** (the fallback is silent).

**Native service: not enforced; creator != driver.**

| step | thread | where |
|---|---|---|
| ring created | the Python caller of `exl3_ram_miss_open` | `N:1935` -> `RamTier::open` `N:1415-1416` -> `RowReader::open` `N:433,482` (`N:1957` `if (!tier->open()) return -1`) |
| ring driven | the `RamThread` service thread | `std::thread` `N:2210`; `run()` `N:2259`; `pump_demand` -> `reader_.read` `N:1806` |
| ring re-created after a failed read | whichever thread called `read()`, in practice the service thread | `drain()` `N:1065-1079` |
| ring destroyed | whichever thread drops the last reference (not traced; expected the Python side at `stop`/`close`) | destructor `N:411-413` |
| ring also driven by | the caller thread, in the non-threaded test paths | `exl3_ram_miss_pump` `N:1969`, `exl3_ram_miss_read_rows*` `N:1132,1159,1197` |

So the plan's constraint "One owning CPU thread drives each ring" is satisfied by convention only, and
"preserve creator-thread ownership" is **not a property the native service has today**. It would be one
the pipeline has to acquire, on the thread that will drive it (§8.4).

### 8.3 Regular kernel entries for deferred task work: new measurement

Revision 1 measured that a spinner on CQ memory never sees a `DEFER_TASKRUN` completion. Which *calls* do enter
usefully was not measured. `defer_probe` (Appendix E, CPU-only, `taskset -c 2`, kernel 6.12 + liburing 2.12)
submits a read on an empty pipe, has a helper thread write to it at 50 ms, waits 120 ms, then performs **one**
call on the owner thread and reports `io_uring_cq_ready()` before and after [M2]:

| call on the owner thread | `flags=0` | `SI\|DTR` | `SI\|DTR\|TASKRUN_FLAG` |
|---|---|---|---|
| `cq_ready()` only (no syscall) | 1 (already there) | **0** | 0 |
| `io_uring_submit()`, SQ empty | 1 | **0** | 1 |
| `io_uring_submit()` with a NOP prepared | 1 -> 2 | **0 -> 1 (the NOP only; the pipe read stays hidden)** | 0 -> 2 |
| `io_uring_submit_and_wait(0)` | 1 | **0** | 1 |
| `io_uring_peek_cqe()` | 1 | **-EAGAIN** | 1 |
| `io_uring_get_events()` | 1 | **0 -> 1** | 0 -> 1 |
| `io_uring_submit_and_wait(1)` | 1 | **0 -> 1** | 0 -> 1 |

Only a call that enters the kernel **with `IORING_ENTER_GETEVENTS`** runs deferred work, and liburing adds
that flag on the non-waiting paths only when the ring advertises pending task work, which needs
`IORING_SETUP_TASKRUN_FLAG`. (I did not disassemble liburing; this is the observed behaviour of 2.12.)
Each cell is a single run of a deterministic probe, not a statistic.

**What this means for the native loop** [C2 + M2, an inference, not a run of the service]:
`RowReader::read()` calls `reap(ready)` every iteration (`N:822`), and `reap` calls `submit(ready ? 0 : 1)`
(`N:841`). With `ready` (a fully read row waiting to pack) that is `io_uring_submit` (`N:1057`), which under
plain `SI|DTR` reaps nothing new; the completions it should have reaped wait until an iteration finds no
ready row and blocks in `submit_and_wait(1)`. While rows pack, each packing pass (about 1.6 ms per row,
`MIRROR_ROWS.md:36`) would run with the CQ frozen and SQ credit unreturned, so storage would starve.
Flags 0 has no such stall today. `DEFER_TASKRUN` therefore needs one of: `IORING_SETUP_TASKRUN_FLAG` at
setup, or an explicit `io_uring_get_events()` before each `for_each_cqe`.

### 8.4 What Task 4/5 must preserve if it adopts either flag (checklist; nothing adopted)

The recommendation from Revision 1 §2.3 stands: **do not adopt `SINGLE_ISSUER`/`DEFER_TASKRUN` or
`READ_FIXED` in the native service for Task 4**, because no wall-time benefit was measured on large
drive-bound reads and the flags add an ownership hazard. If the asynchronous service later wants them, the
audit above turns into these acceptance conditions:

1. The ring is created, or `io_uring_enable_rings`-ed (`IORING_SETUP_R_DISABLED`, measured to work in Revision 1),
   **on the thread that will drive it**; `RowReader::open` (`N:433`) currently runs on the Python thread and
   reports failure synchronously through `exl3_ram_miss_open` (`N:1957`), which constrains where creation can move.
2. Every progress step of the asynchronous interface enters the kernel with `GETEVENTS`
   (`io_uring_get_events`, or `TASKRUN_FLAG`), including when a row is ready to pack and when the idle loop is spinning
   (`_mm_pause`, `N:2290`) with reads still owned by the service. A thread that spins on the mailbox while the ring
   holds work would hang.
3. `drain()`'s ring reset (`N:1078-1079`) repeats the same setup on the same thread; a reset that quietly
   falls back to flags 0 is a silent loss of the mode, and for `READ_FIXED` a silent loss of the registration.
4. The caller-thread entry points (`N:1132,1159,1197,1969`) and the test hooks run on the thread that owns
   the ring, or the tests fail with `-EEXIST` on the first submit (measured, Revision 1).
5. A counter of SQEs by opcode (`READ_FIXED` vs `READ`) and by ring mode, logged once, because a registration
   or flag failure changes no result and is otherwise invisible (the general reader has this problem today, §8.2).
6. `READ_FIXED` only: one iovec over `[bounce_, bounce_ + 16 * slot_bytes)` registered after `N:482`,
   re-registered after `N:1079`, unregistered before `free(bounce_)` (`N:411-413`); registration failure
   falls back with a logged reason. Every extent already lies inside one slot (`N:253`), so the fallback
   rate is 0 by construction; it is a counter to assert zero, not a rate to measure (Revision 1 §3.2).

## 9. The measurements Task 3 still owes: design only, none run

**Nothing in this section has been run.** The plan's second bullet (storage alone, current SM transfer alone,
their simultaneous execution) and its fourth (the registered-bounce decision) need the GPU and/or heavy drive
I/O. Revision 2 was under a standing instruction not to use the GPU (timed arms may be running) and not to
run drive benchmarks on nvme0/2/4. What follows is what I would measure and the conditions that would make
a number valid, so the run can be scheduled without further design.

### 9.0 Preconditions common to every arm

- Schedule GPU time with the owner of the machine and take `cc-gpu.lock` through `$ANA/gpu-run.sh`. **Verify the
  wrapper and `$ANA` exist first**; the plan's own constraint. Do not start or stop production. The storage-only arm (§9.A) needs neither and is
  a heavy-I/O run: it needs the drives to be otherwise idle.
- CPU jobs `taskset -c 0-63`, `OMP_NUM_THREADS=16 MKL_NUM_THREADS=16` (plan) or fewer; **never 64-71**. Confirm with
  `ps -eLo pid,psr,comm | awk '$2>=64'` that none of ours is there. Because NVMe interrupts can still land on 64-71
  (§7.3), record the effective IRQ map before and after each arm (Appendix F) and the rate of change of
  `/proc/interrupts` on cores 64-71 across the run.
- Before and after every arm: `/proc/diskstats` **by resolved device** (§7.1: `nvme0n1`, `nvme3n1`, never a label),
  `fincore` on both mirrors, `nvidia-smi --query-gpu=pcie.link.gen.current,pstate,memory.used`, `/proc/meminfo` and
  per-node `meminfo`, `pgrep` for a foreign process. Require 0 sectors in 3 s on the drives before starting, as Revision 1 §6.
- **Validity gates, each abort-on-fail:** diskstats bytes read equals bytes issued within 1 % (else a foreign reader
  or a buffered path is present); implied per-drive bandwidth at most the link ceiling (3.94 GB/s for x4, 1.97 GB/s
  for the x2 drive; a value above it means a cache, not a drive) and GPU transfer at most the Gen3 x16 15.75 GB/s;
  the GPU link generation read at the end of each GPU arm is 3, not 1; no page-cache growth in the mirrors' `fincore`.
- Arm order randomised, each condition repeated at least 5 times, interleaved (A B B A) so drift is not read as a
  difference; report every repetition, the median, and min-max, never a mean of two.
- Record the state the plan asks for: file age/state (§7.5), cache residency (§7.2), IOMMU (§7.6), versions
  (§1.1), IRQ map (§7.3), and that the drives are at their negotiated Gen3 (nvme2 at x2) (§7.6).

### 9.A Storage alone (heavy drive I/O; no GPU)

- **Question.** What does each mirror deliver alone and both together with the production read geometry, and what
  does the submitting thread spend? Revision 1 has nvme0 alone (3.56 GB/s, 90 % of the Gen3 x4 ceiling) and
  `SCHEDULING.md` has both together; **nvme4 alone has never been recorded**, and mirror B's layout is 47x more
  fragmented than A's (§7.4).
- **Arms.** nvme0 alone; nvme4 alone; both from one thread; both from two threads. O_DIRECT, 6.5 MiB extents
  (`LEN = 6,815,744` in `uring_probe`), queue depth 16 per drive, 128 reads per pass, at least 5 passes per arm after
  one discarded, offsets random over the whole 41-file set (204 GiB, so no drive-side reuse), same seed in every arm.
  Use the existing harnesses (`bench_row_scheduling.py`, `uring_probe read`) rather than a new tool; extend `uring_probe`
  to take two files if needed. Bounce sized as production, 213 MB (16 slots), on node 0, then on node 1.
- **Record:** wall, GB/s, thread CPU (`getrusage(RUSAGE_THREAD)`), diskstats delta by device, IRQ map, and for each
  arm the submitter's core and node. Expected footprint: 0.87 GB per pass, about 13 GB per arm at 15 passes,
  well under 200 GB for the whole matrix, minutes of wall time.
- **Decision it feeds:** the Task 4 resource budget (bytes/s the storage side can supply) and whether mirror B's
  fragmentation needs a fix (re-copy contiguously with `fallocate`), which would be a separate, labelled experiment.

### 9.B Current SM transfer alone (GPU)

- **Question.** What does the production SM (GPU-pull) gather deliver from host to device, at an honest working
  set, on this box's Gen3 link and NUMA layout, and what does the copy engine deliver as a separately labelled comparison?
- **Tool.** `benchmark/kernels/moe/benchmark_expert_cache_transfer.py --backend gpu` (production path; add `--cuda-graph`
  to match capture). **Its defaults do not qualify:** 128 source experts x 65,536 B x 6 tensors is about 50 MB, less than
  any plausible L2 (§7.6, CLAUDE.md's "size the working set past L2"). Set `--source-experts` and `--row-bytes` so the
  pinned source is at least **4 GiB**, and draw rows without replacement across it each iteration (so successive
  iterations do not re-read one row). Take the six tensor segment sizes from the real EXL3 layout, not `--row-bytes 65536`
  (an NVFP4 row); the layout is the per-row `slot_bytes` of about 13.3 MB (`MIRROR_ROWS.md:42`) split into its six segments,
  which must be read from the loaded table.
- **Arms.** rows per submission in {8, 32, 48} (8 is the native batch, 48 the `bench_mirror_rows` default) x pinned
  source on node 0, node 1, interleaved. `--backend dma` is a **separate arm, labelled "copy engine, not the
  production SM path"**, never merged into the SM figures (plan).
- **Placement must be verified, not assumed.** Allocate under `numactl --membind=N` and touch the memory before
  `cudaHostRegister` (Revision 1: `cudaHostRegister`'s first-touch is not established). Confirm with `move_pages`
  or `/proc/self/numa_maps` **of the benchmark's own process** before any timing. Node 1 has 1.29 GiB free (§7.2), so a
  4 GiB node-1 source reclaims page cache; record that.
- **Link.** Idle the GPU is P8 / Gen1 (§7.6). Run at least 2 s of traffic untimed to bring it up, sample
  `pcie.link.gen.current` once at the end, abort the arm if it is not 3.
- **Record:** per-iteration CUDA-event time (p50/p95/p99), GB/s, gen, pstate, source and destination node, whether the
  destination cache rows are reused. Expected footprint: 4 GiB pinned host + the destination rows, a few seconds per
  condition, about 10 minutes total. **Needs the GPU lock and a scheduled slot.**

### 9.C Storage and SM transfer simultaneously (GPU + heavy drive I/O)

- **Question.** Do the two add, or do they contend? Contention is possible in the drive-to-host DMA writes into
  memory, the socket interconnect (drives and the GPU are on different nodes, §1.2), the memory controllers of
  the node holding the slabs, and the cores that run the reader and the gather.
- **Design.** Two processes, started together on a barrier, each reporting per-iteration timestamps on the
  same monotonic clock: the reader process (§9.A geometry, CPU-only, no CUDA) and the GPU process (§9.B). Only the
  window where both are running counts, at least 5 s. Placement matrix, 5 conditions: (bounce node, pinned-source node) in
  {0,1}^2 with the reader on a node-0 core, plus (bounce 1, source 0) with the reader on a node-1 core that avoids 27-35
  and 63 (§7.3).
- **Third load, kept separate:** a variant where the reader thread additionally `memcpy`s each completed row from the bounce
  to a second host buffer (the production scatter, about 1.6 ms per 13.3 MB row), because that is the actual
  memory-bandwidth competitor. Label the two variants "storage only" and "storage + scatter".
- **Report** each side's throughput and p99 latency alone, together, and the ratio together/alone. The decisive
  numbers are storage GB/s and SM p99 under load. **A ratio near 1 is a result** ("they add"), not a failed experiment.
- **Abort** if diskstats shows more than 1 % foreign bytes, if a production process appears, if the GPU link is not
  Gen3 at the end, or if any of our threads is found on cores 64-71.
- **Needs the GPU lock, a scheduled slot, and the drives idle.** Roughly 30-45 min including repetitions.

### 9.D The registered-bounce decision after Task 4

Revision 1 §3.4 says do not build it now and revisit under two conditions. Revision 2 keeps that and adds what the
experiment must record if it is run:

- **Trigger:** the Task 4 timeline shows the owner thread saturated (no idle wait between batches) **and** the recorded bounce
  placement shows 4 KiB backing after node placement is settled (§7.8 rec. 1 may make the second moot).
- **Arms (native service, same binary, flag-selected):** plain `IORING_OP_READ` (today); one registered iovec over the 16-slot
  bounce; the same with a 2 MiB-aligned `MADV_HUGEPAGE` bounce and no registration (the cheaper lever); registered +
  `SI|DTR|TASKRUN_FLAG` only if the ownership work of §8.4 is done.
- **Record per arm:** registration time at start-up and after each ring reset (expect roughly 45-107 ms once, §7.7);
  fixed and plain SQE counters and their sum equalling submitted reads; the fallback count (must be 0, and 100 %
  if a reset dropped the registration); owner-thread CPU per batch and its idle-wait fraction; `AnonHugePages` of the
  bounce; `RLIMIT_MEMLOCK` and pinned bytes (`Unevictable`); end-to-end decode latency p50/p99 against the matched Task 1
  baseline.
- **Adopt only if** the end-to-end p50 difference exceeds the baseline's own repeat spread. Task 1 already records that two-bank
  itself moved throughput about 3 % rather than the 35 % first claimed (`d432533a61`), which is the scale a CPU-side
  saving of 1.6-3.7 ms per 8-row batch (Revision 1 §3.1) has to be measured against.
- **`IOPOLL`/`SQPOLL`** stay optional and measured, never assumed. `IOPOLL` cannot be exercised here (`poll_queues=0`;
  enabling it is a privileged module-parameter change) and is recorded as such. `SQPOLL` would dedicate a spinning
  kernel thread: only on a core in 0-63, only as its own labelled arm, and not before the ownership work.

<details><summary>Appendix A: uring_probe.c</summary>

```c
// CPU-only io_uring probe for divix01 (Task 3, TOPOLOGY.md). Never touches the GPU.
//   gcc -O2 -pthread uring_probe.c -o uring_probe -luring
//   ./uring_probe flags | limits | memlock | cost | read <file> <arm> <nreads> <reps>
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <liburing.h>
#include <pthread.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

static double now_s(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec * 1e-9;
}
static double cpu_s(void) {
  struct rusage ru;
  getrusage(RUSAGE_THREAD, &ru);
  return ru.ru_utime.tv_sec + ru.ru_utime.tv_usec * 1e-6 + ru.ru_stime.tv_sec + ru.ru_stime.tv_usec * 1e-6;
}
static void* aligned(size_t n, int nohuge) {
  void* p = NULL;
  if (posix_memalign(&p, 4096, n)) return NULL;
  if (nohuge) madvise(p, n, MADV_NOHUGEPAGE);
  return p;
}

// ---------- flags ----------
static struct io_uring g_ring;
static int g_rc;
static void* nop_from_thread(void* arg) {
  (void)arg;
  struct io_uring_sqe* sqe = io_uring_get_sqe(&g_ring);
  io_uring_prep_nop(sqe);
  g_rc = io_uring_submit(&g_ring);
  return NULL;
}
static void* enable_then_nop(void* arg) {
  (void)arg;
  int rc = io_uring_enable_rings(&g_ring);
  if (rc < 0) { g_rc = rc; return NULL; }
  struct io_uring_sqe* sqe = io_uring_get_sqe(&g_ring);
  io_uring_prep_nop(sqe);
  g_rc = io_uring_submit_and_wait(&g_ring, 1);
  return NULL;
}
static void try_init(const char* name, unsigned flags) {
  struct io_uring r;
  struct io_uring_params p;
  memset(&p, 0, sizeof p);
  p.flags = flags;
  int rc = io_uring_queue_init_params(64, &r, &p);
  printf("init %-44s rc=%d (%s)\n", name, rc, rc < 0 ? strerror(-rc) : "ok");
  if (rc == 0) io_uring_queue_exit(&r);
}
struct wr { int fd; };
static void* delayed_write(void* a) {
  usleep(50000);
  char c = 'x';
  (void)!write(((struct wr*)a)->fd, &c, 1);
  return NULL;
}
static void deferral_probe(const char* name, unsigned flags) {
  struct io_uring r;
  struct io_uring_params p;
  memset(&p, 0, sizeof p);
  p.flags = flags;
  if (io_uring_queue_init_params(8, &r, &p) < 0) { printf("deferral %s: init failed\n", name); return; }
  int pf[2];
  (void)!pipe(pf);
  char buf[1];
  struct io_uring_sqe* sqe = io_uring_get_sqe(&r);
  io_uring_prep_read(sqe, pf[0], buf, 1, 0);
  io_uring_submit(&r);  // pipe is empty: the read pends in the kernel
  struct wr w = {pf[1]};
  pthread_t t;
  pthread_create(&t, NULL, delayed_write, &w);
  double t0 = now_s(), seen = -1;
  while (now_s() - t0 < 0.5) {  // spin on CQ memory only: no io_uring_enter
    if (io_uring_cq_ready(&r) > 0) { seen = now_s() - t0; break; }
  }
  pthread_join(t, NULL);
  if (seen >= 0) {
    printf("deferral %-30s CQE visible from memory alone after %.1f ms (write was at 50 ms)\n", name, seen * 1e3);
  } else {
    printf("deferral %-30s NO CQE in 500 ms of spinning on CQ memory; ", name);
    struct io_uring_cqe* cqe;
    int rc = io_uring_wait_cqe(&r, &cqe);  // enters the kernel with GETEVENTS
    printf("after io_uring_wait_cqe (kernel entry): rc=%d res=%d\n", rc, rc == 0 ? cqe->res : 0);
  }
  close(pf[0]); close(pf[1]);
  io_uring_queue_exit(&r);
}
static void run_flags(void) {
  printf("liburing %d.%d\n", IO_URING_VERSION_MAJOR, IO_URING_VERSION_MINOR);
  try_init("flags=0", 0);
  try_init("SINGLE_ISSUER", IORING_SETUP_SINGLE_ISSUER);
  try_init("SINGLE_ISSUER|DEFER_TASKRUN", IORING_SETUP_SINGLE_ISSUER | IORING_SETUP_DEFER_TASKRUN);
  try_init("DEFER_TASKRUN alone", IORING_SETUP_DEFER_TASKRUN);
  try_init("SINGLE_ISSUER|DEFER_TASKRUN|R_DISABLED", IORING_SETUP_SINGLE_ISSUER | IORING_SETUP_DEFER_TASKRUN | IORING_SETUP_R_DISABLED);
  try_init("IOPOLL (ring only; needs poll queues for reads)", IORING_SETUP_IOPOLL);
  pthread_t t;
  // (a) plain ring created here, driven from another thread: the native service today
  io_uring_queue_init(8, &g_ring, 0);
  pthread_create(&t, NULL, nop_from_thread, NULL); pthread_join(t, NULL);
  printf("cross-thread submit, flags=0 ring           : rc=%d\n", g_rc);
  io_uring_queue_exit(&g_ring);
  // (b) SINGLE_ISSUER|DEFER_TASKRUN ring created here, submitted from another thread
  struct io_uring_params p; memset(&p, 0, sizeof p);
  p.flags = IORING_SETUP_SINGLE_ISSUER | IORING_SETUP_DEFER_TASKRUN;
  io_uring_queue_init_params(8, &g_ring, &p);
  pthread_create(&t, NULL, nop_from_thread, NULL); pthread_join(t, NULL);
  printf("cross-thread submit, SINGLE_ISSUER ring     : rc=%d (%s)\n", g_rc, g_rc < 0 ? strerror(-g_rc) : "ok");
  io_uring_queue_exit(&g_ring);
  // (c) R_DISABLED: created here, enabled and driven by the service-like thread
  memset(&p, 0, sizeof p);
  p.flags = IORING_SETUP_SINGLE_ISSUER | IORING_SETUP_DEFER_TASKRUN | IORING_SETUP_R_DISABLED;
  io_uring_queue_init_params(8, &g_ring, &p);
  pthread_create(&t, NULL, enable_then_nop, NULL); pthread_join(t, NULL);
  printf("R_DISABLED, enabled+driven by other thread  : rc=%d (%s)\n", g_rc, g_rc < 0 ? strerror(-g_rc) : "ok");
  struct io_uring_sqe* sqe = io_uring_get_sqe(&g_ring);
  if (sqe) { io_uring_prep_nop(sqe); int rc = io_uring_submit(&g_ring); printf("  ...then the creating thread submits       : rc=%d (%s)\n", rc, rc < 0 ? strerror(-rc) : "ok"); }
  io_uring_queue_exit(&g_ring);
  deferral_probe("flags=0", 0);
  deferral_probe("SINGLE_ISSUER|DEFER_TASKRUN", IORING_SETUP_SINGLE_ISSUER | IORING_SETUP_DEFER_TASKRUN);
}

// ---------- limits ----------
static void run_limits(void) {
  struct rlimit rl; getrlimit(RLIMIT_MEMLOCK, &rl);
  printf("RLIMIT_MEMLOCK soft=%llu hard=%llu (~0 = unlimited)\n", (unsigned long long)rl.rlim_cur, (unsigned long long)rl.rlim_max);
  unsigned counts[] = {1024, 16384, 16385, 65536};
  for (int i = 0; i < 4; ++i) {
    struct io_uring r;
    io_uring_queue_init(8, &r, 0);
    int rc = io_uring_register_buffers_sparse(&r, counts[i]);
    printf("sparse registered-buffer table of %5u entries: rc=%d (%s)\n", counts[i], rc, rc < 0 ? strerror(-rc) : "ok");
    io_uring_queue_exit(&r);
  }
  size_t sizes[] = {1ull << 30, (1ull << 30) + 4096};
  for (int i = 0; i < 2; ++i) {
    struct io_uring r;
    io_uring_queue_init(8, &r, 0);
    void* b = aligned(sizes[i], 0);
    struct iovec iov = {b, sizes[i]};
    double t0 = now_s();
    int rc = io_uring_register_buffers(&r, &iov, 1);
    printf("one iovec of %10zu B: rc=%d (%s) in %.1f ms\n", sizes[i], rc, rc < 0 ? strerror(-rc) : "ok", (now_s() - t0) * 1e3);
    io_uring_queue_exit(&r);
    free(b);
  }
}
static void run_memlock(void) {  // lower the limit in this process and see what io_uring enforces
  struct rlimit rl = {64ull << 20, 64ull << 20};
  setrlimit(RLIMIT_MEMLOCK, &rl);
  struct io_uring r;
  int irc = io_uring_queue_init(8, &r, 0);
  printf("queue_init under RLIMIT_MEMLOCK=64MiB: rc=%d (%s)\n", irc, irc < 0 ? strerror(-irc) : "ok");
  if (irc < 0) return;
  size_t sizes[] = {32ull << 20, 32ull << 20, 32ull << 20, 128ull << 20};
  for (int i = 0; i < 4; ++i) {
    void* b = aligned(sizes[i], 0);
    memset(b, 1, sizes[i]);
    struct iovec iov = {b, sizes[i]};
    int rc = io_uring_register_buffers(&r, &iov, 1);
    printf("RLIMIT_MEMLOCK=64MiB, register %3zu MiB (registration #%d): rc=%d (%s)\n", sizes[i] >> 20, i + 1, rc, rc < 0 ? strerror(-rc) : "ok");
    if (rc == 0) io_uring_unregister_buffers(&r);  // one at a time; then charge is dropped
  }
  // hold two at once: the second must fail if charges accumulate
  void* a = aligned(48ull << 20, 0); void* b = aligned(48ull << 20, 0);
  memset(a, 1, 48ull << 20); memset(b, 1, 48ull << 20);
  struct iovec iov2[2] = {{a, 48ull << 20}, {b, 48ull << 20}};
  int rc = io_uring_register_buffers(&r, iov2, 2);
  printf("RLIMIT_MEMLOCK=64MiB, register 2x48 MiB in one call: rc=%d (%s)\n", rc, rc < 0 ? strerror(-rc) : "ok");
}

// ---------- registration cost ----------
static int cmpd(const void* a, const void* b) { double x = *(double*)a, y = *(double*)b; return (x > y) - (x < y); }
static void run_cost(void) {
  size_t mib[] = {8, 64, 104, 256, 1024};
  printf("%8s %9s %8s %12s %12s %12s\n", "MiB", "memory", "faulted", "register ms", "unreg ms", "ns/4K page");
  for (int m = 0; m < 5; ++m)
    for (int nohuge = 0; nohuge < 2; ++nohuge)
      for (int fault = 0; fault < 2; ++fault) {
        size_t n = mib[m] << 20;
        double reg[7], unreg[7];
        for (int rep = 0; rep < 7; ++rep) {
          struct io_uring r; io_uring_queue_init(8, &r, 0);
          void* b = aligned(n, nohuge);
          if (fault) memset(b, 1, n);
          struct iovec iov = {b, n};
          double t0 = now_s();
          int rc = io_uring_register_buffers(&r, &iov, 1);
          double t1 = now_s();
          if (rc) { printf("register failed %d\n", rc); return; }
          io_uring_unregister_buffers(&r);
          double t2 = now_s();
          reg[rep] = (t1 - t0) * 1e3; unreg[rep] = (t2 - t1) * 1e3;
          io_uring_queue_exit(&r); free(b);
        }
        qsort(reg, 7, sizeof(double), cmpd); qsort(unreg, 7, sizeof(double), cmpd);
        printf("%8zu %9s %8s %12.3f %12.3f %12.1f\n", mib[m], nohuge ? "4K-only" : "THP-ok", fault ? "yes" : "no", reg[3], unreg[3], reg[3] * 1e6 / (n / 4096.0));
      }
}

// ---------- read benchmark on one big file ----------
static uint64_t rng = 88172645463325252ull;
static uint64_t xr(void) { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return rng; }
#define QD 16
#define LEN (6815744ull)  // 6.5 MiB: about half of a 13.3 MB row, the extent size of a two-root read
static void run_read(const char* path, const char* arm, long nreads, int reps) {
  int fixed = strstr(arm, "fixed") != NULL, si = strstr(arm, "si") != NULL;
  int fd = open(path, O_RDONLY | O_DIRECT);
  if (fd < 0) { perror("open"); exit(1); }
  struct stat st; fstat(fd, &st);
  uint64_t span = (st.st_size - LEN) / 4096;
  uint8_t* bounce;
  if (getenv("HUGE")) {  // 2 MiB-aligned and madvise(MADV_HUGEPAGE): asks for direct compaction (defrag=madvise)
    void* hp = NULL;
    size_t sz = ((QD * LEN) + (2u << 20) - 1) & ~((size_t)(2u << 20) - 1);
    if (posix_memalign(&hp, 2u << 20, sz)) exit(1);
    madvise(hp, sz, MADV_HUGEPAGE);
    bounce = hp;
  } else {
    bounce = aligned(QD * LEN, getenv("NOHUGE") != NULL);
  }
  memset(bounce, 0, QD * LEN);
  {
    FILE* f = fopen("/proc/self/smaps_rollup", "r");
    char line[256];
    while (f && fgets(line, sizeof line, f))
      if (!strncmp(line, "AnonHugePages:", 14)) { printf("bounce %s: %s", getenv("HUGE") ? "MADV_HUGEPAGE+2MiB-aligned" : getenv("NOHUGE") ? "NOHUGEPAGE" : "default", line); }
    if (f) fclose(f);
  }
  double walls[16], cpus[16];
  for (int rep = -1; rep < reps; ++rep) {  // rep -1: warm-up, discarded
    struct io_uring r; struct io_uring_params p; memset(&p, 0, sizeof p);
    if (si) p.flags = IORING_SETUP_SINGLE_ISSUER | IORING_SETUP_DEFER_TASKRUN;
    int rc = io_uring_queue_init_params(64, &r, &p);
    if (rc) { printf("init %d\n", rc); exit(1); }
    if (fixed) {
      struct iovec iov = {bounce, QD * LEN};
      rc = io_uring_register_buffers(&r, &iov, 1);
      if (rc) { printf("register %d\n", rc); exit(1); }
    }
    rng = 88172645463325252ull;  // same offsets in every arm and rep
    long issued = 0, done = 0; unsigned inflight = 0; long bad = 0;
    double t0 = now_s(), c0 = cpu_s();
    unsigned nsub = 0;
    while (done < nreads) {
      while (inflight < QD && issued < nreads) {
        struct io_uring_sqe* sqe = io_uring_get_sqe(&r);
        if (!sqe) break;
        unsigned slot = issued % QD;
        uint64_t off = (xr() % span) * 4096;
        void* buf = bounce + slot * LEN;
        if (fixed) io_uring_prep_read_fixed(sqe, fd, buf, LEN, off, 0);
        else io_uring_prep_read(sqe, fd, buf, LEN, off);
        sqe->user_data = slot;
        ++issued; ++inflight;
      }
      int s = io_uring_submit_and_wait(&r, 1);
      if (s < 0 && s != -EINTR) { printf("submit %d\n", s); exit(1); }
      ++nsub;
      struct io_uring_cqe* cqe; unsigned head, seen = 0;
      io_uring_for_each_cqe(&r, head, cqe) {
        if (cqe->res != (int)LEN) ++bad;
        ++seen; --inflight; ++done;
      }
      io_uring_cq_advance(&r, seen);
    }
    double t1 = now_s(), c1 = cpu_s();
    if (rep >= 0) { walls[rep] = t1 - t0; cpus[rep] = c1 - c0; }
    if (bad) printf("  rep %d: %ld short/failed reads\n", rep, bad);
    io_uring_queue_exit(&r);
  }
  double bytes = nreads * (double)LEN;
  qsort(walls, reps, sizeof(double), cmpd); qsort(cpus, reps, sizeof(double), cmpd);
  printf("arm=%-9s reads=%ld x %.2f MiB, reps=%d: wall p50 %.1f ms (min %.1f max %.1f)  %.2f GB/s   thread CPU p50 %.2f ms (min %.2f max %.2f)  CPU/GB %.2f ms\n",
         arm, nreads, LEN / 1048576.0, reps, walls[reps / 2] * 1e3, walls[0] * 1e3, walls[reps - 1] * 1e3, bytes / walls[reps / 2] / 1e9,
         cpus[reps / 2] * 1e3, cpus[0] * 1e3, cpus[reps - 1] * 1e3, cpus[reps / 2] * 1e3 / (bytes / 1e9));
}

int main(int argc, char** argv) {
  if (argc < 2) return 2;
  if (!strcmp(argv[1], "flags")) run_flags();
  else if (!strcmp(argv[1], "limits")) run_limits();
  else if (!strcmp(argv[1], "memlock")) run_memlock();
  else if (!strcmp(argv[1], "cost")) run_cost();
  else if (!strcmp(argv[1], "read")) run_read(argv[2], argv[3], atol(argv[4]), atoi(argv[5]));
  return 0;
}
```

</details>

<details><summary>Appendix B: numa_touch.c</summary>

```c
// CPU-only: where do first-touched pages land? gcc -O2 numa_touch.c -o numa_touch
#define _GNU_SOURCE
#include <numaif.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <stdint.h>
int main(int argc, char** argv) {
  int core = atoi(argv[1]); size_t mib = atol(argv[2]);
  cpu_set_t s; CPU_ZERO(&s); CPU_SET(core, &s); sched_setaffinity(0, sizeof s, &s);
  size_t n = mib << 20; void* p; posix_memalign(&p, 4096, n);
  memset(p, 1, n);                       // first touch, on `core`
  size_t pages = n / 4096; void** addr = malloc(pages * sizeof(void*)); int* st = malloc(pages * sizeof(int));
  for (size_t i = 0; i < pages; ++i) addr[i] = (char*)p + i * 4096;
  move_pages(0, pages, addr, NULL, st, 0);   // query only (nodes == NULL)
  size_t c[2] = {0, 0}, other = 0;
  for (size_t i = 0; i < pages; ++i) { if (st[i] == 0 || st[i] == 1) c[st[i]]++; else other++; }
  printf("touched %zu MiB from core %d (node %d): node0 %.1f%%  node1 %.1f%%  other %.1f%%\n", mib, core, core % 36 < 18 ? 0 : 1,
         100.0 * c[0] / pages, 100.0 * c[1] / pages, 100.0 * other / pages);
  return 0;
}
```

</details>

<details><summary>Appendix C: memcpy_numa.c</summary>

```c
// CPU-only: single-thread row memcpy (bounce -> slab scatter proxy) by thread/src/dst NUMA node.
// gcc -O2 memcpy_numa.c -o memcpy_numa -lnuma
#define _GNU_SOURCE
#include <numaif.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
static double now_s(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static void* bound(size_t n, int node) {
  void* p = mmap(NULL, n, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  unsigned long mask = 1ul << node;
  if (mbind(p, n, MPOL_BIND, &mask, 8 * sizeof mask, 0)) { perror("mbind"); exit(1); }
  memset(p, 1, n);
  return p;
}
static int cmpd(const void* a, const void* b) { double x = *(double*)a, y = *(double*)b; return (x > y) - (x < y); }
int main(int argc, char** argv) {
  int core = atoi(argv[1]), sn = atoi(argv[2]), dn = atoi(argv[3]);
  cpu_set_t s; CPU_ZERO(&s); CPU_SET(core, &s); sched_setaffinity(0, sizeof s, &s);
  size_t row = 13300000, rows = 40, n = row * rows;
  char* src = bound(n, sn); char* dst = bound(n, dn);
  double r[7];
  for (int pass = -1; pass < 6; ++pass) {
    double t0 = now_s();
    for (size_t i = 0; i < rows; ++i) memcpy(dst + i * row, src + i * row, row);
    double t = now_s() - t0;
    if (pass >= 0) r[pass] = t / rows * 1e3;
  }
  qsort(r, 6, sizeof(double), cmpd);
  printf("thread node %d  src node %d  dst node %d : %.3f ms/row (min %.3f max %.3f)  %.2f GB/s\n", core >= 18 && core < 36 || core >= 54 ? 1 : 0, sn, dn, r[3], r[0], r[5], row / (r[3] * 1e-3) / 1e9);
  return 0;
}
```

</details>

<details><summary>Appendix D: dio.c</summary>

```c
#define _GNU_SOURCE
#include <fcntl.h>
#include <stdio.h>
#include <sys/stat.h>
int main(int c, char** v) { for (int i = 1; i < c; ++i) { struct statx sx; if (statx(AT_FDCWD, v[i], 0, STATX_DIOALIGN, &sx)) { perror(v[i]); continue; } printf("%s: dio_mem_align=%u dio_offset_align=%u (mask has DIOALIGN: %d)\n", v[i], sx.stx_dio_mem_align, sx.stx_dio_offset_align, !!(sx.stx_mask & STATX_DIOALIGN)); } return 0; }
```

</details>

<details><summary>Appendix E: defer_probe.c and its output (2026-09-21)</summary>

```c
// CPU-only: which calls on a DEFER_TASKRUN ring make a deferred completion visible? No drive, no GPU.
//   gcc -O2 -pthread defer_probe.c -o defer_probe -luring
#define _GNU_SOURCE
#include <liburing.h>
#include <pthread.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
static double now_s(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
static void* w(void*a){usleep(50000);char c='x';(void)!write(*(int*)a,&c,1);return 0;}
static void run(const char*name,unsigned flags,int step){
  struct io_uring r;struct io_uring_params p;memset(&p,0,sizeof p);p.flags=flags;
  if(io_uring_queue_init_params(8,&r,&p)<0){printf("%s init failed\n",name);return;}
  int pf[2];(void)!pipe(pf);char b[1];
  struct io_uring_sqe*s=io_uring_get_sqe(&r);io_uring_prep_read(s,pf[0],b,1,0);io_uring_submit(&r);
  pthread_t t;pthread_create(&t,0,w,&pf[1]);usleep(120000); // write landed at 50 ms; owner has not entered since
  unsigned before=io_uring_cq_ready(&r);
  const char*what="";int rc=0;
  switch(step){
    case 0: what="cq_ready only (no syscall)";break;
    case 1: what="io_uring_submit() with an empty SQ";rc=io_uring_submit(&r);break;
    case 2: {what="io_uring_submit() with a NOP prepared";struct io_uring_sqe*n=io_uring_get_sqe(&r);io_uring_prep_nop(n);rc=io_uring_submit(&r);break;}
    case 3: what="io_uring_get_events()";rc=io_uring_get_events(&r);break;
    case 4: what="io_uring_submit_and_wait(0)";rc=io_uring_submit_and_wait(&r,0);break;
    case 5: {struct io_uring_cqe*c;what="io_uring_peek_cqe()";rc=io_uring_peek_cqe(&r,&c);break;}
    case 6: {what="io_uring_submit_and_wait(1)";rc=io_uring_submit_and_wait(&r,1);break;}
  }
  unsigned after=io_uring_cq_ready(&r);
  printf("%-32s %-40s rc=%d cq_ready before=%u after=%u\n",name,what,rc,before,after);
  pthread_join(t,0);close(pf[0]);close(pf[1]);io_uring_queue_exit(&r);
}
int main(void){
  for(int st=0;st<=6;st++){
    run("flags=0",0,st);
    run("SI|DTR",IORING_SETUP_SINGLE_ISSUER|IORING_SETUP_DEFER_TASKRUN,st);
    run("SI|DTR|TASKRUN_FLAG",IORING_SETUP_SINGLE_ISSUER|IORING_SETUP_DEFER_TASKRUN|IORING_SETUP_TASKRUN_FLAG,st);
  }
  return 0;}
```

Output on divix01 (`CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 taskset -c 2 ./defer_probe`, kernel 6.12.0-211.51.1.el10_2, liburing 2.12):

```
flags=0                          cq_ready only (no syscall)               rc=0 cq_ready before=1 after=1
SI|DTR                           cq_ready only (no syscall)               rc=0 cq_ready before=0 after=0
SI|DTR|TASKRUN_FLAG              cq_ready only (no syscall)               rc=0 cq_ready before=0 after=0
flags=0                          io_uring_submit() with an empty SQ       rc=0 cq_ready before=1 after=1
SI|DTR                           io_uring_submit() with an empty SQ       rc=0 cq_ready before=0 after=0
SI|DTR|TASKRUN_FLAG              io_uring_submit() with an empty SQ       rc=0 cq_ready before=0 after=1
flags=0                          io_uring_submit() with a NOP prepared    rc=1 cq_ready before=1 after=2
SI|DTR                           io_uring_submit() with a NOP prepared    rc=1 cq_ready before=0 after=1
SI|DTR|TASKRUN_FLAG              io_uring_submit() with a NOP prepared    rc=1 cq_ready before=0 after=2
flags=0                          io_uring_get_events()                    rc=0 cq_ready before=1 after=1
SI|DTR                           io_uring_get_events()                    rc=0 cq_ready before=0 after=1
SI|DTR|TASKRUN_FLAG              io_uring_get_events()                    rc=0 cq_ready before=0 after=1
flags=0                          io_uring_submit_and_wait(0)              rc=0 cq_ready before=1 after=1
SI|DTR                           io_uring_submit_and_wait(0)              rc=0 cq_ready before=0 after=0
SI|DTR|TASKRUN_FLAG              io_uring_submit_and_wait(0)              rc=0 cq_ready before=0 after=1
flags=0                          io_uring_peek_cqe()                      rc=0 cq_ready before=1 after=1
SI|DTR                           io_uring_peek_cqe()                      rc=-11 cq_ready before=0 after=0
SI|DTR|TASKRUN_FLAG              io_uring_peek_cqe()                      rc=0 cq_ready before=0 after=1
flags=0                          io_uring_submit_and_wait(1)              rc=0 cq_ready before=1 after=1
SI|DTR                           io_uring_submit_and_wait(1)              rc=0 cq_ready before=0 after=1
SI|DTR|TASKRUN_FLAG              io_uring_submit_and_wait(1)              rc=0 cq_ready before=0 after=1
```

</details>

<details><summary>Appendix F: nvme_irq_map.py and the Revision 2 command list</summary>

```python
#!/usr/bin/env python3
# Which CPU takes the completion interrupt of each NVMe submitting core? Read-only sysfs/procfs.
#   python3 nvme_irq_map.py            # prints, per drive, the submitter cores in 0-63 whose IRQ is on 64-71
import os
def cpus(s):
    out = []
    for p in s.strip().split(","):
        if "-" in p:
            a, b = p.split("-"); out += range(int(a), int(b) + 1)
        elif p:
            out.append(int(p))
    return out
irqs = {}
for line in open("/proc/interrupts"):
    f = line.split()
    if f and f[-1].startswith("nvme"):
        irqs[f[-1]] = f[0].rstrip(":")
for dev in ("nvme0", "nvme2", "nvme3"):        # nvme3 is the drive mounted at /mnt/nvme4
    m = {}
    for q in os.listdir(f"/sys/block/{dev}n1/mq"):
        cl = cpus(open(f"/sys/block/{dev}n1/mq/{q}/cpu_list").read())
        irq = irqs[f"{dev}q{int(q) + 1}"]
        eff = cpus(open(f"/proc/irq/{irq}/effective_affinity_list").read())[0]
        for c in cl:
            m[c] = eff
    print(dev, "submitter cores in 0-63 whose completion IRQ lands on 64-71:", sorted(c for c in m if c < 64 and 64 <= m[c] <= 71))
    print(dev, "node-0 submitters' IRQ nodes:", sorted({("n0" if (m[c] < 18 or 36 <= m[c] < 54) else "n1") for c in m if c < 18 or 36 <= c < 54}))
```

```sh
# Revision 2 commands (all read-only on divix01; no GPU; nothing under /mnt/nvme1 or /mnt/nvme2 beyond metadata)
grep -E "nvme|/data/models" /proc/mounts ; ls /dev/nvme* /sys/block
for r in /mnt/nvme0/dsv41_flash /mnt/nvme4/dsv41_flash /mnt/nvme2/DeepSeek-V4.1-Flash-EXL3-3.0bpw; do fincore -b -n -o RES,SIZE $r/*.safetensors; done
filefrag -v /mnt/nvme4/dsv41_flash/*.safetensors          # FIEMAP: metadata only
xfs_info /mnt/nvme0 ; xfs_info /mnt/nvme2 ; cat /sys/block/nvme*n1/queue/max_sectors_kb
cat /sys/bus/pci/devices/0000:{86,87,88,89}:00.0/current_link_{speed,width}
cat /proc/cmdline ; cat /sys/kernel/iommu_groups/<group of 0000:86:00.0>/type
dd if=<cold shard> of=/dev/null bs=1M count=32 skip=2000 iflag=direct ; fincore -b -n -o RES <shard>   # before and after
python3 nvme_irq_map.py                                    # Appendix F ; then compare /proc/interrupts on CPUs 64-71
gcc -O2 -pthread defer_probe.c -o defer_probe -luring ; taskset -c 2 ./defer_probe    # Appendix E
numactl --physcpubind=2 --membind=0 ./uring_probe cost    # Appendix A, registration cost incl. 256 MiB
```

</details>
