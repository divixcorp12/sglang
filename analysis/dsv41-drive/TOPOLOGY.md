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
| native bounce | `kBounceRows × slot_bytes` = 8 × ~13.3 MB ≈ **106 MB** (slot_bytes from `MIRROR_ROWS.md:42`, not read back from a live table) | [C] `exl3_ram_miss_host.cpp:46,339`; [D] |
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
