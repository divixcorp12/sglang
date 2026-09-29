# Hotpath zero-overhead: suites, GPU tests and hot-path counts at the branch head

Date: 2026-09-29. Branch head verified: `2a3aaee887` (`origin/cc/hotpath-zero-overhead`), checked out detached at
`/data/models/slang/nvfp4-work/wt-hotpath` after `SYNC`. Base for every comparison: `ba01695c35`
(`/data/models/slang/nvfp4-work/wt-hotpath-base`), the same commit `baseline.md` used.

All arithmetic below is recomputed from actual `--collect-only -q` test-ID diffs between `ba01695c35` and `2a3aaee887`
(F17), not from any task report's running total. The per-task reports are cited only to explain *why* a given file's
ID set changed, not for their numbers.

## 1. SUITE: kernels

```bash
cd /data/models/slang/nvfp4-work/wt-hotpath && export PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 \
  CUDA_HOME=/usr/local/cuda-13.4 \
  && /data/models/slang/.venv/bin/python -c "import sglang; print(sglang.__file__)" \
  && flock /data/models/slang/nvfp4-work/cc-gpu.lock taskset -c 32-63 /data/models/slang/.venv/bin/python -m pytest \
     test/registered/unit/kernels -q -p no:randomly --durations=0
```

`sglang.__file__` = `/data/models/slang/nvfp4-work/wt-hotpath/python/sglang/__init__.py` (correct worktree).

**Result: `1119 passed, 21 skipped` in 334.01s (0:05:34), `EXIT=0`** (`${PIPESTATUS[0]}` read explicitly).

Baseline (`baseline.md` §2): `1933 passed, 23 skipped`, `EXIT=0`. Delta: **-814 passed, -2 skipped** (net -816
collected IDs: 1956 -> 1140).

### 1a. Collected-ID diff (ground truth for the delta)

```bash
# base
cd /data/models/slang/nvfp4-work/wt-hotpath-base && export PYTHONPATH=$PWD/python \
  && /data/models/slang/.venv/bin/python -m pytest --collect-only -q test/registered/unit/kernels -p no:randomly \
  | grep "::" | sort > base-ids.txt      # 1956 IDs
# head
cd /data/models/slang/nvfp4-work/wt-hotpath && export PYTHONPATH=$PWD/python \
  && /data/models/slang/.venv/bin/python -m pytest --collect-only -q test/registered/unit/kernels -p no:randomly \
  | grep "::" | sort > head-ids.txt      # 1140 IDs
diff base-ids.txt head-ids.txt           # 944 removed ("<"), 128 added (">")
```

1956 - 944 + 128 = 1140 = 1119 passed + 21 skipped. Exact.

**Removed (944), by file — all from Task 6 deleting the packed path** (`task-6-report.md`, "the packed path is
deleted; RowReader is the only reader"; every removed ID is a packed-path/shard-table/bounce-buffer test whose
subject no longer exists):

| File | removed |
|---|---:|
| test_exl3_ram_miss_pack_workers.py | 550 (file deleted: packing-worker pool tests) |
| test_exl3_ram_miss_row_images.py | 235 (213 shard-vs-image reuse clones + refusal/mode cases whose packed leg is gone) |
| test_exl3_ram_miss_split.py | 53 (bounce-only split functions; the `direct` param collapsed) |
| test_exl3_ram_miss_piece_stream.py | 36 (bounce-only piece-stream functions; workers/chunks params collapsed) |
| test_expert_stream_reader_golden.py | 15 (12 shard shape/mode cases + 3 `direct_split_traced` cases) |
| test_expert_stream_fixed_buffers.py | 12 (`[*-bounce]` cases; renamed to `[*-images]`, counted under added) |
| test_exl3_ram_miss_prefill_fills.py | 10 (`tier[shards]` cases; `tier[row_images]` renamed to no-param, counted under added) |
| test_exl3_ram_miss_tier.py | 9 (8 `pack_pool_affinity`/`pack_worker_cpus` extent cases + 1 rename, see below) |
| test_expert_stream_uring_integration.py | 8 (`[fixed-shard-bounce-*]`; renamed to `[readv-fixed-shared-arena-*]`) |
| test_exl3_ram_miss_task5_item5_ack_independence.py | 6 (`running[inline_pack/two_pack_workers]` collapsed) |
| test_expert_stream_read_cuts.py | 4 (`[*-bounce]`; renamed to `[*-images]`) |
| test_exl3_ram_miss_piece_stream_parts.py | 4 (4 shard shapes; no image equivalent — coverage given up, per Task 6 Concern 4) |
| test_expert_stream_reader_split.py | 1 (`..._through_any_reader`; renamed `..._through_the_row_reader`) |
| test_exl3_ram_miss_device_args.py | 1 (`test_the_exl3_host_file_is_only_bindings`; Task 8 parametrizes it `[exl3_ram_miss_host.cpp]`/`[exl3_ram_miss_host_instr.cpp]`) |

**Added (128), by file:**

| File | added | Source / reason |
|---|---:|---|
| test_expert_stream_build_variants.py | 28 | Task 8 (+6: `ProdBuild`/`InstrBuild` selection) and Task 10 (+21: test-only export refusal on prod, per-build counters, `test_a_faulted_read_refuses_on_prod`), plus `test_an_unknown_variant_is_refused`/`test_a_production_host_reports_only_the_core_counters` counted once each |
| test_exl3_ram_miss_row_images.py | 22 | Task 6 (+18: image-mode reuse clones + 2 refusal tests it kept) and Task 7 (renamed `..._opens_images_only_with_the_flag_mirrors_o_direct_and_leases` -> `..._opens_images_only_with_mirrors_and_o_direct`, dropping the flag/lease legs) |
| test_expert_stream_ring_reset.py | 15 | New file, from the `origin/master` merge (`merge-65754399-report.md`): NOP-drain ring reset, independent of the reader stack |
| test_expert_stream_hotpath_shim.py | 10 | Task 8 build selection, Task 9 (`test_the_prod_service_thread_reads_no_clock_per_request`), Task 12 (copy-thread alloc/condvar `[prod]`/`[instr]`, spinning-copy-thread), Task 15 (`test_the_service_and_copy_threads_take_no_lock_and_never_wait_on_a_condvar[prod]`/`[instr]`) |
| test_expert_stream_ownership.py | 9 | New file, Task 13 (Review Focus 1 and 4: unpaused-eager-call refusal, set_hot burst-past-ring), Task 14 (Review Focus 3: copy released at owner's poll, pause retires a completed copy), Task 15 (Review Focus 2/lock-audit: tier declares only caller's mutex, fill-holds-slots, failed-fill release, stop-mid-pause join), B3 fix round (`test_start_thread_refuses_a_tier_whose_fill_is_still_running`, `b3-fix-report.md`) |
| test_expert_stream_fixed_buffers.py | 7 | Master merge's `test_a_refused_nop_drain_through_the_reader` (converted onto row images) + Task 6's `[*-bounce]`->`[*-images]` renames (6 IDs, matching the 12 removed above 1:1 minus the collapsed pieces param) |
| test_exl3_ram_miss_split.py | 5 | Task 6: `direct` param collapsed from `[True]/[False]` to single-value IDs on the two short-read tests |
| test_exl3_ram_miss_prefill_fills.py | 5 | Task 6: `tier[row_images]` renamed to no-param IDs (5, matching 5 of the 10 removed 1:1; the `[shards]` legs have no successor) |
| test_exl3_ram_miss_piece_stream.py | 5 | Task 6: bounce-only functions' `workers`/`chunks` params collapsed (u2/u3/flag_off cases) |
| test_expert_stream_uring_integration.py | 4 | Task 6: `[fixed-shard-bounce-*]` renamed to `[readv-fixed-shared-arena-*]` |
| test_expert_stream_hotpath_golden.py | 3 | Task 9/10: `test_row_image_reads_prepare_the_golden_sqes_and_land_exact_bytes` (new) + `test_the_scripted_scenario_matches_the_golden[prod]`/`[instr]` (Task 10's ProdBuild/InstrBuild parametrization; no base-commit predecessor existed to remove, since the unparametrized test post-dates `ba01695c35`) |
| test_exl3_ram_miss_task5_item5_ack_independence.py | 3 | Task 6: `running[inline_pack/two_pack_workers]` collapsed to 3 unparametrized IDs |
| test_expert_stream_spsc_ring.py | 2 | Task 12: `test_the_ring_delivers_every_item_in_order_across_threads[plain]`/`[tsan]` (the SPSC ring, "the ring 2" in the brief) |
| test_expert_stream_read_cuts.py | 2 | Task 6: `[*-bounce]` renamed to `[*-images]` |
| test_exl3_ram_miss_device_args.py | 2 | Task 8: `test_the_exl3_host_file_is_only_bindings` parametrized `[exl3_ram_miss_host.cpp]`/`[exl3_ram_miss_host_instr.cpp]`, replacing the single removed ID |
| test_expert_stream_reader_split.py | 1 | Task 6: `..._through_the_row_reader` (renamed from `..._through_any_reader`) |
| test_expert_stream_prod_build_symbols.py | 1 | Task 10: `test_prod_has_no_trace_or_fault_symbols_and_instr_has_them` (the `nm` symbols test) |
| test_expert_stream_hotpath_stress.py | 1 | `test_every_party_against_the_service_keeps_the_tier_invariants` (stress test registration; predates its Task-16 TSan/fills extension, which added no new ID) |
| test_expert_stream_fixed_vec.py | 1 | Task 11: `test_fixed_vec_pushes_assigns_clears_and_throws_on_overflow` (`FixedVec`, "FixedVec 1" in the brief) |
| test_exl3_ram_miss_tier.py | 1 | B3 fix round: `test_an_owned_call_waits_for_a_pump_on_another_thread` (renamed from the removed `test_release_refuses_a_slot_that_is_still_loading`) |
| test_exl3_ram_miss_stage_trace_causal.py | 1 | Task 9: `test_no_ungated_clock_read_remains_on_the_request_path` |

**Every one of the 944 removed and 128 added IDs is accounted for.** No unexplained count.

The brief's category list is fully covered: golden 3 (2 param variants + 1 new function; matches "golden 2 (4 with
Task 10's parametrization)" — the base predates the pre-parametrized single test, so there is no removal to pair
with the +2), shim (10), stress (1, pre-existing registration), build variants (28, covering both Task 8's build
selection and Task 10's test-only refusals), the symbols test (1), FixedVec (1), the ring (2, `spsc_ring.py`),
ownership (9), and the Task 5-7 refusals (the `row_images.py` refusal tests kept through Task 6's -235/+22, and
Task 7's rename dropping the flag/lease legs).

## 2. SUITE: layers/moe (with `--continue-on-collection-errors`, per the baseline's F17-equivalent ruling)

```bash
cd /data/models/slang/nvfp4-work/wt-hotpath && export PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 \
  CUDA_HOME=/usr/local/cuda-13.4 \
  && flock /data/models/slang/nvfp4-work/cc-gpu.lock taskset -c 32-63 /data/models/slang/.venv/bin/python -m pytest \
     test/registered/unit/layers/moe -q -p no:randomly --continue-on-collection-errors
```

**Result: `2 failed, 1087 passed, 23 warnings, 12 errors, 12405 subtests passed` in 88.39s (0:01:28), `EXIT=1`**
(`${PIPESTATUS[0]}` read explicitly).

Baseline (`baseline.md` §1b): `2 failed, 1094 passed, 12 errors, 12405 subtests passed`, `EXIT=1`.

- **The 11 pyarrow collection errors are byte-identical to the base's set**: `test_deepep_v2_buffer_lifecycle.py`,
  `test_deepep_v2_masked_slab.py`, `test_deepep_v2_wire_dtype.py`, `test_flashinfer_a2a_wide_ep.py`,
  `test_fused_moe_native.py`, `test_fused_shared_expert_scaling.py`, `test_mega_moe_deepgemm_api.py`,
  `test_topk_correction_bias_cache.py`, `test_w4afp8_deepep_dtype.py`, `test_w4afp8_deepep_post_reorder.py`,
  `test_w4afp8_requant_geometry.py`. Pre-existing, environmental (pyarrow 25.0.1), unrelated to this branch.
- **The 12th error is the same teardown-time error**, at the same test:
  `test_exl3_ram_miss_service.py::test_graph_routes_are_logged_only_when_the_stage_trace_is_on[trace_on]`
  (`service.shutdown()` -> `_quarantine`, `AttributeError("'types.SimpleNamespace' object has no attribute
  'state'")`). Pre-existing at base, recorded there as undiagnosed, unchanged here.
- **The 2 failures are the same 2 as base**: `test_expert_plugins_cuda.py::TestPinnedTierCuda::
  test_a_split_chunk_copies_its_row_index_to_the_device_once` and `test_expert_row_source.py::TestGatherReadStats::
  test_a_gather_without_host_reads_keeps_its_stats_object`. Pre-existing, undiagnosed at base, unchanged.
- **Passed delta: 1087 - 1094 = -7**, matching the collect-only ID diff exactly (15 removed, 8 added, net -7; see
  §2a). No new failure, no new error, and the collection-error file set is identical to base.

### 2a. Collected-ID diff, layers/moe

```bash
python -m pytest --collect-only -q test/registered/unit/layers/moe -p no:randomly --continue-on-collection-errors \
  | grep "::" | sort
```

Base: 1096 IDs. Head: 1089 IDs. Diff: 15 removed, 8 added (net -7), all in `test_exl3_ram_miss_service.py`,
`test_exl3_ram_miss_tables.py` and `test_exl3_prefill_fills_service.py::test_the_flag_is_refused_without_row_images`
(1 removed, no successor — the flag no longer exists to refuse, per Task 7). The remaining 14 removed / 8 added are
Task 7's mandatory-row-images-and-leases rewrite (`task-7-report.md` §"Per-test list"): the packed-workers/shards
config-refusal tests (`piece_stream_refuses_unless_two_phase_lease_and_pack_workers_all_hold[no_lease/no_pack_workers/
no_two_phase]`, `lease_pdl_without_leases_is_refused`, `lease_switch_defaults_off...`) have no successor (leases and
row images are now mandatory, not configurable), and `test_exl3_ram_miss_tables.py` is rewritten from 11 shard-based
IDs to 8 image-based ones. This exactly matches Task 7's per-test table.

## 3. GPU manual test list (minus `test_exl3_piece_stream_cuda.py`, per Task 1 Step 3)

```bash
cd /data/models/slang/nvfp4-work/wt-hotpath && export PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 \
  CUDA_HOME=/usr/local/cuda-13.4 \
  && flock /data/models/slang/nvfp4-work/cc-gpu.lock taskset -c 32-63 \
  /data/models/slang/.venv/bin/python -m pytest -q -rf -p no:randomly \
    test/manual/dsv41/test_exl3_ram_miss_cuda.py test/manual/dsv41/test_exl3_lease_kernels_cuda.py \
    test/manual/dsv41/test_exl3_copy_engine_cuda.py test/manual/dsv41/test_exl3_piece_stream_row_images_cuda.py \
    test/manual/dsv41/test_exl3_two_phase_parity_cuda.py test/manual/dsv41/test_exl3_two_phase_failure_cuda.py \
    test/manual/dsv41/test_exl3_two_phase_timing_cuda.py test/manual/dsv41/test_exl3_native_prefetch_cuda.py \
    test/manual/dsv41/test_exl3_task5_item4_gpu.py test/manual/dsv41/test_exl3_task5_item6_shutdown_gpu.py \
    test/manual/dsv41/test_exl3_ram_miss_graph_gpu.py
```

**Result: `15 failed, 119 passed, 3 skipped` in 167.41s (0:02:47), `EXIT=1`.**

Baseline (`baseline.md` §3): `24 failed, 119 passed, 3 skipped`, `EXIT=1`. Same 119 passed, same 3 skipped.

**All 15 head failures map onto the base's 24 pre-existing failures; there are 0 new failures.** The 9-failure
reduction is exactly Task 7's deletion of the `leases_off`/lease-A-B parametrizations (`task-7-report.md`, "Manual
GPU" table):

- `test_exl3_task5_item4_gpu.py::..._fatal_demand_error_...[leases_off-timeout]`, `[leases_off-read_fault]` — the
  `leases_off` param no longer exists; `[leases_on-timeout]`/`[leases_on-read_fault]` survive as the unparametrized
  `[timeout]`/`[read_fault]`, both still red (matches head's `test_a_fatal_demand_error_...[timeout]`/`[read_fault]`).
- `test_exl3_ram_miss_graph_gpu.py::test_ram_misses_inside_a_replay_are_served[leases_off]` and
  `test_a_forced_timeout_fails_stop_without_hanging[leases_off]` — same param removal; the `[leases_on]` legs survive
  unparametrized, both still red.
- `test_exl3_ram_miss_graph_gpu.py::test_many_layers_in_one_replay_are_served[leases_off-layers_4]`,
  `[leases_off-layers_20]` — same; `[leases_on-layers_4]`/`[leases_on-layers_20]` survive as `[layers_4]`/`[layers_20]`,
  both still red.
- `test_exl3_ram_miss_graph_gpu.py::test_lease_mode_output_is_byte_exact_against_off` and
  `test_lease_mode_multi_layer_graph_is_byte_exact_against_off[layers_4]`, `[layers_20]` — the leases-off-vs-on A/B is
  deleted outright (Task 7; leases are unconditional, so there is no "off" arm to compare against).

`24 - 9 = 15`, and every surviving ID appears in both lists. Every remaining failure (item4 timeout/read_fault, item6
default_stream/side_stream/waiting-reader, graph_gpu's 6 `direct_insert_replay...` params, `ram_misses_inside_a_
replay_are_served`, `a_forced_timeout_fails_stop_without_hanging`, `many_layers...[layers_4]`/`[layers_20]`) was
already diagnosed as pre-existing and not a lease-invariant product bug in `gpu4-diagnosis.md` (classes (a)/(b)/(d):
test races and harness preconditions exposed by O_DIRECT/row-image timing, not product regressions). No skip count
changed (3, both trees: GPU-unavailable-class skips, unrelated to this branch).

## 4. NVMe and fixed-buffer manual tests (reader; `rowimg-disk.lock` then `cc-gpu.lock`)

```bash
export PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 CUDA_HOME=/usr/local/cuda-13.4 \
  && flock /data/models/slang/nvfp4-work/rowimg-disk.lock flock /data/models/slang/nvfp4-work/cc-gpu.lock \
     taskset -c 32-63 /data/models/slang/.venv/bin/python -m pytest -q -rs -p no:randomly \
    test/manual/dsv41/test_expert_stream_read_cuts_nvme.py test/manual/dsv41/test_expert_stream_fixed_buffers_big.py
```

| Tree | Result | EXIT |
|---|---|---|
| `wt-hotpath-base` (`ba01695c35`) | `3 passed, 2 skipped` in 6.30s | 0 |
| `wt-hotpath` (`2a3aaee887`) | `4 passed, 2 skipped` in 7.33s | 0 |

Both skips are identical and pre-existing: `IOPOLL_CUTS_DIR` is unset (`test_expert_stream_read_cuts_nvme.py:94,104`),
so the two IOPOLL cut cases skip on both trees. The +1 passed is a new test ID,
`test_expert_stream_fixed_buffers_big.py::test_a_ring_reset_keeps_the_big_slab_registered`, which exists only at
head; it comes from the `origin/master` merge's ring-reset NOP-drain work (`merge-65754399-report.md`), same as
§1's `test_expert_stream_ring_reset.py`. No regression, no unexplained delta.

## 5. Hot-path shim and stress (`taskset -c 0-63`)

### 5a. Shim

```bash
cd /data/models/slang/nvfp4-work/wt-hotpath && export PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 \
  CUDA_HOME=/usr/local/cuda-13.4 \
  && taskset -c 0-63 /data/models/slang/.venv/bin/python -c "import sglang; print(sglang.__file__)" \
  && taskset -c 0-63 /data/models/slang/.venv/bin/python -m pytest \
     test/registered/unit/kernels/test_expert_stream_hotpath_shim.py -q -rs -s -p no:randomly
```

`sglang.__file__` = `/data/models/slang/nvfp4-work/wt-hotpath/python/sglang/__init__.py`. **`10 passed`**, EXIT=0,
87.51s. Scenario: 200 measured requests, `requests=200`, `posts=208`, `copies_done=200`, `deferrals=8`,
`copy_jobs=200` in every sub-scenario (unchanged across prod/instr/spinning/no-lock variants — the copy-job and
deferral load is identical regardless of build or lock instrumentation).

### Table: master (baseline.md §6, run 1) vs branch head, prod build, per request

| thread | kind | master (`e3a66073b4`) raw | master / req | branch head (`2a3aaee887`) prod raw | branch / req |
|---|---|---:|---:|---:|---:|
| service | malloc | 2,800 | 14.00 | 0 | **0.00** |
| service | free | 2,800 | 14.00 | 0 | **0.00** |
| service | mutex | 351,658 | 1758.29 | 0 | **0.00** |
| service | cond | 0 | 0.00 | 0 | 0.00 |
| service | clock | 991,680 | 4958.40 | 0 | **0.00** |
| service | sleep | 0 | 0.00 | 0 | 0.00 |
| service | futex | n/a (shim had no futex kind at master) | n/a | 3-4 | 0.015-0.02 |
| copy | malloc | 530 | 2.65 | 0 | **0.00** |
| copy | free | 530 | 2.65 | 0 | **0.00** |
| copy | mutex | 530 | 2.65 | 0 | **0.00** |
| copy | cond | 265 | 1.32 | 0 | **0.00** |
| copy | clock | 795 | 3.98 | 0 | **0.00** |
| copy | sleep | 0 | 0.00 | 0 | 0.00 |
| copy | futex | n/a | n/a | 11-12 | 0.055-0.06 |

Every branch-head prod value the plan's four rules require to be zero is zero: service malloc/free/mutex/cond/clock/
sleep, and copy malloc/free/mutex/cond. This is the plan's target state reached — master's service thread did 14
mallocs/frees and ~1758 mutex ops per request; the branch's does none. The only nonzero counters on either thread are
`futex` (3-4 on service, 11-12 on copy), which the controller ruling requires reported with a note, not asserted
zero: they are the copy engine's documented sleep/wake protocol (`H/spsc_ring.h`'s `futex_wait`/`futex_wake`, Task
12) — the service's `submit()` wakes a sleeping copy thread only when the ring was empty and the thread parked; they
do not scale with `copy_jobs` (200) because the 50 ms Python-driver spin between posts keeps the copy thread mostly
out of its idle sleep, matching baseline's own reading for master's mutex/clock ("dominated by the idle spin, not the
request").

The instr build (not required to be zero) shows the expected InstrBuild-only cost: `service clock == 200` (1/request,
Task 15's assertion) and `copy clock == 600` (3/copy_job, Task 15's assertion), with malloc/free/mutex/cond still 0 on
both threads.

Copy jobs and deferrals (from the child, every scenario): `copies_done=200`, `deferrals=8`, `copy_jobs=200`.

### 5b. Stress (10 runs)

```bash
for i in $(seq 1 10); do taskset -c 0-63 /data/models/slang/.venv/bin/python -m pytest \
  test/registered/unit/kernels/test_expert_stream_hotpath_stress.py -q -p no:randomly; echo "EXIT=${PIPESTATUS[0]}"; done
```

**10/10 `1 passed`, `EXIT=0` every run** (~11.4s each).

## 6. Known flakes (rates, not regressions)

- **`test_fill_wait_returns_as_a_prefix_lands`** (`test_exl3_ram_miss_prefill_fills.py`): pre-existing timing flake
  under load, tracked as a follow-up since Task 6/10. Not rerun for a rate here; it did not fail in this batch's
  kernels-suite run (§1) or the 10-run stress loop (§5b).
- **`test_the_seqlock_reader_never_accepts_a_torn_record`** (`test_exl3_ram_miss_tier.py`): pre-existing
  throughput-floor flake (`seqlock-reader-report.md`). Recorded rate: **190/200 head (`f2ea4d3493`) vs 191/200 base
  (`ba01695c35`)**, every failure a throughput-floor miss (`accepted <= 100`) with `torn == 0` in all cases (no
  correctness failure in 400 combined runs). Statistically indistinguishable from base; load-sensitive, not a
  regression.

## 7. Commit

Commit `analysis(hotpath): suites, GPU tests and hot-path counts at the branch head`, both trailers, pushed with
`git push origin cc/hotpath-zero-overhead`.

## 8. Decode arms A (master) / B (branch) / A2 (master), and C (branch under the counting shim)

Date: 2026-09-29, 08:14-08:31 local. Driver `analysis/dsv41-drive/hotpath/drive_hotpath_arms.sh` at `0c50b9241c`, report
`hotpath_report.py`. Out dir `divix01:/mnt/nvme1/dsv41-hotpath/20260929-081400/` (`arms-report.json`,
driver log `/mnt/nvme1/dsv41-hotpath/driver-20260929-081400.log`).

```bash
ssh divix01 'cd /data/models/slang/nvfp4-work/wt-hotpath && nohup bash analysis/dsv41-drive/hotpath/drive_hotpath_arms.sh \
  /data/models/slang/nvfp4-work/wt-hotpath-master 65754399e394cc6f9a3529ba02b33e3af40f9555 \
  /data/models/slang/nvfp4-work/wt-hotpath $(git rev-parse HEAD) /mnt/nvme1/dsv41-hotpath/$TS 30031 \
  > /mnt/nvme1/dsv41-hotpath/driver-$TS.log 2>&1 &'      # TS=20260929-081400, HEAD=0c50b9241c
```

- **Trees.** A and A2: master `65754399e3` (the master this branch merged; python tree `83e046cc87`, registered
  `hotpath-base`) in the private worktree `wt-hotpath-master`. B and C: the branch at `0c50b9241c` (python tree
  `ee964852b2`, registered `hotpath-zero-overhead`) in `wt-hotpath`. Every arm ran the branch's `run_arm.sh`,
  `arm_env` and `generations.json`. `check_worktree` passed for both (at the commit, clean, `sglang.__file__` under
  each tree's own `python/`).
- **EXL3 gate.** Before every arm the driver resolved the launch's expert-stream requirements with that arm's own
  `python/` tree: `EXL3` for all four (not the silent NVFP4 fallback).
- **Tier: full, `0:61440,1:40960` / `102400`.** Before A: node 0 MemFree 77986 + page cache 9550 = 87536 MiB against
  80896 needed; node 1 52305 + 32380 = 84685 MiB against 55056. Re-checked before B, A2 and C at the same values; all
  passed. The full tier equals `base_env`'s, so it is no diff below.
- **Effective env against the branch's `base_env`** (`<arm>-env-diff.json`):
  - A, A2: `SGLANG_DSV41_ENABLE_RAM_MISS_ROW_IMAGES=1`, `SGLANG_DSV41_ENABLE_RAM_MISS_LEASES=1`,
    `SGLANG_DSV41_RAM_MISS_PACK_WORKERS=8` (master's recipe; the branch's `arm_env` no longer sets them);
  - B: none;
  - C: `LD_PRELOAD=<out>/hotpath_shim.so` (built from the branch's `hotpath_shim.c`),
    `HOTPATH_SHIM_OUT=<out>/C-shim.json`.
  `run_arm.sh` verified each set against the live server's `/proc/<pid>/environ` (51 vars for A).
- **Build check** (the "exl3 RAM miss thread started" line once "fired up"): A and A2 end in `copy engine on` with no
  build field; B and C end in `copy engine on, build prod`.
- `perf_event_paranoid` = 2. No lock waits: every arm started at its first attempt.

### 8a. Decode

| Arm | pooled ms/token | session CDW / ETR | median TTFT s | Δ vs mean(A, A2) | identical to A | served | rows_read | read_errors |
|---|---:|---:|---:|---:|---|---:|---:|---:|
| A | 100.52 | 123.95 / 98.84 | 7.96 | +0.21 | (reference) | 2907 | 4082 | 0 |
| B | **99.99** | 123.64 / 98.30 | 7.97 | **-0.32** | **yes (2/2 turns)** | 2953 | 4169 | 0 |
| A2 | 100.10 | 123.73 / 98.41 | 7.96 | -0.21 | yes | 2922 | 4124 | 0 |
| C (untimed) | 100.47 | 123.89 / 98.80 | 8.01 | (excluded) | yes | 2995 | 4234 | 0 |

90 decode tokens per arm (the harness's 2-session timed set). SM clock over every timed window: 2940-2970 MHz, median
2951-2962. `served`/`rows_read` are the server's lifetime counters (warm-up, prefill and the timed set).

- baseline mean(A, A2) = 100.31 ms/token; drift A2 - A = -0.42; allowance max(1.5, 0.42) = 1.5; limit 101.81.
- **B = 99.99 <= 101.81: B is not slower.** Byte identity holds for B, A2 and C against A. `read_errors` = 0 everywhere.

### 8b. The service thread's CPU and counters

CPU seconds from `thread_sampler.report` over the timed window (~26.6 s). perf stat on the `exl3-ram-miss` thread from
"fired up" to the arm's end, divided by the lifetime `served` (so slightly over-estimated, identically per arm).

| Arm | ram-miss CPU s (timed) | cycles:u / req | instructions:u / req | voluntary switches / req | involuntary / req | migrations / req |
|---|---:|---:|---:|---:|---:|---:|
| A | 7.99 | 34.3 M | 41.5 M | 200.0 | 0.034 | 0.312 |
| B | 7.70 | 33.7 M | **33.5 M** | 198.3 | 0.021 | 0.353 |
| A2 | 7.89 | 33.5 M | 40.0 M | 199.3 | 0.020 | 0.352 |
| C | 6.63 | 33.6 M | 33.3 M | 195.2 | 0.021 | 0.386 |

- **perf's `context-switches` and `cpu-migrations` read 0 in every arm, and that 0 means nothing.** Under
  `perf_event_paranoid` 2 perf adds `:u` to software events, which then never count (checked on divix01 before the
  run: a thread that slept ~27k times in 3 s read 0 and 0). The switch and migration columns above come from
  `/proc/<pid>/task/<tid>/status` and `sched`, sampled every ~5 s by the driver (`<arm>-sched.jsonl`).
- The thread's cycles are its idle loop, not its requests: it spins `spin_us`, then sleeps 50 µs (spec L12), so
  cycles/request is the same in all arms and ~200 voluntary switches/request are idle sleeps. Instructions differ:
  B retires **~17-19% fewer instructions per served request** than A/A2 (33.5 M against 41.5 M / 40.0 M) over the same
  cycles.

### 8c. C: whole-run shim counts of the production server

The shim is armed at load, so it counts each tracked thread from the moment it names itself until process exit,
including thread start-up and teardown. One dump (the scheduler, pid 1067138); both threads recognized once.

| thread | malloc | free | mutex | cond | clock | sleep | futex |
|---|---:|---:|---:|---:|---:|---:|---:|
| service (`exl3-ram-miss`) | **1** | **1** | 0 | 0 | 0 | 854,870 | 8 |
| copy (`exl3-copy-eng`) | **72** | **72** | **90,035,468** | **1** | 0 | 0 | 9,395 |

**The zero rule is not met as a whole-run count: this is an open finding, not a pass.**

To separate thread lifecycle from serving, the same shim mode was run on the branch's production host with no requests
at all (CPU `HostCopyBackend`, `hotpath_script.build_host(variant="prod")`, `start_thread`, idle 0.5 s or 3 s, then
`stop`, under `taskset -c 0-63`). Both runs gave service `{malloc 0, free 1}` and copy `{malloc 0, free 1, mutex 1,
cond 1}`. That lifecycle floor is:
- the `std::thread` state that libstdc++ deletes on the new thread after `run()` returns (one free on each thread);
- `CopyEngine::start`'s handshake on the copy thread (`start_mutex_`, `ready_cv_`: one mutex and one condvar).

Against that floor:
- **Service:** `free 1` and `mutex 0`/`cond 0` are the floor. **`malloc 1` is one allocation beyond it,
  unattributed.** It is one call over the whole run, not per request (2995 served), but the shim cannot say where.
- **Copy:** `cond 1` and one mutex are the floor. **The rest (90.0 M mutex, 72 malloc/free, 9,395 futex) come from the
  CUDA copy backend.** The CPU backend's per-request copy counts are 0 (§5a, Task 15), and the only difference here is
  `CudaCopyBackend`, which calls `cuMemcpyAsync`, `cuEventRecord` and `cuEventQuery` through libcuda. ~90 M mutex ops
  over the server's life is what a copy thread polling `cuEventQuery` in its spin loop would produce if libcuda takes a
  pthread mutex per query. That is an inference: the shim records no call sites.
- `sleep` 854,870 on the service is the idle 50 µs sleep plus the parked 20 µs sleep during pauses: both are
  `sleep_for`, and the shim cannot tell them apart. Copy `sleep` 0: its idle wait is a futex. Service `futex` 8 are
  `submit()`'s wakes of a sleeping copy thread; the copy thread's 9,395 futex calls include its idle waits and any
  that libcuda makes through `syscall()`.

### 8d. Verdict and limits

- **Timed arms pass:** byte-identical output (B, A2 and C against A), `read_errors` 0, and B 0.32 ms/token faster than
  mean(A, A2), inside a 1.5 ms allowance with 0.42 ms drift.
- **C does not meet the whole-run zero rule** for malloc/free/mutex/cond. `arms-report.json`'s `pass` is `false` for
  that reason alone. What is still open:
  - the service thread's single unattributed malloc;
  - the copy thread's libcuda-side mutex, malloc and futex traffic, which production's CUDA copy backend incurs and the
    CPU shim tests cannot see.
  Attributing both needs a shim that records call sites (e.g. a backtrace of the first N calls per kind) or a window
  armed after start-up, in one more server run.
- **Limits:** one pass of 90 decode tokens per arm, so the ms/token comparison resolves about the 1.5 ms allowance and
  no finer. perf and `served` cover different spans. The instruction saving is the only per-request CPU difference
  large enough to read.
