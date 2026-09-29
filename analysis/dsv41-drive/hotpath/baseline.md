# Hotpath zero-overhead: baseline test counts at ba01695c35

Date: 2026-09-29. divix01 kernel: `6.12.0-211.60.1.el10_2.x86_64`. Base commit `ba01695c35` (`origin/master`),
checked out detached at `/data/models/slang/nvfp4-work/wt-hotpath-base`.

Every later task's suite count is compared against the numbers on this page, using the same commands.

## 1. SUITE (kernels + layers/moe together)

```bash
cd /data/models/slang/nvfp4-work/wt-hotpath-base && export PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 \
  CUDA_HOME=/usr/local/cuda-13.4 \
  && /data/models/slang/.venv/bin/python -c "import sglang; print(sglang.__file__)" \
  && flock /data/models/slang/nvfp4-work/cc-gpu.lock taskset -c 32-63 /data/models/slang/.venv/bin/python -m pytest \
     test/registered/unit/kernels test/registered/unit/layers/moe -q -p no:randomly
```

`sglang.__file__` resolved to `/data/models/slang/nvfp4-work/wt-hotpath-base/python/sglang/__init__.py` (correct
worktree). **Result: `Interrupted: 11 errors during collection`, `EXIT=2`.** This is the pre-existing, environmental
`layers/moe` collection failure documented in `.claude/rules/divix01-run-protocol.md` ("Point the registered suite at
`unit/kernels`, not the whole tree"): all 11 errors are

```
AttributeError: module 'pyarrow' has no attribute 'PyExtensionType'. Did you mean: 'ExtensionType'?
```

raised while importing `python/sglang/benchmark/datasets/mmmu.py` -> `datasets` -> `pyarrow` (pyarrow 25.0.1
incompatibility), during collection of:

```
test_deepep_v2_buffer_lifecycle.py, test_deepep_v2_masked_slab.py, test_deepep_v2_wire_dtype.py,
test_flashinfer_a2a_wide_ep.py, test_fused_moe_native.py, test_fused_shared_expert_scaling.py,
test_mega_moe_deepgemm_api.py, test_topk_correction_bias_cache.py, test_w4afp8_deepep_dtype.py,
test_w4afp8_deepep_post_reorder.py, test_w4afp8_requant_geometry.py
```

It reproduces at any commit with none of this branch's work present, and pytest aborts the whole run on collection
errors, so `SUITE` (as the global-context template defines it, `kernels` + `layers/moe` together) gives **no signal**
at `ba01695c35`. This is not a regression to fix; it is a pre-existing environment defect. No `SUITE` pass/skip count
exists to compare against at this base commit.

## 2. kernels-only (the lead's comparison target)

```bash
cd /data/models/slang/nvfp4-work/wt-hotpath-base && export PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 \
  CUDA_HOME=/usr/local/cuda-13.4 \
  && flock /data/models/slang/nvfp4-work/cc-gpu.lock taskset -c 32-63 /data/models/slang/.venv/bin/python -m pytest \
     test/registered/unit/kernels -q -p no:randomly
```

**Result: `1933 passed, 23 skipped` in 522.93s (0:08:42), `EXIT=0`** (`${PIPESTATUS[0]}` read explicitly).

## 3. GPU manual test list

```bash
cd /data/models/slang/nvfp4-work/wt-hotpath-base && export PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 \
  CUDA_HOME=/usr/local/cuda-13.4 && flock /data/models/slang/nvfp4-work/cc-gpu.lock taskset -c 32-63 \
  /data/models/slang/.venv/bin/python -m pytest -q -rf -p no:randomly \
    test/manual/dsv41/test_exl3_ram_miss_cuda.py test/manual/dsv41/test_exl3_lease_kernels_cuda.py \
    test/manual/dsv41/test_exl3_copy_engine_cuda.py test/manual/dsv41/test_exl3_piece_stream_row_images_cuda.py \
    test/manual/dsv41/test_exl3_two_phase_parity_cuda.py test/manual/dsv41/test_exl3_two_phase_failure_cuda.py \
    test/manual/dsv41/test_exl3_two_phase_timing_cuda.py test/manual/dsv41/test_exl3_native_prefetch_cuda.py \
    test/manual/dsv41/test_exl3_task5_item4_gpu.py test/manual/dsv41/test_exl3_task5_item6_shutdown_gpu.py \
    test/manual/dsv41/test_exl3_ram_miss_graph_gpu.py
```

(`test_exl3_piece_stream_cuda.py` is intentionally excluded per the plan: it exercises the packed path, which Task 6
deletes.)

**Result: `24 failed, 119 passed, 3 skipped` in 165.52s (0:02:45), `EXIT=1`.** These failures are present at the base
commit with none of this branch's work applied, so they are a pre-existing baseline condition, not something Task 1
introduced. They are recorded here (not diagnosed — Task 1 is baseline-only, no code changes) so later tasks compare
against `24 failed / 119 passed / 3 skipped`, not `0 failed`. Full FAILED list:

```
test_exl3_task5_item4_gpu.py::test_a_fatal_demand_error_drops_the_failed_layer_and_every_later_layer_and_runs_no_expert[leases_off-timeout]
test_exl3_task5_item4_gpu.py::test_a_fatal_demand_error_drops_the_failed_layer_and_every_later_layer_and_runs_no_expert[leases_off-read_fault]
test_exl3_task5_item4_gpu.py::test_a_fatal_demand_error_drops_the_failed_layer_and_every_later_layer_and_runs_no_expert[leases_on-timeout]
test_exl3_task5_item4_gpu.py::test_a_fatal_demand_error_drops_the_failed_layer_and_every_later_layer_and_runs_no_expert[leases_on-read_fault]
test_exl3_task5_item6_shutdown_gpu.py::test_shutdown_does_not_free_a_tier_while_the_gpu_still_has_work_in_flight[default_stream]
test_exl3_task5_item6_shutdown_gpu.py::test_shutdown_does_not_free_a_tier_while_the_gpu_still_has_work_in_flight[side_stream]
test_exl3_task5_item6_shutdown_gpu.py::test_shutdown_ends_a_gpu_reader_waiting_on_the_service_without_waiting_out_its_timeout
test_exl3_ram_miss_graph_gpu.py::test_direct_insert_replay_hit_evict_refetch_and_prefill_handoff[timeout-generic_routes]
test_exl3_ram_miss_graph_gpu.py::test_direct_insert_replay_hit_evict_refetch_and_prefill_handoff[timeout-fused_routes]
test_exl3_ram_miss_graph_gpu.py::test_direct_insert_replay_hit_evict_refetch_and_prefill_handoff[generation_violation-generic_routes]
test_exl3_ram_miss_graph_gpu.py::test_direct_insert_replay_hit_evict_refetch_and_prefill_handoff[generation_violation-fused_routes]
test_exl3_ram_miss_graph_gpu.py::test_direct_insert_replay_hit_evict_refetch_and_prefill_handoff[two_phase_timeout-generic_routes]
test_exl3_ram_miss_graph_gpu.py::test_direct_insert_replay_hit_evict_refetch_and_prefill_handoff[two_phase_timeout-fused_routes]
test_exl3_ram_miss_graph_gpu.py::test_ram_misses_inside_a_replay_are_served[leases_off]
test_exl3_ram_miss_graph_gpu.py::test_ram_misses_inside_a_replay_are_served[leases_on]
test_exl3_ram_miss_graph_gpu.py::test_a_forced_timeout_fails_stop_without_hanging[leases_off]
test_exl3_ram_miss_graph_gpu.py::test_a_forced_timeout_fails_stop_without_hanging[leases_on]
test_exl3_ram_miss_graph_gpu.py::test_lease_mode_output_is_byte_exact_against_off
test_exl3_ram_miss_graph_gpu.py::test_many_layers_in_one_replay_are_served[leases_off-layers_4]
test_exl3_ram_miss_graph_gpu.py::test_many_layers_in_one_replay_are_served[leases_off-layers_20]
test_exl3_ram_miss_graph_gpu.py::test_many_layers_in_one_replay_are_served[leases_on-layers_4]
test_exl3_ram_miss_graph_gpu.py::test_many_layers_in_one_replay_are_served[leases_on-layers_20]
test_exl3_ram_miss_graph_gpu.py::test_lease_mode_multi_layer_graph_is_byte_exact_against_off[layers_4]
test_exl3_ram_miss_graph_gpu.py::test_lease_mode_multi_layer_graph_is_byte_exact_against_off[layers_20]
```

## 4. Per-file pass/skip counts: `*pack*`, `*row_image*`, `*exl3_ram_miss*` under `test/registered/unit/kernels/`

Later tasks delete the packed path, so this table anchors the test-by-test delta they must explain. Command (per-file
collected counts, cross-checked against a full `-q -rs -p no:randomly` run of the same 29 files, which reported
`1434 passed, 1 skipped` against `1435 collected` -- the counts below are exact since only one test in the whole set
skips):

```bash
cd /data/models/slang/nvfp4-work/wt-hotpath-base && export PYTHONPATH=$PWD/python \
  && /data/models/slang/.venv/bin/python -m pytest --collect-only -q <29 files> -p no:randomly
# cross-check:
cd /data/models/slang/nvfp4-work/wt-hotpath-base && export PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 \
  CUDA_HOME=/usr/local/cuda-13.4 && flock /data/models/slang/nvfp4-work/cc-gpu.lock taskset -c 32-63 \
  /data/models/slang/.venv/bin/python -m pytest -q -rs -p no:randomly <29 files> --durations=0
```

Files matched by name glob `*pack*`, `*row_image*`, `*exl3_ram_miss*` in `test/registered/unit/kernels/`, plus
`test_expert_stream_reader_split.py` (matched by content: the only file naming `PackReader`/`PackPool` directly, even
though its filename does not match a glob):

| File | collected | passed | skipped |
|---|---:|---:|---:|
| test_build_row_images.py | 7 | 7 | 0 |
| test_exl3_ram_miss_advisory.py | 3 | 3 | 0 |
| test_exl3_ram_miss_attach_lanes.py | 5 | 5 | 0 |
| test_exl3_ram_miss_copy_engine.py | 19 | 19 | 0 |
| test_exl3_ram_miss_device_args.py | 35 | 35 | 0 |
| test_exl3_ram_miss_lease_defer.py | 8 | 8 | 0 |
| test_exl3_ram_miss_lease_service.py | 21 | 21 | 0 |
| test_exl3_ram_miss_leases.py | 14 | 14 | 0 |
| test_exl3_ram_miss_lease_thread.py | 8 | 8 | 0 |
| test_exl3_ram_miss_lease_wrap.py | 2 | 2 | 0 |
| test_exl3_ram_miss_pack_workers.py | 550 | 549 | 1 |
| test_exl3_ram_miss_piece_stream_parts.py | 7 | 7 | 0 |
| test_exl3_ram_miss_piece_stream.py | 157 | 157 | 0 |
| test_exl3_ram_miss_prefill_fills.py | 16 | 16 | 0 |
| test_exl3_ram_miss_prefill_share.py | 16 | 16 | 0 |
| test_exl3_ram_miss_row_images.py | 262 | 262 | 0 |
| test_exl3_ram_miss_split.py | 137 | 137 | 0 |
| test_exl3_ram_miss_stage_trace_causal.py | 19 | 19 | 0 |
| test_exl3_ram_miss_stage_trace_lanes.py | 7 | 7 | 0 |
| test_exl3_ram_miss_stage_trace.py | 15 | 15 | 0 |
| test_exl3_ram_miss_task5_item5_ack_independence.py | 7 | 7 | 0 |
| test_exl3_ram_miss_thread.py | 26 | 26 | 0 |
| test_exl3_ram_miss_tier.py | 57 | 57 | 0 |
| test_exl3_ram_miss_trace_export.py | 2 | 2 | 0 |
| test_exl3_ram_miss_two_phase.py | 4 | 4 | 0 |
| test_exl3_ram_miss_two_phase_victim.py | 2 | 2 | 0 |
| test_exl3_ram_miss_wrap.py | 10 | 10 | 0 |
| test_exl3_row_image.py | 15 | 15 | 0 |
| test_expert_stream_reader_split.py | 4 | 4 | 0 |
| **Total** | **1435** | **1434** | **1** |

The single skip is `test_exl3_ram_miss_pack_workers.py:244`, reason: "needs a core with two allowed hyperthreads, and
another core" (an SMT-topology skip, environmental, not related to this plan's work).

### Content search: which of the above (or other kernels files) reference the packed path by symbol

```bash
cd /data/models/slang/nvfp4-work/wt-hotpath-base/test/registered/unit/kernels
grep -lE "PackReader|PackPool" *.py     # class-name references
grep -lE "pack_workers|PACK_WORKERS" *.py  # deprecated-knob references
grep -lE "bounce" *.py                     # bounce-buffer references (shared infra term, NOT packed-path-specific --
                                            # row-image reads also use bounce buffers for O_DIRECT alignment)
```

- `PackReader`/`PackPool` (strongest signal, direct class references): only `test_expert_stream_reader_split.py`.
- `pack_workers`/`PACK_WORKERS` (the deprecated knob name; present in many files as a parametrized/deprecation-warning
  case, not necessarily packed-path test *logic*): `test_exl3_native_prefetch_service.py`,
  `test_exl3_ram_miss_copy_engine.py`, `test_exl3_ram_miss_lease_service.py`, `test_exl3_ram_miss_pack_workers.py`,
  `test_exl3_ram_miss_piece_stream_parts.py`, `test_exl3_ram_miss_piece_stream.py`, `test_exl3_ram_miss_row_images.py`,
  `test_exl3_ram_miss_split.py`, `test_exl3_ram_miss_task5_item5_ack_independence.py`,
  `test_expert_stream_fixed_buffers.py`, `test_expert_stream_read_cuts.py`, `test_expert_stream_reader_golden.py`.
- `bounce` is a broad, shared-infrastructure term (present in 10 files including row-image and general reader-split
  tests) and is **not** by itself evidence of packed-path-specific test logic; whoever deletes the packed path (Task
  6) needs to read each file's content to separate packed-path tests from shared-infra tests that merely share the
  bounce-buffer type. This baseline records the grep hits so that judgment call is not made blind.

### Full test-ID list: `test_exl3_ram_miss_pack_workers.py` (name-glob match, the clearest packed-path-specific file; 550 tests, 1 skip)

<details>
<summary>550 test IDs (click to expand)</summary>

- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_fault_leaves_the_ring_clean[w1c1-fault0-1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_fault_leaves_the_ring_clean[w1c1-fault1-1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_fault_leaves_the_ring_clean[w1c1-fault2-1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_fault_leaves_the_ring_clean[w1c1-fault3-0]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_fault_leaves_the_ring_clean[w1c1-fault4-0]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_fault_leaves_the_ring_clean[w1c1-fault5-1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_fault_leaves_the_ring_clean[w1c1-fault6-1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_fault_leaves_the_ring_clean[w1c1-fault7-0]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_fault_leaves_the_ring_clean[w3c1-fault0-1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_fault_leaves_the_ring_clean[w3c1-fault1-1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_fault_leaves_the_ring_clean[w3c1-fault2-1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_fault_leaves_the_ring_clean[w3c1-fault3-0]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_fault_leaves_the_ring_clean[w3c1-fault4-0]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_fault_leaves_the_ring_clean[w3c1-fault5-1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_fault_leaves_the_ring_clean[w3c1-fault6-1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_fault_leaves_the_ring_clean[w3c1-fault7-0]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_fault_leaves_the_ring_clean[w3c3-fault0-1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_fault_leaves_the_ring_clean[w3c3-fault1-1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_fault_leaves_the_ring_clean[w3c3-fault2-1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_fault_leaves_the_ring_clean[w3c3-fault3-0]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_fault_leaves_the_ring_clean[w3c3-fault4-0]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_fault_leaves_the_ring_clean[w3c3-fault5-1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_fault_leaves_the_ring_clean[w3c3-fault6-1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_fault_leaves_the_ring_clean[w3c3-fault7-0]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_fault_leaves_the_ring_clean[w1c1ps-fault0-1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_fault_leaves_the_ring_clean[w1c1ps-fault1-1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_fault_leaves_the_ring_clean[w1c1ps-fault2-1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_fault_leaves_the_ring_clean[w1c1ps-fault3-0]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_fault_leaves_the_ring_clean[w1c1ps-fault4-0]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_fault_leaves_the_ring_clean[w1c1ps-fault5-1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_fault_leaves_the_ring_clean[w1c1ps-fault6-1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_fault_leaves_the_ring_clean[w1c1ps-fault7-0]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_fault_leaves_the_ring_clean[w3c3ps-fault0-1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_fault_leaves_the_ring_clean[w3c3ps-fault1-1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_fault_leaves_the_ring_clean[w3c3ps-fault2-1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_fault_leaves_the_ring_clean[w3c3ps-fault3-0]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_fault_leaves_the_ring_clean[w3c3ps-fault4-0]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_fault_leaves_the_ring_clean[w3c3ps-fault5-1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_fault_leaves_the_ring_clean[w3c3ps-fault6-1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_fault_leaves_the_ring_clean[w3c3ps-fault7-0]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_batch_needing_more_reads_than_its_credit_completes_through_refill[w1c1-1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_batch_needing_more_reads_than_its_credit_completes_through_refill[w1c1-3]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_batch_needing_more_reads_than_its_credit_completes_through_refill[w1c1-5]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_batch_needing_more_reads_than_its_credit_completes_through_refill[w3c1-1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_batch_needing_more_reads_than_its_credit_completes_through_refill[w3c1-3]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_batch_needing_more_reads_than_its_credit_completes_through_refill[w3c1-5]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_batch_needing_more_reads_than_its_credit_completes_through_refill[w3c3-1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_batch_needing_more_reads_than_its_credit_completes_through_refill[w3c3-3]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_batch_needing_more_reads_than_its_credit_completes_through_refill[w3c3-5]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_batch_needing_more_reads_than_its_credit_completes_through_refill[w1c1ps-1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_batch_needing_more_reads_than_its_credit_completes_through_refill[w1c1ps-3]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_batch_needing_more_reads_than_its_credit_completes_through_refill[w1c1ps-5]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_batch_needing_more_reads_than_its_credit_completes_through_refill[w3c3ps-1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_batch_needing_more_reads_than_its_credit_completes_through_refill[w3c3ps-3]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_batch_needing_more_reads_than_its_credit_completes_through_refill[w3c3ps-5]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_completions_processed_back_to_front_land_the_same_bytes[w1c1-0]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_completions_processed_back_to_front_land_the_same_bytes[w1c1-3]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_completions_processed_back_to_front_land_the_same_bytes[w3c1-0]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_completions_processed_back_to_front_land_the_same_bytes[w3c1-3]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_completions_processed_back_to_front_land_the_same_bytes[w3c3-0]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_completions_processed_back_to_front_land_the_same_bytes[w3c3-3]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_completions_processed_back_to_front_land_the_same_bytes[w1c1ps-0]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_completions_processed_back_to_front_land_the_same_bytes[w1c1ps-3]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_completions_processed_back_to_front_land_the_same_bytes[w3c3ps-0]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_completions_processed_back_to_front_land_the_same_bytes[w3c3ps-3]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_short_read_resubmits_its_own_extent_under_reversed_completions[w1c1-0]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_short_read_resubmits_its_own_extent_under_reversed_completions[w1c1-1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_short_read_resubmits_its_own_extent_under_reversed_completions[w3c1-0]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_short_read_resubmits_its_own_extent_under_reversed_completions[w3c1-1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_short_read_resubmits_its_own_extent_under_reversed_completions[w3c3-0]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_short_read_resubmits_its_own_extent_under_reversed_completions[w3c3-1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_zero_length_part_issues_no_read[w1c1-weights0-2]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_zero_length_part_issues_no_read[w1c1-weights1-1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_zero_length_part_issues_no_read[w1c1-weights2-1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_zero_length_part_issues_no_read[w3c1-weights0-2]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_zero_length_part_issues_no_read[w3c1-weights1-1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_zero_length_part_issues_no_read[w3c1-weights2-1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_zero_length_part_issues_no_read[w3c3-weights0-2]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_zero_length_part_issues_no_read[w3c3-weights1-1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_zero_length_part_issues_no_read[w3c3-weights2-1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_zero_length_part_issues_no_read[w1c1ps-weights0-2]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_zero_length_part_issues_no_read[w1c1ps-weights1-1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_zero_length_part_issues_no_read[w1c1ps-weights2-1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_zero_length_part_issues_no_read[w3c3ps-weights0-2]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_zero_length_part_issues_no_read[w3c3ps-weights1-1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_zero_length_part_issues_no_read[w3c3ps-weights2-1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_failed_part_fails_the_row_and_never_leaves_a_half_packed_one[w1c1-0]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_failed_part_fails_the_row_and_never_leaves_a_half_packed_one[w1c1-1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_failed_part_fails_the_row_and_never_leaves_a_half_packed_one[w3c1-0]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_failed_part_fails_the_row_and_never_leaves_a_half_packed_one[w3c1-1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_failed_part_fails_the_row_and_never_leaves_a_half_packed_one[w3c3-0]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_failed_part_fails_the_row_and_never_leaves_a_half_packed_one[w3c3-1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_failed_part_fails_the_row_and_never_leaves_a_half_packed_one[w1c1ps-0]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_failed_part_fails_the_row_and_never_leaves_a_half_packed_one[w1c1ps-1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_failed_part_fails_the_row_and_never_leaves_a_half_packed_one[w3c3ps-0]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_failed_part_fails_the_row_and_never_leaves_a_half_packed_one[w3c3ps-1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_short_read_resubmits_only_its_own_extent[w1c1-False-0]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_short_read_resubmits_only_its_own_extent[w1c1-False-1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_short_read_resubmits_only_its_own_extent[w1c1-True-0]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_short_read_resubmits_only_its_own_extent[w1c1-True-1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_short_read_resubmits_only_its_own_extent[w3c1-False-0]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_short_read_resubmits_only_its_own_extent[w3c1-False-1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_short_read_resubmits_only_its_own_extent[w3c1-True-0]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_short_read_resubmits_only_its_own_extent[w3c1-True-1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_short_read_resubmits_only_its_own_extent[w3c3-False-0]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_short_read_resubmits_only_its_own_extent[w3c3-False-1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_short_read_resubmits_only_its_own_extent[w3c3-True-0]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_short_read_resubmits_only_its_own_extent[w3c3-True-1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_traced_read_moves_the_same_bytes_as_an_untraced_one[w1c1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_traced_read_moves_the_same_bytes_as_an_untraced_one[w3c1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_traced_read_moves_the_same_bytes_as_an_untraced_one[w3c3]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_traced_read_moves_the_same_bytes_as_an_untraced_one[w1c1ps]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_traced_read_moves_the_same_bytes_as_an_untraced_one[w3c3ps]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_mirrored_reads_are_accounted_per_drive[w1c1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_mirrored_reads_are_accounted_per_drive[w3c1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_mirrored_reads_are_accounted_per_drive[w3c3]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_mirrored_reads_are_accounted_per_drive[w1c1ps]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_mirrored_reads_are_accounted_per_drive[w3c3ps]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_an_empty_part_is_not_an_extent_and_reads_no_drive[w1c1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_an_empty_part_is_not_an_extent_and_reads_no_drive[w3c1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_an_empty_part_is_not_an_extent_and_reads_no_drive[w3c3]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_an_empty_part_is_not_an_extent_and_reads_no_drive[w1c1ps]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_an_empty_part_is_not_an_extent_and_reads_no_drive[w3c3ps]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_fault_leaves_a_consistent_record[w1c1-fault0-1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_fault_leaves_a_consistent_record[w1c1-fault1-1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_fault_leaves_a_consistent_record[w1c1-fault2-1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_fault_leaves_a_consistent_record[w1c1-fault3-0]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_fault_leaves_a_consistent_record[w1c1-fault4-0]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_fault_leaves_a_consistent_record[w3c1-fault0-1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_fault_leaves_a_consistent_record[w3c1-fault1-1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_fault_leaves_a_consistent_record[w3c1-fault2-1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_fault_leaves_a_consistent_record[w3c1-fault3-0]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_fault_leaves_a_consistent_record[w3c1-fault4-0]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_fault_leaves_a_consistent_record[w3c3-fault0-1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_fault_leaves_a_consistent_record[w3c3-fault1-1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_fault_leaves_a_consistent_record[w3c3-fault2-1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_fault_leaves_a_consistent_record[w3c3-fault3-0]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_fault_leaves_a_consistent_record[w3c3-fault4-0]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_fault_leaves_a_consistent_record[w1c1ps-fault0-1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_fault_leaves_a_consistent_record[w1c1ps-fault1-1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_fault_leaves_a_consistent_record[w1c1ps-fault2-1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_fault_leaves_a_consistent_record[w1c1ps-fault3-0]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_fault_leaves_a_consistent_record[w1c1ps-fault4-0]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_fault_leaves_a_consistent_record[w3c3ps-fault0-1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_fault_leaves_a_consistent_record[w3c3ps-fault1-1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_fault_leaves_a_consistent_record[w3c3ps-fault2-1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_fault_leaves_a_consistent_record[w3c3ps-fault3-0]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_fault_leaves_a_consistent_record[w3c3ps-fault4-0]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_the_byte_split_of_a_clean_read[w1c1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_the_byte_split_of_a_clean_read[w3c1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_the_byte_split_of_a_clean_read[w3c3]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_the_byte_split_of_a_clean_read[w1c1ps]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_the_byte_split_of_a_clean_read[w3c3ps]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_short_read_is_retried_bytes_and_adds_nothing_to_useful[w1c1-0]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_short_read_is_retried_bytes_and_adds_nothing_to_useful[w1c1-1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_short_read_is_retried_bytes_and_adds_nothing_to_useful[w3c1-0]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_short_read_is_retried_bytes_and_adds_nothing_to_useful[w3c1-1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_short_read_is_retried_bytes_and_adds_nothing_to_useful[w3c3-0]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_short_read_is_retried_bytes_and_adds_nothing_to_useful[w3c3-1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_an_interrupted_read_is_resubmitted_whole_and_counted_as_retried[w1c1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_an_interrupted_read_is_resubmitted_whole_and_counted_as_retried[w3c1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_an_interrupted_read_is_resubmitted_whole_and_counted_as_retried[w3c3]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_failed_read_cancels_the_bytes_it_never_received[w1c1-fault0]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_failed_read_cancels_the_bytes_it_never_received[w1c1-fault1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_failed_read_cancels_the_bytes_it_never_received[w1c1-fault2]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_failed_read_cancels_the_bytes_it_never_received[w3c1-fault0]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_failed_read_cancels_the_bytes_it_never_received[w3c1-fault1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_failed_read_cancels_the_bytes_it_never_received[w3c1-fault2]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_failed_read_cancels_the_bytes_it_never_received[w3c3-fault0]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_failed_read_cancels_the_bytes_it_never_received[w3c3-fault1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_failed_read_cancels_the_bytes_it_never_received[w3c3-fault2]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_failed_read_cancels_the_bytes_it_never_received[w1c1ps-fault0]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_failed_read_cancels_the_bytes_it_never_received[w1c1ps-fault1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_failed_read_cancels_the_bytes_it_never_received[w1c1ps-fault2]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_failed_read_cancels_the_bytes_it_never_received[w3c3ps-fault0]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_failed_read_cancels_the_bytes_it_never_received[w3c3ps-fault1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_failed_read_cancels_the_bytes_it_never_received[w3c3ps-fault2]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_batch_that_fails_after_a_success_cancels_only_its_own_bytes[w1c1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_batch_that_fails_after_a_success_cancels_only_its_own_bytes[w3c1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_batch_that_fails_after_a_success_cancels_only_its_own_bytes[w3c3]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_batch_that_fails_after_a_success_cancels_only_its_own_bytes[w1c1ps]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_batch_that_fails_after_a_success_cancels_only_its_own_bytes[w3c3ps]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_each_rows_stamps_are_causally_ordered_over_several_batches[w1c1-None-fault0]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_each_rows_stamps_are_causally_ordered_over_several_batches[w1c1-weights1-fault1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_each_rows_stamps_are_causally_ordered_over_several_batches[w1c1-weights2-fault2]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_each_rows_stamps_are_causally_ordered_over_several_batches[w1c1-weights3-fault3]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_each_rows_stamps_are_causally_ordered_over_several_batches[w3c1-None-fault0]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_each_rows_stamps_are_causally_ordered_over_several_batches[w3c1-weights1-fault1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_each_rows_stamps_are_causally_ordered_over_several_batches[w3c1-weights2-fault2]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_each_rows_stamps_are_causally_ordered_over_several_batches[w3c1-weights3-fault3]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_each_rows_stamps_are_causally_ordered_over_several_batches[w3c3-None-fault0]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_each_rows_stamps_are_causally_ordered_over_several_batches[w3c3-weights1-fault1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_each_rows_stamps_are_causally_ordered_over_several_batches[w3c3-weights2-fault2]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_each_rows_stamps_are_causally_ordered_over_several_batches[w3c3-weights3-fault3]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_each_rows_stamps_are_causally_ordered_over_several_batches[w1c1ps-None-fault0]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_each_rows_stamps_are_causally_ordered_over_several_batches[w1c1ps-weights1-fault1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_each_rows_stamps_are_causally_ordered_over_several_batches[w1c1ps-weights2-fault2]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_each_rows_stamps_are_causally_ordered_over_several_batches[w1c1ps-weights3-fault3]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_each_rows_stamps_are_causally_ordered_over_several_batches[w3c3ps-None-fault0]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_each_rows_stamps_are_causally_ordered_over_several_batches[w3c3ps-weights1-fault1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_each_rows_stamps_are_causally_ordered_over_several_batches[w3c3ps-weights2-fault2]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_each_rows_stamps_are_causally_ordered_over_several_batches[w3c3ps-weights3-fault3]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_per_row_and_per_extent_stamps_are_bounded_and_the_overflow_counted[w1c1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_per_row_and_per_extent_stamps_are_bounded_and_the_overflow_counted[w3c1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_per_row_and_per_extent_stamps_are_bounded_and_the_overflow_counted[w3c3]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_per_row_and_per_extent_stamps_are_bounded_and_the_overflow_counted[w1c1ps]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_per_row_and_per_extent_stamps_are_bounded_and_the_overflow_counted[w3c3ps]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_traced_read_is_byte_identical_to_an_untraced_one[w1c1-None-fault0]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_traced_read_is_byte_identical_to_an_untraced_one[w1c1-weights1-fault1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_traced_read_is_byte_identical_to_an_untraced_one[w1c1-weights2-fault2]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_traced_read_is_byte_identical_to_an_untraced_one[w1c1-weights3-fault3]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_traced_read_is_byte_identical_to_an_untraced_one[w3c1-None-fault0]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_traced_read_is_byte_identical_to_an_untraced_one[w3c1-weights1-fault1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_traced_read_is_byte_identical_to_an_untraced_one[w3c1-weights2-fault2]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_traced_read_is_byte_identical_to_an_untraced_one[w3c1-weights3-fault3]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_traced_read_is_byte_identical_to_an_untraced_one[w3c3-None-fault0]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_traced_read_is_byte_identical_to_an_untraced_one[w3c3-weights1-fault1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_traced_read_is_byte_identical_to_an_untraced_one[w3c3-weights2-fault2]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_traced_read_is_byte_identical_to_an_untraced_one[w3c3-weights3-fault3]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_traced_read_is_byte_identical_to_an_untraced_one[w1c1ps-None-fault0]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_traced_read_is_byte_identical_to_an_untraced_one[w1c1ps-weights1-fault1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_traced_read_is_byte_identical_to_an_untraced_one[w1c1ps-weights2-fault2]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_traced_read_is_byte_identical_to_an_untraced_one[w1c1ps-weights3-fault3]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_traced_read_is_byte_identical_to_an_untraced_one[w3c3ps-None-fault0]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_traced_read_is_byte_identical_to_an_untraced_one[w3c3ps-weights1-fault1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_traced_read_is_byte_identical_to_an_untraced_one[w3c3ps-weights2-fault2]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_traced_read_is_byte_identical_to_an_untraced_one[w3c3ps-weights3-fault3]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_row_packs_while_another_rows_read_is_still_outstanding[w1c1-None]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_row_packs_while_another_rows_read_is_still_outstanding[w1c1-weights1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_row_packs_while_another_rows_read_is_still_outstanding[w3c1-None]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_row_packs_while_another_rows_read_is_still_outstanding[w3c1-weights1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_row_packs_while_another_rows_read_is_still_outstanding[w3c3-None]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_row_packs_while_another_rows_read_is_still_outstanding[w3c3-weights1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_row_packs_while_another_rows_read_is_still_outstanding[w1c1ps-None]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_row_packs_while_another_rows_read_is_still_outstanding[w1c1ps-weights1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_row_packs_while_another_rows_read_is_still_outstanding[w3c3ps-None]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_row_packs_while_another_rows_read_is_still_outstanding[w3c3ps-weights1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_bank_is_not_reused_until_every_row_in_it_has_packed[w1c1-fault0]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_bank_is_not_reused_until_every_row_in_it_has_packed[w1c1-fault1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_bank_is_not_reused_until_every_row_in_it_has_packed[w1c1-fault2]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_bank_is_not_reused_until_every_row_in_it_has_packed[w1c1-fault3]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_bank_is_not_reused_until_every_row_in_it_has_packed[w1c1-fault4]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_bank_is_not_reused_until_every_row_in_it_has_packed[w3c1-fault0]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_bank_is_not_reused_until_every_row_in_it_has_packed[w3c1-fault1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_bank_is_not_reused_until_every_row_in_it_has_packed[w3c1-fault2]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_bank_is_not_reused_until_every_row_in_it_has_packed[w3c1-fault3]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_bank_is_not_reused_until_every_row_in_it_has_packed[w3c1-fault4]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_bank_is_not_reused_until_every_row_in_it_has_packed[w3c3-fault0]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_bank_is_not_reused_until_every_row_in_it_has_packed[w3c3-fault1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_bank_is_not_reused_until_every_row_in_it_has_packed[w3c3-fault2]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_bank_is_not_reused_until_every_row_in_it_has_packed[w3c3-fault3]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_bank_is_not_reused_until_every_row_in_it_has_packed[w3c3-fault4]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_bank_is_not_reused_until_every_row_in_it_has_packed[w1c1ps-fault0]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_bank_is_not_reused_until_every_row_in_it_has_packed[w1c1ps-fault1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_bank_is_not_reused_until_every_row_in_it_has_packed[w1c1ps-fault2]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_bank_is_not_reused_until_every_row_in_it_has_packed[w1c1ps-fault3]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_bank_is_not_reused_until_every_row_in_it_has_packed[w1c1ps-fault4]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_bank_is_not_reused_until_every_row_in_it_has_packed[w3c3ps-fault0]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_bank_is_not_reused_until_every_row_in_it_has_packed[w3c3ps-fault1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_bank_is_not_reused_until_every_row_in_it_has_packed[w3c3ps-fault2]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_bank_is_not_reused_until_every_row_in_it_has_packed[w3c3ps-fault3]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_bank_is_not_reused_until_every_row_in_it_has_packed[w3c3ps-fault4]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_ring_credit_is_independent_of_the_banks[w1c1-0]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_ring_credit_is_independent_of_the_banks[w1c1-2]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_ring_credit_is_independent_of_the_banks[w1c1-3]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_ring_credit_is_independent_of_the_banks[w1c1-5]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_ring_credit_is_independent_of_the_banks[w3c1-0]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_ring_credit_is_independent_of_the_banks[w3c1-2]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_ring_credit_is_independent_of_the_banks[w3c1-3]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_ring_credit_is_independent_of_the_banks[w3c1-5]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_ring_credit_is_independent_of_the_banks[w3c3-0]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_ring_credit_is_independent_of_the_banks[w3c3-2]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_ring_credit_is_independent_of_the_banks[w3c3-3]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_ring_credit_is_independent_of_the_banks[w3c3-5]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_ring_credit_is_independent_of_the_banks[w1c1ps-0]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_ring_credit_is_independent_of_the_banks[w1c1ps-2]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_ring_credit_is_independent_of_the_banks[w1c1ps-3]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_ring_credit_is_independent_of_the_banks[w1c1ps-5]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_ring_credit_is_independent_of_the_banks[w3c3ps-0]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_ring_credit_is_independent_of_the_banks[w3c3ps-2]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_ring_credit_is_independent_of_the_banks[w3c3ps-3]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_ring_credit_is_independent_of_the_banks[w3c3ps-5]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_refill_submits_a_credit_freed_read_before_the_next_rows_blocking_pack[w1c1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_refill_submits_a_credit_freed_read_before_the_next_rows_blocking_pack[w3c1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_refill_submits_a_credit_freed_read_before_the_next_rows_blocking_pack[w3c3]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_refill_submits_a_credit_freed_read_before_the_next_rows_blocking_pack[w1c1ps]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_refill_submits_a_credit_freed_read_before_the_next_rows_blocking_pack[w3c3ps]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_poisoned_descriptors_and_slots_are_recycled_many_times_without_a_wrong_byte[w1c1-fault0]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_poisoned_descriptors_and_slots_are_recycled_many_times_without_a_wrong_byte[w1c1-fault1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_poisoned_descriptors_and_slots_are_recycled_many_times_without_a_wrong_byte[w1c1-fault2]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_poisoned_descriptors_and_slots_are_recycled_many_times_without_a_wrong_byte[w1c1-fault3]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_poisoned_descriptors_and_slots_are_recycled_many_times_without_a_wrong_byte[w3c1-fault0]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_poisoned_descriptors_and_slots_are_recycled_many_times_without_a_wrong_byte[w3c1-fault1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_poisoned_descriptors_and_slots_are_recycled_many_times_without_a_wrong_byte[w3c1-fault2]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_poisoned_descriptors_and_slots_are_recycled_many_times_without_a_wrong_byte[w3c1-fault3]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_poisoned_descriptors_and_slots_are_recycled_many_times_without_a_wrong_byte[w3c3-fault0]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_poisoned_descriptors_and_slots_are_recycled_many_times_without_a_wrong_byte[w3c3-fault1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_poisoned_descriptors_and_slots_are_recycled_many_times_without_a_wrong_byte[w3c3-fault2]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_poisoned_descriptors_and_slots_are_recycled_many_times_without_a_wrong_byte[w3c3-fault3]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_poisoned_descriptors_and_slots_are_recycled_many_times_without_a_wrong_byte[w1c1ps-fault0]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_poisoned_descriptors_and_slots_are_recycled_many_times_without_a_wrong_byte[w1c1ps-fault1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_poisoned_descriptors_and_slots_are_recycled_many_times_without_a_wrong_byte[w1c1ps-fault2]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_poisoned_descriptors_and_slots_are_recycled_many_times_without_a_wrong_byte[w1c1ps-fault3]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_poisoned_descriptors_and_slots_are_recycled_many_times_without_a_wrong_byte[w3c3ps-fault0]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_poisoned_descriptors_and_slots_are_recycled_many_times_without_a_wrong_byte[w3c3ps-fault1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_poisoned_descriptors_and_slots_are_recycled_many_times_without_a_wrong_byte[w3c3ps-fault2]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_poisoned_descriptors_and_slots_are_recycled_many_times_without_a_wrong_byte[w3c3ps-fault3]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_the_generation_counter_wraps_and_the_reader_goes_on[w1c1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_the_generation_counter_wraps_and_the_reader_goes_on[w3c1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_the_generation_counter_wraps_and_the_reader_goes_on[w3c3]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_the_generation_counter_wraps_and_the_reader_goes_on[w1c1ps]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_the_generation_counter_wraps_and_the_reader_goes_on[w3c3ps]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_completion_of_a_retired_extent_cannot_finish_the_extent_that_recycled_its_descriptor[w1c1-False]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_completion_of_a_retired_extent_cannot_finish_the_extent_that_recycled_its_descriptor[w1c1-True]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_completion_of_a_retired_extent_cannot_finish_the_extent_that_recycled_its_descriptor[w3c1-False]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_completion_of_a_retired_extent_cannot_finish_the_extent_that_recycled_its_descriptor[w3c1-True]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_completion_of_a_retired_extent_cannot_finish_the_extent_that_recycled_its_descriptor[w3c3-False]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_completion_of_a_retired_extent_cannot_finish_the_extent_that_recycled_its_descriptor[w3c3-True]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_completion_of_a_retired_extent_cannot_finish_the_extent_that_recycled_its_descriptor[w1c1ps-False]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_completion_of_a_retired_extent_cannot_finish_the_extent_that_recycled_its_descriptor[w1c1ps-True]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_completion_of_a_retired_extent_cannot_finish_the_extent_that_recycled_its_descriptor[w3c3ps-False]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_completion_of_a_retired_extent_cannot_finish_the_extent_that_recycled_its_descriptor[w3c3ps-True]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_submit_that_consumes_nothing_is_resubmitted[w1c1-None-1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_submit_that_consumes_nothing_is_resubmitted[w1c1-None-2]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_submit_that_consumes_nothing_is_resubmitted[w1c1-None-3]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_submit_that_consumes_nothing_is_resubmitted[w1c1-weights1-1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_submit_that_consumes_nothing_is_resubmitted[w1c1-weights1-2]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_submit_that_consumes_nothing_is_resubmitted[w1c1-weights1-3]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_submit_that_consumes_nothing_is_resubmitted[w3c1-None-1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_submit_that_consumes_nothing_is_resubmitted[w3c1-None-2]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_submit_that_consumes_nothing_is_resubmitted[w3c1-None-3]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_submit_that_consumes_nothing_is_resubmitted[w3c1-weights1-1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_submit_that_consumes_nothing_is_resubmitted[w3c1-weights1-2]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_submit_that_consumes_nothing_is_resubmitted[w3c1-weights1-3]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_submit_that_consumes_nothing_is_resubmitted[w3c3-None-1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_submit_that_consumes_nothing_is_resubmitted[w3c3-None-2]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_submit_that_consumes_nothing_is_resubmitted[w3c3-None-3]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_submit_that_consumes_nothing_is_resubmitted[w3c3-weights1-1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_submit_that_consumes_nothing_is_resubmitted[w3c3-weights1-2]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_submit_that_consumes_nothing_is_resubmitted[w3c3-weights1-3]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_submit_that_consumes_nothing_is_resubmitted[w1c1ps-None-1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_submit_that_consumes_nothing_is_resubmitted[w1c1ps-None-2]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_submit_that_consumes_nothing_is_resubmitted[w1c1ps-None-3]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_submit_that_consumes_nothing_is_resubmitted[w1c1ps-weights1-1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_submit_that_consumes_nothing_is_resubmitted[w1c1ps-weights1-2]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_submit_that_consumes_nothing_is_resubmitted[w1c1ps-weights1-3]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_submit_that_consumes_nothing_is_resubmitted[w3c3ps-None-1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_submit_that_consumes_nothing_is_resubmitted[w3c3ps-None-2]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_submit_that_consumes_nothing_is_resubmitted[w3c3ps-None-3]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_submit_that_consumes_nothing_is_resubmitted[w3c3ps-weights1-1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_submit_that_consumes_nothing_is_resubmitted[w3c3ps-weights1-2]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_submit_that_consumes_nothing_is_resubmitted[w3c3ps-weights1-3]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_hard_error_with_both_banks_in_flight_leaves_no_half_packed_row_and_a_clean_ring[w1c1-fault0]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_hard_error_with_both_banks_in_flight_leaves_no_half_packed_row_and_a_clean_ring[w1c1-fault1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_hard_error_with_both_banks_in_flight_leaves_no_half_packed_row_and_a_clean_ring[w1c1-fault2]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_hard_error_with_both_banks_in_flight_leaves_no_half_packed_row_and_a_clean_ring[w1c1-fault3]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_hard_error_with_both_banks_in_flight_leaves_no_half_packed_row_and_a_clean_ring[w1c1-fault4]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_hard_error_with_both_banks_in_flight_leaves_no_half_packed_row_and_a_clean_ring[w1c1-fault5]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_hard_error_with_both_banks_in_flight_leaves_no_half_packed_row_and_a_clean_ring[w1c1-fault6]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_hard_error_with_both_banks_in_flight_leaves_no_half_packed_row_and_a_clean_ring[w1c1-fault7]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_hard_error_with_both_banks_in_flight_leaves_no_half_packed_row_and_a_clean_ring[w3c1-fault0]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_hard_error_with_both_banks_in_flight_leaves_no_half_packed_row_and_a_clean_ring[w3c1-fault1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_hard_error_with_both_banks_in_flight_leaves_no_half_packed_row_and_a_clean_ring[w3c1-fault2]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_hard_error_with_both_banks_in_flight_leaves_no_half_packed_row_and_a_clean_ring[w3c1-fault3]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_hard_error_with_both_banks_in_flight_leaves_no_half_packed_row_and_a_clean_ring[w3c1-fault4]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_hard_error_with_both_banks_in_flight_leaves_no_half_packed_row_and_a_clean_ring[w3c1-fault5]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_hard_error_with_both_banks_in_flight_leaves_no_half_packed_row_and_a_clean_ring[w3c1-fault6]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_hard_error_with_both_banks_in_flight_leaves_no_half_packed_row_and_a_clean_ring[w3c1-fault7]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_hard_error_with_both_banks_in_flight_leaves_no_half_packed_row_and_a_clean_ring[w3c3-fault0]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_hard_error_with_both_banks_in_flight_leaves_no_half_packed_row_and_a_clean_ring[w3c3-fault1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_hard_error_with_both_banks_in_flight_leaves_no_half_packed_row_and_a_clean_ring[w3c3-fault2]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_hard_error_with_both_banks_in_flight_leaves_no_half_packed_row_and_a_clean_ring[w3c3-fault3]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_hard_error_with_both_banks_in_flight_leaves_no_half_packed_row_and_a_clean_ring[w3c3-fault4]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_hard_error_with_both_banks_in_flight_leaves_no_half_packed_row_and_a_clean_ring[w3c3-fault5]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_hard_error_with_both_banks_in_flight_leaves_no_half_packed_row_and_a_clean_ring[w3c3-fault6]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_hard_error_with_both_banks_in_flight_leaves_no_half_packed_row_and_a_clean_ring[w3c3-fault7]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_hard_error_with_both_banks_in_flight_leaves_no_half_packed_row_and_a_clean_ring[w1c1ps-fault0]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_hard_error_with_both_banks_in_flight_leaves_no_half_packed_row_and_a_clean_ring[w1c1ps-fault1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_hard_error_with_both_banks_in_flight_leaves_no_half_packed_row_and_a_clean_ring[w1c1ps-fault2]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_hard_error_with_both_banks_in_flight_leaves_no_half_packed_row_and_a_clean_ring[w1c1ps-fault3]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_hard_error_with_both_banks_in_flight_leaves_no_half_packed_row_and_a_clean_ring[w1c1ps-fault4]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_hard_error_with_both_banks_in_flight_leaves_no_half_packed_row_and_a_clean_ring[w1c1ps-fault5]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_hard_error_with_both_banks_in_flight_leaves_no_half_packed_row_and_a_clean_ring[w1c1ps-fault6]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_hard_error_with_both_banks_in_flight_leaves_no_half_packed_row_and_a_clean_ring[w1c1ps-fault7]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_hard_error_with_both_banks_in_flight_leaves_no_half_packed_row_and_a_clean_ring[w3c3ps-fault0]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_hard_error_with_both_banks_in_flight_leaves_no_half_packed_row_and_a_clean_ring[w3c3ps-fault1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_hard_error_with_both_banks_in_flight_leaves_no_half_packed_row_and_a_clean_ring[w3c3ps-fault2]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_hard_error_with_both_banks_in_flight_leaves_no_half_packed_row_and_a_clean_ring[w3c3ps-fault3]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_hard_error_with_both_banks_in_flight_leaves_no_half_packed_row_and_a_clean_ring[w3c3ps-fault4]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_hard_error_with_both_banks_in_flight_leaves_no_half_packed_row_and_a_clean_ring[w3c3ps-fault5]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_hard_error_with_both_banks_in_flight_leaves_no_half_packed_row_and_a_clean_ring[w3c3ps-fault6]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_hard_error_with_both_banks_in_flight_leaves_no_half_packed_row_and_a_clean_ring[w3c3ps-fault7]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_short_read_in_either_bank_resubmits_only_its_own_extent[w1c1-False-2]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_short_read_in_either_bank_resubmits_only_its_own_extent[w1c1-False-9]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_short_read_in_either_bank_resubmits_only_its_own_extent[w1c1-False-15]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_short_read_in_either_bank_resubmits_only_its_own_extent[w1c1-True-2]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_short_read_in_either_bank_resubmits_only_its_own_extent[w1c1-True-9]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_short_read_in_either_bank_resubmits_only_its_own_extent[w1c1-True-15]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_short_read_in_either_bank_resubmits_only_its_own_extent[w3c1-False-2]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_short_read_in_either_bank_resubmits_only_its_own_extent[w3c1-False-9]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_short_read_in_either_bank_resubmits_only_its_own_extent[w3c1-False-15]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_short_read_in_either_bank_resubmits_only_its_own_extent[w3c1-True-2]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_short_read_in_either_bank_resubmits_only_its_own_extent[w3c1-True-9]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_short_read_in_either_bank_resubmits_only_its_own_extent[w3c1-True-15]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_short_read_in_either_bank_resubmits_only_its_own_extent[w3c3-False-2]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_short_read_in_either_bank_resubmits_only_its_own_extent[w3c3-False-9]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_short_read_in_either_bank_resubmits_only_its_own_extent[w3c3-False-15]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_short_read_in_either_bank_resubmits_only_its_own_extent[w3c3-True-2]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_short_read_in_either_bank_resubmits_only_its_own_extent[w3c3-True-9]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_short_read_in_either_bank_resubmits_only_its_own_extent[w3c3-True-15]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_cancellation_reaps_what_was_submitted_and_reads_nothing_more[w1c1-fault0]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_cancellation_reaps_what_was_submitted_and_reads_nothing_more[w1c1-fault1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_cancellation_reaps_what_was_submitted_and_reads_nothing_more[w1c1-fault2]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_cancellation_reaps_what_was_submitted_and_reads_nothing_more[w3c1-fault0]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_cancellation_reaps_what_was_submitted_and_reads_nothing_more[w3c1-fault1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_cancellation_reaps_what_was_submitted_and_reads_nothing_more[w3c1-fault2]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_cancellation_reaps_what_was_submitted_and_reads_nothing_more[w3c3-fault0]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_cancellation_reaps_what_was_submitted_and_reads_nothing_more[w3c3-fault1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_cancellation_reaps_what_was_submitted_and_reads_nothing_more[w3c3-fault2]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_cancellation_reaps_what_was_submitted_and_reads_nothing_more[w1c1ps-fault0]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_cancellation_reaps_what_was_submitted_and_reads_nothing_more[w1c1ps-fault1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_cancellation_reaps_what_was_submitted_and_reads_nothing_more[w1c1ps-fault2]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_cancellation_reaps_what_was_submitted_and_reads_nothing_more[w3c3ps-fault0]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_cancellation_reaps_what_was_submitted_and_reads_nothing_more[w3c3ps-fault1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_cancellation_reaps_what_was_submitted_and_reads_nothing_more[w3c3ps-fault2]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_one_row_batches_are_bounded_by_the_banks[w1c1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_one_row_batches_are_bounded_by_the_banks[w3c1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_one_row_batches_are_bounded_by_the_banks[w3c3]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_one_row_batches_are_bounded_by_the_banks[w1c1ps]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_one_row_batches_are_bounded_by_the_banks[w3c3ps]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_row_whose_extents_deliver_less_than_its_segments_read_fails_instead_of_packing[w1c1-None]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_row_whose_extents_deliver_less_than_its_segments_read_fails_instead_of_packing[w1c1-weights1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_row_whose_extents_deliver_less_than_its_segments_read_fails_instead_of_packing[w3c1-None]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_row_whose_extents_deliver_less_than_its_segments_read_fails_instead_of_packing[w3c1-weights1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_row_whose_extents_deliver_less_than_its_segments_read_fails_instead_of_packing[w3c3-None]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_row_whose_extents_deliver_less_than_its_segments_read_fails_instead_of_packing[w3c3-weights1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_row_whose_extents_deliver_less_than_its_segments_read_fails_instead_of_packing[w1c1ps-None]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_row_whose_extents_deliver_less_than_its_segments_read_fails_instead_of_packing[w1c1ps-weights1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_row_whose_extents_deliver_less_than_its_segments_read_fails_instead_of_packing[w3c3ps-None]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_row_whose_extents_deliver_less_than_its_segments_read_fails_instead_of_packing[w3c3ps-weights1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_the_thread_serves_demands_without_a_pump[w1c1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_the_thread_serves_demands_without_a_pump[w3c1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_the_thread_serves_demands_without_a_pump[w3c3]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_slow_read_times_out_the_wait_and_raises_fatal[w1c1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_slow_read_times_out_the_wait_and_raises_fatal[w3c1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_slow_read_times_out_the_wait_and_raises_fatal[w3c3]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_no_advisory_starts_while_paused[w1c1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_no_advisory_starts_while_paused[w3c1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_no_advisory_starts_while_paused[w3c3]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_pause_waits_for_an_advisory_in_flight_and_cuts_it_short[w1c1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_pause_waits_for_an_advisory_in_flight_and_cuts_it_short[w3c1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_pause_waits_for_an_advisory_in_flight_and_cuts_it_short[w3c3]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_concurrent_eager_use_and_advisories_never_share_a_slot[w1c1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_concurrent_eager_use_and_advisories_never_share_a_slot[w3c1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_concurrent_eager_use_and_advisories_never_share_a_slot[w3c3]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_pause_that_times_out_raises_and_leaves_the_thread_running[w1c1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_pause_that_times_out_raises_and_leaves_the_thread_running[w3c1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_pause_that_times_out_raises_and_leaves_the_thread_running[w3c3]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_stop_while_paused_returns_promptly[w1c1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_stop_while_paused_returns_promptly[w3c1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_stop_while_paused_returns_promptly[w3c3]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_collecting_a_threaded_host_stops_its_thread[w1c1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_collecting_a_threaded_host_stops_its_thread[w3c1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_collecting_a_threaded_host_stops_its_thread[w3c3]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_cancelled_advisory_keeps_the_rows_that_completed_and_releases_the_rest[w1c1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_cancelled_advisory_keeps_the_rows_that_completed_and_releases_the_rest[w3c1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_cancelled_advisory_keeps_the_rows_that_completed_and_releases_the_rest[w3c3]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_rows_kept_from_a_cancelled_advisory_follow_the_usual_eviction_rules[w1c1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_rows_kept_from_a_cancelled_advisory_follow_the_usual_eviction_rules[w3c1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_rows_kept_from_a_cancelled_advisory_follow_the_usual_eviction_rules[w3c3]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_an_advisory_has_one_row_outstanding_and_a_demand_may_have_several[w1c1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_an_advisory_has_one_row_outstanding_and_a_demand_may_have_several[w3c1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_an_advisory_has_one_row_outstanding_and_a_demand_may_have_several[w3c3]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_full_cache_serves_a_demand_that_exactly_fills_it_and_fails_one_that_cannot_fit[w1c1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_full_cache_serves_a_demand_that_exactly_fills_it_and_fails_one_that_cannot_fit[w3c1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_full_cache_serves_a_demand_that_exactly_fills_it_and_fails_one_that_cannot_fit[w3c3]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_resident_expert_outside_protect_is_evicted_by_the_request_that_needs_another[w1c1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_resident_expert_outside_protect_is_evicted_by_the_request_that_needs_another[w3c1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_resident_expert_outside_protect_is_evicted_by_the_request_that_needs_another[w3c3]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_resident_expert_in_protect_is_not_the_victim[w1c1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_resident_expert_in_protect_is_not_the_victim[w3c1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_resident_expert_in_protect_is_not_the_victim[w3c3]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_when_every_resident_expert_is_protected_the_request_fails_and_evicts_nothing[w1c1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_when_every_resident_expert_is_protected_the_request_fails_and_evicts_nothing[w3c1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_when_every_resident_expert_is_protected_the_request_fails_and_evicts_nothing[w3c3]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_read_that_fails_before_it_starts_publishes_nothing_and_frees_every_slot[w1c1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_read_that_fails_before_it_starts_publishes_nothing_and_frees_every_slot[w3c1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_read_that_fails_before_it_starts_publishes_nothing_and_frees_every_slot[w3c3]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_stop_during_an_advisory_returns_promptly_and_leaves_the_tier_consistent[w1c1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_stop_during_an_advisory_returns_promptly_and_leaves_the_tier_consistent[w3c1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_stop_during_an_advisory_returns_promptly_and_leaves_the_tier_consistent[w3c3]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_multi_row_advisories_cut_short_by_pauses_leave_only_whole_rows[w1c1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_multi_row_advisories_cut_short_by_pauses_leave_only_whole_rows[w3c1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_multi_row_advisories_cut_short_by_pauses_leave_only_whole_rows[w3c3]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_the_fixture_puts_a_reused_test_on_the_workers_it_names[w1c1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_the_fixture_puts_a_reused_test_on_the_workers_it_names[w3c1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_the_fixture_puts_a_reused_test_on_the_workers_it_names[w3c3]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_the_fixture_puts_a_reused_test_on_the_workers_it_names[w1c1ps]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_the_fixture_puts_a_reused_test_on_the_workers_it_names[w3c3ps]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_the_flag_defaults_to_off
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_no_worker_thread_exists_unless_asked_for_and_close_joins_them
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_worker_may_not_use_cores_64_to_71
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_the_pool_pins_each_worker_to_its_own_allowed_core_and_refuses_when_too_few_are_left
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_workers_take_separate_physical_cores_before_hyperthread_siblings
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_the_unpinned_service_thread_keeps_off_the_packing_workers_cpus
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_workers_copy_rows_concurrently_and_a_row_is_cut_across_them
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_row_cut_into_many_more_chunks_than_it_has_lines_is_byte_exact[2-64]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_row_cut_into_many_more_chunks_than_it_has_lines_is_byte_exact[1-100000]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_row_cut_into_many_more_chunks_than_it_has_lines_is_byte_exact[4-7]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_failure_returns_only_when_no_copy_is_still_running[w1c1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_failure_returns_only_when_no_copy_is_still_running[w3c1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_failure_returns_only_when_no_copy_is_still_running[w3c3]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_failure_returns_only_when_no_copy_is_still_running[w1c1ps]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_failure_returns_only_when_no_copy_is_still_running[w3c3ps]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_row_short_of_its_segments_is_never_handed_to_a_worker[w1c1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_row_short_of_its_segments_is_never_handed_to_a_worker[w3c1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_row_short_of_its_segments_is_never_handed_to_a_worker[w3c3]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_row_short_of_its_segments_is_never_handed_to_a_worker[w1c1ps]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_row_short_of_its_segments_is_never_handed_to_a_worker[w3c3ps]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_no_extent_reuses_a_bank_before_the_copies_out_of_it_are_done[w1c1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_no_extent_reuses_a_bank_before_the_copies_out_of_it_are_done[w3c1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_no_extent_reuses_a_bank_before_the_copies_out_of_it_are_done[w3c3]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_no_extent_reuses_a_bank_before_the_copies_out_of_it_are_done[w1c1ps]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_no_extent_reuses_a_bank_before_the_copies_out_of_it_are_done[w3c3ps]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_the_ordering_assertion_fires_on_a_record_that_violates_it
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_held_rows_are_released_together_and_read_byte_exact[w1c1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_held_rows_are_released_together_and_read_byte_exact[w3c1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_held_rows_are_released_together_and_read_byte_exact[w3c3]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_held_rows_are_released_together_and_read_byte_exact[w1c1ps]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_held_rows_are_released_together_and_read_byte_exact[w3c3ps]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_pack_ns_is_the_sum_of_the_rows_spans[w1c1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_pack_ns_is_the_sum_of_the_rows_spans[w3c1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_pack_ns_is_the_sum_of_the_rows_spans[w3c3]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_pack_ns_is_the_sum_of_the_rows_spans[w1c1ps]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_pack_ns_is_the_sum_of_the_rows_spans[w3c3ps]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_credit_and_rows_in_flight_do_not_depend_on_where_the_copy_runs[w1c1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_credit_and_rows_in_flight_do_not_depend_on_where_the_copy_runs[w3c1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_credit_and_rows_in_flight_do_not_depend_on_where_the_copy_runs[w3c3]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_credit_and_rows_in_flight_do_not_depend_on_where_the_copy_runs[w1c1ps]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_credit_and_rows_in_flight_do_not_depend_on_where_the_copy_runs[w3c3ps]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_stage_record_carries_the_packing_mode_that_produced_it[0-0-0]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_stage_record_carries_the_packing_mode_that_produced_it[1-0-1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_stage_record_carries_the_packing_mode_that_produced_it[3-0-3]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_stage_record_carries_the_packing_mode_that_produced_it[3-1-1]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_stage_record_carries_the_packing_mode_that_produced_it[3-3-3]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_a_stage_record_carries_the_packing_mode_that_produced_it[2-5-5]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_every_record_a_host_pushes_carries_its_mode_including_the_ones_that_read_nothing[0]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_every_record_a_host_pushes_carries_its_mode_including_the_ones_that_read_nothing[2]
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_the_mode_reaches_the_jsonl_a_trace_writes
- registered/unit/kernels/test_exl3_ram_miss_pack_workers.py::test_the_analysis_refuses_the_metrics_of_a_trace_a_worker_host_wrote_and_keeps_those_of_an_inline_one

</details>

### Full test-ID list: `test_expert_stream_reader_split.py` (content match: direct `PackReader`/`PackPool` references; 4 tests, 0 skip)

- registered/unit/kernels/test_expert_stream_reader_split.py::test_the_core_names_no_path_specific_mechanism
- registered/unit/kernels/test_expert_stream_reader_split.py::test_each_derived_reader_holds_only_its_own_mechanism
- registered/unit/kernels/test_expert_stream_reader_split.py::test_the_bounce_path_keeps_the_scalar_read_opcode
- registered/unit/kernels/test_expert_stream_reader_split.py::test_the_tier_and_ffi_read_through_any_reader

## 5. bpftrace / perf availability on divix01 (for Tasks 12/15/18)

```bash
which bpftrace   # not found
which perf       # /usr/bin/perf
sudo -n true     # "sudo: a password is required" -- passwordless sudo NOT available for this user
```

- `perf` is installed and on `PATH`.
- `bpftrace` is **not installed**.
- `sudo -n` fails (password required); the user has no NOPASSWD sudo entry beyond the specific `nsys-profile` wrapper
  documented in the project's Nsight notes. Any perf/bpftrace use that needs elevated privileges (e.g. kernel
  symbols, `perf_event_paranoid`) will need either a NOPASSWD wrapper added for that specific command, or running as
  the admin interactively -- Tasks 12/15/18 should check `perf_event_paranoid` and plan around this rather than
  assume root.

## Worktrees

- `/data/models/slang/nvfp4-work/wt-hotpath-base`: created this task, detached at `ba01695c35`. Clean.
- `/data/models/slang/nvfp4-work/wt-hotpath`: did not exist before this task; created in Step 5 below, detached at
  `origin/cc/hotpath-zero-overhead` after the push.
