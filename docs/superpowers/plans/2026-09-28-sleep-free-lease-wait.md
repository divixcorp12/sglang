# Sleep-free lease wait implementation plan

**Goal:** Replace the long NVMe polling loops in the conventional, batched lease, and second-stage lease waits with CUDA stream memory dependencies, preserving graph replay and failure handling.

**Authorization:** The user approved the in-chat architecture and requested implementation in an isolated worktree. Branch: `codex/sleep-free-lease-wait`.

**Architecture:** A short preparation kernel publishes the request generation and remaining timeout into a separate pinned completion mailbox. An independent native CPU monitor observes the existing service's completion/fatal/shutdown words and publishes an immutable outcome followed by a 32-bit ready signal. The host launcher enqueues `cuStreamWaitValue32`, or inserts an equivalent graph memory-operation node during capture, followed by a non-polling validation kernel. The CPU I/O service and lease wire ABI remain unchanged. The one producer/one consumer mailbox is reused only after the ready signal has unblocked its stream; concurrent execution chains require separate device objects/mailboxes.

The CPU timeout starts at first observation of the pending mailbox and uses the duration computed on the GPU, not a comparison between CPU and GPU clocks. It may exceed the former GPU deadline by CPU observation/scheduling latency. Final process-level recovery remains the existing watchdog. Existing timeout/lease cleanup protects late I/O; the bridge neither frees slots nor writes service results.

## Ownership and sequence

- [x] Stream enqueue helper and native CPU harness: `stream_wait.cuh`, `test_expert_stream_wait_enqueue.py`.
- [x] Completion monitor and native CPU harness: `host/wait_completion.h`, `expert_stream_wait.cpp`, `test_expert_stream_wait_completion.py`.
- [x] Preparation/validation kernel split and GPU tests: `lease_kernels.cuh`, `test_expert_stream_sleep_free_cuda.py`.
- [x] Integration: shared `wait_layout.h`, Python module linkage/mailbox lifetime/arguments.
- [x] Independent review; CPU suites; CUDA 13.4 compilation and eager/capture/replay/timeout/abort GPU regressions on divix01.

## Constraints and verification

Keep bounded hit-stage polling and piece streaming unchanged: a full-request gate there would remove I/O overlap. Preserve release/acquire operations and all terminal-before-fatal ordering. A ready token is a scheduling signal, not proof of success; validators must check generation/outcome. Prepare must satisfy the gate on no-request/fatal paths. Cancel the monitor, drain queued GPU work while it continues publishing aborts, then join it before releasing pinned tensors. Keep the device object alive through every future replay of graphs that refer to its mailbox. Test stale completion, same-page generations, sequence wrap, timeout, fatal/shutdown, no-request, and replay. No performance improvement is claimed without measurement.

Remote builds follow `.claude/rules/divix01-run-protocol.md`: commit/push/fetch into a private worktree, correct PYTHONPATH, capped CPU affinity, and cc-gpu.lock for GPU execution. No production checkout edits.

## Implementation and verification evidence

The three long waits no longer update the GPU `polls` counter. The monitor uses a CPU thread sleeping 25 microseconds between checks; the GPU waits through a stream dependency and does not keep a polling kernel resident. This is not a claim of an end-to-end latency improvement. There is one mailbox and one sequential execution chain per `ExpertStreamDevice`.

Independent reviews covered the monitor publication/reuse ordering, capture dependencies, validator failures, and Python cancellation/draining/buffer lifetime. No blocking findings remained. Cleanup tests cover both ordered shutdown and buffer retention when CUDA synchronization fails.

Local CPU verification: **31 passed** with:

```bash
PYTHONPATH=$PWD/python python3 -m pytest \
  test/registered/unit/kernels/test_expert_stream_wait_lifecycle.py \
  test/registered/unit/kernels/test_expert_stream_wait_completion.py \
  test/registered/unit/kernels/test_expert_stream_wait_enqueue.py \
  test/registered/unit/kernels/test_exl3_ram_miss_device_args.py \
  test/registered/unit/kernels/test_expert_stream_sleep_free_sources.py \
  test/registered/unit/kernels/test_expert_stream_sync_primitives.py -q -p no:randomly
```

CUDA 13.4 / RTX 5090, in the private divix01 worktree: **52 passed** with:

```bash
export PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 CUDA_HOME=/usr/local/cuda-13.4
flock -w 900 /data/models/slang/nvfp4-work/cc-gpu.lock taskset -c 32-63 \
  /data/models/slang/.venv/bin/python -m pytest \
  test/manual/dsv41/test_expert_stream_sleep_free_cuda.py \
  test/manual/dsv41/test_exl3_ram_miss_cuda.py \
  test/manual/dsv41/test_exl3_lease_kernels_cuda.py -q -x -p no:randomly
```

This compiled the new host bridge and CUDA kernels and exercised delayed I/O, eager execution, repeated graph replay, timeout, sticky bypass, fatal interruption, generation validation, lease acknowledgment, and both single-stage and second-stage copying.

Full MoE graph and two-phase failure verification: **18 passed** with:

```bash
export PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 MAX_JOBS=8 CUDA_HOME=/usr/local/cuda-13.4
export SGLANG_EXL3_SRC=/data/models/slang/nvfp4-work/exllamav3
export SGLANG_EXL3_BUILD_DIR=/data/models/slang/nvfp4-work/cc-expert-prediction/exl3-build
flock -w 900 /data/models/slang/nvfp4-work/cc-gpu.lock taskset -c 32-63 \
  /data/models/slang/.venv/bin/python -m pytest \
  test/manual/dsv41/test_exl3_two_phase_failure_cuda.py \
  test/manual/dsv41/test_exl3_ram_miss_graph_gpu.py -q -x -p no:randomly
```

The initial invocation without `SGLANG_EXL3_SRC` stopped at the dependency configuration check. The corrected invocation above rebuilt EXL3 with CUDA 13.4 and passed, including full graph replay, DIRECT insertion, eviction/refetch, timeout, generation violation, and two-phase failure paths.

Captured topology and bitwise two-phase parity: **3 passed**, using the same exports above and:

```bash
flock -w 900 /data/models/slang/nvfp4-work/cc-gpu.lock taskset -c 32-63 \
  /data/models/slang/.venv/bin/python -m pytest \
  test/manual/dsv41/test_exl3_two_phase_parity_cuda.py -q -x -p no:randomly
```

The topology assertion verifies the exact 12-node/11-edge chain, including preparation → batch memory operation → validation, and orders finalize against all fused-consumer nodes. The output test passes bitwise equality in eager and captured execution. Total targeted evidence: **31 CPU tests and 73 GPU tests passed**. Final independent verification and `git diff --check` found no blockers.
