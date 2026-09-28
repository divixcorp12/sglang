# Sleep-free lease wait implementation plan

**Goal:** Replace the long NVMe polling loops in the conventional, batched lease, and second-stage lease waits with CUDA stream memory dependencies, preserving graph replay and failure handling.

**Authorization:** The user approved the in-chat architecture and requested implementation in an isolated worktree. Branch: `codex/sleep-free-lease-wait`.

**Architecture:** A short preparation kernel publishes the request generation and remaining timeout into a separate pinned completion mailbox. An independent native CPU monitor observes the existing service's completion/fatal/shutdown words and publishes an immutable outcome followed by a 32-bit ready signal. The host launcher enqueues `cuStreamWaitValue32`, or inserts an equivalent graph memory-operation node during capture, followed by a non-polling validation kernel. The CPU I/O service and lease wire ABI remain unchanged. The one producer/one consumer mailbox is reused only after the ready signal has unblocked its stream; concurrent execution chains require separate device objects/mailboxes.

The CPU timeout starts at first observation of the pending mailbox and uses the duration computed on the GPU, not a comparison between CPU and GPU clocks. It may exceed the former GPU deadline by CPU observation/scheduling latency. Final process-level recovery remains the existing watchdog. Existing timeout/lease cleanup protects late I/O; the bridge neither frees slots nor writes service results.

## Ownership and sequence

- [ ] Stream enqueue helper and native CPU harness: `stream_wait.cuh`, `test_expert_stream_wait_enqueue.py`.
- [ ] Completion monitor and native CPU harness: `host/wait_completion.h`, `expert_stream_wait.cpp`, `test_expert_stream_wait_completion.py`.
- [ ] Preparation/validation kernel split and GPU tests: `lease_kernels.cuh`, `test_expert_stream_sleep_free_cuda.py`.
- [ ] Integration: shared `wait_layout.h`, Python module linkage/mailbox lifetime/arguments.
- [ ] Independent review; CPU suites; CUDA 13.4 compilation and eager/capture/replay/timeout/abort GPU regressions on divix01.

## Constraints and verification

Keep bounded hit-stage polling and piece streaming unchanged: a full-request gate there would remove I/O overlap. Preserve release/acquire operations and all terminal-before-fatal ordering. A ready token is a scheduling signal, not proof of success; validators must check generation/outcome. Prepare must satisfy the gate on no-request/fatal paths. Stop the monitor before releasing pinned tensors. Test stale completion, same-page generations, sequence wrap, timeout, fatal/shutdown, no-request, and replay. No performance improvement is claimed without measurement.

Remote builds follow `.claude/rules/divix01-run-protocol.md`: commit/push/fetch into a private worktree, correct PYTHONPATH, capped CPU affinity, and cc-gpu.lock for GPU execution. No production checkout edits.
