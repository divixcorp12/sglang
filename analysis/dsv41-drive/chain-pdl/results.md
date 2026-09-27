# Chain PDL skeleton: divix01 results (RTX 5090, sm_120f, CUDA 13.4, PCIe Gen3 x16)

Run 2026-09-27 at commit 9e0a536251 in the private worktree `wt-xfer`. `sglang.__file__` was
`/data/models/slang/nvfp4-work/wt-xfer/python/sglang/__init__.py`. 200 timed replays after 20 warm-up replays, CUDA
events, median. Command: `README.md`, "Run". The script exited 0, and the every-word-reads-R check passed in all
three modes, so PDL broke no edge.

## Records (`skeleton.jsonl`)

| mode | work_ns | replay us p50 | per layer us p50 |
|---|---:|---:|---:|
| 0 (no PDL) | 0 | 242.35 | 6.059 |
| 1 (PDL, implicit trigger) | 0 | 211.42 | 5.286 |
| 2 (PDL, early trigger) | 0 | 185.81 | 4.645 |
| 0 (no PDL) | 2000 | 953.54 | 23.838 |
| 1 (PDL, implicit trigger) | 2000 | 922.66 | 23.066 |
| 2 (PDL, early trigger) | 2000 | 894.27 | 22.357 |

## Saving against mode 0

The per-step figure is for the 40-layer step. The share is of the untraced 66.8 ms/token decode step.

| mode | work_ns | per layer (us) | per step (us) | share of 66.8 ms |
|---|---:|---:|---:|---:|
| 1 (implicit trigger) | 0 | 0.773 | 30.9 | 0.046% |
| 2 (early trigger) | 0 | 1.413 | 56.5 | 0.085% |
| 1 (implicit trigger) | 2000 | 0.772 | 30.9 | 0.046% |
| 2 (early trigger) | 2000 | 1.482 | 59.3 | 0.089% |

Mode 1 and mode 2 are faster than mode 0 at both work levels, as expected. The saving does not depend on `work_ns`:
PDL overlaps launch latency, not the stage bodies. It comes to about 0.1 us per edge with the implicit trigger and
about 0.18 us with the early one, over the 8 PDL edges of a layer.

This is a **launch-latency-only saving**, not a strict upper bound. Every skeleton stage runs `griddepcontrol.wait`
as its first instruction, so there is no prologue for PDL to overlap. Real stages do prologue work before their
dependent read (parameter loads, address setup, CW's mbarrier init), and a production PDL placement could overlap
that. The margin to the gate is 0.52 us per layer, about 65 ns per edge over 8 edges. If the real prologues average
more than about 65 ns before the wait, the real saving could clear 2 us/layer. Against that, the real stages also
wait on host flags, which PDL cannot shorten. A skeleton mode with 200-500 ns of pre-wait work would measure that
sensitivity; it was not run.

## Task 8 gate

Rule: run Task 8 if the saving is >= 2 us per layer in mode 1 or mode 2 at `work_ns` 2000.

Measured: mode 1 saves 0.772 us per layer and mode 2 saves 1.482 us per layer. Both are below 2 us, so **do not run
Task 8**. Part B stops here on divix01.

Decision-table row "PDL on the chain": **no (bound below threshold)**, subject to the prologue caveat above. The best case, mode 2, saves 59 us per step,
0.09% of 66.8 ms/token, against the 1% (0.67 ms/step) needed to put a ship decision to the user. This verdict is for
Gen3 divix01. Launch latency is a property of the GPU and driver, not the link, so a Gen5 host with the same GPU is not
expected to differ much. That expectation is untested.

Task 8 is gated twice: by this bound, and by the expert-stream-native-sync merge. It was not started.
