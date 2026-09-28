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

## Pre-wait work probe (2026-09-27, commit 3ee723bd8b)

The review's caveat was that the skeleton has no prologue for PDL to overlap. This probe gives every chain stage a
spin of `pre_ns` on all threads before `griddepcontrol.wait`, at `work_ns` 2000, and runs modes 0/1/2 at
`pre_ns` 0, 200 and 500 in one job. The `moe` stage has none. Every-word-reads-R check: passed in all nine records,
`SKEL_EXIT=0`.

```bash
skeleton.py --repo $PWD --out analysis/dsv41-drive/chain-pdl/skeleton-prewait.jsonl --work-ns 2000 --pre-ns 0,200,500
python3 skeleton_report.py skeleton-prewait.jsonl
```

| pre-wait ns | mode 0 us/layer | mode 1 us/layer | mode 2 us/layer | mode 1 saving | mode 2 saving | mode 2 per step | share of 66.8 ms | >= 2 us gate |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 24.168 | 23.387 | 22.461 | 0.782 | 1.708 | 68.3 us | 0.102% | no |
| 200 | 25.912 | 25.136 | 22.649 | 0.776 | **3.263** | 130.5 us | 0.195% | **yes** |
| 500 | 28.212 | 27.446 | 22.962 | 0.766 | **5.250** | 210.0 us | 0.314% | **yes** |

The pre-0 row repeats the first run within noise (mode 2 saving 1.71 vs 1.48 us/layer).

- **With the early trigger (mode 2), the prologue is almost entirely hidden.** Mode 2's per-layer time rises only
  0.19 us at 200 ns and 0.50 us at 500 ns, against 1.74 and 4.04 us for mode 0. So each edge hides about 200-500 ns
  of prologue behind its predecessor's body.
- **With the implicit trigger (mode 1), none of it is hidden.** The saving stays at 0.77-0.78 us. The dependent cannot
  launch until the primary exits, so its prologue still runs after the primary.

**Does the PDL row flip?** Yes, for the early trigger, once real stages do about 200 ns or more of work before their
dependent read: the skeleton's saving crosses the 2 us/layer gate (3.26 us at 200 ns, 5.25 us at 500 ns). The Task 8
gate therefore reads "run Task 8" for mode 2, provided the real chain's pre-wait work is at least about 200 ns per
stage. That is unmeasured: Task 8, or a trace of the real stages' prologues, would measure it.

Two things do not change:
- **Task 8 stays blocked** by the expert-stream-native-sync merge gate (NOT_MERGED at the time of writing).
- **Even the flipped figure is small.** It is 0.20-0.31% of the 66.8 ms step. This is **above** the 0.1% the request
  anticipated, but still below the 1% (0.67 ms/step) needed to put a ship decision to the user.

Decision-table row "PDL on the chain", updated: **gate met for the early trigger if real prologues are >= ~200 ns
(skeleton: 3.26-5.25 us/layer); gate not met for the implicit trigger (0.77 us/layer). Ship bar not met (<= 0.31% of
step). Task 8 remains gated on the merge.**
