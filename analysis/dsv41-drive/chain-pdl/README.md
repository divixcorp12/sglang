# Chain PDL skeleton (Part B, Task 7)

An upper bound on what Programmatic Dependent Launch would save per layer on the expert-stream lease chain, measured
on a skeleton rather than the real chain. Plan: `docs/superpowers/plans/2026-09-27-expert-stream-transfer-measurement.md`,
Task 7. Results: `results.md`.

## What it models

- The chain `post -> W1 -> C1 -> A1 -> S -> A2 -> CW -> F` with the production launch shapes (grid x block):
  1x32, 1x32, 8x256, 1x8, 8x256, 1x8, 1x256, 1x32.
- Forty layers captured in one CUDA graph, with a non-PDL `moe` stage (1x256) between chains, standing in for the
  attention and MoE work between them.
- Each stage waits for its predecessor under PDL (`griddepcontrol.wait`), reads the predecessor's word and writes its
  own, so every edge is a real data dependency. The script checks that after R replays every word reads R.
- Three modes: 0, no PDL; 1, PDL with the trigger implied at exit (SASS `ACQBULK`); 2, PDL with the trigger right
  after the wait (`ACQBULK` + `PREEXIT`).
- `work_ns` (0 or 2000) is a `%globaltimer` spin per stage, standing in for the stage's body.

## What it leaves out

- Real work per stage: stages spin, they don't copy, poll or wait on the host.
- The host round trip. The real chain waits on host flags (demand, CopyDone), and PDL cannot shorten those waits.
- PDL on C1 (`expert_cache_transfer.cuh`), which Task 8's test hook does not cover either.

## Run

```bash
ssh divix01 'cd /data/models/slang/nvfp4-work/wt-xfer && git fetch origin && git checkout --detach origin/expert-stream-transfer-measurement \
  && rm -f analysis/dsv41-drive/chain-pdl/skeleton.jsonl \
  && PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 TMPDIR=/mnt/nvme1/tmp-ests/xfer-tmp flock /data/models/slang/nvfp4-work/cc-gpu.lock taskset -c 32-63 \
     /data/models/slang/.venv/bin/python analysis/dsv41-drive/chain-pdl/skeleton.py --repo $PWD \
     --out analysis/dsv41-drive/chain-pdl/skeleton.jsonl 2>&1 | tail -6; echo "EXIT=${PIPESTATUS[0]}"'
```
