# Copy-mechanism sweep

GB/s against bytes in flight for every way the expert stream could move pinned host bytes into VRAM, plus the
link's ceiling, a serial PCIe read's latency and a host-flag round trip, on whatever host it runs on. Plan:
`docs/superpowers/plans/2026-09-27-expert-stream-transfer-measurement.md`, Part A.

## Run on a new host (the Gen5 one-liner)

```bash
# In a checkout of the branch, with a CUDA toolkit the sglang JIT resolves and one GPU visible:
export PYTHONPATH=$PWD/python D=analysis/dsv41-drive/copy-mechanism H=$(hostname)
python $D/probe.py --repo $PWD --out $D/$H-probe.json \
  && python $D/mech_bench.py --repo $PWD --probe $D/$H-probe.json --out $D/$H.jsonl --node 0 \
  && python3 $D/mech_report.py $D/$H.jsonl
```

Use `--node -1` on a host where NUMA binding is unwanted. On divix01, prefix the Python commands with
`flock /data/models/slang/nvfp4-work/cc-gpu.lock taskset -c 32-63` (run protocol), and set `TMPDIR` to a disk with
room (nvcc writes its temporaries there; divix01's root volume has run full).

## Reading it

- `measured_ceiling_gbs` is the best cell of any method. `theoretical_gbs` is the PCIe payload ceiling.
- `bdp_bytes` = minimum flag round trip x measured ceiling: the bytes in flight a copy needs to fill the link
  (primary; a copy with fixed bytes in flight runs at about in_flight / RTT). `bdp_acquire_bytes` uses the serial
  acquire instead and understates it.
- A method's `knee_bytes` is the fewest bytes in flight within 5% of its best.
- `s_pattern` (grid 8, 1 load per thread, 16 B) is S. `cw_pattern` (grid 1, 4 loads, 16 B, contiguous) is
  contiguous 4-deep, NOT the copy wait's real shape. `cw_real` is: the production `copy_wait_read` over CW's four
  small tensors per lane (20,480, 9,216, 10,240, 4,608 B), 1-8 lanes; read its `us_per_row` (per lane) and GB/s.
- `sm_cv*` use `ld.global.cv`; `sm_weak*` use plain (weak) `ld.global` ordered after the acquire. Both are
  contract-valid. `.nc` is non-coherent and appears only as the fresh check's negative control: not contract-valid,
  informational only.
- `fresh` records: every swept method must read bytes the host rewrote mid-kernel, and the `.nc` control must not.
- The exit status is 1 on an above-ceiling cell, an unsafe method, or a blind fresh check.

## Why the control has its own barrier (the CCTL split)

The fresh check's midpoint is an acquire poll of a host flag. On sm_120, `ld.acquire.sys` compiles to
`LDG.E.STRONG.SYS` followed by `CCTL.IVALL`, which invalidates the whole SM's L1. With that barrier the `.nc` control's
second pass missed L1 and read fresh bytes, so the check could not tell a stale read from a fresh one (first
divix01 smoke, 2026-09-27). The control therefore polls with `ld.relaxed.sys` (no CCTL); the swept kernels keep the
acquire, which is what the visibility contract requires of them. `mech_bench.py` reads the loaded module's SASS and
refuses to sweep unless the control has no `CCTL` and every swept copy kernel has one.

`cudaMemcpyBatchAsync` rejects the legacy NULL stream (`invalid argument`), so the probe and the sweep run on a side
stream.
