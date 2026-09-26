# DSV41 Prefill Route Plan Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the host issue each gathered chunk's expert compute while that chunk's PCIe gather is still running, instead of blocking ~58 ms on it, so the 260-token prefill drops from ~12.9 s toward the ~6 s link floor.

**Architecture:** Behind a new default-off flag, `SGLANG_DSV41_ENABLE_PREFILL_ROUTE_PLAN`, `_apply_streamed` reads a layer's `topk_ids` to the host once, before any of its gathers is queued. It groups the routes by expert into an `Exl3RoutePlan`, and the streamer hands back each chunk's expert ids and `row_of_source` as host lists. Between a chunk's gather launch and its compute launches there is then no readback: neither `chunk.tolist()`, `row_of_source.tolist()`, nor one `torch.where` per expert. The host runs ahead of the gather, and stream order still makes chunk k+1's gather wait for chunk k's consumers.

**Tech Stack:** Python 3.13, PyTorch (CUDA, RTX 5090 SM120), the EXL3 extension (`exl3_ext`), Triton gather kernels, `msgspec`, pytest; Nsight Systems for the before/after traces.

**Spec:** `MOE_PREFILL_OPT.md` (repo root). Also read `DSV41_REFERENCE.md` sections 27.2, 27.4 item 7, 27.6, 27.9 and 27.10.

## Global Constraints

- New env var: `SGLANG_DSV41_ENABLE_PREFILL_ROUTE_PLAN = EnvBool(False)` in `python/sglang/srt/environ.py`, with the field `enable_prefill_route_plan` in `Dsv41Config`. `test_one_field_per_knob` must pass.
- Flag off must be exactly today's path: the same kernels, syncs and outputs.
- Output must be byte-identical: greedy output flag off vs on, and `torch.equal` of the MoE output against the current `exl3_moe_accumulate` path. That includes more than 64 experts (several chunks), repeated experts per token, and dropped routes (`-1`).
- The fp32 accumulation order stays ascending expert id, with each expert's routes in `torch.where` (row-major) order.
- Data containers are `msgspec.Struct`, never `@dataclass`. No defensive `getattr`/`hasattr`. Comments follow `.claude/rules/comment-style.md`.
- Every divix01 run follows `.claude/rules/divix01-run-protocol.md`:
  - code gets there by commit, push, and a private worktree;
  - `PYTHONPATH=$PWD/python`, and print `sglang.__file__`;
  - read `PIPESTATUS[0]`;
  - CPU work under `taskset -c 0-63`;
  - GPU work under `cc-gpu.lock` on cores 32-63;
  - lock order: `rowimg-disk.lock`, then `cc-gpu.lock`.
- Scratch goes on `/mnt/nvme1`, never `/` or `/tmp` on divix01. Never test in `dsv41-direct-prod` or `dsv41-direct-live`.
- **Production (port 7867) is currently running at the user's request and holds `cc-gpu.lock`.** GPU tests and arms (Tasks 6-8) cannot start until the user agrees to stop it. Ask before stopping it, and ask again before restarting it (with `/mnt/nvme1/prod-flags/start_prod_hot.sh`).
- Arms: `benchmarks/dsv41_baseline/run_arm.sh`, A (flag off) then B (flag on), once each, no ABBA, port 30021.
- The traced arm uses `NSYS_TRACE=1 NSYS_CUDA_GRAPH_TRACE=node` (graph mode is refused with the copy engine on) and `NSYS_TMPDIR=/mnt/nvme1/nsys-tmp`.
- Git:
  - never force-push;
  - stage files by name;
  - never commit `.omc/`;
  - fetch and rebase onto `origin/master` before pushing.
- Commit trailer:
  ```
  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01N985sZUTxTE9P7MP2rtJaF
  ```

## What the trace says (the reason for this plan)

Measured 2026-09-25 on `prod-flags-node-20260925-211044.sqlite` (production recipe: fills and cast fusion on). The prefill of the last timed session ran 16.741-29.623 s, 12,882 ms of wall time.

| Where the host blocks | Readbacks | Host blocked |
|---|---:|---:|
| `chunk.tolist()` right after a chunk's gather is queued (`exl3.py:547`) | 115 | **5,970 ms** (median 57.9 ms each) |
| `row_of_source.tolist()` right after it (`exl3.py:548`) | 116 | ~2 ms |
| Per-expert `torch.where` count readbacks (`exl3_ops.py:217`) | ~6,970 | 72 ms |
| Everything else (range checks, lookups, `prefill_fills`) | ~1,050 | ~60 ms |

The ~8,000 small syncs are cheap because they run while the GPU queue is empty. All of the cost is the host waiting for each chunk's ~58 ms gather before it can launch that chunk's compute. After each wait, the host spends a median 38.3 ms before launching the next chunk's gather within a layer (81 chunks, 3,990 ms). Across a layer boundary, which includes attention, it spends a median 62.9 ms (34 chunks, 2,189 ms). The host and GPU strictly take turns: ~6.0 s gather plus ~6.2 s host.

**Expected effect.** Per chunk, the time goes from `P + G + C` to about `P + max(G, C)`:

- G: the gather;
- C: the host time to launch the chunk's compute;
- P: the host's pre-gather work for the next chunk (lookup, admission, fill wait).

Dropping the per-expert `torch.where` also shrinks C, removing ~5 of the ~25 kernels per expert and a sync each. The estimate is TTFT ~12.9 s to ~8-9.5 s. The remaining gap to ~6 s is P, plus the syncs in the next layer's attention, which will wait for the last chunk's queued gather. Task 8's trace measures both and decides what comes next.

## File Structure

- Modify: `analysis/dsv41-drive/pcie-trace/compare_arms.py`: wrap the CLI in `main()` so its `windows()` can be imported.
- Create: `analysis/dsv41-drive/pcie-trace/prefill_sync_sites.py`: readbacks keyed by preceding kernels, sync calls, and the kernel mix.
- Create: `analysis/dsv41-drive/pcie-trace/chunk_host_time.py`: per chunk, host blocked on the gather and host time to the next gather launch.
- Modify: `python/sglang/srt/environ.py`: the flag.
- Modify: `python/sglang/srt/dsv41_config.py`: the field.
- Modify: `test/registered/unit/test_dsv41_config.py`: the defaults test.
- Modify: `python/sglang/srt/layers/quantization/exl3_ops.py`:
  - `Exl3RoutePlan`;
  - `_accumulate_expert`, the shared per-expert body;
  - `exl3_moe_accumulate_planned`.
- Modify: `python/sglang/srt/layers/moe/expert_stream.py`:
  - `host_row_of_source`;
  - `ExpertStreamer.iter_gather_experts_host` and `_gather_experts_host`;
  - a `hot_out` out-parameter threaded through `_gather_eager_rows`, `_dispatch_eager_rows` and `_gather_cached`.
- Modify: `python/sglang/srt/layers/quantization/exl3.py`: `Exl3MoEMethod` reads the flag in `__init__`, and `_apply_streamed` gets a flag-on branch.
- Test:
  - Create `test/registered/unit/layers/quantization/test_exl3_route_plan.py` (CPU).
  - Modify `test/registered/unit/layers/quantization/test_exl3_moe_stream_mode.py` (CPU).
  - Modify `test/registered/unit/layers/moe/test_expert_plugins_cuda.py` (CUDA).
  - Modify `test/manual/dsv41/test_exl3_stream_apply_gpu.py` (GPU, real kernels).
- Docs:
  - `DSV41_REFERENCE.md`: a new section 27.11, plus section 27.4 item 7.
  - `MOE_PREFILL_OPT.md`: mark it done or superseded.

## Review Focus

1. **A token routes the same expert in two slots.** `torch.where` yields `(t, s1), (t, s2)` adjacent, and `index_add_` receives `t` twice. The plan must give the identical index tensors. Pinned in Task 3 (`test_plan_matches_torch_where_row_major`, case `repeated`).
2. **A layer whose routes are all `-1`.** Today `source_ids` is empty, no chunk runs, and the output is zeros. The planned path must also produce zeros and must not call the streamer with an empty list. Pinned in Task 5 (`test_route_plan_all_dropped_routes_give_zeros`).
3. **An all-hot chunk.** The rows are the hot cache's own tensors and `row_of_source` is the hot slots themselves, not positions. The host list must be the slots. Pinned in Task 4 (`host_row_of_source` all-hit case) and Task 6 (`hot_experts=[2, 0, 1]`).
4. **A chunk that evicts pinned rows mid-layer** (a pinned tier smaller than the layer's experts). Row order must still follow the host list. Pinned in Task 5 (the real CPU streamer with `pinned_rows=3`, flag on).
5. **Capture/warmup forwards.** `_apply_streamed` must not record routes while capturing, and the planned path must keep that. Pinned in Task 5 (`test_streamed_apply_skips_route_recording_while_capturing`, parametrized over the flag).

---

### Task 1: Commit the attribution tools

**Files:**
- Modify: `analysis/dsv41-drive/pcie-trace/compare_arms.py` (the CLI loop at the bottom)
- Create: `analysis/dsv41-drive/pcie-trace/prefill_sync_sites.py`
- Create: `analysis/dsv41-drive/pcie-trace/chunk_host_time.py`

**Interfaces:**
- Produces: `compare_arms.windows(db) -> (prefill_start_ns, decode_start_ns)`, now importable. Both scripts take `MAIN_SQLITE` and print the tables used in Task 8.

The laptop worktree already holds uncommitted versions of the first two files, made while writing this plan. The `compare_arms.py` guard is done. `prefill_sync_sites.py` exists, and `chunk_host_time.py` is in the session scratchpad.

- [ ] **Step 1: Check the `compare_arms.py` guard is in place**

Run: `tail -5 analysis/dsv41-drive/pcie-trace/compare_arms.py`
Expected: it ends with
```python
if __name__ == "__main__":
    main()
```
and the old top-level `for arg in sys.argv[1:]:` loop is the indented body of `def main():`.

- [ ] **Step 2: Create `chunk_host_time.py`, importing from its own directory**

```python
"""Per gathered chunk of the last timed session's prefill: host time blocked on the gather, then host time to the
next chunk's gather launch, split by whether that chunk is in the same layer (no deepseek_rope_kernel between).

Usage: chunk_host_time.py MAIN_SQLITE  (run where the report lives; bound memory).
"""

import statistics
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from compare_arms import connect, windows  # noqa: E402

BLOCKED_NS = 5_000_000  # a readback longer than this waited for a gather; the rest are sub-millisecond


def main(path):
    db = connect(path)
    p0, d0 = windows(db)
    names = dict(db.execute("select id, value from StringIds"))
    runtime = {
        corr: (start, end)
        for corr, start, end in db.execute(
            "select correlationId, start, end from CUPTI_ACTIVITY_KIND_RUNTIME where start >= ? and start < ?",
            (p0 - 10**9, d0),
        )
    }
    kernels = db.execute(
        "select shortName, correlationId from CUPTI_ACTIVITY_KIND_KERNEL where start >= ? and start < ? "
        "order by start",
        (p0, d0),
    ).fetchall()
    chunks, previous = [], None
    for name_id, corr in kernels:
        name = names[name_id]
        if name == "_gather_host_rows_kernel" and previous != "_gather_host_rows_kernel":
            chunks.append(runtime[corr][0])
        previous = name
    rope = sorted(runtime[corr][0] for name_id, corr in kernels if names[name_id] == "deepseek_rope_kernel")
    blocks = sorted(
        runtime[corr]
        for (corr,) in db.execute(
            "select correlationId from CUPTI_ACTIVITY_KIND_MEMCPY where copyKind = 2 and start >= ? and start < ?",
            (p0, d0),
        )
        if runtime[corr][1] - runtime[corr][0] > BLOCKED_NS
    )
    blocked, within, across = [], [], []
    for a, b in zip(chunks, chunks[1:]):
        waits = [w for w in blocks if a <= w[0] < b]
        if not waits:
            continue
        blocked.append(waits[0][1] - waits[0][0])
        after = b - waits[0][1]
        (across if any(waits[0][1] <= r < b for r in rope) else within).append(after)
    print(f"prefill {(d0 - p0) / 1e6:.0f} ms; chunks {len(chunks)}")
    if blocked:
        print(f"blocked-on-gather waits {len(blocked)}: total {sum(blocked) / 1e6:.0f} ms, "
              f"median {statistics.median(blocked) / 1e6:.1f} ms")
    else:
        print("blocked-on-gather waits: none")
    for label, xs in (("next chunk same layer", within), ("next chunk next layer", across)):
        if xs:
            print(f"{label}: n={len(xs)}, host after wait {sum(xs) / 1e6:.0f} ms, "
                  f"median {statistics.median(xs) / 1e6:.1f} ms, p90 {sorted(xs)[int(len(xs) * 0.9)] / 1e6:.1f} ms")


if __name__ == "__main__":
    main(sys.argv[1])
```

- [ ] **Step 3: Commit and push, then reproduce the numbers on divix01**

```bash
git add analysis/dsv41-drive/pcie-trace/compare_arms.py analysis/dsv41-drive/pcie-trace/prefill_sync_sites.py analysis/dsv41-drive/pcie-trace/chunk_host_time.py
git commit -m "analysis(dsv41): prefill sync attribution and per-chunk host time from a node-mode trace"
git push origin HEAD:master
```
Then, on divix01, in a private worktree `wt-route-plan` at that commit:
```bash
ulimit -v 8000000
taskset -c 0-31 /data/models/slang/.venv/bin/python analysis/dsv41-drive/pcie-trace/prefill_sync_sites.py /mnt/nvme1/dsv41-nsys/prod-flags-node-20260925-211044.sqlite > /mnt/nvme1/prefill-opt/sites-before.txt; echo EXIT=$?
taskset -c 0-31 /data/models/slang/.venv/bin/python analysis/dsv41-drive/pcie-trace/chunk_host_time.py /mnt/nvme1/dsv41-nsys/prod-flags-node-20260925-211044.sqlite > /mnt/nvme1/prefill-opt/chunks-before.txt; echo EXIT=$?
```
Expected:
- the first row of `sites-before.txt` is `232  5985  <=256B  _scatter_hot_rows_kernel | unrolled_elementwise_kernel`;
- `chunks-before.txt` shows 115 waits, total ~5,970 ms.

---

### Task 2: The flag

**Files:**
- Modify: `python/sglang/srt/environ.py`, after `SGLANG_DSV41_ENABLE_PREFILL_SHARE` (~line 1923)
- Modify: `python/sglang/srt/dsv41_config.py` (the struct and `from_envs`)
- Modify: `test/registered/unit/test_dsv41_config.py` (`test_defaults_match_the_env_declarations`)

**Interfaces:**
- Produces: `envs.SGLANG_DSV41_ENABLE_PREFILL_ROUTE_PLAN` (`EnvBool`, default `False`) and `Dsv41Config.enable_prefill_route_plan: bool`.

Read `.claude/skills/env-var-conventions/SKILL.md` first.

- [ ] **Step 1: Add the field to the defaults test so it fails**

In `test_defaults_match_the_env_declarations`, next to `enable_prefill_fills=False,`, add:
```python
        enable_prefill_route_plan=False,
```

- [ ] **Step 2: Run it and see it fail**

Run: `PYTHONPATH=$PWD/python python -m pytest test/registered/unit/test_dsv41_config.py -q -p no:randomly`
Expected: FAIL (`Dsv41Config` has no `enable_prefill_route_plan`).

- [ ] **Step 3: Declare the env var and the field**

`environ.py`, after `SGLANG_DSV41_ENABLE_PREFILL_SHARE = EnvBool(False)`:
```python
    # Prefill route plan (plan 2026-09-25-dsv41-prefill-route-plan): the eager streamed MoE reads a layer's topk_ids
    # to the host once, before any of its gathers, groups the routes by expert, and takes each chunk's expert ids and
    # row_of_source as host lists, so no readback sits between a chunk's gather and its compute and the host runs
    # ahead of the gather. Outputs are bitwise those of the per-expert torch.where loop. Off by default.
    SGLANG_DSV41_ENABLE_PREFILL_ROUTE_PLAN = EnvBool(False)
```
`dsv41_config.py`: add `enable_prefill_route_plan: bool` after `enable_prefill_share: bool`, and in `from_envs`, after `enable_prefill_share=...`:
```python
            enable_prefill_route_plan=envs.SGLANG_DSV41_ENABLE_PREFILL_ROUTE_PLAN.get(),
```

- [ ] **Step 4: Run it and see it pass**

Run: `PYTHONPATH=$PWD/python python -m pytest test/registered/unit/test_dsv41_config.py -q -p no:randomly`
Expected: PASS, including `test_one_field_per_knob`.

- [ ] **Step 5: Commit**

```bash
git add python/sglang/srt/environ.py python/sglang/srt/dsv41_config.py test/registered/unit/test_dsv41_config.py
git commit -m "feat(dsv41): SGLANG_DSV41_ENABLE_PREFILL_ROUTE_PLAN, default off"
```

---

### Task 3: `Exl3RoutePlan` and the planned accumulate

**Files:**
- Modify: `python/sglang/srt/layers/quantization/exl3_ops.py`:
  - add `import itertools` and `import msgspec`;
  - split `exl3_moe_accumulate` (lines 196-221);
  - add the new class and function after it.
- Create: `test/registered/unit/layers/quantization/test_exl3_route_plan.py`

**Interfaces:**
- Produces:
  - `Exl3RoutePlan.from_topk(topk_ids: torch.Tensor) -> Exl3RoutePlan`, with fields:
    - `experts: list[int]`, ascending and distinct, no `-1`;
    - `offsets: list[int]`, `len(experts) + 1` entries;
    - `index_of: dict[int, int]`;
    - `tokens: torch.Tensor` and `slots: torch.Tensor`, int64 on `topk_ids.device`;
    - `source_ids: torch.Tensor`, `topk_ids.dtype` on `topk_ids.device`.
  - `Exl3RoutePlan.routes_of(expert: int) -> tuple[torch.Tensor, torch.Tensor]`.
  - `exl3_moe_accumulate_planned(out, x, topk_weights, plan, w13, w2, swiglu_limit, experts, linear=exl3_linear) -> None`.

- [ ] **Step 1: Write the failing tests**

```python
"""Exl3RoutePlan: a layer's routes grouped by expert on the host, identical to the per-expert torch.where loop."""

import functools

import pytest
import torch

from sglang.srt.layers.quantization import exl3_ops
from sglang.srt.layers.quantization.exl3_ops import Exl3RoutePlan, Exl3Tensors
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

HIDDEN, INTER = 32, 16


def _fake_linear(x, t, out_dtype=None):
    """Depends on every input row and on its expert's tensors, so a wrong token or expert changes the result."""
    y = x.float() @ t.trellis.float()
    return y.to(out_dtype or x.dtype)


def _weights(num_experts, seed=0):
    g = torch.Generator().manual_seed(seed)

    def t(i, o):
        return Exl3Tensors(trellis=torch.randn(i, o, generator=g), suh=None, svh=None, mul1=True)

    return [(t(HIDDEN, INTER), t(HIDDEN, INTER)) for _ in range(num_experts)], [t(INTER, HIDDEN) for _ in range(num_experts)]


CASES = {
    "distinct": torch.tensor([[5, 0, 3], [3, 1, 5], [0, 4, 1], [5, 3, 4]], dtype=torch.int32),
    "repeated": torch.tensor([[2, 2, 0], [1, 2, 2], [0, 0, 0]], dtype=torch.int32),
    "dropped": torch.tensor([[-1, 3, -1], [3, -1, 1], [-1, -1, -1]], dtype=torch.int32),
    "all_dropped": torch.full((2, 3), -1, dtype=torch.int32),
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_plan_matches_torch_where_row_major(case):
    topk_ids = CASES[case]
    plan = Exl3RoutePlan.from_topk(topk_ids)
    flat = topk_ids.reshape(-1)
    assert plan.experts == sorted(set(flat[flat >= 0].tolist()))
    assert plan.source_ids.dtype == topk_ids.dtype and plan.source_ids.tolist() == plan.experts
    for expert in plan.experts:
        want_token, want_slot = torch.where(topk_ids == expert)
        token, slot = plan.routes_of(expert)
        assert token.dtype == want_token.dtype == torch.int64
        assert torch.equal(token, want_token) and torch.equal(slot, want_slot)


def test_planned_accumulate_is_bitwise_the_where_loop_over_many_experts():
    """80 experts, more than one 64-expert gather chunk, each chunk accumulated in turn."""
    num_experts, tokens, topk = 80, 50, 6
    g = torch.Generator().manual_seed(1)
    topk_ids = torch.stack([torch.randperm(num_experts, generator=g)[:topk] for _ in range(tokens)]).to(torch.int32)
    topk_ids[3, 2] = -1
    topk_ids[7, 1] = topk_ids[7, 0]  # one token routes an expert twice
    x = torch.randn(tokens, HIDDEN, generator=g).to(torch.bfloat16)
    topk_weights = torch.rand(tokens, topk, generator=g)
    w13, w2 = _weights(num_experts)
    experts = sorted(set(topk_ids[topk_ids >= 0].tolist()))
    want = torch.zeros(tokens, HIDDEN)
    got = torch.zeros(tokens, HIDDEN)
    plan = Exl3RoutePlan.from_topk(topk_ids)
    for start in range(0, len(experts), 64):
        chunk = experts[start : start + 64]
        exl3_ops.exl3_moe_accumulate(want, x, topk_weights, topk_ids, w13, w2, 10.0, chunk, _fake_linear)
        exl3_ops.exl3_moe_accumulate_planned(got, x, topk_weights, plan, w13, w2, 10.0, chunk, _fake_linear)
    assert torch.equal(got, want)
```

- [ ] **Step 2: Run them and see them fail**

Run: `PYTHONPATH=$PWD/python python -m pytest test/registered/unit/layers/quantization/test_exl3_route_plan.py -q -p no:randomly`
Expected: FAIL with `ImportError: cannot import name 'Exl3RoutePlan'`.

- [ ] **Step 3: Implement**

Replace `exl3_moe_accumulate`'s loop body with a call to a shared helper; the math and kernels stay exactly as they are. Then add the plan and the planned loop:
```python
def _accumulate_expert(out, x, topk_weights, token, slot, w13e, w2e, swiglu_limit, linear) -> None:
    xe = x[token]
    gate = linear(xe, w13e[0], torch.float32)
    up = linear(xe, w13e[1], torch.float32)
    if swiglu_limit is not None and swiglu_limit > 0:
        up = up.clamp(-swiglu_limit, swiglu_limit)
        gate = gate.clamp(max=swiglu_limit)
    h = F.silu(gate) * up * topk_weights[token, slot].float().unsqueeze(-1)
    out.index_add_(0, token, linear(h.to(x.dtype), w2e, torch.float32))


def exl3_moe_accumulate(out, x, topk_weights, topk_ids, w13, w2, swiglu_limit, experts, linear=exl3_linear) -> None:
    """(docstring unchanged)"""
    for expert in experts:
        token, slot = torch.where(topk_ids == expert)
        if token.numel() == 0:
            continue
        _accumulate_expert(out, x, topk_weights, token, slot, w13[expert], w2[expert], swiglu_limit, linear)


class Exl3RoutePlan(msgspec.Struct, frozen=True):
    """A layer's routes grouped by expert on the host, from one readback, so the expert loop never syncs.

    Expert ``experts[i]``'s routes are ``tokens``/``slots`` over ``offsets[i]:offsets[i + 1]``, row-major within
    the expert: the order ``torch.where(topk_ids == expert)`` returns, so every gather and ``index_add_`` sees the
    same indices as ``exl3_moe_accumulate``.
    """

    experts: list[int]
    offsets: list[int]
    index_of: dict[int, int]
    tokens: torch.Tensor
    slots: torch.Tensor
    source_ids: torch.Tensor

    @classmethod
    def from_topk(cls, topk_ids: torch.Tensor) -> "Exl3RoutePlan":
        width = topk_ids.shape[-1]
        # The layer's one readback; nothing of this layer is queued yet, so the pageable copies back are free too.
        flat = topk_ids.reshape(-1).cpu()
        positions = (flat >= 0).nonzero().flatten()
        positions = positions[torch.argsort(flat[positions], stable=True)]
        experts, counts = torch.unique_consecutive(flat[positions], return_counts=True)
        index = torch.stack((positions // width, positions % width)).to(topk_ids.device)
        expert_list = experts.tolist()
        return cls(
            experts=expert_list,
            offsets=[0, *itertools.accumulate(counts.tolist())],
            index_of={expert: i for i, expert in enumerate(expert_list)},
            tokens=index[0],
            slots=index[1],
            source_ids=experts.to(device=topk_ids.device, dtype=topk_ids.dtype),
        )

    def routes_of(self, expert: int) -> tuple[torch.Tensor, torch.Tensor]:
        i = self.index_of[expert]
        a, b = self.offsets[i], self.offsets[i + 1]
        return self.tokens[a:b], self.slots[a:b]


def exl3_moe_accumulate_planned(
    out, x, topk_weights, plan: Exl3RoutePlan, w13, w2, swiglu_limit, experts: Iterable[int], linear=exl3_linear
) -> None:
    """``exl3_moe_accumulate`` over ``plan``'s routes: the same math and order with no per-expert readback."""
    for expert in experts:
        token, slot = plan.routes_of(expert)
        _accumulate_expert(out, x, topk_weights, token, slot, w13[expert], w2[expert], swiglu_limit, linear)
```
Keep the existing type hints on `exl3_moe_accumulate`'s signature exactly; the sketch above abbreviates them. Check that `msgspec.Struct` accepts `torch.Tensor` annotations at construction (it validates types only on decode); if it objects, annotate those three fields as `object` and say why in one line.

- [ ] **Step 4: Run the new tests and the existing EXL3 CPU tests**

Run: `PYTHONPATH=$PWD/python python -m pytest test/registered/unit/layers/quantization/test_exl3_route_plan.py test/registered/unit/layers/quantization/test_exl3_ops_cpu.py test/registered/unit/layers/quantization/test_exl3_moe_stream_mode.py test/registered/unit/layers/quantization/test_exl3_moe_method.py -q -p no:randomly; echo EXIT=${PIPESTATUS[0]}`
Expected: all pass, `EXIT=0`.

- [ ] **Step 5: Commit**

```bash
git add python/sglang/srt/layers/quantization/exl3_ops.py test/registered/unit/layers/quantization/test_exl3_route_plan.py
git commit -m "feat(dsv41): Exl3RoutePlan, a layer's routes grouped by expert from one readback"
```

---

### Task 4: Host-side chunk ids and `row_of_source` from the streamer

**Files:**
- Modify: `python/sglang/srt/layers/moe/expert_stream.py`:
  - `_gather_cached` (~1445), `_gather_eager_rows` (~1894) and `_dispatch_eager_rows` (~1907): a `hot_out` out-parameter;
  - add `_gather_experts_host` and `iter_gather_experts_host` after `iter_gather_experts` (~1832);
  - add the module-level `host_row_of_source` near `_sum_gather_stats`.
- Test:
  - `test/registered/unit/layers/quantization/test_exl3_route_plan.py` (CPU, for `host_row_of_source` and validation);
  - `test/registered/unit/layers/moe/test_expert_plugins_cuda.py` (CUDA, host lists against the device gather).

**Interfaces:**
- Consumes: nothing from Task 3.
- Produces:
  - `host_row_of_source(slots: list[int], hits: list[bool]) -> list[int]`.
  - `ExpertStreamer.iter_gather_experts_host(source_ids: torch.Tensor, experts: list[int], chunk_rows: int | None = None) -> Iterator[tuple[list[int], list[int], dict[str, torch.Tensor]]]`. It yields `(chunk_experts, row_of_source, rows)`, and `rows[name][row_of_source[i]]` holds `chunk_experts[i]`. The staging-reuse contract is `iter_gather_experts`'s. `last_gather_stats` is summed the same way.

`_gather_cached` today syncs once per chunk, before the gather is queued, on `int(hit_mask.sum().item())`. With `hot_out` given, that same sync instead reads `slots` and `hit_mask` together (`torch.stack(...).tolist()`, one readback, same point in the stream). The host then knows every row's place and needs no readback after the gather. With `hot_out=None`, the code is byte-for-byte today's.

- [ ] **Step 1: Write the failing CPU tests** (append to `test_exl3_route_plan.py`)

```python
from sglang.srt.layers.moe.expert_stream import host_row_of_source


def _device_rows(slots, hits):
    """_gather_cached's own row_of_source computation (expert_stream.py, the hit_rows branch), on CPU tensors."""
    slots_t, hit_mask = torch.tensor(slots), torch.tensor(hits)
    if bool(hit_mask.all()):
        return slots_t.tolist()
    order = torch.cat(((~hit_mask).nonzero().flatten(), hit_mask.nonzero().flatten()))
    row_of_source = torch.empty_like(order)
    row_of_source[order] = torch.arange(len(slots))
    return row_of_source.tolist()


@pytest.mark.parametrize(
    "slots",
    [[2, 0, 1], [-1, -1, -1], [-1, 4, -1, 0, -1], [7, -1]],
    ids=["all_hit", "all_miss", "mixed", "hit_first"],
)
def test_host_row_of_source_matches_the_device_order(slots):
    hits = [slot >= 0 for slot in slots]
    assert host_row_of_source(slots, hits) == _device_rows(slots, hits)
```

- [ ] **Step 2: Run it and see it fail**

Run: `PYTHONPATH=$PWD/python python -m pytest test/registered/unit/layers/quantization/test_exl3_route_plan.py -q -p no:randomly -k host_row`
Expected: FAIL with `ImportError: cannot import name 'host_row_of_source'`.

- [ ] **Step 3: Implement `host_row_of_source`**

```python
def host_row_of_source(slots: list[int], hits: list[bool]) -> list[int]:
    """The staging row ``_gather_cached`` gives each source, from its hot-cache lookup read to the host: the hot slot
    itself when every source hits, else misses first and then hits, each in source order."""
    if all(hits):
        return list(slots)
    order = [i for i, hit in enumerate(hits) if not hit] + [i for i, hit in enumerate(hits) if hit]
    row_of_source = [0] * len(slots)
    for row, source in enumerate(order):
        row_of_source[source] = row
    return row_of_source
```
Run the Step 2 command again. Expected: PASS.

- [ ] **Step 4: Thread `hot_out` through the eager gather**

- `_gather_cached(self, source_ids, compact_ids, topk_ids, hot_out: Optional[list] = None)`. Replace the hit-count block with:
  ```python
  if hot_out is not None:
      # The chunk's one sync, as below, also hands the host every row's place: no readback after its gather.
      slots_host, hits_host = torch.stack((slots.long(), hit_mask.long())).tolist()
      hot_out.extend((slots_host, [bool(hit) for hit in hits_host]))
      hit_rows = routed_hit_rows = sum(hot_out[1])
  elif routed_rows == row_count:
      hit_rows = routed_hit_rows = int(hit_mask.sum().item())
  else:
      ...unchanged...
  ```
  With `hot_out`, the method is only reached from `_gather_experts_host`, where `compact_ids` is an arange, so `routed_rows == row_count`. Assert that at the top of the `hot_out` branch.
- `_gather_eager_rows(self, source_ids, compact_ids, topk_ids, hot_out: Optional[list] = None)` passes `hot_out=hot_out` to `_dispatch_eager_rows`.
- `_dispatch_eager_rows(self, source_ids, compact_ids, topk_ids, hot_out: Optional[list] = None)` passes it to `_gather_cached` only. The pinned and uncached paths ignore it: they return `compact_ids`, an arange, so source i is at row i.

- [ ] **Step 5: Add `_gather_experts_host` and `iter_gather_experts_host`**

```python
    def _gather_experts_host(self, source_ids: torch.Tensor) -> tuple[list[int], dict[str, torch.Tensor]]:
        """``_gather_experts`` returning ``row_of_source`` as a host list, read with the chunk's pre-gather sync."""
        count = source_ids.numel()
        cap = self.format.max_gather_rows
        if cap is not None and count > cap:
            raise ValueError(f"gather of {count} experts exceeds the format's max_gather_rows={cap}")
        if self.prefetch_coordinator is not None or self.next_layer_prefetch is not None:
            raise ValueError("gather_experts does not drive expert prefetch")
        if self.before_eager_gather is not None:
            self.before_eager_gather()
        compact_ids = _cached_arange(count, source_ids.device, source_ids.dtype)
        hot: list = []
        _, rows = self._gather_eager_rows(source_ids, compact_ids, source_ids.reshape(1, -1), hot_out=hot)
        return (host_row_of_source(*hot) if hot else list(range(count))), rows

    def iter_gather_experts_host(
        self, source_ids: torch.Tensor, experts: list[int], chunk_rows: int | None = None
    ) -> Iterator[tuple[list[int], list[int], dict[str, torch.Tensor]]]:
        """``iter_gather_experts`` for a caller holding ``source_ids`` on the host as ``experts``: yields each chunk's
        expert ids and ``row_of_source`` as lists, so consuming a chunk needs no readback. Validated on the host."""
        count = len(experts)
        if source_ids.ndim != 1 or source_ids.numel() != count:
            raise ValueError("iter_gather_experts_host needs source_ids and experts of one length")
        if count == 0:
            return
        if min(experts) < 0 or max(experts) >= self.num_experts:
            raise ValueError(f"selected expert ID is outside [0, {self.num_experts - 1}]")
        if len(set(experts)) != count:
            raise ValueError("iter_gather_experts_host needs distinct expert IDs")
        cap = self.format.max_gather_rows
        chunk_rows = index(chunk_rows if chunk_rows is not None else (cap if cap is not None else count))
        if chunk_rows < 1 or (cap is not None and chunk_rows > cap):
            raise ValueError(f"gather chunks of {chunk_rows} rows need 1 <= rows <= max_gather_rows={cap}")
        chunk_stats = []
        try:
            for start in range(0, count, chunk_rows):
                row_of_source, rows = self._gather_experts_host(source_ids[start : start + chunk_rows])
                chunk_stats.append(self.last_gather_stats)
                yield experts[start : start + chunk_rows], row_of_source, rows
        finally:
            if chunk_stats:
                self.last_gather_stats = _sum_gather_stats(chunk_stats)
```
Before writing the refusal line, check whether `prefetch_coordinator` and `next_layer_prefetch` are always set on `ExpertStreamer`. `_gather_experts` reads them with `getattr(..., None)`. If they are not always set, set both to `None` in `ExpertStreamer.__init__` (per `no-getattr-defensive.md`) in this same commit, or copy `_gather_experts`' existing form. Do not add a new `getattr`.

- [ ] **Step 6: Write the CUDA test** (in `TestGatherExpertsCuda`)

```python
    def test_host_rows_match_the_device_gather_through_hot_and_cold_rows(self):
        from sglang.srt.layers.moe.expert_hot_cache import ExpertHotCache

        for hot in ([1, 4], [6, 1, 3, 4, 0], []):  # mixed, all hit, all miss
            layer = _nvfp4_layer(pinned=False)
            streamer = ExpertStreamer(layer, NVFP4_STREAM_TENSORS)
            ExpertHotCache(streamer, max(len(hot), 1)).reassign(hot)
            ids = [6, 1, 3, 4, 0]
            source_ids = torch.tensor(ids, device="cuda")
            got = [
                (chunk, row_of_source, {n: rows[n][torch.tensor(row_of_source).long()].view(torch.uint8).cpu()
                                        for n in NVFP4_STREAM_TENSORS})
                for chunk, row_of_source, rows in streamer.iter_gather_experts_host(source_ids, ids, chunk_rows=2)
            ]
            want = [
                (chunk.tolist(), row_of_source.tolist(), {n: rows[n][row_of_source.long()].view(torch.uint8).cpu()
                                                          for n in NVFP4_STREAM_TENSORS})
                for chunk, row_of_source, rows in streamer.iter_gather_experts(source_ids, chunk_rows=2)
            ]
            for (gc, gr, grows), (wc, wr, wrows) in zip(got, want, strict=True):
                self.assertEqual((gc, gr), (wc, wr), hot)
                for n in NVFP4_STREAM_TENSORS:
                    self.assertTrue(torch.equal(grows[n], wrows[n]), (hot, n))
```
Check whether `ExpertHotCache(streamer, n).reassign([])` is legal. If it is not, build the all-miss case with a capacity-1 cache holding expert 7, which `ids` never route.

- [ ] **Step 7: Run the CPU tests locally and commit**

Run: `PYTHONPATH=$PWD/python python -m pytest test/registered/unit/layers/quantization/test_exl3_route_plan.py test/registered/unit/layers/moe/test_expert_gather_experts.py -q -p no:randomly; echo EXIT=${PIPESTATUS[0]}`
Expected: PASS, `EXIT=0`. The CUDA test runs in Task 6.
```bash
git add python/sglang/srt/layers/moe/expert_stream.py test/registered/unit/layers/quantization/test_exl3_route_plan.py test/registered/unit/layers/moe/test_expert_plugins_cuda.py
git commit -m "feat(dsv41): iter_gather_experts_host, chunk ids and row_of_source read with the pre-gather sync"
```

---

### Task 5: `_apply_streamed` takes the plan when the flag is on

**Files:**
- Modify: `python/sglang/srt/layers/quantization/exl3.py`:
  - `Exl3MoEMethod.__init__` (~line 311): read the flag once;
  - `_apply_streamed` (~528): add a flag-on branch. The method is a `@staticmethod` today; pass the flag in from its caller.
- Test: `test/registered/unit/layers/quantization/test_exl3_moe_stream_mode.py`

**Interfaces:**
- Consumes:
  - `Exl3RoutePlan.from_topk` and `exl3_moe_accumulate_planned` (Task 3);
  - `ExpertStreamer.iter_gather_experts_host(source_ids, experts)` (Task 4);
  - `envs.SGLANG_DSV41_ENABLE_PREFILL_ROUTE_PLAN` (Task 2).
- Produces: `Exl3MoEMethod.route_plan: bool`.

- [ ] **Step 1: Make the CPU parity tests cover both paths**

In `test_exl3_moe_stream_mode.py`:
- Add `iter_gather_experts_host` to `FakeStreamer`. Assert that its `experts` equal `source_ids.tolist()`, then yield the same chunks as `iter_gather_experts` with `chunk` and `row_of_source` as lists.
- Parametrize these over `route_plan in (False, True)`, building the method inside `envs.SGLANG_DSV41_ENABLE_PREFILL_ROUTE_PLAN.override(route_plan)`:
  - `test_streamed_apply_matches_the_resident_loop`;
  - `test_a_real_streamer_spanning_chunks_matches_the_resident_loop`;
  - `test_streamed_apply_skips_route_recording_while_capturing`.
- In the real-streamer test, record the chunks through whichever iterator the flag selects. For `iter_gather_experts_host`, wrap it as the existing `recording` does, appending `chunk` directly.
- Patch `exl3_mod.exl3_moe_accumulate_planned` with `functools.partial(..., linear=_fake_linear)` next to the existing `exl3_moe_accumulate` patch.

Add:
```python
def test_route_plan_all_dropped_routes_give_zeros(monkeypatch):
    """Every route -1: no chunk is gathered and the output is zeros, as with the flag off."""
    with envs.SGLANG_DSV41_EXPERT_STREAM.override(True), envs.SGLANG_DSV41_ENABLE_PREFILL_ROUTE_PLAN.override(True):
        method = Exl3MoEMethod(Exl3Config.from_config(CFG), streamed=True)
        layer = _layer(method)
    streamer = FakeStreamer({}, chunk_rows=2)
    layer._nvfp4_expert_streamer = streamer
    layer.should_fuse_routed_scaling_factor_in_topk = False
    method.moe_runner_config = types.SimpleNamespace(
        apply_router_weight_on_input=False, swiglu_limit=10.0, routed_scaling_factor=None
    )
    trace = Exl3StreamTrace()
    monkeypatch.setattr(exl3_mod, "get_exl3_stream_trace", lambda: trace)
    topk_ids = torch.full((3, 3), -1, dtype=torch.int32)
    x, _, dispatch = _routed_inputs(topk_ids)
    got = method.apply(layer, dispatch).hidden_states
    assert torch.equal(got, torch.zeros_like(x)) and streamer.chunks == []
```

- [ ] **Step 2: Run them and see the flag-on cases fail**

Run: `PYTHONPATH=$PWD/python python -m pytest test/registered/unit/layers/quantization/test_exl3_moe_stream_mode.py -q -p no:randomly; echo EXIT=${PIPESTATUS[0]}`
Expected: every `route_plan=False` case passes; the `route_plan=True` cases fail (the flag is not read yet, so `iter_gather_experts_host` is never called and the chunk-recording assertions fail).

- [ ] **Step 3: Implement**

In `Exl3MoEMethod.__init__`, next to `self.cast_fusion = ...`:
```python
        self.route_plan = envs.SGLANG_DSV41_ENABLE_PREFILL_ROUTE_PLAN.get()
```
`_apply_streamed` gains a keyword `route_plan: bool`, and its caller passes `route_plan=self.route_plan`. Inside, after `out = torch.zeros(...)` and before the `with streamer.prefill_fills(...)` block:
```python
        if route_plan:
            plan = Exl3RoutePlan.from_topk(topk_ids)
            gathered = bool(plan.experts)
            with streamer.prefill_fills(plan.source_ids):
                for experts, row_of_source, rows in streamer.iter_gather_experts_host(plan.source_ids, plan.experts):
                    w13, w2 = EXL3_ROW_VIEWS.select(rows, experts, row_of_source)
                    exl3_moe_accumulate_planned(out, x, topk_weights, plan, w13, w2, swiglu_limit, experts)
        else:
            (the existing source_ids / prefill_fills / iter_gather_experts block, unchanged)
```
Keep `flat`, `routed`, and the `record_routes(routed)` call before the branch exactly as they are, so residency sees identical routes. Build `plan` before `prefill_fills`, whose own readback then finds the stream already drained. `prefill_fills` of an empty `plan.source_ids` already yields without work (`source_ids.numel() == 0`).

- [ ] **Step 4: Run the stream-mode and route-plan tests**

Run: `PYTHONPATH=$PWD/python python -m pytest test/registered/unit/layers/quantization/ -q -p no:randomly; echo EXIT=${PIPESTATUS[0]}`
Expected: all pass, `EXIT=0`.

- [ ] **Step 5: Commit and push**

```bash
git add python/sglang/srt/layers/quantization/exl3.py test/registered/unit/layers/quantization/test_exl3_moe_stream_mode.py
git commit -m "feat(dsv41): the eager streamed MoE takes the route plan under SGLANG_DSV41_ENABLE_PREFILL_ROUTE_PLAN"
git fetch origin && git rebase origin/master && git push origin HEAD:master
```
If the rebase touches any file above, re-run Step 4 before pushing.

---

### Task 6: GPU parity, no-sync proof, and mutants (divix01, needs production stopped)

**Files:**
- Modify: `test/manual/dsv41/test_exl3_stream_apply_gpu.py`
- Test (run): `test/registered/unit/layers/moe/test_expert_plugins_cuda.py`

**Interfaces:**
- Consumes: everything from Tasks 3-5.

**Before starting:** production holds `cc-gpu.lock`. Ask the user to stop it (or wait until they do). Never stop it on your own.

- [ ] **Step 1: Parametrize the real-kernel apply test over the flag and add a >64-expert case**

In `test_streamed_apply_equals_resident`:
- add `@pytest.mark.parametrize("route_plan", [False, True])`;
- build the method under `envs.SGLANG_DSV41_ENABLE_PREFILL_ROUTE_PLAN.override(route_plan)`;
- make `recording` wrap `iter_gather_experts_host` when `route_plan` is on.

Add a separate test with many experts:
```python
@pytest.mark.parametrize("route_plan", [False, True])
def test_many_experts_route_plan_equals_resident(tmp_path, route_plan):
    """80 experts, 40 tokens x 6 routes with a repeated expert and dropped routes: two 64-expert chunks."""
```
Its body follows `test_streamed_apply_equals_resident`: `write_fake_exl3(..., num_experts=80, ...)`, a hot cache of 8 experts, and a pinned tier of 100 rows. Set `topk_ids[5, 3] = -1` and `topk_ids[9, 2] = topk_ids[9, 1]`, and assert `torch.equal(got, want)` against `exl3_moe_loop`. Check first that `write_fake_exl3` accepts 80 experts at the fake `HIDDEN`/`INTER` without taking minutes; if it is slow, use 70.

- [ ] **Step 2: Add the no-sync proof**

```python
def test_planned_chunk_body_never_syncs():
    """With the plan built, the per-chunk expert compute issues no host sync: the host can run ahead of the gather."""
    import torch.cuda
    from sglang.srt.layers.quantization.exl3_ops import (
        Exl3RoutePlan, exl3_moe_accumulate_planned, random_exl3_tensors,
    )

    experts, hidden, inter, tokens = 8, 5120, 2304, 40
    w13 = [(random_exl3_tensors(hidden, inter, 3, device="cuda", seed=3 * e),
            random_exl3_tensors(hidden, inter, 3, device="cuda", seed=3 * e + 1)) for e in range(experts)]
    w2 = [random_exl3_tensors(inter, hidden, 3, device="cuda", seed=3 * e + 2) for e in range(experts)]
    x = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16) * 0.05
    topk_ids = torch.stack([torch.randperm(experts, device="cuda")[:6] for _ in range(tokens)]).to(torch.int32)
    topk_weights = torch.rand(tokens, 6, device="cuda")
    plan = Exl3RoutePlan.from_topk(topk_ids)
    out = torch.zeros(tokens, hidden, device="cuda")
    exl3_moe_accumulate_planned(out, x, topk_weights, plan, w13, w2, 10.0, plan.experts)  # warm the kernels
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        exl3_moe_accumulate_planned(out, x, topk_weights, plan, w13, w2, 10.0, plan.experts)
    finally:
        torch.cuda.set_sync_debug_mode("default")
```
Controls:
- Run the same body with `exl3_moe_accumulate(out, x, topk_weights, topk_ids, ...)` under `"error"` once by hand and confirm it raises. This proves the check can fail.
- If `exl3_linear` itself trips the check (for example, the `rows <= AUTO_RECONSTRUCT_THRESHOLD` branch reading a shape), report which op before changing anything.

- [ ] **Step 3: Run on divix01 under the GPU lock**

In the private worktree at the pushed commit:
```bash
PYTHONPATH=$PWD/python /data/models/slang/.venv/bin/python -c "import sglang; print(sglang.__file__)"
flock /data/models/slang/nvfp4-work/cc-gpu.lock taskset -c 32-63 env PYTHONPATH=$PWD/python SGLANG_EXL3_SRC=<value of arm_env.EXL3_SRC> \
  /data/models/slang/.venv/bin/python -m pytest test/manual/dsv41/test_exl3_stream_apply_gpu.py test/registered/unit/layers/moe/test_expert_plugins_cuda.py -q -p no:randomly 2>&1 | tail -5; echo EXIT=${PIPESTATUS[0]}
```
Expected: all pass, `EXIT=0`. Record the command and counts.

- [ ] **Step 4: Mutants (in the private worktree only; `git checkout -- <file>` after each)**

Each must turn at least one test red:
1. `Exl3RoutePlan.from_topk`: `stable=True` changed to `stable=False`, plus the `repeated` case reordered so an unstable sort differs. If CPU argsort happens to be stable anyway, use mutant 1b: reverse `positions` within each expert.
2. `host_row_of_source`: hits placed before misses.
3. `_apply_streamed` flag-on branch: the `experts` sequence passed to `exl3_moe_accumulate_planned` replaced with `reversed(experts)` (fp32 order changes, so the output bits change).
4. `_gather_cached`: `hot_out.extend((slots_host, ...))` with the slots list rotated by one.

Record which test caught each. After reverting, re-run Step 3's command and record that it is green again.

- [ ] **Step 5: Registered suite against the merge base**

```bash
PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 taskset -c 0-63 /data/models/slang/.venv/bin/python -m pytest test/registered/unit/kernels -q -p no:randomly 2>&1 | tail -3; echo EXIT=${PIPESTATUS[0]}
```
Run it here and at the merge base (a second private worktree at `git merge-base HEAD 4831d251bc`), then diff the counts. Also run `test/registered/unit/layers/quantization test/registered/unit/layers/moe test/registered/unit/test_dsv41_config.py` at both commits. Record every command next to its counts.

- [ ] **Step 6: Commit the GPU test changes**

```bash
git add test/manual/dsv41/test_exl3_stream_apply_gpu.py
git commit -m "test(dsv41): route plan parity with real EXL3 kernels, and no host sync in the planned chunk body"
git fetch origin && git rebase origin/master && git push origin HEAD:master
```

---

### Task 7: A/B arms (divix01, production stopped)

**Files:**
- Create: `analysis/dsv41-drive/route-plan/drive_arms.sh`

**Interfaces:**
- Consumes: the pushed commit from Task 6, registered as a code generation.

- [ ] **Step 1: Set up the worktree and register the generation**

```bash
git -C /data/models/slang/sglang fetch origin
git -C /data/models/slang/sglang worktree add --detach /data/models/slang/nvfp4-work/wt-route-plan-arms <sha>
cd /data/models/slang/nvfp4-work/wt-route-plan-arms
PYTHONPATH=$PWD/benchmarks/dsv41_baseline /data/models/slang/.venv/bin/python -c "import generations; generations.register('$PWD/python', 'route-plan-<short sha>')"
```

- [ ] **Step 2: Write the driver**

The template is `analysis/dsv41-drive/nvme-load/drive_traced_arm.sh`: the disk lock first, then poll the GPU lock, pinned to `EXPECT_SHA`.
```bash
#!/usr/bin/env bash
# Route plan A/B (plan 2026-09-25-dsv41-prefill-route-plan): A = production recipe, B = + the route plan. A then B, once.
set -u
WT=/data/models/slang/nvfp4-work/wt-route-plan-arms
SHA=$1
say() { echo "$(date +%T) $*"; }
cd $WT
[ "$(git rev-parse HEAD)" = "$SHA" ] || { say "worktree not at $SHA"; exit 1; }
PYTHONPATH=$WT/python /data/models/slang/.venv/bin/python -c "import sglang; print(\"sglang from\", sglang.__file__)"
exec 8>/data/models/slang/nvfp4-work/rowimg-disk.lock
say "waiting for rowimg-disk.lock"; flock 8; say "rowimg-disk.lock held"
while ! flock -n /data/models/slang/nvfp4-work/cc-gpu.lock true; do say "cc-gpu.lock held; waiting"; sleep 60; done
say "arm A"; EXPECT_SHA=$SHA bash benchmarks/dsv41_baseline/run_arm.sh route-plan-A 30021; say "A rc=$?"
while ! flock -n /data/models/slang/nvfp4-work/cc-gpu.lock true; do sleep 30; done
say "arm B"; EXPECT_SHA=$SHA bash benchmarks/dsv41_baseline/run_arm.sh route-plan-B 30021 SGLANG_DSV41_ENABLE_PREFILL_ROUTE_PLAN=1; say "B rc=$?"
say "DRIVER DONE"
```
Commit and push it. Run it on divix01 with `setsid nohup bash analysis/dsv41-drive/route-plan/drive_arms.sh <sha> > /mnt/nvme1/prefill-opt/arms.log 2>&1 < /dev/null &`.

- [ ] **Step 3: Read the results**

From each arm's `run-manifest.json` and verdict, report:
- TTFT for both timed sessions;
- pooled ms/token;
- verdict flags;
- output identity: the greedy token ids of A and B must be identical per session.

A difference in output is a failure: stop and debug with superpowers:systematic-debugging. Expected: TTFT drops in B, decode ms/token unchanged (±1-2 ms; decode never takes this path).

---

### Task 8: Traced arm B and the write-up

**Files:**
- Modify: `analysis/dsv41-drive/route-plan/drive_arms.sh` (a `traced` mode), or create `drive_traced.sh` beside it
- Modify: `DSV41_REFERENCE.md` (new section 27.11; section 27.4 item 7), `MOE_PREFILL_OPT.md`

- [ ] **Step 1: One node-mode traced arm of B**

Same locking as Task 7. Command:
```bash
NSYS_TMPDIR=/mnt/nvme1/nsys-tmp NSYS_TRACE=1 NSYS_CUDA_GRAPH_TRACE=node EXPECT_SHA=$SHA \
  bash benchmarks/dsv41_baseline/run_arm.sh route-plan-B-node 30021 SGLANG_DSV41_ENABLE_PREFILL_ROUTE_PLAN=1
```
Export the report to SQLite on divix01 (`nsys export --type sqlite`, under `taskset -c 0-31`, into `/mnt/nvme1/dsv41-nsys/`).

- [ ] **Step 2: Compare against `prod-flags-node`**

Under `ulimit -v 8000000; taskset -c 0-31`:
```bash
python analysis/dsv41-drive/pcie-trace/compare_arms.py before=/mnt/nvme1/dsv41-nsys/prod-flags-node-20260925-211044.sqlite,/mnt/nvme1/dsv41-nsys/prod-flags-node-20260925-211044-pcie.sqlite after=<B sqlite>,<B pcie sqlite>
python analysis/dsv41-drive/pcie-trace/prefill_sync_sites.py <B sqlite>
python analysis/dsv41-drive/pcie-trace/chunk_host_time.py <B sqlite>
```
If the `-pcie.sqlite` exports do not exist, export them first. Report:
- prefill wall, GPU busy and GPU idle;
- readbacks ≤256 B and host time blocked in them;
- eager kernel count;
- mean PCIe RX;
- the new per-chunk host time.

Name where the host now blocks. Expect the first sync of the next layer's attention to wait for the last chunk's gather, and `_await_fills` waits. Also give the remaining split: link floor, host, and NVMe waits.

- [ ] **Step 3: Write section 27.11 in `DSV41_REFERENCE.md`, and update section 27.4 item 7 and `MOE_PREFILL_OPT.md`**

Section 27.11 must contain:
- the attribution table from this plan's "What the trace says";
- what changed, with the flag name and the commits;
- the parity and test results, each with its command and counts;
- the mutants, and which test caught each;
- the A/B table: TTFT per session, pooled ms/token, output identity;
- the before/after trace table from Step 2;
- the next target the trace names.

Mark section 27.4 item 7 as partly done, with a pointer to 27.11. In `MOE_PREFILL_OPT.md`, add a line at the top saying steps 1, 2 and 4 are done in 27.11 and what remains.

- [ ] **Step 4: Recommend the production recipe change, but don't make it yet**

If B is faster with identical outputs, propose adding `"SGLANG_DSV41_ENABLE_PREFILL_ROUTE_PLAN": "1"` to `benchmarks/dsv41_baseline/arm_env.py` `base_env()`. Ask the user before committing it. Production currently runs from `dsv41-direct-live/launch-hot16282.sh` (hot cache 16282 MiB, `--mem-fraction-static 0.89`), which layers overrides on `base_env()`. Ask the user before restarting production.

- [ ] **Step 5: Commit and push the docs**

```bash
git add DSV41_REFERENCE.md MOE_PREFILL_OPT.md analysis/dsv41-drive/route-plan/
git commit -m "docs(dsv41): reference 27.11, the prefill route plan measured"
git fetch origin && git rebase origin/master && git push origin HEAD:master
```

## Not in this plan (decided by Task 8's trace)

- **Grouped expert compute** (spec step 3). This means one launch for many experts instead of ~20 small kernels each. It matters only if Task 8 shows the host, not the link, still bounding the chunk (C > G).
- **Sync-free pre-gather work for chunk k+1** (`lookup`/`.item()`, `nonzero`, and `ensure_rows` against a host mirror of `expert_to_slot`). It matters if P dominates the remaining idle.
- **The next layer's attention syncs** waiting on the last chunk's gather.
- **Double-buffered staging** (spec step 5). The GPU compute per chunk is ~3 ms against a ~52 ms gather, so overlapping them buys little, and it needs ~852 MB of VRAM that the new 16282 MiB hot cache has taken.
