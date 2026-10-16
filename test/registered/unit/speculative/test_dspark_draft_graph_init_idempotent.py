"""DSparkWorkerV2.init_cuda_graphs may run twice: the startup hot-cache calibration's trial capture
(SGLANG_OPT_HOT_CACHE_CALIBRATE_AT_STARTUP) calls it before the real capture. The folded-proposal draft sampler must be
built once and its capture hook appended once, so the real capture records the same graph as a single call does.

`DSparkWorkerV2` cannot be constructed on CPU, so the method runs unbound on a stand-in carrying the attributes it
reads, with the module's GPU predicates patched.
"""

from contextlib import nullcontext
from types import SimpleNamespace

from sglang.srt.environ import envs
from sglang.srt.speculative.dspark_components import dspark_worker_v2 as dw
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _worker(built):
    captures = []
    return SimpleNamespace(
        _decode_graph_allowed=True,
        draft_model=object(),
        _tp_sync=SimpleNamespace(available_memory_gb=lambda *a, **k: 4.0),
        device="cuda",
        gpu_id=0,
        _draft_graph_group=None,
        ps=SimpleNamespace(tp_rank=0),
        _draft_context=nullcontext,
        draft_model_runner=SimpleNamespace(capture_tail_hooks=[]),
        _draft_sampler=None,
        _draft_sampler_built=False,
        _maybe_build_draft_sampler=lambda *, available_memory_gb: built.append(available_memory_gb) or f"sampler{len(built)}",
        _proposer=SimpleNamespace(attach_draft_sampler=lambda s: captures.append(("attach", s))),
        _draft_worker=SimpleNamespace(init_cuda_graphs=lambda **k: captures.append(("capture", k))),
    ), captures


def test_two_calls_build_one_sampler_and_append_one_hook(monkeypatch):
    monkeypatch.setattr(dw, "draft_graph_allowed", lambda model: True)
    monkeypatch.setattr(dw, "is_cuda_alike", lambda: True)
    monkeypatch.setattr(dw, "draft_runs_exl3", lambda model: False)
    monkeypatch.setattr(dw, "draft_pp_context", nullcontext)
    monkeypatch.setattr(dw, "make_draft_sampler_capture_hook", lambda sampler: ("hook", sampler))
    built = []
    worker, captures = _worker(built)
    with envs.SGLANG_DSPARK_FOLDED_PROPOSAL.override(True):
        dw.DSparkWorkerV2.init_cuda_graphs(worker)
        dw.DSparkWorkerV2.init_cuda_graphs(worker)
    assert built == [4.0]  # one sampler
    assert worker.draft_model_runner.capture_tail_hooks == [("hook", "sampler1")]  # one hook
    assert captures == [("attach", "sampler1"), ("capture", {"capture_decode_cuda_graph": True})] * 2
