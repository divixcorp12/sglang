"""DSpark's graphed verify: a flagged forward is re-run eagerly, and startup skips what cannot be captured (CPU)."""

from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import pytest

from sglang.srt.environ import envs
from sglang.srt.model_executor.runner.decode_cuda_graph_runner import DecodeCudaGraphRunner
from sglang.srt.speculative.dspark_components import dspark_worker_v2
from sglang.srt.speculative.dspark_components.dspark_graphed_verify import (
    draft_graph_allowed,
    draft_runs_exl3,
    forward_verify_with_reverify,
    target_gather_is_narrow,
)
from sglang.srt.speculative.dspark_components.dspark_worker_v2 import DSparkWorkerV2
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _Manager:
    def __init__(self, narrow, overflows):
        self.narrow_graph_gather = narrow
        self._overflows = list(overflows)
        self.suspended = False
        self.reads = 0
        self.graphed = []

    def take_verify_overflow(self, graphed=True):
        self.reads += 1
        self.graphed.append(graphed)
        return self._overflows.pop(0)

    @contextmanager
    def suspend_graph_gather(self):
        self.suspended = True
        try:
            yield
        finally:
            self.suspended = False


def _runner(manager):
    graph_runner = object.__new__(DecodeCudaGraphRunner)
    return SimpleNamespace(expert_hot_cache_manager=manager, decode_cuda_graph_runner=graph_runner)


def _forward(runner, log):
    def forward():
        graphed = runner.decode_cuda_graph_runner.can_run_graph(SimpleNamespace(replace_embeds=None)) is not False
        log.append((graphed, runner.expert_hot_cache_manager.suspended))
        return SimpleNamespace(can_run_cuda_graph=graphed, tag=len(log))

    return forward


def test_a_flagged_graphed_verify_is_re_run_eagerly_with_the_gather_suspended(monkeypatch):
    manager = _Manager(narrow=True, overflows=[True])
    runner = _runner(manager)
    monkeypatch.setattr(DecodeCudaGraphRunner, "can_run_graph", lambda self, batch: not self._eager_only)
    log = []
    out = forward_verify_with_reverify(runner, _forward(runner, log))
    assert log == [(True, False), (False, True)]
    assert out.tag == 2 and not out.can_run_cuda_graph
    assert not manager.suspended and not runner.decode_cuda_graph_runner._eager_only


def test_an_unflagged_graphed_verify_is_kept(monkeypatch):
    manager = _Manager(narrow=True, overflows=[False])
    runner = _runner(manager)
    monkeypatch.setattr(DecodeCudaGraphRunner, "can_run_graph", lambda self, batch: not self._eager_only)
    log = []
    out = forward_verify_with_reverify(runner, _forward(runner, log))
    assert log == [(True, False)] and out.tag == 1 and manager.reads == 1


@pytest.mark.parametrize("graphed", [True, False])
def test_no_narrow_gather_reads_nothing(graphed):
    manager = _Manager(narrow=False, overflows=[])
    runner = SimpleNamespace(expert_hot_cache_manager=manager, decode_cuda_graph_runner=None)
    out = forward_verify_with_reverify(runner, lambda: SimpleNamespace(can_run_cuda_graph=graphed))
    assert manager.reads == 0 and out.can_run_cuda_graph is graphed


@pytest.mark.parametrize("overflowed, runs", [(True, 2), (False, 1)])
def test_a_verify_the_runner_did_not_graph_still_answers_to_the_narrowed_gather(monkeypatch, overflowed, runs):
    """The gather is chosen by route count, not by whether a graph runs: an eager verify (replace_embeds, a refused
    attention key) also takes the narrowed gather and may overflow into slot 0. Its flag is read, and a flagged one is
    re-run with the gather suspended. Mutant: return before the read when the runner did not graph -- red."""
    manager = _Manager(narrow=True, overflows=[overflowed])
    runner = _runner(manager)
    monkeypatch.setattr(DecodeCudaGraphRunner, "can_run_graph", lambda self, batch: False)
    log = []
    out = forward_verify_with_reverify(runner, _forward(runner, log))
    assert log == [(False, False), (False, True)][:runs] and out.tag == runs
    assert manager.graphed == [False] and not manager.suspended


def test_no_hot_cache_reads_nothing():
    runner = SimpleNamespace(expert_hot_cache_manager=None, decode_cuda_graph_runner=None)
    out = forward_verify_with_reverify(runner, lambda: SimpleNamespace(can_run_cuda_graph=True))
    assert out.can_run_cuda_graph


def test_the_test_switch_re_runs_every_graphed_verify(monkeypatch):
    manager = _Manager(narrow=True, overflows=[False])
    runner = _runner(manager)
    monkeypatch.setattr(DecodeCudaGraphRunner, "can_run_graph", lambda self, batch: not self._eager_only)
    log = []
    with envs.SGLANG_TEST_DSPARK_FORCE_REVERIFY.override(True):
        forward_verify_with_reverify(runner, _forward(runner, log))
    assert log == [(True, False), (False, True)] and manager.reads == 1


def test_eager_only_refuses_the_graph_and_restores_on_error():
    runner = object.__new__(DecodeCudaGraphRunner)
    assert runner._eager_only is False
    with pytest.raises(KeyError):
        with runner.eager_only():
            assert runner.can_run_graph(SimpleNamespace(replace_embeds=None)) is False
            raise KeyError("restore")
    assert runner._eager_only is False


def test_draft_runs_exl3():
    exl3 = SimpleNamespace(quant_config=SimpleNamespace(get_name=lambda: "exl3"))
    fp8 = SimpleNamespace(quant_config=SimpleNamespace(get_name=lambda: "fp8"))
    assert draft_runs_exl3(exl3) and not draft_runs_exl3(fp8)
    assert not draft_runs_exl3(SimpleNamespace(quant_config=None)) and not draft_runs_exl3(SimpleNamespace())


def test_target_gather_is_narrow():
    assert target_gather_is_narrow(SimpleNamespace(expert_hot_cache_manager=SimpleNamespace(narrow_graph_gather=True)))
    assert not target_gather_is_narrow(SimpleNamespace(expert_hot_cache_manager=SimpleNamespace(narrow_graph_gather=False)))
    assert not target_gather_is_narrow(SimpleNamespace(expert_hot_cache_manager=None))
    assert not target_gather_is_narrow(SimpleNamespace())


def _draft(name):
    return SimpleNamespace(quant_config=SimpleNamespace(get_name=lambda: name))


def test_an_exl3_draft_captures_unless_disabled():
    assert draft_graph_allowed(_draft("exl3"))
    with envs.SGLANG_DSV41_DISABLE_DSPARK_DRAFT_GRAPH.override(True):
        assert not draft_graph_allowed(_draft("exl3"))
        assert draft_graph_allowed(_draft("fp8"))
    assert draft_graph_allowed(_draft("fp8"))


def _worker(monkeypatch, draft, log):
    monkeypatch.setattr(dspark_worker_v2, "is_cuda_alike", lambda: True)
    monkeypatch.setattr(dspark_worker_v2, "draft_pp_context", nullcontext)
    worker = object.__new__(DSparkWorkerV2)
    worker._decode_graph_allowed = True
    worker.draft_model = draft
    worker._tp_sync = SimpleNamespace(available_memory_gb=lambda *a, **k: 10.0)
    worker.device = "cuda"
    worker.gpu_id = 0
    worker.ps = SimpleNamespace(tp_rank=0)
    worker._draft_graph_group = None
    worker._draft_context = nullcontext
    worker._draft_sampler = None
    worker._proposer = SimpleNamespace(attach_draft_sampler=lambda sampler: None)
    worker.draft_model_runner = SimpleNamespace(capture_tail_hooks=[])
    worker._draft_worker = SimpleNamespace(
        init_cuda_graphs=lambda capture_decode_cuda_graph: log.append(("capture", capture_decode_cuda_graph))
    )
    return worker


def test_init_cuda_graphs_prepares_the_exl3_draft_before_capture(monkeypatch):
    from sglang.srt.layers.quantization.exl3 import draft_moe

    log = []
    monkeypatch.setattr(draft_moe, "prepare_dspark_draft_graph", lambda model: log.append(("prepare", model)) or 3)
    draft = _draft("exl3")
    with envs.SGLANG_DSPARK_FOLDED_PROPOSAL.override(False):
        _worker(monkeypatch, draft, log).init_cuda_graphs()
    assert log == [("prepare", draft), ("capture", True)]


# The EXL3 draft MoE runs its graph-safe path eagerly too (DraftResidentMoe.run refuses before prepare()), so an
# EXL3 draft that is not captured is still prepared: the switch decides capture, never preparation.
def test_init_cuda_graphs_prepares_a_disabled_exl3_draft_and_keeps_it_eager(monkeypatch):
    from sglang.srt.layers.quantization.exl3 import draft_moe

    log = []
    monkeypatch.setattr(draft_moe, "prepare_dspark_draft_graph", lambda model: log.append("prepare") or 3)
    with envs.SGLANG_DSPARK_FOLDED_PROPOSAL.override(False), envs.SGLANG_DSV41_DISABLE_DSPARK_DRAFT_GRAPH.override(True):
        _worker(monkeypatch, _draft("exl3"), log).init_cuda_graphs()
    assert log == ["prepare", ("capture", False)]


def test_init_cuda_graphs_prepares_an_exl3_draft_short_of_memory_for_its_graph(monkeypatch):
    from sglang.srt.layers.quantization.exl3 import draft_moe

    log = []
    monkeypatch.setattr(draft_moe, "prepare_dspark_draft_graph", lambda model: log.append("prepare") or 3)
    worker = _worker(monkeypatch, _draft("exl3"), log)
    worker._tp_sync = SimpleNamespace(available_memory_gb=lambda *a, **k: 0.5)
    with envs.SGLANG_DSPARK_FOLDED_PROPOSAL.override(False):
        worker.init_cuda_graphs()
    assert log == ["prepare", ("capture", False)]


def test_init_cuda_graphs_prepares_an_exl3_draft_whose_decode_graph_is_not_allowed(monkeypatch):
    from sglang.srt.layers.quantization.exl3 import draft_moe

    log = []
    monkeypatch.setattr(draft_moe, "prepare_dspark_draft_graph", lambda model: log.append("prepare") or 3)
    worker = _worker(monkeypatch, _draft("exl3"), log)
    worker._decode_graph_allowed = False
    with envs.SGLANG_DSPARK_FOLDED_PROPOSAL.override(False):
        worker.init_cuda_graphs()
    assert log == ["prepare", ("capture", False)]


def test_init_cuda_graphs_prepares_nothing_for_a_non_exl3_draft(monkeypatch):
    from sglang.srt.layers.quantization.exl3 import draft_moe

    log = []
    monkeypatch.setattr(draft_moe, "prepare_dspark_draft_graph", lambda model: log.append("prepare") or 0)
    with envs.SGLANG_DSPARK_FOLDED_PROPOSAL.override(False):
        _worker(monkeypatch, _draft("fp8"), log).init_cuda_graphs()
    assert log == [("capture", True)]
