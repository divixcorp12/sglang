"""The learned L+2 head served through the device hook (T3 scorer, F123's spec_head kernels) on a real GPU: the record
it publishes into the candidate page equals the offline l2_fast_kernels harness's for the same inputs, the launch stays
on the spec stream and never runs the native select, a captured spec_score replays with new inputs and a new posted
seq, and the row backend's post publishes the head's read.

Run on divix01 under cc-gpu.lock, with PYTHONPATH pointing at the tree under test.
"""

import struct
import sys
from pathlib import Path

import msgspec
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "analysis" / "dsv41-drive" / "prefetch-replay"))

import b1_head  # noqa: E402
import l2_fast_kernels  # noqa: E402
import l2_fast_torch  # noqa: E402
import r1_cost  # noqa: E402
import r1_models  # noqa: E402
from lease_chain_rig import EXPERTS, TOP_K, Chain  # noqa: E402

from sglang.kernels.ops.moe import expert_stream_transport as es  # noqa: E402
from sglang.kernels.ops.moe import spec_head  # noqa: E402
from sglang.srt.layers.moe import ram_prefetch_learned_head as head_file  # noqa: E402
from sglang.srt.layers.moe.expert_prediction.training.pca import PCAStats  # noqa: E402
from sglang.srt.layers.moe.ram_prefetch import SpecScoreRow  # noqa: E402

BF16 = torch.bfloat16
X_DIM = r1_cost.X_DIM
RANK = 128
LAYERS = 4  # map rows 0..3 are model layers 19..22; at horizon 2 row 0 targets row 2 and row 1 targets row 3
LAYER_IDS = [19, 20, 21, 22]
FOLD_ROWS = (
    l2_fast_torch.FoldRow(source_layer=19, target_layer=21, stratum="b"),
    l2_fast_torch.FoldRow(source_layer=20, target_layer=22, stratum="a"),
)
# Head row of each target map row: the fold's rows are sorted by target layer.
ROW_OF_TARGET = {2: 0, 3: 1}
LOW, HIGH = -1e9, 1e9  # thresholds that always read / never read
HOT_CAPACITY = 4
TOKENS_MAX = 6


def _model() -> r1_models.R1Model:
    cfg = b1_head.HeadConfig(variant="E5", pca_rank=RANK, hidden=512, emb_dim=4, num_experts=EXPERTS)
    g = torch.Generator().manual_seed(0)
    pca = PCAStats(
        mean=torch.randn(X_DIM, generator=g) * 0.5,
        components=torch.randn((RANK, X_DIM), generator=g) / X_DIM**0.5,
        explained_variance=torch.ones(RANK),
    )
    head = b1_head.B1Head(cfg, pca=pca, vocab=b1_head.Vocab(layers=tuple(LAYER_IDS), strata=("a", "b")))
    model = r1_models.R1Model(r1_models.R1Config(head=cfg, position=True, aggregator=True), head=head)
    with torch.no_grad():  # nonzero everywhere, so every fold term and both head rows matter
        for p in model.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * 0.05)
    return model.eval()


def _write_head(path: str) -> l2_fast_torch.FoldedHead:
    folded = l2_fast_torch.fold(_model(), rows=FOLD_ROWS)
    weights = l2_fast_kernels.to_kernel_weights(folded, device="cpu")
    meta = head_file.HeadMeta(
        format_version=head_file.HEAD_FORMAT_VERSION,
        checkpoint_sha256="00" * 32,
        threshold=LOW,
        threshold_source={"path": "test", "split": "calib", "scorer": "L2_T3", "budget": 0.75},
        horizon=2,
        rows=tuple(head_file.HeadRow(r.source_layer, r.target_layer, i) for i, r in enumerate(FOLD_ROWS)),
    )
    tensors = {name: getattr(weights, name) for name in head_file.TENSOR_NAMES}
    head_file.write_head_file(path, tensors=tensors, meta=meta)
    return folded


def _serving_head(path: str) -> head_file.ServingHead:
    """The file through the serving loader, as ``ExpertEngineService.attach`` loads it."""
    targets = torch.full((LAYERS, 2), -1, dtype=torch.int64)
    targets[0, 0], targets[1, 0] = 2, 3
    loaded = head_file.load_learned_head(path, layer_ids=LAYER_IDS, targets=targets)
    return head_file.serving_head(loaded, device="cuda", top_k=TOP_K)


def _prime(chain: Chain) -> torch.Tensor:
    """Enable the GPU scorer on the rig's device; returns its candidate page.

    The rig has no CPU experts, so the scorer's score scratch and the head's hidden-size check read these two
    attributes from values ``enable_cpu_experts`` would have set.
    """
    chain.dev.cpu_out_rows = torch.empty((LAYERS, 1, X_DIM))
    chain.dev.cpu_tokens_max = TOKENS_MAX
    page = es.new_candidate_page(pin=True)
    chain.dev.enable_spec_scorer(page, top_k=TOP_K, per_token=2, top_k_only=False)
    return page


class _Rig(msgspec.Struct):
    chain: Chain
    page: torch.Tensor
    head: head_file.ServingHead
    offline: l2_fast_kernels.KernelHead
    seqs: int = 0

    def next_seq(self) -> int:
        self.seqs += 1
        return self.seqs


def _build(tmp_path: Path, *, start: bool) -> _Rig:
    path = str(tmp_path / "t3.safetensors")
    folded = _write_head(path)
    (tmp_path / "chain").mkdir()
    chain = Chain(tmp_path / "chain", layers=LAYERS, start=start)
    try:
        page = _prime(chain)
        head = _serving_head(path)
        chain.dev.enable_spec_head(head)
        ram_slot = chain.dev.map_bank["ram_slot"]
        ram_slot[2, [3, 7]] = 0  # experts mapped in RAM at target row 2 and 3: skipped
        ram_slot[3, [4, 11]] = 0
        torch.cuda.synchronize()
        return _Rig(chain=chain, page=page, head=head, offline=l2_fast_kernels.KernelHead(folded, device="cuda"))
    except BaseException:
        chain.close()
        raise


@pytest.fixture(scope="module")
def rig(tmp_path_factory):
    r = _build(tmp_path_factory.mktemp("t3"), start=False)
    yield r
    r.chain.close()


def _bits(value: float) -> int:
    return struct.unpack("<i", struct.pack("<f", value))[0]


def _record(read: es.CandidateRead) -> tuple[int, int, int]:
    """(count, expert or -1, margin's f32 bits) of a published record; count 0 carries no entry."""
    assert read.status == "ready"
    if read.count == 0:
        return (0, -1, 0)
    expert, _rank, margin = read.picks[0]
    return (read.count, expert, _bits(margin))


class _Inputs(msgspec.Struct):
    x: torch.Tensor  # bf16 [tokens, X_DIM]
    routes: torch.Tensor  # int64 [tokens, TOP_K]
    w: torch.Tensor  # fp32 [tokens, TOP_K]
    entry: SpecScoreRow


def _inputs(seed: int, *, tokens: int, target: int) -> _Inputs:
    g = torch.Generator().manual_seed(seed)
    routes = torch.stack([torch.randperm(EXPERTS, generator=g)[:TOP_K] for _ in range(tokens)])
    hot = torch.full((HOT_CAPACITY,), -1, dtype=torch.int64)
    hot[:2] = torch.randperm(EXPERTS, generator=g)[:2]
    gate = (torch.randn((EXPERTS, X_DIM), generator=g) * 0.05).to(BF16).cuda()
    bias = (torch.randn(EXPERTS, generator=g) * 0.1).cuda()
    return _Inputs(
        x=torch.randn((tokens, X_DIM), generator=g).to(BF16).cuda(),
        routes=routes.cuda(),
        w=torch.rand((tokens, TOP_K), generator=g).cuda(),
        entry=SpecScoreRow(gate, bias, target, hot.cuda(), HOT_CAPACITY),
    )


def _serve(r: _Rig, case: _Inputs, *, seq: int) -> es.CandidateRead:
    dev = r.chain.dev
    dev.state[es.STATE_WORDS["posted"]] = seq
    dev.spec_score(case.entry, case.x, routes=case.routes.reshape(-1), route_weights=case.w)
    dev.spec_join()
    torch.cuda.synchronize()
    return es.read_candidates(r.page, seq)


def _offline(r: _Rig, case: _Inputs, *, row: int) -> tuple[int, int, int]:
    """The offline harness's read of the scores the serving launch just wrote, with the device map's mapped experts
    and the entry's hot experts as its skip mask."""
    dev, entry = r.chain.dev, case.entry
    skip = dev.map_bank["ram_slot"][entry.target] >= 0
    skip[entry.hot_slots[entry.hot_slots >= 0]] = True
    tokens = int(case.x.shape[0])
    count, expert, value, _ = r.offline.score(
        scores=dev.spec_scores, x=case.x, src_ids=case.routes, src_w=case.w, positions=r.head.positions[:tokens],
        row=row, skip=skip, threshold=r.head.threshold,
    )  # fmt: skip
    return (count, expert, _bits(value))


@pytest.mark.parametrize("tokens", [1, 3])
@pytest.mark.parametrize("target", [2, 3])
@pytest.mark.parametrize("seed", range(3))
def test_the_served_record_equals_the_offline_harnesss(rig, seed, target, tokens):
    """Same x, scores, routes, route weights, skip masks and head row through the hook and through KernelHead: the
    published (count, expert, margin bits) are equal, and the two head rows read differently so a wrong row shows.
    Mutant: the launch uses head row 0 for every target (or the target row number) -- red."""
    case = _inputs(seed, tokens=tokens, target=target)
    got = _serve(rig, case, seq=rig.next_seq())
    want = _offline(rig, case, row=ROW_OF_TARGET[target])
    other = _offline(rig, case, row=1 - ROW_OF_TARGET[target])
    assert got.flags == 0 and got.count == 1
    assert want != other, "pick seeds whose two head rows publish different reads"
    assert _record(got) == want


def test_a_score_below_the_threshold_publishes_count_zero(rig):
    """The frozen threshold abstains: count 0, no oversize flag, expert -1 in the offline read. Mutant: the launch
    publishes the best expert without comparing the threshold -- red."""
    case = _inputs(5, tokens=2, target=2)
    rig.head.threshold.fill_(HIGH)
    try:
        got = _serve(rig, case, seq=rig.next_seq())
        want = _offline(rig, case, row=ROW_OF_TARGET[2])
    finally:
        rig.head.threshold.fill_(LOW)
    assert (got.count, got.flags, got.picks) == (0, 0, ())
    assert _record(got) == want == (0, -1, 0)


def test_the_head_never_selects_and_every_launch_runs_on_the_spec_stream(rig, monkeypatch):
    """spec_score then the head kernels on the device's spec stream, forked from the post's; run_spec_select is not
    called. Mutant: launch the head kernels on the current stream, or fall through to run_spec_select -- red."""
    seen = []
    real_score, real_head = es.run_spec_score, spec_head.run_spec_head_serving

    def score(*a, **kw):
        seen.append(("score", torch.cuda.current_stream()))
        return real_score(*a, **kw)

    def head(*a, **kw):
        seen.append(("head", torch.cuda.current_stream()))
        return real_head(*a, **kw)

    def select(*a, **kw):
        raise AssertionError("the head path must not run the native select")

    monkeypatch.setattr(es, "run_spec_score", score)
    monkeypatch.setattr(spec_head, "run_spec_head_serving", head)
    monkeypatch.setattr(es, "run_spec_select", select)
    main = torch.cuda.current_stream()
    case = _inputs(1, tokens=1, target=2)
    got = _serve(rig, case, seq=rig.next_seq())
    assert [name for name, _ in seen] == ["score", "head"]
    stream = rig.chain.dev.spec_stream
    assert all(s == stream and s != main for _, s in seen)
    assert _record(got) == _offline(rig, case, row=ROW_OF_TARGET[2])


def test_a_captured_spec_score_replays_with_new_inputs_and_a_new_seq(rig):
    """spec_score and spec_join captured once; each replay's inputs and posted seq land in that seq's slot and equal
    the eager launch's record and the offline read. Mutant: the slot or seq is a host value baked at capture -- red."""
    dev = rig.chain.dev
    case = _inputs(3, tokens=3, target=3)
    x, routes, w = case.x, case.routes.reshape(-1).clone(), case.w
    live = msgspec.structs.replace(case, routes=routes.view(3, TOP_K))  # the capture's buffers, refilled per replay
    rig.head.threshold.fill_(LOW)
    _serve(rig, live, seq=rig.next_seq())  # eager warm-up loads the kernels outside the capture
    stream, graph = torch.cuda.Stream(), torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
        dev.spec_score(case.entry, x, routes=routes, route_weights=w)
        dev.spec_join()
    slots = set()
    for i, seq in enumerate((7, 18, 33)):
        fresh = _inputs(40 + i, tokens=3, target=3)
        rig.page.zero_()
        torch.cuda.synchronize()
        with torch.cuda.stream(stream):
            x.copy_(fresh.x)
            routes.copy_(fresh.routes.reshape(-1))
            w.copy_(fresh.w)
            case.entry.hot_slots.copy_(fresh.entry.hot_slots)
            dev.state[es.STATE_WORDS["posted"]] = seq
            graph.replay()
        stream.synchronize()
        slots.add(es.candidate_offset(seq))
        replayed = _record(es.read_candidates(rig.page, seq))
        assert replayed == _offline(rig, live, row=ROW_OF_TARGET[3]), seq
        rig.page.zero_()
        assert _record(_serve(rig, live, seq=seq)) == replayed, seq
    assert len(slots) == 3


def test_the_row_backends_post_publishes_the_heads_read(tmp_path):
    """The production post with cpu_input=(x, w): its record is the head's read of x past the routes the plan set,
    the entry's hot experts and the target row's mapped experts. Mutant: post passes no routes or weights to the
    hook, or the native select runs -- red."""
    r = _build(tmp_path, start=True)
    c = r.chain
    try:
        case = _inputs(9, tokens=1, target=2)
        backend = c.backends[0]
        backend.spec_expected = True
        backend.spec_score = case.entry
        backend.cpu_input = (case.x, case.w)
        c.plan([0, 1], 0)
        c.gather(0)
        torch.cuda.synchronize()
        assert c.handled()
        seq = int(c.dev.stats()["posted"]) & 0xFFFFFFFF
        ids = torch.full((1, TOP_K), -1, dtype=torch.int64, device="cuda")
        ids[0, :2] = torch.tensor([0, 1])
        got = es.read_candidates(r.page, seq)
        want = _offline(r, msgspec.structs.replace(case, routes=ids), row=ROW_OF_TARGET[2])
        assert got.status == "ready" and got.count == 1
        assert _record(got) == want
    finally:
        c.close()


def test_enabling_the_head_refuses_a_missing_scorer_and_the_wrong_expert_count(tmp_path, rig):
    """Mutants: enable_spec_head without enable_spec_scorer first, or with a head of another expert count, is
    accepted -- red."""
    c = Chain(tmp_path, layers=LAYERS, start=False)
    try:
        c.dev.cpu_out_rows = torch.empty((LAYERS, 1, X_DIM))
        with pytest.raises(RuntimeError):
            c.dev.enable_spec_head(rig.head)
        _prime(c)
        narrow = msgspec.structs.replace(rig.head.weights, w_out=rig.head.weights.w_out[:8])
        with pytest.raises(ValueError):
            c.dev.enable_spec_head(msgspec.structs.replace(rig.head, weights=narrow))
        assert c.dev.spec_head is None
    finally:
        c.close()
