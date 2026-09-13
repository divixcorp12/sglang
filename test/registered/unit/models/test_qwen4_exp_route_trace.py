"""Tests for the debug-only Qwen4-Exp MoE route trace writer and hooks."""

import enum
import json
import os
import shutil
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn

from sglang.srt.models import qwen4_exp_route_trace as route_trace
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

HC = 2
HIDDEN = 3
EXPERTS = 4
TOP_K = 2
LAYERS = 2


class _Mode:
    def __init__(self, decode):
        self.decode = decode

    def is_decode(self):
        return self.decode


def _batch(token, position, decode=True, request=0):
    return SimpleNamespace(
        forward_mode=_Mode(decode),
        input_ids=torch.tensor([token]),
        positions=torch.tensor([position]),
        seq_lens=torch.tensor([position + 1]),
        req_pool_indices=torch.tensor([request]),
    )


class _HyperConnection(nn.Module):
    config = SimpleNamespace(rms_norm_eps=1e-6)

    def mix(self, hyper_input):
        mixed = hyper_input.view(*hyper_input.shape[:-1], HC, HIDDEN).mean(-2)
        return mixed, hyper_input


class _Gate(nn.Linear):
    def forward(self, x):
        return super().forward(x), None


class _TopK(nn.Module):
    topk_config = SimpleNamespace(top_k=TOP_K)

    def forward(self, hidden_states, router_logits):
        weights, ids = torch.topk(router_logits.softmax(-1), TOP_K)
        return SimpleNamespace(
            topk_ids=ids.to(torch.int32),
            topk_weights=weights / weights.sum(-1, keepdim=True),
        )


class _Experts(nn.Module):
    def forward(self, hidden_states, topk_output):
        return hidden_states * 2


class _Moe(nn.Module):
    num_experts = EXPERTS

    def __init__(self):
        super().__init__()
        self.gate = _Gate(HIDDEN, EXPERTS, bias=False)
        self.topk = _TopK()
        self.experts = _Experts()

    def forward(self, hidden_states, forward_batch=None):
        logits, _ = self.gate(hidden_states)
        output = self.experts(hidden_states, self.topk(hidden_states, logits))
        output += 1.0
        return output


class _Layer(nn.Module):
    hc_count = HC
    hidden_size = HIDDEN

    def __init__(self):
        super().__init__()
        self.mlp_hyper_connection = _HyperConnection()
        self.mlp = _Moe()

    def forward(self, hidden_states, forward_batch):
        mixed, residual = self.mlp_hyper_connection.mix(hidden_states)
        output = self.mlp(mixed, forward_batch)
        return residual + output.repeat(1, HC)


class _LanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.embed = nn.Embedding(16, HC * HIDDEN)
        self.layers = nn.ModuleList([_Layer() for _ in range(LAYERS)])
        self.hyper_connection_mixer = _HyperConnection()

    def forward(self, input_ids=None, positions=None, forward_batch=None):
        hidden_states = self.embed(input_ids)
        self.layer_inputs = []
        for layer in self.layers:
            self.layer_inputs.append(hidden_states)
            hidden_states = layer(hidden_states, forward_batch)
        self.final_hc = hidden_states
        mixed, _ = self.hyper_connection_mixer.mix(hidden_states)
        return mixed


def _run(model, batch):
    with torch.no_grad():
        return model(input_ids=batch.input_ids, forward_batch=batch)


def _load(directory):
    with open(os.path.join(directory, route_trace.MANIFEST_NAME)) as handle:
        manifest = json.load(handle)
    shards = [
        torch.load(os.path.join(directory, shard["file"]))
        for shard in manifest["shards"]
    ]
    return manifest, shards


def test_decode_forward_records_router_tensors_before_in_place_shared_add():
    model = _LanguageModel()
    with tempfile.TemporaryDirectory() as directory:
        trace = route_trace.install_moe_route_trace(model, directory, max_tokens=1)
        final_hidden = _run(model, _batch(token=5, position=7))
        manifest, shards = _load(directory)

    assert manifest["complete"] and manifest["tokens"] == 1
    assert manifest["layer_ids"] == [0, 1]
    assert (manifest["hc_count"], manifest["hidden_size"]) == (HC, HIDDEN)
    assert (manifest["num_experts"], manifest["top_k"]) == (EXPERTS, TOP_K)
    shard = shards[0]
    for layer, layer_input in enumerate(model.layer_inputs):
        mixed = layer_input.view(1, HC, HIDDEN).mean(-2)
        logits = model.layers[layer].mlp.gate(mixed)[0]
        weights, ids = torch.topk(logits.softmax(-1), TOP_K)
        expected = {
            "hc_mid": layer_input,
            "router_input": mixed,
            "router_logits": logits,
            "routed_out": mixed * 2,
            "moe_out": mixed * 2 + 1,
        }
        for field, value in expected.items():
            torch.testing.assert_close(
                shard[field][:, layer].float(), value.to(torch.bfloat16).float()
            )
        assert shard["topk_ids"][0, layer].tolist() == ids[0].tolist()
        torch.testing.assert_close(
            shard["topk_weights"][:, layer], weights / weights.sum(-1, keepdim=True)
        )
    torch.testing.assert_close(
        shard["final_hc"].float(), model.final_hc.to(torch.bfloat16).float()
    )
    torch.testing.assert_close(
        shard["final_hidden"].float(), final_hidden.to(torch.bfloat16).float()
    )
    assert shard["input_ids"].tolist() == [5]
    assert shard["positions"].tolist() == [7]
    assert shard["seq_lens"].tolist() == [8]
    assert shard["forward_index"].tolist() == [0]
    assert trace.writer.closed


def test_only_decode_forwards_are_recorded():
    model = _LanguageModel()
    with tempfile.TemporaryDirectory() as directory:
        route_trace.install_moe_route_trace(model, directory, max_tokens=2)
        _run(model, _batch(token=1, position=0, decode=False))
        _run(model, _batch(token=2, position=1))
        _run(model, _batch(token=3, position=2, decode=False))
        _run(model, _batch(token=4, position=2))
        manifest, shards = _load(directory)

    assert manifest["tokens"] == 2 and manifest["forwards"] == 2
    assert torch.cat([shard["input_ids"] for shard in shards]).tolist() == [2, 4]


def test_trace_stops_at_max_tokens_and_restores_the_model():
    model = _LanguageModel()
    with tempfile.TemporaryDirectory() as directory:
        route_trace.install_moe_route_trace(
            model, directory, max_tokens=3, shard_tokens=2
        )
        for token in range(5):
            _run(model, _batch(token=token, position=token))
        manifest, shards = _load(directory)

    assert manifest["complete"] and manifest["tokens"] == 3
    assert [shard["tokens"] for shard in manifest["shards"]] == [2, 1]
    assert [shard["input_ids"].tolist() for shard in shards] == [[0, 1], [2]]
    for module in model.modules():
        assert not module._forward_hooks
        assert not module._forward_pre_hooks
        assert "mix" not in module.__dict__


def test_writer_refuses_a_directory_with_an_existing_trace():
    with tempfile.TemporaryDirectory() as directory:
        route_trace.MoeRouteTraceWriter(directory, 1, [0], {})
        with pytest.raises(FileExistsError):
            route_trace.MoeRouteTraceWriter(directory, 1, [0], {})


def test_writer_refuses_a_non_empty_directory_without_a_manifest():
    with tempfile.TemporaryDirectory() as directory:
        open(os.path.join(directory, "shard_00000.pt"), "wb").close()
        with pytest.raises(FileExistsError, match="not empty"):
            route_trace.MoeRouteTraceWriter(directory, 1, [0], {})


def test_failed_shard_write_stops_the_trace_and_serving_continues():
    model = _LanguageModel()
    with tempfile.TemporaryDirectory() as directory:
        trace = route_trace.install_moe_route_trace(
            model, directory, max_tokens=4, shard_tokens=1
        )
        real_save = torch.save

        def fail_after_temp_file(obj, path):
            real_save(obj, path)
            raise OSError(28, "No space left on device")

        with patch.object(route_trace.torch, "save", side_effect=fail_after_temp_file):
            _run(model, _batch(token=1, position=0))
        output = _run(model, _batch(token=2, position=1))
        manifest, _ = _load(directory)
        leftovers = sorted(os.listdir(directory))
        trace.writer.close()

    assert output.shape == (1, HIDDEN)
    assert trace.writer.closed and trace.writer.tokens == 1
    assert not manifest["complete"] and "No space left" in manifest["error"]
    assert manifest["shards"] == []
    assert leftovers == [route_trace.MANIFEST_NAME]
    for module in model.modules():
        assert not module._forward_hooks
        assert not module._forward_pre_hooks
        assert "mix" not in module.__dict__


def test_forward_missing_a_layer_field_is_dropped():
    with tempfile.TemporaryDirectory() as directory:
        writer = route_trace.MoeRouteTraceWriter(directory, 1, [0], {})
        writer.begin_forward(_batch(token=1, position=0))
        writer.record_layer("router_input", 0, torch.zeros(1, HIDDEN))
        writer.end_forward()
        assert writer.tokens == 0 and not writer.closed


def test_unset_environment_installs_nothing():
    model = _LanguageModel()
    with patch.dict(os.environ, {"SGLANG_MOE_ROUTE_TRACE_DIR": ""}):
        assert route_trace.maybe_install_moe_route_trace(model) is None
    for module in model.modules():
        assert not module._forward_hooks
        assert not module._forward_pre_hooks


SETTINGS = {
    "speculative_num_steps": 3,
    "speculative_num_draft_tokens": 4,
    "speculative_eagle_topk": 1,
}


class _ForwardMode(enum.Enum):
    EXTEND = 1
    DECODE = 2
    IDLE = 5
    TARGET_VERIFY = 6
    DRAFT_EXTEND_V2 = 7

    def is_decode(self):
        return self is _ForwardMode.DECODE

    def is_target_verify(self):
        return self is _ForwardMode.TARGET_VERIFY

    def is_idle(self):
        return self is _ForwardMode.IDLE


def _spec_batch(
    mode, tokens, positions, requests, extend_seq_lens=None, accept=None, front=0
):
    return SimpleNamespace(
        forward_mode=mode,
        input_ids=torch.tensor(tokens),
        positions=torch.tensor(positions),
        seq_lens=None,
        req_pool_indices=torch.tensor(requests),
        extend_seq_lens=(
            None if extend_seq_lens is None else torch.tensor(extend_seq_lens)
        ),
        spec_info=(
            None
            if accept is None
            else SimpleNamespace(
                num_accept_tokens=torch.tensor(accept), num_front_tokens=front
            )
        ),
    )


def _verify_batch(tokens, positions, requests):
    return _spec_batch(_ForwardMode.TARGET_VERIFY, tokens, positions, requests)


class _LogitsProcessor(nn.Module):
    def forward(self, input_ids, hidden_states, lm_head, logits_metadata):
        return lm_head(hidden_states)


class _MtpModel(nn.Module):
    hidden_size = HIDDEN

    def __init__(self):
        super().__init__()
        torch.manual_seed(1)
        self.embed = nn.Embedding(128, HIDDEN)
        self.lm_head = nn.Linear(HIDDEN, 32, bias=False)
        self.logits_processor = _LogitsProcessor()
        self.hidden_rows = None

    def forward(self, input_ids, positions, forward_batch):
        hidden_states = self.embed(input_ids) * 3
        if self.hidden_rows is not None:
            hidden_states = hidden_states[: self.hidden_rows]
        self.lm_head_input = hidden_states
        return self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch
        )


def _run_mtp(model, batch):
    with torch.no_grad():
        return model(batch.input_ids, batch.positions, batch)


def _mtp_load(directory):
    return _load(os.path.join(directory, route_trace.MTP_SUBDIR))


def _routing(model, layer_inputs, tokens):
    expected = []
    for layer, layer_input in enumerate(layer_inputs):
        mixed = layer_input.view(tokens, HC, HIDDEN).mean(-2)
        logits = model.layers[layer].mlp.gate(mixed)[0]
        weights, ids = torch.topk(logits.softmax(-1), TOP_K)
        expected.append((ids, weights / weights.sum(-1, keepdim=True)))
    return expected


def _assert_unhooked(module):
    for child in module.modules():
        assert not child._forward_hooks
        assert not child._forward_pre_hooks
        assert "mix" not in child.__dict__


def test_speculative_trace_records_every_token_of_target_verify_forwards():
    model = _LanguageModel()
    with tempfile.TemporaryDirectory() as directory:
        trace = route_trace.install_moe_route_trace(
            model, directory, max_tokens=8, speculative=True,
            speculative_settings=SETTINGS,
        )
        _run(model, _spec_batch(_ForwardMode.DECODE, [1], [0], [0]))
        final_hidden = _run(
            model, _verify_batch([3, 4, 5, 6, 7, 8], [10, 11, 12, 20, 21, 22], [5, 9])
        )
        routing = _routing(model, model.layer_inputs, 6)
        _run(model, _verify_batch([9, 10, 11, 12], [13, 14, 23, 24], [5, 9]))
        manifest, shards = _load(directory)

    assert manifest["complete"] and manifest["mode"] == "speculative"
    assert (manifest["tokens"], manifest["forwards"]) == (8, 2)
    assert {name: manifest[name] for name in SETTINGS} == SETTINGS
    assert "closed_by" not in manifest
    assert manifest["layer_fields"] == {
        "topk_ids": "torch.int32",
        "topk_weights": "torch.float32",
    }
    assert manifest["forward_fields"] == {"final_hidden": "torch.bfloat16"}
    assert manifest["token_fields"] == [
        "input_ids",
        "positions",
        "req_pool_indices",
        "forward_index",
        "sequence",
    ]
    (shard,) = shards
    assert set(shard) == {
        "topk_ids",
        "topk_weights",
        "final_hidden",
        "input_ids",
        "positions",
        "req_pool_indices",
        "forward_index",
        "sequence",
    }
    assert shard["topk_ids"].dtype == torch.int32
    assert shard["topk_ids"].shape == (8, LAYERS, TOP_K)
    for layer, (ids, weights) in enumerate(routing):
        assert shard["topk_ids"][:6, layer].tolist() == ids.tolist()
        torch.testing.assert_close(shard["topk_weights"][:6, layer], weights)
    torch.testing.assert_close(
        shard["final_hidden"][:6].float(), final_hidden.to(torch.bfloat16).float()
    )
    assert shard["final_hidden"].shape == (8, HIDDEN)
    assert shard["input_ids"].tolist() == [3, 4, 5, 6, 7, 8, 9, 10]
    assert shard["positions"].tolist() == [10, 11, 12, 20, 21, 22, 13, 14]
    assert shard["req_pool_indices"].tolist() == [5, 5, 5, 9, 9, 9, 5, 5]
    assert shard["forward_index"].tolist() == [0] * 6 + [1] * 2
    sequence = shard["sequence"].tolist()
    assert len(set(sequence[:6])) == 1 and len(set(sequence[6:])) == 1
    assert sequence[6] > sequence[0]
    assert trace.writer.closed
    _assert_unhooked(model)


def test_mtp_trace_keeps_seed_rows_and_marks_accepted_chains():
    target = _LanguageModel()
    mtp = _MtpModel()
    with tempfile.TemporaryDirectory() as directory:
        target_trace = route_trace.install_moe_route_trace(
            target, directory, max_tokens=8, speculative=True
        )
        route_trace.install_mtp_hidden_trace(
            mtp, directory, max_tokens=8, speculative_settings=SETTINGS
        )
        _run_mtp(
            mtp,
            _spec_batch(
                _ForwardMode.EXTEND,
                [1, 2, 3, 4, 5, 6],
                [0, 1, 0, 1, 2, 3],
                [2, 7],
                extend_seq_lens=[2, 4],
            ),
        )
        extend_hidden = mtp.lm_head_input
        _run_mtp(mtp, _spec_batch(_ForwardMode.IDLE, [0], [0], [0]))
        _run_mtp(mtp, _spec_batch(_ForwardMode.TARGET_VERIFY, [1, 2], [0, 1], [0]))
        _run_mtp(
            mtp,
            _spec_batch(
                _ForwardMode.DRAFT_EXTEND_V2,
                [7, 8, 9, 10],
                [2, 3, 4, 5],
                [2, 7],
                accept=[2, 1],
            ),
        )
        draft_extend_hidden = mtp.lm_head_input
        _run_mtp(
            mtp,
            _spec_batch(
                _ForwardMode.DRAFT_EXTEND_V2,
                [21, 22, 23, 24, 25, 26],
                [4, 5, 6, 5, 6, 7],
                [2, 7],
                accept=[1, 2],
                front=1,
            ),
        )
        widened_hidden = mtp.lm_head_input
        _run_mtp(mtp, _spec_batch(_ForwardMode.DECODE, [11, 12], [4, 6], [2, 7]))
        decode_hidden = mtp.lm_head_input
        target_trace.writer.close()
        manifest, shards = _mtp_load(directory)

    assert manifest["format"] == route_trace.MTP_TRACE_FORMAT
    assert manifest["mode"] == "speculative_mtp" and manifest["complete"]
    assert manifest["closed_by"] == "target_complete"
    assert {name: manifest[name] for name in SETTINGS} == SETTINGS
    assert (manifest["tokens"], manifest["forwards"]) == (14, 4)
    assert manifest["max_tokens"] == 32 and manifest["layer_ids"] == []
    assert manifest["forward_fields"] == {"hidden": "torch.bfloat16"}
    assert manifest["forward_modes"] == {"EXTEND": 1, "DECODE": 2, "DRAFT_EXTEND_V2": 7}
    (shard,) = shards
    expected = torch.cat(
        [extend_hidden[[1, 5]], draft_extend_hidden, widened_hidden, decode_hidden]
    )
    torch.testing.assert_close(
        shard["hidden"].float(), expected.to(torch.bfloat16).float()
    )
    assert shard["input_ids"].tolist() == [2, 6, 7, 8, 9, 10] + list(range(21, 27)) + [11, 12]
    assert shard["positions"].tolist() == [1, 3, 2, 3, 4, 5, 4, 5, 6, 5, 6, 7, 4, 6]
    assert shard["req_pool_indices"].tolist() == [2, 7, 2, 2, 7, 7, 2, 2, 2, 7, 7, 7, 2, 7]
    assert shard["forward_index"].tolist() == [0] * 2 + [1] * 4 + [2] * 6 + [3] * 2
    assert shard["forward_mode"].tolist() == [1] * 2 + [7] * 10 + [2] * 2
    assert shard["accept_len"].dtype == torch.int32
    assert shard["accept_len"].tolist() == [-1, -1, 2, 2, 1, 1, 1, 1, 1, 2, 2, 2, -1, -1]
    assert shard["selected"].dtype == torch.int8
    assert shard["selected"].tolist() == [1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1]


def test_draft_extend_without_accept_counts_writes_unknown_markers():
    mtp = _MtpModel()
    with tempfile.TemporaryDirectory() as directory:
        trace = route_trace.install_mtp_hidden_trace(mtp, directory, max_tokens=8)
        _run_mtp(
            mtp, _spec_batch(_ForwardMode.DRAFT_EXTEND_V2, [1, 2, 3, 4], [0, 1, 0, 1], [0, 1])
        )
        trace.writer.close()
        manifest, (shard,) = _mtp_load(directory)

    assert manifest["closed_by"] is None
    assert shard["accept_len"].tolist() == [-1] * 4
    assert shard["selected"].tolist() == [-1] * 4


def test_draft_prefill_prompt_rows_do_not_exhaust_the_mtp_cap():
    mtp = _MtpModel()
    with tempfile.TemporaryDirectory() as directory:
        trace = route_trace.install_mtp_hidden_trace(mtp, directory, max_tokens=1)
        _run_mtp(
            mtp,
            _spec_batch(
                _ForwardMode.EXTEND,
                list(range(40)),
                list(range(40)),
                [0, 1],
                extend_seq_lens=[30, 10],
            ),
        )
        assert not trace.writer.closed and trace.writer.tokens == 2
        _run_mtp(
            mtp,
            _spec_batch(_ForwardMode.EXTEND, list(range(100, 106)), list(range(6)), [0, 1]),
        )
        _run_mtp(mtp, _spec_batch(_ForwardMode.DECODE, [7, 8], [6, 6], [0, 1]))
        manifest, (shard,) = _mtp_load(directory)

    assert trace.writer.closed and manifest["complete"]
    assert manifest["closed_by"] == "cap" and manifest["tokens"] == 4
    assert shard["input_ids"].tolist() == [29, 39, 102, 105]
    assert shard["req_pool_indices"].tolist() == [0, 1, 0, 1]
    _assert_unhooked(mtp)


def test_mtp_forward_with_mismatched_hidden_rows_is_dropped_once_logged():
    mtp = _MtpModel()
    with tempfile.TemporaryDirectory() as directory:
        trace = route_trace.install_mtp_hidden_trace(mtp, directory, max_tokens=8)
        mtp.hidden_rows = 1
        with patch.object(route_trace.logger, "warning") as warning:
            for _ in range(2):
                logits = _run_mtp(
                    mtp, _spec_batch(_ForwardMode.DECODE, [1, 2], [0, 0], [0, 1])
                )
            _run_mtp(mtp, _spec_batch(_ForwardMode.EXTEND, [1, 2, 3], [0, 1, 0], [0, 1]))
        assert warning.call_count == 1
        assert (trace.writer.tokens, trace.writer.forwards) == (0, 0)
        assert not trace.writer.closed and not trace.writer.recording
        mtp.hidden_rows = None
        _run_mtp(mtp, _spec_batch(_ForwardMode.DECODE, [3, 4], [1, 1], [0, 1]))
        trace.writer.close()
        _, (shard,) = _mtp_load(directory)

    assert logits.shape == (1, 32)
    assert shard["input_ids"].tolist() == [3, 4]


def test_sequence_orders_interleaved_target_and_draft_forwards():
    target, mtp = _LanguageModel(), _MtpModel()
    with tempfile.TemporaryDirectory() as directory:
        target_trace = route_trace.install_moe_route_trace(
            target, directory, max_tokens=8, speculative=True
        )
        route_trace.install_mtp_hidden_trace(mtp, directory, max_tokens=8)
        _run_mtp(mtp, _spec_batch(_ForwardMode.EXTEND, [1, 2], [0, 1], [0]))
        _run(target, _verify_batch([3, 4, 5], [2, 3, 4], [0]))
        _run_mtp(mtp, _spec_batch(_ForwardMode.DRAFT_EXTEND_V2, [6, 7, 8], [2, 3, 4], [0], accept=[2]))
        _run(target, _verify_batch([9, 10, 11], [4, 5, 6], [0]))
        _run_mtp(mtp, _spec_batch(_ForwardMode.DECODE, [12], [5], [0]))
        target_trace.writer.close()
        _, (target_shard,) = _load(directory)
        _, (mtp_shard,) = _mtp_load(directory)

    def per_forward(shard):
        values = {}
        for forward, sequence in zip(
            shard["forward_index"].tolist(), shard["sequence"].tolist()
        ):
            values.setdefault(forward, set()).add(sequence)
        assert all(len(group) == 1 for group in values.values())
        return [group.pop() for _, group in sorted(values.items())]

    draft = per_forward(mtp_shard)
    verify = per_forward(target_shard)
    assert len(draft) == 3 and len(verify) == 2
    order = [draft[0], verify[0], draft[1], verify[1], draft[2]]
    assert all(earlier < later for earlier, later in zip(order, order[1:]))


def test_speculative_writers_share_the_trace_directory_in_either_order():
    for mtp_first in (False, True):
        target, mtp = _LanguageModel(), _MtpModel()
        with tempfile.TemporaryDirectory() as directory:
            installs = [
                lambda: route_trace.install_moe_route_trace(
                    target, directory, max_tokens=2, speculative=True
                ),
                lambda: route_trace.install_mtp_hidden_trace(
                    mtp, directory, max_tokens=2
                ),
            ]
            for install in reversed(installs) if mtp_first else installs:
                install()
            assert sorted(os.listdir(directory)) == [
                route_trace.MANIFEST_NAME,
                route_trace.MTP_SUBDIR,
            ]
            _run(target, _verify_batch([1, 2], [0, 1], [0]))
            _assert_unhooked(mtp)


def test_speculative_target_writer_still_refuses_a_stale_trace():
    with tempfile.TemporaryDirectory() as directory:
        open(os.path.join(directory, "shard_00000.pt"), "wb").close()
        with pytest.raises(FileExistsError, match="not empty"):
            route_trace.install_moe_route_trace(
                _LanguageModel(), directory, max_tokens=1, speculative=True
            )


def test_target_completion_closes_the_mtp_trace():
    target, mtp = _LanguageModel(), _MtpModel()
    with tempfile.TemporaryDirectory() as directory:
        mtp_trace = route_trace.install_mtp_hidden_trace(mtp, directory, max_tokens=3)
        target_trace = route_trace.install_moe_route_trace(
            target, directory, max_tokens=3, speculative=True
        )
        _run_mtp(mtp, _spec_batch(_ForwardMode.DRAFT_EXTEND_V2, [1, 2], [0, 1], [0]))
        _run(target, _verify_batch([1, 2, 3], [1, 2, 3], [0]))
        _run_mtp(mtp, _spec_batch(_ForwardMode.DRAFT_EXTEND_V2, [3, 4], [2, 3], [0]))
        manifest, shards = _mtp_load(directory)

    assert target_trace.writer.closed and mtp_trace.writer.closed
    assert manifest["complete"] and manifest["error"] is None
    assert manifest["closed_by"] == "target_complete"
    assert manifest["tokens"] == 2
    assert shards[0]["input_ids"].tolist() == [1, 2]
    _assert_unhooked(mtp)


def test_target_abandon_closes_the_mtp_trace_as_target_abandoned():
    target, mtp = _LanguageModel(), _MtpModel()
    with tempfile.TemporaryDirectory() as directory:
        target_trace = route_trace.install_moe_route_trace(
            target, directory, max_tokens=4, shard_tokens=1, speculative=True
        )
        mtp_trace = route_trace.install_mtp_hidden_trace(mtp, directory, max_tokens=4)
        with patch.object(
            route_trace.torch, "save", side_effect=OSError(28, "No space left on device")
        ):
            _run(target, _verify_batch([1, 2], [0, 1], [0]))
        manifest, _ = _mtp_load(directory)
        target_manifest, _ = _load(directory)

    assert "No space left" in target_manifest["error"]
    assert mtp_trace.writer.closed and manifest["complete"]
    assert manifest["closed_by"] == "target_abandoned" and manifest["error"] is None
    assert target_trace.writer.closed
    _assert_unhooked(mtp)


def test_reinstall_on_the_same_directory_gets_a_fresh_link():
    with tempfile.TemporaryDirectory() as parent:
        directory = os.path.join(parent, "trace")
        first_mtp = _MtpModel()
        first_target = _LanguageModel()
        route_trace.install_moe_route_trace(
            first_target, directory, max_tokens=1, speculative=True
        )
        first = route_trace.install_mtp_hidden_trace(first_mtp, directory, max_tokens=1)
        _run(first_target, _verify_batch([1], [0], [0]))
        assert first.writer.closed
        shutil.rmtree(directory)

        target, mtp = _LanguageModel(), _MtpModel()
        second_target = route_trace.install_moe_route_trace(
            target, directory, max_tokens=1, speculative=True
        )
        second = route_trace.install_mtp_hidden_trace(mtp, directory, max_tokens=1)
        _run_mtp(mtp, _spec_batch(_ForwardMode.DECODE, [5], [3], [0]))
        assert not second.writer.closed and second.writer.tokens == 1
        _run(target, _verify_batch([2], [1], [0]))

    assert second_target.writer.closed and second.writer.closed
    assert second.writer.closed_by == "target_complete"


def test_mtp_write_failure_abandons_only_the_mtp_trace():
    target, mtp = _LanguageModel(), _MtpModel()
    with tempfile.TemporaryDirectory() as directory:
        target_trace = route_trace.install_moe_route_trace(
            target, directory, max_tokens=4, speculative=True
        )
        mtp_trace = route_trace.install_mtp_hidden_trace(
            mtp, directory, max_tokens=4, shard_tokens=1
        )
        with patch.object(
            route_trace.torch, "save", side_effect=OSError(28, "No space left on device")
        ):
            logits = _run_mtp(
                mtp, _spec_batch(_ForwardMode.DRAFT_EXTEND_V2, [1, 2], [0, 1], [0])
            )
        _run_mtp(mtp, _spec_batch(_ForwardMode.DRAFT_EXTEND_V2, [3, 4], [2, 3], [0]))
        assert not target_trace.writer.closed
        _run(target, _verify_batch([1, 2, 3, 4], [1, 2, 3, 4], [0]))
        mtp_manifest, _ = _mtp_load(directory)
        target_manifest, target_shards = _load(directory)

    assert logits.shape == (2, 32)
    assert mtp_trace.writer.closed and mtp_trace.writer.tokens == 2
    assert not mtp_manifest["complete"] and "No space left" in mtp_manifest["error"]
    assert mtp_manifest["closed_by"] == "abandoned"
    assert target_manifest["complete"] and target_manifest["error"] is None
    assert target_shards[0]["input_ids"].tolist() == [1, 2, 3, 4]
    _assert_unhooked(mtp)
    _assert_unhooked(target)


def test_speculative_flag_off_installs_no_mtp_trace():
    mtp = _MtpModel()
    with tempfile.TemporaryDirectory() as directory:
        with patch.dict(
            os.environ,
            {
                "SGLANG_MOE_ROUTE_TRACE_DIR": directory,
                "SGLANG_MOE_ROUTE_TRACE_SPECULATIVE": "0",
            },
        ):
            assert route_trace.maybe_install_mtp_hidden_trace(mtp) is None
        assert os.listdir(directory) == []
    _assert_unhooked(mtp)


def test_speculative_flag_installs_both_traces_from_the_environment():
    target, mtp = _LanguageModel(), _MtpModel()
    with tempfile.TemporaryDirectory() as directory:
        with patch.dict(
            os.environ,
            {
                "SGLANG_MOE_ROUTE_TRACE_DIR": directory,
                "SGLANG_MOE_ROUTE_TRACE_SPECULATIVE": "1",
                "SGLANG_MOE_ROUTE_TRACE_MAX_TOKENS": "2",
            },
        ):
            target_trace = route_trace.maybe_install_moe_route_trace(target)
            mtp_trace = route_trace.maybe_install_mtp_hidden_trace(mtp)
        target_trace.writer.close()
        target_manifest, _ = _load(directory)
        mtp_manifest, _ = _mtp_load(directory)

    assert target_manifest["mode"] == "speculative"
    assert mtp_manifest["max_tokens"] == 8 and mtp_trace.writer.closed
    for manifest in (target_manifest, mtp_manifest):
        assert set(SETTINGS) <= set(manifest)


def test_bytes_per_token_estimate_for_qwen38_flash():
    estimate = route_trace.estimate_bytes_per_token(48, 2560, 4, 512, 10)
    assert estimate == 48 * (5120 + 20480 + 1024 + 20 + 40 + 5120 + 5120) + (
        20480 + 5120 + 40
    )
