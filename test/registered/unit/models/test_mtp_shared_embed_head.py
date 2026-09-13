"""CPU tests: NEXTN drafts bind the target's embedding and lm_head instead of allocating."""

import unittest
from contextlib import ExitStack, nullcontext
from dataclasses import dataclass
from itertools import chain
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.models import qwen3_5, qwen3_5_mtp, qwen4_exp_mtp
from sglang.srt.models.qwen3_5_mtp import Qwen3_5ForCausalLMMTP
from sglang.srt.models.qwen4_exp_mtp import Qwen4ExpForCausalLMMTP
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

VOCAB = 64
HIDDEN = 4


def _tensors(module):
    return [t for _, t in chain(module.named_parameters(), module.named_buffers())]


def _parallel():
    return SimpleNamespace(
        tp_rank=0, tp_size=1, attn_tp_rank=0, attn_tp_size=1, enable_dp_lm_head=False
    )


def _target(rows=VOCAB, dtype=torch.float32):
    return nn.Parameter(torch.randn(rows, HIDDEN, dtype=dtype), requires_grad=False)


class BuildWithTargetWeightTests(unittest.TestCase):
    def test_offered_weight_is_bound_without_meta_or_copies(self):
        from sglang.srt.speculative.draft_shared_weights import build_with_target_weight

        for build in (
            lambda: VocabParallelEmbedding(VOCAB, HIDDEN, enable_tp=False),
            lambda: ParallelLMHead(VOCAB, HIDDEN, enable_tp=False),
        ):
            target = _target()
            module = build_with_target_weight(build, target, "vocab")
            with self.subTest(module=type(module).__name__):
                self.assertIs(module.weight, target)
                self.assertTrue(module.shares_target_weight)
                self.assertEqual(_tensors(module), [target])

    def test_not_shared_paths_allocate_real_weights(self):
        from sglang.srt.speculative.draft_shared_weights import build_with_target_weight

        build = lambda: VocabParallelEmbedding(VOCAB, HIDDEN, enable_tp=False)
        for label, target in (
            ("no offer", None),
            ("shape mismatch", _target(rows=VOCAB // 2)),
            ("dtype mismatch", _target(dtype=torch.float64)),
        ):
            module = build_with_target_weight(build, target, "embed_tokens")
            with self.subTest(label):
                self.assertFalse(module.weight.is_meta)
                self.assertIsNot(module.weight, target)
                self.assertEqual(tuple(module.weight.shape), (VOCAB, HIDDEN))
                self.assertFalse(getattr(module, "shares_target_weight", False))

    def test_declined_offer_is_logged_with_shapes_and_dtypes(self):
        from sglang.srt.speculative.draft_shared_weights import build_with_target_weight

        with self.assertLogs(
            "sglang.srt.speculative.draft_shared_weights", level="INFO"
        ) as logs:
            build_with_target_weight(
                lambda: VocabParallelEmbedding(VOCAB, HIDDEN, enable_tp=False),
                _target(rows=VOCAB // 2, dtype=torch.float64),
                "embed_tokens",
            )
        message = "\n".join(logs.output)
        for fragment in ("embed_tokens", "(64, 4)", "(32, 4)", "float32", "float64"):
            self.assertIn(fragment, message)

    def test_scope_offers_tensors_only_inside(self):
        from sglang.srt.speculative.draft_shared_weights import (
            draft_shares_target_embed_and_head,
            shared_target_embed,
            shared_target_head,
        )

        embed, head = torch.zeros(1), torch.ones(1)
        with draft_shares_target_embed_and_head(embed, head):
            self.assertIs(shared_target_embed(), embed)
            self.assertIs(shared_target_head(), head)
        self.assertIsNone(shared_target_embed())
        self.assertIsNone(shared_target_head())

    def test_vocab_weight_check_names_the_module_and_ignores_other_meta(self):
        from sglang.srt.speculative.draft_shared_weights import (
            require_vocab_weights_materialized,
        )

        model = nn.Module()
        model.model = nn.Module()
        model.model.embed_tokens = nn.Embedding(4, 3)
        model.layer = nn.Linear(3, 3, device="meta")
        require_vocab_weights_materialized(model)
        model.lm_head = nn.Linear(3, 4, bias=False, device="meta")
        with self.assertRaisesRegex(RuntimeError, "draft lm_head.weight is still on meta"):
            require_vocab_weights_materialized(model)


class RealMtpConstructionTests(unittest.TestCase):
    pp_group = SimpleNamespace(
        is_first_rank=True, is_last_rank=True, rank_in_group=0, world_size=1
    )

    def construction_patches(self, stack):
        for owner, name, value in (
            (qwen3_5_mtp, "_mtp_quant_config", lambda q: None),
            (qwen3_5_mtp, "get_parallel", _parallel),
            (qwen3_5_mtp, "get_pp_group", lambda: self.pp_group),
            (qwen3_5_mtp, "LogitsProcessor", lambda c: nn.Identity()),
            (qwen3_5, "get_pp_group", lambda: self.pp_group),
            (qwen3_5, "make_layers", lambda *a, **k: (nn.ModuleList(), 0, 0)),
            (qwen3_5, "get_stream", lambda name: None),
            (qwen3_5, "is_dp_attention_enabled", lambda: False),
            (qwen4_exp_mtp, "Qwen4ExpModel", lambda *a, **k: nn.Module()),
            (qwen4_exp_mtp, "_mtp_quant_config", lambda q: None),
            (qwen4_exp_mtp, "get_parallel", _parallel),
            (qwen4_exp_mtp, "get_pp_group", lambda: self.pp_group),
            (qwen4_exp_mtp, "LogitsProcessor", lambda c: nn.Identity()),
            (qwen4_exp_mtp, "maybe_install_mtp_hidden_trace", lambda m: None),
        ):
            stack.enter_context(patch.object(owner, name, value))
        stack.enter_context(
            patch("sglang.srt.layers.vocab_parallel_embedding.get_parallel", _parallel)
        )

    def config(self, tie_word_embeddings=False):
        return SimpleNamespace(
            vocab_size=VOCAB,
            hidden_size=HIDDEN,
            rms_norm_eps=1e-6,
            hc_count=1,
            tie_word_embeddings=tie_word_embeddings,
            num_hidden_layers=1,
            full_attention_interval=1,
            model_type="qwen3_5_text",
        )

    def build(self, model_class, offer, tie_word_embeddings=False):
        from sglang.srt.speculative.draft_shared_weights import (
            draft_shares_target_embed_and_head,
        )

        embed, head = _target(), _target()
        with ExitStack() as stack:
            self.construction_patches(stack)
            if offer:
                stack.enter_context(draft_shares_target_embed_and_head(embed, head))
            model = model_class(self.config(tie_word_embeddings))
        return model, embed, head

    def test_qwen3_5_mtp_binds_offered_embed_and_head_at_the_real_call_sites(self):
        model, embed, head = self.build(Qwen3_5ForCausalLMMTP, offer=True)
        self.assertIs(model.model.embed_tokens.weight, embed)
        self.assertIs(model.lm_head.weight, head)
        self.assertFalse(any(t.is_meta for t in _tensors(model)))

    def test_qwen3_5_mtp_without_offer_allocates_its_own(self):
        model, embed, head = self.build(Qwen3_5ForCausalLMMTP, offer=False)
        self.assertIsNot(model.model.embed_tokens.weight, embed)
        self.assertIsNot(model.lm_head.weight, head)
        self.assertFalse(any(t.is_meta for t in _tensors(model)))

    def test_qwen4_mtp_binds_offered_untied_head(self):
        model, _, head = self.build(Qwen4ExpForCausalLMMTP, offer=True)
        self.assertIs(model.lm_head.weight, head)
        self.assertFalse(any(t.is_meta for t in _tensors(model)))

    def test_qwen4_model_binds_offered_embed_at_the_real_call_site(self):
        from sglang.srt.models import qwen4_exp
        from sglang.srt.speculative.draft_shared_weights import (
            draft_shares_target_embed_and_head,
        )

        config = self.config()
        config.hc_count = 2
        config.hc_lowrank = 2
        config.ple_layer_ids = []
        config.eos_token_id = 0
        for offer in (True, False):
            embed = _target()
            with ExitStack() as stack:
                self.construction_patches(stack)
                stack.enter_context(
                    patch.object(qwen4_exp, "is_dp_attention_enabled", lambda: False)
                )
                stack.enter_context(
                    patch.object(qwen4_exp, "GatedResidual", lambda *a, **k: nn.Module())
                )
                if offer:
                    stack.enter_context(draft_shares_target_embed_and_head(embed, None))
                model = qwen4_exp.Qwen4ExpModel(config, None, "mtp", is_nextn=True)
            with self.subTest(offer=offer):
                self.assertEqual(model.embed_tokens.weight is embed, offer)
                self.assertFalse(model.embed_tokens.weight.is_meta)

    def test_qwen4_mtp_tied_or_unoffered_head_keeps_a_real_weight(self):
        for tie, offer in ((True, True), (False, False)):
            model, _, head = self.build(
                Qwen4ExpForCausalLMMTP, offer=offer, tie_word_embeddings=tie
            )
            with self.subTest(tie=tie, offer=offer):
                self.assertIsNot(model.lm_head.weight, head)
                self.assertFalse(model.lm_head.weight.is_meta)
                self.assertEqual(tuple(model.lm_head.weight.shape), (VOCAB, HIDDEN))


class _StopAfterDraftBuild(Exception):
    pass


@dataclass
class _ParallelState:
    pp_rank: int = 0
    pp_size: int = 1


class EagleDraftWorkerScopeTests(unittest.TestCase):
    def offered_during_draft_build(self, algorithm):
        from sglang.srt.speculative import eagle_worker_v2
        from sglang.srt.speculative.draft_shared_weights import (
            shared_target_embed,
            shared_target_head,
        )

        embed, head = _target(), _target()
        target_worker = SimpleNamespace(
            model_runner=SimpleNamespace(
                model=SimpleNamespace(get_embed_and_head=lambda: (embed, head)),
                model_config=SimpleNamespace(context_len=16),
            ),
            random_seed=0,
        )
        offered = []

        def build_draft(**kwargs):
            offered.append((shared_target_embed(), shared_target_head()))
            raise _StopAfterDraftBuild

        spec = SimpleNamespace(
            speculative_eagle_topk=1,
            speculative_use_rejection_sampling=False,
            speculative_num_steps=3,
            speculative_num_draft_tokens=4,
            speculative_algorithm=algorithm,
        )
        with ExitStack() as stack:
            for name, value in (
                ("get_device", lambda: SimpleNamespace(device="cpu")),
                ("get_spec", lambda: spec),
                ("get_parallel", lambda: SimpleNamespace(enable_dp_attention=False)),
                ("draft_pp_context", nullcontext),
                ("speculative_moe_backend_context", nullcontext),
                ("speculative_moe_a2a_backend_context", nullcontext),
                ("draft_model_build_scope", nullcontext),
                ("TpModelWorker", build_draft),
            ):
                stack.enter_context(patch.object(eagle_worker_v2, name, value))
            stack.enter_context(
                patch.object(
                    eagle_worker_v2.EagleDraftWorker,
                    "_rebuild_topk1_chain_buffers",
                    lambda self: None,
                )
            )
            with self.assertRaises(_StopAfterDraftBuild):
                eagle_worker_v2.EagleDraftWorker(
                    server_args=None,
                    gpu_id=0,
                    ps=_ParallelState(),
                    nccl_port=0,
                    target_worker=target_worker,
                )
        self.assertIsNone(shared_target_embed())
        self.assertIsNone(shared_target_head())
        return offered, embed, head

    def test_nextn_draft_build_is_offered_the_target_tensors(self):
        offered, embed, head = self.offered_during_draft_build("NEXTN")
        self.assertEqual(len(offered), 1)
        self.assertIs(offered[0][0], embed)
        self.assertIs(offered[0][1], head)

    def test_eagle3_draft_build_is_offered_nothing(self):
        offered, _, _ = self.offered_during_draft_build("EAGLE3")
        self.assertEqual(offered, [(None, None)])


class _DraftModel(nn.Module):
    def __init__(self, leave_meta, offloaded_layer=False):
        super().__init__()
        self.leave_meta = leave_meta
        self.embed_tokens = nn.Embedding(VOCAB, HIDDEN, device="meta")
        if offloaded_layer:
            self.layer = nn.Linear(HIDDEN, HIDDEN, device="meta")

    def set_embed_and_head(self, embed, head):
        if not self.leave_meta:
            del self.embed_tokens.weight
            self.embed_tokens.weight = embed


class InitLmHeadMetaCheckTests(unittest.TestCase):
    def run_init_lm_head(self, leave_meta, offloaded_layer=False):
        from sglang.srt.speculative import eagle_worker_v2
        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

        embed = _target()
        worker = object.__new__(eagle_worker_v2.EagleDraftWorker)
        worker.target_worker = SimpleNamespace(
            model_runner=SimpleNamespace(
                model=SimpleNamespace(
                    get_embed_and_head=lambda: (embed, None), lm_head=None
                )
            )
        )
        worker.draft_runner = SimpleNamespace(
            model=_DraftModel(leave_meta, offloaded_layer)
        )
        worker.speculative_algorithm = SpeculativeAlgorithm.from_string("NEXTN")
        worker.hot_token_id = None
        worker.init_lm_head()
        return worker.draft_runner.model, embed

    def test_init_lm_head_raises_when_a_draft_vocab_weight_stays_on_meta(self):
        with self.assertRaisesRegex(
            RuntimeError, "draft embed_tokens.weight is still on meta"
        ):
            self.run_init_lm_head(leave_meta=True)

    def test_init_lm_head_accepts_a_fully_shared_draft(self):
        model, embed = self.run_init_lm_head(leave_meta=False)
        self.assertIs(model.embed_tokens.weight, embed)

    def test_init_lm_head_ignores_a_meta_offloaded_layer_weight(self):
        model, embed = self.run_init_lm_head(leave_meta=False, offloaded_layer=True)
        self.assertIs(model.embed_tokens.weight, embed)
        self.assertTrue(model.layer.weight.is_meta)


class SharedWeightLoadTests(unittest.TestCase):
    def test_post_load_staging_keeps_the_bound_target_tensor(self):
        from sglang.srt.model_loader.post_load import stage_module_for_post_load
        from sglang.srt.speculative.draft_shared_weights import build_with_target_weight

        for build in (
            lambda: VocabParallelEmbedding(VOCAB, HIDDEN, enable_tp=False),
            lambda: ParallelLMHead(VOCAB, HIDDEN, enable_tp=False),
        ):
            target = _target()
            pointer = target.data_ptr()
            module = build_with_target_weight(build, target, "vocab")
            with stage_module_for_post_load(module, torch.device("cpu")):
                module.quant_method.process_weights_after_loading(module)
            with self.subTest(module=type(module).__name__):
                self.assertIs(module.weight, target)
                self.assertEqual(module.weight.data_ptr(), pointer)

    def test_checkpoint_load_leaves_a_shared_target_embedding_untouched(self):
        model = Qwen3_5ForCausalLMMTP.__new__(Qwen3_5ForCausalLMMTP)
        nn.Module.__init__(model)
        model.model = nn.Module()
        embed_tokens = nn.Embedding(4, 3)
        target = nn.Parameter(torch.full((4, 3), 7.0), requires_grad=False)
        del embed_tokens.weight
        embed_tokens.weight = target
        embed_tokens.shares_target_weight = True
        model.model.embed_tokens = embed_tokens
        model.config = SimpleNamespace(num_experts=None)
        model.quant_config = None

        loaded = model.load_weights([("model.embed_tokens.weight", torch.zeros(4, 3))])

        self.assertEqual(loaded, {"model.embed_tokens.weight"})
        torch.testing.assert_close(target, torch.full((4, 3), 7.0))


if __name__ == "__main__":
    unittest.main()
