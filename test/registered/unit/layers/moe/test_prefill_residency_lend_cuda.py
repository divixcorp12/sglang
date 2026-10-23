"""Lent hot-cache slots on the GPU: the pointer-table copy kernel, and lend -> overwrite -> reload byte for byte (CUDA)."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.moe import expert_stream
from sglang.srt.layers.moe.expert_stream import ExpertPinnedHostCache, ExpertStreamer
from sglang.srt.layers.moe.prefill_residency import Exl3LendingLayerMajorResidency, LentPrefillResidency, LentRow
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=15, stage="base-a", runner_config="1-gpu-small")

EXPERTS = 32
ROW_BYTES = 3000  # not a multiple of the kernel's 1024-byte block


class _Layer(torch.nn.Module):
    pass


class _Manager:
    prefill_lend_active = False

    def __init__(self):
        self.caches = {}
        self.gpu_residency = None

    def quiesce_for_prefill_lend(self):
        torch.cuda.synchronize()

    def set_prefill_lend(self, active):
        self.prefill_lend_active = active


def _layer(seed, hot_experts, capacity=8):
    from sglang.srt.layers.moe.expert_hot_cache import ExpertHotCache

    generator = torch.Generator().manual_seed(seed)
    layer = _Layer()
    layer.host_rows = torch.nn.Parameter(
        torch.randint(0, 256, (EXPERTS, ROW_BYTES), dtype=torch.uint8, generator=generator).pin_memory(),
        requires_grad=False,
    )
    streamer = ExpertStreamer(layer, ("host_rows",))
    cache = ExpertHotCache(streamer, capacity=capacity)
    cache.reassign(hot_experts)
    torch.cuda.synchronize()
    return layer, streamer, cache


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TestLendOnTheGpu(unittest.TestCase):
    def setUp(self):
        expert_stream._STAGING.clear()
        self.manager = _Manager()
        self.early, early_streamer, early_cache = _layer(1, [0, 1, 2, 3])
        self.manager.caches[0] = early_cache
        self.donors = {}
        for layer_id, seed in ((21, 2), (22, 3)):
            layer, streamer, cache = _layer(seed, list(range(8, 16)) if layer_id == 21 else list(range(16, 24)))
            ExpertPinnedHostCache(streamer, EXPERTS)
            self.manager.caches[layer_id] = cache
            self.donors[layer_id] = layer
        self.early_streamer = early_streamer
        self.runner = SimpleNamespace(expert_hot_cache_manager=self.manager)

    def tearDown(self):
        expert_stream._STAGING.clear()

    def _state(self):
        return {
            layer_id: (
                cache.tensors["host_rows"].clone(),
                cache.expert_to_slot.clone(),
                cache.slot_state.clone(),
                cache.slot_generations.clone(),
                list(cache.slot_to_expert),
            )
            for layer_id, cache in self.manager.caches.items()
            if layer_id != 0
        }

    def test_lend_overwrite_reload_leaves_the_slots_byte_identical(self):
        before = self._state()
        residency = Exl3LendingLayerMajorResidency(self.runner, rows=12)
        residency.begin(range(0, 2))
        buffer = residency.buffer
        self.assertEqual(buffer.rows, 12)
        self.assertTrue(self.manager.prefill_lend_active)
        # Layer 0's chunks route to misses: they are adopted into 12 donor slots, overwriting them.
        residency.make_resident(0)
        ids = torch.arange(4, 16, device="cuda")
        misses = [e for e in range(4, 16)]
        row_of_source, rows = self.early_streamer._gather_experts(ids)
        torch.cuda.synchronize()
        self.assertEqual(buffer.written, len(misses))
        mid = self._state()
        self.assertFalse(torch.equal(mid[22][0], before[22][0]))  # the lend really wrote the slots
        # A later chunk of the layer reads the adopted rows back out of the donor slots, byte for byte.
        again_row_of_source, again = self.early_streamer._gather_experts(ids)
        torch.cuda.synchronize()
        self.assertTrue(
            torch.equal(again["host_rows"][again_row_of_source.long()].cpu(), self.early.host_rows[ids.cpu()])
        )
        self.assertEqual(self.early_streamer.last_gather_stats.residency_hit_rows, 12)
        residency.restore()
        after = self._state()
        for layer_id in (21, 22):
            for name, then, now in zip(("rows", "map", "state", "generations"), before[layer_id], after[layer_id]):
                self.assertTrue(torch.equal(now, then), (layer_id, name))
            self.assertEqual(before[layer_id][4], after[layer_id][4])
        self.assertFalse(self.manager.prefill_lend_active)
        self.assertEqual(buffer.evicted_rows, 0)
        self.assertEqual(buffer.reloaded_rows, 12)
        residency.end()

    def test_a_failed_reload_evicts_the_slots_on_the_real_hot_cache(self):
        before = self._state()
        residency = Exl3LendingLayerMajorResidency(self.runner, rows=12)
        residency.begin(range(0, 2))
        residency.make_resident(0)
        self.early_streamer._gather_experts(torch.arange(4, 16, device="cuda"))
        # Layer 22 is the first donor: its eight slots are lent first.
        tier = self.manager.caches[22].streamer.pinned_host_cache

        def fail(ids, outputs):
            raise RuntimeError("tier read failed")

        tier.gather_rows = fail
        residency.restore()
        cache = self.manager.caches[22]
        torch.cuda.synchronize()
        self.assertEqual(residency.buffer.evicted_rows, 8)
        self.assertEqual(cache.slot_to_expert, [-1] * 8)
        self.assertTrue(bool((cache.expert_to_slot[16:24] == -1).all()))
        self.assertTrue(bool((cache.slot_state == 0).all()))
        # The layer whose reload worked is untouched.
        after = self._state()
        self.assertTrue(torch.equal(after[21][0], before[21][0]))
        self.assertEqual(after[21][4], before[21][4])
        self.assertFalse(self.manager.prefill_lend_active)

    def test_the_pointer_kernel_copies_rows_of_several_layers_in_both_directions(self):
        plan = [LentRow(layer_id, self.manager.caches[layer_id], slot) for layer_id in (22, 21) for slot in (0, 3, 7)]
        buffer = LentPrefillResidency(
            manager=self.manager, plan=plan, num_experts=EXPERTS,
            specs={"host_rows": ((ROW_BYTES,), torch.uint8)}, device=torch.device("cuda"),
        )
        staged = {"host_rows": torch.randint(0, 256, (6, ROW_BYTES), dtype=torch.uint8, device="cuda")}
        buffer.adopt(torch.arange(6, device="cuda"), staged, 6)
        for entry, row in enumerate(plan):
            self.assertTrue(torch.equal(row.cache.tensors["host_rows"][row.slot], staged["host_rows"][entry]))
        out = {"host_rows": torch.zeros((9, ROW_BYTES), dtype=torch.uint8, device="cuda")}
        rows = torch.tensor([5, 0, 3], device="cuda")
        positions = torch.tensor([8, 1, 4], device="cuda")
        buffer.scatter(rows, positions, 3, out)
        for entry, position in zip((5, 0, 3), (8, 1, 4)):
            self.assertTrue(torch.equal(out["host_rows"][position], staged["host_rows"][entry]))
        self.assertTrue(bool((out["host_rows"][[0, 2, 3, 5, 6, 7]] == 0).all()))  # nothing else written


if __name__ == "__main__":
    unittest.main()
