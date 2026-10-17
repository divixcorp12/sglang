"""A speculative verify's dead rows post DEAD_ROW_ID Engram ids, so the native store reads nothing for them (CPU)."""

from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers import engram
from sglang.srt.layers.moe.verify_dead_rows import VerifyDeadRows, attach_verify_dead_rows
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

ROWS, COLS = 6, 4


def _batch(*, verify: bool):
    return SimpleNamespace(
        forward_mode=SimpleNamespace(is_decode=lambda: not verify, is_target_verify=lambda: verify)
    )


def _ids(tokens: int) -> torch.Tensor:
    return torch.arange(tokens * COLS, dtype=torch.int64).view(tokens, COLS) + 100


def test_dead_rows_become_the_sentinel_and_live_rows_are_unchanged():
    dead_rows = VerifyDeadRows(max_rows=ROWS, device="cpu")
    dead_rows.set_live_rows(torch.tensor([2]))
    ids = _ids(ROWS)
    got = engram.mask_dead_lookup_rows(ids, dead_rows.dead, _batch(verify=True))
    assert got.dtype == torch.int64
    assert torch.equal(got[:2], ids[:2])
    assert (got[2:] == engram.DEAD_ROW_ID).all()
    dead_rows.clear()
    assert torch.equal(engram.mask_dead_lookup_rows(ids, dead_rows.dead, _batch(verify=True)), ids)


@pytest.mark.parametrize(
    "tokens, verify, attached",
    [(1, False, True), (ROWS, False, True), (2 * ROWS, True, True), (ROWS, True, False)],
)
def test_the_mask_is_the_identity_outside_a_one_request_verify(tokens, verify, attached):
    """Decode and prefill, a bs > 1 verify (more tokens than dead rows) and a layer no cut reached: the ids are
    returned as they are, even with every row past the first marked dead."""
    dead_rows = VerifyDeadRows(max_rows=ROWS, device="cpu")
    dead_rows.set_live_rows(torch.tensor([1]))
    ids = _ids(tokens)
    got = engram.mask_dead_lookup_rows(ids, dead_rows.dead if attached else None, _batch(verify=verify))
    assert got is ids


def test_attach_points_every_engram_embedding_at_the_dead_rows():
    embed = engram.EngramEmbedding.__new__(engram.EngramEmbedding)
    torch.nn.Module.__init__(embed)
    embed.verify_dead_rows = None
    dead_rows = VerifyDeadRows(max_rows=ROWS, device="cpu")
    assert attach_verify_dead_rows(torch.nn.ModuleList([embed]), dead_rows) == 0
    assert embed.verify_dead_rows is dead_rows.dead


def test_the_early_post_carries_the_sentinel_for_dead_rows(monkeypatch):
    """Mutant: post_engram_device_lookups posts the unmasked ids, or masks a layer's ids with another layer's
    columns -- red."""
    posted = []

    class FakeLookup:
        def __init__(self, file_table, indices):
            self.shape = tuple(indices.shape)

        def post(self, indices):
            posted.append(indices.clone())

    monkeypatch.setenv("SGLANG_DSV41_ENABLE_ENGRAM_DEVICE_WAIT", "1")
    monkeypatch.setattr(engram, "_EngramDeviceWaitLookup", FakeLookup)
    monkeypatch.setattr(engram, "_captures_native_lookup", lambda embed, indices, forward_batch: True)
    dead_rows = VerifyDeadRows(max_rows=ROWS, device="cpu")
    dead_rows.set_live_rows(torch.tensor([3]))
    embeds = [
        SimpleNamespace(file_table=object(), verify_dead_rows=dead_rows.dead, _posted_lookup=None) for _ in range(2)
    ]
    layers = [SimpleNamespace(embed=embed, layer_hash_index=i) for i, embed in enumerate(embeds)]
    hash_ids = torch.stack([_ids(ROWS), _ids(ROWS) + 1000], dim=1)
    engram.post_engram_device_lookups(layers, hash_ids, _batch(verify=True))
    assert len(posted) == 2
    for index, ids in enumerate(posted):
        assert torch.equal(ids[:3], hash_ids[:3, index])
        assert (ids[3:] == engram.DEAD_ROW_ID).all()
    assert all(embed._posted_lookup.shape == (ROWS, COLS) for embed in embeds)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
