"""scripts/dsv41/verify_expert_mirror.py: the gate the mirror copies are trusted on.

Every fixture is a few KB under ``tmp_path``; nothing here touches a real drive.
"""

import os
import shutil
import sys

import pytest
import torch

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "..", "..", "scripts", "dsv41"
    ),
)

import verify_expert_mirror as vem  # noqa: E402

from sglang.srt.layers.moe.exl3_expert_format import Exl3ExpertFormat  # noqa: E402
from sglang.srt.layers.moe.exl3_expert_layout import build_exl3_expert_layout  # noqa: E402
from sglang.srt.layers.moe.exl3_row_reader import Exl3RowReader  # noqa: E402
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402
from sglang.test.dsv41_fake_exl3 import write_fake_exl3  # noqa: E402

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

LAYERS, EXPERTS = 3, 5


class _Fixture:
    """A fake checkpoint (3 layers x 5 experts, 3 experts per shard) and two copies.

    Shards: model-00001 = (0,0..2), 00002 = (0,3),(0,4),(1,0), 00003 = (1,1..3),
    00004 = (1,4),(2,0),(2,1), 00005 = (2,2..4).
    """

    def __init__(self, tmp_path, num_roots=2):
        self.source = tmp_path / "ckpt"
        self.source.mkdir()
        self.rows = write_fake_exl3(
            str(self.source), num_layers=LAYERS, num_experts=EXPERTS
        )
        self.layout = build_exl3_expert_layout(str(self.source))
        self.streamed = sum(
            segment.nbytes
            for segment in Exl3ExpertFormat(self.layout, 0, direct=False).segment_map()
        )
        self.roots = []
        for i in range(num_roots):
            root = tmp_path / f"drive_{i}" / "copy"
            shutil.copytree(self.source, root)
            self.roots.append(str(root))

    def verify(self, **kwargs):
        kwargs.setdefault("direct", False)
        return vem.verify_mirror(str(self.source), self.roots, **kwargs)

    def record(self, layer, expert):
        return self.layout.records[(layer, expert)]

    def mirror_file(self, root_index, layer, expert):
        record = self.record(layer, expert)
        return os.path.join(
            self.roots[root_index], os.path.relpath(record.path, str(self.source))
        )

    def row_offset(self, suffix):
        return next(s.rel_offset for s in self.layout.tensors if s.name == suffix)

    def flip(self, root_index, layer, expert, row_offset):
        """XOR one byte of a row in one root's copy; returns (expected, found)."""
        path = self.mirror_file(root_index, layer, expert)
        at = self.record(layer, expert).file_offset + row_offset
        with open(path, "r+b") as f:
            f.seek(at)
            (byte,) = f.read(1)
            f.seek(at)
            f.write(bytes([byte ^ 0xFF]))
        return byte, byte ^ 0xFF


def _first_nonzero_streamed_offset(fx, layer, expert):
    """Lowest row offset a zero-filled copy differs at, ignoring the unread mul1."""
    row = fx.rows[(layer, expert)]
    mul1 = {
        span.rel_offset + i
        for span in fx.layout.tensors
        if span.name.endswith("mul1")
        for i in range(span.nbytes)
    }
    return next(i for i, b in enumerate(row) if b and i not in mul1)


# --- A clean mirror, full mode -----------------------------------------------


def test_identical_copies_pass_in_full_mode(tmp_path):
    fx = _Fixture(tmp_path)
    result = fx.verify()
    assert result.ok and result.full
    assert [r.root for r in result.root_reports] == fx.roots
    for report in result.root_reports:
        assert report.rows_checked == LAYERS * EXPERTS
        assert report.bytes_compared == LAYERS * EXPERTS * fx.streamed
        assert not report.size_problems and not report.mismatches
    text = vem.render(result)
    assert "MODE: FULL" in text and "PARTIAL" not in text
    # Every root is named with what was compared against it.
    assert all(root in text for root in fx.roots)


def test_a_full_passing_run_that_read_the_page_cache_is_not_complete(tmp_path):
    fx = _Fixture(tmp_path)
    result = fx.verify(direct=False)
    assert result.ok and result.full and not result.complete
    assert "BUFFERED" in vem.render(result)


# --- Content, not sizes -------------------------------------------------------


def test_a_sparse_copy_of_the_right_size_is_caught_by_content(tmp_path):
    fx = _Fixture(tmp_path)
    victim = fx.mirror_file(1, 1, 1)  # model-00003: (1,1), (1,2), (1,3)
    size = os.path.getsize(victim)
    with open(victim, "wb") as f:
        f.truncate(size)  # a hole: right length, reads back zeros
    assert os.path.getsize(victim) == size

    result = fx.verify()
    assert not result.ok
    clean, bad = result.root_reports
    assert clean.root == fx.roots[0] and not clean.mismatches
    assert bad.root == fx.roots[1] and not bad.size_problems
    first = bad.mismatches[0]
    assert (first.layer, first.expert) == (1, 1)
    offset = _first_nonzero_streamed_offset(fx, 1, 1)
    assert first.row_offset == offset
    assert first.file_offset == fx.record(1, 1).file_offset + offset
    assert first.mirror_path == victim
    assert first.source_path == fx.record(1, 1).path
    assert first.expected == fx.rows[(1, 1)][offset : offset + len(first.expected)]
    assert first.found == b"\x00" * len(first.found)
    assert first.mirror_row_all_zero
    assert len(first.expected) == len(first.found) >= 1
    # Stops at the first bad row: (0,*) and (1,0) passed, (1,1) is the 7th row.
    assert bad.rows_checked == 1 * EXPERTS + 1 + 1
    assert bad.stopped_early


def test_one_flipped_byte_is_reported_at_its_exact_place(tmp_path):
    fx = _Fixture(tmp_path)
    at = fx.row_offset("w2.trellis") + 7
    expected, found = fx.flip(0, 2, 3, at)
    result = fx.verify()
    assert not result.ok
    bad = result.root_reports[0]
    (first,) = bad.mismatches
    assert (first.layer, first.expert, first.row_offset) == (2, 3, at)
    assert first.file_offset == fx.record(2, 3).file_offset + at
    assert first.tensor == "w2.trellis" and first.tensor_offset == 7
    assert first.expected[0] == expected and first.found[0] == found
    assert first.bad_bytes_in_row == 1 and not first.mirror_row_all_zero
    assert not result.root_reports[1].mismatches
    text = vem.render(result)
    assert fx.roots[0] in text and first.mirror_path in text
    assert "layer 2" in text and "expert 3" in text
    assert str(first.file_offset) in text and str(at) in text
    assert f"{expected:02x}" in text and f"{found:02x}" in text


def test_the_first_bad_row_is_the_lowest_layer_and_expert(tmp_path):
    fx = _Fixture(tmp_path)
    fx.flip(1, 2, 0, 100)
    fx.flip(1, 1, 4, 100)
    fx.flip(1, 1, 2, 100)
    (first, *_) = fx.verify().root_reports[1].mismatches
    assert (first.layer, first.expert) == (1, 2)


def test_the_first_byte_in_a_row_is_lowest_by_row_offset_not_tensor_name(tmp_path):
    """w2.suh is streamed as w2_suh, w3.trellis as w13_trellis (checked first)."""
    fx = _Fixture(tmp_path)
    low = fx.row_offset("w2.suh") + 3
    high = fx.row_offset("w3.trellis") + 9
    assert low < high
    fx.flip(0, 0, 1, high)
    fx.flip(0, 0, 1, low)
    (first,) = fx.verify().root_reports[0].mismatches
    assert first.row_offset == low and first.tensor == "w2.suh"
    assert first.bad_bytes_in_row == 2


def test_a_mismatch_in_a_w3_part_reports_its_row_offset(tmp_path):
    """w3 tensors are the second part of their streamed name; offsets must map back."""
    fx = _Fixture(tmp_path)
    at = fx.row_offset("w3.svh") + 5
    fx.flip(1, 0, 4, at)
    (first,) = fx.verify().root_reports[1].mismatches
    assert (first.row_offset, first.tensor, first.tensor_offset) == (at, "w3.svh", 5)


def test_keep_going_lists_every_bad_row_and_compares_them_all(tmp_path):
    fx = _Fixture(tmp_path)
    fx.flip(0, 0, 0, 50)
    fx.flip(0, 1, 2, 60)
    fx.flip(0, 2, 4, 70)
    stopped = fx.verify().root_reports[0]
    assert len(stopped.mismatches) == 1 and stopped.stopped_early
    assert stopped.rows_checked == 1
    every = fx.verify(keep_going=True).root_reports[0]
    assert [(m.layer, m.expert) for m in every.mismatches] == [(0, 0), (1, 2), (2, 4)]
    assert every.rows_checked == LAYERS * EXPERTS and not every.stopped_early


def test_a_bad_root_does_not_hide_the_other_roots(tmp_path):
    fx = _Fixture(tmp_path, num_roots=3)
    fx.flip(0, 0, 0, 10)
    fx.flip(2, 2, 2, 10)
    result = fx.verify()
    assert [bool(r.mismatches) for r in result.root_reports] == [True, False, True]
    assert result.root_reports[1].rows_checked == LAYERS * EXPERTS


def test_a_read_that_writes_nothing_is_not_mistaken_for_a_match(tmp_path, monkeypatch):
    """Root 1's read leaves its buffer untouched. Without zeroing between reads the
    buffer would still hold root 0's correct bytes and the root would pass."""
    fx = _Fixture(tmp_path)
    real = vem.Exl3MirrorRowSource.for_mirrored_layer

    class _Silent:
        bounce = torch.zeros(1, dtype=torch.uint8)

        def read(self, rows, destinations, destination_rows=None):
            return None

    def build(*args, roots, **kwargs):
        if roots[0] == fx.roots[1]:
            return _Silent()
        return real(*args, roots=roots, **kwargs)

    monkeypatch.setattr(
        vem.Exl3MirrorRowSource, "for_mirrored_layer", staticmethod(build)
    )
    result = fx.verify()
    assert result.root_reports[0].ok
    (first, *_) = result.root_reports[1].mismatches
    assert (first.layer, first.expert) == (0, 0) and first.mirror_row_all_zero


def test_a_reader_that_does_no_io_cannot_pass_on_stale_bounce_bytes(
    tmp_path, monkeypatch
):
    """The real row sources, with the mirror reader patched to do no I/O. The source
    read leaves its rows in the shared bounce ring; unless the ring is cleared the
    mirror source copies those out and the (fully sparse) copy passes."""
    fx = _Fixture(tmp_path)
    for path in {
        fx.mirror_file(1, layer, e) for layer in range(LAYERS) for e in range(EXPERTS)
    }:
        size = os.path.getsize(path)
        with open(path, "wb") as f:
            f.truncate(size)  # a hole: right length, all zeros

    def no_io(self, keys, destinations, *, roots, policy):
        return [self.layout.records[key].aligned_read(4096)[2] for key in keys]

    monkeypatch.setattr(Exl3RowReader, "read_split", no_io)
    result = fx.verify()
    assert not result.ok
    assert result.root_reports[0].mismatches  # nothing was read for it either
    (first, *_) = result.root_reports[1].mismatches
    assert (first.layer, first.expert) == (0, 0) and first.mirror_row_all_zero


def test_batches_smaller_than_a_layer_count_rows_and_find_the_first_bad_one(tmp_path):
    fx = _Fixture(tmp_path)
    fx.flip(1, 1, 3, 40)  # in the second batch of two rows
    fx.flip(1, 1, 4, 40)
    stopped = fx.verify(batch_rows=2).root_reports[1]
    (first,) = stopped.mismatches
    assert (first.layer, first.expert) == (1, 3)
    assert stopped.rows_checked == EXPERTS + 3 + 1  # layer 0, (1,0..2), then (1,3)
    every = fx.verify(batch_rows=2, keep_going=True).root_reports[1]
    assert [(m.layer, m.expert) for m in every.mismatches] == [(1, 3), (1, 4)]
    assert every.rows_checked == LAYERS * EXPERTS
    clean = fx.verify(batch_rows=1, layers=[0, 2])
    assert clean.ok and clean.root_reports[0].rows_checked == 2 * EXPERTS


@pytest.mark.skipif(os.geteuid() == 0, reason="root opens any file")
def test_an_unopenable_copy_is_a_read_failure_not_a_pass(tmp_path):
    fx = _Fixture(tmp_path)
    victim = fx.mirror_file(1, 1, 1)
    os.chmod(victim, 0)
    try:
        result = fx.verify()
    finally:
        os.chmod(victim, 0o644)
    assert not result.ok and result.root_reports[0].ok
    (failure, *_) = result.root_reports[1].read_failures
    assert failure.root == fx.roots[1] and failure.layer == 1
    text = vem.render(result)
    assert "READ FAILED at layer 1" in text and "FAIL" in text
    assert result.exit_code == 2


def test_a_read_that_raises_is_reported_and_the_other_roots_still_run(
    tmp_path, monkeypatch
):
    fx = _Fixture(tmp_path)
    real = vem.Exl3MirrorRowSource.for_mirrored_layer

    class _Broken:
        bounce = torch.zeros(1, dtype=torch.uint8)

        def read(self, *args, **kwargs):
            raise RuntimeError("io_uring read ended early: 10 of 20 bytes")

    def build(*args, roots, **kwargs):
        if roots[0] == fx.roots[0] and args[1] == 2:
            return _Broken()
        return real(*args, roots=roots, **kwargs)

    monkeypatch.setattr(
        vem.Exl3MirrorRowSource, "for_mirrored_layer", staticmethod(build)
    )
    result = fx.verify()
    broken, fine = result.root_reports
    assert not result.ok and fine.ok
    assert fine.rows_checked == LAYERS * EXPERTS
    (failure,) = broken.read_failures
    assert (failure.root, failure.layer) == (fx.roots[0], 2)
    assert "ended early" in failure.message and broken.stopped_early
    assert broken.rows_checked == 2 * EXPERTS
    assert "READ FAILED at layer 2: io_uring read ended early" in vem.render(result)

    def cannot_open(*args, roots, **kwargs):
        raise OSError("mirror root: cannot stat")

    monkeypatch.setattr(
        vem.Exl3MirrorRowSource, "for_mirrored_layer", staticmethod(cannot_open)
    )
    result = fx.verify()
    assert not result.ok
    assert all(r.read_failures and r.rows_checked == 0 for r in result.root_reports)


# --- Sizes ---------------------------------------------------------------------


def test_a_truncated_copy_is_a_size_mismatch_naming_root_file_and_sizes(tmp_path):
    fx = _Fixture(tmp_path)
    victim = fx.mirror_file(1, 1, 1)
    size = os.path.getsize(victim)
    with open(victim, "r+b") as f:
        f.truncate(size - 100)
    result = fx.verify()
    assert not result.ok
    (problem,) = result.root_reports[1].size_problems
    assert problem.kind == "size"
    assert problem.root == fx.roots[1] and problem.mirror_path == victim
    assert problem.path == fx.record(1, 1).path
    assert (problem.source_bytes, problem.mirror_bytes) == (size, size - 100)
    text = vem.render(result)
    assert victim in text and str(size) in text and str(size - 100) in text
    # The layers that use the bad file are not read through the mirror source
    # (it would refuse them); the report says so instead of passing them.
    assert result.root_reports[1].layers_skipped == [1]
    assert "skipped" in text and "layer 1" in text
    # Everything else is still compared.
    assert result.root_reports[1].rows_checked == 2 * EXPERTS


def test_a_missing_copy_and_an_oversized_copy_are_reported(tmp_path):
    fx = _Fixture(tmp_path)
    os.remove(fx.mirror_file(0, 0, 0))
    with open(fx.mirror_file(1, 2, 3), "ab") as f:
        f.write(b"x")
    result = fx.verify()
    (gone,) = result.root_reports[0].size_problems
    assert gone.kind == "missing" and gone.mirror_bytes is None
    (long,) = result.root_reports[1].size_problems
    assert long.kind == "size" and long.mirror_bytes == long.source_bytes + 1
    assert not result.ok


def test_sizes_are_checked_for_every_file_even_when_layers_are_chosen(tmp_path):
    fx = _Fixture(tmp_path)
    with open(fx.mirror_file(0, 2, 3), "ab") as f:  # a layer-2 shard
        f.write(b"x")
    result = fx.verify(layers=[0])
    assert not result.ok and result.root_reports[0].size_problems
    assert result.files_checked == len({r.path for r in fx.layout.records.values()})
    # Layer 0 itself is unaffected and was still compared.
    assert result.root_reports[0].rows_checked == EXPERTS


def test_size_problems_are_listed_in_root_then_path_order(tmp_path):
    fx = _Fixture(tmp_path)
    for root_index in (1, 0):
        for layer, expert in ((2, 4), (0, 0)):
            with open(fx.mirror_file(root_index, layer, expert), "ab") as f:
                f.write(b"x")
    result = fx.verify()
    order = [
        (p.root, os.path.basename(p.path))
        for report in result.root_reports
        for p in report.size_problems
    ]
    assert order == [
        (fx.roots[0], "model-00001.safetensors"),
        (fx.roots[0], "model-00005.safetensors"),
        (fx.roots[1], "model-00001.safetensors"),
        (fx.roots[1], "model-00005.safetensors"),
    ]
    assert vem.render(result).count("model-00001.safetensors") >= 2


def test_a_symlink_to_the_source_is_not_a_copy(tmp_path):
    fx = _Fixture(tmp_path)
    victim = fx.mirror_file(1, 0, 0)
    os.remove(victim)
    os.symlink(fx.record(0, 0).path, victim)
    result = fx.verify()
    assert not result.ok
    (problem,) = result.root_reports[1].size_problems
    assert problem.kind == "same-file" and problem.mirror_path == victim
    assert "same file" in vem.render(result)


# --- Refusing to verify nothing --------------------------------------------------


def test_a_root_that_is_the_source_is_refused(tmp_path):
    fx = _Fixture(tmp_path)
    with pytest.raises(ValueError, match="same directory as the source"):
        vem.verify_mirror(str(fx.source), [fx.roots[0], str(fx.source)], direct=False)
    alias = tmp_path / "alias"
    os.symlink(fx.roots[0], alias)
    with pytest.raises(ValueError, match="same directory as"):
        vem.verify_mirror(str(fx.source), [fx.roots[0], str(alias)], direct=False)


def test_no_roots_are_refused(tmp_path):
    fx = _Fixture(tmp_path)
    with pytest.raises(ValueError, match="no mirror roots"):
        vem.verify_mirror(str(fx.source), [], direct=False)


# --- Bounding the work; the mode is never misreported -----------------------------


def test_chosen_layers_are_a_partial_check_that_says_so(tmp_path):
    fx = _Fixture(tmp_path)
    fx.flip(0, 2, 0, 20)  # outside the chosen layers
    result = fx.verify(layers=[0, 1])
    assert result.ok and not result.full and not result.complete
    assert all(r.rows_checked == 2 * EXPERTS for r in result.root_reports)
    text = vem.render(result)
    assert "MODE: PARTIAL" in text and "FULL" not in text
    assert "layers 0-1 of 3" in text
    assert "NOT a full verification" in text
    # A chosen layer that holds the damage does fail.
    assert not fx.verify(layers=[2]).ok


def test_sampling_is_a_partial_check_that_says_so(tmp_path):
    fx = _Fixture(tmp_path)
    result = fx.verify(sample_experts=2, seed=7)
    assert result.ok and not result.full
    assert all(r.rows_checked == LAYERS * 2 for r in result.root_reports)
    text = vem.render(result)
    assert "MODE: PARTIAL" in text and "FULL" not in text
    assert "2 of 5 experts per layer" in text and "seed 7" in text
    assert "NOT a full verification" in text


def test_sampling_is_deterministic_and_independent_of_the_other_layers(tmp_path):
    fx = _Fixture(tmp_path)
    a = fx.verify(sample_experts=2, seed=1)
    b = fx.verify(sample_experts=2, seed=1)
    assert a.sampled == b.sampled
    assert all(len(v) == 2 and list(v) == sorted(v) for v in a.sampled.values())
    assert fx.verify(layers=[1], sample_experts=2, seed=1).sampled[1] == a.sampled[1]
    assert any(
        fx.verify(sample_experts=2, seed=s).sampled != a.sampled for s in range(2, 12)
    )


def test_sampling_finds_damage_in_a_sampled_row_and_misses_an_unsampled_one(tmp_path):
    fx = _Fixture(tmp_path)
    sampled = fx.verify(sample_experts=2, seed=3).sampled[1]
    unsampled = next(e for e in range(EXPERTS) if e not in sampled)
    fx.flip(0, 1, unsampled, 30)
    assert fx.verify(sample_experts=2, seed=3).ok  # the hole in the check
    fx.flip(0, 1, sampled[0], 30)
    result = fx.verify(sample_experts=2, seed=3)
    assert not result.ok
    assert result.root_reports[0].mismatches[0].expert == sampled[0]


def test_sampling_every_expert_of_every_layer_is_a_full_check(tmp_path):
    fx = _Fixture(tmp_path)
    result = fx.verify(sample_experts=EXPERTS)
    assert result.full and "MODE: FULL" in vem.render(result)
    assert result.root_reports[0].rows_checked == LAYERS * EXPERTS
    assert fx.verify(sample_experts=EXPERTS + 3).full


def test_bad_layer_selections_are_refused(tmp_path):
    fx = _Fixture(tmp_path)
    with pytest.raises(ValueError, match="layer 3"):
        fx.verify(layers=[0, 3])
    with pytest.raises(ValueError, match="no layers"):
        fx.verify(layers=[])
    with pytest.raises(ValueError, match="sample"):
        fx.verify(sample_experts=0)


def test_layer_specs():
    assert vem.parse_layers("all") is None
    assert vem.parse_layers("0,2-3,5") == [0, 2, 3, 5]
    assert vem.parse_layers("3,1,3") == [1, 3]
    for bad in ("", "x", "2-", "4-2", "-1", "1,,2"):
        with pytest.raises(ValueError, match="layer"):
            vem.parse_layers(bad)


# --- The command line ------------------------------------------------------------------


def _argv(fx, *extra):
    return ["--source", str(fx.source), "--roots", *fx.roots, *extra]


def test_exit_codes(tmp_path, capsys):
    fx = _Fixture(tmp_path)
    # Passing but not complete (buffered): 3, never 0.
    assert vem.main(_argv(fx, "--buffered")) == 3
    assert "BUFFERED" in capsys.readouterr().out
    # A partial pass is 3 too.
    assert vem.main(_argv(fx, "--buffered", "--layers", "0")) == 3
    assert "PARTIAL" in capsys.readouterr().out
    # A mismatch is 2.
    fx.flip(1, 0, 0, 5)
    assert vem.main(_argv(fx, "--buffered")) == 2
    assert "FAILED" in capsys.readouterr().out
    # A configuration error is 1, on stderr.
    assert vem.main(["--source", str(fx.source), "--roots", str(fx.source)]) == 1
    assert "same directory" in capsys.readouterr().err


def test_a_complete_verification_is_exit_zero(tmp_path, capsys):
    fx = _Fixture(tmp_path)
    try:
        fd = os.open(
            str(fx.source / "model-00001.safetensors"), os.O_RDONLY | os.O_DIRECT
        )
        os.close(fd)
    except OSError:
        pytest.skip("this filesystem cannot O_DIRECT")
    assert vem.main(_argv(fx)) == 0
    out = capsys.readouterr().out
    assert "MODE: FULL" in out and "O_DIRECT" in out and "VERIFIED" in out


def test_reads_are_direct_unless_buffered_is_asked_for():
    args = vem.parse_args(["--source", "s", "--roots", "a", "b"])
    assert args.buffered is False and args.layers == "all"
    assert args.sample_experts is None and args.keep_going is False
    assert vem.parse_args(["--source", "s", "--roots", "a", "--buffered"]).buffered


def test_roots_default_to_the_environment_knob(tmp_path, capsys):
    from sglang.srt.environ import envs

    fx = _Fixture(tmp_path)
    with envs.SGLANG_MOE_EXPERT_MIRROR_DIRS.override(os.pathsep.join(fx.roots)):
        assert vem.main(["--source", str(fx.source), "--buffered"]) == 3
    out = capsys.readouterr().out
    assert all(root in out for root in fx.roots)
    assert vem.main(["--source", str(fx.source), "--buffered"]) == 1
    assert "no mirror roots" in capsys.readouterr().err


def test_every_line_of_a_buffered_report_agrees_with_the_exit_code(tmp_path):
    fx = _Fixture(tmp_path)
    result = fx.verify()
    text = vem.render(result)
    assert result.exit_code == 3
    assert "MODE: FULL, BUFFERED (not a drive verification)" in text
    assert text.count("PASS (buffered)") == 2
    assert "VERIFIED" not in text and "PASSED, BUT THE READS WERE BUFFERED" in text


def test_partial_and_buffered_passes_are_qualified_on_every_root(tmp_path):
    fx = _Fixture(tmp_path)
    text = vem.render(fx.verify(layers=[0]))
    assert "MODE: PARTIAL, BUFFERED (not a drive verification)" in text
    assert text.count("PASS (partial, buffered)") == 2
    assert "ROOT" in text and "\nROOT " + fx.roots[0] + ": PASS\n" not in text
    assert "also BUFFERED" in text


def test_a_failed_verdict_carries_its_scope(tmp_path):
    fx = _Fixture(tmp_path)
    fx.flip(0, 0, 0, 5)
    text = vem.render(fx.verify(layers=[0]))
    assert "VERDICT: FAILED [PARTIAL, BUFFERED (not a drive verification)]" in text


def test_the_verified_verdict_states_what_it_covers_and_what_it_does_not():
    """Built from a hand-made complete result: O_DIRECT cannot be assumed here."""
    fx_result = vem.VerifyResult(
        source="/s",
        num_layers=1,
        num_experts=1,
        layers=(0,),
        sampled={0: (0,)},
        sample_experts=None,
        seed=0,
        direct=True,
        keep_going=False,
        files_checked=1,
        streamed_row_bytes=10,
        root_reports=[vem.RootReport("/r", rows_checked=1, bytes_compared=10)],
        elapsed_seconds=1.5,
    )
    assert fx_result.complete and fx_result.exit_code == 0
    text = vem.render(fx_result)
    verdict = text[text.index("VERDICT") :]
    assert "every byte the runtime reads matches" in verdict
    assert "does NOT mean the files are identical" in verdict
    for excluded in ("mul1", "headers", "padding", "index", "outside the layout"):
        assert excluded in verdict
    assert "SGLANG_DSV41_EXPERT_DIR is ever pointed at a mirror" in verdict
    assert "PASS\n" in text and "elapsed: 1.5 s" in text


def test_the_header_prints_once_and_the_read_time_is_reported(tmp_path, capsys):
    fx = _Fixture(tmp_path)
    vem.main(_argv(fx, "--buffered"))
    out = capsys.readouterr().out
    assert out.count("MODE: FULL") == 1 and out.count("source: ") == 1
    assert out.count("compared: sizes of all") == 1
    assert out.count(", read ") == 2 and "elapsed: " in out


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
