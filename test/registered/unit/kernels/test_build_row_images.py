"""scripts/dsv41/build_row_images.py: the converter that writes row images onto the mirror roots.

Every fixture is a few hundred KB under ``tmp_path``; the tool's reads and writes are O_DIRECT, as on the drives.
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "scripts", "dsv41"))

import build_row_images as bri  # noqa: E402

from sglang.srt.layers.moe import exl3_row_image as ri  # noqa: E402
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402
from sglang.test.dsv41_fake_exl3 import expert_spec  # noqa: E402

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

LAYERS, EXPERTS = 3, 5
ROWS_PER_SHARD = 8  # layer 1's experts span both shards
PREAMBLE = 13  # bytes of a non-expert tensor ahead of the rows: every row starts at an odd offset
# The real 12-tensor row (w1/w2/w3 x suh/svh/mul1/trellis, 4-byte mul1s), at the smallest dims whose six slab
# rows are all multiples of 512 B, as the real checkpoint's are.
SPEC = expert_spec(hidden=256, inter=256)


def _write_checkpoint(directory):
    """Two shards of unpadded expert rows at odd offsets; returns each (layer, expert)'s raw row."""
    row_bytes = sum(nbytes for *_, nbytes in SPEC)
    keys = [(layer, e) for layer in range(LAYERS) for e in range(EXPERTS)]
    rows = {key: os.urandom(row_bytes) for key in keys}
    weight_map = {}
    for shard, first in enumerate(range(0, len(keys), ROWS_PER_SHARD)):
        tensors = [(f"layers.0.attn.pad{shard}", "U8", (PREAMBLE,), b"\x07" * PREAMBLE)]
        for layer, e in keys[first : first + ROWS_PER_SHARD]:
            at = 0
            for suffix, dtype, shape, nbytes in SPEC:
                tensors.append((f"layers.{layer}.ffn.experts.{e}.{suffix}", dtype, shape, rows[(layer, e)][at : at + nbytes]))
                at += nbytes
        header, offset = {}, 0
        for name, dtype, shape, data in tensors:
            header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [offset, offset + len(data)]}
            offset += len(data)
        blob = json.dumps(header).encode()
        blob += b" " * (-len(blob) % 8)
        filename = f"model-{shard + 1:05d}.safetensors"
        with open(os.path.join(directory, filename), "wb") as f:
            f.write(len(blob).to_bytes(8, "little") + blob + b"".join(t[3] for t in tensors))
        weight_map.update({t[0]: filename for t in tensors})
    with open(os.path.join(directory, "model.safetensors.index.json"), "w") as f:
        json.dump({"weight_map": weight_map}, f)
    return rows


class _Case:
    def __init__(self, tmp_path):
        self.source = str(tmp_path / "ckpt")
        os.mkdir(self.source)
        self.rows = _write_checkpoint(self.source)
        self.src = bri.Source.load(self.source)
        self.roots = [str(tmp_path / "nvme_a"), str(tmp_path / "nvme_b")]
        for root in self.roots:
            os.mkdir(root)

    def run(self, *extra, roots=None):
        roots = self.roots if roots is None else roots
        return bri.main(["--model-dir", self.source, "--roots", os.pathsep.join(roots), "--threads", "2", *extra])

    def file(self, root, layer):
        return bri.layer_path(root, layer)

    def flip(self, root, layer, expert, at):
        """XOR one byte of an expert's image in a root's layer file."""
        with open(self.file(root, layer), "r+b") as f:
            f.seek(expert * self.src.image.row_stride + at)
            (byte,) = f.read(1)
            f.seek(-1, os.SEEK_CUR)
            f.write(bytes([byte ^ 0xFF]))

    def manifests(self):
        return [os.path.exists(os.path.join(ri.row_image_dir(r), ri.MANIFEST)) for r in self.roots]


def test_each_row_lands_as_its_image_and_open_row_images_accepts_the_set(tmp_path):
    case = _Case(tmp_path)
    assert all(r.file_offset % 2 == 1 for r in case.src.layout.records.values())
    assert case.run() == bri.EXIT_OK
    lay = case.src.image
    for root in case.roots:
        for layer in range(LAYERS):
            data = open(case.file(root, layer), "rb").read()
            assert len(data) == EXPERTS * lay.row_stride
            for e in range(EXPERTS):
                at = e * lay.row_stride
                assert data[at : at + lay.image_bytes] == ri.image_of_row(lay, case.rows[(layer, e)]), (root, layer, e)
                assert data[at + lay.image_bytes : at + lay.row_stride] == bytes(lay.row_stride - lay.image_bytes)
    images = ri.open_row_images(case.roots, case.src.layout, case.src.segments, case.source, range(LAYERS))
    assert set(images.paths) == set(range(LAYERS))


def test_an_interrupted_rebuild_leaves_no_manifest(tmp_path, monkeypatch):
    """A rerun that dies part-way must not leave the previous manifest vouching for files it was replacing."""
    case = _Case(tmp_path)
    assert case.run() == bri.EXIT_OK
    case.flip(case.roots[0], 1, 2, 100)  # the rerun must rebuild layer 1 on root a

    def interrupted(job):
        raise RuntimeError("interrupted")

    monkeypatch.setattr(bri, "_finish_layer", interrupted)
    with pytest.raises(RuntimeError, match="interrupted"):
        case.run()
    assert case.manifests() == [False, False]


def test_resume_rereads_finished_layers_and_rebuilds_only_the_bad_one(tmp_path):
    case = _Case(tmp_path)
    assert case.run() == bri.EXIT_OK
    inodes = {(r, layer): os.stat(case.file(r, layer)).st_ino for r in case.roots for layer in range(LAYERS)}
    bad_root = case.roots[0]
    case.flip(bad_root, 1, 2, 100)  # the sidecar still claims the layer is good; only a re-read can tell
    assert case.run() == bri.EXIT_OK
    for (root, layer), inode in inodes.items():
        rebuilt = os.stat(case.file(root, layer)).st_ino != inode
        assert rebuilt == ((root, layer) == (bad_root, 1)), (root, layer)
    ri.open_row_images(case.roots, case.src.layout, case.src.segments, case.source, range(LAYERS))


def test_verify_fails_on_one_corrupted_byte(tmp_path, capsys):
    case = _Case(tmp_path)
    assert case.run() == bri.EXIT_OK
    assert case.run("--verify") == bri.EXIT_OK
    capsys.readouterr()
    # The last byte of the last expert's image: the tail of a partial chunk.
    case.flip(case.roots[1], 2, EXPERTS - 1, case.src.image.image_bytes - 1)
    assert case.run("--verify") == bri.EXIT_MISMATCH
    out = capsys.readouterr().out
    assert f"MISMATCH on {case.roots[1]}: 1 row(s)" in out and f"layer 2 expert {EXPERTS - 1}:" in out


def _foreign_manifest(case):
    """Root a holds a complete manifest of another checkpoint."""
    os.makedirs(ri.row_image_dir(case.roots[0]))
    other = dict(case.src.fingerprint, records_sha256="0" * 64)
    ri.write_manifest(case.roots[0], ri.manifest_json(case.src.image, other, {}))


@pytest.mark.parametrize(
    "setup, roots, message",
    [
        (lambda case, mp: None, lambda case: [case.source], "is the source checkpoint directory"),
        (lambda case, mp: _foreign_manifest(case), None, "different source"),
        (lambda case, mp: mp.setattr(bri, "_free_bytes", lambda path: 0), None, "free"),
    ],
    ids=["root-is-source", "manifest-of-another-source", "no-space"],
)
def test_refuses_a_root_it_must_not_write(tmp_path, monkeypatch, capsys, setup, roots, message):
    case = _Case(tmp_path)
    setup(case, monkeypatch)
    before = {r: sorted(os.listdir(ri.row_image_dir(r))) if os.path.isdir(ri.row_image_dir(r)) else None for r in case.roots}
    assert case.run(roots=roots(case) if roots else None) == bri.EXIT_USAGE
    assert message in capsys.readouterr().err
    after = {r: sorted(os.listdir(ri.row_image_dir(r))) if os.path.isdir(ri.row_image_dir(r)) else None for r in case.roots}
    assert all(not any(n.endswith(".rows") or n.endswith(".rows.tmp") for n in (names or [])) for names in after.values())
    if roots is None and message == "different source":
        assert after == before  # the other source's manifest is left alone
    assert not os.path.exists(os.path.join(ri.row_image_dir(case.source), ri.MANIFEST))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
