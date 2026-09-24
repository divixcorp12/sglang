"""Row images (exl3_row_image): the image layout, the byte mapping from a checkpoint row, and the manifest checks
that decide whether a root's images may be read at all."""

import json
import os

import pytest

from sglang.srt.layers.moe import exl3_row_image as ri
from sglang.srt.layers.moe.exl3_expert_format import EXL3_STREAMED_NAMES, RowSegment
from sglang.srt.layers.moe.exl3_expert_layout import Exl3ExpertLayout, Exl3ExpertRecord, Exl3TensorSpan
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

# A small row shaped like dsv41's: w1/w2/w3 x (suh, svh, mul1, trellis), packed unpadded, with 4-byte mul1s that
# leave the later tensors off 16-byte alignment, as the real checkpoint does.
_SIZES = {"suh": 1024, "svh": 512, "mul1": 4, "trellis": 4096}
_ORDER = [(w, k) for w in ("w1", "w2", "w3") for k in ("suh", "svh", "mul1", "trellis")]
_PREFIX = {"w1": ("w13", 0), "w3": ("w13", 1), "w2": ("w2", 0)}


def _spans():
    spans, at = [], 0
    for w, k in _ORDER:
        spans.append(Exl3TensorSpan(f"{w}.{k}", at, _SIZES[k], "I32" if k == "mul1" else "F16", ()))
        at += _SIZES[k]
    return spans, at


def _segments():
    spans, _ = _spans()
    out = []
    for s in spans:
        w, k = s.name.split(".")
        if k == "mul1":
            continue
        prefix, part = _PREFIX[w]
        out.append(RowSegment(f"{prefix}_{k}", part, part * s.nbytes, s.rel_offset, s.nbytes))
    return sorted(out, key=lambda s: s.src_offset)


def _checkpoint(tmp_path, layers=2, experts=3, head=13):
    """A one-shard checkpoint whose rows start at odd offsets (a ``head``-byte preamble)."""
    spans, row = _spans()
    src = tmp_path / "src"
    src.mkdir()
    shard = src / "model.safetensors"
    data = bytearray(os.urandom(head + layers * experts * row + 7))
    shard.write_bytes(data)
    records = {
        (layer, e): Exl3ExpertRecord(layer, e, str(shard), head + (layer * experts + e) * row, row)
        for layer in range(layers)
        for e in range(experts)
    }
    layout = Exl3ExpertLayout(tuple(spans), row, records, layers, experts)
    return layout, str(src), data


def _build(layout, source_root, data, root, layer_ids=None):
    """A reference build of one root's images: what the converter must produce."""
    image_layout = ri.row_image_layout(_segments())
    os.makedirs(ri.row_image_dir(root), exist_ok=True)
    digests = {}
    for layer in layer_ids if layer_ids is not None else range(layout.num_layers):
        out = bytearray()
        digests[layer] = []
        for e in range(layout.num_experts):
            r = layout.records[(layer, e)]
            image = ri.image_of_row(image_layout, data[r.file_offset : r.file_offset + r.nbytes])
            digests[layer].append(ri.row_digest(image))
            out += image + bytes(image_layout.row_stride - image_layout.image_bytes)
        with open(os.path.join(ri.row_image_dir(root), ri.layer_file_name(layer)), "wb") as f:
            f.write(out)
    ri.write_manifest(root, ri.manifest_json(image_layout, ri.source_fingerprint(layout, source_root), digests))
    return image_layout


def test_layout_puts_names_back_to_back_on_512_byte_boundaries():
    lay = ri.row_image_layout(_segments())
    assert lay.row_bytes == (8192, 2048, 1024, 4096, 1024, 512)  # EXL3_STREAMED_NAMES order
    assert lay.name_offsets == (0, 8192, 10240, 11264, 15360, 16384)
    assert lay.image_bytes == 16896 and lay.row_stride == 20480
    assert all(o % ri.IO_ALIGN == 0 for o in lay.name_offsets)


def test_image_holds_each_segment_where_its_slab_row_holds_it(tmp_path):
    layout, _, data = _checkpoint(tmp_path)
    lay = ri.row_image_layout(_segments())
    r = layout.records[(1, 2)]
    raw = data[r.file_offset : r.file_offset + r.nbytes]
    image = ri.image_of_row(lay, raw)
    for s in _segments():
        n = EXL3_STREAMED_NAMES.index(s.name)
        at = lay.name_offsets[n] + s.dst_offset
        assert image[at : at + s.nbytes] == raw[s.src_offset : s.src_offset + s.nbytes]


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda segs: [s for s in segs if s.name != "w2_svh"], "no segment fills"),
        (lambda segs: [s for s in segs if not (s.name == "w13_suh" and s.part == 0)], "gap or overlap"),
        (lambda segs: [RowSegment(s.name, s.part, s.dst_offset, s.src_offset, s.nbytes - 4) if s.name == "w2_svh" else s for s in segs], "not a multiple of 512"),
        (lambda segs: segs + [RowSegment("w13_mul1", 0, 0, 0, 4)], "not streamed"),
    ],
)
def test_layout_refuses_a_map_a_readv_could_not_land(mutate, message):
    with pytest.raises(ValueError, match=message):
        ri.row_image_layout(mutate(_segments()))


def test_open_accepts_matching_roots_and_names_every_layer_file(tmp_path):
    layout, source, data = _checkpoint(tmp_path)
    roots = [str(tmp_path / "a"), str(tmp_path / "b")]
    for root in roots:
        _build(layout, source, data, root)
    images = ri.open_row_images(roots, layout, _segments(), source, [0, 1])
    assert images.paths[1] == tuple(os.path.join(ri.row_image_dir(r), "layer-001.rows") for r in roots)
    assert images.layout.image_bytes == 16896 and images.num_experts == 3


def _corrupt_manifest(root, edit):
    path = os.path.join(ri.row_image_dir(root), ri.MANIFEST)
    m = json.load(open(path))
    edit(m)
    json.dump(m, open(path, "w"))


@pytest.mark.parametrize(
    "edit, message",
    [
        (lambda m: m.update(complete=False), "not marked complete"),
        (lambda m: m.update(version=99), "format"),
        (lambda m: m["layout"].update(row_stride=4096), "different image layout"),
        (lambda m: m["source"].update(records_sha256="0" * 64), "different source"),
        (lambda m: m["layers"].pop("1"), "no images for layers"),
        (lambda m: m["layers"]["0"]["digests"].__setitem__(0, "0" * 16), "digests differ"),
    ],
)
def test_open_refuses_a_root_whose_manifest_does_not_match(tmp_path, edit, message):
    layout, source, data = _checkpoint(tmp_path)
    roots = [str(tmp_path / "a"), str(tmp_path / "b")]
    for root in roots:
        _build(layout, source, data, root)
    _corrupt_manifest(roots[1], edit)
    with pytest.raises(ValueError, match=message):
        ri.open_row_images(roots, layout, _segments(), source, [0, 1])


def test_open_refuses_a_missing_manifest_or_a_short_file(tmp_path):
    layout, source, data = _checkpoint(tmp_path)
    root = str(tmp_path / "a")
    with pytest.raises(ValueError, match="missing"):
        ri.open_row_images([root], layout, _segments(), source, [0])
    _build(layout, source, data, root)
    with open(os.path.join(ri.row_image_dir(root), "layer-000.rows"), "r+b") as f:
        f.truncate(4096)
    with pytest.raises(ValueError, match="expected"):
        ri.open_row_images([root], layout, _segments(), source, [0])


def test_open_refuses_images_of_a_changed_checkpoint(tmp_path):
    layout, source, data = _checkpoint(tmp_path)
    root = str(tmp_path / "a")
    _build(layout, source, data, root)
    with open(os.path.join(source, "model.safetensors"), "ab") as f:
        f.write(b"x")  # the shard's size no longer matches what the images were built from
    with pytest.raises(ValueError, match="different source"):
        ri.open_row_images([root], layout, _segments(), source, [0])
