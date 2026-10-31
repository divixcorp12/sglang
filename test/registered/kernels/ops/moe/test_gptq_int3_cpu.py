"""Lossless adapter and real typed CPU kernel. No CUDA, server or checkpoint mutation.

GCC builds each supported ISA independently. The same fixture runs through the real Team/JobWindow,
not a Python replacement of the kernel. Optional GPTQ_INT3_REAL_ROOT reads one sealed exported expert.
"""

import ctypes as C
import hashlib
import json
import os
import platform
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import pytest
import torch

from sglang.srt.layers.moe.cpu_experts.gptq_int3 import (
    SLAB_NAMES,
    ExportedExperts,
    GptqInt3CpuTrait,
    load_exported_expert,
    prepare_expert,
    prepare_projection,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=90, suite="base-a-test-cpu")
ROOT = Path(__file__).resolve().parents[5]
CSRC = ROOT / "python/sglang/kernels/jit/csrc"
HERE = Path(__file__).parent
CXX = os.environ.get("CXX") or shutil.which("g++")
NATIVE = platform.system() == "Linux" and platform.machine() == "x86_64" and CXX


def pack(code, dim=0):
    """Independent slow bit-stream writer, deliberately different from the adapter."""
    a = code.movedim(dim, 0).numpy().astype(np.uint32)
    out = np.zeros((len(a) * 3 // 32, *a.shape[1:]), dtype=np.uint32)
    for index, row in enumerate(a):
        word, shift = divmod(index * 3, 32)
        out[word] |= row << shift
        if shift > 29:
            out[word + 1] |= row >> (32 - shift)
    return torch.from_numpy(out.view(np.int32)).movedim(0, dim).contiguous()


def projection(n, k, seed):
    rng = torch.Generator().manual_seed(seed)
    code = torch.randint(0, 8, (n, k), dtype=torch.uint8, generator=rng)
    scale = ((torch.rand((k // 128, n), generator=rng) + 0.1) * 0.02).half()
    scale[:, ::2] *= -1
    t = dict(
        qweight=pack(code.t()),
        qzeros=pack(torch.full((k // 128, n), 3, dtype=torch.uint8), 1),
        scales=scale,
        g_idx=torch.arange(k, dtype=torch.int32) // 128,
    )
    w = (
        (code.to(torch.int16) - 4).float() * scale.t().repeat_interleave(128, 1).float()
    ).half()
    return t, w


@pytest.fixture(scope="module", params=["x86-64", "x86-64-v3", "skylake-avx512"])
def native(request, tmp_path_factory):
    if not NATIVE:
        pytest.skip("requires x86-64 Linux and a C++20 compiler")
    features = Path("/proc/cpuinfo").read_text()
    if request.param != "x86-64" and "avx2" not in features:
        pytest.skip("no AVX2")
    if request.param == "skylake-avx512" and "avx512bw" not in features:
        pytest.skip("no AVX512BW")
    out = tmp_path_factory.mktemp("int3") / "kernel.so"
    subprocess.run(
        [
            CXX,
            "-std=c++20",
            "-O3",
            "-ffp-contract=off",
            "-pthread",
            "-shared",
            "-fPIC",
            f"-march={request.param}",
            f"-I{CSRC}",
            str(HERE / "gptq_int3_cpu_bridge.cpp"),
            str(CSRC / "moe/gptq_int3/kernel.cpp"),
            "-o",
            str(out),
        ],
        check=True,
        capture_output=True,
    )
    lib = C.CDLL(str(out))
    ptr = C.c_void_p
    lib.forward.argtypes = (
        [C.c_int] * 5 + [ptr] * 6 + [C.c_int] * 2 + [C.c_float, C.c_int, ptr]
    )
    lib.forward.restype = C.c_int
    lib.unpack.argtypes = [ptr, ptr, ptr, C.c_int]
    lib.half_convert.argtypes = [ptr, ptr, C.c_int]
    lib.half_roundtrip.argtypes = [ptr]
    return lib


def invoke(lib, slab, x, slots, weights, out, *, threads=3, mode=0, window=0, limit=0):
    spec = GptqInt3CpuTrait(limit).layer_spec(slab, slab[SLAB_NAMES[0]].shape[0])
    pointers = (C.c_void_p * 6)(*(p for p, _ in spec.slabs))
    strides = (C.c_uint64 * 6)(*(b for _, b in spec.slabs))
    err = C.create_string_buffer(512)
    status = lib.forward(
        spec.hidden,
        spec.intermediate,
        spec.capacity,
        x.shape[0],
        slots.shape[1],
        pointers,
        strides,
        x.data_ptr(),
        slots.data_ptr(),
        weights.data_ptr(),
        out.data_ptr(),
        mode,
        threads,
        limit,
        window,
        err,
    )
    return status, err.value.decode()


def fixture(h=256, i=128):
    canonical, decoded = zip(
        *(projection(n, k, s) for n, k, s in ((i, h, 1), (i, h, 2), (h, i, 3)))
    )
    rows = prepare_expert(*(prepare_projection(t) for t in canonical))
    slab = {name: torch.stack((v, v)) for name, v in rows.items()}
    x = torch.linspace(-0.2, 0.2, 3 * h).reshape(3, h).half()
    slots = torch.tensor([[0, 1, 0], [-1, 0, 1], [-1, -1, -1]], dtype=torch.int32)
    weights = torch.tensor(
        [[0.3, 0.4, -0.1], [100, 0, 1], [1, 2, 3]], dtype=torch.float32
    )
    return canonical, decoded, slab, x, slots, weights


def reference(decoded, x, slots, weights, limit=0):
    gate, up, down = decoded
    # Float64 dots avoid hiding a native accumulation error in the oracle. Every declared FP16 boundary is explicit.
    g = (x.double() @ gate.double().t()).half()
    u = (x.double() @ up.double().t()).half()
    if limit:
        g = g.clamp(max=limit)
        u = u.clamp(-limit, limit)
    mid = (torch.nn.functional.silu(g.float()).half() * u).half()
    d = (mid.double() @ down.double().t()).half().float()
    out = torch.zeros_like(d)
    for t in range(len(x)):
        for r in range(slots.shape[1]):
            if slots[t, r] >= 0 and weights[t, r] != 0:
                out[t] += d[t] * weights[t, r]
    return out


def canonical_decode(t):
    # Independent 32-code block decoder (no bit-plane adapter in this oracle).
    packed = (t["qweight"].long() & 0xFFFFFFFF).reshape(-1, 3, t["qweight"].shape[1])
    code = []
    for n in range(32):
        word, bit = divmod(n * 3, 32)
        value = packed[:, word] >> bit
        if bit > 29:
            value = value | (packed[:, word + 1] << (32 - bit))
        code.append(value & 7)
    code = torch.stack(code, dim=1).reshape(-1, packed.shape[-1]).t()
    return (
        (code - 4).float() * t["scales"].t().repeat_interleave(128, 1).float()
    ).half()


def test_adapter_preserves_every_code_and_scale_bit():
    t, w = projection(256, 256, 4)
    p = prepare_projection(t)
    low = p.low2.repeat_interleave(4, 1)
    high = p.high1.repeat_interleave(8, 1)
    code = ((low >> (torch.arange(256) % 4 * 2)) & 3) | (
        ((high >> (torch.arange(256) % 8)) & 1) << 2
    )
    rebuilt = (
        (code.to(torch.int16) - 4).float() * p.scales.repeat_interleave(128, 1).float()
    ).half()
    assert torch.equal(rebuilt.view(torch.int16), w.view(torch.int16))
    assert torch.equal(
        p.scales.view(torch.int16), t["scales"].t().contiguous().view(torch.int16)
    )
    assert p.scales.data_ptr() != t["scales"].data_ptr()


def test_native_repack_equals_reference_adapter():
    """The serving path's native plane repacking (GptqInt3Moe.adapt) must give the reference adapter's exact bytes."""
    if not NATIVE:
        pytest.skip("requires x86-64 Linux and a C++20 compiler")
    from sglang.kernels.ops.moe.gptq_int3_moe import gptq_int3_moe_type

    native = gptq_int3_moe_type()(1)
    for n, k, seed in ((256, 384, 11), (384, 1024, 12)):
        t, _ = projection(n, k, seed)
        reference = prepare_projection(t)
        repacked = prepare_projection(t, repack=native.adapt)
        assert torch.equal(repacked.low2, reference.low2)
        assert torch.equal(repacked.high1, reference.high1)
        assert torch.equal(repacked.scales.view(torch.int16), reference.scales.view(torch.int16))


def test_format_is_explicit_cpu_only_and_keeps_slab_lifetimes():
    from sglang.srt.layers.moe.expert_engine.formats import (
        format_for_quantization_config,
        get_format,
    )

    fmt = get_format("gptq_int3")()
    assert fmt.supports_cpu_experts and not fmt.engine_served
    assert not fmt.supports_draft and not fmt.supports_graph_gather
    assert format_for_quantization_config({"quant_method": "gptq", "bits": 3}) is None
    _, _, slabs, _, _, _ = fixture()
    layer = torch.nn.Module()
    for name, tensor in slabs.items():
        setattr(layer, name, tensor)
    assert fmt.layer_dims(layer) == (2, 256)
    trait = fmt.cpu_trait(source="target")
    spec = trait.layer_spec(slabs, 2)
    assert spec.params == (128,)
    assert trait.act_limit is None and spec.act_limit == 0
    assert tuple(t.data_ptr() for t in spec.keep) == tuple(
        slabs[n].data_ptr() for n in SLAB_NAMES
    )
    with pytest.raises(ValueError):
        fmt.cpu_trait(source="draft")
    with pytest.raises(NotImplementedError):
        fmt.moe_apply(None)


@pytest.mark.parametrize(
    "bad", ["policy", "zero", "nan", "offset", "groups", "dtype", "shape", "overflow"]
)
def test_adapter_rejects_invalid_contract(bad):
    t, _ = projection(128, 128, 0)
    kw = {}
    if bad == "policy":
        kw["bits"] = 4
    elif bad == "zero":
        t["scales"][0, 0] = -0.0
    elif bad == "nan":
        t["scales"][0, 0] = float("nan")
    elif bad == "offset":
        t["qzeros"][0, 0] = 0
    elif bad == "groups":
        t["g_idx"][0] = 1
    elif bad == "dtype":
        t["qweight"] = t["qweight"].long()
    elif bad == "shape":
        t["scales"] = t["scales"][:, :64].contiguous()
    elif bad == "overflow":
        t["scales"][0, 0] = 65504
    with pytest.raises(ValueError):
        prepare_projection(t, **kw)


@pytest.mark.parametrize(
    "bad", [None, "manifest", "policy", "owner", "payload", "escape", "replaced"]
)
def test_export_reader_binds_manifest_payload_and_owner(tmp_path, bad):
    from safetensors.torch import save

    canonical, _, _, _, _, _ = fixture()
    prefix = "model.layers.1.mlp.experts.0."
    if bad == "owner":
        prefix = "model.layers.1.mlp.experts.1."
    tensors = {
        prefix + n + "." + f: t
        for n, p in zip(("gate_proj", "up_proj", "down_proj"), canonical)
        for f, t in p.items()
    }
    relative = "experts/layer-01/expert-000/weights.safetensors"
    target = tmp_path / relative
    target.parent.mkdir(parents=True)
    raw = save(tensors)
    target.write_bytes(raw)
    manifest = {
        "status": "completed",
        "format": "mimo-hybrid-gptq-v1",
        "expert_policy": {
            "quant_method": "gptq",
            "bits": 3,
            "group_size": 128,
            "sym": True,
            "desc_act": False,
        },
        "files": {
            relative: {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
        },
    }
    if bad == "policy":
        manifest["expert_policy"]["group_size"] = 64
    encoded = json.dumps(manifest).encode()
    (tmp_path / "export-manifest.json").write_bytes(encoded)
    expected = hashlib.sha256(encoded).hexdigest()
    if bad == "manifest":
        expected = "0" * 64
    elif bad == "payload":
        target.write_bytes(raw[:-1] + bytes([raw[-1] ^ 1]))
    elif bad == "escape":
        outside = tmp_path.parent / (tmp_path.name + "-outside.safetensors")
        outside.write_bytes(raw)
        target.unlink()
        target.symlink_to(outside)
    elif bad == "replaced":
        # A reload skips the SHA only for the file it already hashed; a same-size swap must be hashed and refused.
        reader = ExportedExperts(tmp_path, expected_manifest_sha256=expected)
        reader.load(1, 0)
        swapped = target.with_suffix(".new")
        swapped.write_bytes(raw[:-1] + bytes([raw[-1] ^ 1]))
        os.replace(swapped, target)
        with pytest.raises(ValueError):
            reader.load(1, 0)
        return
    if bad is not None:
        with pytest.raises(ValueError):
            load_exported_expert(tmp_path, 1, 0, expected_manifest_sha256=expected)
    else:
        actual = load_exported_expert(tmp_path, 1, 0, expected_manifest_sha256=expected)
        reference = prepare_expert(*(prepare_projection(p) for p in canonical))
        for name in SLAB_NAMES:
            assert torch.equal(
                actual[name].reshape(-1).view(torch.uint8),
                reference[name].reshape(-1).view(torch.uint8),
            )


def test_half_and_bestla_bits(native):
    half = np.empty(65536, dtype=np.uint16)
    native.half_roundtrip(half.ctypes.data)
    expected = np.arange(65536, dtype=np.uint16)
    finite = (expected & 0x7C00) != 0x7C00
    np.testing.assert_array_equal(half[finite], expected[finite])
    rng = np.random.default_rng(42)
    values = rng.integers(0, 2**32, 100000, dtype=np.uint32).view(np.float32)
    values = values[np.isfinite(values)].copy()
    # Exact ties at every positive finite adjacent half pair, and both neighboring float32 values, both signs.
    halves = np.arange(0x7BFF + 1, dtype=np.uint16).view(np.float16).astype(np.float32)
    ties = (halves[:-1] + halves[1:]) * 0.5
    edges = np.concatenate(
        (ties, np.nextafter(ties, -np.inf), np.nextafter(ties, np.inf))
    )
    values = np.concatenate((values, edges, -edges))
    actual = np.empty(len(values), dtype=np.uint16)
    native.half_convert(values.ctypes.data, actual.ctypes.data, len(values))
    with np.errstate(over="ignore", invalid="ignore"):
        np.testing.assert_array_equal(actual, values.astype(np.float16).view(np.uint16))
    # Exhaust all byte values and their differing neighbors, not just one repeated code.
    lo = rng.integers(0, 256, 4096, dtype=np.uint8)
    hi = rng.integers(0, 256, 2048, dtype=np.uint8)
    codes = np.empty(16384, dtype=np.int8)
    native.unpack(lo.ctypes.data, hi.ctypes.data, codes.ctypes.data, len(codes))
    idx = np.arange(len(codes))
    ref = (
        ((lo[idx // 4] >> (2 * (idx % 4))) & 3)
        | (((hi[idx // 8] >> (idx % 8)) & 1) << 2)
    ).astype(np.int16) - 4
    np.testing.assert_array_equal(codes, ref)


@pytest.mark.parametrize("mode", [0, 1, 2])
@pytest.mark.parametrize("limit", [0, 0.03])
def test_real_kernel_route_order_modes_and_job_window(native, mode, limit):
    _, decoded, slab, x, slots, weights = fixture()
    expected = reference(decoded, x, slots, weights, limit)
    out = torch.full_like(expected, 0.125)
    status, error = invoke(native, slab, x, slots, weights, out, mode=mode, limit=limit)
    assert status == 0, error
    if mode == 2:
        expected += 0.125
    torch.testing.assert_close(out, expected, atol=2e-6, rtol=0.003)
    # Same-output accumulation dependency plus four independently owned Job states and closed down gates.
    concurrent = torch.full_like(expected, 0.125)
    status, error = invoke(
        native, slab, x, slots, weights, concurrent, mode=2, window=2, limit=limit
    )
    assert status == 0, error
    torch.testing.assert_close(
        concurrent,
        0.125 + reference(decoded, x, slots, weights, limit) * 4,
        atol=8e-6,
        rtol=0.003,
    )


@pytest.mark.parametrize(
    "bad",
    ["x", "scale", "down", "gate_overflow", "slot", "weight", "accumulate", "mode"],
)
def test_native_refuses_without_output_write(native, bad):
    _, _, slab, x, slots, weights = fixture()
    if bad == "x":
        x[0, 0] = float("nan")
    elif bad == "scale":
        slab["w13_scales"][0, 0, 0] = 0
    elif bad == "down":
        slab["w2_scales"][0, 0, 0] = float("inf")
    elif bad == "gate_overflow":
        slab["w13_scales"][0, 0, 0] = 65504
    elif bad == "slot":
        slots[0, 0] = 2
    elif bad == "weight":
        weights[0, 0] = float("nan")
    out = torch.full((3, 256), 123.0, dtype=torch.float32)
    mode = 0
    if bad == "accumulate":
        out[2, 0] = float("inf")
        mode = 2
    elif bad == "mode":
        mode = 9
    original = out.clone()
    status, error = invoke(native, slab, x, slots, weights, out, limit=0.03, mode=mode)
    assert status == 2, error
    assert torch.equal(out, original)


def test_native_bounds_scratch_before_reading_call_data(native):
    h, i = 65536, 128
    ptrs = (C.c_void_p * 6)(*([1] * 6))
    strides = (C.c_uint64 * 6)(
        2 * i * h // 4,
        2 * i * h // 8,
        2 * i * h // 128 * 2,
        h * i // 4,
        h * i // 8,
        h * i // 128 * 2,
    )
    out = torch.tensor([123.0])
    error = C.create_string_buffer(512)
    status = native.forward(
        h,
        i,
        1,
        4096,
        0,
        ptrs,
        strides,
        1,
        None,
        None,
        out.data_ptr(),
        0,
        1,
        0,
        0,
        error,
    )
    assert status == 2 and b"scratch" in error.value, error.value
    assert out.item() == 123.0


@pytest.mark.parametrize("owner", [(1, 0), (32, 17), (69, 0)])
def test_actual_exported_expert(native, owner):
    path = os.environ.get("GPTQ_INT3_REAL_ROOT")
    if not path:
        pytest.skip("set GPTQ_INT3_REAL_ROOT for a read-only exported expert check")
    from safetensors.torch import load_file

    layer, expert = owner
    bundle = Path(path) / f"experts/layer-{layer:02d}/expert-{expert:03d}"
    tensor = load_file(str(bundle / "weights.safetensors"))
    # Actual key naming is explicit; no fallback to another precision or projection.
    projections = []
    for name in ("gate_proj", "up_proj", "down_proj"):
        suffix = f".{name}."
        projections.append(
            {k.split(suffix)[1]: v for k, v in tensor.items() if suffix in k}
        )
    rows = load_exported_expert(
        path,
        layer,
        expert,
        expected_manifest_sha256="7b1445ed9cd90d94cfc98e3cde385c618d4a133dc8adcd0cf031f890aa6b5c65",
    )
    slab = {k: v.unsqueeze(0) for k, v in rows.items()}
    decoded = [canonical_decode(p) for p in projections]
    h = decoded[0].shape[1]
    x = (torch.randn(2, h, generator=torch.Generator().manual_seed(42)) * 0.02).half()
    slots = torch.zeros((2, 1), dtype=torch.int32)
    weights = torch.ones((2, 1))
    out = torch.empty((2, h))
    status, error = invoke(native, slab, x, slots, weights, out, threads=8)
    assert status == 0, error
    expected = reference(decoded, x, slots, weights)
    rel = torch.linalg.vector_norm(out - expected) / torch.linalg.vector_norm(expected)
    assert rel < 0.002, float(rel)
    timings = []
    for _ in range(3):
        start = time.perf_counter()
        status, error = invoke(native, slab, x, slots, weights, out, threads=8)
        assert status == 0, error
        timings.append(time.perf_counter() - start)
    print(
        json.dumps(
            dict(
                expert=f"layer{layer}/expert{expert}",
                relative_l2=float(rel),
                warm_ms=min(timings) * 1000,
                rows=2,
                threads=8,
                sha256=hashlib.sha256(
                    (bundle / "weights.safetensors").read_bytes()
                ).hexdigest(),
            )
        )
    )


def test_model_layer_cpu_experts_on_real_export(monkeypatch):
    """The model's MoE stand-in (verified reader, LRU slots, JIT FFI Team) against the independent decoder.

    Eight slots serve four rows of eight distinct real experts each, so every call evicts the previous row's experts;
    a slot mix-up, a stale lookup or a skipped padding route changes the sum.
    """
    path = os.environ.get("GPTQ_INT3_REAL_ROOT")
    if not path or not NATIVE:
        pytest.skip("set GPTQ_INT3_REAL_ROOT on x86-64 Linux for the model-layer path")
    from types import SimpleNamespace

    from safetensors.torch import load_file

    from sglang.srt.environ import envs
    from sglang.srt.layers.moe.cpu_experts import gptq_int3_moe

    monkeypatch.setattr(gptq_int3_moe, "_RUNTIME", None)
    config = SimpleNamespace(
        name_or_path=path,
        experimental_quantization=dict(
            format="mimo-hybrid-gptq-v1",
            routed_experts=dict(quant_method="gptq", bits=3, group_size=128, sym=True, desc_act=False),
        ),
    )
    layer = 32
    ids = torch.arange(32, dtype=torch.int64).reshape(4, 8) * 9 % 288
    ids[2, 5] = -1
    weights = torch.rand((4, 8), generator=torch.Generator().manual_seed(7)) / 4
    with (
        envs.SGLANG_GPTQ_INT3_MANIFEST_SHA256.override(
            "7b1445ed9cd90d94cfc98e3cde385c618d4a133dc8adcd0cf031f890aa6b5c65"
        ),
        envs.SGLANG_GPTQ_INT3_EXPERTS_PER_LAYER.override(8),
        envs.SGLANG_GPTQ_INT3_THREADS.override(8),
        envs.SGLANG_GPTQ_INT3_LOAD_THREADS.override(4),
    ):
        experts = gptq_int3_moe.GptqInt3CpuExperts(
            config=config, layer_id=layer, num_experts=288, top_k=8, hidden_size=6144, intermediate_size=2048
        )
    assert experts.rows_per_call == 1
    x = (torch.randn(4, 6144, generator=torch.Generator().manual_seed(3)) * 0.02).half().float()
    topk = SimpleNamespace(topk_ids=ids, topk_weights=weights)
    first = experts(x, topk)
    second = experts(x, topk)
    assert torch.equal(first, second)

    expected = torch.zeros_like(first)
    for t in range(4):
        for r in range(8):
            e = int(ids[t, r])
            if e < 0:
                continue
            tensor = load_file(f"{path}/experts/layer-{layer:02d}/expert-{e:03d}/weights.safetensors")
            decoded = [
                canonical_decode({k.split(f".{n}.")[1]: v for k, v in tensor.items() if f".{n}." in k})
                for n in ("gate_proj", "up_proj", "down_proj")
            ]
            one = reference(decoded, x[t : t + 1].half(), torch.zeros((1, 1), dtype=torch.int32), weights[t : t + 1, r : r + 1])
            expected[t] += one[0]
    rel = torch.linalg.vector_norm(first - expected) / torch.linalg.vector_norm(expected)
    assert rel < 0.002, float(rel)
    print(json.dumps(dict(model_layer=layer, relative_l2=float(rel), loads=experts.runtime.loads)))


def test_native_sanitizers(tmp_path):
    if not NATIVE:
        pytest.skip("requires Linux x86-64 compiler")
    exe = tmp_path / "sanitized"
    subprocess.run(
        [
            os.environ.get("GPTQ_INT3_SANITIZER_CXX", CXX),
            "-std=c++20",
            "-O1",
            "-g",
            "-pthread",
            "-march=x86-64",
            f"-I{CSRC}",
            "-DGPTQ_INT3_STANDALONE",
            "-fsanitize=address,undefined",
            "-fno-omit-frame-pointer",
            "-fno-sanitize-recover=all",
            str(HERE / "gptq_int3_cpu_bridge.cpp"),
            str(CSRC / "moe/gptq_int3/kernel.cpp"),
            "-o",
            str(exe),
        ],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        [str(exe)],
        check=True,
        timeout=60,
        env={**os.environ, "ASAN_OPTIONS": "detect_leaks=1"},
    )
