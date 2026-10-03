"""Compile/test the actual C ABI against independently decoded dense experts.

Runnable with stdlib unittest: no CUDA, torch, SGLang import or pytest needed.
"""

import ctypes as C
import importlib.util
import math
import os
from pathlib import Path
import random
import struct
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[4]
SRC = ROOT / "python/sglang/srt/layers/quantization/nvfp4_cpu/optimized"


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


native = module("nvfp4_native", SRC / "native.py")
build = module("nvfp4_build", SRC / "build.py").build
FP4 = [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6]


def decode_scale(b):
    sign = -1 if b >= 128 else 1
    e, m = (b & 127) >> 3, b & 7
    if e == 15 and m == 7:
        return math.nan
    return sign * ((m / 8) * 2**-6 if e == 0 else (1 + m / 8) * 2 ** (e - 7))


class Fixture:
    def __init__(self, h=80, n=80, layout=0, limit=0, separate=False):
        self.h, self.n, self.cap = h, n, 3
        self.rng = random.Random(920 + layout)
        self.buffers, self.dense, self.raw_scales = [], [], []
        for rows, cols in ((2 * n, h), (h, n)):
            physical = list(range(rows))
            if rows == 2 * n:
                if layout == 1:
                    physical = physical[n:] + physical[:n]
                if layout == 2:
                    physical = [
                        i
                        for start in range(0, n, 64)
                        for half in (n, 0)
                        for i in range(start + half, start + half + 64)
                    ]
            matrices, packed_slots, scale_slots, raw_scales = [], [], [], []
            for slot in range(self.cap):
                codes = [
                    [self.rng.randrange(16) for _ in range(cols)] for _ in range(rows)
                ]
                # Include distinct subnormal/normal scales, and cross every
                # 128-row/4-column scale tile boundary.
                scales = [
                    [
                        self.rng.choice([1, 7, 8, 12, 16, 20, 24, 28, 32])
                        for _ in range(cols // 16)
                    ]
                    for _ in range(rows)
                ]
                matrices.append(
                    [
                        [
                            FP4[q] * decode_scale(scales[r][c // 16])
                            for c, q in enumerate(row)
                        ]
                        for r, row in enumerate(codes)
                    ]
                )
                raw_scales.append([scales[r] for r in physical])
                packed = bytearray()
                for r in physical:
                    packed.extend(
                        codes[r][c] | codes[r][c + 1] << 4 for c in range(0, cols, 2)
                    )
                # Independent reshape/transpose, expressed by iteration over
                # output tensor axes [row_tile,k_tile,lane,row_group,k_inner].
                padded_rows = (rows + 127) // 128 * 128
                padded_cols = (cols // 16 + 3) // 4 * 4
                sf = bytearray()
                for rt in range(padded_rows // 128):
                    for kt in range(padded_cols // 4):
                        for lane in range(32):
                            for rg in range(4):
                                for ki in range(4):
                                    r, c = rt * 128 + rg * 32 + lane, kt * 4 + ki
                                    sf.append(
                                        scales[physical[r]][c]
                                        if r < rows and c < cols // 16
                                        else 0
                                    )
                packed_slots.append(packed)
                scale_slots.append(sf)
            self.dense.append(matrices)
            self.raw_scales.append(raw_scales)
            self.buffers.append([packed_slots, scale_slots])
        self.arrays = []
        self.desc = native.LayerDescriptor(
            abi_version=1,
            capacity=self.cap,
            hidden=h,
            intermediate=n,
            w13_layout=layout,
            activation=0,
            act_limit=limit,
            inv_input_scale13=0.5,
            inv_input_scale2=0.25,
        )
        for idx, slots in (
            (0, self.buffers[0][0]),
            (1, self.buffers[1][0]),
            (2, self.buffers[0][1]),
            (3, self.buffers[1][1]),
        ):
            # Padded slot strides exercise addressing independent of shape.
            data = b"".join(bytes(b) + b"\xde" * 16 for b in slots)
            arr = (C.c_uint8 * len(data)).from_buffer_copy(data)
            self.arrays.append(arr)
            self.desc.slabs[idx] = C.addressof(arr)
            self.desc.slot_bytes[idx] = len(slots[0]) + 16
        self.gate_alpha = [1, 0.75, 1.25]
        self.down_alpha = [0.5, 1, 1.5]
        self.up_alpha = [0.25, 1.25, 0.5] if separate else self.gate_alpha
        for idx, values in (
            (4, self.gate_alpha),
            (5, self.down_alpha),
            (6, self.up_alpha),
        ):
            if idx == 6 and not separate:
                continue
            arr = (C.c_float * self.cap)(*values)
            self.arrays.append(arr)
            self.desc.slabs[idx], self.desc.slot_bytes[idx] = C.addressof(arr), 4
        self.x_values = [
            struct.unpack("e", struct.pack("e", self.rng.uniform(-0.3, 0.3)))[0]
            for _ in range(h)
        ]
        self.x = C.create_string_buffer(
            b"".join(struct.pack("e", v) for v in self.x_values)
        )
        self.limit = limit

    def reference(self, slots, weights):
        out = [0.0] * self.h
        for slot, route in zip(slots, weights):
            if slot == -1:
                continue
            matrix = self.dense[0][slot]
            vals = [
                math.fsum(v * x for v, x in zip(row, self.x_values)) for row in matrix
            ]
            intermediate = []
            for i in range(self.n):
                g = vals[i] * self.gate_alpha[slot] * 0.5
                u = vals[self.n + i] * self.up_alpha[slot] * 0.5
                if self.limit:
                    g, u = min(g, self.limit), max(-self.limit, min(u, self.limit))
                sigmoid = (
                    1 / (1 + math.exp(-g))
                    if g >= 0
                    else math.exp(g) / (1 + math.exp(g))
                )
                intermediate.append(g * sigmoid * u)
            for i, row in enumerate(self.dense[1][slot]):
                out[i] += (
                    math.fsum(v * x for v, x in zip(row, intermediate))
                    * self.down_alpha[slot]
                    * 0.25
                    * route
                )
        return out


class TestNvfp4CpuNative(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.path = build(
            Path(cls.tmp.name) / "nvfp4.so",
            native=os.environ.get("NVFP4_TEST_NATIVE") == "1",
        )
        cls.api = native.NativeApi(cls.path)
        if hasattr(os, "sched_getaffinity"):
            cores = sorted(os.sched_getaffinity(0))[:3]
            if len(cores) == 3:
                cls.api.check(
                    cls.api.set_cores((C.c_int32 * 3)(*cores), 3), "set_cores"
                )

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def register(self, f):
        h = C.c_int64(-1)
        self.assertEqual(self.api.register(C.byref(f.desc), C.byref(h)), 0)
        self.addCleanup(lambda: self.api.free(h.value))
        return h.value

    def forward(self, handle, f, slots, weights, out=None, threads=3, accumulate=0):
        out = out if out is not None else (C.c_float * f.h)(*([123.0] * f.h))
        status = self.api.forward(
            handle,
            C.addressof(f.x),
            (C.c_int32 * len(slots))(*slots),
            (C.c_float * len(weights))(*weights),
            len(slots),
            out,
            threads,
            accumulate,
        )
        return status, out

    def close_values(self, actual, expected):
        for a, b in zip(actual, expected):
            self.assertAlmostEqual(a, b, delta=3e-5 * max(1.0, abs(b)))

    def test_01_layouts_padding_routing_and_distinct_scales(self):
        for layout in (0, 1, 2):
            f = Fixture(
                h=144, n=192 if layout == 2 else 80, layout=layout, separate=True
            )
            handle = self.register(f)
            slots, weights = [2, -1, 0, 2, 1], [0.75, 1, -0.5, 0.125, 0]
            status, out = self.forward(handle, f, slots, weights)
            self.assertEqual(status, 0)
            self.close_values(out, f.reference(slots, weights))

    def test_clamp_accumulate_and_thread_counts(self):
        f = Fixture(limit=0.025)
        handle = self.register(f)
        expected = f.reference([1, 0], [0.5, 0.25])
        for threads in (1, 2, 3):
            status, out = self.forward(handle, f, [1, 0], [0.5, 0.25], threads=threads)
            self.assertEqual(status, 0)
            self.close_values(out, expected)
            status, out = self.forward(
                handle, f, [1, 0], [0.5, 0.25], out, threads=threads, accumulate=1
            )
            self.assertEqual(status, 0)
            self.close_values(out, [2 * v for v in expected])

    def test_live_slot_updates_no_reregistration(self):
        f = Fixture()
        handle = self.register(f)
        status, before = self.forward(handle, f, [1], [1])
        self.assertEqual(status, 0)
        C.cast(f.desc.slabs[5], C.POINTER(C.c_float))[1] *= 2
        f.down_alpha[1] *= 2
        status, after = self.forward(handle, f, [1], [1])
        self.assertEqual(status, 0)
        self.close_values(after, [2 * v for v in before])
        self.close_values(after, f.reference([1], [1]))
        # Change packed weights in a reusable slot, not just its scale.
        C.memset(f.desc.slabs[0] + f.desc.slot_bytes[0], 0, f.h * f.n)
        status, out = self.forward(handle, f, [1], [1])
        self.assertEqual(status, 0)
        self.assertEqual(list(out), [0.0] * f.h)

    def test_empty_and_skipped_lanes(self):
        f = Fixture(h=16, n=16)
        handle = self.register(f)
        for slots, weights in (([], []), ([-1], [1]), ([0], [0])):
            status, out = self.forward(handle, f, slots, weights)
            self.assertEqual(status, 0)
            self.assertEqual(list(out), [0.0] * f.h)
            out = (C.c_float * f.h)(*([3.0] * f.h))
            status, out = self.forward(handle, f, slots, weights, out, accumulate=1)
            self.assertEqual(status, 0)
            self.assertEqual(list(out), [3.0] * f.h)

    def test_invalid_jobs_do_not_modify_output(self):
        f = Fixture(h=16, n=16)
        handle = self.register(f)
        for slots, weights in (
            ([3], [1]),
            ([-2], [1]),
            ([0], [math.nan]),
            ([0] * 9, [1] * 9),
        ):
            status, out = self.forward(handle, f, slots, weights)
            self.assertEqual(status, 2)
            self.assertEqual(list(out), [123.0] * f.h)
        self.assertEqual(self.api.free(handle), 0)
        status, out = self.forward(handle, f, [0], [1])
        self.assertEqual(status, 2)
        self.assertEqual(list(out), [123.0] * f.h)

    def test_bad_descriptors(self):
        f = Fixture(h=16, n=16)
        for field, value in (
            ("abi_version", 2),
            ("hidden", 17),
            ("capacity", 0),
            ("activation", 1),
            ("w13_layout", 3),
            ("w13_layout", 2),
            ("inv_input_scale13", 0),
            ("inv_input_scale2", math.nan),
            ("act_limit", -1),
        ):
            old = getattr(f.desc, field)
            setattr(f.desc, field, value)
            self.assertEqual(
                self.api.register(C.byref(f.desc), C.byref(C.c_int64())), 2, field
            )
            setattr(f.desc, field, old)
        f.desc.slot_bytes[2] = 1
        self.assertEqual(self.api.register(C.byref(f.desc), C.byref(C.c_int64())), 2)

    def test_native_affinity_cannot_change_after_workers_start(self):
        # Deliberately first run a job: this test also works when selected alone.
        f = Fixture(h=16, n=16)
        handle = self.register(f)
        self.assertEqual(self.forward(handle, f, [0], [1])[0], 0)
        self.assertEqual(self.api.set_cores((C.c_int32 * 1)(0), 1), 2)

    def test_all_finite_e4m3_codes_and_fp4_nibbles(self):
        # H=N=16, constant FP4 gate/up/down rows: dense oracle checks every
        # finite scale code (including negative, zero, max normal, subnormal).
        f = Fixture(h=16, n=16)
        handle = self.register(f)
        for code in range(256):
            if code & 127 == 127:
                continue
            # Only gate/up scales vary; down scale is a small fixed normal.
            C.memset(f.desc.slabs[2], code, f.desc.slot_bytes[2] - 16)
            for m in range(32):
                for k in range(16):
                    f.dense[0][0][m][k] = FP4[
                        (f.buffers[0][0][0][m * 8 + k // 2] >> (4 * (k % 2))) & 15
                    ] * decode_scale(code)
            status, out = self.forward(handle, f, [0], [1])
            self.assertEqual(status, 0)
            self.close_values(out, f.reference([0], [1]))


try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "torch not installed; native C ABI tests still run")
class TestNvfp4CpuTrait(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.path = build(
            Path(cls.tmp.name) / "nvfp4-trait.so",
            native=os.environ.get("NVFP4_TEST_NATIVE") == "1",
        )
        # Import the real adapter without loading SGLang's unrelated runtime.
        key = "sglang.srt.layers.quantization.nvfp4_cpu.optimized.native"
        old = sys.modules.get(key)
        sys.modules[key] = native
        try:
            cls.Trait = module(
                "nvfp4_trait",
                ROOT / "python/sglang/srt/layers/moe/cpu_experts/nvfp4.py",
            ).Nvfp4CpuQuantTrait
        finally:
            if old is None:
                del sys.modules[key]
            else:
                sys.modules[key] = old

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def slabs(self, f, separate=False):
        names = self.Trait.slab_names + (("g1_alphas_up",) if separate else ())
        shape = [
            (f.cap, 2 * f.n, f.h // 2),
            (f.cap, f.h, f.n // 2),
            (f.cap, (2 * f.n + 127) // 128 * 128, (f.h // 16 + 3) // 4 * 4),
            (f.cap, (f.h + 127) // 128 * 128, (f.n // 16 + 3) // 4 * 4),
        ]
        slabs = {}
        for i, name in enumerate(names):
            if i < 4:
                size = math.prod(shape[i][1:])
                b = b"".join(
                    C.string_at(f.desc.slabs[i] + s * f.desc.slot_bytes[i], size)
                    for s in range(f.cap)
                )
                t = (
                    torch.frombuffer(bytearray(b), dtype=torch.uint8)
                    .clone()
                    .reshape(shape[i])
                )
                if i >= 2:
                    t = t.view(torch.float8_e4m3fn)
            else:
                t = torch.tensor(
                    list(C.cast(f.desc.slabs[i], C.POINTER(C.c_float))[: f.cap]),
                    dtype=torch.float32,
                )
            slabs[name] = t
        return slabs

    def test_trait_uses_real_tensor_views_and_native_address(self):
        for layout, name in enumerate(("gate_up", "up_gate", "up_gate_interleaved64")):
            f = Fixture(h=144, n=128, layout=layout, separate=True)
            trait = self.Trait(
                self.path,
                w13_layout=name,
                inv_input_scale13=0.5,
                inv_input_scale2=0.25,
                act_limit=0,
                separate_up_alpha=True,
            )
            slabs = self.slabs(f, separate=True)
            h = trait.register_layer(slabs, f.cap)
            self.addCleanup(lambda trait=trait, h=h: trait.free_layer(h))
            x = torch.tensor([f.x_values], dtype=torch.float16)
            slots, weights = (
                torch.tensor([[2, -1, 0]]),
                torch.tensor([[0.5, 1, 0.25]], dtype=torch.float32),
            )
            out = torch.empty((1, f.h), dtype=torch.float32)
            trait.forward(h, x, slots, weights, out, 3)
            torch.testing.assert_close(
                out[0],
                torch.tensor(f.reference([2, -1, 0], [0.5, 1, 0.25])),
                rtol=3e-5,
                atol=3e-5,
            )
            self.assertEqual(
                trait.native_forward(), C.cast(trait.api.forward, C.c_void_p).value
            )
            self.assertIs(trait._slabs[h][0], slabs["w13_weight"])
            slabs["w13_weight"][2].zero_()
            trait.forward(h, x, torch.tensor([[2]]), torch.ones((1, 1)), out, 3)
            self.assertEqual(out.abs().max().item(), 0)

    def test_production_swizzle_is_byte_identical(self):
        import ast
        from typing import Optional, Union

        # Execute the checkout's function unchanged, avoiding utils.py's GPU
        # imports. This verifies its actual reshape/permute, not a copy of it.
        path = ROOT / "python/sglang/srt/layers/quantization/utils.py"
        tree = ast.parse(path.read_text())
        function = next(
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "swizzle_blockscale"
        )
        scope = {"torch": torch, "Optional": Optional, "Union": Union}
        exec(
            compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"),
            scope,
        )
        f = Fixture(h=144, n=192, layout=2)
        for i in range(2):
            raw = torch.tensor(f.raw_scales[i], dtype=torch.uint8).view(
                torch.float8_e4m3fn
            )
            actual = scope["swizzle_blockscale"](raw).view(torch.uint8)
            expected = torch.tensor(
                [list(b) for b in f.buffers[i][1]], dtype=torch.uint8
            ).reshape(actual.shape)
            self.assertTrue(torch.equal(actual, expected))

    def test_trait_rejects_unsupported_metadata_and_shapes(self):
        with self.assertRaises(ValueError):
            self.Trait(
                self.path, w13_layout="marlin", inv_input_scale13=1, inv_input_scale2=1
            )
        with self.assertRaises(ValueError):
            self.Trait(
                self.path,
                w13_layout="gate_up",
                inv_input_scale13=[1],
                inv_input_scale2=1,
            )
        f = Fixture(h=16, n=16)
        trait = self.Trait(
            self.path, w13_layout="gate_up", inv_input_scale13=1, inv_input_scale2=1
        )
        slabs = self.slabs(f)
        with self.assertRaises(ValueError):
            trait.register_layer(slabs, f.cap)  # Unspecified clamp is refused.
        trait.act_limit = 0
        slabs["w13_blockscale_swizzled"] = slabs["w13_blockscale_swizzled"].view(
            torch.uint8
        )
        with self.assertRaises(ValueError):
            trait.register_layer(slabs, f.cap)


if __name__ == "__main__":
    unittest.main()
