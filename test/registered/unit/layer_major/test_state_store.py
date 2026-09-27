import dataclasses
import unittest

import msgspec
import torch

from sglang.srt.layer_major.state_store import FieldSpec, StateStore, state_store_bytes
from sglang.srt.layer_major.tensor_tree import map_tensors


@dataclasses.dataclass
class _Meta:
    a: torch.Tensor
    b: list
    late: torch.Tensor = dataclasses.field(init=False, default=None)


class _S(msgspec.Struct):
    x: torch.Tensor
    n: int


class TestTensorTree(unittest.TestCase):
    def test_maps_every_tensor_and_keeps_shared_tensors_shared(self):
        t = torch.arange(4)
        meta = _Meta(a=t, b=[t, {"k": torch.ones(2)}, (torch.zeros(1),)])
        meta.late = torch.full((3,), 7)
        out = map_tensors(meta, lambda x: x + 1)
        self.assertTrue(torch.equal(out.a, t + 1))
        self.assertIs(out.a, out.b[0])
        self.assertTrue(torch.equal(out.b[1]["k"], torch.full((2,), 2.0)))
        self.assertTrue(torch.equal(out.late, torch.full((3,), 8)))
        self.assertTrue(torch.equal(meta.a, t))  # the input is left alone

    def test_maps_msgspec_structs(self):
        out = map_tensors(_S(x=torch.ones(1), n=3), lambda x: x * 5)
        self.assertEqual(out.n, 3)
        self.assertEqual(out.x.item(), 5.0)


class TestStateStore(unittest.TestCase):
    def _store(self, capacity=16):
        fields = [FieldSpec(name="hidden", per_token_shape=(2, 3), dtype="bfloat16"),
                  FieldSpec(name="prev_pre", per_token_shape=(2,), dtype="float32")]
        return StateStore(fields, capacity, numa_node=None, pin=False)

    def test_bytes_formula(self):
        fields = [FieldSpec(name="h", per_token_shape=(4, 5120), dtype="bfloat16"),
                  FieldSpec(name="p", per_token_shape=(4,), dtype="float32")]
        self.assertEqual(state_store_bytes(fields, 10), 10 * (4 * 5120 * 2 + 4 * 4))

    def test_round_trip_at_offsets(self):
        store = self._store()
        rows = torch.randn(5, 2, 3).to(torch.bfloat16)
        store.write("hidden", 4, rows)
        out = torch.empty(5, 2, 3, dtype=torch.bfloat16)
        store.read_into("hidden", 4, out, stream=None)
        self.assertTrue(torch.equal(out, rows))
        store.write_from("hidden", 4, rows * 2, stream=None)
        store.read_into("hidden", 4, out, stream=None)
        self.assertTrue(torch.equal(out, rows * 2))

    def test_refuses_rows_past_capacity(self):
        store = self._store(capacity=4)
        with self.assertRaises(ValueError):
            store.write("prev_pre", 2, torch.zeros(3, 2))

    def test_park_and_unpark_hold_objects_on_host(self):
        store = self._store()
        store.park(0, {"t": torch.arange(3)})
        got = store.unpark(0, torch.device("cpu"))
        self.assertTrue(torch.equal(got["t"], torch.arange(3)))
        store.clear_parked()
        with self.assertRaises(KeyError):
            store.unpark(0, torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
