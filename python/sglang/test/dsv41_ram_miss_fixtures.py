"""A fake EXL3 checkpoint plus per-layer pinned-slab stand-ins for the option C CPU tests."""

from __future__ import annotations

import shutil
from dataclasses import dataclass

import torch

from sglang.srt.layers.moe.exl3_expert_format import EXL3_STREAMED_NAMES, Exl3ExpertFormat
from sglang.srt.layers.moe.exl3_expert_layout import build_exl3_expert_layout
from sglang.srt.layers.moe.exl3_shard_row_source import Exl3ShardRowSource
from sglang.srt.layers.moe.expert_host_tier import allocate_host_slab
from sglang.test.dsv41_fake_exl3 import write_fake_exl3


@dataclass
class RamMissSetup:
    layout: object
    fmt: Exl3ExpertFormat
    specs: dict
    slabs: dict
    tables: object
    roots: tuple = ()  # byte-identical copies of the checkpoint, one per mirror weight

    def reference(self, layer: int, experts: list[int]) -> dict[str, torch.Tensor]:
        """Exl3ShardRowSource's split of ``experts`` of ``layer`` (the byte oracle)."""
        out = {
            name: torch.empty((len(experts),) + self.specs[name].row_shape, dtype=self.specs[name].dtype)
            for name in EXL3_STREAMED_NAMES
        }
        Exl3ShardRowSource.for_layer(self.layout, layer, self.fmt.segment_map(), direct=False).read(
            torch.tensor(experts), out
        )
        return out


def same_bytes(a: torch.Tensor, b: torch.Tensor) -> bool:
    return torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


def ram_miss_setup(
    tmp_path, *, capacity: int = 3, layers: int = 2, experts: int = 6, mirror_weights=None, hidden=None, inter=None
) -> RamMissSetup:
    """``mirror_weights``: one weight per mirror root; copies are made beside ``tmp_path`` and the
    tables split every row across them (``parts == len(mirror_weights)``). ``hidden``/``inter`` size the fake
    experts (write_fake_exl3's defaults when None): larger ones give rows many pages long."""
    from sglang.srt.layers.moe.exl3_ram_miss import exl3_ram_miss_tables
    from sglang.srt.layers.moe.exl3_read_split import StaticSplitPolicy

    dims = {k: v for k, v in (("hidden", hidden), ("inter", inter)) if v is not None}
    write_fake_exl3(str(tmp_path), num_layers=layers, num_experts=experts, **dims)
    layout = build_exl3_expert_layout(str(tmp_path))
    fmt = Exl3ExpertFormat(layout, 0, direct=False)
    specs = {spec.name: spec for spec in fmt.tensor_specs(None)}
    slabs = {
        layer: {
            name: allocate_host_slab(capacity, specs[name].row_shape, specs[name].dtype, register=False)
            for name in EXL3_STREAMED_NAMES
        }
        for layer in range(layers)
    }
    roots = ()
    mirrors = {}
    if mirror_weights is not None:
        roots = tuple(str(tmp_path.parent / f"{tmp_path.name}_mirror{i}") for i in range(len(mirror_weights)))
        for root in roots:
            shutil.copytree(tmp_path, root)
        mirrors = dict(roots=roots, policy=StaticSplitPolicy(mirror_weights), source_root=str(tmp_path))
    tables = exl3_ram_miss_tables(layout, fmt.segment_map(), slabs, **mirrors)
    return RamMissSetup(layout, fmt, specs, slabs, tables, roots)
