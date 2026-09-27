"""The two seams of the layer-major strategy: the model adapter and expert residency. The strategy knows nothing
else about the model or its quantization."""

from __future__ import annotations

from typing import Any, Protocol

from sglang.srt.layer_major.state_store import FieldSpec, StateStore


class LayerMajorModelAdapter(Protocol):
    def field_specs(self) -> list[FieldSpec]: ...

    def begin_pass(self, forward_batch: Any, schedule_batch: Any, store: StateStore) -> Any: ...

    def layer_ids(self, handle: Any) -> range: ...

    def num_chunks(self, handle: Any) -> int: ...

    def run_layer(self, handle: Any, layer_id: int, chunk: int, store: StateStore) -> None: ...

    def finish_pass(self, handle: Any, store: StateStore) -> Any: ...

    def release_pass(self, handle: Any, store: StateStore, *, failed: bool) -> None: ...


class ExpertResidency(Protocol):
    def begin(self, layer_ids: range) -> None: ...

    def make_resident(self, layer_id: int) -> None: ...

    def restore(self) -> None: ...

    def end(self) -> None: ...


class NullResidency:
    """Experts come through the model's normal path; nothing is borrowed, so nothing is restored."""

    def begin(self, layer_ids: range) -> None:
        pass

    def make_resident(self, layer_id: int) -> None:
        pass

    def restore(self) -> None:
        pass

    def end(self) -> None:
        pass
