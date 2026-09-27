"""Copy a nested structure with every tensor replaced, to park per-chunk state off the device and bring it back."""

from __future__ import annotations

import copy
import dataclasses
from typing import Any, Callable

import msgspec
import torch


def map_tensors(obj: Any, fn: Callable[[torch.Tensor], torch.Tensor]) -> Any:
    return _map(obj, fn, {})


def _map(obj: Any, fn: Callable[[torch.Tensor], torch.Tensor], memo: dict[int, Any]) -> Any:
    if id(obj) in memo:
        return memo[id(obj)]
    if isinstance(obj, torch.Tensor):
        out = fn(obj)
    elif dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        out = copy.copy(obj)
        memo[id(obj)] = out
        for f in dataclasses.fields(obj):
            # An init=False field without a default is absent until someone sets it.
            if f.name in obj.__dict__:
                setattr(out, f.name, _map(obj.__dict__[f.name], fn, memo))
        return out
    elif isinstance(obj, msgspec.Struct):
        out = copy.copy(obj)
        memo[id(obj)] = out
        for name in obj.__struct_fields__:
            setattr(out, name, _map(getattr(obj, name), fn, memo))
        return out
    elif isinstance(obj, dict):
        out = {k: _map(v, fn, memo) for k, v in obj.items()}
    elif isinstance(obj, list):
        out = [_map(v, fn, memo) for v in obj]
    elif isinstance(obj, tuple):
        items = [_map(v, fn, memo) for v in obj]
        out = type(obj)(*items) if hasattr(type(obj), "_fields") else tuple(items)
    else:
        return obj
    memo[id(obj)] = out
    return out
