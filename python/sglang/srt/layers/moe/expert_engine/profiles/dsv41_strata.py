"""DSV4.1 architecture strata of a model layer id.

Stdlib only, so the serving header writer and the offline readers share one definition. Loaded by file path from
analysis/dsv41-drive/dspark/spec_margin.py, which must not import torch.
"""


def stratum(layer: int) -> str:
    """The handoff's architecture strata, by target layer: Engram sits at 1 and 14, the encoder ends at 19, the
    bounded prefill replay starts at 21, DSpark taps 37-39."""
    if layer in (1, 14):
        return "enters engram"
    if layer in (2, 15):
        return "leaves engram"
    if layer == 20:
        return "19->20"
    if layer == 21:
        return "replay start"
    if layer >= 37:
        return "late"
    return "interior"
