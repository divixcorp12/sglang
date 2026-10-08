"""The NVMe-to-RAM prefetch's Python side (spec docs/superpowers/specs/2026-10-08-dsv41-ram-prefetch-design.md,
Phase 1): the host's bounds, the model's router gates registered at load, and the per-row target table the host's
speculative threads score with."""

from __future__ import annotations

# The host's bounds: host/gate_scorer.h GateScorer::kDepth and kMaxPerLayer, host/ram_prefetch.h SpecPool::kMaxShare.
MAX_PER_TOKEN = 12
MAX_PER_LAYER = 8
MAX_SPEC_SHARE = 4
