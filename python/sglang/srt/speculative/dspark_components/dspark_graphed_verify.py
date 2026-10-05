"""DSpark's target verify in the breakable decode graph with EXL3 expert caching (DSV41_REFERENCE.md §33.8).

A verify's graph gather serves at most W distinct misses per layer. A verify with more raises a sticky device flag,
and its output (logits, epilogue buffers, draft KV) is not a verify result. The verify is then re-run with the decode
graph and the narrowed gather both off: the EXL3 eager MoE, the path DSpark verify ran on before D2.
"""

from sglang.srt.environ import envs


def forward_verify_with_reverify(model_runner, forward):
    """Run a target verify; re-run it eagerly when its graphed expert gather overflowed.

    ``forward`` runs the verify and returns an object with ``can_run_cuda_graph``. The flag is read only after a
    graphed verify on a narrowed gather: one host read, before anything reads the logits.
    """
    out = forward()
    manager = getattr(model_runner, "expert_hot_cache_manager", None)
    if manager is None or not manager.narrow_graph_gather or not out.can_run_cuda_graph:
        return out
    overflowed = manager.take_verify_overflow()
    if not overflowed and not envs.SGLANG_TEST_DSPARK_FORCE_REVERIFY.get():
        return out
    with manager.suspend_graph_gather(), model_runner.decode_cuda_graph_runner.eager_only():
        return forward()


def draft_runs_exl3(draft_model) -> bool:
    """Whether the draft's MoE is EXL3: exl3_moe_loop reads expert counts on the host and refuses capture."""
    config = getattr(draft_model, "quant_config", None)
    return config is not None and config.get_name() == "exl3"


def target_gather_is_narrow(model_runner) -> bool:
    """Whether the target's verify gather serves fewer misses than its routes, and may flag its forward."""
    manager = getattr(model_runner, "expert_hot_cache_manager", None)
    return manager is not None and bool(manager.narrow_graph_gather)
