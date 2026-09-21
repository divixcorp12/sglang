"""The lease protocol model: the designed protocol is clean over every interleaving, and the model is strong
enough to find each thing that is known to be wrong.

A model that cannot find a known bug is too weak to say "nothing" about the design, so half of this file is
mutants: one protocol rule removed each, with the violation it must produce. Pure Python, no GPU, no torch;
the whole file takes about two minutes and under 1 GiB.

    OMP_NUM_THREADS=1 python -m pytest analysis/dsv41-drive/test_lease_model.py -p no:cacheprovider
"""

import importlib.util
import sys
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location("lease_model", Path(__file__).with_name("lease_model.py"))
lm = importlib.util.module_from_spec(_SPEC)
sys.modules["lease_model"] = lm
_SPEC.loader.exec_module(lm)

# Request shapes (lane experts): enough to force eviction with two slots, duplicate lanes, and idle records.
MENU = ((), (0,), (0, 1), (1, 1))
EVICTING = ((0, 1), (2,), (0, 2), (1, 1))
CLEAN = dict(timeouts=False, io_faults=False)


def run(cfg, everything=False):
    result = lm.explore(cfg, stop_on=(lambda kind: False) if everything else None)
    assert result.complete, f"the state cap ({cfg.max_states}) was hit: raise it or shrink the world"
    return result


def replay(cfg, trace):
    """Follow ``trace`` through the transition relation: every label must be an enabled step."""
    model = lm.Model(cfg)
    state = model.initial()
    for label in trace:
        state = next((n for name, n in model.successors(state) if name == label), None)
        assert state is not None, f"the trace step {label!r} is not enabled"
    return model.d(state)


# ---- the model itself ----


def test_reached_is_the_signed_difference_over_the_model_modulus():
    cfg = lm.Config()
    assert lm.reached(cfg, 5, 5) and lm.reached(cfg, 6, 5) and not lm.reached(cfg, 5, 6)
    assert lm.reached(cfg, 1, 7), "1 follows 7 across the wrap"
    assert not lm.reached(cfg, 7, 1)


def test_the_device_and_a_fixed_service_skip_zero_and_count_the_wrap():
    cfg = lm.Config()
    assert lm.next_seq_device(cfg, 7, 0) == (1, 1)
    assert lm.next_seq_service(cfg, 7, 0) == (1, 1)
    assert lm.next_seq_service(lm.Config(skip0=False), 7, 0) == (0, 1)


def test_a_world_with_more_requests_than_half_the_sequence_range_is_refused():
    with pytest.raises(AssertionError):
        lm.Model(lm.Config(requests=4))


def test_a_counterexample_trace_is_a_real_path_to_its_violation():
    cfg = lm.Config(leases=False, menu=MENU, requests=3)
    result = run(cfg)
    assert result.violation == "RecycledUnderReader"
    assert "RecycledUnderReader" in replay(cfg, result.trace)["viol"]


# ---- the designed protocol is clean ----


def test_the_designed_protocol_holds_with_timeouts_and_read_failures():
    result = run(lm.Config(requests=3, menu=MENU))
    assert result.violation is None
    # The protocol's hard cases were reached, not merely allowed.
    for reached in ("retired by ack", "retired by terminal", "dropped: the device already gave up", "two leases on one slot"):
        assert result.coverage.get(reached, 0) > 0, reached


def test_the_designed_protocol_defers_instead_of_failing_and_never_deadlocks():
    """No timeout and no fault: the run must finish, hold no lease, and never fail stop. A demand whose
    victims are all leased waits for the acknowledgements, and the wait-for chain runs into the past."""
    result = run(lm.Config(requests=3, menu=EVICTING, **CLEAN))
    assert result.violation is None
    assert result.coverage.get("deferred: victims are leased", 0) > 0
    assert result.coverage.get("deferred: request slot not retired", 0) > 0


def test_the_designed_protocol_holds_over_every_request_shape():
    assert run(lm.Config(requests=2)).violation is None


def test_the_designed_protocol_holds_with_a_ring_of_three_across_the_wrap():
    assert run(lm.Config(requests=2, ring=3, seq_space=8, start=7, menu=EVICTING)).violation is None
    assert run(lm.Config(requests=3, ring=3, seq_space=8, start=7, menu=EVICTING, **CLEAN)).violation is None


def test_shutdown_frees_only_after_the_device_completed_and_quarantines_after_a_cuda_error():
    result = run(lm.Config(requests=2, menu=EVICTING, shutdown=True, cuda_error=True))
    assert result.violation is None


def test_a_stale_acknowledgement_a_full_cycle_old_is_not_taken_for_a_new_one():
    cfg = lm.Config(requests=3, menu=MENU, stale_ack=(0, 0, (0, 1)))
    assert run(cfg).violation is None


# ---- mutants: each removed rule must be found ----


@pytest.mark.parametrize(
    "name, cfg, kind",
    [
        ("eviction ignores leases", lm.Config(requests=3, menu=MENU, leases=False), "RecycledUnderReader"),
        ("the acknowledgement is published at commit, before the copy", lm.Config(requests=3, menu=MENU, ack_after_copy=False), "RecycledUnderReader"),
        (
            "32-bit generations meet a stale acknowledgement",
            lm.Config(requests=3, menu=MENU, gen64=False, stale_ack=(0, 0, (0, 1))),
            "RecycledUnderReader",
        ),
        ("a request slot is reused before its leases retire", lm.Config(requests=3, menu=MENU, defer_reuse=False), "LeakedLease"),
        (
            "the service counts epochs itself and misses a lap across the wrap",
            lm.Config(requests=3, menu=MENU, echo_gen=False),
            "ArmedRequestLapped",
        ),
        ("a demand fails instead of waiting for leased victims", lm.Config(requests=3, menu=EVICTING, defer_leased=False, **CLEAN), "SpuriousFatal"),
        ("the service never retires a lease", lm.Config(requests=3, menu=MENU, retire=False, **CLEAN), "Deadlock"),
        (
            "Python frees without establishing completion",
            lm.Config(requests=2, menu=EVICTING, shutdown=True, cuda_error=True, free_needs_sync=False),
            "FreedWhileReading",
        ),
    ],
)
def test_each_removed_rule_is_found(name, cfg, kind):
    result = run(cfg)
    assert result.violation == kind, name
    assert replay(cfg, result.trace)["viol"] or kind in ("Deadlock", "LeakedLease", "SpuriousFatal")


def test_the_detector_turns_a_recycled_slot_into_a_failure_instead_of_a_success():
    """Without leases the mechanism fails either way; what the detector decides is whether the wrong bytes
    are accepted: the shipped-bug shape is a request that succeeds on bytes nobody guaranteed."""
    with_detector = run(lm.Config(requests=2, menu=EVICTING, leases=False), everything=True)
    without = run(lm.Config(requests=2, menu=EVICTING, leases=False, detector=False), everything=True)
    assert "WrongBytesRead" in with_detector.seen_violations and "WrongBytesRead" in without.seen_violations
    assert "WrongBytesAccepted" not in with_detector.seen_violations
    assert "WrongBytesAccepted" in without.seen_violations


def test_a_service_that_does_not_skip_zero_is_noise_not_a_safety_failure():
    """The wrap bug in ``pump_demand`` (``next_demand_ += 1u``): a phantom request 0 is processed. Nothing
    ever waits on sequence 0, so the model finds the phantom and nothing else."""
    result = run(lm.Config(requests=3, menu=MENU, skip0=False), everything=True)
    assert set(result.seen_violations) == {"PhantomSequence"}


# ---- the protocol as it is today ----


def test_today_is_safe_without_faults_because_of_the_temporal_exclusion():
    """Calibration: the model does not cry wolf about the code as it is. With the implicit rule (no eviction
    on a layer between its demand and its consume) and no fault, there is no violation."""
    result = run(lm.today(requests=3, menu=EVICTING, **CLEAN), everything=True)
    assert not result.seen_violations


def test_today_a_timeout_lets_the_copy_read_a_slot_the_service_recycles():
    """D1 in LEASE_PROTOCOL.md: after a timeout the gather still runs, and the service is inside the very
    request that timed out. The output is dropped (keep = 0), so no wrong bytes are accepted."""
    cfg = lm.today(requests=3, menu=MENU, io_faults=False)
    result = run(cfg, everything=True)
    assert "RecycledUnderReader" in result.seen_violations
    assert "WrongBytesAccepted" not in result.seen_violations
    shortest = run(cfg)
    assert shortest.violation == "RecycledUnderReader" and "device aborts: wait times out" in shortest.trace


def test_today_without_the_exclusion_wrong_bytes_are_accepted():
    """D2: the device resolves through the map, and nothing but the temporal exclusion keeps it stable."""
    result = run(lm.today(requests=3, menu=EVICTING, exclusion=False, **CLEAN), everything=True)
    assert "WrongBytesAccepted" in result.seen_violations
