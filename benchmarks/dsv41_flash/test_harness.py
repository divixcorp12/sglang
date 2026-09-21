"""CPU tests of the lease benchmark harness: no Engine, no GPU, no model."""

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import arm_config as ac  # noqa: E402
import compare  # noqa: E402
import counters  # noqa: E402
import metrics  # noqa: E402
import workload  # noqa: E402

PATHS, RES = ac.Paths(), ac.Resources()


@pytest.fixture
def model_dir(tmp_path):
    config = {
        "architectures": ["DeepseekV4ForCausalLM"],
        "quantization_config": {"quant_method": "exl3", "bits": 3.02},
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    return str(tmp_path)


def test_the_import_path_assertion_fires_on_a_foreign_sglang(tmp_path):
    # The venv's default interpreter resolves sglang from an unrelated tree unless PYTHONPATH names this one.
    (tmp_path / "sglang").mkdir()
    (tmp_path / "sglang" / "__init__.py").write_text("")
    probe = "import arm_config as ac; ac.assert_sglang_from_repo()"
    env = {
        **os.environ,
        "PYTHONPATH": f"{tmp_path}{os.pathsep}{Path(__file__).resolve().parent}",
    }
    proc = subprocess.run(
        [sys.executable, "-c", probe], env=env, capture_output=True, text=True
    )
    assert (
        proc.returncode != 0
        and "ImportPathError" in proc.stderr
        and str(tmp_path) in proc.stderr
    )


def test_the_arms_differ_only_in_the_lease_switch_and_both_graph_gather():
    off, on = (ac.arm_env(a, paths=PATHS, res=RES) for a in ac.ARMS)
    assert ac.env_diff(off, on) == {ac.LEASE_ENV: {"lease_off": "0", "lease_on": "1"}}
    assert off[ac.GRAPH_GATHER_ENV] == on[ac.GRAPH_GATHER_ENV] == "1"


def test_the_phase3a_graph_gather_off_recipe_is_refused():
    kwargs = ac.engine_kwargs(paths=PATHS, res=RES)
    env = {**ac.arm_env("lease_on", paths=PATHS, res=RES), ac.GRAPH_GATHER_ENV: "0"}
    with pytest.raises(ac.RecipeError, match="lease-on == lease-off by construction"):
        ac.check_recipe(
            env=env, kwargs=kwargs, concurrency=1, allow_eager_batches=False
        )
    with pytest.raises(ac.RecipeError, match="batch size 1"):
        ac.check_recipe(
            env=ac.arm_env("lease_on", paths=PATHS, res=RES),
            kwargs=kwargs,
            concurrency=2,
            allow_eager_batches=False,
        )


def test_the_exl3_gate_accepts_both_arms_and_refuses_a_full_decode_graph(model_dir):
    kwargs = ac.engine_kwargs(paths=PATHS, res=RES)
    for arm in ac.ARMS:
        assert (
            ac.check_gate(
                env=ac.arm_env(arm, paths=PATHS, res=RES),
                kwargs=kwargs,
                model=model_dir,
            )["requirements"]
            == "EXL3"
        )
    full = {**kwargs, "cuda_graph_backend_decode": "full"}
    with pytest.raises(ValueError, match="breakable"):
        ac.check_gate(
            env=ac.arm_env("lease_on", paths=PATHS, res=RES),
            kwargs=full,
            model=model_dir,
        )
    with pytest.raises(ac.RecipeError, match="not EXL3"):
        ac.check_gate(
            env=ac.arm_env("lease_on", paths=PATHS, res=RES),
            kwargs=kwargs,
            model="/nonexistent/model",
        )


def _snapshot(granted, acked, served=100):
    keys = (
        "served",
        "touch_only",
        "rows_read",
        "read_errors",
        "late_after_fatal",
        "leases_granted",
        "leases_acked",
        "leases_voided",
    )
    return dict.fromkeys(keys, 0) | {
        "served": served,
        "leases_granted": granted,
        "leases_acked": acked,
    }


def _cursor(tmp_path, snapshots):
    path = tmp_path / "trace.jsonl"
    path.write_text(
        "".join(
            json.dumps(
                {
                    "kind": "graph_step",
                    "ram_miss": 3,
                    "layer_ram_rows": [3, 0],
                    "thread": s,
                }
            )
            + "\n"
            for s in snapshots
        )
    )
    return counters.TraceCursor(str(path)).poll()


@pytest.mark.parametrize(
    "arm, snapshots, ok",
    [
        (
            "lease_on",
            [_snapshot(g, g - 1) for g in range(1, 6)] + [_snapshot(6, 6)],
            True,
        ),
        (
            "lease_on",
            [_snapshot(0, 0)] * 3,
            False,
        ),  # the lease path never ran: the red flag, not a finding
        (
            "lease_on",
            [_snapshot(g, g - 1) for g in range(1, 20)],
            False,
        ),  # acks leak for good
        ("lease_off", [_snapshot(0, 0)] * 3, True),
        ("lease_off", [_snapshot(4, 4)] * 3, False),  # switch did not reach the service
    ],
)
def test_the_lease_ledger_check(tmp_path, arm, snapshots, ok):
    assert (
        counters.verify_lease(arm=arm, cursor=_cursor(tmp_path, snapshots))["ok"] is ok
    )


def test_a_run_with_no_graph_steps_is_never_verified(tmp_path):
    (tmp_path / "trace.jsonl").write_text(
        json.dumps({"kind": "ram_miss_request"}) + "\n"
    )
    cursor = counters.TraceCursor(str(tmp_path / "trace.jsonl")).poll()
    assert not counters.verify_lease(arm="lease_off", cursor=cursor)["ok"]


def test_nearest_rank_percentiles_are_observations():
    values = list(range(1, 101))
    assert [metrics.nearest_rank(values, q) for q in (50, 95, 99)] == [50, 95, 99]
    assert metrics.nearest_rank([7.0], 99) == 7.0


class _FakeStream:
    def __init__(self, tokens, dt, clock):
        self.tokens, self.dt, self.clock = tokens, dt, clock

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        for n in range(1, self.tokens + 1):
            self.clock.now += self.dt
            finish = {"type": "length"} if n == self.tokens else None
            yield {
                "text": "a" * n,
                "meta_info": {"completion_tokens": n, "finish_reason": finish},
            }


class _Clock:
    now = 0.0

    def __call__(self):
        return self.now


def test_the_driver_discards_warmup_steps_and_reports_per_token_latency():
    clock = _Clock()

    async def generate(**kwargs):
        return _FakeStream(kwargs["sampling_params"]["max_new_tokens"], 0.5, clock)

    prompts = [{"input_ids": [1, 2, 3]} for _ in range(2)]
    records, wall = asyncio.run(
        workload.run_closed_loop(
            generate,
            prompts=prompts,
            params=workload.sampling_params(output_tokens=41),
            concurrency=1,
            clock=clock,
        )
    )
    summary = metrics.aggregate(records, wall_s=wall, discard_steps=30, blocks=2)
    # 40 steps per request, the first 30 dropped from each
    assert summary["step_s"]["n"] == 20 and summary["step_s"]["min"] == 0.5
    assert (
        summary["discarded_steps_per_request"] == 30
        and summary["tokens_per_s"] == 82 / 41.0
    )


def test_block_spread_sees_a_drift_that_a_pooled_percentile_hides():
    drifting = [1.0] * 50 + [1.1] * 50
    assert metrics.block_spread(drifting, 2)["p50_rel_range"] == pytest.approx(
        0.1 / 1.05
    )
    assert metrics.block_spread([1.0] * 9, 5)["blocks"] == 0


def test_the_window_mix_tells_all_hit_from_mixed():
    def win(missed):
        delta = dict.fromkeys(counters.MIX_KEYS, 0)
        return {
            "graph_steps": 10,
            "layer_steps": 400,
            "layer_steps_missed": missed,
            "demand_rows": missed,
            "counters_delta": delta,
        }

    assert counters.mix(win(0))["label"] == "all-hit"
    assert counters.mix(win(100))["label"] == "mixed"
    assert counters.mix(win(399))["label"] == "all-miss"


def _run(arm, p50, mn, spread, ok=True):
    return {
        "arm": arm,
        "rep": 0,
        "position": 0,
        "lease_check": {"ok": ok, "reasons": [] if ok else ["leases_granted == 0"]},
        "summary": {
            "step_s": {"p50": p50, "min": mn, "n": 1700, "sd": 0.001},
            "step_blocks": {"p50_rel_range": spread},
            "tokens_per_s": 1.0,
        },
    }


@pytest.mark.parametrize(
    "on_p50, on_min, spread, verdict",
    [
        (0.3636, 0.3330, 0.002, "RESOLVED"),  # +1% p50 and min, spread 0.2%
        (
            0.3636,
            0.3330,
            0.02,
            "UNRESOLVED. Any effect of lease mode",
        ),  # same delta buried in 2% spread
        (0.3636, 0.3200, 0.002, "disagree in sign"),  # p50 up, min down: box noise
    ],
)
def test_the_second_pair_decision_rule(on_p50, on_min, spread, verdict):
    off = _run("lease_off", 0.3600, 0.3300, spread)
    on = _run("lease_on", on_p50, on_min, spread)
    text = "\n".join(compare.decision({"lease_off": [off], "lease_on": [on]}))
    assert verdict in text


def test_compare_flags_an_unverified_lease_run():
    good = {
        "arm": "lease_off",
        "rep": 0,
        "summary": {"tokens_per_s": 1.0},
        "lease_check": {"ok": True, "reasons": []},
    }
    bad = {
        "arm": "lease_on",
        "rep": 0,
        "summary": {"tokens_per_s": 1.0},
        "lease_check": {"ok": False, "reasons": ["leases_granted == 0"]},
    }
    assert "INVALID" in compare.render([good, bad]) and not compare.problems(
        [good, {**bad, "lease_check": {"ok": True, "reasons": []}}]
    )


def test_the_projection_uses_the_warmup_request_own_rate_and_sd():
    warm = {"ttft_s": 50.0, "decode_tok_s": 4.0, "step_s": [0.25] * 140}
    got = metrics.project(
        warm=warm,
        requests=4,
        output_tokens=401,
        startup_s=100.0,
        warm_wall_s=160.0,
        discard_steps=30,
        expected_effect_s=0.001,
    )
    assert got["per_request_s"] == 150.0 and got["process_total_s"] == 860.0
    assert got["warmup_ms_per_token"] == 250.0
    assert got["warmup_step_sd_s"] == 0.0 and got["resolvable"] is True
    with pytest.raises(ValueError, match="no decode rate"):
        metrics.project(
            warm={"ttft_s": 1.0, "decode_tok_s": 0.0, "step_s": []},
            requests=1,
            output_tokens=10,
            startup_s=0,
            warm_wall_s=0,
            discard_steps=0,
            expected_effect_s=0.001,
        )


def test_a_noisy_step_time_makes_the_expected_effect_unresolvable_and_says_so():
    # 400 ms steps with a 60 ms sd: the window resolves ~10 ms, far above a 0.68 ms effect.
    steps = [0.34, 0.46] * 100
    warm = {"ttft_s": 50.0, "decode_tok_s": 2.5, "step_s": steps}
    got = metrics.project(
        warm=warm,
        requests=4,
        output_tokens=512,
        startup_s=0,
        warm_wall_s=0,
        discard_steps=30,
        expected_effect_s=0.00068,
    )
    assert got["resolvable"] is False and got["resolvable_delta_s"] > 0.005


def test_the_unresolved_verdict_states_the_bound_from_the_per_token_sd():
    off, on = (
        _run("lease_off", 0.400, 0.35, 0.001),
        _run("lease_on", 0.4004, 0.35, 0.001),
    )
    for r in (off, on):
        r["summary"]["step_s"]["sd"] = 0.05
    text = "\n".join(compare.decision({"lease_off": [off], "lease_on": [on]}))
    assert "UNRESOLVED" in text and "below 1.6" in text
