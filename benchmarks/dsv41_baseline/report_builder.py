"""Assemble a Task-1-shaped `report` dict from this campaign's HTTP-driven arm.

`task1_arm_verdict.check_arm` and friends expect the report `trace_corpus.py` (the
offline-Engine driver) produces: `provenance`, `boundary_samples`, `per_session[]`
with `decode_tok_s` / `ttft_s` / `cpu_s` / `step_latency`. This module builds that
shape from the pieces this campaign's HTTP driver can actually produce, rather than
reshaping the verdict logic to fit — the instruction this campaign is working under.
Two fields are honestly unavailable rather than faked; see `STEP_LATENCY_UNAVAILABLE`.
"""

from __future__ import annotations

import client_latency
import provenance

STEP_LATENCY_UNAVAILABLE = {
    "unavailable": (
        "no engine-side per-step timing: run_capture_sessions.py drives /v1/chat/completions "
        "(OpenAI-compatible SSE), whose streamed chunks carry no per-chunk completion_tokens the "
        "way the native /generate endpoint's meta_info does, and driving that endpoint instead "
        "was ruled out (it is not the serving path the user asked for). A client-side proxy "
        "exists (client_inter_token_latency_s, below) but is a differently-defined quantity and "
        "must never be substituted here. A genuine engine-side source may exist in the decode-log "
        "line at decode_log_interval=1 (gen throughput inverts to per-step latency); under "
        "investigation, not yet wired in — see README 'One harness, not two'."
    )
}

# `check_arm` reads these from the top level of `provenance` as facts about the process that ran
# the arm. `provenance.capture()` samples them in the harness, which has none of the server's
# SGLANG_* vars, so leaving them there had the verdict judge the wrong process: it reported the
# reader as 'mmap' and SGLANG_MOE_EXPERT_MIRROR_DIRS as unset for a server whose /proc/<pid>/environ
# had both set, and whose per-drive read counters showed the mirrors carrying the whole arm.
HARNESS_PROCESS_FIELDS = ("sglang_env", "sglang_env_at_exec", "sglang_env_resolved", "sglang_file")

# Written into provenance["unavailable"], which check_arm already reports rather than guessing.
SERVER_UNAVAILABLE = {
    "sglang_env_resolved": (
        "resolved config values (envs.X.get(), which apply each knob's default) can only be read "
        "inside the server, and it does not emit its own provenance; sglang_env below is the "
        "server's real environment, so a knob this arm set explicitly is still checkable, but one "
        "left at its default is not"
    ),
    "sglang_file": (
        "the server's own sglang import path is not observable from outside it; PYTHONPATH in "
        "sglang_env is the measured constraint on it"
    ),
}

SERVER_ENV_IS_EXEC_TIME_NOTE = (
    "sglang_env and sglang_env_at_exec are both /proc/<pid>/environ, the environment the kernel "
    "handed the server at exec: an edit the server made to its own os.environ afterwards is not "
    "visible from outside and would not appear in either"
)

SERVER_PROVENANCE_CEILING_NOTE = (
    "the server is a separate process from the one that captured the rest of this "
    "provenance (the HTTP driver launches it, then talks to it over the network); only "
    "/proc/<pid>/environ could be read from outside it (see server_env_actual). Resolved "
    "config values (envs.X.get()), the server's own sglang import path, and which harness "
    "files it loaded are unavailable. A named limitation per the team lead's instruction, "
    "not a silent gap — deferred rather than closed by teaching the server to emit its own "
    "provenance.capture(), which is a real code change to the launch path this task does "
    "not own."
)


def merge_sessions(
    *,
    results: list[dict],
    clocks_by_id: dict[str, dict],
    compile_by_id: dict[str, dict],
    cpu_s_by_id: dict[str, float | None],
) -> list[dict]:
    """One task1-shaped per_session record per result, joined by session_id."""
    merged = []
    for record in results:
        session_id = record["session_id"]
        clocks = clocks_by_id.get(session_id, {})
        compile_row = compile_by_id.get(session_id, {})
        merged.append(
            {
                "session_id": session_id,
                "decode_tok_s": record.get("decode_tokens_per_sec"),
                "ttft_s": record.get("ttft"),
                "cpu_s": cpu_s_by_id.get(session_id),
                # Engine-side, Task-1-comparable step latency: acknowledged-absent (see module
                # docstring). NOT the same thing as client_inter_token_latency_s below.
                "step_latency": dict(STEP_LATENCY_UNAVAILABLE),
                # Client-side proxy, its own metric, never compared against step_latency
                # thresholds — see client_latency.py.
                "client_inter_token_latency_s": client_latency.client_inter_token_latency_s(
                    record.get("chunk_times") or []
                ),
                "clock_sm_start_mhz": clocks.get("clock_sm_start_mhz"),
                "clock_sm_end_mhz": clocks.get("clock_sm_end_mhz"),
                "compiled_during_session": compile_row.get("compiled_during_session"),
                "compile_events": compile_row.get("compile_events"),
            }
        )
    return merged


def build_report(
    *,
    harness_provenance: dict,
    server_env_actual: dict,
    server_env_expected: dict,
    boundary_samples: list[dict],
    residency: dict,
    sessions: list[dict],
) -> dict:
    prov = dict(harness_provenance)
    prov["harness_process"] = {k: prov.pop(k) for k in HARNESS_PROCESS_FIELDS if k in prov}
    # The subject of every check_arm env test is the server, so the server's own environment is
    # what has to sit here -- filtered, because /proc/<pid>/environ carries the caller's secrets.
    server_env = provenance.filter_env(server_env_actual)
    prov["sglang_env"] = server_env
    prov["sglang_env_at_exec"] = server_env
    prov["sglang_env_resolved"] = None
    prov["sglang_file"] = None
    prov["unavailable"] = {**(prov.get("unavailable") or {}), **SERVER_UNAVAILABLE}
    # Filtered for the same reason: a report is copied and quoted, and the raw environ the
    # verification step read carries whatever secrets the launching shell held.
    prov["server_env_actual"] = server_env
    prov["server_env_expected"] = server_env_expected
    prov["server_provenance_ceiling"] = SERVER_PROVENANCE_CEILING_NOTE
    prov["server_env_is_exec_time"] = SERVER_ENV_IS_EXEC_TIME_NOTE
    return {
        "provenance": prov,
        "boundary_samples": boundary_samples,
        "residency": residency,
        "per_session": sessions,
    }
