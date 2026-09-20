"""One eager arm of the eager-mirror anomaly (DSV41_REFERENCE section 19).

Same Engine, prompts and settings as scripts/dsv41/trace_corpus.py without --graphs,
but every session is bracketed by ``time.monotonic()`` and per-drive /proc/diskstats
sector counts, so the scheduler's trace lines and cache-counter snapshots (which carry
the same monotonic clock) can be cut at exact session boundaries. The caller sets the
streaming environment and SGLANG_DSV41_EXPERT_TRACE_PATH.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from types import SimpleNamespace

# The harness lives in the worktree; run from its root (env.sh does) or from a copy of this file.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "scripts", "dsv41"))
sys.path.insert(0, os.path.join(os.getcwd(), "scripts", "dsv41"))
import provenance  # noqa: E402
import trace_corpus  # noqa: E402
from provenance import DRIVES, SECTOR_BYTES, read_sectors  # noqa: E402


def capture_text(stream, box):
    """Pass a generate stream through, keeping its last cumulative text in ``box``.

    ``trace_corpus.time_stream`` returns ``output_text`` only from the 099eadba33 harness on;
    an older harness (the section 19 commit) drops it, and greedy parity needs it."""
    for chunk in stream:
        if isinstance(chunk, dict) and chunk.get("text") is not None:
            box["text"] = chunk["text"]
        yield chunk


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--sessions", required=True)
    p.add_argument("--n", type=int, default=4)
    p.add_argument("--skip", type=int, default=0)
    p.add_argument("--prompt-tokens", type=int, default=256)
    p.add_argument("--new-tokens", type=int, default=128)
    p.add_argument("--out", required=True)
    p.add_argument("--arm", required=True)
    args = p.parse_args()
    args.mem_fraction_static = 0.85
    args.chunked_prefill_size = 512
    args.graphs = False
    args.dspark = None

    texts = list(trace_corpus._first_turns(args.sessions, args.n, args.skip))
    if not texts:
        raise SystemExit("no sessions selected")

    import sglang
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prov = provenance.capture({"driver": os.path.abspath(__file__), "trace_corpus": trace_corpus.__file__})
    prov["drive_idle_check"] = provenance.drive_idle_check()
    boot_sectors = read_sectors()
    boot_t = time.monotonic()
    engine = sglang.Engine(**trace_corpus.engine_kwargs(args))
    ready_t = time.monotonic()
    ready_sectors = read_sectors()
    # Engine launch may edit os.environ before it spawns the scheduler; record what it changed.
    prov["sglang_env_drift_at_engine_ready"] = provenance.env_drift(prov["sglang_env"], provenance.process_env())
    sessions = []
    for index, text in enumerate(texts):
        ids = tokenizer(text).input_ids[: args.prompt_tokens]
        before = read_sectors()
        start_t = time.monotonic()
        box = {}
        timing = trace_corpus.time_stream(
            capture_text(
                engine.generate(
                    input_ids=ids, sampling_params=trace_corpus.sampling_params(args), stream=True
                ),
                box,
            ),
            args.new_tokens,
        )
        end_t = time.monotonic()
        after = read_sectors()
        output_text = timing.pop("output_text", box.get("text", ""))
        row = {
            "session": index,
            "prompt_tokens": len(ids),
            "start_t": start_t,
            "end_t": end_t,
            "disk_bytes": {k: (after[k] - before[k]) * SECTOR_BYTES for k in DRIVES},
            "output_sha1": hashlib.sha1(output_text.encode()).hexdigest(),
            **timing,
        }
        sessions.append(row)
        print(json.dumps(row), flush=True)
    engine.shutdown()
    report = {
        "arm": args.arm,
        "provenance": prov,
        "mirror_dirs": os.environ.get("SGLANG_MOE_EXPERT_MIRROR_DIRS", ""),
        "boot_t": boot_t,
        "ready_t": ready_t,
        "startup_disk_bytes": {
            k: (ready_sectors[k] - boot_sectors[k]) * SECTOR_BYTES for k in DRIVES
        },
        "per_session": sessions,
        "mean_decode_tok_s": trace_corpus.mean_decode_tok_s(sessions),
    }
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
