#!/usr/bin/env bash
# Runs one DSV4.1 baseline arm end to end: preflight (clean tree, registered code
# generation), corpus checksum, synthetic-session build, cold HTTP server (production
# context length and prefix-cache settings), health gate, env verification, a
# readiness gate (SM clock stable and no further JIT compilation), the 2-session
# timed set (aborting hard on any
# mid-session compile event; provenance, clocks, cpu_s and page-cache residency
# sampled throughout), a verdict judged by Task 1's own `check_arm`/`contention`/
# `generation`/`session_outliers` (imported, not reimplemented — see verdict.py), and
# a run-manifest.json carrying everything paired.py and a human need to trust the
# number.
#
# Usage: run_arm.sh <arm_name> <port> [KEY=VAL ...]
#   <arm_name>  a short label for this V2 storage-change arm (e.g. baseline, v2-on).
#   <port>      the server's port. Pick one outside production's 7867.
#   KEY=VAL...  env overrides layered onto arm_env.py's base recipe for this arm; the
#               same overrides are what run_arm.sh verifies against the live server's
#               /proc/<pid>/environ afterwards.
#
# Env: DSV41_SESSION_INDICES=<i,j,...> times those indices of CORPUS_8_SESSION_IDS
# instead of the shared default (session_subset.timed_session_indices); the resolved
# indices and ids are recorded in run-manifest.json.
# Env: EXPECT_SHA=<sha> to pin the worktree to an exact commit (refuse otherwise); the
# worktree must always be clean (preflight). The python/ tree must be registered in
# generations.json first (`python -c "import generations; generations.register(TREE,
# LABEL)"`) or the arm refuses to start, matching task1-baseline-arms.sh's
# generation_gate.
# Env: NSYS_TRACE=1 captures the timed set with Nsight Systems (report under
# NSYS_OUT_DIR, default /mnt/nvme1/dsv41-nsys; anything off /mnt/nvme1 is refused).
# NSYS_CUDA_GRAPH_TRACE=graph|node (default graph) picks --cuda-graph-trace; graph is
# refused when the arm's env turns on SGLANG_DSV41_ENABLE_RAM_MISS_COPY_ENGINE.
#
# Cost: ~200 s server startup + a readiness loop (clock stability + JIT compile
# settling; the smoke launch saw a 34 s Triton compile land mid-decode after /health
# was already 200) + 2 sessions x (TTFT ~55-99 s + 127 decode tokens at ~2.2-3.5
# tok/s). The mandatory warm-up means this number may
# differ slightly from the recorded 2.781 tok/s by construction; see README.md.
set -uo pipefail

arm=${1:?arm name}
port=${2:?port}
shift 2
overrides=("$@")

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
worktree=${DSV41_WORKTREE:-$(cd "$here/../.." && pwd)}
py=/data/models/slang/.venv/bin/python
# Read the server's core list from arm_env rather than repeating it: this line and
# arm_env.SERVER_CORES were two copies of "32-63" that could drift, and a taskset
# spanning both NUMA nodes is exactly the startup hang of 2026-09-22. The command
# substitution finishes long before the launch below, so it does not break the
# "nothing may fork between the shell and python" rule that keeps $! the server.
server_cores=$("$py" -c "import sys; sys.path.insert(0, '$here'); import arm_env; print(arm_env.SERVER_CORES)") \
    || { echo "cannot read arm_env.SERVER_CORES" >&2; exit 1; }
gpu_lock=/data/models/slang/nvfp4-work/cc-gpu.lock
out_root=${DSV41_RUN_ROOT:-/data/models/slang/nvfp4-work/cc-expert-prediction/dsv41-baseline}
run_dir=$out_root/servers/$arm/run-$(date +%Y%m%d-%H%M%S)
log=$run_dir/server.log
corpus=/mnt/nvme2/nvfp4-work/benchmarks/full/sessions.jsonl
corpus_checksum=249e8a73a32b69aff563471dbae2f4f3a2a9beaa1a3ae5cb03b4c2c549c16c72
max_tokens=128
warmup_session_id='fb-financebench_id_04209'
max_warmup_rounds=12  # a live trace run showed decode tok/s still climbing at round 5
expert_shard_dir=/mnt/nvme2/DeepSeek-V4.1-Flash-EXL3-3.0bpw

abort() { echo "ABORT: $*" >&2; exit 1; }
pyrun() { PYTHONPATH="$here:$worktree/python:$worktree/scripts/dsv41" "$py" "$@"; }

mkdir -p "$run_dir"
log_pretty=$run_dir/run.log
exec > >(tee -a "$log_pretty") 2>&1
touch "$log"

echo "arm=$arm port=$port run_dir=$run_dir worktree=$worktree overrides=${overrides[*]:-none}"

# --- preflight: refuse a dirty worktree, and a wrong one if EXPECT_SHA is set. Adopted
#     from task1-baseline-arms.sh's preflight()/harness_gate() — a harness launching
#     from an unpinned tree is the worst property the earlier version of this script had. ---
pyrun -c "
import provenance
state = provenance.git_state('$worktree/python')
if state['dirty']:
    raise SystemExit(f\"REFUSE: {state['toplevel']} has {state['dirty_file_count']} tracked changes: {state['dirty_files']}\")
expect = '${EXPECT_SHA:-}'
if expect and state['head'] != expect:
    raise SystemExit(f\"REFUSE: {state['toplevel']} is at {state['head']}, expected {expect}\")
print(f\"preflight ok: {state['toplevel']} at {state['head']}, clean\")
" || abort "$arm preflight failed"

# --- generation gate: the python/ tree under test must be registered first, so a new
#     code generation can never silently be compared as if it were an old one. ---
python_tree=$(git -C "$worktree" rev-parse HEAD:python)
generation_label=$(pyrun -c "
import generations
print(generations.check_registered('$python_tree'))
") || abort "$arm generation gate failed: python tree $python_tree is unregistered (register it: python -c \"import generations; generations.register('$python_tree', '<label>')\")"
echo "generation ok: $python_tree = $generation_label"

# --- corpus checksum (abort on mismatch, never regenerate) ---
got_checksum=$(sha256sum "$corpus" | awk '{print $1}')
[ "$got_checksum" = "$corpus_checksum" ] || abort "corpus checksum mismatch: got $got_checksum expected $corpus_checksum"

# --- the timed set: the shared default, or this arm's DSV41_SESSION_INDICES override.
#     Resolved once, here, and refused on a malformed value before anything starts. ---
timed_ids_path=$run_dir/timed-session-ids.json
pyrun -c "
import json, os
import session_subset as ss
indices = ss.timed_session_indices(os.environ.get(ss.SESSION_INDICES_ENV))
ids = ss.timed_session_ids(indices)
json.dump({'indices': list(indices), 'session_ids': list(ids)}, open('$timed_ids_path', 'w'))
print(f'timed sessions: indices={list(indices)} ids={list(ids)}')
" || abort "bad ${DSV41_SESSION_INDICES:+DSV41_SESSION_INDICES=$DSV41_SESSION_INDICES}"

# --- build the timed + 1 warm-up synthetic sessions (real text, truncated to 256
#     tokens, re-decoded); asserts each fits context_length before anything is sent ---
synthetic_sessions=$run_dir/synthetic-sessions.jsonl
pyrun -c "
import json
from transformers import AutoTokenizer
import arm_env, session_subset as ss, synthetic_corpus as sc

tokenizer = AutoTokenizer.from_pretrained(arm_env.MODEL_PATH)
timed = json.load(open('$timed_ids_path'))
corpus8 = ss.load_raw_sessions(n=len(ss.CORPUS_8_SESSION_IDS), skip=ss.SKIP)
raw = [corpus8[i] for i in timed['indices']]
raw += ss.load_raw_sessions(n=1, skip=ss.WARMUP_SESSION_INDEX)
sessions = sc.build_synthetic_sessions(
    raw,
    tokenize=lambda t: tokenizer(t).input_ids,
    detokenize=lambda ids: tokenizer.decode(ids),
)
got_ids = [s['session_id'] for s in sessions[:-1]]
assert got_ids == timed['session_ids'], (got_ids, timed['session_ids'])
assert sessions[-1]['session_id'] == ss.WARMUP_SESSION_ID, sessions[-1]['session_id']
sc.write_jsonl('$synthetic_sessions', sessions)
print(f'wrote {len(sessions)} synthetic sessions to $synthetic_sessions')
" || abort "synthetic corpus build failed"

# --- tenancy at run start ---
tenancy_start=$(pyrun -c "
import json, msgspec, tenancy
print(json.dumps(msgspec.to_builtins(tenancy.capture_tenancy())))
")
echo "tenancy_start: $tenancy_start"

# --- harness provenance and boundary sample, before the server exists: what this
#     launcher would hand off, captured from a process with sglang importable from
#     the worktree under test. Residency of the expert shard dir, before anything reads it. ---
boundary_path=$run_dir/boundary-samples.jsonl
: > "$boundary_path"
harness_provenance_json=$(pyrun -c "
import sys, json
sys.path.insert(0, '$worktree/python')
import sglang  # noqa: F401  (so provenance.capture() resolves sglang_file/git/sglang_env_resolved)
import provenance
prov = provenance.capture({'run_capture_sessions': '$worktree/scripts/expert_prediction/benchmarks/run_capture_sessions.py', 'synthetic_corpus': '$here/synthetic_corpus.py'})
# Measured here or not at all: the check is 'no other job was reading these drives when the arm
# started', which stops being answerable the moment the server does its first read. Leaving it
# out did not read as unmeasured -- check_arm renders a missing one as 'drives not idle at start: {}'.
prov['drive_idle_check'] = provenance.drive_idle_check()
print(json.dumps(prov))
")
residency_before=$(pyrun -c "
import json, provenance
print(json.dumps(provenance.resident_bytes(['$expert_shard_dir'])))
")
pyrun -c "
import json, provenance
row = {'label': 'before_server', **provenance.system_sample()}
with open('$boundary_path', 'a') as f:
    f.write(json.dumps(row) + chr(10))
"

# --- build this arm's environment (base recipe + overrides) ---
expected_env_json=$(pyrun -c "
import json, sys
import arm_env
overrides = dict(kv.split('=', 1) for kv in sys.argv[1:])
print(json.dumps(arm_env.arm_env(overrides)))
" "${overrides[@]}")

commit=$(git -C "$worktree" rev-parse HEAD)

# --- refuse to start while anything else holds the GPU ---
if [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; then
    abort "GPU is in use; not starting a cold server for $arm"
fi
if ss -ltn 'sport = :7867' | grep -q LISTEN; then
    abort "production (7867) is up; not starting $arm"
fi

# --- launch (cold server per condition) ---
#     $! must be the server itself, so nothing may fork between the shell and python.
#     `flock <file> <cmd>` DOES fork, which is why it is not used here: with it, $! was
#     the flock wrapper, so /proc/$spid/environ read the launcher shell's environment
#     (no SGLANG_* vars at all) and stop_server killed the wrapper and leaked a 27 GiB
#     server holding the port. Take the lock on a descriptor instead and hold it for the
#     whole arm -- which is the semantics wanted anyway: this arm owns the GPU. taskset
#     and env both exec in place, and the python below execvp's, so $! stays the server.
#     The ONE exception is NSYS_TRACE=1: `nsys launch` forks, so $! is the profiler's
#     wrapper and the server pid is resolved by cmdline below instead. Anything else
#     added between the shell and python must exec, or it breaks the same way.
expected_env_path=$run_dir/expected-env.json
printf '%s' "$expected_env_json" > "$expected_env_path"
env_argv=()
while IFS='=' read -r k v; do
    [ -n "$k" ] && env_argv+=("$k=$v")
done < <(pyrun -c "
import json
for k, v in json.load(open('$expected_env_path')).items():
    print(f'{k}={v}')
")

# --- optional Nsight Systems capture, gated so the default arm is untraced ---
#     NSYS_TRACE=1 wraps the launch in `nsys launch`, which starts the application
#     immediately but collects nothing until `nsys start`. That is exactly the shape
#     asked for: warm-up runs untraced, the timed set is captured.
#     NSYS_CUDA_GRAPH_TRACE=graph|node picks --cuda-graph-trace (nsys_capture.py). Graph is
#     the default and the only mode to read ms/token or step-tail idle from -- see CLAUDE.md:
#     `node` makes cudaGraphLaunch cost ~0.77 us per traced node and fabricates a ~6 ms idle
#     gap at the end of every decode step. Node is for per-kernel attribution only, since a
#     graph-mode report's kernel table omits the graph body. Graph is refused when the arm
#     runs the RAM-miss copy engine: graph-mode tracing deadlocks its copy wait.
nsys_session=""
if [ "${NSYS_TRACE:-0}" = 1 ]; then
    command -v nsys >/dev/null 2>&1 || abort "NSYS_TRACE=1 but nsys is not on PATH"
    nsys_graph_trace=$(pyrun -c "
import json, sys
import nsys_capture
try:
    print(nsys_capture.graph_trace_mode(sys.argv[1], json.load(open('$expected_env_path'))))
except ValueError as error:
    raise SystemExit(f'REFUSE: {error}')
" "${NSYS_CUDA_GRAPH_TRACE:-}") || abort "$arm: bad NSYS_CUDA_GRAPH_TRACE"
    nsys_session="dsv41-$arm-$$"
    nsys_out_dir=$(pyrun -c "
import sys
import nsys_capture
try:
    print(nsys_capture.check_report_dir(sys.argv[1]))
except ValueError as error:
    raise SystemExit(f'REFUSE: {error}')
" "${NSYS_OUT_DIR:-/mnt/nvme1/dsv41-nsys}") || abort "$arm: bad NSYS_OUT_DIR"
    mkdir -p "$nsys_out_dir" || abort "cannot create $nsys_out_dir"
    # nsys wants ~200 MiB of scratch; divix01's /tmp is on the ~88% full root volume, and
    # exhausting it kills the server mid-run and leaves a ~361 KB report (CLAUDE.md).
    NSYS_TMPDIR=$(pyrun -c "import nsys_capture; print(nsys_capture.NSYS_TMPDIR)")
    export NSYS_TMPDIR
    mkdir -p "$NSYS_TMPDIR" || abort "cannot create NSYS_TMPDIR=$NSYS_TMPDIR"
    nsys_report=$nsys_out_dir/$arm-$(date +%Y%m%d-%H%M%S)
    # NSYS_LAUNCH_ARGS adds application-scope options, which is where --cudabacktrace and
    # --python-backtrace must go (nsys calls them "Application scope"); both also require
    # CPU sampling, which this harness turns off by default, so NSYS_SAMPLE must be set
    # with them or nsys collects the flags and no backtraces. Word-split deliberately.
    # shellcheck disable=SC2206
    nsys_launch_extra=(${NSYS_LAUNCH_ARGS:-})
    nsys_prefix=(nsys launch --session-new="$nsys_session"
                 --trace=cuda,nvtx,osrt
                 --cuda-graph-trace="$nsys_graph_trace"
                 "${nsys_launch_extra[@]}")
    echo "nsys: --cuda-graph-trace=$nsys_graph_trace NSYS_TMPDIR=$NSYS_TMPDIR report dir=$nsys_out_dir"
else
    nsys_prefix=()
fi

exec 9>"$gpu_lock" || abort "cannot open $gpu_lock"
flock --nonblock 9 || abort "cc-gpu.lock is held by another GPU job; not starting $arm"

cd "$worktree"
# DECODE_LOG_INTERVAL overrides ServerArgs' default (unset here); see
# decode_log_interval_compare.sh, which is the only caller that sets it.
decode_log_interval_py=${DECODE_LOG_INTERVAL:-None}
taskset -c "$server_cores" "${nsys_prefix[@]}" env "${env_argv[@]}" \
    PYTHONPATH="$worktree/python" PYTHONUNBUFFERED=1 \
    "$py" -c "
import sys
sys.path.insert(0, '$here')
import arm_env
argv = arm_env.ServerArgs(port=$port, decode_log_interval=$decode_log_interval_py).argv()
import os
os.execvp(argv[0], argv)
" >> "$log" 2>&1 &
launch_pid=$!
spid=$launch_pid
if [ -n "$nsys_session" ]; then
    # `nsys launch` FORKS -- measured: the wrapper and the application are different
    # pids. So $! is the wrapper here, and using it would reintroduce exactly the bug
    # fixed in eb096a697c: /proc/$spid/environ would read the wrapper's environment and
    # stop_server would kill the wrapper while the server kept the port and the GPU.
    # Resolve the server by its own cmdline instead. The port makes it unambiguous.
    spid=""
    for _ in $(seq 1 180); do
        spid=$(pgrep -f "sglang.launch_server.*--port $port" | head -1)
        [ -n "$spid" ] && break
        sleep 1
    done
    [ -n "$spid" ] || {
        kill -TERM "$launch_pid" 2>/dev/null
        nsys cancel --session="$nsys_session" >/dev/null 2>&1
        abort "$arm: nsys launched but no sglang.launch_server appeared on port $port within 180s (see $log)"
    }
    echo "nsys session=$nsys_session launcher pid=$launch_pid report=$nsys_report"
fi
echo "server pid=$spid"

stop_server() {
    # A capture still running here would leave the session behind and never write a
    # report, so end it first -- and cancel rather than stop, because every caller of
    # stop_server is an abort path whose data is not worth a report.
    if [ -n "$nsys_session" ]; then
        nsys cancel --session="$nsys_session" >/dev/null 2>&1
    fi
    kill -TERM "$spid" 2>/dev/null
    for _ in $(seq 1 120); do kill -0 "$spid" 2>/dev/null || break; sleep 1; done
    kill -KILL "$spid" 2>/dev/null
    wait "$spid" 2>/dev/null
    if [ -n "$nsys_session" ] && [ -n "$launch_pid" ]; then
        kill -TERM "$launch_pid" 2>/dev/null
        wait "$launch_pid" 2>/dev/null
    fi
    for _ in $(seq 1 60); do [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && break; sleep 2; done
}

# --- health gate: up to 900 s (smoke.sh's own shape: 180 x 5 s), curl -f treats any
#     non-2xx (including the 503 the smoke log showed while still starting) as failure,
#     so this already only succeeds on a real 200. Never shorten the per-call timeout. ---
healthy=0
for _ in $(seq 1 180); do
    sleep 5
    kill -0 "$spid" 2>/dev/null || break
    curl -sf -m 60 "http://127.0.0.1:$port/health" >/dev/null 2>&1 && { healthy=1; break; }
done
[ "$healthy" = 1 ] || { stop_server; abort "$arm never became healthy within 900s (see $log)"; }
echo "$arm healthy"

# --- verify env from the live server process, not from what we think we set ---
#     Both documents go through files: interpolating json.dumps output into a Python
#     '''...''' literal unescapes backslashes a second time, so any value containing one
#     (byobu exports a PS0 that does) produced a JSONDecodeError instead of a comparison.
server_env_actual_path=$run_dir/server-env-actual.json
pyrun -c "
import json
import tenancy
actual = tenancy.parse_proc_environ(open('/proc/$spid/environ', 'rb').read())
with open('$server_env_actual_path', 'w') as f:
    json.dump(actual, f)
" || { stop_server; abort "$arm could not read /proc/$spid/environ"; }
pyrun -c "
import json
import tenancy
actual = json.load(open('$server_env_actual_path'))
expected = json.load(open('$expected_env_path'))
tenancy.verify_env(actual=actual, expected=expected)
print('env verified OK: all', len(expected), 'vars match /proc/$spid/environ')
" || { stop_server; abort "$arm env verification failed"; }

# --- readiness gate: SM clock STABLE, decode tok/s STABLE, and no further JIT
# compilation this round -- all three on successive-rounds-agree, never a fixed
# count. /health returning 200 does not mean any of them (smoke.sh: a 34 s Triton
# compile landed inside a request after /health was already 200; a live trace run
# showed decode tok/s still climbing at the 5th identical warm-up request, 1.776 ->
# 3.154 -> 3.286 -> 3.404 -> 3.477 -- a fixed round count would have declared victory
# after request 2 or 3, well before it settled). Warm up with the same
# 256-token-prompt/real-generation shape as the timed work, in rounds, until all
# three conditions hold together in the same round.
#
# Each round writes to its OWN results file. run_capture_sessions.py resumes by
# session_id: reusing one shared warmup results file across rounds would make every
# round after the first a silent no-op (the session already "done", so the driver
# returns without ever contacting the server) -- a real bug caught while adding the
# tok/s check below, before any GPU time was spent on it. ---
prev_compile_count=0
ready=0
clock=0
tok_s=0
recent_clocks=()
recent_tok_s=()
for round in $(seq 1 "$max_warmup_rounds"); do
    round_results="$run_dir/results-warmup-$round.jsonl"
    taskset -c 8-15 env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 "$py" \
        "$worktree/scripts/expert_prediction/benchmarks/run_capture_sessions.py" \
        --port "$port" --sessions "$synthetic_sessions" --session-ids "$warmup_session_id" \
        --max-tokens "$max_tokens" --results "$round_results" --rid-suffix="-w$round"
    rc=$?
    [ "$rc" = 0 ] || { stop_server; abort "$arm warm-up round $round rc=$rc"; }
    new_compile_count=$(pyrun -c "import compile_watch as cw; print(cw.compile_events_in_range('$log', start_byte=0))")
    clock=$(pyrun -c "import clock_ramp as cr; print(cr.sample_sm_clock_mhz())")
    tok_s=$(pyrun -c "
from results_gate import load_results
row = load_results('$round_results')[0]
value = row['decode_tokens_per_sec']
if value is None:
    raise SystemExit(
        f\"warm-up round produced no decode rate: completion_tokens={row.get('completion_tokens')} \"
        f\"finish_reason={row.get('finish_reason')}. A round that generated ~1 token with \"
        f\"finish_reason=abort means the request was killed, not measured.\"
    )
print(value)
") || { stop_server; abort "$arm warm-up round $round produced no decode rate"; }
    recent_clocks+=("$clock")
    recent_tok_s+=("$tok_s")
    # Only the two most recent rounds decide stability; older rounds don't count.
    if [ "${#recent_clocks[@]}" -gt 2 ]; then
        recent_clocks=("${recent_clocks[@]: -2}")
        recent_tok_s=("${recent_tok_s[@]: -2}")
    fi
    read -r clock_stable tok_s_stable < <(pyrun -c "
import clock_ramp as cr
clocks = [$(IFS=,; echo "${recent_clocks[*]}")]
tok_s_samples = [$(IFS=,; echo "${recent_tok_s[*]}")]
# is_stable is generic (successive-samples-agree); reused for tok/s rather than
# duplicating the same check under a second name.
print(1 if cr.is_stable(clocks) else 0, 1 if cr.is_stable(tok_s_samples) else 0)
")
    quiet=$([ "$new_compile_count" = "$prev_compile_count" ] && echo 1 || echo 0)
    echo "warm-up round $round: SM clock ${clock}MHz stable=$clock_stable  decode ${tok_s} tok/s stable=$tok_s_stable  compile_events_total=$new_compile_count quiet=$quiet"
    prev_compile_count=$new_compile_count
    if [ "$clock_stable" = 1 ] && [ "$tok_s_stable" = 1 ] && [ "$quiet" = 1 ]; then
        ready=1
        break
    fi
done
[ "$ready" = 1 ] || { stop_server; abort "$arm never reached a clock-stable, tok/s-stable, compile-quiet state after $max_warmup_rounds warm-up rounds (last: clock=${clock}MHz decode=${tok_s}tok/s compile_events=$prev_compile_count; see $log)"; }
echo "$arm ready: clock=${clock}MHz decode=${tok_s}tok/s compile_events=$prev_compile_count"

if [ -n "$nsys_session" ]; then
    # Collection starts HERE, after warm-up: the clock is stable, the graph is captured
    # and no JIT compile remains, so the report contains steady-state decode only.
    nsys start --session="$nsys_session" --output="$nsys_report" \
        --sample="${NSYS_SAMPLE:-none}" --cpuctxsw="${NSYS_CPUCTXSW:-none}" --force-overwrite=true \
        || { stop_server; abort "$arm: nsys start failed for session $nsys_session"; }
    echo "nsys capture started -> $nsys_report.nsys-rep"
fi

residency_ready=$(pyrun -c "
import json, provenance
print(json.dumps(provenance.resident_bytes(['$expert_shard_dir'])))
")
pyrun -c "
import json, provenance
row = {'label': 'server_ready', **provenance.system_sample()}
with open('$boundary_path', 'a') as f:
    f.write(json.dumps(row) + chr(10))
"

# --- the timed set: one session at a time. Aborts hard on any mid-session compile
#     event. Samples clock, server cpu_s (process_tree_cpu_s of the SERVER pid, not
#     this driver process, since the server is where the decode work happens), and a
#     boundary sample bracketing each session. ---
clocks_path=$run_dir/clocks.jsonl
compile_path=$run_dir/compile.jsonl
cpu_path=$run_dir/cpu.jsonl
: > "$clocks_path"
: > "$compile_path"
: > "$cpu_path"
# DSV41_MAX_SESSIONS shortens the timed set. The verdict still expects every session it
# is given, so a shortened arm is a diagnostic capture, never a baseline number.
timed_sessions_done=0
timed_rc=0
mapfile -t timed_session_ids < <(pyrun -c "import json; print(*json.load(open('$timed_ids_path'))['session_ids'], sep=chr(10))")
for session_id in "${timed_session_ids[@]}"
do
    before_byte=$(pyrun -c "import compile_watch as cw; print(cw.log_size('$log'))")
    clock_start=$(pyrun -c "import clock_ramp as cr; print(cr.sample_sm_clock_mhz())")
    cpu_start=$(pyrun -c "import provenance; v = provenance.process_tree_cpu_s(pid=$spid); print(v if v is not None else 'None')")
    taskset -c 8-15 env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 "$py" \
        "$worktree/scripts/expert_prediction/benchmarks/run_capture_sessions.py" \
        --port "$port" --sessions "$synthetic_sessions" --session-ids "$session_id" \
        --max-tokens "$max_tokens" --results "$run_dir/results.jsonl"
    rc=$?
    clock_end=$(pyrun -c "import clock_ramp as cr; print(cr.sample_sm_clock_mhz())")
    cpu_end=$(pyrun -c "import provenance; v = provenance.process_tree_cpu_s(pid=$spid); print(v if v is not None else 'None')")
    after_byte=$(pyrun -c "import compile_watch as cw; print(cw.log_size('$log'))")
    compile_events=$(pyrun -c "import compile_watch as cw; print(cw.compile_events_in_range('$log', start_byte=$before_byte, end_byte=$after_byte))")
    contaminated=$([ "$compile_events" -gt 0 ] && echo True || echo False)
    pyrun -c "
import json
with open('$clocks_path', 'a') as f:
    f.write(json.dumps({'session_id': '$session_id', 'clock_sm_start_mhz': $clock_start, 'clock_sm_end_mhz': $clock_end}) + chr(10))
with open('$compile_path', 'a') as f:
    f.write(json.dumps({'session_id': '$session_id', 'compiled_during_session': $contaminated, 'compile_events': $compile_events}) + chr(10))
cpu_start, cpu_end = $cpu_start, $cpu_end
cpu_s = None if cpu_start is None or cpu_end is None else cpu_end - cpu_start
with open('$cpu_path', 'a') as f:
    f.write(json.dumps({'session_id': '$session_id', 'cpu_s': cpu_s}) + chr(10))
import provenance
row = {'label': 'session_$session_id', **provenance.system_sample()}
with open('$boundary_path', 'a') as f:
    f.write(json.dumps(row) + chr(10))
"
    if [ "$compile_events" -gt 0 ]; then
        stop_server
        abort "$arm: JIT compilation ($compile_events event(s)) happened during session '$session_id'; its timing is contaminated (see $log bytes $before_byte-$after_byte)"
    fi
    if [ "$rc" != 0 ]; then
        timed_rc=$rc
        break
    fi
    timed_sessions_done=$((timed_sessions_done + 1))
    if [ -n "${DSV41_MAX_SESSIONS:-}" ] && [ "$timed_sessions_done" -ge "$DSV41_MAX_SESSIONS" ]; then
        echo "stopping the timed set after $timed_sessions_done session(s): DSV41_MAX_SESSIONS=$DSV41_MAX_SESSIONS"
        break
    fi
done

if [ -n "$nsys_session" ]; then
    # Stop before the server is torn down; nsys writes the report on stop, and a killed
    # application loses it. This is the one path that stops rather than cancels.
    # Then wait for the report file to exist and stop growing before stop_server runs: a
    # node-mode report of a full arm is hundreds of MB, and killing the server under a
    # report still being written is how other scripts lost theirs.
    nsys stop --session="$nsys_session" || echo "WARNING: nsys stop failed for $nsys_session"
    nsys_session=""
    report_size=-1
    for _ in $(seq 1 180); do
        size=$(stat -c %s "$nsys_report.nsys-rep" 2>/dev/null || echo -1)
        [ "$size" -gt 0 ] && [ "$size" = "$report_size" ] && break
        report_size=$size
        sleep 10
    done
    if [ "$report_size" -gt 0 ]; then
        echo "nsys capture written: $nsys_report.nsys-rep ($report_size bytes)"
        # ~361 KB is the signature of nsys running out of scratch mid-run (CLAUDE.md).
        [ "$report_size" -ge 10000000 ] || echo "WARNING: nsys report is only $report_size bytes; check NSYS_TMPDIR space and $log"
    else
        echo "WARNING: no report at $nsys_report.nsys-rep after 30 min"
    fi
fi

# --- residency after the timed set (whole-arm, not per-session: run_capture_sessions.py
#     is unmodified and has no hook to sample it mid-session); tenancy at run end; stop
#     the server (cold-per-condition). ---
residency_after=$(pyrun -c "
import json, provenance
print(json.dumps(provenance.resident_bytes(['$expert_shard_dir'])))
")
tenancy_end=$(pyrun -c "
import json, msgspec, tenancy
print(json.dumps(msgspec.to_builtins(tenancy.capture_tenancy())))
")
stop_server
[ "$timed_rc" = 0 ] || abort "$arm driver rc=$timed_rc"

# --- result gate: exactly one record per timed session, 0 errors, or abort ---
pyrun -c "
import json
from results_gate import load_results, check_result_gate
n = len(json.load(open('$timed_ids_path'))['session_ids'])
check_result_gate(load_results('$run_dir/results.jsonl'), expected_count=n)
print(f'result gate OK: {n} records, 0 errors')
" || abort "$arm result gate failed"

# --- build the Task-1-shaped report, judge it with Task 1's own check_arm/contention/
#     generation/session_outliers (imported, not reimplemented; see verdict.py), and
#     abort only on a problem outside the acknowledged step-latency gap (report_builder.py) ---
report_path=$run_dir/report.json
verdict_path=$run_dir/verdict.txt
pyrun -c "
import json
import report_builder
import results_gate

results = results_gate.load_results('$run_dir/results.jsonl')
clocks_by_id = {r['session_id']: r for r in (json.loads(l) for l in open('$clocks_path') if l.strip())}
compile_by_id = {r['session_id']: r for r in (json.loads(l) for l in open('$compile_path') if l.strip())}
cpu_s_by_id = {r['session_id']: r['cpu_s'] for r in (json.loads(l) for l in open('$cpu_path') if l.strip())}
sessions = report_builder.merge_sessions(
    results=results, clocks_by_id=clocks_by_id, compile_by_id=compile_by_id, cpu_s_by_id=cpu_s_by_id
)
boundary_samples = [json.loads(l) for l in open('$boundary_path') if l.strip()]
report = report_builder.build_report(
    harness_provenance=json.loads('''$harness_provenance_json'''),
    server_env_actual=json.load(open('$server_env_actual_path')),
    server_env_expected=json.load(open('$expected_env_path')),
    boundary_samples=boundary_samples,
    residency={
        'dir': '$expert_shard_dir',
        'before_server': json.loads('''$residency_before''').get('$expert_shard_dir'),
        'server_ready': json.loads('''$residency_ready''').get('$expert_shard_dir'),
        'after_timed_set': json.loads('''$residency_after''').get('$expert_shard_dir'),
    },
    sessions=sessions,
)
with open('$report_path', 'w') as f:
    json.dump(report, f, indent=2)
print('wrote $report_path')
"

pyrun -c "
import json
import task1_verdict
import verdict

report = json.load(open('$report_path'))
task1_module = task1_verdict.load_task1_verdict()
# Truthy, not merely present: mirroring is on by default in arm_env, and an arm turns it
# off by overriding the var to the empty string (exl3_expert_format.exl3_mirror_config
# reads an empty value as off). Testing for the key would then judge an unmirrored arm
# against the mirrored baseline.
mirror = bool(json.load(open('$expected_env_path')).get('SGLANG_MOE_EXPERT_MIRROR_DIRS'))
result = verdict.judge(
    report, root='$worktree', head='$commit', mirror=mirror, traced=False, task1_module=task1_module,
)
with open('$verdict_path', 'w') as f:
    for n in result['notes']:
        f.write(f'NOTE {n}\n')
    for p in result['acknowledged_problems']:
        f.write(f'ACKNOWLEDGED-PROBLEM {p}\n')
    for p in result['unacknowledged_problems']:
        f.write(f'PROBLEM {p}\n')
    f.write(f\"valid (Task 1's strict definition): {result['valid']}\n\")
    f.write(f\"valid except the acknowledged step-latency gap: {result['valid_except_acknowledged_gaps']}\n\")
print(open('$verdict_path').read())
if result['unacknowledged_problems']:
    raise SystemExit('unacknowledged problems: ' + '; '.join(result['unacknowledged_problems']))
" || abort "$arm verdict failed (see $verdict_path)"

# --- manifest ---
pyrun -c "
import json

env = json.load(open('$expected_env_path'))
timed = json.load(open('$timed_ids_path'))
manifest = {
    'arm': '$arm',
    'port': $port,
    'commit': '$commit',
    'python_tree': '$python_tree',
    'generation_label': '$generation_label',
    'corpus_path': '$corpus',
    'corpus_sha256': '$corpus_checksum',
    'session_ids': timed['session_ids'],
    'session_indices': timed['indices'],
    'session_indices_override': '${DSV41_SESSION_INDICES:-}' or None,
    'warmup_session_id': '$warmup_session_id',
    'max_tokens': $max_tokens,
    'env': env,
    'tenancy_start': json.loads('''$tenancy_start'''),
    'tenancy_end': json.loads('''$tenancy_end'''),
    'report_path': '$report_path',
    'verdict_path': '$verdict_path',
}
with open('$run_dir/run-manifest.json', 'w') as f:
    json.dump(manifest, f, indent=2)
print('wrote $run_dir/run-manifest.json')
"

echo "$arm DONE run_dir=$run_dir"
