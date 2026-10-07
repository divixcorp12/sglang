#!/usr/bin/env bash
# Production DSV4.1 server (port 7867): arm_env.prod_env(): the base recipe, plus the
# DSpark mode when arm_env.PROD_DSPARK is set, from the checkout this script lives in.
# Every flag and env var comes from arm_env.py, so production and the benchmark arms cannot drift apart.
#
# Usage: launch_prod.sh [start] >> server.log 2>&1
#        launch_prod.sh stop        SIGTERM the server on the production port, SIGKILL it after
#                                   LAUNCH_PROD_STOP_TIMEOUT_S (120), then wait for cc-gpu.lock to be released
#        launch_prod.sh restart     stop, then start detached with its log in a new
#                                   $LAUNCH_PROD_LOG_ROOT/prod-<timestamp>/server.log (default: cc-expert-prediction/servers)
#        DRY_RUN=1 launch_prod.sh [start|stop|restart]
#                                   print what it would do (the checkout, cores, env and argv for a start), take no lock,
#                                   stop and start nothing
# Holds cc-gpu.lock for the server's lifetime (fd 9 survives the exec) and refuses to start if it is taken.
# Once it holds the lock, points the SERVER_LOG_LINK symlink at the file its stdout goes to (not when stdout is a
# terminal or pipe), so the link always names the running server's log.
# LAUNCH_PROD_PORT and LAUNCH_PROD_GPU_LOCK exist for test_launch_prod.py, which must never touch the live server.
set -euo pipefail

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
here=$repo/benchmarks/dsv41_baseline
py=/data/models/slang/.venv/bin/python

cmd=${1:-start}
case "$cmd" in
    start | stop | restart) ;;
    *)
        echo "usage: launch_prod.sh [start|stop|restart]" >&2
        exit 2
        ;;
esac

arm() { "$py" -c "import sys; sys.path.insert(0, '$here'); import arm_env; print(arm_env.$1)"; }
server_cores=$(arm SERVER_CORES)
gpu_lock=${LAUNCH_PROD_GPU_LOCK:-$(arm GPU_LOCK)}
port=${LAUNCH_PROD_PORT:-$(arm PROD_PORT)}
dry=${DRY_RUN:-0}

# The production server's pids: a python running sglang.launch_server on the production port. Anchored on the
# interpreter so a shell whose command line merely mentions the pattern is never matched.
server_pids() { pgrep -f "^[^ ]*python[^ ]* -m sglang\.launch_server .*--port $port( |$)" || true; }

stop_server() {
    local pids
    pids=$(server_pids | tr '\n' ' ' | sed 's/ *$//')
    if [ -z "$pids" ]; then
        echo "production is not running (no sglang.launch_server on port $port)"
        return 0
    fi
    if [ "$dry" = 1 ]; then
        echo "would stop production (pid $pids)"
        return 0
    fi
    # shellcheck disable=SC2086
    kill -TERM $pids 2>/dev/null || true
    local deadline=$((SECONDS + ${LAUNCH_PROD_STOP_TIMEOUT_S:-120}))
    while [ -n "$(server_pids)" ] && [ $SECONDS -lt $deadline ]; do sleep 1; done
    if [ -n "$(server_pids)" ]; then
        echo "production (pid $pids) did not exit within ${LAUNCH_PROD_STOP_TIMEOUT_S:-120} s of SIGTERM; sending SIGKILL" >&2
        # shellcheck disable=SC2086
        kill -KILL $(server_pids) 2>/dev/null || true
        deadline=$((SECONDS + 30))
        while [ -n "$(server_pids)" ] && [ $SECONDS -lt $deadline ]; do sleep 1; done
        if [ -n "$(server_pids)" ]; then
            echo "production (pid $(server_pids | tr '\n' ' ')) survived SIGKILL" >&2
            return 1
        fi
    fi
    # The server holds cc-gpu.lock until its last process exits; a start before then would be refused.
    if ! flock --wait "${LAUNCH_PROD_LOCK_WAIT_S:-60}" "$gpu_lock" true; then
        echo "production stopped, but cc-gpu.lock ($gpu_lock) is still held after ${LAUNCH_PROD_LOCK_WAIT_S:-60} s" >&2
        return 1
    fi
    echo "stopped production (pid $pids)"
}

if [ "$cmd" = stop ]; then
    stop_server
    exit
fi

if [ "$cmd" = restart ]; then
    log_dir=${LAUNCH_PROD_LOG_ROOT:-$(arm CC)/servers}/prod-$(date +%Y%m%d-%H%M%S)
    if [ "$dry" = 1 ]; then
        stop_server
        echo "would start production, log $log_dir/server.log"
        exit 0
    fi
    stop_server
    mkdir -p "$log_dir"
    setsid nohup bash "${BASH_SOURCE[0]}" start >>"$log_dir/server.log" 2>&1 </dev/null &
    pid=$!
    sleep 5
    if ! kill -0 "$pid" 2>/dev/null; then
        echo "production failed to start; the end of $log_dir/server.log:" >&2
        tail -n 20 "$log_dir/server.log" >&2
        exit 1
    fi
    echo "started production (pid $pid), log $log_dir/server.log"
    echo "ready when: curl -sf http://127.0.0.1:$port/health (~6 min warm, 10-15 min after a code change)"
    exit 0
fi

if [ "$dry" = 1 ]; then
    echo "checkout: $repo @ $(git -C "$repo" log -1 --format='%h %s')"
    echo "cores: $server_cores  lock: $gpu_lock"
    PYTHONPATH="$repo/python:$here" "$py" -c '
import arm_env, sglang
print("sglang:", sglang.__file__)
print("dspark:", arm_env.PROD_DSPARK)
for k, v in sorted(arm_env.prod_env().items()):
    print(f"env {k}={v}")
print("argv:", " ".join(arm_env.ServerArgs.prod().argv()))
'
    exit 0
fi

exec 9>"$gpu_lock"
flock --nonblock 9 || { echo "cc-gpu.lock is held by another GPU job; not starting production" >&2; exit 1; }

log_link=${SERVER_LOG_LINK:-/data/models/slang/nvfp4-work/server.log}
log_file=$(readlink -f "/proc/$$/fd/1" || true)
if [ -f "$log_file" ]; then
    # A temporary link renamed over the old one, so the link is never missing.
    ln -sfn "$log_file" "$log_link.tmp.$$"
    mv -T "$log_link.tmp.$$" "$log_link"
    echo "$log_link -> $log_file"
fi

cd "$repo"
export PYTHONPATH="$repo/python:$here"
export PYTHONUNBUFFERED=1
exec taskset -c "$server_cores" "$py" -c '
import os
import arm_env

env = os.environ.copy()
env.update(arm_env.prod_env())
argv = arm_env.ServerArgs.prod().argv()
os.execvpe(argv[0], argv, env)
'
