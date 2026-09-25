#!/usr/bin/env bash
# Production DSV4.1 server (port 7867): the arm_env base recipe, unchanged, from the checkout this script lives in.
# Every flag and env var comes from arm_env.py, so production and the benchmark arms cannot drift apart.
#
# Usage: launch_prod.sh >> server.log 2>&1
#        DRY_RUN=1 launch_prod.sh    print the checkout, cores, env and argv, take no lock, start nothing
# Holds cc-gpu.lock for the server's lifetime (fd 9 survives the exec) and refuses to start if it is taken.
set -euo pipefail

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
here=$repo/benchmarks/dsv41_baseline
py=/data/models/slang/.venv/bin/python

server_cores=$("$py" -c "import sys; sys.path.insert(0, '$here'); import arm_env; print(arm_env.SERVER_CORES)")
gpu_lock=$("$py" -c "import sys; sys.path.insert(0, '$here'); import arm_env; print(arm_env.GPU_LOCK)")

if [ "${DRY_RUN:-0}" = 1 ]; then
    echo "checkout: $repo @ $(git -C "$repo" log -1 --format='%h %s')"
    echo "cores: $server_cores  lock: $gpu_lock"
    PYTHONPATH="$repo/python:$here" "$py" -c '
import arm_env, sglang
print("sglang:", sglang.__file__)
for k, v in sorted(arm_env.base_env().items()):
    print(f"env {k}={v}")
print("argv:", " ".join(arm_env.ServerArgs.prod().argv()))
'
    exit 0
fi

exec 9>"$gpu_lock"
flock --nonblock 9 || { echo "cc-gpu.lock is held by another GPU job; not starting production" >&2; exit 1; }

cd "$repo"
export PYTHONPATH="$repo/python:$here"
export PYTHONUNBUFFERED=1
exec taskset -c "$server_cores" "$py" -c '
import os
import arm_env

env = os.environ.copy()
env.update(arm_env.base_env())
argv = arm_env.ServerArgs.prod().argv()
os.execvpe(argv[0], argv, env)
'
