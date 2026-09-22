#!/usr/bin/env bash
# Measures whether --decode-log-interval=1 is an observer effect on decode tok/s,
# before trusting it as a step-latency source (README "One harness, not two", option
# 2). Runs the SAME arm twice, back to back: once at the framework default
# (decode_log_interval=40, one scheduler log-and-format per 40 decode steps) and once
# at decode_log_interval=1 (one per step — every step's `gen throughput (token/s)`
# line, on the path whose latency we would be reading), then pairs the two arms'
# per-session decode tok/s the same way any other V2 storage-change comparison would.
#
# Written per the team lead's request; NOT run by the agent that wrote it — GPU time
# on this card is the team lead's to spend.
#
# Usage: decode_log_interval_compare.sh <port40> <port1>
#   Two distinct ports (default server ports below), so both arms use fresh ones.
#
# Read the result as: if the two arms' decode tok/s agree (within whatever this
# protocol's own noise floor turns out to be — see README's noise-floor section, still
# unmeasured for this protocol as of this writing), interval=1 is free and should be
# preferred over client_inter_token_latency_s for anything gate-related, keeping the
# client-side number only as an independent cross-check. If they disagree, interval=1
# is an observer effect and unusable for timed arms (still fine for one-off diagnosis).
set -uo pipefail

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
port40=${1:?port for the decode_log_interval=40 arm}
port1=${2:?port for the decode_log_interval=1 arm}

echo "=== arm A: decode_log_interval=40 (framework default) ==="
DECODE_LOG_INTERVAL=40 "$here/run_arm.sh" decode-log-interval-40 "$port40"
rc40=$?
[ "$rc40" = 0 ] || { echo "arm A failed (rc=$rc40); not running arm B" >&2; exit "$rc40"; }

echo "=== arm B: decode_log_interval=1 (every step logs) ==="
DECODE_LOG_INTERVAL=1 "$here/run_arm.sh" decode-log-interval-1 "$port1"
rc1=$?
[ "$rc1" = 0 ] || { echo "arm B failed (rc=$rc1)" >&2; exit "$rc1"; }

out_root=${DSV41_RUN_ROOT:-/data/models/slang/nvfp4-work/cc-expert-prediction/dsv41-baseline}
dir40=$(ls -td "$out_root/servers/decode-log-interval-40/run-"* | head -1)
dir1=$(ls -td "$out_root/servers/decode-log-interval-1/run-"* | head -1)

echo "=== paired comparison (arm A vs arm B, per session) ==="
PYTHONPATH="$here" python3 "$here/paired.py" "$dir40" "$dir1" --a-name interval40 --b-name interval1
