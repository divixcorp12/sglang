#!/usr/bin/env bash
# verify_arms.sh TRACE RANKS ARMS OUT_DIR [extra verify_replay flags, e.g. the calibration]
set -euo pipefail
trace=$1; ranks=$2; arms=$3; out=$4; shift 4
here=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$out"; : > "$out/results.jsonl"
while read -r name flags; do
  [ -z "$name" ] && continue
  # shellcheck disable=SC2086
  # The arm's flags come last so they override the calibration (argparse keeps the last value).
  "${PYTHON:-python3}" "$here/verify_replay.py" "$trace" --ranks "$ranks" --name "$name" --out "$out/$name.json" "$@" $flags > /dev/null
  python3 -c "import json,sys; d=json.load(open('$out/$name.json')); d['name']='$name'; print(json.dumps(d))" >> "$out/results.jsonl"
  echo "$name done"
done < "$arms"
