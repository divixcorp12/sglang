#!/usr/bin/env bash
# One lock hold for the registered `c` run (C_MEASUREMENT_PREREG.md sections 10, 13, 16, 22): the pre-flight rehearsal and the harness
# run back to back, so no other lane's job can sit between the GO and the run, and the masks the rehearsal picked are the masks used.
#   gpu-run.sh bash run_c.sh <outdir> [--no-nvme]
# Order of the checks: verify_hashes (aborts on any mismatch), quiet_check --rehearse 40 (must print GO; NO-GO or a lane of ours
# arriving means the window does not open and NOTHING is run), then --check-only, then the run. It never analyses. Analysis is a
# separate step and its first line is the verdict.
set -u
O=${1:?outdir}; NVME=1; [ "${2:-}" = "--no-nvme" ] && NVME=0
D=$(cd "$(dirname "$0")/../c_measurement" && pwd)
REPO=$(cd "$D/../../.." && pwd)
PY=/data/models/slang/.venv/bin/python
export PYTHONPATH=$REPO/python CUDA_HOME=/usr/local/cuda-13.2 SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
mkdir -p "$O"
{ echo "start $(date -Is) load $(cat /proc/loadavg)"; nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader; echo "repo $REPO $(git -C "$REPO" rev-parse HEAD)"; } > "$O/window.txt"
python3 "$D/verify_hashes.py" > "$O/verify_hashes.txt" 2>&1 || { echo "HASH MISMATCH, nothing run"; cat "$O/verify_hashes.txt"; exit 4; }
$PY "$D/quiet_check.py" --rehearse 40 > "$O/quiet_check.txt" 2>&1; QC=$?
if [ $QC -ne 0 ]; then echo "quiet_check did not print GO (rc $QC): the window does not open, nothing run"; tail -6 "$O/quiet_check.txt" | cut -c1-300; exit 5; fi
HM=$(sed -n 's/^then: gpu-run.sh taskset -c \([0-9,]*\) .*--reader-cpus \([0-9,]*\)$/\1/p' "$O/quiet_check.txt"); RM=$(sed -n 's/^then: gpu-run.sh taskset -c \([0-9,]*\) .*--reader-cpus \([0-9,]*\)$/\2/p' "$O/quiet_check.txt")
[ -n "$HM" ] && [ -n "$RM" ] || { echo "could not read the picked masks from quiet_check output"; exit 6; }
echo "masks: harness $HM reader $RM nvme=$NVME" | tee -a "$O/window.txt"
ARGS=(--out "$O" --repo "$REPO" --reader-cpus "$RM"); [ $NVME -eq 1 ] && ARGS+=(--with-nvme)
taskset -c "$HM" $PY "$D/c_harness.py" "${ARGS[@]}" --check-only > "$O/check_only.txt" 2>&1 || { echo "check-only FAILED"; tail -5 "$O/check_only.txt"; exit 7; }
tail -1 "$O/check_only.txt"
rm -f "$O/meta.json"
taskset -c "$HM" $PY "$D/c_harness.py" "${ARGS[@]}" > "$O/harness.log" 2>&1; echo "harness rc $?" | tee -a "$O/window.txt"
{ echo "end $(date -Is) load $(cat /proc/loadavg)"; nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader; } >> "$O/window.txt"
ls "$O"
