#!/usr/bin/env bash
# Plan 2026-09-28-iopoll-read-cuts Task 6 Step 3 (and the review fix round): apply each mutant in the private worktree
# $W, run its detecting tests, record, revert. divix01 only; run from anywhere after
#   git -C /data/models/slang/sglang worktree add --detach /data/models/slang/nvfp4-work/wt-cuts-mutant <commit>
# and remove the worktree afterwards. The first run (4738822c9b) used a copy of this script scp'd to
# /mnt/nvme1/iopoll-cuts-logs (see mutants.md); the anchors below match the review-fix head.
set -u
W=/data/models/slang/nvfp4-work/wt-cuts-mutant
H=python/sglang/kernels/jit/csrc/moe/expert_stream/host
PY=/data/models/slang/.venv/bin/python
cd $W
export PYTHONPATH=$W/python OMP_NUM_THREADS=8 CUDA_HOME=/usr/local/cuda-13.4 IOPOLL_CUTS_DIR=/mnt/nvme4/nvfp4-work/iopoll-cuts-mutant
mkdir -p $IOPOLL_CUTS_DIR
RC=test/registered/unit/kernels/test_expert_stream_read_cuts.py
OPT=test/registered/unit/kernels/test_expert_stream_uring_options.py
MAN=test/manual/dsv41/test_expert_stream_read_cuts_nvme.py
run() {  # <name> <file> <python-replace-old> <new> <tests...>
  local name=$1 file=$2 old=$3 new=$4; shift 4
  $PY - "$file" "$old" "$new" <<'PYEOF' || { echo "$name: APPLY FAILED"; return; }
import sys
p, old, new = sys.argv[1:]
s = open(p).read()
assert s.count(old) == 1, (p, s.count(old))
open(p, "w").write(s.replace(old, new))
PYEOF
  flock /data/models/slang/nvfp4-work/rowimg-disk.lock taskset -c 0-63 $PY -m pytest "$@" -q -p no:randomly --basetemp=/mnt/nvme2/nvfp4-work/iopoll-cuts-mutant > /mnt/nvme1/iopoll-cuts-logs/mutant-$name.log 2>&1
  local rc=$?
  echo "$name: pytest rc=$rc ($(tail -1 /mnt/nvme1/iopoll-cuts-logs/mutant-$name.log)) failed: $(grep -E '^FAILED' /mnt/nvme1/iopoll-cuts-logs/mutant-$name.log | sed 's/^FAILED [^:]*:://; s/ - .*//' | tr '\n' ' ')"
  git checkout -- "$file"
}
run M1 $H/read_cuts.h "const bool gap = i > 0 && prev_end" "const bool gap = false && i > 0 && prev_end" \
  "$RC::test_read_cuts_planner_and_limits" "$RC::test_gap_cuts_at_slab_rows_off_the_page"
run M2 $H/read_cuts.h "static_cast<size_t>(lim.cut_bytes - leg.bytes)" "static_cast<size_t>(lim.cut_bytes + 512 - leg.bytes)" \
  "$RC::test_read_cuts_planner_and_limits" "$RC::test_cut_reads_are_byte_identical_and_within_the_cut" "$MAN::test_iopoll_punts_uncut_reads_and_never_cut_ones"
run M3 $H/uring_options.h "return wait_mode == UringWaitMode::Block && !polls_in_wait();" "return wait_mode == UringWaitMode::Block;" \
  "$OPT::test_uring_reader_driver_contract"
run M4 $H/reader_core.h "const size_t per_read = cuts_ ? leg_stride_ : 1;" "const size_t per_read = 1;" \
  "$RC::test_credit_scales_with_the_leg_bound"
run M5 $H/reader_core.h "const size_t want = cuts_ ? leg_bound(" "const size_t want = false ? leg_bound(" \
  "$RC::test_cut_reads_are_byte_identical_and_within_the_cut"
run M6 $H/read_cuts.h "  DeviceLimits fallback;
  fallback.source = \"fallback: \" + why;
  return fallback;" "  return DeviceLimits{INT64_MAX / 2, 0, \"fallback\"};" \
  "$RC::test_read_cuts_planner_and_limits"
run M7 $H/reader_core.h "    for (unsigned k = 0; k < d.legs; ++k)
      if (legs[k].state != LegState::Done) return;
    retire(index, completion, returned);" "    retire(index, completion, returned);" \
  "$RC::test_one_short_cut_leg_resubmits_only_that_leg" "$RC::test_a_held_cut_leg_keeps_its_read_unretired_and_its_pieces_unpublished"
run M8 $H/read_cuts.h "const bool gap = i > 0 && prev_end != reinterpret_cast<uintptr_t>(at) &&" "const bool gap = i > 0 &&" \
  "$RC::test_read_cuts_planner_and_limits"
run M9 $H/reader_core.h "const size_t want = cuts_ ? leg_bound(longest_read(), iovecs, min_cut_bytes()) : fixed ? iovecs : 1;" \
  "const size_t want = cuts_ ? leg_bound(longest_read(), iovecs, min_cut_bytes()) : iovecs;" \
  "$RC::test_cuts_off_keeps_todays_credit_and_legs"
git status --short
