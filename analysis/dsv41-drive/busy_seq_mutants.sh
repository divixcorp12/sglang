#!/bin/bash
# The mutants that busy_seq_protocol_probe.py and unprotected_lane_eviction.py must go red against.
#
# Each mutant edits a git-archive EXPORT of the tree (never a working tree: it rewrites
# exl3_ram_miss_host.cpp), reruns the named file, and restores the C++. Recompiling the JIT module
# takes ~25 s per mutant.
#
#   git archive HEAD python test analysis | (mkdir -p "$E" && tar -x -C "$E")
#   E=<export dir> bash "$E/analysis/dsv41-drive/busy_seq_mutants.sh"
#
# Result recorded in busy_seq_evidence.out.txt. In short: every mutant goes red EXCEPT
# busy_cleared_after_done_store: the sub-microsecond gap between the clear and the done store cannot be
# hit from a Python poller, so the order "kBusySeq clears before demand_done" is established by reading
# handle_demand and pump_demand (one thread, release stores in that order), not by this probe.
set -u
: "${E:?set E to the export directory}"
export E
cd "$E/analysis/dsv41-drive"
C="$E/python/sglang/kernels/jit/csrc/moe/exl3_ram_miss_host.cpp"
cp "$C" /tmp/busyseq_cpp_orig
trap 'cp /tmp/busyseq_cpp_orig "$C"' EXIT
run() {
  env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 PYTHONPATH="$E/python" SGLANG_JIT_CACHE_DIR="$E/.jit" \
    taskset -c 0-63 /data/models/slang/.venv/bin/python -m pytest "$@" -q -p no:cacheprovider 2>&1 |
    grep -E 'FAILED|passed|failed' | sed 's/.*:://'
}
# mut NAME FILE OLD NEW: replace the one occurrence of OLD in the C++ by NEW, run FILE, restore.
mut() {
  cp /tmp/busyseq_cpp_orig "$C"
  python3 - "$C" "$3" "$4" <<'PY'
import sys
path, old, new = sys.argv[1:4]
s = open(path).read()
assert s.count(old) == 1, (old, s.count(old))
open(path, "w").write(s.replace(old, new))
PY
  echo "== MUTANT $1"
  run "$2"
  cp /tmp/busyseq_cpp_orig "$C"
}

mut victim_skips_resident_planned_lane unprotected_lane_eviction.py \
  'if (listed(protect, expert)) {' 'if (listed(protect, expert) || expert == 0) {'

mut busy_never_cleared busy_seq_protocol_probe.py \
  '    store_release(page_ + kBusySeq, 0);
    busy_since_.store(0);' \
  '    busy_since_.store(0);'

mut busy_never_set busy_seq_protocol_probe.py \
  '    busy_since_.store(now_ns());
    store_release(page_ + kBusySeq, request.seq);' \
  '    busy_since_.store(now_ns());'

mut busy_set_after_serve busy_seq_protocol_probe.py \
  '    busy_since_.store(now_ns());
    store_release(page_ + kBusySeq, request.seq);
    if (load_acquire(page_ + kFatal) != 0) counters_[kLateAfterFatal].fetch_add(1);
    int64_t rows = 0;
    const bool ok = request.armed ? serve(request, false, &rows) : touch_request(request);' \
  '    busy_since_.store(now_ns());
    if (load_acquire(page_ + kFatal) != 0) counters_[kLateAfterFatal].fetch_add(1);
    int64_t rows = 0;
    const bool ok = request.armed ? serve(request, false, &rows) : touch_request(request);
    store_release(page_ + kBusySeq, request.seq);'

mut advisory_sets_the_word busy_seq_protocol_probe.py \
  '      busy_since_.store(now_ns());
      int64_t rows = 0;
      serve(request, true, &rows);' \
  '      busy_since_.store(now_ns());
      store_release(page_ + kBusySeq, next_advice_);
      int64_t rows = 0;
      serve(request, true, &rows);
      store_release(page_ + kBusySeq, 0);'

# The clear moves from handle_demand to just after the done store in pump_demand (three runs: the gap
# it opens is sub-microsecond, so a poller lands in it rarely if ever).
cp /tmp/busyseq_cpp_orig "$C"
python3 - "$C" <<'PY'
import sys
path = sys.argv[1]
s = open(path).read()
old = "    store_release(page_ + kBusySeq, 0);\n    busy_since_.store(0);"
assert s.count(old) == 1
s = s.replace(old, "    busy_since_.store(0);")
done = "    store_release(page_ + kDemandDone, next_demand_);\n"
assert s.count(done) == 1
open(path, "w").write(s.replace(done, done + "    store_release(page_ + kBusySeq, 0);\n"))
PY
echo "== MUTANT busy_cleared_after_done_store (3 runs)"
for _ in 1 2 3; do run busy_seq_protocol_probe.py; done
cp /tmp/busyseq_cpp_orig "$C"

echo "== restored"
run busy_seq_protocol_probe.py unprotected_lane_eviction.py
