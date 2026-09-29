#!/usr/bin/env bash
# Final-fix round mutants (hotpath-zero-overhead), run in a PRIVATE worktree at the head under test, never committed:
#   M1 (item 1): finish_row no longer runs the progress hook -> the prefix test must fail under LOAD=64;
#   M2 (item 5): read() calls std::uncaught_exceptions() again -> the first-request malloc test must fail.
# Each mutant is applied with sed, its target run, then reverted with `git checkout --` and the target re-run green.
# Usage: mutants.sh <private worktree> <out_dir>
set -u
wt=${1:?private worktree}; out=${2:?out dir}
here=$(cd "$(dirname "$0")" && pwd)
H=python/sglang/kernels/jit/csrc/moe/expert_stream/host
PREFIX=test/registered/unit/kernels/test_exl3_ram_miss_prefill_fills.py::test_fill_wait_returns_as_a_prefix_lands
FIRST=test/registered/unit/kernels/test_expert_stream_hotpath_shim.py::test_the_service_thread_allocates_nothing_from_its_first_request
cd "$wt" || exit 1
[ -z "$(git status --porcelain)" ] || { echo "$wt is dirty"; exit 1; }
run() {  # <tag> <runs> <node> [LOAD]
    LOAD=${4:-0} bash "$here/repeat_test.sh" "$wt" "$1" "$2" "$out" "$3"
}
echo "== M1: finish_row without the progress hook"
sed -i 's|^    if (c.progress != nullptr) c.progress(c.progress_closure);$|    // M1: removed|' $H/reader_core.h
git diff --stat; git diff | grep '^[-+] ' 
run M1-mutant 10 "$PREFIX" 64
git checkout -- $H/reader_core.h
run M1-restored 10 "$PREFIX" 64
echo "== M2: read() calls std::uncaught_exceptions() again"
sed -i 's|^    } quiesce_on_exit{this};$|    } quiesce_on_exit{this, std::uncaught_exceptions() < 0};|' $H/reader_core.h
git diff --stat; git diff | grep '^[-+] '
run M2-mutant 1 "$FIRST"
git checkout -- $H/reader_core.h
run M2-restored 1 "$FIRST"
git status --short
