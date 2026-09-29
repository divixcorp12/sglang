"""Lease-invariant mutants of the single-owner tier (plan 2026-09-29-hotpath-zero-overhead Task 16).

Each mutant is one exact-string edit of a host header. The runner applies it in a PRIVATE worktree, runs its targets,
reverts with ``git checkout -- <file>``, runs the same targets again, and prints one table row per mutant: red with the
mutant, green once restored. A mutant is never committed; this runner is.

Run from the root of a private worktree (never cc-expert-prediction/dsv41-direct-prod), e.g. on divix01:

    git -C /data/models/slang/sglang worktree add --detach /data/models/slang/nvfp4-work/wt-hotpath-mut <commit>
    cd /data/models/slang/nvfp4-work/wt-hotpath-mut
    OMP_NUM_THREADS=8 taskset -c 0-63 /data/models/slang/.venv/bin/python \\
        analysis/dsv41-drive/hotpath/mutants.py --tsan-cxx clang++ --out <results.md>

``--tsan-cxx`` is the compiler the TSan targets build with (``CXX`` for load_jit and for the runtime lookup in
test/manual/dsv41/test_expert_stream_hotpath_tsan.py); divix01's GCC has no libtsan runtime installed, its Clang has
one. ``--only M1,M4`` runs a subset; ``--no-tsan`` drops the TSan targets."""

from __future__ import annotations

import argparse
import glob
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

H = "python/sglang/kernels/jit/csrc/moe/expert_stream/host/"
T = "test/registered/unit/kernels/"
OWN = T + "test_expert_stream_ownership.py::"
TSAN = "test/manual/dsv41/test_expert_stream_hotpath_tsan.py::"
TSAN_STRESS = TSAN + "test_the_single_owner_tier_is_race_free_under_tsan"
TSAN_FILLS = TSAN + "test_prefill_fills_and_ownership_are_race_free_under_tsan"


@dataclass
class Mutant:
    id: str
    what: str
    file: str
    old: str
    new: str
    targets: list[str]  # each run on its own; the mutant is killed when any goes red

    def line(self, root: Path) -> int:
        """The first line the edit changes."""
        text = (root / self.file).read_text()
        same = next(i for i, (a, b) in enumerate(zip(self.old + "\0", self.new + "\1")) if a != b)
        return text[: text.index(self.old) + same].count("\n") + 1


MUTANTS = [
    Mutant(
        "M1", "wait_copy_idle_owned skips drain_copy_completions()", H + "ram_tier.h",
        "wait_idle(deadline_ns);\n    drain_copy_completions();\n    return idle;\n  }\n\n  // Any thread",
        "wait_idle(deadline_ns);\n    return idle;\n  }\n\n  // Any thread",
        [OWN + "test_a_pause_retires_a_copy_that_completed_while_parked"],
    ),
    Mutant(
        "M2", "apply_command ignores kSetHot", H + "ram_tier.h",
        "case Command::kSetHot:\n          set_hot_owned(c.row, c.hot);\n          break;",
        "case Command::kSetHot:\n          break;",
        [OWN + "test_a_set_hot_burst_past_the_ring_is_applied_in_order"],
    ),
    Mutant(
        "M3", "run_as_owner always applies directly (no caller_owns() check)", H + "ram_tier.h",
        "    std::lock_guard<std::mutex> caller(caller_mutex_);\n    if (!caller_owns()) {\n      const bool wait",
        "    std::lock_guard<std::mutex> caller(caller_mutex_);\n    if (false) {\n      const bool wait",
        [TSAN_STRESS, OWN + "test_unpaused_eager_calls_refuse_or_snapshot",
         OWN + "test_a_set_hot_burst_past_the_ring_is_applied_in_order"],
    ),
    Mutant(
        "M4", "release_copied_owned releases every held lane, not only copy_engine ones", H + "ram_tier.h",
        "if (held.state == 1 && held.copy_engine) release_lease_locked<kLeasesCopied>(tier, held);",
        "if (held.state == 1) release_lease_locked<kLeasesCopied>(tier, held);",
        [T + "test_exl3_ram_miss_copy_engine.py"],
    ),
    # M4 as literally written is equivalent: a job's lanes are exactly its request's copy_engine lanes (the grant adds
    # a lane to the job only when it tags it COPYING), so the dropped test is always true. M4b is the plan's intent:
    # a completion releases every lane its entry holds, the device's READY/LOADING hit lanes included.
    Mutant(
        "M4b", "a copy completion releases every lane its entry holds (the device's READY/LOADING lanes too)",
        H + "ram_tier.h",
        "    for (int i = 0; i < job.count; ++i) {\n      LaneLease& held = entry.lane[job.lanes[i].lane];\n"
        "      if (held.state == 1 && held.copy_engine) release_lease_locked<kLeasesCopied>(tier, held);\n    }\n",
        "    for (uint32_t l = 0; l < entry.count; ++l) {\n      LaneLease& held = entry.lane[l];\n"
        "      if (held.state == 1) release_lease_locked<kLeasesCopied>(tier, held);\n    }\n",
        [T + "test_exl3_ram_miss_copy_engine.py", T + "test_expert_stream_hotpath_golden.py",
         T + "test_expert_stream_hotpath_stress.py"],
    ),
    Mutant(
        "M5", "resume_locked hands the tier back (pause_epoch_ release) before set_parked(false)", H + "ram_thread.h",
        "    tier_->set_parked(false);\n    const uint64_t epoch = pause_epoch_.load(std::memory_order_relaxed);\n"
        "    if (epoch & 1u) pause_epoch_.store(epoch + 1u, std::memory_order_release);\n",
        "    const uint64_t epoch = pause_epoch_.load(std::memory_order_relaxed);\n"
        "    if (epoch & 1u) pause_epoch_.store(epoch + 1u, std::memory_order_release);\n    tier_->set_parked(false);\n",
        [TSAN_STRESS, T + "test_expert_stream_hotpath_stress.py", T + "test_expert_stream_ownership.py"],
    ),
    Mutant(
        "M5b", "resume_locked hands the tier back before the owner's last writes (fill_join's epilogue)",
        H + "ram_thread.h",
        "  void resume_locked() {\n",
        "  void resume_locked() {\n    { const uint64_t e = pause_epoch_.load(std::memory_order_relaxed);\n"
        "      if (e & 1u) pause_epoch_.store(e + 1u, std::memory_order_release); }\n",
        # Not test_exl3_ram_miss_prefill_fills.py: its test_fill_wait_returns_as_a_prefix_lands is the known timing
        # flake (fails at the restored tree too), so the file's exit says nothing about the mutant.
        [TSAN_STRESS, TSAN_FILLS, T + "test_expert_stream_ownership.py"],
    ),
    Mutant(
        "M6", "drain_copy_completions dropped from pump_demand's top", H + "ram_tier.h",
        "    drain_copy_completions();  // D7: the COPYING leases the copy thread handed back",
        "    // M6: drain dropped  // D7: the COPYING leases the copy thread handed back",
        [OWN + "test_a_copying_lease_is_released_at_the_owners_next_poll_not_by_the_copy_thread",
         T + "test_expert_stream_hotpath_golden.py"],
    ),
    Mutant(
        "M7", "run_fill clears tier.filling itself (the old epilogue on the fill thread)", H + "ram_tier.h",
        "    fill_result_ = result;\n",
        "    fill_result_ = result;\n    for (const int64_t slot : fill_slots_) tiers_[fill_row_].filling[slot] = 0;\n",
        [TSAN_FILLS, TSAN_STRESS, OWN + "test_a_fill_holds_its_slots_until_the_owner_joins_it"],
    ),
]


FRAME = re.compile(r"^\s+#(\d+) .*\((\S+?\.so)\+(0x[0-9a-f]+)\) \(BuildId: ([0-9a-f]+)\)")


def _short(name: str) -> str:
    while True:
        shorter = re.sub(r"<[^<>]*>", "", name)
        if shorter == name:
            return re.sub(r"\([^()]*\)", "()", name.replace("sglang::expert_stream::", ""))
        name = shorter


def _module_by_build_id(build_id: str, module: str) -> str | None:
    """The JIT cache's copy of a module by BuildId: a report can name a staging path that load_jit since removed."""
    if os.path.exists(module):
        return module
    roots = [os.environ.get("SGLANG_JIT_CACHE_DIR", ""), str(Path.home() / ".cache/sglang/jit")]
    for root in filter(None, roots):
        for path in glob.glob(f"{root}/**/{Path(module).name}", recursive=True):
            out = subprocess.run(["readelf", "-n", path], capture_output=True, text=True).stdout
            if build_id in out:
                return path
    return None


def symbolize(report: str) -> str:
    """The first TSan report with the host module's frames symbolized offline (addr2line; function names without
    template arguments). The children run with symbolize=0 under Clang without llvm-symbolizer, and GCC's in-process
    symbolizer cannot read a module whose staging path is gone."""
    lines = report.splitlines()
    start = next((i for i, ln in enumerate(lines) if ln.startswith("WARNING: ThreadSanitizer")), None)
    if start is None:
        return ""
    end = next((i for i in range(start, len(lines)) if lines[i].startswith("SUMMARY:")), len(lines) - 1)
    out = []
    for ln in lines[start : end + 1]:
        match = FRAME.match(ln)
        if match and "expert_stream_host" in match.group(2):
            path = _module_by_build_id(match.group(4), match.group(2))
            if path:
                fn = subprocess.run(["addr2line", "-f", "-C", "-e", path, match.group(3)], capture_output=True,
                                    text=True).stdout.splitlines()
                if fn:
                    loc = fn[1].split("/host/")[-1] if len(fn) > 1 else ""
                    ln = f"    #{match.group(1)} {_short(fn[0])} {loc}"
        out.append(ln)
    return "\n".join(out)


def child_env(root: Path) -> dict[str, str]:
    return dict(os.environ, PYTHONPATH=os.pathsep.join(filter(None, [str(root / "python"), os.environ.get("PYTHONPATH")])))


def run_target(target: str, root: Path, python: str, tsan_cxx: str | None, tag: str) -> tuple[int, str]:
    """Run one pytest target; (pytest's exit status, a one-line summary incl. any TSan report's SUMMARY line)."""
    base = Path(os.environ.get("MUTANTS_TMP", "/tmp")) / f"hotpath-mut-{os.getpid()}" / tag
    env = child_env(root)
    if target.startswith(TSAN) and tsan_cxx:
        env["CXX"] = tsan_cxx
    cmd = [python, "-m", "pytest", target, "-q", "-p", "no:randomly", "-p", "no:cacheprovider", f"--basetemp={base}"]
    base.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    proc = subprocess.run(cmd, cwd=root, env=env, capture_output=True, text=True, timeout=3600)
    base.with_suffix(".log").write_text(proc.stdout + proc.stderr)
    lines = [ln for ln in proc.stdout.splitlines() if re.search(r"\d+ (passed|failed|skipped|error)", ln)]
    summary = lines[-1].strip() if lines else (proc.stdout.strip().splitlines() or ["?"])[-1]
    races = []
    for path in glob.glob(str(base / "**" / "child.stderr"), recursive=True):
        report = symbolize(Path(path).read_text(errors="replace"))
        if report:
            races.append(report)
    if races:
        base.with_suffix(".tsan.txt").write_text(races[0] + "\n")
        kind = races[0].splitlines()[0].removeprefix("WARNING: ThreadSanitizer: ").split(" (pid")[0]
        tops = [ln.strip() for ln in races[0].splitlines() if re.match(r"^\s+#0 ", ln)][:2]
        summary += f" | TSan {kind}: " + " vs ".join(tops)
    if "skipped" in summary and target.startswith(TSAN):
        skip = [ln for ln in proc.stdout.splitlines() if "SKIPPED" in ln]
        summary += " | " + (skip[0][:200] if skip else "")
    return proc.returncode, f"{summary} ({time.monotonic() - t0:.0f} s)"


def git(root: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True, text=True).stdout


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--tsan-cxx", default=None)
    ap.add_argument("--no-tsan", action="store_true")
    ap.add_argument("--only", default="")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    # A stopped runner still reverts the mutant it applied (the finally below runs on SystemExit, not on SIGKILL).
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    signal.signal(signal.SIGHUP, lambda *_: sys.exit(129))
    root = Path(git(Path.cwd(), "rev-parse", "--show-toplevel").strip())
    if "dsv41-direct-prod" in str(root):
        sys.exit("refusing to mutate the production checkout")
    if git(root, "status", "--porcelain", "--", H):
        sys.exit(f"{H} has local changes: mutants need a clean private worktree")
    head = git(root, "log", "-1", "--format=%h %s").strip()
    # The interpreter trap: where the targets' sglang comes from, before trusting a result.
    where = subprocess.run([args.python, "-c", "import sglang; print(sglang.__file__)"], cwd=root, env=child_env(root),
                           capture_output=True, text=True).stdout.strip()
    print("sglang.__file__ =", where, flush=True)
    if not where.startswith(str(root / "python")):
        sys.exit(f"sglang resolves to {where!r}, not this worktree's python/")
    only = set(filter(None, args.only.split(",")))
    rows = [f"Mutants at {head}; python {args.python}; tsan CXX {args.tsan_cxx or os.environ.get('CXX', 'c++')}", "",
            "| mutant | file:line | target | mutant run (exit) | restored run (exit) | killed | restored green |",
            "|---|---|---|---|---|---|---|"]
    for m in MUTANTS:
        if only and m.id not in only:
            continue
        targets = [t for t in m.targets if not (args.no_tsan and t.startswith(TSAN))]
        path = root / m.file
        text = path.read_text()
        assert text.count(m.old) == 1, (m.id, text.count(m.old))
        line = m.line(root)
        results = []
        try:
            path.write_text(text.replace(m.old, m.new))
            for i, target in enumerate(targets):
                results.append(run_target(target, root, args.python, args.tsan_cxx, f"{m.id}-red-{i}"))
                print(m.id, "RED-RUN", target, results[-1], flush=True)
        finally:
            git(root, "checkout", "--", m.file)
        assert path.read_text() == text, f"{m.id}: revert did not restore {m.file}"
        restored = []
        for i, target in enumerate(targets):
            restored.append(run_target(target, root, args.python, args.tsan_cxx, f"{m.id}-green-{i}"))
            print(m.id, "GREEN-RUN", target, restored[-1], flush=True)
        killed = any(code != 0 for code, _ in results)
        green = all(code == 0 for code, _ in restored)
        for i, target in enumerate(targets):
            rows.append(
                f"| {m.id} {m.what if i == 0 else ''} | {m.file.removeprefix(H) + ':' + str(line) if i == 0 else ''} "
                f"| `{target.split('/')[-1]}` | {results[i][1]} ({results[i][0]}) | {restored[i][1]} ({restored[i][0]}) "
                f"| {('KILLED' if killed else 'SURVIVED') if i == 0 else ''} | {('yes' if green else 'NO') if i == 0 else ''} |")
    out = "\n".join(rows)
    print(out)
    if args.out:
        Path(args.out).write_text(out + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
