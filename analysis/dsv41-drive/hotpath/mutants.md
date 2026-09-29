# Hotpath zero-overhead Task 16: ThreadSanitizer and lease-invariant mutants

Date: 2026-09-29. divix01 kernel `6.12.0-211.60.1.el10_2.x86_64`, Clang 21 (`/usr/lib/clang/21`), GCC 14.3.1. The
laptop (kernel 7.0, GCC 15.2) was used for iteration only; every number below is divix01's unless marked laptop.

All runs were in the private worktree `/data/models/slang/nvfp4-work/wt-hotpath-mut`, detached at the pushed branch
commit named in each section. `sglang.__file__` resolved to `.../wt-hotpath-mut/python/sglang/__init__.py` every time.

## 1. TSan

**What runs.** `test/manual/dsv41/test_expert_stream_hotpath_tsan.py` builds the instrumented host TU with
`-fsanitize=thread -O1 -g` (`OPS._host_module_tsan`, reached as `variant="instr_tsan"` only when `OPS._ALLOW_TSAN`
is set). Two children each preload the compiler's TSan runtime:

- **stress child**: `run_stress(variant="instr_tsan", seconds=20, seed=7, fills=True)`. The device side (`sim_post`,
  `sim_wait`, the page helpers) goes through the TSan module too. The fill phase: every pause starts a prefill fill,
  admits a row into the filled row while the fill reads, then ends the fill or leaves it for `resume()`. When it
  leaves one, the device may post before `resume()`.
- **fills child** (preflight F16): pytest over `test_exl3_ram_miss_prefill_fills.py` and
  `test_expert_stream_ownership.py`, with every `instr` host loaded from the TSan module. It deselects (`-k`) the two
  wall-clock tests, `test_fill_wait_returns_as_a_prefix_lands` and `test_a_set_hot_burst_past_the_ring_is_applied_in_order`.

**Toolchain on divix01.** GCC's `libtsan.so` is a linker script, `INPUT ( /usr/lib64/libtsan.so.2.0.0 )`. It names a
runtime package that is not installed, and installing one needs root. So the test honors `$CXX`, as `load_jit` does:

- with `CXX=clang++` it preloads `$(clang++ -print-runtime-dir)/libclang_rt.tsan.so`;
- divix01 has no `llvm-symbolizer`. Clang's runtime then falls back to `addr2line`, trips a CHECK in its reply
  parser, and deadlocks re-symbolizing that failure (every thread parked in the runtime's futex, gdb-verified). So
  under Clang without `llvm-symbolizer`, the child runs with `symbolize=0`. `mutants.py` symbolizes a report offline
  (`addr2line`, finding the module by BuildId in the JIT cache).

**Suppressions** (`test/manual/dsv41/tsan.supp`). They are the brief's `called_from_lib` lines, plus:

- `called_from_lib:libtorch_python.so`;
- `mutex:` reports from `libtorch_python`/`libtorch_cpu`/`torchvision` and a static `python3` executable. These are
  "unlock of an unlocked mutex" reports about torch's and the interpreter's own mutexes, at import and at exit.

No `race:` suppression exists. No report was ever attributed to the host module except under a mutant.

**One false positive was fixed by construction, not suppressed.** divix01's first stress run reported a race:

- the service's read of a demand record, against the write of that record through the *uninstrumented production*
  module's `sim_post`;
- that write was an intercepted memcpy, and its release store was invisible to TSan.

The stress child now sets `_DEFAULT_VARIANT = "instr_tsan"`, so both sides are instrumented.

**Result at `445dca57ae`** (divix01):

```bash
cd /data/models/slang/nvfp4-work/wt-hotpath-mut && export PYTHONPATH=$PWD/python OMP_NUM_THREADS=8 \
  CUDA_HOME=/usr/local/cuda-13.4 CXX=clang++ \
  && taskset -c 0-63 /data/models/slang/.venv/bin/python -c "import sglang; print(sglang.__file__)" \
  && taskset -c 0-63 /data/models/slang/.venv/bin/python -m pytest test/manual/dsv41/test_expert_stream_hotpath_tsan.py \
     -q -rs -p no:randomly -p no:cacheprovider --basetemp=/data/models/slang/nvfp4-work/t16-tmp/final 2>&1 | tail -2; \
  echo "EXIT=${PIPESTATUS[0]}"
```

- **`2 passed`, `EXIT=0`, 0 `WARNING: ThreadSanitizer` lines in either child.**
- The stress child printed `TSAN-STRESS-OK 10039 fills 87`: 10,039 armed demands and 87 fills.
- The fills child printed `17 passed, 2 deselected`.
- The laptop (GCC 15 libtsan, no Clang) gave the same verdict: 0 reports at every restored-tree run below.

**Positive controls.** TSan is not idle. It reported the mutants M3, M5b and M7 below as data races in the host
module, each with a stack pair.

## 2. Mutants

Runner: `analysis/dsv41-drive/hotpath/mutants.py`. It applies each mutant as one exact-string edit, runs every target,
reverts it with `git checkout -- <file>` (and checks the file is byte-identical), then re-runs every target. No mutant
was committed.

```bash
cd /data/models/slang/nvfp4-work/wt-hotpath-mut && export OMP_NUM_THREADS=8 CUDA_HOME=/usr/local/cuda-13.4 \
  MUTANTS_TMP=/data/models/slang/nvfp4-work/t16-tmp/mut3 \
  && taskset -c 0-63 /data/models/slang/.venv/bin/python analysis/dsv41-drive/hotpath/mutants.py \
     --python /data/models/slang/.venv/bin/python --tsan-cxx clang++ --out .../mutants-divix01-final.md
```

`EXIT=0`, and `git status --short` was empty afterwards. The table is the runner's output, verbatim. "killed" means some
target went red. Each target's pytest log and TSan report (`<tag>.tsan.txt`) are in
`/data/models/slang/nvfp4-work/t16-tmp/mut3/`.

Mutants at 445dca57ae test(expert-stream): the TSan fills child runs pytest with -s so a report reaches its stderr; python /data/models/slang/.venv/bin/python; tsan CXX clang++

| mutant | file:line | target | mutant run (exit) | restored run (exit) | killed | restored green |
|---|---|---|---|---|---|---|
| M1 wait_copy_idle_owned skips drain_copy_completions() | ram_tier.h:815 | `test_expert_stream_ownership.py::test_a_pause_retires_a_copy_that_completed_while_parked` | 1 failed, 15 warnings in 3.43s (11 s) (1) | 1 passed, 15 warnings in 3.47s (11 s) (0) | KILLED | yes |
| M2 apply_command ignores kSetHot | ram_tier.h:1405 | `test_expert_stream_ownership.py::test_a_set_hot_burst_past_the_ring_is_applied_in_order` | 1 failed, 15 warnings in 3.61s (11 s) (1) | 1 passed, 15 warnings in 3.58s (11 s) (0) | KILLED | yes |
| M3 run_as_owner always applies directly (no caller_owns() check) | ram_tier.h:209 | `test_expert_stream_hotpath_tsan.py::test_the_single_owner_tier_is_race_free_under_tsan` | 1 failed, 1 warning in 14.59s | TSan data race: #0 RamTier::drain_commands() spsc_ring.h:36 vs #0 RamTier::drain_commands() spsc_ring.h:37 (29 s) (1) | 1 passed, 1 warning in 38.02s (44 s) (0) | KILLED | yes |
| M3  |  | `test_expert_stream_ownership.py::test_unpaused_eager_calls_refuse_or_snapshot` | 1 passed, 15 warnings in 3.26s (11 s) (0) | 1 passed, 15 warnings in 3.21s (11 s) (0) |  |  |
| M3  |  | `test_expert_stream_ownership.py::test_a_set_hot_burst_past_the_ring_is_applied_in_order` | 1 passed, 15 warnings in 3.56s (11 s) (0) | 1 passed, 15 warnings in 3.53s (11 s) (0) |  |  |
| M4 release_copied_owned releases every held lane, not only copy_engine ones | ram_tier.h:1553 | `test_exl3_ram_miss_copy_engine.py` | 19 passed, 15 warnings in 3.81s (11 s) (0) | 19 passed, 15 warnings in 3.74s (11 s) (0) | SURVIVED | yes |
| M4b a copy completion releases every lane its entry holds (the device's READY/LOADING lanes too) | ram_tier.h:1551 | `test_exl3_ram_miss_copy_engine.py` | 1 failed, 18 passed, 15 warnings in 40.31s (48 s) (1) | 19 passed, 15 warnings in 4.11s (12 s) (0) | KILLED | yes |
| M4b  |  | `test_expert_stream_hotpath_golden.py` | 2 failed, 1 passed, 15 warnings in 39.06s (46 s) (1) | 3 passed, 15 warnings in 3.57s (11 s) (0) |  |  |
| M4b  |  | `test_expert_stream_hotpath_stress.py` | 1 passed, 15 warnings in 11.37s (19 s) (0) | 1 passed, 15 warnings in 11.42s (19 s) (0) |  |  |
| M5 resume_locked hands the tier back (pause_epoch_ release) before set_parked(false) | ram_thread.h:132 | `test_expert_stream_hotpath_tsan.py::test_the_single_owner_tier_is_race_free_under_tsan` | 1 passed, 1 warning in 37.67s (44 s) (0) | 1 passed, 1 warning in 37.65s (44 s) (0) | SURVIVED | yes |
| M5  |  | `test_expert_stream_hotpath_stress.py` | 1 passed, 15 warnings in 11.44s (19 s) (0) | 1 passed, 15 warnings in 11.43s (19 s) (0) |  |  |
| M5  |  | `test_expert_stream_ownership.py` | 8 passed, 15 warnings in 4.55s (12 s) (0) | 8 passed, 15 warnings in 4.48s (12 s) (0) |  |  |
| M5b resume_locked hands the tier back before the owner's last writes (fill_join's epilogue) | ram_thread.h:128 | `test_expert_stream_hotpath_tsan.py::test_the_single_owner_tier_is_race_free_under_tsan` | 1 failed, 1 warning in 15.66s | TSan data race: #0 RamTier::census_locked() const ram_tier.h:1790 vs #0 RamTier::finish_fill_owned() ram_tier.h:1850 (35 s) (1) | 1 passed, 1 warning in 37.50s (44 s) (0) | KILLED | yes |
| M5b  |  | `test_expert_stream_hotpath_tsan.py::test_prefill_fills_and_ownership_are_race_free_under_tsan` | 1 passed, 1 warning in 24.24s (31 s) (0) | 1 passed, 1 warning in 23.55s (30 s) (0) |  |  |
| M5b  |  | `test_expert_stream_ownership.py` | 8 passed, 15 warnings in 4.52s (12 s) (0) | 8 passed, 15 warnings in 4.53s (12 s) (0) |  |  |
| M6 drain_copy_completions dropped from pump_demand's top | ram_tier.h:249 | `test_expert_stream_ownership.py::test_a_copying_lease_is_released_at_the_owners_next_poll_not_by_the_copy_thread` | 1 failed, 15 warnings in 3.28s (11 s) (1) | 1 passed, 15 warnings in 3.19s (11 s) (0) | KILLED | yes |
| M6  |  | `test_expert_stream_hotpath_golden.py` | 3 passed, 15 warnings in 3.56s (11 s) (0) | 3 passed, 15 warnings in 3.48s (11 s) (0) |  |  |
| M7 run_fill clears tier.filling itself (the old epilogue on the fill thread) | ram_tier.h:1833 | `test_expert_stream_hotpath_tsan.py::test_prefill_fills_and_ownership_are_race_free_under_tsan` | 1 failed, 1 warning in 16.32s | TSan data race: #0 RamTier::run_fill() ram_tier.h:1833 vs #0 RamTier::take_slot_locked() ram_tier.h:1919 (35 s) (1) | 1 passed, 1 warning in 23.72s (30 s) (0) | KILLED | yes |
| M7  |  | `test_expert_stream_hotpath_tsan.py::test_the_single_owner_tier_is_race_free_under_tsan` | 1 passed, 1 warning in 38.71s (45 s) (0) | 1 passed, 1 warning in 37.44s (44 s) (0) |  |  |
| M7  |  | `test_expert_stream_ownership.py::test_a_fill_holds_its_slots_until_the_owner_joins_it` | 1 failed, 15 warnings in 3.28s (11 s) (1) | 1 passed, 15 warnings in 3.19s (11 s) (0) |  |  |

### Mapping to the real code (Tasks 13-15)

The plan wrote the mutants against its own shapes. Each is mapped to the committed code as follows.

| # | Plan's mutation | Mutation applied (file:line at `445dca57ae`) |
|---|---|---|
| M1 | `wait_copy_idle_owned` skips `drain_copy_completions()` | As written: the drain line is deleted (`ram_tier.h:815`) |
| M2 | `apply_command` ignores `kSetHot` | As written: the `kSetHot` case no longer calls `set_hot_owned` (`ram_tier.h:1405`). The applied-commands count still rises, so only the last-writer check catches it |
| M3 | `run_as_owner` always applies directly | `if (!caller_owns())` becomes `if (false)` (`ram_tier.h:209`): every caller drains the command ring and applies its command on its own thread |
| M4 | `release_copied_owned` releases every held lane, not only `copy_engine` ones | As written: `&& held.copy_engine` is dropped (`ram_tier.h:1553`). **Equivalent**, see below |
| M4b | (M4's intent) | The loop walks the entry's lanes, not the job's, and releases every held one, including the device's READY/LOADING lanes (`ram_tier.h:1551`) |
| M5 | `resume_locked` clears `pause_requested_` before `set_parked(false)` | The code has no `pause_requested_` flag: the handshake is per-pause epochs (Task 13 deviation 3). The analog is the handback, the `pause_epoch_` release, moved before `set_parked(false)` (`ram_thread.h:132`). **Equivalent**, see below |
| M5b | (M5's hazard) | The handback moves to the top of `resume_locked`, before `fill_join` (whose epilogue is the owner's last tier write, Task 15), before `skip_advice_posted_so_far`, and before `set_parked(false)` (`ram_thread.h:128`) |
| M6 | `drain_copy_completions` dropped from `pump_demand`'s top | As written (`ram_tier.h:249`) |
| M7 | `run_fill` keeps writing `tier.filling[slot] = 0` itself | A loop after `fill_result_ = result;` clears `filling` for every claimed slot on the fill thread (`ram_tier.h:1833`) |

### What killed each mutant

- **M1:** `test_a_pause_retires_a_copy_that_completed_while_parked`. Undrained, the COPYING lease stays held, so the
  pause is refused.
- **M2:** `test_a_set_hot_burst_past_the_ring_is_applied_in_order`, through the census's last writer.
- **M3:** TSan only, which is preflight F16's ruling. It is a data race on the command ring's consumer side:
  - one side: `SpscRing::pop` (`spsc_ring.h:36`), in `RamTier::drain_commands` <- `RamTier::set_hot`
    (`ram_tier.h:1343`) <- `HostExports::set_hot` (`ffi_exports.h:948`), on the noise thread;
  - the other side: `SpscRing::pop` (`spsc_ring.h:37`), in `RamTier::drain_commands` <- `RamThread::run`, on the
    service thread.

  Neither non-TSan target caught it:
  - `test_unpaused_eager_calls_refuse_or_snapshot`: its refusals go through `require_owner`, not `run_as_owner`, and a
    snapshot answered on the caller while the service idles is still consistent.
  - `test_a_set_hot_burst_past_the_ring_is_applied_in_order`: see concern 1 in the report; the timing assertion cannot
    tell a queued burst from a direct one.

  **B3 fix round (b1e0201af0): the burst test now kills M3 without TSan.** It times the burst before `sim_wait`, not
  after the 300 ms read. Queued, the 65th `set_hot` waits for the service to drain the full ring at the end of the
  read (~0.25 s); applied directly, the burst takes about 1 ms. Laptop, private worktree
  `…/535dc605…/scratchpad/wt-b3f-red` at `b1e0201af0`. Command:
  `PYTHONPATH=$S/stubs OMP_NUM_THREADS=4 MUTANTS_TMP=$S/b3f_mut_tmp systemd-run --user --scope -q -p MemoryMax=8G -p MemorySwapMax=0 $PY analysis/dsv41-drive/hotpath/mutants.py --python $PY --only M3 --no-tsan`
  (`EXIT=0`; `sglang.__file__` under the worktree's `python/`; `git status --short` empty afterwards).

  | target | mutant run (exit) | restored run (exit) |
  |---|---|---|
  | `test_unpaused_eager_calls_refuse_or_snapshot` | 1 passed (0) | 1 passed (0) |
  | `test_a_set_hot_burst_past_the_ring_is_applied_in_order` | **1 failed (1)**: `the burst returned in 0.001 s, while the read ran: nothing was queued` | 1 passed (0) |

  M3 is now KILLED by a registered test as well as by the TSan stress child.
- **M4b:** `test_exl3_ram_miss_copy_engine.py` (1 failed) and `test_expert_stream_hotpath_golden.py` (2 failed: the
  script's `copy` step then shows the miss lane released).
- **M5b:** TSan's stress child. It is a data race:
  - one side: `RamTier::census_locked` (`ram_tier.h:1790`) <- `defers` (`:932`) <- `pump_demand` (`:279`) <-
    `RamThread::run`, on the service thread, serving a demand the device posted during the pause;
  - the other side: `RamTier::finish_fill_owned` (`ram_tier.h:1850`) <- `fill_join` (`:578`) <- `HostExports::resume`,
    on the resuming owner.

  On the laptop the same mutant raced at `begin_busy` (service `pump_demand` against the fill thread's `run_fill`): the
  service was serving while the fill thread still read. It needed the fill stress's device release before `resume()`
  (commit `d8c9dcc31f`). Without it every post came after `resume()` returned, ordered by the Python event, and M5b
  survived TSan (first divix01 run at `cea2a563fb`).
- **M6:** `test_a_copying_lease_is_released_at_the_owners_next_poll_not_by_the_copy_thread`. The golden stays green, as
  preflight row 45 predicted: the golden drains in `copy_engine_idle`.
- **M7:** two killers.
  - The TSan fills child, in `test_a_slot_being_filled_is_never_a_victim_and_cannot_be_released`, reported a data
    race:
    - one side: the fill thread's write, `RamTier::run_fill` (`ram_tier.h:1833`, the mutant line);
    - the other side: the owner's read, `RamTier::take_slot_locked` (`ram_tier.h:1919` in the mutated tree, 1918 at HEAD: the `tier.filling[slot]` check) <- `take_admit_slot_locked`
      <- `assign` (`:447`).
  - Without TSan, `test_a_fill_holds_its_slots_until_the_owner_joins_it` fails too.
  - The stress child's fill phase did not catch M7: its admission took a free or unfilled slot before it read the
    fill's flags.

### The two survivors are equivalent mutants

- **M4 (as written).** A copy job's lanes are exactly the lanes the grant tagged COPYING. `grant_lane_group_locked`
  adds a lane to `job.lanes` only in the branch that sets `kLeaseTagCopying` (`ram_tier.h` ~1049). So
  `held.copy_engine` is always true for `entry.lane[job.lanes[i].lane]`, and dropping the test changes no execution.
  M4b is the non-equivalent form, and it is killed.
- **M5 (the literal analog).** Between the moved `pause_epoch_` release and `set_parked(false)`, `parked_` reads true
  while the service runs. But every reader of `parked_` holds `caller_mutex_`, which `resume_locked` holds throughout:
  - `caller_owns()` in `run_as_owner` and `wait_copy_idle`;
  - `require_owner` in `has`/`touch`/`assign`/`release`/`fill_begin`.

  The service never reads `parked_`. So no thread can observe the window, and both orders are the same program to every
  observer. The hazard the plan's M5 aims at is a handback that precedes the owner's last write. In this code that is
  M5b, and it is killed.
