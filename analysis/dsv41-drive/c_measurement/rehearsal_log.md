# Rehearsal log for the `c` measurement (CPU-only dress rehearsals of the frozen foreign-CPU gate; no GPU, no measurement of `c`)

Every rehearsal, with what overlapped it. **Contaminated runs are kept and labelled; none is merged with another and none is used as a lower bound.** Times are divix01 wall clock, 2026-09-21; where I did not record a start time it is
reconstructed from the lead's messages and marked "approx.".

| # | time | code | result | overlap / status |
|---|---|---|---|---|
| 1 | 04:57-04:59 | in force (`35dcf8215ae5c2c9…`, whole-mask foreign CPU) | **NO-GO**: harness mask [36, 38] 72% of 80 windows above 10%, reader mask [25, 27, 30] 66%; foreign 21.8 cores; load 25.7 | box clean per the lead's 04:57:40 measurement (0 pytest); the sweep relaunch was found at 05:01:58, after this ended. **Valid.** |
| 2 | approx. 04:59-05:01 | diagnostic on cores 36 and 46 | foreign ticks on the occupied core: 0 in 19/20 windows and 1 tick in one (core 36); 0 in 20/20 (core 46) | no pytest seen. Diagnostic only. |
| 3 | approx. 05:01-05:03 | proposed amendment 6 code | GO-shaped (0% of 80 windows on both masks) but **foreign CPU box-wide 29.9 cores, load 32** | **CONTAMINATED, DISCARDED.** The lead found 16 concurrent `pytest -p orderplug` processes (`t1-ordersweep`, `test_fork.py`) at 05:01:58, killed at 05:02. |
| 4 | approx. 05:04-05:06 | proposed amendment 6 code | GO (0% / 0%), foreign 20.6 cores, load 25; `pgrep` for pytest found 0 **at its end** | **UNVERIFIED, NOT TO BE RELIED ON.** It began after the 05:02 kill on the lead's timeline but I did not record a `pgrep` at its start. |
| 5 | 05:15:11-05:16:37 | in force | NO-GO 85% / 92%; foreign 30.3 cores; load 32.0 | **CONTAMINATED, DISCARDED.** `pgrep` for `orderplug|pytest` found **18 processes at the start and 14 at the end**: `sweep3.sh` / `xargs -P 3` (`t1-ordersweep`, `test_zaya_cca.py`), `taskset -c 0-63`, relaunched a fifth time. |

| 6 | 05:20:54-05:22:20 | amendment 6 v2 (used cores + SMT siblings + reserved-neighbour exclusion) | **NO-GO**: 40% / 40% of 80 windows above 10% (harness mask [46, 50], reader mask [18, 23, 26]); load 27.9-30.5, foreign 21.1 cores; physical cores with both siblings under 10%: harness 0, reader 0 | **CLEAN**: `pgrep orderplug|pytest|sweep3` = 0 at the start and 0 at the end. |

Run 5 output as printed:

```text
candidates from /sys: harness (node 0 within 32-63) [36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53] ; reader (node 1 outside 32-63, not 64-71) [18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
loadavg 32.04 30.19 28.48 over 5 s
foreign CPU box-wide: 30.30 cores (an upper bound: kernel worker time is included). The user's permanent services are the condition, not a NO-GO reason.
  java             1430.4%  pid 3314157  /usr/lib/jvm/java-25-openjdk/bin/java -p /home/dnikolaidis/bin/questdb.jar -DQuestDB-Runtime-66535 -XX:+Unlock
  aggregate-runne   335.4%  pid 2858475  /usr/bin/aggregate-runner --logdir=/home/dnikolaidis/logdir/aggregate-runner --level=2 
  nimbus_beacon_n   112.4%  pid 2518382  /home/user/nimbus_beacon_node --network=mainnet --data-dir=/data --web3-url=http://127.0.0.1:20551 --jwt-secre
  python             98.9%  pid 3192026  /data/models/slang/.venv/bin/python -m pytest -p orderplug test/registered/unit/mem_cache/test_registry.py -q 
  python             98.9%  pid 3192028  /data/models/slang/.venv/bin/python -m pytest -p orderplug test/registered/unit/mem_cache/test_registry.py -q 
  python             98.9%  pid 3192256  /data/models/slang/.venv/bin/python -m pytest -p orderplug test/registered/unit/mem_cache/test_swa_locked_full
NVMe traffic (all whole devices): read 0.0154 GB/s, write 0.0034 GB/s (limit 0.02 for an idle cell)
cores busy % (0-63): 0:18 1:20 2:36 3:30 4:72 5:43 6:21 7:15 8:22 9:14 10:98 11:82 12:16 13:76 14:19 15:25 16:26 17:74 18:59 19:42 20:46 21:42 22:100 23:49 24:14 25:32 26:35 27:36 28:17 29:10 30:9 31:57 32:29 33:13 34:44 35:27 36:22 37:21 38:28 39:65 40:34 41:32 42:21 43:38 44:18 45:98 46:13 47:24 4
harness cores (node 0, within 32-63), quietest 4: [42, 44, 46, 53] -> busy [21, 18, 13, 15]
reader cores (node 1, outside 32-63 and 64-71), quietest 4: [24, 28, 29, 30] -> busy [14, 17, 10, 9]
IDLE-CORE CHECK (informational, my own margin): NO-GO (every picked core under 5% busy and the NVMe drives under 0.02 GB/s)
REHEARSAL (the frozen gate as the harness measures it, our own load present): harness mask [42, 44]: 85% of 80 windows above 10%; reader mask [24, 28, 29]: 92% of 80 windows
NO-GO (at most 5% of rehearsal windows above the gate on either mask, NVMe idle)
then: gpu-run.sh taskset -c 42,44 <python> c_harness.py ... --reader-cpus 24,28,29
```

**Consequence.** The only clean rehearsals are run 1 (in-force code, NO-GO) and run 6 (SMT-corrected code, NO-GO). The GO-shaped runs 3 and 4 are contaminated or unverified and are not evidence; they are kept to show that the un-corrected rule certified physical cores whose SMT siblings were busy (the hole the lead measured).
