---
name: magic-trace
description: Capture, validate, copy, and inspect native CPU execution with magic-trace and Intel Processor Trace. Use for requests such as "magic-trace xyz binary", tracing a running process or worker thread, or capturing execution before a rare slow operation. Covers installation, optimized C/C++ triggers, shared-library/JIT symbols, perf compatibility, and the Perfetto-based viewer.
---

# Magic Trace

Deliver a **validated timeline**, raw recording and provenance, with a direct local file link. Collector exit 0 or a file named `trace.fxt.gz` is insufficient. Read the target repository's instructions for workload, locks, build/deployment and instrumentation gates; keep application-specific choices there.

Resolve `scripts/` and `assets/` relative to this skill directory. Helpers require Python 3.9+ and no third-party packages or SGLang. Capture on a compatible **Linux Intel host**; view on any supported browser, including the user's laptop.

## Route the request

- "magic-trace ./binary args" means run that command under the collector, initially using its normal optimized build.
- For a slow-initializing service, use its existing launch harness and attach after readiness. Keep its workload/trace gate closed until **`[ Attached.`** is observed. For an existing process, attach without a restart.
- Verify `/proc/PID/exe`, command line, start identity and thread membership. Names alone do not prove ownership or role.
- Use the supplied threshold; otherwise choose from observed tails or collect a short bounded snapshot. Do not reuse another application's threshold as a universal default.
- Smoke-test before an expensive model restart. If a real capture fails, inspect symbols, thread selection and saved PT packets before repeating the expensive workload. Keep failed attempts separate.

## Install and preflight

Read [references/setup.md](references/setup.md) for dependencies, permissions and the native probe. The verified reference release is **v1.2.4**. Recheck CLI and compatibility for newer releases. Download from the official release, not a third-party executable mirror:

```bash
mkdir -p "$HOME/.local/bin"
curl -fL --retry 3 \
  https://github.com/janestreet/magic-trace/releases/download/v1.2.4/magic-trace \
  -o "$HOME/.local/bin/magic-trace-v1.2.4"
chmod 755 "$HOME/.local/bin/magic-trace-v1.2.4"
"$HOME/.local/bin/magic-trace-v1.2.4" version
perf version
sha256sum "$HOME/.local/bin/magic-trace-v1.2.4"
```

Record versions and hash. A computed SHA identifies a download; it is not a publisher signature. Inspect an existing installation before replacing it. Do not automatically change system security controls/sysctls. Own-process userspace PT was possible at `perf_event_paranoid=2`; test before seeking broader access.

Check `uname -sm`, `lscpu`, `/sys/bus/event_source/devices/intel_pt/type`, and `perf`. Upstream primarily supports Linux Intel Skylake or later; VM support is limited. Report unsupported PT. `-sampling` is a separately labeled alternative, not equivalent branch history.

## Capture a binary

Use a fresh disk output directory and retain `-working-directory`; otherwise raw files may be deleted. Start with **4M** for a single relevant thread if permitted. This is ring capacity, **not duration**; lookback depends on activity. Avoid `-full-execution` for a long service.

```bash
mkdir -p /absolute/disk/path/capture
magic-trace-v1.2.4 run \
  -snapshot-size 4M \
  -working-directory /absolute/disk/path/capture/raw \
  -output /absolute/disk/path/capture/trace.fxt.gz \
  -trigger magic_trace_stop_indicator \
  -- ./binary arg1 arg2
```

Omit `-trigger` for exit/manual snapshots. Record exact argv, tool/perf versions, binary/build identity, affinity and capture reason. For long-running programs use bounded attach below. Default `run` traces the main thread; `-multi-thread` intentionally expands scope and shortens per-thread lookback.

## Attach with a deadline

```bash
python3 scripts/capture.py \
  --tool /absolute/path/magic-trace-v1.2.4 \
  --pid PROCESS_PID --tid WORKER_TID \
  --trigger magic_trace_stop_indicator \
  --seconds 20 --output /absolute/disk/path/new-capture \
  --ready-file /absolute/disk/path/collector-ready.json
```

`--pid` is TGID; `--tid` is its chosen thread. The helper checks membership, attaches directly to TID, waits for attachment, and writes the optional ready marker. It saves `capture.log`, `capture.json`, `raw/perf.data` and `trace.fxt.gz`. At deadline it explicitly snapshots its **owned perf child** before interrupting the collector, gives decoding a separate timeout, and never signals the target. `--multi-thread` instead selects the process; don't combine it with another TID.

Check the timed fallback with the installed tool/perf pair: failed reference captures printed “Snapshot taken” yet saved **no AUXTRACE packets** after interruption. A successful triggered probe does not validate fallback behavior.

## Loaded library/JIT trigger

Read [references/triggers.md](references/triggers.md) for optimized trigger code and ELF details. After loading, select the **exact active DSO**, not the last matching mapping:

```bash
python3 scripts/resolve_trigger.py --pid PROCESS_PID --tid WORKER_TID \
  --module /absolute/path/active-library.so --symbol magic_trace_stop_indicator
python3 scripts/capture.py --tool /absolute/path/magic-trace-v1.2.4 \
  --pid PROCESS_PID --tid WORKER_TID \
  --module /absolute/path/active-library.so --symbol magic_trace_stop_indicator \
  --seconds 20 --output /absolute/disk/path/new-capture
```

The resolver reports ELF symbol value, runtime address, main-executable PIE bias and magic-trace selection. In v1.2.4 `addr:` is **executable-relative**, including DSO selections. Supplying a raw ASLR address to a PIE process adds its bias twice. Require the collector's printed `@ 0x...` to match the intended runtime address. Re-resolve for each process lifetime. Symbol definitions do not prove the selected module owns the engine; see PLT/interposition caveats in the reference.

Thread names can be inherited by OpenMP workers. Prefer an application-published TID; otherwise verify affinity/creation order against its actual threading model. Do not select the last matching name. Direct TID attachment was verified on main- and worker-thread native probes.

## perf compatibility

v1.2.4 can reject newer perf's `tr strt jmp` branch flag. Upstream master accepts that alias. For this exact failure:

```bash
export MAGIC_TRACE_PERF_PATH=/absolute/path/to/this-skill/scripts/perf_compat.py
```

The executable adapter passes recording through unchanged; only `perf script` output's trace-start flag is normalized. It preserves timestamps, addresses, line length and raw data. Do not hide other parser failures or normalize arbitrary flags. `MAGIC_TRACE_REAL_PERF` optionally selects a real perf binary other than `/usr/bin/perf`.

## Validate, deliver, view

```bash
python3 scripts/verify_trace.py /absolute/path/capture/trace.fxt.gz \
  --perf-data /absolute/path/capture/raw/perf.data \
  --log /absolute/path/capture/capture.log \
  --output /absolute/path/capture/verification.json
```

The bounded streaming validator checks record boundaries, requires FXT timeline events and (when supplied) saved Intel PT AUXTRACE payload, and hashes both files. Reference failures had 87–88 byte gzip files with **zero events**. Even megabytes of perf mappings/AUX bookkeeping need not include saved packets. Reject empty captures.

Inspect `raw/hits.sexp` and application markers to establish the trigger reason/sequence. Timed fallback is not a threshold hit. Preserve decoder errors: PT overflows leave gaps that cannot support continuous-history claims. Native control flow does not prove memory-stall causes or show GPU execution; use scheduling/resource/GPU measurements when needed. Warm-up diagnostics do not establish steady-state throughput.

For a requested remote-to-laptop copy, copy the validated FXT, log, provenance, hit metadata and optionally raw data to disk. Compare source/destination SHA256; a queued transfer is not a completed copy. Exclude huge model/build trees unless needed. Link the actual local file.

Use [magic-trace.org](https://magic-trace.org/), the project's Perfetto-based viewer: **Open trace file**, select `.fxt.gz` directly. `W/S` zoom, `A/D` pan. The viewing laptop needs no Linux collector. Consult upstream for offline hosting; don't silently upload traces to another service.

## Smoke checks and sources

`assets/trigger_probe.cpp` supplies an optimized main/worker trigger probe with an explicit readiness handshake. Follow [references/setup.md](references/setup.md); test automatic triggering **and** the bounded fallback before costly launches. Verify that the validator rejects a metadata-only failed trace.

Sources: [upstream README/support](https://github.com/janestreet/magic-trace), [v1.2.4 release](https://github.com/janestreet/magic-trace/releases/tag/v1.2.4), [current parser](https://github.com/janestreet/magic-trace/blob/master/src/perf_decode.ml), [FXT format](https://fuchsia.dev/fuchsia-src/reference/tracing/trace-format).
