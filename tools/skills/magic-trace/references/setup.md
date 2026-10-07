# Capture-host setup and smoke checks

## Dependencies and permissions

Use Linux x86-64 with Intel PT, the official magic-trace executable, Linux `perf` matching the running kernel, and Python 3.9+ for helpers. Binutils `nm`/`readelf` serve DSO triggers; `g++` is only for the native probe. `fzf` is optional when PID and selector are explicit. Check installed `attach -help`, `run -help`, and `decode -help` after a release change.

Typical packages, only when needed and package installation is authorized:

```bash
# Debian/Ubuntu; use the package matching the actual kernel/distribution.
sudo apt-get install linux-tools-common linux-tools-"$(uname -r)" binutils g++
# Fedora/RHEL family:
sudo dnf install perf binutils gcc-c++
```

For custom/distribution kernels without that package, consult their perf packaging rather than installing an unrelated kernel. A live trace establishes more than package presence.

Read-only preflight:

```bash
uname -sm
lscpu
cat /sys/bus/event_source/devices/intel_pt/type
cat /proc/sys/kernel/perf_event_paranoid
cat /proc/sys/kernel/yama/ptrace_scope
cat /proc/sys/kernel/perf_event_mlock_kb
ulimit -l
perf version
magic-trace-v1.2.4 attach -help
```

Absent PMU, unsupported CPU or VM can prevent PT. Paranoid settings, ptrace ownership, container capabilities or locked-memory budgets can prevent a specific attach/buffer. Inspect the actual error; do not assume root or `perf_event_paranoid=-1` is always necessary. Kernel tracing requires broader privileges than an own-process userspace capture. Request only a demonstrated necessary change; don't disable security controls as blanket “setup.”

Keep the normal optimized binary first and preserve its ELF/build ID and libraries through decoding. Debug info improves names/source mapping but cannot restore boundaries erased by inlining. If later needed, use `-O2/-O3 -g` or `RelWithDebInfo`; frame pointers help sampling/unwinding but aren't prerequisite to Intel PT. Avoid `-O0` for production-latency diagnosis.

## Native smoke probe

```bash
g++ -std=c++20 -O3 -pthread -rdynamic assets/trigger_probe.cpp -o /absolute/disk/path/trigger-probe
nm --defined-only /absolute/disk/path/trigger-probe | rg magic_trace_stop_indicator
objdump -d /absolute/disk/path/trigger-probe | rg 'call.*magic_trace_stop_indicator'
mkdir -p /absolute/disk/path/smoke
/absolute/disk/path/trigger-probe \
  /absolute/disk/path/smoke/probe.ready.json \
  /absolute/disk/path/smoke/collector.ready.json worker trigger &
```

Read the published PID/TID from `probe.ready.json`. Pass them to `scripts/capture.py`, with `--trigger magic_trace_stop_indicator`, a fresh `--output`, and the matching `--ready-file`. The probe waits for that file, executes ~50 ms, calls the trigger and remains alive briefly. Repeat with `main`, and with `worker no-trigger` **omitting `--trigger`** to exercise timed fallback. Each must produce a verified nonempty timeline. Follow the host's CPU affinity/resource policy.

## Diagnosis before another expensive run

| Symptom | Check |
|---|---|
| App marker, no profiler hit | Correct TID, active DSO, runtime attach address, PLT binding, once-only trigger consumed before attachment? |
| Empty FXT after exit 0 | Count FXT type-4 events and raw AUXTRACE packets. Total bytes/collector messages aren't proof. |
| No AUXTRACE payload | Snapshot may not have flushed. Test explicit SIGUSR2 to the owned perf recorder before stopping; no decoder can recover unsaved packets. |
| `tr strt jmp` parser exception | Use the narrow adapter or a verified upstream release accepting that alias. |
| Overflow/errors | Preserve gaps. Smaller scope/lower timing resolution can help; a larger ring doesn't necessarily fix hardware packet loss. |
| Timeout during decoding | Recording has stopped. Use a separate decode deadline; don't keep signaling the decoder. Preserve raw data. |
| Named thread idle/irrelevant | Workers inherit names; verify actual role, TID, affinity and threading model. |

`perf script -D -i raw/perf.data` exposes raw records, but can produce huge text. Prefer the bounded validator's record counts. **PERF_RECORD_AUX (11) is bookkeeping; PERF_RECORD_AUXTRACE (71) includes saved PT payload.**

Join profiler hits to app sequence/reason/time markers. Perf, FXT and GPU clocks can differ in origins/units. Do not subtract unrelated clocks without validated correlation; startup offset alone doesn't bound drift.
