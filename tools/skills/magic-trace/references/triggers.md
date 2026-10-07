# Optimized triggers and ELF modules

## Retain a real call

GNU/Clang Linux C++, with `<cstdint>`:

```cpp
extern "C" __attribute__((noinline, visibility("default")))
void magic_trace_stop_indicator(uint32_t sequence, int64_t delay_ns, int reason) {
  asm volatile("" : : "r"(sequence), "r"(delay_ns), "r"(reason) : "memory");
}
```

Define once in a `.cc` file. A header-only definition can use C++ `inline` for ODR while retaining **`noinline`**; the keyword and optimizer inlining differ. An empty call may disappear despite noinline; volatile asm consumes its arguments. Verify actual optimized code with `nm`, `objdump` and a live probe, including LTO when used.

Give the trigger an explicit compile-time switch. It can share an existing diagnostic build when those counters are wanted; a counters-off comparison needs an independent minimal switch containing just the required timing, gate and trigger call. Neither `noinline`, `used`, nor volatile asm can revive a call in a discarded `if constexpr (false)` branch or a preprocessor-disabled block. Verify both the symbol **and the reachable call site** in the requested build.

Record reason, elapsed duration, request sequence and TID in a bounded buffer when detailed instrumentation is enabled; a minimal mode can publish only one trigger record after the measured interval. Trigger once after a configurable threshold; check the capture gate **before spending the once-only trigger**. No allocation/formatting/IO in the measured interval. Place the trigger after the suspicious interval to capture preceding history. Keep detection, preparation, execution and completion timestamps distinct and specify what the threshold excludes. The uprobe itself costs time: end the measured interval before calling it. A threshold needs at least the clocks needed to measure its interval; describe a minimal-trigger trace as detailed-counters-off, rather than claiming zero instrumentation.

For main-executable symbols use the stable `extern "C"` name, retained/exported (e.g. probe `-rdynamic`). C++ selectors otherwise need a mangled name or fuzzy lookup. Main-executable lookup need not search `dlopen` libraries.

## Loaded library/JIT definitions

Inspect maps after loading; select by exact path, role and build identity, or an app-published function pointer. Several variants can export the same name. An arbitrary matching address can be executable but never called.

For the chosen ELF64 little-endian module:

1. Obtain the defined dynamic symbol's ELF value with `nm -D --defined-only`.
2. Derive ELF load bias from PT_LOAD and the offset-zero mapping; ELF `st_value` isn't a file offset.
3. `runtime_trigger = module_bias + symbol_value`.
4. v1.2.4 interprets `addr:` relative to `/proc/TID/exe`: `selection = runtime_trigger - executable_bias`. Normal ET_EXEC bias is zero.
5. Require logged `@ 0x...` to match the runtime trigger. Recompute after every restart/ASLR change.

`scripts/resolve_trigger.py` implements this restricted ELF64 path and rejects ambiguous basenames. Its JSON includes paths, biases, hash and TID membership. It resolves a **definition**, not arbitrary interposition or the application's role mapping.

For PLT/interposition uncertainty:

```bash
readelf -rDW /absolute/path/caller.so | rg magic_trace_stop_indicator
objdump -d -C /absolute/path/caller.so | rg -B 2 -A 2 magic_trace_stop_indicator
```

`-rDW` includes dynamic relocations that plain `-rW` may miss. If needed/permitted, inspect the owned process's corresponding GOT slot. A pointer to `symbol@plt+6` is **unresolved lazy binding**, not proof of another definition. A pointer after a real call or a profiler hit instruction address provides stronger evidence. A definition alone does not prove which library owns the live engine.

## Thread scope

Verify `/proc/PID/task/TID/status` Tgid. Direct `-pid TID` was validated on a non-main worker with v1.2.4. Multi-thread tracing divides buffer resources. OpenMP workers can inherit the leader's comm; “last matching name” is wrong. Minimum TID/creation order is only a shortcut when the application's thread model establishes it, not a universal Linux identity rule.
