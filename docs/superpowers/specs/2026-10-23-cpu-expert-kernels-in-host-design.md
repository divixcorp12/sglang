# CPU expert kernels compiled into the expert-stream host

Status: approved in conversation 2026-10-23 (owner). Branch `cpu-kernel-in-host`, cut from `ram-prefetch-pool-serve` at
3157590740. Supersedes the "Decision" of `2026-10-04-cpu-expert-kernel-interface-design.md` (a virtual
`CpuExpertKernel` crossing library boundaries).

## Why

Each format's CPU expert kernel is a separate library (EXL3: inside the exllamav3 torch extension; NVFP4: a tvm-ffi
`.so`) with its own flags. Python reads the kernel's address from that library (`kernel_address`) and hands it to the
expert-stream host, which calls it through a vtable. Around that boundary sit: `binding.hpp`'s `ExternalCpuKernel`,
params packed as at most 16 bytes (`params_as<>`), ABI-restricted signatures (`std::span`/`std::array` only), torch ops
for the EXL3 kernel's options/plans, seven link stubs the exllamav3 extension needs because our kernel replaces its
`cpu/moe_mul1.cpp`, and runtime ISA detection with a cap (`SGLANG_EXPERT_STREAM_CPU_MAX_ISA`).

The 2026-10-04 reasons for that shape no longer hold:

- Flags: one flag set serves both formats now. The owner dropped NVFP4's bit-exactness against GGML (2026-10-23).
- Template propagation: the host is already compiled once per format (`-DSGLANG_EXPERT_FORMAT=<name>`,
  `expert_stream/format.h`), so `F::CpuKernel` can name the kernel type directly, with no template parameter threaded
  through `RamTier`/`NumaGroup`/`CopyEngine`.
- Build time: compiling only the build machine's ISA tier makes the EXL3 kernel cheap enough to compile into every host
  variant (measured below).

## Decisions

- D1. The ISA tier is a compile-time constant from the compiler's macros under the resolved `-march`
  (`__AVX512VBMI__` -> Vbmi, `__AVX512VNNI__` -> Vnni, `__AVX512BW__` -> Bw, `__AVX2__` -> Avx2, else Scalar). Runtime
  detection (`detect_isa`, `__builtin_cpu_supports`), `configure_isa`, the cap and its report are removed. A lower tier is
  tested by building at a lower `-march` (`SGLANG_JIT_HOST_MARCH`, e.g. `skylake-avx512` Bw, `x86-64-v3` Avx2, `x86-64`
  Scalar), which the build key already includes. `SGLANG_EXPERT_STREAM_CPU_MAX_ISA` and `_REPORT_ISA` are removed.
- D2. Each build compiles both plan shapes at that one tier: EXL3's DSV4.1 plan (only when the tier is Bw or above;
  `if constexpr`) and its generic plan. A mixed-K checkpoint (DSV4.1-1017: K3 layers on the DSV4.1 plan, layers 18-22
  K2 on the generic plan) needs both in one process. No per-model plan selection in the build.
- D3. Two flag sets, chosen per source file (owner, 2026-10-23, after step 1's bisection). Kernel files: `-Ofast
  -march=<resolved>`, exactly EXL3's flags today. Host files keep the host module's current flags (no fast-math), so
  host numerics (calibration fits, the gate scorer) and NaN checks (`gate_scorer.h`, `ram_tier_fill_copy_cpu.inc.h`)
  do not change. One flag set was tried first:
  `-fno-finite-math-only` in the EXL3 kernel made 70 of 72 VNNI cases differ, because glibc declares the vector `expf`
  (`_ZGVdN8v_expf`, `_ZGVbN4v_expf`) only under full fast-math, so SiLU fell back to scalar `expf`. A kernel file's own
  finite checks (the call check's non-finite-weight refusal, `cpu_experts/expert_forward.hpp`) use a bit-pattern test
  that finite-math cannot fold. Step 2 adds per-source flags to `load_jit`.
- D4. NVFP4 drops its bitwise contract with GGML and its `-ffp-contract=off`/baseline-arch build, and takes D3's kernel
  flags; its GGML checks become tolerance checks. Its row-vs-one-row invariance test stays only if it still holds.
- D5. The format's kernel sources are compiled straight into the host module (extra `cpp_files` of its `load_jit`): no
  separate library, no static archive, no address hand-off. Each kernel stays one translation unit; its internals may
  keep internal linkage, and only the entry type the host calls has a named namespace.
- D6. The host calls a concrete, non-virtual `F::CpuKernel` with a typed `Params`. Removed: `kernel.hpp`'s virtual
  `CpuExpertKernel`, `binding.hpp`'s `ExternalCpuKernel`, `params_as<>` and the 16-byte cap, `kernel_address` (EXL3 torch
  op, NVFP4 `ffi.cpp`), the EXL3 kernel's torch dependency (`c10::Half`, `TORCH_CHECK`, ATen in `moe_mul1.h`), and the
  exllamav3 extension's CPU replacement: the extension compiles upstream's own `cpu/moe_mul1.cpp` again, so the seven link
  stubs in `exl3/optimized/kernel.cpp` go. `UnsupportedCpuKernel` stays, concrete, for formats with no CPU kernel.
- D7. Python keeps configuring the engine once at startup (it owns the pinned tensors and parses checkpoint metadata,
  per-layer K included): `enable_cpu_experts`/`draft_cpu_open` lose the kernel argument; `set_cpu_layer`/
  `draft_cpu_set_layer` take the format's params as an int64 list that C++ converts to the typed `Params` and validates;
  the EXL3 torch ops `params`/`plans`/`plan_calls`/`chunk_m`/`configure` become host FFI exports.
- D8. Tests inject a fake kernel through a test-only `fake` format (EXL3's row layout, a fake `CpuKernel`), replacing
  `test_kernel_address`.

## Measurements (divix01, Xeon 8280 = VNNI tier, GCC 15, D3's flags)

| EXL3 `optimized/kernel.cpp` as one TU, containing | wall | peak RSS |
|---|---|---|
| all 7 plans (today: DSV4.1 Bw and Bw+VNNI, generic x 5 tiers) | 100.4 s | 2.5 GB |
| DSV4.1@VNNI + generic@VNNI (D1 + D2 on this host) | 68.5 s | 1.9 GB |
| NVFP4 `optimized/kernel.cpp` / vendored `nvfp4.c` | 6.3 s / 0.4 s | 0.4 GB |

The kernel TU compiles in parallel with the host TU (50-100 s per variant), so a cold host build grows little.
Scratch: `divix01:/mnt/nvme1/ksplit-20261023/`.

## Steps

Each step lands as commits on `cpu-kernel-in-host` and is verified before the next starts.

1. Compile-time tier, in place (kernels still in their current libraries). D1, D2, D3 for both kernels, D4. The EXL3
   extension's `-march=native` becomes the resolved `-march` so `SGLANG_JIT_HOST_MARCH` reaches it. Python drops the
   cap/report envs and options (`environ.py`, `expert_engine/config.py`, `removed_env.py` gets the migration entry).
   Tests and benches that capped the tier build at a lower `-march` instead.
2. Kernels into the host module (D5) and the Python plumbing (D7): per-source flags in `load_jit` (NVFP4's vendored
   `nvfp4.c` compiled as C through them); the EXL3 kernel without torch; its torch ops become host exports; the
   exllamav3 extension compiles upstream's `cpu/moe_mul1.cpp` again and the link stubs go; Python passes no kernel
   address. The virtual interface stays for this step, so tests keep injecting a fake kernel (an override honoured
   only outside the prod variant).
3. Concrete kernel type and typed params (D6) with the fake-kernel test format (D8); delete the old interface and the
   override; docs (both kernel READMEs, the 2026-10-04 spec marked superseded).

## Verification

- EXL3 at the VNNI tier is bit-identical before and after each step (its existing exact checks: `test/manual/dsv41`
  CPU forward checks, `test_cpu_expert_engines_exl3.py`), and the CPU forward bench
  (`expert_stream/bench`, `exl3_cpu_*`) is unchanged within noise. Perf runs take `cc-gpu.lock` (performance-sensitive).
- Tiers below VNNI: the same checks built at `skylake-avx512`, `x86-64-v3`, `x86-64`.
- NVFP4: its kernel tests at tolerance.
- Per step, only the test files the step touches (run protocol), on divix01 in a private worktree, GPU files under
  `-n 8` after a serial JIT warm-up; the merge base's counts measured once.
- After step 4: a production smoke serve of DSV4.1-1017.

## Non-goals

- The exllamav3 extension's vendored-kernel build flavor and the bench's baseline backend (`exl3/moe_mul1.cpp`): a
  separate decision (memory note "exl3 moe_mul1.h cleanup", option 3).
- Moving kernels into `kernels/aot/csrc/cpu` (upstream's prebuilt CPU backend).
- Any change to kernel arithmetic or plan logic.
