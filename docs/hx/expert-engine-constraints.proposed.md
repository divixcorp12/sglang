# Proposed ST13 expert-engine constraints

## purpose
Record the existing expert-engine boundaries needed to plan F146, GPTQ INT3 CPU experts using BesTLA primitives. This is a proposal for owner review and import, not an adopted architecture or approval to change production.

## system_shape
- **SYSTEM_SHAPE-1** Extend the existing quant- and model-agnostic expert engine rather than creating a second scheduler or worker pool. *Why:* Existing formats share admission, transport, CPU jobs and completion ownership. *Applies:* component:expert_engine, component:cpu_experts, component:formats *Source:* Current typed format and JobWindow interfaces; F146 R2 and R3.

## external_api_usage
- **EXTERNAL_API_USAGE-1** Pin adapted upstream numerical primitives and preserve their license, attribution and documented layout/arithmetic changes; do not import upstream threading or silently quantize activations. *Why:* BesTLA reuse must preserve the checkpoint contract and engine ownership. *Applies:* component:formats, component:cpu_experts *Source:* User-approved BesTLA direction; F146 R6.

## architecture_boundaries
| component | paths |
| --- | --- |
| expert_engine | python/sglang/srt/layers/moe/expert_engine/**, python/sglang/kernels/ops/moe/expert_stream_transport.py |
| cpu_experts | python/sglang/kernels/jit/csrc/moe/expert_stream/host/cpu_experts/**, python/sglang/srt/layers/moe/cpu_experts/**, python/sglang/kernels/ops/moe/cpu_kernel_build.py |
| formats | python/sglang/kernels/jit/csrc/moe/formats/**, python/sglang/kernels/jit/csrc/exl3/**, python/sglang/kernels/jit/csrc/nvfp4/**, python/sglang/kernels/jit/csrc/gptq_int3/** |

- **ARCHITECTURE_BOUNDARIES-1** Keep numerical format kernels behind the concrete F::CpuKernel and CpuExpertJobKernel contracts; Python owns configuration and tensor lifetimes, while existing native jobs own execution and bounded scratch. *Why:* The current engine directly calls typed kernels and manages their lifetime and task dependencies. *Applies:* component:expert_engine, component:cpu_experts, component:formats *Source:* Current format.h, binding.hpp and CpuExpertLayerSpec.

## state_and_data
- **STATE_AND_DATA-1** Preserve immutable source checkpoints, quantized codes, scale bit patterns, provenance and prior failed evidence; runtime row adaptation must be lossless. *Why:* Adding an execution backend must not silently create a different quantized model. *Applies:* component:formats, component:cpu_experts *Source:* User explicitly requested preservation of the existing quantized weights; F146 C1/C2/R4.

## invariants
- **INVARIANTS-1** Preserve route order, leases, stage-two readiness, output modes, bounded admission and completion publication. *Why:* Kernel integration must honor the existing concurrency and ownership contract. *Applies:* component:expert_engine, component:cpu_experts, component:formats *Source:* Existing JobWindow and CPU expert interfaces; F146 C3/C6.

## errors_and_failure
- **ERRORS_AND_FAILURE-1** Reject unsupported numerical policies and malformed layer/call metadata explicitly rather than silently falling back to another quantization or arithmetic path. *Why:* A successful execution must name the actual supported representation and semantics. *Applies:* component:formats, component:cpu_experts *Source:* Existing layer registration/admission contracts; F146 C1/R4.

## code_style
- **CODE_STYLE-1** Keep format arithmetic and native scheduling as separate responsibilities, retaining the host's compiler flags and explicit finite-value validation. *Why:* Numerical optimization must not weaken host validation or change unrelated formats. *Applies:* component:formats, component:cpu_experts *Source:* Current cpu_kernel_build.py and native CPU expert interfaces.

## testing_constraints
- **TESTING_CONSTRAINTS-1** Prove lossless format adaptation exactly, then validate numerical execution and scheduling separately; report numerical tolerances and bounded benchmark conditions without promoting component results to serving or whole-model quality. *Why:* Storage parity, arithmetic parity, concurrency and performance are different claims. *Applies:* component:formats, component:cpu_experts, component:expert_engine *Source:* F146 acceptance criteria and testing section.

## operations
- **OPERATIONS-1** Preserve unrelated working-tree edits and keep kernel tests/benchmarks isolated from active serving processes and immutable model artifacts. *Why:* This feature authorizes development and bounded verification, not production disruption. *Applies:* component:formats, component:cpu_experts, component:expert_engine *Source:* F146 R1/R7.

## operator_decisions
- **OPERATOR_DECISIONS-1** Require explicit owner direction before production deployment, checkpoint overwrite, activation-quantized variants or changing the approved numerical policy. *Why:* These expand the user-approved scope beyond lossless GPTQ INT3 CPU execution. *Applies:* component:formats, component:cpu_experts, component:expert_engine *Source:* User approval and F146 non-goals.

## changing_the_constraints
- **CHANGING_THE_CONSTRAINTS-1** Present proposed constraint changes for owner review instead of silently installing them while implementing a feature. *Why:* The story's adopted principles are owner-controlled. *Applies:* component:expert_engine, component:cpu_experts, component:formats *Source:* hx-running skill's story-principle ownership rules.
