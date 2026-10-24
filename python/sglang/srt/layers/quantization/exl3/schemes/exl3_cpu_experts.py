"""EXL3 binding of the CPU expert kernel, as a ``CpuExpertQuantTrait``.

The kernel is the optimized build of exllamav3's CPU MoE kernel, ``csrc/exl3/optimized/kernel.cpp``, compiled into the
expert-stream host module. ``Exl3CpuQuantTrait`` describes each streamed layer's pinned slabs and parameters for the
kernel's ``make_layer`` (``layer_spec``); the host takes the kernel in-process and its options from the module
(``expert_stream_transport._install_host_options``).

Each layer runs one of the kernel's compiled plans, chosen here once, at registration: the first available fixed plan
whose facts (the host's ``cpu_kernel_plans``) equal the layer's, read off its slabs, else the generic plan. The kernel
refuses a plan that does not accept the layer and never re-chooses per call.
"""

from typing import Any, Mapping, Optional, Sequence

import msgspec
import torch

from sglang.srt.layers.moe.cpu_experts.trait import CpuExpertLayerSpec
from sglang.srt.layers.moe.expert_engine.config import ResolvedConfig, resolved_config
from sglang.srt.layers.moe.expert_engine.formats.exl3 import EXL3_STREAMED_NAMES


class Exl3LayerFacts(msgspec.Struct, frozen=True, kw_only=True):
    """What a plan must match: the layer's shape, its gate/up (w13) and down (w2) bits, its clamp and layout."""

    hidden: int
    intermediate: int
    bits_w13: int
    bits_w2: int
    act_limit: float
    swizzled: int

    def describe(self) -> str:
        return (
            f"hidden {self.hidden}, intermediate {self.intermediate}, K w13 {self.bits_w13}, K w2 {self.bits_w2}, "
            f"act_limit {self.act_limit:g}, swizzled {self.swizzled}"
        )


class Exl3CpuPlan(msgspec.Struct, frozen=True, kw_only=True):
    """One compiled plan, as the host's ``cpu_kernel_plans`` lists it (kernel.h's ``SglangExl3CpuPlanInfo``). A fixed plan
    accepts exactly its facts; the generic plan (``fixed`` false) accepts any layer."""

    id: int
    name: str
    fixed: bool
    hidden: int
    intermediate: int
    bits_w13: int
    bits_w2: int
    swizzled: int
    act_limit: float
    min_isa: str
    available: bool

    def accepts(self, facts: Exl3LayerFacts) -> bool:
        if not self.fixed:
            return True
        return (self.hidden, self.intermediate, self.bits_w13, self.bits_w2, self.act_limit, self.swizzled) == (
            facts.hidden,
            facts.intermediate,
            facts.bits_w13,
            facts.bits_w2,
            facts.act_limit,
            facts.swizzled,
        )


def choose_exl3_cpu_plan(plans: Sequence[Exl3CpuPlan], facts: Exl3LayerFacts) -> Exl3CpuPlan:
    """The first available fixed plan that accepts ``facts``, else the generic plan."""
    for plan in plans:
        if plan.fixed and plan.available and plan.accepts(facts):
            return plan
    generic = [plan for plan in plans if not plan.fixed]
    if len(generic) != 1 or not generic[0].available:
        raise RuntimeError(f"the EXL3 CPU kernel lists no available generic plan: {plans}")
    return generic[0]


class Exl3CpuQuantTrait:
    """The EXL3 CPU kernel's trait: fp16 activations in, fp32 routed sum out.

    ``act_limit`` is the SwiGLU clamp the layers run with; the RAM-miss service fills it in at the first registration when it is
    ``None``. ``swizzled`` selects the band-contiguous weight layout, which is
    bit-identical to the native one. ``config`` is the process's resolved config (default
    ``resolved_config()`` when first needed). The plan list comes from the host module the layers run on
    (:meth:`attach_host`).
    """

    name = "exl3"
    slab_names = EXL3_STREAMED_NAMES
    x_dtype = torch.float16
    weights_dtype = torch.float16
    out_dtype = torch.float32

    def __init__(
        self,
        act_limit: Optional[float],
        swizzled: bool = False,
        config: Optional[ResolvedConfig] = None,
    ):
        self.act_limit = act_limit
        self.swizzled = swizzled
        self._config = config
        self._host: Any = None
        self._plan_list: Optional[tuple[Exl3CpuPlan, ...]] = None

    def _resolved(self) -> ResolvedConfig:
        if self._config is None:
            self._config = resolved_config()
        return self._config

    def check_environment(self) -> None:
        """Require ``SGLANG_EXL3_MOE_CPU_PIN=0``; else the kernel pins to the first cores."""
        if self._resolved().exl3.cpu.pin != "0":
            raise ValueError(
                "SGLANG_EXL3_MOE_CPU_PIN=0 must be set: the kernel would otherwise pin its workers to the first cores"
            )

    def hidden_size(self, slabs: Mapping[str, torch.Tensor]) -> int:
        """The hidden size, read off the gate's input sign vector."""
        return int(slabs["w13_suh"].shape[-1])  # the gate's input sign vector

    def _facts(self, slabs: Mapping[str, torch.Tensor], capacity: int) -> Exl3LayerFacts:
        """The facts of a layer whose first ``capacity`` slab rows the kernel can address; raises if the activation
        limit is not known yet or a slab would be misaddressed."""
        if self.act_limit is None:
            raise ValueError(
                "the EXL3 CPU kernel needs the layers' activation limit before a layer registers"
            )
        # Each trellis is [..., k/16, n/16, 16 * bits]: w13 (gate then up) fixes hidden, intermediate and their bits,
        # w2 its own bits; every other slab is checked against those.
        w13, w2 = slabs["w13_trellis"], slabs["w2_trellis"]
        hidden, intermediate, bits_w13 = int(w13.shape[-3]) * 16, int(w13.shape[-2]) * 16, int(w13.shape[-1]) // 16
        bits_w2 = int(w2.shape[-1]) // 16
        weights = hidden * intermediate  # one projection's weights; a trellis holds bits / 16 int16 elements each
        row = {  # elements per slot row: w13 rows hold gate then up, w2 rows hold down
            "w13_trellis": 2 * weights * bits_w13 // 16,
            "w13_suh": 2 * hidden,
            "w13_svh": 2 * intermediate,
            "w2_trellis": weights * bits_w2 // 16,
            "w2_suh": intermediate,
            "w2_svh": hidden,
        }
        for name in self.slab_names:
            slab = slabs[name]
            dtype = torch.int16 if name.endswith("_trellis") else torch.float16
            if (
                slab.device.type != "cpu"
                or not slab.is_contiguous()
                or slab.dtype != dtype
                or slab.shape[0] < capacity
                or slab[0].numel() != row[name]
            ):
                raise ValueError(
                    f"EXL3 slab {name} {tuple(slab.shape)} {slab.dtype} is not {capacity} contiguous CPU rows of "
                    f"{row[name]} {dtype} elements"
                )
        if tuple(w2.shape[-3:-1]) != (intermediate // 16, hidden // 16):
            raise ValueError(
                f"EXL3 slab w2_trellis {tuple(w2.shape)} is not down's [{intermediate // 16}, {hidden // 16}, 16 * K]"
            )
        return Exl3LayerFacts(
            hidden=hidden,
            intermediate=intermediate,
            bits_w13=bits_w13,
            bits_w2=bits_w2,
            act_limit=float(self.act_limit),
            swizzled=int(self.swizzled),
        )

    def attach_host(self, host) -> None:
        """Take the host (or ``CpuKernelOptions``) whose module's kernel the layers run on, for its plan list."""
        self._host = host
        self._plan_list = None

    def _plans(self) -> tuple[Exl3CpuPlan, ...]:
        """The kernel's compiled plans (the host's ``cpu_kernel_plans``), read once; their availability follows the
        module's -march."""
        if self._plan_list is None:
            if self._host is None:
                raise RuntimeError("the EXL3 CPU trait needs a host module (attach_host) to read the kernel's plans")
            self._plan_list = tuple(msgspec.json.decode(self._host.cpu_kernel_plans(), type=list[Exl3CpuPlan]))
        return self._plan_list

    def _params(self, facts: Exl3LayerFacts, plan: int) -> tuple[int, ...]:
        """The kernel's parameter list (kernel.h's ``SglangExl3CpuParams`` fields in order); the host builds and
        validates the struct."""
        return (facts.bits_w13, facts.bits_w2, facts.swizzled, plan)

    def layer_spec(self, slabs: Mapping[str, torch.Tensor], capacity: int) -> CpuExpertLayerSpec:
        """The layer's six slabs by base pointer and row size, and the kernel's parameter list, naming the plan
        chosen for the layer's facts."""
        facts = self._facts(slabs, capacity)
        plan = choose_exl3_cpu_plan(self._plans(), facts)
        views = [slabs[name] for name in self.slab_names]
        return CpuExpertLayerSpec(
            capacity=capacity,
            hidden=facts.hidden,
            intermediate=facts.intermediate,
            act_limit=facts.act_limit,
            # One slot's row: the slab is contiguous (checked), so this is its stride(0), which PyTorch does not keep
            # meaningful for a one-slot slab.
            slabs=tuple((v.data_ptr(), v[0].numel() * v.element_size()) for v in views),
            params=self._params(facts, plan.id),
            keep=tuple(views),
            plan=f"EXL3 CPU plan {plan.name} ({facts.describe()})",
        )
