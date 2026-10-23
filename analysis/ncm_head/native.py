"""Optional native EXL3 bridge. Importing this module does not load SGLang or CUDA.

The quantized forward is unchanged. Backward uses the reconstructed frozen linear
weight: an explicitly declared straight-through low-precision surrogate. It does
not claim a mathematical derivative through trellis kernels or floating-point casts.
Native execution is not part of CPU setup validation.
"""

import torch
from torch import nn

from .model import rms


class NativeProjector(nn.Module):
    def __init__(self, tensors, norm_weight, eps, reconstruction):
        super().__init__()
        self.register_buffer("trellis", tensors.trellis)
        self.register_buffer("suh", tensors.suh)
        self.register_buffer("svh", tensors.svh)
        self.register_buffer("norm_weight", norm_weight.detach().clone())
        self.register_buffer("backward_weight", reconstruction.detach())  # [in,out]
        self.eps = eps
        self.mul1 = tensors.mul1

    def forward(self, hidden):
        from sglang.srt.layers.quantization.exl3.ops import Exl3Tensors, exl3_linear

        native = Exl3Tensors(self.trellis, self.suh, self.svh, self.mul1)
        normalized = rms(hidden, self.eps) * self.norm_weight.to(hidden.dtype)
        weight = self.backward_weight

        class FrozenExl3(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                ctx.save_for_backward(weight)
                ctx.input_dtype = x.dtype
                return exl3_linear(x, native)

            @staticmethod
            def backward(ctx, grad):
                (w,) = ctx.saved_tensors
                return (grad.float() @ w.float().T).to(ctx.input_dtype)

        return FrozenExl3.apply(normalized).float()


def from_loaded_dspark(draft_model):
    """Call only during an explicitly authorized native probe/capture, TP=1.

    Returns frozen tensors/projector for an isolated research Head. No replacement
    of the model's runtime modules is performed. Reconstruction allocates ~1.23 GiB
    for the 5120x129280 fp16 head and can initialize the extension/GPU.
    """
    from sglang.srt.layers.quantization.exl3.ops import exl3_dense_weight

    lm = draft_model.lm_head
    if getattr(lm, "tp_size", 1) != 1:
        raise ValueError("native reference supports TP=1 only")
    parts = getattr(lm, "exl3_tensors", None)
    if parts is None or len(parts) != 1:
        raise ValueError("expected one loaded native EXL3 vocabulary projection")
    t = parts[0]
    reconstructed = exl3_dense_weight(t)
    projector = NativeProjector(
        t, draft_model.stages[-1].norm.weight, draft_model.norm_eps, reconstructed
    )
    markov = draft_model.markov_head
    return {
        "embedding": draft_model.embed_tokens.weight.detach(),
        "projector": projector,
        "dependency_embedding": markov.markov_w1.weight.detach(),
        "dependency_output": markov.markov_w2.weight.detach(),
    }


@torch.inference_mode(False)
@torch.enable_grad()
def check_loaded_boundary(draft_model, inputs, tolerances):
    """Future native probe on actual captured post-HC rows; does not self-approve readiness."""
    parts = from_loaded_dspark(draft_model)
    x = inputs.detach().clone().requires_grad_(True)
    output = parts["projector"](x)
    reference = draft_model._logits_from_x_post_hc(inputs).detach().float()
    loss = output.square().mean()
    grad = torch.autograd.grad(loss, x)[0]
    diff = (output - reference).abs()
    result = {
        "max_abs_logit_error": float(diff.max()),
        "mean_abs_logit_error": float(diff.mean()),
        "gradient_finite": bool(torch.isfinite(grad).all()),
        "gradient_nonzero": bool(grad.abs().sum() > 0),
        "backward_semantics": "reconstructed-weight straight-through surrogate",
    }
    result["passed"] = (
        result["gradient_finite"]
        and result["gradient_nonzero"]
        and bool(
            torch.allclose(
                output, reference, atol=tolerances["atol"], rtol=tolerances["rtol"]
            )
        )
    )
    return result
