"""Report which DSV4 prefill indexer-logits path this machine takes (phase 0 of the layer-major plan).

Usage: PYTHONPATH=<wt>/python python phase0_probe.py
"""

import importlib

from sglang.srt.environ import envs

print("SGLANG_OPT_USE_TOPK_V2 =", envs.SGLANG_OPT_USE_TOPK_V2.get())
try:
    deep_gemm = importlib.import_module("deep_gemm")
    print("deep_gemm.fp8_fp4_mqa_logits:", hasattr(deep_gemm, "fp8_fp4_mqa_logits"))
except ImportError as exc:
    print("deep_gemm not importable:", exc)
