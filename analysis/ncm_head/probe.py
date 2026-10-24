"""Prepare, or explicitly execute, the first native EXL3 weight/gradient probe.

Default is metadata-only. --execute is a future GPU operation, NOT used by setup.
This deliberately does not load the target transformer or launch a serving process.
"""

import argparse
import json
from pathlib import Path

from .protocol import inventory, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(Path(__file__).with_name("pilot.json")))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    plan = {
        "operation": "EXL3 LM-head forward and surrogate-input-gradient probe",
        "execute": args.execute,
        "target": config["target_path"],
        "draft": config["draft_path"],
        "gpu_lock": "/data/models/slang/nvfp4-work/cc-gpu.lock",
        "inputs": "synthetic [5,5120] correctness fixture; not representative acceptance data",
        "starts_server": False,
        "trains_model": False,
    }
    if not args.execute:
        print(json.dumps(plan, indent=2))
        return
    if args.output is None or args.output.exists():
        parser.error("--execute requires a fresh --output directory")
    # Refuse the machine's older editable import even when model directories match.
    import importlib.util

    spec = importlib.util.find_spec("sglang")
    if spec is None or spec.origin is None:
        raise RuntimeError("pinned SGLang source is absent")
    package = Path(spec.origin).parent
    for relative, expected_hash in config["native_source_hashes"].items():
        path = package / relative
        if not path.is_file() or sha256(path) != expected_hash:
            raise RuntimeError(
                f"SGLang source mismatch: {relative}; use run-native-probe.sh"
            )
    # Lock before importing/loading CUDA extensions; fail fast, never wait behind an ancestor.
    import fcntl

    with open(plan["gpu_lock"], "a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(
                "GPU lock busy; no work started (do not wrap this probe in another flock)"
            )
        target, draft = (
            inventory(config["target_path"]),
            inventory(config["draft_path"]),
        )
        for name, meta in [("target", target), ("draft", draft)]:
            if meta["config_sha256"] != config["native_config_sha256"][name]:
                raise ValueError(
                    f"{name} config drift: update and review manifest before execution"
                )
        import torch
        from safetensors import safe_open
        from sglang.srt.layers.layernorm import RMSNorm
        from sglang.srt.layers.quantization.exl3.ops import (
            Exl3Tensors,
            exl3_dense_weight,
            exl3_linear,
        )

        from .native import NativeProjector

        def tensor(root, key):
            root = Path(root)
            index = json.loads((root / "model.safetensors.index.json").read_text())[
                "weight_map"
            ]
            with safe_open(str(root / index[key]), framework="pt", device="cpu") as f:
                return f.get_tensor(key).to("cuda")

        packed = Exl3Tensors(
            tensor(config["target_path"], "head.trellis"),
            tensor(config["target_path"], "head.suh"),
            tensor(config["target_path"], "head.svh"),
            True,
        )
        norm_weight = tensor(config["draft_path"], "mtp.2.norm.weight")
        dense = exl3_dense_weight(packed)
        projector = NativeProjector(
            packed, norm_weight, config["expected_native"]["norm_eps"], dense
        )
        native_norm = RMSNorm(
            norm_weight.numel(), eps=config["expected_native"]["norm_eps"]
        ).to(device="cuda", dtype=norm_weight.dtype)
        native_norm.weight.data.copy_(norm_weight)
        generator = torch.Generator(device="cuda").manual_seed(config["seed"])
        x = torch.randn(
            5,
            norm_weight.numel(),
            device="cuda",
            dtype=norm_weight.dtype,
            generator=generator,
            requires_grad=True,
        )
        actual = projector(x)
        expected = exl3_linear(native_norm(x.detach()), packed).float()
        gradient = torch.autograd.grad(actual.square().mean(), x)[0]
        result = {
            **plan,
            "target_metadata": target,
            "draft_metadata": draft,
            "source_config_sha256": sha256(args.config),
            "torch_version": torch.__version__,
            "max_abs_logit_error": float((actual - expected).abs().max()),
            "mean_abs_logit_error": float((actual - expected).abs().mean()),
            "gradient_finite": bool(torch.isfinite(gradient).all()),
            "gradient_nonzero": bool(gradient.abs().sum() > 0),
            "feature_alignment_validated": False,
            "pilot_ready": False,
            "note": "Weight-level evidence only. Requires real-row parity, feature alignment, corpus, captures and complete hashes before training.",
        }
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / "weight-probe.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
