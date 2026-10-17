"""Replay the confidence cut over an STS collection's verifies: tokens per verify, verify ms and ms per token for the
cut under each STS file given, against verifying every row and against the oracle width.

A collection verifies every row, so each shard row holds the head's pre-STS logits and how many drafts the verify's
own accept took (``sampled_prefix_mask``: the rejection-sampling outcome for a sampled request). A cut verify of w
rows runs the same accept with the same coins and caps it (dspark_confidence_cut.py's docstring), so it would have
emitted min(m, w - 1) + 1 tokens: the replay is exact for tokens, given the drafts. Its time is the T(w) fit given
(--cost a,b: T(w) = a + b (w - 1) ms, a served "T(w) ms" line's fit) plus --draft-ms per verify. E(w*) next to the
tokens shows each file's calibration under the cut.

Usage: sts_cut_replay.py --data-glob 'OUT/sts/confidence.*.pt' --cost 96.5,18.3 [--draft-ms 0] [--temperature 0.6]
                         [--label sampled|argmax] NAME=STS.json [NAME=STS.json ...]
"""

from __future__ import annotations

import argparse
import glob
import json
import os

# CPU only: run as a script, hide the GPU before torch (and sglang's imports) can open a CUDA context outside
# cc-gpu.lock. (Imported by a test, the caller's devices are left alone.)
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch  # noqa: E402

from sglang.srt.speculative.dspark_components.dspark_confidence_cut import stop_rule_verify_lens
from sglang.srt.speculative.dspark_components.dspark_sts import load_sts_calibration_from_path


def load(data_glob: str, label: str, temperature: float | None) -> tuple[torch.Tensor, torch.Tensor]:
    """[n, gamma] pre-STS logits and [n] drafts the accept took."""
    logits, taken = [], []
    for path in sorted(glob.glob(data_glob)):
        shard = torch.load(path, map_location="cpu")
        mask = shard["sampled_prefix_mask" if label == "sampled" else "prefix_mask"]
        keep = torch.ones(mask.shape[0], dtype=torch.bool)
        if temperature is not None:
            keep = (shard["temperatures"] - temperature).abs() < 1e-3
        logits.append(shard["logits"][keep])
        taken.append(mask[keep].sum(dim=1).to(torch.int64))
    if not logits:
        raise SystemExit(f"no shard matches {data_glob!r}")
    return torch.cat(logits), torch.cat(taken)


def summarize(widths: torch.Tensor, taken: torch.Tensor, cost: torch.Tensor, draft_ms: float,
              expected: torch.Tensor | None = None) -> dict:
    tokens = torch.minimum(taken, widths - 1) + 1
    ms = cost[widths - 1] + draft_ms
    out = {
        "tokens_per_verify": round(float(tokens.float().mean()), 4),
        "ms_per_verify": round(float(ms.mean()), 3),
        "ms_per_token": round(float(ms.sum() / tokens.sum()), 3),
        "mean_width": round(float(widths.float().mean()), 3),
        "widths": torch.bincount(widths, minlength=cost.numel() + 1)[1:].tolist(),
    }
    if expected is not None:
        out["expected_tokens"] = round(float(expected.mean()), 4)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-glob", required=True)
    p.add_argument("--cost", required=True, help="a,b: T(w) = a + b (w - 1) ms")
    p.add_argument("--draft-ms", type=float, default=0.0)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--label", choices=("sampled", "argmax"), default="sampled")
    p.add_argument("--json", default=None)
    p.add_argument("sts", nargs="+", metavar="NAME=STS.json")
    a = p.parse_args(argv)
    logits, taken = load(a.data_glob, a.label, a.temperature)
    n, gamma = logits.shape
    base, per_row = (float(x) for x in a.cost.split(","))
    cost = torch.tensor([base + per_row * w for w in range(gamma + 1)], dtype=torch.float64)
    report = {
        "verifies": n,
        "cost": [base, per_row],
        "draft_ms": a.draft_ms,
        "every_row": summarize(torch.full((n,), gamma + 1), taken, cost, a.draft_ms),
        "oracle": summarize(taken + 1, taken, cost, a.draft_ms),
    }
    for spec in a.sts:
        name, path = spec.split("=", 1)
        temperatures = torch.tensor(load_sts_calibration_from_path(path).temperatures)
        confidence = torch.sigmoid(logits / temperatures)
        widths = stop_rule_verify_lens(confidence=confidence, cost_ms=cost + a.draft_ms)
        survival = torch.cumprod(confidence.double(), dim=1)
        expected = 1 + (survival * (torch.arange(gamma)[None, :] < (widths - 1)[:, None])).sum(dim=1)
        report[name] = summarize(widths, taken, cost, a.draft_ms, expected) | {"temperatures": temperatures.tolist()}
    text = json.dumps(report, indent=1)
    print(text)
    if a.json:
        with open(a.json, "w") as f:
            f.write(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
