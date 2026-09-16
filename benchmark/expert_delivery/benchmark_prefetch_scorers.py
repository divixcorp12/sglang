#!/usr/bin/env python3
"""Time captured representative scorer outputs plus candidate selection per target.

The input is a ``torch.save`` mapping of ``target-layer`` to an already
captured score tensor ``[rows, experts]``; this keeps timing separate from
capture and reports one CUDA-event median for every target layer.
"""

import argparse
import json
import statistics

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--captured-scores", required=True)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    from sglang.srt.layers.moe.expert_prediction.serving.candidates import PrefetchCandidateBank
    scores_by_layer = torch.load(args.captured_scores, map_location="cuda", weights_only=True)
    bank = PrefetchCandidateBank(layer_ids=[int(layer) for layer in scores_by_layer], width=args.width, device=torch.device("cuda"))
    for layer, scores in scores_by_layer.items():
        layer = int(layer)
        residency = torch.full((scores.shape[1],), -1, dtype=torch.int32, device="cuda")
        for _ in range(10): bank.write(layer, scores, expert_to_slot=residency)
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(args.iterations)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(args.iterations)]
        for start, end in zip(starts, ends):
            start.record(); bank.write(layer, scores, expert_to_slot=residency); end.record()
        torch.cuda.synchronize()
        print(json.dumps({"target_layer": layer, "scorer_plus_selection_ms": statistics.median(s.elapsed_time(e) for s, e in zip(starts, ends)), "width": args.width}, sort_keys=True))


if __name__ == "__main__":
    main()
