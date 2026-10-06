"""Make a DSpark draft resident-set file from a routes probe: each stage's N most-routed experts stay on the GPU.

  python analysis/dsv41-drive/dspark/draft_resident_set.py ROUTES.jsonl N OUT.json
ROUTES.jsonl is SGLANG_DSPARK_DEBUG_DRAFT_ROUTES_PATH's output (ab_cpu_draft.py's "routes" arm). The server reads
OUT.json through SGLANG_DSV41_DSPARK_DRAFT_RESIDENT_PATH.
"""

import sys

from sglang.srt.layers.moe.cpu_experts.draft_resident import top_n_resident_set, write_resident_set


def main():
    routes, n, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    with open(routes) as f:
        stages = top_n_resident_set(f, n)
    write_resident_set(out, stages, n=n, source=routes)
    for layer, ids in stages.items():
        print(f"stage {layer}: {len(ids)} resident: {ids}")


if __name__ == "__main__":
    main()
