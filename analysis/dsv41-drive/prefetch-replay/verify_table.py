"""One markdown row per arm of verify_arms.sh's results.jsonl, saved ms against the ``none`` arm."""

import json
import sys

KEYS = ["precision_target", "precision_any_use", "spec_rows_per_step", "ram_misses_per_step", "late_per_step",
        "harmful_evictions_per_step", "demand_rows_delayed_per_step", "demand_delay_mean_ms", "gate_ms_per_step",
        "step_ms_per_step", "ms_per_token", "nvme_busy_frac"]


def main() -> None:
    rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
    base = {r["args"]["nvme_row_ms"]: r for r in rows if r["name"].startswith("none")}
    print("| arm | " + " | ".join(KEYS) + " | saved ms/token |")
    print("|---|" + "---:|" * (len(KEYS) + 1))
    for r in rows:
        b = base.get(r["args"]["nvme_row_ms"])
        saved = b["ms_per_token"] - r["ms_per_token"] if b else float("nan")
        cells = [f"{r[k]:.3f}" if isinstance(r[k], float) else str(r[k]) for k in KEYS]
        print(f"| {r['name']} | " + " | ".join(cells) + f" | {saved:+.2f} |")


if __name__ == "__main__":
    main()
