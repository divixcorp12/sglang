import argparse
import json
from pathlib import Path

from .protocol import (
    inventory,
    paired_report,
    preflight,
    screen_budget,
    validate_groups,
)


def main():
    p = argparse.ArgumentParser(
        description="NCM setup: metadata checks/analysis only; never launches experiments"
    )
    sub = p.add_subparsers(dest="command", required=True)
    i = sub.add_parser("inventory")
    i.add_argument("checkpoint")
    r = sub.add_parser("preflight")
    r.add_argument("--config", default=str(Path(__file__).with_name("pilot.json")))
    r.add_argument("--corpus")
    r.add_argument("--bundle")
    s = sub.add_parser("screen-budget")
    s.add_argument("--sd", type=float, required=True)
    s.add_argument("--gain", type=float, required=True)
    a = sub.add_parser("report")
    a.add_argument("prompt_means")
    a.add_argument("--resamples", type=int, default=5000)
    c = sub.add_parser("check-corpus")
    c.add_argument("corpus")
    args = p.parse_args()
    if args.command == "inventory":
        result = inventory(args.checkpoint)
    elif args.command == "preflight":
        result = preflight(
            json.loads(Path(args.config).read_text()), args.corpus, args.bundle
        )
    elif args.command == "screen-budget":
        result = screen_budget(args.sd, args.gain)
    elif args.command == "report":
        result = paired_report(
            [
                json.loads(x)
                for x in Path(args.prompt_means).read_text().splitlines()
                if x.strip()
            ],
            draws=args.resamples,
        )
    else:
        result = validate_groups(
            [
                json.loads(x)
                for x in Path(args.corpus).read_text().splitlines()
                if x.strip()
            ]
        )
    print(json.dumps(result, indent=2, allow_nan=False))
    if args.command == "preflight" and not result["ready"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
