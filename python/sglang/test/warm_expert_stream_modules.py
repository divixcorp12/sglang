"""Build the expert-stream JIT host modules a test run loads, cold ones side by side, before the run starts.

Run it once ahead of ``pytest -n 8`` (it is a no-op when everything is cached)::

    python -m sglang.test.warm_expert_stream_modules --variants instr,prod --lanes 8,16 --nodes 1,2

Each cold host variant costs 50-100 s to compile; loaded by the tests themselves they queue behind the per-module
build lock and inside child-process timeouts (see ``dsv41_ram_miss_fixtures.warm_host_modules``)."""

from __future__ import annotations

import argparse
import logging
import time

from sglang.test.dsv41_ram_miss_fixtures import warm_host_module_set


def _ints(text: str) -> list[int]:
    return [int(part) for part in text.split(",") if part]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--variants", default="instr,prod")
    parser.add_argument("--lanes", type=_ints, default=[8])
    parser.add_argument("--nodes", type=_ints, default=[1])
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    start = time.monotonic()
    warm_host_module_set(args.variants.split(","), args.lanes, args.nodes)
    print(f"warm: {time.monotonic() - start:.1f} s", flush=True)


if __name__ == "__main__":
    main()
