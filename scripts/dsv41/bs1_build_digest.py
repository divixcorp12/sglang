"""The one-token (BS1) build's machine code, digested (plan 2026-10-06-dsv41-dspark-both-cpu-experts Task 2).

Widening the wire past 32 lanes edits source the BS1 build compiles too. Its instantiations must compile to the same
machine code, which is what keeps BS1 outputs and timing unchanged. This runs the BS1 suites into a fresh JIT cache (so
every module they load is built here), then records:
  - for every kernel in the BS1 device modules and the shared DIRECT and route-table modules: the sha256 of its SASS
    (cuobjdump -sass), instruction words and encodings, addresses stripped, keyed by its demangled name with the wide
    template arguments this plan adds normalised away;
  - for every 8-lane host module: the sha256 of its .text section.
Run on divix01 under cc-gpu.lock from the worktree root:
    python scripts/dsv41/bs1_build_digest.py --write OUT.json      (record)
    python scripts/dsv41/bs1_build_digest.py --compare GOLDEN.json (exit 1 on any difference; keys only in the new
                                                                    build, the wide kernels, are ignored)
    ... --compare GOLDEN.json --permanent   the kernels no later task of the plan changes: every device kernel but the
                                            post (Tasks 4-5 change it on purpose, inert at one token), no host .text
                                            (Tasks 6 and 9 change the host on purpose; test_expert_stream_hotpath_golden
                                            pins its behaviour)
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SUITES = [
    "test/manual/dsv41/test_exl3_lease_kernels_cuda.py",
    "test/manual/dsv41/test_exl3_cpu_lane_order_cuda.py",
    "test/manual/dsv41/test_exl3_ram_miss_graph_gpu.py",
    "test/manual/dsv41/test_exl3_moe_split_parity_cuda.py",
    "test/registered/unit/kernels/test_expert_stream_prod_build_symbols.py",
]
DEVICE = re.compile(r"^(expert_stream_exl3_l8(_n2)?|expert_residency_direct_.*|exl3_moe_route_tables.*)$")
HOST = re.compile(r"^expert_stream_host_exl3_(prod|instr)_l8(_n2)?$")
# The wide template arguments this plan adds to shared kernels; their narrow instantiation is the old kernel.
RENAMES = [
    (re.compile(r"direct_commit_gather_kernel<unsigned int, 32>"), "direct_commit_gather_kernel"),
    (re.compile(r"(exl3_moe_route_tables_kernel<[^<>]*?), unsigned int>"), r"\1>"),
]


def _demangle(name: str) -> str:
    out = subprocess.run(["c++filt", name], capture_output=True, text=True, check=True).stdout.strip()
    for pattern, repl in RENAMES:
        out = pattern.sub(repl, out)
    return out


def _sass(so: str) -> dict[str, str]:
    text = subprocess.run(["cuobjdump", "-sass", so], capture_output=True, text=True, check=True).stdout
    digests, name, lines = {}, None, []

    def close():
        if name is not None:
            digests[_demangle(name)] = hashlib.sha256("\n".join(lines).encode()).hexdigest()

    for line in text.splitlines():
        match = re.match(r"\s*Function : (\S+)", line)
        if match:
            close()
            name, lines = match.group(1), []
        elif name is not None and "/*" in line:
            lines.append(re.sub(r"/\*[0-9a-f]{4,}\*/", "", line).strip())  # drop the address column
    close()
    return digests


def _text(so: str) -> str:
    with tempfile.NamedTemporaryFile(suffix=".bin") as out:
        subprocess.run(["objcopy", "-O", "binary", "--only-section=.text", so, out.name], check=True)
        return hashlib.sha256(open(out.name, "rb").read()).hexdigest()


def collect() -> dict:
    cache = os.path.join(REPO, ".bs1-digest-cache")
    shutil.rmtree(cache, ignore_errors=True)
    env = os.environ | {"SGLANG_JIT_CACHE_DIR": cache, "PYTHONPATH": os.path.join(REPO, "python")}
    rc = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:randomly", *SUITES], env=env, cwd=REPO).returncode
    if rc != 0:
        raise SystemExit(f"the BS1 suites failed (exit {rc}); a digest of a red build proves nothing")
    result = {"nvcc": subprocess.run(["nvcc", "--version"], capture_output=True, text=True).stdout.splitlines()[-1],
              "device": {}, "host": {}}
    import torch

    result["arch"] = "sm_%d%d" % torch.cuda.get_device_capability()
    for root, _, files in os.walk(cache):
        for f in files:
            if not f.endswith(".so"):
                continue
            module, path = f[:-3], os.path.join(root, f)
            if DEVICE.match(module):
                for kernel, digest in _sass(path).items():
                    result["device"][f"{module}::{kernel}"] = digest
            elif HOST.match(module):
                result["host"][module] = _text(path)
    shutil.rmtree(cache, ignore_errors=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--write")
    group.add_argument("--compare")
    parser.add_argument("--permanent", action="store_true")
    args = parser.parse_args()
    now = collect()
    if args.write:
        with open(args.write, "w") as f:
            json.dump(now, f, indent=1, sort_keys=True)
        return
    with open(args.compare) as f:
        golden = json.load(f)
    if (golden["arch"], golden["nvcc"]) != (now["arch"], now["nvcc"]):
        print(f"toolchain differs: golden {golden['arch']} {golden['nvcc']}, now {now['arch']} {now['nvcc']}")
        raise SystemExit(2)
    kinds = ("device",) if args.permanent else ("device", "host")
    diffs = [f"{kind} {key}" for kind in kinds for key, digest in golden[kind].items()
             if now[kind].get(key) != digest and not (args.permanent and "exl3_ram_miss_post_kernel" in key)]
    print("\n".join(diffs) or "BS1 build unchanged")
    raise SystemExit(1 if diffs else 0)


if __name__ == "__main__":
    main()
