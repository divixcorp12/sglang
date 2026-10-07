#!/usr/bin/env python3
"""Resolve a specified ELF64 module definition for magic-trace v1.2.4 addr:."""
import argparse
import json
from pathlib import Path
import struct
import subprocess

from verify_trace import sha256


def target_identity(pid, tid):
    proc = Path("/proc") / str(pid)
    fields = {}
    for line in (proc / "task" / str(tid) / "status").read_text().splitlines():
        key, _, value = line.partition(":")
        fields[key] = value.strip()
    if int(fields["Tgid"]) != pid:
        raise ValueError("TID does not belong to the specified process")
    executable = str((proc / "exe").resolve())
    return dict(pid=pid, tid=tid, executable=executable,
                executable_sha256=sha256(executable),
                command_line=(proc / "cmdline").read_bytes().decode(errors="replace").rstrip("\0").split("\0"),
                start_ticks=(proc / "stat").read_text().rsplit(")", 1)[1].split()[19],
                thread_start_ticks=(proc / "task" / str(tid) / "stat").read_text().rsplit(")", 1)[1].split()[19],
                allowed_cpus=fields.get("Cpus_allowed_list"), comm=fields["Name"])


def elf_base(path):
    with Path(path).open("rb") as binary:
        header = binary.read(64)
        if len(header) != 64 or header[:6] != b"\x7fELF\x02\x01":
            raise ValueError("resolver supports little-endian ELF64 only")
        kind = struct.unpack_from("<H", header, 16)[0]
        offset = struct.unpack_from("<Q", header, 32)[0]
        length, count = struct.unpack_from("<HH", header, 54)
        if kind not in (2, 3) or length < 56 or count > 4096:
            raise ValueError("unsupported ELF header")
        binary.seek(offset)
        loads = []
        for _ in range(count):
            entry = binary.read(length)
            if len(entry) != length:
                raise ValueError("truncated ELF program headers")
            if struct.unpack_from("<I", entry)[0] == 1:
                file_offset, virtual = struct.unpack_from("<QQ", entry, 8)
                if file_offset == 0:
                    loads.append(virtual)
        if not loads:
            raise ValueError("no offset-zero PT_LOAD; inspect this ELF manually")
        return kind, min(loads)


def resolve(pid, tid, module, symbol):
    identity = target_identity(pid, tid)
    mappings = []
    for line in (Path("/proc") / str(pid) / "maps").read_text().splitlines():
        parts = line.split(maxsplit=5)
        if len(parts) == 6 and parts[5].startswith("/"):
            mappings.append((int(parts[0].split("-")[0], 16), int(parts[2], 16), parts[5]))
    def bias(path):
        _, base = elf_base(path)
        starts = {start for start, offset, name in mappings if offset == 0 and name == path}
        if len(starts) != 1:
            raise ValueError(f"expected one offset-zero mapping: {path}")
        return next(iter(starts)) - base
    exe = identity["executable"]
    if module is None:
        chosen = exe
    else:
        # Exact path or exact basename; multiple matching paths are refused.
        candidates = {name for _, _, name in mappings
                      if name == module or ("/" not in module and Path(name).name == module)}
        if len(candidates) != 1:
            raise ValueError(f"expected one exact module, found {sorted(candidates)}")
        chosen = next(iter(candidates))
    if chosen.endswith(" (deleted)"):
        raise ValueError("mapped binary was replaced/deleted; preserve the actual ELF")
    command = ["nm", "--defined-only", *( ["-D"] if chosen != exe else []), chosen]
    lines = subprocess.check_output(command, text=True).splitlines()
    values = {int(parts[0], 16) for line in lines if len(parts := line.split()) == 3 and parts[2] == symbol}
    if len(values) != 1:
        raise ValueError(f"expected one defined symbol {symbol!r} in {chosen}")
    value = next(iter(values))
    module_bias, exe_bias = bias(chosen), bias(exe)
    address = module_bias + value
    selection = address - exe_bias
    if selection < 0:
        raise ValueError("negative executable-relative address; inspect tool support manually")
    return dict(**identity, module=chosen, module_sha256=sha256(chosen),
                elf_symbol_value=value, module_bias=module_bias, executable_bias=exe_bias,
                runtime_trigger=address, selection="addr:" + hex(selection), symbol=symbol)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--tid", type=int)
    parser.add_argument("--module")
    parser.add_argument("--symbol", default="magic_trace_stop_indicator")
    args = parser.parse_args()
    try:
        print(json.dumps(resolve(args.pid, args.tid or args.pid, args.module, args.symbol), indent=2))
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        raise SystemExit(str(error))


if __name__ == "__main__":
    main()
