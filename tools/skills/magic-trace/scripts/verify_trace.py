#!/usr/bin/env python3
"""Bounded FXT/perf validation. Empty timelines or unsaved PT packets fail."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import struct


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_fxt(path, max_bytes=256 * 1024 * 1024):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    counts, names, total = Counter(), [], 0
    timestamps, frequency = [], 1_000_000_000
    with opener(path, "rb") as stream:
        while True:
            raw = stream.read(8)
            if not raw:
                break
            if len(raw) != 8:
                raise ValueError("truncated FXT header")
            header = struct.unpack("<Q", raw)[0]
            kind = header & 15
            size = ((header >> 4) & (0xFFFFFFFF if kind == 15 else 0xFFF)) * 8
            if size < 8 or total + size > max_bytes:
                raise ValueError("invalid FXT record size or decoded byte limit exceeded")
            # Standard records are at most 32 KiB. Large records are skipped in chunks.
            first = stream.read(min(size - 8, 32760))
            if len(first) != min(size - 8, 32760):
                raise ValueError("truncated FXT record")
            remaining = size - 8 - len(first)
            while remaining:
                chunk = stream.read(min(remaining, 65536))
                if not chunk:
                    raise ValueError("truncated large FXT record")
                remaining -= len(chunk)
            counts[kind] += 1
            total += size
            if kind == 1 and len(first) >= 8:
                frequency = struct.unpack_from("<Q", first)[0]
            if kind == 2 and len(names) < 4096:
                names.append(first[:(header >> 32) & 32767].decode(errors="replace"))
            if kind == 4:
                if len(first) < 8:
                    raise ValueError("FXT event has no timestamp")
                stamp = struct.unpack_from("<Q", first)[0]
                if not timestamps:
                    timestamps = [stamp, stamp]
                else:
                    timestamps = [min(timestamps[0], stamp), max(timestamps[1], stamp)]
    if not counts[4]:
        raise ValueError("empty FXT timeline: no type-4 event records")
    return dict(path=str(path.resolve()), sha256=sha256(path), file_bytes=path.stat().st_size,
                decoded_bytes=total, record_counts=dict(counts), timeline_events=counts[4],
                timestamp_bounds_ticks=timestamps, ticks_per_second=frequency, strings=names)


def inspect_perf(path, max_records=5_000_000):
    path = Path(path)
    counts, aux_bytes, records = Counter(), 0, 0
    with path.open("rb") as stream:
        header = stream.read(104)
        if len(header) < 104 or header[:8] != b"PERFILE2":
            raise ValueError("expected little-endian perf.data v2 file")
        offset, size = struct.unpack_from("<QQ", header, 40)
        end = offset + size
        if offset < 104 or end > path.stat().st_size:
            raise ValueError("invalid perf data section")
        stream.seek(offset)
        while stream.tell() < end:
            raw = stream.read(8)
            if len(raw) != 8:
                raise ValueError("truncated perf record header")
            kind, _, length = struct.unpack("<IHH", raw)
            if length < 8 or stream.tell() + length - 8 > end:
                raise ValueError("invalid perf record size")
            body = stream.read(length - 8)
            counts[kind] += 1
            records += 1
            if records > max_records:
                raise ValueError("perf record limit exceeded")
            if kind == 71:  # AUXTRACE payload follows the record header/body.
                if len(body) < 8:
                    raise ValueError("AUXTRACE record has no payload length")
                n = struct.unpack_from("<Q", body)[0]
                if stream.tell() + n > end:
                    raise ValueError("AUXTRACE payload outside perf data section")
                aux_bytes += n
                stream.seek(n, 1)
    if not aux_bytes:
        raise ValueError("no saved Intel PT AUXTRACE payload; AUX bookkeeping is not trace data")
    return dict(path=str(path.resolve()), sha256=sha256(path), file_bytes=path.stat().st_size,
                record_counts=dict(counts), auxtrace_payload_bytes=aux_bytes)


def verify(path, perf_data=None, log=None):
    result = dict(fxt=inspect_fxt(path))
    if perf_data is not None:
        result["perf"] = inspect_perf(perf_data)
    if log is not None:
        warnings = []
        with Path(log).open(errors="replace") as stream:
            for line in stream:
                if any(word in line.lower() for word in ("warning", "overflow", "trace errors")):
                    if len(warnings) < 100:
                        warnings.append(line.strip())
        result["decoder_warnings"] = warnings
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--perf-data", type=Path)
    parser.add_argument("--log", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = verify(args.trace, args.perf_data, args.log)
    except (ValueError, OSError, EOFError) as error:
        raise SystemExit(str(error))
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result["fxt"].items() if k != "strings"}, indent=2))


if __name__ == "__main__":
    main()
