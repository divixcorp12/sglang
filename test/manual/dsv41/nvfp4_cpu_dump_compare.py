"""Compare two nvfp4_cpu_forward_ab dumps at a tolerance: python nvfp4_cpu_dump_compare.py WANT GOT [RTOL].

The kernel is built -Ofast (FMA contraction, reassociation), so a revision's dump differs from another's in the last
bits. Statuses and case names must match exactly; each case's values must satisfy |got - want| <= RTOL * max|want|
over that case's output row (plus a floor of 1e-30 for an all-zero row). Default RTOL 1e-2: an fp32 rounding change
anywhere before a Q8_0 requantization (input, or the gated-SiLU intermediate) can flip one value to the next
quantum, which is 1/127 = 7.9e-3 of its block's amax, and that flip is the largest single change the forward can
see. Measured against the pre-change bit-exact build (2026-10-23, divix01): worst 3.03e-3 at x86-64-v3 and x86-64.
"""

import struct
import sys

import numpy as np


def read(path):
    cases = {}
    with open(path, "rb") as f:
        data = f.read()
    pos = 0
    while pos < len(data):
        (length,) = struct.unpack_from("<I", data, pos)
        pos += 4
        name = data[pos : pos + length].decode()
        pos += length
        status, count = struct.unpack_from("<iI", data, pos)
        pos += 8
        values = np.frombuffer(data, dtype="<f4", count=count, offset=pos)
        pos += 4 * count
        cases[name] = (status, values)
    return cases


def main():
    want, got = read(sys.argv[1]), read(sys.argv[2])
    rtol = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-2
    if want.keys() != got.keys():
        sys.exit("the dumps hold different cases")
    worst, worst_case, bad = 0.0, "", []
    for name, (status, a) in want.items():
        status_got, b = got[name]
        if status != status_got or a.shape != b.shape:
            bad.append(f"{name}: status {status} vs {status_got}")
            continue
        scale = max(float(np.max(np.abs(a))) if a.size else 0.0, 1e-30)
        error = float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))) / scale if a.size else 0.0
        if error > worst:
            worst, worst_case = error, name
        if error > rtol:
            bad.append(f"{name}: max relative difference {error:.3e}")
    print(f"{len(want)} cases, worst relative difference {worst:.3e} ({worst_case}), tolerance {rtol:g}")
    for line in bad:
        print("MISMATCH", line)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
