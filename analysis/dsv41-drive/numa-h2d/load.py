"""Host memory load for numa_h2d.py: copy a 256 MiB buffer to another until SIGTERM or the time limit.

Prints {"gbs": ...}, counting read plus write bytes, so a copy of N bytes counts 2N.
"""

import json
import signal
import sys
import time

import numpy as np

stop = False


def _stop(*_):
    global stop
    stop = True


signal.signal(signal.SIGTERM, _stop)
limit = float(sys.argv[1])
a = np.ones(256 << 20, dtype=np.uint8)
b = np.zeros_like(a)
moved = 0
t0 = time.monotonic()
while not stop and time.monotonic() - t0 < limit:
    np.copyto(b, a)
    moved += 2 * a.nbytes
print(json.dumps({"gbs": moved / (time.monotonic() - t0) / 1e9}), flush=True)
