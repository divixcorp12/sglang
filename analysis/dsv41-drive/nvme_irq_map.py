#!/usr/bin/env python3
# Which CPU takes the completion interrupt of each NVMe submitting core? Read-only sysfs/procfs.
#   python3 nvme_irq_map.py            # prints, per drive, the submitter cores in 0-63 whose IRQ is on 64-71
import os
def cpus(s):
    out = []
    for p in s.strip().split(","):
        if "-" in p:
            a, b = p.split("-"); out += range(int(a), int(b) + 1)
        elif p:
            out.append(int(p))
    return out
irqs = {}
for line in open("/proc/interrupts"):
    f = line.split()
    if f and f[-1].startswith("nvme"):
        irqs[f[-1]] = f[0].rstrip(":")
for dev in ("nvme0", "nvme2", "nvme3"):        # nvme3 is the drive mounted at /mnt/nvme4
    m = {}
    for q in os.listdir(f"/sys/block/{dev}n1/mq"):
        cl = cpus(open(f"/sys/block/{dev}n1/mq/{q}/cpu_list").read())
        irq = irqs[f"{dev}q{int(q) + 1}"]
        eff = cpus(open(f"/proc/irq/{irq}/effective_affinity_list").read())[0]
        for c in cl:
            m[c] = eff
    print(dev, "submitter cores in 0-63 whose completion IRQ lands on 64-71:", sorted(c for c in m if c < 64 and 64 <= m[c] <= 71))
    print(dev, "node-0 submitters' IRQ nodes:", sorted({("n0" if (m[c] < 18 or 36 <= m[c] < 54) else "n1") for c in m if c < 18 or 36 <= c < 54}))
