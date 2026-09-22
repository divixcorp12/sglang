"""Offline (reads logs and sources only). For each killed mutant, print the killing tests and the assertion line each failed on;
flag a kill whose failing assertion line reads a counter (counters(), ['...'] of a counters dict) and nothing else."""
import json, re, sys
D = "/data/models/slang/nvfp4-work/t2-mutants"
results, tree, prefix = sys.argv[1:4]
for line in open(f"{D}/{results}"):
    r = json.loads(line)
    if r["status"] != "KILLED":
        print(r["id"], r["status"], r.get("tail")); continue
    log = open(f"{D}/log-{prefix}{r['id']}.txt").read()
    locs = re.findall(r"^(\S+\.py):(\d+): (.*)$", log, re.M)
    kills = []
    for path, no, msg in locs:
        rel = path.split(f"{tree}/")[-1]
        try: src = open(f"{D}/{tree}/{rel}").read().split("\n")[int(no) - 1].strip()
        except Exception: src = "?"
        counter_only = bool(re.search(r"counters|\['(?:overruns|deferred|rows_read|version|advis)", src)) and "==" in src
        kills.append((rel.split('/')[-1] + ":" + no, src[:110], "COUNTER-ONLY?" if counter_only else ""))
    print(r["id"], "KILLED", r.get("tail"))
    for k in kills[:12]: print("   ", *k)

# Invocation check: every run's collected total must equal its baseline's, or the run did not exercise what it claims.
import glob
def total(log):
    nums = dict((k, int(v)) for v, k in re.findall(r"(\d+) (passed|failed|skipped|errors?)", log[-500:]))
    return sum(nums.get(k, 0) for k in ("passed", "failed", "skipped", "error", "errors"))
print("\nINVOCATION CHECK (collected total per run vs its baseline):")
for f in sorted(glob.glob(f"{D}/log-*.txt")):
    if not f.split("/")[-1].startswith(f"log-{prefix}"): continue
    t = total(open(f).read())
    print("  ", f.split("/")[-1], t)
