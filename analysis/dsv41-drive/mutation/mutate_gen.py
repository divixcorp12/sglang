"""usage: mutate_gen.py <base> <tree> <results_file> <label_prefix> ID [ID ...]
Applies each mutant to <tree> (a copy of <base>), runs run_gen.sh, records the result, restores the file."""
import json, os, re, shutil, subprocess, sys, time
D = "/data/models/slang/nvfp4-work/t2-mutants"
base, tree, results, prefix, *ids = sys.argv[1:]
src = open(f"{D}/mutate.py").read().split("only = set")[0]
exec(src)  # defines MUTANTS (base-1 texts, including H19/H20)
H14_BASE3 = (
 "    } else {\n      counters_[kOverruns].fetch_add(1);  // status stays pending: a waiting layer fails stop\n    }\n    deferred_seq_ = 0;\n",
 "    } else {\n      counters_[kOverruns].fetch_add(1);  // status stays pending: a waiting layer fails stop\n      end_stage();\n      next_demand_ = skip_zero(next_demand_ + 1u);\n      return true;\n    }\n    deferred_seq_ = 0;\n")
table = {m[0]: list(m) for m in MUTANTS}
if base == "base3":
    table["H14"][2], table["H14"][3] = H14_BASE3
    table["H14"][6] = "pump_demand tail skipped for an unreadable record (base-3 rewrite)"
DRY = bool(os.environ.get("DRYRUN"))
RESUME = bool(os.environ.get("RESUME"))
bad = 0
already = set()
if RESUME and os.path.exists(f"{D}/{results}"):
    already = {json.loads(l)["id"] for l in open(f"{D}/{results}") if l.strip()}
for mid in ids:
    if os.path.exists(f"{D}/stop.flag") and not DRY:
        print("stop.flag present: stopping before", mid, "(finished mutants are in", results + ")"); sys.exit(0)
    if mid in already:
        print("RESUME: skipping", mid, "(already in", results + ")"); continue
    _, path, old, new, all_, tests, note = table[mid]
    orig = open(f"{D}/{base}/{path}").read()
    n = orig.count(old)
    if DRY:
        ok = n == 1 or (n > 1 and all_)
        print(base, mid, n, "OK" if ok else "BAD"); bad += 0 if ok else 1; continue
    rec = {"base": base, "id": mid, "file": path, "note": note, "occurrences": n}
    if n < 1 or (n > 1 and not all_):
        rec["status"] = "NOT_APPLIED"; open(f"{D}/{results}", "a").write(json.dumps(rec) + "\n"); continue
    open(f"{D}/{tree}/{path}", "w").write(orig.replace(old, new))
    env = dict(os.environ)
    if tests: env["FILES_OVERRIDE"] = tests
    t0 = time.time()
    load0 = open("/proc/loadavg").read().split()[:3]
    subprocess.run([f"{D}/run_gen.sh", tree, f"{prefix}{mid}"], env=env)
    log = open(f"{D}/log-{prefix}{mid}.txt").read()
    failed = sorted(set(re.findall(r"^FAILED (\S+)", log, re.M)))
    errors = sorted(set(re.findall(r"^ERROR (\S+)", log, re.M)))
    rec.update({"loadavg_at_start": load0, "loadavg_at_end": open("/proc/loadavg").read().split()[:3], "seconds": round(time.time() - t0), "tail": " ".join(re.findall(r"\d+ (?:passed|failed|skipped|errors?)", log[-500:])),
                "failed": failed, "errors": errors, "status": "KILLED" if (failed or errors) else "SURVIVED"})
    open(f"{D}/{results}", "a").write(json.dumps(rec) + "\n")
    shutil.copy(f"{D}/{base}/{path}", f"{D}/{tree}/{path}")
print("done", base, ids)
if DRY and bad: sys.exit(2)
