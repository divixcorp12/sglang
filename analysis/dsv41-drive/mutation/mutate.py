import json, os, re, shutil, subprocess, sys, time
D = "/data/models/slang/nvfp4-work/t2-mutants"
HOST = "python/sglang/kernels/jit/csrc/moe/exl3_ram_miss_host.cpp"
VER = "scripts/dsv41/verify_expert_mirror.py"
LAY = "python/sglang/srt/layers/moe/exl3_expert_layout.py"
URING = "python/sglang/kernels/jit/csrc/io/uring_file_reader.cpp"
K = "test/registered/unit/kernels/"
M = "test/registered/unit/layers/moe/"
HOST_TESTS = None  # the default set in run.sh
VERIFY_TESTS = M + "test_exl3_verify_expert_mirror.py"
MUTANTS = [
 # id, file, old, new, replace_all, tests, note
 ("H01", HOST, "if (ok || (cancelled && i < packed.size() && packed[i] != 0)) {",
  "if (ok || (i < packed.size() && packed[i] != 0)) {", False, None,
  "publish gate widened: rows packed before a hard failure are published (Task 6 V2's edit)"),
 ("H02", HOST, "if (ok || (cancelled && i < packed.size() && packed[i] != 0)) {",
  "if (ok || (false && i < packed.size() && packed[i] != 0)) {", False, None,
  "control: a cancelled advisory publishes none of its completed rows"),
 ("H03", HOST, "          } else {\n            release_locked(request.row, slots[i]);\n          }\n        }\n        (advisory ? tier.rows_advisory",
  "          } else {\n          }\n        }\n        (advisory ? tier.rows_advisory", False, None,
  "unpublished slots are not released: LOADING slots leak"),
 ("H04", HOST, "next_demand_ = skip_zero(head - kDemandRecords + 2u);", "next_demand_ = head - kDemandRecords + 2u;", False, None,
  "demand lap resume: no zero skip"),
 ("H05", HOST, "next_demand_ = skip_zero(next_demand_ + 1u);", "next_demand_ = next_demand_ + 1u;", False, None,
  "demand advance: no zero skip"),
 ("H06", HOST, "next_advice_ = skip_zero(head - kAdviseRecords + 2u);", "next_advice_ = head - kAdviseRecords + 2u;", False, None,
  "advisory lap resume: no zero skip"),
 ("H07", HOST, "next_advice_ = skip_zero(next_advice_ + 1u);", "next_advice_ = next_advice_ + 1u;", False, None,
  "advisory advance: no zero skip"),
 ("H08", HOST, "next_demand_ = skip_zero(head - kDemandRecords + 2u);", "next_demand_ = skip_zero(head - kDemandRecords + 1u);", False, None,
  "demand lap resumes one record early (head-15, may be mid-rewrite)"),
 ("H09", HOST, "next_advice_ = skip_zero(head - kAdviseRecords + 2u);", "next_advice_ = skip_zero(head - kAdviseRecords + 1u);", False, None,
  "advisory lap resumes one record early"),
 ("H10", HOST, "counters_[kOverruns].fetch_add(head - next_demand_ - (kDemandRecords - 2));", "counters_[kOverruns].fetch_add(1);", False, None,
  "lap overrun count is 1, not the skipped count"),
 ("H11", HOST, "      if (tier.hot[expert]) continue;\n", "", False, None,
  "take_slot_locked evicts hot rows"),
 ("H12", HOST, "    publish_map(row, victim, -1);  // unmapped before its bytes are overwritten (D11)\n", "", False, None,
  "eviction does not unmap the victim before its slot is reused"),
 ("H13", HOST, "    tier.expert_slot[victim] = -1;\n    tier.slot_to_expert[best] = -1;", "    tier.slot_to_expert[best] = -1;", False, None,
  "eviction leaves the victim's expert_slot pointing at the reused slot"),
 ("H14", HOST, "      counters_[kOverruns].fetch_add(1);  // status stays pending: a waiting layer fails stop\n    }\n    _mm_sfence();\n    store_release(page_ + kDemandDone, next_demand_);\n",
  "      counters_[kOverruns].fetch_add(1);  // status stays pending: a waiting layer fails stop\n      end_stage();\n      next_demand_ = skip_zero(next_demand_ + 1u);\n      return true;\n    }\n    _mm_sfence();\n    store_release(page_ + kDemandDone, next_demand_);\n", False, None,
  "pump_demand tail is skipped for an unreadable record (Done never advances for it)"),
 ("H15", HOST, "          if (c.trace->extent_submit[d.trace_slot] == 0) {\n            if (prepared == 0) prepared = stamp(c.trace);\n            c.trace->extent_submit[d.trace_slot] = prepared;\n          } else {\n            ++c.trace->extent_attempts[d.trace_slot];\n          }\n",
  "          if (c.trace->extent_submit[d.trace_slot] != 0) ++c.trace->extent_attempts[d.trace_slot];\n          if (prepared == 0) prepared = stamp(c.trace);\n          c.trace->extent_submit[d.trace_slot] = prepared;\n", False, None,
  "a resubmitted extent overwrites its first submit stamp"),
 ("H16", HOST, "file_drive_.push_back(static_cast<uint8_t>(std::min<size_t>(drive, kMaxDrives - 1)));", "file_drive_.push_back(0);", False, None,
  "every file is attributed to drive 0"),
 ("H17", HOST, "c.trace->drive_bytes[drive] += d.done;", "c.trace->drive_bytes[0] += d.done;", True, None,
  "per-drive bytes all land on drive 0 (both sites)"),
 ("H18", HOST, "      thread_.join();\n      tier_->set_threaded(false);\n      throw std::runtime_error(", "      thread_.join();\n      throw std::runtime_error(", False, None,
  "a failed thread start leaves the tier marked threaded"),
 ("P01", VER, "direct=direct,\n                    roots=[root],", "direct=False,\n                    roots=[root],", False, VERIFY_TESTS,
  "verifier reads mirror rows buffered, reports direct"),
 ("P02", VER, "layout, layer, comparer.segments, direct=direct\n        )", "layout, layer, comparer.segments, direct=False\n        )", False, VERIFY_TESTS,
  "verifier reads source rows buffered, reports direct"),
 ("P03", LAY, 'rf"^{re.escape(prefix)}\\.(\\d+)\\.ffn\\.experts\\.(\\d+)\\."', 'rf"^layers\\.(\\d+)\\.ffn\\.experts\\.(\\d+)\\."', False, M + "test_exl3_expert_layout.py",
  "layout ignores the prefix (draft path would read the target's experts)"),
 ("U01", URING, "(direct ? O_DIRECT : 0)", "0", False, K + "test_uring_file_reader.py " + VERIFY_TESTS + " " + M + "test_exl3_mirror_row_source.py " + M + "test_exl3_row_reader.py",
  "the reader never opens O_DIRECT, whatever `direct` says"),

 ("H19", HOST, "if (ok || (cancelled && i < packed.size() && packed[i] != 0)) {",
  "if (true) {", False, None,
  "gate always true"),
 ("H20", HOST, "if (ok || (cancelled && i < packed.size() && packed[i] != 0)) {",
  "if (ok || cancelled) {", False, None,
  "gate ignores packed[i]"),
]
only = set(sys.argv[1:])
results_path = f"{D}/results.jsonl"
def restore(path):
    shutil.copy(f"{D}/base/{path}", f"{D}/tree/{path}")
for mid, path, old, new, all_, tests, note in MUTANTS:
    if only and mid not in only: continue
    src = open(f"{D}/base/{path}").read()
    n = src.count(old)
    rec = {"id": mid, "file": path, "note": note, "occurrences": n}
    if n < 1 or (n > 1 and not all_):
        rec["status"] = "NOT_APPLIED"; open(results_path, "a").write(json.dumps(rec) + "\n"); continue
    open(f"{D}/tree/{path}", "w").write(src.replace(old, new))
    env = dict(os.environ)
    if tests: env["FILES_OVERRIDE"] = tests
    t0 = time.time()
    subprocess.run([f"{D}/run.sh", mid], env=env)
    log = open(f"{D}/log-{mid}.txt").read()
    m = re.search(r"=+ (.*?) in [\d.]+s", log) or re.search(r"(\d+ (?:passed|failed|error).*?) in [\d.]+s", log)
    summary = re.findall(r"(\d+) (passed|failed|skipped|error|errors)", log.split("\n")[-4] if False else " ".join(re.findall(r"\d+ (?:passed|failed|skipped|errors?)", log[-400:])))
    failed = sorted(set(re.findall(r"^FAILED (\S+)", log, re.M)))
    errors = sorted(set(re.findall(r"^ERROR (\S+)", log, re.M)))
    rec.update({"seconds": round(time.time() - t0), "tail": " ".join(re.findall(r"\d+ (?:passed|failed|skipped|errors?)", log[-500:])),
                "failed": failed, "errors": errors, "status": "KILLED" if (failed or errors) else "SURVIVED"})
    open(results_path, "a").write(json.dumps(rec) + "\n")
    restore(path)
print("done")
