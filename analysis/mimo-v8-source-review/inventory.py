"""Public technical tool candidate inventory. No expert scores, GPU or source execution."""
import os
os.environ["CUDA_VISIBLE_DEVICES"]=""
import sys,json,re,heapq,collections,time,importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/"calibration-mix-v3"));import common as C
P=C.read(OUT/"policy.json");PARTS=("primary","repeat");SEED=P["seed"]
def save(n,x):C.save(OUT/n,x)
def read(p):return C.read(p)
def log(event,**kw):
 d=dict(event=event,at=time.time(),**kw);save("progress.json",d)
 with(OUT/"steps.jsonl").open("a")as f:f.write(C.canonical(d)+"\n")
 print(C.canonical(d),flush=True)
def base():
 rows={p:[]for p in PARTS}
 for x in map(json.loads,(ROOT/"calibration-mix-v5/all-selected.jsonl").open()):rows[x["partition"]].append(x["record"])
 return rows
def owners():
 b=base();return C.owner_table(b)
def partition(g,own):return own.get(g.lower(),PARTS[int(C.sha([SEED,"partition",g.lower()])[:8],16)%2])
def clean_goal(text):
 text=C.issue_body(text)
 return re.sub(r"https?://\S+","",text).split("Current terminal state:",1)[0].strip()
def family(text,st):
 goal=clean_goal(text).lower()
 if st=="research":
  if re.search(r"\b(?:algorithm|numerical|scientific computing|compiler|distributed|database|concurren|quantiz|control|optimization|optimisation|simulation|signal processing|fourier|radar|statistic|linear algebra|software engineering|program synthesis|programming|monte carlo|time.series|portfolio|backtest|risk model|python|c\+\+|julia|dockerfile|benchmark|retrieval|neural|machine learning|tensor|finite.element)",goal):return "evidence_research"
  return None
 # Semantic screening aids only; admitted cases require source-linked review.
 if re.search(r"\b(?:airbnb|hotel|restaurant|dinner party|vacation|getaway|train tickets|touris|princely state|voivodeship|cookbook|molecular gastronomy|workshop registration|wellness|astrology|pet.friendly)\b",goal):return None
 if re.search(r"\b(?:backup|restore|recovery|recover|durab\w*|rollback|transactions?|fsync|write.ahead|crash.consisten|atomic (?:write|file)|database migration|incremental (?:backup|snapshot)|sqlite|postgresql|mysql|redis|mongodb)\b",goal):return "persistence_recovery"
 if re.search(r"\b(?:numerical|linalg|linear algebra|covariance|regression|least.squares|statistical|monte.carlo|fourier|fft|eigen\w*|differential equation|finite.element|scipy|ndarray|simulation|signal processing|interpolat\w*|quadrature|kalman|time.series|backtest|portfolio optimization|risk model)\b",goal):return "scientific_numeric"
 if re.search(r"\b(?:dependency|dependencies|install\w*|packag\w*|cmake|compiler|compilation|linker|toolchain|wheel|pip|conda|npm|cargo|build\w*|debug\w*|segfault|stack.trace|unit.test|pytest|test.suite|profil\w*)\b",goal):return "build_package_debug"
 if re.search(r"\b(?:http\w*|apis?|sdk|oauth|tls|ssl|authenticat\w*|authoriz\w*|permissions?|rbac|kubernetes|cloud|endpoint|requests?|client|server|socket|network|protocol|sparql|graphql|rdf|json.schema|websocket|asyncio)\b",goal):return "api_platform_security"
 return None
def inventory():
 assert not(OUT/"raw-candidates.jsonl").exists()
 own=owners();b=base();non=[r for p in PARTS for r in b[p]if r["domain"]!="general_tool_use"]
 counts=collections.Counter(r["split_group"].lower()for r in non if r["split_group"].startswith("repo:"))
 repo_tokens=collections.Counter()
 for r in non:
  if r["split_group"].startswith("repo:"):repo_tokens[r["split_group"].lower()]+=r["valid_tokens"]
 nonraw={r.get("source_record_sha256")for r in non};nonproblems={C.sha(C.PUBLIC.problem(clean_goal(C.first_user(r))))for r in non}
 heaps=collections.defaultdict(list);stats=collections.Counter();sources={};seq=0
 def add(c):
  nonlocal seq
  stats[c["stratum"]+":scanned_candidates"]+=1
  raw=c["record"];text=c.get("task_text")or C.first_user(raw);fam=family(text,c["stratum"])
  if not fam:stats["outside_positive_scope"]+=1;return
  g=c["split_group"].lower()
  if g.startswith("repo:")and(counts[g]>=P["public_repository_max_episodes"]or repo_tokens[g]>=P["public_repository_max_tokens"]):stats["existing_repository_at_cap"]+=1;return
  if c["source_record_sha256"]in nonraw or C.sha(C.PUBLIC.problem(clean_goal(text)))in nonproblems:stats["duplicates_preserved_non_tool"]+=1;return
  chars=len(C.canonical(raw))
  if not 1500<=chars<=400000:stats["source_character_bounds"]+=1;return
  c.update(partition=partition(g,own),workflow_family=fam,workflow_families=[fam],objective_text=clean_goal(text))
  priority=C.sha([SEED,c["source_record_sha256"]]);c["metadata_priority"]=priority
  k=(c["partition"],fam,c["stratum"]);h=heaps[k];seq+=1;item=(-int(priority,16),seq,c)
  if len(h)<P["reservoir_per_partition_family_source"]:heapq.heappush(h,item)
  elif item[0]>h[0][0]:heapq.heapreplace(h,item)
  stats["eligible:"+c["partition"]+":"+fam+":"+c["stratum"]]+=1
 # Immutable pinned SWE training shards already held in the workspace.
 metas=[]
 for name in ["nvidia-calibration-v4/source.lock.json","calibration-gap-v1/swe-extension-source.lock.json"]:
  for m in read(ROOT/name)["files"]:
   if "SFT-SWE"in m["dataset"]:
    sources[m["path"]]=m
    if m["upstream_file"].endswith(".parquet")and m["path"]not in {x["path"]for x in metas}:metas.append(m)
 for m in metas:
  assert C.file_sha(ROOT/m["path"])==m["sha256"]
  for i,r in C.PUBLIC.rows(m):
   text=C.first_user(r)
   try:g="repo:"+C.PUBLIC.repo_of(text)
   except ValueError:stats["missing_repo"]+=1;continue
   add(dict(record=r,stratum="swe",source_file=m["path"],source_row=i,dataset=m["dataset"],revision=m["revision"],upstream_file=m["upstream_file"],source_record_sha256=C.sha(r),split_group=g,problem_sha256=C.sha(C.PUBLIC.problem(text)),repository_evidence="explicit_workspace_path"))
  log("swe_inventory",file=m["upstream_file"],stats=dict(stats))
 # Previously scanned complete-terminal candidates retain original raw row coordinates.
 for m in read(ROOT/"calibration-gap-v1/terminal-source.lock.json")["files"]:sources[m["path"]]=m
 for c in map(json.loads,(ROOT/"calibration-gap-v1/terminal-raw.jsonl").open()):add(c)
 log("terminal_inventory",stats=dict(stats))
 research=read(ROOT/"calibration-gap-v1/research-source.lock.json")
 for m in research["files"]:sources[m["path"]]=m
 meta=next(m for m in research["files"]if m["upstream_file"]=="train.jsonl")
 assert C.file_sha(ROOT/meta["path"])==meta["sha256"]
 for i,line in enumerate((ROOT/meta["path"]).open()):
  raw=json.loads(line)
  if raw["num_tool_calls"]<4 or raw["type"]!="long_form":continue
  q=raw["question"];g="research_question:"+C.sha(C.PUBLIC.problem(q))
  add(dict(record=raw,stratum="research",source_file=meta["path"],source_row=i,dataset=meta["dataset"],revision=meta["revision"],upstream_file=meta["upstream_file"],source_record_sha256=C.sha(raw),split_group=g,problem_sha256=C.sha(C.PUBLIC.problem(q)),task_text=q))
 selected=[c for k in sorted(heaps)for _,_,c in sorted(heaps[k],key=lambda x:-x[0])]
 with(OUT/"raw-candidates.jsonl").open("w")as f:
  for c in selected:f.write(C.canonical(c)+"\n")
 save("sources.lock.json",dict(files=list(sources.values()),source_locks_used=["nvidia-calibration-v4/source.lock.json","calibration-gap-v1/swe-extension-source.lock.json","calibration-gap-v1/terminal-source.lock.json","calibration-gap-v1/research-source.lock.json"]))
 save("owners.json",own);save("inventory.json",dict(status="candidate_inventory",candidates=len(selected),cells={":".join(k):len(v)for k,v in heaps.items()},stats=stats,policy_sha256=C.file_sha(OUT/"policy.json"),raw_candidates_sha256=C.file_sha(OUT/"raw-candidates.jsonl")))
 with(OUT/"candidate-goals.jsonl").open("w")as f:
  for c in selected:f.write(C.canonical({k:c[k]for k in ["source_record_sha256","partition","workflow_family","stratum","split_group","source_file","source_row","objective_text","metadata_priority"]})+"\n")
 log("inventory_complete",candidates=len(selected),cells={":".join(k):len(v)for k,v in heaps.items()})
if __name__=="__main__":inventory()
