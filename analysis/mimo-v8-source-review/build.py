"""Additive, score-blind public workflow inventory and normalization."""
import os
os.environ["CUDA_VISIBLE_DEVICES"]="";os.environ["TOKENIZERS_PARALLELISM"]="false"
import sys,json,re,collections,concurrent.futures,time,copy,hashlib
from pathlib import Path
ROOT=Path("/mnt/nvme4/mimo-reap");OUT=Path(__file__).parent;DEST=ROOT/"calibration-mix-v7"
sys.path[:0]=[str(ROOT/"public-tool-workflows-v1"),str(ROOT/"calibration-mix-v3"),str(ROOT/"calibration-gap-v1")]
import inventory as I,normalize as N,checks as K,selection as S,pipeline as W,verify_bound as V
from finish import overlap_inputs,visible
C=I.C;PARTS=("primary","repeat")
def read(p):return C.read(p)
def rows(p):return list(map(json.loads,Path(p).open()))
def emit(p,rs):
 p=Path(p);assert not p.exists(),p
 with p.open("w")as f:
  for r in rs:f.write(C.canonical(r)+"\n")
def log(phase,**kw):
 d=dict(phase=phase,time=time.time(),**kw);C.save(OUT/"progress.json",d)
 with(OUT/"steps.jsonl").open("a")as f:f.write(C.canonical(d)+"\n")
 print(C.canonical(d),flush=True)
def base():
 return {p:[x["record"]for x in rows(ROOT/"calibration-mix-v6/all-selected.jsonl")if x["partition"]==p]for p in PARTS}
def policy():return read(OUT/"policy.json")
def classify(goal,st,group):
 p=policy();text=goal.lower();out=[]
 for cat,spec in p["categories"].items():
  if (st=="research")!=spec["research"]:continue
  if re.search(spec["pattern"],text,re.I):
   if cat=="mesh_numerical"and not re.search(r"pyvista|mesh|fem|scipy|nilearn|numpy|astropy|rioxarray|geopandas|xarray|sympy|numerical|scikit|uxarray|vtk|gmsh|dolfin|fenics|jax",group,re.I):continue
   out.append(cat)
 return out
def inventory():
 assert not(OUT/"raw-candidates.jsonl").exists()
 b=base();own=C.owner_table(b);C.save(OUT/"owners.json",own)
 existing={r.get("source_record_sha256")for rs in b.values()for r in rs};rc=collections.Counter();rt=collections.Counter()
 for rs in b.values():
  for r in rs:rc[r["split_group"].lower()]+=1;rt[r["split_group"].lower()]+=r["valid_tokens"]
 pools=collections.defaultdict(list);stats=collections.Counter();metas=read(ROOT/"calibration-mix-v6/sources.lock.json")["files"]
 def add(raw,m,i,st):
  goal=I.clean_goal(raw["question"]if st=="research"else C.first_user(raw))
  try:g="research_question:"+C.sha(C.PUBLIC.problem(goal))if st=="research"else"repo:"+C.PUBLIC.repo_of(C.first_user(raw))
  except ValueError:return
  cats=classify(goal,st,g)
  if not cats:return
  h=C.sha(raw)
  if h in existing:return
  if g.startswith("repo:")and(rc[g]>=2 or rt[g]>=100000):return
  if not 1500<=len(C.canonical(raw))<=400000:return
  part=I.partition(g,own)
  for cat in cats:
   c=dict(record=raw,stratum=st,source_file=m["path"],source_row=i,dataset=m["dataset"],revision=m["revision"],upstream_file=m["upstream_file"],source_record_sha256=h,split_group=g,problem_sha256=C.sha(C.PUBLIC.problem(C.first_user(raw)if st=="swe"else raw["question"])),repository_evidence="explicit_workspace_path"if st=="swe"else"source_question_hash",partition=part,workflow_family=policy()["categories"][cat]["family"],workflow_families=[policy()["categories"][cat]["family"]],objective_text=goal,metadata_priority=C.sha([policy()["seed"],h]),gap_category=cat)
   if st=="research":c["task_text"]=raw["question"]
   key=(cat,part,g);pools[key].append(c);pools[key]=sorted(pools[key],key=lambda c:c["metadata_priority"])[:2];stats[cat+":"+part]+=1
 for m in metas:
  if m["path"].endswith(".parquet")and"SFT-SWE"in m["dataset"]:
   assert C.file_sha(ROOT/m["path"])==m["sha256"]
   for i,r in C.PUBLIC.rows(m):add(r,m,i,"swe")
   log("inventory_source",file=m["path"],eligible=dict(stats))
  elif m["path"].endswith("train.jsonl")and"dr-tulu"in m["dataset"]:
   assert C.file_sha(ROOT/m["path"])==m["sha256"]
   for i,line in enumerate((ROOT/m["path"]).open()):
    r=json.loads(line)
    if r["num_tool_calls"]>=4 and r["type"]=="long_form":add(r,m,i,"research")
   log("inventory_research",eligible=dict(stats))
 candidates=[r for k,rs in sorted(pools.items())for r in rs]
 emit(OUT/"raw-candidates.jsonl",candidates);C.save(OUT/"sources.lock.json",dict(files=metas))
 C.save(OUT/"inventory.json",dict(candidates=len(candidates),eligible=dict(stats),policy_sha256=C.file_sha(OUT/"policy.json"),score_inputs_read=False))
 log("inventory_complete",candidates=len(candidates))
def norm_init():N.init()
def norm_work(c):
 v=N.work(c)
 if v["ok"]:
  r=v["row"];r["gap_category"]=c["gap_category"];r["public_workflow_policy_sha256"]=C.file_sha(OUT/"policy.json")
  if r["stratum"]=="research":
   titles=sorted(set(re.findall(r'<(?:snippet|webpage)[^>]*>\s*Title:\s*([^\n]+)',r["messages"][-1]["content"])))
   r["retrieved_paper_titles"]=titles;r["source_papers"]=sorted({"paper_title:"+C.sha(" ".join(re.findall(r"\w+",t.lower())))for t in titles})
 return v
def normalize():
 assert not(OUT/"normalized.jsonl").exists();raw=rows(OUT/"raw-candidates.jsonl");ok=[];bad=[]
 with concurrent.futures.ProcessPoolExecutor(max_workers=4,initializer=norm_init)as ex:
  for i,v in enumerate(ex.map(norm_work,raw,chunksize=2),1):
   (ok if v["ok"]else bad).append(v["row"]if v["ok"]else v)
   if i%40==0:log("normalizing",completed=i,total=len(raw),admitted=len(ok))
 emit(OUT/"normalized.jsonl",ok);emit(OUT/"normalization-rejections.jsonl",bad);log("normalized",records=len(ok),rejected=len(bad))
def screen():
 assert not(OUT/"screened.jsonl").exists()
 pins,_,held,reserved,question,forbidden,grams=overlap_inputs()
 b=base();old=[r for rs in b.values()for r in rs];features=[];index=collections.defaultdict(set);exact=set()
 for r in old+held:
  q=S.objective(r)if r in old else question(r)
  exact.add(C.sha(C.PUBLIC.problem(q)));fs=C.PUBLIC.features(q);j=len(features);features.append(fs)
  for g in fs:index[g].add(j)
 owners=read(OUT/"owners.json");lineage={p:{k:{str(v).lower()for r in b[p]for v in r.get(k,[])}for k in ["lineage_aliases","source_papers","dataset_groups"]}for p in PARTS}
 accept=[];bad=[];seen=set()
 for n,r in enumerate(rows(OUT/"normalized.jsonl"),1):
  key=(r["sample_id"],r["gap_category"])
  if key in seen:continue
  seen.add(key);why=None
  try:K.record(r)
  except Exception as e:why="structure:"+str(e)
  q=S.objective(r);fs=C.PUBLIC.features(q);hits=collections.Counter(j for g in fs for j in index.get(g,()))
  if r["split_group"].lower()in reserved:why="heldout_group"
  if C.sha(C.PUBLIC.problem(q))in exact:why="existing_or_heldout_exact"
  if any(v>=8 and C.PUBLIC.similar(fs,features[j])for j,v in hits.items()):why="existing_or_heldout_near"
  if grams(visible(r))&forbidden:why="heldout_visible_span"
  other=PARTS[1-PARTS.index(r["partition"])]
  if any(set(map(str.lower,r.get(k,[])))&lineage[other][k]for k in lineage[other]):why="opposite_parent_lineage"
  if why:bad.append(dict(sample_id=r["sample_id"],category=r["gap_category"],reason=why))
  else:accept.append(r)
 emit(OUT/"screened.jsonl",accept);C.save(OUT/"screening.json",dict(accepted=len(accept),rejected=bad,input_pins=pins))
 briefs=[]
 for r in accept:
  brief=S.brief(r);brief["category"]=r["gap_category"];brief["priority"]=r["metadata_priority"];briefs.append(brief)
 C.save(OUT/"review-queue.json",briefs);log("screen_complete",accepted=len(accept),rejected=len(bad))
if __name__=="__main__":
 globals()[sys.argv[1]]()

