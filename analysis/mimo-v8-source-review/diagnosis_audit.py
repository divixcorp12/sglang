"""CPU-only fixed-score attribution of v10's paired layer-32 tool gate."""
import os
os.environ["CUDA_VISIBLE_DEVICES"]=""
os.environ["OPENBLAS_NUM_THREADS"]="1"
import collections,datetime,hashlib,json,math,time
from pathlib import Path
import numpy as np
ROOT=Path("/mnt/nvme4/mimo-reap"); RUN=ROOT/"full-scoring-v10"; OUT=Path(__file__).parent
P=("primary","repeat"); KEEP=344; TH=.95; pins={}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):
 p=Path(p);pins[str(p)]=sha(p);return json.loads(p.read_text())
def save(n,d):(OUT/n).write_text(json.dumps(d,indent=2,allow_nan=False)+"\n")
def order(x):return np.lexsort((np.arange(384),-np.asarray(x)))
def rank(x):
 r=np.empty(384,dtype=int);r[order(x)]=np.arange(1,385);return r
def chosen(x):return set(map(int,order(x)[:KEEP]))
def jac(a,b):return len(a&b)/len(a|b)
def scalar(x):return set(sorted(range(384),key=lambda e:(-x[e],e))[:KEEP])
def desc(s):
 return dict(sample_id=s["sample_id"],repository=s["split_group"],dataset=s.get("dataset"),family=s.get("workflow_family"),tokens=s["valid_tokens"],task_key=s["task_key"],source_file=s.get("source_file"),source_row=s.get("source_row"),source_record_sha256=s.get("source_record_sha256"),objective=(s.get("objective_text") or next((str(m.get("content","")) for m in s["messages"] if m["role"]=="user"),""))[:1800])
def main():
 start=time.time();gate=read(RUN/"early-gate.json");lock=read(RUN/"inputs.lock.json");state=read(RUN/"state.json")
 assert state["status"]=="early_gate_failed"
 assert gate["threshold"]==TH and gate["keep"]==KEEP
 evidence={p:read(RUN/(p+"-early-evidence.json")) for p in P}
 contexts={};indices={}
 for p in P:
  assert sha(RUN/(p+"-early-evidence.json"))==gate["evidence_sha256"][p]
  planpath=ROOT/lock["partitions"][p]["plan"];plan=read(planpath)
  assert sha(planpath)==lock["inputs_sha256"][str(planpath.relative_to(ROOT))]
  contexts[p]={}
  for g in plan["groups"]:
   if g["domain"]!="general_tool_use":continue
   path=ROOT/g["manifest"];m=read(path)
   assert sha(path)==lock["inputs_sha256"][str(path.relative_to(ROOT))]
   for s in m["samples"]:contexts[p][s["sample_id"]]=desc(s)
  assert len(contexts[p])==40
  ptr=read(RUN/p/"checkpoints/latest.json");folder=RUN/p/"checkpoints"/ptr["generation"];manifest=read(folder/"manifest.json")
  assert sha(folder/"manifest.json")==ptr["manifest_sha256"]==evidence[p]["manifest_sha256"]
  assert manifest["next_stage"]==67
  # The index is checked against hashes embedded in the committed observer state.
  import torch
  sp=folder/"state.pt";assert sha(sp)==manifest["files"]["state.pt"]["sha256"];pins[str(sp)]=sha(sp)
  st=torch.load(sp,map_location="cpu",weights_only=True)["observer"]
  indices[p]=st["written"]
 composition={}
 for p in P:
  composition[p]={}
  for key in ["family","dataset"]:
   out={}
   for r in contexts[p].values():
    v=out.setdefault(str(r[key]),dict(episodes=0,tokens=0));v["episodes"]+=1;v["tokens"]+=r["tokens"]
   composition[p][key]=out
 assert not ({r["repository"] for r in contexts["primary"].values()} & {r["repository"] for r in contexts["repeat"].values()})
 comparisons={};details={};influences=collections.defaultdict(list);group_rows=[];scalar_checks=0;matrix={}
 for layer in range(1,33):
  tab={}
  for p in P:
   path=RUN/p/"trajectories"/f"layer-{layer:02d}.json";doc=read(path)
   assert sha(path)==indices[p][str(layer)]["sha256"] and doc["aggregate_check_passed"]
   rr=[r for r in doc["records"] if r["domain"]=="general_tool_use"]
   assert {r["sample_id"] for r in rr}==set(contexts[p])
   for r in rr:
    assert r["valid_tokens"]==contexts[p][r["sample_id"]]["tokens"]
   x=np.asarray([r["gate_weighted_output_norm_sum"] for r in rr],dtype=np.float64);counts=np.asarray([r["route_counts"] for r in rr])
   agg=evidence[p]["layers"][str(layer)]["general_tool_use"];total=np.asarray(agg["gate_weighted_output_norm_sum"])
   assert np.isfinite(x).all() and (x>=0).all()
   assert np.allclose(x.sum(0),total,rtol=1e-12,atol=1e-8)
   assert np.array_equal(counts.sum(0),agg["route_counts"])
   assert np.array_equal(counts.sum(1),[8*r["valid_tokens"] for r in rr])
   tab[p]=dict(rr=rr,x=x,total=total,selected=chosen(total),ranks=rank(total))
  baseline=jac(tab["primary"]["selected"],tab["repeat"]["selected"]);assert baseline==gate["jaccard_by_layer_domain"][f"{layer}:general_tool_use"]
  comparisons[str(layer)]=baseline
  for p in P:
   t=tab[p];other=tab[P[1-P.index(p)]]
   def probe(mask):
    rem=t["x"][mask].sum(0);sel=chosen(rem);value=jac(sel,other["selected"])
    return dict(layer=layer,jaccard=value,delta=value-baseline,rescued=baseline<TH<=value,new_failure=value<TH<=baseline,changed_experts=sorted(t["selected"]^sel))
   for i,r in enumerate(t["rr"]):
    mask=np.arange(40)!=i;res=probe(mask)
    # Independent high-accuracy sums and Python stable ordering validate all exclusions.
    summed=[math.fsum(float(v) for v in t["x"][mask,e]) for e in range(384)]
    assert scalar(summed)==chosen(t["x"][mask].sum(0));scalar_checks+=1
    influences[(p,r["sample_id"])].append(res)
   for field in ["repository","family","dataset"]:
    for value in sorted({str(contexts[p][r["sample_id"]][field]) for r in t["rr"]}):
     mask=np.array([str(contexts[p][r["sample_id"]][field])!=value for r in t["rr"]])
     if not mask.any():continue
     group_rows.append(dict(partition=p,field=field,value=value,**probe(mask)))
  if baseline<TH:
   exclusive={p:sorted(tab[p]["selected"]-tab[P[1-P.index(p)]]["selected"]) for p in P}
   disputed=sorted(set().union(*map(set,exclusive.values()))); experts=[]
   for e in disputed:
    ex=dict(expert=e,retained_in=next(p for p in P if e in exclusive[p]),partitions={})
    for p in P:
     t=tab[p];cut=float(t["total"][order(t["total"])[343]]);con=t["x"][:,e]/t["total"][e];top=np.argsort(-con)[:5]
     ex["partitions"][p]=dict(rank=int(t["ranks"][e]),relative_to_cutoff=float(t["total"][e]/cut-1),normalized_score=float(t["total"][e]/t["total"].sum()),top_trajectories=[dict(**contexts[p][t["rr"][i]["sample_id"]],expert_score_share=float(con[i]),token_share=t["rr"][i]["valid_tokens"]/sum(r["valid_tokens"] for r in t["rr"])) for i in top],effective_trajectory_count=float(1/np.dot(con,con)))
    experts.append(ex)
   families=sorted({r["family"] for c in contexts.values() for r in c.values()});ft={p:sum(r["tokens"] for r in contexts[p].values()) for p in P}
   target={f:sum(sum(r["tokens"] for r in contexts[p].values() if r["family"]==f)/ft[p] for p in P)/2 for f in families}
   matched={}
   for p in P:
    t=tab[p];matched[p]=sum(target[f]*t["x"][[contexts[p][r["sample_id"]]["family"]==f for r in t["rr"]]].sum(0)/sum(r["tokens"] for r in contexts[p].values() if r["family"]==f) for f in families)
   details[str(layer)]=dict(jaccard=baseline,exclusive=exclusive,experts=experts,matched_family_token_share_jaccard=jac(chosen(matched["primary"]),chosen(matched["repeat"])),boundary_gap={p:float(1-tab[p]["total"][order(tab[p]["total"])[344]]/tab[p]["total"][order(tab[p]["total"])[343]]) for p in P})
   matrix[str(layer)]=tab
  print("Completed layer",layer,flush=True)
 summaries=[]
 for (p,sid),rows in influences.items():
  summaries.append(dict(partition=p,**contexts[p][sid],rescued_layers=[r["layer"] for r in rows if r["rescued"]],new_failed_layers=[r["layer"] for r in rows if r["new_failure"]],minimum_jaccard=min(r["jaccard"] for r in rows),sum_absolute_delta=sum(abs(r["delta"]) for r in rows),by_layer=rows))
 summaries.sort(key=lambda r:(len(r["rescued_layers"]),-len(r["new_failed_layers"]),r["sum_absolute_delta"]),reverse=True)
 report=dict(status="diagnostic_only",run=str(RUN),gpu_used=False,keep=KEEP,threshold=TH,failed_layers=sorted(map(int,details)),checks_reproduced=32,scalar_counterfactuals_verified=scalar_checks,composition=composition,layer_details=details,episode_summaries=summaries,group_counterfactuals=group_rows,limitations=["Fixed-score attribution identifies influence, not task-quality loss or a reason to delete examples.","Family share matching is post-hoc diagnosis only, not a revised acceptance gate.","Complete saved history contributions cannot localize particular tool calls without GPU rescoring.","Public correctness remains upstream-attributed; no independent task execution or regrading."],selection_allowed=False)
 save("report.json",report);save("inputs.sha256.json",pins)
 changed=[p for p,h in pins.items() if sha(p)!=h];assert not changed,changed
 save("verification.json",dict(status="passed",files_verified=len(pins),protected_changes=changed,scalar_counterfactuals=scalar_checks,script_sha256=sha(__file__),elapsed_seconds=time.time()-start))
 save("steps.json",dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),command=".venv-compressor/bin/python rank-diagnosis-v6/audit.py",action="Read committed statistics, reaggregate and attribute tool-use ranking failures",gpu_used=False,protected_files_changed=False))
 print("DONE",json.dumps({k:dict(exclusive=v["exclusive"],matched_family=v["matched_family_token_share_jaccard"]) for k,v in details.items()}),flush=True)
if __name__=="__main__":main()

