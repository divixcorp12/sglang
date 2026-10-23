"""CPU-only source-preserving normalization and structural observation checks."""
import os
os.environ["CUDA_VISIBLE_DEVICES"]="";os.environ["TOKENIZERS_PARALLELISM"]="false"
import sys,json,collections,concurrent.futures,importlib.util,time,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=Path(__file__).resolve().parent
sys.path[:0]=[str(OUT),str(ROOT/"calibration-mix-v3"),str(ROOT/"calibration-gap-v1")]
import inventory as I
C=I.C;N=None
def init():
 global N
 C.PUBLIC.init()
 spec=importlib.util.spec_from_file_location("public_native_normalizer",ROOT/"calibration-gap-v1/normalize_native.py");N=importlib.util.module_from_spec(spec);spec.loader.exec_module(N);N.init()
def work(c):
 try:
  result=C.PUBLIC.convert(c)if c["stratum"]=="swe"else N.convert(c)
  if not result["ok"]:return dict(ok=False,source_record_sha256=c["source_record_sha256"],reason=result.get("reason","adapter_rejection"),cell=[c["partition"],c["workflow_family"],c["stratum"]])
  r=result["row"];source_domain=r["domain"];r.update(domain="general_tool_use",source_adapter_domain=source_domain,workflow_family=c["workflow_family"],workflow_families=[c["workflow_family"]],objective_text=c["objective_text"],metadata_priority=c["metadata_priority"],public_workflow_policy_sha256=C.file_sha(OUT/"policy.json"),calibration_scope="public_technical_tools_candidate",calibration_eligible=False)
  if r["valid_tokens"]<I.P["minimum_tokens"]:raise ValueError("below_context_floor")
  if c["stratum"]=="swe":
   observed=sum(m["role"]=="tool"and m.get("name")not in ["think","finish"]for m in r["messages"])
   # Missing tool names are still actual source result messages, validated by adapter IDs.
   if observed<I.P["minimum_observed_action_results"]:raise ValueError("insufficient_observed_actions")
   r["observed_action_results"]=observed
  else:r["observed_action_results"]=r.get("terminal_observation_count",0)+r.get("embedded_retrieval_results",0)
  return dict(ok=True,row=r)
 except Exception as e:return dict(ok=False,source_record_sha256=c["source_record_sha256"],reason=type(e).__name__+":"+str(e),cell=[c["partition"],c["workflow_family"],c["stratum"]])
def main():
 assert not(OUT/"normalized.jsonl").exists()
 rows=list(map(json.loads,(OUT/"raw-candidates.jsonl").open()));stats=collections.Counter();cells=collections.Counter();started=time.time()
 with(OUT/"normalized.jsonl").open("w")as dest,(OUT/"normalization-rejections.jsonl").open("w")as rejected,concurrent.futures.ProcessPoolExecutor(max_workers=4,initializer=init)as ex:
  for n,result in enumerate(ex.map(work,rows,chunksize=2),1):
   if result["ok"]:
    r=result["row"];dest.write(C.canonical(r)+"\n");stats["normalized"]+=1;cells[r["partition"]+":"+r["workflow_family"]+":"+r["stratum"]]+=1
   else:rejected.write(C.canonical(result)+"\n");stats[result["reason"]]+=1
   if n%16==0:I.log("normalizing",completed=n,total=len(rows),stats=dict(stats),elapsed_seconds=time.time()-started)
 C.save(OUT/"normalization.json",dict(status="completed_candidate_normalization",stats=stats,cells=cells,rows=len(rows),normalized_sha256=C.file_sha(OUT/"normalized.jsonl"),policy_sha256=C.file_sha(OUT/"policy.json"),elapsed_seconds=time.time()-started))
 I.log("normalization_complete",stats=dict(stats),cells=dict(cells))
if __name__=="__main__":main()
