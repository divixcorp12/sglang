"""Versioned public-only tool replacement, independent CPU verification and manifests."""
import os
os.environ["CUDA_VISIBLE_DEVICES"]=""
os.environ["TOKENIZERS_PARALLELISM"]="false"
import sys,json,copy,collections,re,time,concurrent.futures,argparse,unittest,io
from pathlib import Path
OUT=Path(__file__).resolve().parent;ROOT=OUT.parent;DEST=ROOT/"calibration-mix-v6"
sys.path[:0]=[str(OUT),str(ROOT/"calibration-mix-v3"),str(ROOT/"calibration-gap-v1")]
import inventory as I,checks as K,selection as S,verify_bound as V
from finish import overlap_inputs,visible
C=I.C;P=I.P
def read(p):return C.read(p)
def rows(p):return list(map(json.loads,Path(p).open()))
def save(n,v):C.save(DEST/n,v)
def emit(n,rs):
 path=DEST/n;assert not path.exists(),path
 with path.open("w")as f:
  for r in rs:f.write(C.canonical(r)+"\n")
def log(event,**kw):
 obj=dict(event=event,at=time.time(),**kw)
 with(DEST/"steps.jsonl").open("a")as f:f.write(C.canonical(obj)+"\n")
 save("progress.json",obj);print(C.canonical(obj),flush=True)
def checkpins():
 pins=read(DEST/"inputs.lock.json")
 for p,h in pins.items():
  path=Path(p);path=path if path.is_absolute()else ROOT/path
  assert C.file_sha(path)==h,"Input changed: "+p
 return pins
def build():
 assert not DEST.exists(),"Version already exists; preserve it and use a new revision"
 proposal=read(OUT/"proposal.json");assert not proposal["unfilled_slots"]
 review=read(OUT/"content-review.json");selected=rows(OUT/"proposed.jsonl");decisions={r["sample_id"]:r for r in review["records"]}
 assert review["proposed_sha256"]==C.file_sha(OUT/"proposed.jsonl")
 assert review["status"]=="admitted_diagnostic_cohort"and len(decisions)==len(selected)==80
 parent=I.base();parts={p:[r for r in rs if r["domain"]!="general_tool_use"]for p,rs in parent.items()}
 annotations=read(ROOT/"calibration-mix-v5/annotations.json")
 oldann={r["sample_id"]:annotations[r["sample_id"]]for rs in parts.values()for r in rs}
 assert C.file_sha(ROOT/"calibration-mix-v5/all-selected.jsonl")=="5f78bd10555784c0ecd47448c979ffe5f68da53b037876b58351bf2460f4aa8a"
 for r in selected:
  K.record(r);K.review_bound(r,decisions[r["sample_id"]])
  new=copy.deepcopy(r);new.update(calibration_eligible=True,calibration_scope="diagnostic_only",admission=dict(decision="admit_reviewed_public_technical_tools",policy_sha256=C.file_sha(OUT/"policy.json"),content_review_sha256=C.file_sha(OUT/"content-review.json"),source_correctness="upstream_attributed_not_independently_reexecuted"))
  parts[r["partition"]].append(new)
  oldann[r["sample_id"]]=dict(partition=r["partition"],origin="public_technical_tools_v1",domain="general_tool_use",families=[r["workflow_family"]],quota_family=r["workflow_family"],length_bucket=C.bucket(r["valid_tokens"]))
 assert K.preserved_non_tool(parent,parts)==509
 DEST.mkdir();C.save(DEST/"policy.json",dict(parent="calibration-mix-v5",replacement_policy=read(OUT/"policy.json"),replacement_policy_sha256=C.file_sha(OUT/"policy.json"),non_tool_records_immutable=True,production_calibration_ready=False,selection_allowed=False,native_precision_verified=False))
 emit("all-selected.jsonl",[dict(partition=p,record=r)for p in I.PARTS for r in parts[p]])
 emit("selected-tools.jsonl",selected);save("annotations.json",oldann);save("content-review.json",review)
 sourcefiles={}
 for lock in ["sources.lock.json","extension-01/sources.lock.json","extension-02/sources.lock.json"]:
  for m in read(OUT/lock)["files"]:
   if m["path"]in sourcefiles:assert m["sha256"]==sourcefiles[m["path"]]["sha256"]
   sourcefiles[m["path"]]=m
 save("sources.lock.json",dict(files=list(sourcefiles.values())))
 pins=dict(read(ROOT/"calibration-mix-v5/inputs.lock.json"))
 for name,h in read(ROOT/"calibration-mix-v5/evidence.lock.json")["files"].items():
  assert name not in pins or pins[name]==h;pins[name]=h
 for m in sourcefiles.values():
  assert m["path"]not in pins or pins[m["path"]]==m["sha256"];pins[m["path"]]=m["sha256"]
 files=list(OUT.glob("*.py"))+[OUT/n for n in ["policy.json","classification-review.json","selection-design-v2.json","review-reservations.json","proposed.jsonl","proposal.json","content-review.json"]]
 for folder in [OUT,OUT/"extension-01",OUT/"extension-02"]:
  files += [folder/n for n in ["raw-candidates.jsonl","normalized.jsonl","screened.jsonl","screening.json"]]
 files += [ROOT/"calibration-mix-v5"/n for n in ["all-selected.jsonl","annotations.json","verification.json","coverage.json","completion.json"]]
 files += [ROOT/"calibration-mix-v3"/n for n in ["common.py","verify_bound.py","finish.py"]]+[ROOT/"calibration-gap-v1/normalize_native.py",ROOT/"nvidia-calibration-v4/pipeline.py",ROOT/"build_trajectory_bank.py",ROOT/"prepare_observer_manifests.py"]
 files += [C.MODEL/"config.json",C.MODEL/"model.safetensors.index.json"]+[p for p in set(C.MODEL.glob("*token*"))|set(C.MODEL.glob("*template*"))if p.is_file()]
 for f in files:
  name=str(f.relative_to(ROOT))if f.is_relative_to(ROOT)else str(f)
  h=C.file_sha(f);assert name not in pins or pins[name]==h; pins[name]=h
 save("inputs.lock.json",pins)
 save("build-report.json",dict(parent_records=637,preserved_non_tool_records=509,replaced_tool_records=128,new_tool_records=80,all_selected_sha256=C.file_sha(DEST/"all-selected.jsonl"),partitions={p:dict(episodes=len(rs),tokens=sum(r["valid_tokens"]for r in rs))for p,rs in parts.items()},scoring_launched=False))
 log("build_complete",**read(DEST/"build-report.json"))
def sourceverify(selected):
 by=collections.defaultdict(dict);metas={m["path"]:m for m in read(DEST/"sources.lock.json")["files"]}
 for r in selected:by[r["source_file"]][r["source_row"]]=r
 count=0
 for file,wanted in by.items():
  meta=metas[file]
  assert C.file_sha(ROOT/file)==meta["sha256"]
  if file.endswith(".parquet"):source=C.PUBLIC.rows(meta)
  else:source=((i,json.loads(line))for i,line in enumerate((ROOT/file).open()))
  seen=set()
  for i,raw in source:
   if i not in wanted:
    if i>max(wanted):break
    continue
   r=wanted[i];assert C.sha(raw)==r["source_record_sha256"]
   assert r["sample_id"]==C.sha([r["dataset"],r["revision"],r["source_record_sha256"]])
   for k in ["dataset","revision","upstream_file"]:assert r[k]==meta[k]
   if r["stratum"]=="swe":
    V.check_lossless(raw,r);assert r["split_group"]=="repo:"+C.PUBLIC.repo_of(C.first_user(raw))
    assert C.PUBLIC.content_rejection(raw,"swe")is None
   else:
    assert r["messages"]==json.loads(C.canonical(raw["conversations"])),"Native history altered"
    if r["stratum"]=="research":
     assert I.clean_goal(raw["question"])==r["objective_text"]
     assert r["split_group"]=="research_question:"+C.sha(C.PUBLIC.problem(raw["question"]))
     assert raw["num_tool_calls"]==K.native_observations(r)
     titles=sorted(set(re.findall(r'<(?:snippet|webpage)[^>]*>\s*Title:\s*([^\n]+)',r["messages"][-1]["content"])))
     assert titles==r["retrieved_paper_titles"]
     assert sorted({"paper_title:"+C.sha(" ".join(re.findall(r"\w+",t.lower())))for t in titles})==r["source_papers"]
    else:
     assert r["split_group"]=="terminal_project:nvidia-terminal-corpus/"+raw["task"]
   count+=1;seen.add(i)
  assert seen==set(wanted);log("raw_source_verified",file=file,records=len(seen))
 assert count==len(selected);return count
def screenverify(selected):
 pins,_,held,reserved,question,forbidden,grams=overlap_inputs()
 parent=[r for rs in I.base().values()for r in rs if r["domain"]!="general_tool_use"]
 exact=set();features=[];index=collections.defaultdict(set)
 for r,q in [(r,S.objective(r))for r in parent]+[(r,question(r))for r in held]:
  h=C.sha(C.PUBLIC.problem(q))
  if h in exact:continue
  exact.add(h);fs=C.PUBLIC.features(q);j=len(features);features.append(fs)
  for g in fs:index[g].add(j)
 previous=[]
 for n,r in enumerate(selected,1):
  fs=C.PUBLIC.features(S.objective(r));hits=collections.Counter(j for g in fs for j in index.get(g,()))
  assert r["split_group"].lower()not in reserved
  assert C.sha(C.PUBLIC.problem(S.objective(r)))not in exact
  assert not any(v>=8 and C.PUBLIC.similar(fs,features[j])for j,v in hits.items())
  assert not any(C.PUBLIC.similar(fs,f)for f in previous)
  assert not grams(visible(r))&forbidden,"Uncommon held-out content span"
  previous.append(fs)
  if n%20==0:log("independent_overlap_screen",completed=n,total=len(selected))
 for name,h in pins.items():
  path=Path(name);assert C.file_sha(path if path.is_absolute()else ROOT/path)==h
 return dict(status="passed",heldout_records=len(held),preserved_non_tool_records=len(parent),novel_tool_records=len(selected),checks="All visible assistant/tool content uncommon spans; exact/near objectives; held-out groups; no new wrapper exception",limits="Does not establish semantic or pretraining decontamination")
def coverage(parts,selected,annotations):
 cov={};tool={}
 for p,rs in parts.items():
  cov[p]={}
  for domain in C.DOMAINS:
   subset=[r for r in rs if r["domain"]==domain];cov[p][domain]=dict(episodes=len(subset),tokens=sum(r["valid_tokens"]for r in subset),lengths=dict(collections.Counter(C.bucket(r["valid_tokens"])for r in subset)))
  local=[r for r in selected if r["partition"]==p];tool[p]={}
  assert len(local)==40
  for fam in P["families"]:
   rr=[r for r in local if r["workflow_family"]==fam];groups={r["split_group"].lower()for r in rr}
   assert len(rr)==P["target_episodes_per_family_per_partition"]and len(groups)>=P["minimum_distinct_groups_per_family_per_partition"]
   tool[p][fam]=dict(episodes=len(rr),tokens=sum(r["valid_tokens"]for r in rr),distinct_groups=len(groups),real_repositories=sum(g.startswith("repo:")for g in groups),sources=dict(collections.Counter(r["dataset"]for r in rr)))
  assert sum(r["valid_tokens"]>=16384 for r in local)>=P["minimum_16k_episodes_each_partition"]
  assert sum(r["valid_tokens"]>=32768 for r in local)>=P["minimum_32k_episodes_each_partition"]
  coding=[r for r in rs if r["domain"]=="coding"];ct=collections.Counter()
  for r in coding:
   if r["split_group"].lower().startswith("repo:"):ct[r["split_group"].lower()]+=r["valid_tokens"]
  assert max(ct.values(),default=0)/sum(r["valid_tokens"]for r in coding)<=P["combined_coding_repository_token_share_cap"]
 # Preserve pre-existing coding gaps as explicit independent concerns.
 oldgaps=read(ROOT/"calibration-mix-v5/coverage.json")["quota_and_coverage_gaps"]
 preserved=[g for g in oldgaps if ":general_tool_use:"not in g]
 return dict(partitions=cov,public_tool_families=tool,public_tool_policy_passed=True,preserved_non_tool_coverage_gaps=preserved,coverage_requirements_passed=not preserved,old_tool_quota_definition_superseded_by_frozen_public_tool_policy=True,selection_uses_expert_scores=False)
def verify():
 assert not(DEST/"verification.json").exists()
 began=time.time();pins=checkpins();selected=rows(DEST/"selected-tools.jsonl")
 report=read(DEST/"test-report.json")
 assert report["status"]=="passed"and report["test_code_sha256"]==C.file_sha(OUT/"test_checks.py")and report["checks_code_sha256"]==C.file_sha(OUT/"checks.py")
 objects=rows(DEST/"all-selected.jsonl");parts={p:[o["record"]for o in objects if o["partition"]==p]for p in I.PARTS}
 parent=I.base();assert read(OUT/"owners.json")==I.owners(),"Historical ownership changed";preserved=K.preserved_non_tool(parent,parts);review=read(DEST/"content-review.json")
 decisions={r["sample_id"]:r for r in review["records"]};scope=read(OUT/"classification-review.json")["decisions"];annotations=read(DEST/"annotations.json")
 orig=read(ROOT/"calibration-mix-v5/annotations.json")
 for rs in parts.values():
  for r in rs:
   if r["domain"]!="general_tool_use":assert annotations[r["sample_id"]]==orig[r["sample_id"]]
   else:
    src=next(s for s in selected if s["sample_id"]==r["sample_id"])
    expected=copy.deepcopy(src);expected.update(calibration_eligible=True,calibration_scope="diagnostic_only",admission=dict(decision="admit_reviewed_public_technical_tools",policy_sha256=C.file_sha(OUT/"policy.json"),content_review_sha256=C.file_sha(OUT/"content-review.json"),source_correctness="upstream_attributed_not_independently_reexecuted"))
    assert r==expected
    assert annotations[r["sample_id"]]==dict(partition=r["partition"],origin="public_technical_tools_v1",domain="general_tool_use",families=[r["workflow_family"]],quota_family=r["workflow_family"],length_bucket=C.bucket(r["valid_tokens"]))
 for r in selected:
  K.record(r);K.review_bound(r,decisions[r["sample_id"]]);assert r["scope_review_sha256"]==C.sha(scope[r["sample_id"]]);assert scope[r["sample_id"]]["workflow_family"]==r["workflow_family"]
  assert I.partition(r["split_group"],read(OUT/"owners.json"))==r["partition"]
 cap=K.repo_caps(parent,selected);overlap=K.partition_overlap(parts);source_count=sourceverify(selected);screen=screenverify(selected)
 allrows=[r for rs in parts.values()for r in rs]
 with concurrent.futures.ProcessPoolExecutor(max_workers=4,initializer=V.init)as ex:
  for n,_ in enumerate(ex.map(V.check_token,allrows,chunksize=4),1):
   if n%100==0:log("independent_exact_retokenization",completed=n,total=len(allrows))
 cov=coverage(parts,selected,annotations);save("coverage.json",cov)
 save("overlap-audit.json",dict(status="passed",primary_repeat_overlap=overlap,combined_repository_caps=cap,novel_source_rows=source_count,screen=screen))
 assert C.file_sha(DEST/"all-selected.jsonl")==read(DEST/"build-report.json")["all_selected_sha256"]
 checkpins()
 result=dict(status="passed",raw_source_rows_verified=source_count,retokenized=len(allrows),preserved_non_tool_records_exactly=preserved,new_public_tool_records=len(selected),episodes={p:len(rs)for p,rs in parts.items()},tokens={p:sum(r["valid_tokens"]for r in rs)for p,rs in parts.items()},tool_policy_requirements_passed=True,coverage_requirements_passed=cov["coverage_requirements_passed"],coverage_gaps=cov["preserved_non_tool_coverage_gaps"],all_selected_sha256=C.file_sha(DEST/"all-selected.jsonl"),pipeline_sha256=C.file_sha(__file__),source_correctness="upstream_attributed_not_independently_reexecuted",source_selection_uses_expert_scores=False,production_calibration_ready=False,selection_allowed=False,native_precision_verified=False,scoring_launched=False,inputs_rechecked=len(pins),elapsed_seconds=time.time()-began)
 save("verification.json",result);log("verification_complete",**result)
def assemble():
 assert not(DEST/"completion.json").exists()
 v=read(DEST/"verification.json");assert v["status"]=="passed"and v["pipeline_sha256"]==C.file_sha(__file__);checkpins()
 objects=rows(DEST/"all-selected.jsonl");old=read(ROOT/"calibration-mix-v5/completion.json")["plans"];plans={};artifacts={}
 for part in I.PARTS:
  rr=[o["record"]for o in objects if o["partition"]==part];dest=ROOT/"data"/("observer-manifests-representative-v6-"+part);dest.mkdir(exist_ok=False)
  groups=[];counts={}
  for domain in C.DOMAINS:
   dr=[r for r in rr if r["domain"]==domain];buf=[];n=0;ix=0
   for r in dr:
    if buf and n+r["valid_tokens"]>65536:groups.append(C.write(dest/f"{domain}-{ix:02d}.json",buf,domain,DEST/"overlap-audit.json"));buf=[];n=0;ix+=1
    buf.append(r);n+=r["valid_tokens"]
   if buf:groups.append(C.write(dest/f"{domain}-{ix:02d}.json",buf,domain,DEST/"overlap-audit.json"))
   counts[domain]=dict(sequences=len(dr),tokens=sum(r["valid_tokens"]for r in dr))
  plan=dict(schema="mimo_observation_execution_plan_v1",partition=part,domain_weights=read(ROOT/old[part])["domain_weights"],groups=groups,coverage=counts,maximum_sequence_tokens=max(r["valid_tokens"]for r in rr),group_token_limit=65536,production_calibration_ready=False,diagnostic_coverage_ready=v["coverage_requirements_passed"],tool_diagnostic_coverage_ready=v["tool_policy_requirements_passed"],coverage_requirements_passed=v["coverage_requirements_passed"],native_precision_verified=False,selection_allowed=False,scoring_launched=False,purpose="Public-only reviewed technical tool cohort replaces v5 tools; all 509 coding/science/finance records unchanged",source_verification_sha256=C.file_sha(DEST/"verification.json"),workflow_annotations=str((DEST/"annotations.json").relative_to(ROOT)),pending=["Preserved non-tool coverage gaps in coverage.json","Fresh scoring storage/GPU availability preflight","Independent rank stability","Native precision and paired objective-quality evaluation"])
  C.save(dest/"plan.json",plan);plans[part]=str((dest/"plan.json").relative_to(ROOT))
  emitted=collections.defaultdict(list)
  for group in groups:
   m=read(ROOT/group["manifest"]);assert m["external_overlap_audit_sha256"]==C.file_sha(DEST/"overlap-audit.json");assert m["config_sha256"]==C.file_sha(C.MODEL/"config.json")and m["index_sha256"]==C.file_sha(C.MODEL/"model.safetensors.index.json")
   for name,h in m["tokenizer_sha256"].items():assert C.file_sha(C.MODEL/name)==h
   assert sum(r["valid_tokens"]for r in m["samples"])==group["tokens"]<=65536;emitted[group["domain"]]+=m["samples"]
  for domain in C.DOMAINS:assert emitted[domain]==[r for r in rr if r["domain"]==domain]
  for path in dest.glob("*.json"):artifacts[str(path.relative_to(ROOT))]=C.file_sha(path)
 save("manifest-audit.json",dict(status="passed",exact_manifest_records_verified=len(objects),model_and_tokenizer_bound=True))
 save("completion.json",dict(status="diagnostic_mix_assembled",plans=plans,artifacts_sha256=artifacts,verification_sha256=C.file_sha(DEST/"verification.json"),tool_policy_requirements_passed=True,coverage_requirements_passed=v["coverage_requirements_passed"],remaining_gaps=v["coverage_gaps"],scoring_launched=False,active_pointers_changed=False,native_precision_verified=False,selection_allowed=False))
 log("assembly_complete",plans=plans)
def tests():
 assert DEST.exists()
 from test_checks import Suite
 began=time.time();buf=io.StringIO();result=unittest.TextTestRunner(stream=buf,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Suite))
 save("test-report.json",dict(status="passed"if result.wasSuccessful()else"failed",tests=result.testsRun,details=buf.getvalue(),test_code_sha256=C.file_sha(OUT/"test_checks.py"),checks_code_sha256=C.file_sha(OUT/"checks.py"),elapsed_seconds=time.time()-began))
 print(buf.getvalue(),flush=True);assert result.wasSuccessful()
if __name__=="__main__":
 parser=argparse.ArgumentParser();parser.add_argument("phase",choices=["build","tests","verify","assemble"]);args=parser.parse_args();globals()[args.phase]()

