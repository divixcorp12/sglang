"""Additive release, guards, source/token verification and deterministic replay."""
import os
os.environ["CUDA_VISIBLE_DEVICES"]=""
import sys,importlib.util,collections,copy,json,math,concurrent.futures,time
from pathlib import Path
spec=importlib.util.spec_from_file_location("gapbuilder",Path(__file__).with_name("build.py"));B=importlib.util.module_from_spec(spec);spec.loader.exec_module(B)
C=B.C;K=B.K;S=B.S;W=B.W;V=B.V;ROOT=B.ROOT;OUT=B.OUT;DEST=B.DEST;P=B.PARTS
def assert_preserved(parent,parts,ann,oldann):
 for p in P:
  old={r["sample_id"]:r for r in parent[p]};new={r["sample_id"]:r for r in parts[p]}
  if len(new)!=len(parts[p]):raise ValueError("duplicate_sample")
  for sid,r in old.items():
   if new.get(sid)!=r:raise ValueError("parent_record_changed")
   if ann[sid]!=oldann[sid]:raise ValueError("parent_annotation_changed")
 if any(C.lineage_overlap(parts).values()):raise ValueError("partition_lineage_overlap")
def caps(parent,selected,owners=None):
 own=dict(owners)if owners is not None else C.owner_table(parent);rc=collections.Counter();rt=collections.Counter();touched=set()
 for p,rs in parent.items():
  for r in rs:
   g=r["split_group"].lower();rc[g]+=1;rt[g]+=r["valid_tokens"]
 for r in selected:
  p=r["partition"]
  for g in [r["split_group"]]+r.get("lineage_aliases",[]):
   g=g.lower()
   if g in own and own[g]!=p:raise ValueError("repository_cross_partition")
   own[g]=p
  g=r["split_group"].lower();rc[g]+=1;rt[g]+=r["valid_tokens"];touched.add(g)
 for g in touched:
  if g.startswith("repo:")and(rc[g]>2 or rt[g]>100000):raise ValueError("combined_repository_cap")
 return {g:dict(episodes=rc[g],tokens=rt[g])for g in sorted(touched)}
def admissible(r,selected,parent,owners=None):
 try:caps(parent,selected+[r],owners)
 except ValueError:return False
 if any(x["split_group"]==r["split_group"]or x["sample_id"]==r["sample_id"]for x in selected):return False
 for x in selected:
  if C.PUBLIC.similar(C.PUBLIC.features(S.objective(r)),C.PUBLIC.features(S.objective(x))):return False
  if x["partition"]!=r["partition"]and any(set(map(str.lower,r.get(k,[])))&set(map(str.lower,x.get(k,[])))for k in ["source_papers","lineage_aliases","dataset_groups"]):return False
 return True
def propose(output=None):
 dest=Path(output)if output else OUT
 review=B.read(OUT/"scope-review.json")["decisions"]
 pools=B.rows(OUT/"screened.jsonl")+B.rows(OUT/"extension-01/screened.jsonl")+B.rows(OUT/"extension-02/screened.jsonl")
 candidates=[r for r in pools if review.get(r["sample_id"]+":"+r["gap_category"],{}).get("decision")=="admit"]
 candidates=list({(r["sample_id"],r["gap_category"]):r for r in candidates}.values())
 b=B.base();owners=C.owner_table(b);selected=[];gaps=[]
 # Scarce numerical and research categories first, before broader API categories.
 for cat in ["mesh_numerical","battery_research","molecular_research","database_migration","parser_sanitization","api_context"]:
  n=B.policy()["categories"][cat]["target_per_partition"]
  for i in range(n):
   pool={p:[r for r in candidates if r["partition"]==p and r["gap_category"]==cat and admissible(r,selected,b,owners)and(cat!="parser_sanitization"or i!=0 or"html_markup"in review[r["sample_id"]+":"+cat].get("capabilities",[]))and(cat!="parser_sanitization"or i==0 or"parser_semantics"in review[r["sample_id"]+":"+cat].get("capabilities",[]))and(cat!="mesh_numerical"or i!=0 or"mesh_discretization"in review[r["sample_id"]+":"+cat].get("capabilities",[]))and(cat!="molecular_research"or i!=0 or p!="primary"or"molecular_simulation"in review[r["sample_id"]+":"+cat].get("capabilities",[]))]for p in P}
   pairs=[]
   for a in pool["primary"]:
    for z in pool["repeat"]:
     if C.bucket(a["valid_tokens"])!=C.bucket(z["valid_tokens"])or max(a["valid_tokens"],z["valid_tokens"])/min(a["valid_tokens"],z["valid_tokens"])>1.5:continue
     if not admissible(z,selected+[a],b,owners):continue
     h=C.sha([B.policy()["seed"],cat,i,a["source_record_sha256"],z["source_record_sha256"]])
     pairs.append((h,a,z))
   if not pairs:gaps.append(dict(category=cat,pairs_filled=i,target=n));break
   _,a,z=min(pairs,key=lambda x:x[0]);selected.extend([a,z])
 B.emit(dest/"proposed.jsonl",selected)
 C.save(dest/"proposal.json",dict(status="requires_final_review"if not gaps else"incomplete",unfilled=gaps,records=len(selected),tokens=sum(r["valid_tokens"]for r in selected),score_inputs_read=False,policy_sha256=C.file_sha(OUT/"policy.json"),scope_review_sha256=C.file_sha(OUT/"scope-review.json")))
 C.save(dest/"review-queue.json",[dict(**S.brief(r),category=r["gap_category"])for r in selected])
 print(C.canonical(B.read(dest/"proposal.json")),flush=True)
 return selected
def build():
 assert not DEST.exists()
 selected=B.rows(OUT/"proposed.jsonl");assert not B.read(OUT/"proposal.json")["unfilled"]
 review=B.read(OUT/"content-review.json");assert review["proposed_sha256"]==C.file_sha(OUT/"proposed.jsonl")
 decisions={d["sample_id"]:d for d in review["records"]};assert len(decisions)==len(selected)
 parent=B.base();parts=copy.deepcopy(parent);oldann=B.read(ROOT/"calibration-mix-v6/annotations.json");ann=copy.deepcopy(oldann)
 for r in selected:
  d=decisions[r["sample_id"]];assert d["decision"]=="admit"and d["source_record_sha256"]==r["source_record_sha256"]and d["gap_category"]==r["gap_category"]
  K.record(r);new=copy.deepcopy(r)
  new.update(calibration_eligible=True,calibration_scope="diagnostic_only",admission=dict(decision="admit_public_gap_workflow",policy_sha256=C.file_sha(OUT/"policy.json"),content_review_sha256=C.file_sha(OUT/"content-review.json"),source_correctness="upstream_attributed_not_independently_reexecuted"))
  parts[r["partition"]].append(new)
  ann[r["sample_id"]]=dict(partition=r["partition"],origin="public_tool_gap_fill_v1",domain=r["domain"],families=[r["workflow_family"]],fine_grained_family=r["gap_category"],length_bucket=C.bucket(r["valid_tokens"]))
 assert_preserved(parent,parts,ann,oldann);cap=caps(parent,selected)
 DEST.mkdir();B.emit(DEST/"all-selected.jsonl",[dict(partition=p,record=r)for p in P for r in parts[p]])
 B.emit(DEST/"selected-additions.jsonl",selected);C.save(DEST/"annotations.json",ann)
 for n in ["policy.json","content-review.json","protected-before.json"]:C.save(DEST/n,B.read(OUT/n))
 source_files={m["path"]:m for path in [OUT/"sources.lock.json",OUT/"extension-01/sources.lock.json",OUT/"extension-02/sources.lock.json"]for m in B.read(path)["files"]}
 C.save(DEST/"sources.lock.json",dict(files=list(source_files.values())))
 pins={}
 for folder,names in [(ROOT/"calibration-mix-v6",["all-selected.jsonl","annotations.json","completion.json","coverage.json","verification.json"]),(OUT,["build.py","release.py","test_invariants.py","policy.json","owners.json","raw-candidates.jsonl","normalized.jsonl","screened.jsonl","screening.json","scope-review.json","selection-design.json","content-review.json","proposed.jsonl","proposal.json"])]:
  for n in names:pins[str((folder/n).relative_to(ROOT))]=C.file_sha(folder/n)
 for m in source_files.values():pins[m["path"]]=m["sha256"]
 for extension in ["extension-01","extension-02"]:
  for name in ["build.py","policy.json","raw-candidates.jsonl","normalized.jsonl","screened.jsonl","screening.json","sources.lock.json"]:
   f=OUT/extension/name;pins[str(f.relative_to(ROOT))]=C.file_sha(f)
 pins[str((OUT/"expand_numeric.py").relative_to(ROOT))]=C.file_sha(OUT/"expand_numeric.py")
 pins[str((OUT/"extend.py").relative_to(ROOT))]=C.file_sha(OUT/"extend.py")
 for f in [C.MODEL/"config.json",C.MODEL/"model.safetensors.index.json"]+[p for p in set(C.MODEL.glob("*token*"))|set(C.MODEL.glob("*template*"))if p.is_file()]:pins[str(f)]=C.file_sha(f)
 C.save(DEST/"inputs.lock.json",pins);C.save(DEST/"build-report.json",dict(parent_records=589,preserved_records=589,additions=len(selected),partition_episodes={p:len(rs)for p,rs in parts.items()},partition_tokens={p:sum(r["valid_tokens"]for r in rs)for p,rs in parts.items()},repository_caps=cap,all_selected_sha256=C.file_sha(DEST/"all-selected.jsonl")))
 B.log("built",**B.read(DEST/"build-report.json"))
def verify():
 assert not(DEST/"verification.json").exists()
 pins=B.read(DEST/"inputs.lock.json")
 def checkpins():
  for n,h in pins.items():
   p=Path(n);assert C.file_sha(p if p.is_absolute()else ROOT/p)==h,n
 checkpins()
 parent=B.base();objects=B.rows(DEST/"all-selected.jsonl");parts={p:[x["record"]for x in objects if x["partition"]==p]for p in P}
 ann=B.read(DEST/"annotations.json");assert_preserved(parent,parts,ann,B.read(ROOT/"calibration-mix-v6/annotations.json"))
 adds=B.rows(DEST/"selected-additions.jsonl");cap=caps(parent,adds)
 own=B.read(OUT/"owners.json")
 for r in adds:assert B.I.partition(r["split_group"],own)==r["partition"];K.record(r)
 W.DEST=DEST
 # Reuse the established independent raw-row verifier, not cached normalization claims.
 source_count=W.sourceverify(adds)
 # Recompute selection in a new workspace path, compare bytes.
 replay=OUT/"replay";replay.mkdir(exist_ok=False);propose(replay)
 assert C.file_sha(replay/"proposed.jsonl")==C.file_sha(OUT/"proposed.jsonl")
 expected=copy.deepcopy(parent)
 for r in adds:
  actual=next(x for x in parts[r["partition"]]if x["sample_id"]==r["sample_id"])
  test=copy.deepcopy(r);test.update(calibration_eligible=True,calibration_scope="diagnostic_only",admission=dict(decision="admit_public_gap_workflow",policy_sha256=C.file_sha(OUT/"policy.json"),content_review_sha256=C.file_sha(OUT/"content-review.json"),source_correctness="upstream_attributed_not_independently_reexecuted"))
  assert actual==test;expected[r["partition"]].append(test)
 assert parts==expected
 reconstructed=[dict(partition=p,record=r)for p in P for r in expected[p]]
 B.emit(replay/"all-selected.jsonl",reconstructed)
 assert C.file_sha(replay/"all-selected.jsonl")==C.file_sha(DEST/"all-selected.jsonl")
 # Independently repeat leakage screens against all preserved examples, held-out tasks and new pairs.
 sp,_,held,reserved,question,forbidden,grams=B.overlap_inputs()
 old=[r for rs in parent.values()for r in rs]
 features=[C.PUBLIC.features(S.objective(r))for r in old]+[C.PUBLIC.features(question(r))for r in held]
 for i,r in enumerate(adds):
  assert r["split_group"].lower()not in reserved
  f=C.PUBLIC.features(S.objective(r));assert all(not C.PUBLIC.similar(f,x)for x in features)
  assert not grams(B.visible(r))&forbidden
  assert all(not C.PUBLIC.similar(f,C.PUBLIC.features(S.objective(x)))for x in adds[:i])
 for n,h in sp.items():
  pp=Path(n);assert C.file_sha(pp if pp.is_absolute()else ROOT/pp)==h
 allrows=[r for p in P for r in parts[p]]
 with concurrent.futures.ProcessPoolExecutor(max_workers=4,initializer=V.init)as ex:
  for i,_ in enumerate(ex.map(V.check_token,allrows,chunksize=4),1):
   if i%100==0:B.log("verified_exact_tokens",completed=i,total=len(allrows))
 test=B.read(OUT/"tests.json");assert test["status"]=="passed"and test["test_sha256"]==C.file_sha(OUT/"test_invariants.py")
 for n,h in B.read(OUT/"protected-before.json").items():assert C.file_sha(ROOT/n)==h,n
 checkpins()
 cov=copy.deepcopy(B.read(ROOT/"calibration-mix-v6/coverage.json"));cov["parent_coverage_preserved"]=True
 cov["fine_grained_additions"]={p:{cat:dict(episodes=len(rr),tokens=sum(r["valid_tokens"]for r in rr),repositories=sorted({r["split_group"]for r in rr}))for cat in B.policy()["categories"]for rr in [[r for r in adds if r["partition"]==p and r["gap_category"]==cat]]}for p in P}
 cov["public_tool_families"]={p:{fam:dict(episodes=len(rr),tokens=sum(r["valid_tokens"]for r in rr),distinct_groups=len({r["split_group"]for r in rr}),sources=dict(collections.Counter(r.get("dataset","unknown")for r in rr)))for fam in B.I.P["families"]for rr in [[r for r in parts[p]if r["domain"]=="general_tool_use"and r.get("workflow_family")==fam]]}for p in P}
 cov["partitions"]={p:{d:dict(episodes=len(rr),tokens=sum(r["valid_tokens"]for r in rr))for d in C.DOMAINS for rr in [[r for r in parts[p]if r["domain"]==d]]}for p in P}
 C.save(DEST/"coverage.json",cov);C.save(DEST/"overlap-audit.json",dict(status="passed",partition_overlap=C.lineage_overlap(parts),repository_caps=cap,full_visible_heldout_screen=True,all_existing_examples_screened=True))
 C.save(DEST/"verification.json",dict(status="passed",preserved_records_exactly=589,preserved_annotations=True,retokenized=len(allrows),raw_source_rows_verified=source_count,deterministic_selection_replay=True,byte_identical_reconstruction=True,protected_files_unchanged=True,all_selected_sha256=C.file_sha(DEST/"all-selected.jsonl"),coverage_requirements_passed=cov["coverage_requirements_passed"],remaining_gaps=cov["preserved_non_tool_coverage_gaps"],scoring_launched=False,native_precision_verified=False,selection_allowed=False))
 B.log("verification_complete",records=len(allrows),additions=len(adds))
def assemble():
 v=B.read(DEST/"verification.json");assert v["status"]=="passed"
 objects=B.rows(DEST/"all-selected.jsonl");old=B.read(ROOT/"calibration-mix-v6/completion.json")["plans"];plans={};artifacts={}
 for p in P:
  rr=[x["record"]for x in objects if x["partition"]==p];dest=ROOT/"data"/("observer-manifests-representative-v7-"+p);dest.mkdir(exist_ok=False);groups=[];counts={}
  for d in C.DOMAINS:
   dr=[r for r in rr if r["domain"]==d];buf=[];n=0;ix=0
   for r in dr:
    if buf and n+r["valid_tokens"]>65536:groups.append(C.write(dest/f"{d}-{ix:02d}.json",buf,d,DEST/"overlap-audit.json"));buf=[];n=0;ix+=1
    buf.append(r);n+=r["valid_tokens"]
   if buf:groups.append(C.write(dest/f"{d}-{ix:02d}.json",buf,d,DEST/"overlap-audit.json"))
   counts[d]=dict(sequences=len(dr),tokens=sum(r["valid_tokens"]for r in dr))
  plan=dict(schema="mimo_observation_execution_plan_v1",partition=p,domain_weights=B.read(ROOT/old[p])["domain_weights"],groups=groups,coverage=counts,maximum_sequence_tokens=max(r["valid_tokens"]for r in rr),group_token_limit=65536,production_calibration_ready=False,diagnostic_coverage_ready=v["coverage_requirements_passed"],tool_diagnostic_coverage_ready=True,coverage_requirements_passed=v["coverage_requirements_passed"],native_precision_verified=False,selection_allowed=False,scoring_launched=False,purpose="Add fine-grained public tool workflows, preserve all 589 v6 records and original partition ownership",source_verification_sha256=C.file_sha(DEST/"verification.json"),workflow_annotations=str((DEST/"annotations.json").relative_to(ROOT)),pending=["Fresh scoring required; do not resume old checkpoint with changed data","Independent rank stability","Inherited non-tool coverage gaps","Native precision and objective quality gates"])
  C.save(dest/"plan.json",plan);plans[p]=str((dest/"plan.json").relative_to(ROOT))
  emitted=[r for g in groups for r in B.read(ROOT/g["manifest"])["samples"]]
  assert {r["sample_id"]:r for r in emitted}=={r["sample_id"]:r for r in rr}
  for f in dest.glob("*.json"):artifacts[str(f.relative_to(ROOT))]=C.file_sha(f)
 C.save(DEST/"completion.json",dict(status="diagnostic_mix_assembled",plans=plans,artifacts_sha256=artifacts,remaining_gaps=v["remaining_gaps"],coverage_requirements_passed=v["coverage_requirements_passed"],scoring_launched=False,selection_allowed=False,native_precision_verified=False,active_pointers_changed=False))
 files={str(f.relative_to(ROOT)):C.file_sha(f)for f in DEST.glob("*")if f.is_file()}
 C.save(DEST/"evidence.lock.json",dict(files=files))
 B.log("assembled",plans=plans)
if __name__=="__main__":globals()[sys.argv[1]]()

