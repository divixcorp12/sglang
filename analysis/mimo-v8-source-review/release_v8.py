"""Review-bound additive release and independent validation for v8."""
import collections
import concurrent.futures
import copy
import importlib.util
import json
import sys
from pathlib import Path

ROOT=Path('/mnt/nvme4/mimo-reap')
OUT=ROOT/'public-workflow-gap-fill-v2'
DEST=ROOT/'calibration-mix-v8'
spec=importlib.util.spec_from_file_location('v8preparation',OUT/'prepare.py')
A=importlib.util.module_from_spec(spec);sys.modules[spec.name]=A;spec.loader.exec_module(A)
B=A.B;C=A.C;P=A.P

def check_additions(adds, par):
    owners=B.read(OUT/'owners.json'); counts=collections.Counter();tokens=collections.Counter();used=set()
    source_ids={r.get('source_record_sha256') for rs in par.values() for r in rs}
    for rs in par.values():
        for r in rs:counts[r['split_group'].lower()]+=1;tokens[r['split_group'].lower()]+=r['valid_tokens']
    parts={p:list(rs) for p,rs in par.items()}
    for r in adds:
        group=r['split_group'].lower();part=r['partition']
        if B.I.partition(group,owners)!=part:raise ValueError('partition ownership')
        if group in used:raise ValueError('one addition per group')
        if r['source_record_sha256'] in source_ids:raise ValueError('duplicate source history')
        if group.startswith('repo:') and (counts[group]+1>2 or tokens[group]+r['valid_tokens']>100000):raise ValueError('repository cap')
        used.add(group);source_ids.add(r['source_record_sha256']);parts[part].append(r)
    B.K.partition_overlap(parts)
    return parts

def propose(destination=OUT):
    destination.mkdir(exist_ok=True)
    review=B.read(OUT/'scope-review.json');assert review['screened_sha256']==C.file_sha(OUT/'screened.jsonl')
    approved=set(review['approved_candidates']);par=A.parent();pool=A.rows(OUT/'screened.jsonl')
    selected=[];pairs=[];gaps=[];pol=A.policy()
    for cat in pol['categories']:
        candidates={p:sorted([r for r in pool if r['partition']==p and r['gap_category']==cat and r['sample_id']+':'+cat in approved],key=lambda x:x['metadata_priority'])for p in P}
        for n in range(pol['categories'][cat].get('target_per_partition', pol['target_per_partition_category'])):
            options=[]
            for a in candidates['primary']:
                for b in candidates['repeat']:
                    if C.bucket(a['valid_tokens'])!=C.bucket(b['valid_tokens']) or max(a['valid_tokens'],b['valid_tokens'])/min(a['valid_tokens'],b['valid_tokens'])>1.5:continue
                    try:check_additions(selected+[a,b],par)
                    except ValueError:continue
                    options.append((C.sha([pol['seed'],cat,n,a['source_record_sha256'],b['source_record_sha256']]),a,b))
            if not options:gaps.append(dict(category=cat,pair=n));break
            _,a,b=min(options,key=lambda x:x[0]);selected.extend([a,b]);pairs.append(dict(category=cat,primary=a['sample_id'],repeat=b['sample_id'],tokens=[a['valid_tokens'],b['valid_tokens']]))
    B.emit(destination/'proposed.jsonl',selected);C.save(destination/'proposal.json',dict(pairs=pairs,unfilled=gaps,additions=len(selected),score_inputs_read=False))
    return selected

def build():
    assert not DEST.exists()
    adds=A.rows(OUT/'proposed.jsonl');assert not B.read(OUT/'proposal.json')['unfilled']
    review=B.read(OUT/'content-review.json');assert review['proposed_sha256']==C.file_sha(OUT/'proposed.jsonl')
    decisions={r['sample_id']:r for r in review['records']};assert len(decisions)==len(adds)
    par=A.parent();check_additions(adds,par);parts=copy.deepcopy(par);ann=B.read(ROOT/'calibration-mix-v7/annotations.json')
    for r in adds:
        decision=decisions[r['sample_id']];assert decision['decision']=='admit' and decision['source_record_sha256']==r['source_record_sha256']
        new=copy.deepcopy(r);new.update(domain=r['target_domain'],calibration_eligible=True,calibration_scope='diagnostic_only',admission=dict(decision='admit_public_full_model_gap_workflow',content_review_sha256=C.file_sha(OUT/'content-review.json'),policy_sha256=C.file_sha(OUT/'policy.json'),source_correctness='upstream_attributed_not_independently_reexecuted'))
        parts[r['partition']].append(new)
        ann[r['sample_id']]=dict(partition=r['partition'],origin='public_workflow_gap_fill_v2',domain=r['target_domain'],families=[r['gap_category']],fine_grained_family=r['gap_category'],length_bucket=C.bucket(r['valid_tokens']))
    DEST.mkdir();B.emit(DEST/'all-selected.jsonl',[dict(partition=p,record=r)for p in P for r in parts[p]])
    B.emit(DEST/'selected-additions.jsonl',adds);C.save(DEST/'annotations.json',ann)
    for name in ('policy.json','content-review.json','protected-before.json','sources.lock.json'):C.save(DEST/name,B.read(OUT/name))
    C.save(DEST/'build-report.json',dict(parent_records=sum(map(len,par.values())),preserved_records=sum(map(len,par.values())),additions=len(adds),partition_episodes={p:len(rs) for p,rs in parts.items()},partition_tokens={p:sum(r['valid_tokens']for r in rs)for p,rs in parts.items()},categories=collections.Counter(r['gap_category'] for r in adds),all_selected_sha256=C.file_sha(DEST/'all-selected.jsonl')))
    protected=B.read(OUT/'protected-before.json');pins=dict(protected)
    for name in ('prepare.py','release.py','policy.json','scope-review.json','content-review.json','screened.jsonl','proposed.jsonl','proposal.json','owners.json','sources.lock.json'):
        pins[str((OUT/name).relative_to(ROOT))]=C.file_sha(OUT/name)
    for m in B.read(OUT/'sources.lock.json')['files']:
        if m['path'] in {r['source_file']for r in adds}:pins[m['path']]=m['sha256']
    C.save(DEST/'inputs.lock.json',pins)
    A.log('built',**B.read(DEST/'build-report.json'))

def verify():
    assert not (DEST/'verification.json').exists()
    pins=B.read(DEST/'inputs.lock.json')
    def verify_pins():
        for n,h in pins.items():assert C.file_sha(ROOT/n)==h,n
    verify_pins();par=A.parent();adds=A.rows(DEST/'selected-additions.jsonl');check_additions(adds,par)
    assert B.read(OUT/'owners.json')==C.owner_table(par)
    objects=A.rows(DEST/'all-selected.jsonl');parts={p:[x['record']for x in objects if x['partition']==p]for p in P}
    oldann=B.read(ROOT/'calibration-mix-v7/annotations.json');ann=B.read(DEST/'annotations.json')
    for part in P:
        original={r['sample_id']:r for r in par[part]};actual={r['sample_id']:r for r in parts[part]}
        assert all(actual[sid]==r for sid,r in original.items())
    assert all(ann[sid]==v for sid,v in oldann.items())
    replay=OUT/'replay';assert not replay.exists();propose(replay)
    assert C.file_sha(replay/'proposed.jsonl')==C.file_sha(OUT/'proposed.jsonl')
    expected=copy.deepcopy(par)
    for r in adds:
        new=copy.deepcopy(r);new.update(domain=r['target_domain'],calibration_eligible=True,calibration_scope='diagnostic_only',admission=dict(decision='admit_public_full_model_gap_workflow',content_review_sha256=C.file_sha(OUT/'content-review.json'),policy_sha256=C.file_sha(OUT/'policy.json'),source_correctness='upstream_attributed_not_independently_reexecuted'))
        expected[r['partition']].append(new)
    assert expected==parts
    B.emit(replay/'all-selected.jsonl',[dict(partition=p,record=r)for p in P for r in expected[p]])
    assert C.file_sha(replay/'all-selected.jsonl')==C.file_sha(DEST/'all-selected.jsonl')
    B.W.DEST=DEST;B.W.OUT=OUT
    sources=B.W.sourceverify(adds)
    overlap_pins,_,held,reserved,question,forbidden,grams=B.overlap_inputs()
    old=[r for rs in par.values()for r in rs]
    features=[C.PUBLIC.features(B.S.objective(r))for r in old]+[C.PUBLIC.features(question(r))for r in held]
    for i,r in enumerate(adds):
        assert r['split_group'].lower()not in reserved
        f=C.PUBLIC.features(B.S.objective(r));assert all(not C.PUBLIC.similar(f,x)for x in features)
        assert not grams(B.visible(r))&forbidden
        assert all(not C.PUBLIC.similar(f,C.PUBLIC.features(B.S.objective(x)))for x in adds[:i])
    for n,h in overlap_pins.items():
        pp=Path(n);assert C.file_sha(pp if pp.is_absolute()else ROOT/pp)==h
    B.K.partition_overlap(parts)
    allrows=[r for p in P for r in parts[p]]
    with concurrent.futures.ProcessPoolExecutor(max_workers=4,initializer=B.V.init)as ex:
        for i,_ in enumerate(ex.map(B.V.check_token,allrows,chunksize=4),1):
            if i%100==0:A.log('retokenized',completed=i,total=len(allrows))
    verify_pins()
    cov=copy.deepcopy(B.read(ROOT/'calibration-mix-v7/coverage.json'))
    cov['parent_coverage_preserved']=True
    cov['full_model_additions']={p:{cat:dict(episodes=len(rr),tokens=sum(r['valid_tokens']for r in rr))for cat in A.policy()['categories']for rr in [[r for r in adds if r['partition']==p and r['gap_category']==cat]]}for p in P}
    cov['partitions']={p:{d:dict(episodes=len(rr),tokens=sum(r['valid_tokens']for r in rr))for d in C.DOMAINS for rr in [[r for r in parts[p]if r['domain']==d]]}for p in P}
    C.save(DEST/'coverage.json',cov)
    C.save(DEST/'overlap-audit.json',dict(status='passed',partition_overlap=C.lineage_overlap(parts),full_visible_heldout_screen=True,parent_preserved=True))
    C.save(DEST/'verification.json',dict(status='passed',preserved_records_exactly=sum(map(len,par.values())),preserved_annotations=True,retokenized=len(allrows),raw_source_rows_verified=sources,deterministic_selection_replay=True,byte_identical_reconstruction=True,protected_files_unchanged=True,all_selected_sha256=C.file_sha(DEST/'all-selected.jsonl'),coverage_requirements_passed=False,remaining_gaps=cov.get('preserved_non_tool_coverage_gaps',[]),scoring_launched=False,native_precision_verified=False,selection_allowed=False))
    A.log('verification_complete',records=len(allrows),additions=len(adds))

def assemble():
    v=B.read(DEST/'verification.json');assert v['status']=='passed'
    objects=A.rows(DEST/'all-selected.jsonl');old=B.read(ROOT/'calibration-mix-v7/completion.json')['plans'];plans={};artifacts={}
    for part in P:
        rr=[x['record']for x in objects if x['partition']==part];dest=ROOT/'data'/('observer-manifests-representative-v8-'+part);dest.mkdir(exist_ok=False);groups=[];counts={}
        for domain in C.DOMAINS:
            dr=[r for r in rr if r['domain']==domain];buf=[];n=0;ix=0
            for r in dr:
                if buf and n+r['valid_tokens']>65536:groups.append(C.write(dest/f'{domain}-{ix:02d}.json',buf,domain,DEST/'overlap-audit.json'));buf=[];n=0;ix+=1
                buf.append(r);n+=r['valid_tokens']
            if buf:groups.append(C.write(dest/f'{domain}-{ix:02d}.json',buf,domain,DEST/'overlap-audit.json'))
            counts[domain]=dict(sequences=len(dr),tokens=sum(r['valid_tokens']for r in dr))
        plan=dict(schema='mimo_observation_execution_plan_v1',partition=part,domain_weights=B.read(ROOT/old[part])['domain_weights'],groups=groups,coverage=counts,maximum_sequence_tokens=max(r['valid_tokens']for r in rr),group_token_limit=65536,production_calibration_ready=False,diagnostic_coverage_ready=False,coverage_requirements_passed=False,native_precision_verified=False,selection_allowed=False,scoring_launched=False,purpose='Add public workflows across diagnosed full-model gaps; preserve all 621 v7 records.',source_verification_sha256=C.file_sha(DEST/'verification.json'),workflow_annotations=str((DEST/'annotations.json').relative_to(ROOT)),pending=['Fresh scoring from stage zero','Independent full-model ranking gate','Coverage/native precision/objective quality gates'])
        C.save(dest/'plan.json',plan);plans[part]=str((dest/'plan.json').relative_to(ROOT))
        emitted=[r for g in groups for r in B.read(ROOT/g['manifest'])['samples']]
        assert {r['sample_id']:r for r in emitted}=={r['sample_id']:r for r in rr}
        for f in dest.glob('*.json'):artifacts[str(f.relative_to(ROOT))]=C.file_sha(f)
    C.save(DEST/'completion.json',dict(status='diagnostic_mix_assembled',plans=plans,artifacts_sha256=artifacts,scoring_launched=False,selection_allowed=False,native_precision_verified=False,active_pointers_changed=False))
    C.save(DEST/'evidence.lock.json',dict(files={str(f.relative_to(ROOT)):C.file_sha(f)for f in DEST.iterdir()if f.is_file()}))
    A.log('assembled',plans=plans)

if __name__=='__main__':
    {'propose':propose,'build':build,'verify':verify,'assemble':assemble}[sys.argv[1]]()
