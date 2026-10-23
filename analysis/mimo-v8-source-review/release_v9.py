"""Separate additive v9 release, reusing pinned source validators, never scores."""
import collections
import concurrent.futures
import copy
import importlib.util
import json
from pathlib import Path
import sys

ROOT=Path('/mnt/nvme4/mimo-reap');OUT=ROOT/'public-workflow-gap-fill-v3';DEST=ROOT/'calibration-mix-v9';PARENT=ROOT/'calibration-mix-v8'
spec=importlib.util.spec_from_file_location('v9prepare',OUT/'prepare.py');A=importlib.util.module_from_spec(spec);sys.modules[spec.name]=A;spec.loader.exec_module(A)
spec=importlib.util.spec_from_file_location('v9release_helpers',ROOT/'public-workflow-gap-fill-v2/release.py');R=importlib.util.module_from_spec(spec);spec.loader.exec_module(R)
B=A.B;C=A.C;P=A.A.P
R.A=A.A;R.B=B;R.C=C;R.P=P;R.OUT=OUT;R.DEST=DEST


def propose():
    return R.propose(OUT)


def construct():
    par=A.parent();adds=A.A.rows(OUT/'proposed.jsonl');R.check_additions(adds,par)
    review=B.read(OUT/'content-review.json')
    assert review['proposed_sha256']==C.file_sha(OUT/'proposed.jsonl')
    decisions={r['sample_id']:r for r in review['records']};assert len(decisions)==len(adds)
    parts=copy.deepcopy(par);ann=B.read(PARENT/'annotations.json')
    for r in adds:
        d=decisions[r['sample_id']];assert d['decision']=='admit' and d['source_record_sha256']==r['source_record_sha256']
        new=copy.deepcopy(r);new.update(domain=r['target_domain'],calibration_eligible=True,calibration_scope='diagnostic_only',admission=dict(decision='admit_public_layer32_gap_workflow',content_review_sha256=C.file_sha(OUT/'content-review.json'),policy_sha256=C.file_sha(OUT/'policy.json'),source_correctness='upstream_attributed_not_independently_reexecuted'))
        parts[r['partition']].append(new)
        ann[r['sample_id']]=dict(partition=r['partition'],origin='public_workflow_gap_fill_v3',domain=r['target_domain'],families=[r['gap_category']],fine_grained_family=r['gap_category'],length_bucket=C.bucket(r['valid_tokens']))
    return parts,ann,adds


def build():
    assert not DEST.exists()
    assert not B.read(OUT/'proposal.json')['unfilled']
    parts,ann,adds=construct();DEST.mkdir()
    B.emit(DEST/'all-selected.jsonl',[dict(partition=p,record=r) for p in P for r in parts[p]])
    B.emit(DEST/'selected-additions.jsonl',adds);C.save(DEST/'annotations.json',ann)
    for name in ('policy.json','content-review.json','protected-before.json','sources.lock.json'):
        C.save(DEST/name,B.read(OUT/name))
    pins=dict(B.read(OUT/'protected-before.json'))
    for name in ('prepare.py','release.py','policy.json','scope-review.json','content-review.json','screened.jsonl','proposed.jsonl','proposal.json','owners.json','sources.lock.json'):
        pins[str((OUT/name).relative_to(ROOT))]=C.file_sha(OUT/name)
    for meta in B.read(OUT/'sources.lock.json')['files']:
        if meta['path'] in {r['source_file'] for r in adds}:pins[meta['path']]=meta['sha256']
    C.save(DEST/'inputs.lock.json',pins)
    C.save(DEST/'build-report.json',dict(parent_records=667,preserved_records=667,additions=len(adds),partition_episodes={p:len(rows) for p,rows in parts.items()},partition_tokens={p:sum(r['valid_tokens'] for r in rows) for p,rows in parts.items()},categories=dict(collections.Counter(r['gap_category'] for r in adds)),all_selected_sha256=C.file_sha(DEST/'all-selected.jsonl')))


def verify():
    assert not (DEST/'verification.json').exists()
    pins=B.read(DEST/'inputs.lock.json')
    for name,h in pins.items():assert C.file_sha(ROOT/name)==h,name
    parts,ann,adds=construct()
    actual=A.A.rows(DEST/'all-selected.jsonl')
    assert actual==[dict(partition=p,record=r) for p in P for r in parts[p]]
    assert B.read(DEST/'annotations.json')==ann
    old=A.parent();original={r['sample_id']:r for rows in old.values() for r in rows}
    assert all(r==original[r['sample_id']] for obj in actual for r in [obj['record']] if r['sample_id'] in original)
    oldann=B.read(PARENT/'annotations.json');assert all(ann[sid]==v for sid,v in oldann.items())
    assert B.read(OUT/'owners.json')==C.owner_table(old)
    replay=OUT/'replay';assert not replay.exists();R.propose(replay)
    assert C.file_sha(replay/'proposed.jsonl')==C.file_sha(OUT/'proposed.jsonl')
    B.emit(replay/'all-selected.jsonl',actual)
    assert C.file_sha(replay/'all-selected.jsonl')==C.file_sha(DEST/'all-selected.jsonl')
    B.W.DEST=DEST;B.W.OUT=OUT;sources=B.W.sourceverify(adds)
    overlap_pins,_,held,reserved,question,forbidden,grams=B.overlap_inputs()
    existing=[r for rows in old.values() for r in rows]
    features=[C.PUBLIC.features(B.S.objective(r)) for r in existing]+[C.PUBLIC.features(question(r)) for r in held]
    for i,r in enumerate(adds):
        assert r['split_group'].lower() not in reserved
        f=C.PUBLIC.features(B.S.objective(r));assert all(not C.PUBLIC.similar(f,x) for x in features)
        assert not grams(B.visible(r))&forbidden
        assert all(not C.PUBLIC.similar(f,C.PUBLIC.features(B.S.objective(other))) for other in adds[:i])
    for name,h in overlap_pins.items():
        pp=Path(name);assert C.file_sha(pp if pp.is_absolute() else ROOT/pp)==h
    B.K.partition_overlap(parts)
    rows=[r for p in P for r in parts[p]]
    with concurrent.futures.ProcessPoolExecutor(max_workers=4,initializer=B.V.init) as pool:
        for _ in pool.map(B.V.check_token,rows,chunksize=4):pass
    for name,h in pins.items():assert C.file_sha(ROOT/name)==h,name
    C.save(DEST/'coverage.json',dict(parent_coverage_preserved=True,parent_coverage_sha256=C.file_sha(PARENT/'coverage.json'),coverage_requirements_passed=False,remaining_gaps=B.read(PARENT/'final-audit.json').get('coverage_scope_pending',[]),partitions={p:{d:dict(episodes=len(rr),tokens=sum(r['valid_tokens'] for r in rr)) for d in C.DOMAINS for rr in [[r for r in parts[p] if r['domain']==d]]} for p in P}))
    C.save(DEST/'overlap-audit.json',dict(status='passed',partition_overlap=C.lineage_overlap(parts),full_visible_heldout_screen=True,parent_preserved=True))
    C.save(DEST/'verification.json',dict(status='passed',preserved_records_exactly=667,preserved_annotations=True,retokenized=len(rows),raw_source_rows_verified=sources,deterministic_selection_replay=True,byte_identical_reconstruction=True,repository_and_lineage_separation=True,all_selected_sha256=C.file_sha(DEST/'all-selected.jsonl'),coverage_requirements_passed=False,scoring_launched=False,native_precision_verified=False,selection_allowed=False))


def assemble():
    verified=B.read(DEST/'verification.json');assert verified['status']=='passed'
    objects=A.A.rows(DEST/'all-selected.jsonl');plans={};artifacts={}
    for part in P:
        rows=[x['record'] for x in objects if x['partition']==part]
        folder=ROOT/'data'/('observer-manifests-representative-v9-'+part);folder.mkdir(exist_ok=False);groups=[];coverage={}
        for domain in C.DOMAINS:
            dr=[r for r in rows if r['domain']==domain];buffer=[];tokens=0;index=0
            for record in dr:
                if buffer and tokens+record['valid_tokens']>65536:
                    groups.append(C.write(folder/f'{domain}-{index:02d}.json',buffer,domain,DEST/'overlap-audit.json'));buffer=[];tokens=0;index+=1
                buffer.append(record);tokens+=record['valid_tokens']
            if buffer:groups.append(C.write(folder/f'{domain}-{index:02d}.json',buffer,domain,DEST/'overlap-audit.json'))
            coverage[domain]=dict(sequences=len(dr),tokens=sum(r['valid_tokens'] for r in dr))
        parent_plan=B.read(ROOT/B.read(PARENT/'completion.json')['plans'][part])
        plan=dict(schema='mimo_observation_execution_plan_v1',partition=part,domain_weights=parent_plan['domain_weights'],groups=groups,coverage=coverage,maximum_sequence_tokens=max(r['valid_tokens'] for r in rows),group_token_limit=65536,production_calibration_ready=False,diagnostic_coverage_ready=False,coverage_requirements_passed=False,native_precision_verified=False,selection_allowed=False,scoring_launched=False,purpose='Add content-reviewed public finance and tool workflows; preserve all 667 v8 histories and annotations.',source_verification_sha256=C.file_sha(DEST/'verification.json'),workflow_annotations=str((DEST/'annotations.json').relative_to(ROOT)),pending=['Fresh versioned incremental scoring for changed histories','Independent calibration/rank gates','Inherited coverage, native precision and objective-quality gates'])
        C.save(folder/'plan.json',plan);plans[part]=str((folder/'plan.json').relative_to(ROOT))
        emitted=[r for group in groups for r in B.read(ROOT/group['manifest'])['samples']]
        assert {r['sample_id']:r for r in emitted}=={r['sample_id']:r for r in rows}
        for path in folder.glob('*.json'):artifacts[str(path.relative_to(ROOT))]=C.file_sha(path)
    C.save(DEST/'completion.json',dict(status='diagnostic_mix_assembled',plans=plans,artifacts_sha256=artifacts,scoring_launched=False,native_precision_verified=False,selection_allowed=False))
    runtime_paths=[OUT/'prepare.py',OUT/'release.py',ROOT/'public-workflow-gap-fill-v2/prepare.py',ROOT/'public-workflow-gap-fill-v2/release.py',ROOT/'public-tool-gap-fill-v1/build.py']
    C.save(DEST/'release-runtime.lock.json',dict(files={str(p.relative_to(ROOT)):C.file_sha(p) for p in runtime_paths}))
    files={str(p.relative_to(ROOT)):C.file_sha(p) for p in DEST.iterdir() if p.is_file()}
    files.update(artifacts)
    C.save(DEST/'evidence.lock.json',dict(files=files))
    C.save(DEST/'release-seal.json',dict(status='sealed_diagnostic_revision',evidence_lock_sha256=C.file_sha(DEST/'evidence.lock.json'),plans=plans,scoring_launched=False,selection_allowed=False))


if __name__=='__main__':
    {'propose':propose,'build':build,'verify':verify,'assemble':assemble}[sys.argv[1]]()
