"""Independent completed-artifact audit and release seal; no GPU access."""
import collections
import hashlib
import json
from pathlib import Path
ROOT=Path('/mnt/nvme4/mimo-reap');OUT=ROOT/'public-workflow-gap-fill-v2';DEST=ROOT/'calibration-mix-v8'

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(p):return [json.loads(x)for x in p.open()]

def main():
    completion=read(DEST/'completion.json');assert completion['status']=='diagnostic_mix_assembled'
    protected=read(OUT/'protected-before.json')
    assert all(sha(ROOT/n)==h for n,h in protected.items())
    assert all(sha(ROOT/n)==h for n,h in read(DEST/'inputs.lock.json').items())
    original=rows(ROOT/'calibration-mix-v7/all-selected.jsonl');objects=rows(DEST/'all-selected.jsonl')
    old={x['record']['sample_id']:x for x in original};current={x['record']['sample_id']:x for x in objects}
    assert len(current)==len(objects)==667 and len(old)==621
    assert all(current[sid]==x for sid,x in old.items())
    oldann=read(ROOT/'calibration-mix-v7/annotations.json');ann=read(DEST/'annotations.json')
    assert all(ann[k]==v for k,v in oldann.items()) and set(ann)==set(current)
    review=read(OUT/'content-review.json');assert review['proposed_sha256']==sha(OUT/'proposed.jsonl')
    assert review['evidence_sha256']==sha(OUT/'final-review-evidence.json')
    decisions={r['sample_id']:r for r in review['records']};adds=rows(DEST/'selected-additions.jsonl')
    assert len(adds)==46 and {r['sample_id']for r in adds}==set(current)-set(old)
    for r in adds:assert decisions[r['sample_id']]['decision']=='admit' and decisions[r['sample_id']]['source_record_sha256']==r['source_record_sha256']
    pol=read(OUT/'policy.json');counts=collections.Counter((r['partition'],r['gap_category'])for r in adds)
    for p in ('primary','repeat'):
        for c,rule in pol['categories'].items():assert counts[p,c]==rule.get('target_per_partition',pol['target_per_partition_category'])
    for pair in read(OUT/'proposal.json')['pairs']:
        a,b=pair['tokens'];assert max(a,b)/min(a,b)<=1.5
        bucket=lambda n:0 if n<8192 else 1 if n<16384 else 2 if n<32768 else 3
        assert bucket(a)==bucket(b)
    capchecks={};repo_owner={}
    for x in objects:
        r=x['record'];g=r['split_group'].lower();p=x['partition']
        if g.startswith('repo:'):
            assert g not in repo_owner or repo_owner[g]==p;repo_owner[g]=p
            capchecks.setdefault(g,[]).append(r)
    for r in adds:
        g=r['split_group'].lower()
        if g.startswith('repo:'):assert len(capchecks[g])<=2 and sum(x['valid_tokens']for x in capchecks[g])<=100000
    for key in ('split_group','sample_id','task_key','prompt_sha256','token_sha256'):
        a={str(x['record'].get(key,'')).lower()for x in objects if x['partition']=='primary' and x['record'].get(key)}
        b={str(x['record'].get(key,'')).lower()for x in objects if x['partition']=='repeat' and x['record'].get(key)}
        assert not a&b,key
    for key in ('source_papers','dataset_groups','lineage_aliases'):
        sets=[{str(v).lower()for x in objects if x['partition']==p for v in x['record'].get(key,[])}for p in ('primary','repeat')]
        assert not sets[0]&sets[1],key
    manifests=0
    model=Path('/mnt/nvme4/mimi-v26-pro')
    config_sha=sha(model/'config.json');index_sha=sha(model/'model.safetensors.index.json')
    tokenizer_sha={n:sha(model/n)for n in ('chat_template.jinja','tokenizer.json','tokenizer_config.json')}
    parent_completion=read(ROOT/'calibration-mix-v7/completion.json')
    for p,name in completion['plans'].items():
        plan=read(ROOT/name);oldplan=read(ROOT/parent_completion['plans'][p]);assert plan['domain_weights']==oldplan['domain_weights']
        assert plan['native_precision_verified'] is False and plan['selection_allowed'] is False and plan['group_token_limit']==65536
        emitted=[]
        for group in plan['groups']:
            m=read(ROOT/group['manifest']);emitted.extend(m['samples']);manifests+=1
            assert m['source_model']==str(model) and m['config_sha256']==config_sha and m['index_sha256']==index_sha
            assert m['tokenizer_sha256']==tokenizer_sha and m['external_overlap_audit_sha256']==sha(DEST/'overlap-audit.json')
        assert {r['sample_id']:r for r in emitted}=={x['record']['sample_id']:x['record']for x in objects if x['partition']==p}
    assert all(sha(ROOT/n)==h for n,h in completion['artifacts_sha256'].items())
    assert read(OUT/'checks.json')['status']=='passed'
    assert read(DEST/'verification.json')['retokenized']==667
    report=dict(status='passed',parent_records_preserved=621,parent_annotations_preserved=True,additions=46,exact_retokenized=667,all_selected_sha256=sha(DEST/'all-selected.jsonl'),manifests_verified=manifests,unchanged_domain_weights=True,repository_and_lineage_separation=True,source_review_bound=True,admission_checks=7,scoring_launched=False,selection_allowed=False,native_precision_verified=False,coverage_scope_pending=read(DEST/'semantic-label-review.json')['fixed_floor_checks'])
    (DEST/'final-audit.json').write_text(json.dumps(report,indent=2)+'\n')
    scripts={str(p.relative_to(ROOT)):sha(p)for p in OUT.glob('*.py')}
    helpers=['public-tool-gap-fill-v1/build.py','public-tool-workflows-v1/inventory.py','public-tool-workflows-v1/normalize.py','public-tool-workflows-v1/checks.py','public-tool-workflows-v1/pipeline.py','calibration-mix-v3/common.py','calibration-mix-v3/verify_bound.py','calibration-gap-v1/normalize_native.py']
    scripts.update({n:sha(ROOT/n)for n in helpers})
    (DEST/'release-runtime.lock.json').write_text(json.dumps(dict(files=scripts),indent=2)+'\n')
    files={str(f.relative_to(ROOT)):sha(f)for f in DEST.iterdir()if f.is_file()and f.name not in ('release-seal.json','evidence.lock.json')}
    (DEST/'evidence.lock.json').write_text(json.dumps(dict(files=files),indent=2)+'\n')
    (DEST/'release-seal.json').write_text(json.dumps(dict(status='sealed_diagnostic_revision',evidence_lock_sha256=sha(DEST/'evidence.lock.json'),plans=completion['plans'],scoring_launched=False,selection_allowed=False),indent=2)+'\n')
    print(json.dumps(report))

if __name__=='__main__':main()
