"""Additive source/content-only calibration preparation. Never reads expert scores."""
import collections
import concurrent.futures
import copy
import importlib.util
import json
import re
import sys
from pathlib import Path

ROOT = Path('/mnt/nvme4/mimo-reap')
OUT = ROOT / 'public-workflow-gap-fill-v2'
DEST = ROOT / 'calibration-mix-v8'
spec = importlib.util.spec_from_file_location('existing_gap_helpers', ROOT/'public-tool-gap-fill-v1/build.py')
B = importlib.util.module_from_spec(spec); spec.loader.exec_module(B)
C = B.C
P = ('primary', 'repeat')

def rows(path):
    return [json.loads(x) for x in Path(path).open()]

def save(name, value):
    C.save(OUT/name, value)

def emit(name, value):
    B.emit(OUT/name, value)

def log(event, **kw):
    v = dict(event=event, **kw)
    print(json.dumps(v), flush=True)
    with (OUT/'steps.jsonl').open('a') as f: f.write(json.dumps(v)+'\n')

def parent():
    objects = rows(ROOT/'calibration-mix-v7/all-selected.jsonl')
    return {p:[x['record'] for x in objects if x['partition']==p] for p in P}

def policy():
    return B.read(OUT/'policy.json')

def inventory():
    assert not (OUT/'raw-candidates.jsonl').exists()
    par=parent(); owners=C.owner_table(par); save('owners.json',owners)
    existing={r.get('source_record_sha256') for rs in par.values() for r in rs}
    counts=collections.Counter(r['split_group'].lower() for rs in par.values() for r in rs)
    tokens=collections.Counter()
    for rs in par.values():
        for r in rs: tokens[r['split_group'].lower()]+=r['valid_tokens']
    metas=B.read(ROOT/'calibration-mix-v7/sources.lock.json')['files']
    pools=collections.defaultdict(list); stats=collections.Counter(); pol=policy()
    def add(raw,meta,i,stratum):
        goal=B.I.clean_goal(raw['question'] if stratum=='research' else C.first_user(raw))
        try: group='research_question:'+C.sha(C.PUBLIC.problem(goal)) if stratum=='research' else 'repo:'+C.PUBLIC.repo_of(C.first_user(raw))
        except ValueError: return
        group=group.lower()
        if counts[group]>=2 or tokens[group]>=100000: return
        h=C.sha(raw)
        if h in existing or not 2000<=len(C.canonical(raw))<=500000:return
        part=B.I.partition(group,owners)
        for category,rule in pol['categories'].items():
            if rule['stratum']!=stratum or not re.search(rule['pattern'],goal,re.I):continue
            priority=C.sha([pol['seed'],h,category])
            c=dict(record=raw,stratum=stratum,source_file=meta['path'],source_row=i,dataset=meta['dataset'],revision=meta['revision'],upstream_file=meta['upstream_file'],source_record_sha256=h,split_group=group,problem_sha256=C.sha(C.PUBLIC.problem(goal)),repository_evidence='explicit_workspace_path' if stratum=='swe' else 'source_question_hash',partition=part,workflow_family=rule['adapter_family'],workflow_families=[rule['adapter_family']],objective_text=goal,metadata_priority=priority,gap_category=category,target_domain=rule['domain'])
            if stratum=='research':c['task_text']=raw['question']
            key=(category,part,group); pools[key].append(c);pools[key]=sorted(pools[key],key=lambda r:r['metadata_priority'])[:1]
            stats[category+':'+part]+=1
    for m in metas:
        if m['path'].endswith('.parquet') and 'SFT-SWE' in m['dataset']:
            assert C.file_sha(ROOT/m['path'])==m['sha256']
            for i,raw in C.PUBLIC.rows(m):add(raw,m,i,'swe')
        elif m['path'].endswith('train.jsonl') and 'dr-tulu' in m['dataset']:
            assert C.file_sha(ROOT/m['path'])==m['sha256']
            for i,line in enumerate((ROOT/m['path']).open()):
                raw=json.loads(line)
                if raw['num_tool_calls']>=4 and raw['type']=='long_form':add(raw,m,i,'research')
        log('inventory_source',path=m['path'],eligible=dict(stats))
    cells=collections.defaultdict(list)
    for (cat,part,group),rs in pools.items():cells[(cat,part)].extend(rs)
    candidates=[r for cell,rs in sorted(cells.items()) for r in sorted(rs,key=lambda x:x['metadata_priority'])[:pol['candidate_limit_per_cell']]]
    emit('raw-candidates.jsonl',candidates);save('sources.lock.json',dict(files=metas));save('inventory.json',dict(candidates=len(candidates),eligible=dict(stats),score_inputs_read=False))

def init():
    B.N.init()

def work(c):
    v=B.norm_work(c)
    if v['ok']:
        r=v['row'];r['target_domain']=c['target_domain'];r['gap_category']=c['gap_category']
        r['preparation_policy_sha256']=C.file_sha(OUT/'policy.json')
        if not 8192<=r['valid_tokens']<=65536:return dict(ok=False,reason='context_bounds')
    return v

def normalize():
    assert not (OUT/'normalized.jsonl').exists()
    candidates=rows(OUT/'raw-candidates.jsonl');ok=[];bad=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=4,initializer=init) as ex:
        for i,v in enumerate(ex.map(work,candidates,chunksize=1),1):
            (ok if v['ok'] else bad).append(v['row'] if v['ok'] else v)
            if i%20==0:log('normalized',completed=i,total=len(candidates),accepted=len(ok))
    emit('normalized.jsonl',ok);save('normalization-rejections.json',bad)

def screen():
    assert not (OUT/'screened.jsonl').exists()
    pins,_,held,reserved,question,forbidden,grams=B.overlap_inputs()
    par=parent();old=[r for rs in par.values() for r in rs];features=[];index=collections.defaultdict(set);exact=set()
    for r in old+held:
        q=B.S.objective(r) if r in old else question(r)
        exact.add(C.sha(C.PUBLIC.problem(q)));fs=C.PUBLIC.features(q);j=len(features);features.append(fs)
        for g in fs:index[g].add(j)
    lineage={p:{k:{str(v).lower() for r in par[p] for v in r.get(k,[])} for k in ['lineage_aliases','source_papers','dataset_groups']} for p in P}
    accepted=[];bad=[]
    for r in rows(OUT/'normalized.jsonl'):
        why=None
        try:B.K.record(r)
        except Exception as e:why='structure:'+str(e)
        q=B.S.objective(r);fs=C.PUBLIC.features(q);hits=collections.Counter(j for g in fs for j in index.get(g,()))
        if r['split_group'].lower() in reserved:why='heldout_group'
        if C.sha(C.PUBLIC.problem(q)) in exact:why='existing_or_heldout_exact'
        if any(n>=8 and C.PUBLIC.similar(fs,features[j]) for j,n in hits.items()):why='existing_or_heldout_near'
        if grams(B.visible(r))&forbidden:why='heldout_visible_span'
        other=P[1-P.index(r['partition'])]
        if any(set(map(str.lower,r.get(k,[])))&lineage[other][k] for k in lineage[other]):why='opposite_parent_lineage'
        if why:bad.append(dict(sample_id=r['sample_id'],category=r['gap_category'],reason=why))
        else:accepted.append(r)
    emit('screened.jsonl',accepted);save('screening.json',dict(accepted=len(accepted),rejected=bad,input_pins=pins))
    save('review-queue.json',[dict(sample_id=r['sample_id'],category=r['gap_category'],partition=r['partition'],group=r['split_group'],tokens=r['valid_tokens'],goal=r['objective_text'],priority=r['metadata_priority']) for r in accepted])
    log('screened',accepted=len(accepted),cells=dict(collections.Counter(r['gap_category']+':'+r['partition'] for r in accepted)))

if __name__=='__main__':
    {'inventory':inventory,'normalize':normalize,'screen':screen}[sys.argv[1]]()
