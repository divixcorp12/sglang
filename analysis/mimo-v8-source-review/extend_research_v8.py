"""Broaden statistical-method discovery; no admission or score access."""
import collections
import importlib.util
import json
import re
from pathlib import Path
ROOT=Path('/mnt/nvme4/mimo-reap');OUT=ROOT/'public-workflow-gap-fill-v2'
spec=importlib.util.spec_from_file_location('prep',OUT/'prepare.py');A=importlib.util.module_from_spec(spec);spec.loader.exec_module(A)
B=A.B;C=A.C

def main():
    assert not (OUT/'research-extension.jsonl').exists()
    pol=A.policy();par=A.parent();owners=C.owner_table(par);existing={r.get('source_record_sha256')for rs in par.values()for r in rs}
    pattern=r'sample complexity|concentration bound|hypothesis test|bootstrap|bandit|statistical|causal inference|causal effect|uncertainty quantification|monte carlo|bayesian|maximum likelihood|unbiased estimat|parameter estimation|active learning'
    C.save(OUT/'research-extension-policy.json',dict(reason='Initial cached long-form statistical-method inventory has only 2 primary and 1 repeat matches; broaden content discovery, retain source/length/result/leakage admission rules.',category='tool_statistical_research',pattern=pattern,source_types='All source types with >=4 observed results and complete cited answer; same native validator.',score_inputs_read=False))
    pools=collections.defaultdict(list)
    for m in B.read(ROOT/'calibration-mix-v7/sources.lock.json')['files']:
        if not(m['path'].endswith('train.jsonl') and 'dr-tulu' in m['dataset']):continue
        assert C.file_sha(ROOT/m['path'])==m['sha256']
        for i,line in enumerate((ROOT/m['path']).open()):
            raw=json.loads(line);goal=B.I.clean_goal(raw['question']);h=C.sha(raw)
            if h in existing or raw['num_tool_calls']<4:continue
            cats=[]
            if re.search(pattern,goal,re.I):cats.append('tool_statistical_research')
            if raw.get('type')!='long_form':
                cats += [k for k,v in pol['categories'].items()if v['domain']=='finance' and re.search(v['pattern'],goal,re.I)]
            for cat in cats:
                rule=pol['categories'][cat];g='research_question:'+C.sha(C.PUBLIC.problem(goal));part=B.I.partition(g,owners)
                c=dict(record=raw,stratum='research',source_file=m['path'],source_row=i,dataset=m['dataset'],revision=m['revision'],upstream_file=m['upstream_file'],source_record_sha256=h,split_group=g,problem_sha256=C.sha(C.PUBLIC.problem(goal)),repository_evidence='source_question_hash',partition=part,workflow_family=rule['adapter_family'],workflow_families=[rule['adapter_family']],objective_text=goal,task_text=raw['question'],metadata_priority=C.sha([pol['seed'],h,cat]),gap_category=cat,target_domain=rule['domain'],extension_policy_sha256=C.file_sha(OUT/'research-extension-policy.json'))
                pools[(cat,part)].append(c)
    result=[r for key,rs in sorted(pools.items())for r in sorted(rs,key=lambda x:x['metadata_priority'])[:40]]
    B.emit(OUT/'research-extension.jsonl',result)
    A.log('research_extension_inventory',candidates=len(result),cells={str(k):len(rs)for k,rs in pools.items()})

if __name__=='__main__':main()
