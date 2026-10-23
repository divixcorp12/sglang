"""Adversarial admission checks against real v8 records; CPU only."""
import copy
import importlib.util
import sys
from pathlib import Path
ROOT=Path('/mnt/nvme4/mimo-reap');OUT=ROOT/'public-workflow-gap-fill-v2'
spec=importlib.util.spec_from_file_location('v8_release_checks',OUT/'release.py')
R=importlib.util.module_from_spec(spec);sys.modules[spec.name]=R;spec.loader.exec_module(R)

def main():
    par=R.A.parent();adds=R.A.rows(OUT/'proposed.jsonl');results=[]
    R.check_additions(adds,par);results.append('valid real additions accepted')
    def reject(name,records,text):
        try:R.check_additions(records,par)
        except ValueError as e:
            assert text in str(e),(name,str(e));results.append(name);return
        raise AssertionError(name+' was accepted')
    r=copy.copy(adds[0]);r['partition']=R.P[1-R.P.index(r['partition'])]
    reject('opposite partition rejected',[r],'partition ownership')
    reject('repeated group rejected',[adds[0],adds[0]],'one addition per group')
    p=next(p for p in R.P if any(r.get('source_record_sha256')for r in par[p]))
    r=copy.copy(next(r for r in par[p]if r.get('source_record_sha256')));r['partition']=p
    reject('preserved source duplicate rejected',[r],'duplicate source history')
    r=copy.copy(next(r for r in adds if r['split_group'].startswith('repo:')));r['valid_tokens']=100001
    reject('combined repository token cap rejected',[r],'repository cap')
    r=copy.copy(next(r for r in adds if r.get('source_papers')));other=R.P[1-R.P.index(r['partition'])]
    paper=next(x for row in par[other]for x in row.get('source_papers',[]));r['source_papers']=[paper]
    reject('opposite partition paper rejected',[r],'partition_lineage_overlap')
    R.B.V.init();r=copy.deepcopy(adds[0]);r['token_ids'][0]^=1
    try:R.B.V.check_token(r)
    except ValueError as e:
        assert 'Exact token mismatch' in str(e);results.append('altered token stream rejected')
    else:raise AssertionError('Altered token stream accepted')
    R.C.save(OUT/'checks.json',dict(status='passed',checks=results,script_sha256=R.C.file_sha(__file__)))
    print(results)

if __name__=='__main__':main()
