"""Source/content-reviewed labels for additions; preserve all parent labels."""
import copy
import importlib.util
import json
import re
import sys
from pathlib import Path
ROOT=Path('/mnt/nvme4/mimo-reap');OUT=ROOT/'public-workflow-gap-fill-v2';DEST=ROOT/'calibration-mix-v8'
spec=importlib.util.spec_from_file_location('v8_labels',OUT/'release.py');R=importlib.util.module_from_spec(spec);sys.modules[spec.name]=R;spec.loader.exec_module(R)

def main():
    assert R.B.read(DEST/'verification.json')['status']=='passed'
    original=R.B.read(ROOT/'calibration-mix-v7/annotations.json');ann=R.B.read(DEST/'annotations.json');before=copy.deepcopy(ann)
    objects=R.A.rows(DEST/'all-selected.jsonl');adds=R.A.rows(DEST/'selected-additions.jsonl')
    mapping={'coding_parser_metadata':['memory_parsing_state'],'coding_numeric_arrays':[], 'coding_memory_state':['memory_parsing_state'],'science_simulation_measurement':['evidence_research','scientific_methods'],'science_chemical_biological':['evidence_research','chemical_biological'],'finance_quant_risk':['quantitative_research'],'finance_data_methods':['quantitative_research'],'tool_statistical_research':['evidence_research'],'tool_typed_api_sql':['api_platform_security']}
    labels=[]
    for r in adds:
        tags=[r['gap_category']]+mapping[r['gap_category']]
        if r['split_group'].lower() in {'repo:bqskit/bqskit','repo:mne-tools/mne-python','repo:astropy/astropy','repo:mikgroup/sigpy','repo:craffel/mir_eval','repo:microsoft/qsharp'}:tags.append('scientific_numeric')
        if r['split_group'].lower()=='repo:wemake-services/wemake-python-styleguide':
            assert '__aiter__' in r['objective_text'];tags.append('concurrency_async')
        ann[r['sample_id']]['families']=sorted(set(tags))
        labels.append(dict(sample_id=r['sample_id'],source_record_sha256=r['source_record_sha256'],families=ann[r['sample_id']]['families'],rationale='Reviewed task mechanism; scientific numeric labels limited to real scientific/quantum/array methods, and async label to actual async-iterator semantics.'))
    assert all(ann[sid]==v for sid,v in original.items())
    R.C.save(DEST/'annotations.pre-semantic-review.json',before);R.C.save(DEST/'annotations.json',ann)
    oldcov=R.B.read(ROOT/'calibration-mix-v7/coverage.json');checks=[]
    for gap in oldcov.get('preserved_non_tool_coverage_gaps',[]):
        m=re.fullmatch(r'(primary|repeat):(\w+):(\w+):quota_(\d+)_below_(\d+)',gap)
        if not m:checks.append(dict(gap=gap,status='requires_separate_review'));continue
        p,d,f,expected,floor=m.groups();expected=int(expected);floor=int(floor)
        oldrs=[x['record']for x in R.A.rows(ROOT/'calibration-mix-v7/all-selected.jsonl')if x['partition']==p and x['record']['domain']==d]
        rs=[x['record']for x in objects if x['partition']==p and x['record']['domain']==d]
        oldtotal=sum(r['valid_tokens']for r in oldrs if f in original[r['sample_id']]['families'])
        total=sum(r['valid_tokens']for r in rs if f in ann[r['sample_id']]['families'])
        checks.append(dict(gap=gap,inherited_reported_tokens=expected,recomputed_parent_tokens=oldtotal,current_tokens=total,floor=floor,status='fixed_floor_met'if oldtotal==expected and total>=floor else 'requires_separate_review'))
    R.C.save(DEST/'semantic-label-review.json',dict(status='reviewed',parent_annotations_preserved=True,records=labels,fixed_floor_checks=checks,script_sha256=R.C.file_sha(__file__),scope='Content coverage only; no scoring, native precision, task-quality or pruning acceptance.'))
    cov=R.B.read(DEST/'coverage.json');cov['additive_fixed_floor_audit']=checks;cov['coverage_requirements_passed']=False
    R.C.save(DEST/'coverage.json',cov)
    print(json.dumps(checks))

if __name__=='__main__':main()
