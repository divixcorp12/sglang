"""CPU-only immutable failed-prefix attribution; counterfactuals are diagnostic."""
import importlib.util
import json
import math
from pathlib import Path

ROOT=Path('/mnt/nvme4/mimo-reap')
OUT=ROOT/'rank-diagnosis-v8-prefix'


def main():
    spec=importlib.util.spec_from_file_location('production',ROOT/'incremental-production-v1/production_incremental.py')
    runtime=importlib.util.module_from_spec(spec);spec.loader.exec_module(runtime)
    run=runtime.Production(ROOT,ROOT/'incremental-calibration-v1')
    run.locked();snapshot=run.out/'merges/prefix-32';run.verify_merge(snapshot)
    gate=runtime.read(snapshot/'early-gate.json')
    added={json.loads(line)['sample_id'] for line in (ROOT/'calibration-mix-v8/selected-additions.jsonl').read_text().splitlines()}
    source={obj['record']['sample_id']:obj['record'] for obj in map(json.loads,(ROOT/'calibration-mix-v8/all-selected.jsonl').read_text().splitlines())}
    reports=[]
    def order(scores):
        return sorted(range(384),key=lambda i:(-scores[i],i))
    for key,jaccard in gate['failed'].items():
        layer,domain=key.split(':')
        observations={part:runtime.read(snapshot/part/'observations.json') for part in runtime.PARTS}
        scores={part:observations[part]['layers'][layer][domain]['gate_weighted_output_norm_sum'] for part in runtime.PARTS}
        ranks={part:{expert:rank+1 for rank,expert in enumerate(order(s))} for part,s in scores.items()}
        retained={part:set(order(s)[:344]) for part,s in scores.items()}
        a,b=retained.values();runtime.require(len(a&b)/len(a|b)==jaccard,'Gate reproduction differs')
        records={part:[r for r in runtime.read(snapshot/part/'trajectories'/f'layer-{int(layer):02d}.json')['records'] if r['domain']==domain] for part in runtime.PARTS}
        disputed=sorted(a^b);experts=[]
        for expert in disputed:
            report=dict(expert=expert,ranks={part:r[expert] for part,r in ranks.items()},partitions={})
            for part,rows in records.items():
                total=scores[part][expert]
                sorted_rows=sorted(rows,key=lambda r:-r['gate_weighted_output_norm_sum'][expert])
                influential=[]
                for row in sorted_rows[:5]:
                    original=source[row['sample_id']]
                    goal=original.get('source_question') or original.get('objective_text') or next((m.get('content','') for m in original.get('messages',[]) if m.get('role')=='user'),'')
                    influential.append(dict(sample_id=row['sample_id'],new_in_v8=row['sample_id'] in added,share=row['gate_weighted_output_norm_sum'][expert]/total if total else 0,valid_tokens=row['valid_tokens'],dataset=row.get('dataset'),split_group=row.get('split_group'),stratum=original.get('calibration_stratum'),question=str(goal)[:2000]))
                report['partitions'][part]=dict(new_history_share=math.fsum(r['gate_weighted_output_norm_sum'][expert] for r in rows if r['sample_id'] in added)/total if total else 0,top_histories=influential)
            experts.append(report)
        old=runtime.read(ROOT/'full-scoring-v11/rank-stability.json')['jaccard_by_layer_domain'][key]
        reports.append(dict(check=key,jaccard=jaccard,previous_v11_jaccard=old,disputed_experts=experts,domain_histories={part:len(rows) for part,rows in records.items()},new_domain_histories={part:sum(r['sample_id'] in added for r in rows) for part,rows in records.items()}))
    OUT.mkdir(exist_ok=True)
    report=dict(status='diagnosed_failed_gate',failed_checks=reports,parent_snapshot=str(snapshot),snapshot_seal_sha256=runtime.sha(snapshot/'seal.json'),threshold_unchanged=.95,keep_unchanged=344,source_records_removed=False,selection_allowed=False)
    runtime.save(OUT/'report.json',report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
