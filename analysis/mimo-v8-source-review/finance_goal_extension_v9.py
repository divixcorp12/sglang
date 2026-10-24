"""Separate source-only quantitative-method discovery in pinned public train data."""
import collections
import importlib.util
import json
from pathlib import Path
import re
import sys

ROOT = Path('/mnt/nvme4/mimo-reap')
OUT = ROOT / 'public-workflow-gap-fill-v3/financial-goal-extension-v1'
spec = importlib.util.spec_from_file_location('goal_extension_prepare', ROOT / 'public-workflow-gap-fill-v3/prepare.py')
A = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = A
spec.loader.exec_module(A)
B, C = A.B, A.C
A.A.OUT = OUT

PATTERNS = {
    'finance_risk_methods': r'VIX.term.structure|loan.and.insurance.underwriting|credit.scoring|financial.portfolio.optimization|(?:GARCH|expected.shortfall|value.at.risk|yield.curve|risk.premi)',
    'finance_empirical_methods': r'stock.buyback.*Inflation.Reduction|stock.price.syncronisity|earnings.qualit|monetary.value.of.patents|immediate.stock.market.impact',
}


def discover():
    OUT.mkdir(exist_ok=False)
    policy = B.read(ROOT / 'public-workflow-gap-fill-v3/policy.json')
    policy['categories'] = {k: dict(policy['categories'][k], pattern=p) for k, p in PATTERNS.items()}
    C.save(OUT / 'policy.json', policy)
    parent = A.parent()
    owners = C.owner_table(parent)
    C.save(OUT / 'owners.json', owners)
    existing = {r.get('source_record_sha256') for rows in parent.values() for r in rows}
    sources = [m for m in B.read(ROOT / 'calibration-mix-v8/sources.lock.json')['files']
               if 'dr-tulu' in m['dataset'] and m['path'].endswith('train.jsonl')]
    records = []
    for meta in sources:
        assert C.file_sha(ROOT / meta['path']) == meta['sha256']
        for i, line in enumerate((ROOT / meta['path']).open()):
            raw = json.loads(line)
            goal = B.I.clean_goal(raw['question'])
            h = C.sha(raw)
            if h in existing or raw.get('num_tool_calls', 0) < 4:
                continue
            for category, pattern in PATTERNS.items():
                if not re.search(pattern, goal, re.I):
                    continue
                group = 'research_question:' + C.sha(C.PUBLIC.problem(goal))
                part = B.I.partition(group, owners)
                records.append(dict(record=raw, stratum='research', source_file=meta['path'],
                    source_row=i, dataset=meta['dataset'], revision=meta['revision'],
                    upstream_file=meta['upstream_file'], source_record_sha256=h,
                    split_group=group, problem_sha256=C.sha(C.PUBLIC.problem(goal)),
                    repository_evidence='source_question_hash', partition=part,
                    workflow_family='evidence_research', workflow_families=['evidence_research'],
                    objective_text=goal, task_text=raw['question'],
                    metadata_priority=C.sha([policy['seed'], h, category]),
                    gap_category=category, target_domain='finance'))
    B.emit(OUT / 'raw-candidates.jsonl', sorted(records, key=lambda r: r['metadata_priority']))
    C.save(OUT / 'sources.lock.json', dict(files=sources))
    C.save(OUT / 'inventory.json', dict(candidates=len(records), score_inputs_read=False,
        admitted=False, cells=dict(collections.Counter(r['gap_category'] + ':' + r['partition'] for r in records))))


if __name__ == '__main__':
    {'discover': discover, 'normalize': A.A.normalize, 'screen': A.A.screen}[sys.argv[1]]()
