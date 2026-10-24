"""Fresh source/content-only v9 preparation; no saliency inputs are read."""
import collections
import importlib.util
import json
from pathlib import Path
import re
import sys

ROOT=Path('/mnt/nvme4/mimo-reap')
OUT=ROOT/'public-workflow-gap-fill-v3'
PARENT=ROOT/'calibration-mix-v8'
spec=importlib.util.spec_from_file_location('v9preparation',ROOT/'public-workflow-gap-fill-v2/prepare.py')
A=importlib.util.module_from_spec(spec);sys.modules[spec.name]=A;spec.loader.exec_module(A)
B=A.B;C=A.C;A.OUT=OUT;A.DEST=ROOT/'calibration-mix-v9'


def parent():
    objects=A.rows(PARENT/'all-selected.jsonl')
    return {p:[x['record'] for x in objects if x['partition']==p] for p in A.P}


A.parent=parent


def setup():
    OUT.mkdir(exist_ok=False)
    policy=dict(parent='calibration-mix-v8',seed='paired-layer32-content-coverage-20261021-v3',candidate_limit_per_cell=60,target_per_partition_category=3,minimum_tokens=8192,maximum_tokens=65536,minimum_observed_results=4,matching='Same context bucket and <=1.5 length ratio; deterministic source hash after content review. No score inputs.',repository_caps=dict(one_addition_per_repository=True,maximum_combined_episodes=2,maximum_combined_tokens=100000),categories={
        'finance_risk_methods':dict(domain='finance',stratum='research',adapter_family='evidence_research',target_per_partition=2,pattern=r'portfolio|backtest|econometric|volatility|asset.pric|credit.risk|financial.risk|financial.time.series|financial.data|stock.return|market.risk',content_rule='Quantitative financial risk, estimation, optimization, validation or privacy methods. Reserve general investing/news and unrelated social/medical risk.'),
        'finance_empirical_methods':dict(domain='finance',stratum='research',adapter_family='evidence_research',target_per_partition=2,pattern=r'econometric|consumer.demand|financial.*(?:model|data|estimat|predict|forecast)|(?:causal|regression|time.series|factor).*(?:financial|stock|credit|economic)',content_rule='Empirical financial/economic modeling and inference with substantive methods and observed technical research. No general business summaries.'),
        'tool_persistence_indexer':dict(domain='general_tool_use',stratum='swe',adapter_family='persistence_recovery',pattern=r'database|indexer|indexing|migration|checkpoint|cursor|ingest|replicat|transaction|persist',content_rule='Executed database/indexer initialization, migration, persistence or recovery investigation; include test/build/tool feedback.'),
        'tool_typed_api':dict(domain='general_tool_use',stratum='swe',adapter_family='api_platform_security',pattern=r'graphql|openapi|query.parameter|schema|serializ|http.*(?:client|request)|api.*(?:validat|type|response)',content_rule='Executed typed API/query/schema integration, request/response validation or debugging, with observed tool feedback.'),
        'tool_statistical_evidence':dict(domain='general_tool_use',stratum='research',adapter_family='evidence_research',pattern=r'sample.complexity|concentration.bound|bootstrap|statistical|causal.inference|uncertainty.quantification|monte.carlo|bayesian|maximum.likelihood|unbiased.estimat|parameter.estimat',content_rule='Substantive statistical-method source retrieval and evidence synthesis, complete cited continuation and >=4 observed results.')})
    C.save(OUT/'policy.json',policy)
    pins={str(p.relative_to(ROOT)):C.file_sha(p) for p in PARENT.iterdir() if p.is_file()}
    C.save(OUT/'protected-before.json',pins)
    C.save(OUT/'source-discovery.json',dict(preferred_sources='Already pinned public train trajectories: NVIDIA Nemotron SFT-SWE and DR-Tulu SFT. Broaden the train inventory before considering new corpora.',reserved=[dict(dataset='thinkwee/DDRBench_10K_trajectory',reason='Benchmark-derived; reserve pending held-out audit. Not admitted.'),dict(dataset='obaydata/mcp-agent-trajectory-benchmark',reason='Benchmark trajectories; not admitted.'),dict(dataset='coslinedev/FinCode-Reasoning-v1',reason='Code/reasoning examples do not by themselves establish long agent action/result trajectories. Not admitted.')],score_inputs_read=False))


def inventory():
    A.inventory()
    initial=OUT/'raw-candidates.inventory.jsonl'
    (OUT/'raw-candidates.jsonl').rename(initial)
    policy=A.policy();par=parent();owners=C.owner_table(par)
    existing={r.get('source_record_sha256') for rows in par.values() for r in rows}
    pools=collections.defaultdict(list)
    for meta in B.read(PARENT/'sources.lock.json')['files']:
        if not(meta['path'].endswith('train.jsonl') and 'dr-tulu' in meta['dataset']):continue
        if C.file_sha(ROOT/meta['path'])!=meta['sha256']:raise ValueError('Changed public source')
        for i,line in enumerate((ROOT/meta['path']).open()):
            raw=json.loads(line);goal=B.I.clean_goal(raw['question']);h=C.sha(raw)
            if h in existing or raw.get('num_tool_calls',0)<4:continue
            for category,rule in policy['categories'].items():
                if rule['stratum']!='research' or not re.search(rule['pattern'],goal,re.I):continue
                group='research_question:'+C.sha(C.PUBLIC.problem(goal));part=B.I.partition(group,owners)
                candidate=dict(record=raw,stratum='research',source_file=meta['path'],source_row=i,dataset=meta['dataset'],revision=meta['revision'],upstream_file=meta['upstream_file'],source_record_sha256=h,split_group=group,problem_sha256=C.sha(C.PUBLIC.problem(goal)),repository_evidence='source_question_hash',partition=part,workflow_family=rule['adapter_family'],workflow_families=[rule['adapter_family']],objective_text=goal,task_text=raw['question'],metadata_priority=C.sha([policy['seed'],h,category]),gap_category=category,target_domain=rule['domain'])
                pools[(category,part)].append(candidate)
    extension=[r for _,rows in sorted(pools.items()) for r in sorted(rows,key=lambda r:r['metadata_priority'])[:60]]
    B.emit(OUT/'research-extension.jsonl',extension)
    merged={(r['source_record_sha256'],r['gap_category']):r for r in A.rows(initial)+extension}
    B.emit(OUT/'raw-candidates.jsonl',list(merged.values()))
    C.save(OUT/'candidate-merge.json',dict(initial_sha256=C.file_sha(initial),extension_sha256=C.file_sha(OUT/'research-extension.jsonl'),candidates=len(merged),cells=dict(collections.Counter(r['gap_category']+':'+r['partition'] for r in merged.values())),score_inputs_read=False))


if __name__=='__main__':
    {'setup':setup,'inventory':inventory,'normalize':A.normalize,'screen':A.screen}[sys.argv[1]]()
