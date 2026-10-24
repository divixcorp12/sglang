"""Broaden pinned public training-shard discovery for quantitative software tasks."""
import collections
import importlib.util
import json
from pathlib import Path
import re
import sys

ROOT=Path('/mnt/nvme4/mimo-reap');OUT=ROOT/'public-workflow-gap-fill-v3/financial-extension'
spec=importlib.util.spec_from_file_location('finance_v9_prepare',ROOT/'public-workflow-gap-fill-v3/prepare.py');A=importlib.util.module_from_spec(spec);sys.modules[spec.name]=A;spec.loader.exec_module(A)
B=A.B;C=A.C;A.A.OUT=OUT
PATTERN=r'\b(?:backtest\w*|portfolio|quantlib|yfinance|riskfolio|pyfolio|backtrader|freqtrade|qlib|stock.price|stock.market|trading.strateg\w*|financial.time.series|ohlcv|sharpe|drawdown|option.pric\w*|asset.pric\w*|trading.bot|interest.rate.model)\b'


def discover():
    from huggingface_hub import hf_hub_download
    OUT.mkdir(exist_ok=False)
    policy=B.read(ROOT/'public-workflow-gap-fill-v3/policy.json')
    policy['categories']={'finance_research_software':dict(domain='finance',stratum='swe',adapter_family='scientific_numeric',target_per_partition=2,pattern=PATTERN,content_rule='Actual financial modeling, backtest/data-alignment/risk or quantitative research software correctness. Reserve generic business CRUD, incidental dependency mentions, promotional trading and unexecuted answers.')}
    C.save(OUT/'policy.json',policy)
    revision='3f73de64c1fe928a8f538fe45ccc10c228cc4c6a'
    parent=A.parent();owners=C.owner_table(parent);C.save(OUT/'owners.json',owners)
    existing={r.get('source_record_sha256') for rows in parent.values() for r in rows}
    counts=collections.Counter(r['split_group'].lower() for rows in parent.values() for r in rows)
    tokens=collections.Counter()
    for rows in parent.values():
        for r in rows:tokens[r['split_group'].lower()]+=r['valid_tokens']
    metas=[];candidates=[]
    for index in range(24,40):
        filename=f'data/train-{index:05d}-of-00096.parquet'
        path=Path(hf_hub_download('nvidia/Nemotron-SFT-SWE-v3',filename,repo_type='dataset',revision=revision,local_dir=OUT/'sources'))
        meta=dict(dataset='nvidia/Nemotron-SFT-SWE-v3',revision=revision,upstream_file=filename,path=str(path.relative_to(ROOT)),sha256=C.file_sha(path),bytes=path.stat().st_size,url=f'https://huggingface.co/datasets/nvidia/Nemotron-SFT-SWE-v3/resolve/{revision}/{filename}')
        metas.append(meta)
        for i,raw in C.PUBLIC.rows(meta):
            goal=B.I.clean_goal(C.first_user(raw))
            if not re.search(PATTERN,goal,re.I):continue
            try:group='repo:'+C.PUBLIC.repo_of(C.first_user(raw))
            except ValueError:continue
            group=group.lower();h=C.sha(raw)
            if h in existing or counts[group]>=2 or tokens[group]>=100000:continue
            part=B.I.partition(group,owners);category='finance_research_software'
            candidates.append(dict(record=raw,stratum='swe',source_file=meta['path'],source_row=i,dataset=meta['dataset'],revision=revision,upstream_file=filename,source_record_sha256=h,split_group=group,problem_sha256=C.sha(C.PUBLIC.problem(goal)),repository_evidence='explicit_workspace_path',partition=part,workflow_family='scientific_numeric',workflow_families=['scientific_numeric'],objective_text=goal,metadata_priority=C.sha([policy['seed'],h,category]),gap_category=category,target_domain='finance'))
        C.save(OUT/'sources.lock.json',dict(files=metas))
        A.A.log('financial_source_inventory',shard=index,candidates=len(candidates),partitions=dict(collections.Counter(r['partition'] for r in candidates)))
    selected=sorted(candidates,key=lambda r:r['metadata_priority'])
    B.emit(OUT/'raw-candidates.jsonl',selected)
    C.save(OUT/'inventory.json',dict(candidates=len(selected),score_inputs_read=False,new_training_shards=len(metas)))


if __name__=='__main__':
    if sys.argv[1]=='discover':discover()
    elif sys.argv[1]=='normalize':A.A.normalize()
    elif sys.argv[1]=='screen':A.A.screen()
