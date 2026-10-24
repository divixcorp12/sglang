"""Repository-aware public train discovery; repository identity is not admission."""
import collections
import concurrent.futures
import importlib.util
import json
from pathlib import Path
import re
import sys

ROOT=Path('/mnt/nvme4/mimo-reap');OUT=ROOT/'public-workflow-gap-fill-v3/financial-repo-scan-v1'
spec=importlib.util.spec_from_file_location('repo_scan_prepare',ROOT/'public-workflow-gap-fill-v3/prepare.py');A=importlib.util.module_from_spec(spec);sys.modules[spec.name]=A;spec.loader.exec_module(A)
B=A.B;C=A.C;A.A.OUT=OUT
REPOS=r'(?:quantlib|qlib|freqtrade|backtrader|zipline|finrl|quantstats|pyportfolioopt|riskfolio|nautilus.trader|vectorbt|openbb|backtesting\.py|yfinance|qstrader|pyfolio|bt\.py|financetoolkit|financepy|pysabr)'
GOALS=r'backtest\w*|ohlcv|sharpe|drawdown|option.pric\w*|asset.pric\w*|financial.time.series|trading.strateg\w*|interest.rate.model'


def scan(meta):
    par=A.parent();owners=C.owner_table(par)
    existing={r.get('source_record_sha256') for rows in par.values() for r in rows}
    counts=collections.Counter(r['split_group'].lower() for rows in par.values() for r in rows)
    if C.file_sha(ROOT/meta['path'])!=meta['sha256']:raise ValueError('Changed source')
    candidates=[];hits=[];stats=collections.Counter()
    for index,raw in C.PUBLIC.rows(meta):
        user=C.first_user(raw);goal=B.I.clean_goal(user)
        try:group=('repo:'+C.PUBLIC.repo_of(user)).lower()
        except ValueError:continue
        if not(re.search(REPOS,group,re.I) or re.search(GOALS,goal,re.I)):continue
        stats[group]+=1
        h=C.sha(raw)
        hits.append(dict(group=group,row=index,source_record_sha256=h,goal=goal[:1000],already_in_parent=h in existing,parent_group_records=counts[group]))
        if h in existing or counts[group]>=2:continue
        part=B.I.partition(group,owners);category='finance_research_software'
        candidates.append(dict(record=raw,stratum='swe',source_file=meta['path'],source_row=index,dataset=meta['dataset'],revision=meta['revision'],upstream_file=meta['upstream_file'],source_record_sha256=h,split_group=group,problem_sha256=C.sha(C.PUBLIC.problem(goal)),repository_evidence='explicit_workspace_path',partition=part,workflow_family='scientific_numeric',workflow_families=['scientific_numeric'],objective_text=goal,metadata_priority=C.sha(['finance-repository-content-discovery-20261021',h]),gap_category=category,target_domain='finance'))
    return candidates,hits,dict(stats)


def inventory():
    OUT.mkdir(exist_ok=False)
    policy=B.read(ROOT/'public-workflow-gap-fill-v3/financial-extension/policy.json')
    policy.update(seed='finance-repository-content-discovery-20261021',repository_pattern=REPOS,goal_pattern=GOALS,discovery_only='Repository and objective matches are review candidates only; substantive quant-finance action/result/final review is required.')
    C.save(OUT/'policy.json',policy);C.save(OUT/'owners.json',C.owner_table(A.parent()))
    sources=B.read(ROOT/'calibration-mix-v8/sources.lock.json')['files']+B.read(ROOT/'public-workflow-gap-fill-v3/financial-extension/sources.lock.json')['files']
    sources={m['sha256']:m for m in sources if m['dataset']=='nvidia/Nemotron-SFT-SWE-v3' and m['path'].endswith('.parquet')}
    C.save(OUT/'sources.lock.json',dict(files=list(sources.values())))
    candidates=[];hits=[];counts=collections.Counter()
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
        for i,(rows,found,groups) in enumerate(pool.map(scan,sources.values(),chunksize=1),1):
            candidates.extend(rows);hits.extend(found);counts.update(groups)
            A.A.log('repository_scan',completed_sources=i,total_sources=len(sources),candidate_histories=len(candidates),matched_groups=dict(counts))
    unique={r['source_record_sha256']:r for r in candidates}
    B.emit(OUT/'raw-candidates.jsonl',sorted(unique.values(),key=lambda r:r['metadata_priority']))
    C.save(OUT/'hits.json',hits)
    C.save(OUT/'inventory.json',dict(sources=len(sources),candidates=len(unique),matched_groups=dict(counts),score_inputs_read=False))


if __name__=='__main__':
    {'inventory':inventory,'normalize':A.A.normalize,'screen':A.A.screen}[sys.argv[1]]()
