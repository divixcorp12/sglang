"""Independent raw-source, token, identity and coverage verification before assembly."""
import json,sys,collections,concurrent.futures,copy,time,math,re
from pathlib import Path
from common import *
from build import verify_pins
from finish import overlap_inputs,visible,select_rows
TOKENIZER=None
def init():
 global TOKENIZER
 from transformers import AutoTokenizer
 TOKENIZER=AutoTokenizer.from_pretrained(MODEL,local_files_only=True,trust_remote_code=False)
def check_token(r):
 text=TOKENIZER.apply_chat_template(r['messages'],tokenize=False,add_generation_prompt=False)
 ids=TOKENIZER(text,add_special_tokens=False)['input_ids']
 if ids!=r['token_ids']or token_hash(ids)!=r['token_sha256']or len(ids)!=r['valid_tokens']or r['truncated']:
  raise ValueError('Exact token mismatch '+r['sample_id'])
 for m in r['messages']:
  for key in ['content','reasoning_content']:
   if m.get(key)and m[key]not in text:raise ValueError('Lost message content')
 return r['sample_id']
def check_lossless(raw,row):
 old=raw['messages'];new=row['messages'];offset=0
 if len(new)==len(old)+1:
  assert new[0]['role']=='system'and new[0]['content']=='';offset=1
 else:assert len(new)==len(old)
 for a,b in zip(old,new[offset:]):
  assert a['role']==b['role'];content=a.get('content')or'';assert isinstance(content,str)
  match=re.fullmatch(r'\s*<think>(.*?)</think>(.*)',content,re.S)if a['role']=='assistant'else None
  if match:assert b.get('reasoning_content')==match[1]and b['content']==match[2]
  else:
   assert b['content']==content
   if a.get('reasoning_content'):assert a['reasoning_content']==b.get('reasoning_content')
  assert not a.get('function_call'),'Legacy function call cannot be silently discarded'
  calls=a.get('tool_calls')or[]
  if isinstance(calls,str):calls=json.loads(calls)
  assert len(calls)==len(b.get('tool_calls',[]))
  for c,d in zip(calls,b.get('tool_calls',[])):
   assert c['function']['name']==d['function']['name']and c.get('id')==d.get('id')
   args=c['function'].get('arguments',{});args=json.loads(args)if isinstance(args,str)else args
   assert args==d['function']['arguments']
  if a.get('tools'):assert a['tools']==b.get('tools')
 if raw.get('tools'):
  tools=raw['tools'];tools=json.loads(tools)if isinstance(tools,str)else tools
  assert next(m for m in new if m['role']=='system')['tools']==tools

def reviewed_owners():
 owners=dict(read(OUT/'inventory.json')['owners'])
 suite={t['id']:t for t in read(ROOT/'teacher-python-v2/suite.lock.json')['tasks']}
 review=read(ROOT/'teacher-python-v2/content-review-v1/review.json')
 original={r['sample_id']:r for r in map(json.loads,(ROOT/'teacher-python-v2/candidates.jsonl').open())}
 selected={r['sample_id']:r for r in map(json.loads,(OUT/'focused-python.jsonl').open())}
 evidence=[]
 for d in review['records']:
  r=original[d['sample_id']];current=selected[d['sample_id']];t=suite[d['task']]
  assert r['split_group']==current['split_group']
  assert r['partition']==current['partition']==d['partition']==t['partition']
  group=r['split_group'].lower();part=t['partition']
  assert group not in owners or owners[group]==part
  owners[group]=part;evidence.append(dict(sample_id=r['sample_id'],group=group,partition=part,task=t['id']))
 assert len(evidence)==6
 save(OUT/'authored-ownership-audit.json',dict(status='passed',groups=evidence,rule='Reviewed authored suite ownership is established provenance; hash fallback applies only to unseen public groups.'))
 return owners

def main():
 if(OUT/'verification.json').exists():raise ValueError('Refuse overwrite completed verification')
 started=time.time();report=read(OUT/'build-report.json');verify_pins(read(OUT/'source-inputs.lock.json'));verify_pins(read(OUT/'python-source-inputs.lock.json'))
 assert report['all_selected_sha256']==file_sha(OUT/'all-selected.jsonl')
 assert file_sha(OUT/'public-reservoir.jsonl')==read(OUT/'inventory.json')['reservoir_sha256']
 assert file_sha(OUT/'public-normalized.jsonl')==read(OUT/'normalization.json')['sha256']
 assert file_sha(OUT/'screened-public.jsonl')==read(OUT/'screening.json')['screened_sha256']
 parts={p:[]for p in PARTS}
 for obj in map(json.loads,(OUT/'all-selected.jsonl').open()):parts[obj['partition']].append(obj['record'])
 base,oldplans=load_base();annotations=read(OUT/'annotations.json')
 for part,rows in parts.items():
  actual={r['sample_id']:r for r in rows};assert len(actual)==len(rows)
  for r in base[part]:assert actual[r['sample_id']]==r,'Anchor changed'
  assert [r for r in rows if r['domain']=='finance']==[r for r in base[part]if r['domain']=='finance']
  assert all(r['valid_tokens']<=65536 and not r['truncated']for r in rows)
 overlap=lineage_overlap(parts);assert not any(overlap.values())
 owners=reviewed_owners()
 for part,rows in parts.items():
  for r in rows:assert partition(r['split_group'],owners)==part
 selected=[json.loads(l)for l in(OUT/'selected-public.jsonl').open()]
 review=read(OUT/'content-review.json');decisions={r['sample_id']:r for r in review['records']}
 reservations={k:v['reason']for k,v in decisions.items()if v['decision']=='reserve'}
 _,_,_,recomputed,_,_,_=select_rows(reservations)
 assert {r['sample_id']for r in recomputed}=={r['sample_id']for r in selected},'Selection differs from deterministic content/quota policy'
 raw_pool={}
 wanted={r['source_record_sha256']:r for r in selected}
 for c in map(json.loads,(OUT/'public-reservoir.jsonl').open()):
  if c['source_record_sha256']in wanted:raw_pool[c['source_record_sha256']]=c
 assert set(raw_pool)==set(wanted)
 bysource=collections.defaultdict(dict)
 for rawhash,r in wanted.items():
  c=raw_pool[rawhash];assert sha(c['record'])==rawhash
  assert r['sample_id']==sha([r['dataset'],r['revision'],rawhash])
  assert PUBLIC.content_rejection(c['record'],r['stratum'])is None
  family,tags,why=classify(c['record'],r['stratum']);assert why is None and family==r['workflow_family']and tags==r['workflow_families']
  if r['stratum']=='swe':assert r['split_group']=='repo:'+PUBLIC.repo_of(first_user(c['record']))
  check_lossless(c['record'],r)
  assert decisions[r['sample_id']]['decision']=='admit'and decisions[r['sample_id']]['source_record_sha256']==rawhash
  bysource[r['source_file']][r['source_row']]=c
 lock=read(ROOT/'nvidia-calibration-v4/source.lock.json')
 source_rows=0
 for filename,rows in bysource.items():
  meta=next(m for m in lock['files']if m['path']==filename);assert file_sha(ROOT/filename)==meta['sha256']
  checked=set();last=max(rows)
  for index,raw in PUBLIC.rows(meta):
   if index in rows:
    assert raw==rows[index]['record']and sha(raw)==rows[index]['source_record_sha256']
    checked.add(index);source_rows+=1
   if index>=last:break
  assert checked==set(rows);log('raw_source_verified',file=filename,rows=len(checked))
 allrows=[r for p in PARTS for r in parts[p]]
 with concurrent.futures.ProcessPoolExecutor(max_workers=4,initializer=init)as pool:
  for n,_ in enumerate(pool.map(check_token,allrows,chunksize=2),1):
   if n%100==0:log('independent_retokenization',completed=n,total=len(allrows))
 # Independent problem and visible-span checks cover novel admissions; inherited anchors keep prior audits.
 pins,existing,held,reserved,question,forbidden,grams=overlap_inputs()
 existing_features=[PUBLIC.features(question(r))for r in existing]
 focused=[json.loads(l)for l in(OUT/'focused-python.jsonl').open()]
 newrows=selected+focused
 for n,r in enumerate(newrows):
  assert r['split_group'].lower()not in reserved
  fs=PUBLIC.features(question(r))
  assert all(not PUBLIC.similar(fs,x)for x in existing_features),'New row near existing/heldout problem'
  assert not grams(visible(r))&forbidden,'Uncommon heldout visible span'
 for i,a in enumerate(newrows):
  fa=PUBLIC.features(question(a))
  for b in newrows[:i]:assert not PUBLIC.similar(fa,PUBLIC.features(question(b))),'Novel rows near duplicates'
 gaps=[];coverage={};concentration={};contexts={};cohorts={}
 for part,rows in parts.items():
  coverage[part]={};contexts[part]={};cohorts[part]={}
  for domain in DOMAINS:
   rr=[r for r in rows if r['domain']==domain];n=sum(r['valid_tokens']for r in rr)
   families=collections.defaultdict(lambda:dict(episodes=0,tokens=0,groups=set(),origins=collections.Counter()))
   origins=collections.Counter();buckets=collections.Counter();repo=collections.Counter()
   for r in rr:
    ann=annotations[r['sample_id']];origins[ann['origin']]+=r['valid_tokens'];buckets[bucket(r['valid_tokens'])]+=r['valid_tokens']
    if r['split_group'].lower().startswith('repo:'):repo[r['split_group'].lower()]+=r['valid_tokens']
    for fam in ann['families']:
     x=families[fam];x['episodes']+=1;x['tokens']+=r['valid_tokens'];x['groups'].add(r['split_group'].lower());x['origins'][ann['origin']]+=1
   for fam,x in families.items():x['groups']=sorted(x['groups']);x['origins']=dict(x['origins'])
   coverage[part][domain]=dict(episodes=len(rr),tokens=n,families=dict(families))
   contexts[part][domain]=dict(tokens=dict(buckets));cohorts[part][domain]=dict(tokens=dict(origins))
   concentration[part+':'+domain]=[dict(group=g,tokens=nt,share=nt/n)for g,nt in repo.most_common()]
   if domain=='coding':assert max(repo.values(),default=0)/n<=POLICY['final_coding_repository_token_share_cap']
   for fam in POLICY['required_workflow_groups_each_partition'].get(domain,[]):
    if len(families.get(fam,{}).get('groups',[]))<2:gaps.append(part+':'+domain+':'+fam+':fewer_than_two_groups')
 for part in PARTS:
  for domain,families in POLICY['public_quotas_per_partition'].items():
   for fam,target in families.items():
    n=sum(r['valid_tokens']for r in selected if r['partition']==part and r['domain']==domain and r['workflow_family']==fam)
    if n<target*.8:gaps.append(f'{part}:{domain}:{fam}:quota_{n}_below_{int(target*.8)}')
    assert n<=target*1.1
 verify_pins(read(OUT/'source-inputs.lock.json'));verify_pins(read(OUT/'python-source-inputs.lock.json'));verify_pins(pins)
 save(OUT/'coverage.json',dict(partitions=coverage,contexts=contexts,source_cohorts=cohorts,repository_concentration=concentration,quota_and_coverage_gaps=gaps,source_selection_uses_expert_scores=False))
 save(OUT/'overlap-audit.json',dict(status='passed',primary_repeat_overlap=overlap,novel_source_rows=source_rows,novel_episodes_checked=len(newrows),heldout_records=len(held),checks='Repository/task identities, exact/near problem inputs, uncommon full visible spans including tool arguments.',limits=POLICY['selection_limits'],formatting_exception=read(OUT/'screening.json')['formatting_exception']))
 result=dict(status='passed',verifier_code_sha256=file_sha(OUT/'verify_bound.py'),authored_ownership_audit_sha256=file_sha(OUT/'authored-ownership-audit.json'),retokenized=len(allrows),episodes={p:len(rs)for p,rs in parts.items()},tokens={p:sum(r['valid_tokens']for r in rs)for p,rs in parts.items()},all_selected_sha256=file_sha(OUT/'all-selected.jsonl'),raw_source_rows_verified=source_rows,all_v8_records_preserved_exactly=True,focused_python_admitted=6,coverage_requirements_passed=not gaps,coverage_gaps=gaps,input_files_rechecked=len(read(OUT/'source-inputs.lock.json'))+len(read(OUT/'python-source-inputs.lock.json')),policy_sha256=file_sha(OUT/'policy.json'),elapsed_seconds=time.time()-started,native_precision_verified=False,selection_allowed=False,scoring_launched=False,limits=POLICY['selection_limits'])
 save(OUT/'verification.json',result);log('verification_complete',**result)
if __name__=='__main__':main()
