"""Lossless adapters for complete native terminal and XML research histories."""
import sys,json,re,copy,collections,concurrent.futures
from pathlib import Path
sys.path.insert(0,'/mnt/nvme4/mimo-reap/calibration-mix-v3')
import common as C
from scan_terminal import action
OUT=Path(__file__).resolve().parent;TOKENIZER=None
TERMINAL_PREFIXES=['be054008ecdf','011360fc8a2a','3bbf60f360d1','235367c95249','20f63a8ec272','400b6eb5c484','008bfc254f9c','82fe7521b4bc','4acbc3349890','63e4f876bc84','2000ec545a82','0e8874078334','346c6dc8ca40','7516365d4b32']
RESEARCH_ROWS={791,1181,1991,3640,5360,5522,5834,6534,7468,12244,12704,12865}
def init():
 global TOKENIZER
 from transformers import AutoTokenizer
 TOKENIZER=AutoTokenizer.from_pretrained(C.MODEL,local_files_only=True,trust_remote_code=False)
def convert(c):
 try:
  raw=c['record'];msgs=copy.deepcopy(raw['conversations'])
  assert all(set(m)<= {'role','content'}and m['role']in ['user','system','assistant']and isinstance(m['content'],str)for m in msgs)
  assert msgs[-1]['role']=='assistant'
  batches=obs=commands=0;research_calls=0
  if c['stratum']=='terminal':
   final=action(msgs[-1]['content']);assert final.get('task_complete')is True and not final.get('commands')
   for i,m in enumerate(msgs[:-1]):
    if m['role']=='assistant':
     try:a=action(m['content'])
     except (ValueError,TypeError):continue
     if not isinstance(a,dict):continue
     cc=a.get('commands',[])
     if cc:
      assert i+1<len(msgs)and msgs[i+1]['role']=='user'and msgs[i+1]['content'].startswith('New Terminal Output:'),'unobserved_action_batch'
      batches+=1;commands+=sum(bool(x.get('keystrokes','').strip())for x in cc);obs+=1
   assert batches>=4 and obs>=4 and commands>=4,'insufficient_terminal_observations'
   kind='agent_terminal_native_trajectory';native='terminus2_json_commands_and_user_observations'
  else:
   assert len(msgs)==3 and [m['role']for m in msgs]==['system','user','assistant']
   body=msgs[-1]['content'];research_calls=len(re.findall(r'<call_tool\b',body))
   assert research_calls==raw['num_tool_calls']>=4
   assert body.count('<tool_output>')==body.count('</tool_output>')==research_calls,'unbalanced_retrieval_results'
   assert body.rstrip().endswith('</answer>')and body.count('<answer>')==body.count('</answer>')==1,'incomplete_research_answer'
   defined=set(re.findall(r'<(?:snippet|webpage)\s+id=["\x27]?([^"\x27>\s]+)',body))
   cited={v.strip()for x in re.findall(r'<cite\s+id=["\x27]([^"\x27]+)',body)for v in x.split(',')}
   assert cited and cited<=defined,'undefined_citation_ids'
   kind='agent_research_native_trajectory';native='dr_tulu_embedded_xml_actions_results_and_citations'
  msgs=json.loads(C.canonical(msgs));text=TOKENIZER.apply_chat_template(msgs,tokenize=False,add_generation_prompt=False)
  assert all(m['content']in text for m in msgs),'lost_source_content'
  ids=TOKENIZER(text,add_special_tokens=False)['input_ids'];n=len(ids);assert 4096<=n<=65536,'episode_length'
  r={k:v for k,v in c.items()if k not in ['record','task_text']}
  r.update(sample_id=C.sha([c['dataset'],c['revision'],c['source_record_sha256']]),source='public_gap_workflows',source_revision=c['revision'],source_split='train',source_license='CC-BY-4.0'if c['stratum']=='terminal'else'ODC-BY; upstream research/educational intent retained',task_key=c['split_group']+':'+c['problem_sha256'],prompt_sha256=C.sha(msgs[0]['content']if msgs[0]['role']=='user'else msgs[1]['content']),domain='general_tool_use',messages=msgs,token_ids=ids,token_sha256=C.token_hash(ids),valid_tokens=n,truncated=False,synthetic_repetition=False,teacher_generated=True,synthetic=True,trajectory_kind=kind,native_action_format=native,tool_call_count=0,tool_result_count=0,embedded_terminal_action_batches=batches,terminal_observation_count=obs,terminal_command_count=commands,embedded_retrieval_calls=research_calls,embedded_retrieval_results=research_calls,objective_text=c.get('task_text',raw.get('question','')),teacher_model=raw.get('model','DR-Tulu source teacher; see source card'),success_verification='upstream_attributed_not_independently_reexecuted',calibration_eligible=False,native_precision_verified=False,selection_allowed=False)
  return dict(ok=True,row=r)
 except (AssertionError,ValueError,KeyError,TypeError)as e:return dict(ok=False,hash=c['source_record_sha256'][:12],reason=str(e))
def main():
 raw=[]
 for c in map(json.loads,(OUT/'terminal-raw.jsonl').open()):
  if any(c['source_record_sha256'].startswith(p)for p in TERMINAL_PREFIXES):raw.append(c)
 meta=next(x for x in C.read(OUT/'research-source.lock.json')['files']if x['upstream_file']=='train.jsonl')
 for i,line in enumerate((C.ROOT/meta['path']).open()):
  if i not in RESEARCH_ROWS:continue
  r=json.loads(line);q=r['question'];group='research_question:'+C.sha(C.PUBLIC.problem(q));raw.append(dict(record=r,stratum='research',dataset=meta['dataset'],revision=meta['revision'],upstream_file=meta['upstream_file'],source_file=meta['path'],source_row=i,source_record_sha256=C.sha(r),split_group=group,problem_sha256=C.sha(C.PUBLIC.problem(q)),task_text=q,partition=C.partition(group,C.read(OUT/'owners.json'))))
 with(OUT/'native-raw.jsonl').open('w')as f:
  for c in raw:f.write(C.canonical(c)+'\n')
 stats=collections.Counter()
 with(OUT/'native-normalized.jsonl').open('w')as f,concurrent.futures.ProcessPoolExecutor(max_workers=3,initializer=init)as ex:
  for result in ex.map(convert,raw):
   if not result['ok']:print(result,flush=True);stats[result['reason']]+=1;continue
   r=result['row'];f.write(C.canonical(r)+'\n');stats['normalized']+=1
   print(r['source_record_sha256'][:12],r['partition'],r['stratum'],r['valid_tokens'],r['embedded_terminal_action_batches'],r['embedded_retrieval_calls'],r['objective_text'][:130].replace('\n',' '),flush=True)
 C.save(OUT/'native-normalization.json',dict(stats=stats,raw_sha256=C.file_sha(OUT/'native-raw.jsonl'),normalized_sha256=C.file_sha(OUT/'native-normalized.jsonl')))
if __name__=='__main__':main()
