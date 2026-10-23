"""Score-blind helpers for a versioned whole-episode calibration blend."""
import json,re,hashlib,importlib.util,sys,collections
from pathlib import Path
ROOT=Path('/mnt/nvme4/mimo-reap');OUT=Path(__file__).resolve().parent;MODEL=Path('/mnt/nvme4/mimi-v26-pro')
sys.path.insert(0,str(ROOT))
from build_trajectory_bank import canonical,sha,file_sha,save
from prepare_observer_manifests import token_hash,write
def read(p):return json.loads(Path(p).read_text())
POLICY=read(OUT/'policy.json')
spec=importlib.util.spec_from_file_location('mix_public_source_helpers',ROOT/'nvidia-calibration-v4/pipeline.py')
PUBLIC=importlib.util.module_from_spec(spec);sys.modules[spec.name]=PUBLIC;spec.loader.exec_module(PUBLIC)
DOMAINS=('finance','science','coding','general_tool_use');PARTS=('primary','repeat')
def log(event,**kw):
 import time
 data=dict(event=event,at=time.time(),**kw);save(OUT/'progress.json',data)
 with(OUT/'steps.jsonl').open('a')as f:f.write(json.dumps(data)+'\n')
 print(json.dumps(data),flush=True)
def load_base():
 rows={};plans={}
 for part,name in zip(PARTS,['observer-manifests-v8','observer-manifests-repeat-v8']):
  plans[part]=read(ROOT/'data'/name/'plan.json')
  rows[part]=[r for g in plans[part]['groups']for r in read(ROOT/g['manifest'])['samples']]
 return rows,plans
def first_user(raw):return PUBLIC.user(raw)
def issue_body(text):
 for tag in ['issue_description','issue_details','github_issue_description','pr_description','issue']:
  matches=re.findall('<'+tag+r'>((?:(?!<'+tag+r'>).)*?)</'+tag+'>',text,re.S|re.I)
  if matches:return matches[-1]
 return text.split('Implement /workspace/solution.py')[0]
def task_text(raw):return issue_body(first_user(raw)).lower()
# Fix extraction locally for both new and comparison questions; upstream helpers stay untouched.
def corrected_problem(text):return re.sub(r'\\s+',' ',issue_body(text).lower()).strip()
PUBLIC.problem=corrected_problem
PATTERNS={
 'concurrency_async':r'\b(?:concurr\w*|asynchron\w*|async|threads?|threadpool\w*|deadlock\w*|semaphor\w*|race condition|taskgroup|scheduler|cancell?ation|retry|retries)\b',
 'persistence_recovery':r'\b(?:persist\w*|durab\w*|sqlite|database|transactions?|checkpoint\w*|backfill|fsync|write.ahead|wal|crash recovery|recover\w*|idempoten\w*|migration|backup|restore|event ledger)\b',
 'scientific_numeric':r'\b(?:energy distance|energy statistic|median|numerical|numpy|pandas|dataframe|ndarray|matrix|matrices|covariance|regression|scipy|simulation|statistical|monte carlo|signal processing|chemical|smiles|descriptors?|finite element|differential equation|tensor|quantitative|time.series|backtest\w*|portfolio|risk model|physical model)\b',
 'api_platform_security':r'\b(?:apis?|sdk|http|oauth|authenticat\w*|authoriz\w*|rbac|credentials?|permissions?|azure|aws|ec2|kubernetes|starknet|blockchain|protocol|network state|cloud|endpoint|server|client|socket|firestore|graphql|security|ssl|tls|mfa)\b',
 'memory_parsing_state':r'\b(?:memory|ownership|pointer|buffer|copy|clone|snapshot|serializ\w*|deserializ\w*|parse\w*|parsing|grammar|lexer|tokenizer|bitwise|codec|image|palette|json|xml|toml|binary|state machine)\b',
 'build_package_debug':r'\b(?:cmake|toolchain|linker|compiler|compilation|compile|packaging|package|dependencies|dependency|installation|install|build|distribution|wheel|cargo|pip|conda|debugg\w*)\b',
 'evidence_research':r'\b(?:literature|research paper|scientific paper|academic|pubmed|arxiv|doi|citation|documentation|technical reference|clinical trial|machine learning research)\b'
}
SCIENCE={
 'statistical_numeric':r'\b(?:variance|covariance|probability|regression|estimator|hypothesis|likelihood|confidence interval|statistical|numerical|eigenvalue|matrix|matrices)\b',
 'physical_science':r'\b(?:quantum|velocity|acceleration|momentum|kinetic|thermodynamic|electromagnetic|magnetic|electric field|force|mass|energy|oscillat\w*|wave|photon|relativ\w*)\b',
 'chemical_biological':r'\b(?:stoichiometr\w*|molar|molecule|molecular|chemical|chemistry|reaction|equilibrium|acid|base|ph|enzyme|protein|cell|gene|biology)\b'
}
def observed_tools(raw):
 tools=set()
 for m in raw.get('messages',[]):
  calls=m.get('tool_calls')or[]
  if isinstance(calls,str):calls=json.loads(calls)
  for c in calls:
   if isinstance(c,dict):tools.add(c.get('function',{}).get('name','').lower())
 return tools
TOOL_PATTERNS={
 'api_platform_security':r'\b(?:apis?|sdk|http|oauth|authenticat\w*|rbac|credentials?|azure|aws|ec2|kubernetes|starknet|blockchain|firestore|graphql|ssl|tls|mfa|cloud infrastructure|cloud computing|network security|cybersecurity|server deployment)\b',
 'persistence_recovery':r'\b(?:database|sqlite|postgres\w*|sql server|mongodb|mysql|redis|backup|restore|write.ahead|fsync|data migration|disaster recovery)\b',
 'build_package_debug':r'\b(?:cmake|toolchain|linker|compiler|compilation|python package|pip install|conda|npm|cargo|software installation|software build|debugging|stack trace|dependency conflict|unit tests?|segfault)\b',
 'scientific_numeric':r'\b(?:numpy|pandas|scipy|covariance|regression|monte carlo|finite element|differential equation|quantitative|time.series|backtest\w*|risk model|smiles|molecular|molecules|chemical reaction|chemical structure|compound descriptors|statistical (?:analysis|test|model)|portfolio (?:optimization|rebalance|risk)|vix|protein structure|neural network)\b',
 'evidence_research':r'\b(?:pubmed|arxiv|scientific papers?|research papers?|scientific literature|clinical trials?|machine learning research|systematic review)\b'
}
def classify(raw,st):
 text=task_text(raw);tools=observed_tools(raw)
 if st in ['cpp','python']:return 'algorithmic_reasoning',['algorithmic_reasoning'],None
 if st=='stem':
  tags=[k for k,p in SCIENCE.items()if re.search(p,text,re.I)]
  return (tags[0],tags,None)if tags else(None,[],'outside_quantitative_science_families')
 if st=='swe':
  # URLs and general acknowledgement lines are not the engineering objective.
  goal=re.sub(r'https?://\S+','',text)
  goal='\n'.join(line for line in goal.splitlines()if not re.search(r'thank.*(?:package|library|work)',line))
  tags=[k for k,p in PATTERNS.items()if k!='evidence_research'and re.search(p,goal,re.I)]
  return (tags[0],tags,None)if tags else(None,[],'unclassified_engineering_issue')
 if any('lotuswisdom'in t for t in tools):return None,[],'outside_engineering_scientific_quantitative_workload'
 meta=re.compile(r'thought|thinking|mentalmodel|metacogniti|decisionframework|visualreasoning|designpattern|programmingparadigm|debuggingapproach')
 if tools and all(meta.search(t)or t=='think'for t in tools):return None,[],'thought_only_tool_workflow'
 tags=[k for k,p in TOOL_PATTERNS.items()if re.search(p,text,re.I)]
 # Persistence/installation require an actual developer/operations context.
 if 'persistence_recovery'in tags and not re.search(r'database|sqlite|postgres|mysql|mongodb|redis|server|data (?:backup|restore|migration)|backup (?:job|schedule|throughput|restore)|disaster recovery|sql|infrastructure',text,re.I):tags.remove('persistence_recovery')
 if 'scientific_numeric'in tags and 'persistence_recovery'in tags and not re.search(r'backup|restore|recover|migration|transaction|durability|fsync|write.ahead',text,re.I):tags.remove('persistence_recovery')
 allowed=list(POLICY['public_quotas_per_partition']['general_tool_use'])
 tags=[t for t in tags if t in allowed]
 if 'evidence_research'in tags:tags=['evidence_research']+[t for t in tags if t!='evidence_research']
 return (tags[0],tags,None)if tags else(None,[],'outside_engineering_scientific_quantitative_workload')
def owner_table(base):
 owner={}
 for part,rows in base.items():
  for r in rows:
   g=r['split_group'].lower()
   if g in owner and owner[g]!=part:raise ValueError('Existing group ownership conflict '+g)
   owner[g]=part
 # Historical public partitions stay fixed, including records not selected into this revision.
 for r in map(json.loads,(ROOT/'nvidia-calibration-v4/selected.jsonl').open()):
  g=r['split_group'].lower();part=r['partition']
  if g in owner and owner[g]!=part:raise ValueError('Historical ownership conflict '+g)
  owner[g]=part
 return owner
def partition(group,owners):
 return owners.get(group.lower(),PARTS[int(sha([POLICY['seed'],'partition',group.lower()])[:8],16)%2])
def bucket(n):
 if n<8192:return 'under8k'
 if n<16384:return '8-16k'
 if n<32768:return '16-32k'
 return '32-64k'
def lineage_overlap(parts):
 overlap={}
 for key in ['sample_id','task_key','prompt_sha256','token_sha256','split_group']:
  sets=[{str(r.get(key,'')).lower()for r in parts[p]if r.get(key)}for p in PARTS]
  overlap[key]=sorted(sets[0]&sets[1])
 for key in ['lineage_aliases','source_papers','dataset_groups']:
  sets=[{str(v).lower()for r in parts[p]for v in r.get(key,[])}for p in PARTS]
  overlap[key]=sorted(sets[0]&sets[1])
 return overlap
def anchor_labels(r):
 st=r.get('calibration_stratum','');mech=r.get('workflow_mechanisms',[])
 tags=[]
 if 'concurrency'in mech:tags.append('concurrency_async')
 if 'persistence'in mech:tags.append('persistence_recovery')
 if 'memory_state'in mech:tags.append('memory_parsing_state')
 if 'platform_integration'in mech:tags.append('api_platform_security')
 if any(x in mech for x in ['installation','debugging']):tags.append('build_package_debug')
 if 'api_research'in mech:tags.append('api_platform_security')
 if r['domain']=='finance':tags+=['quantitative_research']
 if 'literature'in st:tags+=['evidence_research']
 if 'physics'in st:tags+=['physical_science']
 if any(x in st for x in ['chemistry','biology']):tags+=['chemical_biological']
 if any(x in st for x in ['statistics','notebook','optimization']):tags+=['statistical_numeric','scientific_numeric']
 if any(x in st for x in ['reasoning_code','algorithmic_reasoning']):tags+=['algorithmic_reasoning']
 # Issue/body classification supplies multi-label coverage, not authoritative expert semantics.
 _,inferred,_=classify(r,'swe'if r['domain']=='coding'else'tool_calling')
 return sorted(set(tags+inferred))
