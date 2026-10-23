"""Independent invariants for the public-only tool revision; no score inputs."""
import os
os.environ["CUDA_VISIBLE_DEVICES"]=""
import sys,json,re,collections,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=Path(__file__).resolve().parent
sys.path[:0]=[str(OUT),str(ROOT/"calibration-mix-v3"),str(ROOT/"calibration-gap-v1")]
import inventory as I
from scan_terminal import action
C=I.C;P=I.P
def structured_observations(messages):
 pending={};n=0
 for m in messages:
  if m["role"]=="assistant":
   if pending:raise ValueError("assistant_before_observation")
   for t in m.get("tool_calls")or[]:
    cid=t.get("id");name=t.get("function",{}).get("name")
    if not cid or cid in pending:raise ValueError("invalid_tool_identity")
    pending[cid]=name
  elif m["role"]=="tool":
   cid=m.get("tool_call_id")
   if cid not in pending:raise ValueError("unmatched_observation")
   if m.get("name")and m["name"]!=pending[cid]:raise ValueError("tool_name_changed")
   del pending[cid];n+=1
 if pending and set(pending.values())!={"finish"}:raise ValueError("unobserved_final_command")
 return n
def native_observations(r):
 msgs=r["messages"]
 if r["stratum"]=="terminal":
  final=action(msgs[-1]["content"])
  if not isinstance(final,dict)or final.get("task_complete")is not True or final.get("commands"):raise ValueError("incomplete_terminal")
  count=0
  for i,m in enumerate(msgs[:-1]):
   if m["role"]!="assistant":continue
   try:a=action(m["content"])
   except(ValueError,TypeError):continue
   if isinstance(a,dict)and a.get("commands"):
    if msgs[i+1]["role"]!="user"or not msgs[i+1]["content"].startswith("New Terminal Output:"):raise ValueError("unobserved_native_action")
    if not all(isinstance(c,dict)and isinstance(c.get("keystrokes"),str)for c in a["commands"]):raise ValueError("malformed_commands")
    count+=1
  return count
 if [m["role"]for m in msgs]!=["system","user","assistant"]:raise ValueError("research_shape")
 body=msgs[-1]["content"];count=len(re.findall(r"<call_tool\b",body))
 if count!=body.count("<tool_output>")or count!=body.count("</tool_output>"):raise ValueError("missing_research_result")
 if not body.rstrip().endswith("</answer>")or body.count("<answer>")!=1:raise ValueError("unfinished_research")
 defined=set(re.findall(r'<(?:snippet|webpage)\s+id=["\x27]?([^"\x27>\s]+)',body))
 cites={v.strip()for x in re.findall(r'<cite\s+id=["\x27]([^"\x27]+)',body)for v in x.split(",")}
 if not cites or not cites<=defined:raise ValueError("undefined_research_citation")
 answer=body.split("<answer>",1)[1]
 depth=0
 for tag in re.findall(r"<cite\b[^>]*>|</cite>",answer):
  depth+= -1 if tag=="</cite>"else 1
  if depth not in [0,1]:raise ValueError("malformed_citation_markup")
 if depth:raise ValueError("malformed_citation_markup")
 return count
def record(r):
 if r.get("dataset")not in P["sources"]:raise ValueError("not_public_source")
 if r.get("domain")!="general_tool_use"or r.get("workflow_family")not in P["families"]:raise ValueError("invalid_scope")
 if r.get("truncated")or not P["minimum_tokens"]<=r["valid_tokens"]<=P["maximum_tokens"]:raise ValueError("invalid_length")
 if len(r["token_ids"])!=r["valid_tokens"]or C.token_hash(r["token_ids"])!=r["token_sha256"]:raise ValueError("token_hash")
 n=structured_observations(r["messages"])if r["stratum"]=="swe"else native_observations(r)
 if n<P["minimum_observed_action_results"]:raise ValueError("insufficient_observations")
 if r.get("selection_allowed")or r.get("native_precision_verified"):raise ValueError("scope_promoted")
 return n
def review_bound(r,decision):
 if decision["decision"]!="admit":raise ValueError("not_admitted")
 for k in ["sample_id","source_record_sha256","source_file","source_row","partition","workflow_family"]:
  if decision[k]!=r[k]:raise ValueError("review_binding_"+k)
 if decision["policy_sha256"]!=C.file_sha(OUT/"policy.json"):raise ValueError("review_policy_changed")
def preserved_non_tool(parent,mixed):
 old={p:[r for r in rows if r["domain"]!="general_tool_use"]for p,rows in parent.items()}
 actual={p:[r for r in rows if r["domain"]!="general_tool_use"]for p,rows in mixed.items()}
 if old!=actual:raise ValueError("non_tool_records_changed")
 return sum(map(len,old.values()))
def repo_caps(parent,selected):
 stats=collections.defaultdict(lambda:[0,0]);touched=set();owners={}
 for part,rows in parent.items():
  for r in rows:
   if r["domain"]=="general_tool_use":continue
   g=r["split_group"].lower()
   if g.startswith("repo:"):
    if g in owners and owners[g]!=part:raise ValueError("parent_repository_overlap")
    owners[g]=part;stats[g][0]+=1;stats[g][1]+=r["valid_tokens"]
 for r in selected:
  for g in [r["split_group"]]+r.get("lineage_aliases",[]):
   g=g.lower()
   if g.startswith("repo:"):
    if g in owners and owners[g]!=r["partition"]:raise ValueError("repository_cross_partition")
    owners[g]=r["partition"]
  g=r["split_group"].lower()
  if g.startswith("repo:"):stats[g][0]+=1;stats[g][1]+=r["valid_tokens"];touched.add(g)
 for g in touched:
  if stats[g][0]>P["public_repository_max_episodes"]or stats[g][1]>P["public_repository_max_tokens"]:raise ValueError("combined_repository_cap")
 baseline_exceptions={g:dict(episodes=n,tokens=t)for g,(n,t)in stats.items()if g not in touched and(n>P["public_repository_max_episodes"]or t>P["public_repository_max_tokens"])}
 return dict(touched_repository_totals={g:dict(episodes=stats[g][0],tokens=stats[g][1])for g in sorted(touched)},unchanged_baseline_only_groups_above_new_addition_cap=baseline_exceptions)
def partition_overlap(parts):
 out=C.lineage_overlap(parts)
 if any(out.values()):raise ValueError("partition_lineage_overlap")
 for p,rows in parts.items():
  for k in ["sample_id","task_key","prompt_sha256","token_sha256"]:
   values=[r[k]for r in rows if r.get(k)]
   if len(values)!=len(set(values)):raise ValueError("within_partition_duplicate_"+k)
 return out
