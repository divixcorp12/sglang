"""Protocol/guard tests. Synthetic fixtures are never production evidence."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
from unittest.mock import patch

import pytest
import torch
import production_incremental as p


def record(sid, domain):
    ids=[1,2]
    return dict(sample_id=sid,domain=domain,valid_tokens=2,token_ids=ids,token_sha256=hashlib.sha256(b'[1,2]').hexdigest(),split_group=sid)


def scored(row,scale=1):
    return {**{k:row[k] for k in p.META},'gate_weighted_output_norm_sum':[scale*(384-i) for i in range(384)],'route_counts':[16]+[0]*383}


def producer(root,out,model,kind,part):
    out.mkdir(parents=True)
    groups=[]; samples={}
    for domain in p.DOMAINS:
        row=record(f'{part}-{kind}-{domain}',domain); samples[row['sample_id']]=row
        path=out/'inputs'/(domain+'.json')
        p.save(path,dict(domain=domain,samples=[row]))
        groups.append(dict(domain=domain,manifest=str(path)))
    plan=out/'inputs/plan.json';p.save(plan,dict(groups=groups,domain_weights=dict(finance=.3,science=.3,coding=.3,general_tool_use=.1)))
    files={str(model/n):p.sha(model/n) for n in ('config.json','model.safetensors.index.json')}
    files.update({path:p.sha(path) for path in p.plan_files(root,plan)})
    stat=(model/'fixture.safetensors').stat()
    identity=dict(files=files,layers=70,precision='sglang_sm120',invariant='observation_only; propagate_error=false; original source weights unchanged',source_shard_stats={'fixture.safetensors':dict(size=stat.st_size,mtime_ns=stat.st_mtime_ns)},torch='synthetic_protocol_fixture',cuda='none',gpu='none')
    p.save(out/'checkpoint-identity.json',identity)
    layers={}; captured={}; written={}
    for layer in range(1,70):
        rows=[scored(row,1 if kind=='old' else 2) for row in samples.values()]
        path=out/'trajectories'/f'layer-{layer:02d}.json'
        p.save(path,dict(layer=layer,records=rows,aggregate_check_passed=True))
        written[str(layer)]=dict(path=str(path),sha256=p.sha(path),episodes=len(rows))
        layers[str(layer)]={}
        for row in rows:
            stat={k:row[k] for k in ('gate_weighted_output_norm_sum','route_counts')}
            stat['routed_assignments']=16
            captured[f'layers.{layer}.feed_forward_stage.mlp::{row["domain"]}']=stat
            layers[str(layer)][row['domain']]=dict(valid_tokens=2,**stat)
    folder=out/'checkpoints/generation-0141-fixture';folder.mkdir(parents=True)
    torch.save(dict(observer=dict(captured=captured,written=written,completed=141)),folder/'state.pt')
    payloads={'state.pt':dict(bytes=(folder/'state.pt').stat().st_size,sha256=p.sha(folder/'state.pt'))}
    for i in range(4):
        path=folder/f'batch-{i:05d}.pt';path.write_bytes(b'protocol-fixture-only')
        payloads[path.name]=dict(bytes=path.stat().st_size,sha256=p.sha(path))
    p.save(folder/'manifest.json',dict(schema='mimo_sequential_checkpoint_v1',offload_device='cpu',identity=identity,graph=[{}]*141,batches=4,next_stage=141,files=payloads,previous=None))
    p.save(out/'checkpoints/latest.json',dict(generation=folder.name,manifest_sha256=p.sha(folder/'manifest.json')))
    p.save(out/'trajectory-index.json',dict(status='completed',layers=written))
    observation=dict(config_sha256=files[str(model/'config.json')],index_sha256=files[str(model/'model.safetensors.index.json')],routing='mimo_sigmoid_bias_selection_normalized_unbiased_weights',precision_profile='sglang_sm120',sample_ids=sorted(samples),layers={k:{d:{f:s[f] for f in ('valid_tokens','route_counts','gate_weighted_output_norm_sum')} for d,s in domains.items()} for k,domains in layers.items()},native_precision_verified=False,selection_allowed=False,synthetic=False,full_stack_completed=True,calibration_manifest_sha256=p.sha(plan))
    p.save(out/'observations.json',observation)
    p.save(out/'report.json',dict(status='passed',layers=70,precision_profile='sglang_sm120',native_precision_verified=False,selection_allowed=False,sample_ids=sorted(samples),statistics=layers,offload_new_files=0,identity=dict(plan_sha256=p.sha(plan)),domain_tokens={d:2 for d in p.DOMAINS}))
    return plan,samples,identity


@pytest.fixture
def production(tmp_path):
    root=tmp_path/'mimo-reap';root.mkdir()
    model=tmp_path/'mimi-v26-pro';model.mkdir()
    for name in ('config.json','model.safetensors.index.json','fixture.safetensors'):
        (model/name).write_text('{}')
    out=root/'incremental-fixture';out.mkdir()
    instance=p.Production(root,out)
    instance.parent=root/'full-scoring-v11'
    actual=Path('/mnt/nvme4/mimo-reap')
    for name in ('full-scoring-v11/common.py','full-scoring-v3/common.py','rank_stability.py'):
        target=root/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(actual/name,target)
    lock=dict(keep=344,minimum_jaccard=.95,graph=[{}]*141,pins={},partitions={},storage=dict(required_total_bytes=1024))
    for part in p.PARTS:
        old_plan,old,identity=producer(root,instance.parent/part,model,'old',part)
        new_plan,new,_=producer(root,out/part,model,'new',part)
        combined=out/'inputs'/part/'combined.json'
        groups=[]
        for domain in p.DOMAINS:
            path=combined.parent/(domain+'.json')
            p.save(path,dict(domain=domain,samples=[s for s in list(old.values())+list(new.values()) if s['domain']==domain]))
            groups.append(dict(domain=domain,manifest=str(path)))
        p.save(combined,dict(groups=groups,domain_weights=read_weights(old_plan)))
        lock['partitions'][part]=dict(parent_plan=str(old_plan),plan=str(new_plan),combined_plan=str(combined))
        lock['compute']=p.compute_identity(root,identity,old_plan)
    p.save(out/'inputs.lock.json',lock)
    p.save(out/'preparation.json',dict(status='passed',inputs_lock_sha256=p.sha(out/'inputs.lock.json')))
    return instance


def read_weights(path):
    return p.read(path)['domain_weights']


@pytest.mark.parametrize('change', ['duplicate','missing','negative','nan','fractional_count','wrong_token','wrong_domain','wrong_group','wrong_count','wrong_shape'])
def test_reject_bad_trajectory(change):
    sample=record('fixture','finance');rows=[scored(sample)]
    if change=='duplicate': rows*=2
    if change=='missing': rows=[]
    if change=='negative': rows[0]['gate_weighted_output_norm_sum'][0]=-1
    if change=='nan': rows[0]['gate_weighted_output_norm_sum'][0]=float('nan')
    if change=='fractional_count': rows[0]['route_counts'][0]=15.5
    if change=='wrong_token': rows[0]['token_sha256']='bad'
    if change=='wrong_domain': rows[0]['domain']='science'
    if change=='wrong_group': rows[0]['split_group']='other'
    if change=='wrong_count': rows[0]['route_counts'][0]=8
    if change=='wrong_shape': rows[0]['route_counts'].pop()
    with pytest.raises(ValueError): p.validate_records(rows,{'fixture':sample})


def test_full_merge_and_recovery_metadata_survives_checkpoint_retirement(production):
    early=production.merge(32);final=production.merge(69)
    assert early['early_gate_passed'] and len(early['jaccard_by_layer_domain'])==128
    assert final['rank_stability_verified'] and len(final['jaccard_by_layer_domain'])==276
    target=production.out/'merges/full-69'
    obs=p.read(target/'primary/observations.json')
    assert len(obs['sample_ids'])==8 and obs['layers']['1']['finance']['valid_tokens']==4
    assert obs['layers']['1']['finance']['gate_weighted_output_norm_sum'][0]==1152
    assert obs['layers']['1']['finance']['route_counts'][0]==32
    assert obs['native_precision_verified'] is False and obs['selection_allowed'] is False
    # Only synthetic fixture generations are removed. Snapshot proofs must live on.
    for part in p.PARTS:
        shutil.rmtree(production.out/part/'checkpoints/generation-0141-fixture')
    production.verify_merge(target)
    assert production.merge(32)==early


@pytest.mark.parametrize('mutation',['changed_precision','changed_code','changed_weights','bad_stage','bad_graph','bad_state_hash','forged_gate','report_mismatch','corrupt_trajectory','missing_layer'])
def test_reject_bad_producer(production,mutation):
    path=production.out/'primary';plan=production.locked()['partitions']['primary']['plan'];lock=production.locked()
    identity=p.read(path/'checkpoint-identity.json')
    if mutation=='changed_precision':
        identity['precision']='decoded_bf16';p.save(path/'checkpoint-identity.json',identity)
    elif mutation=='changed_code':
        identity['torch']='other';p.save(path/'checkpoint-identity.json',identity)
    elif mutation=='changed_weights':
        identity['source_shard_stats']['fixture.safetensors']['size']+=1;p.save(path/'checkpoint-identity.json',identity)
    elif mutation=='bad_stage':
        folder,manifest=production.header(path);manifest['next_stage']=65;p.save(folder/'manifest.json',manifest);p.save(path/'checkpoints/latest.json',dict(generation=folder.name,manifest_sha256=p.sha(folder/'manifest.json')))
    elif mutation=='bad_graph':
        folder,manifest=production.header(path);manifest['graph'][0]={'unexpected':'stage'};p.save(folder/'manifest.json',manifest);p.save(path/'checkpoints/latest.json',dict(generation=folder.name,manifest_sha256=p.sha(folder/'manifest.json')))
    elif mutation=='bad_state_hash':
        folder,_=production.header(path);(folder/'state.pt').write_bytes(b'corrupt')
    elif mutation=='forged_gate':
        obs=p.read(path/'observations.json');obs['selection_allowed']=True;p.save(path/'observations.json',obs)
    elif mutation=='report_mismatch':
        report=p.read(path/'report.json');report['domain_tokens']['finance']+=1;p.save(path/'report.json',report)
    elif mutation=='corrupt_trajectory':
        (path/'trajectories/layer-01.json').write_text('{}')
    elif mutation=='missing_layer':
        (path/'trajectories/layer-69.json').unlink()
    with pytest.raises((ValueError,FileNotFoundError)):
        production.evidence(path,plan,69,lock)


def test_overlap_rejected(production):
    lock=production.locked();config=lock['partitions']['primary']
    old,_,_=production.evidence(production.parent/'primary',config['parent_plan'],32,lock)
    with pytest.raises(ValueError,match='Overlapping'):
        p.merge_layers(old,old,32)


def test_sealed_output_corruption_rejected(production):
    production.merge(32)
    target=production.out/'merges/prefix-32'
    (target/'primary/observations.json').write_text('{}')
    with pytest.raises(ValueError,match='Merged artifact changed'):
        production.verify_merge(target)


def test_interrupted_merge_preserved_and_rebuilt(production):
    pending=production.out/'merges/prefix-32.pending';pending.mkdir(parents=True)
    (pending/'partial').write_text('preserve interrupted output')
    production.merge(32)
    assert list((production.out/'merges/uncommitted').glob('*/partial'))


def test_layer32_failure_prevents_full_scoring(production):
    with patch.object(production,'storage',return_value={}),patch.object(production,'at_boundary',return_value=True),patch.object(production,'merge',return_value=dict(early_gate_passed=False)) as merge,patch.object(production,'observe') as observe,patch.object(p.signal,'signal'):
        (production.root/'performance-v1').mkdir()
        production.controller()
        assert merge.call_args.args==(32,)
        observe.assert_not_called()
        assert p.read(production.out/'state.json')['status']=='calibration_failed'


def test_early_pass_continues_both_partitions_and_preserves_pruning_gates(production):
    gates=[dict(early_gate_passed=True,minimum_jaccard=1),dict(rank_stability_verified=True)]
    for part in p.PARTS:
        (production.out/part/'report.json').unlink()
    with patch.object(production,'storage',return_value={}),patch.object(production,'at_boundary',return_value=True),patch.object(production,'merge',side_effect=gates),patch.object(production,'observe',return_value=0) as observe,patch.object(p.signal,'signal'):
        (production.root/'performance-v1').mkdir()
        production.controller()
        assert [call.args[:2] for call in observe.call_args_list]==[('primary',False),('repeat',False)]
        result=p.read(production.out/'post-calibration.json')
        assert result['status']=='reap_prerequisites_pending' and result['selection_allowed'] is False


def test_user_pause_prevents_launch(production):
    p.save(production.out/'pause.request',dict(user=True))
    with patch.object(production,'storage',return_value={}),patch.object(production,'observe') as observe,patch.object(p.signal,'signal'):
        (production.root/'performance-v1').mkdir()
        production.controller();observe.assert_not_called()
        assert p.read(production.out/'state.json')['status']=='paused'


def test_reducer_against_real_bounded_gpu_outputs():
    root=Path('/mnt/nvme4/mimo-reap/incremental-scoring-v1/runs')
    for part in p.PARTS:
        docs={mode:p.read(root/part/mode/'observations.json') for mode in ('joined','old','new')}
        merged=p.merge_layers(docs['old'],docs['new'],1)
        for domain in p.DOMAINS:
            a=merged['layers']['1'][domain];b=docs['joined']['layers']['1'][domain]
            assert a['valid_tokens']==b['valid_tokens'] and a['route_counts']==b['route_counts']
            assert all(math_close(x,y) for x,y in zip(a['gate_weighted_output_norm_sum'],b['gate_weighted_output_norm_sum']))
            assert sorted(range(384),key=lambda i:(-a['gate_weighted_output_norm_sum'][i],i))==sorted(range(384),key=lambda i:(-b['gate_weighted_output_norm_sum'][i],i))


def math_close(a,b):
    import math
    return math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-8)
