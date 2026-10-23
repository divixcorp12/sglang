"""Versioned additions-only MiMo scoring with immutable, checked evidence merging."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

PARTS = ('primary', 'repeat')
DOMAINS = ('finance', 'science', 'coding', 'general_tool_use')
META = ('sample_id', 'domain', 'valid_tokens', 'token_sha256', 'split_group')
FLAGS = ('native_precision_verified', 'selection_allowed')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.writing')
    with temporary.open('w') as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def verify_pins(pins):
    def verify(item):
        path, expected = item
        require(sha(path) == expected, 'Frozen file changed: ' + path)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(verify, pins.items()))


def sample_map(plan_path):
    plan = read(plan_path)
    result = {}
    for group in plan['groups']:
        document = read(group['manifest'])
        require(document['domain'] == group['domain'], 'Manifest domain mismatch')
        for sample in document['samples']:
            sid = sample['sample_id']
            require(sid not in result, 'Duplicate sample ID')
            require(sample['domain'] == group['domain'] and sample['valid_tokens'] == len(sample['token_ids']), 'Invalid token coverage')
            require(0 < sample['valid_tokens'] <= 65536, 'Invalid context length')
            digest = hashlib.sha256(json.dumps(sample['token_ids'], separators=(',', ':')).encode()).hexdigest()
            require(digest == sample['token_sha256'], 'Invalid token SHA')
            result[sid] = sample
    require({s['domain'] for s in result.values()} == set(DOMAINS), 'Incomplete domains')
    return result


def plan_files(root, plan_path):
    plan_path = Path(plan_path).resolve()
    return {str(plan_path), *(str((root/g['manifest']).resolve()) for g in read(plan_path)['groups'])}


def compute_identity(root, identity, plan_path):
    require(identity['layers'] == 70 and identity['precision'] == 'sglang_sm120', 'Incompatible compute depth/precision')
    require(identity['invariant'] == 'observation_only; propagate_error=false; original source weights unchanged', 'Observation invariant changed')
    data_paths = plan_files(root, plan_path)
    require(data_paths <= set(identity['files']), 'Unbound input plan')
    # Only the explicit input plan/manifests may differ. All policies, source
    # shards, model/tokenizer files, compute code, torch/CUDA and GPU must match.
    return {**identity, 'files': {p:h for p,h in identity['files'].items() if p not in data_paths}}


def validate_records(records, expected):
    require(len(records) == len(expected), 'Missing or duplicated trajectories')
    require({r['sample_id'] for r in records} == set(expected), 'Trajectory IDs differ')
    for row in records:
        require(all(row.get(k) == expected[row['sample_id']].get(k) for k in META), 'Trajectory metadata differs')
        scores, counts = row['gate_weighted_output_norm_sum'], row['route_counts']
        require(len(scores) == len(counts) == 384, 'Wrong expert count')
        require(all(isinstance(x, (int,float)) and not isinstance(x,bool) and math.isfinite(x) and x >= 0 for x in scores), 'Invalid saliency')
        require(all(isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x) and x >= 0 and int(x) == x for x in counts), 'Invalid route counts')
        require(sum(counts) == 8*row['valid_tokens'], 'Route coverage differs')


def validate_aggregate(records, layers, last):
    require(set(layers) == {str(i) for i in range(1,last+1)}, 'Missing aggregate layers')
    for layer in range(1,last+1):
        ds = layers[str(layer)]
        require(set(ds) == set(DOMAINS), 'Missing aggregate domains')
        for domain in DOMAINS:
            rows = [r for r in records[str(layer)] if r['domain'] == domain]
            stat = ds[domain]
            require(stat['valid_tokens'] == sum(r['valid_tokens'] for r in rows), 'Aggregate token mismatch')
            require(len(stat['route_counts']) == len(stat['gate_weighted_output_norm_sum']) == 384, 'Wrong aggregate shape')
            for i in range(384):
                target = stat['gate_weighted_output_norm_sum'][i]
                require(math.isfinite(target) and target >= 0, 'Invalid aggregate score')
                require(sum(r['route_counts'][i] for r in rows) == stat['route_counts'][i], 'Aggregate count mismatch')
                require(math.isclose(math.fsum(r['gate_weighted_output_norm_sum'][i] for r in rows), target, rel_tol=1e-12, abs_tol=1e-8), 'Aggregate score mismatch')
            require(sum(stat['gate_weighted_output_norm_sum']) > 0, 'Empty domain saliency')


def merge_layers(old, added, last):
    require(not set(old['sample_ids']) & set(added['sample_ids']), 'Overlapping parent sample IDs')
    for key in ('config_sha256','index_sha256','routing','precision_profile'):
        require(old[key] == added[key], 'Observation identity mismatch: '+key)
    require(all(old.get(k) is False and added.get(k) is False for k in FLAGS), 'Unexpected gate promotion')
    layers = {}
    for i in range(1,last+1):
        key = str(i)
        require(set(old['layers'][key]) == set(added['layers'][key]) == set(DOMAINS), 'Incomplete merge domains')
        layers[key] = {}
        for domain in DOMAINS:
            a,b = old['layers'][key][domain], added['layers'][key][domain]
            layers[key][domain] = dict(
                valid_tokens=a['valid_tokens']+b['valid_tokens'],
                route_counts=[x+y for x,y in zip(a['route_counts'],b['route_counts'],strict=True)],
                gate_weighted_output_norm_sum=[math.fsum((x,y)) for x,y in zip(a['gate_weighted_output_norm_sum'],b['gate_weighted_output_norm_sum'],strict=True)])
    return dict(schema='mimo_upstream_multi_domain_observation_v1', config_sha256=old['config_sha256'], index_sha256=old['index_sha256'], routing=old['routing'], precision_profile=old['precision_profile'], sample_ids=sorted(old['sample_ids']+added['sample_ids']), synthetic=False, full_stack_completed=last==69, native_precision_verified=False, selection_allowed=False, layers=layers)


class Production:
    def __init__(self, root, directory):
        self.root, self.out = Path(root).resolve(), Path(directory).resolve()
        require(self.out.is_relative_to(self.root), 'Output must be in the workspace')
        self.parent = self.root/'full-scoring-v11'
        self.revision = self.root/'calibration-mix-v8'
        self.child = None

    def pin(self, pins, path):
        path = Path(path).resolve()
        pins[str(path)] = sha(path)

    def prepare(self):
        require(not self.out.exists(), 'Use a fresh versioned run directory')
        pins = {}
        seal = read(self.revision/'release-seal.json')
        require(sha(self.revision/'evidence.lock.json') == seal['evidence_lock_sha256'], 'Invalid revision seal')
        for lock in ('evidence.lock.json','release-runtime.lock.json'):
            for path,h in read(self.revision/lock)['files'].items():
                require(sha(self.root/path) == h, 'Changed release artifact: '+path)
                self.pin(pins,self.root/path)
        for path in (self.revision/'release-seal.json',self.revision/'evidence.lock.json',self.root/'incremental-scoring-v1/completion.json',self.root/'rank_stability.py',self.parent/'common.py',Path(__file__)):
            self.pin(pins,path)
        proof = read(self.root/'incremental-scoring-v1/completion.json')
        require(proof['status'] == 'bounded_incremental_verification_passed', 'Missing split/join verification')
        for path,h in proof['artifacts_sha256'].items():
            require(sha(self.root/'incremental-scoring-v1'/path) == h, 'Verification evidence changed')
            self.pin(pins,self.root/'incremental-scoring-v1'/path)
        audit = read(self.revision/'final-audit.json')
        require(audit['status']=='passed' and audit['repository_and_lineage_separation'] is True and audit['parent_records_preserved']==621 and audit['additions']==46, 'Revision admission incomplete')
        for path,h in read(self.parent/'inputs.lock.json')['inputs_sha256'].items():
            require(sha(self.root/path)==h, 'Parent frozen input changed: '+path)
            self.pin(pins,self.root/path)
        self.pin(pins,self.parent/'inputs.lock.json')
        parent_rows = [json.loads(s) for s in (self.root/'calibration-mix-v7/all-selected.jsonl').read_text().splitlines()]
        rows = [json.loads(s) for s in (self.revision/'all-selected.jsonl').read_text().splitlines()]
        combined = {r['record']['sample_id']:r for r in rows}
        require(len(combined)==len(rows)==667, 'Wrong combined revision')
        require(all(combined.get(r['record']['sample_id']) == r for r in parent_rows), 'Parent examples or partitions changed')
        additions = [json.loads(s) for s in (self.revision/'selected-additions.jsonl').read_text().splitlines()]
        added_ids = {r.get('record',r)['sample_id'] for r in additions}
        require(len(added_ids)==46 and added_ids == set(combined)-{r['record']['sample_id'] for r in parent_rows}, 'Additions are not an exact set difference')
        partition_data = {}
        identities = {}
        graphs = {}
        for part in PARTS:
            old_plan = self.root/f'data/observer-manifests-representative-v7-{part}/plan.json'
            full_plan = self.root/seal['plans'][part]
            old = sample_map(old_plan)
            full = sample_map(full_plan)
            require(set(old) <= set(full), 'Original histories missing')
            require(all(all(full[s][k]==old[s][k] for k in (*META,'token_ids')) for s in old), 'Original token identity changed')
            new = {sid:r for sid,r in full.items() if sid not in old}
            require(len(new)==23 and set(new)<=added_ids, 'Wrong additions partition')
            require(set(full)=={sid for sid,r in combined.items() if r['partition']==part}, 'Combined partition differs')
            require(read(old_plan)['domain_weights'] == read(full_plan)['domain_weights'], 'Domain weights changed')
            for plan in (old_plan,full_plan):
                for path in plan_files(self.root,plan):
                    self.pin(pins,path)
            identity = read(self.parent/part/'checkpoint-identity.json')
            identities[part] = compute_identity(self.root,identity,old_plan)
            verify_pins(identity['files'])
            for path in identity['files']:
                self.pin(pins,path)
            for path in (self.parent/part).glob('trajectories/layer-*.json'):
                self.pin(pins,path)
            for name in ('checkpoint-identity.json','trajectory-index.json','observations.json','report.json','checkpoints/latest.json'):
                self.pin(pins,self.parent/part/name)
            pointer=read(self.parent/part/'checkpoints/latest.json')
            folder=self.parent/part/'checkpoints'/pointer['generation']
            graphs[part]=read(folder/'manifest.json')['graph']
            for name in ('manifest.json','state.pt'):
                self.pin(pins,folder/name)
            partition_data[part] = (old_plan,full_plan,old,full,new)
        require(identities['primary']==identities['repeat'], 'Parent compute differs across partitions')
        require(graphs['primary']==graphs['repeat'], 'Parent execution graphs differ')
        require(not set(partition_data['primary'][3]) & set(partition_data['repeat'][3]), 'Partitions overlap')
        self.out.mkdir(parents=True)
        lock = dict(schema='mimo_incremental_run_v1', parent=str(self.parent), revision=str(self.revision), keep=344, minimum_jaccard=.95, compute=identities['primary'], graph=graphs['primary'], partitions={}, pins=pins, native_precision_verified=False, selection_allowed=False)
        generations = {}
        for part,(old_plan,full_plan,old,full,new) in partition_data.items():
            groups=[]
            for domain in DOMAINS:
                template = next(read(g['manifest']) for g in read(full_plan)['groups'] if g['domain']==domain)
                samples = [s for s in new.values() if s['domain']==domain]
                require(bool(samples), 'Missing additions domain')
                path=self.out/'inputs'/part/(domain+'.json')
                save(path,dict(template,samples=samples,valid_tokens=sum(s['valid_tokens'] for s in samples),production_calibration_ready=False))
                groups.append(dict(domain=domain,manifest=str(path)))
                self.pin(pins,path)
            plan=self.out/'inputs'/part/'plan.json'
            save(plan,dict(schema='mimo_observation_execution_plan_v1',partition=part,groups=groups,domain_weights=read(full_plan)['domain_weights'],purpose='Frozen v8 additions only; fresh stage-zero checkpoints',native_precision_verified=False,selection_allowed=False))
            self.pin(pins,plan)
            lock['partitions'][part] = dict(parent_plan=str(old_plan),combined_plan=str(full_plan),plan=str(plan),episodes=len(new),sample_ids=sorted(new),domain_tokens={d:sum(s['valid_tokens'] for s in new.values() if s['domain']==d) for d in DOMAINS},combined_sample_ids=sorted(full))
            generations[part] = sum(s['valid_tokens'] for s in new.values())*6144*2
        # Two recovery generations per partition and one in-flight generation.
        reserve=math.ceil((2*sum(generations.values())+max(generations.values()))*1.1)+(8<<30)
        lock['storage'] = dict(required_total_bytes=reserve,generation_tensor_bytes=generations,retained_generations_per_partition=2,inflight_generations=1)
        save(self.out/'inputs.lock.json',lock)
        save(self.out/'state.json',dict(status='prepared',full_calibration_launched=False,partitions={p:lock['partitions'][p]['episodes'] for p in PARTS},**{k:False for k in FLAGS}))
        # Validate all parent statistics now; no GPU allocation or scoring.
        for part in PARTS:
            self.evidence(self.parent/part,lock['partitions'][part]['parent_plan'],69,lock)
        save(self.out/'preparation.json',dict(status='passed',inputs_lock_sha256=sha(self.out/'inputs.lock.json'),additions=46,combined_histories=667,storage=lock['storage'],source_weights_unchanged=True,selection_allowed=False))
        return lock

    def locked(self):
        lock=read(self.out/'inputs.lock.json')
        prepared=read(self.out/'preparation.json')
        require(prepared['status']=='passed' and prepared['inputs_lock_sha256']==sha(self.out/'inputs.lock.json'),'Preparation not verified')
        require(lock['keep']==344 and lock['minimum_jaccard']==.95, 'Gate configuration changed')
        verify_pins(lock['pins'])
        source=Path(next(p for p in lock['compute']['files'] if p.endswith('/mimi-v26-pro/config.json'))).parent
        # Source location comes from the pinned model config, not an inferred alias.
        for name, expected in lock['compute']['source_shard_stats'].items():
            stat=(source/name).stat()
            require(stat.st_size==expected['size'] and stat.st_mtime_ns==expected['mtime_ns'], 'Weight shard identity changed')
        return lock

    def header(self,path):
        pointer=read(path/'checkpoints/latest.json')
        name=pointer['generation']
        require(Path(name).name==name and name.startswith('generation-'), 'Invalid checkpoint pointer')
        folder=path/'checkpoints'/name
        require(sha(folder/'manifest.json')==pointer['manifest_sha256'], 'Invalid checkpoint manifest')
        manifest=read(folder/'manifest.json')
        require(manifest['schema']=='mimo_sequential_checkpoint_v1' and manifest['offload_device']=='cpu','Checkpoint schema/offload differs')
        require(manifest['identity']==read(path/'checkpoint-identity.json'), 'Checkpoint identity mismatch')
        return folder,manifest

    def evidence(self,path,plan,last,lock):
        import torch
        path,plan=Path(path),Path(plan)
        identity=read(path/'checkpoint-identity.json')
        require(compute_identity(self.root,identity,plan)==lock['compute'], 'Producer compute identity differs')
        verify_pins(identity['files'])
        expected=sample_map(plan)
        folder,manifest=self.header(path)
        require(len(manifest['graph'])==141 and manifest['batches']==len(expected), 'Checkpoint graph/batch mismatch')
        require(manifest['graph']==lock['graph'],'Producer execution graph differs')
        require(manifest['next_stage']>=2*(last+1)+1, 'Required layers not committed')
        require(sha(folder/'state.pt')==manifest['files']['state.pt']['sha256'], 'Corrupt committed observer state')
        state=torch.load(folder/'state.pt',map_location='cpu',weights_only=True)['observer']
        require(state['completed']==manifest['next_stage'], 'Observer stage mismatch')
        written=state['written']
        require(set(manifest['files'])=={'state.pt',*(f'batch-{i:05d}.pt' for i in range(len(expected)))},'Checkpoint inventory differs')
        require(set(state['captured'])=={f'layers.{i}.feed_forward_stage.mlp::{d}' for i in range(1,(manifest['next_stage']-1)//2) for d in DOMAINS},'Unexpected captured statistics')
        committed_layers=(manifest['next_stage']-1)//2
        require(set(written)=={str(i) for i in range(1,committed_layers)}, 'Committed trajectory coverage differs')
        records,layers={},{}
        for i in range(1,last+1):
            key=str(i);meta=written[key];p=Path(meta['path'])
            require(p.resolve().parent==(path/'trajectories').resolve() and p.name==f'layer-{i:02d}.json', 'Trajectory path differs')
            require(sha(p)==meta['sha256'], 'Trajectory checksum differs')
            doc=read(p)
            require(doc['layer']==i and doc['aggregate_check_passed'] is True, 'Invalid trajectory layer')
            validate_records(doc['records'],expected)
            records[key]=doc['records']
            layers[key]={}
            for domain in DOMAINS:
                stat=state['captured'][f'layers.{i}.feed_forward_stage.mlp::{domain}']
                tokens=sum(s['valid_tokens'] for s in expected.values() if s['domain']==domain)
                require(stat['routed_assignments']==8*tokens,'Committed routed assignment mismatch')
                layers[key][domain]=dict(valid_tokens=tokens,**stat)
        validate_aggregate(records,layers,last)
        if last==69:
            require(manifest['next_stage']==141, 'Incomplete full checkpoint')
            report,obs,index=[read(path/n) for n in ('report.json','observations.json','trajectory-index.json')]
            require(report['status']=='passed' and report['layers']==70 and index['status']=='completed', 'Incomplete producer report')
            require(set(index['layers'])==set(written) and index['layers']==written, 'Report trajectory index differs')
            require(obs['full_stack_completed'] is True and obs['synthetic'] is False, 'Incomplete/non-real observations')
            require(all(obs.get(k) is False and report.get(k) is False for k in FLAGS), 'Producer promoted a gate')
            require(len(obs['sample_ids'])==len(report['sample_ids'])==len(expected) and set(obs['sample_ids'])==set(expected)==set(report['sample_ids']), 'Producer sample IDs differ')
            require(obs['layers']=={k:{d:{f:s[f] for f in ('valid_tokens','route_counts','gate_weighted_output_norm_sum')} for d,s in v.items()} for k,v in layers.items()}, 'Final observations differ from committed state')
            require(report['precision_profile']==identity['precision'] and obs['precision_profile']==identity['precision'], 'Report precision differs')
            require(report['offload_new_files']==0 and report['identity']['plan_sha256']==sha(plan) and obs['calibration_manifest_sha256']==sha(plan),'Final report plan/offload differs')
            require(report['domain_tokens']=={d:layers['1'][d]['valid_tokens'] for d in DOMAINS},'Report tokens differ')
            require(report['statistics']==layers,'Report statistics differ')
            require(obs['routing']=='mimo_sigmoid_bias_selection_normalized_unbiased_weights','Report routing differs')
            require(obs['config_sha256']==identity['files'][next(p for p in identity['files'] if p.endswith('/mimi-v26-pro/config.json'))] and obs['index_sha256']==identity['files'][next(p for p in identity['files'] if p.endswith('/mimi-v26-pro/model.safetensors.index.json'))],'Report source differs')
        model_config=next(p for p in identity['files'] if p.endswith('/mimi-v26-pro/config.json'))
        model_index=str(Path(model_config).with_name('model.safetensors.index.json'))
        observation=dict(config_sha256=identity['files'][model_config],index_sha256=identity['files'][model_index],routing='mimo_sigmoid_bias_selection_normalized_unbiased_weights',precision_profile=identity['precision'],sample_ids=sorted(expected),layers=layers,native_precision_verified=False,selection_allowed=False)
        receipt=dict(plan=str(plan),plan_sha256=sha(plan),checkpoint_manifest=str(folder/'manifest.json'),checkpoint_manifest_sha256=sha(folder/'manifest.json'),state_sha256=sha(folder/'state.pt'),checkpoint_identity_sha256=sha(path/'checkpoint-identity.json'),trajectory_hashes={str(i):written[str(i)]['sha256'] for i in range(1,last+1)})
        return observation,records,receipt

    def merge(self,last):
        require(last in (32,69),'Unsupported merge boundary')
        lock=self.locked()
        target=self.out/'merges'/('prefix-32' if last==32 else 'full-69')
        if target.exists():
            self.verify_merge(target)
            return read(target/('early-gate.json' if last==32 else 'rank-stability.json'))
        temporary=target.with_name(target.name+'.pending')
        if temporary.exists():
            archive=self.out/'merges'/'uncommitted'/str(time.time_ns())
            archive.parent.mkdir(parents=True,exist_ok=True)
            os.replace(temporary,archive)
        temporary.mkdir(parents=True)
        observations={}
        receipts={}
        for part in PARTS:
            config=lock['partitions'][part]
            old,old_rows,old_receipt=self.evidence(self.parent/part,config['parent_plan'],last,lock)
            added,new_rows,new_receipt=self.evidence(self.out/part,config['plan'],last,lock)
            for index,receipt in enumerate((old_receipt,new_receipt)):
                snapshot=temporary/part/'parents'/str(index)
                snapshot.mkdir(parents=True)
                original=Path(receipt['checkpoint_manifest'])
                shutil.copyfile(original,snapshot/'manifest.json')
                shutil.copyfile(original.parent/'state.pt',snapshot/'state.pt')
                require(sha(snapshot/'manifest.json')==receipt['checkpoint_manifest_sha256'] and sha(snapshot/'state.pt')==receipt['state_sha256'],'Parent changed while snapshotting')
                receipt['snapshot_manifest']=str((snapshot/'manifest.json').relative_to(temporary))
                receipt['snapshot_state']=str((snapshot/'state.pt').relative_to(temporary))
            result=merge_layers(old,added,last)
            expected=sample_map(config['combined_plan'])
            require(result['sample_ids']==sorted(expected),'Combined coverage differs')
            combined_rows={key:old_rows[key]+new_rows[key] for key in old_rows}
            for key,rows in combined_rows.items():
                validate_records(rows,expected)
                save(temporary/part/'trajectories'/f'layer-{int(key):02d}.json',dict(schema='mimo_per_trajectory_saliency_v1',layer=int(key),records=rows,aggregate_check_passed=True,method='Immutable concatenation of parent records; raw domain accumulators added'))
            validate_aggregate(combined_rows,result['layers'],last)
            result.update(calibration_manifest_sha256=sha(config['combined_plan']),identity=dict(kind='incremental_union',parents=[old_receipt,new_receipt],inputs_lock_sha256=sha(self.out/'inputs.lock.json'),domain_weights=read(config['combined_plan'])['domain_weights']))
            save(temporary/part/'observations.json',result)
            observations[part]=result;receipts[part]=[old_receipt,new_receipt]
        if last==32:
            legacy=module(self.parent/'common.py','incremental_legacy_common')
            gate=legacy.compare_prefix(observations['primary'],observations['repeat'])
            gate_name='early-gate.json'
        else:
            sys.path.insert(0,str(self.root))
            gate=module(self.root/'rank_stability.py','incremental_original_rank').compare(observations['primary'],observations['repeat'])
            gate_name='rank-stability.json'
        save(temporary/gate_name,gate)
        save(temporary/'parents.json',receipts)
        hashes={str(p.relative_to(temporary)):sha(p) for p in temporary.rglob('*') if p.is_file()}
        save(temporary/'seal.json',dict(inputs_lock_sha256=sha(self.out/'inputs.lock.json'),last_layer=last,files=hashes))
        os.replace(temporary,target)
        return gate

    def verify_merge(self,target):
        seal=read(target/'seal.json')
        require(seal['inputs_lock_sha256']==sha(self.out/'inputs.lock.json'),'Merged input lock changed')
        for name,h in seal['files'].items():
            require(Path(name).is_absolute() is False and '..' not in Path(name).parts,'Invalid sealed path')
            require(sha(target/name)==h,'Merged artifact changed')
        for receipts in read(target/'parents.json').values():
            for r in receipts:
                require(sha(r['plan'])==r['plan_sha256'],'Merge parent plan changed')
                for key,expected in [('snapshot_manifest',r['checkpoint_manifest_sha256']),('snapshot_state',r['state_sha256'])]:
                    relative=Path(r[key])
                    require(not relative.is_absolute() and '..' not in relative.parts,'Invalid parent snapshot path')
                    require(sha(target/relative)==expected,'Merge parent snapshot changed')
                folder=Path(r['checkpoint_manifest']).parent
                part_path=folder.parent.parent
                require(sha(part_path/'checkpoint-identity.json')==r['checkpoint_identity_sha256'],'Merge producer identity changed')
                for layer,h in r['trajectory_hashes'].items():
                    require(sha(part_path/'trajectories'/f'layer-{int(layer):02d}.json')==h,'Merge trajectory parent changed')

    def storage(self,lock):
        committed=0
        for part in PARTS:
            pointer=self.out/part/'checkpoints/latest.json'
            if pointer.exists():
                folder,manifest=self.header(self.out/part)
                folders=[folder]
                if manifest.get('previous'):
                    name=manifest['previous']
                    require(Path(name).name==name and name.startswith('generation-'),'Invalid recovery generation')
                    folders.append(folder.parent/name)
                for generation in folders:
                    m=read(generation/'manifest.json')
                    require(m['identity']==manifest['identity'],'Recovery checkpoint identity differs')
                    for name,info in m['files'].items():
                        require(Path(name).name==name,'Invalid checkpoint file')
                        p=generation/name
                        require(not p.is_symlink() and p.stat().st_size==info['bytes'],'Invalid recovery payload')
                        committed+=info['bytes']
        required=max(0,lock['storage']['required_total_bytes']-committed)
        free=shutil.disk_usage(self.root).free
        require(free>=required,f'Checkpoint reserve needs {required/2**30:.1f} GiB; {free/2**30:.1f} GiB free')
        return dict(required_free_bytes=required,free_bytes=free,committed_bytes=committed)

    def preflight(self,lock):
        import psutil
        module(self.root/'fused-attention-v1/guard.py','incremental_gpu_guard').require_gpu_release()
        apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True).strip()
        require(not apps,'GPU busy; existing jobs unchanged: '+apps)
        free=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).splitlines()
        require(len(free)==1 and int(free[0])>=30720,'Requires one GPU with 30 GiB free')
        require(psutil.virtual_memory().available>=100*(1<<30),'Requires 100 GiB available RAM')
        return self.storage(lock)

    def event(self,status,**fields):
        row=dict(status=status,pid=os.getpid(),time=time.time(),native_precision_verified=False,selection_allowed=False,**fields)
        save(self.out/'state.json',row)
        with (self.out/'events.jsonl').open('a') as stream:
            stream.write(json.dumps(row)+'\n')

    def observe(self,part,early,lock):
        self.preflight(lock)
        path=self.out/part
        command=[sys.executable,'-u',str(self.root/'performance-v9/observe.py'),'--plan',lock['partitions'][part]['plan'],'--out',str(path),'--layers','70','--precision','sglang_sm120','--checkpoint-every-layers','8','--pause-file',str(self.out/'pause.request')]
        if early:
            command+=['--stop-after-stage','67']
        if (path/'checkpoint-identity.json').exists():
            command+=['--resume']
        env=dict(os.environ,FLASHINFER_WORKSPACE_BASE=str(self.root),TMPDIR=str(self.out/'scratch'),MAX_JOBS='16',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
        (self.out/'scratch').mkdir(exist_ok=True)
        with (self.out/(part+'.log')).open('a') as stream:
            self.child=subprocess.Popen(command,cwd=self.root,env=env,stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
            self.event('running',partition=part,early_check=early,child_pid=self.child.pid,command=command)
            code=self.child.wait();self.child=None
        require(code in (0,75),'Observer failed; inspect partition log: '+str(code))
        return code

    def at_boundary(self,part,stage):
        path=self.out/part
        return (path/'checkpoints/latest.json').exists() and self.header(path)[1]['next_stage']>=stage

    def controller(self):
        lock=self.locked()
        with (self.out/'controller.lock').open('a') as controller,(self.root/'performance-v1/benchmark.lock').open('a') as gpu:
            fcntl.flock(controller,fcntl.LOCK_EX|fcntl.LOCK_NB)
            fcntl.flock(gpu,fcntl.LOCK_EX|fcntl.LOCK_NB)
            def pause(signum,frame):
                save(self.out/'pause.request',dict(signal=signum,time=time.time()))
            for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGUSR1):
                signal.signal(sig,pause)
            try:
                self.event('preflight',storage=self.storage(lock))
                for part in PARTS:
                    if (self.out/'pause.request').exists():
                        self.event('paused');return
                    if not self.at_boundary(part,67):
                        self.observe(part,True,lock)
                    if (self.out/'pause.request').exists():
                        self.event('paused',partition=part);return
                early=self.merge(32)
                if not early['early_gate_passed']:
                    self.event('calibration_failed',phase='paired_layer32',gate=early);return
                self.event('early_gate_passed',minimum_jaccard=early['minimum_jaccard'])
                for part in PARTS:
                    if (self.out/'pause.request').exists():
                        self.event('paused');return
                    if not (self.out/part/'report.json').exists():
                        code=self.observe(part,False,lock)
                        if code==75:
                            self.event('paused',partition=part);return
                final=self.merge(69)
                if not final['rank_stability_verified']:
                    self.event('calibration_failed',phase='full_model',gate=final);return
                result=dict(status='reap_prerequisites_pending',calibration_passed=True,combined_evidence=str(self.out/'merges/full-69'),required_gates=['native_precision_verified','objective_baseline_verified','calibration_coverage_verified'],reason='Merged calibration does not establish native precision, coverage or objective quality; no pruning/export gate is promoted',native_precision_verified=False,selection_allowed=False)
                save(self.out/'post-calibration.json',result)
                self.event('calibration_completed',result=result)
            except BaseException as error:
                self.event('failed',error=repr(error));raise


def controller_live(out):
    import psutil
    state_path=out/'state.json'
    if not state_path.exists():
        return False
    try:
        process=psutil.Process(read(state_path)['pid'])
        args=process.cmdline()
        return process.status()!=psutil.STATUS_ZOMBIE and 'run' in args and str(out) in args and any(Path(arg).name==Path(__file__).name for arg in args)
    except (KeyError,OSError,psutil.Error):
        return False


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['prepare','preflight','run','start','pause','resume','status','merge-prefix','merge-full'])
    parser.add_argument('--workspace',default='/mnt/nvme4/mimo-reap')
    parser.add_argument('--run-dir',default='/mnt/nvme4/mimo-reap/incremental-calibration-v1')
    args=parser.parse_args();production=Production(args.workspace,args.run_dir)
    if args.action=='prepare':
        production.prepare()
    elif args.action=='preflight':
        print(json.dumps(production.preflight(production.locked()),indent=2))
    elif args.action=='run':
        production.controller()
    elif args.action in ('start','resume'):
        require(not controller_live(production.out),'Controller already running')
        require((production.out/'preparation.json').exists(),'Prepare and verify before launch')
        production.preflight(production.locked())
        if args.action=='resume':
            (production.out/'pause.request').unlink(missing_ok=True)
        else:
            require(not (production.out/'pause.request').exists(),'Paused: use resume explicitly')
        command=[sys.executable,'-u',str(Path(__file__).resolve()),'run','--workspace',str(production.root),'--run-dir',str(production.out)]
        with (production.out/'controller.log').open('a') as stream:
            process=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,cwd=production.root)
        save(production.out/'launch.json',dict(pid=process.pid,command=command,time=time.time()))
        print(json.dumps(dict(status='submitted',pid=process.pid)))
    elif args.action=='pause':
        save(production.out/'pause.request',dict(reason='User pause',time=time.time()))
        print('Pause requested; observer commits at the next stage boundary and then releases the GPU.')
    elif args.action=='status':
        data=dict(controller=read(production.out/'state.json'),controller_live=controller_live(production.out),partitions={})
        for part in PARTS:
            path=production.out/part
            data['partitions'][part]={name:read(path/name) for name in ('progress.json','checkpoint-status.json') if (path/name).exists()}
        print(json.dumps(data,indent=2))
    else:
        print(json.dumps(production.merge(32 if args.action=='merge-prefix' else 69),indent=2))


if __name__=='__main__':
    main()
