"""Bounded additive-scoring verification; never resumes a changed-data checkpoint."""
import argparse
import contextlib
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/mnt/nvme4/mimo-reap')
HERE = ROOT / 'incremental-scoring-v1'
DOMAINS = ['finance', 'science', 'coding', 'general_tool_use']
PARTS = ['primary', 'repeat']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.writing')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def ranking(values):
    assert len(values) == 384 and all(math.isfinite(x) and x >= 0 for x in values)
    return sorted(range(384), key=lambda i: (-values[i], i))


def close_scores(a, b):
    assert len(a) == len(b) == 384
    for x, y in zip(a, b):
        assert math.isclose(x, y, rel_tol=1e-12, abs_tol=1e-8), (x, y)
    return max(abs(x-y) / max(abs(x), abs(y), 1e-30) for x, y in zip(a, b))


def protected():
    paths = set()
    for part in PARTS:
        run = ROOT / 'full-scoring-v11' / part
        paths.update(run.glob('trajectories/layer-*.json'))
        paths.update(run.glob('checkpoints/*/manifest.json'))
        paths.update(run.glob('checkpoints/latest.json'))
        paths.update(run / p for p in ['observations.json', 'checkpoint-identity.json', 'trajectory-index.json'])
    paths.update(ROOT / 'full-scoring-v11' / p for p in ['inputs.lock.json', 'rank-stability.json', 'early-gate.json', 'post-calibration.json'])
    for name in ['calibration-mix-v7', 'calibration-mix-v8']:
        paths.update((ROOT / name).glob('*.json'))
        paths.update((ROOT / name).glob('*.jsonl'))
    return {str(p): sha(p) for p in sorted(paths)}


def frozen_identity():
    seen = {}
    for part in PARTS:
        identity = read(ROOT / 'full-scoring-v11' / part / 'checkpoint-identity.json')
        assert identity['invariant'] == 'observation_only; propagate_error=false; original source weights unchanged'
        assert identity['precision'] == 'sglang_sm120'
        for path, expected in identity['files'].items():
            if path not in seen:
                assert sha(path) == expected, path
                seen[path] = expected
        for name, expected in identity['source_shard_stats'].items():
            stat = (Path('/mnt/nvme4/mimi-v26-pro') / name).stat()
            assert stat.st_size == expected['size'] and stat.st_mtime_ns == expected['mtime_ns'], name
    return {'pinned_files_verified': len(seen), 'source_shard_stat_identity_verified': True}


def prepare():
    assert not (HERE / 'prepared.json').exists(), 'Do not overwrite prepared evidence'
    before = protected()
    identity = frozen_identity()
    additions = {json.loads(line)['sample_id'] for line in (ROOT / 'calibration-mix-v8/selected-additions.jsonl').read_text().splitlines()}
    selected = {}
    pins = dict(before)
    for part in PARTS:
        path = ROOT / f'data/observer-manifests-representative-v8-{part}/plan.json'
        plan = read(path)
        pins[str(path)] = sha(path)
        pools = {kind: {domain: [] for domain in DOMAINS} for kind in ['old', 'new']}
        templates = {}
        for group in plan['groups']:
            manifest_path = ROOT / group['manifest']
            manifest = read(manifest_path)
            pins[str(manifest_path)] = sha(manifest_path)
            templates[group['domain']] = manifest
            for sample in manifest['samples']:
                kind = 'new' if sample['sample_id'] in additions else 'old'
                pools[kind][sample['domain']].append(sample)
        selected[part] = {}
        for kind in ['old', 'new']:
            selected[part][kind] = []
            for domain in DOMAINS:
                pool = pools[kind][domain]
                # Coverage selection uses lengths only, never saliency/rank results.
                if domain == 'coding':
                    long = [s for s in pool if s['valid_tokens'] > 32768]
                    sample = min(long or pool, key=lambda s: (s['valid_tokens'], s['sample_id']))
                else:
                    substantial = [s for s in pool if s['valid_tokens'] >= 8192]
                    sample = min(substantial or pool, key=lambda s: (s['valid_tokens'], s['sample_id']))
                selected[part][kind].append(sample)
        for mode in ['joined', 'old', 'new']:
            groups = []
            # New-before-old changes domain accumulation history deliberately.
            for domain in DOMAINS:
                samples = [s for kind in (['new', 'old'] if mode == 'joined' else [mode]) for s in selected[part][kind] if s['domain'] == domain]
                manifest = dict(templates[domain], samples=samples, valid_tokens=sum(s['valid_tokens'] for s in samples), production_calibration_ready=False)
                dest = HERE / 'inputs' / part / mode / (domain + '.json')
                save(dest, manifest)
                groups.append({'domain': domain, 'manifest': str(dest)})
            save(HERE / 'inputs' / part / mode / 'plan.json', dict(groups=groups, purpose='bounded_incremental_verification_only', native_precision_verified=False, selection_allowed=False))
    save(HERE / 'protected-before.json', before)
    save(HERE / 'input-pins.json', pins)
    metadata = {part: {kind: [{k: s[k] for k in ['sample_id', 'domain', 'valid_tokens', 'token_sha256']} for s in samples] for kind, samples in kinds.items()} for part, kinds in selected.items()}
    save(HERE / 'prepared.json', dict(identity=identity, selected=metadata, layers=2, script_sha256=sha(__file__), no_full_calibration=True))


def cpu_audit():
    import numpy as np
    started = time.time()
    expected = read(ROOT / 'full-scoring-v11/rank-stability.json')
    recovered = {}
    domain_checks = 0
    worst = 0.0
    full_order_exact = 0
    for part in PARTS:
        run = ROOT / 'full-scoring-v11' / part
        observations = read(run / 'observations.json')
        index = read(run / 'trajectory-index.json')['layers']
        recovered[part] = {}
        for layer in range(1, 70):
            path = run / 'trajectories' / f'layer-{layer:02d}.json'
            assert sha(path) == index[str(layer)]['sha256']
            document = read(path)
            assert document['aggregate_check_passed'] is True
            records = document['records']
            assert len({r['sample_id'] for r in records}) == len(records)
            assert sorted(r['sample_id'] for r in records) == sorted(observations['sample_ids'])
            recovered[part][str(layer)] = {}
            for domain in DOMAINS:
                rows = [r for r in records if r['domain'] == domain]
                baseline = observations['layers'][str(layer)][domain]
                for row in rows:
                    assert sum(row['route_counts']) == row['valid_tokens'] * 8
                values = np.asarray([r['gate_weighted_output_norm_sum'] for r in rows], dtype=np.float64)
                counts = np.asarray([r['route_counts'] for r in rows], dtype=np.int64)
                total = [math.fsum(column) for column in zip(*values)]
                worst = max(worst, close_scores(total, baseline['gate_weighted_output_norm_sum']))
                assert counts.sum(0).tolist() == baseline['route_counts']
                assert sum(r['valid_tokens'] for r in rows) == baseline['valid_tokens']
                order = ranking(total)
                assert set(order[:344]) == set(ranking(baseline['gate_weighted_output_norm_sum'])[:344])
                full_order_exact += order == ranking(baseline['gate_weighted_output_norm_sum'])
                # Repartition/recombine fixed saved rows with several groupings.
                for stride in [2, 3, 7]:
                    chunks = [values[offset::stride].sum(0) for offset in range(stride)]
                    merged = np.asarray(chunks).sum(0).tolist()
                    close_scores(total, merged)
                    assert ranking(merged) == order
                recovered[part][str(layer)][domain] = order[:344]
                domain_checks += 1
    gates = {}
    for layer in range(1, 70):
        for domain in DOMAINS:
            a, b = [set(recovered[part][str(layer)][domain]) for part in PARTS]
            value = len(a & b) / len(a | b)
            key = f'{layer}:{domain}'
            assert value == expected['jaccard_by_layer_domain'][key]
            gates[key] = value
    result = dict(status='passed', layer_domain_reconstructions=domain_checks, full_expert_orders_exact=full_order_exact, partition_recombination_checks=domain_checks*3, paired_gate_values_exact=len(gates), failed_gates_preserved=[k for k,v in gates.items() if v < expected['threshold']], max_score_relative_difference=worst, seconds=time.time()-started, scope='CPU audit of existing captured evidence; not fresh GPU parity')
    save(HERE / 'cpu-audit.json', result)
    print(json.dumps(result), flush=True)


def observe():
    # Same install contexts as performance-v9/observe.py. Hooks only hash outputs.
    out = Path(sys.argv[sys.argv.index('--out')+1])
    sys.path[:0] = [str(ROOT / 'performance-v9'), str(ROOT)]
    spec = importlib.util.spec_from_file_location('incremental_v9_runtime', ROOT / 'performance-v9/observe.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    v7 = module.v7
    observer = v7.v4.v3.v2.fast.observer
    import torch
    identity = read(ROOT / 'full-scoring-v11/primary/checkpoint-identity.json')
    assert str(torch.__version__) == identity['torch'] and str(torch.version.cuda) == identity['cuda']
    assert torch.cuda.get_device_name() == identity['gpu']
    def tensor_hash(tensor):
        tensor = tensor.detach().contiguous().cpu()
        return dict(shape=list(tensor.shape), dtype=str(tensor.dtype), sha256=hashlib.sha256(tensor.view(torch.uint8).numpy().tobytes()).hexdigest())
    hooks = []
    original = observer.base.TrajectoryREAP._experts_block_hook
    def capture(self, name, model, args, output):
        sample = self._samples[self._state.current_batch_idx]
        record = dict(sample_id=sample['sample_id'], layer=name, routes=tensor_hash(args[1]), gate_weights=tensor_hash(args[2]), expert_block_output=tensor_hash(output))
        result = original(self, name, model, args, output)
        hooks.append(record)
        return result
    observer.base.TrajectoryREAP._experts_block_hook = capture
    sys.argv.pop(1)  # Remove this harness's observe subcommand.
    v7.v4.v3.v2.cadence(sys.argv)
    try:
        with v7.v4.expert_pool.install(), v7.v4.pinned_staging.install(), v7.softmax_candidates.install('fp32_inplace'), v7.v4.v3.v2.fast.install(observer.sequential_pipeline, v7.v4.v3.v2.fast.event), v7.v4.v3.v2.install_saliency(), v7.v4.v3.writers(), module.parallel_verify.install():
            code = observer.main()
        assert code == 0
        save(out / 'raw-hook-hashes.json', hooks)
    finally:
        observer.base.TrajectoryREAP._experts_block_hook = original


def compare_gpu():
    import torch
    checks = 0
    worst = 0.0
    original_worst = 0.0
    for part in PARTS:
        runs = {mode: HERE / 'runs' / part / mode for mode in ['joined', 'old', 'new']}
        documents = {mode: read(path / 'trajectories/layer-01.json') for mode, path in runs.items()}
        records = {mode: {r['sample_id']: r for r in doc['records']} for mode, doc in documents.items()}
        assert set(records['joined']) == set(records['old']) | set(records['new'])
        hashes = {mode: {r['sample_id']: r for r in read(path / 'raw-hook-hashes.json')} for mode, path in runs.items()}
        original = {r['sample_id']: r for r in read(ROOT / 'full-scoring-v11' / part / 'trajectories/layer-01.json')['records']}
        for mode in ['old', 'new']:
            for sid, row in records[mode].items():
                joined = records['joined'][sid]
                assert hashes[mode][sid] == hashes['joined'][sid], sid
                assert row['route_counts'] == joined['route_counts']
                assert row['token_sha256'] == joined['token_sha256']
                worst = max(worst, close_scores(row['gate_weighted_output_norm_sum'], joined['gate_weighted_output_norm_sum']))
                if mode == 'old':
                    assert row['route_counts'] == original[sid]['route_counts']
                    original_worst = max(original_worst, close_scores(row['gate_weighted_output_norm_sum'], original[sid]['gate_weighted_output_norm_sum']))
                checks += 1
        observations = {mode: read(path/'observations.json')['layers']['1'] for mode, path in runs.items()}
        for domain in DOMAINS:
            baseline = observations['joined'][domain]
            combined = [math.fsum(pair) for pair in zip(observations['old'][domain]['gate_weighted_output_norm_sum'], observations['new'][domain]['gate_weighted_output_norm_sum'])]
            close_scores(baseline['gate_weighted_output_norm_sum'], combined)
            assert ranking(combined) == ranking(baseline['gate_weighted_output_norm_sum'])
            assert [a+b for a,b in zip(observations['old'][domain]['route_counts'], observations['new'][domain]['route_counts'])] == baseline['route_counts']
        # Compare actual saved final activations by sample ID, not batch number.
        payloads = {}
        for mode, path in runs.items():
            samples = read(path/'instrumentation.json')['sample_metadata']
            pointer = read(path/'checkpoints/latest.json')
            folder = path/'checkpoints'/pointer['generation']
            assert sha(folder/'manifest.json') == pointer['manifest_sha256']
            manifest = read(folder/'manifest.json')
            assert manifest['next_stage'] == 5
            data = {}
            for i, sample in enumerate(samples):
                name = f'batch-{i:05d}.pt'
                assert sha(folder/name) == manifest['files'][name]['sha256']
                data[sample['sample_id']] = torch.load(folder/name, map_location='cpu', weights_only=True)
            payloads[mode] = data
        for mode in ['old','new']:
            for sid, payload in payloads[mode].items():
                joined = payloads['joined'][sid]
                assert payload.keys() == joined.keys()
                for key, item in payload.items():
                    assert item['kind'] == joined[key]['kind']
                    if item['kind'] == 'tensor':
                        assert torch.equal(item['value'], joined[key]['value']), (sid,key)
    result = dict(status='passed', independent_workflows=checks, token_routes_bitwise_exact=True, gate_weights_bitwise_exact=True, expert_outputs_bitwise_exact=True, final_cached_tensors_bitwise_exact=True, domain_full_rankings_exact=8, max_split_join_score_relative_difference=worst, max_original_v11_score_relative_difference=original_worst, scope='Both partitions, four domains, whole long histories, dense layer 0 and MoE layer 1; later layers covered by source-independence and saved-evidence aggregation audit, not fresh full-depth replay', native_precision_verified=False, selection_allowed=False)
    save(HERE/'gpu-parity.json', result)
    return result


def gpu():
    import fcntl
    assert read(HERE/'cpu-audit.json')['status'] == 'passed'
    pins = read(HERE/'input-pins.json')
    for path, expected in pins.items():
        assert sha(path) == expected, path
    frozen_identity()
    sys.path.insert(0, str(ROOT/'fused-attention-v1'))
    from guard import require_gpu_release
    require_gpu_release()
    # Never stop a service to make room for this bounded test.
    apps = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True).strip()
    assert not apps, 'Unrelated GPU processes present: '+apps
    assert int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True)) >= 30720
    env = dict(os.environ, TMPDIR=str(HERE/'scratch'), FLASHINFER_WORKSPACE_BASE=str(ROOT), TRITON_CACHE_DIR=str(ROOT/'performance-v4/scratch/triton'), MAX_JOBS='4', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
    (HERE/'scratch').mkdir(exist_ok=True)
    receipts = []
    with (ROOT/'performance-v1/benchmark.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        for part in PARTS:
            for mode in ['joined','old','new']:
                require_gpu_release()
                out = HERE/'runs'/part/mode
                assert not out.exists(), 'Use a new version for reruns'
                command = [sys.executable,'-u',str(HERE/'verify_incremental.py'),'observe','--plan',str(HERE/'inputs'/part/mode/'plan.json'),'--out',str(out),'--layers','2','--precision','sglang_sm120','--checkpoint-every-layers','8']
                save(HERE/'status.json', dict(status='running',partition=part,mode=mode,time=time.time(),completed_runs=len(receipts),total_runs=6))
                started = time.time()
                with (HERE/f'{part}-{mode}.log').open('w') as stream:
                    p = subprocess.run(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=1200)
                receipts.append(dict(command=command,returncode=p.returncode,seconds=time.time()-started))
                save(HERE/'commands.json', receipts)
                assert p.returncode == 0, (part,mode,p.returncode)
        result = compare_gpu()
        assert protected() == read(HERE/'protected-before.json'), 'Protected artifact changed'
        save(HERE/'protected-after.json', protected())
        save(HERE/'status.json', dict(status='completed',gpu=result,time=time.time(),no_full_calibration=True))
    print(json.dumps(result),flush=True)


if __name__ == '__main__':
    action = sys.argv[1]
    if action == 'observe':
        observe()
    else:
        try:
            {'prepare':prepare,'cpu':cpu_audit,'gpu':gpu,'compare':compare_gpu}[action]()
        except BaseException as error:
            save(HERE/'failure.json',dict(action=action,error=repr(error),time=time.time()))
            raise
