"""CPU-only full-model reaggregation and failure attribution; no data selection."""
import collections
import hashlib
import json
import math
import time
from pathlib import Path
import numpy as np

ROOT = Path('/mnt/nvme4/mimo-reap')
RUN = ROOT / 'full-scoring-v11'
OUT = ROOT / 'rank-diagnosis-v7'
P = ('primary', 'repeat')
pins = {}

def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def read(p):
    pins[str(p)] = digest(p)
    return json.loads(Path(p).read_text())

def write(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2) + '\n')

def order(x):
    return sorted(range(len(x)), key=lambda e: (-float(x[e]), e))

def selected(x):
    return set(order(x)[:344])

def jaccard(a, b):
    return len(a & b) / len(a | b)

def description(r):
    goal = r.get('objective_text') or next((str(m.get('content', '')) for m in r['messages'] if m['role'] == 'user'), '')
    return dict(sample_id=r['sample_id'], domain=r['domain'], group=r['split_group'], dataset=r.get('dataset'), family=r.get('workflow_family', r.get('calibration_stratum')), tokens=r['valid_tokens'], language=r.get('language'), objective=goal[:2400])

def main():
    started = time.time()
    OUT.mkdir(exist_ok=False)
    gate = read(RUN / 'rank-stability.json')
    assert read(RUN / 'state.json')['status'] == 'calibration_failed'
    assert gate['threshold'] == .95 and gate['keep'] == 344
    lock = read(RUN / 'inputs.lock.json')
    contexts, obs, indices = {}, {}, {}
    import torch
    for part in P:
        obs[part] = read(RUN / part / 'observations.json')
        assert digest(RUN / part / 'observations.json') == gate['first_sha256' if part == 'primary' else 'second_sha256']
        assert obs[part]['full_stack_completed']
        planpath = ROOT / lock['partitions'][part]['plan']
        plan = read(planpath)
        assert digest(planpath) == lock['inputs_sha256'][str(planpath.relative_to(ROOT))]
        contexts[part] = {}
        for group in plan['groups']:
            path = ROOT / group['manifest']
            manifest = read(path)
            assert digest(path) == lock['inputs_sha256'][str(path.relative_to(ROOT))]
            for r in manifest['samples']:
                assert r['sample_id'] not in contexts[part]
                contexts[part][r['sample_id']] = description(r)
        cp = RUN / part / 'checkpoints'
        ptr = read(cp / 'latest.json'); folder = cp / ptr['generation']
        manifest = read(folder / 'manifest.json')
        assert digest(folder / 'manifest.json') == ptr['manifest_sha256'] and manifest['next_stage'] == 141
        assert manifest['identity'] == read(RUN / part / 'checkpoint-identity.json')
        sp = folder / 'state.pt'; assert digest(sp) == manifest['files']['state.pt']['sha256']
        pins[str(sp)] = digest(sp)
        indices[part] = torch.load(sp, map_location='cpu', weights_only=True)['observer']['written']
    comparisons, details, influences = {}, {}, []
    scalar_checks = 0
    for layer in range(1, 70):
        docs = {}
        for part in P:
            path = RUN / part / 'trajectories' / f'layer-{layer:02d}.json'
            doc = read(path)
            assert digest(path) == indices[part][str(layer)]['sha256'] and doc['aggregate_check_passed']
            assert {r['sample_id'] for r in doc['records']} == set(contexts[part])
            docs[part] = doc
        for domain in ('coding', 'science', 'finance', 'general_tool_use'):
            key = f'{layer}:{domain}'; tab = {}
            for part in P:
                rr = [r for r in docs[part]['records'] if r['domain'] == domain]
                x = np.asarray([r['gate_weighted_output_norm_sum'] for r in rr], dtype=np.float64)
                routes = np.asarray([r['route_counts'] for r in rr])
                aggregate = obs[part]['layers'][str(layer)][domain]
                total = np.asarray(aggregate['gate_weighted_output_norm_sum'])
                assert np.isfinite(x).all() and (x >= 0).all()
                assert np.allclose(x.sum(0), total, rtol=1e-12, atol=1e-8)
                assert np.array_equal(routes.sum(0), aggregate['route_counts'])
                assert np.array_equal(routes.sum(1), [8*r['valid_tokens'] for r in rr])
                assert all(r['valid_tokens'] == contexts[part][r['sample_id']]['tokens'] for r in rr)
                tab[part] = dict(rr=rr, x=x, total=total, chosen=selected(total), rank={e:i+1 for i,e in enumerate(order(total))})
            baseline = jaccard(tab[P[0]]['chosen'], tab[P[1]]['chosen'])
            assert baseline == gate['jaccard_by_layer_domain'][key]
            comparisons[key] = baseline
            if baseline >= .95:
                continue
            exclusive = {part: sorted(tab[part]['chosen'] - tab[P[1-P.index(part)]]['chosen']) for part in P}
            experts = []
            for e in sorted(set().union(*map(set, exclusive.values()))):
                item = dict(expert=e, partitions={})
                for part in P:
                    t = tab[part]; shares = t['x'][:,e] / t['total'][e]
                    top = np.argsort(-shares, kind='stable')[:5]
                    item['partitions'][part] = dict(rank=t['rank'][e], relative_to_cutoff=float(t['total'][e]/t['total'][order(t['total'])[343]]-1), effective_episodes=float(1/np.dot(shares,shares)), top=[dict(**contexts[part][t['rr'][i]['sample_id']], expert_score_share=float(shares[i]), token_share=t['rr'][i]['valid_tokens']/sum(r['valid_tokens'] for r in t['rr'])) for i in top])
                experts.append(item)
            for part in P:
                t = tab[part]; other = tab[P[1-P.index(part)]]
                for i,r in enumerate(t['rr']):
                    reduced = np.delete(t['x'], i, 0)
                    scalar = [math.fsum(reduced[:,e]) for e in range(384)]
                    assert selected(scalar) == selected(reduced.sum(0)); scalar_checks += 1
                    value = jaccard(selected(scalar), other['chosen'])
                    influences.append(dict(check=key, partition=part, **contexts[part][r['sample_id']], diagnostic_exclusion_jaccard=value, delta=value-baseline))
            details[key] = dict(jaccard=baseline, exclusive=exclusive, experts=experts)
        print('verified layer', layer, flush=True)
    assert len(comparisons) == 276 and len(details) == 8
    write('report.json', dict(status='diagnostic_only', gpu_used=False, checks_reproduced=len(comparisons), failed_checks=details, episode_counterfactuals=influences, scalar_counterfactuals_verified=scalar_checks, limitations=['Fixed-score influence does not prove incorrect examples, task-quality loss or justify deletion.', 'Whole-history attribution cannot identify particular token regions.', 'Public correctness remains upstream attributed; no execution or scientific regrading.'], selection_allowed=False))
    write('inputs.sha256.json', pins)
    assert all(digest(p) == h for p,h in pins.items())
    write('verification.json', dict(status='passed', files_verified=len(pins), scalar_checks=scalar_checks, elapsed_seconds=time.time()-started, script_sha256=digest(__file__)))
    print('DONE', list(details), flush=True)

if __name__ == '__main__':
    main()
