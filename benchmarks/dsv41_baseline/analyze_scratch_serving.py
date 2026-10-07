"""Analyze a completed gated serving capture, with bounded, streaming input reads.

Units are tile-count times token rows, a workload proxy rather than a kernel cost
model. CPU time includes hardware memory stalls. This script makes no speedup claim.
"""
import argparse
import bisect
from collections import Counter, defaultdict
import json
from pathlib import Path
import statistics


def records(path):
    with path.open() as source:
        for line in source:
            if line.strip():
                yield json.loads(line)


def stats(values, scale=1000):
    values = sorted(values)
    if not values:
        return None
    return dict(n=len(values), median=statistics.median(values) / scale,
                p95=values[int((len(values) - 1) * .95)] / scale,
                max=values[-1] / scale, sum=sum(values) / scale)


def correlation(xs, ys):
    if len(set(xs)) < 2 or len(set(ys)) < 2:
        return None
    return statistics.correlation(xs, ys)


def analyze(root):
    run = next(root.glob('servers/*/*/results.jsonl')).parent
    low = json.loads((root / 'timed.start').read_text())['ns']
    high = int(next(r['monotonic'] for r in records(run / 'boundary-samples.jsonl')
                    if r['label'].startswith('session_')) * 1e9)
    output = dict(window_ns=[low, high], results=list(records(run / 'results.jsonl')),
                  groups={}, footers={}, top_forwards=[], top_scratch=[], growths=[], shapes={})
    jobs = defaultdict(list)
    # Compact companions retain complete CPU/copy identities for the existing DMA
    # classifier, while omitting hundreds of thousands of idle polling events.
    for path in sorted(root.glob('events.*.jsonl')):
        group = 0 if 'cpu-exp0' in path.name else 1
        pending, submits, shapes = {}, {}, {}
        with (root / ('compact-' + path.name)).open('w') as compact:
            for r in records(path):
                kind = r.get('event', '')
                if not kind or kind in ('cpu_submit', 'cpu_start', 'cpu_shape', 'cpu_end') or (
                        'cpu-exp' not in path.name):
                    compact.write(json.dumps(r) + '\n')
                if not kind:
                    if 'dropped' in r:
                        output['footers'][path.name] = r
                        if r['dropped']:
                            raise ValueError(f'dropped job events: {path}')
                    continue
                key = (kind.split('_')[0], r['seq'], r['row'])
                if kind == 'cpu_submit':
                    submits[key] = r['ns']
                if kind == 'cpu_shape':
                    shapes[key] = [r[c] for c in 'abc']
                if kind in ('cpu_start', 'draft_start'):
                    pending[key] = r
                if kind in ('cpu_end', 'draft_end') and key in pending:
                    first = pending.pop(key)
                    if low <= first['ns'] <= r['ns'] <= high:
                        jobs[group].append(dict(begin=first['ns'], end=r['ns'], kind=key[0],
                            row=r['row'], seq=r['seq'], submit=submits.get(key),
                            shape=shapes.get(key, [first[c] for c in 'abc'])))
        if path.name not in output['footers']:
            raise ValueError(f'missing job footer: {path}')
    for js in jobs.values():
        js.sort(key=lambda j: j['begin'])
        previous = None
        for job in js:
            if job['submit'] is not None:
                available = max(job['submit'], previous['end'] if previous else job['submit'])
                job['available_to_start'] = job['begin'] - available
                job['previous_kind'] = previous['kind'] if previous else None
                job['previous_end'] = previous['end'] if previous else None
            previous = job
    starts = {g: [j['begin'] for j in js] for g, js in jobs.items()}
    scratch = {}
    for path in root.glob('scratch.*.jsonl'):
        leader = int(path.name.split('.')[-2])
        for r in records(path):
            if r.get('kind') == 'scratch':
                scratch[leader, r['forward']] = r
            elif r.get('kind') == 'growth' and low <= r['begin'] <= r['end'] <= high:
                output['growths'].append(dict(r, leader=leader))
            elif r.get('footer'):
                output['footers'][path.name] = r
                if r['dropped_growths']:
                    raise ValueError(f'dropped growth records: {path}')
        if path.name not in output['footers']:
            raise ValueError(f'missing scratch footer: {path}')
    frames = defaultdict(list)
    for path in root.glob('worker-phases.*.jsonl'):
        leader = int(path.name.split('.')[-2])
        by_forward = defaultdict(list)
        for r in records(path):
            if 'forward' in r and low <= r['forward_begin'] <= r['forward_end'] <= high:
                by_forward[r['forward']].append(r)
            elif r.get('footer'):
                output['footers'][path.name] = r
                if r['dropped_forwards']:
                    raise ValueError(f'dropped worker records: {path}')
        if path.name not in output['footers']:
            raise ValueError(f'missing worker footer: {path}')
        for serial, rs in by_forward.items():
            first = rs[0]
            if len(rs) != first['threads'] * 5:
                raise ValueError('partial forward')
            group = 0 if first['cpu'] < 18 else 1
            index = bisect.bisect_right(starts.get(group, []), first['forward_begin']) - 1
            job = jobs[group][index] if index >= 0 else None
            if job and not job['begin'] <= first['forward_begin'] <= first['forward_end'] <= job['end']:
                job = None
            frame = dict(group=group, leader=leader, forward=serial, begin=first['forward_begin'],
                end=first['forward_end'], wall=first['forward_end'] - first['forward_begin'],
                rows=first['rows'], chunks=first['chunks'], job=job,
                scratch=scratch.get((leader, serial)), phases=[],
                entry=max(r['ready'] for r in rs) - first['team_begin'],
                join=first['forward_end'] - max(r['end'] for r in rs))
            for p in (0, 1, 2, 3, 5):
                rr = [r for r in rs if r['phase'] == p]
                workers = [dict(worker=r['worker'], tid=r['tid'], cpu=r['cpu'], units=r['units'],
                    begin=r['begin'], end=r['work_end'], wall=r['work_end'] - r['begin'],
                    cpu_ns=r['cpu_work_end'] - r['cpu_begin'],
                    minflt=r['minflt'], majflt=r['majflt'], nvcsw=r['nvcsw'], nivcsw=r['nivcsw']) for r in rr]
                units = [w['units'] for w in workers]
                last = max(workers, key=lambda w: w['end'])
                phase = dict(phase=p, begin=min(r['begin'] for r in rr),
                    end=max(r['work_end'] for r in rr),
                    wall=max(r['work_end'] for r in rr) - min(r['begin'] for r in rr),
                    tail=max(r['end'] for r in rr) - max(r['work_end'] for r in rr),
                    max_mean_units=max(units) / statistics.mean(units) if sum(units) else None,
                    last_mean_units=last['units'] / statistics.mean(units) if sum(units) else None,
                    units_cpu_correlation=correlation(units, [w['cpu_ns'] for w in workers]),
                    last_worker=last, workers=workers)
                frame['phases'].append(phase)
            frames[group].append(frame)
    for group, fs in frames.items():
        matrices = [p for f in fs for p in f['phases'] if p['phase'] in (1, 3)]
        scratches = [f['scratch'] for f in fs if f['scratch']]
        tails = [max(p['tail'] for p in f['phases']) for f in fs]
        output['groups'][str(group)] = dict(forwards=len(fs), matched_jobs=sum(f['job'] is not None for f in fs),
            forward_us=stats([f['wall'] for f in fs]), entry_us=stats([f['entry'] for f in fs]),
            join_us=stats([f['join'] for f in fs]), barrier_tail_us=stats(tails),
            barrier_over_1ms=sum(t > 1000000 for t in tails),
            scratch_us=stats([r['end'] - r['begin'] for r in scratches]),
            scratch_with_growth=sum(bool(r['growths']) for r in scratches),
            scratch_without_growth_us=stats([r['end'] - r['begin'] for r in scratches if not r['growths']]),
            matrix_us=stats([p['wall'] for p in matrices]),
            max_mean_units=stats([p['max_mean_units'] for p in matrices], scale=1),
            last_mean_units=stats([p['last_mean_units'] for p in matrices], scale=1),
            units_cpu_correlation=stats([p['units_cpu_correlation'] for p in matrices
                                       if p['units_cpu_correlation'] is not None], scale=1),
            jobs_us={kind: stats([j['end'] - j['begin'] for j in jobs[group] if j['kind'] == kind])
                     for kind in ('cpu', 'draft')},
            submit_to_start_us=stats([j['begin'] - j['submit'] for j in jobs[group] if j['submit']]),
            available_to_start_us=stats([j['available_to_start'] for j in jobs[group] if j['submit']]),
            queue_over_1ms=sum(j['begin'] - j['submit'] > 1000000 for j in jobs[group] if j['submit']),
            available_to_start_over_1ms=sum(j['available_to_start'] > 1000000 for j in jobs[group] if j['submit']),
            top_queues=sorted([j for j in jobs[group] if j['submit']],
                              key=lambda j: j['begin'] - j['submit'], reverse=True)[:10],
            top_available_to_start=sorted([j for j in jobs[group] if j['submit']],
                                          key=lambda j: j['available_to_start'], reverse=True)[:10])
        shaped = defaultdict(list)
        for f in fs:
            key = (f['job']['kind'] if f['job'] else 'unknown', f['rows'], f['chunks'])
            shaped[key].append(f)
        output['shapes'][str(group)] = [dict(kind=k[0], rows=k[1], chunks=k[2], n=len(v),
            forward_us=stats([f['wall'] for f in v]),
            scratch_us=stats([f['scratch']['end'] - f['scratch']['begin'] for f in v if f['scratch']]),
            max_mean_units=stats([p['max_mean_units'] for f in v for p in f['phases']
                                 if p['phase'] in (1, 3)], scale=1)) for k, v in sorted(shaped.items())]
    holds = defaultdict(list)
    for path in root.glob('hold.*.jsonl'):
        by_hold = defaultdict(list)
        for r in records(path):
            if 'hold' in r and low <= r['begin'] <= r['end'] <= high:
                by_hold[r['hold']].append(r)
            elif r.get('footer'):
                output['footers'][path.name] = r
                if r['dropped_holds']:
                    raise ValueError(f'dropped hold records: {path}')
        if path.name not in output['footers']:
            raise ValueError(f'missing hold footer: {path}')
        for serial, rs in by_hold.items():
            first = rs[0]
            if len(rs) != first['threads']:
                raise ValueError('partial hold')
            group = 0 if first['cpu'] < 18 else 1
            holds[group].append(dict(hold=serial, begin=first['begin'], end=first['end'],
                join=first['end'] - max(r['exit'] for r in rs),
                entry=max(r['ready'] for r in rs) - first['begin']))
    for group, hs in holds.items():
        output['groups'].setdefault(str(group), {})['holds'] = dict(n=len(hs),
            join_us=stats([h['join'] for h in hs]), entry_us=stats([h['entry'] for h in hs]),
            top_joins=sorted(hs, key=lambda h: h['join'], reverse=True)[:10])
    all_frames = [f for fs in frames.values() for f in fs]
    output['top_forwards'] = sorted(all_frames, key=lambda f: f['wall'], reverse=True)[:20]
    output['top_scratch'] = sorted([f for f in all_frames if f['scratch']],
        key=lambda f: f['scratch']['end'] - f['scratch']['begin'], reverse=True)[:20]
    output['runtime'] = dict(Counter(tuple(sorted(r['counters'].items()))
                            for r in records(root / 'runtime-samples.jsonl') if low <= r['ns'] <= high))
    output['runtime'] = [dict(counters=dict(k), samples=v) for k, v in output['runtime'].items()]
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    result = analyze(args.root)
    (args.root / 'scratch-analysis.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result['groups'], indent=2))


if __name__ == '__main__':
    main()
