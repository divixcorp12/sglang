"""Union screened candidate pools under a fresh draft, without releasing data."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path('/mnt/nvme4/mimo-reap')
BASE = ROOT / 'public-workflow-gap-fill-v3'
EXT = BASE / 'financial-goal-extension-v1'
OUT = BASE / 'finance-union-v1'


def merge_rows(original, extension):
    merged = {}
    source_by_id = {}
    duplicates = []
    for origin, rows in [('original', original), ('extension', extension)]:
        for row in rows:
            sid, source, category = (row[k] for k in
                ('sample_id', 'source_record_sha256', 'gap_category'))
            if sid in source_by_id and source_by_id[sid] != source:
                raise ValueError('sample identity has conflicting raw source')
            source_by_id[sid] = source
            key = (sid, category)
            if key in merged:
                # Discovery rules differ, while the complete normalized history must agree.
                clean = lambda r: {k: v for k, v in r.items() if k != 'preparation_policy_sha256'}
                if clean(merged[key]) != clean(row):
                    raise ValueError('conflicting duplicate normalized history')
                duplicates.append(dict(sample_id=sid, category=category, origin=origin,
                    preparation_policy_sha256=row.get('preparation_policy_sha256')))
            else:
                merged[key] = copy.deepcopy(row)
    return list(merged.values()), duplicates


def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    inputs = [BASE / n for n in ('screened.jsonl', 'scope-review.json', 'policy.json',
        'owners.json', 'protected-before.json', 'sources.lock.json', 'prepare.py', 'release.py')]
    inputs += [EXT / n for n in ('screened.jsonl', 'scope-review.json', 'policy.json',
        'sources.lock.json')]
    inputs += [Path(__file__).absolute()]
    pins = {str(p.relative_to(ROOT)): file_sha(p) for p in inputs}
    read = lambda p: json.loads(p.read_text())
    rows = lambda p: [json.loads(s) for s in p.read_text().splitlines()]
    scope, extra = read(BASE / 'scope-review.json'), read(EXT / 'scope-review.json')
    if scope['screened_sha256'] != file_sha(BASE / 'screened.jsonl'):
        raise ValueError('original scope review pin')
    if extra['screened_sha256'] != file_sha(EXT / 'screened.jsonl'):
        raise ValueError('extension scope review pin')
    original = rows(BASE / 'screened.jsonl')
    extended = rows(EXT / 'screened.jsonl')
    merged, duplicates = merge_rows(original, extended)
    by_key = {r['sample_id'] + ':' + r['gap_category']: r for r in merged}
    approved = set(scope['approved_candidates'])
    for decision in extra['records']:
        key = decision['sample_id'] + ':' + decision['category']
        if by_key[key]['source_record_sha256'] != decision['source_record_sha256']:
            raise ValueError('extension review raw source identity')
        if decision['decision'] == 'scope_admit_pending_full_content':
            approved.add(key)
    if not approved <= by_key.keys():
        raise ValueError('scope approval outside candidate union')
    for name, h in read(BASE / 'protected-before.json').items():
        if file_sha(ROOT / name) != h:
            raise ValueError('changed frozen parent')
    sources = {m['path']: m for m in read(BASE / 'sources.lock.json')['files']}
    for m in read(EXT / 'sources.lock.json')['files']:
        if m['path'] in sources and sources[m['path']]['sha256'] != m['sha256']:
            raise ValueError('source file identity conflict')
        sources.setdefault(m['path'], m)
    OUT.mkdir(exist_ok=False)
    save = lambda p, v: p.write_text(json.dumps(v, indent=2) + '\n')
    for name in ('policy.json', 'owners.json', 'protected-before.json'):
        save(OUT / name, read(BASE / name))
    save(OUT / 'sources.lock.json', dict(files=list(sources.values())))
    (OUT / 'screened.jsonl').write_text(''.join(json.dumps(r, sort_keys=True) + '\n' for r in merged))
    save(OUT / 'scope-review.json', dict(screened_sha256=file_sha(OUT / 'screened.jsonl'),
        approved_candidates=sorted(approved), full_content_review=False,
        selection_uses_scores=False, input_reviews=[pins[str(p.relative_to(ROOT))]
            for p in (BASE / 'scope-review.json', EXT / 'scope-review.json')]))
    spec = importlib.util.spec_from_file_location('union_release_helpers', BASE / 'release.py')
    release = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = release
    spec.loader.exec_module(release)
    release.R.OUT = OUT
    release.A.A.OUT = OUT
    release.R.propose(OUT)
    replay = OUT / 'replay'
    release.R.propose(replay)
    for name in ('proposed.jsonl', 'proposal.json'):
        if file_sha(OUT / name) != file_sha(replay / name):
            raise ValueError('proposal replay mismatch')
    for name, h in pins.items():
        if file_sha(ROOT / name) != h:
            raise ValueError('input changed during union')
    save(OUT / 'inputs.lock.json', pins)
    save(OUT / 'union-audit.json', dict(original_candidates=len(original),
        extension_candidates=len(extended), union_candidates=len(merged),
        duplicates=duplicates, original_candidates_preserved=True,
        unchanged_policy_sha256=file_sha(BASE / 'policy.json'),
        deterministic_proposal_replay=True, release_ready=False,
        full_content_review=False, scoring_launched=False, selection_allowed=False))
    print(json.dumps(read(OUT / 'proposal.json')), flush=True)


if __name__ == '__main__':
    main()
