"""Allocate only new, unscored finance research groups before any expert scoring."""
import collections
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path('/mnt/nvme4/mimo-reap')
BASE = ROOT / 'public-workflow-gap-fill-v3'
SOURCE = BASE / 'finance-union-v1'
OUT = BASE / 'finance-balanced-draft-v1'


def extended_owners(parent_owners, additions):
    owners = dict(parent_owners)
    for r in additions:
        group, part = r['split_group'].lower(), r['partition']
        if group in owners and owners[group] != part:
            raise ValueError('Existing group ownership cannot move')
        owners[group] = part
    return owners


def allocate(pool, locked, parent_owners, targets, seed, bucket, validate):
    selected = copy.deepcopy(locked)
    ledger, gaps = [], []
    for category, target in targets.items():
        counts = collections.Counter(r['partition'] for r in selected if r['gap_category'] == category)
        if counts['primary'] != counts['repeat'] or counts['primary'] > target:
            raise ValueError('Locked cohort cannot be moved to repair imbalance')
        for pair_index in range(counts['primary'], target):
            used_sources = {r['source_record_sha256'] for r in selected}
            used_groups = {r['split_group'].lower() for r in selected}
            candidates = [r for r in pool if r['gap_category'] == category
                and r['stratum'] == 'research' and r['target_domain'] == 'finance'
                and r['source_record_sha256'] not in used_sources
                and r['split_group'].lower() not in used_groups]
            options = []
            for a in candidates:
                for b in candidates:
                    if a['sample_id'] == b['sample_id'] or a['source_record_sha256'] == b['source_record_sha256']:
                        continue
                    lengths = [a['valid_tokens'], b['valid_tokens']]
                    if not all(8192 <= n <= 65536 for n in lengths):
                        continue
                    if bucket(lengths[0]) != bucket(lengths[1]) or max(lengths) / min(lengths) > 1.5:
                        continue
                    pair = [dict(copy.deepcopy(a), partition='primary'), dict(copy.deepcopy(b), partition='repeat')]
                    try:
                        owners = extended_owners(parent_owners, selected + pair)
                        validate(selected + pair, owners)
                    except ValueError:
                        continue
                    changes = int(a['partition'] != 'primary') + int(b['partition'] != 'repeat')
                    tie = hashlib.sha256(json.dumps([seed, category, pair_index,
                        a['source_record_sha256'], b['source_record_sha256']], separators=(',', ':')).encode()).hexdigest()
                    options.append(((changes, tie), pair, [a['partition'], b['partition']]))
            if not options:
                gaps.append(dict(category=category, pair=pair_index))
                break
            _, pair, original_parts = min(options, key=lambda item: item[0])
            for r, original in zip(pair, original_parts):
                ledger.append(dict(sample_id=r['sample_id'], source_record_sha256=r['source_record_sha256'],
                    group=r['split_group'].lower(), category=category, valid_tokens=r['valid_tokens'],
                    discovery_partition=original, allocated_partition=r['partition'],
                    previously_parent_owned=r['split_group'].lower() in parent_owners))
            selected.extend(pair)
    return selected, extended_owners(parent_owners, selected), ledger, gaps


def main():
    spec = importlib.util.spec_from_file_location('balanced_draft_helpers', BASE / 'release.py')
    R = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = R
    spec.loader.exec_module(R)
    C, B = R.C, R.B
    pins = dict(B.read(SOURCE / 'inputs.lock.json'))
    for name in ('screened.jsonl', 'scope-review.json', 'proposed.jsonl', 'proposal.json',
            'policy.json', 'owners.json', 'sources.lock.json', 'partial-validation.json'):
        pins[str((SOURCE / name).relative_to(ROOT))] = C.file_sha(SOURCE / name)
    pins[str(Path(__file__).absolute().relative_to(ROOT))] = C.file_sha(Path(__file__))
    def check_pins():
        for name, h in pins.items():
            if C.file_sha(ROOT / name) != h:
                raise ValueError('Changed draft input: ' + name)
    check_pins()
    review = B.read(SOURCE / 'scope-review.json')
    if review['screened_sha256'] != C.file_sha(SOURCE / 'screened.jsonl'):
        raise ValueError('Candidate scope review pin')
    approved = set(review['approved_candidates'])
    pool = [r for r in R.A.A.rows(SOURCE / 'screened.jsonl')
        if r['sample_id'] + ':' + r['gap_category'] in approved]
    locked = R.A.A.rows(SOURCE / 'proposed.jsonl')
    parent = R.A.parent()
    parent_owners = C.owner_table(parent)
    frozen_sources = {r.get('source_record_sha256') for rows in parent.values() for r in rows}
    parent_counts = collections.Counter(r['split_group'].lower() for rows in parent.values() for r in rows)
    parent_tokens = collections.Counter()
    for rows in parent.values():
        for r in rows:
            parent_tokens[r['split_group'].lower()] += r['valid_tokens']
    def validate(additions, owners):
        seen_sources, seen_groups = set(frozen_sources), set()
        parts = {p: list(rows) for p, rows in parent.items()}
        for r in additions:
            group, part = r['split_group'].lower(), r['partition']
            if r['source_record_sha256'] in seen_sources or group in seen_groups:
                raise ValueError('Duplicate source or group')
            if B.I.partition(group, owners) != part:
                raise ValueError('Partition ownership')
            if group.startswith('repo:') and (parent_counts[group] + 1 > 2
                    or parent_tokens[group] + r['valid_tokens'] > 100000):
                raise ValueError('Repository cap')
            seen_sources.add(r['source_record_sha256'])
            seen_groups.add(group)
            parts[part].append(r)
        B.K.partition_overlap(parts)
    policy = B.read(SOURCE / 'policy.json')
    targets = {cat: v.get('target_per_partition', policy['target_per_partition_category'])
        for cat, v in policy['categories'].items()}
    result = allocate(pool, locked, parent_owners, targets, policy['seed'], C.bucket, validate)
    again = allocate(list(reversed(pool)), locked, parent_owners, targets, policy['seed'], C.bucket, validate)
    if result != again:
        raise ValueError('Allocation depends on source enumeration order')
    selected, owners, ledger, gaps = result
    if selected[:len(locked)] != locked:
        raise ValueError('Locked draft histories changed')
    check_pins()
    OUT.mkdir(exist_ok=False)
    policy['allocation'] = dict(method='minimum_changes_then_seeded_hash_among_content_and_lineage_feasible_pairs',
        scope='Only previously unselected, unscored finance research groups; all parent ownership and 24 locked histories fixed',
        base_policy_sha256=C.file_sha(SOURCE / 'policy.json'), expert_scores_used=False,
        length_buckets_and_ratio_unchanged=True)
    C.save(OUT / 'policy.json', policy)
    C.save(OUT / 'owners.json', owners)
    C.save(OUT / 'sources.lock.json', B.read(SOURCE / 'sources.lock.json'))
    C.save(OUT / 'allocation-ledger.json', dict(parent_owners_sha256=C.sha(parent_owners),
        locked_draft_sha256=C.file_sha(SOURCE / 'proposed.jsonl'), assignments=ledger,
        deterministic_replay=True, input_order_independent=True, expert_scores_used=False))
    B.emit(OUT / 'proposed.jsonl', selected)
    C.save(OUT / 'proposal.json', dict(additions=len(selected), unfilled=gaps,
        original_draft_histories_preserved=len(locked), parent_histories_preserved=667,
        full_content_review=False, release_ready=False, scoring_launched=False,
        selection_allowed=False))
    C.save(OUT / 'inputs.lock.json', pins)
    R.R.OUT = OUT
    R.R.check_additions(selected, parent)
    print(json.dumps(B.read(OUT / 'proposal.json')), flush=True)


if __name__ == '__main__':
    main()
