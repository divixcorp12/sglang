"""Lossless review copies of the selected histories, without content approval."""
import collections
import hashlib
import json
from pathlib import Path

ROOT = Path('/mnt/nvme4/mimo-reap/public-workflow-gap-fill-v3/finance-balanced-draft-v1')


def main():
    source = ROOT / 'proposed.jsonl'
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    out = ROOT / 'review-materials'
    out.mkdir(exist_ok=False)
    index = []
    for r in rows:
        folder = out / r['sample_id']
        folder.mkdir()
        path = folder / 'messages.json'
        path.write_text(json.dumps(r['messages'], ensure_ascii=False, indent=2) + '\n')
        if json.loads(path.read_text()) != r['messages']:
            raise ValueError('Lossy review copy')
        provenance = {k: r.get(k) for k in ('sample_id', 'source_record_sha256',
            'dataset', 'source_revision', 'source_file', 'source_row', 'upstream_file',
            'source_split', 'source_license', 'split_group', 'source_papers',
            'retrieved_paper_titles', 'partition', 'gap_category', 'target_domain',
            'valid_tokens', 'token_sha256', 'objective_text', 'success_verification')}
        (folder / 'provenance.json').write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + '\n')
        index.append(dict(sample_id=r['sample_id'], partition=r['partition'],
            category=r['gap_category'], tokens=r['valid_tokens'], goal=r['objective_text'],
            messages=str(path.relative_to(ROOT)), messages_file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            roles=dict(collections.Counter(m['role'] for m in r['messages'])),
            content_review_status='pending'))
    (out / 'index.json').write_text(json.dumps(dict(
        proposed_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), records=index,
        native_history_copies_lossless=True, full_content_review=False,
        release_ready=False), ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(histories=len(index), full_content_review=False, release_ready=False)))


if __name__ == '__main__':
    main()
