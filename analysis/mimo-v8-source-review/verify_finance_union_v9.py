"""CPU evidence checks on the incomplete candidate draft; never declares a release."""
import concurrent.futures
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path('/mnt/nvme4/mimo-reap')
BASE = ROOT / 'public-workflow-gap-fill-v3'
OUT = BASE / 'finance-union-v1'


def main():
    spec = importlib.util.spec_from_file_location('partial_union_release', BASE / 'release.py')
    R = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = R
    spec.loader.exec_module(R)
    C, B = R.C, R.B
    pins = B.read(OUT / 'inputs.lock.json')
    def check_pins():
        for name, h in pins.items():
            if C.file_sha(ROOT / name) != h:
                raise ValueError('Changed candidate input: ' + name)
    check_pins()
    selected = R.A.A.rows(OUT / 'proposed.jsonl')
    before = C.file_sha(OUT / 'proposed.jsonl')
    R.R.OUT = OUT
    R.R.check_additions(selected, R.A.parent())
    for row in selected:
        B.K.record(row)
    original = R.A.A.rows(BASE / 'proposed.jsonl')
    actual = {(r['sample_id'], r['gap_category']): r for r in selected}
    if not all(actual.get((r['sample_id'], r['gap_category'])) == r for r in original):
        raise ValueError('An original draft history changed or disappeared')
    B.W.OUT = OUT
    B.W.DEST = OUT
    verified_sources = B.W.sourceverify(selected)
    with concurrent.futures.ProcessPoolExecutor(max_workers=4, initializer=B.V.init) as pool:
        for _ in pool.map(B.V.check_token, selected, chunksize=1):
            pass
    check_pins()
    if C.file_sha(OUT / 'proposed.jsonl') != before:
        raise ValueError('Proposal changed during verification')
    result = OUT / 'partial-validation.json'
    if result.exists():
        raise ValueError('Preserve previous validation evidence')
    C.save(result, dict(status='partial_draft_checks_passed', source_rows=verified_sources,
        exact_retokenized=len(selected), original_proposed_histories_preserved=len(original),
        complete_native_structure_checked=True, repository_and_lineage_caps_checked=True,
        proposed_sha256=before, unfilled=B.read(OUT / 'proposal.json')['unfilled'],
        full_content_review=False, release_ready=False, scoring_launched=False,
        native_precision_verified=False, selection_allowed=False))
    print(json.dumps(B.read(result)), flush=True)


if __name__ == '__main__':
    main()
