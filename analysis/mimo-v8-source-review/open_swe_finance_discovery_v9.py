"""Bounded public training-source discovery; never admits a history by repo name."""
import collections
import concurrent.futures
import hashlib
import json
from pathlib import Path
import re

from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq

ROOT = Path('/mnt/nvme4/mimo-reap')
SOURCE = ROOT / 'public-workflow-gap-fill-v3/external-source-discovery/open-swe-traces'
OUT = SOURCE / 'finance-scan-v1'
REPOS = re.compile(r'quantlib|portfolio|backtest|freqtrade|finrl|qlib|yfinance|finance|financ|trading|pyfolio|riskfolio|backtrader|\bbt\b|zipline|vectorbt|ta-lib|pandas-ta|hummingbot|ccxt', re.I)


def main():
    meta = json.loads((SOURCE / 'metadata.json').read_text())
    files = [f for f in meta['files'] if
        f['path'].startswith('data/sweagent/qwen38_27b/scale-swe/')
        and f['path'].endswith('.parquet')][:2]
    assert len(files) == 2
    assert sum(f['size'] for f in files) < 700_000_000
    OUT.mkdir(exist_ok=False)
    (OUT / 'plan.json').write_text(json.dumps(dict(dataset=meta['dataset'],
        revision=meta['revision'], files=files, source_only=True, admitted=False), indent=2))

    def scan(f):
        p = Path(hf_hub_download(meta['dataset'], f['path'], repo_type='dataset',
            revision=meta['revision'], local_dir=str(OUT / 'sources')))
        assert p.stat().st_size == f['size']
        table = pq.read_table(p, columns=['repo', 'instance_id', 'resolved'])
        counts = collections.Counter(table['repo'].to_pylist())
        hits = [dict(source_row=i, **r) for i, r in enumerate(table.to_pylist())
                if REPOS.search(r['repo'])]
        receipt = dict(path=str(p.relative_to(ROOT)),
            sha256=hashlib.sha256(p.read_bytes()).hexdigest(), revision=meta['revision'],
            upstream_file=f['path'], rows=len(table), repository_counts=dict(counts),
            discovery_hits=hits, admitted=False)
        (OUT / (p.stem + '.json')).write_text(json.dumps(receipt, indent=2))
        print(json.dumps(dict(event='source_scanned', source=f['path'], rows=len(table),
            discovery_hits=hits, admitted=False)), flush=True)
        return receipt

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        receipts = list(pool.map(scan, files))
    (OUT / 'inventory.json').write_text(json.dumps(dict(sources=receipts,
        admitted=False, source_only=True), indent=2))


if __name__ == '__main__':
    main()
