"""Discover explicit FX hedge sizing and credit-risk interpretation histories."""
import importlib.util
from pathlib import Path
import sys

ROOT = Path('/mnt/nvme4/mimo-reap')
spec = importlib.util.spec_from_file_location('hedge_goal_helpers',
    ROOT / 'public-workflow-gap-fill-v3/finance_goal_extension.py')
G = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = G
spec.loader.exec_module(G)
G.OUT = ROOT / 'public-workflow-gap-fill-v3/financial-hedge-extension-v1'
G.A.A.OUT = G.OUT
G.PATTERNS = {'finance_risk_methods': r'CHF.futures.contract.*hedge|current.credit.spread'}

if __name__ == '__main__':
    {'discover': G.discover, 'normalize': G.A.A.normalize, 'screen': G.A.A.screen}[sys.argv[1]]()
