"""Discover financial software workflows without lowering research-content standards."""
import copy
import importlib.util
import sys
from pathlib import Path
ROOT=Path('/mnt/nvme4/mimo-reap');OUT=ROOT/'public-workflow-gap-fill-v2'
spec=importlib.util.spec_from_file_location('finance_preparation',OUT/'prepare.py')
A=importlib.util.module_from_spec(spec);sys.modules[spec.name]=A;spec.loader.exec_module(A)

if __name__=='__main__':
    folder=OUT/'financial-software';folder.mkdir(exist_ok=False)
    pol=copy.deepcopy(A.policy())
    pol['categories']={'finance_research_software':dict(domain='finance',stratum='swe',adapter_family='scientific_numeric',pattern=r'\b(?:backtest\w*|portfolio|quantlib|yfinance|riskfolio|pyfolio|backtrader|freqtrade|qlib|stock price|stock market|trading strateg\w*|financial time series|ohlcv|sharpe|drawdown|option pric\w*|asset pric\w*)\b',content_rule='Actual quantitative finance/trading-data/research-software correctness with observed development tools. Generic business CRUD and crypto promotion are reserved.')}
    A.C.save(folder/'policy.json',pol)
    A.OUT=folder
    A.inventory();A.normalize();A.screen()
