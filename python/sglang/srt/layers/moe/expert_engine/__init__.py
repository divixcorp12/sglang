"""The expert engine: quant-agnostic expert streaming.

``formats`` holds the expert format registry (what a quantization format's rows, GPU apply and CPU
kernel are), ``profiles`` the model profile registry (which modules stream, which router they use,
which draft adapter serves them). ``service`` is the process-wide engine built from one of each.
This package imports nothing at load; import the submodule you need.
"""
