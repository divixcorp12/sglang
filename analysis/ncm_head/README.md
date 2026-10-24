# NCM head experiment preparation

This is an isolated research implementation for the corrected [experiment protocol](/Users/dnikolaidis/Desktop/divix/sglang-nvfp4/docs/hx/ncm-head-experiments.draft.md). It is not connected to production serving. The user requested direct setup without hx and no experiment launch yet.

**Current status: CPU reference checks pass; the first native weight/gradient probe is prepared but unrun. The acceptance pilot is not yet launch-ready.** Real prompt groups, correctly aligned native captures, and native forward/gradient evidence are still required. The capture writer is an explicit integration API, not an already-wired SGLang capture driver. A native scheduler/driver must supply committed processed rows and actual-prefix target distributions before training can start.

## What is prepared

- `model.py`: revision-3 memory encoder and FP32 normalized EMAs; hierarchical and flat four-segment planners; zero-initialized token adapter; single-gate prefix head; detached confidence module; fixed future labels including EOS padding.
- `sampling.py`: transformed proposals, causal pre-draw admission and fallback, EOS-aware reference verification, and candidate-ratio acceptance estimator. This is eager reference code, not a graph-optimized serving path.
- `train.py`: bounded cached-feature training stages, separate reference/rollout conditioning, fixed projection labels, valid-label normalization, optimizer-state continuity and per-branch gradient diagnostics. It has no implicit execution or model-loading entry point.
- `data.py`: typed-by-contract capture records, tensor hashes, split/kind identities and an explicit processed-prefix boundary. The native capture driver is still to be integrated and validated.
- `protocol.py` / `__main__.py`: checkpoint-header inventory without loading tensors, grouped-corpus validation, fail-closed artifact preflight, analytical screen sizing, and paired crossed seed/prompt bootstrap.
- `native.py`: native EXL3 forward with an explicit reconstructed-weight straight-through backward surrogate. Native tensor/gradient parity has not been run.
- `probe.py` / `run-native-probe.sh`: first native weight probe, metadata-only by default; explicit `--execute` required for GPU work. It takes the existing GPU lock nonblockingly, refuses mismatched source/configuration, and does not launch a server.
- `pilot.json`: fixed-final feasibility pilot, C/F-only producer banks, separate target/proposal transforms, native checkpoint dimensions/hashes and exact control widths.

The native frozen projection preserves the EXL3 forward. Its backward is the reconstructed linear-weight gradient through low-precision arithmetic; it is a declared surrogate, not a claim that a custom quantized kernel has an exact differentiable implementation. The original normalization boundary also needs native parity checks. A synthetic weight probe alone cannot establish feature alignment or full-head parity on real draft rows.

## Astra corrections now applied

The 128-training/32-development pilot estimates feasibility, training behavior and variability; it cannot reject the architecture for failing a +0.10 threshold. Its scheduled final endpoint is used. The other 64 development groups stay out of selection until one independently budgeted screen. Meaningful effect size derives from measured reward/cost, and the screen's size is frozen from prompt variability before its outcomes are seen.

The flat controls use the same four segments and observed input access as F. `F-flat-aux` receives the same future labels, completing hierarchy × supervision. Every comparison uses C/F-only producer banks. Replication retains the same recipe; three seeds provide weak evidence about training-population tails, reported separately from prompt uncertainty. A bounded paired training extension is specified if the first recipe is inadequate.

Native dimensions were read from checkpoint configuration and tensor headers on divix01, without loading the model: projected context/draft width 5,120; vocabulary 129,280; dependency rank 256; gamma 5; target layers 37/38/39; final norm epsilon 1e-20; EOS 1. The target vocabulary head is a 6-bit EXL3 trellis, not a dense `weight` parameter.

| Arm | Active non-vocabulary proposal parameters | Total trainable parameters | Adjustable hidden width |
| --- | ---: | ---: | ---: |
| C | 3,609,216 | 69,800,576 | 1,200 |
| F / F-noaux | 3,606,144 | 69,797,504 | hierarchical: 256 |
| F-flat / F-flat-aux | 3,606,144 | 69,797,504 | 384 |

Counts exclude frozen vocabulary weights and confidence-only parameters. C differs by about 0.085%; both flat arms match F exactly. These are algebraic parameter counts, not measured memory or latency. Float32 dependency tables dominate trainable storage; Adam states and gradients add substantial memory. The reconstructed native head alone occupies approximately 1.23 GiB in FP16, with extra temporary/backward memory to measure.

## Local CPU checks

From the repository root, use an isolated environment. These commands test tiny fixtures; they do not load native weights or estimate model acceptance:

```bash
uv venv --python 3.12 /tmp/ncm-head-venv
uv pip install --python /tmp/ncm-head-venv/bin/python -r analysis/ncm_head/requirements.txt
/tmp/ncm-head-venv/bin/python -m pytest analysis/ncm_head/test_ncm_head.py -q
/tmp/ncm-head-venv/bin/python -m analysis.ncm_head preflight
```

The last command currently exits 2 with missing real corpus and captured native bundle. A report of `ready: false` is expected. It will not launch a process to manufacture those artifacts.

The suite covers stock-logit initialization, state ordering/chunking/watermarks, FP32 EMA coefficients under BF16 conversion, zero-mean residuals and the flat-supervision identity, EOS/truncation, prefix expressivity, confidence gradient isolation, causal cutoff and invalid-score fallback, exact rational sampled-distribution enumeration with a deliberately biased negative control, filter gradients, capacity matching, group leakage, crossed pairing, capture roundtrip, and a tiny optimizer plumbing check. The native backward bridge has a CPU dense oracle check; that is not an EXL3 GPU validation.

Observed setup results: **31 tests passed locally** using PyTorch 2.14.1; **the same 31 passed on divix01 with CUDA hidden** using its existing PyTorch 2.13.0 environment. Ruff checks passed for this package. No repository-wide suite or native GPU test was run.

## Staged native environment

Isolated remote staging root:

```text
divix01:/tmp/ncm-head-setup-20261018.MHdoQR
```

Its `python/` directory is a source snapshot from local commit `d82ae69d0368478b5018d4cbfa92ab4643a43f54`; `analysis/ncm_head/` contains this setup. This avoids the remote interpreter's default editable installation, which points at an older worktree lacking the required EXL3 module. The wrapper sets `PYTHONPATH` only for its own process, uses a separate extension build directory, and does not change the global installation. The local package is authoritative; `/tmp` staging is disposable.

Verified through read-only inspection: both checkpoint directories and configuration hashes, source hashes for the key EXL3/model files, the pinned exllamav3 checkout `02aef45cd681b960a00afcd0749a4ab99e6c1bfe`, CUDA 13.2 compiler path, and GCC toolset-15 compiler path. Whole checkpoint weights have not been hashed, and free GPU capacity has not been claimed.

The following remote command is **dry-run only**, and has been exercised during setup:

```bash
cd /tmp/ncm-head-setup-20261018.MHdoQR
bash analysis/ncm_head/run-native-probe.sh
```

When a native probe is explicitly requested later, the prepared command is:

```bash
cd /tmp/ncm-head-setup-20261018.MHdoQR
bash analysis/ncm_head/run-native-probe.sh --execute --output /tmp/ncm-head-weight-probe-20261018
```

That command has **not** been run. It uses the native head with synthetic five-row inputs to inspect forward differences and finite/nonzero surrogate gradients. It does not train a head, capture target continuations, benchmark throughput, start/stop a production service, or mark the pilot ready. The result must be reviewed before native capture work.

## Data contract and remaining native preparation

A real prompt registry is JSONL with `id`, `group`, `split`, `domain`, `length_bin`, `source`, and `text`. Related variants share a group and cannot cross splits. The split names are `train`, `select`, `screen`, `calibrate`, `threshold`, and `test`. No real registry has been assembled in this setup; test fixtures are not a substitute. Full-study sources and selection versions must be frozen as required by the protocol.

Native capture must produce the projected target features from `project_target_hidden`, post-HC/pre-norm backbone rows, anchor IDs, reference future rows/tokens, and target distributions at each recorded actual draft prefix. It must prove watermark/commit alignment and preserve separate target, proposal and label laws. `CaptureWriter` expects these explicit inputs and intentionally does not guess them from physical KV or padded tensor lengths. The existing top-probability trace cannot supply this data.

Before the acceptance pilot can launch:

1. Run and assess the prepared native weight/gradient probe after authorization.
2. Integrate the capture writer at the native logical commit/anchor boundary, validate it on actual rows and verify zero-initialized full-head logits against stock.
3. Assemble and freeze the real grouped corpus; collect reference and own-prefix rollout records under their distinct laws after capture is authorized.
4. Save frozen weight/projection artifacts with complete hashes and evidence for feature alignment, forward parity and input gradients. Measure capture/training-step resource cost.
5. Freeze the resource budget and run C/F warm start, collect a shared producer bank, then run each declared refinement stage with fresh bank identities. Neither stage orchestration nor serving integration should be inferred from the reference primitives.

These are actual remaining requirements, not completed checks. Readiness remains false until they are met. Later confidence calibration, independent screening, expanded controls and closed-loop serving follow the revised protocol; none were started.
