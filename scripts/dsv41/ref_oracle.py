"""Reference oracle: DeepSeek's own `inference/model.py`, truncated to N layers + head, running on
EXL3 weights dequantized to bf16 (so quantization error cancels against SGLang's own EXL3 path in
Task 9's comparison -- neither side compares to a from-scratch bf16 checkpoint).

Two classes the reference builds by name are patched before construction:
  - `ref.Expert -> LazyExpert`: dequantizes its EXL3 trellis tensors on first use instead of
    holding a dense state dict (the routed-expert dense weights would otherwise be huge).
  - `ref.ParallelEngramEmbedding -> FileEngram`: reads rows from the memmapped
    `EngramFileTable` (Task 6) instead of loading the whole (101.5 GB) table into memory.

CLI:
  --make-prompts K   write ANA/prompts.jsonl from the corpus's first K sessions (CPU only: reads
                     the JSONL and the tokenizer, builds no model).
  --router-corpus N  run the first N sessions' first turns through the model, accumulate routed
                     expert counts, and write ANA/router-<tag>.json (router_stats.py's summaries).
  --prompts PATH     run each prompt (a JSONL of {"tokens": [...]}) through the model and write
                     ANA/oracle-<tag>.npz (top-20 logits, hc-mean hidden states, router ids).

The model-building paths (--router-corpus, --prompts) need a GPU and the EXL3/Engram data on
/mnt/nvme2; they only run inside a granted GPU window (Tasks 10/11). --make-prompts and
`reference_args` need neither and are what this task's own tests exercise.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

import numpy as np

# All EXL3-dequantizing pieces of this module (Exl3Tensors, exl3_dense_weight) are pure GPU
# kernels; importing sglang itself does not need a GPU.
from sglang.srt.layers.engram_file_table import EngramFileTable
from sglang.srt.layers.quantization.exl3_ops import Exl3Tensors, exl3_dense_weight

_DEFAULT_SESSIONS = "/mnt/nvme2/nvfp4-work/benchmarks/full/sessions.jsonl"


def reference_args(ref_config: dict, n_layers: int, max_seq_len: int) -> dict:
    """ModelArgs kwargs for the truncated model: the reference's own `inference/config.json`
    (whose keys already match `ModelArgs` field names -- unlike the EXL3 checkpoint's HF-style
    config that Task 7's `truncate_text_config` edits) cut down to the first `n_layers` decoder
    layers plus head, with MTP/vision disabled.
    """
    cfg = dict(ref_config)
    cfg["n_layers"] = n_layers
    cfg["n_mtp_layers"] = 0
    cfg["dspark_block_size"] = 0  # Transformer.__init__ only builds DSparkBlocks when this is set
    cfg["compress_ratios"] = list(cfg["compress_ratios"])[:n_layers]
    cfg["kv_source_layers"] = [i for i in cfg["kv_source_layers"] if i < n_layers]
    cfg["index_source_layers"] = [i for i in cfg["index_source_layers"] if i < n_layers]
    if cfg["candidate_source_layer"] >= n_layers:
        cfg["candidate_source_layer"] = -1
    kept = [(i, r) for i, r in zip(cfg["engram_layer_ids"], cfg["engram_num_embeddings"]) if i < n_layers]
    cfg["engram_layer_ids"] = [i for i, _ in kept]
    cfg["engram_num_embeddings"] = [r for _, r in kept]
    cfg["vision_n_layers"] = 0
    cfg["dtype"] = "bf16"
    cfg["expert_dtype"] = None
    cfg["max_batch_size"] = 1
    cfg["max_seq_len"] = max_seq_len
    cfg["temperature"] = 0  # only the logits are used; no sampling
    return cfg


def _import_reference(snapshot: str):
    """Import the reference `model.py`/`engram.py` by their own module names (they import each
    other that way) and switch the default dtype to bf16 for every weight the reference allocates
    without an explicit dtype."""
    import torch

    inference_dir = os.path.join(snapshot, "inference")
    if inference_dir not in sys.path:
        sys.path.insert(0, inference_dir)
    import engram as ref_engram  # noqa: F401 -- imported for parity with model.py's own import
    import model as ref

    torch.set_default_dtype(torch.bfloat16)
    return ref, ref_engram


def _make_lazy_expert():
    import torch
    import torch.nn.functional as F

    class LazyExpert(torch.nn.Module):
        """Drop-in for `ref.Expert` that keeps its three EXL3 tensors and dequantizes on every
        call instead of holding a dense state dict. `keep_dense=True` (the shared expert) instead
        dequantizes once at bind time and caches the result."""

        def __init__(self, dim: int, inter_dim: int, dtype=None, swiglu_limit: float = 0.0):
            super().__init__()
            self.dim = dim
            self.inter_dim = inter_dim
            self.swiglu_limit = swiglu_limit
            self._tensors: tuple[Exl3Tensors, Exl3Tensors, Exl3Tensors] | None = None
            self._keep_dense = False
            self._dense: tuple = ()

        def bind(self, w1: Exl3Tensors, w2: Exl3Tensors, w3: Exl3Tensors, keep_dense: bool) -> None:
            self._tensors = (w1, w2, w3)
            self._keep_dense = keep_dense
            self._dense = self._dequant() if keep_dense else ()

        def _dequant(self):
            w1, w2, w3 = self._tensors
            # exl3_dense_weight(t) is [in, out]; nn.Linear's convention (and F.linear) wants
            # [out, in], matching ref.Expert's own w1/w2/w3 = Linear(dim, inter_dim)-shaped weights.
            return tuple(exl3_dense_weight(t).t().to(torch.bfloat16) for t in (w1, w2, w3))

        def forward(self, x, weights=None):
            # Mirrors ref.Expert.forward exactly (model.py:841-851): up clamped both ways, gate
            # clamped from above, both in fp32; the route weight is applied before w2.
            w1, w2, w3 = self._dense if self._keep_dense else self._dequant()
            dtype = x.dtype
            gate = F.linear(x, w1).float()
            up = F.linear(x, w3).float()
            if self.swiglu_limit > 0:
                up = torch.clamp(up, min=-self.swiglu_limit, max=self.swiglu_limit)
                gate = torch.clamp(gate, max=self.swiglu_limit)
            h = F.silu(gate) * up
            if weights is not None:
                h = weights * h
            return F.linear(h.to(dtype), w2)

    return LazyExpert


def _make_file_engram():
    import torch

    class FileEngram(torch.nn.Module):
        """Drop-in for `ref.ParallelEngramEmbedding`, backed by a memmapped `EngramFileTable`
        instead of an in-memory nn.Parameter (the layer-1 table alone is 101.5 GB)."""

        def __init__(self, num_embeddings: int, dim: int):
            super().__init__()
            self.num_embeddings = num_embeddings
            self.dim = dim
            self._table: EngramFileTable | None = None

        def bind(self, table: EngramFileTable) -> None:
            self._table = table

        def forward(self, indices):
            assert self._table is not None, "FileEngram.bind() was never called"
            return self._table.lookup(indices)

    return FileEngram


# --- checkpoint loading -----------------------------------------------------------------------

_ROUTED_RE = re.compile(r"^(layers\.\d+\.ffn\.experts\.\d+)\.(w[123])\.(suh|svh|mul1|trellis)$")
_SHARED_RE = re.compile(r"^(layers\.\d+\.ffn\.shared_experts)\.(w[123])\.(suh|svh|mul1|trellis)$")
_SLICE_RE = re.compile(r"^(layers\.\d+\.attn\.wo_a)\.slice\.(\d+)\.(suh|svh|mul1|trellis)$")
_EXL3_RE = re.compile(r"^(.+)\.(suh|svh|mul1|trellis)$")
_LAYER_EXPERT_RE = re.compile(r"^layers\.(\d+)\.ffn\.experts\.(\d+)$")
_LAYER_RE = re.compile(r"^layers\.(\d+)\.ffn\.shared_experts$")


def _exl3_tensors(parts: dict) -> Exl3Tensors:
    return Exl3Tensors(
        trellis=parts["trellis"],
        suh=parts["suh"],
        svh=parts["svh"],
        mul1=bool(parts["mul1"].item()),
    )


def _load_checkpoint(trunc_dir: str, device: str):
    """Read every tensor in `trunc_dir`'s (Task 7-filtered) index, split into: a plain `state`
    dict ready for `load_state_dict` (dense/plain tensors, including every dequantized-then-
    concatenated EXL3 stem), plus the routed/shared expert `Exl3Tensors` grouped by (layer,
    expert) so the caller can `.bind()` them onto the already-built model."""
    from safetensors import safe_open

    with open(os.path.join(trunc_dir, "model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]

    handles = {
        shard: safe_open(os.path.join(trunc_dir, shard), framework="pt", device=device)
        for shard in sorted(set(weight_map.values()))
    }

    def get(name: str):
        return handles[weight_map[name]].get_tensor(name)

    routed_parts: dict[tuple, dict] = {}
    shared_parts: dict[tuple, dict] = {}
    slice_parts: dict[tuple, dict] = {}
    generic_parts: dict[str, dict] = {}
    plain: dict = {}

    for name in weight_map:
        m = _ROUTED_RE.match(name)
        if m:
            routed_parts.setdefault((m.group(1), m.group(2)), {})[m.group(3)] = get(name)
            continue
        m = _SHARED_RE.match(name)
        if m:
            shared_parts.setdefault((m.group(1), m.group(2)), {})[m.group(3)] = get(name)
            continue
        m = _SLICE_RE.match(name)
        if m:
            slice_parts.setdefault((m.group(1), int(m.group(2))), {})[m.group(3)] = get(name)
            continue
        m = _EXL3_RE.match(name)
        if m:
            generic_parts.setdefault(m.group(1), {})[m.group(2)] = get(name)
            continue
        plain[name] = get(name)

    import torch

    state = dict(plain)

    # Every other EXL3 stem -> dense [out, in] bf16 as "<stem>.weight".
    for stem, parts in generic_parts.items():
        state[f"{stem}.weight"] = exl3_dense_weight(_exl3_tensors(parts)).t().to(torch.bfloat16)

    # attn.wo_a.slice.g -> dense, concatenated to attn.wo_a.weight [G*R, D] (o_groups groups of
    # o_lora_rank rows each; see Attention.__init__'s ColumnParallelLinear(..., o_groups*o_lora_rank)).
    by_stem: dict[str, dict[int, dict]] = {}
    for (stem, group), parts in slice_parts.items():
        by_stem.setdefault(stem, {})[group] = parts
    for stem, groups in by_stem.items():
        pieces = [exl3_dense_weight(_exl3_tensors(groups[g])).t().to(torch.bfloat16) for g in sorted(groups)]
        state[f"{stem}.weight"] = torch.cat(pieces, dim=0)

    routed = {}
    for (stem, wk), parts in routed_parts.items():
        m = _LAYER_EXPERT_RE.match(stem)
        layer, expert = int(m.group(1)), int(m.group(2))
        routed.setdefault((layer, expert), {})[wk] = _exl3_tensors(parts)

    shared = {}
    for (stem, wk), parts in shared_parts.items():
        m = _LAYER_RE.match(stem)
        layer = int(m.group(1))
        shared.setdefault(layer, {})[wk] = _exl3_tensors(parts)

    return state, routed, shared


def build_model(ref, snapshot: str, trunc_dir: str, engram_dir: str, max_seq_len: int):
    """Build the truncated `ref.Transformer` on `cuda`, load the (Task 7-filtered) checkpoint,
    and bind the routed/shared experts plus the layer-1 Engram table. Needs a GPU."""
    import torch

    with open(os.path.join(trunc_dir, "config.json")) as f:
        checkpoint_config = json.load(f)
    with open(os.path.join(snapshot, "inference", "config.json")) as f:
        ref_config = json.load(f)
    n_layers = checkpoint_config["text_config"]["num_hidden_layers"]
    kwargs = reference_args(ref_config, n_layers, max_seq_len)

    ref.Expert = _make_lazy_expert()
    ref.ParallelEngramEmbedding = _make_file_engram()

    from transformers import AutoTokenizer

    # The reference Engram builds its compressed token map from the tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(trunc_dir)
    with torch.device("cuda"):
        model = ref.Transformer(ref.ModelArgs(**kwargs), tokenizer=tokenizer)

    state, routed, shared = _load_checkpoint(trunc_dir, device="cuda")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise AssertionError(f"unexpected_keys was not empty: {unexpected}")
    buffer_names = {name for name, _ in model.named_buffers()}
    bad_missing = [name for name in missing if name not in buffer_names]
    if bad_missing:
        raise AssertionError(f"missing_keys had non-buffer entries: {bad_missing}")
    print(f"missing_keys: {missing}")
    print(f"unexpected_keys: {unexpected}")

    for (layer, expert), tensors in routed.items():
        model.layers[layer].ffn.experts[expert].bind(tensors["w1"], tensors["w2"], tensors["w3"], keep_dense=False)
    for layer, tensors in shared.items():
        model.layers[layer].ffn.shared_experts.bind(tensors["w1"], tensors["w2"], tensors["w3"], keep_dense=True)

    # Bind straight off the layout the model actually built (not a second, independently
    # filtered read of the config), so a mismatch between the two config files can't silently
    # bind the wrong layer or row count.
    if model.engram_layout is not None:
        for layer_id, rows in zip(model.engram_layout.layer_ids, model.engram_layout.num_embeddings):
            table = EngramFileTable.open(engram_dir, layer_id, rows, model.engram_layout.head_dim)
            model.layers[layer_id].engram.embed.bind(table)

    model.eval()
    return model


# --- forward with captures ---------------------------------------------------------------------


def run_oracle_forward(ref, model, input_ids):
    """Replicates `Transformer.forward` (inference/model.py:1241-1273) for `start_pos=0`, adding:
      - `hidden`: hc-mean of the residual stream before layer 0 and after every block, [L+1,T,D];
      - `router_ids`: each block's `MoE.gate` top-k expert indices, [L,T,topk] (via forward hooks);
      - full-vocabulary top-20 log-probabilities instead of the reference's last-position sampling.
    """
    import torch
    import torch.nn.functional as F

    router_indices = []

    def _hook(_module, _inputs, output):
        router_indices.append(output[1].detach())

    handles = [layer.ffn.gate.register_forward_hook(_hook) for layer in model.layers]
    try:
        with torch.inference_mode():
            engram_hashes = (
                model.engram_hash(input_ids, 0, None) if model.engram_hash is not None else None
            )
            h = model.embed(input_ids)
            h = h.unsqueeze(2).repeat(1, 1, model.hc_mult, 1)
            hiddens = [h.mean(dim=2).float()]
            pre_mix = ref.make_identity_pre_mix(h, model.hc_mult)
            layer = None
            for i, layer in enumerate(model.layers):
                if layer.engram is not None:
                    h = layer.engram(h, engram_hashes[:, :, layer.engram.layer_hash_index, :], None)
                h, pre_mix = layer(h, 0, pre_mix, None)
                hiddens.append(h.mean(dim=2).float())
            h = layer.hc_pre(h, pre_mix)
            logits = model.head(model.norm(h), full_logits=True)
    finally:
        for handle in handles:
            handle.remove()

    hidden = torch.stack(hiddens, dim=0)[:, 0]  # [L+1, T, D]
    # Gate.forward runs on x.view(-1, dim) (MoE.forward), and batch=1 here, so each hooked
    # output is already [T, topk] with no batch dim to index out.
    router_ids = torch.stack(router_indices, dim=0)  # [L, T, topk]
    logprobs = F.log_softmax(logits.float(), dim=-1)[0]  # [T, V]
    top = logprobs.topk(20, dim=-1)
    return hidden, router_ids, top.indices, top.values


# --- CLI -----------------------------------------------------------------------------------


def _load_first_turns(sessions_path: str, n: int):
    with open(sessions_path) as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            yield json.loads(line)["turns"][0]


def _write_prompts(trunc_dir: str, sessions_path: str, k: int, out_dir: str, max_tokens: int = 256) -> str:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(trunc_dir)
    out_path = os.path.join(out_dir, "prompts.jsonl")
    count = 0
    with open(out_path, "w") as out:
        for text in _load_first_turns(sessions_path, k):
            ids = tokenizer(text).input_ids[:max_tokens]
            out.write(json.dumps({"tokens": ids}) + "\n")
            count += 1
    print(f"wrote {count} prompts to {out_path}")
    return out_path


def _run_prompts(ref, model, prompts_path: str, tag: str, out_dir: str) -> str:
    import torch

    device = next(model.parameters()).device
    out = {}
    with open(prompts_path) as f:
        rows = [json.loads(line)["tokens"] for line in f]
    for i, tokens in enumerate(rows):
        input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
        hidden, router_ids, top_ids, top_lp = run_oracle_forward(ref, model, input_ids)
        out[f"tokens_{i}"] = np.asarray(tokens, dtype=np.int32)
        out[f"top_ids_{i}"] = top_ids.to(torch.int32).cpu().numpy()
        out[f"top_logprobs_{i}"] = top_lp.to(torch.float32).cpu().numpy()
        out[f"hidden_{i}"] = hidden.to(torch.float16).cpu().numpy()
        out[f"router_ids_{i}"] = router_ids.to(torch.int16).cpu().numpy()
    out_path = os.path.join(out_dir, f"oracle-{tag}.npz")
    np.savez(out_path, **out)
    print(f"wrote {len(rows)} prompts' captures to {out_path}")
    return out_path


def _run_router_corpus(ref, model, trunc_dir: str, sessions_path: str, n_sessions: int, tag: str, out_dir: str) -> str:
    import torch

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import router_stats as rs  # scripts/dsv41/router_stats.py

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(trunc_dir)
    device = next(model.parameters()).device
    n_layers = len(model.layers)
    n_experts = model.layers[0].ffn.n_routed_experts
    counts = np.zeros((n_layers, n_experts), dtype=np.int64)

    for text in _load_first_turns(sessions_path, n_sessions):
        ids = tokenizer(text).input_ids[:1024]
        if not ids:
            continue
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        _, router_ids, _, _ = run_oracle_forward(ref, model, input_ids)
        flat = router_ids.reshape(n_layers, -1).cpu().numpy()
        for layer in range(n_layers):
            counts[layer] += np.bincount(flat[layer], minlength=n_experts)

    summary = rs.skew_summary(counts)
    residencies = {"7.9%": 0.079, "11.5%": 0.115, "36.7%": 0.367}
    hit_rates = {label: rs.cache_hit_rate(counts, frac) for label, frac in residencies.items()}
    report = {
        "n_sessions": n_sessions,
        "n_layers": n_layers,
        "n_experts": n_experts,
        "skew_summary": summary,
        "cache_hit_rate": hit_rates,
    }
    out_path = os.path.join(out_dir, f"router-{tag}.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"wrote {out_path}")
    return out_path


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True, help="official DeepSeek-V4.1-Flash snapshot dir")
    parser.add_argument("--trunc", required=True, help="truncated EXL3 model dir (scripts/dsv41/make_truncated_model.py)")
    parser.add_argument("--engram-dir", required=True, help="dir holding the layer-1 Engram file-table shard")
    parser.add_argument("--out-dir", default="ANA")
    parser.add_argument("--tag", help="suffix for oracle-<tag>.npz / router-<tag>.json")
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--sessions", default=_DEFAULT_SESSIONS, help="sessions.jsonl for --router-corpus/--make-prompts")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prompts", help="JSONL of {\"tokens\": [...]} to run through the model")
    group.add_argument("--router-corpus", type=int, metavar="N", help="first N sessions -> router-<tag>.json")
    group.add_argument("--make-prompts", type=int, metavar="K", help="write ANA/prompts.jsonl from the first K sessions")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    if args.make_prompts is not None:
        # CPU only: reads the JSONL and the tokenizer, builds no model.
        _write_prompts(args.trunc, args.sessions, args.make_prompts, args.out_dir)
        return 0

    if not args.tag:
        raise SystemExit("--tag is required for --prompts/--router-corpus")

    ref, _ = _import_reference(args.snapshot)
    model = build_model(ref, args.snapshot, args.trunc, args.engram_dir, args.max_seq_len)

    if args.router_corpus is not None:
        _run_router_corpus(ref, model, args.trunc, args.sessions, args.router_corpus, args.tag, args.out_dir)
    else:
        _run_prompts(ref, model, args.prompts, args.tag, args.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
