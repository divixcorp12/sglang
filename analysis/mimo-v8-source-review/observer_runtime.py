"""Resumable unpruned single-sequence text observation. Pilot artifacts only.

Reuses LLM Compressor's actual REAP hooks, preserving sums and counts, not its
conditional means or pruning callbacks. Full oneshot integration remains separate.
"""
import hashlib
import json
import os
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from llmcompressor.core import State
from llmcompressor.modifiers.pruning import REAPPruningModifier

from .adapter import register_reap_config
from .decoder import build_layer, load_config, position_inputs, source_core
from .packed import PackedCheckpoint


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity(root, token_ids, domain, attention_backend="eager", synthetic=False):
    return dict(schema="mimo_observer_pilot_v1", calibration_qualified=False, domain=domain,
                token_ids=token_ids, precision="decoded_bf16_no_fp8_activation_simulation", attention_backend=attention_backend, synthetic=synthetic,
                source={n:digest(Path(root)/n) for n in ("config.json","model.safetensors.index.json","modeling_mimo_v2.py")},
                implementation={p.name:digest(p) for p in Path(__file__).parent.glob("*.py")})


def check_identity(expected, existing):
    if expected != existing: raise ValueError("Resume identity mismatch: source, input, domain or code changed")


def atomic_json(path, data):
    path=Path(path)
    temporary=path.with_suffix(path.suffix+".partial")
    with temporary.open("w") as f:
        json.dump(data,f,indent=2); f.write("\n"); f.flush(); os.fsync(f.fileno())
    temporary.replace(path)


def save_layer(directory, layer, x, stats):
    path=Path(directory)/f"layer-{layer:02d}.pt"
    temp=path.with_suffix(".partial")
    with temp.open("wb") as f:
        torch.save({"hidden_states":x.cpu(),"statistics":stats},f)
        f.flush(); os.fsync(f.fileno())
    temp.replace(path)
    # A commit marker comes last. An interrupted payload is never resumable.
    atomic_json(path.with_suffix(".json"),dict(layer=layer,payload_sha256=digest(path)))


def load_layer(directory, layer):
    path=Path(directory)/f"layer-{layer:02d}.pt"
    marker=path.with_suffix(".json")
    if not marker.exists(): return None
    record=json.loads(marker.read_text())
    if record["layer"] != layer or record["payload_sha256"] != digest(path):
        raise ValueError("Corrupted layer cache")
    return torch.load(path,weights_only=True,map_location="cpu")


class ObserveOnlyREAP(REAPPruningModifier):
    def on_sequential_epoch_end(self,state,event,**kwargs):
        pass  # Never prune the original observation pass.

    def on_finalize(self,state,**kwargs):
        self.remove_hooks()
        self._saliency_trackers.clear()
        self._norm_buffers.clear()
        return True  # Deliberately do not update expert-count config.


def run(root, token_ids, domain, directory, stop_layer=None, progress=None, attention_backend="eager", synthetic=False):
    directory=Path(directory); directory.mkdir(parents=True,exist_ok=True)
    current=identity(root,token_ids,domain,attention_backend,synthetic)
    manifest=directory/"identity.json"
    if manifest.exists(): check_identity(current,json.loads(manifest.read_text()))
    else: atomic_json(manifest,current)
    checkpoint=PackedCheckpoint(root)
    cfg=load_config(root,attention_backend)
    if domain not in {"finance","science","coding","general_tool_use"}: raise ValueError("Unknown domain")
    if not token_ids: raise ValueError("Empty input")
    last=cfg.num_hidden_layers if stop_layer is None else stop_layer
    if not 1 <= last <= cfg.num_hidden_layers: raise ValueError("Invalid stop layer")
    x=checkpoint.rows("model.embed_tokens.weight",token_ids,"cuda").unsqueeze(0)
    if x.dtype != torch.bfloat16: raise ValueError("Only source BF16 embeddings are supported")
    register_reap_config()
    layer_reports={}
    with torch.inference_mode():
        for layer_id in range(last):
            cached=load_layer(directory,layer_id)
            if cached is not None:
                x=cached["hidden_states"].to("cuda")
                stats=cached["statistics"]
                if x.shape != (1,len(token_ids),cfg.hidden_size) or not torch.isfinite(x).all():
                    raise ValueError("Invalid cached activation")
                if progress: progress(layer_id,True,checkpoint)
            else:
                layer=build_layer(checkpoint,cfg,layer_id)
                state=None; observer=None
                if cfg.moe_layer_freq[layer_id]:
                    model=nn.Module(); model.config=cfg; model.mlp=layer.mlp
                    state=State(model=model)
                    observer=ObserveOnlyREAP(sparsity=.1)  # Required upstream field; unused.
                    observer.on_initialize(state); observer.on_calibration_start(state,None)
                try:
                    x=layer(x,**position_inputs(cfg,layer_id,x))
                    if not torch.isfinite(x).all(): raise ValueError(f"Nonfinite layer {layer_id}")
                    stats=None
                    if observer is not None:
                        tracker=observer._saliency_trackers["mlp"]
                        if tracker.total_count != len(token_ids)*cfg.num_experts_per_tok:
                            raise ValueError("Routed token count mismatch")
                        stats=dict(domain=domain,valid_tokens=len(token_ids),
                                   gate_weighted_output_norm_sum=tracker.sum_saliency.cpu().tolist(),
                                   route_counts=tracker.count.cpu().tolist())
                        assert len(layer.mlp.experts)==cfg.n_routed_experts==384
                    save_layer(directory,layer_id,x,stats)
                finally:
                    if observer is not None: observer.on_calibration_end(state,None); observer.on_finalize(state)
                del layer
                if observer is not None: del observer,state,model
                if progress: progress(layer_id,False,checkpoint)
            if stats is not None: layer_reports[str(layer_id)]=stats
    report=dict(schema="mimo_observer_pilot_v1",calibration_qualified=False,synthetic=synthetic,
                token_count=len(token_ids),domain=domain,completed_layers=last,
                full_stack_completed=(last==cfg.num_hidden_layers),statistics=layer_reports,
                checkpoint_bytes_read=checkpoint.bytes_read,checkpoint_reads=checkpoint.reads)
    if last==cfg.num_hidden_layers:
        norm=source_core(checkpoint.root)["MiMoV2RMSNorm"](cfg.hidden_size,eps=cfg.layernorm_epsilon).to("cuda")
        norm.weight=nn.Parameter(checkpoint.tensor("model.norm.weight","cuda"),requires_grad=False)
        # Last-position logits only. Stream the head in vocabulary row chunks.
        with torch.inference_mode():
            final=norm(x)[:,-1,:]
            logits=[]
            for _,weight in checkpoint.row_chunks("lm_head.weight",device="cuda"):
                if weight.dtype!=torch.bfloat16: raise ValueError("Only BF16 source head is supported")
                logits.append(F.linear(final,weight).float().cpu())
            logits=torch.cat(logits,-1)
            if not torch.isfinite(logits).all(): raise ValueError("Nonfinite logits")
            top=torch.topk(logits,5,dim=-1)
            report["last_position_top5_ids"]=top.indices.tolist()
            report["last_position_top5_logits"]=top.values.tolist()
            report["checkpoint_bytes_read"]=checkpoint.bytes_read
            report["checkpoint_reads"]=checkpoint.reads
    atomic_json(directory/"observation.json",report)
    return report
