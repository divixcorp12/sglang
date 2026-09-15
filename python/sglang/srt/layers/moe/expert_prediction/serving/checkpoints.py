"""Load trained LLaPor and APEX checkpoints and validate them against the live MoE layer specs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import msgspec
import torch

from sglang.srt.layers.moe.expert_prediction.contracts import MoeLayerSpec
from sglang.srt.layers.moe.expert_prediction.training import apex, llapor
from sglang.srt.layers.moe.expert_prediction.training.pca import PCAStats

PREFETCH_PREDICTORS = ("llapor", "apex")


class LlaporCheckpoint(msgspec.Struct, frozen=True):
    source_layer: int
    target_layer: int
    group: str
    pca: PCAStats
    model: torch.nn.Module


class ApexCheckpoint(msgspec.Struct, frozen=True):
    layer_id: int
    top_k: int
    ranker: apex.Ranker
    cdf: apex.OrdinalCDF


def state_dict_sha256(state_dict: Mapping[str, torch.Tensor]) -> str:
    """The digest the training scripts store under ``tensor_checksums``."""
    digest = hashlib.sha256()
    for key in sorted(state_dict):
        digest.update(key.encode())
        digest.update(state_dict[key].cpu().numpy().tobytes())
    return digest.hexdigest()


def _manifest(directory: Path) -> dict:
    if not (directory / "DONE").exists():
        raise ValueError(f"prefetch checkpoint {directory} is incomplete: no DONE marker")
    return json.loads((directory / "manifest.json").read_text())


def _verified_state(path: Path, expected_sha256: str) -> dict[str, torch.Tensor]:
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state_dict_sha256(state) != expected_sha256:
        raise ValueError(f"prefetch checkpoint {path} does not match its manifest checksum")
    return state


def _spec(specs: Mapping[int, MoeLayerSpec], layer_id: int, directory: Path) -> MoeLayerSpec:
    if layer_id not in specs:
        raise ValueError(f"prefetch checkpoint {directory} names MoE layer {layer_id}, which the model lacks")
    return specs[layer_id]


def _load_llapor(directory: Path, specs: Mapping[int, MoeLayerSpec]) -> LlaporCheckpoint:
    manifest = _manifest(directory)
    architecture, grouping = manifest["architecture"], manifest["grouping"]
    source = _spec(specs, grouping["source_layer"], directory)
    target = _spec(specs, grouping["target_layer"], directory)
    experts = {architecture["num_experts"], source.num_experts, target.num_experts}
    if len(experts) != 1:
        raise ValueError(f"prefetch checkpoint {directory} expert count {architecture['num_experts']} "
                         f"does not match layers {source.layer_id}/{target.layer_id}")
    pca_state = torch.load(directory / "pca.pt", map_location="cpu", weights_only=True)
    if tuple(pca_state["mean"].shape) != (source.hidden_size,):
        raise ValueError(f"prefetch checkpoint {directory} PCA hidden width {tuple(pca_state['mean'].shape)} "
                         f"does not match layer {source.layer_id} hidden size {source.hidden_size}")
    model = llapor.build_predictor(
        architecture["group"], pca_rank=architecture["pca_rank"], num_experts=target.num_experts
    )
    model.load_state_dict(_verified_state(directory / "model.pt", manifest["tensor_checksums"]["model"]))
    pca = PCAStats(
        mean=pca_state["mean"],
        components=pca_state["components"],
        explained_variance=torch.tensor(manifest["pca_stats"]["explained_variance"]),
    )
    return LlaporCheckpoint(
        source_layer=source.layer_id,
        target_layer=target.layer_id,
        group=architecture["group"],
        pca=pca,
        model=model.eval(),
    )


def _load_apex(directory: Path, specs: Mapping[int, MoeLayerSpec]) -> ApexCheckpoint:
    manifest = _manifest(directory)
    architecture = manifest["architecture"]
    spec = _spec(specs, manifest["layer_id"], directory)
    found = (architecture["hidden_size"], architecture["num_experts"], architecture["top_k"])
    expected = (spec.hidden_size, spec.num_experts, spec.top_k)
    if found != expected:
        raise ValueError(f"prefetch checkpoint {directory} (hidden, experts, top_k)={found} "
                         f"does not match layer {spec.layer_id} {expected}")
    ranker = apex.Ranker(spec.hidden_size, spec.num_experts)
    ranker.load_state_dict(_verified_state(directory / "ranker.pt", manifest["tensor_checksums"]["ranker"]))
    # The training run stores no checksum for cdf.pt.
    cdf = apex.OrdinalCDF(spec.hidden_size, spec.num_experts - spec.top_k + 1)
    cdf.load_state_dict(torch.load(directory / "cdf.pt", map_location="cpu", weights_only=True))
    return ApexCheckpoint(layer_id=spec.layer_id, top_k=spec.top_k, ranker=ranker.eval(), cdf=cdf.eval())


def load_prefetch_checkpoints(
    model_dir: Path, *, predictor: str, specs: Sequence[MoeLayerSpec]
) -> dict[int, LlaporCheckpoint | ApexCheckpoint]:
    """Checkpoints keyed by the MoE layer whose experts they predict."""
    by_layer = {spec.layer_id: spec for spec in specs}
    if predictor == "llapor":
        pairs = [_load_llapor(path, by_layer) for path in sorted((model_dir / "llapor").glob("pair-*"))]
        loaded = {pair.target_layer: pair for pair in pairs}
    elif predictor == "apex":
        layers = [_load_apex(path, by_layer) for path in sorted((model_dir / "apex").glob("layer-*"))]
        loaded = {layer.layer_id: layer for layer in layers}
    else:
        raise ValueError(f"unknown prefetch predictor {predictor!r}; expected one of {PREFETCH_PREDICTORS}")
    if not loaded:
        raise ValueError(f"no {predictor} prefetch checkpoints under {model_dir}")
    return loaded
