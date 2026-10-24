"""Explicit capture contract. No automatic interception of speculative/padded rows."""

import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from .protocol import sha256


class CaptureWriter:
    """Writer invoked by a native capture driver AFTER logical commit/alignment.

    The scheduler owns commit classification. This API deliberately cannot infer it
    from a tensor's row count. Reference and actual-prefix rollout records are distinct.
    """

    def __init__(self, root, identity):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=False)
        self.identity = identity
        self.records = []

    def add(
        self,
        *,
        record_id,
        group_id,
        split,
        kind,
        anchor_id,
        processed_ids,
        history,
        hidden,
        tokens,
        future_features=None,
        target_probabilities=None,
        label_law=None,
        producer=None,
        target_sampling=None,
        proposal_sampling=None,
    ):
        if kind not in {"reference", "rollout"} or split not in {
            "train",
            "select",
            "screen",
            "calibrate",
            "threshold",
            "test",
        }:
            raise ValueError("invalid record kind/split")
        if len(processed_ids) != len(history) or history.ndim != 2 or hidden.ndim != 2:
            raise ValueError("processed-token/feature alignment")
        if processed_ids != list(range(len(processed_ids))):
            raise ValueError("capture must contain full contiguous processed prefix")
        if not 0 < len(tokens) <= len(hidden):
            raise ValueError("invalid proposal/reference length")
        if any(r["id"] == record_id for r in self.records):
            raise ValueError("duplicate record id")
        if kind == "reference" and (
            future_features is None
            or len(future_features) != len(tokens)
            or label_law
            != {"temperature": 1.0, "top_p": 1.0, "top_k": 0, "greedy": False}
        ):
            raise ValueError("reference requires fixed-law aligned future features")
        if kind == "rollout" and (
            target_probabilities is None
            or len(target_probabilities) != len(tokens)
            or producer is None
            or target_sampling is None
            or proposal_sampling is None
        ):
            raise ValueError(
                "rollout needs own-prefix probabilities, producer and sampling identities"
            )
        tensors = {
            "history": history.detach().cpu().contiguous(),
            "hidden": hidden.detach().cpu().contiguous(),
            "tokens": torch.tensor(tokens, dtype=torch.long),
        }
        if future_features is not None:
            tensors["future_features"] = future_features.detach().cpu().contiguous()
        if target_probabilities is not None:
            p = target_probabilities.detach().float().cpu()
            if (
                not torch.isfinite(p).all()
                or (p < 0).any()
                or not torch.allclose(
                    p.sum(-1), torch.ones(len(p)), atol=2e-6, rtol=2e-6
                )
            ):
                raise ValueError("invalid target probabilities")
            tensors["target_probabilities"] = p.contiguous()
        if any(not torch.isfinite(t).all() for t in tensors.values()):
            raise ValueError("nonfinite capture tensor")
        # File names are generated; arbitrary record IDs cannot escape the capture directory.
        name = f"record-{len(self.records):06d}.safetensors"
        save_file(tensors, str(self.root / name))
        self.records.append(
            {
                "id": record_id,
                "group": group_id,
                "split": split,
                "kind": kind,
                "anchor": anchor_id,
                "processed_count": len(history),
                "file": name,
                "sha256": sha256(self.root / name),
                "label_law": label_law,
                "producer": producer,
                "target_sampling": target_sampling,
                "proposal_sampling": proposal_sampling,
            }
        )

    def close(self):
        # Proof gates cannot be self-certified by merely writing capture tensors.
        manifest = {
            "schema": 1,
            "identity": self.identity,
            "records": self.records,
            "feature_alignment_validated": False,
            "native_logits_parity_validated": False,
            "input_gradient_validated": False,
            "checkpoint_hashes_complete": False,
            "synthetic": False,
        }
        (self.root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def read_records(root, split, kind):
    root = Path(root).resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    selected = [
        r for r in manifest["records"] if r["split"] == split and r["kind"] == kind
    ]
    if not selected:
        raise ValueError(f"no {split}/{kind} records")
    for record in selected:
        path = (root / record["file"]).resolve()
        if path.parent != root or sha256(path) != record["sha256"]:
            raise ValueError("record path/hash mismatch")
        yield record, load_file(str(path))
