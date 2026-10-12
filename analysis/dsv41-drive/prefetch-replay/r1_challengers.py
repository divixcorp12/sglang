"""R1 challengers: T5 (trainable projection in place of the PCA) and G0 (full 384-score residual head).

T5 is an R1Model whose 128-wide projection is the head's pca_components, promoted from a buffer to a parameter under
the same state key, so its checkpoints load through r1_models.load_r1 unchanged. It centres with the train-only mean
and reads raw x from ModelInput.x, so step 0 equals the parent. G0 is z = a[t]*s_native + b[t] + B[t] GELU(A x + pos)
with a=1, b=0, B=0 at step 0 (z equals the native margins), fed to the BlockAggregator.

Inputs: ModelInput with x (and s_native for G0) from r1_cache.collate. Training lives in r1_train.
Usage: m = r1_challengers.with_projection(parent); new = r1_challengers.projection_params(m)
"""

from __future__ import annotations

import hashlib
import os
import sys

import msgspec
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "scripts", "dsv41"))
sys.path.insert(0, os.path.join(REPO, "analysis", "dsv41-drive", "router-capture"))
sys.path.insert(0, HERE)
from r1_aggregator import BlockAggregator  # noqa: E402
from r1_groups import MAX_WIDTH, ModelInput  # noqa: E402
from r1_models import R1Model, load_r1  # noqa: E402

_PROJECTION = "head.pca_components"
_HIDDEN = 5120  # DSV4.1 router input width
_NUM_EXPERTS = 384


class _ProjectedR1Model(R1Model):
    """R1Model whose first pca_rank encoder features are (x - mean) @ components.T, with components trainable."""

    def _encoder_input(self, inp: ModelInput) -> torch.Tensor:
        base = super()._encoder_input(inp)
        if inp.x is None:
            raise ValueError("the projection arm needs ModelInput.x")
        live = self._index(inp.batch.live).bool().unsqueeze(-1)
        z = (inp.x - self.head.pca_mean) @ self.head.pca_components.T
        z = torch.where(live, z, torch.zeros_like(z))
        return torch.cat([z, base[..., z.shape[-1]:]], dim=-1)


def _promote_components(model: R1Model) -> None:
    """Replace the pca_components buffer by a parameter of the same name, keeping the state key."""
    comp = model.head.pca_components.detach().clone()
    del model.head.pca_components
    model.head.register_parameter("pca_components", nn.Parameter(comp))


def with_projection(parent: R1Model) -> R1Model:
    """T5: a deep copy of the parent with a trainable projection initialised from its PCA components."""
    model = _ProjectedR1Model.from_parent(parent.cfg, parent=parent)
    _promote_components(model)
    return model


def projection_params(model: R1Model) -> set[str]:
    """Parameter names of the projection; the trainer gives them lr x0.1 (D5)."""
    return {_PROJECTION}


def load_with_projection(path) -> tuple[R1Model, str, str]:
    """Load a T5 checkpoint written by r1_models.save_r1 as a projection model (eval mode, sha, parent sha)."""
    plain, sha, parent_sha = load_r1(path)
    model = _ProjectedR1Model(plain.cfg, head=plain.head)
    model.load_state_dict(plain.state_dict(), strict=False)
    _promote_components(model)
    return model.eval(), sha, parent_sha


class G0Config(msgspec.Struct, frozen=True):
    rank: int = 64
    pos_dim: int = 64
    emb_dim: int = 16
    hidden: int = _HIDDEN
    num_experts: int = _NUM_EXPERTS


class G0Model(nn.Module):
    def __init__(self, cfg: G0Config, *, n_layers: int, n_strata: int):
        super().__init__()
        self.cfg = cfg
        self.n_layers = n_layers
        self.n_strata = n_strata
        self.proj_a = nn.Linear(cfg.hidden, cfg.rank)
        self.pos_emb = nn.Embedding(MAX_WIDTH, cfg.pos_dim)
        self.pos_proj = nn.Linear(cfg.pos_dim, cfg.rank, bias=False)
        self.scale_a = nn.Parameter(torch.ones(n_layers))
        self.shift_b = nn.Parameter(torch.zeros(n_layers))
        self.out_b = nn.Parameter(torch.zeros(n_layers, cfg.num_experts, cfg.rank))
        self.aggregator = BlockAggregator(n_layers=n_layers, n_strata=n_strata, emb_dim=cfg.emb_dim)

    @property
    def device(self) -> torch.device:
        return self.proj_a.weight.device

    def token_logits(self, inp: ModelInput) -> torch.Tensor:
        """z f32 [G,6,E]; padding rows hold values nobody may read (mask by live)."""
        if inp.x is None or inp.s_native is None:
            raise ValueError("G0 needs ModelInput.x and ModelInput.s_native")
        b = inp.batch
        layer = torch.as_tensor(b.target_layer, device=self.device)
        pos = torch.as_tensor(b.position, device=self.device).clamp(0, MAX_WIDTH - 1)
        hid = F.gelu(self.proj_a(inp.x) + self.pos_proj(self.pos_emb(pos)))  # [G,6,rank]
        resid = torch.bmm(hid, self.out_b[layer].transpose(1, 2))  # [G,6,E]
        a = self.scale_a[layer].view(-1, 1, 1)
        shift = self.shift_b[layer].view(-1, 1, 1)
        return a * inp.s_native + shift + resid

    def forward(self, inp: ModelInput) -> torch.Tensor:
        """Block logits f32 [G,E]."""
        return self.aggregator(self.token_logits(inp), batch=inp.batch)


def _file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def save_g0(path, model: G0Model, *, parent_sha: str, arm: str = "G0") -> str:
    """Write config, state and the parent's sha256 to ``path``; returns the file's sha256."""
    path = os.fspath(path)
    payload = {
        "cfg": msgspec.to_builtins(model.cfg),
        "n_layers": model.n_layers,
        "n_strata": model.n_strata,
        "parent_sha": parent_sha,
        "arm": arm,
        "state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
    }
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)
    return _file_sha256(path)


def load_g0(path) -> tuple[G0Model, str, str]:
    """(model in eval mode, file sha256, parent sha256)."""
    path = os.fspath(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    cfg = msgspec.convert(payload["cfg"], G0Config)
    model = G0Model(cfg, n_layers=payload["n_layers"], n_strata=payload["n_strata"])
    model.load_state_dict(payload["state"])
    return model.eval(), _file_sha256(path), payload["parent_sha"]
