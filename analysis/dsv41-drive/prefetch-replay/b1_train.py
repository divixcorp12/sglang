"""B1 training loop: one head per horizon, AdamW with a cosine schedule, early stop on dev recall at 1 read (S1.2).

Training groups stream from shards each epoch; groups are shuffled inside a bounded buffer, never across the corpus.
Dev recall@1 is the shared b1_score.evaluate pooled cell; the best epoch's weights are restored. The PCA is fit on the
training groups only. ``nested_subsets`` gives the seeded nested session subsets for the learning curve.

Inputs: a dataset dir from b1_dataset.py (train and dev splits) and a horizon.
Usage: python3 b1_train.py --dataset D --horizon 1 --out OUT [--variant E2|E5] [--device cuda]
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import math
import os
import sys
from collections.abc import Callable, Iterable, Iterator

import msgspec
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "scripts", "dsv41"))
sys.path.insert(0, os.path.join(REPO, "analysis", "dsv41-drive", "router-capture"))
sys.path.insert(0, HERE)
import b1_head  # noqa: E402
import b1_rows  # noqa: E402
import b1_score  # noqa: E402

PCA_MAX_ROWS = 100_000  # keeps the PCA fit under the 8 GiB analysis cap (see b1_head.fit_train_pca)
_SHUFFLE_BUFFER_BATCHES = 8


class TrainConfig(msgspec.Struct, frozen=True):
    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 1e-2
    batch_rows: int = 4096
    patience: int = 3
    seed: int = 0
    device: str = "cpu"


def nested_subsets(sessions: list[str], *, fractions=(0.25, 0.5, 1.0), seed: int) -> list[list[str]]:
    """Prefixes of one seeded permutation, so every subset contains the smaller ones; each returned sorted."""
    ordered = sorted(sessions)
    perm = np.random.default_rng(seed).permutation(len(ordered))
    shuffled = [ordered[i] for i in perm]
    sizes = [max(1, round(f * len(ordered))) for f in fractions]
    return [sorted(shuffled[:size]) for size in sizes]


def _pack(groups: list[b1_rows.Group], *, batch_rows: int) -> Iterator[list[b1_rows.Group]]:
    batch: list[b1_rows.Group] = []
    rows = 0
    for g in groups:
        batch.append(g)
        rows += len(g.tokens)
        if rows >= batch_rows:
            yield batch
            batch, rows = [], 0
    if batch:
        yield batch


def _batches(groups: Iterable[b1_rows.Group], *, batch_rows: int, rng: np.random.Generator
             ) -> Iterator[list[b1_rows.Group]]:
    """Group lists of about batch_rows rows, shuffled within a buffer of _SHUFFLE_BUFFER_BATCHES batches."""
    buf: list[b1_rows.Group] = []
    rows = 0
    for g in groups:
        buf.append(g)
        rows += len(g.tokens)
        if rows >= _SHUFFLE_BUFFER_BATCHES * batch_rows:
            yield from _pack([buf[i] for i in rng.permutation(len(buf))], batch_rows=batch_rows)
            buf, rows = [], 0
    if buf:
        yield from _pack([buf[i] for i in rng.permutation(len(buf))], batch_rows=batch_rows)


def _steps_per_epoch(groups: Iterable[b1_rows.Group], *, batch_rows: int) -> int:
    return sum(1 for _ in _batches(groups, batch_rows=batch_rows, rng=np.random.default_rng(0)))


def _cosine(total_steps: int) -> Callable[[int], float]:
    return lambda step: 0.5 * (1.0 + math.cos(math.pi * min(1.0, step / max(1, total_steps))))


def _train_epoch(model: b1_head.B1Head, opt: torch.optim.Optimizer, sched, *, groups: Iterable[b1_rows.Group],
                 x_of, tcfg: TrainConfig, rng: np.random.Generator) -> float:
    model.train()
    total, count = 0.0, 0
    for batch_groups in _batches(groups, batch_rows=tcfg.batch_rows, rng=rng):
        batch = b1_head.make_batch(batch_groups, x_of=x_of, pca=model.pca, cfg=model.cfg, vocab=model.vocab)
        loss = b1_head.head_loss(model(batch), batch, cfg=model.cfg)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        rows = len(batch.u)
        total += float(loss.item()) * rows
        count += rows
    return total / max(1, count)


def _dev_table(model: b1_head.B1Head, *, groups: Iterable[b1_rows.Group], x_of) -> b1_score.Table:
    scorer = b1_head.HeadScorer(model, x_of=x_of, name="head")
    return b1_score.evaluate(groups, scorer, reads=(1,))


def train_head(cfg: b1_head.HeadConfig, tcfg: TrainConfig, *, train_groups: Callable[[], Iterable[b1_rows.Group]],
               dev_groups: Callable[[], Iterable[b1_rows.Group]], x_of, pca=None
               ) -> tuple[b1_head.B1Head, dict]:
    torch.manual_seed(tcfg.seed)
    rng = np.random.default_rng(tcfg.seed)
    if pca is None:
        pca = b1_head.fit_train_pca(train_groups(), x_of=x_of, rank=cfg.pca_rank, max_rows=PCA_MAX_ROWS,
                                    seed=tcfg.seed)
    vocab = b1_head.Vocab.from_rows(itertools.chain(train_groups(), dev_groups()))
    model = b1_head.B1Head(cfg, pca=pca, vocab=vocab).to(tcfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay)
    total = tcfg.epochs * _steps_per_epoch(train_groups(), batch_rows=tcfg.batch_rows)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, _cosine(total))

    history: list[dict] = []
    best_recall, best_epoch, best_state, best_table = -1.0, -1, None, None
    for epoch in range(tcfg.epochs):
        loss = _train_epoch(model, opt, sched, groups=train_groups(), x_of=x_of, tcfg=tcfg, rng=rng)
        table = _dev_table(model, groups=dev_groups(), x_of=x_of)
        recall = table.cells["reads=1"]["pooled"].recall
        history.append({"epoch": epoch, "loss": loss, "dev_recall_at_1": recall})
        if recall > best_recall:
            best_recall, best_epoch, best_table = recall, epoch, table
            best_state = copy.deepcopy(model.state_dict())
        elif epoch - best_epoch >= tcfg.patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    return model, {"history": history, "best_epoch": best_epoch, "best_dev_recall_at_1": best_recall,
                   "best_dev_table": b1_score.table_to_json(best_table)}


def make_x_of(dataset_dir: str) -> Callable[[b1_rows.Group], np.ndarray]:
    """x_of reading each group's rows from its session's memmapped x shard."""
    maps: dict[tuple[str, str], np.ndarray] = {}

    def x_of(g: b1_rows.Group) -> np.ndarray:
        key = (g.split, g.session)
        if key not in maps:
            maps[key] = b1_rows.read_x(dataset_dir, split=g.split, session=g.session)
        return b1_rows.x_rows(maps[key], g)

    return x_of


def _parse(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dataset", required=True)
    p.add_argument("--horizon", type=int, required=True, choices=b1_rows.TRAINED_HORIZONS)
    p.add_argument("--out", required=True)
    p.add_argument("--variant", choices=("E2", "E5"), default="E2")
    p.add_argument("--pca-rank", type=int, default=256)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--hard-neg-weight", type=float, default=1.0)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-rows", type=int, default=4096)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse(argv)
    cfg = b1_head.HeadConfig(variant=args.variant, pca_rank=args.pca_rank, hidden=args.hidden,
                             hard_neg_weight=args.hard_neg_weight)
    tcfg = TrainConfig(epochs=args.epochs, lr=args.lr, batch_rows=args.batch_rows, patience=args.patience,
                       seed=args.seed, device=args.device)

    def groups(split: str) -> Callable[[], Iterable[b1_rows.Group]]:
        return lambda: b1_rows.read_groups(args.dataset, split=split, horizon=args.horizon)

    model, history = train_head(cfg, tcfg, train_groups=groups("train"), dev_groups=groups("dev"),
                                x_of=make_x_of(args.dataset))
    os.makedirs(args.out, exist_ok=True)
    head_path = os.path.join(args.out, f"head_h{args.horizon}.pt")
    history["head_sha256"] = b1_head.save_head(head_path, model.cpu())
    history["config"] = {"head": msgspec.to_builtins(cfg), "train": msgspec.to_builtins(tcfg)}
    with open(os.path.join(args.out, f"history_h{args.horizon}.json"), "w") as f:
        json.dump(history, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
