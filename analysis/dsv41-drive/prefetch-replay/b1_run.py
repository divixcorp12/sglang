"""B1 run driver: dataset, fit, baselines, train, curve, cuts, freeze, locked, cost and report, resumable (D8).

Each stage writes its outputs and ``stages/<name>/DONE.json`` under --out; a rerun skips stages that have one and
resumes at the first without. ``locked`` reads the locked_test split only through b1_select.require_frozen, and the
results record the frozen selection file's sha256. ``--smoke`` relabels 5 real train-split sessions as 2 train / 1
dev / 1 calib / 1 stand-in locked_test (scratch blocks whose rids.json names only those sessions, so no real
locked-test session is extracted) with a reduced grid and epochs.

Frozen selection keys: horizon h and variant E<n> are stored as h * 10 + n (L+1 E2 is 12, L+4 E5 is 45). Cuts use the
keys 'reads=r' (head), 'gate:reads=r', 'gate_floors:reads=r' and 'at_gate_precision' (head cut whose calib precision
matches the gate with floors at 1 read).

Inputs: MODEL_PATH (gates), the B0 corpus dir with manifest.json and block dirs, the integrity report, or --dataset.
The train and cost stages need the GPU: run the whole driver under cc-gpu.lock on cores 32-63.
Usage: flock /data/models/slang/nvfp4-work/cc-gpu.lock taskset -c 32-63 python3 b1_run.py MODEL_PATH --corpus C
       --integrity R --out /mnt/nvme1/nvfp4-work/b1-run-<date> [--dataset D] [--stages ...] [--device cuda] [--smoke]
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import itertools
import json
import math
import os
import sys
from collections.abc import Callable

import msgspec
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "scripts", "dsv41"))
sys.path.insert(0, os.path.join(REPO, "analysis", "dsv41-drive", "router-capture"))
sys.path.insert(0, HERE)
import b1_baselines  # noqa: E402
import b1_cache  # noqa: E402
import b1_cost  # noqa: E402
import b1_dataset  # noqa: E402
import b1_head  # noqa: E402
import b1_report  # noqa: E402
import b1_rows  # noqa: E402
import b1_score  # noqa: E402
import b1_select  # noqa: E402
import b1_train  # noqa: E402

from sglang.srt.layers.moe.expert_prediction.training.pca import PCAStats  # noqa: E402

OUT_ROOT = "/mnt/nvme1"
SELECTION_NAME = "selection.json"
LEARNING_LABELS = ("15", "30", "60")  # the nominal sizes; smoke subsets are smaller but keep the labels
SMOKE_SESSIONS = {"train": 2, "dev": 1, "calib": 1, "locked_test": 1}
VARIANTS = ("E2", "E5")
HEAD_CUTS = tuple(round(0.01 * i, 2) for i in range(1, 100))
# Margins are biased-score minus the 6th score; the range is a guess wide enough for the packaged floors.
GATE_CUTS = tuple(round(-4.0 + 0.05 * i, 2) for i in range(161))
# Arbitrary: a head whose L+1 precision gain is at least half the bar counts as near it (D5).
NEAR_BAR_FRACTION = 0.5


class RunConfig(msgspec.Struct, frozen=True):
    horizons: tuple[int, ...]
    grids: dict[str, list[dict]]
    train: b1_train.TrainConfig
    num_experts: int = b1_rows.NUM_EXPERTS
    bootstrap_n: int = 10_000
    curve_horizon: int = 1
    curve_fractions: tuple[float, ...] = (0.25, 0.5, 1.0)
    cost_runs: int = 1000
    cost_warmup: int = 50
    seed: int = 0
    floors: dict[int, float] | None = None  # None loads the packaged margin floors


class _Ctx(msgspec.Struct, frozen=True):
    model_path: str
    corpus: str
    integrity: str
    out: str
    dataset_dir: str
    cfg: RunConfig
    smoke: bool
    device: str


def _full_config(*, device: str, compile: bool = False) -> RunConfig:
    return RunConfig(horizons=b1_rows.TRAINED_HORIZONS, grids={v: b1_select.grid(variant=v) for v in VARIANTS},
                     train=b1_train.TrainConfig(device=device, compile=compile))


def _smoke_config(*, device: str, compile: bool = False) -> RunConfig:
    return RunConfig(horizons=(1, 2), grids={v: b1_select.grid(variant=v)[:1] for v in VARIANTS},
                     train=b1_train.TrainConfig(epochs=2, patience=1, device=device, compile=compile),
                     bootstrap_n=200, cost_runs=50, cost_warmup=5)


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _load_json(path: str):
    with open(path) as f:
        return json.load(f)


def _write_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)


def _save_tables(path: str, tables: dict[str, b1_score.Table]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as f:
        f.write(msgspec.json.format(msgspec.json.encode(tables), indent=1))


def _load_tables(path: str) -> dict[str, b1_score.Table]:
    with open(path, "rb") as f:
        return msgspec.json.decode(f.read(), type=dict[str, b1_score.Table])


def _stage_dir(ctx: _Ctx, name: str) -> str:
    return os.path.join(ctx.out, "stages", name)


def _done(ctx: _Ctx, name: str) -> dict:
    return _load_json(os.path.join(_stage_dir(ctx, name), "DONE.json"))


def _cut_budget(cut: float) -> str:
    return f"cut={float(cut)!r}"  # the key format of b1_score's cut cells


def _sel_key(horizon: int, variant: str) -> int:
    return horizon * 10 + int(variant[1:])


def _x_of(ctx: _Ctx, *, frozen_sha256: str | None = None) -> Callable[[b1_rows.Group], object]:
    """Reads each group's rows from its session's memmapped x shard; locked_test needs the frozen sha256."""
    maps: dict[tuple[str, str], object] = {}

    def x_of(g: b1_rows.Group):
        key = (g.split, g.session)
        if key not in maps:
            maps[key] = b1_rows.read_x(ctx.dataset_dir, split=g.split, session=g.session,
                                       frozen_sha256=frozen_sha256)
        return b1_rows.x_rows(maps[key], g)

    return x_of


def _groups(ctx: _Ctx, *, split: str, horizon: int, sessions: list[str] | None = None,
            frozen_sha256: str | None = None):
    return b1_rows.read_groups(ctx.dataset_dir, split=split, horizon=horizon, sessions=sessions,
                               frozen_sha256=frozen_sha256)


def _evaluate(ctx: _Ctx, scorers: list, *, split: str, horizon: int, cuts=(),
              frozen_sha256: str | None = None) -> dict[str, b1_score.Table]:
    groups = _groups(ctx, split=split, horizon=horizon, frozen_sha256=frozen_sha256)
    return b1_score.evaluate_many(groups, scorers, reads=b1_score.READS, cuts=cuts)


def _dataset_info(ctx: _Ctx) -> dict:
    return _load_json(os.path.join(ctx.dataset_dir, b1_dataset.DATASET_NAME))


def _train_sessions(ctx: _Ctx) -> list[str]:
    info = _dataset_info(ctx)
    built = set(info["built_sessions"])
    return [s for s in info["sessions_by_split"]["train"] if s in built]


def _floors(ctx: _Ctx) -> dict[int, float]:
    return b1_baselines.load_floors() if ctx.cfg.floors is None else dict(ctx.cfg.floors)


def _load_pop(ctx: _Ctx, *, horizon: int) -> b1_baselines.PopScorer:
    doc = _load_json(os.path.join(_stage_dir(ctx, "fit"), "pop.json"))
    return b1_baselines.PopScorer.from_json(json.dumps(doc[str(horizon)]))


def _load_pca(ctx: _Ctx, *, rank: int) -> PCAStats:
    path = os.path.join(_stage_dir(ctx, "fit"), f"pca_r{rank}.pt")
    t = torch.load(path, map_location="cpu", weights_only=True)
    return PCAStats(mean=t["mean"], components=t["components"], explained_variance=t["explained_variance"])


# ---- smoke inputs -------------------------------------------------------------------------------------------------


def _pick_smoke_sessions(manifest: dict) -> dict[str, list[str]]:
    """Real train-split sessions only, in sorted order, dealt out to the smoke splits."""
    pool = [s for s in sorted(manifest["split"]["train"]) if s in manifest["session_block"]]
    need = sum(SMOKE_SESSIONS.values())
    if len(pool) < need:
        raise ValueError(f"smoke needs {need} train-split sessions in the corpus, found {len(pool)}")
    picked, start = {}, 0
    for split, n in SMOKE_SESSIONS.items():
        picked[split] = pool[start:start + n]
        start += n
    return picked


def _block_dir(*, corpus: str, name: str) -> str:
    # b0_capture writes OUT_DIR/blocks/<block-id>/; manifest session_block holds the bare id.
    return os.path.join(corpus, "blocks", name)


def _scratch_block(src: str, dst: str, *, sessions: set[str]) -> bool:
    """Symlink the block's capture files and write a rids.json naming only ``sessions``; False when none match."""
    rids = {rid: v for rid, v in _load_json(os.path.join(src, "rids.json")).items() if v["session_id"] in sessions}
    if not rids:
        return False
    os.makedirs(dst, exist_ok=True)
    for name in sorted(os.listdir(src)):
        link = os.path.join(dst, name)
        if name != "rids.json" and not os.path.lexists(link):
            os.symlink(os.path.join(src, name), link)
    _write_json(os.path.join(dst, "rids.json"), rids)
    return True


def _smoke_inputs(ctx: _Ctx) -> tuple[str, str, list[str]]:
    """(manifest path, integrity path, block dirs) of the relabelled smoke corpus under out/smoke."""
    manifest = _load_json(os.path.join(ctx.corpus, "manifest.json"))
    integrity = _load_json(ctx.integrity)
    picked = _pick_smoke_sessions(manifest)
    chosen = {s for sessions in picked.values() for s in sessions}
    root = os.path.join(ctx.out, "smoke")
    blocks, rewritten = [], []
    for name in sorted({manifest["session_block"][s] for s in chosen}):
        src, dst = _block_dir(corpus=ctx.corpus, name=name), _block_dir(corpus=root, name=name)
        if not _scratch_block(src, dst, sessions=chosen):
            continue
        blocks.append(dst)
        rewritten += [{**r, "out_dir": dst} for r in integrity["per_session"]
                      if os.path.realpath(r["out_dir"]) == os.path.realpath(src)]
    small = {"split": picked, "warm_start": sorted(chosen & set(manifest["warm_start"])),
             "flagged": [f for f in manifest["flagged"] if f["session_id"] in chosen]}
    _write_json(os.path.join(root, "manifest.json"), small)
    _write_json(os.path.join(root, "integrity.json"), {"per_session": rewritten})
    return os.path.join(root, "manifest.json"), os.path.join(root, "integrity.json"), blocks


# ---- stages -------------------------------------------------------------------------------------------------------


def _stage_dataset(ctx: _Ctx) -> dict:
    """Build the shards (or adopt --dataset) and report the manifest's counts."""
    if not os.path.isfile(os.path.join(ctx.dataset_dir, b1_dataset.DATASET_NAME)):
        if ctx.smoke:
            manifest, integrity, blocks = _smoke_inputs(ctx)
        else:
            manifest, integrity = os.path.join(ctx.corpus, "manifest.json"), ctx.integrity
            names = sorted(set(_load_json(manifest)["session_block"].values()))
            blocks = [_block_dir(corpus=ctx.corpus, name=n) for n in names]
        b1_dataset.build_dataset(model_path=ctx.model_path, block_dirs=blocks, manifest_path=manifest,
                                 integrity_path=integrity, out_dir=ctx.dataset_dir)
    info = _dataset_info(ctx)
    return {"dataset_dir": ctx.dataset_dir, "counts": info["counts"], "warmup_forwards": info["warmup_forwards"],
            "dropped_cells": info["dropped_cells"]}


def _stage_fit(ctx: _Ctx) -> dict:
    """Popularity per horizon and one PCA per rank, from train-split rows only (R2).

    x is stored once per session, so the horizons share one input distribution and one PCA per rank.
    """
    out = _stage_dir(ctx, "fit")
    ranks = sorted({g["pca_rank"] for grid in ctx.cfg.grids.values() for g in grid})
    pops = {}
    for h in ctx.cfg.horizons:
        pop = b1_baselines.PopScorer.fit(_groups(ctx, split="train", horizon=h), num_experts=ctx.cfg.num_experts)
        pops[str(h)] = json.loads(pop.to_json())
    for rank in ranks:
        pca = b1_head.fit_train_pca_from_shards(ctx.dataset_dir, sessions=_train_sessions(ctx), rank=rank,
                                                max_rows=b1_train.PCA_MAX_ROWS, seed=ctx.cfg.train.seed)
        torch.save({"mean": pca.mean.cpu(), "components": pca.components.cpu(),
                    "explained_variance": pca.explained_variance.cpu()}, os.path.join(out, f"pca_r{rank}.pt"))
    _write_json(os.path.join(out, "pop.json"), pops)
    return {"horizons": list(ctx.cfg.horizons), "ranks": ranks}


def _stage_baselines(ctx: _Ctx) -> dict:
    """gate (+floors), prev and pop on dev and calib; the gate family also gets a margin cut sweep."""
    out = _stage_dir(ctx, "baselines")
    floors = _floors(ctx)
    written = []
    for split in ("dev", "calib"):
        for h in ctx.cfg.horizons:
            pop = _load_pop(ctx, horizon=h)
            scorers = b1_baselines.baseline_scorers(horizon=h, floors=floors, pop=pop,
                                                    num_experts=ctx.cfg.num_experts)
            gates = [s for s in scorers if s.name.startswith("gate")]
            counts = [s for s in scorers if not s.name.startswith("gate")]
            tables = _evaluate(ctx, gates, split=split, horizon=h, cuts=GATE_CUTS)
            tables.update(_evaluate(ctx, counts, split=split, horizon=h))
            path = os.path.join(out, f"{split}_h{h}.json")
            _save_tables(path, tables)
            written.append(path)
    return {"tables": written}


def _head_key(kwargs: dict) -> str:
    return f"{kwargs['variant']}_r{kwargs['pca_rank']}_h{kwargs['hidden']}_w{kwargs['hard_neg_weight']}"


class _Caches(msgspec.Struct):
    train: b1_cache.SplitCache
    dev: b1_cache.SplitCache
    pca: PCAStats
    vocab: b1_head.Vocab


def _caches(ctx: _Ctx, *, horizon: int, pca: PCAStats, vocab: b1_head.Vocab,
            sessions: list[str] | None = None) -> _Caches:
    """Device-resident train (optionally a session subset) and dev caches built with ``pca``."""
    device = ctx.cfg.train.device
    pca = PCAStats(mean=pca.mean.to(device), components=pca.components.to(device),
                   explained_variance=pca.explained_variance.to(device))

    def build(split: str, subset: list[str] | None) -> b1_cache.SplitCache:
        return b1_cache.build(_groups(ctx, split=split, horizon=horizon, sessions=subset), x_of=_x_of(ctx), pca=pca,
                              vocab=vocab, num_experts=ctx.cfg.num_experts)

    return _Caches(train=build("train", sessions), dev=build("dev", None), pca=pca, vocab=vocab)


def _vocab(ctx: _Ctx, *, horizon: int) -> b1_head.Vocab:
    return b1_head.Vocab.from_rows(itertools.chain(_groups(ctx, split="train", horizon=horizon),
                                                   _groups(ctx, split="dev", horizon=horizon)))


def _train_one(ctx: _Ctx, *, horizon: int, kwargs: dict, out: str, caches: _Caches) -> tuple[dict, b1_score.Table]:
    cfg = b1_head.HeadConfig(**{**kwargs, "num_experts": ctx.cfg.num_experts})
    model, history = b1_train.train_head_cached(cfg, ctx.cfg.train, train_cache=caches.train, dev_cache=caches.dev,
                                                pca=caches.pca, vocab=caches.vocab)
    key = _head_key(kwargs)
    table = b1_train.dev_table(model, dev_cache=caches.dev, dev_groups=_groups(ctx, split="dev", horizon=horizon),
                               name=key)
    path = os.path.join(out, "heads", f"h{horizon}_{key}.pt")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sha = b1_head.save_head(path, model.cpu())
    entry = {"variant": cfg.variant, "config": msgspec.to_builtins(cfg), "path": path, "sha256": sha,
             "best_epoch": history["best_epoch"], "history": history["history"],
             "dev_precision_r1": table.cells["reads=1"]["pooled"].precision}
    return entry, table


def _release_device_memory() -> None:
    # Callers drop their cache reference first; a rank-256 E5 train cache is several GB.
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _stage_train(ctx: _Ctx) -> dict:
    """Grid and train each head per horizon on train, early-stopped on dev; pick by dev precision at 1 read.

    Caches are built once per (horizon, PCA rank) and shared by that rank's configs. Needs the GPU when --device
    cuda: run under cc-gpu.lock.
    """
    out = _stage_dir(ctx, "train")
    summary = {}
    for h in ctx.cfg.horizons:
        vocab = _vocab(ctx, horizon=h)
        configs = [kwargs for variant in VARIANTS for kwargs in ctx.cfg.grids[variant]]
        done: dict[str, tuple[dict, b1_score.Table]] = {}
        for rank in sorted({kw["pca_rank"] for kw in configs}):
            caches = _caches(ctx, horizon=h, pca=_load_pca(ctx, rank=rank), vocab=vocab)
            for kwargs in (kw for kw in configs if kw["pca_rank"] == rank):
                done[_head_key(kwargs)] = _train_one(ctx, horizon=h, kwargs=kwargs, out=out, caches=caches)
            del caches
            _release_device_memory()
        entries = {_head_key(kw): done[_head_key(kw)][0] for kw in configs}
        tables = {_head_key(kw): done[_head_key(kw)][1] for kw in configs}
        best_by_variant = {v: b1_select.pick_best({k: t for k, t in tables.items() if entries[k]["variant"] == v})
                           for v in VARIANTS}
        _save_tables(os.path.join(out, f"dev_h{h}.json"), tables)
        summary[str(h)] = {"configs": entries, "best_by_variant": best_by_variant,
                           "best": b1_select.pick_best(tables)}
    _write_json(os.path.join(out, "train.json"), summary)
    return summary


def _stage_curve(ctx: _Ctx) -> dict:
    """Dev precision at 1 read for the best E2 and E5 config trained on nested 25/50/100% of the train sessions."""
    h = ctx.cfg.curve_horizon
    train = _load_json(os.path.join(_stage_dir(ctx, "train"), "train.json"))[str(h)]
    sessions = _train_sessions(ctx)
    subsets = b1_train.nested_subsets(sessions, fractions=ctx.cfg.curve_fractions, seed=ctx.cfg.seed)
    vocab = _vocab(ctx, horizon=h)
    curve, sizes, improving = {}, [len(s) for s in subsets], {}
    for variant in VARIANTS:
        entry = train["configs"][train["best_by_variant"][variant]]
        cfg = msgspec.convert(entry["config"], b1_head.HeadConfig)
        values = []
        for subset in subsets:
            if len(subset) == len(sessions):
                values.append(entry["dev_precision_r1"])
                continue
            # The subset's own PCA: no statistic of the sessions it leaves out reaches it.
            pca = b1_head.fit_train_pca_from_shards(ctx.dataset_dir, sessions=subset, rank=cfg.pca_rank,
                                                    max_rows=b1_train.PCA_MAX_ROWS, seed=ctx.cfg.train.seed)
            caches = _caches(ctx, horizon=h, pca=pca, vocab=vocab, sessions=subset)
            model, _ = b1_train.train_head_cached(cfg, ctx.cfg.train, train_cache=caches.train,
                                                  dev_cache=caches.dev, pca=caches.pca, vocab=vocab)
            table = b1_train.dev_table(model, dev_cache=caches.dev, dev_groups=_groups(ctx, split="dev", horizon=h),
                                       name=variant)
            values.append(table.cells["reads=1"]["pooled"].precision)
            del caches, model
            _release_device_memory()
        curve[variant] = dict(zip(LEARNING_LABELS, values, strict=True))
        improving[variant] = b1_select.still_improving(*values)
    result = {"curve": curve, "sizes": sizes, "still_improving": improving}
    _write_json(os.path.join(_stage_dir(ctx, "curve"), "curve.json"), result)
    return result


def _stage_cuts(ctx: _Ctx) -> dict:
    """Calibrate cuts on calib only: gate and gate_floors margin cuts, head cuts, and the head cut at gate precision."""
    out = _stage_dir(ctx, "cuts")
    train = _load_json(os.path.join(_stage_dir(ctx, "train"), "train.json"))
    summary = {}
    for h in ctx.cfg.horizons:
        base = _load_tables(os.path.join(_stage_dir(ctx, "baselines"), f"calib_h{h}.json"))
        cuts = {name: b1_select.calibrate_cuts(base[name]) for name in ("gate", "gate_floors") if name in base}
        reference = base["gate_floors" if "gate_floors" in base else "gate"]
        gate_precision = reference.cells["reads=1"]["pooled"].precision
        per_h = train[str(h)]
        calib_tables: dict[str, b1_score.Table] = {}
        for variant in VARIANTS:
            entry = per_h["configs"][per_h["best_by_variant"][variant]]
            model, _ = b1_head.load_head(entry["path"])
            model = model.to(ctx.cfg.train.device)
            scorer = b1_head.HeadScorer(model, x_of=_x_of(ctx), name=variant)
            table = _evaluate(ctx, [scorer], split="calib", horizon=h, cuts=HEAD_CUTS)[variant]
            calib_tables[variant] = table
            cuts[variant] = b1_select.calibrate_cuts(table)
            at_p = b1_score.cut_for_precision(table, gate_precision)
            cuts[variant]["at_gate_precision"] = at_p
        _save_tables(os.path.join(out, f"calib_h{h}.json"), calib_tables)
        summary[str(h)] = {"cuts": cuts, "gate_precision_calib_r1": gate_precision,
                           "calibration": {v: [list(r) for r in t.calibration] for v, t in calib_tables.items()}}
    _write_json(os.path.join(out, "cuts.json"), summary)
    return summary


def _selection_cuts(per_h: dict, *, variant: str) -> dict[str, float]:
    cuts = {k: v for k, v in per_h["cuts"][variant].items() if v is not None}
    for gate in ("gate", "gate_floors"):
        if gate in per_h["cuts"]:
            cuts.update({f"{gate}:{k}": v for k, v in per_h["cuts"][gate].items()})
    return cuts


def _stage_freeze(ctx: _Ctx) -> dict:
    """Write the frozen selection (chosen configs, cuts, model and popularity checksums) once."""
    path = os.path.join(ctx.out, SELECTION_NAME)
    train = _load_json(os.path.join(_stage_dir(ctx, "train"), "train.json"))
    cuts = _load_json(os.path.join(_stage_dir(ctx, "cuts"), "cuts.json"))
    pop_path = os.path.join(_stage_dir(ctx, "fit"), "pop.json")
    horizons, chosen = {}, {}
    for h in ctx.cfg.horizons:
        per_h = train[str(h)]
        chosen[str(h)] = per_h["configs"][per_h["best"]]["variant"]
        for variant in VARIANTS:
            entry = per_h["configs"][per_h["best_by_variant"][variant]]
            horizons[_sel_key(h, variant)] = b1_select.HorizonSelection(
                variant=variant, config=entry["config"], cuts=_selection_cuts(cuts[str(h)], variant=variant),
                model_path=entry["path"], model_sha256=entry["sha256"])
    selection = b1_select.Selection(horizons=horizons, pop_path=pop_path, pop_sha256=_sha256_file(pop_path),
                                    created=datetime.date.today().isoformat())
    if not os.path.exists(path):
        b1_select.freeze_selection(path, selection)
    frozen, sha = b1_select.load_frozen(path)
    if frozen.horizons != selection.horizons:
        raise ValueError(f"{path} was frozen from a different training or cuts run; remove it to refreeze")
    return {"selection": path, "sha256": sha, "chosen": chosen}


def _locked_scorers(ctx: _Ctx, sel: b1_select.Selection, *, horizon: int, x_of) -> list[tuple[list, tuple]]:
    """(scorers, cuts) per evaluation pass; checksums of the pop file and heads are verified against the freeze."""
    if _sha256_file(sel.pop_path) != sel.pop_sha256:
        raise ValueError(f"{sel.pop_path} does not match the frozen popularity checksum")
    pop = b1_baselines.PopScorer.from_json(json.dumps(_load_json(sel.pop_path)[str(horizon)]))
    scorers = b1_baselines.baseline_scorers(horizon=horizon, floors=_floors(ctx), pop=pop,
                                            num_experts=ctx.cfg.num_experts)
    ref = sel.horizons[_sel_key(horizon, VARIANTS[0])].cuts
    passes = []
    for s in scorers:
        if s.name.startswith("gate"):
            passes.append(([s], tuple(ref[f"{s.name}:reads={r}"] for r in b1_score.READS)))
    passes.append(([s for s in scorers if not s.name.startswith("gate")], ()))
    for variant in VARIANTS:
        entry = sel.horizons[_sel_key(horizon, variant)]
        model, sha = b1_head.load_head(entry.model_path)
        if sha != entry.model_sha256:
            raise ValueError(f"{entry.model_path} does not match the frozen checksum")
        cuts = [entry.cuts[f"reads={r}"] for r in b1_score.READS]
        cuts += [entry.cuts[k] for k in ("at_gate_precision",) if k in entry.cuts]
        scorer = b1_head.HeadScorer(model.to(ctx.cfg.train.device), x_of=x_of, name=variant)
        passes.append(([scorer], tuple(cuts)))
    return passes


def _stage_locked(ctx: _Ctx) -> dict:
    """Evaluate every scorer once on locked_test; refuses unless the frozen selection file exists (R1)."""
    frozen_path = os.path.join(ctx.out, SELECTION_NAME)
    sha = b1_select.require_frozen(b1_rows.LOCKED_SPLIT, frozen_path)
    sel, _ = b1_select.load_frozen(frozen_path)
    chosen = _done(ctx, "freeze")["chosen"]
    x_of = _x_of(ctx, frozen_sha256=sha)
    bootstrap: dict = {}
    for h in ctx.cfg.horizons:
        tables: dict[str, b1_score.Table] = {}
        for scorers, cuts in _locked_scorers(ctx, sel, horizon=h, x_of=x_of):
            tables.update(_evaluate(ctx, scorers, split=b1_rows.LOCKED_SPLIT, horizon=h, cuts=cuts,
                                    frozen_sha256=sha))
        _save_tables(os.path.join(_stage_dir(ctx, "locked"), f"h{h}.json"), tables)
        if h == 1:
            diffs = b1_select.paired_diffs(tables[chosen["1"]], tables["gate_floors"])
            median, lo, hi = b1_select.session_bootstrap(diffs, n=ctx.cfg.bootstrap_n, seed=ctx.cfg.seed)
            bootstrap = {"median": median, "lo": lo, "hi": hi, "sessions": len(diffs)}
    return {"frozen_sha256": sha, "bootstrap": bootstrap, "horizons": list(ctx.cfg.horizons)}


def _stage_cost(ctx: _Ctx) -> dict:
    """Scorer cost of the chosen L+1 head on the RTX 5090: bf16 bytes and CUDA-graph median ms for 6 tokens.

    Runs under cc-gpu.lock (cores 32-63). Without CUDA it raises, so no DONE marker hides an unmeasured cost.
    """
    if not (ctx.device.startswith("cuda") and torch.cuda.is_available()):
        raise RuntimeError(f"the cost stage needs CUDA (got device {ctx.device!r}): "
                           "run it with --device cuda under cc-gpu.lock on cores 32-63")
    sel, _ = b1_select.load_frozen(os.path.join(ctx.out, SELECTION_NAME))
    chosen = _done(ctx, "freeze")["chosen"]
    entry = sel.horizons[_sel_key(1, chosen["1"])]
    model, sha = b1_head.load_head(entry.model_path)
    result = b1_cost.measure_cost(model, runs=ctx.cfg.cost_runs, warmup=ctx.cfg.cost_warmup, device=ctx.device)
    return {**result, "head_sha256": sha, "gpu": torch.cuda.get_device_name()}


def _verdict(ctx: _Ctx, *, tables: dict[str, b1_score.Table], bootstrap: dict, cost: dict, cuts: dict) -> dict:
    head, gate = tables[cuts["chosen"]], tables["gate_floors"]
    r1 = "reads=1"
    at_p = cuts.get("at_gate_precision")
    head_recall = None if at_p is None else head.cells[_cut_budget(at_p)]["pooled"].recall
    return b1_select.verdict(
        gate_floor_precision_r1=gate.cells[r1]["pooled"].precision,
        head_precision_r1=head.cells[r1]["pooled"].precision,
        gate_floor_recall_at_p=gate.cells[r1]["pooled"].recall, head_recall_at_p=head_recall,
        ci_lo=bootstrap["lo"], cost_ms_per_layer=cost.get("median_ms", math.inf))


def _second_capture(*, verdict: dict, curve: dict, chosen: str, head_precision: float, gate_precision: float) -> dict:
    """D5: a second capture only if the head is near the bar and its dev curve is still rising."""
    near = (not verdict["advance"]
            and head_precision - gate_precision >= NEAR_BAR_FRACTION * b1_select.PRECISION_GAIN)
    improving = curve["still_improving"][chosen]
    return {"near_bar": near, "still_improving": improving, "warranted": near and improving}


def _stage_report(ctx: _Ctx) -> dict:
    """Assemble results.json in b1_report's schema and render REPORT.md."""
    locked, curve = _done(ctx, "locked"), _done(ctx, "curve")
    cost, freeze = _done(ctx, "cost"), _done(ctx, "freeze")
    cuts = _load_json(os.path.join(_stage_dir(ctx, "cuts"), "cuts.json"))
    results: dict = {
        "horizons": {}, "learning_curve": curve["curve"], "bootstrap": locked["bootstrap"], "cost": cost,
        "verdict": {}, "frozen_sha256": locked["frozen_sha256"], "smoke": ctx.smoke,
        "dataset": _done(ctx, "dataset")["counts"], "still_improving": curve["still_improving"],
        "calibration": {h: per_h["calibration"] for h, per_h in cuts.items()},
        "paths": {"dataset": ctx.dataset_dir, "selection": os.path.join(ctx.out, SELECTION_NAME),
                  "out": ctx.out, "model_path": ctx.model_path},
    }
    for h in ctx.cfg.horizons:
        tables = _load_tables(os.path.join(_stage_dir(ctx, "locked"), f"h{h}.json"))
        results["horizons"][str(h)] = {"tables": {n: b1_score.table_to_json(t) for n, t in tables.items()}}
        if h == 1:
            chosen = freeze["chosen"]["1"]
            sel_cuts = {"chosen": chosen, **cuts["1"]["cuts"][chosen]}
            results["verdict"] = _verdict(ctx, tables=tables, bootstrap=locked["bootstrap"], cost=cost, cuts=sel_cuts)
            results["second_capture"] = _second_capture(
                verdict=results["verdict"], curve=curve, chosen=chosen,
                head_precision=tables[chosen].cells["reads=1"]["pooled"].precision,
                gate_precision=tables["gate_floors"].cells["reads=1"]["pooled"].precision)
    out = _stage_dir(ctx, "report")
    _write_json(os.path.join(out, "results.json"), results)
    with open(os.path.join(out, "REPORT.md"), "w") as f:
        f.write(b1_report.render(results))
    return {"results": os.path.join(out, "results.json"), "report": os.path.join(out, "REPORT.md")}


_STAGE_FNS: dict[str, Callable[[_Ctx], dict]] = {
    "dataset": _stage_dataset, "fit": _stage_fit, "baselines": _stage_baselines, "train": _stage_train,
    "curve": _stage_curve, "cuts": _stage_cuts, "freeze": _stage_freeze, "locked": _stage_locked,
    "cost": _stage_cost, "report": _stage_report,
}
STAGES = tuple(_STAGE_FNS)


def _run_stages(*, ctx, fns: dict[str, Callable], selected: list[str]) -> dict[str, bool]:
    """Run the selected stages in order, skipping those with a DONE marker; returns name -> ran."""
    order = list(fns)
    last = max(order.index(s) for s in selected)
    for name in order[:last]:
        if name not in selected and not os.path.isfile(os.path.join(_stage_dir(ctx, name), "DONE.json")):
            raise RuntimeError(f"stage {name!r} has no DONE marker; it must run before {order[last]!r}")
    ran = {}
    for name in order:
        if name not in selected:
            continue
        marker = os.path.join(_stage_dir(ctx, name), "DONE.json")
        ran[name] = not os.path.isfile(marker)
        if ran[name]:
            os.makedirs(_stage_dir(ctx, name), exist_ok=True)
            _write_json(marker, fns[name](ctx))
    return ran


def run(args: argparse.Namespace, *, cfg: RunConfig | None = None) -> dict:
    """Run the selected stages (resuming) and return results.json when the report stage has been done."""
    if cfg is None:
        cfg = (_smoke_config if args.smoke else _full_config)(device=args.device, compile=args.compile)
    dataset_dir = args.dataset or os.path.join(args.out, "dataset")
    ctx = _Ctx(model_path=args.model_path, corpus=args.corpus, integrity=args.integrity, out=args.out,
               dataset_dir=dataset_dir, cfg=cfg, smoke=args.smoke, device=args.device)
    _run_stages(ctx=ctx, fns=_STAGE_FNS, selected=list(args.stages or STAGES))
    results = os.path.join(_stage_dir(ctx, "report"), "results.json")
    return _load_json(results) if os.path.isfile(results) else {}


def _parse(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("model_path")
    p.add_argument("--corpus", required=True)
    p.add_argument("--integrity", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--dataset", help="an existing dataset dir from b1_dataset.py; default builds under --out")
    p.add_argument("--stages", nargs="+", choices=STAGES, help="default: all, in order")
    p.add_argument("--device", default="cuda")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--compile", action="store_true", help="torch.compile the head's MLP for training")
    p.add_argument("--allow-any-out", action="store_true", help="tests only: accept an --out off /mnt/nvme1")
    args = p.parse_args(argv)
    if args.smoke and args.dataset:
        p.error("--smoke builds its own relabelled dataset; do not pass --dataset")
    if not args.allow_any_out and not os.path.realpath(args.out).startswith(OUT_ROOT + os.sep):
        p.error(f"--out must be under {OUT_ROOT} (pass --allow-any-out in tests)")
    return args


def main(argv: list[str] | None = None) -> int:
    results = run(_parse(argv))
    print(json.dumps({"verdict": results.get("verdict"), "frozen_sha256": results.get("frozen_sha256")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
