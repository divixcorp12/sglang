"""A gather-time head for the VRAM hot cache: P(a routed expert of this verify is routed again by the next verify).

The question it serves (vram_cache_replay.py): at a layer's gather in verify v, which of the misses should take the
GPU lanes (and so be inserted), and which residents should be the victims. A one-verify oracle is worth +68 hits per
verify at today's lane count; the gather-time lookup table (position, acceptance odds, token count, score, gate
weight) gets a fifth of that. This head adds what only the router input carries: the context of the token that routes
the expert, as PCA(x) of layer L's gate input, with a per-(layer, expert) bias and, in variant B, a per-expert
direction in PCA space.

Data: B0 corpus blocks (``trace.jsonl`` schema-2 ``graph_routes`` with ``router`` indices, ``router.{json,ids,w,x,
seq}.bin``, ``verify-accept.*.jsonl``) for training, and spec_margin captures (``stages.jsonl`` plus the same router
files) held out for evaluation. One example per (verify v, layer L, expert e routed at v) whose next verify in the
same request exists: label ``e in routes(v+1, L)``; input ``z = PCA_L(x[v, L, i*])`` for the token ``i*`` routing e
with the largest gate weight, and the dense features the table used plus age, residency and live-token count. The
per-layer PCA (rank ``--pca-rank``) is fit on a sample of training rows only.

Model per layer: ``logit = b[e] + w.f + a.z (+ U[e].z in variant B)``. Trained with BCE (AdamW, cosine), early stop on
the ``--dev`` split's log loss. ``predict`` writes an npz keyed by (router index, layer, expert) that
``vram_cache_replay.py --pred`` reads.

Usage:
  next_verify_head.py extract --out D --train BLOCK... [--eval NAME=CAPTURE ...] [--pca-rank 64] [--pca-every 8]
  next_verify_head.py train --data D --out H [--variant A|B] [--epochs 8] [--dev-frac 0.1]
  next_verify_head.py predict --data D --head H --split NAME --out PRED.npz
  next_verify_head.py report --data D --head H --split NAME
CPU only; set OMP/torch threads from the environment (``OMP_NUM_THREADS``).
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import sys

import numpy as np

DENSE = ("cnt1", "cnt2", "cnt3p", "pos0", "pos1", "pos2", "pos3", "pos4", "pos5", "gate_w", "log_score", "score0",
         "log_age", "first_touch", "resident", "live", "odds")
NUM_EXPERTS = 384
DECAY = 0.98


# ---- reading a capture or block ------------------------------------------------------------------------------------

def bf16_rows(words: np.ndarray) -> np.ndarray:
    return (words.astype(np.uint32) << 16).view(np.float32)


class Source:
    """One directory's verify sequence with its router capture."""

    def __init__(self, d: str):
        self.dir = d
        log = os.path.join(d, "trace.jsonl") if os.path.exists(os.path.join(d, "trace.jsonl")) else os.path.join(d, "stages.jsonl")
        self.header, recs = None, []
        with open(log) as f:
            for line in f:
                r = json.loads(line)
                k = r.get("kind")
                if k == "graph_routes_header":
                    self.header = r
                elif k == "graph_routes" and r.get("phase") == "target_verify" and "hot" in r and r.get("router") is not None:
                    recs.append(r)
        if self.header is None or not recs:
            raise ValueError(f"{d}: no schema-2 graph_routes")
        with open(os.path.join(d, "router.json")) as f:
            meta = json.load(f)
        self.layers, self.tokens, self.topk, self.hidden = (len(meta["layer_ids"]), int(meta["tokens"]),
                                                              int(meta["topk"]), int(meta["hidden"]))
        self.seq = np.fromfile(os.path.join(d, "router.seq.bin"), dtype=np.int64).reshape(-1, 3)
        n = self.seq.shape[0]
        self.ids = np.fromfile(os.path.join(d, "router.ids.bin"), dtype=np.int32).reshape(n, self.layers, self.tokens, self.topk)
        self.w = np.fromfile(os.path.join(d, "router.w.bin"), dtype=np.float32).reshape(n, self.layers, self.tokens, self.topk)
        self.x = np.memmap(os.path.join(d, "router.x.bin"), dtype=np.uint16, mode="r",
                           shape=(n, self.layers, self.tokens, self.hidden))
        accepts = []
        for path in glob.glob(os.path.join(d, "verify-accept.*.jsonl")):
            with open(path) as f:
                accepts = [json.loads(line) for line in f]
        confs = {}
        for path in glob.glob(os.path.join(d, "draft-top-prob.*.jsonl")):
            with open(path) as f:
                for e in map(json.loads, f):
                    confs[(e["rid"], e["k"])] = e["top_prob"]
        # the i-th verify record pairs with the i-th accept line (both written at the verify's end)
        self.recs = []
        for i, r in enumerate(recs):
            ri = r["router"]
            if ri >= n or self.seq[ri, 0] != r["seq"]:
                continue
            acc = accepts[i] if i < len(accepts) else None
            rids = r.get("rids") or ([acc["rid"]] if acc else [])
            if len(rids) != 1:
                continue  # warm-up or ambiguous
            r["rid"] = rids[0]
            r["live"] = int(self.seq[ri, 2]) or int(r.get("forward_tokens") or self.tokens)
            r["conf"] = confs.get((acc["rid"], acc["k"])) if acc else None
            self.recs.append(r)

    def pairs(self):
        """(record, next record) for consecutive verifies of the same request."""
        for a, b in zip(self.recs, self.recs[1:]):
            if a["rid"] == b["rid"]:
                yield a, b

    def x_rows(self, router: int, layer: int, live: int) -> np.ndarray:
        return bf16_rows(np.asarray(self.x[router, layer, :live, :]))


# ---- PCA -----------------------------------------------------------------------------------------------------------

def fit_pca(sources: list, rank: int, every: int, max_rows: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-layer mean [L, hidden] and projection [L, hidden, rank] from every ``every``-th training verify."""
    layers, hidden = sources[0].layers, sources[0].hidden
    mean = np.zeros((layers, hidden), dtype=np.float32)
    proj = np.zeros((layers, hidden, rank), dtype=np.float32)
    for L in range(layers):
        rows = []
        total = 0
        for s in sources:
            for i, r in enumerate(s.recs):
                if i % every:
                    continue
                rows.append(s.x_rows(r["router"], L, r["live"]))
                total += rows[-1].shape[0]
                if total >= max_rows:
                    break
            if total >= max_rows:
                break
        X = np.concatenate(rows, axis=0)
        mu = X.mean(axis=0)
        _, _, vt = np.linalg.svd(X - mu, full_matrices=False)
        mean[L] = mu
        proj[L] = vt[:rank].T
        print(f"pca layer {L}: {X.shape[0]} rows", file=sys.stderr, flush=True)
    return mean, proj


# ---- feature extraction --------------------------------------------------------------------------------------------

def score_bucket(s: float) -> int:
    return 0 if s <= 0 else 1 if s < 1 else 2 if s < 3 else 3 if s < 8 else 4


def extract_source(s: Source, mean: np.ndarray, proj: np.ndarray, out_dir: str, name: str) -> dict:
    """Write per-layer shards ``out_dir/name/L<LL>.npz`` (z, f, e, y, router, miss, tbl) for every routed expert of
    every verify that has a next verify in its request."""
    layers, topk = s.layers, s.topk
    per_layer = [collections.defaultdict(list) for _ in range(layers)]
    score = [collections.defaultdict(float) for _ in range(layers)]
    last = [dict() for _ in range(layers)]
    cur_rid = None
    n_pairs = 0
    for idx, r in enumerate(s.recs):
        if r["rid"] != cur_rid:
            cur_rid = r["rid"]
            score = [collections.defaultdict(float) for _ in range(layers)]
            last = [dict() for _ in range(layers)]
            t_in_rid = 0
        nxt = s.recs[idx + 1] if idx + 1 < len(s.recs) and s.recs[idx + 1]["rid"] == r["rid"] else None
        live = r["live"]
        ri = r["router"]
        probs = r.get("conf")
        cum = [1.0]
        if probs:
            for p in probs:
                cum.append(cum[-1] * p)
        for L in range(layers):
            ids = s.ids[ri, L, :live, :]
            w = s.w[ri, L, :live, :]
            hot = set(r["hot"][L])
            nxt_set = set(e for e in nxt["routes"][L] if e >= 0) if nxt is not None else None
            best_tok, best_w, cnt, fpos = {}, {}, collections.Counter(), {}
            for tok in range(live):
                for e, wt in zip(ids[tok].tolist(), w[tok].tolist()):
                    if e < 0:
                        continue
                    cnt[e] += 1
                    if e not in best_w or wt > best_w[e]:
                        best_w[e], best_tok[e] = wt, tok
                    if e not in fpos:
                        fpos[e] = tok
            if nxt_set is not None:
                z_all = (s.x_rows(ri, L, live) - mean[L]) @ proj[L]
                P = per_layer[L]
                for e in cnt:
                    sc = score[L][e]
                    age = t_in_rid - last[L][e] if e in last[L] else -1
                    pos = fpos[e]
                    odds = cum[min(pos, len(cum) - 1)] if probs else 1.0
                    f = np.zeros(len(DENSE), dtype=np.float32)
                    f[0 if cnt[e] == 1 else 1 if cnt[e] == 2 else 2] = 1.0
                    f[3 + min(pos, 5)] = 1.0
                    f[9] = best_w[e]
                    f[10] = np.log1p(sc)
                    f[11] = 1.0 if sc <= 0 else 0.0
                    f[12] = np.log1p(age) if age >= 0 else 0.0
                    f[13] = 1.0 if age < 0 else 0.0
                    f[14] = 1.0 if e in hot else 0.0
                    f[15] = live / 6.0
                    f[16] = odds
                    P["z"].append(z_all[best_tok[e]].astype(np.float32))
                    P["f"].append(f)
                    P["e"].append(e)
                    P["y"].append(1 if e in nxt_set else 0)
                    P["router"].append(ri)
                    P["miss"].append(0 if e in hot else 1)
                    P["tbl"].append((min(cnt[e], 3), min(pos, 3), 0 if odds < 0.2 else 1 if odds < 0.5 else 2 if odds < 0.8 else 3,
                                     score_bucket(sc), 0 if best_w[e] < 0.1 else 1 if best_w[e] < 0.2 else 2 if best_w[e] < 0.3 else 3))
            # advance the policy's state as the boundary would
            d = DECAY ** live
            for e in list(score[L]):
                score[L][e] *= d
            for e, c in cnt.items():
                score[L][e] += c
                last[L][e] = t_in_rid
        if nxt is not None:
            n_pairs += 1
        t_in_rid += 1
    os.makedirs(os.path.join(out_dir, name), exist_ok=True)
    counts = {}
    for L in range(layers):
        P = per_layer[L]
        if not P["y"]:
            continue
        np.savez(os.path.join(out_dir, name, f"L{L:02d}.npz"),
                 z=np.stack(P["z"]), f=np.stack(P["f"]), e=np.asarray(P["e"], dtype=np.int16),
                 y=np.asarray(P["y"], dtype=np.int8), router=np.asarray(P["router"], dtype=np.int32),
                 miss=np.asarray(P["miss"], dtype=np.int8), tbl=np.asarray(P["tbl"], dtype=np.int8))
        counts[L] = len(P["y"])
    return {"pairs": n_pairs, "verifies": len(s.recs), "examples": sum(counts.values()), "has_conf": any(r.get("conf") for r in s.recs)}


def cmd_extract(args) -> int:
    os.makedirs(args.out, exist_ok=True)
    train = [Source(d) for d in args.train]
    print({d: len(s.recs) for d, s in zip(args.train, train)}, file=sys.stderr, flush=True)
    mean, proj = fit_pca(train, args.pca_rank, args.pca_every, args.pca_rows)
    np.savez(os.path.join(args.out, "pca.npz"), mean=mean, proj=proj)
    manifest = {"train": {}, "eval": {}, "pca_rank": args.pca_rank, "dense": DENSE}
    for d, s in zip(args.train, train):
        name = "train-" + os.path.basename(os.path.normpath(d))
        manifest["train"][name] = extract_source(s, mean, proj, args.out, name) | {"dir": d}
        print(name, manifest["train"][name], file=sys.stderr, flush=True)
    for spec in args.eval or []:
        name, d = spec.split("=", 1)
        s = Source(d)
        manifest["eval"][name] = extract_source(s, mean, proj, args.out, name) | {"dir": d}
        print(name, manifest["eval"][name], file=sys.stderr, flush=True)
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    return 0


# ---- training ------------------------------------------------------------------------------------------------------

def load_split(data: str, names: list, layer: int):
    zs, fs, es, ys, rs, ms, tb = [], [], [], [], [], [], []
    for name in names:
        path = os.path.join(data, name, f"L{layer:02d}.npz")
        if not os.path.exists(path):
            continue
        a = np.load(path)
        zs.append(a["z"]); fs.append(a["f"]); es.append(a["e"]); ys.append(a["y"]); rs.append(a["router"]); ms.append(a["miss"]); tb.append(a["tbl"])
    if not ys:
        return None
    return {"z": np.concatenate(zs), "f": np.concatenate(fs), "e": np.concatenate(es).astype(np.int64),
            "y": np.concatenate(ys).astype(np.float32), "router": np.concatenate(rs), "miss": np.concatenate(ms),
            "tbl": np.concatenate(tb)}


def build_model(rank: int, variant: str):
    import torch

    class Head(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = torch.nn.Embedding(NUM_EXPERTS, 1)
            self.dense = torch.nn.Linear(len(DENSE), 1)
            self.z = torch.nn.Linear(rank, 1, bias=False)
            self.u = torch.nn.Embedding(NUM_EXPERTS, rank) if variant == "B" else None
            torch.nn.init.zeros_(self.bias.weight)
            if self.u is not None:
                torch.nn.init.zeros_(self.u.weight)

        def forward(self, z, f, e):
            out = self.bias(e).squeeze(-1) + self.dense(f).squeeze(-1) + self.z(z).squeeze(-1)
            if self.u is not None:
                out = out + (self.u(e) * z).sum(-1)
            return out

    return Head()


def standardize(train_f: np.ndarray, train_z: np.ndarray):
    fm, fs = train_f.mean(0), train_f.std(0) + 1e-6
    zs = train_z.std(0) + 1e-6
    return fm, fs, zs


def train_layer(data: str, names: list, layer: int, variant: str, epochs: int, dev_frac: float, seed: int, rank: int):
    import torch

    torch.manual_seed(seed)
    d = load_split(data, names, layer)
    if d is None:
        return None
    n = len(d["y"])
    # hold out whole forwards (router indices) for early stopping
    routers = np.unique(d["router"])
    rng = np.random.default_rng(seed)
    dev_r = set(rng.choice(routers, size=max(1, int(len(routers) * dev_frac)), replace=False).tolist())
    is_dev = np.fromiter((r in dev_r for r in d["router"]), dtype=bool, count=n)
    fm, fs, zs = standardize(d["f"][~is_dev], d["z"][~is_dev])
    F = torch.from_numpy((d["f"] - fm) / fs)
    Z = torch.from_numpy(d["z"] / zs)
    E = torch.from_numpy(d["e"])
    Y = torch.from_numpy(d["y"])
    tr = torch.from_numpy(np.flatnonzero(~is_dev))
    dv = torch.from_numpy(np.flatnonzero(is_dev))
    model = build_model(rank, variant)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    steps = epochs * ((len(tr) + 4095) // 4096)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, steps))
    loss_fn = torch.nn.BCEWithLogitsLoss()
    best, best_state = float("inf"), None
    for ep in range(epochs):
        model.train()
        perm = tr[torch.randperm(len(tr))]
        for i in range(0, len(perm), 4096):
            b = perm[i:i + 4096]
            opt.zero_grad()
            loss = loss_fn(model(Z[b], F[b], E[b]), Y[b])
            loss.backward()
            opt.step()
            sched.step()
        model.eval()
        with torch.no_grad():
            dl = loss_fn(model(Z[dv], F[dv], E[dv]), Y[dv]).item() if len(dv) else 0.0
        if dl < best:
            best, best_state = dl, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    base_rate = float(d["y"][is_dev].mean()) if len(dv) else float("nan")
    return {"state": {k: v.numpy() for k, v in best_state.items()}, "fm": fm, "fs": fs, "zs": zs, "dev_logloss": best,
            "dev_base_logloss": float(-(base_rate * np.log(base_rate + 1e-9) + (1 - base_rate) * np.log(1 - base_rate + 1e-9))),
            "n": n, "n_dev": int(len(dv))}


def cmd_train(args) -> int:
    import torch

    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    with open(os.path.join(args.data, "manifest.json")) as f:
        manifest = json.load(f)
    names = sorted(manifest["train"])
    rank = int(manifest["pca_rank"])
    os.makedirs(args.out, exist_ok=True)
    summary = {"variant": args.variant, "epochs": args.epochs, "layers": {}}
    for L in range(40):
        r = train_layer(args.data, names, L, args.variant, args.epochs, args.dev_frac, args.seed, rank)
        if r is None:
            continue
        np.savez(os.path.join(args.out, f"L{L:02d}.npz"), fm=r["fm"], fs=r["fs"], zs=r["zs"],
                 **{"state_" + k: v for k, v in r["state"].items()})
        summary["layers"][L] = {k: r[k] for k in ("dev_logloss", "dev_base_logloss", "n", "n_dev")}
        print(f"layer {L}: n={r['n']} dev logloss {r['dev_logloss']:.4f} (base {r['dev_base_logloss']:.4f})", file=sys.stderr, flush=True)
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    return 0


def load_layer_model(head_dir: str, layer: int, rank: int, variant: str):
    import torch

    path = os.path.join(head_dir, f"L{layer:02d}.npz")
    if not os.path.exists(path):
        return None
    a = np.load(path)
    model = build_model(rank, variant)
    model.load_state_dict({k[len("state_"):]: torch.from_numpy(a[k]) for k in a.files if k.startswith("state_")})
    model.eval()
    return model, a["fm"], a["fs"], a["zs"]


def predict_split(data: str, head: str, split: str) -> dict:
    """Per layer: arrays router, e, p, y, miss, tbl for the split."""
    import torch

    with open(os.path.join(data, "manifest.json")) as f:
        manifest = json.load(f)
    with open(os.path.join(head, "summary.json")) as f:
        summary = json.load(f)
    rank, variant = int(manifest["pca_rank"]), summary["variant"]
    out = {}
    for L in range(40):
        d = load_split(data, [split], L)
        m = load_layer_model(head, L, rank, variant)
        if d is None or m is None:
            continue
        model, fm, fs, zs = m
        with torch.no_grad():
            logits = model(torch.from_numpy(d["z"] / zs), torch.from_numpy((d["f"] - fm) / fs), torch.from_numpy(d["e"]))
            p = torch.sigmoid(logits).numpy()
        out[L] = {"router": d["router"], "e": d["e"], "p": p, "y": d["y"], "miss": d["miss"], "tbl": d["tbl"], "f": d["f"]}
    return out


def cmd_predict(args) -> int:
    pred = predict_split(args.data, args.head, args.split)
    routers = np.concatenate([np.full(len(v["e"]), 0, dtype=np.int32) + v["router"] for v in pred.values()])
    layers = np.concatenate([np.full(len(v["e"]), L, dtype=np.int16) for L, v in pred.items()])
    experts = np.concatenate([v["e"].astype(np.int16) for v in pred.values()])
    ps = np.concatenate([v["p"].astype(np.float32) for v in pred.values()])
    np.savez(args.out, router=routers, layer=layers, expert=experts, p=ps)
    print(f"{args.out}: {len(ps)} predictions over {len(pred)} layers", file=sys.stderr)
    return 0


def cmd_report(args) -> int:
    """Precision/recall at k on the split, k = the misses the capture inserted (hot[t+1]), against the table and the
    oracle; the table is fit on the training shards. Only misses (not resident at v) are ranked, as the gather would."""
    pred = predict_split(args.data, args.head, args.split)
    with open(os.path.join(args.data, "manifest.json")) as f:
        manifest = json.load(f)
    # table from the training shards
    tbl_n, tbl_y = collections.Counter(), collections.Counter()
    for L in range(40):
        d = load_split(args.data, sorted(manifest["train"]), L)
        if d is None:
            continue
        for key, y in zip(map(tuple, d["tbl"].tolist()), d["y"].tolist()):
            tbl_n[key] += 1
            tbl_y[key] += y
    rate = sum(tbl_y.values()) / max(1, sum(tbl_n.values()))
    table = {k: (tbl_y[k] + rate) / (tbl_n[k] + 1) for k in tbl_n}
    # the inserted set per (router, layer) comes from the eval source's hot snapshots
    src = Source(manifest["eval"][args.split]["dir"])
    by_router = {r["router"]: i for i, r in enumerate(src.recs)}
    agg = {k: collections.Counter() for k in ("head", "table", "score", "oracle", "actual")}
    ll = collections.Counter()
    for L, v in pred.items():
        groups = collections.defaultdict(list)
        for i in range(len(v["e"])):
            groups[int(v["router"][i])].append(i)
        for ri, idxs in groups.items():
            t = by_router.get(ri)
            if t is None or t + 1 >= len(src.recs) or src.recs[t + 1]["rid"] != src.recs[t]["rid"]:
                continue
            nxt_hot = set(src.recs[t + 1]["hot"][L])
            miss_idx = [i for i in idxs if v["miss"][i]]
            if not miss_idx:
                continue
            inserted = [i for i in miss_idx if int(v["e"][i]) in nxt_hot]
            k = len(inserted)
            recur = sum(int(v["y"][i]) for i in miss_idx)
            for i in miss_idx:
                p = float(v["p"][i])
                y = float(v["y"][i])
                ll["head"] += -(y * np.log(p + 1e-9) + (1 - y) * np.log(1 - p + 1e-9))
                pt = table.get(tuple(v["tbl"][i].tolist()), rate)
                ll["table"] += -(y * np.log(pt + 1e-9) + (1 - y) * np.log(1 - pt + 1e-9))
                ll["n"] += 1
            rank = {
                "head": sorted(miss_idx, key=lambda i: -v["p"][i])[:k],
                "table": sorted(miss_idx, key=lambda i: -table.get(tuple(v["tbl"][i].tolist()), rate))[:k],
                "score": sorted(miss_idx, key=lambda i: -v["f"][i][10])[:k],
                "oracle": sorted(miss_idx, key=lambda i: -v["y"][i])[:k],
                "actual": inserted,
            }
            for key, chosen in rank.items():
                a = agg[key]
                a["k"] += k
                a["tp"] += sum(int(v["y"][i]) for i in chosen)
                a["recur"] += recur
    res = {key: {"precision": a["tp"] / max(1, a["k"]), "recall": a["tp"] / max(1, a["recur"]), "k": a["k"]} for key, a in agg.items()}
    res["logloss"] = {"head": ll["head"] / max(1, ll["n"]), "table": ll["table"] / max(1, ll["n"]), "n": ll["n"]}
    print(json.dumps(res, indent=1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--out", required=True)
    e.add_argument("--train", nargs="+", required=True)
    e.add_argument("--eval", nargs="*")
    e.add_argument("--pca-rank", type=int, default=64)
    e.add_argument("--pca-every", type=int, default=8)
    e.add_argument("--pca-rows", type=int, default=6000)
    e.set_defaults(fn=cmd_extract)
    t = sub.add_parser("train")
    t.add_argument("--data", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--variant", default="B", choices=("A", "B"))
    t.add_argument("--epochs", type=int, default=8)
    t.add_argument("--dev-frac", type=float, default=0.1)
    t.add_argument("--seed", type=int, default=0)
    t.set_defaults(fn=cmd_train)
    p = sub.add_parser("predict")
    p.add_argument("--data", required=True)
    p.add_argument("--head", required=True)
    p.add_argument("--split", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_predict)
    r = sub.add_parser("report")
    r.add_argument("--data", required=True)
    r.add_argument("--head", required=True)
    r.add_argument("--split", required=True)
    r.set_defaults(fn=cmd_report)
    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
