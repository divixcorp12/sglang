"""Can a verify's token ids predict layer 0's routed experts before the verify starts? Offline, CPU only.

Layer 0 has the most forced NVMe misses of any layer, and the in-forward lookahead (layer T's gate on layer T-1's
input) never reaches it. The verify's tokens -- the last accepted token and DSpark's drafts -- are known before the
verify runs, so layer 0's experts might be predicted from the ids alone and read from NVMe ahead of the demand.

Inputs:

* router captures (``router.{json,x.bin,ids.bin,seq.bin}``) with their route log (``stages.jsonl`` or
  ``trace.jsonl``, ``graph_routes`` lines): every graph forward's layer-0 MoE input and routed top-6 per token;
* a decode capture (schema 1, one token a forward, ``responses.jsonl``) as extra training data whose ids are also
  checked against the re-tokenized response text;
* for a capture with an InstrBuild job trace (``events.*.jsonl``): each record's ``miss_expert`` events (the routed
  experts read from NVMe, the true tier labels) and the draft / demand timestamps for the lead.

Token ids are not logged, so they are recovered as the nearest normed embedding (cosine) to the layer-0 MoE input,
special tokens excluded; the decode capture's text gives the accuracy. Predictors, each a function of the token id
only (so a 129,280-row table the host can index during the draft):

* ``prev``: the previous verify's layer-0 experts (same request);
* ``table``: P(expert | token) counted on the training split, shrunk to the popularity prior;
* ``emb_gate``: layer 0's gate on ffn_norm(embedding), attention skipped (the hyper-connection streams are copies of
  the embedding before layer 0, so this is exactly the gate without attention's contribution);
* ``emb_gate_off``: the same with the training split's mean (MoE input - normed embedding) added;
* ``emb_gate_off_bias``: plus a per-expert bias fitted so each expert's predicted top-k rate matches its routed rate;
* ``table_gate``: the table shrunk toward softmax(tau * emb_gate_off_bias) instead of popularity.

The job trace's forwards are joined to the route log's by ``match_forwards``. A verify's prediction is the union of each live token's top-k. Scored per verify: recall on all routed experts and on
NVMe rows (``miss_expert``), precision, useful NVMe reads (predicted NVMe rows) and wasted ones (predicted experts
the verify does not route that sit in NVMe). The tier of an unrouted expert is not logged: it is approximated as
"not VRAM-hot and not routed at layer 0 in the last W forwards", W calibrated on the routed experts' true labels;
``wasted_any`` (every unrouted prediction) bounds it from above.

Usage: layer0_draft_predict.py OUT_DIR --decode DIR --verify NAME=DIR [...] --split TRAIN,..>TEST,.. [...]
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import statistics
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import layer_misses  # noqa: E402

LAYER, HIDDEN, N_EXPERTS, TOPK = 0, 5120, 384, 6
EPS = 1e-20  # config rms_norm_eps
KS = (6, 8, 12)
EDGES = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0001)
MODEL = "/data/models/slang/nvfp4-work/cc-expert-prediction/dsv41-full40"


def bf16_to_f32(bits: np.ndarray) -> np.ndarray:
    return (np.asarray(bits, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def rms_norm(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    return x / np.sqrt(np.square(x).mean(-1, keepdims=True) + EPS) * w


def _unit(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def nearest_ids(x: np.ndarray, emb, w: np.ndarray, exclude, chunk: int = 8192):
    """Each row's nearest normed embedding by cosine: (ids, best cosine, second cosine). ``emb`` is [V, H] in any
    dtype ``rms_norm`` takes, or a callable (start, stop) -> rows; ``exclude`` ids are never chosen."""
    vocab = emb.shape[0] if not callable(emb) else emb.vocab
    get = emb if callable(emb) else (lambda a, b: emb[a:b])
    xu = _unit(x.astype(np.float32))
    best = np.full((len(x), 2), -2.0, np.float32)
    best_i = np.zeros((len(x), 2), np.int64)
    excl = np.asarray(sorted(exclude), np.int64)
    for a in range(0, vocab, chunk):
        b = min(vocab, a + chunk)
        s = xu @ _unit(rms_norm(get(a, b), w)).T
        drop = excl[(excl >= a) & (excl < b)] - a
        s[:, drop] = -2.0
        j = np.argsort(-s, axis=1)[:, :2]
        cand = np.concatenate([best, np.take_along_axis(s, j, 1)], 1)
        cand_i = np.concatenate([best_i, j + a], 1)
        o = np.argsort(-cand, axis=1, kind="stable")[:, :2]
        best, best_i = np.take_along_axis(cand, o, 1), np.take_along_axis(cand_i, o, 1)
    return best_i[:, 0], best[:, 0], best[:, 1]


def gate_scores(x: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    """DSV4.1's router score: sqrt(softplus(W x)) + b (the bias only selects; it is not a weight)."""
    z = x.astype(np.float32) @ W.T
    return np.sqrt(np.logaddexp(0.0, z)) + b


def topk(scores: np.ndarray, k: int) -> np.ndarray:
    """Top-k columns of each row, best first."""
    j = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    return np.take_along_axis(j, np.argsort(-np.take_along_axis(scores, j, 1), axis=1, kind="stable"), 1)


def margin_conf(scores: np.ndarray, k: int = TOPK, sharp: float = 20.0) -> np.ndarray:
    """sigmoid(sharp * (score - k-th score)): the router-capture study's confidence for a gate's candidates."""
    kth = -np.partition(-scores, k - 1, axis=1)[:, k - 1:k]
    return 1.0 / (1.0 + np.exp(-sharp * (scores - kth)))


class TokenTable:
    """Per token id, how often each expert was routed in the training split."""

    def __init__(self, counts: dict, n_experts: int, k: int):
        self.counts, self.n_experts, self.k = counts, n_experts, k

    @classmethod
    def fit(cls, tokens: np.ndarray, routes: np.ndarray, n_experts: int = N_EXPERTS) -> "TokenTable":
        counts: dict = {}
        for t, r in zip(np.asarray(tokens).tolist(), np.asarray(routes)):
            row = counts.setdefault(t, np.zeros(n_experts + 1, np.float32))  # last column: times seen
            row[r] += 1.0
            row[-1] += 1.0
        return cls(counts, n_experts, np.asarray(routes).shape[1])

    def seen(self, tokens: np.ndarray) -> np.ndarray:
        return np.array([int(self.counts[t][-1]) if t in self.counts else 0 for t in np.asarray(tokens).tolist()])

    def scores(self, tokens: np.ndarray, prior: np.ndarray, beta: float = 1.0) -> np.ndarray:
        """Posterior mean P(routed | token) with ``beta`` pseudo-observations of ``prior`` ([n, E], each row's
        P(routed) for one routing, i.e. summing to 1)."""
        out = np.empty((len(tokens), self.n_experts), np.float32)
        for i, t in enumerate(np.asarray(tokens).tolist()):
            row = self.counts.get(t)
            cnt, n = (row[:-1], row[-1]) if row is not None else (0.0, 0.0)
            out[i] = (cnt + beta * self.k * prior[i]) / (n + beta)
        return out


def fit_bias(scores: np.ndarray, routes: np.ndarray, k: int = TOPK, iters: int = 300, step: float = 0.02):
    """Per-expert additive bias moving each expert's top-k rate toward its routed rate on the training split."""
    n, e = scores.shape
    want = np.bincount(np.asarray(routes).ravel(), minlength=e) / n
    bias = np.zeros(e, np.float32)
    for _ in range(iters):
        got = np.bincount(topk(scores + bias, k).ravel(), minlength=e) / n
        bias += step * np.sign(want - got) * (np.abs(want - got) > 0.5 / n)
    return bias


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(1, keepdims=True)


def fit_tau(scores: np.ndarray, routes: np.ndarray, grid=(1, 2, 4, 8, 16, 32, 64)) -> float:
    """The temperature whose softmax(tau * score) gives the routed experts the highest log-likelihood."""
    rows = np.arange(len(scores))[:, None]
    return float(max(grid, key=lambda t: np.log(softmax(t * scores)[rows, routes] + 1e-12).sum()))


def union_topk(scores: np.ndarray, conf: np.ndarray, k: int, live: int) -> dict:
    """The verify's prediction: each live token's top-k, an expert keeping its best confidence over tokens."""
    pred: dict = {}
    top = topk(scores[:live], k)
    for t in range(live):
        for e in top[t].tolist():
            pred[e] = max(pred.get(e, 0.0), float(conf[t, e]))
    return pred


def score_verify(pred: dict, routed: set, nvme, unrouted_nvme) -> dict:
    """Counts for one verify; the NVMe ones only when the capture logs the routed experts' tier."""
    p = set(pred)
    r = {"predicted": len(p), "routed": len(routed), "hit_routed": len(p & routed), "wasted_any": len(p - routed)}
    if nvme is not None:
        r.update(labelled=1, nvme=len(nvme), hit_nvme=len(p & nvme), wasted_nvme=len((p - routed) & unrouted_nvme))
    return r


def bin_verify(pred: dict, routed: set, nvme: set, unrouted_nvme: set, edges=EDGES) -> list:
    out = [{"candidates": 0, "routed": 0, "useful": 0, "wasted": 0} for _ in range(len(edges) - 1)]
    for e, c in pred.items():
        i = max(0, min(len(out) - 1, int(np.searchsorted(edges, c, side="right")) - 1))
        out[i]["candidates"] += 1
        out[i]["routed"] += e in routed
        out[i]["useful"] += e in nvme
        out[i]["wasted"] += (e not in routed) and (e in unrouted_nvme)
    return out


def recency_nvme(history: list, window: int, hot: set, n_experts: int = N_EXPERTS) -> set:
    """Approximate NVMe tier: not VRAM-hot and not routed at layer 0 by any of the last ``window`` forwards."""
    recent = set().union(*history[-window:]) if window and history else set()
    return set(range(n_experts)) - recent - set(hot)


def leads(events: list) -> dict:
    """Per verify (row 0's gen): ms from the last ``draft_end``, the last ``draft_observed``, and the previous
    forward's row-39 ``gate_open`` to row 0's first ``copy_submit`` (its demand)."""
    out, last = {}, {"draft_end": None, "draft_observed": None, "gate39": None}
    for e in sorted(events, key=lambda e: e["ns"]):
        kind = e.get("event")
        if kind in ("draft_end", "draft_observed"):
            last[kind] = e["ns"]
        elif kind == "gate_open" and e["row"] == 39:
            last["gate39"] = e["ns"]
        elif kind == "copy_submit" and e["row"] == 0 and e["gen"] not in out:
            ms = lambda t: None if t is None else round((e["ns"] - t) / 1e6, 6)
            out[e["gen"]] = {"draft_end_ms": ms(last["draft_end"]), "draft_observed_ms": ms(last["draft_observed"]),
                             "prev_verify_end_ms": ms(last["gate39"])}
    return out


def match_forwards(job: list, routes: list, window: int = 40) -> list:
    """(job forward, route line) pairs, in order: each job forward's first route line at or after the previous
    match whose per-layer routes hold every expert it missed. The job trace drops forwards it could not complete, so
    no single offset joins the two logs (verify_split.align's assumption fails on cut-timeline: 0.689)."""
    out, j = [], 0
    for i, fwd in enumerate(job):
        for jj in range(j, min(len(routes), j + window)):
            if all(m <= routes[jj][row] for row, m in enumerate(fwd)):
                out.append((i, jj))
                j = jj + 1
                break
    return out


# ---- loading ---------------------------------------------------------------------------------------------------


def _route_lines(d: str) -> tuple[list, list]:
    """(graph_routes lines, layer-0 stage lines with their experts) of a capture's route log."""
    path = next(p for p in (os.path.join(d, "stages.jsonl"), os.path.join(d, "trace.jsonl")) if os.path.exists(p))
    routes, stage0 = [], []
    with open(path) as f:
        for line in f:
            if '"graph_routes"' in line[:40]:
                x = json.loads(line)
                if x.get("kind") == "graph_routes":
                    routes.append(x)
            elif line.startswith('{"forward"') and '"layer": 0,' in line[:60]:
                x = json.loads(line)
                if "experts" in x:
                    stage0.append(x)
    return routes, stage0


def load_verify(name: str, d: str) -> list:
    with open(os.path.join(d, "router.json")) as f:
        h = json.load(f)
    t, k, layers = h["tokens"], h["topk"], len(h["layer_ids"])
    ids = np.fromfile(os.path.join(d, "router.ids.bin"), dtype=np.int32).reshape(-1, layers, t, k)
    xs = np.memmap(os.path.join(d, "router.x.bin"), dtype=np.uint16, mode="r").reshape(-1, layers, t, HIDDEN)
    lines, stage0 = _route_lines(d)
    accepts = {}
    for p in glob.glob(os.path.join(d, "verify-accept.*.jsonl")):
        with open(p) as f:
            for e in map(json.loads, f):
                accepts[(e["rid"], e["k"])] = e["num_correct_drafts"]
    # Every forward's layer-0 experts in order, for the recency tier: graph forwards from the route log, eager ones
    # (prefill) from the stage trace, which logs only those.
    order = sorted([(x["forward_pass_id"], set(x["experts"])) for x in stage0 if x.get("forward_pass_id") is not None]
                   + [(x["forward_pass_id"], set(x["routes"][LAYER])) for x in lines
                      if x.get("forward_pass_id") is not None and x["forward_pass_id"] >= 0], key=lambda o: o[0])
    pos_of = {fp: i for i, (fp, _) in enumerate(order)}
    job_nvme, job_gen = {}, {}
    paths = sorted(glob.glob(os.path.join(d, "events.*.jsonl")))
    if paths:
        recs = layer_misses.load(paths)
        missed = collections.defaultdict(set)
        for p in paths:
            with open(p) as f:
                for line in f:
                    if '"miss_expert"' in line:
                        e = json.loads(line)
                        missed[(e["row"], e["gen"])].add(e["a"])
        fwds = layer_misses.forwards(recs, layers)
        job = [[missed.get((r["row"], r["gen"]), set()) for r in fw] for fw in fwds]
        pairs = match_forwards(job, [[set(r) for r in x["routes"]] for x in lines])
        if len(pairs) < 0.95 * len(job):
            raise ValueError(f"{name}: only {len(pairs)} of {len(job)} job forwards found their route line")
        for i, li in pairs:
            job_nvme[li] = job[i][LAYER]
            job_gen[li] = fwds[i][0]["gen"]
    out, seen = [], collections.Counter()
    for li, x in enumerate(lines):
        if x.get("phase") != "target_verify" or x.get("router") is None:
            continue
        rid = (x.get("rids") or [None])[0]
        kk = seen[rid]
        seen[rid] += 1
        r = x["router"]
        live = int(x.get("forward_tokens") or t)
        # A few captured rows inside the live count are all zero (an unwritten ring slot): not a token.
        while live and not np.any(np.asarray(xs[r, LAYER, live - 1])):
            live -= 1
        if not live:
            continue
        fp = x.get("forward_pass_id")
        hist = [s for _, s in order[:pos_of[fp]]] if fp in pos_of else None
        out.append({
            "capture": name, "rid": rid, "k": kk, "live": live, "num_correct_drafts": accepts.get((rid, kk)),
            "x": bf16_to_f32(np.asarray(xs[r, LAYER, :live])), "routes": ids[r, LAYER, :live].astype(np.int64),
            "hot": set(x["hot"][LAYER]) if x.get("hot") else set(), "history": hist,
            "nvme": job_nvme.get(li) if paths else None, "gen": job_gen.get(li),
        })
    return out


def load_decode(d: str) -> dict:
    with open(os.path.join(d, "router.json")) as f:
        layers = len(json.load(f)["layer_ids"])
    xs = np.memmap(os.path.join(d, "router.x.bin"), dtype=np.uint16, mode="r").reshape(-1, layers, HIDDEN)
    lines, _ = _route_lines(d)
    keep = [x for x in lines if x.get("phase") == "decode" and x.get("router") is not None]
    rec = np.array([x["router"] for x in keep])
    return {"x": bf16_to_f32(np.asarray(xs[rec, LAYER])), "routes": np.array([x["routes"][LAYER] for x in keep]),
            "rids": [x["rids"][0] for x in keep]}


class Model:
    """Layer 0's gate, ffn_norm and the token embedding, read tensor by tensor."""

    def __init__(self, path: str):
        from safetensors import safe_open

        with open(os.path.join(path, "model.safetensors.index.json")) as f:
            index = json.load(f)["weight_map"]

        def get(key):
            with safe_open(os.path.join(path, index[key]), "pt") as f:
                return f.get_tensor(key)

        self.W = get("layers.0.ffn.gate.weight").float().numpy()
        self.b = get("layers.0.ffn.gate.bias").float().numpy()
        self.norm_w = get("layers.0.ffn_norm.weight").float().numpy()
        self._emb = get("embed.weight")  # bf16 torch tensor, about 1.3 GB
        self.vocab = self._emb.shape[0]
        with open(os.path.join(path, "tokenizer.json")) as f:
            self.special = [t["id"] for t in json.load(f).get("added_tokens", []) if t.get("special")]

    def __call__(self, a, b):
        return self._emb[a:b].float().numpy()

    shape = property(lambda self: (self.vocab, HIDDEN))

    def normed(self, tokens: np.ndarray) -> np.ndarray:
        import torch

        return rms_norm(self._emb[torch.as_tensor(np.asarray(tokens))].float().numpy(), self.norm_w)


# ---- analysis --------------------------------------------------------------------------------------------------


def recover(model: Model, xs: list) -> tuple[list, dict]:
    cat = np.concatenate(xs)
    ids, c1, c2 = nearest_ids(cat, model, model.norm_w, model.special)
    out, a = [], 0
    for x in xs:
        out.append(ids[a:a + len(x)])
        a += len(x)
    q = lambda v: [round(float(z), 3) for z in np.quantile(v, [0.1, 0.5, 0.9])]
    return out, {"queries": len(cat), "cos_best_q10_50_90": q(c1), "margin_q10_50_90": q(c1 - c2)}


def check_decode_ids(model_path: str, d: str, dec: dict, ids: np.ndarray) -> dict:
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(os.path.join(model_path, "tokenizer.json"))
    by_rid = collections.defaultdict(list)
    for i, r in enumerate(dec["rids"]):
        by_rid[r].append(i)
    match = total = 0
    with open(os.path.join(d, "responses.jsonl")) as f:
        for rsp in map(json.loads, f):
            want = tok.encode(rsp["text"], add_special_tokens=False).ids
            got = [int(ids[i]) for i in by_rid.get(rsp["id"], [])]
            n = min(len(want), len(got))
            total += n
            match += sum(a == b for a, b in zip(want[:n], got[:n]))
    return {"tokens": total, "match": match, "share": round(match / max(1, total), 4)}


class Predictors:
    def __init__(self, model: Model, train_tok: np.ndarray, train_x: np.ndarray, train_routes: np.ndarray):
        self.model = model
        en = model.normed(train_tok)
        self.offset = (train_x - en).mean(0)
        s_off = gate_scores(en + self.offset, model.W, model.b)
        self.bias = fit_bias(s_off, train_routes)
        self.tau = fit_tau(s_off + self.bias, train_routes)
        self.table = TokenTable.fit(train_tok, train_routes)
        self.pop = np.bincount(train_routes.ravel(), minlength=N_EXPERTS) / train_routes.size
        cos = (_unit(train_x) * _unit(en)).sum(1)
        self.fit_info = {"train_tokens": int(len(train_tok)), "tau": self.tau,
                         "cos_x_vs_normed_emb_q10_50_90": [round(float(z), 3) for z in np.quantile(cos, [.1, .5, .9])],
                         "cos_x_vs_normed_emb_plus_offset_median": round(float(np.median(
                             (_unit(train_x) * _unit(en + self.offset)).sum(1))), 3),
                         "bias_abs_mean": round(float(np.abs(self.bias).mean()), 4)}

    def all(self, tok: np.ndarray) -> dict:
        """name -> (scores [n, E], confidence [n, E]) for these token ids."""
        en = self.model.normed(tok)
        s_emb = gate_scores(en, self.model.W, self.model.b)
        s_off = gate_scores(en + self.offset, self.model.W, self.model.b)
        s_bias = s_off + self.bias
        pop = np.broadcast_to(self.pop, (len(tok), N_EXPERTS))
        tab = self.table.scores(tok, pop, beta=1.0)
        tab_gate = self.table.scores(tok, softmax(self.tau * s_bias), beta=1.0)
        return {"table": (tab, tab), "emb_gate": (s_emb, margin_conf(s_emb)), "emb_gate_off": (s_off, margin_conf(s_off)),
                "emb_gate_off_bias": (s_bias, margin_conf(s_bias)), "table_gate": (tab_gate, tab_gate)}


def evaluate(pred: Predictors, verifies: list, tokens: list, window: int) -> dict:
    sums = collections.defaultdict(lambda: collections.Counter())
    bins = collections.defaultdict(lambda: [collections.Counter() for _ in range(len(EDGES) - 1)])
    prev_by_rid: dict = {}
    pos0 = collections.Counter()
    for v, tok in zip(verifies, tokens):
        routed = set(v["routes"].ravel().tolist())
        nvme = v["nvme"]
        unrouted_nvme = None
        if nvme is not None and v["history"] is not None:
            unrouted_nvme = recency_nvme(v["history"], window, v["hot"]) - routed
        elif nvme is not None:
            unrouted_nvme = set(range(N_EXPERTS)) - v["hot"] - routed
        prev = prev_by_rid.get(v["rid"])
        prev_by_rid[v["rid"]] = routed
        if prev is not None:
            sums[("prev", 0)].update(score_verify(dict.fromkeys(prev, 1.0), routed, nvme, unrouted_nvme or set()))
            sums[("prev", 0)]["verifies"] += 1
        if nvme is not None:
            p0 = set(v["routes"][0].tolist())
            pos0.update(nvme=len(nvme), nvme_pos0=len(nvme & p0), verifies=1)
        for name, (s, c) in pred.all(tok).items():
            for k in KS:
                p = union_topk(s, c, k, v["live"])
                sums[(name, k)].update(score_verify(p, routed, nvme, unrouted_nvme or set()))
                sums[(name, k)]["verifies"] += 1
                if nvme is not None:
                    q = union_topk(s[:1], c[:1], k, 1)
                    sums[(name, k)].update(hit_nvme_pos0_pred=len(set(q) & nvme), pred_pos0=len(q),
                                           wasted_nvme_pos0=len((set(q) - routed) & unrouted_nvme))
                    if k == max(KS):
                        for b, x in zip(bins[name], bin_verify(p, routed, nvme, unrouted_nvme)):
                            b.update(x)
                            b["verifies"] += 1
    table = {}
    for (name, k), c in sums.items():
        n = c["verifies"]
        row = {"verifies": n, "predicted_per_verify": round(c["predicted"] / n, 2),
               "routed_per_verify": round(c["routed"] / n, 2),
               "recall_routed": round(c["hit_routed"] / max(1, c["routed"]), 3),
               "precision_routed": round(c["hit_routed"] / max(1, c["predicted"]), 3),
               "wasted_any_per_verify": round(c["wasted_any"] / n, 2)}
        if c["nvme"]:
            n = c["labelled"]  # NVMe rates are per verify whose job forward joined (tier labels known)
            row.update(labelled_verifies=n, nvme_per_verify=round(c["nvme"] / n, 2), recall_nvme=round(c["hit_nvme"] / c["nvme"], 3),
                       useful_nvme_per_verify=round(c["hit_nvme"] / n, 2),
                       wasted_nvme_per_verify=round(c["wasted_nvme"] / n, 2))
            if "pred_pos0" in c:
                row.update(pos0_useful_nvme_per_verify=round(c["hit_nvme_pos0_pred"] / n, 2),
                           pos0_wasted_nvme_per_verify=round(c["wasted_nvme_pos0"] / n, 2))
        table[f"{name}/k{k}" if k else name] = row
    conf = {}
    for name, bs in bins.items():
        conf[name] = []
        for lo, hi, b in zip(EDGES[:-1], EDGES[1:], bs):
            n = max(1, b["verifies"])
            conf[name].append({"bin": [lo, round(min(hi, 1.0), 2)], "candidates_per_verify": round(b["candidates"] / n, 2),
                               "precision_routed": round(b["routed"] / max(1, b["candidates"]), 3),
                               "useful_nvme_per_verify": round(b["useful"] / n, 2),
                               "wasted_nvme_per_verify": round(b["wasted"] / n, 2)})
    out = {"predictors": table, "confidence_k12": conf}
    if pos0["verifies"]:
        out["nvme_rows_routed_by_position_0"] = round(pos0["nvme_pos0"] / pos0["nvme"], 3)
    return out


def calibrate_window(verifies: list, grid=(1, 2, 4, 8, 16, 32, 64, 128, 256, 512)) -> dict:
    """Pick W so the recency tier best labels the routed, non-hot experts whose true tier the job trace gives."""
    res = {}
    for w in grid:
        tp = tn = fp = fn = 0
        for v in verifies:
            if v["nvme"] is None or v["history"] is None:
                continue
            routed = set(v["routes"].ravel().tolist()) - v["hot"]
            guess = recency_nvme(v["history"], w, v["hot"]) & routed
            tp += len(guess & v["nvme"])
            fp += len(guess - v["nvme"])
            fn += len((routed - guess) & v["nvme"])
            tn += len(routed - guess - v["nvme"])
        if tp + fn and tn + fp:
            res[w] = {"balanced_accuracy": round(0.5 * (tp / (tp + fn) + tn / (tn + fp)), 3),
                      "nvme_recall": round(tp / (tp + fn), 3), "ram_recall": round(tn / (tn + fp), 3)}
    if not res:
        return {"window": None}
    best = max(res, key=lambda w: res[w]["balanced_accuracy"])
    return {"window": best, "grid": res}


def lead_summary(paths: list, gens: set) -> dict:
    ev = []
    for p in paths:
        with open(p) as f:
            for line in f:
                if any(s in line for s in ('"draft_end"', '"draft_observed"', '"gate_open"', '"copy_submit"')):
                    ev.append(json.loads(line))
    ls = [v for g, v in leads(ev).items() if g in gens]
    out = {"verifies": len(ls)}
    for key in ("draft_end_ms", "draft_observed_ms", "prev_verify_end_ms"):
        vals = sorted(v[key] for v in ls if v[key] is not None and v[key] > 0)
        if vals:
            out[key] = {"p10": round(vals[len(vals) // 10], 3), "p50": round(statistics.median(vals), 3),
                        "p90": round(vals[(9 * len(vals)) // 10], 3), "n": len(vals)}
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("out_dir")
    p.add_argument("--model", default=MODEL)
    p.add_argument("--decode", help="schema-1 decode router capture with responses.jsonl (training only)")
    p.add_argument("--verify", action="append", default=[], help="NAME=DIR of a verify router capture")
    p.add_argument("--split", action="append", default=[], help="TRAIN,..>TEST,.. ('decode' names the decode capture)")
    a = p.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    model = Model(a.model)
    caps = {n: load_verify(n, d) for n, d in (s.split("=", 1) for s in a.verify)}
    dec = load_decode(a.decode) if a.decode else None
    xs = ([dec["x"]] if dec else []) + [v["x"] for n in caps for v in caps[n]]
    ids, rec_info = recover(model, xs)
    if dec:
        dec["tok"] = ids.pop(0)
        rec_info["decode_text_check"] = check_decode_ids(a.model, a.decode, dec, dec["tok"])
    for n in caps:
        for v in caps[n]:
            v["tok"] = ids.pop(0)
    result = {"recovery": rec_info, "captures": {n: len(v) for n, v in caps.items()}, "splits": {}}
    with_tier = [v for n in caps for v in caps[n] if v["nvme"] is not None]
    result["tier_window"] = calibrate_window(with_tier)
    window = result["tier_window"]["window"] or 64
    for spec in a.split:
        train, test = (s.split(",") for s in spec.split(">"))
        tt, tx, tr = [], [], []
        for n in train:
            if n == "decode":
                tt.append(dec["tok"]), tx.append(dec["x"]), tr.append(dec["routes"])
            else:
                for v in caps[n]:
                    tt.append(v["tok"]), tx.append(v["x"]), tr.append(v["routes"])
        pred = Predictors(model, np.concatenate(tt), np.concatenate(tx), np.concatenate(tr))
        res = {"fit": pred.fit_info}
        for n in test:
            vs = caps[n]
            seen = pred.table.seen(np.concatenate([v["tok"] for v in vs]))
            res[n] = {"test_tokens_seen_in_train": round(float((seen > 0).mean()), 3),
                      **evaluate(pred, vs, [v["tok"] for v in vs], window)}
        result["splits"][spec] = res
        print(spec, json.dumps(res["fit"]), flush=True)
    for n, vs in caps.items():
        paths = sorted(glob.glob(os.path.join(dict(s.split("=", 1) for s in a.verify)[n], "events.*.jsonl")))
        if paths:
            result.setdefault("leads", {})[n] = lead_summary(paths, {v["gen"] for v in vs if v["gen"] is not None})
            nv = [len(v["nvme"]) for v in vs if v["nvme"] is not None]
            result.setdefault("layer0_nvme_rows_per_verify", {})[n] = round(statistics.mean(nv), 2) if nv else None
    with open(os.path.join(a.out_dir, "layer0_draft_predict.json"), "w") as f:
        json.dump(result, f, indent=1)
    print(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
