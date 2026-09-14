"""Accuracy harness for online FP8 of the BF16 GPU weights (BF16 server vs FP8 server).

Subcommands, each talking to one running SGLang server or comparing saved results:

* ``generate``: greedy replies to the fixed prompt set; saves prompt and output token ids.
* ``support``: on the BF16 server, teacher-forced top-K ids per scored position; their union
  per sequence is the token support that ``score`` asks every server for.
* ``score``: teacher-forced logprobs of the support ids plus the full-vocab top-2 at every
  scored position of every generated sequence.
* ``compare-logits``: per-position KL(BF16 || test) over the support plus one tail bucket (a
  lower bound on the full-vocabulary KL), top-1 agreement, and gate verdicts.
* ``compare-greedy``: greedy-match length between two ``generate`` runs.
* ``gsm8k`` / ``compare-gsm8k``: fixed GSM8K subset, paired accuracy and exact McNemar test.

Only the standard library, numpy and transformers (tokenizer) are needed; nothing touches the GPU.
"""

import argparse
import json
import math
import re
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

KL_MEAN_GATE = 0.005
TOP1_GATE = 0.977
TOP1_NOISE_MARGIN = 0.01
GREEDY_MEDIAN_FRACTION = 0.5
GSM8K_MAX_DROP = 0.02
GSM8K_MCNEMAR_ALPHA = 0.05
SCORED_PROMPT_TAIL = 64


def _records_text() -> str:
    return "\n".join(
        f"Record {i}: the warehouse in district {i} shipped {i * 37 % 101} crates of copper wire, "
        f"{i * 53 % 97} pallets of glass, and {i * 11 % 89} barrels of oil on day {i} of the quarter."
        for i in range(1, 61)
    )


def prompt_set() -> list:
    """Stage 3 workload (count deduplicated), E1 route-trace workload, then 12 broader prompts."""
    records = _records_text()
    return [
        ("s3_count", "Count from 1 to 40 separated by spaces."),
        ("s3_long_prompt", f"{records}\n\nSummarize the records above in one short sentence."),
        ("s3_code", "Write a Python function that parses an ISO-8601 duration string such as P3DT4H12M into total seconds, with a short docstring and three doctest examples."),
        ("s3_explain", "Explain in about 300 words why the sky looks blue during the day and red at sunset."),
        ("s3_math", "A train leaves at 09:40 travelling 72 km/h; a second leaves the same station at 10:05 travelling 90 km/h on the same track. When and where does the second catch the first? Show the steps."),
        ("s3_translate", "Translate into French: The committee postponed the vote until the auditors confirm last quarter's figures."),
        ("e1_code", "Write a Python module implementing an LRU cache with a maximum byte budget, per-entry sizes, thread safety, and an eviction callback. Include type hints, docstrings and a short pytest test file."),
        ("e1_summary", f"{records}\n\nSummarize the records above: identify the districts with the highest and lowest shipments of each good and describe any pattern."),
        ("e1_math", "A tank is filled by two pipes and drained by a third. Pipe A fills it in 6 hours, pipe B in 9 hours, and pipe C drains it in 12 hours. Starting empty with all three open, after 2 hours pipe B is closed. How long in total until the tank is full? Show each step."),
        ("e1_chat", "I'm planning a week-long cycling trip through the Loire Valley in late September. Suggest a daily route, where to stay, what to pack, and how to handle bike repairs on the road."),
        ("e1_translate", "Translate into French and then German, preserving formatting: 'Maintenance window: the database cluster will be read-only from 02:00 to 03:30 UTC on Saturday. Writes during this window will fail with error 503; clients should retry with exponential backoff.'"),
        ("e1_chat2", "Explain to a new engineer how a hash map handles collisions, resizing and iteration order, with small examples."),
        ("x_network", "Explain the difference between TCP and UDP to a student, with two concrete examples of each."),
        ("x_chinese", "请用中文简要介绍长城的历史，并列出三个著名的段落。"),
        ("x_json", "Return a JSON object describing a fictional book with fields title, author, year, genres (array) and a one-sentence summary. Output only JSON."),
        ("x_debug", "Find and fix the bug in this Python function and explain the fix:\n\ndef mean(xs):\n    total = 0\n    for x in xs:\n        total += x\n    return total / len(xs) if xs else total / 0"),
        ("x_logic", "Alice is older than Bob. Carol is younger than Bob but older than Dave. Who is the second youngest? Explain briefly."),
        ("x_arith", "Compute 487 * 369 step by step, then verify the result with a different method."),
        ("x_poem", "Write a 12-line poem about a lighthouse keeper in winter, with an ABAB rhyme scheme."),
        ("x_sql", "Write a SQL query that returns, for each customer, their total order value in 2024 and their rank by that value, using tables customers(id, name) and orders(id, customer_id, amount, created_at)."),
        ("x_rust", "Write a Rust function that reverses the words in a string slice without allocating more than one String, and add a unit test."),
        ("x_german", "Fasse die Vor- und Nachteile von Kernenergie in fünf Stichpunkten zusammen."),
        ("x_mrna", "Describe how mRNA vaccines work, from injection to immune response, in about 200 words."),
        ("x_postgres", "List the steps to migrate a PostgreSQL database from version 13 to 16 with minimal downtime, including rollback considerations."),
    ]


class Chat:
    """Applies the model's chat template (thinking off) to produce prompt token ids."""

    def __init__(self, model_path: str):
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    def ids(self, content: str) -> list:
        text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        return list(self.tokenizer(text, add_special_tokens=False)["input_ids"])


def post_generate(base: str, payload: dict, timeout: float = 3600.0) -> dict:
    request = urllib.request.Request(
        f"{base}/generate",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def output_ids(result: dict) -> list:
    if result.get("output_ids") is not None:
        return list(result["output_ids"])
    return [int(entry[1]) for entry in result["meta_info"]["output_token_logprobs"]]


def read_jsonl(path: str) -> list:
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: str, value) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def cmd_generate(args) -> int:
    chat = Chat(args.model_path)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as handle:
        for name, content in prompt_set():
            prompt = chat.ids(content)
            start = time.monotonic()
            result = post_generate(
                args.base,
                {
                    "input_ids": prompt,
                    "sampling_params": {"temperature": 0, "max_new_tokens": args.max_new_tokens},
                    "return_logprob": True,
                },
            )
            generated = output_ids(result)
            if not generated:
                print(f"EMPTY_REPLY[{name}]", file=sys.stderr)
                return 1
            record = {
                "name": name,
                "prompt_ids": prompt,
                "output_ids": generated,
                "text": result.get("text", ""),
                "finish_reason": result["meta_info"].get("finish_reason"),
                "elapsed_s": round(time.monotonic() - start, 3),
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"{name}: prompt={len(prompt)} output={len(generated)} {record['elapsed_s']}s")
    return 0


def _score_request(base: str, record: dict, top_k: int, support=None) -> dict:
    ids = record["prompt_ids"] + record["output_ids"]
    start = max(1, len(record["prompt_ids"]) - SCORED_PROMPT_TAIL - 1)
    payload = {
        "input_ids": ids,
        "sampling_params": {"temperature": 1.0, "max_new_tokens": 1},
        "return_logprob": True,
        "logprob_start_len": start,
        "top_logprobs_num": top_k,
    }
    if support is not None:
        payload["token_ids_logprob"] = support
    result = post_generate(base, payload)
    return {"start": start, "meta": result["meta_info"]}


def _positions(entries) -> list:
    """Scored rows without the scheduler's leading ``None`` placeholder; any other empty row is an error."""
    if entries and entries[0] is None:
        entries = entries[1:]
    if any(not entry for entry in entries):
        raise RuntimeError("input logprobs contain an empty row after the leading placeholder")
    return list(entries)


def cmd_support(args) -> int:
    support = {}
    for record in read_jsonl(args.gen):
        scored = _score_request(args.base, record, args.top_k)
        top = _positions(scored["meta"]["input_top_logprobs"])
        ids = sorted({int(entry[1]) for position in top for entry in position})
        support[record["name"]] = {"start": scored["start"], "positions": len(top), "ids": ids}
        print(f"{record['name']}: positions={len(top)} support={len(ids)}")
    write_json(args.out, support)
    return 0


def cmd_score(args) -> int:
    support = json.loads(Path(args.support).read_text())
    arrays = {}
    for record in read_jsonl(args.gen):
        entry = support[record["name"]]
        scored = _score_request(args.base, record, 2, entry["ids"])
        if scored["start"] != entry["start"]:
            raise RuntimeError(f"{record['name']}: start {scored['start']} != {entry['start']}")
        position_to_col = {token: col for col, token in enumerate(entry["ids"])}
        rows = _positions(scored["meta"]["input_token_ids_logprobs"])
        top = _positions(scored["meta"]["input_top_logprobs"])
        if len(rows) != len(top):
            raise RuntimeError(f"{record['name']}: {len(rows)} support rows vs {len(top)} top rows")
        logprobs = np.full((len(rows), len(entry["ids"])), np.nan, dtype=np.float64)
        top2 = np.full((len(rows), 2, 2), np.nan, dtype=np.float64)
        for i, (row, best) in enumerate(zip(rows, top)):
            for value, token, _ in row:
                logprobs[i, position_to_col[int(token)]] = value
            for j, (value, token, _) in enumerate(best[:2]):
                top2[i, j] = (value, token)
        if np.isnan(logprobs).all(axis=1).any():
            raise RuntimeError(f"{record['name']}: a scored position returned no support logprobs")
        arrays[f"{record['name']}/logprobs"] = logprobs
        arrays[f"{record['name']}/top2"] = top2
        print(f"{record['name']}: positions={len(rows)}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **arrays)
    return 0


def position_metrics(ref_logprobs, test_logprobs, ref_top2, test_top2) -> dict:
    """KL(ref || test) over support plus a tail bucket, top-1 agreement, ref top-2 gap."""
    p = np.exp(np.nan_to_num(ref_logprobs, nan=-np.inf))
    q = np.exp(np.nan_to_num(test_logprobs, nan=-np.inf))
    tail_p = np.clip(1.0 - p.sum(axis=1), 0.0, 1.0)
    tail_q = np.clip(1.0 - q.sum(axis=1), 0.0, 1.0)
    floor = 1e-30
    with np.errstate(divide="ignore", invalid="ignore"):
        body = np.where(p > 0, p * (np.log(np.maximum(p, floor)) - np.log(np.maximum(q, floor))), 0.0)
        tail = np.where(
            tail_p > 0,
            tail_p * (np.log(np.maximum(tail_p, floor)) - np.log(np.maximum(tail_q, floor))),
            0.0,
        )
    kl = np.maximum(body.sum(axis=1) + tail, 0.0)
    agree = ref_top2[:, 0, 1] == test_top2[:, 0, 1]
    gap = np.exp(ref_top2[:, 0, 0]) - np.exp(ref_top2[:, 1, 0])
    return {"kl": kl, "agree": agree, "tail_mass": tail_p, "ref_top2_prob_gap": gap}


def cmd_compare_logits(args) -> int:
    ref = np.load(args.ref)
    test = np.load(args.test)
    names = sorted({key.split("/")[0] for key in ref.files})
    per_request = {}
    kl_all, agree_all, tail_all, gap_all = [], [], [], []
    for name in names:
        if ref[f"{name}/logprobs"].shape != test[f"{name}/logprobs"].shape:
            raise RuntimeError(f"{name}: shape mismatch, not the same sequences or support")
        metrics = position_metrics(
            ref[f"{name}/logprobs"], test[f"{name}/logprobs"], ref[f"{name}/top2"], test[f"{name}/top2"]
        )
        per_request[name] = {
            "positions": int(metrics["kl"].size),
            "kl_mean": float(metrics["kl"].mean()),
            "kl_max": float(metrics["kl"].max()),
            "top1_agreement": float(metrics["agree"].mean()),
        }
        kl_all.append(metrics["kl"])
        agree_all.append(metrics["agree"])
        tail_all.append(metrics["tail_mass"])
        gap_all.append(metrics["ref_top2_prob_gap"])
    kl = np.concatenate(kl_all)
    agree = np.concatenate(agree_all)
    tail = np.concatenate(tail_all)
    gap = np.concatenate(gap_all)
    flips = ~agree
    pinsker = 2.0 * np.sqrt(kl / 2.0)
    summary = {
        "ref": args.ref,
        "test": args.test,
        "positions": int(kl.size),
        "kl_mean": float(kl.mean()),
        "kl_p50": float(np.percentile(kl, 50)),
        "kl_p95": float(np.percentile(kl, 95)),
        "kl_p99": float(np.percentile(kl, 99)),
        "kl_max": float(kl.max()),
        "top1_agreement": float(agree.mean()),
        "support_tail_mass_mean": float(tail.mean()),
        "support_tail_mass_p99": float(np.percentile(tail, 99)),
        "flips": int(flips.sum()),
        "flips_with_ref_gap_over_pinsker_of_kl_lower_bound": int((flips & (gap > pinsker)).sum()),
        "flip_ref_gap_median": float(np.median(gap[flips])) if flips.any() else None,
    }
    noise_top1 = None
    if args.noise:
        noise_top1 = json.loads(Path(args.noise).read_text())["top1_agreement"]
    top1_floor = TOP1_GATE if noise_top1 is None else min(TOP1_GATE, noise_top1 - TOP1_NOISE_MARGIN)
    summary["gates"] = {
        "kl_mean": {"limit": KL_MEAN_GATE, "pass": summary["kl_mean"] <= KL_MEAN_GATE},
        "top1_agreement": {"limit": top1_floor, "pass": summary["top1_agreement"] >= top1_floor},
    }
    summary["pass"] = all(gate["pass"] for gate in summary["gates"].values())
    summary["per_request"] = per_request
    write_json(args.out, summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "per_request"}, indent=2))
    return 0


def cmd_compare_greedy(args) -> int:
    ref = {record["name"]: record for record in read_jsonl(args.ref)}
    test = {record["name"]: record for record in read_jsonl(args.test)}
    rows = {}
    for name, record in ref.items():
        a, b = record["output_ids"], test[name]["output_ids"]
        match = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
        rows[name] = {
            "match_tokens": match,
            "ref_tokens": len(a),
            "test_tokens": len(b),
            "identical": a == b,
        }
    lengths = np.array([row["match_tokens"] for row in rows.values()])
    summary = {
        "ref": args.ref,
        "test": args.test,
        "prompts": len(rows),
        "identical": sum(row["identical"] for row in rows.values()),
        "match_tokens_median": float(np.median(lengths)),
        "match_tokens_mean": float(lengths.mean()),
        "per_prompt": rows,
    }
    if args.noise:
        noise = json.loads(Path(args.noise).read_text())
        limit = GREEDY_MEDIAN_FRACTION * noise["match_tokens_median"]
        summary["gate"] = {"median_limit": limit, "pass": summary["match_tokens_median"] >= limit}
    write_json(args.out, summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "per_prompt"}, indent=2))
    return 0


def gsm8k_subset(path: str, count: int) -> list:
    questions = read_jsonl(path)
    step = len(questions) / count
    return [(int(i * step), questions[int(i * step)]) for i in range(count)]


_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def parse_number(text: str):
    answer = re.findall(r"Answer:\s*\$?(-?\d[\d,]*(?:\.\d+)?)", text)
    candidates = answer or _NUMBER.findall(text)
    if not candidates:
        return None
    try:
        return float(candidates[-1].replace(",", ""))
    except ValueError:
        return None


def cmd_gsm8k(args) -> int:
    chat = Chat(args.model_path)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as handle:
        for index, item in gsm8k_subset(args.data, args.count):
            content = (
                f"{item['question']}\n\nSolve the problem step by step, then write the final "
                "answer on the last line as: Answer: <number>"
            )
            result = post_generate(
                args.base,
                {
                    "input_ids": chat.ids(content),
                    "sampling_params": {"temperature": 0, "max_new_tokens": args.max_new_tokens},
                },
            )
            gold = float(item["answer"].split("####")[-1].strip().replace(",", ""))
            predicted = parse_number(result.get("text", ""))
            correct = predicted is not None and math.isclose(predicted, gold, rel_tol=0, abs_tol=1e-6)
            handle.write(
                json.dumps(
                    {
                        "index": index,
                        "gold": gold,
                        "predicted": predicted,
                        "correct": correct,
                        "completion_tokens": result["meta_info"].get("completion_tokens"),
                        "finish_reason": result["meta_info"].get("finish_reason"),
                        "text": result.get("text", ""),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            handle.flush()
    return 0


def mcnemar_exact_p(ref_only: int, test_only: int) -> float:
    n = ref_only + test_only
    if n == 0:
        return 1.0
    k = min(ref_only, test_only)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2.0 * tail)


def cmd_compare_gsm8k(args) -> int:
    ref = {row["index"]: row for row in read_jsonl(args.ref)}
    test = {row["index"]: row for row in read_jsonl(args.test)}
    if ref.keys() != test.keys():
        raise RuntimeError("GSM8K runs cover different questions")
    ref_only = sum(ref[i]["correct"] and not test[i]["correct"] for i in ref)
    test_only = sum(test[i]["correct"] and not ref[i]["correct"] for i in ref)
    ref_acc = sum(row["correct"] for row in ref.values()) / len(ref)
    test_acc = sum(row["correct"] for row in test.values()) / len(test)
    p_value = mcnemar_exact_p(ref_only, test_only)
    drop = ref_acc - test_acc
    significant_drop = ref_only > test_only and p_value < GSM8K_MCNEMAR_ALPHA
    summary = {
        "ref": args.ref,
        "test": args.test,
        "questions": len(ref),
        "ref_accuracy": ref_acc,
        "test_accuracy": test_acc,
        "ref_only_correct": ref_only,
        "test_only_correct": test_only,
        "mcnemar_exact_p": p_value,
        "truncated_ref": sum(row["finish_reason"] and row["finish_reason"].get("type") == "length" for row in ref.values()),
        "truncated_test": sum(row["finish_reason"] and row["finish_reason"].get("type") == "length" for row in test.values()),
        "gate": {
            "max_drop": GSM8K_MAX_DROP,
            "pass": drop <= GSM8K_MAX_DROP and not significant_drop,
        },
    }
    write_json(args.out, summary)
    print(json.dumps(summary, indent=2))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    generate = sub.add_parser("generate")
    generate.add_argument("--base", required=True)
    generate.add_argument("--model-path", required=True)
    generate.add_argument("--out", required=True)
    generate.add_argument("--max-new-tokens", type=int, default=256)
    generate.set_defaults(func=cmd_generate)

    support = sub.add_parser("support")
    support.add_argument("--base", required=True)
    support.add_argument("--gen", required=True)
    support.add_argument("--out", required=True)
    support.add_argument("--top-k", type=int, default=20)
    support.set_defaults(func=cmd_support)

    score = sub.add_parser("score")
    score.add_argument("--base", required=True)
    score.add_argument("--gen", required=True)
    score.add_argument("--support", required=True)
    score.add_argument("--out", required=True)
    score.set_defaults(func=cmd_score)

    compare_logits = sub.add_parser("compare-logits")
    compare_logits.add_argument("--ref", required=True)
    compare_logits.add_argument("--test", required=True)
    compare_logits.add_argument("--noise", help="compare-logits JSON of BF16 vs BF16")
    compare_logits.add_argument("--out", required=True)
    compare_logits.set_defaults(func=cmd_compare_logits)

    compare_greedy = sub.add_parser("compare-greedy")
    compare_greedy.add_argument("--ref", required=True)
    compare_greedy.add_argument("--test", required=True)
    compare_greedy.add_argument("--noise", help="compare-greedy JSON of BF16 vs BF16")
    compare_greedy.add_argument("--out", required=True)
    compare_greedy.set_defaults(func=cmd_compare_greedy)

    gsm8k = sub.add_parser("gsm8k")
    gsm8k.add_argument("--base", required=True)
    gsm8k.add_argument("--model-path", required=True)
    gsm8k.add_argument("--data", required=True)
    gsm8k.add_argument("--out", required=True)
    gsm8k.add_argument("--count", type=int, default=200)
    gsm8k.add_argument("--max-new-tokens", type=int, default=512)
    gsm8k.set_defaults(func=cmd_gsm8k)

    compare_gsm8k = sub.add_parser("compare-gsm8k")
    compare_gsm8k.add_argument("--ref", required=True)
    compare_gsm8k.add_argument("--test", required=True)
    compare_gsm8k.add_argument("--out", required=True)
    compare_gsm8k.set_defaults(func=cmd_compare_gsm8k)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
