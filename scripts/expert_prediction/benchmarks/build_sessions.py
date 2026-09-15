"""Build a session set for the MoE expert-capture benchmark pilot.

Samples multi-turn chat sessions from two sources:
  - ConvFinQA dev.json: multi-turn numeric QA over a financial report excerpt.
  - FinanceBench financebench_open_source.jsonl: single-question QA over a
    10-K/10-Q filing excerpt, split into a two-turn session (answer, then
    show-your-work follow-up).

Writes one JSON object per line to --out. Use --dry-run to preview per-session
turn counts and context sizes without writing anything.
"""

import argparse
import hashlib
import json
import os
import random
import subprocess
import urllib.request

CONVFINQA_PATH = "/mnt/nvme2/nvfp4-work/benchmarks/convfinqa/data/dev.json"
FINANCEBENCH_PATH = "/mnt/nvme2/nvfp4-work/benchmarks/financebench/financebench_open_source.jsonl"
FINANCEBENCH_PDF_CACHE = "/mnt/nvme2/nvfp4-work/benchmarks/financebench/pdfs"
FINANCEBENCH_PDF_URL = "https://raw.githubusercontent.com/patronus-ai/financebench/main/pdfs/{doc_name}.pdf"
FINANCEBENCH_MAX_CONTEXT_CHARS = 96_000
FINANCEBENCH_PAGE_MARGIN = 6

ANSWER_INSTRUCTION = (
    "Answer using the financial report excerpt below. Give the final numeric "
    "answer on the last line as `ANSWER: <number>`.\n\n"
)


def _render_table_markdown(table):
    if not table:
        return ""
    lines = []
    header = table[0]
    lines.append("| " + " | ".join(str(c) for c in header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for row in table[1:]:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)


def _load_convfinqa_records():
    with open(CONVFINQA_PATH) as f:
        return json.load(f)


def _load_financebench_records():
    records = []
    with open(FINANCEBENCH_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def build_convfinqa_sessions(n, seed):
    records = _load_convfinqa_records()
    eligible = []
    for r in records:
        questions = r.get("annotation", {}).get("dialogue_break") or []
        answers = r.get("annotation", {}).get("exe_ans_list") or []
        if 3 <= len(questions) <= 6 and len(answers) == len(questions):
            eligible.append(r)
    eligible.sort(key=lambda r: r["id"])
    rng = random.Random(seed)
    sampled_indices = sorted(rng.sample(range(len(eligible)), min(n, len(eligible))))
    sampled = [eligible[i] for i in sampled_indices]
    sampled.sort(key=lambda r: r["id"])

    n_train = int(len(sampled) * 0.8)
    sessions = []
    for idx, r in enumerate(sampled):
        split = "train" if idx < n_train else "val"
        questions = r["annotation"]["dialogue_break"]
        answers = r["annotation"]["exe_ans_list"]
        pre_text = "\n".join(r.get("pre_text") or [])
        post_text = "\n".join(r.get("post_text") or [])
        table_md = _render_table_markdown(r.get("table") or [])

        turns = []
        for qi, question in enumerate(questions):
            if qi == 0:
                content = (
                    ANSWER_INSTRUCTION
                    + pre_text
                    + "\n\n"
                    + table_md
                    + "\n\n"
                    + post_text
                    + "\n\nQuestion: "
                    + question
                )
            else:
                content = "Question: " + question + "\n\n" + ANSWER_INSTRUCTION.rstrip("\n")
            turns.append(content)

        context_chars = len(pre_text) + len(table_md) + len(post_text)
        sessions.append(
            {
                "session_id": f"cfq-{idx}",
                "source": r["id"],
                "domain": "convfinqa",
                "split": split,
                "turns": turns,
                "expected": list(answers),
                "context_chars": context_chars,
            }
        )
    return sessions


def _pdf_cache_path(doc_name):
    return os.path.join(FINANCEBENCH_PDF_CACHE, f"{doc_name}.pdf")


def _download_pdf(doc_name):
    path = _pdf_cache_path(doc_name)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    os.makedirs(FINANCEBENCH_PDF_CACHE, exist_ok=True)
    url = FINANCEBENCH_PDF_URL.format(doc_name=doc_name)
    tmp_path = path + ".tmp"
    with urllib.request.urlopen(url, timeout=120) as response:
        data = response.read()
    with open(tmp_path, "wb") as f:
        f.write(data)
    os.replace(tmp_path, path)
    return path


def _pdf_page_count(pdf_path):
    out = subprocess.run(
        ["pdfinfo", pdf_path], capture_output=True, text=True, check=True
    ).stdout
    for line in out.splitlines():
        if line.startswith("Pages:"):
            return int(line.split(":", 1)[1].strip())
    raise RuntimeError(f"could not determine page count for {pdf_path}")


def _extract_page_text(pdf_path, page):
    out = subprocess.run(
        ["pdftotext", "-f", str(page), "-l", str(page), "-layout", pdf_path, "-"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return out


def _build_financebench_context(pdf_path, evidence_pages, num_pages):
    wanted = set()
    for p in evidence_pages:
        for q in range(p - FINANCEBENCH_PAGE_MARGIN, p + FINANCEBENCH_PAGE_MARGIN + 1):
            if 1 <= q <= num_pages:
                wanted.add(q)
    ordered_pages = sorted(wanted)

    # Rank pages by distance to the nearest evidence page; drop farthest first
    # when the context exceeds the cap.
    def distance(p):
        return min(abs(p - e) for e in evidence_pages)

    page_texts = {p: _extract_page_text(pdf_path, p) for p in ordered_pages}
    kept = list(ordered_pages)
    while True:
        total_chars = sum(len(page_texts[p]) for p in kept)
        if total_chars <= FINANCEBENCH_MAX_CONTEXT_CHARS or len(kept) <= 1:
            break
        kept.sort(key=distance)
        kept.pop()  # drop the farthest page
        kept.sort()

    context = "\n\n".join(page_texts[p] for p in kept)
    if len(context) > FINANCEBENCH_MAX_CONTEXT_CHARS:
        context = context[:FINANCEBENCH_MAX_CONTEXT_CHARS]
    return context


def build_financebench_sessions(m, seed):
    records = _load_financebench_records()
    by_doc = {}
    for r in records:
        by_doc.setdefault(r["doc_name"], r)
    doc_names = sorted(by_doc.keys())
    rng = random.Random(seed)
    sampled_docs = rng.sample(doc_names, min(m, len(doc_names)))
    sampled_docs.sort()

    sessions = []
    for doc_name in sampled_docs:
        r = by_doc[doc_name]
        pdf_path = _download_pdf(doc_name)
        num_pages = _pdf_page_count(pdf_path)
        evidence_pages = sorted(
            {e["evidence_page_num"] for e in r.get("evidence") or [] if e.get("evidence_page_num")}
        )
        if not evidence_pages:
            evidence_pages = [1]
        context = _build_financebench_context(pdf_path, evidence_pages, num_pages)

        turn1 = (
            "Answer the question using the filing excerpt below.\n\n"
            + context
            + "\n\nQuestion: "
            + r["question"]
        )
        turn2 = "Show the calculation or the exact line items you used, and state any caveats."

        sessions.append(
            {
                "session_id": f"fb-{r['financebench_id']}",
                "source": doc_name,
                "domain": "financebench",
                "split": "holdout",
                "turns": [turn1, turn2],
                "expected": [r["answer"], None],
                "context_chars": len(context),
            }
        )
    return sessions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--convfinqa", type=int, default=0)
    parser.add_argument("--financebench", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    sessions = []
    if args.convfinqa:
        sessions.extend(build_convfinqa_sessions(args.convfinqa, args.seed))
    if args.financebench:
        sessions.extend(build_financebench_sessions(args.financebench, args.seed))

    if args.dry_run:
        for s in sessions:
            print(
                json.dumps(
                    {
                        "session_id": s["session_id"],
                        "domain": s["domain"],
                        "split": s["split"],
                        "turns": len(s["turns"]),
                        "context_chars": s["context_chars"],
                    }
                )
            )
        return

    with open(args.out, "w") as f:
        for s in sessions:
            f.write(json.dumps(s) + "\n")
    print(f"wrote {len(sessions)} sessions to {args.out}")


if __name__ == "__main__":
    main()
