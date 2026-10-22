"""Teacher-forced logprob dump for the DSV4.1 parity check (F145).

Usage:
  dump_logits.py --server-url http://127.0.0.1:7867 --out DIR [--server-pid PID] [--prompts prompts.jsonl]
  dump_logits.py --engine-model-path PATH --out DIR      (offline sglang.Engine; defaults, not the production recipe)

Each prompt row is tokenized (prompt, then the first `continuation_tokens` tokens of the continuation text) with the
served model's tokenizer, and prompt + continuation goes to /generate as `input_ids` with `return_logprob=True`,
`logprob_start_len=len(prompt_ids) - 1`, `top_logprobs_num=20` and `max_new_tokens=1`. The prefill returns, in
`meta_info`, `input_token_logprobs` and `input_top_logprobs`; the last N entries of each belong to the continuation
tokens (entry i is the distribution that predicts token i). The dump keeps those, K=20 best first.

Writes DIR/logits.npz and DIR/manifest.json (format: dump_format.py); compare.py reads two such directories.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

import dump_format

HERE = Path(__file__).resolve().parent
REQUEST_TIMEOUT_S = 3600
# Where chosen_lp and the top-K list disagree by more than this, the entries are misaligned.
ALIGNMENT_TOLERANCE = 1e-2


def load_prompts(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def tokenize_prompt(*, tokenizer, row: dict) -> tuple[list[int], list[int]]:
    prompt_ids = tokenizer.encode(row["prompt"], add_special_tokens=True)
    continuation_ids = tokenizer.encode(row["continuation"], add_special_tokens=False)
    count = row["continuation_tokens"]
    if len(continuation_ids) < count:
        raise ValueError(f"{row['id']}: continuation has {len(continuation_ids)} tokens, need {count}")
    return prompt_ids, continuation_ids[:count]


def teacher_forced_request(*, prompt_ids: list[int], continuation_ids: list[int]) -> dict:
    return {
        "input_ids": prompt_ids + continuation_ids,
        "sampling_params": {"temperature": 0.0, "max_new_tokens": 1},
        "return_logprob": True,
        "logprob_start_len": len(prompt_ids) - 1,
        "top_logprobs_num": dump_format.TOP_K,
        "return_text_in_logprobs": False,
    }


def extract_arrays(*, meta_info: dict, continuation_ids: list[int]) -> dict[str, np.ndarray]:
    """The continuation's rows from one /generate response; raises when the response is not aligned."""
    count = len(continuation_ids)
    token_rows = meta_info["input_token_logprobs"][-count:]
    top_rows = meta_info["input_top_logprobs"][-count:]
    returned_ids = [row[1] for row in token_rows]
    if returned_ids != continuation_ids:
        raise ValueError("input_token_logprobs ids do not match the teacher-forced continuation")
    chosen = np.array([row[0] for row in token_rows], dtype=np.float32)
    top_ids = np.full((count, dump_format.TOP_K), -1, dtype=np.int32)
    top_lp = np.full((count, dump_format.TOP_K), -np.inf, dtype=np.float32)
    for j, entries in enumerate(top_rows):
        ranked = sorted(entries, key=lambda entry: -entry[0])[: dump_format.TOP_K]
        top_ids[j, : len(ranked)] = [entry[1] for entry in ranked]
        top_lp[j, : len(ranked)] = [entry[0] for entry in ranked]
    check_alignment(tokens=continuation_ids, chosen=chosen, top_ids=top_ids, top_lp=top_lp)
    return {
        "tokens": np.array(continuation_ids, dtype=np.int32),
        "chosen_lp": chosen,
        "top_ids": top_ids,
        "top_lp": top_lp,
    }


def check_alignment(*, tokens: list[int], chosen: np.ndarray, top_ids: np.ndarray, top_lp: np.ndarray) -> None:
    """A token inside its own top-K list must carry the same logprob as chosen_lp."""
    checked = 0
    for j, token in enumerate(tokens):
        hit = np.nonzero(top_ids[j] == token)[0]
        if hit.size:
            checked += 1
            if abs(float(top_lp[j, hit[0]]) - float(chosen[j])) > ALIGNMENT_TOLERANCE:
                raise ValueError(f"position {j}: chosen logprob and top-K entry disagree; rows are shifted")
    if checked == 0 and len(tokens) > 8:
        raise ValueError("no continuation token appears in its own top-K list; rows are shifted")


def server_get(url: str, path: str):
    import requests

    return requests.get(url + path, timeout=60).json()


def post_generate(url: str, body: dict) -> dict:
    import requests

    requests.post(url + "/flush_cache", timeout=60)
    response = requests.post(url + "/generate", json=body, timeout=REQUEST_TIMEOUT_S)
    response.raise_for_status()
    return response.json()


def server_request_fn(url: str):
    return lambda body: post_generate(url, body)["meta_info"]


def engine_request_fn(model_path: str):
    import sglang

    engine = sglang.Engine(model_path=model_path, skip_server_warmup=True)

    def request(body: dict) -> dict:
        engine.flush_cache()
        keys = ("sampling_params", "return_logprob", "logprob_start_len", "top_logprobs_num", "return_text_in_logprobs")
        return engine.generate(input_ids=body["input_ids"], **{key: body[key] for key in keys})["meta_info"]

    return request


def git_state() -> dict:
    root = HERE.parents[2]

    def run(*args: str) -> str:
        return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True).stdout.strip()

    return {"git_commit": run("rev-parse", "HEAD"), "git_dirty": bool(run("status", "--porcelain", "--untracked=no"))}


def read_proc_environ(pid: int) -> dict[str, str]:
    raw = Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
    return dict(item.decode().split("=", 1) for item in raw if b"=" in item)


def sglang_file() -> str | None:
    try:
        import sglang
    except ImportError:
        return None
    return sglang.__file__


def build_manifest(*, args: argparse.Namespace, prompts_path: Path, server_info: dict | None) -> dict:
    server_env = read_proc_environ(args.server_pid) if args.server_pid else dict(os.environ)
    manifest = {
        **git_state(),
        "created_unix": int(time.time()),
        "prompt_set_sha256": dump_format.file_sha256(prompts_path),
        "sglang_file": sglang_file(),
        "sglang_env": {k: v for k, v in sorted(server_env.items()) if k.startswith("SGLANG_")},
        "env_source": f"/proc/{args.server_pid}/environ" if args.server_pid else "this process (not the server's)",
        "server_pythonpath": server_env.get("PYTHONPATH"),
        "server_url": args.server_url,
        "model_path": (server_info or {}).get("model_path") or args.engine_model_path or args.model_path,
        "top_k": dump_format.TOP_K,
    }
    return manifest


def dump(*, rows: list[dict], tokenizer, request) -> dict[str, dict[str, np.ndarray]]:
    result = {}
    for row in rows:
        prompt_ids, continuation_ids = tokenize_prompt(tokenizer=tokenizer, row=row)
        started = time.time()
        meta_info = request(teacher_forced_request(prompt_ids=prompt_ids, continuation_ids=continuation_ids))
        result[row["id"]] = extract_arrays(meta_info=meta_info, continuation_ids=continuation_ids)
        print(f"{row['id']}: {len(prompt_ids)}+{len(continuation_ids)} tokens in {time.time() - started:.1f}s", flush=True)
    return result


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--server-url")
    source.add_argument("--engine-model-path")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, default=HERE / "prompts.jsonl")
    parser.add_argument("--server-pid", type=int, help="server process whose SGLANG_* env goes in the manifest")
    parser.add_argument("--model-path", help="tokenizer/model path; defaults to the server's /server_info model_path")
    parser.add_argument("--limit", type=int, help="first N prompts only (smoke runs; the manifest still hashes the full set)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    from transformers import AutoTokenizer

    args = parse_args(argv)
    server_info = server_get(args.server_url, "/server_info") if args.server_url else None
    model_path = args.model_path or args.engine_model_path or server_info["model_path"]
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    request = server_request_fn(args.server_url) if args.server_url else engine_request_fn(args.engine_model_path)
    rows = load_prompts(args.prompts)[: args.limit]
    arrays = dump(rows=rows, tokenizer=tokenizer, request=request)
    manifest = build_manifest(args=args, prompts_path=args.prompts, server_info=server_info)
    manifest["model_path"] = model_path
    manifest["prompt_ids"] = [row["id"] for row in rows]
    dump_format.write_dump(out_dir=args.out, manifest=manifest, prompts=arrays)
    print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
