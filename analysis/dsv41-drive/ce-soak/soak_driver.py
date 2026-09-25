#!/usr/bin/env python3
"""Copy-engine soak driver: a long, seeded, varied request stream against one DSV4.1 EXL3 server.

The copy engine (LEASE_PROTOCOL.md 7.6) fail-stops the server when a driver call that takes the context lock and
waits for the device runs while an armed decode step spins in its copy wait: a kernel module loaded for the first
time, ``empty_cache``, host registration. This driver covers the request features whose GPU paths a warm-up does not
take (sampling variants, penalties, logit bias, logprobs, grammars, n>1, stop handling, long prompts, streaming,
queued bursts, client aborts) and records, per request, what came back and how fast. It does not decide anything
about the server; soak_report.py reads its output.

    soak_driver.py --port P --out DIR --seed S --tokenizer DIR [--max-requests N] [--max-minutes M]
                   [--server-pid PID] [--only-kinds k1,k2] [--plan-only]

Writes DIR/plan.jsonl (the seeded plan, before anything runs), DIR/requests.jsonl (one record per request, as it
finishes), DIR/captures/ (py-spy and nvidia-smi taken while a stream is stalled) and DIR/driver_summary.json.
Exit status: 0 the plan (or its time cap) ran, 3 the server died, 2 usage.

Stdlib plus ``tokenizers`` only. CPU only; run under taskset with capped threads.
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import random
import re
import subprocess
import sys
import threading
import time

# Client inter-token gap that triggers an evidence capture (py-spy of the scheduler, nvidia-smi) while the stream is
# still stalled. The RAM-miss deadline is 2 s, so a capture has to start well before it to see the hang.
CAPTURE_GAP_S = 0.7
STALL_S = (0.5, 2.0)
REQUEST_TIMEOUT_S = 1800
CONTEXT = 32768

DETERMINISM_PROMPT = "List the first ten prime numbers, then explain in two sentences why 1 is not prime."
DETERMINISM_EVERY = 20

TOPICS = [
    "Summarise the main drivers of revenue growth for a regional bank over a decade, with figures.",
    "Explain how a refinery's crack spread affects its quarterly earnings, step by step.",
    "Write a short Python function that merges two sorted lists, then explain its complexity.",
    "Describe the water cycle to a ten-year-old.",
    "Compare TCP and QUIC congestion control in a short table.",
    "Write a haiku about a lighthouse in winter.",
    "What are the trade-offs of write-ahead logging versus shadow paging?",
    "Give three arguments for and against congestion pricing in large cities.",
    "Explain the difference between a mutex and a semaphore with an example in C.",
    "Draft a polite email declining a meeting invitation because of a scheduling conflict.",
    "Why does ice float on water? Answer in one paragraph.",
    "Outline a one-week study plan for learning linear algebra.",
    "Translate 'The early bird catches the worm' into French, German and Spanish, and explain each.",
    "Write a SQL query that finds the second-highest salary per department.",
    "Tell a very short story about a robot that learns to paint.",
    "What is the time complexity of Dijkstra's algorithm with a binary heap, and why?",
    "Explain how a CUDA stream differs from a CUDA graph.",
    "List five common logical fallacies with one-line examples.",
    "Describe how vaccines train the immune system.",
    "Write a Rust function that reverses the words in a string.",
    "Explain what a yield curve inversion signals and its historical record.",
    "Give a recipe for a simple tomato soup.",
    "Summarise the plot of Hamlet in five sentences.",
    "What is the difference between precision and recall?",
]
TINY = ["Hi", "2+2?", "Name a colour.", "Yes or no: is the sky blue?", "Continue: 1, 1, 2, 3, 5,", "Say hello.", "Why?"]
SYSTEMS = [
    "You are a terse assistant. Answer in as few words as possible.",
    "You are a helpful assistant who always answers in bullet points.",
    "You are a pirate. Stay in character.",
    "You answer only in formal English and never use contractions.",
    "You are an expert financial analyst. Cite numbers when you can.",
]
WORDS = (
    "account ledger river mountain signal cache engine vector matrix harbor lantern orbit quartz meadow copper "
    "falcon ember glacier velvet summit canyon thistle marble beacon cobalt prairie willow nickel cedar tundra "
    "saffron pewter granite ripple compass anchor bramble cinder dune fjord grove heron ivory jasper kelp lagoon"
).split()

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}, "unit": {"type": "string", "enum": ["c", "f"]}},
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "convert_currency",
            "description": "Convert an amount between currencies.",
            "parameters": {
                "type": "object",
                "properties": {"amount": {"type": "number"}, "from": {"type": "string"}, "to": {"type": "string"}},
                "required": ["amount", "from", "to"],
            },
        },
    },
]

JSON_SCHEMAS = [
    {
        "name": "person",
        "schema": {
            "type": "object",
            "properties": {"name": {"type": "string"}, "age": {"type": "integer"}, "city": {"type": "string"}},
            "required": ["name", "age", "city"],
            "additionalProperties": False,
        },
        "prompt": "Invent a fictional person and describe them as JSON with name, age and city.",
    },
    {
        "name": "tool_call",
        "schema": {
            "type": "object",
            "properties": {
                "tool": {"type": "string", "enum": ["get_weather", "convert_currency"]},
                "arguments": {"type": "object"},
            },
            "required": ["tool", "arguments"],
        },
        "prompt": "I want to know how much 250 euros is in yen. Reply with the tool call to make, as JSON.",
    },
    {
        "name": "inventory",
        "schema": {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"sku": {"type": "string"}, "qty": {"type": "integer", "minimum": 0}},
                        "required": ["sku", "qty"],
                    },
                    "minItems": 1,
                    "maxItems": 4,
                }
            },
            "required": ["items"],
        },
        "prompt": "Produce a small warehouse inventory of up to four items as JSON.",
    },
]
REGEXES = [
    (r"(yes|no), because [a-z ]{5,60}\.", "Is Paris the capital of France? Answer yes or no, then give a reason."),
    (r"[0-9]{3}-[0-9]{3}-[0-9]{4}", "Make up a US phone number."),
    (r"(19|20)[0-9]{2}-(0[1-9]|1[0-2])-(0[1-9]|[12][0-9])", "Give a plausible date for a historical event, ISO format."),
    (r"\{\"answer\": [0-9]{1,4}\}", "What is 17 times 23? Reply as {\"answer\": N}."),
]
EBNFS = [
    (
        'root ::= "The answer is " choice "."\nchoice ::= "yes" | "no" | "maybe"',
        r"The answer is (yes|no|maybe)\.",
        "Will it rain tomorrow in London?",
    ),
    (
        'root ::= item ("," " " item)*\nitem ::= [a-z]+',
        r"[a-z]+(, [a-z]+)*",
        "List some fruits, lowercase, comma separated.",
    ),
]


class Tok:
    """Token counts and filler of a chosen token length, from the model's own tokenizer.json."""

    def __init__(self, path: str):
        from tokenizers import Tokenizer

        self.t = Tokenizer.from_file(os.path.join(path, "tokenizer.json"))

    def count(self, text: str) -> int:
        return len(self.t.encode(text, add_special_tokens=False).ids)

    def single_id(self, text: str):
        ids = self.t.encode(text, add_special_tokens=False).ids
        return ids[0] if len(ids) == 1 else None

    def filler(self, rng: random.Random, tokens: int) -> str:
        """Deterministic varied text of about ``tokens`` tokens: ledger lines, prose and code, mixed."""
        parts, have = [], 0
        k = rng.randrange(10**6)
        while have < tokens:
            style = rng.randrange(3)
            if style == 0:
                line = (
                    f"Ledger line {k}: account {k * 7919 % 10007} moved {k * 104729 % 99991} units to account "
                    f"{k * 15485863 % 10009} on day {k % 365}, reference {k * 32452843 % 1000003}."
                )
            elif style == 1:
                line = " ".join(rng.choice(WORDS) for _ in range(rng.randint(8, 24))).capitalize() + "."
            else:
                line = f"def f{k}(x):\n    return (x * {k % 97} + {k % 13}) % {k % 89 + 2}\n"
            parts.append(line)
            have += self.count(line) + 1
            k += 1
        text = "\n".join(parts)
        ids = self.t.encode(text, add_special_tokens=False).ids[:tokens]
        return self.t.decode(ids)


class Plan:
    """The seeded request plan: items run in order; an item is one request, a burst, or a conversation."""

    def __init__(self, rng: random.Random, tok: Tok, n_requests: int, only_kinds=None):
        self.rng, self.tok = rng, tok
        self.stop_ids = [i for i in (tok.single_id("\n\n"), tok.single_id("."), tok.single_id(",")) if i is not None]
        self.bias_ids = [i for i in (tok.single_id(" the"), tok.single_id(" and"), tok.single_id("The")) if i is not None]
        kinds = {
            "greedy": 10, "sampled": 22, "penalties": 8, "logit_bias": 5, "logprobs": 8, "seeded": 4,
            "stop": 6, "max1": 3, "json_schema": 6, "regex": 4, "ebnf": 3, "tools": 3, "system": 5,
            "conversation": 4, "long": 5, "burst": 5, "abort": 4, "n2": 1, "completions": 3, "generate": 3,
        }
        if only_kinds:
            kinds = {k: v for k, v in kinds.items() if k in only_kinds}
        self.kinds, self.weights = list(kinds), list(kinds.values())
        self.items = []
        count, next_control = 0, 0
        while count < n_requests:
            if count >= next_control:
                self.items.append({"kind": "determinism", "requests": [self.determinism()]})
                count += 1
                next_control += DETERMINISM_EVERY
            kind = rng.choices(self.kinds, self.weights)[0]
            item = {"kind": kind, "requests": getattr(self, "make_" + kind)()}
            self.items.append(item)
            count += sum(len(r.get("burst") or r.get("conversation") or [r]) for r in item["requests"])

    # --- pieces ---------------------------------------------------------------------------------------------------
    def max_tokens(self):
        r = self.rng.random()
        if r < 0.04:
            return 1
        if r < 0.28:
            return self.rng.randint(2, 32)
        if r < 0.70:
            return self.rng.randint(33, 128)
        return self.rng.randint(129, 400)

    def prompt(self):
        r = self.rng.random()
        if r < 0.15:
            return self.rng.choice(TINY)
        if r < 0.65:
            return self.rng.choice(TOPICS)
        n = self.rng.randint(300, 4000)
        return "Read the following notes, then summarise them in three bullets.\n\n" + self.tok.filler(self.rng, n)

    def sampling(self, greedy=False):
        rng, p = self.rng, {}
        if greedy:
            p["temperature"] = 0.0
        else:
            p["temperature"] = round(rng.uniform(0.3, 1.2), 3)
            if rng.random() < 0.5:
                p["top_p"] = round(rng.uniform(0.5, 0.98), 3)
            if rng.random() < 0.4:
                p["top_k"] = rng.choice([1, 5, 20, 40, 100])
            if rng.random() < 0.3:
                p["min_p"] = round(rng.uniform(0.01, 0.2), 3)
        return p

    def chat(self, messages, stream=None, **extra):
        body = {"model": "default", "messages": messages, "max_tokens": extra.pop("max_tokens", self.max_tokens())}
        body.update(extra)
        stream = self.rng.random() < 0.7 if stream is None else stream
        body["stream"] = stream
        if stream:
            body["stream_options"] = {"include_usage": True}
        return {"endpoint": "/v1/chat/completions", "body": body}

    def user(self, text, system=None):
        m = [{"role": "system", "content": system}] if system else []
        return m + [{"role": "user", "content": text}]

    # --- kinds ----------------------------------------------------------------------------------------------------
    def determinism(self):
        r = self.chat(self.user(DETERMINISM_PROMPT), stream=False, max_tokens=64, temperature=0.0)
        r["determinism"] = True
        return r

    def make_greedy(self):
        return [self.chat(self.user(self.prompt()), **self.sampling(greedy=True))]

    def make_sampled(self):
        return [self.chat(self.user(self.prompt()), **self.sampling())]

    def make_penalties(self):
        rng = self.rng
        p = self.sampling(greedy=rng.random() < 0.3)
        for key, lo, hi in (("frequency_penalty", -0.5, 1.5), ("presence_penalty", -0.5, 1.5), ("repetition_penalty", 0.8, 1.5)):
            if rng.random() < 0.6:
                p[key] = round(rng.uniform(lo, hi), 3)
        if not any(k in p for k in ("frequency_penalty", "presence_penalty", "repetition_penalty")):
            p["repetition_penalty"] = 1.2
        return [self.chat(self.user(self.prompt()), **p)]

    def make_logit_bias(self):
        p = self.sampling(greedy=self.rng.random() < 0.4)
        p["logit_bias"] = {str(i): self.rng.choice([-100, -5, 3, 8]) for i in self.rng.sample(self.bias_ids, k=min(2, len(self.bias_ids)))}
        return [self.chat(self.user(self.rng.choice(TOPICS)), **p)]

    def make_logprobs(self):
        p = self.sampling(greedy=self.rng.random() < 0.4)
        p["logprobs"] = True
        p["top_logprobs"] = self.rng.choice([0, 1, 3, 5, 10, 20])
        r = self.chat(self.user(self.prompt()), **p)
        r["expect_top_logprobs"] = p["top_logprobs"]
        return [r]

    def make_seeded(self):
        p = self.sampling()
        p["seed"] = self.rng.randint(0, 2**31 - 1)
        return [self.chat(self.user(self.rng.choice(TOPICS)), **p)]

    def make_stop(self):
        p = self.sampling(greedy=self.rng.random() < 0.5)
        if self.rng.random() < 0.5:
            p["stop"] = self.rng.choice([["\n\n"], ["."], [" and ", "\n"], ["```"]])
        else:
            p["stop_token_ids"] = self.rng.sample(self.stop_ids, k=min(len(self.stop_ids), self.rng.randint(1, 2)))
        p["max_tokens"] = self.rng.randint(32, 256)
        r = self.chat(self.user(self.rng.choice(TOPICS)), **p)
        r["expect_stop"] = p.get("stop")
        return [r]

    def make_max1(self):
        return [self.chat(self.user(self.prompt()), max_tokens=1, **self.sampling(greedy=self.rng.random() < 0.5))]

    def make_json_schema(self):
        s = self.rng.choice(JSON_SCHEMAS)
        p = self.sampling(greedy=self.rng.random() < 0.5)
        p["response_format"] = {"type": "json_schema", "json_schema": {"name": s["name"], "schema": s["schema"]}}
        r = self.chat(self.user(s["prompt"]), max_tokens=256, **p)
        r["validate"] = {"type": "json", "required": s["schema"].get("required", [])}
        return [r]

    def make_regex(self):
        pattern, prompt = self.rng.choice(REGEXES)
        r = self.chat(self.user(prompt), max_tokens=96, regex=pattern, **self.sampling(greedy=self.rng.random() < 0.5))
        r["validate"] = {"type": "regex", "pattern": pattern}
        return [r]

    def make_ebnf(self):
        grammar, pattern, prompt = self.rng.choice(EBNFS)
        r = self.chat(self.user(prompt), max_tokens=64, ebnf=grammar, **self.sampling(greedy=self.rng.random() < 0.5))
        r["validate"] = {"type": "regex", "pattern": pattern}
        return [r]

    def make_tools(self):
        system = (
            "You can call these functions. To call one, reply with only a JSON object "
            '{"name": ..., "arguments": {...}}.\n' + json.dumps([t["function"] for t in TOOLS])
        )
        ask = self.rng.choice(["What's the weather in Oslo in celsius?", "Convert 99.5 USD to GBP.", "Weather in Lima?"])
        extra = {"tools": TOOLS} if self.rng.random() < 0.5 else {}
        return [self.chat(self.user(ask, system=system), max_tokens=128, **extra, **self.sampling(greedy=self.rng.random() < 0.6))]

    def make_system(self):
        return [self.chat(self.user(self.rng.choice(TOPICS), system=self.rng.choice(SYSTEMS)), **self.sampling(greedy=self.rng.random() < 0.4))]

    def make_conversation(self):
        system = self.rng.choice(SYSTEMS) if self.rng.random() < 0.5 else None
        first = self.rng.choice(TOPICS)
        if self.rng.random() < 0.5:
            first = "Context notes:\n" + self.tok.filler(self.rng, self.rng.randint(500, 3000)) + "\n\n" + first
        follow = ["Now make it shorter.", "Give one more example.", "What did you assume?", "Summarise that in one line."]
        turns = [first] + self.rng.sample(follow, k=self.rng.randint(2, 3))
        p = self.sampling(greedy=self.rng.random() < 0.5)
        reqs = []
        for i, t in enumerate(turns):
            r = self.chat(self.user(t, system=system if i == 0 else None), max_tokens=self.rng.randint(24, 160), **p)
            r["turn"] = i
            reqs.append(r)
        return [{"conversation": reqs}]

    def make_long(self):
        n = self.rng.choice([self.rng.randint(4000, 16000)] * 3 + [self.rng.randint(16000, 28000)])
        text = self.tok.filler(self.rng, n)
        ask = self.rng.choice(["Summarise these notes in three bullets.", "Which ledger line moved the most units?",
                               "Count the function definitions above."])
        mt = self.rng.randint(8, 128)
        mt = min(mt, CONTEXT - n - 256)
        return [self.chat(self.user(text + "\n\n" + ask), max_tokens=mt, **self.sampling(greedy=self.rng.random() < 0.5))]

    def make_burst(self):
        reqs = []
        for _ in range(self.rng.randint(3, 5)):
            r = self.chat(self.user(self.rng.choice(TOPICS + TINY)), max_tokens=self.rng.randint(4, 64),
                          **self.sampling(greedy=self.rng.random() < 0.5))
            reqs.append(r)
        return [{"burst": reqs}]

    def make_abort(self):
        r = self.chat(self.user(self.rng.choice(TOPICS)), stream=True, max_tokens=300, **self.sampling(greedy=self.rng.random() < 0.5))
        r["abort_after"] = self.rng.randint(1, 40)
        return [r]

    def make_n2(self):
        return [self.chat(self.user(self.rng.choice(TOPICS)), stream=False, n=2, max_tokens=48, temperature=0.8)]

    def make_completions(self):
        body = {"model": "default", "prompt": self.rng.choice(TOPICS), "max_tokens": self.rng.randint(8, 96),
                **self.sampling(greedy=self.rng.random() < 0.5)}
        if self.rng.random() < 0.7:
            body["logprobs"] = self.rng.choice([1, 3, 5])
            body["echo"] = self.rng.random() < 0.5
        body["stream"] = self.rng.random() < 0.5
        return [{"endpoint": "/v1/completions", "body": body}]

    def make_generate(self):
        sp = {"max_new_tokens": self.rng.randint(8, 96), **self.sampling(greedy=self.rng.random() < 0.5)}
        if self.rng.random() < 0.3:
            sp["min_new_tokens"] = min(sp["max_new_tokens"], 8)
        if self.rng.random() < 0.3:
            sp["json_schema"] = json.dumps(JSON_SCHEMAS[0]["schema"])
        body = {"text": self.rng.choice(TOPICS), "sampling_params": sp, "return_logprob": True,
                "top_logprobs_num": self.rng.choice([0, 2, 5]), "logprob_start_len": self.rng.choice([-1, 0]),
                "stream": self.rng.random() < 0.5}
        r = {"endpoint": "/generate", "body": body}
        if "json_schema" in sp:
            r["validate"] = {"type": "json", "required": JSON_SCHEMAS[0]["schema"]["required"]}
        return [r]


class Runner:
    def __init__(self, port: int, out: str, server_pid, tok: Tok):
        self.port, self.out, self.server_pid, self.tok = port, out, server_pid, tok
        self.t0 = time.monotonic()
        self.lock = threading.Lock()
        self.records = open(os.path.join(out, "requests.jsonl"), "a")
        self.active = {}  # request idx -> [last chunk time, captured]
        self.stop = threading.Event()
        self.n = 0
        self.determinism_texts = []
        os.makedirs(os.path.join(out, "captures"), exist_ok=True)
        threading.Thread(target=self._watch, daemon=True).start()

    def now(self):
        return time.monotonic() - self.t0

    # --- evidence while a stream is stalled ------------------------------------------------------------------------
    def _watch(self):
        while not self.stop.is_set():
            time.sleep(0.05)
            with self.lock:
                stalled = [(i, a) for i, a in self.active.items() if a[0] and not a[1] and time.monotonic() - a[0] > CAPTURE_GAP_S]
                for _, a in stalled:
                    a[1] = True
            for i, _ in stalled:
                threading.Thread(target=self.capture, args=(f"req{i}-stall",), daemon=True).start()

    def capture(self, tag):
        path = os.path.join(self.out, "captures", f"{int(self.now()):06d}-{tag}.txt")
        pids = subprocess.run(["pgrep", "-f", "sglang::scheduler"], capture_output=True, text=True).stdout.split()
        with open(path, "w") as f:
            f.write(f"t={self.now():.3f} tag={tag} scheduler pids={pids}\n")
            f.flush()
            for pid in pids[:1]:
                for cmd in (
                    ["/data/models/slang/.venv/bin/py-spy", "dump", "--native", "--pid", pid],
                    ["eu-stack", "-p", pid],
                ):
                    try:
                        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                        f.write(f"\n===== {' '.join(cmd)} rc={r.returncode}\n{r.stdout}\n{r.stderr}\n")
                    except Exception as e:  # noqa: BLE001 - evidence is best effort
                        f.write(f"\n===== {' '.join(cmd)} failed: {e!r}\n")
                    f.flush()
            r = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
            f.write("\n===== nvidia-smi\n" + r.stdout)
            if os.environ.get("SOAK_CAPTURE_CUDA_GDB") == "1" and pids:
                # Diagnosis runs only (a long RAM-miss deadline): which kernels are resident while the step is stuck.
                cmd = ["/usr/local/cuda-13.2/bin/cuda-gdb", "-p", pids[0], "-batch", "-ex", "info cuda kernels",
                       "-ex", "thread apply all bt 25", "-ex", "detach"]
                try:
                    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                    f.write(f"\n===== cuda-gdb rc={r.returncode}\n{r.stdout}\n{r.stderr[-4000:]}\n")
                except Exception as e:  # noqa: BLE001
                    f.write(f"\n===== cuda-gdb failed: {e!r}\n")

    def server_alive(self):
        if self.server_pid is None:
            return True
        try:
            os.kill(self.server_pid, 0)
        except OSError:
            return False
        return True

    # --- one request ---------------------------------------------------------------------------------------------
    def run(self, req, item_idx, kind, extra_messages=None):
        with self.lock:
            idx = self.n
            self.n += 1
        body = dict(req["body"])
        if extra_messages is not None:
            body["messages"] = extra_messages
        rec = {"idx": idx, "item": item_idx, "kind": kind, "endpoint": req["endpoint"], "stream": bool(body.get("stream")),
               "t_start": round(self.now(), 3), "params": {k: v for k, v in body.items() if k not in ("messages", "prompt", "text")}}
        if "messages" in body:
            rec["prompt_chars"] = sum(len(m["content"]) for m in body["messages"])
        for k in ("determinism", "abort_after", "turn", "expect_top_logprobs", "expect_stop", "validate"):
            if k in req:
                rec[k] = req[k]
        chunks, text, t_send = [], [], time.monotonic()
        try:
            conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=REQUEST_TIMEOUT_S)
            conn.request("POST", req["endpoint"], json.dumps(body), {"Content-Type": "application/json"})
            resp = conn.getresponse()
            rec["status"] = resp.status
            if resp.status != 200:
                rec["error"] = resp.read().decode(errors="replace")[:2000]
            elif body.get("stream"):
                with self.lock:
                    self.active[idx] = [None, False]
                usage, finish, aborted, lp = None, None, False, []
                for raw in resp:
                    line = raw.decode(errors="replace").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    d = json.loads(data)
                    piece = self._piece(req["endpoint"], d, text)
                    if d.get("usage"):
                        usage = d["usage"]
                    if req["endpoint"] == "/generate":
                        meta = d.get("meta_info", {})
                        usage = {"prompt_tokens": meta.get("prompt_tokens"), "completion_tokens": meta.get("completion_tokens")}
                        finish = (meta.get("finish_reason") or {}).get("type") if meta.get("finish_reason") else finish
                    else:
                        for c in d.get("choices", []):
                            finish = c.get("finish_reason") or finish
                            if c.get("logprobs"):
                                lp.append(c["logprobs"])
                    if piece:
                        t = time.monotonic()
                        chunks.append(t)
                        with self.lock:
                            self.active[idx][0] = t
                        if req.get("abort_after") and len(chunks) >= req["abort_after"]:
                            aborted = True
                            break
                if aborted:
                    conn.sock.close()
                    rec["aborted_after_chunks"] = len(chunks)
                rec["usage"], rec["finish_reason"] = usage, finish
                rec["logprob_chunks"] = len(lp)
                if lp and req.get("expect_top_logprobs") is not None:
                    tops = [len(c.get("top_logprobs") or []) for x in lp for c in (x.get("content") or [])]
                    rec["top_logprobs_ok"] = bool(tops) and all(n == req["expect_top_logprobs"] for n in tops)
            else:
                d = json.loads(resp.read())
                chunks.append(time.monotonic())
                self._final(req["endpoint"], d, rec, text, req)
            conn.close()
        except Exception as e:  # noqa: BLE001 - every failure is a result to record
            rec["error"] = f"{type(e).__name__}: {e}"
        finally:
            with self.lock:
                self.active.pop(idx, None)
        rec["text"] = "".join(text)
        rec["wall_s"] = round(time.monotonic() - t_send, 3)
        if chunks:
            rec["ttft_s"] = round(chunks[0] - t_send, 3)
        if body.get("stream") and len(chunks) >= 2:
            gaps = [b - a for a, b in zip(chunks, chunks[1:])]
            rec["chunks"] = len(chunks)
            rec["ms_per_token"] = round(1000 * (chunks[-1] - chunks[0]) / (len(chunks) - 1), 2)
            rec["gap_ms_max"] = round(1000 * max(gaps), 1)
            for s in STALL_S:
                rec[f"gaps_ge_{s}s"] = sum(g >= s for g in gaps)
        rec["ok"] = rec.get("status") == 200 and "error" not in rec
        if rec["ok"] and "validate" in req:
            rec["valid"] = validate(req["validate"], rec["text"])
        if rec.get("expect_stop") and rec["ok"]:
            rec["stop_absent"] = not any(s in rec["text"] for s in rec["expect_stop"])
        if req.get("determinism"):
            self.determinism_texts.append(rec["text"])
            rec["determinism_identical"] = rec["text"] == self.determinism_texts[0]
        # A refused or dropped connection is how a fail-stop looks from here; the process may take a while to exit.
        rec["connection_lost"] = any(e in rec.get("error", "") for e in ("ConnectionRefused", "RemoteDisconnected", "ConnectionReset"))
        rec["server_alive"] = self.server_alive()
        with self.lock:
            self.records.write(json.dumps(rec) + "\n")
            self.records.flush()
        return rec

    @staticmethod
    def _piece(endpoint, d, text):
        if endpoint == "/generate":
            s = d.get("text", "")
            # /generate streams the cumulative text unless incremental output is on.
            prev = "".join(text)
            new = s[len(prev):] if s.startswith(prev) else s
            if new:
                text.clear()
                text.append(s)
            return new
        piece = ""
        for c in d.get("choices", []):
            piece += (c.get("delta") or {}).get("content") or c.get("text") or ""
        text.append(piece)
        return piece

    @staticmethod
    def _final(endpoint, d, rec, text, req):
        if endpoint == "/generate":
            text.append(d.get("text", ""))
            meta = d.get("meta_info", {})
            rec["usage"] = {"prompt_tokens": meta.get("prompt_tokens"), "completion_tokens": meta.get("completion_tokens")}
            rec["finish_reason"] = (meta.get("finish_reason") or {}).get("type")
            rec["input_logprobs"] = len(meta.get("input_token_logprobs") or [])
            rec["output_logprobs"] = len(meta.get("output_token_logprobs") or [])
            return
        choices = d.get("choices", [])
        rec["n_choices"] = len(choices)
        if choices:
            c = choices[0]
            text.append((c.get("message") or {}).get("content") or c.get("text") or "")
            rec["finish_reason"] = c.get("finish_reason")
            if len(choices) > 1:
                rec["other_choices"] = [(x.get("message") or {}).get("content") for x in choices[1:]]
            lp = c.get("logprobs")
            if lp and req.get("expect_top_logprobs") is not None:
                tops = [len(x.get("top_logprobs") or []) for x in (lp.get("content") or [])]
                rec["top_logprobs_ok"] = bool(tops) and all(n == req["expect_top_logprobs"] for n in tops)
            elif lp:
                rec["logprobs_tokens"] = len(lp.get("tokens") or lp.get("content") or [])
        rec["usage"] = d.get("usage")


def validate(v, text):
    if v["type"] == "json":
        try:
            obj = json.loads(text)
        except ValueError:
            return False
        return isinstance(obj, dict) and all(k in obj for k in v["required"])
    return re.fullmatch(v["pattern"], text) is not None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--max-requests", type=int, default=320)
    ap.add_argument("--max-minutes", type=float, default=115)
    ap.add_argument("--server-pid", type=int)
    ap.add_argument("--only-kinds")
    ap.add_argument("--start-item", type=int, default=0, help="skip the plan's first items (a rerun of one item)")
    ap.add_argument("--plan-only", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tok = Tok(a.tokenizer)
    plan = Plan(random.Random(a.seed), tok, a.max_requests, a.only_kinds.split(",") if a.only_kinds else None)
    with open(os.path.join(a.out, "plan.jsonl"), "w") as f:
        for i, item in enumerate(plan.items):
            f.write(json.dumps({"item": i, **item}) + "\n")
    if a.plan_only:
        print(f"{len(plan.items)} items")
        return 0
    runner = Runner(a.port, a.out, a.server_pid, tok)
    deadline = time.monotonic() + 60 * a.max_minutes
    died, items_run = False, 0
    for i, item in enumerate(plan.items):
        if i < a.start_item:
            continue
        if time.monotonic() > deadline:
            print(f"time cap reached before item {i}", flush=True)
            break
        kind = item["kind"]
        recs = []
        for req in item["requests"]:
            if "burst" in req:
                out = [None] * len(req["burst"])

                def go(j, r):
                    out[j] = runner.run(r, i, kind)

                ts = [threading.Thread(target=go, args=(j, r)) for j, r in enumerate(req["burst"])]
                for t in ts:
                    t.start()
                for t in ts:
                    t.join()
                recs += out
            elif "conversation" in req:
                history = []
                for turn in req["conversation"]:
                    msgs = history + turn["body"]["messages"]
                    rec = runner.run(turn, i, kind, extra_messages=msgs)
                    recs.append(rec)
                    if not rec["ok"]:
                        break
                    history = msgs + [{"role": "assistant", "content": rec["text"]}]
            else:
                recs.append(runner.run(req, i, kind))
        items_run += 1
        for r in recs:
            flag = "" if r["ok"] else f" ERROR {r.get('status')} {r.get('error', '')[:200]}"
            print(f"[{r['t_start']:8.1f}s] item {i} req {r['idx']} {kind:12s} {r['endpoint']:22s} stream={int(r['stream'])} "
                  f"tok={((r.get('usage') or {}).get('completion_tokens'))} {r['wall_s']:.1f}s "
                  f"mspt={r.get('ms_per_token')} gapmax={r.get('gap_ms_max')}{flag}", flush=True)
        if any(r["connection_lost"] and not r.get("aborted_after_chunks") for r in recs) or not all(r["server_alive"] for r in recs) or not runner.server_alive():
            died = True
            print(f"SERVER DIED during item {i} ({kind}) at {runner.now():.1f}s", flush=True)
            break
    runner.stop.set()
    ids = [t == runner.determinism_texts[0] for t in runner.determinism_texts] if runner.determinism_texts else []
    summary = {"seed": a.seed, "items_planned": len(plan.items), "items_run": items_run, "requests": runner.n,
               "elapsed_s": round(runner.now(), 1), "server_died": died,
               "determinism": {"runs": len(ids), "identical": sum(ids)}}
    with open(os.path.join(a.out, "driver_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary), flush=True)
    return 3 if died else 0


if __name__ == "__main__":
    sys.exit(main())
