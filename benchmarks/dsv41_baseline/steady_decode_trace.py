"""A long streaming request with a separately recorded decode-only admission boundary.

Run only after run_arm's unchanged warm-up stability gate. Client progress is not
engine step latency: speculative decoding may deliver several tokens per update.
"""
import argparse
import json
from pathlib import Path
import time
import urllib.request

import compile_watch


class DecodeBurnIn:
    def __init__(self, min_seconds=30, min_tokens=256, min_updates=100):
        self.min_seconds, self.min_tokens, self.min_updates = min_seconds, min_tokens, min_updates
        self.first_ns = None
        self.tokens = self.updates = 0

    def observe(self, ns, tokens, has_content):
        if not has_content or not isinstance(tokens, int) or tokens <= self.tokens:
            return None
        if self.first_ns is None:
            self.first_ns = ns
        self.tokens = tokens
        self.updates += 1
        elapsed = (ns - self.first_ns) / 1e9
        if elapsed < self.min_seconds or tokens < self.min_tokens or self.updates < self.min_updates:
            return None
        return dict(ready_ns=ns, first_content_ns=self.first_ns, decode_seconds=elapsed,
                    completion_tokens=tokens, progress_updates=self.updates,
                    min_seconds=self.min_seconds, min_tokens=self.min_tokens, min_updates=self.min_updates,
                    observation="client stream progress; not engine step latency")


def post(port, endpoint, payload, timeout=600):
    return urllib.request.urlopen(urllib.request.Request(
        f"http://127.0.0.1:{port}{endpoint}", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}), timeout=timeout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--sessions", type=Path, required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ready-file", type=Path, required=True)
    parser.add_argument("--stop-file", type=Path, required=True)
    parser.add_argument("--server-log", type=Path, required=True)
    args = parser.parse_args()
    if args.ready_file.exists() or args.stop_file.exists():
        parser.error("refusing an existing admission or stop marker")
    session = next(d for line in args.sessions.read_text().splitlines()
                   if (d := json.loads(line))["session_id"] == args.session_id)
    rid = "steady-trace-" + str(time.monotonic_ns())
    payload = dict(model="default", messages=[dict(role="user", content=session["turns"][0])],
                   temperature=0, max_tokens=4096, ignore_eos=True, rid=rid, stream=True,
                   stream_options=dict(include_usage=True, continuous_usage_stats=True))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "request.json").write_text(json.dumps(payload, indent=2) + "\n")
    start_byte = compile_watch.log_size(args.server_log)
    start = time.monotonic_ns()
    burn = DecodeBurnIn()
    admitted = stopped = False
    with (args.output / "progress.jsonl").open("w") as progress:
        try:
            with post(args.port, "/v1/chat/completions", payload) as response:
                for raw in response:
                    if time.monotonic_ns() - start > 600_000_000_000:
                        raise TimeoutError("steady decode admission/capture timeout")
                    line = raw.decode().strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    chunk = json.loads(data)
                    choices = chunk.get("choices") or []
                    delta = choices[0].get("delta", {}) if choices else {}
                    tokens = (chunk.get("usage") or {}).get("completion_tokens")
                    ns = time.monotonic_ns()
                    content = bool(delta.get("content") or delta.get("reasoning_content"))
                    previous = burn.updates
                    ready = burn.observe(ns, tokens, content)
                    if burn.updates != previous:
                        progress.write(json.dumps(dict(ns=ns, completion_tokens=tokens,
                                                       progress_updates=burn.updates)) + "\n")
                        progress.flush()
                    if ready is not None and not admitted:
                        if compile_watch.compile_events_in_range(args.server_log, start_byte=start_byte):
                            raise RuntimeError("compilation during long request; admission refused")
                        ready.update(request_id=rid, request_start_ns=start, formal_warmup_passed=True,
                                     compile_events_since_request=0)
                        temporary = args.ready_file.with_suffix(".tmp")
                        temporary.write_text(json.dumps(ready, indent=2) + "\n")
                        temporary.replace(args.ready_file)
                        admitted = True
                        print("steady decode burn-in complete: " + json.dumps(ready), flush=True)
                    if args.stop_file.exists():
                        stopped = True
                        break
        finally:
            # Cancel only this diagnostic request before the harness stops its server.
            with post(args.port, "/abort_request", dict(rid=rid), timeout=30):
                pass
    compiles = compile_watch.compile_events_in_range(args.server_log, start_byte=start_byte)
    result = dict(diagnostic_only=True, admitted=admitted, stopped_after_capture=stopped,
                  completion_tokens=burn.tokens, progress_updates=burn.updates,
                  request_start_ns=start, end_ns=time.monotonic_ns(), compile_events=compiles)
    (args.output / "completion.json").write_text(json.dumps(result, indent=2) + "\n")
    if not admitted or not stopped or compiles:
        raise SystemExit("steady decode capture did not complete with clean admission: " + str(result))


if __name__ == "__main__":
    main()
