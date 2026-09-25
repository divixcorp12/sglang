# Publish and run the SGLang servers on divix01

Two servers can run on divix01's port 7867, one at a time:

- **Qwen3.8 NVFP4** (below): the earlier production server, kept so it can be run again.
- **DeepSeek-V4.1 Flash EXL3** ([DSV4.1 production](#dsv41-production)): the current
  production server.

The Qwen guide publishes the `master` branch from the local
SGLang checkout, updates the matching checkout on divix01, and launches the
NVFP4 server with dynamic expert residency.

## Paths and endpoints

| Item | Value |
| --- | --- |
| Local checkout | `/home/dimitri/data/divix/sglang-nvfp4` |
| divix01 checkout | `/data/models/slang/nvfp4-work/main-port-probe-7bc4eb` (as of 2026-09-25 detached at `797be6f678`, the last Qwen-served commit; the update step below refuses a detached checkout, so run `git switch master` there first only if Qwen should run on current `master`) |
| Remote | `origin` = `git@github.com:divixcorp12/sglang.git` (the divix01 bare repo is retired) |
| Branch | `master` |
| Launch script | `/data/models/slang/nvfp4-work/run-nvfp4-expert-dynamic-hot10g.sh` |
| tmux session | `cc-nvfp4-dynamic` |
| Server endpoint on divix01 | `http://127.0.0.1:7867` |
| Logs | `/data/models/slang/nvfp4-stream-logs/` |

The launch script uses a 10 GiB GPU expert hot tier, a 35 GiB pinned-host tier,
seeded dynamic residency, and disabled expert prefetch. Its source of record is
`/home/dimitri/data/divix/crypto/prototypes/sglang-nvfp4/run-nvfp4-expert-dynamic-hot10g.sh`;
copy it over the host-local script whenever it changes.

## Storage layout on divix01

The model, the PLE table, and the expert files live on `/mnt/nvme2`. The expert
and PLE cache manifests hash the resolved model path, and they were built from
`/data/models/huggingface_hub/...`. The launch script therefore sets
`SGLANG_FILE_CACHE_MODEL_PATH` to that old path; without it, the move rebuilds
115 GB of caches. Change that variable together with `model`, never alone,
because a stale value makes a different checkpoint reuse these caches. The
launch script refuses to start when either cache directory has no manifests,
instead of silently rebuilding.

File reads are selected by environment in the launch script:

| Variable | Value | Why |
| --- | --- | --- |
| `SGLANG_MOE_EXPERT_FILE_READER` | `uring_direct` | Expert rows are page multiples read straight into pinned slots with `O_DIRECT`, so the 64 GB file does not also fill the page cache. |
| `SGLANG_QWEN4_PLE_FILE_READER` | `uring` | PLE rows are 160 B; buffered page reads let repeated tokens hit the page cache. |

Either variable set to `mmap` restores the previous shared-mapping reads.

Decode CUDA graphs replay the MoE layers without breaking when two more
variables are set:

| Variable | Value | Why |
| --- | --- | --- |
| `SGLANG_MOE_EXPERT_HOST_ARENA` | `1` | Startup copies all 63.3 GiB of host expert rows into page-aligned memory registered with CUDA, so a captured kernel can pull any expert. Requires `SGLANG_MOE_PINNED_HOST_MB=0` and about 64 GiB of free RAM. |
| `SGLANG_MOE_EXPERT_GRAPH_GATHER` | `1` | Decode gathers of up to `--cuda-graph-max-bs-decode` × top-k routes run without host syncs. Each layer reserves that many scratch rows from `SGLANG_MOE_HOT_GPU_MB` (about 1.2 GiB at max batch size 1). |
| `SGLANG_MOE_HOT_UPDATE_DECODE_FORWARDS` | `16` | With `SGLANG_MOE_HOT_DYNAMIC=1`, re-ranks each layer's hot experts every 16 decode forwards and copies the promoted rows in. Without it only a prefill of `SGLANG_MOE_HOT_UPDATE_PREFILL_TOKENS` tokens re-ranks, so a long decode keeps missing the experts it keeps using. `0` disables it. |
| `SGLANG_MOE_EXPERT_COPY_BACKEND` | `dma` | Eager (uncaptured) misses of layers with a hot cache, and hot cache promotions, copy registered host expert rows through the CUDA copy engine, merged into runs of consecutive experts that never cross a host registration: about 12.8 GiB/s on this PCIe 3.0 x16 link, against 6.6–8.3 GiB/s for the GPU pull kernel. Decode graphs always use the pull kernel. The metrics file's `gather_copy_engine_bytes` stays 0 when the copy engine is not built. |
| `SGLANG_MOE_HOT_DECAY_TOKENS` | `16` | Older scores decay by 0.95 per 16 routed tokens, whether the tokens arrived in one prefill or in 16 decode forwards; 16 keeps the half-life of about 216 tokens that decode updates were first measured with. A boundary's own counts still enter at full weight, so a 1,024-token prefill chunk leaves about 4% of earlier history, the seed included. Unset, that chunk would decay history only as much as one decode update. |
| `SGLANG_MOE_HOT_PROMOTION_SIGMAS` | `2` | An expert replaces a resident only when its score leads by 2 standard deviations of routing-count noise, at most about `sqrt(a + b)` counts for scores `a` and `b`. Without it the first measured run moved 1.9 GiB of experts in 20 decode updates. |
| `SGLANG_QWEN4_PLE_STAGE_BEFORE_REPLAY` | `1` | The file-backed PLE rows are read before each decode replay instead of inside a graph break. Requires `--disable-overlap-schedule` and no speculative decoding. |

Unset the expert variables, and restore `SGLANG_MOE_PINNED_HOST_MB`, to return
to the pinned LRU tier with one graph break per MoE layer; unset the PLE
variable to put the PLE read back inside a break. Done when startup logs
`Expert host arena startup`, and the `Breakable CUDA graph captured` lines
report fewer breaks than the 48 streamed MoE layers:

```bash
ssh divix01 'log=/data/models/slang/nvfp4-stream-logs/nvfp4-expert-dynamic-hot10g.latest.log; grep -E "Expert host arena startup|Breakable CUDA graph captured" "$log"'
```

One-time cut-over, with the server stopped (see
[Stop the server](#8-stop-the-server)). The SATA copies stay in place:

```bash
ssh divix01 'mkdir -p /mnt/nvme2/nvfp4-work && rsync -aH --info=progress2 /data/models/slang/nvfp4-work/qwen38-nvfp4-expert-cache-v1 /mnt/nvme2/nvfp4-work/'
scp /home/dimitri/data/divix/crypto/prototypes/sglang-nvfp4/run-nvfp4-expert-dynamic-hot10g.sh divix01:/data/models/slang/nvfp4-work/run-nvfp4-expert-dynamic-hot10g.sh
```

Done when the first startup log reports `outcome=verified_hit` for every
`File tensor cache` line and contains no `building_miss`:

```bash
ssh divix01 'log=/data/models/slang/nvfp4-stream-logs/nvfp4-expert-dynamic-hot10g.latest.log; grep -o "outcome=[a-z_]*" "$log" | sort | uniq -c'
```

## Testing on divix01 without touching the served checkout

Tests run from a copy of the local tree, with pytest installed outside the
server's venv:

```bash
rsync -a --delete --exclude __pycache__ python test divix01:/data/models/slang/nvfp4-work/uring-test-tree/
ssh divix01 'cd /data/models/slang/nvfp4-work/uring-test-tree && source /data/models/slang/.venv/bin/activate && PYTHONPATH=/data/models/slang/nvfp4-work/flashinfer-0.6.18-cu130-overlay:$PWD/python:/data/models/slang/nvfp4-work/uring-test-deps python -m pytest -q -p no:cacheprovider <test paths>'
```

## 1. Publish the local branch

Start from the local checkout and confirm that the branch and changes are the
ones intended for divix01:

```bash
cd /home/dimitri/data/divix/sglang-nvfp4
git status --short --branch
git diff --check
```

Commit any intended changes before publishing them:

```bash
git add --patch
git add run_server.md
git diff --cached --check
git commit
```

Add every new file explicitly by name; `git add --patch` only stages changes to
files that Git already tracks.

Push the branch to GitHub:

```bash
git push origin master
```

Confirm that the local branch is synchronized:

```bash
git status --short --branch
git rev-parse HEAD
```

The status should not show an ahead/behind count or uncommitted files.

## 2. Update the divix01 checkout

Do not update source files underneath a running server. Check first:

```bash
ssh divix01 'pgrep -af "[s]glang serve.*--port 7867" || true'
```

If this finds the server, either keep serving the current commit or stop it
intentionally using the procedure in [Stop the server](#8-stop-the-server)
before pulling the new commit.

Inspect the remote checkout before changing it:

```bash
ssh divix01 'cd /data/models/slang/nvfp4-work/main-port-probe-7bc4eb && git status --short --branch && git rev-parse HEAD'
```

If it is clean, update it with:

```bash
ssh divix01 'cd /data/models/slang/nvfp4-work/main-port-probe-7bc4eb && branch=$(git branch --show-current) && test "$branch" = master || { echo "WRONG_BRANCH: $branch"; exit 1; }; git pull --rebase origin master'
```

If it is dirty, do not reset or overwrite it. Those changes may be work created
directly on divix01. First make sure they are represented in the local checkout
and committed. From the local checkout, compare the contents of every dirty
remote path with the local tree:

```bash
rsync -inc --dry-run --out-format='%i %n' --files-from=<(ssh divix01 'cd /data/models/slang/nvfp4-work/main-port-probe-7bc4eb && git ls-files -m -o --exclude-standard') ./ divix01:/data/models/slang/nvfp4-work/main-port-probe-7bc4eb/
```

No output means those remote paths have identical local contents. Any output
must be investigated and copied or committed deliberately before continuing.
When the remote changes are known to match the published commit, preserve them
in a recoverable stash before pulling:

```bash
ssh divix01 'cd /data/models/slang/nvfp4-work/main-port-probe-7bc4eb && branch=$(git branch --show-current) && test "$branch" = master || { echo "WRONG_BRANCH: $branch"; exit 1; }; git stash push -u -m pre-shared-update-$(date +%Y%m%d-%H%M%S) && git pull --rebase origin master'
```

Keep that stash until the updated server has been verified. Do not immediately
`git stash pop`: the same changes are already present in the published commit.

Verify that local and divix01 now use the same commit:

```bash
git -C /home/dimitri/data/divix/sglang-nvfp4 rev-parse HEAD
ssh divix01 'cd /data/models/slang/nvfp4-work/main-port-probe-7bc4eb && git status --short --branch && git rev-parse HEAD'
```

## 3. Check the host before launch

Do not launch a duplicate server. Check the existing process, port, sessions,
and available GPU memory:

```bash
ssh divix01 'pgrep -af "[s]glang serve.*--port 7867" || true'
ssh divix01 'ss -ltn 2>/dev/null | grep ":7867[[:space:]]" || true'
ssh divix01 'tmux list-sessions 2>/dev/null || echo NO_SESSIONS'
ssh divix01 'nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader'
```

If an SGLang process is already serving on port 7867, use that process or stop
it intentionally before launching the updated checkout.

## 4. Launch the server

The server is long-running, so run it in the existing agent-owned tmux session
or create that session if it does not exist:

```bash
ssh divix01 'tmux has-session -t cc-nvfp4-dynamic 2>/dev/null || tmux new-session -d -s cc-nvfp4-dynamic'
ssh divix01 'tmux list-panes -t cc-nvfp4-dynamic -F "command=#{pane_current_command} dead=#{pane_dead} pid=#{pane_pid}"'
ssh divix01 'readlink -f /data/models/slang/nvfp4-stream-logs/nvfp4-expert-dynamic-hot10g.latest.log 2>/dev/null || echo NO_PREVIOUS_LOG'
```

Launch with a guard that refuses to send keys unless the pane is alive and
running an idle shell:

```bash
ssh divix01 'state=$(tmux display-message -p -t cc-nvfp4-dynamic "#{pane_dead}:#{pane_current_command}") || exit 1; case "$state" in 0:bash|0:zsh|0:sh) tmux send-keys -t cc-nvfp4-dynamic "bash /data/models/slang/nvfp4-work/run-nvfp4-expert-dynamic-hot10g.sh" Enter ;; *) echo "REFUSING_BUSY_PANE: $state"; exit 1 ;; esac'
ssh divix01 'tmux has-session -t cc-nvfp4-dynamic 2>/dev/null && echo SESSION_OK || echo SESSION_FAILED'
ssh divix01 'readlink -f /data/models/slang/nvfp4-stream-logs/nvfp4-expert-dynamic-hot10g.latest.log; stat -Lc "started=%y bytes=%s" /data/models/slang/nvfp4-stream-logs/nvfp4-expert-dynamic-hot10g.latest.log'
```

Confirm that `latest.log` now resolves to a newly timestamped file instead of
the previous run. The host-local launch script is outside the Git checkout;
verify it separately whenever its model paths or launch settings change.

Do not attach to the session or use `tmux capture-pane`. The launch script logs
stdout and stderr to timestamped files and updates a `latest.log` symlink.

## 5. Monitor startup

Read only the relevant startup events; the full `server_args` line is extremely
large:

```bash
ssh divix01 'log=/data/models/slang/nvfp4-stream-logs/nvfp4-expert-dynamic-hot10g.latest.log; grep -E "Load weight|Pinned host expert cache startup|Expert hot cache startup|Uvicorn running|Server is ready|Traceback|NotImplemented|CUDA out of memory|OutOfMemory|Received sigquit|kill_process_tree|Killed" "$log" | grep -v "server_args=" | tail -40 | cut -c1-1000'
```

Also verify the process, listener, and GPU allocation:

```bash
ssh divix01 'pgrep -af "[s]glang serve.*--port 7867" | head -3'
ssh divix01 'ss -ltn 2>/dev/null | grep ":7867[[:space:]]"'
ssh divix01 'nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader'
```

A successful startup includes all of these events:

1. `Load weight end`
2. `Pinned host expert cache startup`
3. `Expert hot cache startup`
4. `Uvicorn running on http://127.0.0.1:7867`

## 6. Check health and inference

Check the health endpoint from divix01:

```bash
ssh divix01 'curl --fail --silent --show-error --max-time 10 http://127.0.0.1:7867/health >/dev/null && echo HEALTH_OK'
```

Send a short OpenAI-compatible chat request:

```bash
ssh divix01 'model=$(curl --fail --silent --show-error --max-time 10 http://127.0.0.1:7867/v1/models | jq -r ".data[0].id"); curl --fail --silent --show-error --max-time 180 http://127.0.0.1:7867/v1/chat/completions -H "Content-Type: application/json" -d "{\"model\":\"$model\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with exactly: NVFP4 OK\"}],\"temperature\":0,\"max_tokens\":16,\"chat_template_kwargs\":{\"enable_thinking\":false}}" | jq -c "{finish_reason: .choices[0].finish_reason, content: .choices[0].message.content, usage}"'
```

`enable_thinking` is disabled for this smoke test so a small token budget is not
consumed entirely by hidden reasoning. The expected content is `NVFP4 OK`.

## 7. Open the server locally

The server binds only to divix01 loopback. Forward it to the local machine:

```bash
ssh -N -L 7867:127.0.0.1:7867 divix01
```

While that command is running, use `http://127.0.0.1:7867` locally.

## 8. Stop the server

Stopping a production process is destructive. Verify the target before sending
an interrupt:

```bash
ssh divix01 'pgrep -af "[s]glang serve.*--port 7867"'
ssh divix01 "tmux send-keys -t cc-nvfp4-dynamic C-c"
```

Then confirm that the process and listener are gone:

```bash
ssh divix01 'pgrep -af "[s]glang serve.*--port 7867" || echo SERVER_STOPPED'
ssh divix01 'ss -ltn 2>/dev/null | grep ":7867[[:space:]]" || echo PORT_7867_FREE'
```

Do not run `tmux kill-server`, `byobu kill-server`, or `tmux capture-pane`.

---

## DSV4.1 production

DeepSeek-V4.1 Flash EXL3 on port 7867. Every flag and environment variable comes from
[`benchmarks/dsv41_baseline/arm_env.py`](benchmarks/dsv41_baseline/arm_env.py) (`base_env()`
and `ServerArgs.prod()`), launched by
[`benchmarks/dsv41_baseline/launch_prod.sh`](benchmarks/dsv41_baseline/launch_prod.sh) with no
overrides, so production and every benchmark arm run one recipe. To change a production
setting, change `arm_env.py`, not the launcher. Current settings and their evidence:
`DSV41_REFERENCE.md` §23.1 and §25.

| Item | Value |
| --- | --- |
| Remote and branch | `origin` = `git@github.com:divixcorp12/sglang.git`, `master` |
| divix01 checkout | `/data/models/slang/nvfp4-work/cc-expert-prediction/dsv41-direct-prod` (tracks `origin/master`) |
| Launcher | `/data/models/slang/nvfp4-work/cc-expert-prediction/dsv41-direct-live/launch.sh`, which execs the checkout's `launch_prod.sh` |
| Log and pid | `dsv41-direct-live/server.log` (appended), `dsv41-direct-live/server.pid` |
| Endpoint | `http://<divix01>:7867`, bound to all interfaces |
| GPU lock | `/data/models/slang/nvfp4-work/cc-gpu.lock`, held by the server for its lifetime |

Rules that apply to this server:

- Never test in the production checkout. Tests and benchmark arms run in private worktrees
  (`.claude/rules/divix01-run-protocol.md`).
- The server holds `cc-gpu.lock`, so no GPU test or benchmark can run while it is up, and it
  refuses to start while one does.
- The copy engine requires `CUDA_MODULE_LOADING=EAGER`, which `arm_env` sets. Never run
  graph-mode nsys against this server.
- Memory fraction is 0.83. A ~30k-token prompt at this setting has **not** been tested (§25.3).

### D1. Publish from the laptop

Commit on `master` in the local checkout and push:

```bash
cd /home/dimitri/data/divix/sglang-nvfp4
git status --short --branch
git push origin master
git status --short --branch   # no ahead/behind count
```

From a worktree whose `HEAD` is the commit to ship, `git push origin HEAD:master` does the
same. It must fast-forward; never force-push `master`.

### D2. Check that production is stopped

Never update the checkout under a running server:

```bash
ssh divix01 'pgrep -af "[s]glang.launch_server.*--port 7867" || echo DSV41_STOPPED'
ssh divix01 'ss -ltn 2>/dev/null | grep ":7867[[:space:]]" || echo PORT_7867_FREE'
ssh divix01 'flock --nonblock /data/models/slang/nvfp4-work/cc-gpu.lock true && echo GPU_LOCK_FREE || echo GPU_LOCK_HELD'
```

If the server is running, stop it first (D7). If the GPU lock is held by a test or benchmark,
wait for it; do not kill another job's process.

### D3. Update the production checkout

Inspect it, then fast-forward it to `origin/master`:

```bash
ssh divix01 'cd /data/models/slang/nvfp4-work/cc-expert-prediction/dsv41-direct-prod && git status --short --branch && git log -1 --oneline'
ssh divix01 'cd /data/models/slang/nvfp4-work/cc-expert-prediction/dsv41-direct-prod && git pull --ff-only origin master && git log -1 --oneline'
```

The only expected untracked file is `benchmarks/dsv41_baseline/generations.json`. Any tracked
change is work done directly on divix01: stop and commit it from the laptop instead of
overwriting it. If the pull refuses to fast-forward, someone committed on divix01 or
rewrote `master`; investigate, do not reset.

Confirm the laptop and divix01 agree:

```bash
git -C /home/dimitri/data/divix/sglang-nvfp4 rev-parse --short origin/master
ssh divix01 'cd /data/models/slang/nvfp4-work/cc-expert-prediction/dsv41-direct-prod && git rev-parse --short HEAD'
```

### D4. Dry-run the launch

Prints the checkout, cores, every environment variable and the full argv. It takes no lock
and starts nothing:

```bash
ssh divix01 'DRY_RUN=1 bash /data/models/slang/nvfp4-work/cc-expert-prediction/dsv41-direct-prod/benchmarks/dsv41_baseline/launch_prod.sh'
```

Check that `sglang:` resolves inside `dsv41-direct-prod`, and that the argv has
`--host 0.0.0.0 --port 7867`.

### D5. Start

```bash
ssh divix01 'cd /data/models/slang/nvfp4-work/cc-expert-prediction/dsv41-direct-live || exit 1; setsid nohup bash launch.sh >> server.log 2>&1 < /dev/null & echo $! > server.pid; sleep 5; ps -o pid,args -p $(cat server.pid) | cut -c1-120'
```

Every step from `setsid` to the server `exec`s in place, so `server.pid` is the server's own
pid; the `ps` line should show `sglang.launch_server ... --port 7867`. If the GPU lock is held, the launcher exits at once with
`cc-gpu.lock is held by another GPU job; not starting production` in `server.log`.

### D6. Monitor startup and check health

Startup takes several minutes (weights, pinned tier, graph capture). Follow the events, not
the whole log:

```bash
ssh divix01 'grep -E "Uvicorn running|fired up|copy engine|Traceback|Error|out of memory|refus" /data/models/slang/nvfp4-work/cc-expert-prediction/dsv41-direct-live/server.log | tail -20 | cut -c1-300'
```

A good start logs, in order: `Uvicorn running on http://0.0.0.0:7867`, `The server is fired up
and ready to roll!`, and after the first 16 decode forwards `exl3 RAM miss copy engine armed
after 16 decode forwards since capture`.

```bash
ssh divix01 'curl --fail --silent --show-error --max-time 900 http://127.0.0.1:7867/health >/dev/null && echo HEALTH_OK'
ssh divix01 'curl --fail --silent --show-error --max-time 300 http://127.0.0.1:7867/v1/chat/completions -H "Content-Type: application/json" -d "{\"model\":\"default\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with exactly: DSV41 OK\"}],\"temperature\":0,\"max_tokens\":32}" | jq -c "{finish_reason: .choices[0].finish_reason, content: .choices[0].message.content}"'
```

`/health` runs a real generation, so it can take minutes on a cold server.

### D7. Stop

Confirm the pid is the server before signalling it:

```bash
ssh divix01 'cd /data/models/slang/nvfp4-work/cc-expert-prediction/dsv41-direct-live && ps -o pid,args -p $(cat server.pid) | cut -c1-120'
ssh divix01 'cd /data/models/slang/nvfp4-work/cc-expert-prediction/dsv41-direct-live && kill $(cat server.pid)'
ssh divix01 'for i in $(seq 60); do pgrep -f "[s]glang.launch_server.*--port 7867" >/dev/null || { echo DSV41_STOPPED; break; }; sleep 2; done; nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader'
```

Only kill the pid that the first command shows as `sglang.launch_server ... --port 7867`.
