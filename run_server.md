# Publish and run the NVFP4 SGLang server on divix01

This guide publishes the `master` branch from the local
SGLang checkout, updates the matching checkout on divix01, and launches the
NVFP4 server with dynamic expert residency.

## Paths and endpoints

| Item | Value |
| --- | --- |
| Local checkout | `/home/dimitri/data/divix/sglang-nvfp4` |
| divix01 checkout | `/data/models/slang/nvfp4-work/main-port-probe-7bc4eb` |
| Shared bare remote on divix01 | `/data/models/slang/nvfp4-work/remotes/sglang-nvfp4.git` |
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

Push the branch to the shared remote:

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
ssh divix01 'cd /data/models/slang/nvfp4-work/main-port-probe-7bc4eb && branch=$(git branch --show-current) && test "$branch" = master || { echo "WRONG_BRANCH: $branch"; exit 1; }; git pull --rebase shared master'
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
ssh divix01 'cd /data/models/slang/nvfp4-work/main-port-probe-7bc4eb && branch=$(git branch --show-current) && test "$branch" = master || { echo "WRONG_BRANCH: $branch"; exit 1; }; git stash push -u -m pre-shared-update-$(date +%Y%m%d-%H%M%S) && git pull --rebase shared master'
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
