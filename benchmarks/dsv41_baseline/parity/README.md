# F145 parity: teacher-forced logprobs and config snapshot

Acceptance: mean per-token KL(base || branch) <= 1e-3 nats and top-1 agreement >= 99.5% on `prompts.jsonl`
(32 prompts: 8 code, 6 math, 8 prose, 6 tool-call, 4 long of about 4k tokens; 256 teacher-forced tokens each).
Base is origin/master 570958cb71. Follow `.claude/rules/divix01-run-protocol.md`: push, then use one private
worktree per arm; never run in the production checkout.

```bash
# laptop
git push origin expert-engine-template

# divix01: one private worktree per arm
ssh divix01 'git -C /data/models/slang/sglang fetch origin \
  && git -C /data/models/slang/sglang worktree add --detach /data/models/slang/nvfp4-work/wt-parity-base 570958cb71 \
  && git -C /data/models/slang/sglang worktree add --detach /data/models/slang/nvfp4-work/wt-parity-branch origin/expert-engine-template'
```

The parity tools exist only on the branch. Run them from the branch worktree against either server
(`PARITY=/data/models/slang/nvfp4-work/wt-parity-branch/benchmarks/dsv41_baseline/parity`).

## 1. Config snapshot at the base commit (CPU only)

The base worktree lacks these scripts, so run the branch's copy and point `--tree` at the base worktree: it loads
that tree's `environ.py` and `arm_env.py` by file path.

```bash
mkdir -p /data/models/slang/nvfp4-work/parity
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 taskset -c 0-63 \
  /data/models/slang/.venv-upstream/bin/python -B $PARITY/config_snapshot.py \
  --tree /data/models/slang/nvfp4-work/wt-parity-base \
  --out /data/models/slang/nvfp4-work/parity/base-config-snapshot.json
```

Check `git_commit` and `environ_path` in the JSON name the base worktree. `resolved.<recipe>.resolved` is
`{name: value}` for every `Envs` field under `base`, `dspark` and `prod`; `unregistered_recipe_names` are SGLANG_*
names the recipe sets that `Envs` does not declare. The branch's test maps old names to new config fields with the
merge table and compares against this file.

## 2. Logprob dump per arm

The server is the production recipe from the arm's own worktree. It binds production's port 7867 and holds
`cc-gpu.lock`, so production must be stopped first (owner's call), and restarted after:

```bash
# in the arm's worktree, e.g. wt-parity-base
cd /data/models/slang/nvfp4-work/wt-parity-base
benchmarks/dsv41_baseline/launch_prod.sh stop      # only with the owner's approval
mkdir -p /data/models/slang/nvfp4-work/parity
benchmarks/dsv41_baseline/launch_prod.sh start > /data/models/slang/nvfp4-work/parity/base-server.log 2>&1 &
until curl -sf http://127.0.0.1:7867/health >/dev/null; do sleep 15; done   # ~15+ min for DSpark
```

Then dump, with the branch's tools but pointing at the running server. Take the server's pid from
`ss -ltnp 'sport = :7867'` so the manifest records its SGLANG_* env:

```bash
SERVER_PID=$(ss -ltnp 'sport = :7867' | sed -n 's/.*pid=\([0-9]*\).*/\1/p' | head -1)
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 PYTHONPATH=/data/models/slang/nvfp4-work/wt-parity-base/python \
  taskset -c 22-25 /data/models/slang/.venv-upstream/bin/python $PARITY/dump_logits.py \
  --server-url http://127.0.0.1:7867 --server-pid "$SERVER_PID" \
  --out /data/models/slang/nvfp4-work/parity/base-dump
benchmarks/dsv41_baseline/launch_prod.sh stop
```

Repeat with `wt-parity-branch` and `--out .../branch-dump`. Print `sglang_file` from each manifest and check it is
the arm's own worktree (the interpreter trap in the run protocol). `PYTHONPATH` on the client only decides which
tree the manifest names; the server's tree is the worktree its launcher ran from, and `server_pythonpath` records it.

Requests are sequential (`MAX_RUNNING_REQUESTS = 1`), flush the cache before each prompt, and take minutes for the
long prompts. `--limit 4` is a smoke run.

Noise floor: dump the base twice (`base-dump`, `base-dump2`) and compare them. CPU experts, hot-cache residency and
accumulation order can make two runs of one commit differ; if base vs base is already near the thresholds, the
branch result cannot be read against them.

## 3. Compare

```bash
python3 $PARITY/compare.py /data/models/slang/nvfp4-work/parity/base-dump /data/models/slang/nvfp4-work/parity/branch-dump
echo "EXIT=$?"      # 0 pass, 1 threshold failed, 2 dumps not comparable
```

NumPy only; `--max-kl 1e-3 --min-top1 0.995 --worst 5` are the defaults. Dumps from different prompt sets or token
ids are refused (exit 2). KL is computed on the union of the two top-20 supports plus a lumped tail bucket; the
approximation is documented in `compare.py`. CPU tests: `python -m pytest benchmarks/dsv41_baseline/parity/test_compare.py -q`.

`build_prompts.py` regenerates `prompts.jsonl` from repo files; do not rerun it for a real comparison, since the
checked-in file is the fixed set.
