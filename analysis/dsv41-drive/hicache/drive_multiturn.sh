#!/usr/bin/env bash
# Hierarchical KV cache under GPU-KV overflow: three cold servers on the production recipe, each running
# multiturn.py (two interleaved conversations over 4096-token documents, three turns each):
#   big            the recipe's KV pool (~200k tokens): every prefix stays on the GPU, the best case;
#   small          --max-total-tokens 6144, one conversation fits and two do not: revisits recompute their prefix;
#   small-hicache  as small, plus --enable-hierarchical-cache --hicache-ratio 2 --hicache-size 0
#                  --hicache-write-policy write_through: revisits can reload their prefix from host memory.
# Two diagnostic arms, two turns each, for why big's revisits reused nothing (0 cached tokens on 4k prompts):
#   big-noreplay   big without --enable-decoder-swa-bounded-replay;
#   big-2k         big over 2048-token documents, the prompt size where earlier soaks did reuse prefixes;
#   big-tails16    big with --swa-prefix-tails 16: a larger SWA pool, in case a 4k prefill evicts the other
#                  conversation's SWA tail from the 3584-slot pool;
#   big-hicache    big plus the hierarchical cache: a host-backed SWA tail is a valid match boundary.
# Output under /mnt/nvme1/hicache/mt-<arm>/. Production must be stopped.
# Usage: drive_multiturn.sh <worktree> [arm ...]
set -u
WT=${1:?worktree}
shift
ARMS=${*:-big small small-hicache}
H=$WT/benchmarks/dsv41_baseline
PY=/data/models/slang/.venv/bin/python
T=/mnt/nvme1/hicache
PORT=30026
SMALL="--max-total-tokens 6144"
HICACHE="--enable-hierarchical-cache --hicache-ratio 2 --hicache-size 0 --hicache-write-policy write_through"
say() { echo "$(date +%T) $*" | tee -a $T/drive-multiturn.log; }
mkdir -p $T

# Disk lock first, then the GPU lock: the order the other drivers on this box use.
exec 8>/data/models/slang/nvfp4-work/rowimg-disk.lock
say "waiting for rowimg-disk.lock"; flock 8
exec 9>/data/models/slang/nvfp4-work/cc-gpu.lock
say "waiting for cc-gpu.lock"; flock 9
say "locks held; wt=$WT head=$(git -C $WT rev-parse HEAD) dirty=$(git -C $WT status --porcelain --untracked-files=no | wc -l)"
ss -ltn 'sport = :7867' | grep -q LISTEN && { say "production up; refusing"; exit 1; }
PYTHONPATH=$WT/python $PY -c 'import sglang; print("sglang from", sglang.__file__)' | tee -a $T/drive-multiturn.log

mapfile -t ENV < <(PYTHONPATH=$H $PY -c "
import arm_env
e = arm_env.base_env()
e['PYTHONPATH'] = '$WT/python'
e['PYTHONUNBUFFERED'] = '1'
for k, v in e.items(): print(f'{k}={v}')
")
[ ${#ENV[@]} -gt 0 ] || { say "arm_env failed"; exit 1; }
mapfile -t BASE_ARGV < <(PYTHONPATH=$H $PY -c "
import arm_env
print(*arm_env.ServerArgs(port=$PORT).argv(), sep='\n')
")
CORES=$(PYTHONPATH=$H $PY -c "import arm_env; print(arm_env.SERVER_CORES)")
MODEL=$(PYTHONPATH=$H $PY -c "import arm_env; print(arm_env.MODEL_PATH)")

for arm in $ARMS; do
  doc=4096; turns=3; drop=""
  case $arm in
    big) extra="" ;;
    small) extra="$SMALL" ;;
    small-hicache) extra="$SMALL $HICACHE" ;;
    big-noreplay) extra=""; turns=2; drop="--enable-decoder-swa-bounded-replay" ;;
    big-2k) extra=""; turns=2; doc=2048 ;;
    big-tails16) extra="--swa-prefix-tails 16"; turns=2 ;;
    big-hicache) extra="$HICACHE"; turns=2 ;;
    *) say "unknown arm $arm"; exit 2 ;;
  esac
  OUT=$T/mt-$arm
  rm -rf $OUT; mkdir -p $OUT
  printf '%s\n' "${ENV[@]}" > $OUT/env.txt
  ARGV=()
  for a in "${BASE_ARGV[@]}"; do [ "$a" = "$drop" ] || ARGV+=("$a"); done
  ARGV+=($extra)
  printf '%s\n' "${ARGV[@]}" > $OUT/argv.txt
  say "arm $arm: extra='$extra' drop='$drop' doc=$doc turns=$turns"
  cd $WT
  taskset -c $CORES env "${ENV[@]}" "${ARGV[@]}" > $OUT/server.log 2>&1 &
  SPID=$!
  healthy=0
  for i in $(seq 1 180); do
    sleep 5
    kill -0 $SPID 2>/dev/null || break
    curl -sf -m 60 localhost:$PORT/health >/dev/null && { healthy=1; break; }
  done
  if [ $healthy = 1 ]; then
    say "arm $arm healthy"
    taskset -c 8-15 $PY $WT/analysis/dsv41-drive/hicache/multiturn.py --port $PORT --model $MODEL \
      --text $WT/DSV41_REFERENCE.md --doc-tokens $doc --turns $turns --out $OUT/turns.jsonl 2>&1 | tee -a $OUT/client.log
    say "arm $arm client rc=${PIPESTATUS[0]}"
  else
    say "arm $arm never became healthy (see $OUT/server.log)"
  fi
  kill -TERM $SPID 2>/dev/null
  for i in $(seq 1 120); do kill -0 $SPID 2>/dev/null || break; sleep 1; done
  kill -KILL $SPID 2>/dev/null
  pkill -f "sglang.launch_server.*--port $PORT" 2>/dev/null
  for i in $(seq 1 60); do [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && break; sleep 2; done
  say "arm $arm stopped; gpu apps: '$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)'"
done
say "DRIVER DONE"
