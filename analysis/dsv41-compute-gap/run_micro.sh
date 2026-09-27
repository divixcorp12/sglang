#!/usr/bin/env bash
# E1-E3 microbenchmarks on divix01. Holds cc-gpu.lock (blocking) for the whole run, cores 32-63.
# Usage (from the worktree root): run_micro.sh OUTDIR [wo_a] [gemv] [moe]
set -u
OUT=${1:?outdir}; shift
PARTS=${*:-wo_a gemv moe}
WT=$(pwd)
PY=/data/models/slang/.venv/bin/python
D=$WT/analysis/dsv41-compute-gap
export SGLANG_EXL3_SRC=/data/models/slang/nvfp4-work/exllamav3
export SGLANG_EXL3_BUILD_DIR=/data/models/slang/nvfp4-work/cc-expert-prediction/exl3-build
export EXL3_VARIANT_SRC=/data/models/slang/nvfp4-work/exl3-variant-src
export CUDA_HOME=/usr/local/cuda-13.2 PATH=/usr/local/cuda-13.2/bin:$PATH TORCH_CUDA_ARCH_LIST=12.0
export OMP_NUM_THREADS=8 PYTHONPATH=$WT/python
mkdir -p $OUT
exec 9>/data/models/slang/nvfp4-work/cc-gpu.lock
echo "$(date '+%F %T') waiting for cc-gpu.lock" >> $OUT/run.log; flock 9
echo "$(date '+%F %T') cc-gpu.lock held" >> $OUT/run.log
git log -1 --oneline > $OUT/commit.txt
$PY -c 'import sglang; print("sglang from", sglang.__file__)' > $OUT/sglang_file.txt
nvidia-smi --query-gpu=name,clocks.sm,clocks.mem,temperature.gpu,power.draw --format=csv > $OUT/gpu.txt
r() {  # r LOGNAME cmd...
    local log=$OUT/$1.log; shift
    echo "cmd: $*" > $log
    taskset -c 32-63 "$@" >> $log 2>&1
    echo "EXIT=$?" >> $log
    echo "$(date '+%F %T') $(basename $log) $(tail -1 $log)" >> $OUT/run.log
}
for part in $PARTS; do
  case $part in
  wo_a) r wo_a $PY $D/wo_a_bench.py --out $OUT/wo_a.json ;;
  gemv)
    for v in prod knobs gemv_d2 gemv_d3 gemv_d6 gemv_nostage prod knobs; do
      tag=$v; [ -e $OUT/gemv-$v.json ] && tag=$v-rep
      r gemv-$tag $PY $D/gemv_bench.py --variant $v --out $OUT/gemv-$tag.json --save $OUT/gemv-$tag.pt
      [ $v = prod ] || r gemv-cmp-$tag $PY $D/gemv_bench.py --compare $OUT/gemv-prod.pt $OUT/gemv-$tag.pt
    done ;;
  moe)
    r moe-prod $PY $D/moe_bench.py --variant prod --out $OUT/moe-prod.json --save $OUT/moe-prod.pt
    for cfg in knobs:0:0 knobs:8:0 knobs:16:0 knobs:24:0 knobs:0:128 moe_sh2:0:0 moe_sh4:0:0 moe_fs2:0:0 knobs:0:0 prod:0:0; do
      IFS=: read v w t <<< "$cfg"
      tag=$v-w$w-t$t; [ -e $OUT/moe-$tag.json ] && tag=$tag-rep
      [ $v = prod ] && tag=prod-rep
      r moe-$tag env EXL3_MOE_GROUP_WIDTH=$w EXL3_MOE_TILE_N=$t $PY $D/moe_bench.py --variant $v --out $OUT/moe-$tag.json --save $OUT/moe-$tag.pt
      r moe-cmp-$tag $PY $D/moe_bench.py --compare $OUT/moe-prod.pt $OUT/moe-$tag.pt
    done ;;
  moe2)
    [ -e $OUT/moe-prod.pt ] || r moe-prod $PY $D/moe_bench.py --variant prod --out $OUT/moe-prod.json --save $OUT/moe-prod.pt
    for v in moe_sh4 moe_sh5 moe_sh6 moe_sh4_fs4 moe_sh4_fs2 prod moe_sh4; do
      tag=$v; [ -e $OUT/moe-$tag.json ] && tag=$v-rep
      r moe-$tag $PY $D/moe_bench.py --variant $v --out $OUT/moe-$tag.json --save $OUT/moe-$tag.pt
      r moe-cmp-$tag $PY $D/moe_bench.py --compare $OUT/moe-prod.pt $OUT/moe-$tag.pt
    done ;;
  esac
done
echo "$(date '+%F %T') MICRO DONE" >> $OUT/run.log
