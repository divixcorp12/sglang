#!/usr/bin/env bash
# usage: run_gen.sh <tree> <label>   host-facing tests against $D/<tree>; log $D/log-<label>.txt
D=/data/models/slang/nvfp4-work/t2-mutants
tree=$1; label=$2; shift 2
cd $D/$tree
export CUDA_VISIBLE_DEVICES=9 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 SGLANG_JIT_CACHE_DIR=$D/jit PYTHONPATH=$D/$tree/python PYTHONDONTWRITEBYTECODE=1
K=test/registered/unit/kernels
M=test/registered/unit/layers/moe
FILES="$K/test_exl3_ram_miss_split.py $K/test_exl3_ram_miss_thread.py $K/test_exl3_ram_miss_tier.py $K/test_exl3_ram_miss_advisory.py $K/test_exl3_ram_miss_attach_lanes.py $K/test_exl3_ram_miss_wrap.py $K/test_exl3_ram_miss_device_args.py $K/test_exl3_lease_block.py $K/test_exl3_ram_miss_stage_trace.py $K/test_exl3_ram_miss_stage_trace_causal.py $K/test_exl3_ram_miss_stage_trace_lanes.py $K/test_exl3_ram_miss_trace_export.py $M/test_exl3_ram_miss_service.py $M/test_exl3_ram_miss_tables.py"
[ -n "$FILES_OVERRIDE" ] && FILES="$FILES_OVERRIDE"
# every test file that exists in this tree (later bases add lease tests; the host-facing set is fixed to the 14 above)
rm -rf $D/bt/$label
taskset -c 0-63 /data/models/slang/.venv/bin/python -m pytest $FILES -p no:cacheprovider -q --tb=line -rfE --basetemp $D/bt/$label "$@" > $D/log-$label.txt 2>&1
echo "exit=$?" >> $D/log-$label.txt
