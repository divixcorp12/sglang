#!/usr/bin/env bash
# Run on the laptop, once, when released. Exports base3 at the current HEAD sha, builds the three trees, launches ONE chain.
set -e
D=/data/models/slang/nvfp4-work/t2-mutants
cd /home/dimitri/data/divix/sglang-nvfp4-worktrees/dsv41
HERE=$(cd "$(dirname "$0")" && pwd)
SHA=$(git rev-parse HEAD)
echo "base3 sha: $SHA"
scp -q $HERE/mutate.py $HERE/mutate_gen.py $HERE/run_gen.sh $HERE/chain.sh $HERE/summarize3.py $HERE/cond.py divix01:$D/
ssh divix01 "cd $D && chmod +x run_gen.sh chain.sh && test ! -e chain.lock && rm -rf base3 tree3 tree2b tree1x && mkdir base3 && echo $SHA > base3.sha"
git archive $SHA python test scripts analysis/dsv41-drive | ssh divix01 "tar -x -C $D/base3"
ssh divix01 "cd $D && cp -r base3 tree3 && cp -r base2 tree2b && cp -r base tree1x && for t in base tree1x base2 tree2b base3 tree3; do printf '%s ' \$t; md5sum \$t/python/sglang/kernels/jit/csrc/moe/exl3_ram_miss_host.cpp | cut -c1-8; done"
ssh divix01 "cd $D && python3 cond.py"
ssh -f divix01 "cd $D && nohup ./chain.sh > chain.out 2>&1 < /dev/null &"
sleep 5
ssh divix01 "pgrep -af 'chain.sh' | grep -v pgrep; echo count=\$(pgrep -f 'bash ./chain.sh' | wc -l); ls $D/chain.lock $D/chain.refused 2>&1"
