#!/usr/bin/env bash
# ONE launch. base3 = the sha given as $1 (exported before this script runs).
D=/data/models/slang/nvfp4-work/t2-mutants
cd $D
[ -e chain.lock ] && { echo "chain.lock exists: another chain was started" > chain.refused; exit 3; }
echo $$ > chain.lock
G="python3 mutate_gen.py"
ALL3="H01 H02 H03 H04 H05 H06 H07 H08 H09 H10 H11 H12 H13 H14 H15 H16 H17 H18 P01 P02 P03 U01 H19 H20"
# preflight: every base dir and tree exists and every mutant applies to every base, BEFORE anything runs
for b in base base2 base3; do test -d $D/$b || { echo "preflight: $D/$b missing" > chain.refused; rm -f chain.lock; exit 4; }; done
for t in tree1x tree2b tree3; do test -d $D/$t || { echo "preflight: $D/$t missing" > chain.refused; rm -f chain.lock; exit 4; }; done
DRYRUN=1 $G base3 tree3 /dev/null pf3- $ALL3 > preflight.out 2>&1 || { echo "preflight: a mutant does not apply on base3 (see preflight.out)" > chain.refused; rm -f chain.lock; exit 5; }
DRYRUN=1 $G base2 tree2b /dev/null pf2- H01 H02 H19 H20 >> preflight.out 2>&1 || { echo "preflight: base2" > chain.refused; rm -f chain.lock; exit 5; }
DRYRUN=1 $G base tree1x /dev/null pf1- H19 H20 >> preflight.out 2>&1 || { echo "preflight: base (base 1)" > chain.refused; rm -f chain.lock; exit 5; }
[ -z "$RESUME" ] && rm -f results3.jsonl results2b.jsonl results1x.jsonl
rm -f alldone3.txt chain.stopped
stopped() { [ -e stop.flag ] && { echo "stopped by stop.flag after the last finished mutant" > chain.stopped; rm -f chain.lock; exit 0; }; }
./run_gen.sh tree3 base3-baseline
$G base3 tree3 results3.jsonl b3- $ALL3
stopped
./run_gen.sh tree2b base2-baseline
stopped
$G base2 tree2b results2b.jsonl b2- H01 H02 H19 H20
stopped
stopped
./run_gen.sh tree1x b1-baseline-host
$G base tree1x results1x.jsonl b1- H19 H20
# invocation baselines for the file-specific mutants (P01/P02, P03, U01) on pristine base 1, so a run whose
# selector matched fewer tests than intended cannot be mistaken for a kill or a survivor
K=test/registered/unit/kernels; M=test/registered/unit/layers/moe
FILES_OVERRIDE="$M/test_exl3_verify_expert_mirror.py" ./run_gen.sh tree1x b1-baseline-verify
FILES_OVERRIDE="$M/test_exl3_expert_layout.py" ./run_gen.sh tree1x b1-baseline-layout
FILES_OVERRIDE="$K/test_uring_file_reader.py $M/test_exl3_verify_expert_mirror.py $M/test_exl3_mirror_row_source.py $M/test_exl3_row_reader.py" ./run_gen.sh tree1x b1-baseline-uring
echo ALLDONE > alldone3.txt
rm -f chain.lock
