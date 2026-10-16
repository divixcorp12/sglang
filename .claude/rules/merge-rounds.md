# Merge rounds and agent hand-offs

A merge itself takes minutes. What makes it take an hour is re-testing: a C++ change gives every JIT module a new build
key (50-100 s per expert-stream host variant), and the GPU groups run one at a time behind `cc-gpu.lock`. On 2026-10-16
a consolidation merge to master took about an hour. The git merges were quick; the time went to three avoidable things:

1. Six branches and a settings change were added one at a time **after** the test round had started. Each addition
   meant another round.
2. The lead's additions reached the merging agent while it was inside a long test run. It reported the old state and
   went idle, and about 20 minutes passed before it picked them up.
3. The final round reran every GPU group, although the late additions touched no GPU kernel: one was a settings file,
   the others Python and analysis code.

## Rules

- **Fix the merge set before the first test round.** List every branch and settings change in one message, get the
  owner's yes, then start. Something found later waits for the next round, unless the owner says it must go in.
- **When something must go in mid-round, send it as one consolidated message.** Don't drip it in over several messages.
  Confirm the agent has acted on it, from its next report naming it, before assuming it did. An idle notification
  that doesn't mention it means the agent hasn't seen it: send a short pointer.
- **Retest only what the late addition affects.** Follow `divix01-run-protocol.md`, "Run only the tests your change
  affects":
  - A settings-only change (`arm_env.py`, `launch_prod.sh`) reruns `benchmarks/dsv41_baseline` tests and no GPU group.
  - A Python or analysis change reruns its own test files.
  - A C++ or kernel change reruns the GPU groups that build it.
- **The merge base is measured once per merge.** Its counts carry over to later rounds of the same merge. Don't rerun
  the base for an addition.
- **When the owner is waiting, say so in the instruction**, and name the exact test selection, so the agent doesn't
  default to the full set.
