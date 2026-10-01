On-demand isolation service for divix01
======================================

Install once with administrator privileges. After that, dnikolaidis in group
exl3bench can start/stop exactly exl3bench.service via polkit without sudo or an
authentication prompt. The rule grants no daemon-reload, unit-file management,
transient-service creation, property changes or management of other services.
Root-owned helpers have no user-supplied cgroup/CPU paths. Benchmark executables,
the runner, arguments and output files execute as dnikolaidis, never as root.

The unit is directly under the root slice, CPUs18..33 and their SMT siblings
54..69. ExecStartPre reserves the isolated partition and verifies effective masks
before launching any benchmark. The benchmark uses workers18..33; siblings are
reserved but idle. Both memory nodes0 and1 remain allowed, without numactl policy.
ExecStopPost returns the cpuset to member and clears exclusivity after benchmark
descendants finish or are killed. Startup failures also roll back the partition.
The helper explicitly refuses overlap with the existing isolated legacy scope.
Do not run the old configure_isolation.sh in parallel with the new service.

Installation
------------
From a normal terminal on divix01 (outside the old isolated shell):

  sudo bash /data/models/slang/nvfp4-work/cc-expert-prediction/dsv41-direct-prod/python/sglang/kernels/jit/csrc/moe/expert_stream/bench/service/install.sh --release-idle-legacy

The flag permits releasing the old reservation only when its entire scope has
at most one dnikolaidis-owned bash process and no benchmark jobs. The installer
does not kill the old shell. It freezes that scope briefly, rechecks the process
list, releases isolation and thaws it. If a job is present, installation refuses
the transition. Finish that job first. Later installs with a member/inactive old
scope do not need the flag. Installing never starts the benchmark automatically.

Files installed root-owned:
  /etc/systemd/system/exl3bench.service
  /etc/polkit-1/rules.d/49-exl3bench.rules
  /usr/local/libexec/exl3bench-isolation
  /usr/local/libexec/exl3bench-run
The installer adds group exl3bench and user membership, applies SELinux contexts,
verifies the unit and reloads systemd. Polkit watches its rules directory.
Open a new login after installation, or run newgrp exl3bench in the current shell.
The service is installed persistently but not enabled at boot: idle machines
retain all CPUs for normal work.

Everyday use (no sudo)
---------------------
  systemctl --no-ask-password start exl3bench.service
  journalctl -fu exl3bench.service
  systemctl status exl3bench.service

Start returns once launched; the benchmark continues in the service. Ctrl-C on
journalctl stops log viewing, not the benchmark. Completion/failure automatically
releases the CPUs. To cancel and release them early:

  systemctl --no-ask-password stop exl3bench.service

After completion:
  systemctl show exl3bench.service -p Result -p ExecMainStatus -p ActiveState

No restart privilege is granted; use stop then start. A second start while running
does not launch another benchmark. On a failed run, examine the journal and the
partial results before starting again. The service uses control-group termination
and a 30s stop timeout; a process stuck in uninterruptible kernel sleep can delay
its termination/release, as with other Linux services. Do not manually release
the partition while a benchmark is still executing.

Settings and output
-------------------
Default: existing native baseline and optimized binaries, 8 alternating rounds,
512 forwards per count1/3/5, 12 baseline /16 optimized workers. Results live in:
  /data/models/exl3_exp/google_benchmark/service-TIMESTAMP-INVOCATION_ID/
The service journal prints that path. All runs use fresh directories.

The launcher reads optional, user-owned files at the start of each invocation:
  /data/models/exl3_exp/google_benchmark/service-rounds.txt
    One positive integer, default8.
  /data/models/exl3_exp/google_benchmark/service-args.txt
    One literal CLI argument per line; blank lines and # comments skipped.
    These are benchmark arguments, never shell code (no source/eval).
Example short first run, create these as your normal user:

  printf '1\n' > /data/models/exl3_exp/google_benchmark/service-rounds.txt
  printf '%s\n' '--benchmark_min_time=8x' '--warmup-forwards=8' \
    > /data/models/exl3_exp/google_benchmark/service-args.txt

Start the service, then verify individual worker CPU lists in each process log.
Remove the two settings files to return to full defaults. Rebuild the benchmark
with CMake before starting if source changed; the service does not compile.
The installed launcher points to this checkout's bench/run.sh. If moving that
checkout, rerun the installer while the service is stopped to update its path.

Verify live isolation while the benchmark runs:
  cat /sys/fs/cgroup/exl3bench.service/cpuset.cpus.partition
  cat /sys/fs/cgroup/exl3bench.service/cpuset.cpus.effective
  cat /sys/fs/cgroup/exl3bench.service/cpuset.mems.effective
Expected: isolated, 18-33,54-69, 0-1. The cgroup may disappear after completion.
The benchmark checks worker affinity and saved outputs. Unit parsing and helper
rollback/cleanup were checked before installation; actual kernel transitions and
polkit authorization need the administrator installation/first service run.

Administrator removal
---------------------
  sudo systemctl stop exl3bench.service
  sudo rm -f /etc/systemd/system/exl3bench.service \
    /etc/polkit-1/rules.d/49-exl3bench.rules \
    /usr/local/libexec/exl3bench-isolation /usr/local/libexec/exl3bench-run
  sudo systemctl daemon-reload
Group membership and saved results can remain; without the rule it grants no
service-management permission.

References:
https://www.kernel.org/doc/html/latest/admin-guide/cgroup-v2.html
https://github.com/systemd/systemd/blob/v257/src/core/dbus-util.c
