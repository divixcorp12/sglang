#!/bin/bash
# One-time administrator install. Later start/stop uses narrowly scoped polkit.
set -euo pipefail
export PATH=/usr/sbin:/usr/bin:/sbin:/bin
[[ $EUID == 0 ]] || { echo 'Run this installer once with sudo.' >&2; exit 1; }
[[ $# -le 1 && (${1:-} == '' || ${1:-} == --release-idle-legacy) ]] || {
    echo 'Usage: install.sh [--release-idle-legacy]' >&2; exit 2;
}
source_dir=$(cd -- "$(dirname -- "$0")" && pwd)
runner=$(realpath "$source_dir/../run.sh")
root=/data/models/exl3_exp/google_benchmark
legacy=/sys/fs/cgroup/exl3bench.scope
release_legacy=false
[[ -f $runner && -x $root/build/exl3_cpu_baseline && -x $root/build/exl3_cpu_optimized ]]
id dnikolaidis >/dev/null
if systemctl is-active --quiet exl3bench.service; then
    echo 'Stop exl3bench.service before updating its installed files.' >&2; exit 1;
fi

# Do not kill the user's existing shell. Refuse a transition if anything other
# than one idle shell remains in the old scope, including nested cgroups.
if [[ -f $legacy/cpuset.cpus.partition && $(< "$legacy/cpuset.cpus.partition") != member ]]; then
    [[ ${1:-} == --release-idle-legacy ]] || {
        echo 'Legacy scope is isolated. Close its benchmark jobs, then rerun with --release-idle-legacy from a normal terminal.' >&2
        exit 1
    }
    mapfile -t pids < <(find "$legacy" -name cgroup.procs -exec cat {} + | sort -un)
    [[ ${#pids[@]} -le 1 ]] || {
        echo 'Legacy scope contains more than an idle shell. Finish its jobs first.' >&2; exit 1;
    }
    for pid in "${pids[@]}"; do
        [[ -f /proc/$pid/comm && $(< "/proc/$pid/comm") == bash ]] || {
            echo "Legacy scope has an active process: $pid" >&2; exit 1;
        }
        [[ $(stat -c %u "/proc/$pid") == $(id -u dnikolaidis) ]]
    done
    release_legacy=true
fi

getent group exl3bench >/dev/null || groupadd --system exl3bench
usermod -aG exl3bench dnikolaidis
install -d -o root -g root -m 0755 /usr/local/libexec
install -o root -g root -m 0755 "$source_dir/exl3bench-isolation" /usr/local/libexec/exl3bench-isolation
sed "s|@BENCH_RUNNER@|$runner|" "$source_dir/exl3bench-run" > /usr/local/libexec/exl3bench-run
chown root:root /usr/local/libexec/exl3bench-run
chmod 0755 /usr/local/libexec/exl3bench-run
install -o root -g root -m 0644 "$source_dir/exl3bench.service" /etc/systemd/system/exl3bench.service
install -o root -g root -m 0644 "$source_dir/49-exl3bench.rules" /etc/polkit-1/rules.d/49-exl3bench.rules
restorecon /usr/local/libexec/exl3bench-{isolation,run} /etc/systemd/system/exl3bench.service /etc/polkit-1/rules.d/49-exl3bench.rules
systemd-analyze verify /etc/systemd/system/exl3bench.service
systemctl daemon-reload

if $release_legacy; then
    # Recheck just before releasing, in case a job started during installation.
    systemctl freeze exl3bench.scope
    trap 'systemctl thaw exl3bench.scope' EXIT
    mapfile -t now < <(find "$legacy" -name cgroup.procs -exec cat {} + | sort -un)
    [[ ${now[*]} == "${pids[*]}" ]] || { echo 'Legacy processes changed; reservation left intact.' >&2; exit 1; }
    printf '%s\n' member > "$legacy/cpuset.cpus.partition"
    printf '\n' > "$legacy/cpuset.cpus.exclusive"
    [[ $(< "$legacy/cpuset.cpus.partition") == member ]]
    systemctl thaw exl3bench.scope
    trap - EXIT
    echo 'Released the old scope reservation; its shell was not killed.'
fi
echo 'Installed. Open a new login or use newgrp exl3bench, then:'
echo '  systemctl --no-ask-password start exl3bench.service'
echo '  journalctl -fu exl3bench.service'
echo '  systemctl --no-ask-password stop exl3bench.service'
