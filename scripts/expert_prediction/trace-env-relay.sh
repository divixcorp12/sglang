#!/usr/bin/env bash
# Carries the launcher's environment across a privileged trace wrapper.
# sudo resets the environment (env_reset), so a root profiler would start the
# server without its SGLANG_* configuration.
#   save <env-file> <command...>     write this environment (NUL-separated, 0600), exec command
#   restore <env-file> <command...>  re-export the saved environment, delete the file, exec command
# On restore, variables the profiler set for injection keep the profiler's value.
set -euo pipefail

mode=${1:?save or restore}
env_file=${2:?env file}
shift 2
[ "$#" -gt 0 ] || { echo "trace-env-relay: command is required" >&2; exit 2; }

case "$mode" in
save)
    (umask 077 && env -0 > "$env_file")
    exec "$@"
    ;;
restore)
    [ -r "$env_file" ] || { echo "trace-env-relay: cannot read $env_file" >&2; exit 2; }
    while IFS= read -r -d '' entry; do
        name=${entry%%=*}
        [[ "$name" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || continue
        case "$name" in
        LD_PRELOAD | *INJECTION* | NSYS_* | NSIGHT_*)
            [ -n "${!name+x}" ] && continue
            ;;
        esac
        export "$entry"
    done < "$env_file"
    rm -f -- "$env_file"
    exec "$@"
    ;;
*)
    echo "trace-env-relay: mode must be save or restore" >&2
    exit 2
    ;;
esac
