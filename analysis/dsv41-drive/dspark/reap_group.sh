#!/bin/bash
# Sourced by run_capture_pair.sh. A capture starts its server as the leader of its own session and records the group
# in OUT/NAME/server.pgid (spec_margin_capture.launch_server). If the capture dies without stopping it (it was killed,
# or crashed), the server and its scheduler, detokenizer and resource_tracker children outlive it, reparented to PID 1,
# and hold the GPU.
#
# reap_server_group PGIDFILE LABEL [TERM_WAIT_S=30]: if that group has a live member, log it, SIGTERM the group, wait up
# to TERM_WAIT_S, SIGKILL it, and wait until no member is left. By PGID only, never by name. Returns 0 when the group
# is gone (whether or not it had to be killed), 1 when a member survived the SIGKILL.
reap_server_group() {
  local file=$1 label=$2 wait_s=${3:-30} pgid i
  [ -r "$file" ] || return 0
  pgid=$(<"$file")
  [[ $pgid =~ ^[0-9]+$ ]] && [ "$pgid" -gt 1 ] || return 0
  kill -0 -- "-$pgid" 2>/dev/null || return 0
  echo "REAP $label: server group $pgid still has members after the capture ended: $(ps -o pid=,comm= -g "$pgid" | tr -s ' ' | tr '\n' ';')"
  kill -TERM -- "-$pgid" 2>/dev/null
  for ((i = 0; i < wait_s * 10; i++)); do
    kill -0 -- "-$pgid" 2>/dev/null || { echo "REAP $label: group $pgid ended on SIGTERM"; return 0; }
    sleep 0.1
  done
  echo "REAP $label: group $pgid still alive after ${wait_s} s; SIGKILL"
  kill -KILL -- "-$pgid" 2>/dev/null
  for ((i = 0; i < 100; i++)); do
    kill -0 -- "-$pgid" 2>/dev/null || { echo "REAP $label: group $pgid gone"; return 0; }
    sleep 0.1
  done
  echo "REAP $label: group $pgid survived SIGKILL: $(ps -o pid=,stat=,comm= -g "$pgid" | tr -s ' ' | tr '\n' ';')"
  return 1
}

# server_clear PORT LABEL: nothing listens on PORT and no process holds the GPU; otherwise say what does. 0 when clear.
server_clear() {
  local port=$1 label=$2 bad=0 apps
  if ss -ltnH "sport = :$port" 2>/dev/null | grep -q .; then
    echo "NOT CLEAR $label: something still listens on port $port: $(ss -ltnpH "sport = :$port" 2>/dev/null | tr -s ' ')"
    bad=1
  fi
  apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr '\n' ' ')
  if [ -n "${apps// /}" ]; then
    echo "NOT CLEAR $label: GPU still held by pid(s) $apps"
    bad=1
  fi
  return $bad
}
