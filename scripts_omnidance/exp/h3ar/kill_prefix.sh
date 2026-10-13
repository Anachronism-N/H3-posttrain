#!/bin/bash
# kill_prefix.sh <cmdline-prefix>...: kill processes whose full cmdline STARTS with one of the prefixes (never the caller's shell)
for pid in $(ps -eo pid=); do
  [ "$pid" = "$$" ] || [ "$pid" = "$PPID" ] && continue
  a=$(tr '\0' ' ' < /proc/$pid/cmdline 2>/dev/null) || continue
  for p in "$@"; do case "$a" in "$p"*) echo "kill $pid ${a:0:80}"; kill $pid ;; esac; done
done
