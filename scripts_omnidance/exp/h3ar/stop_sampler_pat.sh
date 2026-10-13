#!/bin/bash
# stop_sampler_pat.sh <pattern>: kill only run_h3ar_node.sh wrappers / sample_h3_ar.py processes whose cmdline contains <pattern>
# (matches by program prefix, so the calling ssh shell, whose cmdline also contains <pattern>, is never killed)
P=$1; [ -z "$P" ] && exit 1
for pid in $(ps -eo pid=); do
  [ "$pid" = "$$" ] && continue
  a=$(tr '\0' ' ' < /proc/$pid/cmdline 2>/dev/null) || continue
  case "$a" in
    "bash /"*run_h3ar_node.sh*"$P"*|"/opt/venv_h3/bin/python /"*sample_h3_ar.py*"$P"*) echo "kill $pid"; kill $pid ;;
  esac
done
