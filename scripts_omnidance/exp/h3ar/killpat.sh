#!/bin/bash
# killpat.sh <python-script-basename...>: kill python processes whose argv contains one of the names (excludes shells)
for n in "$@"; do
  for p in $(pgrep -f "python.*$n"); do
    [ "$p" = "$$" ] && continue
    grep -q "python" /proc/$p/comm 2>/dev/null || continue
    kill $p 2>/dev/null
  done
done
sleep 3; echo "left: $(for n in "$@"; do pgrep -f "python.*$n"; done | wc -l)"
