#!/bin/bash
# fanout_h3.sh <src> <dst...>: copy /dev/shm/h3 and /opt/venv_h3 node-to-node
S=$1; shift
for D in "$@"; do
  ( ssh $S "cd /dev/shm && tar cf - h3" | ssh $D "mkdir -p /dev/shm && cd /dev/shm && tar xf - && echo $D h3 \$(du -sh /dev/shm/h3 | cut -f1)"
    ssh $S "cd /opt && tar cf - venv_h3" | ssh $D "cd /opt && rm -rf venv_h3 && tar xf - && echo $D venv_h3 ok" ) &
done
wait; echo FANOUT_DONE
