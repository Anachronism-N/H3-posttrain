#!/bin/bash
# evalw_after_train.sh: wait until no training runs on this node, then become an eval worker (frozen copy of eval_sched.sh)
while ps -eo args | grep -q '^/opt/venv_h3/bin/python.*model_training/train.py'; do sleep 120; done; sleep 60
cp /apdcephfs_gy4/share_302533218/cedricnie/diff/scripts_omnidance/exp/h3ar/eval_sched.sh /tmp/eval_sched_run.sh; exec bash /tmp/eval_sched_run.sh
