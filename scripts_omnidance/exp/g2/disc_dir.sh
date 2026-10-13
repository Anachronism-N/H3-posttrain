#!/bin/bash
# disc_dir.sh <eval_out_dir ...>: G2 discriminator scores for eval dirs that have manifest.csv + metrics/kps
R=/apdcephfs_gy4/share_302533218/cedricnie/diff; A=""
for D in "$@"; do A="$A $D/manifest.csv $D/metrics/kps"; done
cd $R; /opt/venv_od/bin/python scripts_omnidance/exp/g2/mert_val.py $(for D in "$@"; do echo $D/manifest.csv; done) > /dev/null 2>&1
/opt/venv_od/bin/python scripts_omnidance/exp/g2/score_disc.py $R/logs_omnidance/exp/G2/disc_v1_best.pt $A 2>/dev/null | grep "^|"
