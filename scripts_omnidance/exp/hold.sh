#!/bin/bash
# GPU holder control on the local node.
#   hold.sh stop  "0,1,2"   -> kill holder processes bound to these GPUs (by PID, holder only)
#   hold.sh start "0,1,2"   -> start holders on these GPUs if no other process is using them
#   hold.sh idle            -> start holders on every GPU that has no compute process
#   hold.sh status
PY=/apdcephfs_gy2/share_303214315/cedricnie/miniconda3/envs/longlive/bin/python
HOLDER=/apdcephfs_gy4/share_302533218/cedricnie/diff/scripts_omnidance/exp/hold_gpu.py
cmd=$1; gpus=${2:-}

holder_pids_for_gpu() {  # $1 = physical gpu index
  for p in $(pgrep -f "[h]old_gpu.py"); do
    cvd=$(tr '\0' '\n' < /proc/$p/environ 2>/dev/null | sed -n 's/^CUDA_VISIBLE_DEVICES=//p')
    og=$(tr '\0' '\n' < /proc/$p/environ 2>/dev/null | sed -n 's/^OCCUPY_GPU=//p')
    if [ -n "$cvd" ]; then phys=$cvd; else phys=$og; fi
    [ "$phys" = "$1" ] && echo $p
  done
}

gpu_busy_nonholder() {  # returns 0 if a non-holder compute process is on gpu $1
  uuid=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i $1)
  for p in $(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | awk -F', ' -v u="$uuid" '$1==u{print $2}'); do
    grep -q hold_gpu /proc/$p/cmdline 2>/dev/null || return 0
  done
  return 1
}

start_one() {
  g=$1
  [ -n "$(holder_pids_for_gpu $g)" ] && return
  gpu_busy_nonholder $g && { echo "gpu $g busy, skip holder"; return; }
  CUDA_VISIBLE_DEVICES=$g OCCUPY_GPU=0 setsid nohup $PY $HOLDER > /tmp/hold_gpu_$g.log 2>&1 < /dev/null &
  echo "holder started gpu $g"
}

case $cmd in
  stop)
    for g in ${gpus//,/ }; do
      touch /tmp/od_reserve_$g
      pids=$(holder_pids_for_gpu $g)
      [ -n "$pids" ] && kill $pids && echo "holder stopped gpu $g ($pids)"
    done
    sleep 3 ;;
  start)
    for g in ${gpus//,/ }; do start_one $g; done ;;
  idle)
    for g in 0 1 2 3 4 5 6 7; do
      if [ -f /tmp/od_reserve_$g ] && [ $(( $(date +%s) - $(stat -c %Y /tmp/od_reserve_$g) )) -lt 1200 ]; then continue; fi
      start_one $g
    done ;;
  status)
    for g in 0 1 2 3 4 5 6 7; do echo "gpu $g holder=[$(holder_pids_for_gpu $g | tr '\n' ' ')] $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader -i $g)"; done ;;
esac
