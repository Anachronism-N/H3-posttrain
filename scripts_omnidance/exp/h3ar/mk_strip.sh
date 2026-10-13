#!/bin/bash
# mk.sh <case> <frames "a b c"> <out.png> <dir1> <dir2> ...  -> rows = dirs, cols = frames, upper body crop
E=/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp
c=$1; fr=$2; out=$3; shift 3; W=/tmp/mk_$$; mkdir -p $W; i=0; ins=""
sel=$(for f in $fr; do printf "eq(n\\\\,%s)+" $f; done); sel=${sel%+}; n=$(echo $fr | wc -w)
for d in "$@"; do cp $E/$d/joint/${c}_${c}.mp4 $W/$i.mp4 || continue
  timeout 120 ffmpeg -loglevel error -y -i $W/$i.mp4 -vf "select='$sel',scale=480:832,crop=480:400:0:40,tile=${n}x1" -frames:v 1 $W/$i.png && ins="$ins -i $W/$i.png" && i=$((i+1)); done
ffmpeg -loglevel error -y $ins -filter_complex vstack=$i $out; rm -rf $W; echo done $i
