#!/bin/bash
# stage_base.sh <src fused dir on cephfs> <dst local dir>: copy fused transformer shards to local disk, symlink every other part to /dev/shm/h3/Ref2VA
SRC=$1; DST=$2
mkdir -p $DST/transformer
for d in /dev/shm/h3/Ref2VA/*; do b=$(basename $d); [ "$b" = transformer ] || ln -sfn $d $DST/$b; done
cp -n $SRC/transformer/*.safetensors $DST/transformer/
for f in /dev/shm/h3/Ref2VA/transformer/*; do case $f in *.safetensors) ;; *) ln -sfn $f $DST/transformer/;; esac; done
echo "[stage] done $(ls $DST/transformer/*.safetensors | wc -l) shards $(date)"
