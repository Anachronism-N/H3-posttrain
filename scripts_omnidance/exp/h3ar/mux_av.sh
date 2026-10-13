#!/bin/bash
# mux_av.sh <data_dir>: video/<sid>.mp4 + audio/<sid>.wav -> av/<sid>.mp4 (video copy, stereo AAC), 16 parallel; skips done
D=$1; mkdir -p $D/av
ls $D/video | grep -v tmp | sed 's/\.mp4$//' | xargs -P 16 -I{} sh -c "[ -s $D/av/{}.mp4 ] || /usr/local/bin/ffmpeg -loglevel error -y -i $D/video/{}.mp4 -i $D/audio/{}.wav -map 0:v -map 1:a -c:v copy -c:a aac -ac 2 -b:a 192k $D/av/{}.tmp.mp4 && mv $D/av/{}.tmp.mp4 $D/av/{}.mp4"
echo "[mux] $(ls $D/av | grep -vc tmp) done"
