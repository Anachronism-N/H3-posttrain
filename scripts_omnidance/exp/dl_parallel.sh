#!/bin/bash
# dl_parallel.sh <url> <out> <nparts> : resumable multi-connection range download through star-proxy (no aria2c on the nodes).
# Each part is fetched with curl -C - in a retry loop; parts are concatenated and the size is checked at the end.
U=$1; O=$2; N=${3:-32}
export http_proxy=http://star-proxy.oa.com:3128 https_proxy=http://star-proxy.oa.com:3128
L=$(curl -sI "$U" | tr -d '\r' | awk 'tolower($1)=="content-length:"{print $2}' | tail -1)
[ -n "$L" ] || { echo "no content-length"; exit 1; }
P=$O.parts; mkdir -p $P; S=$(( (L + N - 1) / N ))
for i in $(seq 0 $((N-1))); do
  ( a=$((i*S)); b=$(( (i+1)*S - 1 )); [ $b -ge $L ] && b=$((L-1)); want=$((b-a+1)); f=$P/$(printf %03d $i)
    for t in $(seq 1 500); do
      have=$(stat -c %s $f 2>/dev/null || echo 0); [ "$have" -ge "$want" ] && break
      curl -sS -L --max-time 1200 -r $((a+have))-$b "$U" >> $f 2>/dev/null; sleep 2
    done ) &
done
wait
tot=$(cat $P/* | wc -c); echo "[dl] $O got $tot / $L"
[ "$tot" = "$L" ] && cat $P/* > $O.tmp && mv $O.tmp $O && rm -rf $P && echo "[dl] done $O"
