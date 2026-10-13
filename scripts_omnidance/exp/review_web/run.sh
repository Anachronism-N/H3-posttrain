#!/bin/bash
# 常驻启动评审服务（崩溃自动重启）。用法: bash run.sh [port]
# 停止: kill $(cat /tmp/review_web_8765.pid)
PORT=${1:-8765}
cd "$(dirname "$0")"
LOG=/tmp/review_web_${PORT}.log
echo $$ > /tmp/review_web_${PORT}.pid
while true; do
    echo "[$(date '+%F %T')] start server.py --port $PORT" >> "$LOG"
    python3 server.py --port "$PORT" >> "$LOG" 2>&1
    sleep 3
done
