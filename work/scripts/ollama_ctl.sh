#!/usr/bin/env bash
# start/stop a private experiment server (pid file per port; avoids pkill -f matching unrelated shells)
# usage: work/scripts/ollama_ctl.sh start [num_parallel] [port] | stop [port]
W="$(cd "$(dirname "$0")/.." && pwd)"
case "$1" in
  start) P=${3:-11435}; nohup $W/scripts/ollama_serve.sh "${2:-1}" "$P" > $W/ollama_serve_$P.log 2>&1 &
         echo $! > $W/ollama_serve_$P.pid; sleep 3; curl -s -o /dev/null -w "server $P %{http_code}\n" 127.0.0.1:$P/api/tags ;;
  stop)  P=${2:-11435}; [ -f $W/ollama_serve_$P.pid ] && kill "$(cat $W/ollama_serve_$P.pid)" 2>/dev/null; rm -f $W/ollama_serve_$P.pid; sleep 2 ;;
esac
