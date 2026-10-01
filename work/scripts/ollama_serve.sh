#!/usr/bin/env bash
# Private Ollama server for the experiments: reads the system model store, listens on 127.0.0.1:11435.
# NUM_PARALLEL / MAX_LOADED_MODELS keep the SLM footprint inside the 8 GB budget (checked in results/resources/).
# usage: work/scripts/ollama_serve.sh [num_parallel] [port]   (defaults 1, 11435)
export OLLAMA_HOST=127.0.0.1:${2:-11435}
export OLLAMA_MODELS=${OLLAMA_MODELS:-/usr/share/ollama/.ollama/models}
export OLLAMA_NOPRUNE=1
export OLLAMA_NUM_PARALLEL=${1:-1}
export OLLAMA_MAX_LOADED_MODELS=1
export OLLAMA_KEEP_ALIVE=30m
export OLLAMA_FLASH_ATTENTION=1
exec ${OLLAMA_BIN:-ollama} serve
