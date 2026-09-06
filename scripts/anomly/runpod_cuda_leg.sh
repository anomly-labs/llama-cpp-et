#!/usr/bin/env bash
# Copyright (c) 2026 Anomly, Inc. All rights reserved. Author: Ry Bruscoe.
# Self-contained NVIDIA exact-profile leg for a fresh RunPod CUDA-devel pod: build the fork's
# default (exact-profile) CUDA backend, run the canonical prompt through the public 0.5B b-posit8
# GGUF, print the whole-graph dump hash and the board verdict. One marker line at the end.
set -uo pipefail
export PATH=/usr/local/cuda/bin:$PATH CUDACXX=/usr/local/cuda/bin/nvcc
export DEBIAN_FRONTEND=noninteractive
BOARD=96edc94d772072a9
say(){ echo ">> $*"; }
say "apt deps"; apt-get -qq update >/dev/null 2>&1; apt-get -qq install -y git cmake build-essential curl >/dev/null 2>&1
say "gpu"; nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader 2>/dev/null | head -1
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')
say "clone"; rm -rf /w && git clone -q --depth 1 https://github.com/anomly-labs/llama-cpp-et /w
say "build CUDA sm_$CC"; cmake -S /w -B /w/b -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="$CC" > /tmp/cfg.log 2>&1 && cmake --build /w/b --target llama-cli -j"$(nproc)" > /tmp/bld.log 2>&1 || { say "BUILD FAIL"; tail -5 /tmp/bld.log; echo "RUNPOD_RESULT arch=sm_$CC verdict=BUILD_FAIL"; sleep 20; exit 1; }
grep -c "exact profile: forcing" /tmp/cfg.log | sed 's/^/>> inexact backends forced off: /'
say "model"; curl -sL -o /w/m.gguf https://huggingface.co/Anomly/Qwen2.5-0.5B-Instruct-bposit8/resolve/main/Qwen2.5-0.5B-Instruct-bposit8.gguf
P="Write a Python function that returns the n-th Fibonacci number, with a docstring."
INVAR_LOGITS_OUT=/w/d.jsonl INVAR_LOGITS_LAYERS=1 INVAR_LOGITS_MATMULS=1 /w/b/bin/llama-cli -m /w/m.gguf -p "$P" --no-escape -n 8 --temp 0 --seed 1 -st --simple-io -t 4 -fa off -ngl 99 > /w/d.out 2>/dev/null
H=$(sha256sum /w/d.jsonl | cut -c1-16)
V=$([ "$H" = "$BOARD" ] && echo IDENTICAL || echo DIFFERENT)
echo "RUNPOD_RESULT arch=sm_$CC sha=$H board=$BOARD verdict=$V"
sleep 30
