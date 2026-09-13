#!/usr/bin/env bash
# Copyright (c) 2026 Anomly, Inc. All rights reserved. Author: Ry Bruscoe.
# Metal exact-profile regression gate: build the Metal exact build fresh and assert the whole-graph
# dump on the Apple GPU is bit-identical to the recorded x86/CUDA references, and that no exact
# kernel is missing from the shader. This locks in the 2026-09-13 fix (missing ANOMLY_METAL_EXACT
# compile def + kernel_bin_exact + kernel_geglu) so it cannot silently regress again.
set -u
export PATH=/opt/homebrew/bin:$PATH
cd "$(dirname "$0")/../.." || exit 2
PROMPT="Write a Python function that returns the n-th Fibonacci number, with a docstring."
# model path : expected sorted-sha256 (first 16 hex)
declare -a CASES=(
  "$HOME/development/SmolLM2-135M-Instruct-bposit8.gguf:68edecdc52a5e19e"
  "$HOME/development/Gemma-3-270M-it-bposit8.gguf:6e742d52fe1ffc51"
)
echo "== configuring/building build-metal =="
cmake -B build-metal -DGGML_METAL=ON -DANOMLY_METAL_EXACT=ON -DCMAKE_BUILD_TYPE=Release >/dev/null 2>&1
cmake --build build-metal --target llama-cli -j >/tmp/mgate_build.log 2>&1 || { echo "BUILD FAILED"; tail -5 /tmp/mgate_build.log; exit 1; }
fail=0
for c in "${CASES[@]}"; do
  m="${c%%:*}"; want="${c##*:}"
  [ -f "$m" ] || { echo "SKIP (missing model): $m"; continue; }
  d=$(mktemp); e=$(mktemp)
  INVAR_LOGITS_OUT="$d" INVAR_LOGITS_LAYERS=1 INVAR_LOGITS_MATMULS=1 \
    ./build-metal/bin/llama-cli -m "$m" -p "$PROMPT" -n 8 --temp 0 --seed 1 -st --simple-io -t 4 -fa off -ngl 99 2>"$e" >/dev/null
  miss=$(grep -c "not found in the library" "$e")
  got=$(LC_ALL=C sort "$d" | shasum -a 256 | cut -c1-16)
  bn=$(basename "$m")
  if [ "$got" = "$want" ] && [ "$miss" = "0" ]; then echo "PASS  $bn  $got"
  else echo "FAIL  $bn  got=$got want=$want missing_kernels=$miss"; fail=1; fi
  rm -f "$d" "$e"
done
[ "$fail" = "0" ] && { echo "METAL EXACT GATE: PASS"; exit 0; } || { echo "METAL EXACT GATE: FAIL"; exit 1; }
