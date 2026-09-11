#!/usr/bin/env bash
set -euo pipefail

# Run from the selected SGLang source checkout. MODEL_PATH is the local snapshot.
: "${MODEL_PATH:?Set MODEL_PATH to the DeepSeek-V4.1-Flash checkpoint}"
unset SGLANG_SIMULATE_ACC_LEN SGLANG_SIMULATE_ACC_METHOD
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}"
export MAX_JOBS=16

dspark_args=()
if [[ "${DSPARK:-1}" == 1 ]]; then
  dspark_args=(--speculative-algorithm DSPARK --speculative-dspark-block-size 5)
fi

python -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --served-model-name deepseek-ai/DeepSeek-V4.1-Flash \
  --tp 4 --ep-size 4 --trust-remote-code \
  --mem-fraction-static 0.80 --max-total-tokens 33554432 \
  --chunked-prefill-size 4096 \
  --cuda-graph-bs-decode 1 2 4 8 16 32 64 \
  --max-running-requests 128 --skip-server-warmup \
  --reasoning-parser deepseek-v41 --random-seed 42 \
  --decode-log-interval 10 --host 127.0.0.1 --port 30021 \
  "${dspark_args[@]}"
