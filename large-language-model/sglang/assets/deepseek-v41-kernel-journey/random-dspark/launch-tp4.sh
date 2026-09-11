#!/usr/bin/env bash
set -euo pipefail

# MoE TP4 + padding (EP1), with attention TP4; run from checkout 835c3909.
# MODEL_PATH is the local DeepSeek-V4.1-Flash snapshot.
: "${MODEL_PATH:?Set MODEL_PATH to the DeepSeek-V4.1-Flash checkpoint}"
unset SGLANG_SIMULATE_ACC_LEN SGLANG_SIMULATE_ACC_METHOD SGLANG_RAGGED_VERIFY_MODE
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}"
export MAX_JOBS=16

dspark_args=()
if [[ "${DSPARK:-1}" == 1 ]]; then
  export SGLANG_RAGGED_VERIFY_MODE=static
  export SGLANG_SIMULATE_ACC_LEN=5.5
  export SGLANG_SIMULATE_ACC_METHOD=match-expected
  dspark_args=(--speculative-algorithm DSPARK --speculative-dspark-block-size 5)
fi

python -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --served-model-name deepseek-ai/DeepSeek-V4.1-Flash \
  --tp 4 --ep-size 1 --trust-remote-code \
  --moe-a2a-backend none --moe-runner-backend flashinfer_mxfp4 \
  --mem-fraction-static 0.80 --max-total-tokens 33554432 \
  --chunked-prefill-size 4096 \
  --cuda-graph-bs-decode 1 2 4 8 16 32 64 \
  --max-running-requests 128 --skip-server-warmup \
  --reasoning-parser deepseek-v41 --random-seed 42 \
  --decode-log-interval 10 --host 127.0.0.1 --port 30021 \
  "${dspark_args[@]}"
