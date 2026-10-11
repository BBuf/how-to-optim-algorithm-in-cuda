#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT=/scratch/bbuf-diffusion-20261011
framework=$1
run_label=$2
shift 2
if [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; then
  echo 'GPU is occupied; refusing to overlap benchmark processes.'
  exit 7
fi
export HF_HOME="$TASK_ROOT/models/qwen21"
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8
export FLASHINFER_DISABLE_VERSION_CHECK=1
export SGLANG_DIFFUSION_SYNC_STAGE_PROFILING=0
export TORCHINDUCTOR_CACHE_DIR="$TASK_ROOT/compile-cache/$framework"
export TRITON_CACHE_DIR="$TASK_ROOT/compile-cache/triton-$framework"
export TORCH_LOGS=recompiles
export VLLM_LOGGING_LEVEL=INFO
export IMAGE_BENCH_FRAMEWORK="$framework"
export IMAGE_BENCH_MODEL=qwen21
export IMAGE_BENCH_OUTPUT="$TASK_ROOT/results/$run_label"
MODEL_PATH=$(/opt/sglang/bin/python3 -c 'import json;print(json.load(open("/scratch/bbuf-diffusion-20261011/results/qwen21-checkpoint.json"))["snapshot_path"])')
if [ "$framework" = sglang ]; then
  export PYTHONPATH="$TASK_ROOT/source/sglang/python"
  bench_python=/opt/sglang/bin/python3
else
  export MASTER_PORT=29651
  bench_python="$TASK_ROOT/envs/vllm/bin/python"
fi
exec numactl --physcpubind=8-15 --membind=0 "$bench_python" "$TASK_ROOT/bench_qwen21_sdk_uncached.py" \
  --framework "$framework" --model "$MODEL_PATH" --output "$TASK_ROOT/results/$run_label" "$@"
