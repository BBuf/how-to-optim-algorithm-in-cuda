# PR39068 accuracy validation

The accuracy lane uses genuine DSpark acceptance. The blog's random-input,
simulated-acceptance performance measurements are a separate experiment.

Candidate: `835c39094ad017c2f54f8ea598002e669e6fa30d`, on 4×GB300, TP4/EP4.
Model revision: `dba1be0a40aa45a94ad051997016db3960a90277`.
Historical reference: `0669e3d9464c`, on 4×B300. The historical reference was
reused without rerunning it; this is not a new paired comparison on identical
hardware. Dataset, prompt recipe and sampling parameters are recorded below.

Aggregate results are in `results.json`; `samples.jsonl` contains compact records
for all 3138 evaluated responses (1314 + 1314 + 30 + 480). The records retain
incorrect and truncated responses.

## Server

Use the dependencies listed in `../random-dspark/README.md`. Start in the
candidate SGLang checkout, with the checkpoint path set in `MODEL_PATH`.
Clear inherited `SGLANG_*` overrides, then run:

```bash
unset SGLANG_SIMULATE_ACC_LEN SGLANG_SIMULATE_ACC_METHOD
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH="$PWD/python" MAX_JOBS=16 \
SGLANG_RAGGED_VERIFY_MODE=static \
python -m sglang.launch_server \
  --model-path "$MODEL_PATH" --tp 4 --ep-size 4 --trust-remote-code \
  --mem-fraction-static 0.80 --max-total-tokens 33554432 \
  --chunked-prefill-size 4096 \
  --cuda-graph-bs-decode 1 2 4 8 16 32 64 --max-running-requests 128 \
  --speculative-algorithm DSPARK --speculative-dspark-block-size 5 \
  --skip-server-warmup --reasoning-parser deepseek-v41 \
  --random-seed 42 --decode-log-interval 10 \
  --host 127.0.0.1 --port 30021
```

In the client terminal, use the candidate checkout on `PYTHONPATH` too.
The server must be healthy before the following commands. Run the lanes
sequentially. No benchmark or profiler runs concurrently with accuracy.

```bash
curl -f -X POST http://127.0.0.1:30021/freeze_gc
```

## GSM8K

`gsm_eval.py` is the evaluator used for this run. It uses the candidate's legacy
`simple_eval_mixed_prefix_gsm8k` prompt helpers and numeric answer scorer.
The first five test rows are demonstrations; rows 5 through 1318 are scored
(zero-based indices, 1314 questions). Chat API, temperature 0, top-p 1,
seed 0, maximum output 4096 tokens. EOS stopping is enabled.

```bash
curl -fL https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl -o gsm8k-test.jsonl
python - <<'PY'
import hashlib
from pathlib import Path
assert hashlib.sha256(Path('gsm8k-test.jsonl').read_bytes()).hexdigest() == '3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14'
PY
curl -f -X POST http://127.0.0.1:30021/flush_cache
python gsm_eval.py --out gsm-serial1314 --threads 1 --count 1314
curl -f -X POST http://127.0.0.1:30021/flush_cache
python gsm_eval.py --out gsm-concurrent1314 --threads 64 --count 1314
```

Keep the dataset beside `gsm_eval.py`. Every request, including errors, is
retained in `samples.jsonl`; `summary.json` records counts, truncations and
empty answers. Failed requests count as incorrect and also fail the evaluator.

## AIME 2026

Evaluator `sgl-eval==0.1.0`, `math-verify==0.9.0`,
`latex2sympy2_extended==1.11.0`, `antlr4-python3-runtime==4.9.3`,
`editdistance==0.8.1`. Use the bundled AIME 2026 dataset and MathArena prompt:

```bash
python -m pip install sgl-eval==0.1.0 math-verify==0.9.0 \
  latex2sympy2_extended==1.11.0 antlr4-python3-runtime==4.9.3 editdistance==0.8.1
EVAL_DIR=$(python -c 'from pathlib import Path; import sgl_eval; print(Path(sgl_eval.__file__).resolve().parent)')
AIME_DATA="$EVAL_DIR/_vendored/nemo_skills/dataset/aime26/test.txt"
AIME_PROMPT="$EVAL_DIR/_vendored/nemo_skills/prompts/matharena-aime.yaml"
export OPENAI_API_KEY=EMPTY
curl -f -X POST http://127.0.0.1:30021/flush_cache
python -m sgl_eval.cli run aime26 \
  --base-url http://127.0.0.1:30021/v1 --model "$MODEL_PATH" \
  --from-dataset "$AIME_DATA" --prompt "$AIME_PROMPT" \
  --num-threads 1 --n-repeats 1 --thinking --reasoning-effort max \
  --temperature 0 --top-p 0.95 --seed 0 --max-tokens 65536 \
  --out-dir aime-serial30
python -m sgl_eval.cli run aime26 \
  --base-url http://127.0.0.1:30021/v1 --model "$MODEL_PATH" \
  --from-dataset "$AIME_DATA" --prompt "$AIME_PROMPT" \
  --num-threads 64 --n-repeats 16 --thinking --reasoning-effort max \
  --temperature 1 --top-p 0.95 --max-tokens 65536 \
  --out-dir aime-repeat16
```

- Dataset SHA256: `6a43e48d55eb7736003f53074fef041a3da2b78d902d3fcd0b1974b2523221fc`.
- Prompt SHA256: `12d8c6381b0191d2e8f6b1ecd39a65229f35af6f7aff21716bb951d456133afb`.

The serial lane tests 30 questions. The repeated lane retains all 480 responses
(16 per question), with no fixed request seed. Report average correctness over
all responses, not pass@16. Truncations remain in the denominator and are
reported separately. Full local response records and source/package manifests
were retained for auditing; the public compact records include their hashes.
