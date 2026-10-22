#!/bin/bash
set -euo pipefail
cd "$HOME/prefill-awareness"
export HF_HOME="$HOME/.cache/huggingface"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1

"$HOME/prefill-venv/bin/python" -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'"

manifest=data/gsm8k-test-200.jsonl
test -s "$manifest"
run_dir=runs/sft-qwen3-32b-dolly-all-17568019.0-1789588306
output_dir=runs/capability-gsm8k-200
timeout --signal=TERM 10800 "$HOME/prefill-venv/bin/python" -m prefill_awareness.gsm8k_eval run \
  --manifest "$manifest" --run "$run_dir" --out "$output_dir" --batch-size 8
"$HOME/prefill-venv/bin/python" -m prefill_awareness.gsm8k_eval report \
  --manifest "$manifest" --out "$output_dir"
