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

run_dir=runs/sft-qwen3-32b-dolly-all-17568019.0-1789588306
output_dir=runs/capability-dolly-128
manifest="$output_dir/manifest.jsonl"
mkdir -p "$output_dir"
if [ ! -f "$manifest" ]; then
  "$HOME/prefill-venv/bin/python" -m prefill_awareness.capability_eval prepare \
    --data data/qwen3-32b-dolly-brainstorming-17567181.0-1789569567.jsonl \
           data/qwen3-32b-dolly-rest-17567560.0-1789577293.jsonl \
    --out "$manifest" --n 128 --recovery-n 32 --seed 0
fi

timeout --signal=TERM 7200 "$HOME/prefill-venv/bin/python" -m prefill_awareness.capability_eval run \
  --manifest "$manifest" --run "$run_dir" --out "$output_dir" --batch-size 8
"$HOME/prefill-venv/bin/python" -m prefill_awareness.capability_eval report \
  --manifest "$manifest" --out "$output_dir"
