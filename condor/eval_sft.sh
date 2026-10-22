#!/bin/bash
set -euo pipefail
cd "$HOME/prefill-awareness"
export HF_HOME="$HOME/.cache/huggingface"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
nvidia-smi
# A faulty GPU leaves torch without CUDA, and the model then runs on CPU for
# hours. Fail instead, which holds the job.
"$HOME/prefill-venv/bin/python" -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'"
# Every generation of the first SFT adapter, on the whole test split and on as
# many training examples: learned-but-not-generalised looks different from
# not-learned-at-all.
run="runs/sft-qwen3-32b-dolly-17567334.0-1789573461"
data="data/qwen3-32b-dolly-brainstorming-17567181.0-1789569567.jsonl"
for split in test train; do
  timeout --signal=TERM 1200 "$HOME/prefill-venv/bin/python" -m prefill_awareness.evaluate \
    --model Qwen/Qwen3-32B --adapter "$run/adapter" --data "$data" --split "$split" --n 162 \
    --batch-size 8 --max-new-tokens 64 --out "$run/eval-$split-${1:-manual}.json"
done
