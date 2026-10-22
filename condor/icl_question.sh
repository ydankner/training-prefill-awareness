#!/bin/bash
set -euo pipefail
# Runs in the checkout it was submitted from, so parallel clones do not collide.
cd "$(dirname "$0")/.."
export HF_HOME="$HOME/.cache/huggingface"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
mkdir -p data runs
nvidia-smi
# A faulty GPU leaves torch without CUDA, and the model then runs on CPU for
# hours. Fail instead, which holds the job.
"$HOME/prefill-venv/bin/python" -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'"
# In-context learning baseline of the untuned model on run 2's data: the same 50
# test records (25 tampered, 25 clean) with 0-64 random train records shown as
# examples in the follow-up turn, before the detection question, so the tampered
# turn is read in its usual context (prefill_awareness.icl). Same seed as
# condor/icl.sh: the same test records and the same shots per record.
timeout --signal=TERM 5400 "$HOME/prefill-venv/bin/python" -m prefill_awareness.icl \
  --model Qwen/Qwen3-32B \
  --data data/qwen3-32b-dolly-brainstorming-17567181.0-1789569567.jsonl \
         data/qwen3-32b-dolly-rest-17567560.0-1789577293.jsonl \
  --shots 0 1 2 4 8 16 32 64 --n-test 50 --placement question \
  --out "runs/icl-question-qwen3-32b-${1:-manual}-$(date +%s)"
