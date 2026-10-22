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
# Ablation: the data and settings of run 2 (job 17568019), but the answer also
# names the original token, "changed -> original" (formats.apply_answer_format).
# The planted token stays first, so the in-loop learning curve (same 128 test
# records, same seed) is directly comparable with run 2 and the keep-think run.
out="runs/sft-qwen3-32b-dolly-all-plantorig-${1:-manual}-$(date +%s)"
timeout --signal=TERM 18000 "$HOME/prefill-venv/bin/python" -m prefill_awareness.train \
  --model Qwen/Qwen3-32B \
  --data data/qwen3-32b-dolly-brainstorming-17567181.0-1789569567.jsonl \
         data/qwen3-32b-dolly-rest-17567560.0-1789577293.jsonl \
  --out "$out" --gradient-checkpointing --save-strategy epoch --answer-format plant-orig
