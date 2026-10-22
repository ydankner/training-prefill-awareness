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
# Both checkpoints of the keep-think run (job 17607642) on run 2's data: every
# test record, an equal train sample, surprisal in both contexts. The think
# block is kept, as in training (read from the run's meta.json).
timeout --signal=TERM 10800 "$HOME/prefill-venv/bin/python" -m prefill_awareness.eval_run \
  --model Qwen/Qwen3-32B \
  --run runs/sft-qwen3-32b-dolly-all-keepthink-17607642.0-1790190403 \
  --data data/qwen3-32b-dolly-brainstorming-17567181.0-1789569567.jsonl \
         data/qwen3-32b-dolly-rest-17567560.0-1789577293.jsonl \
  --tag "${1:-manual}"
