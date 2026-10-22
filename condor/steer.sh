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
RUN=runs/sft-qwen3-32b-dolly-all-17568019.0-1789588306
# Steer 3 clean test answers of checkpoint-1108 at one chosen token, at every layer,
# along the base and fine-tuned plant directions (plant vs rest, minimal pairs)
# and a random control, over a fine grid of coefficients.
timeout --signal=TERM 7200 "$HOME/prefill-venv/bin/python" -m prefill_awareness.steer \
  --model Qwen/Qwen3-32B \
  --run "$RUN" \
  --checkpoint checkpoint-1108 \
  --data data/qwen3-32b-dolly-brainstorming-17567181.0-1789569567.jsonl \
         data/qwen3-32b-dolly-rest-17567560.0-1789577293.jsonl \
  --eval "$RUN/eval-checkpoint-1108-test-17568216.0.json" \
  --directions "$RUN/hidden-17569245.0/directions-17569599.0.pt" \
  --n-samples 3 \
  --tag "${1:-manual}"
