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
# Attention to plants (base and checkpoint-1108, all 997 test records), the geometry
# of the plant-reading heads along the fine-tuned minimal-pair direction, and
# attention to the steered token of the 3 steering answers.
timeout --signal=TERM 7200 "$HOME/prefill-venv/bin/python" -m prefill_awareness.attention \
  --model Qwen/Qwen3-32B \
  --run "$RUN" \
  --checkpoint checkpoint-1108 \
  --data data/qwen3-32b-dolly-brainstorming-17567181.0-1789569567.jsonl \
         data/qwen3-32b-dolly-rest-17567560.0-1789577293.jsonl \
  --eval "$RUN/eval-checkpoint-1108-test-17568216.0.json" \
  --dump "$RUN/hidden-17569245.0" \
  --directions "$RUN/hidden-17569245.0/directions-17569599.0.pt" \
  --steering-meta "$RUN/steering-meta-17569669.0.json" \
  --tag "${1:-manual}"
