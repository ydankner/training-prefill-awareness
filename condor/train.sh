#!/bin/bash
set -euo pipefail
cd "$HOME/prefill-awareness"
export HF_HOME="$HOME/.cache/huggingface"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
mkdir -p runs
nvidia-smi
# A faulty GPU leaves torch without CUDA, and the model then runs on CPU for
# hours. Fail instead, which holds the job.
"$HOME/prefill-venv/bin/python" -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'"
# LoRA SFT of Qwen3-32B on both Dolly datasets (brainstorming plus the remaining
# no-context categories), with the default LoRA and optimiser settings and
# in-loop evaluation on the test split. The first run (job 17567334, commit
# 513c2ee) trained on the brainstorming dataset alone.
out="runs/sft-qwen3-32b-dolly-all-${1:-manual}-$(date +%s)"
timeout --signal=TERM 18000 "$HOME/prefill-venv/bin/python" -m prefill_awareness.train \
  --model Qwen/Qwen3-32B \
  --data data/qwen3-32b-dolly-brainstorming-17567181.0-1789569567.jsonl \
         data/qwen3-32b-dolly-rest-17567560.0-1789577293.jsonl \
  --out "$out" --gradient-checkpointing --save-strategy epoch
