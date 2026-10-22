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
# Pre-SFT baseline on every record of the pilot dataset (train and test alike).
out="runs/baseline-qwen3-32b-pilot-${1:-manual}-$(date +%s).json"
timeout --signal=TERM 1200 "$HOME/prefill-venv/bin/python" -m prefill_awareness.evaluate \
  --model Qwen/Qwen3-32B \
  --data data/qwen3-32b-pilot-17566671.0-1789561019.jsonl --split all \
  --batch-size 8 --max-new-tokens 64 --out "$out"
