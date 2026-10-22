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
# Ablation: SFT with the settings of run 2 (job 17568019) on the datasets
# regenerated with T=1 sampling (job 17607643), all records, no filtering on the
# plant's surprisal rank.
out="runs/sft-qwen3-32b-dolly-all-sampled-${1:-manual}-$(date +%s)"
timeout --signal=TERM 18000 "$HOME/prefill-venv/bin/python" -m prefill_awareness.train \
  --model Qwen/Qwen3-32B \
  --data data/qwen3-32b-dolly-brainstorming-sampled-17607643.0-1790190404.jsonl \
         data/qwen3-32b-dolly-rest-sampled-17607643.1-1790190405.000.jsonl \
         data/qwen3-32b-dolly-rest-sampled-17607643.2-1790190408.001.jsonl \
  --out "$out" --gradient-checkpointing --save-strategy epoch
