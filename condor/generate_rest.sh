#!/bin/bash
set -euo pipefail
cd "$HOME/prefill-awareness"
export HF_HOME="$HOME/.cache/huggingface"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
mkdir -p data
nvidia-smi
# A faulty GPU leaves torch without CUDA, and the model then runs on CPU for
# hours. Fail instead, which holds the job.
"$HOME/prefill-venv/bin/python" -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'"
# Second training dataset: the Dolly categories without a context passage that
# were not generated yet, keeping only mid-length answers (>= 100 tokens) for
# clean and modified alike. A unique output per execution preserves records
# after a retry.
out="data/qwen3-32b-dolly-rest-${1:-manual}-$(date +%s).jsonl"
timeout --signal=TERM 21600 "$HOME/prefill-venv/bin/python" -m prefill_awareness.data_gen \
  --model Qwen/Qwen3-32B --no-enable-thinking \
  --prompts data/dolly-rest-prompts.jsonl --out "$out" \
  --n 100000 --batch-size 32 --greedy --max-words 200 --max-new-tokens 512 \
  --min-response-tokens 100
