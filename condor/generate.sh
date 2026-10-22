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
# Training dataset: every Dolly brainstorming prompt, same settings as the pilot
# but batched. A unique output per execution preserves records after a retry.
out="data/qwen3-32b-dolly-brainstorming-${1:-manual}-$(date +%s).jsonl"
timeout --signal=TERM 10800 "$HOME/prefill-venv/bin/python" -m prefill_awareness.data_gen \
  --model Qwen/Qwen3-32B --no-enable-thinking \
  --prompts data/dolly-brainstorming-prompts.jsonl --out "$out" \
  --n 1751 --batch-size 32 --greedy --max-words 200 --max-new-tokens 512
