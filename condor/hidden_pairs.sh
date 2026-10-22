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
df -h "$HOME" | tail -1
# Minimal pairs at the plant: the state at the plant position with the plant and with
# the original token put back, for the base model and checkpoint-1108 (~1.3 GB).
timeout --signal=TERM 1800 "$HOME/prefill-venv/bin/python" -m prefill_awareness.hidden_states \
  --model Qwen/Qwen3-32B \
  --run runs/sft-qwen3-32b-dolly-all-17568019.0-1789588306 \
  --checkpoint checkpoint-1108 \
  --data data/qwen3-32b-dolly-brainstorming-17567181.0-1789569567.jsonl \
         data/qwen3-32b-dolly-rest-17567560.0-1789577293.jsonl \
  --out runs/sft-qwen3-32b-dolly-all-17568019.0-1789588306/hidden-17569245.0 --pairs
