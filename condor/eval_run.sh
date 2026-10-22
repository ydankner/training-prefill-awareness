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
# Both checkpoints of the second SFT run (end of epoch 1 and 2): every test
# record, an equally large random train sample, and per-record surprisal ranks.
timeout --signal=TERM 7200 "$HOME/prefill-venv/bin/python" -m prefill_awareness.eval_run \
  --model Qwen/Qwen3-32B \
  --run runs/sft-qwen3-32b-dolly-all-17568019.0-1789588306 \
  --data data/qwen3-32b-dolly-brainstorming-17567181.0-1789569567.jsonl \
         data/qwen3-32b-dolly-rest-17567560.0-1789577293.jsonl \
  --tag "${1:-manual}"
