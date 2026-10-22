#!/bin/bash
set -euo pipefail
cd "$HOME/prefill-awareness"
export HF_HOME="$HOME/.cache/huggingface"
export HF_HUB_OFFLINE=1
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
RUN=runs/sft-qwen3-32b-dolly-all-17568019.0-1789588306
# Plant directions at every layer of the hidden-state dump (base and checkpoint-1108),
# including the minimal pairs; wrong names come from the full test evaluation.
timeout --signal=TERM 10800 "$HOME/prefill-venv/bin/python" -m prefill_awareness.directions \
  --dump "$RUN/hidden-17569245.0" \
  --eval "$RUN/eval-checkpoint-1108-test-17568216.0.json" \
  --tokenizer Qwen/Qwen3-32B \
  --tag "${1:-manual}"
