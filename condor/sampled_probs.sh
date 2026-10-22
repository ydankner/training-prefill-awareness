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
# Teacher-forced answer probabilities and per-token statistics (surprisal,
# entropy, top-1 probability, rank) of the sampled-data run on its test split,
# for both checkpoints and the base model: the analyses of run 2's answer_probs
# (17568222) and token_stats (17569177) jobs, for the run summary.
runs=(runs/sft-qwen3-32b-dolly-all-sampled-*)
[ "${#runs[@]}" = 1 ] || { echo "expected one sampled run, got ${runs[*]}"; exit 1; }
timeout --signal=TERM 7200 "$HOME/prefill-venv/bin/python" -m prefill_awareness.eval_run \
  --model Qwen/Qwen3-32B --run "${runs[0]}" \
  --data data/qwen3-32b-dolly-brainstorming-sampled-17607643.0-1790190404.jsonl \
         data/qwen3-32b-dolly-rest-sampled-17607643.1-1790190405.000.jsonl \
         data/qwen3-32b-dolly-rest-sampled-17607643.2-1790190408.001.jsonl \
  --no-generate --no-surprisal --answer-probs --token-stats --tag "${1:-manual}"
