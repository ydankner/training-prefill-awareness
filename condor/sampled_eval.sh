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
# The sampled-data run on its own test split, then both ways across datasets:
# the sampled run on the greedy data of run 2, and run 2 on the sampled data.
# Run 2's checkpoints are symlinked into this checkout's copy of its run
# directory, so outputs land here.
sampled=(data/qwen3-32b-dolly-brainstorming-sampled-17607643.0-1790190404.jsonl
         data/qwen3-32b-dolly-rest-sampled-17607643.1-1790190405.000.jsonl
         data/qwen3-32b-dolly-rest-sampled-17607643.2-1790190408.001.jsonl)
greedy=(data/qwen3-32b-dolly-brainstorming-17567181.0-1789569567.jsonl
        data/qwen3-32b-dolly-rest-17567560.0-1789577293.jsonl)
runs=(runs/sft-qwen3-32b-dolly-all-sampled-*)
[ "${#runs[@]}" = 1 ] || { echo "expected one sampled run, got ${runs[*]}"; exit 1; }
py() { timeout --signal=TERM 9000 "$HOME/prefill-venv/bin/python" -m prefill_awareness.eval_run --model Qwen/Qwen3-32B "$@"; }
py --run "${runs[0]}" --data "${sampled[@]}" --tag "${1:-manual}"
py --run "${runs[0]}" --data "${greedy[@]}" --tag "${1:-manual}-greedydata"
py --run runs/sft-qwen3-32b-dolly-all-17568019.0-1789588306 --data "${sampled[@]}" --tag "${1:-manual}-sampleddata"
